from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.category_repository import CategoryRepository
from app.catalog.product_repository import ProductRepository
from app.core.config import LlmSettings, get_llm_settings
from app.distributors.category_refresh import (
    CategoryRefreshResult,
    refresh_distributor_categories,
)
from app.llm.base import LlmClient, LlmError, LlmHttpError
from app.llm.full_category_composer import (
    V3_CODE_VALIDATION_BYPASSED,
    V3_FULL_CATEGORY_MATRIX_MODE,
    V3_MECHANICAL_VALIDATION_FAILED,
    V3_NO_RECOMMENDATION,
    V3_PROVIDER_ERROR,
    V3_PROVIDER_NOT_CONFIGURED,
    V3_SCHEMA_VALIDATION_FAILED,
    V3_VALIDATED,
    compose_full_category_quote,
)
from app.llm.openai_compatible import OpenAICompatibleLlmClient
from app.matching.full_category_matrix import (
    MATRIX_EMPTY_AFTER_CATEGORY_SELECTION,
    MATRIX_TOO_LARGE_FOR_MODEL,
    build_full_category_matrix_group_package,
)
from app.matching.stock_refresh_fallback import (
    add_stock_refresh_cache_warning,
    stock_refresh_cached_fallback_diagnostics,
    stock_refresh_used_cached_fallback,
)
from app.matching.v3_full_category_profiles import (
    V3_FULL_CATEGORY_PROFILE_ALIASES,
    V3_FULL_CATEGORY_PROFILES,
    resolve_v3_full_category_profile,
)

QUOTE_DRAFT_REVIEW_REQUIRED = "quote_draft_review_required"
QUOTE_CANDIDATE_CUSTOMER_READY = "quote_candidate_customer_ready"
NO_RECOMMENDATION = "no_recommendation"
MATRIX_TOO_LARGE_FOR_MODEL_STATE = "matrix_too_large_for_model"
MATRIX_EMPTY_AFTER_CATEGORY_SELECTION_STATE = "matrix_empty_after_category_selection"
PROVIDER_ERROR = "provider_error"
PROVIDER_NOT_CONFIGURED = "provider_not_configured"
MECHANICAL_VALIDATION_FAILED = "mechanical_validation_failed"
SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
V3_REQUEST_INTAKE_MODE = "v3_request_intake"
REQUEST_INTAKE_PROMPT_VERSION_V7 = "request_intake_v7"
RESOLVED_REQUEST_SCHEMA_VERSION_V7 = "resolved_request_schema_v7"
REQUEST_INTAKE_PROMPT_VERSION_V7_1 = "request_intake_v7_1"
RESOLVED_REQUEST_SCHEMA_VERSION_V7_1 = "resolved_request_schema_v7_1"
V3_CATEGORY_INTAKE_NO_MATCHING_CATEGORY = "v3_category_intake_no_matching_category"
V3_REQUEST_INTAKE_SCHEMA_VALIDATION_FAILED = "v3_request_intake_schema_validation_failed"
V3_STOCK_REFRESH_FAILED = "v3_stock_refresh_failed"
STOCK_REFRESH_FAILED_STATE = "stock_refresh_failed"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class V3FullCategoryQuoteResult:
    profile: str | None
    category_ids: list[str]
    distributor_code: str
    result_state: str
    report_json: dict[str, Any]


@dataclass(frozen=True)
class V3RequestIntakeDecision:
    status: str
    profile: str | None = None
    category_ids: list[str] = field(default_factory=list)
    reason: str | None = None
    resolved_request: dict[str, Any] = field(default_factory=dict)
    raw_output: dict[str, Any] | None = None
    error_type: str | None = None
    http_status: int | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    canonical_input_hash: str | None = None
    input_char_count: int | None = None


async def run_v3_full_category_quote(
    *,
    text: str,
    session: AsyncSession,
    profile: str | None = None,
    category_ids: Sequence[str] | None = None,
    distributor_code: str = "ocs",
    settings: LlmSettings | None = None,
) -> V3FullCategoryQuoteResult:
    effective_settings = settings or get_llm_settings()
    intake_decision: V3RequestIntakeDecision | None = None
    category_repository = CategoryRepository(session)
    if not profile and not category_ids:
        category_catalog = _category_catalog_for_intake(
            await category_repository.list_all_categories(distributor_code)
        )
        if not category_catalog:
            intake_decision = V3RequestIntakeDecision(
                status="no_matching_category",
                reason="No distributor category catalog is available for v3 request intake.",
            )
            return _routing_failure_result(
                text=text,
                distributor_code=distributor_code,
                router_decision=intake_decision,
                settings=effective_settings,
            )
        intake_decision = await asyncio.to_thread(
            route_v3_full_category_target,
            text=text,
            settings=effective_settings,
            category_catalog=category_catalog,
        )
        if (
            intake_decision.status not in {"selected_category", "selected_profile"}
            or not intake_decision.category_ids
        ):
            return _routing_failure_result(
                text=text,
                distributor_code=distributor_code,
                router_decision=intake_decision,
                settings=effective_settings,
            )
        profile = intake_decision.profile
        category_ids = intake_decision.category_ids

    profile_name, resolved_category_ids = resolve_v3_full_category_profile(
        profile=profile,
        category_ids=category_ids,
    )
    root_category_ids = list(resolved_category_ids)
    resolved_category_ids = await category_repository.list_category_ids_with_descendants(
        distributor_code=distributor_code,
        category_ids=root_category_ids,
    )
    resolved_request = _resolved_request_json(
        intake_decision.resolved_request if intake_decision is not None else {},
        text=text,
        profile=profile_name,
        source="request_intake" if intake_decision is not None else "profile_override",
    )

    stock_refresh_diagnostics: dict[str, Any] = {
        "enabled": bool(effective_settings.v3_refresh_categories_before_llm),
        "status": "disabled",
    }
    repository = ProductRepository(session)
    rows: list[Any] | None = None
    if effective_settings.v3_refresh_categories_before_llm:
        refresh_result = await refresh_distributor_categories(
            session,
            distributor_code=distributor_code,
            category_ids=resolved_category_ids,
        )
        stock_refresh_diagnostics = {
            "enabled": True,
            **refresh_result.to_diagnostics(),
        }
        if not refresh_result.success:
            rows = await repository.list_latest_full_category_group_matrix(
                distributor_code,
                resolved_category_ids,
            )
            if rows:
                stock_refresh_diagnostics = stock_refresh_cached_fallback_diagnostics(
                    stock_refresh_diagnostics,
                    cached_matrix_row_count=len(rows),
                )
            else:
                return _stock_refresh_failure_result(
                    text=text,
                    distributor_code=distributor_code,
                    profile_name=profile_name,
                    root_category_ids=root_category_ids,
                    resolved_category_ids=resolved_category_ids,
                    resolved_request=resolved_request,
                    stock_refresh_result=refresh_result,
                    settings=effective_settings,
                    intake_decision=intake_decision,
                )

    if rows is None:
        rows = await repository.list_latest_full_category_group_matrix(
            distributor_code,
            resolved_category_ids,
        )
    matrix_package = build_full_category_matrix_group_package(
        distributor_code=distributor_code,
        category_ids=resolved_category_ids,
        rows=rows,
        max_package_chars=effective_settings.llm_configurator_max_package_chars,
        model=effective_settings.llm_model,
    )

    outcome = await asyncio.to_thread(
        _compose_v3_full_category_quote_sync,
        text=text,
        resolved_request=resolved_request,
        matrix_package=matrix_package,
        settings=effective_settings,
    )
    report_json = outcome.to_report_json()
    report_json["v3_result_state"] = v3_result_state(report_json)
    report_json["v3_profile"] = profile_name
    report_json["category_ids"] = resolved_category_ids
    report_json["root_category_ids"] = root_category_ids
    report_json["distributor_code"] = distributor_code
    report_json["source_text"] = text
    report_json["resolved_request"] = resolved_request
    if intake_decision is not None:
        report_json["v3_request_intake"] = _routing_decision_json(intake_decision)
    if stock_refresh_used_cached_fallback(stock_refresh_diagnostics):
        add_stock_refresh_cache_warning(report_json)

    diagnostics = report_json.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics.setdefault("category_ids", resolved_category_ids)
        diagnostics.setdefault("root_category_ids", root_category_ids)
        diagnostics.setdefault("matrix_row_count", len(rows))
        diagnostics.setdefault(
            "matrix_component_count",
            matrix_package.payload.get("diagnostics", {}).get("component_count"),
        )
        diagnostics.setdefault(
            "matrix_stock_row_count",
            matrix_package.payload.get("diagnostics", {}).get("stock_row_count"),
        )
        diagnostics.setdefault("matrix_char_count", matrix_package.char_count)
        diagnostics.setdefault("matrix_status", matrix_package.status)
        diagnostics.setdefault("model", effective_settings.llm_model)
        diagnostics.setdefault("resolved_request", resolved_request)
        diagnostics.setdefault("stock_refresh", stock_refresh_diagnostics)
        if intake_decision is not None:
            diagnostics.setdefault("request_intake_used", True)
            diagnostics.setdefault(
                "request_intake_decision",
                _routing_decision_json(intake_decision),
            )

    if report_json.get("final_status_source") == V3_MECHANICAL_VALIDATION_FAILED:
        logger.warning(
            "v3_full_category_mechanical_validation_failed "
            "profile=%s category_count=%s errors=%s "
            "matrix_rows=%s prompt_chars=%s",
            profile_name,
            len(resolved_category_ids),
            report_json.get("v3_validation_errors", [])[:6],
            diagnostics.get("matrix_row_count") if isinstance(diagnostics, dict) else None,
            diagnostics.get("prompt_char_count") if isinstance(diagnostics, dict) else None,
        )

    return V3FullCategoryQuoteResult(
        profile=profile_name,
        category_ids=resolved_category_ids,
        distributor_code=distributor_code,
        result_state=report_json["v3_result_state"],
        report_json=report_json,
    )


def route_v3_full_category_target(
    *,
    text: str,
    settings: LlmSettings | None = None,
    category_catalog: Sequence[Mapping[str, Any]] | None = None,
    llm_client: LlmClient | None = None,
) -> V3RequestIntakeDecision:
    effective_settings = settings or get_llm_settings()
    client = llm_client or _create_llm_client(effective_settings)
    if client is None:
        return V3RequestIntakeDecision(
            status="provider_not_configured",
            reason="LLM provider is not configured for v3 request intake.",
        )

    try:
        system_prompt = _request_intake_system_prompt(effective_settings)
        user_prompt = _request_intake_user_prompt(
            text,
            category_catalog=category_catalog or [],
            settings=effective_settings,
        )
        raw_output = client.generate_json(
            system_prompt,
            user_prompt,
        )
    except LlmHttpError as exc:
        return V3RequestIntakeDecision(
            status="provider_error",
            reason="LLM provider error before v3 request intake completed.",
            error_type=type(exc).__name__,
            http_status=exc.status_code,
        )
    except LlmError as exc:
        return V3RequestIntakeDecision(
            status="provider_error",
            reason="LLM error before v3 request intake completed.",
            error_type=type(exc).__name__,
        )
    finally:
        if llm_client is None and isinstance(client, OpenAICompatibleLlmClient):
            client.close()

    return _parse_request_intake_output(
        raw_output,
        category_catalog=category_catalog or [],
        prompt_version=_request_intake_prompt_version(effective_settings),
        schema_version=(
            RESOLVED_REQUEST_SCHEMA_VERSION_V7_1
            if _request_intake_prompt_version(effective_settings)
            == REQUEST_INTAKE_PROMPT_VERSION_V7_1
            else RESOLVED_REQUEST_SCHEMA_VERSION_V7
        ),
        canonical_input_hash=_sha256_text(user_prompt),
        input_char_count=len(system_prompt) + len(user_prompt),
    )


def v3_result_state(report_json: dict[str, Any]) -> str:
    final_status_source = str(report_json.get("final_status_source") or "")
    primary_status = str(report_json.get("primary_recommendation_status") or "")
    quote = report_json.get("validated_quote")

    if (
        final_status_source in {V3_VALIDATED, V3_CODE_VALIDATION_BYPASSED}
        and primary_status == "valid"
    ):
        engineering_review_required = True
        if isinstance(quote, dict):
            engineering_review_required = bool(
                quote.get("engineering_review_required", True)
            )
        if engineering_review_required:
            return QUOTE_DRAFT_REVIEW_REQUIRED
        return QUOTE_CANDIDATE_CUSTOMER_READY
    if final_status_source == V3_NO_RECOMMENDATION:
        return NO_RECOMMENDATION
    if final_status_source == MATRIX_TOO_LARGE_FOR_MODEL:
        return MATRIX_TOO_LARGE_FOR_MODEL_STATE
    if final_status_source == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION:
        return MATRIX_EMPTY_AFTER_CATEGORY_SELECTION_STATE
    if final_status_source == V3_MECHANICAL_VALIDATION_FAILED:
        return MECHANICAL_VALIDATION_FAILED
    if final_status_source == V3_SCHEMA_VALIDATION_FAILED:
        return SCHEMA_VALIDATION_FAILED
    if final_status_source == V3_PROVIDER_NOT_CONFIGURED:
        return PROVIDER_NOT_CONFIGURED
    if final_status_source == V3_PROVIDER_ERROR:
        return PROVIDER_ERROR
    if final_status_source == V3_REQUEST_INTAKE_SCHEMA_VALIDATION_FAILED:
        return SCHEMA_VALIDATION_FAILED
    if final_status_source == V3_CATEGORY_INTAKE_NO_MATCHING_CATEGORY:
        return NO_RECOMMENDATION
    if final_status_source == V3_STOCK_REFRESH_FAILED:
        return STOCK_REFRESH_FAILED_STATE
    return NO_RECOMMENDATION


def _compose_v3_full_category_quote_sync(
    *,
    text: str,
    resolved_request: Mapping[str, Any],
    matrix_package: Any,
    settings: LlmSettings,
) -> Any:
    llm_client = _create_llm_client(settings)
    try:
        return compose_full_category_quote(
            user_request=text,
            resolved_request=resolved_request,
            matrix_package=matrix_package,
            settings=settings,
            llm_client=llm_client,
        )
    finally:
        if llm_client is not None:
            llm_client.close()


def _create_llm_client(settings: LlmSettings) -> OpenAICompatibleLlmClient | None:
    if not (
        settings.llm_base_url.strip()
        and settings.llm_api_key.strip()
        and settings.llm_model.strip()
    ):
        return None

    return OpenAICompatibleLlmClient(
        settings,
        timeout_seconds=settings.llm_configurator_timeout_seconds,
        read_timeout_seconds=settings.llm_configurator_read_timeout_seconds,
        max_output_tokens=settings.llm_configurator_max_output_tokens,
    )


def _routing_failure_result(
    *,
    text: str,
    distributor_code: str,
    router_decision: V3RequestIntakeDecision,
    settings: LlmSettings,
) -> V3FullCategoryQuoteResult:
    if router_decision.status == "provider_not_configured":
        final_status_source = V3_PROVIDER_NOT_CONFIGURED
    elif router_decision.status == "provider_error":
        final_status_source = V3_PROVIDER_ERROR
    elif router_decision.status == "schema_error":
        final_status_source = V3_REQUEST_INTAKE_SCHEMA_VALIDATION_FAILED
    else:
        final_status_source = V3_CATEGORY_INTAKE_NO_MATCHING_CATEGORY

    report_json: dict[str, Any] = {
        "pipeline_version": V3_FULL_CATEGORY_MATRIX_MODE,
        "composer_mode": V3_REQUEST_INTAKE_MODE,
        "llm_configurator_used": router_decision.status != "provider_not_configured",
        "primary_recommendation_status": "no_recommendation",
        "final_status_source": final_status_source,
        "validated_quote": {},
        "primary_recommendation": {},
        "no_recommendation_reason": {
            "summary": router_decision.reason
            or "LLM could not select a target v3 category.",
            "fallback_reason": final_status_source,
        },
        "v3_llm_output": router_decision.raw_output or {},
        "v3_validation_errors": [],
        "v3_validation_warnings": [],
        "diagnostics": {
            "request_intake_used": True,
            "request_intake_decision": _routing_decision_json(router_decision),
            "resolved_request": _resolved_request_json(
                router_decision.resolved_request,
                text=text,
                profile=router_decision.profile,
                source="request_intake",
            ),
            "model": settings.llm_model,
            "distributor_code": distributor_code,
        },
        "llm_error_type": router_decision.error_type,
        "llm_http_status": router_decision.http_status,
        "v3_request_intake": _routing_decision_json(router_decision),
        "resolved_request": _resolved_request_json(
            router_decision.resolved_request,
            text=text,
            profile=router_decision.profile,
            source="request_intake",
        ),
        "v3_profile": None,
        "category_ids": [],
        "distributor_code": distributor_code,
        "source_text": text,
    }
    report_json["v3_result_state"] = v3_result_state(report_json)
    return V3FullCategoryQuoteResult(
        profile=None,
        category_ids=[],
        distributor_code=distributor_code,
        result_state=report_json["v3_result_state"],
        report_json=report_json,
    )


def _stock_refresh_failure_result(
    *,
    text: str,
    distributor_code: str,
    profile_name: str | None,
    root_category_ids: Sequence[str],
    resolved_category_ids: Sequence[str],
    resolved_request: Mapping[str, Any],
    stock_refresh_result: CategoryRefreshResult,
    settings: LlmSettings,
    intake_decision: V3RequestIntakeDecision | None,
) -> V3FullCategoryQuoteResult:
    reason = stock_refresh_result.error_message or (
        "Could not refresh selected distributor categories before matrix build."
    )
    report_json: dict[str, Any] = {
        "pipeline_version": V3_FULL_CATEGORY_MATRIX_MODE,
        "composer_mode": V3_REQUEST_INTAKE_MODE,
        "llm_configurator_used": False,
        "primary_recommendation_status": "no_recommendation",
        "final_status_source": V3_STOCK_REFRESH_FAILED,
        "validated_quote": {},
        "primary_recommendation": {},
        "no_recommendation_reason": {
            "summary": (
                "Selected distributor stock could not be refreshed before the "
                "paid v3 Composer call."
            ),
            "fallback_reason": V3_STOCK_REFRESH_FAILED,
            "details": reason,
        },
        "v3_llm_output": {},
        "v3_validation_errors": [],
        "v3_validation_warnings": [],
        "diagnostics": {
            "category_ids": list(resolved_category_ids),
            "root_category_ids": list(root_category_ids),
            "model": settings.llm_model,
            "distributor_code": distributor_code,
            "resolved_request": dict(resolved_request),
            "stock_refresh": {
                "enabled": True,
                **stock_refresh_result.to_diagnostics(),
            },
        },
        "resolved_request": dict(resolved_request),
        "v3_profile": profile_name,
        "category_ids": list(resolved_category_ids),
        "root_category_ids": list(root_category_ids),
        "distributor_code": distributor_code,
        "source_text": text,
    }
    if intake_decision is not None:
        report_json["v3_request_intake"] = _routing_decision_json(intake_decision)
        report_json["diagnostics"]["request_intake_used"] = True
        report_json["diagnostics"]["request_intake_decision"] = _routing_decision_json(
            intake_decision
        )
    report_json["v3_result_state"] = v3_result_state(report_json)
    return V3FullCategoryQuoteResult(
        profile=profile_name,
        category_ids=list(resolved_category_ids),
        distributor_code=distributor_code,
        result_state=report_json["v3_result_state"],
        report_json=report_json,
    )


def _parse_request_intake_output(
    raw_output: dict[str, Any],
    *,
    category_catalog: Sequence[Mapping[str, Any]],
    prompt_version: str | None = None,
    schema_version: str | None = None,
    canonical_input_hash: str | None = None,
    input_char_count: int | None = None,
) -> V3RequestIntakeDecision:
    status = str(raw_output.get("status") or "").strip()
    category_ids = _clean_category_ids(
        raw_output.get("category_ids") or raw_output.get("target_category_ids") or []
    )
    reason = str(raw_output.get("reason") or "").strip() or None
    resolved_request = _mapping_or_empty(raw_output.get("resolved_request"))
    if not category_ids:
        retrieval_plan = _mapping_or_empty(resolved_request.get("retrieval_plan"))
        category_ids = _clean_category_ids(
            [
                *_raw_sequence(retrieval_plan.get("anchor_category_ids")),
                *_raw_sequence(retrieval_plan.get("component_category_ids")),
                *_raw_sequence(retrieval_plan.get("fallback_category_ids")),
            ]
        )
    profile_name = _clean_intake_profile(
        raw_output.get("profile") or resolved_request.get("profile")
    )
    valid_category_ids = {
        str(entry.get("category_id") or "").strip()
        for entry in category_catalog
        if str(entry.get("category_id") or "").strip()
    }
    available_profiles = {
        str(entry.get("profile") or "").strip()
        for entry in _profile_catalog_for_intake(category_catalog=category_catalog)
    }
    if status == "selected_profile":
        unknown_category_ids = [
            category_id for category_id in category_ids if category_id not in valid_category_ids
        ]
        if (
            category_ids
            and not unknown_category_ids
            and (profile_name is None or profile_name not in available_profiles)
        ):
            resolved_request["profile"] = None
            resolved_request["target_category_ids"] = category_ids
            return V3RequestIntakeDecision(
                status="selected_category",
                category_ids=category_ids,
                reason=reason
                or "Profile is unavailable for this distributor; using selected categories.",
                resolved_request=resolved_request,
                raw_output=raw_output,
                prompt_version=prompt_version,
                schema_version=schema_version,
                canonical_input_hash=canonical_input_hash,
                input_char_count=input_char_count,
            )
        if profile_name is None:
            return V3RequestIntakeDecision(
                status="schema_error",
                reason="LLM request intake returned an unknown v3 product profile.",
                category_ids=category_ids,
                resolved_request=resolved_request,
                raw_output=raw_output,
                error_type="V3CategoryIntakeUnknownProfile",
                prompt_version=prompt_version,
                schema_version=schema_version,
                canonical_input_hash=canonical_input_hash,
                input_char_count=input_char_count,
            )
        if profile_name not in available_profiles:
            return V3RequestIntakeDecision(
                status="schema_error",
                reason=(
                    "LLM request intake returned a profile that is not available "
                    "for the supplied distributor category catalog."
                ),
                category_ids=category_ids,
                profile=profile_name,
                resolved_request=resolved_request,
                raw_output=raw_output,
                error_type="V3CategoryIntakeUnavailableProfile",
                prompt_version=prompt_version,
                schema_version=schema_version,
                canonical_input_hash=canonical_input_hash,
                input_char_count=input_char_count,
            )
        profile_category_ids = list(V3_FULL_CATEGORY_PROFILES[profile_name].category_ids)
        resolved_request["profile"] = profile_name
        resolved_request["target_category_ids"] = profile_category_ids
        return V3RequestIntakeDecision(
            status="selected_profile",
            profile=profile_name,
            category_ids=profile_category_ids,
            reason=reason,
            resolved_request=resolved_request,
            raw_output=raw_output,
            prompt_version=prompt_version,
            schema_version=schema_version,
            canonical_input_hash=canonical_input_hash,
            input_char_count=input_char_count,
        )
    if status == "selected_category" and category_ids:
        unknown_category_ids = [
            category_id for category_id in category_ids if category_id not in valid_category_ids
        ]
        if unknown_category_ids:
            return V3RequestIntakeDecision(
                status="schema_error",
                reason="LLM request intake returned category IDs outside the catalog.",
                category_ids=category_ids,
                resolved_request=resolved_request,
                raw_output=raw_output,
                error_type="V3CategoryIntakeUnknownCategoryIds",
                prompt_version=prompt_version,
                schema_version=schema_version,
                canonical_input_hash=canonical_input_hash,
                input_char_count=input_char_count,
            )
        resolved_request["profile"] = None
        resolved_request["target_category_ids"] = category_ids
        return V3RequestIntakeDecision(
            status="selected_category",
            category_ids=category_ids,
            reason=reason,
            resolved_request=resolved_request,
            raw_output=raw_output,
            prompt_version=prompt_version,
            schema_version=schema_version,
            canonical_input_hash=canonical_input_hash,
            input_char_count=input_char_count,
        )
    if status == "no_matching_category":
        resolved_request["profile"] = None
        return V3RequestIntakeDecision(
            status="no_matching_category",
            reason=reason or "The request does not match available v3 categories.",
            resolved_request=resolved_request,
            raw_output=raw_output,
            prompt_version=prompt_version,
            schema_version=schema_version,
            canonical_input_hash=canonical_input_hash,
            input_char_count=input_char_count,
        )
    return V3RequestIntakeDecision(
        status="schema_error",
        reason="LLM request intake returned an invalid response.",
        resolved_request=resolved_request,
        raw_output=raw_output,
        error_type="V3RequestIntakeSchemaError",
        prompt_version=prompt_version,
        schema_version=schema_version,
        canonical_input_hash=canonical_input_hash,
        input_char_count=input_char_count,
    )


def _routing_decision_json(decision: V3RequestIntakeDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "profile": decision.profile,
        "category_ids": decision.category_ids,
        "reason": decision.reason,
        "resolved_request": decision.resolved_request,
        "raw_output": decision.raw_output or {},
        "error_type": decision.error_type,
        "http_status": decision.http_status,
        "prompt_version": decision.prompt_version,
        "schema_version": decision.schema_version,
        "canonical_input_hash": decision.canonical_input_hash,
        "input_char_count": decision.input_char_count,
    }


def _request_intake_prompt_version(settings: LlmSettings) -> str:
    version = str(settings.v3_full_category_contract_version or "").strip().lower()
    if version in {"v7_1", "v7.1", "7.1"}:
        return REQUEST_INTAKE_PROMPT_VERSION_V7_1
    return REQUEST_INTAKE_PROMPT_VERSION_V7 if version == "v7" else "request_intake_v6"


def _request_intake_system_prompt(settings: LlmSettings) -> str:
    if _request_intake_prompt_version(settings) in {
        REQUEST_INTAKE_PROMPT_VERSION_V7,
        REQUEST_INTAKE_PROMPT_VERSION_V7_1,
    }:
        return _request_intake_system_prompt_v7()
    return _request_intake_system_prompt_v6()


def _request_intake_user_prompt(
    text: str,
    *,
    category_catalog: Sequence[Mapping[str, Any]],
    settings: LlmSettings,
) -> str:
    prompt_version = _request_intake_prompt_version(settings)
    if prompt_version in {
        REQUEST_INTAKE_PROMPT_VERSION_V7,
        REQUEST_INTAKE_PROMPT_VERSION_V7_1,
    }:
        return _request_intake_user_prompt_v7(
            text,
            category_catalog=category_catalog,
            prompt_version=prompt_version,
        )
    return _request_intake_user_prompt_v6(
        text,
        category_catalog=category_catalog,
    )


def _request_intake_system_prompt_v7() -> str:
    return """
You are Stock Configurator Request Intake v7.1.

Responsibility:
- Understand the customer's procurement task.
- Select one existing product profile when a supplied profile covers the task.
- Otherwise select only catalog category IDs supplied in the payload.
- Build one structured requirement ledger.

Forbidden in intake:
- Do not choose products, SKUs, stock rows, prices or brands not written by the user.
- Do not build a BOM.
- Do not prove compatibility.
- Do not duplicate one requirement into hard/soft/target arrays.
- Do not narrow the future matrix to a product shortlist.

Profile rule:
- A profile is only a transport preset for the full matrix.
- If one profile covers the functional class, return selected_profile and that profile.
- Do not manually assemble a smaller category set when a profile covers the class.
- Use selected_category only when no profile fits.

Request mode:
- Detailed specification does not mean exact-only.
- Default request_mode is best_available.
- allow_partial_offer is true by default.
- Named vendor, model, generation, part number and ecosystem are targets or
  preferences unless the user explicitly says only, strictly, no analogs,
  do not replace, exactly this SKU/model, or otherwise forbids substitution.
- Local strict wording applies only to the related constraint.

Ledger contract:
- resolved_request must contain objects[].requested_items[].constraints[].
- requested_item is user intent, not a recommended catalog row.
- Use item_kind from: primary_product, component, accessory, feature,
  consumable, license_service, spare, other.
- role is a short natural-language role, not a closed server enum.
- quantity is stored once on requested_item.
- quantity_basis is one of: total, per_primary_unit, spare_total,
  not_applicable.
- For explicit quantity, set quantity_requirement_id on the item. Do not
  duplicate the same quantity as a constraint.
- Every explicit fact becomes exactly one atomic constraint with:
  requirement_id, dimension, operator, value_text, value_number only when
  explicit, unit, strictness, substitution, source_phrase.
- strictness values: locked, core, target, preference.
- substitution values: forbidden, equivalent_only, downgrade_allowed.
- Operational quantity and spare quantity are separate requested_items.
- Multi-product requests use multiple objects or multiple requested_items.

Classification:
- locked means the user explicitly forbids violation of that exact boundary.
- core means product class or business task.
- target means measurable requested property, quantity-independent option,
  interface, protocol, speed, capacity or component role.
- preference means brand, series, generation, model, ecosystem, appearance or
  optional feature when substitution is not forbidden.
- Do not invent numeric limits from vague words.
- Preserve compound lines: count, per-unit capacity, total capacity, interface,
  form factor, port count, speed, operational/spare role and source phrase.

Retrieval plan:
- For selected_profile, category_ids may be empty; code will expand the profile.
- For selected_category, use only supplied category IDs.
- retrieval_plan may contain anchor_category_ids, component_category_ids,
  fallback_category_ids and retrieval_intents. These are transport groups, not
  product recommendations.
- For every object return object_quantity, primary_item_id and anchor_policy.
- anchor_policy values:
  required: complete_system, configured_system and the main configured device in
  multi_product_solution.
  self: standalone_product.
  not_required: accessory, replacement_component or expansion_or_upgrade when
  the base device is not supplied.
- Return retrieval_plan.objects[] with object_id, anchor_category_ids,
  component_category_ids and fallback_category_ids. These are category transport
  scopes only, never SKU recommendations.
- Optimize for recall of the functional class, not for the smallest matrix.

Output language:
- All natural-language fields in Russian.
- Keep technical names, interfaces, standards and part numbers as written.
- Return JSON only.

Return shape:
{
  "status": "selected_profile | selected_category | no_matching_category",
  "profile": "profile name or null",
  "category_ids": ["category_id"],
  "reason": "short Russian reason",
  "resolved_request": {
    "schema_version": "resolved_request_schema_v7_1",
    "request_mode": "best_available",
    "allow_partial_offer": true,
    "price_objective": "lowest_price_after_best_fit",
    "customer_task_summary": "...",
    "objects": [
      {
        "object_id": "O1",
        "functional_class": "...",
        "deliverable_scope": "configured_system",
        "object_quantity": 1,
        "primary_item_id": "I1",
        "anchor_policy": "required",
        "requested_items": [
          {
            "item_id": "I1",
            "item_kind": "primary_product",
            "role": "...",
            "supply_class": "operational",
            "quantity": 1,
            "quantity_basis": "total",
            "quantity_requirement_id": "R1",
            "partial_quantity_allowed": true,
            "source_phrase": "...",
            "constraints": [
              {
                "requirement_id": "R2",
                "dimension": "vendor_or_model",
                "operator": "semantic_match",
                "value_text": "...",
                "value_number": null,
                "unit": null,
                "strictness": "preference",
                "substitution": "downgrade_allowed",
                "source_phrase": "..."
              }
            ]
          }
        ]
      }
    ],
    "unknowns": [],
    "retrieval_plan": {
      "objects": [
        {
          "object_id": "O1",
          "anchor_category_ids": [],
          "component_category_ids": [],
          "fallback_category_ids": []
        }
      ],
      "anchor_category_ids": [],
      "component_category_ids": [],
      "fallback_category_ids": [],
      "retrieval_intents": []
    }
  }
}
""".strip()


def _request_intake_system_prompt_v6() -> str:
    return """
You are Stock Configurator Request Intake v6.

Your job is not to recommend products or SKUs. Your job is to understand the
customer's functional procurement task, create one structured requirement
ledger, determine whether the request is exact-only or best-available, and
select a matrix retrieval scope broad enough to contain complete, downgraded
and partial analogs. The next step will include every stocked/priced product
from each selected category and all of its subcategories, grouped by those
distributor subcategories.

Write all natural-language fields in Russian. Keep technical model names,
vendor names, interfaces and part numbers as written by the user or catalog.

Do not recommend products. Do not choose SKUs. Do not invent category IDs. Do
not mention stock_row_id, component_candidate_id, part numbers or brands unless
the user explicitly wrote them. Do not narrow the future matrix.

First check the supplied profile_catalog. If the request matches one known
product profile listed there, return status="selected_profile" and that profile
name. This is preferred for configurable product groups such as servers,
storage systems, NAS, switches or other groups represented in profile_catalog.
A profile is a transport preset for the full stocked matrix, not a SKU choice
and not a technical recommendation. If the product group is clear but the
matching profile is not listed in profile_catalog, use status="selected_category"
with real category_ids from category_catalog instead.

Use status="selected_category" with raw category_ids only when no supplied
profile covers the request. Choose category_ids only from the supplied
category_catalog. Do not optimize for the smallest possible category set;
optimize for recall of anchor products, normal configurable components,
enablement items and same-function substitute categories when substitution is
allowed. If the request specifies a
configurable bundle with base device plus CPU, memory, drives, adapters,
licenses, modules, optics, power or other add-ons, do not try to manually build
the component category set if a supplied profile already covers that product
group. If no profile covers it, include the visible component categories too.
Do not assume a ready device category covers sibling component categories
unless they are descendants. Prefer specific child/component categories over
broad top-level roots when those child categories are visible in the catalog.
Do not choose a top-level root like all servers, all storage or all network
unless the user truly asks for the whole root group.
If the request needs product groups from more than one root category and no
profile covers it, choose all needed target categories. If no profile or
category in the catalog matches the request, return status="no_matching_category".

Build resolved_request as one requirement ledger, not as a BOM.
The primary v6 ledger is requirements[]. Keep backward-compatible arrays
non_negotiable_requirements, targets, preferences, hard_requirements and
soft_preferences only as derived aliases for older code and reports.
Classify explicit request facts into:
- locked: only explicit no-substitution/no-violation wording, safety/legal
  boundaries, explicitly hard installation boundaries, or strict budget
  ceilings.
- core: requested product class or functional task.
- important: requested quantity, capacity, performance, interface, component,
  option or feature that should be covered but may be short, downgraded or
  omitted in a partial offer unless explicitly locked.
- preference: vendor, family, model, generation, ecosystem, appearance or
  optional feature when not explicitly locked.

Legacy aliases:
- non_negotiable_requirements: exact-only/no-analog wording, "только",
  "обязательно", "строго", "без аналогов" and strict blockers only.
- targets: core and important requested characteristics that should be matched where possible but
  can be substituted when substitution_policy permits it.
- preferences: preferred vendor, model, generation, ecosystem, color, series or
  part number when exact-only wording is absent.
Put unclear details into unknowns_or_missing_facts instead of guessing. Keep
source_phrase short so Composer can trace each requirement back to the input.
For quantity requirements, keep source_phrase neutral and free of preferred
vendor/model wording. Example: use "1 server" as the non_negotiable quantity
source, and put the named server model into preferences/targets separately.

If the user provides a concrete characteristics set, preserve each explicit
number, unit, workload, capacity, form factor, warehouse, budget or vendor
constraint as a separate ledger item. Do not silently delete or rewrite explicit
requirements in request intake. Do not turn vague words like "fast", "normal"
or "powerful" into invented numeric thresholds. Do not turn a named brand,
product line, product model, generation or part number into non_negotiable
merely because it appears in a spec-like phrase such as
"<model> in the following configuration". Unless the user explicitly forbids
substitutes, those names belong in preferences and the measurable technical
minima belong in targets or non_negotiable only when the wording makes them so.
For compound component lines, preserve all explicit attributes instead of
collapsing the line to one number. Examples:
- "8 x SSD 1.92TB SAS 2.5" must preserve count=8, role=operational storage,
  capacity=1.92TB per drive, interface=SAS and form_factor=2.5".
- "4 x SSD 1.92TB SAS 2.5 (ZIP/spare)" must preserve count=4 and role=spare.
- "8 x 16GB DDR5 RDIMM 4800MHz" must preserve module_count=8,
  per_module_capacity=16GB, total_capacity=128GB, memory_type=DDR5 RDIMM and
  speed=4800MHz.
When useful, put these parsed attributes into the requirement object under an
attributes map while keeping requested and source_phrase human-readable. If a
line contains interface/protocol/form_factor/count, do not encode it as
capacity-only.

Default integrator substitution policy:
- procurement_mode is "best_available" by default.
- allow_partial_quote is true by default.
- allow_partial_quantity is true by default unless the user explicitly rejects
  smaller partial supply.
- If the user names a concrete model/generation, CPU, drive, adapter, vendor or
  part number but does not say "strictly this exact model", "only this model",
  "no analogs", "do not substitute" or similar, treat the name as a preferred
  reference, not a hard exact-SKU requirement. This applies both to the primary
  procurement object and to components inside the requested configuration.
- In that default case, put the exact model/generation name into
  preferences or compatibility_attention_points and record the underlying
  measurable requirements as targets: capacity, cores, memory amount,
  interface, form factor, port count, redundancy, warehouse and other technical
  targets. Use non_negotiable_requirements only for explicit no-violation
  constraints.
- Requested licenses, vendor options, rail kits, adapters, controllers,
  interfaces and protocols are targets by default unless the text explicitly
  forbids substitutes or states a no-violation boundary. Do not classify them
  as non_negotiable merely because they appear in a detailed customer spec.
- In that default case, assumptions_allowed_for_draft should explicitly say
  that technically equivalent or better stocked substitutes are allowed for
  preferred model/vendor/part references. If no equal-or-better stocked option
  exists, the Composer may propose the closest coherent stocked analog only
  with explicit deviation notes and approval impact.
- In that default case, compatibility_attention_points must not anchor the
  whole future BOM to the named preferred model. Phrase them around the selected
  platform/system/component that Composer will choose from the matrix.
- In that default case, non_goals must not forbid analogs, substitutes or other
  vendors unless the original user text explicitly forbids them.
- If the user explicitly forbids analogs or requires the exact SKU/model, then
  record exact_model as a hard requirement with source_phrase.
- This is not product selection. It only tells the next Composer whether
  technically equivalent or better stocked substitutes, or the closest coherent
  analog with disclosed deviations, are allowed.

Exact-only policy:
- procurement_mode="exact_only" only when the user explicitly says words
  equivalent to "только", "строго", "без аналогов", "не заменять",
  "точно этот SKU" or "иначе не предлагать".
- Do not infer exact_only merely from a detailed specification.
- A shortage of quantity, memory, drives, licenses, modules, adapters or other
  requested items does not cancel matrix retrieval in best_available mode. The
  Composer may later produce a partial stock offer and list the missing items.

Use objective="cheapest_minimum_viable" by default because v3 is meant to quote
the lowest-price technically workable minimum spec, not a premium configuration.

Important: resolved_request is not a BOM and not a product shortlist. It is
only a clarified reading of the customer request. The later Composer must still
study the complete category/subcategory product matrix and choose from supplied
products and stock rows only.

Return only JSON:
{
  "status": "selected_profile" | "selected_category" | "no_matching_category",
  "profile": "profile name or null",
  "category_ids": ["category_id"],
  "reason": "short reason",
  "resolved_request": {
    "objective": "cheapest_minimum_viable",
    "procurement_mode": "best_available",
    "allow_partial_quote": true,
    "allow_partial_quantity": true,
    "price_policy": "lowest_price_after_best_fit",
    "profile": "profile name or null",
    "target_category_ids": ["category_id"],
    "customer_task_summary": "one sentence",
    "deliverable_scope": "complete_system",
    "requested_quantity": 1,
    "substitution_policy": "allowed_with_disclosed_downgrade",
    "objects": [
      {
        "object_id": "O1",
        "functional_class": "...",
        "deliverable_scope": "complete_system",
        "requested_quantity": 1,
        "allow_partial_quantity": true
      }
    ],
    "requirements": [
      {
        "id": "R1",
        "object_id": "O1",
        "dimension": "product_type",
        "requested": "...",
        "attributes": {},
        "comparison": "semantic",
        "value": null,
        "unit": null,
        "priority": "core",
        "substitution": "degrade_allowed",
        "source_phrase": "..."
      }
    ],
    "unknowns": [],
    "retrieval_plan": {
      "anchor_category_ids": [],
      "component_category_ids": [],
      "fallback_category_ids": []
    },
    "non_negotiable_requirements": [
      {
        "requirement_id": "R1",
        "key": "quantity",
        "value": 1,
        "unit": "pcs",
        "source_phrase": "1 сервер",
        "explicit": true
      }
    ],
    "targets": [
      {
        "requirement_id": "R2",
        "key": "cpu.cores",
        "value": 32,
        "unit": "cores",
        "source_phrase": "32 ядра",
        "explicit": true
      }
    ],
    "preferences": [],
    "hard_requirements": [],
    "soft_preferences": [],
    "unknowns_or_missing_facts": [],
    "assumptions_allowed_for_draft": [],
    "fulfillment_policy": "exact -> equivalent -> disclosed analog -> no_recommendation",
    "compatibility_attention_points": [],
    "non_goals": []
  }
}
""".strip()


def _request_intake_user_prompt_v7(
    text: str,
    *,
    category_catalog: Sequence[Mapping[str, Any]],
    prompt_version: str = REQUEST_INTAKE_PROMPT_VERSION_V7,
) -> str:
    schema_version = (
        RESOLVED_REQUEST_SCHEMA_VERSION_V7_1
        if prompt_version == REQUEST_INTAKE_PROMPT_VERSION_V7_1
        else RESOLVED_REQUEST_SCHEMA_VERSION_V7
    )
    return _stable_json(
        {
            "TASK_CAPSULE": (
                "Request Intake v7.1: choose profile/category transport scope, "
                "build one ledger and object anchor policy only."
            ),
            "user_request": text.strip(),
            "prompt_version": prompt_version,
            "resolved_request_schema_version": schema_version,
            "intake_policy": {
                "product_selection": "forbidden",
                "bom_selection": "forbidden",
                "sku_selection": "forbidden",
                "profile_first": True,
                "single_ledger": True,
                "legacy_hard_soft_arrays": "do_not_return",
                "default_request_mode": "best_available",
                "default_allow_partial_offer": True,
                "named_models_are_locked_only_when_explicit": True,
                "object_anchor_policy_required": True,
                "object_retrieval_plan_required": True,
            },
            "profile_catalog": _profile_catalog_for_intake(
                category_catalog=category_catalog
            ),
            "category_catalog": list(category_catalog),
        }
    )


def _request_intake_user_prompt_v6(
    text: str,
    *,
    category_catalog: Sequence[Mapping[str, Any]],
) -> str:
    return _stable_json(
        {
            "user_request": text.strip(),
            "category_catalog_policy": {
                "selection_scope": (
                    "prefer known product profile; use target category IDs as fallback"
                ),
                "profile_selection": (
                    "if one profile in profile_catalog covers the request, return "
                    "selected_profile with that profile"
                ),
                "matrix_expansion_after_selection": (
                    "code will map selected profiles to category IDs, then include "
                    "selected categories and all descendants"
                ),
                "product_selection": "do not choose products in request intake",
                "natural_language_output": "ru",
                "substitution_policy": (
                    "named models, generations and part numbers are preferred "
                    "references unless the user explicitly forbids analogs or "
                    "requires an exact model; if exact fulfillment is unavailable, "
                    "Composer may quote the best available stocked analog with "
                    "explicit structured deviation_notes"
                ),
                "v6_retrieval_plan": (
                    "return anchor_category_ids, component_category_ids and "
                    "fallback_category_ids inside resolved_request.retrieval_plan "
                    "when useful; these are matrix transport groups, not SKU "
                    "recommendations"
                ),
                "partial_quote": (
                    "best_available mode allows a partial stocked offer with "
                    "visible gaps unless the user explicitly rejects analogs or "
                    "partial supply"
                ),
            },
            "profile_catalog": _profile_catalog_for_intake(
                category_catalog=category_catalog
            ),
            "category_catalog": list(category_catalog),
        }
    )


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _resolved_request_json(
    raw_resolved_request: Mapping[str, Any],
    *,
    text: str,
    profile: str | None,
    source: str,
) -> dict[str, Any]:
    resolved = _mapping_or_empty(raw_resolved_request)
    if _is_v7_resolved_request(resolved):
        resolved.setdefault("schema_version", RESOLVED_REQUEST_SCHEMA_VERSION_V7_1)
        resolved.setdefault("request_mode", "best_available")
        resolved.setdefault("allow_partial_offer", True)
        resolved.setdefault("price_objective", "lowest_price_after_best_fit")
        resolved.setdefault("customer_task_summary", text.strip())
        resolved.setdefault("unknowns", [])
        resolved["objects"] = _normalize_v7_1_objects(resolved)
        resolved.setdefault(
            "retrieval_plan",
            {
                "anchor_category_ids": [],
                "component_category_ids": list(resolved.get("target_category_ids") or []),
                "fallback_category_ids": [],
                "retrieval_intents": [],
                "objects": _default_retrieval_plan_objects(resolved),
            },
        )
        resolved["retrieval_plan"] = _normalize_v7_1_retrieval_plan(resolved)
        resolved["profile"] = profile
        resolved["source"] = source
        return resolved

    resolved.setdefault("objective", "cheapest_minimum_viable")
    resolved["profile"] = profile
    resolved.setdefault("customer_task_summary", text.strip())
    resolved.setdefault("deliverable_scope", "multi_product_solution")
    resolved.setdefault("requested_quantity", 1)
    resolved.setdefault("substitution_policy", "allowed_with_disclosed_downgrade")
    exact_only = (
        str(resolved.get("procurement_mode") or "").strip() == "exact_only"
        or str(resolved.get("substitution_policy") or "").strip() == "forbidden"
    )
    resolved.setdefault("procurement_mode", "exact_only" if exact_only else "best_available")
    resolved.setdefault("allow_partial_quote", not exact_only)
    resolved.setdefault("allow_partial_quantity", not exact_only)
    resolved.setdefault("price_policy", "lowest_price_after_best_fit")
    resolved.setdefault(
        "objects",
        [
            {
                "object_id": "O1",
                "functional_class": str(profile or resolved.get("deliverable_scope") or "product"),
                "deliverable_scope": resolved.get("deliverable_scope"),
                "requested_quantity": resolved.get("requested_quantity"),
                "object_quantity": resolved.get("requested_quantity") or 1,
                "primary_item_id": "I1",
                "anchor_policy": _anchor_policy_for_scope(
                    str(resolved.get("deliverable_scope") or "multi_product_solution")
                ),
                "allow_partial_quantity": resolved.get("allow_partial_quantity"),
            }
        ],
    )
    resolved["objects"] = _normalize_v7_1_objects(resolved)
    resolved.setdefault("requirements", _requirements_from_legacy_request(resolved))
    resolved.setdefault("unknowns", list(resolved.get("unknowns_or_missing_facts") or []))
    resolved.setdefault(
        "retrieval_plan",
        {
            "anchor_category_ids": [],
            "component_category_ids": list(resolved.get("target_category_ids") or []),
            "fallback_category_ids": [],
            "objects": _default_retrieval_plan_objects(resolved),
        },
    )
    resolved["retrieval_plan"] = _normalize_v7_1_retrieval_plan(resolved)
    resolved.setdefault("non_negotiable_requirements", [])
    resolved.setdefault("targets", list(resolved.get("hard_requirements") or []))
    resolved.setdefault("preferences", list(resolved.get("soft_preferences") or []))
    resolved.setdefault("hard_requirements", [])
    resolved.setdefault("soft_preferences", [])
    resolved.setdefault("unknowns_or_missing_facts", [])
    resolved.setdefault("assumptions_allowed_for_draft", [])
    resolved.setdefault(
        "fulfillment_policy",
        (
            "Сначала искать точное закрытие ТЗ. Если точного закрытия нет и "
            "аналоги не запрещены явно, можно предложить лучший доступный "
            "складской аналог с явным описанием отклонений."
        ),
    )
    resolved.setdefault("compatibility_attention_points", [])
    resolved.setdefault("non_goals", [])
    resolved["source"] = source
    return resolved


def _is_v7_resolved_request(resolved: Mapping[str, Any]) -> bool:
    if str(resolved.get("schema_version") or "") in {
        RESOLVED_REQUEST_SCHEMA_VERSION_V7,
        RESOLVED_REQUEST_SCHEMA_VERSION_V7_1,
    }:
        return True
    objects = resolved.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes, bytearray)):
        return False
    return any(
        isinstance(item, Mapping) and isinstance(item.get("requested_items"), Sequence)
        for item in objects
    )


def _normalize_v7_1_objects(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_objects = resolved.get("objects")
    if not isinstance(raw_objects, Sequence) or isinstance(
        raw_objects,
        (str, bytes, bytearray),
    ):
        raw_objects = []
    result: list[dict[str, Any]] = []
    root_scope = str(resolved.get("deliverable_scope") or "").strip()
    for index, raw_object in enumerate(raw_objects, start=1):
        if not isinstance(raw_object, Mapping):
            continue
        item = dict(raw_object)
        object_id = str(item.get("object_id") or f"O{index}").strip() or f"O{index}"
        deliverable_scope = str(
            item.get("deliverable_scope") or root_scope or "standalone_product"
        ).strip()
        primary_item_id = str(item.get("primary_item_id") or "").strip()
        if not primary_item_id:
            primary_item_id = _primary_item_id_from_requested_items(
                item.get("requested_items")
            )
        item["object_id"] = object_id
        item["deliverable_scope"] = deliverable_scope
        item["object_quantity"] = _int_or_default(
            item.get("object_quantity")
            or item.get("requested_quantity")
            or resolved.get("requested_quantity"),
            default=1,
        )
        item["primary_item_id"] = primary_item_id
        item["anchor_policy"] = str(item.get("anchor_policy") or "").strip() or (
            _anchor_policy_for_scope(deliverable_scope)
        )
        result.append(item)
    if result:
        return result
    return [
        {
            "object_id": "O1",
            "functional_class": str(resolved.get("profile") or root_scope or "product"),
            "deliverable_scope": root_scope or "standalone_product",
            "object_quantity": _int_or_default(
                resolved.get("requested_quantity"),
                default=1,
            ),
            "primary_item_id": "I1",
            "anchor_policy": _anchor_policy_for_scope(root_scope or "standalone_product"),
            "requested_items": [],
        }
    ]


def _primary_item_id_from_requested_items(value: Any) -> str:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ""
    first_item_id = ""
    for raw_item in value:
        if not isinstance(raw_item, Mapping):
            continue
        item_id = str(raw_item.get("item_id") or "").strip()
        if item_id and not first_item_id:
            first_item_id = item_id
        if item_id and str(raw_item.get("item_kind") or "").strip() == "primary_product":
            return item_id
    return first_item_id


def _normalize_v7_1_retrieval_plan(resolved: Mapping[str, Any]) -> dict[str, Any]:
    retrieval_plan = _mapping_or_empty(resolved.get("retrieval_plan"))
    retrieval_plan.setdefault("anchor_category_ids", [])
    retrieval_plan.setdefault(
        "component_category_ids",
        list(resolved.get("target_category_ids") or []),
    )
    retrieval_plan.setdefault("fallback_category_ids", [])
    retrieval_plan.setdefault("retrieval_intents", [])
    retrieval_plan["objects"] = _default_retrieval_plan_objects(resolved)
    return retrieval_plan


def _default_retrieval_plan_objects(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    retrieval_plan = _mapping_or_empty(resolved.get("retrieval_plan"))
    raw_object_plans = retrieval_plan.get("objects")
    existing_by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_object_plans, Sequence) and not isinstance(
        raw_object_plans,
        (str, bytes, bytearray),
    ):
        for raw_item in raw_object_plans:
            if not isinstance(raw_item, Mapping):
                continue
            object_id = str(raw_item.get("object_id") or "").strip()
            if object_id:
                existing_by_id[object_id] = raw_item
    root_anchor_ids = list(retrieval_plan.get("anchor_category_ids") or [])
    root_component_ids = list(retrieval_plan.get("component_category_ids") or [])
    root_fallback_ids = list(retrieval_plan.get("fallback_category_ids") or [])
    result: list[dict[str, Any]] = []
    for raw_object in resolved.get("objects") or []:
        if not isinstance(raw_object, Mapping):
            continue
        object_id = str(raw_object.get("object_id") or "").strip()
        if not object_id:
            continue
        existing = existing_by_id.get(object_id, {})
        result.append(
            {
                "object_id": object_id,
                "anchor_category_ids": list(
                    existing.get("anchor_category_ids") or root_anchor_ids
                ),
                "component_category_ids": list(
                    existing.get("component_category_ids") or root_component_ids
                ),
                "fallback_category_ids": list(
                    existing.get("fallback_category_ids") or root_fallback_ids
                ),
            }
        )
    return result


def _anchor_policy_for_scope(scope: str) -> str:
    normalized = str(scope or "").strip()
    if normalized in {"complete_system", "configured_system", "multi_product_solution"}:
        return "required"
    if normalized == "standalone_product":
        return "self"
    if normalized in {"accessory", "replacement_component", "expansion_or_upgrade"}:
        return "not_required"
    return "required"


def _int_or_default(value: Any, *, default: int) -> int:
    try:
        if isinstance(value, bool) or value in (None, ""):
            return default
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _requirements_from_legacy_request(resolved: Mapping[str, Any]) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    legacy_groups = [
        ("non_negotiable_requirements", "locked", "forbidden"),
        ("hard_requirements", "important", "degrade_allowed"),
        ("targets", "important", "degrade_allowed"),
        ("preferences", "preference", "omission_allowed"),
        ("soft_preferences", "preference", "omission_allowed"),
    ]
    seen: set[str] = set()
    for field_name, priority, substitution in legacy_groups:
        raw_items = resolved.get(field_name) or []
        if isinstance(raw_items, Mapping):
            raw_items = [raw_items]
        if isinstance(raw_items, str):
            raw_items = [raw_items]
        if not isinstance(raw_items, Sequence):
            continue
        for raw_item in raw_items:
            item = _mapping_or_empty(raw_item) if isinstance(raw_item, Mapping) else {}
            requested = (
                str(
                    item.get("source_phrase")
                    or item.get("requested")
                    or item.get("key")
                    or raw_item
                )
                .strip()
            )
            if not requested or requested in seen:
                continue
            seen.add(requested)
            value = item.get("value")
            requirements.append(
                {
                    "id": item.get("requirement_id") or f"R{len(requirements) + 1}",
                    "object_id": item.get("object_id") or "O1",
                    "dimension": str(item.get("key") or item.get("dimension") or "other"),
                    "requested": requested,
                    "comparison": item.get("comparison") or "semantic",
                    "value": value if value != "" else None,
                    "unit": item.get("unit"),
                    "priority": priority,
                    "substitution": substitution,
                    "source_phrase": str(item.get("source_phrase") or requested),
                }
            )
    if not requirements:
        requirements.append(
            {
                "id": "R1",
                "object_id": "O1",
                "dimension": "product_type",
                "requested": str(resolved.get("customer_task_summary") or "Запрос клиента"),
                "comparison": "semantic",
                "value": None,
                "unit": None,
                "priority": "core",
                "substitution": "degrade_allowed",
                "source_phrase": str(resolved.get("customer_task_summary") or ""),
            }
        )
    return requirements


def _category_catalog_for_intake(categories: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "category_id": str(category.category_id or "").strip(),
            "parent_category_id": _string_or_none(category.parent_category_id),
            "name": str(category.name or "").strip(),
            "level": int(category.level or 0),
            "path": _category_path_text(category.path_json, fallback=category.name),
            "enabled_for_sync": bool(category.enabled_for_sync),
        }
        for category in categories
        if str(category.category_id or "").strip()
    ]


def _profile_catalog_for_intake(
    *,
    category_catalog: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    available_category_ids = {
        str(entry.get("category_id") or "").strip()
        for entry in category_catalog
        if str(entry.get("category_id") or "").strip()
    }
    return [
        {
            "profile": profile.name,
            "description": profile.description,
            "category_ids": list(profile.category_ids),
            "profile_name": profile.name,
            "anchor_category_ids": [],
            "component_category_ids": list(profile.category_ids),
            "fallback_category_ids": [],
        }
        for profile in V3_FULL_CATEGORY_PROFILES.values()
        if set(profile.category_ids).issubset(available_category_ids)
    ]


def _category_path_text(value: Any, *, fallback: str | None = None) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                for key in ("name", "title", "category_name", "id", "category_id"):
                    text = str(item.get(key) or "").strip()
                    if text:
                        parts.append(text)
                        break
            else:
                text = str(item or "").strip()
                if text:
                    parts.append(text)
        if parts:
            return " / ".join(parts)
    return str(fallback or "").strip()


def _clean_category_ids(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [
        category_id
        for category_id in dict.fromkeys(str(item or "").strip() for item in value)
        if category_id
    ]


def _clean_intake_profile(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    profile_name = V3_FULL_CATEGORY_PROFILE_ALIASES.get(text, text)
    if profile_name in V3_FULL_CATEGORY_PROFILES:
        return profile_name
    return None


def _raw_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _string_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _stable_json(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "MECHANICAL_VALIDATION_FAILED",
    "MATRIX_TOO_LARGE_FOR_MODEL_STATE",
    "MATRIX_EMPTY_AFTER_CATEGORY_SELECTION_STATE",
    "NO_RECOMMENDATION",
    "PROVIDER_ERROR",
    "PROVIDER_NOT_CONFIGURED",
    "QUOTE_CANDIDATE_CUSTOMER_READY",
    "QUOTE_DRAFT_REVIEW_REQUIRED",
    "SCHEMA_VALIDATION_FAILED",
    "V3_FULL_CATEGORY_MATRIX_MODE",
    "V3_REQUEST_INTAKE_MODE",
    "V3FullCategoryQuoteResult",
    "V3RequestIntakeDecision",
    "route_v3_full_category_target",
    "run_v3_full_category_quote",
    "v3_result_state",
]
