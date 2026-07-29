from __future__ import annotations

import asyncio
import copy
import hashlib
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.category_repository import CategoryRepository
from app.catalog.product_repository import ProductRepository
from app.core.config import LlmSettings, get_llm_settings
from app.distributors.category_refresh import refresh_distributor_categories
from app.llm.base import LlmClient, LlmError, LlmHttpError
from app.llm.openai_compatible import OpenAICompatibleLlmClient
from app.llm.simple_stock_composer import (
    SIMPLE_STOCK_MECHANICAL_ERROR,
    SIMPLE_STOCK_NO_RECOMMENDATION,
    SIMPLE_STOCK_QUOTE_ACCEPTED,
    SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
    compose_simple_stock_quote,
)
from app.matching.full_category_matrix import (
    MATRIX_EMPTY_AFTER_CATEGORY_SELECTION,
    MATRIX_TOO_LARGE_FOR_MODEL,
)
from app.matching.simple_stock_matrix import build_simple_stock_matrix_group_package
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
from app.matching.v3_full_category_quote_service import (
    MATRIX_EMPTY_AFTER_CATEGORY_SELECTION_STATE,
    MATRIX_TOO_LARGE_FOR_MODEL_STATE,
    NO_RECOMMENDATION,
    PROVIDER_ERROR,
    PROVIDER_NOT_CONFIGURED,
    QUOTE_CANDIDATE_CUSTOMER_READY,
    QUOTE_DRAFT_REVIEW_REQUIRED,
    SCHEMA_VALIDATION_FAILED,
    STOCK_REFRESH_FAILED_STATE,
    V3FullCategoryQuoteResult,
)

SIMPLE_STOCK_ROUTE_MODE = "simple_stock_route"
SIMPLE_STOCK_ROUTE_PROMPT_VERSION = "simple_stock_route_v19"
SIMPLE_STOCK_ROUTE_MAX_CATEGORY_IDS = 12
SIMPLE_STOCK_ROUTE_MAX_TOTAL_SUBTREE_POSITIONS = 500
SIMPLE_STOCK_ROUTE_CACHE_MAX_SIZE = 256
SIMPLE_STOCK_ROUTE_SCHEMA_FAILED = "simple_stock_route_schema_failed"
SIMPLE_STOCK_ROUTE_NO_MATCHING_CATEGORY = "simple_stock_route_no_matching_category"
SIMPLE_STOCK_ROUTE_PROVIDER_NOT_CONFIGURED = "simple_stock_route_provider_not_configured"
SIMPLE_STOCK_ROUTE_PROVIDER_ERROR = "simple_stock_route_provider_error"
SIMPLE_STOCK_REFRESH_FAILED = "simple_stock_refresh_failed"

_SIMPLE_STOCK_ROUTE_CACHE: OrderedDict[str, SimpleStockRouteDecision] = OrderedDict()
_SIMPLE_STOCK_ROUTE_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class SimpleStockRouteDecision:
    status: str
    profile: str | None = None
    category_ids: list[str] = field(default_factory=list)
    reason: str | None = None
    resolved_request: dict[str, Any] = field(default_factory=dict)
    raw_output: dict[str, Any] | None = None
    error_type: str | None = None
    http_status: int | None = None
    prompt_version: str | None = None
    canonical_input_hash: str | None = None
    input_char_count: int | None = None
    route_cache_status: str | None = None
    route_cache_key: str | None = None
    selected_subtree_position_count: int | None = None
    max_total_subtree_positions: int | None = None
    routing_over_budget: bool | None = None


async def run_simple_stock_quote(
    *,
    text: str,
    session: AsyncSession,
    profile: str | None = None,
    category_ids: Sequence[str] | None = None,
    distributor_code: str = "ocs",
    settings: LlmSettings | None = None,
) -> V3FullCategoryQuoteResult:
    effective_settings = settings or get_llm_settings()
    category_repository = CategoryRepository(session)
    product_repository = ProductRepository(session)
    route_decision: SimpleStockRouteDecision | None = None
    category_catalog = _category_catalog_for_router(
        await category_repository.list_all_categories(distributor_code)
    )

    if not profile and not category_ids:
        if not category_catalog:
            route_decision = SimpleStockRouteDecision(
                status="no_matching_category",
                reason="No distributor category catalog is available for simple stock route.",
            )
            return _routing_failure_result(
                text=text,
                distributor_code=distributor_code,
                route_decision=route_decision,
                settings=effective_settings,
            )
        category_stock_counts = await product_repository.list_latest_stock_counts_by_category(
            distributor_code,
            [entry["category_id"] for entry in category_catalog],
        )
        route_decision = await asyncio.to_thread(
            route_simple_stock_target,
            text=text,
            distributor_code=distributor_code,
            settings=effective_settings,
            category_catalog=category_catalog,
            category_stock_counts=category_stock_counts,
        )
        if (
            route_decision.status not in {"selected_profile", "selected_category"}
            or not route_decision.category_ids
        ):
            return _routing_failure_result(
                text=text,
                distributor_code=distributor_code,
                route_decision=route_decision,
                settings=effective_settings,
            )
        profile = route_decision.profile
        category_ids = route_decision.category_ids

    profile_name, root_category_ids = resolve_v3_full_category_profile(
        profile=profile,
        category_ids=category_ids,
    )
    root_category_ids = _non_overlapping_category_ids(
        _clean_category_ids(root_category_ids),
        category_catalog=category_catalog,
    )[:SIMPLE_STOCK_ROUTE_MAX_CATEGORY_IDS]
    resolved_category_ids = await category_repository.list_category_ids_with_descendants(
        distributor_code=distributor_code,
        category_ids=root_category_ids,
    )
    resolved_request = _resolved_request_json(
        route_decision.resolved_request if route_decision is not None else {},
        text=text,
        profile=profile_name,
        source="simple_route" if route_decision is not None else "profile_override",
    )
    resolved_request["target_category_ids"] = list(root_category_ids)

    stock_refresh_diagnostics: dict[str, Any] = {
        "enabled": bool(effective_settings.v3_refresh_categories_before_llm),
        "status": "disabled",
    }
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
            rows = await product_repository.list_latest_full_category_group_matrix(
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
                    stock_refresh_diagnostics=stock_refresh_diagnostics,
                    settings=effective_settings,
                    route_decision=route_decision,
                )

    if rows is None:
        rows = await product_repository.list_latest_full_category_group_matrix(
            distributor_code,
            resolved_category_ids,
        )
    matrix_package = build_simple_stock_matrix_group_package(
        distributor_code=distributor_code,
        category_ids=resolved_category_ids,
        rows=rows,
        max_package_chars=effective_settings.llm_configurator_max_package_chars,
        model=effective_settings.llm_model,
    )

    outcome = await asyncio.to_thread(
        _compose_simple_stock_quote_sync,
        text=text,
        resolved_request=resolved_request,
        matrix_package=matrix_package,
        settings=effective_settings,
    )
    report_json = outcome.to_report_json()
    report_json["v3_result_state"] = simple_stock_result_state(report_json)
    report_json["v3_profile"] = profile_name
    report_json["category_ids"] = resolved_category_ids
    report_json["root_category_ids"] = root_category_ids
    report_json["distributor_code"] = distributor_code
    report_json["source_text"] = text
    report_json["resolved_request"] = resolved_request
    if route_decision is not None:
        report_json["simple_route_decision"] = _route_decision_json(route_decision)
    if stock_refresh_used_cached_fallback(stock_refresh_diagnostics):
        add_stock_refresh_cache_warning(report_json)

    diagnostics = report_json.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics.setdefault("route", SIMPLE_STOCK_ROUTE_MODE)
        diagnostics.setdefault("simple_clean_route", True)
        diagnostics.setdefault("legacy_v7_1_contract_used", False)
        diagnostics.setdefault("category_ids", resolved_category_ids)
        diagnostics.setdefault("root_category_ids", root_category_ids)
        diagnostics.setdefault("matrix_row_count", len(rows))
        diagnostics.setdefault(
            "matrix_position_count",
            matrix_package.payload.get("diagnostics", {}).get("position_count"),
        )
        diagnostics.setdefault("matrix_char_count", matrix_package.char_count)
        diagnostics.setdefault("matrix_status", matrix_package.status)
        diagnostics.setdefault("model", effective_settings.llm_model)
        diagnostics.setdefault("resolved_request", resolved_request)
        diagnostics.setdefault("stock_refresh", stock_refresh_diagnostics)
        diagnostics.setdefault(
            "category_augmentation",
            {
                "enabled": False,
                "reason": "llm_selected_category_set",
                "added_category_ids": [],
                "matched_groups": [],
            },
        )
        if route_decision is not None:
            diagnostics.setdefault("simple_route_used", True)
            diagnostics.setdefault("simple_route_decision", _route_decision_json(route_decision))

    return V3FullCategoryQuoteResult(
        profile=profile_name,
        category_ids=resolved_category_ids,
        distributor_code=distributor_code,
        result_state=report_json["v3_result_state"],
        report_json=report_json,
    )


def route_simple_stock_target(
    *,
    text: str,
    distributor_code: str | None = None,
    settings: LlmSettings | None = None,
    category_catalog: Sequence[Mapping[str, Any]],
    category_stock_counts: Sequence[Mapping[str, Any]] | None = None,
    llm_client: LlmClient | None = None,
) -> SimpleStockRouteDecision:
    effective_settings = settings or get_llm_settings()
    client = llm_client or _create_llm_client(effective_settings)
    if client is None:
        return SimpleStockRouteDecision(
            status="provider_not_configured",
            reason="LLM provider is not configured for simple stock route.",
        )

    system_prompt = _simple_route_system_prompt()
    routing_index = _category_routing_index_for_router(
        category_catalog,
        category_stock_counts=category_stock_counts or [],
    )
    user_prompt = _simple_route_user_prompt(
        text,
        routing_index=routing_index,
    )
    canonical_input_hash = _sha256_text(user_prompt)
    route_cache_key = _simple_route_cache_key(
        canonical_input_hash=canonical_input_hash,
        distributor_code=distributor_code,
        model=effective_settings.llm_model,
    )
    if llm_client is None:
        cached_decision = _simple_route_cache_get(route_cache_key)
        if cached_decision is not None:
            return cached_decision

    try:
        raw_output = client.generate_json(system_prompt, user_prompt)
    except LlmHttpError as exc:
        return SimpleStockRouteDecision(
            status="provider_error",
            reason="LLM provider error before simple stock route completed.",
            error_type=type(exc).__name__,
            http_status=exc.status_code,
        )
    except LlmError as exc:
        return SimpleStockRouteDecision(
            status="provider_error",
            reason="LLM error before simple stock route completed.",
            error_type=type(exc).__name__,
        )
    finally:
        if llm_client is None and isinstance(client, OpenAICompatibleLlmClient):
            client.close()

    decision = _parse_simple_route_output(
        raw_output,
        category_catalog=category_catalog,
        routing_index=routing_index,
        prompt_version=SIMPLE_STOCK_ROUTE_PROMPT_VERSION,
        canonical_input_hash=canonical_input_hash,
        input_char_count=len(system_prompt) + len(user_prompt),
        route_cache_key=route_cache_key,
        route_cache_status="miss" if llm_client is None else "bypassed",
        max_total_subtree_positions=SIMPLE_STOCK_ROUTE_MAX_TOTAL_SUBTREE_POSITIONS,
    )
    if llm_client is None and decision.status == "selected_category" and decision.category_ids:
        _simple_route_cache_put(route_cache_key, decision)
    return decision


def simple_stock_result_state(report_json: Mapping[str, Any]) -> str:
    final_status_source = str(report_json.get("final_status_source") or "")
    primary_status = str(report_json.get("primary_recommendation_status") or "")
    quote = report_json.get("validated_quote")
    if final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED and primary_status in {
        "valid",
        "llm_final",
    }:
        engineering_review_required = True
        if isinstance(quote, Mapping):
            engineering_review_required = bool(
                quote.get("engineering_review_required", True)
            )
        return (
            QUOTE_DRAFT_REVIEW_REQUIRED
            if engineering_review_required
            else QUOTE_CANDIDATE_CUSTOMER_READY
        )
    if final_status_source in {
        SIMPLE_STOCK_NO_RECOMMENDATION,
        SIMPLE_STOCK_ROUTE_NO_MATCHING_CATEGORY,
    }:
        return NO_RECOMMENDATION
    if final_status_source in {
        SIMPLE_STOCK_ROUTE_SCHEMA_FAILED,
        "simple_stock_quote_schema_failed",
        SIMPLE_STOCK_MECHANICAL_ERROR,
    }:
        return SCHEMA_VALIDATION_FAILED
    if final_status_source in {
        SIMPLE_STOCK_ROUTE_PROVIDER_NOT_CONFIGURED,
        "simple_stock_quote_provider_not_configured",
    }:
        return PROVIDER_NOT_CONFIGURED
    if final_status_source in {
        SIMPLE_STOCK_ROUTE_PROVIDER_ERROR,
        "simple_stock_quote_provider_error",
    }:
        return PROVIDER_ERROR
    if final_status_source == SIMPLE_STOCK_REFRESH_FAILED:
        return STOCK_REFRESH_FAILED_STATE
    if final_status_source == MATRIX_TOO_LARGE_FOR_MODEL:
        return MATRIX_TOO_LARGE_FOR_MODEL_STATE
    if final_status_source == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION:
        return MATRIX_EMPTY_AFTER_CATEGORY_SELECTION_STATE
    return NO_RECOMMENDATION


def _compose_simple_stock_quote_sync(
    *,
    text: str,
    resolved_request: Mapping[str, Any],
    matrix_package: Any,
    settings: LlmSettings,
) -> Any:
    llm_client = _create_llm_client(settings)
    try:
        return compose_simple_stock_quote(
            user_request=text,
            resolved_request=resolved_request,
            matrix_package=matrix_package,
            settings=settings,
            llm_client=llm_client,
        )
    finally:
        if llm_client is not None:
            llm_client.close()


def _routing_failure_result(
    *,
    text: str,
    distributor_code: str,
    route_decision: SimpleStockRouteDecision,
    settings: LlmSettings,
) -> V3FullCategoryQuoteResult:
    if route_decision.status == "provider_not_configured":
        final_status_source = SIMPLE_STOCK_ROUTE_PROVIDER_NOT_CONFIGURED
    elif route_decision.status == "provider_error":
        final_status_source = SIMPLE_STOCK_ROUTE_PROVIDER_ERROR
    elif route_decision.status == "schema_error":
        final_status_source = SIMPLE_STOCK_ROUTE_SCHEMA_FAILED
    else:
        final_status_source = SIMPLE_STOCK_ROUTE_NO_MATCHING_CATEGORY

    report_json: dict[str, Any] = {
        "pipeline_version": SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
        "composer_mode": SIMPLE_STOCK_ROUTE_MODE,
        "llm_configurator_used": route_decision.status != "provider_not_configured",
        "primary_recommendation_status": "no_recommendation",
        "final_status_source": final_status_source,
        "validated_quote": {},
        "primary_recommendation": {},
        "no_recommendation_reason": {
            "summary": route_decision.reason
            or "LLM could not select a stock category for the simple route.",
            "fallback_reason": final_status_source,
        },
        "v3_llm_output": route_decision.raw_output or {},
        "v3_validation_errors": [],
        "v3_validation_warnings": [],
        "diagnostics": {
            "route": SIMPLE_STOCK_ROUTE_MODE,
            "simple_clean_route": True,
            "legacy_v7_1_contract_used": False,
            "simple_route_used": True,
            "simple_route_decision": _route_decision_json(route_decision),
            "resolved_request": _resolved_request_json(
                route_decision.resolved_request,
                text=text,
                profile=route_decision.profile,
                source="simple_route",
            ),
            "model": settings.llm_model,
        },
        "simple_route_decision": _route_decision_json(route_decision),
        "resolved_request": _resolved_request_json(
            route_decision.resolved_request,
            text=text,
            profile=route_decision.profile,
            source="simple_route",
        ),
        "source_text": text,
        "distributor_code": distributor_code,
    }
    report_json["v3_result_state"] = simple_stock_result_state(report_json)
    return V3FullCategoryQuoteResult(
        profile=route_decision.profile,
        category_ids=route_decision.category_ids,
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
    stock_refresh_diagnostics: Mapping[str, Any],
    settings: LlmSettings,
    route_decision: SimpleStockRouteDecision | None,
) -> V3FullCategoryQuoteResult:
    report_json: dict[str, Any] = {
        "pipeline_version": SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
        "composer_mode": SIMPLE_STOCK_ROUTE_MODE,
        "llm_configurator_used": False,
        "primary_recommendation_status": "no_recommendation",
        "final_status_source": SIMPLE_STOCK_REFRESH_FAILED,
        "validated_quote": {},
        "primary_recommendation": {},
        "no_recommendation_reason": {
            "summary": "Selected distributor stock could not be refreshed before simple quote.",
            "fallback_reason": SIMPLE_STOCK_REFRESH_FAILED,
        },
        "v3_validation_errors": [],
        "v3_validation_warnings": [],
        "diagnostics": {
            "route": SIMPLE_STOCK_ROUTE_MODE,
            "simple_clean_route": True,
            "legacy_v7_1_contract_used": False,
            "category_ids": list(resolved_category_ids),
            "root_category_ids": list(root_category_ids),
            "resolved_request": dict(resolved_request),
            "stock_refresh": dict(stock_refresh_diagnostics),
            "model": settings.llm_model,
        },
        "v3_profile": profile_name,
        "category_ids": list(resolved_category_ids),
        "root_category_ids": list(root_category_ids),
        "distributor_code": distributor_code,
        "source_text": text,
        "resolved_request": dict(resolved_request),
    }
    if route_decision is not None:
        report_json["simple_route_decision"] = _route_decision_json(route_decision)
        report_json["diagnostics"]["simple_route_decision"] = _route_decision_json(
            route_decision
        )
    report_json["v3_result_state"] = simple_stock_result_state(report_json)
    return V3FullCategoryQuoteResult(
        profile=profile_name,
        category_ids=list(resolved_category_ids),
        distributor_code=distributor_code,
        result_state=report_json["v3_result_state"],
        report_json=report_json,
    )


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


def _simple_route_system_prompt() -> str:
    return """
You are Stock Configurator Simple Route.

Given a free-form Telegram request and distributor category routing index,
choose the minimal set of distributor category subtrees whose descendants
should be sent to the stock quote composer.

Do not choose SKUs, products, prices or stock rows. Do not build a BOM. Your
job is request understanding, target object detection and category routing.

First split the request into commercially material quote roles and fitment
notes. Quote roles are primary systems, high-value components, functional
options, licenses, consumables or spares that deserve their own stock lookup
and can materially change the quote. Fitment notes are low-value installation
or mounting details for a larger configured object. For each quote role, choose
a real category_id only when that category subtree is likely to contain
relevant stock. If a role cannot be mapped to the catalog, put it into unmapped
requirements instead of inventing a category.

Also identify target_objects: the concrete object(s) the customer expects to
receive. A target object can be a complete system, device, license, component
set, accessory set or explicitly requested standalone component. Set
expects_anchor_line=true when this object should normally be represented by
one stock row before supporting options are added. Set false for pure sets of
equal standalone components or consumables.

Use category_routing_index. Each row is a real distributor category path with
subtree_in_stock_positions: how many stocked positions are under that category
after descendants are expanded.

Rules:
- Select 1 to 12 known, non-overlapping category_ids. Cover every explicit
  product role first; only then minimize scope.
- Category-role discipline: map each requirement to categories by the product
  role sold in the category path, not by specifications or compatibility terms
  merely mentioned inside another product role. If no stocked category for the
  requested product role is visible, put the role into unmapped_requirements
  instead of guessing.
- Whole-object priority: if the request asks to supply, quote, build or
  configure a complete target object such as a server, storage system,
  workstation, switch, UPS, appliance or other finished device, first select
  the category subtree for ready, assembled, configured or finished devices
  when such a subtree is visible and stocked. Component/platform categories
  are not a substitute for that finished-device category. Include both the
  finished-device category and explicit component/option/accessory categories
  when the request asks for a configured system.
- Anchor/base completeness: when you identify a target object that should be
  returned in resolved_request.target_objects[] with expects_anchor_line=true,
  that target/base object is its own explicit product role. The selected
  category set must include a category subtree whose path is likely to contain
  stocked base/platform/product candidates for that target object itself.
  Supporting component, option, license or accessory branches do not cover the
  target object by implication. Do not replace an assembled/ready system
  category with a platform, chassis, component or spare-parts branch.
  For such a target, a components-only route is incomplete even if CPU, RAM,
  storage, network, license or accessory roles are covered. Choose the most
  specific visible stocked base/platform subtree that still preserves
  reasonable choices. Do not choose a catalog-wide root when a narrower stocked
  base subtree is visible. Then add only the additional component, option,
  license and material option subtrees needed for explicit roles.
- Commercial materiality boundary: keep core BOM roles first-class. Always
  preserve explicit base/system, CPU, RAM, drive, controller, HBA, NIC,
  transceiver, PSU, battery, license, support, high-performance cooling kit and
  other functional option roles as must_have when requested. Treat rails, cable
  management arms, generic mounting kits, caddies, brackets, screws, minor
  cables, generic risers and similar installation hardware for a larger
  configured object as fitment_notes, not must_have roles, unless the customer
  is buying that accessory itself as the primary standalone object.
- Do not move explicit high-performance cooling requests into fitment_notes.
  Requested high-performance fans, fan kits, heat sinks, heatsinks, cooling
  kits or other platform cooling options are material quote roles even when
  they appear in the same phrase as rails, risers or cable management arms.
  Keep each requested cooling role in must_have and map visible cooling
  categories when the distributor catalog has them.
- For fitment_notes, do not select dedicated accessory subtrees merely to chase
  those low-value installation details. Preserve them in
  resolved_request.fitment_notes so the final quote can mention them as a
  single manual reserve or engineer check.
- Atomic request roles: resolved_request.must_have must be a flat list of
  atomic, independently fulfillable commercial roles. Each item must represent
  exactly one material product, option, component, license, service,
  consumable or spare that can independently become a quote line, available
  alternative, procurement gap or
  engineer check. Split comma, semicolon, plus and "and"/"и"-separated lists
  only for material quote roles with distinct stock outcomes. Do not atomize
  fitment notes into must_have. Do not split product model names, part numbers,
  quantity+spec phrases or technical descriptors that belong to one role.
  Prefer object entries with requirement_id and requirement text.
- Choose the smallest subtree or set of descendant subtrees that preserves the
  same complete role coverage. Use a parent only when no smaller descendant set
  covers those roles; never choose a broad root merely as insurance.
- For multi-role requests, select separate branches when roles live apart
  instead of their common ancestor. For single-role requests, choose one
  sufficiently narrow subtree that still preserves reasonable choices.
- Prefer a complete selection within max_total_subtree_positions, but never
  drop an explicit role only to meet the budget.
- If a complete selection exceeds max_total_subtree_positions, re-check broad
  parent choices and replace them with specific non-overlapping child branches
  when the same explicit role coverage is preserved.
- Break ties by lower total subtree_in_stock_positions, then fewer category_ids,
  then lexicographically smaller sorted category_ids.
- Never select both an ancestor and its descendant. Do not select SKUs or reason
  about product compatibility.

Return status="selected_category" with real category_ids from
category_routing_index. If no category is suitable, return
status="no_matching_category".

Return one JSON object:
{
  "status": "selected_category | no_matching_category",
  "profile": null,
  "category_ids": ["1 to 12 real category_id values"],
  "reason": "short reason",
  "resolved_request": {
    "customer_task_summary": "Russian summary",
    "target_objects": [
      {
        "target_id": "T1",
        "label": "requested object",
        "quantity": 1,
        "expects_anchor_line": true,
        "requirement_ids": ["optional short IDs"]
      }
    ],
    "must_have": [
      {
        "requirement_id": "R1",
        "requirement": "one atomic requirement that cannot be ignored"
      }
    ],
    "targets": ["requested models, quantities, options and preferences"],
    "fitment_notes": [
      "low-value installation or mounting details to mention but not quote as core roles"
    ],
    "requirement_category_map": [
      {"requirement": "...", "category_ids": ["..."]}
    ],
    "unmapped_requirements": ["roles not visible in category_routing_index"],
    "unknowns": ["missing details to clarify"],
    "allow_analogs": true,
    "allow_partial_offer": true
  }
}
"""


def _simple_route_user_prompt(
    text: str,
    *,
    routing_index: Sequence[Mapping[str, Any]],
) -> str:
    return _stable_json(
        {
            "user_request": text.strip(),
            "prompt_version": SIMPLE_STOCK_ROUTE_PROMPT_VERSION,
            "routing_policy": {
                "sku_selection": "forbidden",
                "profile_first": False,
                "max_category_ids": SIMPLE_STOCK_ROUTE_MAX_CATEGORY_IDS,
                "max_total_subtree_positions": (
                    SIMPLE_STOCK_ROUTE_MAX_TOTAL_SUBTREE_POSITIONS
                ),
                "matrix_budget": (
                    "soft: prefer complete selections within this total subtree size; "
                    "do not drop explicit roles only to meet it"
                ),
                "category_descendants": "code expands selected categories after your answer",
                "extra_category_augmentation": "disabled",
                "target_objects": "return in resolved_request; code passes them to composer",
                "default_allow_analogs": True,
                "default_allow_partial_offer": True,
            },
            "category_routing_index": list(routing_index),
        }
    )


def _parse_simple_route_output(
    raw_output: Mapping[str, Any],
    *,
    category_catalog: Sequence[Mapping[str, Any]],
    routing_index: Sequence[Mapping[str, Any]],
    prompt_version: str,
    canonical_input_hash: str,
    input_char_count: int,
    route_cache_key: str | None = None,
    route_cache_status: str | None = None,
    max_total_subtree_positions: int | None = None,
) -> SimpleStockRouteDecision:
    status = str(raw_output.get("status") or "").strip()
    category_ids = _clean_category_ids(raw_output.get("category_ids") or [])
    reason = str(raw_output.get("reason") or "").strip() or None
    resolved_request = _mapping_or_empty(raw_output.get("resolved_request"))
    profile_name = _clean_profile(raw_output.get("profile") or resolved_request.get("profile"))
    valid_category_ids = {
        str(entry.get("category_id") or "").strip()
        for entry in category_catalog
        if str(entry.get("category_id") or "").strip()
    }
    available_profiles = {
        str(entry.get("profile") or "").strip()
        for entry in _profile_catalog_for_router(category_catalog=category_catalog)
    }

    if status == "selected_profile":
        unknown_category_ids = [
            category_id for category_id in category_ids if category_id not in valid_category_ids
        ]
        if category_ids and unknown_category_ids:
            return SimpleStockRouteDecision(
                status="schema_error",
                profile=profile_name,
                category_ids=category_ids,
                reason="LLM returned category IDs outside the catalog.",
                resolved_request=resolved_request,
                raw_output=dict(raw_output),
                error_type="SimpleStockRouteUnknownCategoryIds",
                prompt_version=prompt_version,
                canonical_input_hash=canonical_input_hash,
                input_char_count=input_char_count,
                route_cache_status=route_cache_status,
                route_cache_key=route_cache_key,
                max_total_subtree_positions=max_total_subtree_positions,
            )
        if category_ids:
            selected_category_ids = _non_overlapping_category_ids(
                category_ids,
                category_catalog=category_catalog,
            )
        elif profile_name is not None and profile_name in available_profiles:
            selected_category_ids = _non_overlapping_category_ids(
                V3_FULL_CATEGORY_PROFILES[profile_name].category_ids,
                category_catalog=category_catalog,
            )
        else:
            return SimpleStockRouteDecision(
                status="schema_error",
                profile=profile_name,
                category_ids=category_ids,
                reason="Simple route requires one category ID.",
                resolved_request=resolved_request,
                raw_output=dict(raw_output),
                error_type="SimpleStockRouteCategoryRequired",
                prompt_version=prompt_version,
                canonical_input_hash=canonical_input_hash,
                input_char_count=input_char_count,
                route_cache_status=route_cache_status,
                route_cache_key=route_cache_key,
                max_total_subtree_positions=max_total_subtree_positions,
            )
        resolved_request["profile"] = None
        resolved_request["target_category_ids"] = selected_category_ids
        selected_subtree_position_count = _selected_subtree_position_count(
            selected_category_ids,
            routing_index=routing_index,
        )
        return SimpleStockRouteDecision(
            status="selected_category",
            profile=None,
            category_ids=selected_category_ids,
            reason=reason,
            resolved_request=resolved_request,
            raw_output=dict(raw_output),
            prompt_version=prompt_version,
            canonical_input_hash=canonical_input_hash,
            input_char_count=input_char_count,
            route_cache_status=route_cache_status,
            route_cache_key=route_cache_key,
            selected_subtree_position_count=selected_subtree_position_count,
            max_total_subtree_positions=max_total_subtree_positions,
            routing_over_budget=_routing_over_budget(
                selected_subtree_position_count,
                max_total_subtree_positions,
            ),
        )

    if status == "selected_category" and category_ids:
        unknown_category_ids = [
            category_id for category_id in category_ids if category_id not in valid_category_ids
        ]
        if unknown_category_ids:
            return SimpleStockRouteDecision(
                status="schema_error",
                category_ids=category_ids,
                reason="LLM returned category IDs outside the catalog.",
                resolved_request=resolved_request,
                raw_output=dict(raw_output),
                error_type="SimpleStockRouteUnknownCategoryIds",
                prompt_version=prompt_version,
                canonical_input_hash=canonical_input_hash,
                input_char_count=input_char_count,
                route_cache_status=route_cache_status,
                route_cache_key=route_cache_key,
                max_total_subtree_positions=max_total_subtree_positions,
            )
        selected_category_ids = _non_overlapping_category_ids(
            category_ids,
            category_catalog=category_catalog,
        )
        resolved_request["profile"] = None
        resolved_request["target_category_ids"] = selected_category_ids
        selected_subtree_position_count = _selected_subtree_position_count(
            selected_category_ids,
            routing_index=routing_index,
        )
        return SimpleStockRouteDecision(
            status="selected_category",
            category_ids=selected_category_ids,
            reason=reason,
            resolved_request=resolved_request,
            raw_output=dict(raw_output),
            prompt_version=prompt_version,
            canonical_input_hash=canonical_input_hash,
            input_char_count=input_char_count,
            route_cache_status=route_cache_status,
            route_cache_key=route_cache_key,
            selected_subtree_position_count=selected_subtree_position_count,
            max_total_subtree_positions=max_total_subtree_positions,
            routing_over_budget=_routing_over_budget(
                selected_subtree_position_count,
                max_total_subtree_positions,
            ),
        )

    if status == "no_matching_category":
        return SimpleStockRouteDecision(
            status="no_matching_category",
            reason=reason or "The request does not match available stock categories.",
            resolved_request=resolved_request,
            raw_output=dict(raw_output),
            prompt_version=prompt_version,
            canonical_input_hash=canonical_input_hash,
            input_char_count=input_char_count,
            route_cache_status=route_cache_status,
            route_cache_key=route_cache_key,
            max_total_subtree_positions=max_total_subtree_positions,
        )

    return SimpleStockRouteDecision(
        status="schema_error",
        reason="LLM returned an invalid simple route response.",
        resolved_request=resolved_request,
        raw_output=dict(raw_output),
        error_type="SimpleStockRouteSchemaError",
        prompt_version=prompt_version,
        canonical_input_hash=canonical_input_hash,
        input_char_count=input_char_count,
        route_cache_status=route_cache_status,
        route_cache_key=route_cache_key,
        max_total_subtree_positions=max_total_subtree_positions,
    )


def _category_catalog_for_router(categories: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "category_id": str(category.category_id or "").strip(),
            "parent_category_id": _string_or_none(category.parent_category_id),
            "name": str(category.name or "").strip(),
            "level": int(category.level or 0),
            "path": _category_path_text(category.path_json, fallback=category.name),
        }
        for category in categories
        if str(category.category_id or "").strip()
    ]


def _category_routing_index_for_router(
    category_catalog: Sequence[Mapping[str, Any]],
    *,
    category_stock_counts: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    entries = [
        {
            "category_id": str(entry.get("category_id") or "").strip(),
            "parent_category_id": _string_or_none(entry.get("parent_category_id")),
            "name": str(entry.get("name") or "").strip(),
            "level": _int_or_zero(entry.get("level")),
            "path": str(entry.get("path") or "").strip(),
        }
        for entry in category_catalog
        if str(entry.get("category_id") or "").strip()
    ]
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        parent_id = entry.get("parent_category_id")
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(entry)

    stock_position_count_by_category = _stock_position_count_by_category(
        category_stock_counts
    )
    routing_index: list[dict[str, Any]] = []
    for entry in entries:
        category_id = entry["category_id"]
        descendants = _descendant_category_entries(
            category_id,
            children_by_parent=children_by_parent,
        )
        subtree_category_ids = [category_id]
        subtree_category_ids.extend(descendant["category_id"] for descendant in descendants)
        subtree_stock_positions = sum(
            stock_position_count_by_category.get(subtree_category_id, 0)
            for subtree_category_id in subtree_category_ids
        )
        routing_index.append(
            _compact_dict(
                {
                    "category_id": category_id,
                    "parent_category_id": entry.get("parent_category_id"),
                    "category_path": entry["path"] or entry["name"],
                    "level": entry["level"],
                    "children_count": len(children_by_parent.get(category_id, [])),
                    "descendant_count": len(descendants),
                    "subtree_in_stock_positions": subtree_stock_positions,
                }
            )
        )
    return routing_index


def _stock_position_count_by_category(
    category_stock_counts: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in category_stock_counts:
        category_id = str(row.get("category_id") or "").strip()
        if not category_id:
            continue
        try:
            count = int(row.get("position_count") or row.get("stock_row_count") or 0)
        except (TypeError, ValueError):
            count = 0
        result[category_id] = max(0, count)
    return result


def _descendant_category_entries(
    category_id: str,
    *,
    children_by_parent: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    stack = list(reversed(list(children_by_parent.get(category_id, []))))
    seen: set[str] = set()
    while stack:
        entry = dict(stack.pop())
        child_id = str(entry.get("category_id") or "").strip()
        if not child_id or child_id in seen:
            continue
        seen.add(child_id)
        result.append(entry)
        stack.extend(reversed(list(children_by_parent.get(child_id, []))))
    return result


def _profile_catalog_for_router(
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
        }
        for profile in V3_FULL_CATEGORY_PROFILES.values()
        if set(profile.category_ids).issubset(available_category_ids)
    ]


def _resolved_request_json(
    value: Mapping[str, Any],
    *,
    text: str,
    profile: str | None,
    source: str,
) -> dict[str, Any]:
    resolved = dict(value)
    resolved.setdefault("customer_task_summary", text.strip())
    resolved.setdefault("profile", profile)
    resolved.setdefault("source", source)
    resolved.setdefault("allow_analogs", True)
    resolved.setdefault("allow_partial_offer", True)
    if not isinstance(resolved.get("target_objects"), list):
        resolved["target_objects"] = []
    return resolved


def _route_decision_json(decision: SimpleStockRouteDecision) -> dict[str, Any]:
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
        "canonical_input_hash": decision.canonical_input_hash,
        "input_char_count": decision.input_char_count,
        "route_cache_status": decision.route_cache_status,
        "route_cache_key": decision.route_cache_key,
        "selected_subtree_position_count": decision.selected_subtree_position_count,
        "max_total_subtree_positions": decision.max_total_subtree_positions,
        "routing_over_budget": decision.routing_over_budget,
    }


def _simple_route_cache_key(
    *,
    canonical_input_hash: str,
    distributor_code: str | None,
    model: str,
) -> str:
    return _sha256_text(
        _stable_json(
            {
                "route_prompt_version": SIMPLE_STOCK_ROUTE_PROMPT_VERSION,
                "canonical_input_hash": canonical_input_hash,
                "distributor_code": distributor_code or "",
                "model": model,
            }
        )
    )


def _simple_route_cache_get(cache_key: str) -> SimpleStockRouteDecision | None:
    with _SIMPLE_STOCK_ROUTE_CACHE_LOCK:
        cached = _SIMPLE_STOCK_ROUTE_CACHE.get(cache_key)
        if cached is None:
            return None
        _SIMPLE_STOCK_ROUTE_CACHE.move_to_end(cache_key)
    return _copy_route_decision(
        cached,
        route_cache_status="hit",
        route_cache_key=cache_key,
    )


def _simple_route_cache_put(
    cache_key: str,
    decision: SimpleStockRouteDecision,
) -> None:
    cached = _copy_route_decision(
        decision,
        route_cache_status="stored",
        route_cache_key=cache_key,
    )
    with _SIMPLE_STOCK_ROUTE_CACHE_LOCK:
        _SIMPLE_STOCK_ROUTE_CACHE[cache_key] = cached
        _SIMPLE_STOCK_ROUTE_CACHE.move_to_end(cache_key)
        while len(_SIMPLE_STOCK_ROUTE_CACHE) > SIMPLE_STOCK_ROUTE_CACHE_MAX_SIZE:
            _SIMPLE_STOCK_ROUTE_CACHE.popitem(last=False)


def _copy_route_decision(
    decision: SimpleStockRouteDecision,
    **overrides: Any,
) -> SimpleStockRouteDecision:
    copied = replace(
        decision,
        category_ids=list(decision.category_ids),
        resolved_request=copy.deepcopy(decision.resolved_request),
        raw_output=copy.deepcopy(decision.raw_output)
        if decision.raw_output is not None
        else None,
    )
    return replace(copied, **overrides) if overrides else copied


def _selected_subtree_position_count(
    category_ids: Sequence[str],
    *,
    routing_index: Sequence[Mapping[str, Any]],
) -> int:
    positions_by_category = {
        str(row.get("category_id") or "").strip(): _int_or_zero(
            row.get("subtree_in_stock_positions")
        )
        for row in routing_index
        if str(row.get("category_id") or "").strip()
    }
    return sum(positions_by_category.get(category_id, 0) for category_id in category_ids)


def _routing_over_budget(
    selected_subtree_position_count: int,
    max_total_subtree_positions: int | None,
) -> bool:
    return bool(
        max_total_subtree_positions
        and selected_subtree_position_count > max_total_subtree_positions
    )



def _clean_category_ids(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [
        category_id
        for category_id in dict.fromkeys(str(item or "").strip() for item in value)
        if category_id
    ]


def _non_overlapping_category_ids(
    category_ids: Sequence[str],
    *,
    category_catalog: Sequence[Mapping[str, Any]],
) -> list[str]:
    parent_by_id = {
        str(entry.get("category_id") or "").strip(): _string_or_none(
            entry.get("parent_category_id")
        )
        for entry in category_catalog
        if str(entry.get("category_id") or "").strip()
    }
    result: list[str] = []

    for category_id in _clean_category_ids(category_ids):
        ancestors = _ancestor_ids(category_id, parent_by_id=parent_by_id)
        if any(parent_id in result for parent_id in ancestors):
            continue

        result = [
            existing_id
            for existing_id in result
            if category_id
            not in _ancestor_ids(existing_id, parent_by_id=parent_by_id)
        ]
        result.append(category_id)

    return result


def _ancestor_ids(
    category_id: str,
    *,
    parent_by_id: Mapping[str, str | None],
) -> list[str]:
    ancestors: list[str] = []
    seen: set[str] = set()
    parent_id = parent_by_id.get(category_id)
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        ancestors.append(parent_id)
        parent_id = parent_by_id.get(parent_id)
    return ancestors


def _clean_profile(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text or text == "null":
        return None
    profile_name = V3_FULL_CATEGORY_PROFILE_ALIASES.get(text, text)
    return profile_name if profile_name in V3_FULL_CATEGORY_PROFILES else None


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _compact_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value not in (None, "", [], {})
    }


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


def _string_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _stable_json(payload: Mapping[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "SIMPLE_STOCK_QUOTE_PIPELINE_VERSION",
    "SimpleStockRouteDecision",
    "route_simple_stock_target",
    "run_simple_stock_quote",
    "simple_stock_result_state",
]
