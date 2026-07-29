from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.core.config import get_llm_settings
from app.llm.base import LlmError, LlmHttpError, LlmInvalidJsonError
from app.matching.spec_schema import StockSpec, StockSpecItem
from app.planning import role_lifecycle
from app.planning.network_facts import network_requirement_from_sources
from app.policies.product_group_policy import (
    PRODUCT_GROUP_PROFILES,
    ProductGroupProfile,
    get_product_group_profile,
)

SERVER_PRODUCT_GROUP = "server"
NETWORK_PRODUCT_GROUP = "network"
STORAGE_PRODUCT_GROUP = "storage"
CLASS_HARD_TECHNICAL = "hard_technical_requirement"
CLASS_SOFT_PREFERENCE = "soft_preference"
CLASS_WORKLOAD_CONTEXT = "workload_context"
CLASS_LOGISTICS_CONSTRAINT = "logistics_constraint"
CLASS_COMMERCIAL_INSTRUCTION = "commercial_instruction"
CLASS_RESPONSE_INSTRUCTION = "response_instruction"
CLASS_OUTPUT_INSTRUCTION = CLASS_RESPONSE_INSTRUCTION
CLASS_ENGINEER_REVIEW_INSTRUCTION = "engineer_review_instruction"
CLASS_UNSUPPORTED_HARD = "unsupported_hard_requirement"
UNMAPPED_ROLE = "unmapped"
UNKNOWN_FACT = "unknown"
REQ_CLASS_PURCHASABLE_COMPONENT_ROLE = "purchasable_component_role"
REQ_CLASS_PRIMARY_OBJECT_FEATURE = "primary_object_feature"
REQ_CLASS_ACCESSORY_OR_CONSUMABLE = "accessory_or_consumable"
REQ_CLASS_SERVICE_OR_SUPPORT = "service_or_support"
REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT = "logistics_or_commercial_constraint"
REQ_CLASS_ENGINEERING_CHECK = "engineering_check"
REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING = (
    "out_of_scope_or_unmapped_non_blocking"
)
REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE = (
    "blocking_unmapped_purchasable_role"
)
REQ_HARD = "hard"
REQ_OPTIONAL = "optional"
FULFILLMENT_SEPARATE_COMPONENT_REQUIRED = "separate_component_required"
FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT = "included_in_primary_object"
FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT = "included_in_selected_component"
FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT = "included_in_bundle_or_kit"
FULFILLMENT_SERVICE_OR_SUPPORT = "service_or_support"
FULFILLMENT_LOGISTICS_CONSTRAINT = "logistics_constraint"
FULFILLMENT_ENGINEERING_CHECK_ONLY = "engineering_check_only"
FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION = "unverified_requires_confirmation"
FULFILLMENT_NOT_APPLICABLE = "not_applicable"
FULFILLMENT_VALUES = {
    FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
    FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
    FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
    FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
    FULFILLMENT_SERVICE_OR_SUPPORT,
    FULFILLMENT_LOGISTICS_CONSTRAINT,
    FULFILLMENT_ENGINEERING_CHECK_ONLY,
    FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
    FULFILLMENT_NOT_APPLICABLE,
}
FULFILLMENT_INCLUDED_MODES = {
    FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
    FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
    FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
}
PRIMARY_PRODUCT_GROUPS = {SERVER_PRODUCT_GROUP, NETWORK_PRODUCT_GROUP, STORAGE_PRODUCT_GROUP}
PRIMARY_OBJECTS = {
    "server",
    "switch",
    "router",
    "firewall",
    "access_point",
    "storage_system",
    "nas",
    "dac_cable",
    "transceiver",
    "other",
}
SEMANTIC_CONFIDENCE_VALUES = {"high", "medium", "low"}
SEMANTIC_FORBIDDEN_KEYS = {
    "category_id",
    "category_ids",
    "selected_category_ids",
    "component_candidate_id",
    "component_candidate_ids",
    "selected_component_candidate_ids",
    "price",
    "prices",
    "price_value",
    "price_currency",
    "stock",
    "available_quantity",
    "reservable_locations",
}
SEMANTIC_COMPLEX_FALLBACK_REASON = "complex_request_requires_llm_semantic_planner"
SEMANTIC_SOURCE_LLM = "llm"
SEMANTIC_SOURCE_LLM_REPAIRED = "llm_repaired"
SEMANTIC_SOURCE_LLM_MINIMAL_FALLBACK = "llm_minimal_fallback"
SEMANTIC_SOURCE_DETERMINISTIC_FALLBACK = "deterministic_fallback"
SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_EMPTY = "fallback_after_llm_empty"
SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_INVALID = "fallback_after_llm_invalid"
SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_ERROR = "fallback_after_llm_error"
SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT = "fallback_after_llm_timeout"
SEMANTIC_LLM_SOURCES = {
    SEMANTIC_SOURCE_LLM,
    SEMANTIC_SOURCE_LLM_REPAIRED,
    SEMANTIC_SOURCE_LLM_MINIMAL_FALLBACK,
}
SEMANTIC_STAGE_INTENT_ROUTER = "intent_router"
SEMANTIC_STAGE_INTENT_ROUTER_REPAIR = "intent_router_repair"
SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER = "requirement_classifier"
SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER_REPAIR = "requirement_classifier_repair"
SEMANTIC_STAGE_MINIMAL_FALLBACK = "minimal_fallback"
SEMANTIC_EMPTY_RESPONSE_AFTER_REPAIR = "semantic_planner_empty_response_after_repair"
SEMANTIC_PLANNER_TIMEOUT_ERROR_TYPE = "SemanticPlannerTimeout"
SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON = "semantic_planner_timeout"
SEMANTIC_PLANNER_PARSE_STATUS_TIMEOUT = "timeout"
SEMANTIC_PLANNER_TIMEOUT_REASON_STAGE = "stage_timeout"
SEMANTIC_PLANNER_TIMEOUT_REASON_DEADLINE = "overall_deadline_exceeded"
REQUIREMENT_CLASSIFIER_STATUS_COMPLETE = "complete"
REQUIREMENT_CLASSIFIER_STATUS_REPAIRED = "repaired"
REQUIREMENT_CLASSIFIER_STATUS_PARTIAL = "partial"
REQUIREMENT_CLASSIFIER_STATUS_INCOMPLETE_REPAIR = "incomplete_repair"
REQUIREMENT_CLASSIFIER_STATUS_FAILED = "failed"
SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE = (
    "Не удалось безопасно разобрать сложный запрос без AI semantic planner. "
    "Повторите позже или проверьте настройки LLM."
)
SEMANTIC_LLM_UNAVAILABLE_REASON = (
    "AI semantic planner недоступен; использован безопасный детерминированный fallback."
)

CLASSIFICATION_VALUES = {
    CLASS_HARD_TECHNICAL,
    CLASS_SOFT_PREFERENCE,
    CLASS_WORKLOAD_CONTEXT,
    CLASS_LOGISTICS_CONSTRAINT,
    CLASS_COMMERCIAL_INSTRUCTION,
    CLASS_RESPONSE_INSTRUCTION,
    CLASS_ENGINEER_REVIEW_INSTRUCTION,
    CLASS_UNSUPPORTED_HARD,
}

REQUIREMENT_CLASSIFICATION_VALUES = {
    REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
    REQ_CLASS_PRIMARY_OBJECT_FEATURE,
    REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
    REQ_CLASS_SERVICE_OR_SUPPORT,
    REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
    REQ_CLASS_ENGINEERING_CHECK,
    REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
    REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE,
}
NON_BLOCKING_REQUIREMENT_CLASSIFICATIONS = {
    REQ_CLASS_PRIMARY_OBJECT_FEATURE,
    REQ_CLASS_ENGINEERING_CHECK,
    REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
    REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
}


@runtime_checkable
class RequirementPlannerClient(Protocol):
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Return a structured requirement plan as JSON."""


@dataclass
class _SemanticPlannerRunBudget:
    started_at: float
    max_seconds: float
    stage_timeout_seconds: float

    def elapsed_ms(self) -> int:
        return max(0, int((time.monotonic() - self.started_at) * 1000))

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - (time.monotonic() - self.started_at))

    def stage_timeout_for_call(self) -> float:
        return max(0.0, min(self.stage_timeout_seconds, self.remaining_seconds()))


def _plan_semantic_matrix_requirements_v2(
    spec: StockSpec | Mapping[str, Any] | str | None = None,
    *,
    source_text: str | None = None,
    normalized_spec: Mapping[str, Any] | None = None,
    distributor_code: str | None = None,
    planner_client: RequirementPlannerClient | None = None,
    deterministic_product_group_hint: str | None = None,
) -> dict[str, Any]:
    """Plan product group and matrix roles with LLM authority, then coerce to role_plan."""

    request_text = source_text or _source_text_from_spec(spec)
    fallback_plan = _deterministic_fallback_plan_for_semantic_planner(
        spec,
        source_text=source_text,
        normalized_spec=normalized_spec,
        deterministic_product_group_hint=deterministic_product_group_hint,
    )
    deterministic_hint = str(
        deterministic_product_group_hint or fallback_plan.get("product_group") or "unknown"
    ).strip()
    fallback_confidence = _deterministic_semantic_fallback_confidence(
        fallback_plan,
        request_text,
    )
    fallback_plan = _with_semantic_diagnostics(
        fallback_plan,
        source=SEMANTIC_SOURCE_DETERMINISTIC_FALLBACK,
        confidence=fallback_confidence,
        reason=SEMANTIC_LLM_UNAVAILABLE_REASON,
        deterministic_product_group_hint=deterministic_hint,
        semantic_plan=None,
        warning=None,
        fallback_reason="llm_client_not_configured",
    )
    if planner_client is None:
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=SEMANTIC_SOURCE_DETERMINISTIC_FALLBACK,
            confidence=fallback_confidence,
            reason=SEMANTIC_LLM_UNAVAILABLE_REASON,
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=None,
            warning=None,
            fallback_reason="llm_client_not_configured",
        )

    payload = {
        "source_text": request_text,
        "normalized_spec": dict(normalized_spec or {}),
        "product_group_profiles": _semantic_profiles_prompt_payload(),
        "distributor_code": distributor_code,
        "deterministic_product_group_hint": deterministic_hint,
        "deterministic_fallback_plan": _safe_plan_for_prompt(fallback_plan),
        "contract": {
            "stage": "AI Semantic Matrix Planner V2",
            "forbidden_fields": sorted(SEMANTIC_FORBIDDEN_KEYS),
            "required_output": _semantic_matrix_json_contract(),
        },
    }
    try:
        response = planner_client.generate_json(
            _semantic_matrix_planner_system_prompt(),
            json.dumps(payload, ensure_ascii=False),
        )
    except (LlmError, ValueError, TypeError) as exc:
        error_diagnostics = _semantic_error_diagnostics(exc)
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=_semantic_llm_exception_source(exc),
            confidence=fallback_confidence,
            reason=(
                "AI semantic planner вернул ошибку; сохранен безопасный "
                "детерминированный fallback."
            ),
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=None,
            warning=f"semantic_matrix_planner_llm_fallback:{type(exc).__name__}",
            fallback_reason=_semantic_llm_exception_fallback_reason(exc),
            error_type=error_diagnostics["error_type"],
            http_status=error_diagnostics["http_status"],
            parse_status=error_diagnostics["parse_status"],
        )

    if not isinstance(response, Mapping) or not response:
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_EMPTY,
            confidence=fallback_confidence,
            reason=(
                "AI semantic planner вернул пустой план; сохранен безопасный "
                "детерминированный fallback."
            ),
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=None,
            warning="semantic_matrix_planner_empty",
            fallback_reason="llm_empty_response",
            error_type=None,
            http_status=None,
            parse_status="empty_response",
        )

    try:
        semantic_plan = _coerce_semantic_matrix_plan(response)
    except (ValueError, TypeError) as exc:
        error_diagnostics = _semantic_error_diagnostics(exc, parse_status="invalid_contract")
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_INVALID,
            confidence=fallback_confidence,
            reason=(
                "AI semantic planner вернул некорректный контракт; сохранен безопасный "
                "детерминированный fallback."
            ),
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=None,
            warning=f"semantic_matrix_planner_invalid:{type(exc).__name__}",
            fallback_reason="llm_invalid_contract",
            error_type=error_diagnostics["error_type"],
            http_status=error_diagnostics["http_status"],
            parse_status=error_diagnostics["parse_status"],
        )

    if not _semantic_plan_is_actionable(semantic_plan):
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_EMPTY,
            confidence=semantic_plan["confidence"],
            reason=semantic_plan["classification_reason"]
            or "AI semantic planner не выбрал пригодную товарную группу или blueprint.",
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=semantic_plan,
            warning="semantic_matrix_planner_not_actionable",
            fallback_reason="llm_not_actionable",
            error_type=None,
            http_status=None,
            parse_status="not_actionable",
        )

    profile = get_product_group_profile(semantic_plan["primary_product_group"])
    if profile is None:
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_INVALID,
            confidence=semantic_plan["confidence"],
            reason=semantic_plan["classification_reason"]
            or "AI semantic planner выбрал неподдерживаемую товарную группу.",
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=semantic_plan,
            warning="semantic_matrix_planner_product_group_unsupported",
            fallback_reason="llm_unsupported_product_group",
            error_type=None,
            http_status=None,
            parse_status="unsupported_product_group",
        )

    role_plan = _requirement_plan_from_semantic_matrix_plan(
        semantic_plan,
        profile=profile,
    )
    return _with_semantic_diagnostics(
        role_plan,
        source=SEMANTIC_SOURCE_LLM,
        confidence=semantic_plan["confidence"],
        reason=semantic_plan["classification_reason"],
        deterministic_product_group_hint=deterministic_hint,
        semantic_plan=semantic_plan,
        warning=None,
        fallback_reason=None,
    )


def plan_semantic_matrix_requirements(
    spec: StockSpec | Mapping[str, Any] | str | None = None,
    *,
    source_text: str | None = None,
    normalized_spec: Mapping[str, Any] | None = None,
    distributor_code: str | None = None,
    planner_client: RequirementPlannerClient | None = None,
    deterministic_product_group_hint: str | None = None,
    semantic_planner_max_seconds: float | None = None,
    semantic_planner_stage_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Plan product group and matrix roles through resilient Semantic Planner V3."""

    budget = _semantic_planner_run_budget(
        max_seconds=semantic_planner_max_seconds,
        stage_timeout_seconds=semantic_planner_stage_timeout_seconds,
    )
    request_text = source_text or _source_text_from_spec(spec)
    fallback_plan = _deterministic_fallback_plan_for_semantic_planner(
        spec,
        source_text=source_text,
        normalized_spec=normalized_spec,
        deterministic_product_group_hint=deterministic_product_group_hint,
    )
    deterministic_hint = str(
        deterministic_product_group_hint or fallback_plan.get("product_group") or "unknown"
    ).strip()
    fallback_confidence = _deterministic_semantic_fallback_confidence(
        fallback_plan,
        request_text,
    )
    fallback_plan = _with_semantic_diagnostics(
        fallback_plan,
        source=SEMANTIC_SOURCE_DETERMINISTIC_FALLBACK,
        confidence=fallback_confidence,
        reason=SEMANTIC_LLM_UNAVAILABLE_REASON,
        deterministic_product_group_hint=deterministic_hint,
        semantic_plan=None,
        warning=None,
        fallback_reason="llm_client_not_configured",
        diagnostics=_semantic_v3_diagnostics(
            attempts=[],
            stage="deterministic_fallback",
            repair_attempted=False,
            repair_success=False,
            minimal_router_used=False,
            minimal_fallback_used=False,
            empty_response_count=0,
            empty_response_reason=None,
            requirement_classifier_status=REQUIREMENT_CLASSIFIER_STATUS_FAILED,
            requirement_classifier_error_type=None,
            requirement_classifier_parse_status=None,
            elapsed_ms=budget.elapsed_ms(),
        ),
    )
    if planner_client is None:
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=SEMANTIC_SOURCE_DETERMINISTIC_FALLBACK,
            confidence=fallback_confidence,
            reason=SEMANTIC_LLM_UNAVAILABLE_REASON,
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=None,
            warning=None,
            fallback_reason="llm_client_not_configured",
            diagnostics=_semantic_v3_diagnostics(
                attempts=[],
                stage="deterministic_fallback",
                repair_attempted=False,
                repair_success=False,
                minimal_router_used=False,
                minimal_fallback_used=False,
                empty_response_count=0,
                empty_response_reason=None,
                requirement_classifier_status=REQUIREMENT_CLASSIFIER_STATUS_FAILED,
                requirement_classifier_error_type=None,
                requirement_classifier_parse_status=None,
                elapsed_ms=budget.elapsed_ms(),
            ),
        )

    attempts: list[dict[str, Any]] = []
    empty_response_count = 0
    repair_attempted = False
    repair_success = False

    intent_result = _run_semantic_intent_router(
        planner_client,
        request_text=request_text,
        normalized_spec=normalized_spec,
        distributor_code=distributor_code,
        deterministic_hint=deterministic_hint,
        fallback_plan=fallback_plan,
        attempts=attempts,
        budget=budget,
    )
    empty_response_count += intent_result["empty_response_count"]
    repair_attempted = repair_attempted or bool(intent_result["repair_attempted"])
    repair_success = repair_success or bool(intent_result["repair_success"])
    intent = intent_result.get("intent")
    if not isinstance(intent, Mapping):
        diagnostics = _semantic_v3_diagnostics(
            attempts=attempts,
            stage=str(intent_result.get("stage") or SEMANTIC_STAGE_INTENT_ROUTER),
            repair_attempted=repair_attempted,
            repair_success=repair_success,
            minimal_router_used=False,
            minimal_fallback_used=False,
            empty_response_count=empty_response_count,
            empty_response_reason=_text_or_none(intent_result.get("empty_response_reason")),
            requirement_classifier_status=REQUIREMENT_CLASSIFIER_STATUS_FAILED,
            requirement_classifier_error_type=None,
            requirement_classifier_parse_status=None,
            elapsed_ms=budget.elapsed_ms(),
        )
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=str(intent_result.get("source") or SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_EMPTY),
            confidence=fallback_confidence,
            reason=SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE,
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=None,
            warning=_text_or_none(intent_result.get("warning")),
            fallback_reason=_semantic_stage_fallback_reason(intent_result),
            error_type=_text_or_none(intent_result.get("error_type")),
            http_status=_as_int(intent_result.get("http_status")),
            parse_status=_text_or_none(intent_result.get("parse_status")),
            diagnostics=diagnostics,
        )

    profile = get_product_group_profile(intent["product_group"])
    if profile is None:
        diagnostics = _semantic_v3_diagnostics(
            attempts=attempts,
            stage=SEMANTIC_STAGE_INTENT_ROUTER,
            repair_attempted=repair_attempted,
            repair_success=repair_success,
            minimal_router_used=True,
            minimal_fallback_used=False,
            empty_response_count=empty_response_count,
            empty_response_reason=None,
            requirement_classifier_status=REQUIREMENT_CLASSIFIER_STATUS_FAILED,
            requirement_classifier_error_type=None,
            requirement_classifier_parse_status="unsupported_product_group",
            elapsed_ms=budget.elapsed_ms(),
        )
        return _semantic_fallback_or_fail_closed(
            fallback_plan,
            request_text=request_text,
            source=SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_INVALID,
            confidence=str(intent.get("confidence") or "low"),
            reason=str(intent.get("reason") or SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE),
            deterministic_product_group_hint=deterministic_hint,
            semantic_plan=None,
            warning="semantic_intent_router_product_group_unsupported",
            fallback_reason=SEMANTIC_COMPLEX_FALLBACK_REASON,
            parse_status="unsupported_product_group",
            diagnostics=diagnostics,
        )

    classifier_result = _run_semantic_requirement_classifier(
        planner_client,
        request_text=request_text,
        normalized_spec=normalized_spec,
        distributor_code=distributor_code,
        intent=intent,
        profile=profile,
        attempts=attempts,
        budget=budget,
    )
    empty_response_count += classifier_result["empty_response_count"]
    semantic_plan = classifier_result.get("semantic_plan")
    source = (
        SEMANTIC_SOURCE_LLM_REPAIRED
        if repair_success
        else SEMANTIC_SOURCE_LLM
    )
    stage = SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER
    warning: str | None = None
    semantic_fallback_reason: str | None = None
    semantic_error_type: str | None = None
    semantic_parse_status: str | None = None
    requirement_classifier_status = REQUIREMENT_CLASSIFIER_STATUS_COMPLETE
    classifier_error_type = _text_or_none(classifier_result.get("error_type"))
    classifier_parse_status = _text_or_none(classifier_result.get("parse_status"))
    classifier_empty_reason = _text_or_none(
        classifier_result.get("empty_response_reason")
    )

    if not isinstance(semantic_plan, Mapping):
        if _semantic_stage_timed_out(classifier_result):
            semantic_plan = _minimal_semantic_plan_from_intent(
                intent,
                request_text=request_text,
                profile=profile,
            )
            _log_semantic_progress(
                "semantic_minimal_fallback_used",
                {
                    "stage": SEMANTIC_STAGE_MINIMAL_FALLBACK,
                    "reason": _semantic_stage_fallback_reason(classifier_result),
                },
            )
            source = SEMANTIC_SOURCE_LLM_MINIMAL_FALLBACK
            stage = SEMANTIC_STAGE_MINIMAL_FALLBACK
            warning = "semantic_requirement_classifier_timeout_minimal_fallback"
            requirement_classifier_status = REQUIREMENT_CLASSIFIER_STATUS_PARTIAL
            semantic_fallback_reason = SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
            semantic_error_type = SEMANTIC_PLANNER_TIMEOUT_ERROR_TYPE
            semantic_parse_status = SEMANTIC_PLANNER_PARSE_STATUS_TIMEOUT
        else:
            repair_attempted = True
            repaired_result = _run_semantic_requirement_classifier(
                planner_client,
                request_text=request_text,
                normalized_spec=normalized_spec,
                distributor_code=distributor_code,
                intent=intent,
                profile=profile,
                attempts=attempts,
                budget=budget,
                repair=True,
                previous_parse_status=classifier_parse_status,
            )
            empty_response_count += repaired_result["empty_response_count"]
            semantic_plan = repaired_result.get("semantic_plan")
            classifier_error_type = (
                _text_or_none(repaired_result.get("error_type")) or classifier_error_type
            )
            classifier_parse_status = (
                _text_or_none(repaired_result.get("parse_status"))
                or classifier_parse_status
            )
            classifier_empty_reason = (
                _text_or_none(repaired_result.get("empty_response_reason"))
                or classifier_empty_reason
            )
            if isinstance(semantic_plan, Mapping):
                repair_success = True
                source = SEMANTIC_SOURCE_LLM_REPAIRED
                stage = SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER_REPAIR
                requirement_classifier_status = REQUIREMENT_CLASSIFIER_STATUS_REPAIRED
            else:
                semantic_plan = _minimal_semantic_plan_from_intent(
                    intent,
                    request_text=request_text,
                    profile=profile,
                )
                _log_semantic_progress(
                    "semantic_minimal_fallback_used",
                    {
                        "stage": SEMANTIC_STAGE_MINIMAL_FALLBACK,
                        "reason": _semantic_stage_fallback_reason(repaired_result),
                    },
                )
                source = SEMANTIC_SOURCE_LLM_MINIMAL_FALLBACK
                stage = SEMANTIC_STAGE_MINIMAL_FALLBACK
                warning = "semantic_requirement_classifier_minimal_fallback"
                requirement_classifier_status = REQUIREMENT_CLASSIFIER_STATUS_PARTIAL
                if _semantic_stage_timed_out(repaired_result):
                    semantic_fallback_reason = SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
                    semantic_error_type = SEMANTIC_PLANNER_TIMEOUT_ERROR_TYPE
                    semantic_parse_status = SEMANTIC_PLANNER_PARSE_STATUS_TIMEOUT

    role_plan = _requirement_plan_from_semantic_matrix_plan(
        semantic_plan,
        profile=profile,
    )
    role_plan = _apply_role_lifecycle_policy(
        role_plan,
        intent=intent,
        semantic_plan=semantic_plan,
        profile=profile,
    )
    source_coverage_diagnostics = _requirement_source_coverage_diagnostics(
        request_text=request_text,
        intent=intent,
        classified_requirements=_mapping_rows(role_plan.get("classified_requirements")),
        profile=profile,
        repair_attempted=repair_attempted,
        repair_stage=stage == SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER_REPAIR,
    )
    role_plan = {**role_plan, **source_coverage_diagnostics}
    if (
        stage == SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER_REPAIR
        and not source_coverage_diagnostics["requirement_classifier_repair_accepted"]
    ):
        repair_success = False
        requirement_classifier_status = (
            REQUIREMENT_CLASSIFIER_STATUS_INCOMPLETE_REPAIR
        )
        incomplete_warning = "semantic_requirement_classifier_repair_incomplete"
        role_plan["planner_warnings"] = _unique(
            [
                *_string_list(role_plan.get("planner_warnings")),
                incomplete_warning,
            ]
        )
        warning = warning or incomplete_warning
    diagnostics = {
        **_semantic_v3_diagnostics(
            attempts=attempts,
            stage=stage,
            repair_attempted=repair_attempted,
            repair_success=repair_success,
            minimal_router_used=True,
            minimal_fallback_used=source == SEMANTIC_SOURCE_LLM_MINIMAL_FALLBACK,
            empty_response_count=empty_response_count,
            empty_response_reason=classifier_empty_reason,
            requirement_classifier_status=requirement_classifier_status,
            requirement_classifier_error_type=classifier_error_type,
            requirement_classifier_parse_status=classifier_parse_status,
            elapsed_ms=budget.elapsed_ms(),
        ),
        **source_coverage_diagnostics,
    }
    return _with_semantic_diagnostics(
        role_plan,
        source=source,
        confidence=str(semantic_plan.get("confidence") or "low"),
        reason=str(semantic_plan.get("classification_reason") or ""),
        deterministic_product_group_hint=deterministic_hint,
        semantic_plan=semantic_plan,
        warning=warning,
        fallback_reason=semantic_fallback_reason,
        error_type=semantic_error_type,
        parse_status=semantic_parse_status,
        diagnostics=diagnostics,
    )


def plan_universal_requirements(
    spec: StockSpec | Mapping[str, Any] | str | None = None,
    *,
    source_text: str | None = None,
    normalized_spec: Mapping[str, Any] | None = None,
    product_group_profile: ProductGroupProfile | None = None,
    distributor_code: str | None = None,
    planner_client: RequirementPlannerClient | None = None,
) -> dict[str, Any]:
    request_text = source_text or _source_text_from_spec(spec)
    profile = product_group_profile or _profile_for_request(request_text, spec)
    if profile is None:
        return _unknown_product_group_plan(source_text or str(spec or ""))

    base_plan = _deterministic_requirement_plan(
        spec,
        source_text=source_text,
        normalized_spec=normalized_spec,
        product_group_profile=profile,
    )
    warnings = list(base_plan.get("planner_warnings") or [])
    if planner_client is None:
        return base_plan

    payload = {
        "source_text": request_text,
        "normalized_spec": dict(normalized_spec or {}),
        "product_group_profile": _profile_prompt_payload(profile),
        "distributor_code": distributor_code,
        "deterministic_fallback_plan": _safe_plan_for_prompt(base_plan),
    }
    try:
        response = planner_client.generate_json(
            _requirement_planner_system_prompt(profile),
            json.dumps(payload, ensure_ascii=False),
        )
        llm_plan = _coerce_requirement_plan(response, profile=profile)
        llm_plan = _preserve_deterministic_semantic_fallback(
            base_plan,
            llm_plan,
            profile=profile,
        )
        llm_plan["planner_warnings"] = _unique(
            [
                *warnings,
                *_string_list(llm_plan.get("planner_warnings")),
            ]
        )
        return llm_plan
    except (LlmError, ValueError, TypeError) as exc:
        base_plan["planner_warnings"] = _unique(
            [*warnings, f"requirement_planner_llm_fallback:{type(exc).__name__}"]
        )
        return base_plan


def _deterministic_fallback_plan_for_semantic_planner(
    spec: StockSpec | Mapping[str, Any] | str | None,
    *,
    source_text: str | None,
    normalized_spec: Mapping[str, Any] | None,
    deterministic_product_group_hint: str | None,
) -> dict[str, Any]:
    hint = str(deterministic_product_group_hint or "").strip()
    profile = get_product_group_profile(hint) if hint else None
    request_text = source_text or _source_text_from_spec(spec)
    if profile is None:
        profile = _profile_for_request(request_text, spec)
    if profile is None:
        return _unknown_product_group_plan(source_text or str(spec or ""))
    return _deterministic_requirement_plan(
        spec,
        source_text=source_text,
        normalized_spec=normalized_spec,
        product_group_profile=profile,
    )


def _semantic_planner_run_budget(
    *,
    max_seconds: float | None,
    stage_timeout_seconds: float | None,
) -> _SemanticPlannerRunBudget:
    settings = get_llm_settings()
    configured_max = (
        max_seconds
        if max_seconds is not None
        else settings.llm_semantic_planner_max_seconds
    )
    configured_stage = (
        stage_timeout_seconds
        if stage_timeout_seconds is not None
        else settings.llm_semantic_planner_stage_timeout_seconds
    )
    max_value = _positive_float_or_default(configured_max, 300.0)
    stage_value = _positive_float_or_default(configured_stage, 120.0)
    return _SemanticPlannerRunBudget(
        started_at=time.monotonic(),
        max_seconds=max_value,
        stage_timeout_seconds=stage_value,
    )


def _positive_float_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _semantic_profiles_prompt_payload() -> list[dict[str, Any]]:
    return [
        _profile_prompt_payload(profile)
        for group_id, profile in PRODUCT_GROUP_PROFILES.items()
        if group_id in PRIMARY_PRODUCT_GROUPS
    ]


def _run_semantic_intent_router(
    planner_client: RequirementPlannerClient,
    *,
    request_text: str,
    normalized_spec: Mapping[str, Any] | None,
    distributor_code: str | None,
    deterministic_hint: str,
    fallback_plan: Mapping[str, Any],
    attempts: list[dict[str, Any]],
    budget: _SemanticPlannerRunBudget,
) -> dict[str, Any]:
    payload = {
        "source_text": request_text,
        "normalized_spec": dict(normalized_spec or {}),
        "product_group_profiles": _semantic_profiles_prompt_payload(),
        "distributor_code": distributor_code,
        "deterministic_product_group_hint": deterministic_hint,
        "deterministic_fallback_plan": _safe_plan_for_prompt(fallback_plan),
        "contract": {
            "stage": "Semantic Planner V3 Stage A Minimal AI Intent Router",
            "forbidden_fields": sorted(SEMANTIC_FORBIDDEN_KEYS),
            "required_output": _semantic_intent_router_json_contract(),
        },
    }
    result = _call_semantic_stage(
        planner_client,
        system_prompt=_semantic_intent_router_system_prompt(repair=False),
        payload=payload,
        stage=SEMANTIC_STAGE_INTENT_ROUTER,
        attempts=attempts,
        budget=budget,
    )
    if _semantic_stage_timed_out(result):
        return {
            "intent": None,
            "source": SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT,
            "stage": SEMANTIC_STAGE_INTENT_ROUTER,
            "repair_attempted": False,
            "repair_success": False,
            "empty_response_count": 0,
            "empty_response_reason": None,
            "error_type": SEMANTIC_PLANNER_TIMEOUT_ERROR_TYPE,
            "http_status": None,
            "parse_status": SEMANTIC_PLANNER_PARSE_STATUS_TIMEOUT,
            "warning": "semantic_intent_router_timeout",
            "timeout_reason": _text_or_none(result.get("timeout_reason")),
        }
    if result.get("status") == "ok":
        try:
            intent = _coerce_semantic_intent_router_response(
                result["response"],
                fallback_plan=fallback_plan,
            )
            intent["deterministic_product_group_hint"] = deterministic_hint
            if _semantic_intent_is_actionable(intent):
                attempts[-1]["status"] = "actionable"
                return {
                    "intent": intent,
                    "source": SEMANTIC_SOURCE_LLM,
                    "stage": SEMANTIC_STAGE_INTENT_ROUTER,
                    "repair_attempted": False,
                    "repair_success": False,
                    "empty_response_count": 0,
                    "empty_response_reason": None,
                    "error_type": None,
                    "http_status": None,
                    "parse_status": None,
                    "warning": None,
                }
            attempts[-1]["status"] = "not_actionable"
            attempts[-1]["parse_status"] = "not_actionable"
            result = {**result, "status": "not_actionable", "parse_status": "not_actionable"}
        except (ValueError, TypeError) as exc:
            diagnostics = _semantic_error_diagnostics(
                exc,
                parse_status="invalid_contract",
            )
            attempts[-1].update(
                {
                    "status": "invalid",
                    "error_type": diagnostics["error_type"],
                    "parse_status": diagnostics["parse_status"],
                }
            )
            result = {
                **result,
                "status": "invalid",
                "error_type": diagnostics["error_type"],
                "http_status": diagnostics["http_status"],
                "parse_status": diagnostics["parse_status"],
            }

    repair_payload = {
        **payload,
        "repair_instruction": (
            "Your previous intent/router response was empty, invalid, or not actionable. "
            "Return only the small router JSON. Determine product_group and primary_object "
            "from the whole request. Do not invent SKUs, category_id, component_candidate_id, "
            "prices, stock, or availability."
        ),
        "previous_parse_status": result.get("parse_status") or result.get("status"),
    }
    repaired = _call_semantic_stage(
        planner_client,
        system_prompt=_semantic_intent_router_system_prompt(repair=True),
        payload=repair_payload,
        stage=SEMANTIC_STAGE_INTENT_ROUTER_REPAIR,
        attempts=attempts,
        budget=budget,
    )
    empty_count = _empty_response_count(result) + _empty_response_count(repaired)
    if _semantic_stage_timed_out(repaired):
        return {
            "intent": None,
            "source": SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT,
            "stage": SEMANTIC_STAGE_INTENT_ROUTER_REPAIR,
            "repair_attempted": True,
            "repair_success": False,
            "empty_response_count": empty_count,
            "empty_response_reason": _text_or_none(result.get("empty_response_reason")),
            "error_type": SEMANTIC_PLANNER_TIMEOUT_ERROR_TYPE,
            "http_status": None,
            "parse_status": SEMANTIC_PLANNER_PARSE_STATUS_TIMEOUT,
            "warning": "semantic_intent_router_repair_timeout",
            "timeout_reason": _text_or_none(repaired.get("timeout_reason")),
        }
    if repaired.get("status") == "ok":
        try:
            intent = _coerce_semantic_intent_router_response(
                repaired["response"],
                fallback_plan=fallback_plan,
            )
            intent["deterministic_product_group_hint"] = deterministic_hint
            if _semantic_intent_is_actionable(intent):
                attempts[-1]["status"] = "actionable"
                return {
                    "intent": intent,
                    "source": SEMANTIC_SOURCE_LLM_REPAIRED,
                    "stage": SEMANTIC_STAGE_INTENT_ROUTER_REPAIR,
                    "repair_attempted": True,
                    "repair_success": True,
                    "empty_response_count": empty_count,
                    "empty_response_reason": None,
                    "error_type": None,
                    "http_status": None,
                    "parse_status": None,
                    "warning": None,
                }
            attempts[-1]["status"] = "not_actionable"
            attempts[-1]["parse_status"] = "not_actionable"
            repaired = {**repaired, "status": "not_actionable", "parse_status": "not_actionable"}
        except (ValueError, TypeError) as exc:
            diagnostics = _semantic_error_diagnostics(
                exc,
                parse_status="invalid_contract",
            )
            attempts[-1].update(
                {
                    "status": "invalid",
                    "error_type": diagnostics["error_type"],
                    "parse_status": diagnostics["parse_status"],
                }
            )
            repaired = {
                **repaired,
                "status": "invalid",
                "error_type": diagnostics["error_type"],
                "http_status": diagnostics["http_status"],
                "parse_status": diagnostics["parse_status"],
            }

    parse_status = _text_or_none(repaired.get("parse_status")) or _text_or_none(
        result.get("parse_status")
    )
    if parse_status == "empty_response" and empty_count >= 2:
        parse_status = SEMANTIC_EMPTY_RESPONSE_AFTER_REPAIR
    return {
        "intent": None,
        "source": _semantic_source_for_stage_failure(repaired, fallback_empty=True),
        "stage": SEMANTIC_STAGE_INTENT_ROUTER_REPAIR,
        "repair_attempted": True,
        "repair_success": False,
        "empty_response_count": empty_count,
        "empty_response_reason": (
            SEMANTIC_EMPTY_RESPONSE_AFTER_REPAIR
            if empty_count
            else _text_or_none(repaired.get("empty_response_reason"))
        ),
        "error_type": _text_or_none(repaired.get("error_type")),
        "http_status": _as_int(repaired.get("http_status")),
        "parse_status": parse_status,
        "warning": (
            f"semantic_matrix_planner_invalid:{repaired.get('error_type')}"
            if parse_status == "invalid_contract"
            else "semantic_intent_router_failed"
        ),
    }


def _run_semantic_requirement_classifier(
    planner_client: RequirementPlannerClient,
    *,
    request_text: str,
    normalized_spec: Mapping[str, Any] | None,
    distributor_code: str | None,
    intent: Mapping[str, Any],
    profile: ProductGroupProfile,
    attempts: list[dict[str, Any]],
    budget: _SemanticPlannerRunBudget,
    repair: bool = False,
    previous_parse_status: str | None = None,
) -> dict[str, Any]:
    stage = (
        SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER_REPAIR
        if repair
        else SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER
    )
    source_fragments = _semantic_classifier_source_fragments(request_text, intent)
    payload = {
        "source_text": request_text,
        "normalized_spec": dict(normalized_spec or {}),
        "intent_router": _json_safe_mapping(intent),
        "stage_a_output": _json_safe_mapping(intent),
        "source_fragments": source_fragments,
        "deterministic_product_group_hint": (
            intent.get("deterministic_product_group_hint") or intent.get("product_group")
        ),
        "product_group_profiles": _semantic_profiles_prompt_payload(),
        "product_group_profile": _profile_prompt_payload(profile),
        "distributor_code": distributor_code,
        "contract": {
            "stage": "Semantic Planner V3 Stage B Requirement Classifier",
            "forbidden_fields": sorted(SEMANTIC_FORBIDDEN_KEYS),
            "required_output": _semantic_requirement_classifier_json_contract(),
        },
    }
    if repair:
        payload["repair_instruction"] = (
            "You returned empty or non-actionable requirement classification. Use the "
            "already determined product_group and roles from Stage A. The original "
            "request is in source_text, Stage A output is in stage_a_output, and "
            "source_fragments contains extracted request fragments and hints. Classify "
            "every source fragment; do not return only role names. Preserve exact "
            "source_text from the user request whenever the fragment exists there. "
            "Primary object features are not separate BOM roles, but they must be kept "
            "for Composer and post-Composer validation. Accessories may be separate "
            "components or bundle requirements and must keep fulfillment_mode. Do not "
            "invent SKUs. Do not create blocking unmapped for primary object features."
        )
        payload["previous_parse_status"] = previous_parse_status
    result = _call_semantic_stage(
        planner_client,
        system_prompt=_semantic_requirement_classifier_system_prompt(repair=repair),
        payload=payload,
        stage=stage,
        attempts=attempts,
        budget=budget,
    )
    empty_count = _empty_response_count(result)
    if result.get("status") != "ok":
        return {
            "status": _text_or_none(result.get("status")),
            "semantic_plan": None,
            "empty_response_count": empty_count,
            "empty_response_reason": _text_or_none(result.get("empty_response_reason")),
            "error_type": _text_or_none(result.get("error_type")),
            "http_status": _as_int(result.get("http_status")),
            "parse_status": _text_or_none(result.get("parse_status")),
            "timeout_reason": _text_or_none(result.get("timeout_reason")),
            "timeout_seconds": result.get("timeout_seconds"),
        }
    try:
        semantic_plan = _coerce_semantic_classifier_response(
            result["response"],
            intent=intent,
            profile=profile,
        )
    except (ValueError, TypeError) as exc:
        diagnostics = _semantic_error_diagnostics(
            exc,
            parse_status="invalid_contract",
        )
        attempts[-1].update(
            {
                "status": "invalid",
                "error_type": diagnostics["error_type"],
                "parse_status": diagnostics["parse_status"],
            }
        )
        return {
            "status": "invalid",
            "semantic_plan": None,
            "empty_response_count": empty_count,
            "empty_response_reason": None,
            "error_type": diagnostics["error_type"],
            "http_status": diagnostics["http_status"],
            "parse_status": diagnostics["parse_status"],
        }
    if not _semantic_plan_is_actionable(semantic_plan):
        attempts[-1].update({"status": "not_actionable", "parse_status": "not_actionable"})
        return {
            "status": "not_actionable",
            "semantic_plan": None,
            "empty_response_count": empty_count,
            "empty_response_reason": None,
            "error_type": None,
            "http_status": None,
            "parse_status": "not_actionable",
        }
    attempts[-1]["status"] = "actionable"
    return {
        "status": "ok",
        "semantic_plan": semantic_plan,
        "empty_response_count": empty_count,
        "empty_response_reason": None,
        "error_type": None,
        "http_status": None,
        "parse_status": None,
    }


def _call_semantic_stage(
    planner_client: RequirementPlannerClient,
    *,
    system_prompt: str,
    payload: Mapping[str, Any],
    stage: str,
    attempts: list[dict[str, Any]],
    budget: _SemanticPlannerRunBudget,
) -> dict[str, Any]:
    stage_timeout_seconds = budget.stage_timeout_for_call()
    attempt: dict[str, Any] = {
        "stage": stage,
        "status": "started",
        "timeout_seconds": round(stage_timeout_seconds, 3),
    }
    attempts.append(attempt)
    start_elapsed_ms = budget.elapsed_ms()
    progress_event = (
        "semantic_repair_start" if _semantic_stage_is_repair(stage) else "semantic_stage_start"
    )
    _log_semantic_progress(
        progress_event,
        {"stage": stage, "timeout_seconds": round(stage_timeout_seconds, 3)},
    )
    if stage_timeout_seconds <= 0:
        return _semantic_stage_timeout_result(
            attempt,
            stage=stage,
            timeout_seconds=0.0,
            timeout_reason=SEMANTIC_PLANNER_TIMEOUT_REASON_DEADLINE,
            elapsed_ms=budget.elapsed_ms(),
        )

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def run_call() -> None:
        try:
            response = planner_client.generate_json(
                system_prompt,
                json.dumps(payload, ensure_ascii=False),
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced on the caller thread.
            result_queue.put(("error", exc))
            return
        result_queue.put(("ok", response))

    worker = threading.Thread(
        target=run_call,
        name=f"semantic-planner-{stage}",
        daemon=True,
    )
    worker.start()
    worker.join(stage_timeout_seconds)
    if worker.is_alive():
        reason = (
            SEMANTIC_PLANNER_TIMEOUT_REASON_DEADLINE
            if budget.remaining_seconds() <= 0
            else SEMANTIC_PLANNER_TIMEOUT_REASON_STAGE
        )
        return _semantic_stage_timeout_result(
            attempt,
            stage=stage,
            timeout_seconds=stage_timeout_seconds,
            timeout_reason=reason,
            elapsed_ms=budget.elapsed_ms(),
        )

    try:
        status, payload_or_exc = result_queue.get_nowait()
    except queue.Empty:
        return _semantic_stage_timeout_result(
            attempt,
            stage=stage,
            timeout_seconds=stage_timeout_seconds,
            timeout_reason=SEMANTIC_PLANNER_TIMEOUT_REASON_STAGE,
            elapsed_ms=budget.elapsed_ms(),
        )
    if status == "error":
        exc = payload_or_exc
        if not isinstance(exc, (LlmError, ValueError, TypeError)):
            raise exc
        elapsed_ms = max(0, budget.elapsed_ms() - start_elapsed_ms)
        diagnostics = _semantic_error_diagnostics(exc)
        attempt.update(
            {
                "status": "error",
                "error_type": diagnostics["error_type"],
                "http_status": diagnostics["http_status"],
                "parse_status": diagnostics["parse_status"],
            }
        )
        _log_semantic_progress(
            "semantic_repair_done" if _semantic_stage_is_repair(stage) else "semantic_stage_done",
            {
                "stage": stage,
                "status": "error",
                "elapsed_ms": elapsed_ms,
            },
        )
        return {
            "status": "error",
            "error_type": diagnostics["error_type"],
            "http_status": diagnostics["http_status"],
            "parse_status": diagnostics["parse_status"],
        }
    response = payload_or_exc
    if not isinstance(response, Mapping) or not response:
        elapsed_ms = max(0, budget.elapsed_ms() - start_elapsed_ms)
        attempt.update(
            {
                "status": "empty",
                "parse_status": "empty_response",
            }
        )
        _log_semantic_progress(
            "semantic_repair_done" if _semantic_stage_is_repair(stage) else "semantic_stage_done",
            {
                "stage": stage,
                "status": "empty",
                "elapsed_ms": elapsed_ms,
            },
        )
        return {
            "status": "empty",
            "parse_status": "empty_response",
            "empty_response_reason": "llm_empty_response",
        }
    elapsed_ms = max(0, budget.elapsed_ms() - start_elapsed_ms)
    attempt["status"] = "received"
    _log_semantic_progress(
        "semantic_repair_done" if _semantic_stage_is_repair(stage) else "semantic_stage_done",
        {"stage": stage, "status": "received", "elapsed_ms": elapsed_ms},
    )
    return {"status": "ok", "response": response}


def _semantic_stage_timeout_result(
    attempt: dict[str, Any],
    *,
    stage: str,
    timeout_seconds: float,
    timeout_reason: str,
    elapsed_ms: int,
) -> dict[str, Any]:
    attempt.update(
        {
            "status": "timeout",
            "error_type": SEMANTIC_PLANNER_TIMEOUT_ERROR_TYPE,
            "parse_status": SEMANTIC_PLANNER_PARSE_STATUS_TIMEOUT,
            "timeout_seconds": round(timeout_seconds, 3),
            "timeout_reason": timeout_reason,
            "elapsed_ms": elapsed_ms,
        }
    )
    _log_semantic_progress(
        "semantic_repair_timeout" if _semantic_stage_is_repair(stage) else "semantic_stage_timeout",
        {
            "stage": stage,
            "timeout_seconds": round(timeout_seconds, 3),
            "reason": timeout_reason,
            "elapsed_ms": elapsed_ms,
        },
    )
    return {
        "status": "timeout",
        "error_type": SEMANTIC_PLANNER_TIMEOUT_ERROR_TYPE,
        "http_status": None,
        "parse_status": SEMANTIC_PLANNER_PARSE_STATUS_TIMEOUT,
        "timeout_seconds": round(timeout_seconds, 3),
        "timeout_reason": timeout_reason,
    }


def _semantic_stage_timed_out(result: Mapping[str, Any]) -> bool:
    return (
        result.get("status") == "timeout"
        or result.get("error_type") == SEMANTIC_PLANNER_TIMEOUT_ERROR_TYPE
    )


def _semantic_stage_is_repair(stage: str) -> bool:
    return stage in {
        SEMANTIC_STAGE_INTENT_ROUTER_REPAIR,
        SEMANTIC_STAGE_REQUIREMENT_CLASSIFIER_REPAIR,
    }


def _log_semantic_progress(event: str, fields: Mapping[str, Any] | None = None) -> None:
    parts = [event]
    for key, value in (fields or {}).items():
        if value in (None, "", [], {}):
            continue
        text = str(value).replace("\n", " ").replace("\r", " ")
        parts.append(f"{key}={text}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _semantic_source_for_stage_failure(
    result: Mapping[str, Any],
    *,
    fallback_empty: bool,
) -> str:
    if _semantic_stage_timed_out(result):
        return SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT
    if fallback_empty and result.get("status") in {"empty", "not_actionable"}:
        return SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_EMPTY
    if result.get("status") == "invalid":
        return SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_INVALID
    if result.get("status") == "error":
        return SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_ERROR
    return SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_EMPTY


def _semantic_stage_fallback_reason(result: Mapping[str, Any]) -> str:
    status = str(result.get("source") or result.get("status") or "")
    parse_status = str(result.get("parse_status") or "")
    if _semantic_stage_timed_out(result) or parse_status == SEMANTIC_PLANNER_PARSE_STATUS_TIMEOUT:
        return SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
    if "invalid" in status or parse_status == "invalid_contract":
        return "llm_invalid_contract"
    if "error" in status:
        return "llm_call_failed"
    if parse_status == SEMANTIC_EMPTY_RESPONSE_AFTER_REPAIR:
        return "llm_empty_response"
    if "empty" in status or parse_status == "empty_response":
        return "llm_empty_response"
    if parse_status == "not_actionable":
        return "llm_not_actionable"
    return "llm_semantic_planner_failed"


def _empty_response_count(result: Mapping[str, Any]) -> int:
    return 1 if result.get("status") == "empty" else 0


def _semantic_v3_diagnostics(
    *,
    attempts: Sequence[Mapping[str, Any]],
    stage: str,
    repair_attempted: bool,
    repair_success: bool,
    minimal_router_used: bool,
    minimal_fallback_used: bool,
    empty_response_count: int,
    empty_response_reason: str | None,
    requirement_classifier_status: str,
    requirement_classifier_error_type: str | None,
    requirement_classifier_parse_status: str | None,
    elapsed_ms: int | None = None,
) -> dict[str, Any]:
    stage_timeouts = [
        {
            "stage": _text_or_none(row.get("stage")),
            "timeout_seconds": row.get("timeout_seconds"),
            "timeout_reason": _text_or_none(row.get("timeout_reason")),
            "elapsed_ms": row.get("elapsed_ms"),
        }
        for row in attempts
        if row.get("status") == "timeout"
    ]
    timeout_reason = next(
        (
            _text_or_none(row.get("timeout_reason"))
            for row in attempts
            if row.get("status") == "timeout"
        ),
        None,
    )
    timeout_seconds = next(
        (
            row.get("timeout_seconds")
            for row in attempts
            if row.get("status") == "timeout"
        ),
        None,
    )
    return {
        "semantic_planner_attempts": [_json_safe_mapping(row) for row in attempts],
        "semantic_planner_stage": stage,
        "semantic_planner_stage_timeouts": stage_timeouts,
        "semantic_planner_timeout_reason": timeout_reason,
        "semantic_planner_timeout_seconds": timeout_seconds,
        "semantic_planner_elapsed_ms": elapsed_ms if stage_timeouts else None,
        "semantic_planner_repair_attempted": repair_attempted,
        "semantic_planner_repair_success": repair_success,
        "semantic_planner_minimal_router_used": minimal_router_used,
        "semantic_planner_minimal_fallback_used": minimal_fallback_used,
        "semantic_planner_empty_response_count": empty_response_count,
        "semantic_planner_empty_response_reason": empty_response_reason,
        "requirement_classifier_status": requirement_classifier_status,
        "requirement_classifier_error_type": requirement_classifier_error_type,
        "requirement_classifier_parse_status": requirement_classifier_parse_status,
    }


def _semantic_intent_router_json_contract() -> dict[str, Any]:
    return {
        "product_group": "server|network|storage|unknown",
        "primary_object": (
            "server|switch|router|firewall|access_point|storage_system|nas|"
            "dac_cable|transceiver|other"
        ),
        "language": "short language tag such as ru or en",
        "complexity": "simple|medium|complex",
        "required_bom_roles_guess": ["role ids from the selected product group profile"],
        "primary_object_feature_hints": ["raw feature text, not purchasable lines"],
        "accessory_hints": ["raw accessory/consumable text"],
        "service_support_hints": ["raw service/support text"],
        "logistics_hints": ["raw logistics/commercial text"],
        "confidence": "high|medium|low",
        "reason": "short safe explanation",
    }


def _semantic_requirement_classifier_json_contract() -> dict[str, Any]:
    return {
        "classified_requirements": _semantic_matrix_json_contract()[
            "classified_requirements"
        ],
        "matrix_blueprint": "optional; roles only when useful",
        "required_capabilities": "optional generic capabilities",
        "optional_capabilities": "optional generic capabilities",
        "embedded_requirements": [],
        "not_primary_product_groups": [],
        "logistics_constraints": {},
        "commercial_instructions": [],
        "response_instructions": [],
        "engineer_review_instructions": [],
        "unsupported_or_unmapped_requirements": [],
        "planner_warnings": [],
    }


def _semantic_matrix_json_contract() -> dict[str, Any]:
    return {
        "primary_product_group": "server|network|storage|unknown",
        "primary_object": (
            "server|switch|router|firewall|access_point|storage_system|nas|"
            "dac_cable|transceiver|other"
        ),
        "confidence": "high|medium|low",
        "classification_reason": "short safe explanation",
        "matrix_blueprint": {
            "roles": [
                {
                    "role": "role from selected profile or unmapped",
                    "required": True,
                    "source_text": "exact request fragment",
                    "characteristics_to_match": {},
                    "hard_capability_ids": [],
                }
            ]
        },
        "required_capabilities": [],
        "optional_capabilities": [],
        "classified_requirements": [
            {
                "requirement_id": "stable id",
                "source_text": "exact request fragment",
                "classification": (
                    "purchasable_component_role|primary_object_feature|"
                    "accessory_or_consumable|service_or_support|"
                    "logistics_or_commercial_constraint|engineering_check|"
                    "out_of_scope_or_unmapped_non_blocking|"
                    "blocking_unmapped_purchasable_role"
                ),
                "product_group": "selected product group",
                "target_role": "known role when applicable",
                "target_primary_object": "primary object when applicable",
                "hard_or_optional": "hard|optional",
                "reason": "short explanation",
                "confidence": "high|medium|low",
                "fulfillment_mode": (
                    "separate_component_required|included_in_primary_object|"
                    "included_in_selected_component|included_in_bundle_or_kit|"
                    "service_or_support|logistics_constraint|engineering_check_only|"
                    "unverified_requires_confirmation|not_applicable"
                ),
                "fulfillment_target_role": "role that closes the requirement when applicable",
                "fulfillment_target_component_candidate_id": (
                    "optional; only when a later Composer-selected candidate is known"
                ),
                "evidence_source": (
                    "request_text|product_card|package_json|content_properties|"
                    "ai_reasoning|none"
                ),
                "evidence_text": "short quote/fact proving the fulfillment mode, or empty",
                "should_create_bom_role": False,
                "should_block_before_composer": False,
                "should_appear_in_composer_brief": True,
                "should_validate_after_composer": True,
                "should_be_validated_after_composer": True,
                "engineer_check_ru": "",
                "suggested_engineer_check_ru": "",
                "category_needed": False,
                "parsed_requirements": {},
            }
        ],
        "embedded_requirements": [],
        "not_primary_product_groups": [],
        "logistics_constraints": {},
        "commercial_instructions": [],
        "response_instructions": [],
        "engineer_review_instructions": [],
        "unsupported_or_unmapped_requirements": [],
        "planner_warnings": [],
    }


def _semantic_matrix_planner_system_prompt() -> str:
    return (
        "You are AI Semantic Matrix Planner V2. Return strict JSON only. "
        "You are the semantic authority for primary_product_group, primary_object, "
        "roles, and characteristics_to_match. Application code will only validate your "
        "contract and materialize it against the real distributor catalog. Determine the "
        "primary product group from the whole request meaning, not from isolated tokens. "
        "Separate embedded requirements from the top-level product group: a NIC, SFP+, "
        "power cables, PSU, fans, controllers, or management ports inside a server do not "
        "make the request a standalone network/cable/storage request. Do not choose or "
        "mention category_id, component_candidate_id, prices, stock, or availability. "
        "Classify every explicit requirement in classified_requirements. Not every hard "
        "requirement is a separate purchasable BOM role: platform/device/system features "
        "must be primary_object_feature attached to the relevant target_role or "
        "target_primary_object. For every classified requirement choose fulfillment_mode "
        "and set should_create_bom_role=true only for requirements that need a separate "
        "BOM role/category/matrix line. If you claim the requirement is included in a "
        "platform, bundle, kit, or selected component, cite evidence_source and "
        "evidence_text; otherwise use unverified_requires_confirmation and require an "
        "engineer check. Only separate BOM items become required roles. Do not "
        "create role=\"unmapped\" for platform features such as cooling redundancy, "
        "USB/VGA/management ports, form factor, bays, socket count, switch PoE/L3/"
        "stacking/ports, storage usable/raw capacity/RAID/controllers/host ports, or "
        "UPS power/runtime/form factor. If unsure, classify as engineering_check or "
        "out_of_scope_or_unmapped_non_blocking, not as a blocking role. Use "
        "blocking_unmapped_purchasable_role only when you explicitly decide the fragment "
        "needs a separate purchasable line but no role/category can be mapped. "
        "matrix_blueprint.roles is the candidate-matrix contract and must contain only "
        "separate selectable roles plus primary object roles that need candidates. "
        "Example 1: complex 1U 2-socket server with Intel CPUs, DDR5 RDIMM, SATA SSDs, "
        "LSI HBA, Intel X710-DA2 2x10GbE SFP+, 2x2000W PSU, C13-C14/C13-Schuko cables, "
        "fans, USB/serial/VGA/management. Expected primary_product_group=server, "
        "primary_object=server, roles include server_platform, cpu, ram, storage, "
        "storage_controller, network_adapter, power_supply, cable or other_accessory, "
        "and cooling/management as primary_object_feature target_role=server_platform, "
        "not role=unmapped. "
        "not_primary_product_groups must say network because SFP+ belongs to a NIC inside "
        "the server, and network cable/DAC because C13-C14/C13-Schuko are power cables. "
        "Server examples: '8 fans N+1' -> primary_object_feature target_role="
        "server_platform; 'USB 3.0 x3, VGA, RJ-45 management' -> "
        "primary_object_feature target_role=server_platform; '2 x Intel Xeon' -> "
        "purchasable_component_role target_role=cpu; 'C13-C14 cables' -> "
        "accessory_or_consumable target_role=cable when separate cables are requested. "
        "Example 2: \"Нужен 1 коммутатор 48 портов 1G PoE, 4 uplink 10G SFP+, L3\" "
        "=> primary_product_group=network, primary_object=switch, role=switch, "
        "PoE/L3/stacking/port count as primary_object_feature target_role=switch. "
        "Network SFP+ modules 4 pcs -> accessory_or_consumable target_role=transceiver; "
        "support 3 years -> service_or_support. "
        "Example 3: \"Нужен DAC SFP+ 10G 3 м\" => primary_product_group=network, "
        "primary_object=dac_cable, role=dac_cable. "
        "Example 4: \"Нужна СХД 100 ТБ usable, 2 контроллера, SSD, FC 32G\" "
        "=> primary_product_group=storage, primary_object=storage_system, roles include "
        "storage_system, controller, drive/ssd, host_port/protocol_module, support if "
        "requested; usable capacity/RAID/controller count/host ports are "
        "primary_object_feature of storage_system unless separate shelves/modules are "
        "requested. "
        "UPS examples for future groups: '10kVA online rack' is primary_object_feature "
        "target_role=ups; 'battery module' is purchasable_component_role "
        "target_role=battery_module; 'SNMP card' is accessory_or_consumable "
        "target_role=management_card. "
        "Example 5: \"Нужен сервер 2U с Intel X710-DA2 2x10GbE SFP+\" "
        "=> primary_product_group=server, role network_adapter, not network switch/cable. "
        "Return exactly the requested JSON shape."
    )


def _semantic_intent_router_system_prompt(*, repair: bool) -> str:
    repair_text = (
        "This is a repair attempt after an empty, invalid, or non-actionable router "
        "answer. "
        if repair
        else ""
    )
    return (
        "You are Semantic Planner V3 Stage A Minimal AI Intent Router. "
        "Compatibility marker: AI Semantic Matrix Planner V2. Return strict JSON only. "
        + repair_text
        + "Read the whole request and return only the small intent/router contract. "
        "Decide product_group and primary_object first. Separate primary-object "
        "features from separate BOM roles: form factor, socket count, bays, cooling "
        "redundancy, management ports, switch port/L3/PoE/stacking properties, and "
        "storage usable capacity/RAID are feature hints, not blocking unmapped roles. "
        "required_bom_roles_guess must contain only role ids from the selected product "
        "group profile. If a fragment looks like a feature, put its raw text in "
        "primary_object_feature_hints. Do not choose or mention category_id, "
        "component_candidate_id, prices, stock, or availability."
    )


def _semantic_requirement_classifier_system_prompt(*, repair: bool) -> str:
    repair_text = (
        "This is a repair attempt. You returned empty or non-actionable requirement "
        "classification. Use the already determined product_group and roles from "
        "Stage A. The payload includes source_text with the original request, "
        "stage_a_output with Stage A, and source_fragments with extracted request "
        "fragments/hints. Classify every source fragment; do not return only role "
        "names. Preserve exact source_text from the user request when original text "
        "exists. Primary object features are not separate BOM roles, but must be kept "
        "for Composer and validation. Accessories may be separate components or bundle "
        "requirements with fulfillment_mode. "
        if repair
        else ""
    )
    return (
        "You are Semantic Planner V3 Stage B AI Requirement Classifier. "
        "Compatibility marker: AI Semantic Matrix Planner V2. Return strict JSON only. "
        + repair_text
        + "Classify each explicit requirement from the original request into one of: "
        "purchasable_component_role, primary_object_feature, accessory_or_consumable, "
        "service_or_support, logistics_or_commercial_constraint, engineering_check, "
        "out_of_scope_or_unmapped_non_blocking, blocking_unmapped_purchasable_role. "
        "For each requirement choose fulfillment_mode: separate_component_required, "
        "included_in_primary_object, included_in_selected_component, "
        "included_in_bundle_or_kit, service_or_support, logistics_constraint, "
        "engineering_check_only, unverified_requires_confirmation, or not_applicable. "
        "Set should_create_bom_role=true only when a separate role must be sent to "
        "category planning and Composer. If you claim inclusion in a platform, bundle, "
        "kit, or selected component, cite evidence_source and evidence_text; if the "
        "evidence is absent or only assumed, use unverified_requires_confirmation and "
        "add engineer_check_ru. "
        "Do not invent SKUs. Do not choose or mention category_id, "
        "component_candidate_id, prices, stock, or availability. Do not create "
        "blocking unmapped for primary object features. Separate BOM items become "
        "purchasable roles; device/platform/system properties remain "
        "primary_object_feature and should not block before composer."
    )


def _coerce_semantic_intent_router_response(
    plan: Mapping[str, Any],
    *,
    fallback_plan: Mapping[str, Any],
) -> dict[str, Any]:
    forbidden_paths = _forbidden_semantic_key_paths(plan)
    if forbidden_paths:
        raise ValueError(f"semantic_intent_forbidden_fields:{','.join(forbidden_paths[:5])}")
    product_group = _normalize_primary_product_group(
        plan.get("product_group") or plan.get("primary_product_group")
    )
    primary_object = str(plan.get("primary_object") or "other").strip()
    if primary_object not in PRIMARY_OBJECTS:
        primary_object = "other"
    confidence = str(plan.get("confidence") or "low").strip().casefold()
    if confidence not in SEMANTIC_CONFIDENCE_VALUES:
        confidence = "low"
    roles = _semantic_router_roles(
        plan.get("required_bom_roles_guess"),
        product_group=product_group,
    )
    if not roles:
        roles = _semantic_router_roles_from_matrix_plan(plan, product_group=product_group)
    feature_hints = _semantic_hint_rows(plan.get("primary_object_feature_hints"))
    if not feature_hints:
        feature_hints = _semantic_feature_hints_from_classified(plan)
    return {
        "product_group": product_group,
        "primary_object": primary_object,
        "language": str(plan.get("language") or "").strip(),
        "complexity": str(plan.get("complexity") or "").strip() or "unknown",
        "required_bom_roles_guess": roles,
        "primary_object_feature_hints": feature_hints,
        "accessory_hints": _semantic_hint_rows(plan.get("accessory_hints")),
        "service_support_hints": _semantic_hint_rows(plan.get("service_support_hints")),
        "logistics_hints": _semantic_hint_rows(plan.get("logistics_hints")),
        "confidence": confidence,
        "reason": str(
            plan.get("reason") or plan.get("classification_reason") or ""
        ).strip(),
    }


def _semantic_intent_is_actionable(intent: Mapping[str, Any]) -> bool:
    if intent.get("product_group") not in PRIMARY_PRODUCT_GROUPS:
        return False
    if str(intent.get("primary_object") or "").strip() in {"", "other"}:
        return False
    return bool(
        _string_list(intent.get("required_bom_roles_guess"))
        or _safe_semantic_list(intent.get("primary_object_feature_hints"))
        or str(intent.get("confidence") or "") in {"high", "medium"}
    )


def _coerce_semantic_classifier_response(
    response: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    profile: ProductGroupProfile,
) -> dict[str, Any]:
    if response.get("primary_product_group") or response.get("product_group"):
        merged = {
            "primary_product_group": intent.get("product_group"),
            "primary_object": intent.get("primary_object"),
            "confidence": intent.get("confidence"),
            "classification_reason": intent.get("reason"),
            **dict(response),
        }
        return _coerce_semantic_matrix_plan(merged)
    semantic_plan = _semantic_matrix_plan_from_intent_and_classifier(
        intent=intent,
        classifier=response,
        profile=profile,
    )
    return _coerce_semantic_matrix_plan(semantic_plan)


def _semantic_matrix_plan_from_intent_and_classifier(
    *,
    intent: Mapping[str, Any],
    classifier: Mapping[str, Any],
    profile: ProductGroupProfile,
) -> dict[str, Any]:
    product_group = str(intent.get("product_group") or "unknown").strip()
    primary_object = str(intent.get("primary_object") or "other").strip()
    classified = _coerce_classified_requirements(
        classifier.get("classified_requirements") or classifier.get("requirements"),
        profile=profile,
        product_group=product_group,
        primary_object=primary_object,
    )
    matrix_blueprint = classifier.get("matrix_blueprint")
    if not _mapping_rows(_mapping(matrix_blueprint).get("roles")):
        matrix_blueprint = _matrix_blueprint_from_intent_and_classified(
            intent,
            classified,
            profile=profile,
        )
    return {
        "primary_product_group": product_group,
        "primary_object": primary_object,
        "confidence": str(
            classifier.get("confidence") or intent.get("confidence") or "low"
        ),
        "classification_reason": str(
            classifier.get("reason")
            or classifier.get("classification_reason")
            or intent.get("reason")
            or ""
        ),
        "matrix_blueprint": matrix_blueprint,
        "required_capabilities": _mapping_rows(classifier.get("required_capabilities")),
        "optional_capabilities": _mapping_rows(classifier.get("optional_capabilities")),
        "classified_requirements": classified,
        "embedded_requirements": _safe_semantic_list(
            classifier.get("embedded_requirements")
        ),
        "not_primary_product_groups": _safe_semantic_list(
            classifier.get("not_primary_product_groups")
        ),
        "logistics_constraints": _mapping(classifier.get("logistics_constraints")),
        "commercial_instructions": _safe_semantic_list(
            classifier.get("commercial_instructions")
        ),
        "response_instructions": _string_list(classifier.get("response_instructions")),
        "engineer_review_instructions": _string_list(
            classifier.get("engineer_review_instructions")
        ),
        "unsupported_or_unmapped_requirements": _string_list(
            classifier.get("unsupported_or_unmapped_requirements")
        ),
        "planner_warnings": _string_list(classifier.get("planner_warnings")),
    }


def _minimal_semantic_plan_from_intent(
    intent: Mapping[str, Any],
    *,
    request_text: str,
    profile: ProductGroupProfile,
) -> dict[str, Any]:
    product_group = str(intent.get("product_group") or profile.product_group_id).strip()
    primary_object = str(intent.get("primary_object") or "other").strip()
    primary_role = _primary_object_role(profile, primary_object) or (
        profile.required_roles[0] if profile.required_roles else ""
    )
    classified: list[dict[str, Any]] = []
    for index, hint in enumerate(
        _safe_semantic_list(intent.get("primary_object_feature_hints")),
        start=1,
    ):
        text = _semantic_hint_text(hint)
        if not text:
            continue
        classified.append(
            {
                "requirement_id": f"minimal_feature_{index}",
                "source_text": text,
                "classification": REQ_CLASS_PRIMARY_OBJECT_FEATURE,
                "product_group": product_group,
                "target_role": primary_role,
                "target_primary_object": primary_object,
                "hard_or_optional": REQ_HARD,
                "reason": "Carried from Stage A router as an unverified primary-object feature.",
                "confidence": intent.get("confidence") or "medium",
                "should_block_before_composer": False,
                "should_appear_in_composer_brief": True,
                "should_be_validated_after_composer": True,
                "category_needed": False,
                "parsed_requirements": {"unverified_constraint": True},
            }
        )
    for index, hint in enumerate(
        _safe_semantic_list(intent.get("accessory_hints")),
        start=1,
    ):
        text = _semantic_hint_text(hint)
        role = _role_from_hint(hint, profile=profile) or ""
        if not text:
            continue
        classified.append(
            {
                "requirement_id": f"minimal_accessory_{index}",
                "source_text": text,
                "classification": REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
                "product_group": product_group,
                "target_role": role,
                "target_primary_object": primary_object,
                "hard_or_optional": REQ_HARD,
                "reason": "Carried from Stage A router as an unverified accessory hint.",
                "confidence": intent.get("confidence") or "medium",
                "category_needed": bool(role),
                "parsed_requirements": {"unverified_constraint": True},
            }
        )
    for index, hint in enumerate(
        _safe_semantic_list(intent.get("service_support_hints")),
        start=1,
    ):
        text = _semantic_hint_text(hint)
        role = "support" if "support" in profile.role_catalog else ""
        if not text:
            continue
        classified.append(
            {
                "requirement_id": f"minimal_service_{index}",
                "source_text": text,
                "classification": REQ_CLASS_SERVICE_OR_SUPPORT,
                "product_group": product_group,
                "target_role": role,
                "target_primary_object": primary_object,
                "hard_or_optional": REQ_HARD,
                "reason": "Carried from Stage A router as an unverified service/support hint.",
                "confidence": intent.get("confidence") or "medium",
                "category_needed": bool(role),
                "parsed_requirements": {"unverified_constraint": True},
            }
        )
    for index, hint in enumerate(
        _safe_semantic_list(intent.get("logistics_hints")),
        start=1,
    ):
        text = _semantic_hint_text(hint)
        if not text:
            continue
        classified.append(
            {
                "requirement_id": f"minimal_logistics_{index}",
                "source_text": text,
                "classification": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
                "product_group": product_group,
                "target_role": "",
                "target_primary_object": primary_object,
                "hard_or_optional": REQ_OPTIONAL,
                "reason": "Carried from Stage A router as an unverified logistics/commercial hint.",
                "confidence": intent.get("confidence") or "medium",
                "category_needed": False,
                "parsed_requirements": {"unverified_constraint": True},
            }
        )
    return _coerce_semantic_matrix_plan(
        {
            "primary_product_group": product_group,
            "primary_object": primary_object,
            "confidence": str(intent.get("confidence") or "medium"),
            "classification_reason": str(
                intent.get("reason")
                or "Minimal AI semantic fallback from Stage A router."
            ),
            "matrix_blueprint": _matrix_blueprint_from_intent_and_classified(
                intent,
                classified,
                profile=profile,
                request_text=request_text,
            ),
            "required_capabilities": [],
            "optional_capabilities": [],
            "classified_requirements": classified,
            "embedded_requirements": [],
            "not_primary_product_groups": [],
            "logistics_constraints": {},
            "commercial_instructions": [],
            "response_instructions": [],
            "engineer_review_instructions": [],
            "unsupported_or_unmapped_requirements": [],
            "planner_warnings": ["semantic_requirement_classifier_partial"],
        }
    )


def _matrix_blueprint_from_intent_and_classified(
    intent: Mapping[str, Any],
    classified_requirements: Sequence[Mapping[str, Any]],
    *,
    profile: ProductGroupProfile,
    request_text: str | None = None,
) -> dict[str, Any]:
    roles = _semantic_router_roles(
        intent.get("required_bom_roles_guess"),
        product_group=profile.product_group_id,
    )
    for row in classified_requirements:
        classification = str(row.get("classification") or "")
        if classification not in {
            REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
            REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
            REQ_CLASS_SERVICE_OR_SUPPORT,
        }:
            continue
        if not bool(row.get("should_create_bom_role", True)):
            continue
        role = _normalize_semantic_role(
            row.get("target_role"),
            product_group=profile.product_group_id,
        )
        if role in profile.role_catalog and role not in roles:
            roles.append(role)
    blueprint_rows: list[dict[str, Any]] = []
    for role in roles:
        source_texts = [
            str(row.get("source_text") or "").strip()
            for row in classified_requirements
            if _normalize_semantic_role(
                row.get("target_role"),
                product_group=profile.product_group_id,
            )
            == role
            and str(row.get("classification") or "")
            in {
                REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
                REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
                REQ_CLASS_SERVICE_OR_SUPPORT,
            }
            and bool(row.get("should_create_bom_role", True))
        ]
        source_text = "; ".join(_unique([text for text in source_texts if text]))
        if not source_text:
            source_text = role
        parsed = {"unverified_constraints": source_texts} if source_texts else {}
        if request_text and not source_texts:
            parsed = {"unverified_from_router": True}
        blueprint_rows.append(
            {
                "role": role,
                "required": True,
                "source_text": source_text,
                "characteristics_to_match": parsed,
                "hard_capability_ids": [f"{role}.requested"],
            }
        )
    return {"roles": blueprint_rows}


def _semantic_router_roles(
    value: Any,
    *,
    product_group: str,
) -> list[str]:
    profile = get_product_group_profile(product_group)
    if profile is None:
        return []
    roles: list[str] = []
    for item in _safe_semantic_list(value):
        role_value: Any
        if isinstance(item, Mapping):
            role_value = item.get("role") or item.get("role_id") or item.get("target_role")
        else:
            role_value = item
        role = _normalize_semantic_role(role_value, product_group=product_group)
        if role in profile.role_catalog and role not in roles:
            roles.append(role)
    return roles


def _semantic_router_roles_from_matrix_plan(
    plan: Mapping[str, Any],
    *,
    product_group: str,
) -> list[str]:
    matrix_blueprint = _mapping(plan.get("matrix_blueprint"))
    return _semantic_router_roles(
        [
            row.get("role") or row.get("role_id")
            for row in _mapping_rows(matrix_blueprint.get("roles"))
        ],
        product_group=product_group,
    )


def _semantic_hint_rows(value: Any) -> list[Any]:
    rows: list[Any] = []
    if not isinstance(value, list | tuple):
        return rows
    for item in value:
        if isinstance(item, Mapping):
            safe = _json_safe_mapping(item)
            if _semantic_hint_text(safe):
                rows.append(safe)
            continue
        text = str(item or "").strip()
        if text:
            rows.append(text)
    return rows


def _semantic_feature_hints_from_classified(plan: Mapping[str, Any]) -> list[Any]:
    rows: list[Any] = []
    for row in _mapping_rows(plan.get("classified_requirements")):
        if str(row.get("classification") or "") != REQ_CLASS_PRIMARY_OBJECT_FEATURE:
            continue
        text = str(row.get("source_text") or row.get("requirement_text") or "").strip()
        if text:
            rows.append(text)
    return rows


def _semantic_hint_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("source_text", "text", "requirement_text", "hint", "role"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return ""
    return str(value or "").strip()


def _role_from_hint(value: Any, *, profile: ProductGroupProfile) -> str | None:
    if not isinstance(value, Mapping):
        return None
    role = _normalize_semantic_role(
        value.get("target_role") or value.get("role") or value.get("role_id"),
        product_group=profile.product_group_id,
    )
    return role if role in profile.role_catalog else None


def _semantic_classifier_source_fragments(
    request_text: str,
    intent: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(text: str, source: str) -> None:
        for fragment in _split_requirement_source_fragment(text):
            if not _source_fragment_has_requirement_signal(fragment):
                continue
            normalized = _normalize_requirement_source(fragment)
            if not normalized:
                continue
            if any(
                _normalize_requirement_source(row.get("source_text")) == normalized
                for row in rows
            ):
                continue
            rows.append({"source_text": fragment, "source": source})

    for key in (
        "primary_object_feature_hints",
        "accessory_hints",
        "service_support_hints",
        "logistics_hints",
    ):
        for hint in _safe_semantic_list((intent or {}).get(key)):
            add(_semantic_hint_text(hint), f"stage_a.{key}")

    for line in str(request_text or "").splitlines():
        add(line, "request_text")

    if not rows:
        add(request_text, "request_text")
    return rows


def _split_requirement_source_fragment(text: str) -> list[str]:
    stripped = " ".join(str(text or "").strip().split())
    if not stripped:
        return []
    if stripped.endswith(":") and not stripped.split(":", 1)[-1].strip():
        return []
    first_pass = [
        part.strip(" ,;")
        for part in re.split(r"\s*;\s*", stripped)
        if part.strip(" ,;")
    ]
    result: list[str] = []
    for part in first_pass or [stripped]:
        comma_parts = [
            item.strip(" ,")
            for item in re.split(r"\s*,\s*", part)
            if item.strip(" ,")
        ]
        meaningful_parts = [
            item for item in comma_parts if _source_fragment_has_requirement_signal(item)
        ]
        if len(meaningful_parts) >= 2:
            result.extend(meaningful_parts)
        else:
            result.append(part)
    return _unique([item for item in result if item])


def _source_fragment_has_requirement_signal(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 2:
        return False
    if stripped.endswith(":") and not stripped.split(":", 1)[-1].strip():
        return False
    if re.search(r"\d", stripped):
        return True
    if re.search(r"[A-ZА-ЯЁ]{2,}", stripped):
        return True
    if re.search(r"[+/-]", stripped):
        return True
    if ":" in stripped and stripped.split(":", 1)[-1].strip():
        return True
    return _known_requirement_fragment(stripped)


def _coerce_semantic_matrix_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_paths = _forbidden_semantic_key_paths(plan)
    if forbidden_paths:
        raise ValueError(f"semantic_plan_forbidden_fields:{','.join(forbidden_paths[:5])}")

    primary_product_group = _normalize_primary_product_group(
        plan.get("primary_product_group") or plan.get("product_group")
    )
    primary_object = str(plan.get("primary_object") or "other").strip()
    if primary_object not in PRIMARY_OBJECTS:
        primary_object = "other"
    confidence = str(plan.get("confidence") or "low").strip().casefold()
    if confidence not in SEMANTIC_CONFIDENCE_VALUES:
        confidence = "low"
    matrix_blueprint = _coerce_matrix_blueprint(
        plan.get("matrix_blueprint"),
        product_group=primary_product_group,
    )
    profile = get_product_group_profile(primary_product_group)
    required_capabilities = _coerce_semantic_capabilities(
        plan.get("required_capabilities"),
        product_group=primary_product_group,
        hard=True,
    )
    optional_capabilities = _coerce_semantic_capabilities(
        plan.get("optional_capabilities"),
        product_group=primary_product_group,
        hard=False,
    )
    classified_requirements = _coerce_classified_requirements(
        plan.get("classified_requirements"),
        profile=profile,
        product_group=primary_product_group,
        primary_object=primary_object,
    )
    unsupported = _string_list(plan.get("unsupported_or_unmapped_requirements"))
    return {
        "primary_product_group": primary_product_group,
        "primary_object": primary_object,
        "confidence": confidence,
        "classification_reason": str(plan.get("classification_reason") or "").strip(),
        "matrix_blueprint": matrix_blueprint,
        "required_capabilities": required_capabilities,
        "optional_capabilities": optional_capabilities,
        "classified_requirements": classified_requirements,
        "embedded_requirements": _safe_semantic_list(plan.get("embedded_requirements")),
        "not_primary_product_groups": _safe_semantic_list(
            plan.get("not_primary_product_groups")
        ),
        "logistics_constraints": _mapping(plan.get("logistics_constraints")),
        "commercial_instructions": _safe_semantic_list(plan.get("commercial_instructions")),
        "response_instructions": _string_list(plan.get("response_instructions")),
        "engineer_review_instructions": _string_list(
            plan.get("engineer_review_instructions")
        ),
        "unsupported_or_unmapped_requirements": unsupported,
        "planner_warnings": _string_list(plan.get("planner_warnings")),
        "role_catalog": list(profile.role_catalog) if profile is not None else [],
    }


def _forbidden_semantic_key_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            key_path = f"{prefix}.{key_text}" if prefix else key_text
            if key_text.strip().casefold() in SEMANTIC_FORBIDDEN_KEYS:
                paths.append(key_path)
            paths.extend(_forbidden_semantic_key_paths(item, prefix=key_path))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            item_path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_forbidden_semantic_key_paths(item, prefix=item_path))
    return paths


def _coerce_matrix_blueprint(
    value: Any,
    *,
    product_group: str,
) -> dict[str, Any]:
    raw = _mapping(value)
    profile = get_product_group_profile(product_group)
    allowed_roles = set(profile.role_catalog) if profile is not None else set()
    roles: list[dict[str, Any]] = []
    for row in _mapping_rows(raw.get("roles")):
        source_text = str(row.get("source_text") or row.get("requirement_text") or "").strip()
        original_role = str(row.get("role") or row.get("role_id") or "").strip()
        role = _normalize_semantic_role(original_role, product_group=product_group)
        if role not in allowed_roles and role != UNMAPPED_ROLE:
            role = UNMAPPED_ROLE if (role or source_text) else ""
        characteristics = _mapping(
            row.get("characteristics_to_match") or row.get("parsed_requirements")
        )
        hard_capability_ids = _string_list(row.get("hard_capability_ids"))
        if not role and not source_text and not characteristics and not hard_capability_ids:
            continue
        result = {
            "role": role or UNMAPPED_ROLE,
            "required": _semantic_required_value(row.get("required")),
            "source_text": source_text,
            "characteristics_to_match": characteristics,
            "hard_capability_ids": hard_capability_ids,
        }
        if original_role and original_role != result["role"]:
            result["original_role"] = original_role
        roles.append(result)
    return {"roles": _unique_blueprint_roles(roles)}


def _coerce_semantic_capabilities(
    value: Any,
    *,
    product_group: str,
    hard: bool,
) -> list[dict[str, Any]]:
    profile = get_product_group_profile(product_group)
    if profile is None:
        return []
    allowed_roles = set(profile.role_catalog)
    result: list[dict[str, Any]] = []
    for row in _mapping_rows(value):
        original_role = str(row.get("role") or row.get("role_id") or "").strip()
        role = _normalize_semantic_role(original_role, product_group=product_group)
        normalized = dict(row)
        normalized["role"] = role
        capability = _coerce_capability(normalized, allowed_roles=allowed_roles, hard=hard)
        if capability is None:
            continue
        if original_role and _normalize_semantic_role(
            original_role,
            product_group=product_group,
        ) != capability["role"]:
            capability["original_role"] = original_role
        result.append(capability)
    return _unique_capabilities(result)


def _semantic_required_value(value: Any) -> bool:
    if value is None:
        return True
    return _bool_value(value)


def _normalize_primary_product_group(value: Any) -> str:
    text = str(value or "").strip().casefold()
    aliases = {
        "servers": SERVER_PRODUCT_GROUP,
        "switch": NETWORK_PRODUCT_GROUP,
        "networking": NETWORK_PRODUCT_GROUP,
        "network_equipment": NETWORK_PRODUCT_GROUP,
        "san": STORAGE_PRODUCT_GROUP,
        "nas": STORAGE_PRODUCT_GROUP,
        "storage_array": STORAGE_PRODUCT_GROUP,
        "storage_system": STORAGE_PRODUCT_GROUP,
    }
    text = aliases.get(text, text)
    if text in PRIMARY_PRODUCT_GROUPS:
        return text
    return "unknown"


def _normalize_semantic_role(value: Any, *, product_group: str) -> str:
    role = str(value or "").strip()
    normalized = role.casefold().replace("-", "_").replace("/", "_").replace(" ", "_")
    aliases = {
        "platform": "server_platform",
        "barebone": "server_platform",
        "chassis": "server_platform",
        "processor": "cpu",
        "processors": "cpu",
        "memory": "ram",
        "nic": "network_adapter",
        "network": "network_adapter" if product_group == SERVER_PRODUCT_GROUP else "switch",
        "network_card": "network_adapter",
        "network_interface_card": "network_adapter",
        "psu": "power_supply",
        "power": "power_supply",
        "power_cable": "cable",
        "c13_c14": "cable",
        "c13_schuko": "cable",
        "accessory": "other_accessory",
        "accessories": "other_accessory",
        "other": "other_accessory",
        "unknown": UNMAPPED_ROLE,
        "unmapped": UNMAPPED_ROLE,
    }
    if product_group == SERVER_PRODUCT_GROUP:
        aliases.update(
            {
                "drive": "storage",
                "drives": "storage",
                "disk": "storage",
                "disks": "storage",
                "ssd": "storage",
                "storage_ssd": "storage",
                "storage_drive": "storage",
                "hdd": "storage",
                "controller": "storage_controller",
                "hba": "storage_controller",
                "raid_controller": "storage_controller",
                "management": "server_platform",
                "management_ports": "server_platform",
                "remote_management": "server_platform",
            }
        )
    elif product_group == STORAGE_PRODUCT_GROUP:
        aliases.update(
            {
                "storage_array": "storage_system",
                "system": "storage_system",
                "shelf": "disk_shelf",
                "drive_shelf": "disk_shelf",
                "expansion_shelf": "disk_shelf",
                "drives": "drive",
                "disks": "drive",
                "host_ports": "host_port",
                "ports": "host_port",
                "host_interface": "host_port",
                "protocol": "protocol_module",
                "interface_module": "protocol_module",
            }
        )
    elif product_group == NETWORK_PRODUCT_GROUP:
        aliases.update(
            {
                "network_switch": "switch",
                "ethernet_switch": "switch",
                "direct_attach": "dac_cable",
                "dac": "dac_cable",
                "network_cable": "cable",
            }
        )
    return aliases.get(normalized, role)


def _unique_blueprint_roles(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        key = (
            str(value.get("role") or ""),
            str(value.get("source_text") or ""),
            json.dumps(
                value.get("characteristics_to_match") or {},
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result


def _safe_semantic_list(value: Any) -> list[Any]:
    if not isinstance(value, list | tuple):
        return []
    result: list[Any] = []
    for item in value:
        if isinstance(item, Mapping):
            result.append(_json_safe_mapping(item))
        elif isinstance(item, str | int | float | bool):
            text = str(item).strip()
            if text:
                result.append(text)
    return result


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if isinstance(item, Mapping):
            result[key_text] = _json_safe_mapping(item)
        elif isinstance(item, list | tuple):
            result[key_text] = _safe_semantic_list(item)
        elif item is None or isinstance(item, str | int | float | bool):
            result[key_text] = item
        else:
            result[key_text] = str(item)
    return result


def _semantic_plan_is_actionable(semantic_plan: Mapping[str, Any]) -> bool:
    if semantic_plan.get("primary_product_group") not in PRIMARY_PRODUCT_GROUPS:
        return False
    blueprint = _mapping(semantic_plan.get("matrix_blueprint"))
    return bool(
        _mapping_rows(blueprint.get("roles"))
        or _mapping_rows(semantic_plan.get("required_capabilities"))
        or _mapping_rows(semantic_plan.get("optional_capabilities"))
        or _mapping_rows(semantic_plan.get("classified_requirements"))
    )


def _requirement_plan_from_semantic_matrix_plan(
    semantic_plan: Mapping[str, Any],
    *,
    profile: ProductGroupProfile,
) -> dict[str, Any]:
    blueprint_required, blueprint_optional = _capabilities_from_matrix_blueprint(
        semantic_plan.get("matrix_blueprint"),
        profile=profile,
        classified_requirements=_mapping_rows(
            semantic_plan.get("classified_requirements")
        ),
    )
    payload = {
        "product_group": semantic_plan["primary_product_group"],
        "required_capabilities": [
            *blueprint_required,
            *_mapping_rows(semantic_plan.get("required_capabilities")),
        ],
        "optional_capabilities": [
            *blueprint_optional,
            *_mapping_rows(semantic_plan.get("optional_capabilities")),
        ],
        "logistics_constraints": _mapping(semantic_plan.get("logistics_constraints")),
        "commercial_instructions": semantic_plan.get("commercial_instructions"),
        "response_instructions": semantic_plan.get("response_instructions"),
        "engineer_review_instructions": semantic_plan.get("engineer_review_instructions"),
        "engineer_review_required": bool(
            _string_list(semantic_plan.get("engineer_review_instructions"))
        ),
        "classified_requirements": semantic_plan.get("classified_requirements"),
        "unsupported_or_unmapped_requirements": semantic_plan.get(
            "unsupported_or_unmapped_requirements"
        ),
        "planner_warnings": semantic_plan.get("planner_warnings"),
    }
    return _coerce_requirement_plan(payload, profile=profile)


def _apply_role_lifecycle_policy(
    role_plan: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    semantic_plan: Mapping[str, Any],
    profile: ProductGroupProfile,
) -> dict[str, Any]:
    """Preserve broad roles across Stage A, classifier, and category planning."""

    product_group = profile.product_group_id
    stage_a_roles = role_lifecycle.unique_roles(
        _string_list(intent.get("required_bom_roles_guess")),
        product_group=product_group,
    )
    profile_roles = role_lifecycle.product_group_profile_broad_roles(product_group)
    semantic_matrix_blueprint_roles = role_lifecycle.roles_from_blueprint(
        semantic_plan.get("matrix_blueprint"),
        product_group=product_group,
    )
    semantic_matrix_blueprint_roles = [
        role
        for role in semantic_matrix_blueprint_roles
        if role != role_lifecycle.UNMAPPED_ROLE
    ]
    requirement_classifier_roles = role_lifecycle.unique_roles(
        [
            *semantic_matrix_blueprint_roles,
            *role_lifecycle.roles_from_capabilities(
                semantic_plan.get("required_capabilities"),
                product_group=product_group,
            ),
            *role_lifecycle.roles_from_capabilities(
                semantic_plan.get("optional_capabilities"),
                product_group=product_group,
            ),
            *role_lifecycle.roles_from_classified_requirements(
                semantic_plan.get("classified_requirements"),
                product_group=product_group,
            ),
        ],
        product_group=product_group,
    )
    requirement_classifier_roles = [
        role
        for role in requirement_classifier_roles
        if role != role_lifecycle.UNMAPPED_ROLE
    ]
    accessory_roles = role_lifecycle.accessory_hint_roles(
        intent,
        product_group=product_group,
    )
    not_applicable_reasons = _classifier_not_applicable_reason_by_role(
        semantic_plan.get("classified_requirements"),
        product_group=product_group,
    )
    effective_roles = role_lifecycle.unique_roles(
        [
            *stage_a_roles,
            *profile_roles,
            *requirement_classifier_roles,
            *accessory_roles,
        ],
        product_group=product_group,
    )
    effective_roles = [
        role
        for role in effective_roles
        if role != role_lifecycle.UNMAPPED_ROLE
        and (role not in not_applicable_reasons or role in profile_roles)
    ]
    role_sources = role_lifecycle.merge_role_sources(
        (role_lifecycle.ROLE_SOURCE_STAGE_A, stage_a_roles),
        (role_lifecycle.ROLE_SOURCE_PRODUCT_GROUP_PROFILE, profile_roles),
        (
            role_lifecycle.ROLE_SOURCE_REQUIREMENT_CLASSIFIER,
            requirement_classifier_roles,
        ),
        (role_lifecycle.ROLE_SOURCE_ACCESSORY_HINT, accessory_roles),
        existing=_mapping(role_plan.get("role_source_by_role")),
    )

    required_capabilities = _mapping_rows(role_plan.get("required_capabilities"))
    optional_capabilities = _mapping_rows(role_plan.get("optional_capabilities"))
    capability_roles = {
        _normalize_role(row.get("role"))
        for row in [*required_capabilities, *optional_capabilities]
    }
    required_roles = _string_list(role_plan.get("required_roles"))
    for role in effective_roles:
        if role in capability_roles:
            continue
        required_capabilities.append(
            _capability(
                role=role,
                capability_id=f"{role}.broad_role",
                requirement_text=f"Broad role preserved for {product_group}: {role}",
                parsed_requirements={
                    "required": True,
                    "role_lifecycle_source": role_sources.get(role, []),
                },
            )
        )
        capability_roles.add(role)
    required_roles = _unique([*required_roles, *effective_roles])
    requirements_by_role = dict(_mapping(role_plan.get("requirements_by_role")))
    for role in effective_roles:
        requirements_by_role.setdefault(
            role,
            {
                "required": True,
                "role_lifecycle_source": role_sources.get(role, []),
            },
        )
    stage_a_missing_from_classifier = role_lifecycle.dropped_roles(
        stage_a_roles,
        requirement_classifier_roles,
    )
    roles_dropped_after_stage_a = [
        role for role in stage_a_missing_from_classifier if role not in not_applicable_reasons
    ]
    reason_by_role = role_lifecycle.merge_drop_reasons(
        {
            role: "not_emitted_by_requirement_classifier_preserved_by_union"
            for role in roles_dropped_after_stage_a
            if role in effective_roles
        },
        {
            role: f"classifier_marked_not_applicable:{reason}"
            for role, reason in not_applicable_reasons.items()
        },
        existing=_mapping(role_plan.get("roles_dropped_reason_by_role")),
    )
    result = dict(role_plan)
    result["required_capabilities"] = _unique_capabilities(required_capabilities)
    result["optional_capabilities"] = _unique_capabilities(optional_capabilities)
    result["required_roles"] = required_roles
    result["requirements_by_role"] = requirements_by_role
    result["stage_a_broad_roles"] = stage_a_roles
    result["semantic_matrix_blueprint_roles"] = semantic_matrix_blueprint_roles
    result["requirement_classifier_roles"] = requirement_classifier_roles
    result["effective_matrix_roles_before_category_planner"] = effective_roles
    result["category_planner_input_roles"] = effective_roles
    result["roles_dropped_after_stage_a"] = roles_dropped_after_stage_a
    result["roles_dropped_before_category_planner"] = []
    result["role_source_by_role"] = role_sources
    result["roles_dropped_reason_by_role"] = reason_by_role
    result["role_lifecycle_trace"] = role_lifecycle.build_role_lifecycle_trace(
        effective_roles,
        role_source_by_role=role_sources,
        stage_a_roles=stage_a_roles,
        semantic_matrix_blueprint_roles=semantic_matrix_blueprint_roles,
        requirement_classifier_roles=requirement_classifier_roles,
        before_category_planner_roles=effective_roles,
        category_planner_input_roles=effective_roles,
        dropped_reason_by_role=reason_by_role,
    )
    return result


def _classifier_not_applicable_reason_by_role(
    classified_requirements: Any,
    *,
    product_group: str,
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for row in _mapping_rows(classified_requirements):
        role = role_lifecycle.normalize_role(
            row.get("target_role") or row.get("role") or row.get("role_id"),
            product_group=product_group,
        )
        if not role:
            continue
        hard_or_optional = str(row.get("hard_or_optional") or "").strip()
        fulfillment_mode = str(row.get("fulfillment_mode") or "").strip()
        reason = str(row.get("reason") or row.get("evidence_text") or "").strip()
        if fulfillment_mode == FULFILLMENT_NOT_APPLICABLE and reason:
            reasons[role] = reason
        elif hard_or_optional == REQ_OPTIONAL and reason:
            reasons[role] = f"optional:{reason}"
    return reasons


def _capabilities_from_matrix_blueprint(
    value: Any,
    *,
    profile: ProductGroupProfile,
    classified_requirements: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blueprint = _mapping(value)
    required: list[dict[str, Any]] = []
    optional: list[dict[str, Any]] = []
    allowed_roles = set(profile.role_catalog)
    for row in _mapping_rows(blueprint.get("roles")):
        role = _normalize_semantic_role(
            row.get("role"),
            product_group=profile.product_group_id,
        )
        if role not in allowed_roles and role != UNMAPPED_ROLE:
            role = UNMAPPED_ROLE
        source_text = str(row.get("source_text") or row.get("role") or role).strip()
        characteristics = _mapping(row.get("characteristics_to_match"))
        capability_ids = _string_list(row.get("hard_capability_ids"))
        if not capability_ids:
            capability_ids = [_blueprint_capability_id(role, profile, characteristics)]
        classified = _classified_requirement_for_source(
            source_text,
            classified_requirements,
        )
        if classified:
            if not _classified_requirement_should_materialize_capability(classified):
                continue
            role = _role_for_classified_capability(
                role,
                classified,
                profile=profile,
            )
        for capability_id in capability_ids:
            capability = _capability(
                role=role,
                capability_id=capability_id,
                requirement_text=source_text or role,
                parsed_requirements=characteristics or {"required": True},
                hard=_semantic_required_value(row.get("required")),
            )
            if row.get("original_role"):
                capability["original_role"] = str(row.get("original_role"))
            if capability["hard"]:
                required.append(capability)
            else:
                optional.append(capability)
    return _unique_capabilities(required), _unique_capabilities(optional)


def _blueprint_capability_id(
    role: str,
    profile: ProductGroupProfile,
    characteristics: Mapping[str, Any],
) -> str:
    if role in profile.required_roles and not characteristics:
        return f"{role}.base"
    return f"{role}.requested"


def _semantic_fallback_or_fail_closed(
    fallback_plan: Mapping[str, Any],
    *,
    request_text: str,
    source: str,
    confidence: str,
    reason: str,
    deterministic_product_group_hint: str,
    semantic_plan: Mapping[str, Any] | None,
    warning: str | None,
    fallback_reason: str | None,
    error_type: str | None = None,
    http_status: int | None = None,
    parse_status: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if _semantic_fallback_is_unsafe(fallback_plan, request_text, confidence=confidence):
        fail_closed = _semantic_planner_fail_closed_plan()
        fail_warning = _unique(
            [
                *(_string_list(fallback_plan.get("planner_warnings"))),
                *([warning] if warning else []),
                "semantic_matrix_planner_fallback_unsafe",
            ]
        )
        fail_closed["planner_warnings"] = fail_warning
        return _with_semantic_diagnostics(
            fail_closed,
            source=_semantic_fail_closed_source(source),
            confidence="low",
            reason=SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE,
            deterministic_product_group_hint=deterministic_product_group_hint,
            semantic_plan=None,
            warning=None,
            fallback_reason=(
                SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
                if fallback_reason == SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
                or source == SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT
                else SEMANTIC_COMPLEX_FALLBACK_REASON
            ),
            error_type=error_type,
            http_status=http_status,
            parse_status=parse_status,
            diagnostics=diagnostics,
        )
    return _with_semantic_diagnostics(
        fallback_plan,
        source=source,
        confidence=confidence,
        reason=reason,
        deterministic_product_group_hint=deterministic_product_group_hint,
        semantic_plan=semantic_plan,
        warning=warning,
        fallback_reason=fallback_reason,
        error_type=error_type,
        http_status=http_status,
        parse_status=parse_status,
        diagnostics=diagnostics,
    )


def _semantic_llm_exception_source(exc: BaseException) -> str:
    if isinstance(exc, LlmInvalidJsonError):
        return SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_INVALID
    return SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_ERROR


def _semantic_llm_exception_fallback_reason(exc: BaseException) -> str:
    if isinstance(exc, LlmInvalidJsonError):
        return "llm_invalid_json"
    return "llm_call_failed"


def _semantic_fail_closed_source(source: str) -> str:
    if source == SEMANTIC_SOURCE_DETERMINISTIC_FALLBACK:
        return SEMANTIC_COMPLEX_FALLBACK_REASON
    if source in {
        SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_EMPTY,
        SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_INVALID,
        SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_ERROR,
        SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT,
    }:
        return source
    return SEMANTIC_COMPLEX_FALLBACK_REASON


def _semantic_fallback_is_unsafe(
    plan: Mapping[str, Any],
    request_text: str,
    *,
    confidence: str,
) -> bool:
    if confidence != "low":
        return False
    if str(plan.get("product_group") or "").strip() == "unknown":
        return False
    if _looks_like_complex_mixed_request(request_text):
        return True
    return (
        _fallback_capability_coverage_ratio(plan, request_text) < 0.08
        and len(request_text) >= 500
        and _technical_domain_count(request_text) > 1
    )


def _deterministic_semantic_fallback_confidence(
    plan: Mapping[str, Any],
    request_text: str,
) -> str:
    if str(plan.get("product_group") or "").strip() == "unknown":
        return "low"
    if _looks_like_complex_mixed_request(request_text):
        return "low"
    roles = set(_string_list(plan.get("required_roles")))
    product_group = str(plan.get("product_group") or "").strip()
    if product_group == NETWORK_PRODUCT_GROUP and roles.intersection(
        {"switch", "router", "firewall", "access_point", "dac_cable", "transceiver"}
    ):
        return "high"
    if product_group == STORAGE_PRODUCT_GROUP and roles.intersection(
        {
            "storage_system",
            "controller",
            "controller_module",
            "disk_shelf",
            "drive",
            "host_port",
            "protocol_module",
        }
    ):
        return "high"
    if product_group == SERVER_PRODUCT_GROUP and roles.intersection(
        {"server_platform", "cpu", "ram", "storage"}
    ):
        text = request_text.casefold()
        if re.search(r"\bserver|сервер", text, re.I) or _technical_domain_count(text) <= 1:
            return "high"
    return "medium" if _fallback_capability_coverage_ratio(plan, request_text) >= 0.2 else "low"


def _looks_like_complex_mixed_request(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    line_count = len([line for line in normalized.splitlines() if line.strip()])
    section_count = _technical_section_count(normalized)
    domain_count = _technical_domain_count(normalized)
    long_request = len(normalized) >= 500 or line_count >= 8
    has_many_sections = section_count >= 3
    return domain_count > 1 and (long_request or has_many_sections)


def _technical_section_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) <= 40 and stripped.endswith(":"):
            count += 1
            continue
        if re.fullmatch(r"[A-ZА-ЯЁ0-9 .+/-]{3,40}", stripped):
            count += 1
    return count


def _technical_domain_count(text: str) -> int:
    lowered = str(text or "").casefold()
    patterns = {
        "server": (
            r"\b(?:server|cpu|processor|xeon|epyc|ram|memory|rdimm|ddr[345]?)\b|"
            r"сервер|процессор|оператив|памят"
        ),
        "network": (
            r"\b(?:switch|router|firewall|nic|ethernet|sfp\+?|sfp28|qsfp|dac|"
            r"10gbe|25gbe|uplink|rj-?45)\b|"
            r"коммутатор|свитч|маршрутизатор|сетев|трансивер|аплинк"
        ),
        "storage": (
            r"\b(?:ssd|hdd|nvme|sata|sas|raid|hba|jbod|storage|nas|san)\b|"
            r"схд|диск|накопител|контроллер"
        ),
        "power_accessory": (
            r"\b(?:psu|power\s+supply|c13|c14|schuko|pdu|platinum|fan|cooling)\b|"
            r"бп|питан|охлажд|вентилятор"
        ),
    }
    return sum(1 for pattern in patterns.values() if re.search(pattern, lowered, re.I))


def _fallback_capability_coverage_ratio(plan: Mapping[str, Any], request_text: str) -> float:
    text_length = max(1, len(" ".join(str(request_text or "").split())))
    covered = 0
    for capability in _mapping_rows(plan.get("required_capabilities")):
        capability_id = str(capability.get("capability_id") or "")
        if capability_id.endswith(".base"):
            continue
        source = str(
            capability.get("source_text")
            or capability.get("requirement_text")
            or capability.get("category_search_intent")
            or ""
        ).strip()
        if source:
            covered += min(len(source), 180)
    return covered / text_length


def _semantic_planner_fail_closed_plan() -> dict[str, Any]:
    return {
        "product_group": "unknown",
        "requirements": [],
        "required_capabilities": [],
        "optional_capabilities": [],
        **_requirement_classification_diagnostics([]),
        "workload_context": [],
        "logistics_constraints": {},
        "commercial_instructions": [],
        "response_instructions": [SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE],
        "engineer_review_instructions": [SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE],
        "engineer_review_required": True,
        "unsupported_or_unmapped_requirements": [],
        "planner_warnings": [],
        "required_roles": [],
        "optional_roles": [],
        "requirements_by_role": {},
        "role_catalog": [],
        "matrix_blueprint": {"roles": []},
        "embedded_requirements": [],
        "not_primary_product_groups": [],
    }


def _semantic_error_diagnostics(
    exc: BaseException,
    *,
    parse_status: str | None = None,
) -> dict[str, Any]:
    status = parse_status
    if status is None and isinstance(exc, LlmInvalidJsonError):
        status = exc.json_extract_status or exc.parse_stage or "invalid_json"
    return {
        "error_type": type(exc).__name__,
        "http_status": exc.status_code if isinstance(exc, LlmHttpError) else None,
        "parse_status": status,
    }


def _with_semantic_diagnostics(
    plan: Mapping[str, Any],
    *,
    source: str,
    confidence: str,
    reason: str,
    deterministic_product_group_hint: str,
    semantic_plan: Mapping[str, Any] | None,
    warning: str | None,
    fallback_reason: str | None,
    error_type: str | None = None,
    http_status: int | None = None,
    parse_status: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(plan)
    selected_product_group = str(
        result.get("product_group")
        or (semantic_plan or {}).get("primary_product_group")
        or "unknown"
    ).strip()
    matrix_blueprint = _mapping(
        (semantic_plan or {}).get("matrix_blueprint") or result.get("matrix_blueprint")
    )
    result["primary_product_group"] = selected_product_group
    result["primary_object"] = str(
        (semantic_plan or {}).get("primary_object") or result.get("primary_object") or "other"
    ).strip()
    result["semantic_planner_source"] = source
    result["semantic_planner_used"] = source in SEMANTIC_LLM_SOURCES
    result["semantic_planner_confidence"] = (
        confidence if confidence in SEMANTIC_CONFIDENCE_VALUES else "low"
    )
    result["semantic_planner_error_type"] = error_type
    result["semantic_planner_http_status"] = http_status
    result["semantic_planner_parse_status"] = parse_status
    result["semantic_planner_fallback_reason"] = fallback_reason
    if diagnostics:
        result.update(_json_safe_mapping(diagnostics))
    result["selected_product_group_reason"] = str(reason or "").strip()
    result["deterministic_product_group_hint"] = deterministic_product_group_hint
    result["semantic_planner_disagreement"] = bool(
        deterministic_product_group_hint
        and deterministic_product_group_hint != "unknown"
        and selected_product_group
        and selected_product_group != deterministic_product_group_hint
    )
    result["matrix_blueprint"] = matrix_blueprint if matrix_blueprint else {"roles": []}
    result["matrix_blueprint_roles"] = _matrix_blueprint_role_ids(
        result["matrix_blueprint"]
    )
    result["embedded_requirements"] = _safe_semantic_list(
        (semantic_plan or {}).get("embedded_requirements")
        or result.get("embedded_requirements")
    )
    classified_requirements = _mapping_rows(
        result.get("classified_requirements")
        or (semantic_plan or {}).get("classified_requirements")
    )
    result.update(_requirement_classification_diagnostics(classified_requirements))
    result["not_primary_product_groups"] = _safe_semantic_list(
        (semantic_plan or {}).get("not_primary_product_groups")
        or result.get("not_primary_product_groups")
    )
    warnings = _string_list(result.get("planner_warnings"))
    if warning:
        warnings.append(warning)
    if result["semantic_planner_disagreement"]:
        warnings.append("semantic_matrix_planner_disagreement_with_deterministic_hint")
    result["planner_warnings"] = _unique(warnings)
    return result


def _matrix_blueprint_role_ids(matrix_blueprint: Mapping[str, Any]) -> list[str]:
    return _unique(
        [
            str(row.get("role") or "").strip()
            for row in _mapping_rows(matrix_blueprint.get("roles"))
            if str(row.get("role") or "").strip()
        ]
    )


def _deterministic_requirement_plan(
    spec: StockSpec | Mapping[str, Any] | str | None,
    *,
    source_text: str | None,
    normalized_spec: Mapping[str, Any] | None,
    product_group_profile: ProductGroupProfile,
) -> dict[str, Any]:
    items, spec_source_text, global_requirements = _spec_parts(spec)
    request_source_text = _join_unique_text_parts(source_text, spec_source_text)
    structured_context = (
        ""
        if request_source_text
        else _join_unique_text_parts(
            str(normalized_spec or ""),
            str(global_requirements or ""),
        )
    )
    full_source_text = _join_unique_text_parts(request_source_text, structured_context)
    product_group = _detect_product_group(full_source_text, product_group_profile)
    required_capabilities: list[dict[str, Any]] = []
    optional_capabilities: list[dict[str, Any]] = []
    unsupported: list[str] = []
    warnings: list[str] = []

    if product_group == product_group_profile.product_group_id:
        for role in product_group_profile.required_roles:
            required_capabilities.append(
                _capability(
                    role=role,
                    capability_id=f"{role}.base",
                    requirement_text=f"Base {product_group} role",
                    parsed_requirements={"required": True},
                )
            )

    texts = [full_source_text]
    include_item_requirements = not bool(full_source_text)
    for item in items:
        texts.append(
            _request_text(
                "",
                item,
                include_requirements=include_item_requirements,
            )
        )
    combined_text = _join_unique_text_parts(*texts)

    if product_group_profile.product_group_id == NETWORK_PRODUCT_GROUP:
        required_capabilities.extend(
            _network_product_capabilities(
                combined_text,
                items=items,
                global_requirements=global_requirements,
            )
        )
        optional_capabilities.extend(_network_optional_capabilities(combined_text))
        unsupported.extend(_unknown_hard_requirement_markers(combined_text))
        return _coerce_requirement_plan(
            {
                "product_group": product_group,
                "required_capabilities": required_capabilities,
                "optional_capabilities": optional_capabilities,
                "logistics_constraints": _deterministic_logistics_constraints(
                    combined_text
                ),
                "commercial_instructions": _deterministic_commercial_instructions(
                    combined_text
                ),
                "unsupported_or_unmapped_requirements": unsupported,
                "planner_warnings": warnings,
            },
            profile=product_group_profile,
        )

    if product_group_profile.product_group_id == STORAGE_PRODUCT_GROUP:
        required_capabilities.extend(
            _storage_product_capabilities(
                combined_text,
                items=items,
                global_requirements=global_requirements,
            )
        )
        optional_capabilities.extend(_storage_optional_capabilities(combined_text))
        unsupported.extend(_unknown_hard_requirement_markers(combined_text))
        return _coerce_requirement_plan(
            {
                "product_group": product_group,
                "required_capabilities": required_capabilities,
                "optional_capabilities": optional_capabilities,
                "logistics_constraints": _deterministic_logistics_constraints(
                    combined_text
                ),
                "commercial_instructions": _deterministic_commercial_instructions(
                    combined_text
                ),
                "engineer_review_required": True,
                "engineer_review_instructions": _deterministic_engineer_review_instructions(
                    combined_text
                ),
                "unsupported_or_unmapped_requirements": unsupported,
                "planner_warnings": warnings,
            },
            profile=product_group_profile,
        )

    network_requirement: dict[str, Any] | None = None
    for item in items:
        item_requirements = _item_requirements(item)
        network = network_requirement_from_sources(
            text=_request_text(full_source_text, item),
            explicit=_mapping(item_requirements.get("network")),
        )
        if network.get("required"):
            network_requirement = _merge_network_requirement(network_requirement, network)
    if network_requirement is None:
        network_requirement = network_requirement_from_sources(text=combined_text)
    if network_requirement.get("required"):
        required_capabilities.append(
            _capability(
                role="network_adapter",
                capability_id=_network_capability_id(network_requirement),
                requirement_text=_capability_text(combined_text, "network"),
                parsed_requirements=network_requirement,
            )
        )

    dynamic_detectors = (
        (
            "storage_controller",
            _requires_storage_controller,
            _storage_controller_parsed_requirements,
            "storage_controller.requested",
        ),
        ("gpu", _requires_gpu, _empty_parsed_requirements, "gpu.requested"),
        (
            "transceiver",
            _requires_transceiver,
            _empty_parsed_requirements,
            "transceiver.requested",
        ),
        ("cable", _requires_cable, _cable_parsed_requirements, "cable.requested"),
        (
            "power_supply",
            _requires_power_supply,
            _power_supply_parsed_requirements,
            "power_supply.requested",
        ),
        ("rail_kit", _requires_rail_kit, _empty_parsed_requirements, "rail_kit.requested"),
        ("license", _requires_license, _empty_parsed_requirements, "license.requested"),
        ("support", _requires_support, _empty_parsed_requirements, "support.requested"),
    )
    for role, detector, parser, capability_id in dynamic_detectors:
        if detector(combined_text, global_requirements):
            parsed_requirements = parser(combined_text)
            current_capability_id = capability_id
            if role == "power_supply" and (
                _as_int(parsed_requirements.get("psu_count_per_server"))
                or _as_int(parsed_requirements.get("count"))
                or 0
            ) >= 2:
                current_capability_id = "power_supply.min_2"
            required_capabilities.append(
                _capability(
                    role=role,
                    capability_id=current_capability_id,
                    requirement_text=_capability_text(combined_text, role),
                    parsed_requirements=parsed_requirements,
                )
            )

    unsupported.extend(_unknown_hard_requirement_markers(combined_text))
    plan = _coerce_requirement_plan(
        {
            "product_group": product_group,
            "required_capabilities": required_capabilities,
            "optional_capabilities": optional_capabilities,
            "unsupported_or_unmapped_requirements": unsupported,
            "planner_warnings": warnings,
        },
        profile=product_group_profile,
    )
    return plan


def _coerce_requirement_plan(
    plan: Mapping[str, Any],
    *,
    profile: ProductGroupProfile,
) -> dict[str, Any]:
    allowed_roles = set(profile.role_catalog)
    original_product_group = str(plan.get("product_group") or profile.product_group_id).strip()
    product_group = original_product_group
    if product_group not in {profile.product_group_id, "unknown"}:
        product_group = profile.product_group_id

    requirements: list[dict[str, Any]] = []
    required: list[dict[str, Any]] = []
    optional: list[dict[str, Any]] = []
    unsupported = _string_list(plan.get("unsupported_or_unmapped_requirements"))
    warnings = _string_list(plan.get("planner_warnings"))
    if original_product_group not in {profile.product_group_id, "unknown"}:
        warnings.append(
            f"requirement_planner_product_group_normalized:{original_product_group}"
        )
    workload_context = _string_list(plan.get("workload_context"))
    logistics_constraints = _mapping(plan.get("logistics_constraints"))
    commercial_instructions = _instruction_rows(plan.get("commercial_instructions"))
    response_instructions = _string_list(plan.get("response_instructions"))
    engineer_review_instructions = _string_list(plan.get("engineer_review_instructions"))
    engineer_review_required = _bool_value(plan.get("engineer_review_required"))
    classified_requirements = _coerce_classified_requirements(
        plan.get("classified_requirements"),
        profile=profile,
        product_group=product_group,
        primary_object=str(plan.get("primary_object") or "").strip(),
    )

    for index, raw in enumerate(_mapping_rows(plan.get("requirements")), start=1):
        item = _coerce_requirement_item(
            raw,
            allowed_roles=allowed_roles,
            fallback_id=f"req_{index}",
        )
        if item is None:
            continue
        requirements.append(item)
        classification = item["classification"]
        source_text = str(item.get("source_text") or "").strip()
        parsed = _mapping(item.get("parsed_requirements"))
        if classification == CLASS_HARD_TECHNICAL:
            capability = _capability_from_requirement_item(item, allowed_roles=allowed_roles)
            if capability is None:
                if source_text:
                    unsupported.append(source_text)
                continue
            required.append(capability)
            continue
        if classification == CLASS_SOFT_PREFERENCE:
            capability = _capability_from_requirement_item(
                item,
                allowed_roles=allowed_roles,
                hard=False,
            )
            if capability is not None:
                optional.append(capability)
            continue
        if classification == CLASS_WORKLOAD_CONTEXT:
            if source_text:
                workload_context.append(source_text)
            continue
        if classification == CLASS_LOGISTICS_CONSTRAINT:
            logistics_constraints = {
                **logistics_constraints,
                **parsed,
            }
            if source_text and not parsed:
                logistics_constraints.setdefault("notes", [])
                notes = logistics_constraints.get("notes")
                if isinstance(notes, list):
                    notes.append(source_text)
            continue
        if classification == CLASS_COMMERCIAL_INSTRUCTION:
            commercial_instructions.append(_instruction_row(source_text, parsed))
            continue
        if classification == CLASS_OUTPUT_INSTRUCTION:
            if source_text:
                response_instructions.append(source_text)
            continue
        if classification == CLASS_ENGINEER_REVIEW_INSTRUCTION:
            engineer_review_required = True
            if source_text:
                engineer_review_instructions.append(source_text)
            continue
        if classification == CLASS_UNSUPPORTED_HARD and source_text:
            unsupported.append(source_text)

    for raw in _mapping_rows(plan.get("required_capabilities")):
        capability = _coerce_capability(raw, allowed_roles=allowed_roles, hard=True)
        if capability is None:
            text = str(
                raw.get("source_text")
                or raw.get("requirement_text")
                or raw.get("role")
                or ""
            ).strip()
            if text:
                unsupported.append(text)
            continue
        required.append(capability)
        requirement = _requirement_item_from_capability(
            capability,
            classification=CLASS_HARD_TECHNICAL,
        )
        if not _has_requirement(requirements, requirement):
            requirements.append(requirement)

    for raw in _mapping_rows(plan.get("optional_capabilities")):
        capability = _coerce_capability(raw, allowed_roles=allowed_roles, hard=False)
        if capability is not None:
            optional.append(capability)
            requirement = _requirement_item_from_capability(
                capability,
                classification=CLASS_SOFT_PREFERENCE,
            )
            if not _has_requirement(requirements, requirement):
                requirements.append(requirement)

    if product_group == profile.product_group_id:
        required = _ensure_profile_required_capabilities(required, profile)

    classified_requirements = _derive_missing_classified_requirements(
        classified_requirements,
        requirements=requirements,
        required_capabilities=required,
        optional_capabilities=optional,
        profile=profile,
        product_group=product_group,
        primary_object=str(plan.get("primary_object") or "").strip(),
    )
    required.extend(
        _capabilities_from_classified_requirements(
            classified_requirements,
            profile=profile,
            hard=True,
            existing_capabilities=[*required, *optional],
        )
    )
    optional.extend(
        _capabilities_from_classified_requirements(
            classified_requirements,
            profile=profile,
            hard=False,
            existing_capabilities=[*required, *optional],
        )
    )
    required, optional, unsupported, requirements, engineer_review_instructions = (
        _apply_classified_requirement_policy(
            required_capabilities=required,
            optional_capabilities=optional,
            unsupported=unsupported,
            requirements=requirements,
            engineer_review_instructions=engineer_review_instructions,
            classified_requirements=classified_requirements,
            profile=profile,
        )
    )
    engineer_review_required = engineer_review_required or bool(
        engineer_review_instructions
    )

    requirements_by_role: dict[str, dict[str, Any]] = {}
    required_roles: list[str] = []
    optional_roles: list[str] = []
    for capability in required:
        role = capability["role"]
        if role not in required_roles:
            required_roles.append(role)
        requirements_by_role[role] = _merge_role_requirements(
            requirements_by_role.get(role),
            capability,
        )
    for capability in optional:
        role = capability["role"]
        if role not in optional_roles:
            optional_roles.append(role)

    classification_diagnostics = _requirement_classification_diagnostics(
        classified_requirements
    )
    return {
        "product_group": product_group,
        "requirements": _unique_requirements(requirements),
        "required_capabilities": _unique_capabilities(required),
        "optional_capabilities": _unique_capabilities(optional),
        **classification_diagnostics,
        "workload_context": _unique(workload_context),
        "logistics_constraints": _dedupe_mapping_lists(logistics_constraints),
        "commercial_instructions": _unique_instruction_rows(commercial_instructions),
        "response_instructions": _unique(response_instructions),
        "engineer_review_instructions": _unique(engineer_review_instructions),
        "engineer_review_required": engineer_review_required,
        "unsupported_or_unmapped_requirements": _unique(unsupported),
        "planner_warnings": _unique(warnings),
        "required_roles": required_roles,
        "optional_roles": optional_roles,
        "requirements_by_role": requirements_by_role,
        "role_catalog": list(profile.role_catalog),
    }


def _coerce_capability(
    raw: Mapping[str, Any],
    *,
    allowed_roles: set[str],
    hard: bool,
) -> dict[str, Any] | None:
    role = _normalize_role(raw.get("role") or raw.get("role_id"))
    original_role = role
    if role not in allowed_roles and role != UNMAPPED_ROLE:
        if hard and role:
            role = UNMAPPED_ROLE
        else:
            return None
    if not role:
        return None
    capability_id = str(raw.get("capability_id") or f"{role}.requested").strip()
    parsed = raw.get("parsed_requirements")
    text = str(
        raw.get("source_text")
        or raw.get("requirement_text")
        or raw.get("text")
        or role
    ).strip()
    result = {
        "capability_id": capability_id,
        "role": role,
        "requirement_text": text,
        "source_text": text,
        "hard": _bool_value(raw.get("hard")) if "hard" in raw else hard,
        "parsed_requirements": dict(parsed) if isinstance(parsed, Mapping) else {},
    }
    raw_original_role = str(raw.get("original_role") or "").strip()
    if raw_original_role and raw_original_role != role:
        result["original_role"] = raw_original_role
    elif original_role and original_role != role:
        result["original_role"] = original_role
    category_search_intent = str(raw.get("category_search_intent") or "").strip()
    if category_search_intent:
        result["category_search_intent"] = category_search_intent
    confidence = str(raw.get("confidence") or "").strip()
    if confidence in {"high", "medium", "low"}:
        result["confidence"] = confidence
    result["can_be_satisfied_by_platform"] = _capability_can_be_satisfied_by_platform(
        result
    )
    return result


def _coerce_requirement_item(
    raw: Mapping[str, Any],
    *,
    allowed_roles: set[str],
    fallback_id: str,
) -> dict[str, Any] | None:
    source_text = str(
        raw.get("source_text")
        or raw.get("requirement_text")
        or raw.get("text")
        or ""
    ).strip()
    classification = _normalize_classification(
        raw.get("classification"),
        hard=_bool_value(raw.get("hard")),
    )
    parsed = raw.get("parsed_requirements")
    role = _normalize_role(raw.get("role") or raw.get("role_id"))
    original_role = role
    if (
        role
        and role not in allowed_roles
        and role != UNMAPPED_ROLE
        and classification == CLASS_HARD_TECHNICAL
    ):
        role = UNMAPPED_ROLE
    result: dict[str, Any] = {
        "requirement_id": str(raw.get("requirement_id") or fallback_id).strip(),
        "source_text": source_text,
        "classification": classification,
        "hard": _bool_value(raw.get("hard"))
        if "hard" in raw
        else classification in {CLASS_HARD_TECHNICAL, CLASS_UNSUPPORTED_HARD},
        "parsed_requirements": dict(parsed) if isinstance(parsed, Mapping) else {},
    }
    if role in allowed_roles or role == UNMAPPED_ROLE:
        result["role"] = role
    if original_role and original_role != role:
        result["original_role"] = original_role
    capability_id = str(raw.get("capability_id") or "").strip()
    if capability_id:
        result["capability_id"] = capability_id
    category_search_intent = str(raw.get("category_search_intent") or "").strip()
    if category_search_intent:
        result["category_search_intent"] = category_search_intent
    confidence = str(raw.get("confidence") or "").strip()
    if confidence in {"high", "medium", "low"}:
        result["confidence"] = confidence
    if source_text or role or capability_id or result["parsed_requirements"]:
        return result
    return None


def _normalize_classification(value: Any, *, hard: bool) -> str:
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().casefold())
    aliases = {
        "hard_requirement": CLASS_HARD_TECHNICAL,
        "technical_requirement": CLASS_HARD_TECHNICAL,
        "hard_technical": CLASS_HARD_TECHNICAL,
        "soft": CLASS_SOFT_PREFERENCE,
        "preference": CLASS_SOFT_PREFERENCE,
        "context": CLASS_WORKLOAD_CONTEXT,
        "workload": CLASS_WORKLOAD_CONTEXT,
        "use_case": CLASS_WORKLOAD_CONTEXT,
        "logistics": CLASS_LOGISTICS_CONSTRAINT,
        "logistic_constraint": CLASS_LOGISTICS_CONSTRAINT,
        "commercial": CLASS_COMMERCIAL_INSTRUCTION,
        "commercial_optimization": CLASS_COMMERCIAL_INSTRUCTION,
        "optimization_instruction": CLASS_COMMERCIAL_INSTRUCTION,
        "response": CLASS_OUTPUT_INSTRUCTION,
        "response_instruction": CLASS_RESPONSE_INSTRUCTION,
        "output": CLASS_OUTPUT_INSTRUCTION,
        "output_instruction": CLASS_RESPONSE_INSTRUCTION,
        "engineer_review": CLASS_ENGINEER_REVIEW_INSTRUCTION,
        "engineering_review_instruction": CLASS_ENGINEER_REVIEW_INSTRUCTION,
        "manual_review_instruction": CLASS_ENGINEER_REVIEW_INSTRUCTION,
        "unsupported_requirement": CLASS_UNSUPPORTED_HARD,
        "unsupported_hard_technical_requirement": CLASS_UNSUPPORTED_HARD,
        "unsupported_technical_requirement": CLASS_UNSUPPORTED_HARD,
    }
    normalized = aliases.get(text, text)
    if normalized in CLASSIFICATION_VALUES:
        return normalized
    return CLASS_HARD_TECHNICAL if hard else CLASS_SOFT_PREFERENCE


def _normalize_role(value: Any) -> str:
    role = str(value or "").strip()
    if role == "platform":
        return "server_platform"
    if role == "license/support":
        return "support"
    if role.casefold() in {"unknown", "unmapped"}:
        return UNMAPPED_ROLE
    return role


def _capability_from_requirement_item(
    item: Mapping[str, Any],
    *,
    allowed_roles: set[str],
    hard: bool = True,
) -> dict[str, Any] | None:
    role = _normalize_role(item.get("role"))
    if role not in allowed_roles and role != UNMAPPED_ROLE:
        return None
    capability_id = str(item.get("capability_id") or f"{role}.requested").strip()
    parsed = item.get("parsed_requirements")
    source_text = str(item.get("source_text") or role).strip()
    result = {
        "capability_id": capability_id,
        "role": role,
        "requirement_text": source_text,
        "source_text": source_text,
        "hard": _bool_value(item.get("hard")) if "hard" in item else hard,
        "parsed_requirements": dict(parsed) if isinstance(parsed, Mapping) else {},
    }
    if item.get("original_role"):
        result["original_role"] = str(item.get("original_role"))
    category_search_intent = str(item.get("category_search_intent") or "").strip()
    if category_search_intent:
        result["category_search_intent"] = category_search_intent
    confidence = str(item.get("confidence") or "").strip()
    if confidence in {"high", "medium", "low"}:
        result["confidence"] = confidence
    result["can_be_satisfied_by_platform"] = _capability_can_be_satisfied_by_platform(
        result
    )
    return result


def _requirement_item_from_capability(
    capability: Mapping[str, Any],
    *,
    classification: str,
) -> dict[str, Any]:
    source_text = str(
        capability.get("source_text")
        or capability.get("requirement_text")
        or capability.get("role")
        or ""
    ).strip()
    result = {
        "requirement_id": str(
            capability.get("requirement_id")
            or capability.get("capability_id")
            or source_text
        ).strip(),
        "source_text": source_text,
        "classification": classification,
        "role": str(capability.get("role") or "").strip(),
        "capability_id": str(capability.get("capability_id") or "").strip(),
        "hard": _bool_value(capability.get("hard"))
        if "hard" in capability
        else classification == CLASS_HARD_TECHNICAL,
        "parsed_requirements": dict(_mapping(capability.get("parsed_requirements"))),
    }
    category_search_intent = str(
        capability.get("category_search_intent") or ""
    ).strip()
    if category_search_intent:
        result["category_search_intent"] = category_search_intent
    confidence = str(capability.get("confidence") or "").strip()
    if confidence in {"high", "medium", "low"}:
        result["confidence"] = confidence
    if capability.get("original_role"):
        result["original_role"] = str(capability.get("original_role"))
    return result


def _has_requirement(
    requirements: Sequence[Mapping[str, Any]],
    requirement: Mapping[str, Any],
) -> bool:
    key = (
        str(requirement.get("source_text") or ""),
        str(requirement.get("classification") or ""),
        str(requirement.get("role") or ""),
        str(requirement.get("capability_id") or ""),
    )
    for current in requirements:
        current_key = (
            str(current.get("source_text") or ""),
            str(current.get("classification") or ""),
            str(current.get("role") or ""),
            str(current.get("capability_id") or ""),
        )
        if current_key == key:
            return True
    return False


def _ensure_profile_required_capabilities(
    capabilities: list[dict[str, Any]],
    profile: ProductGroupProfile,
) -> list[dict[str, Any]]:
    result = list(capabilities)
    roles = {str(capability.get("role") or "") for capability in capabilities}
    for role in profile.required_roles:
        if role in roles:
            continue
        result.append(
            _capability(
                role=role,
                capability_id=f"{role}.base",
                requirement_text=f"Base {profile.product_group_id} role",
                parsed_requirements={"required": True},
            )
        )
        roles.add(role)
    return result


def _capability(
    *,
    role: str,
    capability_id: str,
    requirement_text: str,
    parsed_requirements: Mapping[str, Any],
    hard: bool = True,
) -> dict[str, Any]:
    result = {
        "capability_id": capability_id,
        "role": role,
        "requirement_text": requirement_text,
        "source_text": requirement_text,
        "hard": hard,
        "parsed_requirements": dict(parsed_requirements),
    }
    result["can_be_satisfied_by_platform"] = _capability_can_be_satisfied_by_platform(
        result
    )
    return result


def _capability_can_be_satisfied_by_platform(capability: Mapping[str, Any]) -> bool:
    role = str(capability.get("role") or "").strip()
    parsed = _mapping(capability.get("parsed_requirements"))
    if role in {"server_platform", "network_adapter"}:
        return True
    if role != "power_supply":
        return False
    separate_markers = {
        "separate_psu",
        "spare_psu",
        "replacement_psu",
        "extra_psu",
        "additional_psu",
    }
    return not any(_bool_value(parsed.get(key)) for key in separate_markers)


def _coerce_classified_requirements(
    value: Any,
    *,
    profile: ProductGroupProfile | None,
    product_group: str,
    primary_object: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    allowed_roles = set(profile.role_catalog) if profile is not None else set()
    for index, row in enumerate(_mapping_rows(value), start=1):
        source_text = str(
            row.get("source_text")
            or row.get("requirement_text")
            or row.get("text")
            or ""
        ).strip()
        classification = _normalize_requirement_classification(
            row.get("classification")
        )
        hard_or_optional = _normalize_hard_or_optional(row)
        target_role = _normalize_role(
            row.get("target_role")
            or row.get("role")
            or row.get("role_id")
            or ""
        )
        original_role = str(
            row.get("target_role") or row.get("role") or row.get("role_id") or ""
        ).strip()
        if target_role and target_role not in allowed_roles and target_role != UNMAPPED_ROLE:
            target_role = UNMAPPED_ROLE
        target_primary_object = str(
            row.get("target_primary_object") or row.get("primary_object") or primary_object
        ).strip()
        parsed = _mapping(
            row.get("parsed_requirements")
            or row.get("characteristics_to_match")
            or row.get("requirements")
        )
        fulfillment_mode = _normalize_fulfillment_mode(
            row.get("fulfillment_mode") or row.get("fulfillment"),
            classification=classification,
        )
        evidence_source = str(row.get("evidence_source") or "").strip()
        evidence_text = str(row.get("evidence_text") or "").strip()
        if (
            fulfillment_mode in FULFILLMENT_INCLUDED_MODES
            and not evidence_text
            and not _explicit_fulfillment_assumption(row)
            and not (
                classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE
                and fulfillment_mode == FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT
            )
        ):
            fulfillment_mode = FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION
        should_create_bom_role = _classified_should_create_bom_role(
            row,
            classification=classification,
            fulfillment_mode=fulfillment_mode,
        )
        category_needed = _classified_category_needed(
            row,
            classification=classification,
            should_create_bom_role=should_create_bom_role,
        )
        should_block = _classified_should_block(
            row,
            classification=classification,
            hard_or_optional=hard_or_optional,
            category_needed=category_needed,
            fulfillment_mode=fulfillment_mode,
        )
        should_validate = _classified_should_validate(
            row,
            classification=classification,
            hard_or_optional=hard_or_optional,
            fulfillment_mode=fulfillment_mode,
        )
        confidence = str(row.get("confidence") or "").strip().casefold()
        if confidence not in SEMANTIC_CONFIDENCE_VALUES:
            confidence = "low"
        engineer_check = str(
            row.get("engineer_check_ru")
            or row.get("suggested_engineer_check_ru")
            or row.get("engineer_check")
            or ""
        ).strip()
        fulfillment_target_role = _normalize_role(
            row.get("fulfillment_target_role")
            or row.get("fulfillment_role")
            or target_role
            or ""
        )
        if (
            fulfillment_target_role
            and fulfillment_target_role not in allowed_roles
            and fulfillment_target_role != UNMAPPED_ROLE
        ):
            fulfillment_target_role = UNMAPPED_ROLE
        item = {
            "requirement_id": str(
                row.get("requirement_id") or row.get("id") or f"req_{index}"
            ).strip(),
            "source_text": source_text,
            "classification": classification,
            "product_group": str(row.get("product_group") or product_group or "").strip(),
            "target_role": target_role,
            "target_primary_object": target_primary_object,
            "hard_or_optional": hard_or_optional,
            "reason": str(row.get("reason") or "").strip(),
            "confidence": confidence,
            "fulfillment_mode": fulfillment_mode,
            "fulfillment_target_role": fulfillment_target_role,
            "fulfillment_target_component_candidate_id": str(
                row.get("fulfillment_target_component_candidate_id") or ""
            ).strip(),
            "evidence_source": evidence_source,
            "evidence_text": evidence_text,
            "should_create_bom_role": should_create_bom_role,
            "should_block_before_composer": should_block,
            "should_appear_in_composer_brief": _classified_should_appear(row),
            "should_validate_after_composer": should_validate,
            "should_be_validated_after_composer": should_validate,
            "engineer_check_ru": engineer_check,
            "suggested_engineer_check_ru": engineer_check,
            "category_needed": category_needed,
            "parsed_requirements": dict(parsed),
        }
        if original_role and original_role != target_role:
            item["original_role"] = original_role
        if source_text or target_role or parsed:
            result.append(item)
    return _unique_classified_requirements(result)


def _normalize_requirement_classification(value: Any) -> str:
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().casefold())
    aliases = {
        "purchasable": REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        "purchasable_role": REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        "component_role": REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        "bom_role": REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        "primary_feature": REQ_CLASS_PRIMARY_OBJECT_FEATURE,
        "platform_feature": REQ_CLASS_PRIMARY_OBJECT_FEATURE,
        "device_feature": REQ_CLASS_PRIMARY_OBJECT_FEATURE,
        "system_feature": REQ_CLASS_PRIMARY_OBJECT_FEATURE,
        "feature_constraint": REQ_CLASS_PRIMARY_OBJECT_FEATURE,
        "accessory": REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
        "consumable": REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
        "service": REQ_CLASS_SERVICE_OR_SUPPORT,
        "support": REQ_CLASS_SERVICE_OR_SUPPORT,
        "license": REQ_CLASS_SERVICE_OR_SUPPORT,
        "logistics": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "logistics_constraint": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "commercial": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "commercial_instruction": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "engineering": REQ_CLASS_ENGINEERING_CHECK,
        "engineer_review": REQ_CLASS_ENGINEERING_CHECK,
        "engineering_review_instruction": REQ_CLASS_ENGINEERING_CHECK,
        "manual_review_instruction": REQ_CLASS_ENGINEERING_CHECK,
        "unmapped_non_blocking": REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
        "unsupported_non_blocking": REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
        "out_of_scope": REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
        "blocking_unmapped": REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE,
        "unmapped_purchasable": REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE,
    }
    normalized = aliases.get(text, text)
    if normalized in REQUIREMENT_CLASSIFICATION_VALUES:
        return normalized
    return REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING


def _normalize_fulfillment_mode(value: Any, *, classification: str) -> str:
    text = re.sub(r"[\s-]+", "_", str(value or "").strip().casefold())
    aliases = {
        "separate": FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        "separate_component": FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        "separate_role": FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        "separate_bom_role": FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        "purchasable_component_role": FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        "included": FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        "included_in_platform": FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
        "included_in_primary": FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
        "primary_object_feature": FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
        "included_in_component": FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
        "selected_component": FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
        "included_in_selected": FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
        "bundle": FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        "kit": FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        "package": FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        "platform_package": FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        "included_in_package": FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        "included_in_platform_bundle": FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        "included_in_bundle": FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        "service": FULFILLMENT_SERVICE_OR_SUPPORT,
        "support": FULFILLMENT_SERVICE_OR_SUPPORT,
        "service_support": FULFILLMENT_SERVICE_OR_SUPPORT,
        "logistics": FULFILLMENT_LOGISTICS_CONSTRAINT,
        "commercial_constraint": FULFILLMENT_LOGISTICS_CONSTRAINT,
        "engineering": FULFILLMENT_ENGINEERING_CHECK_ONLY,
        "engineering_check": FULFILLMENT_ENGINEERING_CHECK_ONLY,
        "engineer_check": FULFILLMENT_ENGINEERING_CHECK_ONLY,
        "unverified": FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
        "unknown": FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
        "requires_confirmation": FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
        "needs_confirmation": FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
        "not_applicable": FULFILLMENT_NOT_APPLICABLE,
        "n/a": FULFILLMENT_NOT_APPLICABLE,
        "none": FULFILLMENT_NOT_APPLICABLE,
    }
    normalized = aliases.get(text, text)
    if normalized in FULFILLMENT_VALUES:
        return normalized
    if classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE:
        return FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT
    if classification == REQ_CLASS_SERVICE_OR_SUPPORT:
        return FULFILLMENT_SERVICE_OR_SUPPORT
    if classification == REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT:
        return FULFILLMENT_LOGISTICS_CONSTRAINT
    if classification == REQ_CLASS_ENGINEERING_CHECK:
        return FULFILLMENT_ENGINEERING_CHECK_ONLY
    if classification == REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING:
        return FULFILLMENT_NOT_APPLICABLE
    return FULFILLMENT_SEPARATE_COMPONENT_REQUIRED


def _explicit_fulfillment_assumption(row: Mapping[str, Any]) -> bool:
    if "explicit_assumption" in row:
        return _bool_value(row.get("explicit_assumption"))
    if "assumption_explicit" in row:
        return _bool_value(row.get("assumption_explicit"))
    assumption = str(row.get("assumption") or "").strip()
    return bool(assumption)


def _normalize_hard_or_optional(row: Mapping[str, Any]) -> str:
    text = str(
        row.get("hard_or_optional")
        or row.get("hardness")
        or row.get("priority")
        or ""
    ).strip().casefold()
    if text in {"hard", "required", "mandatory", "must"}:
        return REQ_HARD
    if text in {"optional", "soft", "nice_to_have", "preference"}:
        return REQ_OPTIONAL
    if "hard" in row:
        return REQ_HARD if _bool_value(row.get("hard")) else REQ_OPTIONAL
    classification = _normalize_requirement_classification(row.get("classification"))
    if classification in {
        REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        REQ_CLASS_ENGINEERING_CHECK,
        REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
    }:
        return REQ_OPTIONAL
    return REQ_HARD


def _classified_category_needed(
    row: Mapping[str, Any],
    *,
    classification: str,
    should_create_bom_role: bool | None = None,
) -> bool:
    if should_create_bom_role is False:
        return False
    if "category_needed" in row:
        return _bool_value(row.get("category_needed"))
    if should_create_bom_role is True:
        return True
    return classification in {
        REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
        REQ_CLASS_SERVICE_OR_SUPPORT,
        REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE,
    }


def _classified_should_create_bom_role(
    row: Mapping[str, Any],
    *,
    classification: str,
    fulfillment_mode: str,
) -> bool:
    if "should_create_bom_role" in row:
        return _bool_value(row.get("should_create_bom_role"))
    if fulfillment_mode in {
        FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        FULFILLMENT_SERVICE_OR_SUPPORT,
    }:
        return classification in {
            REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
            REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
            REQ_CLASS_SERVICE_OR_SUPPORT,
            REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE,
        }
    return False


def _classified_should_block(
    row: Mapping[str, Any],
    *,
    classification: str,
    hard_or_optional: str,
    category_needed: bool,
    fulfillment_mode: str,
) -> bool:
    if fulfillment_mode in {
        FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
        FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
        FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
        FULFILLMENT_NOT_APPLICABLE,
        FULFILLMENT_LOGISTICS_CONSTRAINT,
        FULFILLMENT_ENGINEERING_CHECK_ONLY,
    }:
        return False
    if hard_or_optional != REQ_HARD or not category_needed:
        return False
    if classification == REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE:
        return True
    if classification != REQ_CLASS_PURCHASABLE_COMPONENT_ROLE:
        return False
    return _bool_value(row.get("should_block_before_composer", True))


def _classified_should_appear(row: Mapping[str, Any]) -> bool:
    if "should_appear_in_composer_brief" in row:
        return _bool_value(row.get("should_appear_in_composer_brief"))
    return True


def _classified_should_validate(
    row: Mapping[str, Any],
    *,
    classification: str,
    hard_or_optional: str,
    fulfillment_mode: str,
) -> bool:
    if "should_validate_after_composer" in row:
        return _bool_value(row.get("should_validate_after_composer"))
    if "should_be_validated_after_composer" in row:
        return _bool_value(row.get("should_be_validated_after_composer"))
    if fulfillment_mode in {
        FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
        FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
        FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
        FULFILLMENT_ENGINEERING_CHECK_ONLY,
    }:
        return True
    return hard_or_optional == REQ_HARD or classification == REQ_CLASS_ENGINEERING_CHECK


def _unique_classified_requirements(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for index, value in enumerate(values, start=1):
        key = _classified_requirement_identity(value, fallback_index=index)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result


def _classified_requirement_identity(
    value: Mapping[str, Any],
    *,
    fallback_index: int,
) -> tuple[str, str, str, str]:
    return (
        str(value.get("requirement_id") or f"req_{fallback_index}"),
        _normalize_requirement_source(value.get("source_text")),
        str(value.get("target_role") or value.get("role") or ""),
        str(value.get("classification") or ""),
    )


def _derive_missing_classified_requirements(
    classified_requirements: Sequence[Mapping[str, Any]],
    *,
    requirements: Sequence[Mapping[str, Any]],
    required_capabilities: Sequence[Mapping[str, Any]],
    optional_capabilities: Sequence[Mapping[str, Any]],
    profile: ProductGroupProfile,
    product_group: str,
    primary_object: str,
) -> list[dict[str, Any]]:
    result = [dict(row) for row in classified_requirements]
    known_keys = {
        _classified_requirement_identity(row, fallback_index=index)
        for index, row in enumerate(result, start=1)
    }
    primary_role = _primary_object_role(profile, primary_object)
    rows = [
        *_mapping_rows(requirements),
        *_mapping_rows(required_capabilities),
        *_mapping_rows(optional_capabilities),
    ]
    for index, row in enumerate(rows, start=1):
        source_text = str(
            row.get("source_text")
            or row.get("requirement_text")
            or row.get("text")
            or ""
        ).strip()
        if not _normalize_requirement_source(source_text):
            continue
        role = _normalize_role(row.get("target_role") or row.get("role"))
        if not role or role == UNMAPPED_ROLE:
            continue
        parsed = _mapping(row.get("parsed_requirements"))
        if source_text.startswith("Base "):
            continue
        classification = _default_requirement_classification_for_role(
            role,
            profile=profile,
            primary_role=primary_role,
            parsed_requirements=parsed,
        )
        requirement_id = str(
            row.get("requirement_id") or row.get("capability_id") or f"derived_{index}"
        ).strip()
        identity = _classified_requirement_identity(
            {
                "requirement_id": requirement_id,
                "source_text": source_text,
                "target_role": role,
                "classification": classification,
            },
            fallback_index=index,
        )
        if identity in known_keys:
            continue
        hard_or_optional = (
            REQ_HARD if _bool_value(row.get("hard", True)) else REQ_OPTIONAL
        )
        coerced = _coerce_classified_requirements(
            [
                {
                    "requirement_id": requirement_id,
                    "source_text": source_text,
                    "classification": classification,
                    "product_group": product_group,
                    "target_role": role,
                    "target_primary_object": primary_object,
                    "hard_or_optional": hard_or_optional,
                    "reason": "Derived from role plan for diagnostics.",
                    "confidence": row.get("confidence") or "medium",
                    "parsed_requirements": parsed,
                }
            ],
            profile=profile,
            product_group=product_group,
            primary_object=primary_object,
        )
        if coerced:
            result.extend(coerced)
            known_keys.add(identity)
    return _unique_classified_requirements(result)


def _default_requirement_classification_for_role(
    role: str,
    *,
    profile: ProductGroupProfile,
    primary_role: str | None,
    parsed_requirements: Mapping[str, Any] | None = None,
) -> str:
    if role == primary_role:
        if _primary_role_capability_has_feature_constraints(parsed_requirements):
            return REQ_CLASS_PRIMARY_OBJECT_FEATURE
        return REQ_CLASS_PURCHASABLE_COMPONENT_ROLE
    if role in {"license", "support"}:
        return REQ_CLASS_SERVICE_OR_SUPPORT
    catalog_entry = profile.role_catalog.get(role)
    if catalog_entry is not None and catalog_entry.behavior == "optional":
        return REQ_CLASS_ACCESSORY_OR_CONSUMABLE
    return REQ_CLASS_PURCHASABLE_COMPONENT_ROLE


def _primary_role_capability_has_feature_constraints(
    parsed_requirements: Mapping[str, Any] | None,
) -> bool:
    parsed = _mapping(parsed_requirements)
    for key, value in parsed.items():
        if key in {"required", "count", "quantity", "system_count"}:
            continue
        if value in (None, "", [], {}, UNKNOWN_FACT):
            continue
        return True
    return False


def _capabilities_from_classified_requirements(
    classified_requirements: Sequence[Mapping[str, Any]],
    *,
    profile: ProductGroupProfile,
    hard: bool,
    existing_capabilities: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    existing_keys = {
        (
            _normalize_requirement_source(
                row.get("source_text") or row.get("requirement_text")
            ),
            _normalize_role(row.get("role")),
        )
        for row in existing_capabilities
    }
    for row in classified_requirements:
        hard_row = str(row.get("hard_or_optional") or "").strip() == REQ_HARD
        if hard_row != hard:
            continue
        classification = str(row.get("classification") or "").strip()
        if classification in {
            REQ_CLASS_ENGINEERING_CHECK,
            REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
            REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
        }:
            continue
        if not _classified_requirement_should_materialize_capability(row):
            continue
        role = _target_role_for_classified_requirement(row, profile)
        if not role:
            continue
        row_key = (_normalize_requirement_source(row.get("source_text")), role)
        if row_key in existing_keys:
            continue
        if (
            classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE
            and role == UNMAPPED_ROLE
        ):
            continue
        capability_id = str(
            row.get("capability_id")
            or f"{role}.classified.{row.get('requirement_id') or 'requirement'}"
        ).strip()
        capability = _capability(
            role=role,
            capability_id=capability_id,
            requirement_text=str(row.get("source_text") or role).strip(),
            parsed_requirements=_mapping(row.get("parsed_requirements"))
            or {"required": True},
            hard=hard,
        )
        capability.update(_classification_fields_for_capability(row))
        result.append(capability)
    return result


def _classified_requirement_should_materialize_capability(
    classified: Mapping[str, Any],
) -> bool:
    classification = str(classified.get("classification") or "").strip()
    fulfillment_mode = str(classified.get("fulfillment_mode") or "").strip()
    if fulfillment_mode == FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION:
        return False
    if fulfillment_mode in {
        FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
        FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        FULFILLMENT_NOT_APPLICABLE,
        FULFILLMENT_LOGISTICS_CONSTRAINT,
        FULFILLMENT_ENGINEERING_CHECK_ONLY,
    }:
        return False
    if classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE:
        return True
    return bool(classified.get("should_create_bom_role"))


def _apply_classified_requirement_policy(
    *,
    required_capabilities: list[dict[str, Any]],
    optional_capabilities: list[dict[str, Any]],
    unsupported: list[str],
    requirements: list[dict[str, Any]],
    engineer_review_instructions: list[str],
    classified_requirements: Sequence[Mapping[str, Any]],
    profile: ProductGroupProfile,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
    list[str],
]:
    required = _rewrite_capabilities_by_classification(
        required_capabilities,
        classified_requirements=classified_requirements,
        profile=profile,
    )
    optional = _rewrite_capabilities_by_classification(
        optional_capabilities,
        classified_requirements=classified_requirements,
        profile=profile,
    )
    non_blocking_sources = {
        _normalize_requirement_source(row.get("source_text"))
        for row in classified_requirements
        if str(row.get("classification") or "") in NON_BLOCKING_REQUIREMENT_CLASSIFICATIONS
    }
    unsupported = [
        item
        for item in unsupported
        if _normalize_requirement_source(item) not in non_blocking_sources
    ]
    requirements = _rewrite_requirement_items_by_classification(
        requirements,
        classified_requirements=classified_requirements,
        profile=profile,
    )
    checks = [
        str(
            row.get("engineer_check_ru")
            or row.get("suggested_engineer_check_ru")
            or row.get("source_text")
            or ""
        ).strip()
        for row in classified_requirements
        if str(row.get("classification") or "") == REQ_CLASS_ENGINEERING_CHECK
        or str(row.get("fulfillment_mode") or "")
        == FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION
        or (
            str(row.get("classification") or "") == REQ_CLASS_PRIMARY_OBJECT_FEATURE
            and str(row.get("hard_or_optional") or "") == REQ_HARD
        )
    ]
    engineer_review_instructions = _unique(
        [*engineer_review_instructions, *[check for check in checks if check]]
    )
    return required, optional, unsupported, requirements, engineer_review_instructions


def _rewrite_capabilities_by_classification(
    capabilities: Sequence[Mapping[str, Any]],
    *,
    classified_requirements: Sequence[Mapping[str, Any]],
    profile: ProductGroupProfile,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for capability in capabilities:
        classified = _classified_requirement_for_capability(
            capability,
            classified_requirements,
        )
        if not classified:
            result.append(dict(capability))
            continue
        classification = str(classified.get("classification") or "").strip()
        if classification in {
            REQ_CLASS_ENGINEERING_CHECK,
            REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
            REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
        }:
            continue
        if not _classified_requirement_should_materialize_capability(classified):
            continue
        updated = dict(capability)
        role = _role_for_classified_capability(
            str(updated.get("role") or ""),
            classified,
            profile=profile,
        )
        if classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE and role == UNMAPPED_ROLE:
            continue
        if role:
            updated["role"] = role
        updated.update(_classification_fields_for_capability(classified))
        updated["can_be_satisfied_by_platform"] = (
            classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE
            or _capability_can_be_satisfied_by_platform(updated)
        )
        result.append(updated)
    return _unique_capabilities([dict(row) for row in result])


def _rewrite_requirement_items_by_classification(
    requirements: Sequence[Mapping[str, Any]],
    *,
    classified_requirements: Sequence[Mapping[str, Any]],
    profile: ProductGroupProfile,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for requirement in requirements:
        classified = _classified_requirement_for_capability(
            requirement,
            classified_requirements,
        )
        if not classified:
            result.append(dict(requirement))
            continue
        classification = str(classified.get("classification") or "").strip()
        if classification in {
            REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
            REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
        }:
            result.append(dict(requirement))
            continue
        if classification == REQ_CLASS_ENGINEERING_CHECK:
            updated = dict(requirement)
            updated["classification"] = CLASS_ENGINEER_REVIEW_INSTRUCTION
            result.append(updated)
            continue
        updated = dict(requirement)
        role = _role_for_classified_capability(
            str(updated.get("role") or ""),
            classified,
            profile=profile,
        )
        if role:
            updated["role"] = role
        updated["requirement_classification"] = classification
        updated.update(_classification_fields_for_capability(classified))
        result.append(updated)
    return [dict(row) for row in result]


def _classification_fields_for_capability(
    classified: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "classified_requirement_id": classified.get("requirement_id"),
        "requirement_classification": classified.get("classification"),
        "target_primary_object": classified.get("target_primary_object"),
        "fulfillment_mode": classified.get("fulfillment_mode"),
        "fulfillment_target_role": classified.get("fulfillment_target_role"),
        "fulfillment_target_component_candidate_id": classified.get(
            "fulfillment_target_component_candidate_id"
        ),
        "evidence_source": classified.get("evidence_source"),
        "evidence_text": classified.get("evidence_text"),
        "should_create_bom_role": bool(classified.get("should_create_bom_role")),
        "category_needed": bool(classified.get("category_needed")),
        "should_block_before_composer": bool(
            classified.get("should_block_before_composer")
        ),
        "should_validate_after_composer": bool(
            classified.get("should_validate_after_composer")
        ),
        "should_be_validated_after_composer": bool(
            classified.get("should_be_validated_after_composer")
        ),
        "engineer_check_ru": str(
            classified.get("engineer_check_ru")
            or classified.get("suggested_engineer_check_ru")
            or ""
        ).strip(),
        "suggested_engineer_check_ru": str(
            classified.get("suggested_engineer_check_ru") or ""
        ).strip(),
    }


def _role_for_classified_capability(
    current_role: str,
    classified: Mapping[str, Any],
    *,
    profile: ProductGroupProfile,
) -> str:
    classification = str(classified.get("classification") or "").strip()
    role = _target_role_for_classified_requirement(classified, profile)
    if role:
        return role
    if classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE:
        return _primary_object_role(
            profile,
            str(classified.get("target_primary_object") or ""),
        ) or current_role
    return current_role


def _target_role_for_classified_requirement(
    classified: Mapping[str, Any],
    profile: ProductGroupProfile,
) -> str:
    role = _normalize_role(classified.get("target_role") or classified.get("role"))
    if role in profile.role_catalog or role == UNMAPPED_ROLE:
        return role
    primary_role = _primary_object_role(
        profile,
        str(classified.get("target_primary_object") or ""),
    )
    if str(classified.get("classification") or "") == REQ_CLASS_PRIMARY_OBJECT_FEATURE:
        return primary_role or ""
    return ""


def _primary_object_role(
    profile: ProductGroupProfile,
    primary_object: str,
) -> str | None:
    normalized = _normalize_role(primary_object)
    if normalized in profile.role_catalog:
        return normalized
    if profile.required_roles:
        return profile.required_roles[0]
    return None


def _classified_requirement_for_capability(
    capability: Mapping[str, Any],
    classified_requirements: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    capability_id = str(capability.get("capability_id") or "").strip()
    if capability_id:
        for row in classified_requirements:
            if str(row.get("capability_id") or "").strip() == capability_id:
                return row
    source_text = str(
        capability.get("source_text")
        or capability.get("requirement_text")
        or capability.get("text")
        or ""
    ).strip()
    source_key = _normalize_requirement_source(source_text)
    if not source_key:
        return None
    role = _normalize_role(capability.get("role"))
    source_matches = [
        row
        for row in classified_requirements
        if _normalize_requirement_source(row.get("source_text")) == source_key
    ]
    for row in source_matches:
        if _classified_requirement_matches_role(row, role):
            return row
    if len(source_matches) == 1 and _classified_requirement_can_claim_source(
        source_matches[0],
        role,
    ):
        return source_matches[0]
    return None


def _classified_requirement_matches_role(
    classified: Mapping[str, Any],
    role: str,
) -> bool:
    target_role = _normalize_role(
        classified.get("target_role") or classified.get("role")
    )
    if target_role and target_role == role:
        return True
    classification = str(classified.get("classification") or "").strip()
    if classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE and role == UNMAPPED_ROLE:
        return True
    if classification == REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE:
        return role == UNMAPPED_ROLE
    return False


def _classified_requirement_can_claim_source(
    classified: Mapping[str, Any],
    role: str,
) -> bool:
    if _classified_requirement_matches_role(classified, role):
        return True
    target_role = _normalize_role(
        classified.get("target_role") or classified.get("role")
    )
    classification = str(classified.get("classification") or "").strip()
    if target_role and role and target_role != role:
        return False
    return classification in {
        REQ_CLASS_ENGINEERING_CHECK,
        REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
    }


def _classified_requirement_for_source(
    source_text: str,
    classified_requirements: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    source_key = _normalize_requirement_source(source_text)
    if not source_key:
        return None
    for row in classified_requirements:
        if _normalize_requirement_source(row.get("source_text")) == source_key:
            return row
    return None


def _normalize_requirement_source(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _requirement_source_coverage_diagnostics(
    *,
    request_text: str,
    intent: Mapping[str, Any] | None,
    classified_requirements: Sequence[Mapping[str, Any]],
    profile: ProductGroupProfile,
    repair_attempted: bool,
    repair_stage: bool,
) -> dict[str, Any]:
    fragments = _semantic_classifier_source_fragments(request_text, intent)
    rows = [dict(row) for row in classified_requirements]
    coverage: list[dict[str, Any]] = []
    covered_count = 0
    for fragment in fragments:
        fragment_text = str(fragment.get("source_text") or "").strip()
        matched = _source_fragment_match(
            fragment_text,
            rows,
            request_text=request_text,
            profile=profile,
        )
        if matched is not None:
            covered_count += 1
        coverage.append(
            {
                "source_text": fragment_text,
                "source": fragment.get("source"),
                "covered": matched is not None,
                "requirement_id": matched.get("requirement_id") if matched else None,
                "classified_source_text": (
                    matched.get("source_text") if matched else None
                ),
                "classification": matched.get("classification") if matched else None,
                "target_role": matched.get("target_role") if matched else None,
            }
        )
    fragment_count = len(fragments)
    coverage_percent = (
        round((covered_count / fragment_count) * 100, 2) if fragment_count else 100.0
    )
    synthetic_count = sum(
        1
        for row in rows
        if _classified_requirement_is_synthetic(
            row,
            request_text=request_text,
            profile=profile,
        )
    )
    source_backed_count = sum(
        1
        for row in rows
        if _classified_requirement_is_source_backed(
            row,
            request_text=request_text,
            profile=profile,
        )
    )
    unclassified = [
        str(row.get("source_text") or "").strip()
        for row in coverage
        if not row.get("covered") and str(row.get("source_text") or "").strip()
    ]
    quality, accepted, incomplete_reason = _requirement_classifier_repair_quality(
        coverage_percent=coverage_percent,
        fragment_count=fragment_count,
        synthetic_requirement_count=synthetic_count,
        source_backed_requirement_count=source_backed_count,
        unclassified_source_fragments=unclassified,
        repair_attempted=repair_attempted,
        repair_stage=repair_stage,
    )
    return {
        "requirement_source_coverage": coverage,
        "requirement_source_coverage_percent": coverage_percent,
        "unclassified_source_fragments": unclassified,
        "synthetic_requirement_count": synthetic_count,
        "source_backed_requirement_count": source_backed_count,
        "requirement_classifier_repair_quality": quality,
        "requirement_classifier_repair_accepted": accepted,
        "requirement_classifier_incomplete_reason": incomplete_reason,
    }


def _source_fragment_match(
    fragment_text: str,
    classified_requirements: Sequence[Mapping[str, Any]],
    *,
    request_text: str,
    profile: ProductGroupProfile,
) -> Mapping[str, Any] | None:
    for row in classified_requirements:
        if _classified_requirement_is_synthetic(
            row,
            request_text=request_text,
            profile=profile,
        ):
            continue
        if _source_fragment_covered_by_requirement(fragment_text, row):
            return row
    return None


def _source_fragment_covered_by_requirement(
    fragment_text: str,
    requirement: Mapping[str, Any],
) -> bool:
    fragment = _normalize_requirement_source(fragment_text)
    source = _normalize_requirement_source(requirement.get("source_text"))
    if not fragment or not source:
        return False
    if fragment in source or (len(source) >= 4 and source in fragment):
        return True
    fragment_tokens = _requirement_source_tokens(fragment)
    source_tokens = set(_requirement_source_tokens(source))
    if not fragment_tokens or not source_tokens:
        return False
    overlap = [token for token in fragment_tokens if token in source_tokens]
    if len(fragment_tokens) <= 2:
        return bool(overlap) and any(_requirement_token_is_specific(token) for token in overlap)
    return len(overlap) / len(fragment_tokens) >= 0.7


def _classified_requirement_is_synthetic(
    requirement: Mapping[str, Any],
    *,
    request_text: str,
    profile: ProductGroupProfile,
) -> bool:
    source = _normalize_requirement_source(requirement.get("source_text"))
    if not source:
        return True
    role = _normalize_role(
        requirement.get("target_role")
        or requirement.get("role")
        or requirement.get("fulfillment_target_role")
    )
    role_texts = _synthetic_role_source_texts(role, profile=profile)
    if source in role_texts:
        return True
    if len(_requirement_source_tokens(source)) == 1 and source in role_texts:
        return True
    request = _normalize_requirement_source(request_text)
    return not request and source in role_texts


def _classified_requirement_is_source_backed(
    requirement: Mapping[str, Any],
    *,
    request_text: str,
    profile: ProductGroupProfile,
) -> bool:
    if _classified_requirement_is_synthetic(
        requirement,
        request_text=request_text,
        profile=profile,
    ):
        return False
    source = _normalize_requirement_source(requirement.get("source_text"))
    request = _normalize_requirement_source(request_text)
    if source and request and source in request:
        return True
    source_tokens = _requirement_source_tokens(source)
    request_tokens = set(_requirement_source_tokens(request))
    if not source_tokens:
        return False
    overlap = [token for token in source_tokens if token in request_tokens]
    if len(source_tokens) <= 2:
        return bool(overlap) and any(_requirement_token_is_specific(token) for token in overlap)
    return len(overlap) / len(source_tokens) >= 0.5


def _synthetic_role_source_texts(
    role: str,
    *,
    profile: ProductGroupProfile,
) -> set[str]:
    normalized_role = _normalize_requirement_source(role)
    values = {
        normalized_role,
        _normalize_requirement_source(role.replace("_", " ")),
        _normalize_requirement_source(f"{role}s"),
        _normalize_requirement_source(f"{role.replace('_', ' ')}s"),
    }
    catalog_entry = profile.role_catalog.get(role)
    if catalog_entry is not None:
        values.add(_normalize_requirement_source(catalog_entry.role_id))
        values.add(_normalize_requirement_source(catalog_entry.display_name_ru))
        for synonym in catalog_entry.synonyms:
            normalized = _normalize_requirement_source(synonym)
            values.add(normalized)
            values.add(_normalize_requirement_source(f"{synonym}s"))
    return {value for value in values if value}


def _requirement_source_tokens(text: str) -> list[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "in",
        "least",
        "need",
        "of",
        "or",
        "pcs",
        "the",
        "with",
        "и",
        "или",
        "на",
        "не",
        "менее",
        "нужно",
        "по",
        "с",
    }
    tokens = [
        token.strip("_").casefold()
        for token in re.findall(r"[0-9a-zA-ZА-Яа-яЁё]+", str(text or ""))
    ]
    return [
        token
        for token in tokens
        if token and (token not in stop_words or _requirement_token_is_specific(token))
    ]


def _requirement_token_is_specific(token: str) -> bool:
    return bool(re.search(r"\d", token) or len(token) >= 3)


def _requirement_classifier_repair_quality(
    *,
    coverage_percent: float,
    fragment_count: int,
    synthetic_requirement_count: int,
    source_backed_requirement_count: int,
    unclassified_source_fragments: Sequence[str],
    repair_attempted: bool,
    repair_stage: bool,
) -> tuple[str, bool, str | None]:
    if not repair_stage:
        return ("not_repair", False, None)
    if fragment_count == 0:
        return ("no_source_fragments", True, None)
    if source_backed_requirement_count == 0 and synthetic_requirement_count > 0:
        return (
            "synthetic_only",
            False,
            "repair_returned_only_synthetic_role_requirements",
        )
    if coverage_percent < 80.0:
        return (
            "incomplete_source_coverage",
            False,
            "repair_source_coverage_below_threshold",
        )
    if unclassified_source_fragments:
        return (
            "incomplete_source_coverage",
            False,
            "repair_left_unclassified_source_fragments",
        )
    return ("accepted", bool(repair_attempted), None)


def _requirement_classification_diagnostics(
    classified_requirements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in classified_requirements]
    buckets = {
        "purchasable_role_requirements": REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        "primary_object_feature_requirements": REQ_CLASS_PRIMARY_OBJECT_FEATURE,
        "accessory_or_consumable_requirements": REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
        "service_or_support_requirements": REQ_CLASS_SERVICE_OR_SUPPORT,
        "logistics_or_commercial_constraints": (
            REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT
        ),
        "engineering_check_requirements": REQ_CLASS_ENGINEERING_CHECK,
        "unmapped_requirements_non_blocking": (
            REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING
        ),
        "unmapped_requirements_blocking": (
            REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE
        ),
    }
    diagnostics = {
        "classified_requirements": rows,
        "requirement_role_mapping_decision": [
            {
                "requirement_id": row.get("requirement_id"),
                "source_text": row.get("source_text"),
                "classification": row.get("classification"),
                "target_role": row.get("target_role"),
                "target_primary_object": row.get("target_primary_object"),
                "fulfillment_mode": row.get("fulfillment_mode"),
                "fulfillment_target_role": row.get("fulfillment_target_role"),
                "fulfillment_target_component_candidate_id": row.get(
                    "fulfillment_target_component_candidate_id"
                ),
                "evidence_source": row.get("evidence_source"),
                "evidence_text": row.get("evidence_text"),
                "should_create_bom_role": row.get("should_create_bom_role"),
                "category_needed": row.get("category_needed"),
                "should_block_before_composer": row.get(
                    "should_block_before_composer"
                ),
                "should_validate_after_composer": row.get(
                    "should_validate_after_composer"
                ),
                "reason": row.get("reason"),
            }
            for row in rows
        ],
        "requirement_fulfillment_decision": [
            {
                "requirement_id": row.get("requirement_id"),
                "source_text": row.get("source_text"),
                "classification": row.get("classification"),
                "fulfillment_mode": row.get("fulfillment_mode"),
                "target_role": row.get("target_role"),
                "fulfillment_target_role": row.get("fulfillment_target_role"),
                "evidence_source": row.get("evidence_source"),
                "evidence_text": row.get("evidence_text"),
                "should_create_bom_role": row.get("should_create_bom_role"),
                "should_validate_after_composer": row.get(
                    "should_validate_after_composer"
                ),
                "engineer_check_ru": row.get("engineer_check_ru"),
            }
            for row in rows
        ],
    }
    for key, classification in buckets.items():
        diagnostics[key] = [
            row for row in rows if row.get("classification") == classification
        ]
    for mode in sorted(FULFILLMENT_VALUES):
        diagnostics[f"requirements_fulfillment_{mode}"] = [
            row for row in rows if row.get("fulfillment_mode") == mode
        ]
    return diagnostics


def _merge_role_requirements(
    current: dict[str, Any] | None,
    capability: Mapping[str, Any],
) -> dict[str, Any]:
    parsed = capability.get("parsed_requirements")
    merged = dict(current or {})
    if isinstance(parsed, Mapping):
        merged.update(parsed)
    merged["required"] = bool(capability.get("hard", True))
    return merged


def _unknown_product_group_plan(source_text: str) -> dict[str, Any]:
    return {
        "product_group": "unknown",
        "requirements": [],
        "required_capabilities": [],
        "optional_capabilities": [],
        **_requirement_classification_diagnostics([]),
        "workload_context": [],
        "logistics_constraints": {},
        "commercial_instructions": [],
        "response_instructions": [],
        "engineer_review_instructions": [],
        "engineer_review_required": False,
        "unsupported_or_unmapped_requirements": [source_text] if source_text else [],
        "planner_warnings": ["product_group_profile_missing"],
        "required_roles": [],
        "optional_roles": [],
        "requirements_by_role": {},
        "role_catalog": [],
    }


def _profile_for_request(
    request_text: str,
    spec: StockSpec | Mapping[str, Any] | str | None,
) -> ProductGroupProfile | None:
    items, spec_source_text, global_requirements = _spec_parts(spec)
    item_type_values = {
        str(_item_value(item, "item_type") or "").strip().casefold()
        for item in items
    }
    if "server" in item_type_values:
        return get_product_group_profile(SERVER_PRODUCT_GROUP)
    if item_type_values.intersection(
        {
            "storage",
            "storage_system",
            "storage_array",
            "san",
            "nas",
            "схд",
        }
    ):
        return get_product_group_profile(STORAGE_PRODUCT_GROUP)
    item_types = " ".join(str(_item_value(item, "item_type") or "") for item in items)
    item_names = " ".join(
        str(_item_value(item, "name") or _item_value(item, "category") or "")
        for item in items
    )
    haystack = " ".join(
        part
        for part in [
            request_text,
            spec_source_text,
            item_types,
            item_names,
            str(global_requirements or ""),
        ]
        if part
    )
    if _looks_like_server_product_group(haystack):
        return get_product_group_profile(SERVER_PRODUCT_GROUP)
    if _looks_like_storage_product_group(haystack):
        return get_product_group_profile(STORAGE_PRODUCT_GROUP)
    if _looks_like_network_product_group(haystack):
        return get_product_group_profile(NETWORK_PRODUCT_GROUP)
    return get_product_group_profile(SERVER_PRODUCT_GROUP)


def _looks_like_server_product_group(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:server|servers|cpu|processor|xeon|epyc|ram|memory)\b|"
            r"сервер|процессор|оператив|памят",
            text,
            re.I,
        )
    )


def _looks_like_network_product_group(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:switch|router|firewall|ngfw|utm|access\s*point|wi-?fi|"
            r"transceiver|optic|sfp\+?|sfp28|qsfp\+?|qsfp28|dac|uplink|poe|"
            r"stacking|stack)\b|"
            r"коммутатор|свитч|маршрутизатор|роутер|межсетев|фаервол|"
            r"точк[аи]\s+доступа|трансивер|аплинк|стек|стекир|"
            r"\bL[23]\b",
            text,
            re.I,
        )
    )


def _looks_like_storage_product_group(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:storage\s+(?:system|array)|san|nas|disk\s+shelf|drive\s+shelf|"
            r"storage\s+controller|usable\s+capacity|raw\s+capacity|nvme-?of|"
            r"fc\s*\d{1,3}\s*g|iscsi\s*\d{1,3}\s*g)\b|"
            r"схд|систем[ауы]\s+хранени|дисков\w+\s+массив|полезн\w+\s+емкост|"
            r"сыр\w+\s+емкост|raw\s*\d|полк[аи]\s+расширени|"
            r"fc\s*\d{1,3}\s*g|iscsi\s*\d{1,3}\s*g",
            text,
            re.I,
        )
    )


def _detect_product_group(text: str, profile: ProductGroupProfile) -> str:
    lowered = text.casefold()
    if profile.product_group_id == "server":
        if any(marker in lowered for marker in ("server", "сервер", "srv", "cpu", "ram")):
            return "server"
        return "server"
    if profile.product_group_id == NETWORK_PRODUCT_GROUP:
        if _looks_like_network_product_group(text):
            return NETWORK_PRODUCT_GROUP
        return NETWORK_PRODUCT_GROUP if lowered else "unknown"
    if profile.product_group_id == STORAGE_PRODUCT_GROUP:
        if _looks_like_storage_product_group(text):
            return STORAGE_PRODUCT_GROUP
        return STORAGE_PRODUCT_GROUP if lowered else "unknown"
    return profile.product_group_id if lowered else "unknown"


def _deterministic_logistics_constraints(text: str) -> dict[str, Any]:
    if re.search(r"склад\s+москв|warehouse\s+moscow|moscow\s+warehouse", text, re.I):
        return {"shipment_city": "Москва"}
    return {}


def _deterministic_commercial_instructions(text: str) -> list[dict[str, Any]]:
    if not re.search(r"сам(?:ый|ая|ое)\s+дешев|cheapest|lowest\s+cost", text, re.I):
        return []
    return [
        _instruction_row(
            "один самый дешевый вариант",
            {
                "optimization_goal": "cheapest_valid_stock_quote",
                "alternatives_required": False,
            },
        )
    ]


def _deterministic_engineer_review_instructions(text: str) -> list[str]:
    if re.search(r"проверить\s+инженер|инженер\w+\s+провер|engineer", text, re.I):
        return ["проверить инженером"]
    return []


def _requirement_planner_system_prompt(profile: ProductGroupProfile) -> str:
    group_specific = ""
    if profile.product_group_id == SERVER_PRODUCT_GROUP:
        group_specific = (
            "For server PSU redundancy/completeness inside each server, use role=\"power_supply\" "
            "with parsed psu_count_per_server/count and do not require separate PSU modules "
            "unless the user asks for spare, replacement, extra, or separate PSUs. Include "
            "base server roles needed for a complete build. "
        )
    elif profile.product_group_id == NETWORK_PRODUCT_GROUP:
        group_specific = (
            "For network requests, preserve explicit hard asks such as port_count, port_speed, "
            "port_media, uplink_count, uplink_speed, uplink_media, poe_required, poe_budget_w, "
            "poe_standard, l2_required, l3_required, stacking_required, airflow, redundant_psu, "
            "license_required and support_required. Use roles from the network role catalog "
            "such as switch, router, firewall, access_point, transceiver, dac_cable, cable, "
            "license, support, power_supply, stacking_module and other_accessory. Do not turn "
            "warehouse/city, cheapest option, КП/quote wording, support review wording, or "
            "manual engineering review instructions into unsupported hard requirements. "
        )
    elif profile.product_group_id == STORAGE_PRODUCT_GROUP:
        group_specific = (
            "For storage/SAN/NAS/СХД requests, preserve explicit hard asks such as "
            "usable_capacity_tb, raw_capacity_tb, redundancy_level, controller_count, "
            "drive_count, drive_capacity_tb, drive_type, drive_interface, host_protocol, "
            "host_port_count, host_port_speed, host_port_media, nvme_of_required, "
            "fc_required, iscsi_required, sas_required, license_required, "
            "support_required, warranty_months, rail_kit_required and redundant_psu. "
            "Use roles from the storage role catalog such as storage_system, controller, "
            "controller_module, disk_shelf, drive, ssd, hdd, host_port, protocol_module, "
            "transceiver, cable, license, support, power_supply, rail_kit and "
            "other_accessory. Do not turn warehouse/city, cheapest option, КП/quote "
            "wording, one-option wording, or manual engineering review instructions into "
            "unsupported hard requirements. "
        )
    return (
        "You are Universal Requirement Planner. Return JSON only. Extract every explicit "
        "fragment from the user request and classify it semantically as one of: "
        "hard_technical_requirement, soft_preference, workload_context, "
        "logistics_constraint, commercial_instruction, response_instruction, "
        "engineer_review_instruction, unsupported_hard_requirement. Only hard technical "
        "requirements become required_capabilities. Soft preferences become "
        "optional_capabilities only when they map to a role. Workload/use-case context, "
        "logistics such as city or warehouse, KP/quote wording, cheapest-stock "
        "commercial goals, response formatting, and manual/engineering review "
        "instructions are never unsupported hard requirements. The known role_catalog "
        "helps you choose stable roles, but it must not limit your reasoning: use known "
        "roles when they fit; if none fit, set role=\"unmapped\" and preserve the hard "
        "technical requirement as an unmapped required capability. Do not invent "
        "category_id or component_candidate_id values. Do not drop explicit hard "
        "technical requirements. Also return classified_requirements using the universal "
        "classification contract: purchasable_component_role, primary_object_feature, "
        "accessory_or_consumable, service_or_support, logistics_or_commercial_constraint, "
        "engineering_check, out_of_scope_or_unmapped_non_blocking, or "
        "blocking_unmapped_purchasable_role. For each classified requirement also choose "
        "fulfillment_mode: separate_component_required, included_in_primary_object, "
        "included_in_selected_component, included_in_bundle_or_kit, service_or_support, "
        "logistics_constraint, engineering_check_only, unverified_requires_confirmation, "
        "or not_applicable. Use should_create_bom_role=true only when a separate BOM "
        "role/category/matrix line is required. If you believe a requirement is included "
        "in a platform, bundle, kit, or selected component, provide evidence_source and "
        "evidence_text; without evidence, use unverified_requires_confirmation. Feature "
        "constraints of the primary object must not become role=\"unmapped\". "
        f"{group_specific}"
        "Product group is "
        f"{profile.product_group_id} unless clearly unknown. Return JSON only."
    )


def _profile_prompt_payload(profile: ProductGroupProfile) -> dict[str, Any]:
    return {
        "product_group_id": profile.product_group_id,
        "required_roles": list(profile.required_roles),
        "role_catalog": [
            {
                "role_id": entry.role_id,
                "display_name_ru": entry.display_name_ru,
                "synonyms": list(entry.synonyms),
                "behavior": entry.behavior,
                "quantity_rule": entry.quantity_rule,
                "validation_capabilities": list(entry.validation_capabilities),
            }
            for entry in profile.role_catalog.values()
        ],
    }


def _safe_plan_for_prompt(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "product_group": plan.get("product_group"),
        "requirements": plan.get("requirements"),
        "classified_requirements": plan.get("classified_requirements"),
        "required_capabilities": plan.get("required_capabilities"),
        "optional_capabilities": plan.get("optional_capabilities"),
        "workload_context": plan.get("workload_context"),
        "logistics_constraints": plan.get("logistics_constraints"),
        "commercial_instructions": plan.get("commercial_instructions"),
        "response_instructions": plan.get("response_instructions"),
        "engineer_review_required": plan.get("engineer_review_required"),
        "unsupported_or_unmapped_requirements": plan.get(
            "unsupported_or_unmapped_requirements"
        ),
        "planner_warnings": plan.get("planner_warnings"),
    }


def _preserve_deterministic_semantic_fallback(
    base_plan: Mapping[str, Any],
    llm_plan: Mapping[str, Any],
    *,
    profile: ProductGroupProfile,
) -> dict[str, Any]:
    if profile.product_group_id not in {NETWORK_PRODUCT_GROUP, STORAGE_PRODUCT_GROUP}:
        return dict(llm_plan)

    base_required = [dict(row) for row in _mapping_rows(base_plan.get("required_capabilities"))]
    if not base_required:
        return dict(llm_plan)

    llm_required = [dict(row) for row in _mapping_rows(llm_plan.get("required_capabilities"))]
    base_roles = _string_list(base_plan.get("required_roles"))
    llm_roles = set(_string_list(llm_plan.get("required_roles")))
    missing_base_roles = [role for role in base_roles if role not in llm_roles]
    if (
        llm_required
        and not missing_base_roles
        and _llm_capabilities_cover_base_requirements(base_required, llm_required)
    ):
        return dict(llm_plan)

    merged = dict(llm_plan)
    merged["required_capabilities"] = _unique_capabilities_prefer_later(
        [*llm_required, *base_required]
    )
    merged["optional_capabilities"] = _unique_capabilities_prefer_later(
        [
            *[dict(row) for row in _mapping_rows(llm_plan.get("optional_capabilities"))],
            *[dict(row) for row in _mapping_rows(base_plan.get("optional_capabilities"))],
        ]
    )
    if not _mapping(llm_plan.get("logistics_constraints")):
        merged["logistics_constraints"] = base_plan.get("logistics_constraints")
    if not _instruction_rows(llm_plan.get("commercial_instructions")):
        merged["commercial_instructions"] = base_plan.get("commercial_instructions")
    merged["planner_warnings"] = _unique(
        [
            *_string_list(llm_plan.get("planner_warnings")),
            "requirement_planner_deterministic_semantic_fallback_preserved",
        ]
    )
    return _coerce_requirement_plan(merged, profile=profile)


def _llm_capabilities_cover_base_requirements(
    base_required: Sequence[Mapping[str, Any]],
    llm_required: Sequence[Mapping[str, Any]],
) -> bool:
    for base_capability in base_required:
        role = str(base_capability.get("role") or "").strip()
        if not role:
            continue
        if not any(
            _capability_covers_required_parsed_values(candidate, base_capability)
            for candidate in llm_required
            if str(candidate.get("role") or "").strip() == role
        ):
            return False
    return True


def _capability_covers_required_parsed_values(
    candidate: Mapping[str, Any],
    required: Mapping[str, Any],
) -> bool:
    candidate_parsed = _mapping(candidate.get("parsed_requirements"))
    required_parsed = _mapping(required.get("parsed_requirements"))
    for key, required_value in required_parsed.items():
        if required_value in (None, "", UNKNOWN_FACT):
            continue
        if key == "required":
            continue
        candidate_value = candidate_parsed.get(key)
        if candidate_value in (None, "", UNKNOWN_FACT):
            return False
        if str(candidate_value).casefold() != str(required_value).casefold():
            return False
    return True


def _spec_parts(
    spec: StockSpec | Mapping[str, Any] | str | None,
) -> tuple[list[Any], str, Mapping[str, Any]]:
    if isinstance(spec, StockSpec):
        return list(spec.items), spec.source_text or "", spec.requirements
    if isinstance(spec, Mapping):
        return (
            list(spec.get("items") or []),
            str(spec.get("source_text") or ""),
            _mapping(spec.get("requirements")),
        )
    return [], str(spec or ""), {}


def _source_text_from_spec(spec: StockSpec | Mapping[str, Any] | str | None) -> str:
    if isinstance(spec, StockSpec):
        return spec.source_text or ""
    if isinstance(spec, Mapping):
        return str(spec.get("source_text") or "")
    return str(spec or "")


def _request_text(
    source_text: str,
    item: Any,
    *,
    include_requirements: bool = True,
) -> str:
    requirements = _item_requirements(item)
    parts = [
        source_text,
        _item_value(item, "name"),
        _item_value(item, "category"),
    ]
    if include_requirements:
        parts.append(str(requirements or ""))
    return " ".join(str(part) for part in parts if part)


def _item_requirements(item: Any) -> Mapping[str, Any]:
    if isinstance(item, StockSpecItem):
        return item.requirements
    if isinstance(item, Mapping):
        return _mapping(item.get("requirements"))
    return {}


def _item_value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _requires_storage_controller(text: str, requirements: Mapping[str, Any]) -> bool:
    storage = _mapping(requirements.get("storage"))
    if storage.get("controller") or storage.get("raid") or storage.get("hba"):
        return True
    return bool(
        re.search(
            r"\b(?:RAID|HBA|FC\s+HBA|Fibre\s+Channel|tri[- ]?mode|controller)\b|контроллер",
            text,
            re.I,
        )
    )


def _storage_controller_parsed_requirements(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    adapter_type = "fc_hba" if "fc hba" in lowered or "fibre channel" in lowered else "raid_hba"
    return {"required": True, "adapter_type": adapter_type}


def _requires_gpu(text: str, requirements: Mapping[str, Any]) -> bool:
    if requirements.get("gpu") or requirements.get("accelerator"):
        return True
    return bool(
        re.search(
            r"\b(?:GPU|NVIDIA|CUDA|accelerator|A100|H100|L40S|graphics card)\b|видеокарт",
            text,
            re.I,
        )
    )


def _requires_transceiver(text: str, requirements: Mapping[str, Any]) -> bool:
    if requirements.get("transceiver"):
        return True
    return bool(
        re.search(
            r"\b(?:transceiver|SFP module|QSFP module|optic(?:al)? module)\b|трансивер",
            text,
            re.I,
        )
    )


def _requires_cable(text: str, requirements: Mapping[str, Any]) -> bool:
    if requirements.get("cable") or requirements.get("cables"):
        return True
    return bool(re.search(r"\b(?:DAC|AOC|cables?)\b|кабел", text, re.I))


def _cable_parsed_requirements(text: str) -> dict[str, Any]:
    if re.search(r"\bDAC\b", text, re.I):
        return {"required": True, "cable_type": "DAC"}
    if re.search(r"\bAOC\b", text, re.I):
        return {"required": True, "cable_type": "AOC"}
    return {"required": True}


def _requires_power_supply(text: str, requirements: Mapping[str, Any]) -> bool:
    power = _mapping(requirements.get("power"))
    if requirements.get("power_supply"):
        return True
    if power.get("separate_psu") or power.get("spare_psu"):
        return True
    psu_count = _as_int(
        power.get("psu_count_per_server")
        or power.get("min_count")
        or power.get("count")
    )
    if psu_count is not None and psu_count >= 2:
        return True
    return bool(
        re.search(
            r"\b(?:spare|extra|separate)\s+(?:PSU|power supply)\b|"
            r"\b(?:1\+1|redundant|redundancy|2x\s*(?:PSU|power supply)|"
            r"2\s*(?:PSU|power supplies?))\b|"
            r"(?:2|\u0434\u0432\u0430)\s+(?:\u0431\u043f|"
            r"\u0431\u043b\u043e\u043a\w+\s+\u043f\u0438\u0442\u0430\u043d\w+)|"
            r"\bpower supply module\b|отдельн\w+\s+блок\w+\s+питания",
            text,
            re.I,
        )
    )


def _power_supply_parsed_requirements(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    separate = bool(
        re.search(
            r"\b(?:spare|extra|separate|replacement)\s+(?:PSU|power supply)\b|"
            r"\u0437\u0430\u043f\u0430\u0441\w*\s+"
            r"(?:\u0431\u043f|\u0431\u043b\u043e\u043a\w+\s+"
            r"\u043f\u0438\u0442\u0430\u043d\w+)",
            text,
            re.I,
        )
    )
    if (
        "1+1" in lowered
        or "redundant" in lowered
        or "redundancy" in lowered
        or re.search(r"\b2x\s*(?:PSU|power supply)", text, re.I)
        or re.search(r"\b2\s*(?:PSU|power supplies?)\b", text, re.I)
        or re.search(
            r"(?:2|\u0434\u0432\u0430)\s+(?:\u0431\u043f|"
            r"\u0431\u043b\u043e\u043a\w+\s+\u043f\u0438\u0442\u0430\u043d\w+)",
            text,
            re.I,
        )
    ):
        return {
            "required": True,
            "psu_count_per_server": 2,
            "count": 2,
            "separate_psu": separate,
        }
    if separate:
        return {"required": True, "separate_psu": True}
    return {"required": True}


def _requires_rail_kit(text: str, requirements: Mapping[str, Any]) -> bool:
    if requirements.get("rail_kit") or requirements.get("rails"):
        return True
    return bool(re.search(r"\b(?:rail kit|rails?)\b|рельс", text, re.I))


def _requires_license(text: str, requirements: Mapping[str, Any]) -> bool:
    if requirements.get("license"):
        return True
    return bool(re.search(r"\b(?:license|licence|subscription)\b|лицензи", text, re.I))


def _requires_support(text: str, requirements: Mapping[str, Any]) -> bool:
    if requirements.get("support"):
        return True
    return bool(
        re.search(
            r"\b(?:support|warranty extension|service pack)\b|поддержк|гаранти",
            text,
            re.I,
        )
    )


def _empty_parsed_requirements(text: str) -> dict[str, Any]:
    return {"required": True}


def _unknown_hard_requirement_markers(text: str) -> list[str]:
    markers: list[str] = []
    for match in re.finditer(
        r"(?:must have|required|обязательно|требуется)\s+([^.;\n]{1,120})",
        text,
        re.I,
    ):
        segment = match.group(1).strip()
        if not segment:
            continue
        for part in re.split(r"\s+(?:и|and)\s+|,", segment, flags=re.I):
            normalized = part.strip(" ,")
            if not normalized or _known_requirement_fragment(normalized):
                continue
            markers.append(normalized)
    return markers


def _known_requirement_fragment(text: str) -> bool:
    return bool(
        re.search(
            r"(?:cpu|processor|ram|memory|ssd|hdd|nvme|server|network|nic|port|"
            r"raid|hba|gpu|nvidia|cable|dac|aoc|transceiver|license|support|"
            r"switch|router|firewall|access\s*point|uplink|poe|stack|l2|l3|"
            r"storage\s+array|storage\s+system|san|nas|shelf|usable|raw|fc|iscsi|"
            r"nvme-?of|raid|controller|drive|"
            r"сервер|процессор|памят|накоп|диск|сет|порт|контроллер|кабел|"
            r"трансивер|лицензи|поддержк|гаранти|бп|питани|коммутатор|"
            r"маршрутизатор|роутер|межсетев|точк[аи]\s+доступа|аплинк|стек|"
            r"схд|полезн|сыр\w+\s+емкост|полк[аи]|рейд)",
            text,
            re.I,
        )
    )


def _storage_product_capabilities(
    text: str,
    *,
    items: Sequence[Any],
    global_requirements: Mapping[str, Any],
) -> list[dict[str, Any]]:
    roles = _storage_roles_from_sources(text, items, global_requirements)
    if not roles and _looks_like_storage_product_group(text):
        roles = ["storage_system"]

    common = _storage_common_parsed_requirements(text, global_requirements)
    if common.get("acceptable_drive_types"):
        roles = _storage_roles_with_drive_alternatives(roles)
    capabilities: list[dict[str, Any]] = []
    for role in roles:
        parsed = _storage_role_parsed_requirements(role, common, text)
        if not parsed.get("required", True):
            continue
        capabilities.append(
            _capability(
                role=role,
                capability_id=_storage_product_capability_id(role, parsed),
                requirement_text=_storage_role_requirement_text(text, role),
                parsed_requirements=parsed,
            )
        )
    return capabilities


def _storage_optional_capabilities(text: str) -> list[dict[str, Any]]:
    optional: list[dict[str, Any]] = []
    if _optional_10gbe_requested(text):
        optional.append(
            _capability(
                role="host_port",
                capability_id="host_port.10gbe.optional",
                requirement_text=_capability_text(text, "10GbE"),
                parsed_requirements={
                    "host_protocol": "Ethernet",
                    "host_port_speed": "10G",
                    "host_port_speed_gbps": 10,
                    "optional": True,
                },
                hard=False,
            )
        )
    return optional


def _storage_roles_from_sources(
    text: str,
    items: Sequence[Any],
    global_requirements: Mapping[str, Any],
) -> list[str]:
    roles: list[str] = []
    for item in items:
        item_type = str(_item_value(item, "item_type") or "").strip().casefold()
        category = str(_item_value(item, "category") or "").strip().casefold()
        name = str(_item_value(item, "name") or "").strip()
        for value in (item_type, category, name):
            role = _storage_role_from_text(value)
            if role and role not in roles:
                roles.append(role)
        requested_role = _storage_role_alias(
            str(_item_requirements(item).get("role") or "").strip()
        )
        if requested_role in _STORAGE_ROLE_IDS and requested_role not in roles:
            roles.append(requested_role)

    for key in ("role", "product_role", "product_type"):
        requested_role = _storage_role_alias(str(global_requirements.get(key) or "").strip())
        if requested_role in _STORAGE_ROLE_IDS and requested_role not in roles:
            roles.append(requested_role)

    for role in _STORAGE_ROLE_IDS:
        if _storage_role_requested(text, role) and role not in roles:
            roles.append(role)
    return roles


def _storage_roles_with_drive_alternatives(roles: Sequence[str]) -> list[str]:
    result: list[str] = []
    drive_seen = False
    for role in roles:
        if role in {"drive", "ssd", "hdd"}:
            if not drive_seen:
                result.append("drive")
                drive_seen = True
            continue
        result.append(role)
    return _unique(result)


_STORAGE_ROLE_IDS = (
    "storage_system",
    "controller",
    "controller_module",
    "disk_shelf",
    "drive",
    "ssd",
    "hdd",
    "cache",
    "host_port",
    "protocol_module",
    "transceiver",
    "cable",
    "license",
    "support",
    "power_supply",
    "rail_kit",
    "other_accessory",
)


def _storage_role_alias(role: str) -> str:
    aliases = {
        "storage": "storage_system",
        "storage_array": "storage_system",
        "san": "storage_system",
        "nas": "storage_system",
        "shelf": "disk_shelf",
        "drive_shelf": "disk_shelf",
        "drives": "drive",
        "disk": "drive",
        "disks": "drive",
        "host_ports": "host_port",
        "fc": "host_port",
        "iscsi": "host_port",
        "nvme_of": "protocol_module",
    }
    normalized = role.strip().casefold().replace("-", "_").replace(" ", "_")
    return aliases.get(normalized, normalized)


def _storage_role_from_text(value: str) -> str | None:
    for role in _STORAGE_ROLE_IDS:
        if _storage_role_requested(value, role):
            return role
    if value.strip().casefold() == "storage":
        return "storage_system"
    return None


def _storage_role_requested(text: str, role: str) -> bool:
    patterns = {
        "storage_system": (
            r"\b(?:storage\s+(?:system|array)|san|nas)\b|схд|"
            r"систем[ауы]\s+хранени|дисков\w+\s+массив"
        ),
        "controller": r"\bcontrollers?\b|контроллер",
        "controller_module": (
            r"\bcontroller\s+module\b|контроллерн\w+\s+модул|"
            r"модул\w+\s+контроллер"
        ),
        "disk_shelf": (
            r"\b(?:disk|drive|expansion)\s+shelf\b|"
            r"полк[аи]\s+(?:расширени|диск)|дисков\w+\s+полк"
        ),
        "drive": r"\b(?:drive|disk)s?\b|накопител|диск",
        "ssd": r"\bssd\b|ссд",
        "hdd": r"\bhdd\b|жестк\w+\s+диск|nl-?sas",
        "cache": r"\bcache\b|кэш|кеш",
        "host_port": (
            r"\b(?:host\s+ports?|fc\s*\d{1,3}\s*g|"
            r"iscsi\s*(?:10|25|40|100)\s*g|sas\s+host)\b|порт\w+\s+хост"
        ),
        "protocol_module": (
            r"\b(?:protocol\s+module|fc\s+module|iscsi\s+module|nvme-?of)\b|"
            r"модул\w+\s+(?:fc|iscsi|протокол)"
        ),
        "transceiver": r"\b(?:transceiver|optic(?:al)?\s+module|sfp|qsfp)\b|трансивер|оптик",
        "cable": r"\b(?:cable|dac|aoc)\b|кабел",
        "license": r"\b(?:license|licence|subscription)\b|лицензи|подписк",
        "support": r"\b(?:support|warranty|service)\b|поддержк|гаранти|сервис",
        "power_supply": (
            r"\b(?:redundant|spare|extra|separate)?\s*"
            r"(?:psu|power\s+supply)\b|бп|блок\w+\s+питани"
        ),
        "rail_kit": r"\b(?:rail kit|rails?)\b|рельс",
        "other_accessory": r"\baccessor(?:y|ies)\b|аксессуар|опци[яи]",
    }
    return bool(re.search(patterns[role], text, re.I))


def _storage_common_parsed_requirements(
    text: str,
    global_requirements: Mapping[str, Any],
) -> dict[str, Any]:
    storage = _mapping(global_requirements.get("storage"))
    parsed: dict[str, Any] = {"required": True}
    parsed.update({key: value for key, value in storage.items() if value not in (None, "")})

    system_count = _storage_system_count(text)
    if system_count is not None:
        parsed.setdefault("count", system_count)
        parsed.setdefault("system_count", system_count)

    usable_capacity_tb = _storage_capacity_tb(text, kind="usable")
    if usable_capacity_tb is not None:
        parsed.setdefault("usable_capacity_tb", usable_capacity_tb)
    raw_capacity_tb = _storage_capacity_tb(text, kind="raw")
    if raw_capacity_tb is not None:
        parsed.setdefault("raw_capacity_tb", raw_capacity_tb)
    redundancy = _storage_redundancy_level(text)
    if redundancy:
        parsed.setdefault("redundancy_level", redundancy)

    controller_count = _storage_controller_count(text)
    if controller_count is not None:
        parsed.setdefault("controller_count", controller_count)
        parsed.setdefault("controller_redundancy", controller_count >= 2)

    shelf_count = _storage_shelf_count(text)
    if shelf_count is not None:
        parsed.setdefault("shelf_count", shelf_count)
    elif re.search(r"полк[аи]\s+расширени|expansion\s+shelf", text, re.I):
        parsed.setdefault("shelf_required", True)

    drive_count = _storage_drive_count(text)
    if drive_count is not None:
        parsed.setdefault("drive_count", drive_count)
    drive_capacity_tb = _storage_drive_capacity_tb(text)
    if drive_capacity_tb is not None:
        parsed.setdefault("drive_capacity_tb", drive_capacity_tb)
    acceptable_drive_types = _storage_drive_type_alternatives(text)
    if acceptable_drive_types:
        parsed.setdefault("acceptable_drive_types", acceptable_drive_types)
        parsed.setdefault("drive_type", "any")
    else:
        drive_type = _storage_drive_type(text)
        if drive_type:
            parsed.setdefault("drive_type", drive_type)
    drive_interface = _storage_drive_interface(text)
    if drive_interface:
        parsed.setdefault("drive_interface", drive_interface)

    protocol = _storage_host_protocol(text)
    if protocol:
        parsed.setdefault("host_protocol", protocol)
        if protocol == "FC":
            parsed.setdefault("fc_required", True)
        elif protocol == "iSCSI":
            parsed.setdefault("iscsi_required", True)
        elif protocol == "NVMe-oF":
            parsed.setdefault("nvme_of_required", True)
        elif protocol == "SAS":
            parsed.setdefault("sas_required", True)
    port_speed = _storage_host_port_speed(text)
    if port_speed:
        parsed.setdefault("host_port_speed", port_speed)
    port_count = _storage_host_port_count(text)
    if port_count is not None:
        parsed.setdefault("host_port_count", port_count)
    port_media = _storage_host_port_media(text)
    if port_media:
        parsed.setdefault("host_port_media", port_media)

    if _requires_license(text, global_requirements):
        parsed.setdefault("license_required", True)
    if re.search(r"в\s+комплект|included|in\s+bundle", text, re.I):
        parsed.setdefault("included", True)
    if _requires_support(text, global_requirements):
        parsed.setdefault("support_required", True)
    term_years = _term_years(text)
    if term_years is not None:
        parsed.setdefault("term_years", term_years)
        parsed.setdefault("warranty_months", term_years * 12)
    if _requires_rail_kit(text, global_requirements):
        parsed.setdefault("rail_kit_required", True)
    redundant_psu_pattern = (
        r"\b(?:redundant|1\+1|dual)\s+(?:psu|power\s+supply)\b|"
        r"резерв\w+\s+бп"
    )
    if re.search(redundant_psu_pattern, text, re.I):
        parsed.setdefault("redundant_psu", True)
    return parsed


def _storage_role_parsed_requirements(
    role: str,
    common: Mapping[str, Any],
    text: str,
) -> dict[str, Any]:
    parsed = {"required": True}
    system_count = _as_int(common.get("system_count") or common.get("count")) or 1
    if role == "storage_system":
        parsed.update(common)
        parsed.setdefault("count", system_count)
        return parsed
    if role in {"controller", "controller_module"}:
        for key in ("controller_count", "controller_redundancy"):
            if common.get(key) not in (None, ""):
                parsed[key] = common[key]
        parsed["count"] = _as_int(common.get("controller_count")) or system_count
        return parsed
    if role == "disk_shelf":
        parsed["count"] = _as_int(common.get("shelf_count")) or system_count
        if common.get("shelf_required"):
            parsed["shelf_required"] = True
        return parsed
    if role in {"drive", "ssd", "hdd"}:
        for key in (
            "drive_count",
            "drive_capacity_tb",
            "drive_type",
            "drive_interface",
            "acceptable_drive_types",
            "raw_capacity_tb",
            "usable_capacity_tb",
            "redundancy_level",
        ):
            if common.get(key) not in (None, ""):
                parsed[key] = common[key]
        count = _as_int(common.get("drive_count"))
        if count is not None:
            parsed["count"] = count
        if role == "ssd":
            parsed.setdefault("drive_type", "SSD")
        elif role == "hdd":
            parsed.setdefault("drive_type", "HDD")
        parsed.setdefault("drive_interface", "unknown")
        return parsed
    if role in {"host_port", "protocol_module"}:
        for key in (
            "host_protocol",
            "host_port_count",
            "host_port_speed",
            "host_port_media",
            "nvme_of_required",
            "fc_required",
            "iscsi_required",
            "sas_required",
        ):
            if common.get(key) not in (None, ""):
                parsed[key] = common[key]
        count = _as_int(common.get("host_port_count"))
        if count is not None:
            parsed["count"] = count
        return parsed
    if role in {"license", "support"}:
        parsed["count"] = _explicit_role_quantity(text, role) or system_count
        if role == "license":
            parsed["license_required"] = True
        if role == "support":
            parsed["support_required"] = True
        if common.get("term_years") is not None:
            parsed["term_years"] = common["term_years"]
        if role == "support" and common.get("warranty_months") is not None:
            parsed["warranty_months"] = common["warranty_months"]
        if role == "license" and common.get("included"):
            parsed["included"] = True
        return parsed
    if role in {"transceiver", "cable"}:
        quantity = _explicit_role_quantity(text, role)
        if quantity is not None:
            parsed["count"] = quantity
        for key in ("host_protocol", "host_port_speed", "host_port_media"):
            if common.get(key) not in (None, ""):
                parsed[key] = common[key]
        return parsed
    if role == "power_supply":
        parsed["count"] = _explicit_role_quantity(text, role) or system_count
        if common.get("redundant_psu"):
            parsed["redundant_psu"] = True
        return parsed
    if role == "rail_kit":
        parsed["count"] = _explicit_role_quantity(text, role) or system_count
        parsed["rail_kit_required"] = True
        return parsed
    return parsed


def _storage_product_capability_id(role: str, parsed: Mapping[str, Any]) -> str:
    parts = [role]
    for key in (
        "usable_capacity_tb",
        "raw_capacity_tb",
        "redundancy_level",
        "controller_count",
        "drive_count",
        "drive_capacity_tb",
        "drive_type",
        "acceptable_drive_types",
        "drive_interface",
        "host_protocol",
        "host_port_speed",
        "host_port_media",
        "term_years",
    ):
        value = parsed.get(key)
        if value not in (None, "", UNKNOWN_FACT):
            parts.append(str(value).strip().lower().replace(" ", "").replace("/", "_"))
    if len(parts) == 1:
        parts.append("requested")
    return ".".join(parts)


def _storage_role_requirement_text(text: str, role: str) -> str:
    if not text:
        return role
    marker = {
        "storage_system": "СХД",
        "controller": "контроллер",
        "controller_module": "модуль",
        "disk_shelf": "полк",
        "drive": "диск",
        "ssd": "SSD",
        "hdd": "HDD",
        "cache": "cache",
        "host_port": "FC",
        "protocol_module": "NVMe",
        "transceiver": "трансивер",
        "cable": "кабель",
        "license": "лицензи",
        "support": "поддержк",
        "power_supply": "бп",
        "rail_kit": "рельс",
    }.get(role, role)
    return _capability_text(text, marker)


def _storage_system_count(text: str) -> int | None:
    patterns = (
        r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:storage\s+(?:systems?|arrays?)|san|nas)\b",
        r"\b(\d{1,3})\s*(?:шт\.?|штук)?\s*(?:схд|систем[ауы]\s+хранени|дисков\w+\s+массив)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 999:
                return value
    return 1 if _looks_like_storage_product_group(text) else None


def _storage_controller_count(text: str) -> int | None:
    pattern = (
        r"\b(\d{1,2})\s*(?:x|шт\.?)?\s*контроллер|"
        r"\b(\d{1,2})\s*controllers?\b"
    )
    match = re.search(pattern, text, re.I)
    if not match:
        return None
    value = int(match.group(1) or match.group(2))
    return value if 1 <= value <= 16 else None


def _storage_shelf_count(text: str) -> int | None:
    match = re.search(
        r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:полк[аи]|(?:disk|drive|expansion)\s+shelves?)",
        text,
        re.I,
    )
    if not match:
        return None
    value = int(match.group(1))
    return value if 1 <= value <= 999 else None


def _storage_drive_count(text: str) -> int | None:
    match = re.search(
        r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:ssd|hdd|drive|disk|диск\w*|накопител\w*)",
        text,
        re.I,
    )
    if not match:
        return None
    value = int(match.group(1))
    return value if 1 <= value <= 999 else None


def _storage_drive_capacity_tb(text: str) -> float | None:
    patterns = (
        r"(?:ssd|hdd|drive|disk|диск\w*|накопител\w*)[^\n,;]{0,40}?(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)",
        r"(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)[^\n,;]{0,40}?(?:ssd|hdd|drive|disk|диск\w*|накопител\w*)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def _storage_capacity_tb(text: str, *, kind: str) -> float | None:
    if kind == "usable":
        patterns = (
            r"(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)[^\n,;]{0,60}?(?:usable|полезн\w+\s+емкост)",
            r"(?:usable|полезн\w+\s+емкост)[^\n,;]{0,60}?(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)",
            r"схд\s+на\s+(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)[^\n,;]{0,60}?полезн",
        )
    else:
        patterns = (
            r"\braw[^\n,;]{0,30}?(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)",
            r"(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)[^\n,;]{0,60}?\braw\b",
            r"(?:сыр\w+\s+емкост)[^\n,;]{0,60}?(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)",
        )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def _storage_drive_type(text: str) -> str | None:
    if _storage_drive_type_alternatives(text):
        return None
    if re.search(r"\bssd\b|ссд", text, re.I):
        return "SSD"
    if re.search(r"\bhdd\b|жестк\w+\s+диск|nl-?sas", text, re.I):
        return "HDD"
    return None


def _storage_drive_type_alternatives(text: str) -> list[str]:
    if (
        re.search(r"\bssd\b", text, re.I)
        and re.search(r"\bhdd\b", text, re.I)
        and re.search(
            r"\bssd\s*(?:/|or|или)\s*hdd\b|\bhdd\s*(?:/|or|или)\s*ssd\b|"
            r"(?:ssd|hdd)[^\n,;]{0,30}\b(?:можно|допустимо|подойдет|подойдут)\b|"
            r"\b(?:можно|допустимо|подойдет|подойдут)\b[^\n,;]{0,30}(?:ssd|hdd)",
            text,
            re.I,
        )
    ):
        return ["SSD", "HDD"]
    return []


def _storage_drive_interface(text: str) -> str | None:
    if re.search(r"\bnvme\b|\bu\.?2\b|\bu\.?3\b", text, re.I):
        return "NVMe"
    if re.search(r"\bsas\b", text, re.I):
        return "SAS"
    if re.search(r"\bsata\b", text, re.I):
        return "SATA"
    return None


def _storage_host_protocol(text: str) -> str | None:
    if re.search(r"\bnvme-?of\b", text, re.I):
        return "NVMe-oF"
    if re.search(r"\bfc\s*\d{1,3}\s*g|\bfibre\s+channel\b", text, re.I):
        return "FC"
    if re.search(r"\biscsi\b", text, re.I):
        return "iSCSI"
    if re.search(r"\bsas\s+host\b|\bhost\s+sas\b", text, re.I):
        return "SAS"
    return None


def _storage_host_port_speed(text: str) -> str | None:
    for pattern in (
        r"\bFC\s*(\d{1,3})\s*G\b",
        r"\biSCSI\s*(\d{1,3}(?:\s*/\s*\d{1,3})?)\s*G\b",
        r"\b(\d{1,3})\s*G\s*(?:FC|iSCSI)\b",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            value = re.sub(r"\s+", "", match.group(1).upper())
            return f"{value}G"
    return None


def _storage_host_port_count(text: str) -> int | None:
    match = re.search(
        r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:host\s+ports?|fc\s+ports?|iscsi\s+ports?|порт\w+\s+хост)",
        text,
        re.I,
    )
    if not match:
        return None
    value = int(match.group(1))
    return value if 1 <= value <= 512 else None


def _storage_host_port_media(text: str) -> str | None:
    if re.search(r"\bSFP\s*28\b|\bSFP28\b", text, re.I):
        return "SFP28"
    if re.search(r"\bSFP\s*\+|\bSFP\+\b", text, re.I):
        return "SFP+"
    if re.search(r"\bQSFP\s*28\b|\bQSFP28\b", text, re.I):
        return "QSFP28"
    if re.search(r"\bRJ\s*-?45\b|\bBASE\s*-?T\b", text, re.I):
        return "RJ45"
    return None


def _storage_redundancy_level(text: str) -> str | None:
    match = re.search(r"\bRAID\s*([0-9]+(?:\+[0-9]+)?)\b|рейд\s*([0-9]+)", text, re.I)
    if match:
        return f"RAID {match.group(1) or match.group(2)}"
    if re.search(r"erasure\s+coding|ec\b|кодировани\w+\s+стирани", text, re.I):
        return "erasure_coding"
    return None


def _network_product_capabilities(
    text: str,
    *,
    items: Sequence[Any],
    global_requirements: Mapping[str, Any],
) -> list[dict[str, Any]]:
    roles = _network_roles_from_sources(text, items, global_requirements)
    if not roles and _looks_like_network_product_group(text):
        roles = ["switch"]

    device_requirements = _network_device_parsed_requirements(text, global_requirements)
    capabilities: list[dict[str, Any]] = []
    for role in roles:
        parsed = _network_role_parsed_requirements(role, device_requirements, text)
        if not parsed.get("required", True):
            continue
        capabilities.append(
            _capability(
                role=role,
                capability_id=_network_product_capability_id(role, parsed),
                requirement_text=_network_role_requirement_text(text, role),
                parsed_requirements=parsed,
            )
        )
    return capabilities


def _network_optional_capabilities(text: str) -> list[dict[str, Any]]:
    optional: list[dict[str, Any]] = []
    if _poe_optional_requested(text):
        standard = _poe_optional_standard(text) or "PoE"
        optional.append(
            _capability(
                role="switch",
                capability_id=f"switch.{_poe_standard_id(standard)}.optional",
                requirement_text=_capability_text(text, "PoE"),
                parsed_requirements={
                    "poe_required": True,
                    "poe_standard": standard,
                    "optional": True,
                },
                hard=False,
            )
        )
    if _stacking_optional_requested(text):
        optional.append(
            _capability(
                role="switch",
                capability_id="switch.stacking.optional",
                requirement_text=_capability_text(text, "stacking"),
                parsed_requirements={
                    "stacking_required": True,
                    "optional": True,
                },
                hard=False,
            )
        )
    if _optional_10gbe_requested(text):
        optional.append(
            _capability(
                role="switch",
                capability_id="switch.10gbe.optional",
                requirement_text=_capability_text(text, "10GbE"),
                parsed_requirements={
                    "uplink_speed": "10GbE",
                    "network_speed": "10GbE",
                    "optional": True,
                },
                hard=False,
            )
        )
    return optional


def _network_roles_from_sources(
    text: str,
    items: Sequence[Any],
    global_requirements: Mapping[str, Any],
) -> list[str]:
    roles: list[str] = []
    for item in items:
        item_type = str(_item_value(item, "item_type") or "").strip().casefold()
        category = str(_item_value(item, "category") or "").strip().casefold()
        name = str(_item_value(item, "name") or "").strip()
        for value in (item_type, category, name):
            role = _network_role_from_text(value)
            if role and role not in roles:
                roles.append(role)
        requested_role = str(_item_requirements(item).get("role") or "").strip()
        if requested_role in _NETWORK_ROLE_IDS and requested_role not in roles:
            roles.append(requested_role)

    for key in ("role", "product_role", "product_type"):
        requested_role = str(global_requirements.get(key) or "").strip()
        if requested_role in _NETWORK_ROLE_IDS and requested_role not in roles:
            roles.append(requested_role)

    for role in _NETWORK_ROLE_IDS:
        if _network_role_requested(text, role) and role not in roles:
            roles.append(role)
    return roles


_NETWORK_ROLE_IDS = (
    "switch",
    "router",
    "firewall",
    "access_point",
    "transceiver",
    "dac_cable",
    "cable",
    "license",
    "support",
    "power_supply",
    "stacking_module",
    "other_accessory",
)


def _network_role_from_text(value: str) -> str | None:
    for role in _NETWORK_ROLE_IDS:
        if _network_role_requested(value, role):
            return role
    if value.strip().casefold() == "network":
        return "switch"
    return None


def _network_role_requested(text: str, role: str) -> bool:
    patterns = {
        "switch": r"\b(?:switch|ethernet\s+switch)\b|коммутатор|свитч",
        "router": r"\brouter\b|маршрутизатор|роутер",
        "firewall": r"\b(?:firewall|ngfw|utm)\b|межсетев|фаервол",
        "access_point": r"\b(?:access\s*point|wi-?fi\s+ap)\b|точк[аи]\s+доступа",
        "transceiver": (
            r"\b(?:transceiver|optic(?:al)?\s+module|sfp\s+module|qsfp\s+module)\b|"
            r"трансивер"
        ),
        "dac_cable": r"\b(?:dac|direct\s+attach)\b|dac-?кабел",
        "cable": r"\b(?:aoc|patch\s*cord|cable)\b|кабел|патч-?корд",
        "license": r"\b(?:license|licence|subscription)\b|лицензи|подписк",
        "support": r"\b(?:support|warranty|service)\b|поддержк|гаранти|сервис",
        "power_supply": (
            r"\b(?:spare|extra|separate|replacement)\s+(?:psu|power\s+supply)\b|"
            r"запас\w*\s+(?:бп|блок\w+\s+питани)|отдельн\w+\s+блок\w+\s+питани"
        ),
        "stacking_module": r"\b(?:stacking|stack)\s+module\b|модул\w+\s+стек",
        "other_accessory": r"\baccessor(?:y|ies)\b|аксессуар|опци[яи]",
    }
    return bool(re.search(patterns[role], text, re.I))


def _network_device_parsed_requirements(
    text: str,
    global_requirements: Mapping[str, Any],
) -> dict[str, Any]:
    network = _mapping(global_requirements.get("network"))
    parsed: dict[str, Any] = {"required": True}
    parsed.update({key: value for key, value in network.items() if value not in (None, "")})

    device_count = _network_device_count(text)
    if device_count is not None:
        parsed.setdefault("count", device_count)
        parsed.setdefault("device_count", device_count)

    port_count, port_segment = _network_access_port_segment(text)
    if port_count is not None:
        parsed.setdefault("port_count", port_count)
        speed = _network_speed_from_text(port_segment)
        if speed:
            parsed.setdefault("port_speed", speed)
        media = _network_media_from_text(port_segment)
        if media:
            parsed.setdefault("port_media", media)

    uplink_count, uplink_segment = _network_uplink_segment(text)
    if uplink_count is not None:
        parsed.setdefault("uplink_count", uplink_count)
        speed = _network_speed_from_text(uplink_segment)
        if speed:
            parsed.setdefault("uplink_speed", speed)
        media = _network_media_from_text(uplink_segment)
        if media:
            parsed.setdefault("uplink_media", media)

    if _poe_required(text):
        parsed.setdefault("poe_required", True)
        standard = _poe_standard(text)
        if standard:
            parsed.setdefault("poe_standard", standard)
    poe_budget = _poe_budget_w(text)
    if poe_budget is not None:
        parsed.setdefault("poe_required", True)
        parsed.setdefault("poe_budget_w", poe_budget)
    if re.search(r"\bL2\b|layer\s*2|уровень\s*2", text, re.I):
        parsed.setdefault("l2_required", True)
    if re.search(r"\bL3\b|layer\s*3|уровень\s*3", text, re.I):
        parsed.setdefault("l3_required", True)
    if _stacking_required(text):
        parsed.setdefault("stacking_required", True)
    airflow = _network_airflow(text)
    if airflow:
        parsed.setdefault("airflow", airflow)
    if re.search(
        r"\b(?:redundant|1\+1|dual)\s+(?:psu|power\s+supply)\b|резерв\w+\s+бп",
        text,
        re.I,
    ):
        parsed.setdefault("redundant_psu", True)
    term_years = _term_years(text)
    if term_years is not None:
        parsed.setdefault("term_years", term_years)
    if re.search(r"в\s+комплект|included|in\s+bundle", text, re.I):
        parsed.setdefault("included", True)
    return parsed


def _network_role_parsed_requirements(
    role: str,
    device_requirements: Mapping[str, Any],
    text: str,
) -> dict[str, Any]:
    if role in {"switch", "router", "firewall", "access_point"}:
        parsed = dict(device_requirements)
        parsed["required"] = True
        device_count = _as_int(parsed.get("device_count") or parsed.get("count")) or 1
        parsed.setdefault("count", device_count)
        return parsed
    parsed = {"required": True}
    device_count = (
        _as_int(device_requirements.get("device_count") or device_requirements.get("count"))
        or 1
    )
    if role == "transceiver":
        quantity = (
            _explicit_role_quantity(text, role)
            or _as_int(device_requirements.get("transceiver_count"))
            or (
                _as_int(device_requirements.get("uplink_count"))
                if device_requirements.get("included")
                or re.search(r"трансивер\w+\s+в\s+комплект", text, re.I)
                else None
            )
        )
        if quantity is not None:
            parsed["count"] = quantity
        speed = device_requirements.get("uplink_speed") or device_requirements.get("port_speed")
        if speed:
            parsed.setdefault("port_speed", speed)
        form_factor = device_requirements.get("uplink_media") or device_requirements.get(
            "port_media"
        )
        if form_factor:
            parsed.setdefault("transceiver_form_factor", form_factor)
        return parsed
    if role in {"dac_cable", "cable"}:
        quantity = _explicit_role_quantity(text, role)
        if quantity is None and re.search(
            r"one\s+per\s+(?:uplink|port)|по\s+одн\w+\s+на\s+(?:аплинк|порт)",
            text,
            re.I,
        ):
            quantity = _as_int(
                device_requirements.get("uplink_count") or device_requirements.get("port_count")
            )
        if quantity is not None:
            parsed["count"] = quantity
        speed = device_requirements.get("uplink_speed") or device_requirements.get("port_speed")
        media = device_requirements.get("uplink_media") or device_requirements.get("port_media")
        if speed:
            parsed.setdefault("port_speed", speed)
        if media:
            parsed.setdefault("port_media", media)
        parsed.setdefault("cable_type", "DAC" if role == "dac_cable" else "cable")
        return parsed
    if role in {"license", "support"}:
        parsed["count"] = _explicit_role_quantity(text, role) or device_count
        term_years = _term_years(text)
        if term_years is not None:
            parsed["term_years"] = term_years
        return parsed
    if role in {"power_supply", "stacking_module"}:
        parsed["count"] = _explicit_role_quantity(text, role) or device_count
        return parsed
    return parsed


def _network_product_capability_id(role: str, parsed: Mapping[str, Any]) -> str:
    parts = [role]
    for key in (
        "port_count",
        "port_speed",
        "port_media",
        "uplink_count",
        "uplink_speed",
        "uplink_media",
    ):
        value = parsed.get(key)
        if value not in (None, "", UNKNOWN_FACT):
            parts.append(str(value).strip().lower().replace(" ", ""))
    if parsed.get("poe_required"):
        parts.append("poe")
    if parsed.get("l3_required"):
        parts.append("l3")
    if parsed.get("stacking_required"):
        parts.append("stacking")
    if len(parts) == 1:
        parts.append("requested")
    return ".".join(parts)


def _network_role_requirement_text(text: str, role: str) -> str:
    if not text:
        return role
    if role in {"switch", "router", "firewall", "access_point"}:
        device_text = _network_device_requirement_text(text)
        if device_text:
            return device_text
    marker = {
        "switch": "коммутатор",
        "router": "router",
        "firewall": "firewall",
        "access_point": "точка",
        "transceiver": "трансивер",
        "dac_cable": "DAC",
        "cable": "кабель",
        "license": "лицензи",
        "support": "support",
        "power_supply": "блок",
        "stacking_module": "stack",
    }.get(role, role)
    return _capability_text(text, marker)


def _network_device_requirement_text(text: str) -> str:
    technical_segments: list[str] = []
    technical_pattern = (
        r"\b(?:switch|router|firewall|ngfw|utm|access\s*point|wi-?fi|"
        r"ports?|uplinks?|PoE(?:\+\+|\+)?|Power\s+over\s+Ethernet|"
        r"802\.3(?:af|at|bt)|SFP\+?|SFP28|QSFP\+?|QSFP28|L[23]|"
        r"stacking|stackable|stack)\b|"
        r"коммутатор|свитч|маршрутизатор|роутер|межсетев|фаервол|"
        r"точк[аи]\s+доступа|порт|аплинк|стек|стекир"
    )
    non_technical_pattern = (
        r"склад|москв|warehouse|moscow|сам(?:ый|ая|ое)\s+дешев|"
        r"дешев|вариант\s+для\s+кп|\bкп\b|cheapest|lowest\s+cost|quote"
    )
    for segment in _split_requirement_segments(text):
        if not re.search(technical_pattern, segment, re.I):
            continue
        if re.search(non_technical_pattern, segment, re.I):
            continue
        technical_segments.append(segment)
    if not technical_segments:
        return ""
    return " ".join(" ".join(technical_segments).split())[:180]


def _network_device_count(text: str) -> int | None:
    patterns = (
        r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:switch(?:es)?|router(?:s)?|firewall(?:s)?|access\s*point(?:s)?)\b",
        r"\b(\d{1,3})\s*(?:шт\.?|штук)?\s*(?:коммутатор\w*|маршрутизатор\w*|роутер\w*|точк[аи]\s+доступа)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 999:
                return value
    return 1 if _looks_like_network_product_group(text) else None


_NETWORK_PORT_MULTIPLIER_RE = r"[xх×*]"
_NETWORK_BASE_RATE_RE = r"(?:1000|100|(?:1|2\.5|5|10|25|40|100)\s*g?)"
_NETWORK_G_RATE_RE = r"(?:1|2\.5|5|10|25|40|100)"
_NETWORK_UPLINK_MEDIA_RE = r"(?:sfp\+?|sfp28|qsfp\+?|qsfp28)"
_NETWORK_SPEED_PREFIX_RE = r"(?:\b|(?<=[xх×*]))"


def _network_access_port_segment(text: str) -> tuple[int | None, str]:
    patterns = (
        rf"(\d{{1,3}})\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*"
        rf"{_NETWORK_BASE_RATE_RE}\s*base\s*-?\s*t[x]?\b[^\n,;]{{0,50}}",
        rf"(\d{{1,3}})\s*(?:{_NETWORK_PORT_MULTIPLIER_RE})?\s*"
        r"(?:порт\w*|ports?)(?!\s*(?:uplink|аплинк))[^\n,;]{0,80}",
        rf"(\d{{1,3}})\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*"
        rf"{_NETWORK_G_RATE_RE}\s*g(?:b(?:it)?(?:e|/s|ps)?)?"
        r"(?!\s*(?:base\s*-?\s*x|sfp|qsfp|uplink|аплинк))[^\n,;]{0,50}",
        rf"(\d{{1,3}})\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*"
        r"(?:gigabit|гигабит)[^\n,;]{0,50}",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 512:
                return value, match.group(0)
    return None, ""


def _network_uplink_segment(text: str) -> tuple[int | None, str]:
    patterns = (
        rf"(\d{{1,3}})\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*"
        rf"(?:10|25|40|100)\s*g\s*base\s*-?\s*x\s*"
        rf"(?:{_NETWORK_UPLINK_MEDIA_RE})?[^\n,;]{{0,50}}",
        rf"(\d{{1,3}})\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*"
        rf"(?:10|25|40|100)\s*g\s*{_NETWORK_UPLINK_MEDIA_RE}[^\n,;]{{0,50}}",
        rf"(\d{{1,3}})(?:\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*|\s+)"
        rf"{_NETWORK_UPLINK_MEDIA_RE}\b[^\n,;]{{0,50}}",
        rf"(\d{{1,3}})\s*(?:{_NETWORK_PORT_MULTIPLIER_RE})?\s*"
        r"(?:uplinks?|аплинк\w*)[^\n,;]{0,80}",
        r"(?:uplinks?|аплинк\w*)[^\n,;]{0,20}(\d{1,3})[^\n,;]{0,80}",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 128:
                return value, match.group(0)
    return None, ""


def _network_speed_from_text(text: str) -> str | None:
    if re.search(
        rf"{_NETWORK_SPEED_PREFIX_RE}100\s*base\s*-?\s*t[x]?\b"
        r"|\bfast\s+ethernet\b",
        text,
        re.I,
    ):
        return "100MbE"
    if re.search(
        rf"{_NETWORK_SPEED_PREFIX_RE}1000\s*base\s*-?\s*t[x]?\b"
        rf"|{_NETWORK_SPEED_PREFIX_RE}(?:gigabit|гигабит)",
        text,
        re.I,
    ):
        return "1GbE"
    match = re.search(
        rf"{_NETWORK_SPEED_PREFIX_RE}"
        r"(1|2\.5|5|10|25|40|100|200|400)\s*g(?:b(?:it)?(?:e|/s|ps)?)?\b",
        text,
        re.I,
    )
    if not match:
        return None
    value = match.group(1)
    return f"{value}GbE"


def _network_media_from_text(text: str) -> str | None:
    if re.search(r"\bSFP\s*28\b|\bSFP28\b", text, re.I):
        return "SFP28"
    if re.search(r"\bSFP\s*\+(?=\W|$)", text, re.I):
        return "SFP+"
    if re.search(r"\bQSFP\s*28\b|\bQSFP28\b", text, re.I):
        return "QSFP28"
    if re.search(r"\bQSFP\s*\+(?=\W|$)", text, re.I):
        return "QSFP+"
    if re.search(r"\bRJ\s*-?45\b|BASE\s*-?T(?:X)?\b|PoE", text, re.I):
        return "RJ45"
    return None


_OPTIONAL_REQUIREMENT_MARKER_RE = (
    r"желательно|по\s+возможности|если\s+есть|лучше(?:\s+с)?|"
    r"можно(?:\s+с)?|optional|prefer(?:red|ably)?|nice\s+to\s+have|"
    r"if\s+available"
)
_POE_REQUIREMENT_RE = (
    r"\bPoE(?:\+\+|\+)?(?=\W|$)|\bPower\s+over\s+Ethernet\b|802\.3(?:af|at|bt)"
)
_STACKING_REQUIREMENT_RE = r"\b(?:stacking|stackable|stack)\b|стек|стекир"


def _poe_required(text: str) -> bool:
    return _hard_requirement_requested(text, _POE_REQUIREMENT_RE)


def _poe_standard(text: str) -> str | None:
    return _poe_standard_from_text(
        " ".join(_requirement_segments(text, _POE_REQUIREMENT_RE, optional=False))
    )


def _poe_optional_requested(text: str) -> bool:
    return _optional_requirement_requested(text, _POE_REQUIREMENT_RE)


def _poe_optional_standard(text: str) -> str | None:
    return _poe_standard_from_text(
        " ".join(_requirement_segments(text, _POE_REQUIREMENT_RE, optional=True))
    )


def _poe_standard_from_text(text: str) -> str | None:
    if re.search(r"\bPoE\+\+(?=\W|$)|802\.3bt", text, re.I):
        return "PoE++"
    if re.search(r"\bPoE\+(?=\W|$)|802\.3at", text, re.I):
        return "PoE+"
    if re.search(r"\bPoE\b|\bPower\s+over\s+Ethernet\b|802\.3af", text, re.I):
        return "PoE"
    return None


def _poe_standard_id(standard: str) -> str:
    normalized = standard.strip().casefold()
    if normalized == "poe++":
        return "poeplusplus"
    if normalized == "poe+":
        return "poeplus"
    return "poe"


def _stacking_required(text: str) -> bool:
    return _hard_requirement_requested(text, _STACKING_REQUIREMENT_RE)


def _stacking_optional_requested(text: str) -> bool:
    return _optional_requirement_requested(text, _STACKING_REQUIREMENT_RE)


def _hard_requirement_requested(text: str, requirement_pattern: str) -> bool:
    return bool(_requirement_segments(text, requirement_pattern, optional=False))


def _optional_requirement_requested(text: str, requirement_pattern: str) -> bool:
    return bool(_requirement_segments(text, requirement_pattern, optional=True))


def _requirement_segments(
    text: str,
    requirement_pattern: str,
    *,
    optional: bool,
) -> list[str]:
    segments: list[str] = []
    for segment in _split_requirement_segments(text):
        has_requirement = re.search(requirement_pattern, segment, re.I)
        if not has_requirement:
            continue
        has_optional_marker = re.search(
            _OPTIONAL_REQUIREMENT_MARKER_RE,
            segment,
            re.I,
        )
        if optional == bool(has_optional_marker):
            segments.append(segment)
    return segments


def _split_requirement_segments(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"[,;\n]+", text) if segment.strip()]


def _poe_budget_w(text: str) -> int | None:
    match = re.search(r"(?:poe[^\n,;]{0,40})?(\d{2,5})\s*w(?:att|atts)?\b", text, re.I)
    if not match:
        match = re.search(r"(\d{2,5})\s*Вт", text, re.I)
    if match:
        value = int(match.group(1))
        if 1 <= value <= 100000:
            return value
    return None


def _network_airflow(text: str) -> str | None:
    lowered = text.casefold()
    if re.search(r"front\s*[- ]?to\s*[- ]?back|порт\w*\s+назад", lowered):
        return "front-to-back"
    if re.search(r"back\s*[- ]?to\s*[- ]?front|порт\w*\s+вперед", lowered):
        return "back-to-front"
    if "port-side intake" in lowered:
        return "port-side-intake"
    if "port-side exhaust" in lowered:
        return "port-side-exhaust"
    return None


def _term_years(text: str) -> int | None:
    match = re.search(r"\b([1-9])\s*(?:year|years|yr|года?|лет)\b", text, re.I)
    if match:
        return int(match.group(1))
    return None


def _optional_10gbe_requested(text: str) -> bool:
    optional_marker = (
        r"желательно|по\s+возможности|если\s+есть|лучше\s+с|можно\s+с|"
        r"optional|prefer(?:red|ably)?|nice\s+to\s+have|if\s+available"
    )
    speed_marker = r"(?:10\s*g(?:be|bit|bps)?|10gbe|10g)"
    return bool(
        re.search(
            rf"{speed_marker}[^\n,;]{{0,40}}(?:{optional_marker})|"
            rf"(?:{optional_marker})[^\n,;]{{0,40}}{speed_marker}",
            text,
            re.I,
        )
    )


def _explicit_role_quantity(text: str, role: str) -> int | None:
    patterns = {
        "transceiver": r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:transceiver|трансивер)",
        "dac_cable": r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:dac|direct\s+attach)",
        "cable": r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:cables?|кабел)",
        "license": r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:license|licence|лицензи)",
        "support": r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:support|поддержк|гаранти)",
        "power_supply": r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:psu|power\s+supply|бп|блок\w+\s+питани)",
        "stacking_module": r"\b(\d{1,3})\s*(?:x|шт\.?)?\s*(?:stacking|stack|модул\w+\s+стек)",
    }
    pattern = patterns.get(role)
    if not pattern:
        return None
    match = re.search(pattern, text, re.I)
    if match:
        value = int(match.group(1))
        if 1 <= value <= 999:
            return value
    return None


def _merge_network_requirement(
    current: dict[str, Any] | None,
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    if current is None:
        return dict(incoming)
    merged = dict(current)
    current_ports = _int_value(merged.get("min_ports_per_server")) or 0
    incoming_ports = _int_value(incoming.get("min_ports_per_server")) or 0
    if incoming_ports > current_ports:
        merged["min_ports_per_server"] = incoming_ports
    for key in ("speed", "media", "interface"):
        value = incoming.get(key)
        if value and value != "unknown":
            merged[key] = value
    merged["required"] = True
    return merged


def _network_capability_id(requirement: Mapping[str, Any]) -> str:
    parts = ["network_adapter"]
    speed = str(requirement.get("speed") or "").strip().lower()
    media = str(requirement.get("media") or "").strip().lower()
    if speed and speed != "unknown":
        parts.append(speed)
    if media and media != "unknown":
        parts.append(media)
    if len(parts) == 1:
        parts.append("advanced")
    return ".".join(parts)


def _capability_text(text: str, marker: str) -> str:
    if not text:
        return marker
    pattern = re.escape(marker).replace("_", r"[-_\s]?")
    match = re.search(rf"[^.;\n]{{0,60}}{pattern}[^.;\n]{{0,80}}", text, re.I)
    return " ".join((match.group(0) if match else text[:160]).split())


def _unique_capabilities(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        key = (str(value.get("capability_id")), str(value.get("role")))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _unique_capabilities_prefer_later(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in reversed(values):
        key = (str(value.get("capability_id")), str(value.get("role")))
        if key in seen:
            continue
        seen.add(key)
        result.insert(0, dict(value))
    return result


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _instruction_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            source_text = str(item.get("source_text") or item.get("text") or "").strip()
            parsed = item.get("parsed_requirements")
            rows.append(
                _instruction_row(
                    source_text,
                    dict(parsed) if isinstance(parsed, Mapping) else dict(item),
                )
            )
        else:
            text = str(item or "").strip()
            if text:
                rows.append(_instruction_row(text, {}))
    return rows


def _instruction_row(source_text: str, parsed_requirements: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "source_text": source_text,
        "parsed_requirements": dict(parsed_requirements),
    }
    if not source_text and not parsed_requirements:
        return {}
    return row


def _unique_requirements(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for value in values:
        key = (
            str(value.get("source_text") or ""),
            str(value.get("classification") or ""),
            str(value.get("role") or ""),
            str(value.get("capability_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _unique_instruction_rows(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _dedupe_mapping_lists(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, list):
            result[str(key)] = _unique([str(row) for row in item if str(row).strip()])
        else:
            result[str(key)] = item
    return result


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _join_unique_text_parts(*parts: str | None) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        text = " ".join(str(part or "").split())
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        if any(key in existing or existing in key for existing in seen):
            continue
        seen.add(key)
        result.append(text)
    return " ".join(result)
