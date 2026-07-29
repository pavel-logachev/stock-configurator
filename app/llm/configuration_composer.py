from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from math import ceil
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from app.core.config import (
    LlmSettings,
    WebEvidenceSettings,
    get_llm_settings,
    get_web_evidence_settings,
)
from app.evidence.web_evidence import (
    EvidencePack,
    EvidenceSearchCache,
    WebSearchProvider,
    build_evidence_tasks_for_proposals,
    build_relation_evidence_tasks_for_recommendations,
    collect_web_evidence,
    evidence_components_by_id,
    evidence_pack_has_found_sources,
    evidence_relations_by_recommendation_id,
    safe_evidence_diagnostics,
)
from app.llm.base import (
    LlmClient,
    LlmConfigurationError,
    LlmError,
    LlmHttpError,
    LlmInvalidJsonError,
    LlmReadTimeoutError,
)
from app.llm.composer_package_compactor import (
    COMPACT_FULL_MATRIX_MODE,
    VERBOSE_FULL_MATRIX_MODE,
    compact_composer_package,
    composer_package_compaction_diagnostics,
    json_size_chars,
)
from app.llm.openai_compatible import OpenAICompatibleLlmClient
from app.planning import role_lifecycle
from app.planning.network_facts import (
    network_adapter_facts_satisfy_requirement,
    network_facts_satisfy_requirement,
    required_network_adapter_quantity,
)
from app.planning.power_facts import platform_power_bundle_satisfies
from app.reports.commercial_summary import build_primary_commercial_summary
from app.reports.composer_result_normalizer import (
    COMPOSER_NO_SAFE_COMPLETE_BOM,
    COMPOSER_STRUCTURED_NO_RECOMMENDATION,
    normalize_composer_result,
)
from app.user_facing_text import (
    deduplicate_engineer_checks,
    human_engineering_confidence_label,
    sanitize_engineer_checks_for_product_group,
    sanitize_user_facing_text,
)

logger = logging.getLogger(__name__)


class LlmCallBudgetExceededError(LlmError):
    """Raised before a match would exceed its bounded LLM call budget."""


@dataclass
class LlmCallBudget:
    max_calls: int
    llm_call_count: int = 0
    llm_call_stages: list[str] = field(default_factory=list)
    llm_call_budget_exceeded: bool = False
    llm_call_budget_exceeded_stage: str | None = None

    def reserve(self, stage: str) -> None:
        stage_name = str(stage or "unknown").strip() or "unknown"
        if self.llm_call_count >= self.max_calls:
            self.llm_call_budget_exceeded = True
            self.llm_call_budget_exceeded_stage = stage_name
            raise LlmCallBudgetExceededError(
                f"LLM call budget exceeded before stage {stage_name}"
            )
        self.llm_call_count += 1
        self.llm_call_stages.append(stage_name)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "llm_call_count": self.llm_call_count,
            "llm_call_stages": list(self.llm_call_stages),
            "llm_call_budget_exceeded": bool(self.llm_call_budget_exceeded),
            "llm_call_budget_exceeded_stage": self.llm_call_budget_exceeded_stage,
            "max_llm_calls_per_match": self.max_calls,
        }


class BudgetedLlmClient:
    def __init__(self, client: LlmClient, budget: LlmCallBudget) -> None:
        self._client = client
        self._budget = budget

    @property
    def budget(self) -> LlmCallBudget:
        return self._budget

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self._budget.reserve(_infer_llm_call_stage(system_prompt, user_prompt))
        return self._client.generate_json(system_prompt, user_prompt)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def safe_diagnostics(self) -> dict[str, Any]:
        diagnostics_method = getattr(self._client, "safe_diagnostics", None)
        if callable(diagnostics_method):
            return _safe_mapping(diagnostics_method())
        return {}


def budgeted_llm_client(
    client: LlmClient | None,
    budget: LlmCallBudget | None,
) -> LlmClient | None:
    if client is None or budget is None:
        return client
    if isinstance(client, BudgetedLlmClient):
        return client if client.budget is budget else BudgetedLlmClient(client, budget)
    return BudgetedLlmClient(client, budget)


def llm_call_budget_diagnostics(
    budget: LlmCallBudget | None,
) -> dict[str, Any]:
    if budget is None:
        return {
            "llm_call_count": 0,
            "llm_call_stages": [],
            "llm_call_budget_exceeded": False,
            "llm_call_budget_exceeded_stage": None,
            "max_llm_calls_per_match": None,
        }
    return budget.diagnostics()


def _infer_llm_call_stage(system_prompt: str, user_prompt: str) -> str:
    payload: dict[str, Any] = {}
    try:
        loaded = json.loads(user_prompt)
        if isinstance(loaded, Mapping):
            payload = dict(loaded)
    except (TypeError, ValueError):
        payload = {}

    stage = str(payload.get("multi_pass_stage") or "").strip()
    if stage == "requirement_contract":
        return "requirement_contract"
    if stage == "role_evaluation":
        return "role_evaluation"
    if stage == "bom_composition":
        return "main_composer"
    if stage == "completeness_critic":
        return "completeness_critic"
    if stage == "repair":
        return "repair"
    if payload.get("empty_response_repair_attempt"):
        return "empty_response_repair"
    if payload.get("no_recommendation_coverage_repair_attempt"):
        return "no_recommendation_coverage_repair"
    if payload.get("repair_attempt"):
        return "post_validation_repair"
    if payload.get("evidence_pack"):
        return "evidence_review"
    if payload.get("category_catalog") is not None:
        return (
            "candidate_universe_planner_repair"
            if "repair pass" in system_prompt.casefold()
            else "candidate_universe_planner"
        )

    prompt = system_prompt.casefold()
    if "candidate universe planner" in prompt:
        return (
            "candidate_universe_planner_repair"
            if "repair pass" in prompt
            else "candidate_universe_planner"
        )
    if "requirement contract" in prompt:
        return "requirement_contract"
    if "role evaluation" in prompt:
        return "role_evaluation"
    if "completeness critic" in prompt:
        return "completeness_critic"
    if "repair pass" in prompt:
        return "repair"
    if "evidence review" in prompt:
        return "evidence_review"
    return "main_composer"


READY_SERVER_CANDIDATE_TYPE = "ready_server"
BUILD_CANDIDATE_TYPE = "build_from_parts"
PARTIAL_BUILD_CANDIDATE_TYPE = "partial_build"
SERVER_PLATFORM_ROLE = "server_platform"
CPU_ROLE = "cpu"
RAM_ROLE = "ram"
SSD_ROLE = "ssd"
HDD_ROLE = "hdd"
STORAGE_CONTROLLER_ROLE = "storage_controller"
NETWORK_ADAPTER_ROLE = "network_adapter"
GPU_ROLE = "gpu"
TRANSCEIVER_ROLE = "transceiver"
CABLE_ROLE = "cable"
POWER_SUPPLY_ROLE = "power_supply"
RAIL_KIT_ROLE = "rail_kit"
LICENSE_ROLE = "license"
SUPPORT_ROLE = "support"
OTHER_ACCESSORY_ROLE = "other_accessory"
UNMAPPED_ROLE = "unmapped"
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
FULFILLMENT_SEPARATE_COMPONENT_REQUIRED = "separate_component_required"
FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT = "included_in_primary_object"
FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT = "included_in_selected_component"
FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT = "included_in_bundle_or_kit"
FULFILLMENT_SERVICE_OR_SUPPORT = "service_or_support"
FULFILLMENT_LOGISTICS_CONSTRAINT = "logistics_constraint"
FULFILLMENT_ENGINEERING_CHECK_ONLY = "engineering_check_only"
FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION = "unverified_requires_confirmation"
FULFILLMENT_NOT_APPLICABLE = "not_applicable"
FULFILLMENT_INCLUDED_MODES = {
    FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
    FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
    FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
}
SWITCH_ROLE = "switch"
ROUTER_ROLE = "router"
FIREWALL_ROLE = "firewall"
ACCESS_POINT_ROLE = "access_point"
DAC_CABLE_ROLE = "dac_cable"
STACKING_MODULE_ROLE = "stacking_module"
NETWORK_PRODUCT_GROUP = "network"
SERVER_PRODUCT_GROUP = "server"
STORAGE_PRODUCT_GROUP = "storage"
SEMANTIC_COMPLEX_FALLBACK_REASON = "complex_request_requires_llm_semantic_planner"
SEMANTIC_PACKAGE_SKIPPED_PLANNING_UNAVAILABLE = "planning_unavailable_empty"
SEMANTIC_SOURCE_PLANNER_UNAVAILABLE = "planner_unavailable"
SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT = "fallback_after_llm_timeout"
SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON = "semantic_planner_timeout"
STORAGE_SYSTEM_ROLE = "storage_system"
STORAGE_ARRAY_CONTROLLER_ROLE = "controller"
CONTROLLER_MODULE_ROLE = "controller_module"
DISK_SHELF_ROLE = "disk_shelf"
DRIVE_ROLE = "drive"
CACHE_ROLE = "cache"
HOST_PORT_ROLE = "host_port"
PROTOCOL_MODULE_ROLE = "protocol_module"
UNKNOWN_FACT = "unknown"
OPTIMIZATION_MODE_COST_MINIMAL_FIT = "cost_minimal_fit"
OUTPUT_MODE_SINGLE_BEST_COST_VALID = "single_best_cost_valid"
OUTPUT_MODE_GROUPED_PRESALES = "grouped_presales"
OUTPUT_MODE_LEGACY_MULTI_OPTION = "legacy_multi_option"
SUPPORTED_OUTPUT_MODES = {
    OUTPUT_MODE_SINGLE_BEST_COST_VALID,
    OUTPUT_MODE_GROUPED_PRESALES,
    OUTPUT_MODE_LEGACY_MULTI_OPTION,
    "multi_option",
}
DISTILLER_OVER_BUDGET_SKIP_REASON = "package_over_budget_after_distillation"
DISTILLER_FAILED_SKIP_REASON = "matrix_distiller_failed"
DISTILLER_FALLBACK_PACKAGE_SOURCES = {
    "fallback_compact_package",
    "fallback_compact_package_after_distiller_error",
}
PACKAGE_OVER_BUDGET_FALLBACK_REASON = "llm_configurator_package_over_budget"
INCOMPLETE_MATRIX_COVERAGE_FALLBACK_REASON = (
    "llm_configurator_incomplete_matrix_coverage"
)
INCOMPLETE_MATRIX_EXPOSURE_REASON = "incomplete_matrix_exposure"
CATEGORY_PLAN_MISSING_REQUIRED_ROLES_REASON = "category_plan_missing_required_roles"
MISSING_REQUIRED_CATEGORY_BEFORE_COMPOSER_REASON = (
    "missing_required_category_before_composer"
)
HIGH_QUALITY_BROAD_PACKAGE_UNDER_LIMIT_REASON = (
    "broad_package_under_high_quality_limit"
)
SKIPPED_FULL_BROAD_PACKAGE_UNDER_HIGH_QUALITY_LIMIT_REASON = (
    "skipped_full_broad_package_under_high_quality_limit"
)
FULL_BROAD_MATRIX_EXPOSURE_MODE = "full_broad_matrix"
PROVIDER_CONTEXT_LIMIT_ERROR_TYPE = "context_limit"
PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON = (
    "provider_context_limit_fallback_to_full_matrix"
)
COMPACT_FULL_MATRIX_CONTEXT_LIMIT_FALLBACK_REASON = (
    "compact_full_matrix_context_limit"
)
COMPOSER_SCHEMA_VALIDATION_FAILED = "composer_schema_validation_failed"
COMPOSER_PROVIDER_TIMEOUT = "composer_provider_timeout"
COMPACT_FULL_MATRIX_AUTO_CHAR_THRESHOLD = 1_500_000
COMPACT_FULL_MATRIX_AUTO_CANDIDATE_THRESHOLD = 300
TIMEOUT_FALLBACK_REDUCED_PACKAGE_TYPE = "role_aware_reduced_package"
TIMEOUT_FALLBACK_REDUCED_PACKAGE_MODE = "timeout_fallback_role_aware_reduced_matrix"
DEFAULT_TIMEOUT_FALLBACK_MIN_REQUIRED_ROLE_CANDIDATES = 12
DEFAULT_TIMEOUT_FALLBACK_MAX_REQUIRED_ROLE_CANDIDATES = 48
DEFAULT_TIMEOUT_FALLBACK_MAX_OPTIONAL_ROLE_CANDIDATES = 12
DEFAULT_TIMEOUT_FALLBACK_MAX_TOTAL_CANDIDATES = 240
PACKAGE_OVER_BUDGET_BEFORE_COMPOSER_REASON = "package_over_budget_before_composer"
READY_CANDIDATES_EXCLUDED_AI_CATEGORY_PLAN_REASON = "ai_category_plan_matrix_path"
READY_SERVER_CANDIDATES_LIMIT = 3
DEFAULT_NO_RECOMMENDATION_FULL_COVERAGE_LIMIT = 12
DEFAULT_NO_RECOMMENDATION_MIN_LARGE_ROLE_CANDIDATES = 12
DEFAULT_NO_RECOMMENDATION_MIN_LARGE_ROLE_FRACTION = 0.5
NO_RECOMMENDATION_COVERAGE_REPAIR_INSTRUCTION = (
    "Your no_recommendation considered too few candidates. Re-evaluate using all "
    "provided role candidates and produce either a BOM or a no_recommendation with "
    "sufficient role coverage."
)
NO_RECOMMENDATION_COVERAGE_DIAGNOSTIC_KEYS = (
    "no_recommendation_coverage_gate_passed",
    "no_recommendation_coverage_repair_attempted",
    "no_recommendation_coverage_repair_success",
    "no_recommendation_coverage_rejected",
    "no_recommendation_coverage_thresholds",
    "no_recommendation_coverage_repair_reason",
)
MULTI_PASS_DIAGNOSTIC_KEYS = (
    "composer_mode",
    "requirement_contract",
    "requirement_contract_used",
    "requirement_contract_fallback_used",
    "requirement_contract_source",
    "requirement_contract_error_stage",
    "requirement_contract_error_type",
    "requirement_contract_validation_errors",
    "main_composer_used",
    "role_evaluation_used",
    "role_evaluation_skipped_reason",
    "role_evaluation_count_by_role",
    "role_evaluation_coverage_by_role",
    "role_evaluation_failed_chunks",
    "bom_composer_used",
    "completeness_critic_used",
    "completeness_critic_result",
    "code_completeness_result",
    "code_completeness_after_repair",
    "repair_composer_used",
    "final_bom_after_repair",
    "validation_repair_attempted",
    "validation_repair_used",
    "validation_repair_success",
    "validation_repair_returned_no_recommendation",
    "validation_repair_fallback_reason",
    "validation_repair_failure_reason",
    "validation_repair_empty_output",
    "validation_repair_schema_failure",
    "validation_repair_initial_validation_summary",
    "validation_repair_final_validation_summary",
    "validation_repair_rejected_candidate_ids",
    "validation_repair_concrete_errors",
    "validation_repair_forbidden_component_combinations",
    "validation_repair_rejected_debug_safe",
    "final_status_source",
    "llm_parse_stage",
    "llm_schema_validation_errors",
    "composer_failure_stage",
    "composer_failure_error_type",
    "parse_status",
    "multi_pass_pass_count",
    "multi_pass_chunk_size",
    "multi_pass_role_order",
    "llm_call_count",
    "llm_call_stages",
    "llm_call_budget_exceeded",
    "llm_call_budget_exceeded_stage",
    "max_llm_calls_per_match",
    "no_recommendation_reason",
    "composer_provider_timeout",
    "composer_timeout_fallback_attempted",
    "composer_timeout_fallback_type",
    "composer_timeout_fallback_success",
    "composer_timeout_fallback_reason",
    "composer_timeout_original_fallback_reason",
    "composer_timeout_original_package_mode",
    "composer_timeout_prior_fallback_type",
    "composer_timeout_prior_fallback_reason",
    "composer_timeout_retry_package_mode",
    "composer_timeout_original_context_chars",
    "composer_timeout_retry_context_chars",
    "composer_timeout_fallback_original_candidate_count_by_role",
    "composer_timeout_fallback_candidate_count_by_role",
    "composer_timeout_fallback_dropped_before_fallback_count_by_role",
    "composer_timeout_fallback_dropped_before_fallback_reasons",
    "composer_timeout_fallback_coverage_ratio_by_role",
    "original_candidate_count_by_role",
    "fallback_candidate_count_by_role",
    "dropped_before_fallback_count_by_role",
    "dropped_before_fallback_reasons",
    "timeout_fallback_coverage_ratio_by_role",
    "package_candidate_exposure_policy",
)
COMPACT_PACKAGE_DIAGNOSTIC_KEYS = (
    "v2_package_mode",
    "selected_package_mode",
    "verbose_context_chars",
    "compact_context_chars",
    "selected_context_chars",
    "verbose_context_size",
    "compact_context_size",
    "selected_context_size",
    "chars_by_section",
    "avg_chars_per_candidate_by_role",
    "removed_verbose_fields",
    "removed_verbose_field_counts",
    "verbose_candidate_count_by_role",
    "verbose_candidate_count_total",
    "compact_candidate_count_by_role",
    "compact_candidate_total",
    "compact_candidate_ids_by_role",
    "compact_candidate_ids_hash",
    "compact_package_full_matrix_used",
    "package_candidate_loss",
    "package_candidate_loss_details",
    "provider_context_limit_retry_compact_attempted",
    "provider_context_limit_retry_compact_success",
    "provider_context_limit_original_chars",
    "provider_context_limit_compact_chars",
    "provider_context_limit_after_compact",
)
FULL_MATRIX_FAILED_CHUNKS_KEY = "full_matrix_failed_chunks"
FIT_EXACT_OR_CLOSE = "exact_or_close_fit"
FIT_ACCEPTABLE_OVERFIT = "acceptable_overfit"
FIT_EXCESSIVE_OVERFIT = "excessive_overfit"
FIT_UNKNOWN = "unknown_fit"
FIT_TIER_STRONG = "strong_fit"
FIT_TIER_POSSIBLE = "possible_fit"
FIT_TIER_FALLBACK_UNKNOWN = "fallback_unknown"
FIT_TIER_EXPLICIT_MISMATCH = "explicit_mismatch"
FIT_TIER_WRONG_ROLE = "wrong_role"
SELECTABLE_FIT_TIERS = {FIT_TIER_STRONG, FIT_TIER_POSSIBLE, FIT_TIER_FALLBACK_UNKNOWN}
FIT_TIER_RANK = {
    FIT_TIER_STRONG: 0,
    FIT_TIER_POSSIBLE: 1,
    FIT_TIER_FALLBACK_UNKNOWN: 2,
    FIT_TIER_EXPLICIT_MISMATCH: 98,
    FIT_TIER_WRONG_ROLE: 99,
}
FINAL_SAFE_RECOMMENDATIONS_LIMIT = 3
VALIDATION_REJECTION_KEYS = (
    "rejected_fatal",
    "rejected_missing_required_role",
    "rejected_stock_shortage",
    "rejected_role_mismatch",
    "rejected_unknown_component",
    "rejected_platform_cpu_mismatch",
    "rejected_ram_capacity_unknown",
    "rejected_quantity_materialization_failed",
    "rejected_optional_core_conflict",
    "rejected_invalid_schema",
    "rejected_invalid_candidate_type",
    "rejected_invalid_price_or_currency",
    "rejected_right_size_rejected",
    "rejected_other",
)
SELECTION_SKIP_KEYS = (
    "selection_skipped_duplicate",
    "selection_skipped_dominated_by_cheaper_equivalent",
    "selection_skipped_worse_by_price",
    "selection_skipped_same_platform_without_meaningful_difference",
    "selection_skipped_lower_ranked_alternative",
)
LEGACY_REJECTION_ALIAS_KEYS = (
    "rejected_missing_required",
    "rejected_stock",
    "rejected_right_size",
    "rejected_duplicate",
)
REJECTION_SUMMARY_KEYS = (
    "accepted",
    "accepted_after_validation",
    *VALIDATION_REJECTION_KEYS,
    *SELECTION_SKIP_KEYS,
    *LEGACY_REJECTION_ALIAS_KEYS,
    "validation_rejected_count",
    "selection_skipped_count",
)
REJECTION_REASON_ORDER = (
    *VALIDATION_REJECTION_KEYS,
    *SELECTION_SKIP_KEYS,
)
REJECTION_REASON_MESSAGES = {
    "rejected_missing_required_role": "Missing required core BOM role",
    "rejected_stock_shortage": "Insufficient or unknown stock",
    "rejected_platform_cpu_mismatch": "Platform and CPU are incompatible",
    "rejected_ram_capacity_unknown": "RAM module capacity is unknown",
    "rejected_quantity_materialization_failed": "Could not safely materialize BOM quantity",
    "rejected_optional_core_conflict": "Optional component was selected in core BOM",
    "rejected_invalid_schema": "Composer JSON/schema contract failed",
    "rejected_invalid_candidate_type": "Invalid candidate/source type",
    "rejected_invalid_price_or_currency": "Missing safe price or currency",
    "rejected_right_size_rejected": "Rejected by right-size validator",
    "rejected_fatal": "Совместимость или фатальный риск",
    "rejected_missing_required": "Неполная обязательная комплектация",
    "rejected_stock": "Недостаточный или неизвестный складской остаток",
    "rejected_role_mismatch": "Компонент указан не в своей роли",
    "rejected_unknown_component": "Неизвестный component_candidate_id или source_candidate_id",
    "rejected_other": "Не прошел обязательную валидацию",
    "selection_skipped_duplicate": "Дубликат уже выбранной комплектации",
    "selection_skipped_dominated_by_cheaper_equivalent": (
        "Уступает более близкой или дешевой эквивалентной комплектации"
    ),
    "selection_skipped_worse_by_price": "Уступает выбранному варианту по цене",
    "selection_skipped_same_platform_without_meaningful_difference": (
        "Та же платформа без существенного отличия"
    ),
    "selection_skipped_lower_ranked_alternative": (
        "Не выбран после безопасного deterministic top selection"
    ),
    "rejected_right_size": "Хуже по right-size или цене при наличии более близкой альтернативы",
    "rejected_duplicate": "Дубликат уже проверенной комплектации",
}

SUPPORTED_PROVIDERS = {"openai", "openai-compatible", "openai_compatible"}
SUPPORTED_MODES = {"composer"}
ROLE_ORDER = [
    SERVER_PLATFORM_ROLE,
    SWITCH_ROLE,
    ROUTER_ROLE,
    FIREWALL_ROLE,
    ACCESS_POINT_ROLE,
    STORAGE_SYSTEM_ROLE,
    STORAGE_ARRAY_CONTROLLER_ROLE,
    CONTROLLER_MODULE_ROLE,
    DISK_SHELF_ROLE,
    DRIVE_ROLE,
    CACHE_ROLE,
    HOST_PORT_ROLE,
    PROTOCOL_MODULE_ROLE,
    CPU_ROLE,
    RAM_ROLE,
    SSD_ROLE,
    HDD_ROLE,
    STORAGE_CONTROLLER_ROLE,
    NETWORK_ADAPTER_ROLE,
    GPU_ROLE,
    TRANSCEIVER_ROLE,
    DAC_CABLE_ROLE,
    CABLE_ROLE,
    POWER_SUPPLY_ROLE,
    RAIL_KIT_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
    STACKING_MODULE_ROLE,
    OTHER_ACCESSORY_ROLE,
    UNMAPPED_ROLE,
]
CORE_BOM_ROLE_ORDER = [
    SERVER_PLATFORM_ROLE,
    SWITCH_ROLE,
    ROUTER_ROLE,
    FIREWALL_ROLE,
    ACCESS_POINT_ROLE,
    STORAGE_SYSTEM_ROLE,
    STORAGE_ARRAY_CONTROLLER_ROLE,
    CONTROLLER_MODULE_ROLE,
    DISK_SHELF_ROLE,
    DRIVE_ROLE,
    CACHE_ROLE,
    HOST_PORT_ROLE,
    PROTOCOL_MODULE_ROLE,
    CPU_ROLE,
    RAM_ROLE,
    SSD_ROLE,
    HDD_ROLE,
    STORAGE_CONTROLLER_ROLE,
    NETWORK_ADAPTER_ROLE,
    GPU_ROLE,
    TRANSCEIVER_ROLE,
    DAC_CABLE_ROLE,
    CABLE_ROLE,
    POWER_SUPPLY_ROLE,
    RAIL_KIT_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
    STACKING_MODULE_ROLE,
    OTHER_ACCESSORY_ROLE,
    UNMAPPED_ROLE,
]
OPTIONAL_ENGINEER_CHECK_ROLES = {
    STORAGE_CONTROLLER_ROLE,
    STORAGE_ARRAY_CONTROLLER_ROLE,
    CONTROLLER_MODULE_ROLE,
    DISK_SHELF_ROLE,
    CACHE_ROLE,
    HOST_PORT_ROLE,
    PROTOCOL_MODULE_ROLE,
    NETWORK_ADAPTER_ROLE,
    GPU_ROLE,
    TRANSCEIVER_ROLE,
    CABLE_ROLE,
    POWER_SUPPLY_ROLE,
    RAIL_KIT_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
    OTHER_ACCESSORY_ROLE,
    UNMAPPED_ROLE,
}
PROMPT_ROLE_BY_INTERNAL_ROLE = {
    SERVER_PLATFORM_ROLE: "platform",
    SWITCH_ROLE: SWITCH_ROLE,
    ROUTER_ROLE: ROUTER_ROLE,
    FIREWALL_ROLE: FIREWALL_ROLE,
    ACCESS_POINT_ROLE: ACCESS_POINT_ROLE,
    STORAGE_SYSTEM_ROLE: STORAGE_SYSTEM_ROLE,
    STORAGE_ARRAY_CONTROLLER_ROLE: STORAGE_ARRAY_CONTROLLER_ROLE,
    CONTROLLER_MODULE_ROLE: CONTROLLER_MODULE_ROLE,
    DISK_SHELF_ROLE: DISK_SHELF_ROLE,
    DRIVE_ROLE: DRIVE_ROLE,
    CACHE_ROLE: CACHE_ROLE,
    HOST_PORT_ROLE: HOST_PORT_ROLE,
    PROTOCOL_MODULE_ROLE: PROTOCOL_MODULE_ROLE,
    CPU_ROLE: CPU_ROLE,
    RAM_ROLE: RAM_ROLE,
    SSD_ROLE: SSD_ROLE,
    HDD_ROLE: HDD_ROLE,
    STORAGE_CONTROLLER_ROLE: STORAGE_CONTROLLER_ROLE,
    NETWORK_ADAPTER_ROLE: NETWORK_ADAPTER_ROLE,
    GPU_ROLE: GPU_ROLE,
    TRANSCEIVER_ROLE: TRANSCEIVER_ROLE,
    DAC_CABLE_ROLE: DAC_CABLE_ROLE,
    CABLE_ROLE: CABLE_ROLE,
    POWER_SUPPLY_ROLE: POWER_SUPPLY_ROLE,
    RAIL_KIT_ROLE: RAIL_KIT_ROLE,
    LICENSE_ROLE: LICENSE_ROLE,
    SUPPORT_ROLE: SUPPORT_ROLE,
    STACKING_MODULE_ROLE: STACKING_MODULE_ROLE,
    OTHER_ACCESSORY_ROLE: OTHER_ACCESSORY_ROLE,
    UNMAPPED_ROLE: UNMAPPED_ROLE,
}
INTERNAL_ROLE_BY_PROMPT_ROLE = {
    "platform": SERVER_PLATFORM_ROLE,
    SERVER_PLATFORM_ROLE: SERVER_PLATFORM_ROLE,
    SWITCH_ROLE: SWITCH_ROLE,
    ROUTER_ROLE: ROUTER_ROLE,
    FIREWALL_ROLE: FIREWALL_ROLE,
    ACCESS_POINT_ROLE: ACCESS_POINT_ROLE,
    "storage_array": STORAGE_SYSTEM_ROLE,
    "storage": STORAGE_SYSTEM_ROLE,
    STORAGE_SYSTEM_ROLE: STORAGE_SYSTEM_ROLE,
    STORAGE_ARRAY_CONTROLLER_ROLE: STORAGE_ARRAY_CONTROLLER_ROLE,
    "controller": STORAGE_ARRAY_CONTROLLER_ROLE,
    CONTROLLER_MODULE_ROLE: CONTROLLER_MODULE_ROLE,
    "shelf": DISK_SHELF_ROLE,
    "drive_shelf": DISK_SHELF_ROLE,
    DISK_SHELF_ROLE: DISK_SHELF_ROLE,
    "drives": DRIVE_ROLE,
    DRIVE_ROLE: DRIVE_ROLE,
    CACHE_ROLE: CACHE_ROLE,
    "host_ports": HOST_PORT_ROLE,
    HOST_PORT_ROLE: HOST_PORT_ROLE,
    "protocol": PROTOCOL_MODULE_ROLE,
    PROTOCOL_MODULE_ROLE: PROTOCOL_MODULE_ROLE,
    CPU_ROLE: CPU_ROLE,
    RAM_ROLE: RAM_ROLE,
    SSD_ROLE: SSD_ROLE,
    HDD_ROLE: HDD_ROLE,
    STORAGE_CONTROLLER_ROLE: STORAGE_CONTROLLER_ROLE,
    NETWORK_ADAPTER_ROLE: NETWORK_ADAPTER_ROLE,
    GPU_ROLE: GPU_ROLE,
    TRANSCEIVER_ROLE: TRANSCEIVER_ROLE,
    DAC_CABLE_ROLE: DAC_CABLE_ROLE,
    CABLE_ROLE: CABLE_ROLE,
    POWER_SUPPLY_ROLE: POWER_SUPPLY_ROLE,
    RAIL_KIT_ROLE: RAIL_KIT_ROLE,
    LICENSE_ROLE: LICENSE_ROLE,
    SUPPORT_ROLE: SUPPORT_ROLE,
    STACKING_MODULE_ROLE: STACKING_MODULE_ROLE,
    OTHER_ACCESSORY_ROLE: OTHER_ACCESSORY_ROLE,
    UNMAPPED_ROLE: UNMAPPED_ROLE,
}
GENERIC_COMPONENT_ROLE_ALIASES = {
    "platform",
    "base_device",
    "device",
    "main_device",
    "chassis",
}
NETWORK_BASE_DEVICE_ROLES = {
    SWITCH_ROLE,
    ROUTER_ROLE,
    FIREWALL_ROLE,
    ACCESS_POINT_ROLE,
}
NETWORK_COMPOSER_ROLE_KEYS = (
    SWITCH_ROLE,
    ROUTER_ROLE,
    FIREWALL_ROLE,
    ACCESS_POINT_ROLE,
    TRANSCEIVER_ROLE,
    DAC_CABLE_ROLE,
    CABLE_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
    POWER_SUPPLY_ROLE,
    STACKING_MODULE_ROLE,
    OTHER_ACCESSORY_ROLE,
)
STORAGE_COMPOSER_ROLE_KEYS = (
    STORAGE_SYSTEM_ROLE,
    STORAGE_ARRAY_CONTROLLER_ROLE,
    CONTROLLER_MODULE_ROLE,
    DISK_SHELF_ROLE,
    DRIVE_ROLE,
    SSD_ROLE,
    HDD_ROLE,
    HOST_PORT_ROLE,
    PROTOCOL_MODULE_ROLE,
    TRANSCEIVER_ROLE,
    CABLE_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
    POWER_SUPPLY_ROLE,
    RAIL_KIT_ROLE,
    OTHER_ACCESSORY_ROLE,
)
SERVER_COMPOSER_ROLE_KEYS = (
    SERVER_PLATFORM_ROLE,
    "platform",
    CPU_ROLE,
    RAM_ROLE,
    "storage",
    SSD_ROLE,
    HDD_ROLE,
    STORAGE_CONTROLLER_ROLE,
    NETWORK_ADAPTER_ROLE,
    GPU_ROLE,
    TRANSCEIVER_ROLE,
    CABLE_ROLE,
    POWER_SUPPLY_ROLE,
    RAIL_KIT_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
    OTHER_ACCESSORY_ROLE,
)
PROPOSAL_ROLE_ALIASES = {
    "lower_price_with_tradeoff": "explicit_tradeoff",
    "budget_option": "cheapest_fit",
    "cheapest": "cheapest_fit",
    "price_optimal": "cheapest_fit",
    "technical": "technical_clean",
    "technical_clean_option": "technical_clean",
    "alternative": "alternative_platform",
    "alternative_vendor_or_platform": "alternative_platform",
    "fallback": "partial_fallback",
    "partial": "partial_fallback",
}
RECOMMENDATION_SLOT_ALIASES = {
    "lower_price_with_tradeoff": "alternative",
    "alternative_vendor_or_platform": "alternative",
    "cheapest_fit": "price_optimal",
    "budget_option": "price_optimal",
    "cheapest": "price_optimal",
    "technical_clean": "technical_clean",
    "technical": "technical_clean",
    "alternative_platform": "alternative",
    "explicit_tradeoff": "alternative",
    "partial_fallback": "partial_fallback",
}
MATRIX_KEYS = [
    ("platform", "platform_candidates", SERVER_PLATFORM_ROLE),
    (SWITCH_ROLE, "switch_candidates", SWITCH_ROLE),
    (ROUTER_ROLE, "router_candidates", ROUTER_ROLE),
    (FIREWALL_ROLE, "firewall_candidates", FIREWALL_ROLE),
    (ACCESS_POINT_ROLE, "access_point_candidates", ACCESS_POINT_ROLE),
    (STORAGE_SYSTEM_ROLE, "storage_system_candidates", STORAGE_SYSTEM_ROLE),
    (STORAGE_ARRAY_CONTROLLER_ROLE, "controller_candidates", STORAGE_ARRAY_CONTROLLER_ROLE),
    (CONTROLLER_MODULE_ROLE, "controller_module_candidates", CONTROLLER_MODULE_ROLE),
    (DISK_SHELF_ROLE, "disk_shelf_candidates", DISK_SHELF_ROLE),
    (DRIVE_ROLE, "drive_candidates", DRIVE_ROLE),
    (CACHE_ROLE, "cache_candidates", CACHE_ROLE),
    (HOST_PORT_ROLE, "host_port_candidates", HOST_PORT_ROLE),
    (PROTOCOL_MODULE_ROLE, "protocol_module_candidates", PROTOCOL_MODULE_ROLE),
    (CPU_ROLE, "cpu_candidates", CPU_ROLE),
    (RAM_ROLE, "ram_candidates", RAM_ROLE),
    (SSD_ROLE, "ssd_candidates", SSD_ROLE),
    (HDD_ROLE, "hdd_candidates", HDD_ROLE),
    (STORAGE_CONTROLLER_ROLE, "storage_controller_candidates", STORAGE_CONTROLLER_ROLE),
    (NETWORK_ADAPTER_ROLE, "network_adapter_candidates", NETWORK_ADAPTER_ROLE),
    (GPU_ROLE, "gpu_candidates", GPU_ROLE),
    (TRANSCEIVER_ROLE, "transceiver_candidates", TRANSCEIVER_ROLE),
    (DAC_CABLE_ROLE, "dac_cable_candidates", DAC_CABLE_ROLE),
    (CABLE_ROLE, "cable_candidates", CABLE_ROLE),
    (POWER_SUPPLY_ROLE, "power_supply_candidates", POWER_SUPPLY_ROLE),
    (RAIL_KIT_ROLE, "rail_kit_candidates", RAIL_KIT_ROLE),
    (LICENSE_ROLE, "license_candidates", LICENSE_ROLE),
    (SUPPORT_ROLE, "support_candidates", SUPPORT_ROLE),
    (STACKING_MODULE_ROLE, "stacking_module_candidates", STACKING_MODULE_ROLE),
    (OTHER_ACCESSORY_ROLE, "other_accessory_candidates", OTHER_ACCESSORY_ROLE),
    (UNMAPPED_ROLE, "unmapped_candidates", UNMAPPED_ROLE),
]
PACKAGE_MATRIX_KEYS = [
    (READY_SERVER_CANDIDATE_TYPE, "ready_server_candidates", READY_SERVER_CANDIDATE_TYPE),
    *MATRIX_KEYS,
]

LLM_CONFIGURATOR_SYSTEM_PROMPT = """
You are LLM Configuration Composer V3 for stable unified stock recommendations.
You are the main semantic reasoning stage. The application code before you only
plans a broad catalog/matrix and provides diagnostics; it is not the source of
truth for every requirement.

Return only a strict JSON object:
{
  "requirement_analysis": {
    "classified_requirements": [],
    "primary_object_feature_requirements": [],
    "purchasable_role_requirements": [],
    "accessory_or_consumable_requirements": [],
    "service_or_support_requirements": [],
    "logistics_or_commercial_constraints": [],
    "engineering_check_requirements": [],
    "fulfillment_decisions": [],
    "unverified_requirements": []
  },
  "fulfillment_decisions": [],
  "selected_components": [],
  "quantities": {},
  "assumptions": [],
  "engineer_checks": [],
  "hard_mismatch_risks": [],
  "unverified_requirements": [],
  "considered_candidate_count_by_role": {},
  "chosen_candidate_ids": [],
  "recommendations": [
    {
      "recommendation_id": "llm_rec_1",
      "proposal_role": "cheapest_fit|technical_clean|alternative_platform|partial_fallback|...",
      "recommendation_slot": "proposal strategy from hard rules",
      "source_type": "ready_server|build_from_parts|partial_build",
      "source_candidate_id": null,
      "selected_component_candidate_ids": {
        "platform": "core BOM component id from component_candidate_matrix",
        "cpu": "core BOM component id from component_candidate_matrix",
        "ram": "core BOM component id from component_candidate_matrix",
        "storage": "core BOM component id from component_candidate_matrix"
      },
      "component_candidate_ids": {
        "platform": "component id from component_candidate_matrix",
        "cpu": "component id from component_candidate_matrix",
        "ram": "component id from component_candidate_matrix",
        "storage": "component id from component_candidate_matrix"
      },
      "optional_component_candidate_ids": {
        "storage_controller": "optional component id only when it is not a hard requirement",
        "network_adapter": "optional component id only when it is not a hard requirement"
      },
      "title": "Оптимальный по цене вариант",
      "quantities": {"platform": 2, "cpu": 4, "ram": 8},
      "why_selected": "...",
      "why_selected_short": "...",
      "right_size_note": "...",
      "commercial_tradeoff": "...",
      "requirement_fulfillment_summary": [
        {
          "requirement_id": "req_id from package.classified_requirements",
          "fulfillment_mode": "separate_component_required|included_in_bundle_or_kit|...",
          "closed_by": "separate_component|primary_object|bundle_or_kit|unverified",
          "component_candidate_id": "component id when relevant",
          "evidence_text": "provided package/card/content evidence or empty",
          "engineer_check_ru": "required check when unverified"
        }
      ],
      "critical_checks": [],
      "engineer_checks": [],
      "engineering_review_required": true,
      "engineering_confidence": "preliminary_requires_engineer_review",
      "confidence": "low|medium|high"
    }
  ],
  "no_recommendation": null,
  "general_notes": []
}

QWEN STRICT COMPOSER PROTOCOL:
- You are not a free-form chatbot.
- You are a procurement/presales reasoning engine.
- Your task is to select component_candidate_id values from the provided matrix.
- Use package.original_request_text / package.user_request as the source of truth.
- You may receive compact candidate rows. Compact rows intentionally omit verbose,
  empty, null, and unknown facts.
- Absence of a fact means unknown, not satisfied.
- Reconstruct the user's requirements from the original request text. Do not rely
  only on package.classified_requirements or other pre-classified diagnostics.
- Determine fulfillment_mode yourself for each reconstructed requirement:
  separate_component_required, included_in_primary_object,
  included_in_selected_component, included_in_bundle_or_kit,
  unverified_requires_confirmation, service_or_support, logistics_constraint,
  or engineering_check_only.
- Include requirement_analysis with classified requirements, primary object
  feature requirements, purchasable role requirements, accessory/consumable
  requirements, service/support requirements, logistics/commercial constraints,
  engineering check requirements, fulfillment_decisions, and unverified_requirements.
- You must not invent products, prices, stock, part numbers, compatibility, quantities,
  or source facts.
- Application code is the source of truth for price, stock, quantities, totals, and
  final BOM materialization.
- Prefer minimal sufficient fit over overpowered configurations.
- Optimize for the cheapest commercially reasonable configuration that satisfies hard
  requirements.
- Treat prefer/preferred/if-possible vendor or model wording as optional, not hard,
  unless the user says must/only/exact.
- Use only component_candidate_id values from package.component_candidate_matrix,
  package.ready_stock_candidates, or package.rule_based_build_candidates.
- Choose the cheapest safe complete BOM. If no safe complete BOM can be built from
  the provided matrix, return structured no_recommendation.
- Return requirement_analysis and fulfillment_decisions. If a hard requirement cannot
  be proven, mark it unverified or engineer_check.
- If a hard requirement cannot be proven by selected component facts, package/card
  facts, or content facts, mark it unverified or engineer_check. Do not mark hard
  requirements satisfied just because a component was selected.
- Explain assumptions and engineering checks in short structured fields.
- Return a proposal pool up to proposal_pool_limit, not only final 3 user-facing
  options.
- Each proposal must have a clear commercial proposal_role: cheapest_fit,
  technical_clean, alternative_platform, partial_fallback, fallback_partial, or
  explicit_tradeoff.
- Do not create multiple proposals that only differ by optional NIC/controller/cable/rail.

CONFIGURATION FAMILY THINKING:
- First identify compatible architecture families from the matrix.
- A family is usually defined by platform CPU ecosystem/socket/chipset, RAM type,
  and storage/backplane class.
- Examples:
  Intel LGA4677/C741 + DDR5 + NVMe
  AMD SP5 + DDR5 + NVMe
- Within one family, reuse the best minimal sufficient CPU/RAM/storage base when
  it is commercially optimal.
- You do not have to choose different CPU/RAM/SSD in every proposal.
- You also do not have to force one common component base across all platforms.
- Different platform families may need different CPUs, RAM, or storage; that is a
  valid separate configuration family.
- Diversity should come from architecture family, platform, price, risk,
  TDP/PSU/backplane/DIMM/headroom, stock, and commercial tradeoff, not cosmetic
  component churn.
- Do not create fake diversity by changing components without a real price,
  stock, compatibility, or engineering reason.
- If RAM/SSD/CPU differs, explain the tradeoff in commercial_tradeoff or
  right_size_note.
- If only RAM or SSD changes without a strong reason, treat it as a tradeoff or
  alternative component, not a separate full recommendation.
- Return enough proposals for the application to build grouped presales output:
  configuration families with platform options inside compatible families.

CORE BOM RULES:
- Core BOM may include only components required by the user request or required to make
  the selected platform satisfy the request.
- For server build requests, core roles are normally: server_platform, cpu, ram, and
  ssd/hdd storage when storage is required.
- selected_component_candidate_ids/component_candidate_ids are core BOM only.
- Do not include storage_controller, network_adapter, cable, rail, GPU, HBA, OCP card,
  transceiver, or extra option in core BOM unless:
  a) user explicitly requested it, or
  b) the matrix clearly says it is mandatory for the chosen platform to satisfy the
     requested storage/network requirement.
- If such a component may be useful, put it into optional_component_candidate_ids,
  engineer_check_component_candidate_ids, or engineer_checks, not core components.
- Optional components must not be used to make the proposal look different from another
  otherwise identical proposal.

PRODUCT-GROUP ROLE CONTRACT:
- Always read package.product_group and package.component_role_contract before selecting
  component_candidate_ids.
- Pre-composer requirement classifier fields, source coverage, repair quality, and
  unclassified source fragments are diagnostics only. Use them as warnings, then
  reason from the original request text and the provided matrix.
- Read package.classified_requirements before selecting anything. Primary object
  features are hard constraints on the selected platform/device/system, not separate
  BOM roles. Engineering checks and logistics/commercial constraints must be carried
  into assumptions/checks instead of ignored.
- If package.requirement_classifier_status is partial or incomplete_repair, or
  package.unclassified_source_fragments is non-empty, say that requirement
  classification is partial and keep those source fragments as unverified engineer
  checks. Do not claim full feature validation from roles alone.
- For every classified requirement, read fulfillment_mode and should_create_bom_role.
  separate_component_required/service_or_support means select a separate component or
  service role when required. included_in_primary_object, included_in_selected_component,
  and included_in_bundle_or_kit must be treated as included only when package evidence
  or the requirement evidence_text supports it; otherwise keep it unverified and add an
  engineer check.
- In each recommendation, include requirement_fulfillment_summary rows that say which
  requirement_id values are closed by separate components, which are bundle/kit/platform
  inclusions with evidence, and which remain unverified_requires_confirmation.
- You must consider primary_object_feature requirements when choosing the primary
  platform/system/device. Do not ignore feature requirements just because they are not
  separate components.
- If the selected primary object lacks evidence for a hard feature, mark the
  assumption/engineer check. If candidate facts contradict a hard feature, do not
  recommend it.
- Do not treat role="unmapped" as required unless classified_requirements explicitly
  contains blocking_unmapped_purchasable_role.
- Use role names exactly as they appear in component_candidate_matrix. The matrix role is
  the source of truth for selected component roles.
- For product_group=server, use server roles: platform/server_platform, cpu, ram,
  storage/ssd/hdd, network_adapter, storage_controller, gpu, power_supply, license,
  support, and other server accessories. Legacy key platform is accepted only as the
  server_platform slot.
- For product_group=network, use real network roles: switch, router, firewall,
  access_point, transceiver, dac_cable, cable, license, support, power_supply,
  stacking_module, other_accessory. For a switch recommendation, emit
  component_candidate_ids with "switch": "...".
- Do not use server roles such as platform/cpu/ram/storage for network recommendations.
- For product_group=storage, use real storage roles: storage_system, controller,
  controller_module, disk_shelf, drive, ssd, hdd, host_port, protocol_module,
  transceiver, cable, license, support, power_supply, other_accessory. For a storage
  system recommendation, emit component_candidate_ids with "storage_system": "...".
- Do not use platform for storage_system and do not use server CPU/RAM/platform roles
  for storage recommendations.
- Generic legacy aliases platform/base_device/device/main_device/chassis may appear in
  older examples, but you should not emit them for network or storage. Application code
  may safely normalize such aliases only by actual matrix role and product_group.

QUANTITY REASONING RULES:
- You may explain intended per-server composition, but your quantities are advisory only.
- Do not overselect RAM/SSD/CPU.
- For RAM, calculate mentally:
  required_modules_per_server = ceil(required_ram_gb_per_server / module_capacity_gb).
  total_modules = required_modules_per_server * requested_server_count.
- For SSD/HDD:
  total_drives = required_drives_per_server * requested_server_count.
- For CPU:
  total_cpu = cpu_per_server * requested_server_count.
- If you are unsure about quantity, choose the component ID and state that application
  code must materialize minimal sufficient quantity.
- Never double RAM quantity unless the user explicitly requests mirrored memory, spare
  modules, special population rules, or there is matrix evidence requiring it.
- For 2 servers, 512 GB RAM per server, 32 GB modules: expected total is 32 modules.
- For 2 servers, 512 GB RAM per server, 64 GB modules: expected total is 16 modules.

Before returning JSON, silently verify:
- Does every selected component_candidate_id exist in the input matrix?
- Does core BOM satisfy hard requirements without optional components?
- Are there duplicate proposals that differ only by optional peripherals?
- Did you avoid overpaying for unnecessary RAM/storage/CPU?
- Did you avoid adding network/storage controllers unless requested or mandatory?
- Did you avoid claiming engineering compatibility as confirmed without evidence?
- Did you explain what an engineer must verify?

OUTPUT SCHEMA INTENT:
- proposal_role is commercial role only: cheapest_fit, technical_clean,
  alternative_platform, partial_fallback, or explicit_tradeoff.
- selected_component_candidate_ids/component_candidate_ids contain core BOM only.
- optional_component_candidate_ids are separate from core BOM.
- why_selected explains the commercial logic.
- right_size_note explains why this is minimally sufficient or why an overage is accepted.
- commercial_tradeoff explains the price/risk/availability tradeoff.
- engineer_checks lists what an engineer must verify before quotation.
- confidence means commercial fit only, not engineering validation.
- Do not use high confidence to mean engineering compatibility.
- engineering_confidence must be preliminary_requires_engineer_review unless external
  evidence confirms more.
- Every recommendation remains preliminary.
- Engineer review is mandatory before quotation.
- Missing QVL/support list is not fatal, but must be listed as engineer check.
- Explicit socket/family/RAM/storage mismatch must be avoided and should not be proposed.

Hard rules:
- You receive a broad but compact stock component matrix. It intentionally covers
  multiple CPU/RAM/storage/platform buckets instead of a narrow top-N shortlist.
- Think like a presales/procurement engineer:
  1. choose the hard core component base for the request;
  2. compare viable platforms against that base;
  3. decide what should go into the commercial offer;
  4. separately name optional/additional components;
  5. separately name engineer checks.
- Return a proposal pool, not the final user-visible shortlist. The user prompt includes
  proposal_pool_limit and final_display_limit. Aim to return proposal_pool_limit diverse
  proposals, usually 8-10, not only 3 final cards. The application will validate and
  deterministically select the safe top recommendations for display.
- Use component_candidate_matrix as the main source for new build_from_parts and
  partial_build recommendations. This is the primary feature: compose a new BOM
  for the selected product_group from component_candidate_id values.
- Use ready_stock_candidates and rule_based_build_candidates as context/baseline.
- ready_server candidates are just one stock candidate type; do not prefer them only
  because they are ready-made.
- ready_server rows in component_candidate_matrix are context only. Recommend a ready
  server through source_candidate_id, not through component_candidate_ids.
- Do not prefer a build only because it is cheaper if it is technically too risky.
- Select the best commercial shortlist for the user with minimal sufficient fit.
- For ready_server recommendations, source_candidate_id is required.
- For build_from_parts and partial_build recommendations, source_candidate_id may be
  null; component_candidate_ids are the source of truth.
- For build recommendations, use only component_candidate_id values already present
  in component_candidate_matrix. You may also reference a rule_based_build_candidate
  as baseline/context with source_candidate_id.
- title and why_selected are explanatory only. The application will display validated
  components/source candidates as the source of truth.
- Do not invent products, item IDs, part numbers, prices, currencies, or stock.
- Do not use component IDs for the wrong role.
- Use component_candidate_ids.platform for platform candidates, cpu for CPU, ram for
  memory, and storage or ssd/hdd for drives.
- Core BOM must contain only hard requirements: platform, CPU, RAM, requested storage,
  and platform completeness/PSU when expressed by the platform candidate. Do not include
  components in the core BOM only because they look useful.
- Do not put storage_controller, network_adapter, cables, rails, risers, NICs, HBAs,
  RAID cards, or other add-ons into core component_candidate_ids unless the user request
  explicitly requires them or the chosen platform/build cannot work without them.
- If storage_controller, network_adapter, cable, rail, or another extra option looks useful
  but is not a hard requirement, put it in optional_component_candidate_ids or mention it
  as optional_or_engineer_check in critical_checks. It must not increase the mandatory
  minimum BOM price.
- Do not invent characteristics or final compatibility.
- Do not claim final engineering compatibility without vendor support lists.
- Without external evidence, treat engineering compatibility as preliminary and requiring
  engineer review even when commercial fit is high.
- Do not assert compatibility when a socket or CPU family mismatch is visible.
- If a candidate has fatal mismatch warnings, use decision=do_not_use.
- If a build lacks any required role, set source_type=partial_build and list missing roles.
- If the user requests Intel Xeon, do not recommend an AMD platform/CPU mix.
- If the user requests DDR5 RAM, do not present a build without RAM as complete.
- Provide why_selected_short as one short complete sentence.
- Do not hide the need for engineering review.
- Set engineering_review_required to true for every recommendation.
- Calculate no totals; the application will calculate totals after validation.
- Cover different proposal strategies when the matrix allows it:
  1. recommendation_slot=price_optimal - the likely lowest-price full safe option.
  2. recommendation_slot=technical_clean - fewer visible technical risks, clearer
     form factor, CPU platform, DDR5, NVMe/backplane and PSU evidence.
  3. recommendation_slot=alternative_vendor_or_platform or alternative - useful
     different vendor/platform/CPU/RAM/SSD combination.
  4. recommendation_slot=exact_cpu_if_available - exact requested CPU core count,
     if such stocked CPU exists and is compatible.
  5. recommendation_slot=lower_price_with_tradeoff - cheaper nearby option with a
     clear tradeoff.
  6. recommendation_slot=partial_only_if_no_full - partial_build only when no full
     build is commercially useful.
- Do not stop after the first good option. Return several different full candidates
  and partial candidates only when they are useful fallback proposals.
- Do not change ranking criteria between runs. Apply criteria in this order:
  right-size first, then price, then technical cleanliness, then diversity.
- Optimize for the minimally sufficient configuration for the user's requirements, not
  the strongest possible product.
- Optional vendor/model preferences are not blockers unless must/only/exact.
- Primary mode is cost_minimal_fit: meet hard requirements at the lowest reasonable
  total price and avoid paying for unnecessary CPU cores, RAM, or storage capacity.
- Do not choose a much stronger CPU or larger drive when a closer, cheaper stocked
  candidate satisfies the requirement.
- Do not choose a more powerful or more expensive component if the matrix contains
  a closer and cheaper option that has enough stock and no visible compatibility
  conflict.
- If you choose a stronger CPU/RAM/drive, explain from matrix evidence: no closer
  option exists, the closer option is more expensive, the closer option lacks stock,
  the closer option has a visible compatibility conflict, or the stronger option is
  technically cleaner.
- If a ready_server does not close RAM/CPU/storage/stock key requirements, do not
  recommend it to the user.
- If a ready_server is useful but has checks, use recommend_with_checks and list them.
- If a build is incomplete, set source_type=partial_build and list missing roles.
- If you choose a component above the requirement, explain the reason in why_selected or
  right_size_note: no closer stocked option, cheaper than closer alternatives, better
  platform/vendor compatibility, or closer candidate stock is insufficient.
- Use decision=do_not_use only for internal/debug; those items will not be shown.

FINAL OUTPUT CONTRACT:
- The final answer must be exactly one JSON object.
- No markdown.
- No code fences.
- No prose before or after JSON.
- No <think> tags.
- No chain-of-thought.
- No comments.
- No trailing commas.
- Use only double quotes.
- The root object must contain a recommendations array. The parser also accepts
  proposals/proposal_pool aliases, but recommendations is preferred.
- The root object should contain requirement_analysis. Keep it concise and
  structured; do not include hidden reasoning.
- The model must not return an empty recommendations list without structured
  no_recommendation.
- Valid outcomes are exactly one of:
  1. A proposed primary_recommendation/recommendations BOM using only provided
     component_candidate_id values.
  2. A structured no_recommendation object explaining why no safe BOM can be built
     from the provided matrix.
- If no complete valid BOM exists, return:
{
  "recommendations": [],
  "no_recommendation": {
    "summary": "Short business summary.",
    "missing_roles": ["role_name"],
    "missing_required_capabilities": [
      {
        "role": "role_name",
        "capability_id": "capability id when provided",
        "requirement_text": "hard requirement text",
        "reason": "why the provided matrix cannot satisfy it"
      }
    ],
    "hard_mismatches": [
      {
        "role": "role_name",
        "component_candidate_id": "candidate id when relevant",
        "requirement": "hard requirement",
        "candidate_fact": "matrix fact that conflicts",
        "reason": "why this blocks a safe BOM"
      }
    ],
    "stock_shortages": [
      {
        "role": "role_name",
        "component_candidate_id": "candidate id when relevant",
        "required_quantity": 1,
        "available_quantity": 0,
        "reason": "why stock blocks the BOM"
      }
    ],
    "role_analysis": [
      {
        "role": "role_name",
        "status": "satisfied|missing|mismatch|stock_shortage|uncertain",
        "considered_candidate_ids": ["component_candidate_id"],
        "explanation": "matrix-level explanation"
      }
    ],
    "considered_candidate_ids": {
      "role_name": ["component_candidate_id"]
    },
    "explanation_ru": "Explain in Russian which hard requirements the matrix cannot satisfy."
  },
  "general_notes": []
}
- no_recommendation must cite role names and relevant component_candidate_id values
  where possible.
- no_recommendation must explain which hard requirements cannot be satisfied by the
  provided matrix.
- Reasoning must be compressed into short fields: why_selected, right_size_note,
  commercial_tradeoff, engineer_checks.
- Do not output hidden reasoning.
- Minimal valid shape:
{
  "recommendations": [
    {
      "recommendation_id": "llm_rec_1",
      "proposal_role": "cheapest_fit",
      "source_type": "build_from_parts",
      "source_candidate_id": null,
      "selected_component_candidate_ids": {
        "platform": "component_candidate_id",
        "cpu": "component_candidate_id",
        "ram": "component_candidate_id",
        "storage": "component_candidate_id"
      },
      "optional_component_candidate_ids": {},
      "quantities": {},
      "title": "Short title",
      "why_selected": "Commercial reason in one or two sentences.",
      "why_selected_short": "Short commercial reason.",
      "right_size_note": "Minimal sufficient fit; code materializes quantities.",
      "commercial_tradeoff": "Price/risk/availability tradeoff.",
      "requirement_fulfillment_summary": [],
      "engineer_checks": ["Check vendor support/QVL before quotation."],
      "engineering_review_required": true,
      "engineering_confidence": "preliminary_requires_engineer_review",
      "confidence": "medium"
    }
  ],
  "general_notes": []
}
""".strip()

LLM_CONFIGURATOR_SINGLE_BEST_SYSTEM_PROMPT = f"""
You are LLM Configuration Composer V4 for one cheapest valid universal stock quote.

DEFAULT TASK:
Return exactly ONE primary recommendation for the primary procurement object.

Rules:
- Read package.product_group before selecting roles. The primary product group may be
  server, network, storage, or another supported stock domain.
- Find the cheapest complete stocked solution that satisfies the hard request for
  that product group.
- Use package.original_request_text / package.user_request as the source of truth.
- Prefer/preferred/if-possible vendor or model wording is optional unless must/only/exact.
- Candidate rows may be compact. Omitted facts are unknown, not satisfied.
- You are the main semantic reasoning stage: reconstruct requirements from the
  original request and determine fulfillment decisions yourself.
- Required capabilities, required roles, and classified requirements are planner
  hints/diagnostics. Use them, but do not rely only on them and do not let partial
  pre-classification hide requirements present in the original text.
- Pre-composer requirement_classifier_status, source coverage, repair quality, and
  unclassified_source_fragments are diagnostic warnings, not blockers.
- Determine for every hard requirement whether it is a primary_object_feature,
  separate purchasable role, accessory/consumable, service/support,
  logistics/commercial constraint, or engineering check.
- Determine fulfillment_mode yourself. Do not create a separate BOM component for
  requirements with included_in_primary_object, included_in_selected_component,
  included_in_bundle_or_kit, or unverified_requires_confirmation unless you have
  matrix/package facts showing a separate purchasable component is required.
  Included bundle or kit requirements require evidence_text from the requirement or
  package/card/content facts; without evidence, leave them unverified and add an
  engineer check.
- Return root requirement_analysis with classified_requirements,
  primary_object_feature_requirements, purchasable_role_requirements,
  accessory_or_consumable_requirements, service_or_support_requirements,
  logistics_or_commercial_constraints, engineering_check_requirements,
  fulfillment_decisions, and unverified_requirements.
- Return requirement_fulfillment_summary in the recommendation: requirement_id,
  fulfillment_mode, closed_by=separate_component|primary_object|selected_component|
  bundle_or_kit|unverified, component_candidate_id when relevant, evidence_text, and
  engineer_check_ru when unverified.
- Use package.category_plan and role_coverage_summary to understand which distributor
  categories produced candidates for each capability.
- Include component IDs for every required role unless a platform onboard or bundled
  feature locally satisfies that hard capability.
- For product_group=network, use network role keys exactly as component_candidate_matrix
  exposes them. A switch quote must use component_candidate_ids.switch, not platform.
- Do not use server roles such as platform/cpu/ram/storage for network recommendations.
- For product_group=storage, use component_candidate_ids.storage_system for the base
  storage system, not platform.
- If required_roles contains network_adapter and platform onboard network does not
  satisfy the requested speed/media/ports, include a network_adapter candidate in
  component_candidate_ids.
- Do not ignore any hard capability, including network_adapter, storage_controller,
  GPU, transceiver, cable, license, support, or future hard roles.
- Do not create multiple alternatives.
- Do not create configuration families.
- Do not create fake diversity.
- Do not prefer brand over price.
- Gooxi is allowed if it satisfies hard requirements.
- ASUS/Supermicro/Vandor are not preferred unless they are cheapest valid or explicitly requested.
- Use only component_candidate_id values from the matrix.
- Return one primary_recommendation JSON object for the selected product group.
- If no safe complete recommendation exists, return structured no_recommendation with
  role-level reasons.
- No markdown, no prose, no chain of thought, JSON only.

Required primary_recommendation shape:
{{
  "primary_recommendation": {{
    "source_type": "build_from_parts",
    "title": "Cheapest valid complete stock build",
    "component_candidate_ids": {{
      "platform": "...",
      "cpu": "...",
      "ram": "...",
      "ssd": "..."
    }},
    "why_selected": "Cheapest complete stocked build that satisfies hard requirements.",
    "assumptions": [],
    "engineer_checks": []
  }},
  "general_notes": []
}}

Compatibility contract:
- Use canonical field source_type, not candidate_type. candidate_type is accepted
  only as a legacy alias and should not be emitted by new responses.
- Inside recommendations/primary_recommendation do not emit selected_components,
  general_notes, tradeoffs, or unverified_requirements. Use component_candidate_ids,
  assumptions, commercial_tradeoff, engineer_checks, and
  requirement_fulfillment_summary instead.
- The example above is the server shape. For network, selected keys must be switch,
  router, firewall, access_point, transceiver, dac_cable, cable, license, support,
  power_supply, stacking_module, or other_accessory. For storage, selected keys must
  include storage_system for the base system.
- Application code is the source of truth for quantities, stock, totals and final validation.
- You may use candidate_type="ready_server" only when a ready server fully satisfies
  the hard request.
- For ready_server, include source_candidate_id.
- For build_from_parts, component_candidate_ids must include the hard core BOM only.
- Do not put optional controllers, NICs, cables, rails, HBAs, or OCP cards into the core BOM
  unless explicitly requested or required by matrix facts.
- If any hard capability cannot be satisfied, return no_recommendation.
- Hard capabilities may be satisfied by a selected component, platform_onboard facts,
  or a platform_bundle such as redundant 1+1 / 2x PSU included with the platform.
- Keep engineering_review_required=true.
- Do not calculate prices or quantities; code will materialize them.
- If no safe complete recommendation exists, return recommendations=[] plus
  no_recommendation with these fields:
  summary, missing_roles, missing_required_capabilities, hard_mismatches,
  stock_shortages, role_analysis, considered_candidate_ids, explanation_ru.
- Do not return an empty recommendations list unless no_recommendation is present and
  explains the exact missing or mismatched hard requirements from the matrix.

Legacy parser compatibility:
- If you cannot emit primary_recommendation, emit a recommendations array with exactly one item.
- Do not emit more than one recommendation.

{LLM_CONFIGURATOR_SYSTEM_PROMPT}
""".strip()

LLM_CONFIGURATOR_REPAIR_SYSTEM_PROMPT = """
You are revising already materialized stock proposals after the deterministic
validator found cost/fit issues.

Do not invent products, component IDs, prices, stock, quantities, compatibility, or facts.
Use only component_candidate_id values from allowed_candidate_alternatives or from the
original_accepted_proposals. Application code remains the source of truth for price,
stock, quantities, totals, and final BOM materialization.

Repair policy:
- Preserve hard requirements.
- Optimize for cheapest valid quote first: cost_minimal_valid_fit.
- Gooxi is allowed as a normal cheapest platform if there is no hard incompatibility.
- Do not prefer brand over price unless creating a separate branded_safe or
  engineering_clear alternative.
- ASUS, Supermicro, and other branded platforms are alternatives, not automatic
  replacements for the cheapest quote.
- If critique_facts show a cheaper equivalent for the cheapest_quote or price_optimal
  proposal, use it unless you explain a concrete technical tradeoff in
  commercial_tradeoff.
- Use only critique_facts marked as cheaper_equivalent. Do not choose matrix_note,
  engineer_check, or not_equivalent_requires_engineering_review alternatives for
  cheapest_quote.
- Unknown hard compatibility is not equivalent.
- Do not use a cheaper platform for cheapest_quote if form factor, CPU sockets,
  RAM type, storage/backplane, or PSU/completeness are unknown or contradicted.
- If you keep a more expensive equivalent in the cheapest quote, name the specific
  technical reason; do not cite brand preference alone.
- Every proposal remains preliminary and requires engineering review.
- Do not add storage controllers, NICs, cables, rails, or other add-ons to core BOM
  unless the request explicitly requires them or the proposal cannot satisfy hard
  requirements without them.
- Repair must return either a complete canonical BOM recommendation or
  recommendations=[] with structured no_recommendation. An empty recommendations
  list without no_recommendation is an invalid repair.

Return JSON only, using the same proposal schema as the original Composer:
{
  "recommendations": [
    {
      "recommendation_id": "llm_rec_1",
      "proposal_role": "cheapest_fit|technical_clean|alternative_platform|...",
      "source_type": "build_from_parts|partial_build|ready_server",
      "source_candidate_id": null,
      "component_candidate_ids": {
        "platform": "component_candidate_id",
        "cpu": "component_candidate_id",
        "ram": "component_candidate_id",
        "storage": "component_candidate_id"
      },
      "optional_component_candidate_ids": {},
      "quantities": {},
      "title": "Short title",
      "why_selected": "Commercial reason in one or two sentences.",
      "why_selected_short": "Short commercial reason.",
      "right_size_note": "Minimal sufficient fit; code materializes quantities.",
      "commercial_tradeoff": "Price/risk/availability tradeoff.",
      "requirement_fulfillment_summary": [],
      "engineer_checks": ["Check vendor support/QVL before quotation."],
      "engineering_review_required": true,
      "engineering_confidence": "preliminary_requires_engineer_review",
      "confidence": "medium"
    }
  ],
  "no_recommendation": null,
  "general_notes": []
}

No markdown. No prose. No code fences. No <think>. No chain-of-thought. No comments.
""".strip()

LLM_ONLINE_COMPOSER_SYSTEM_PROMPT = """
Ты Online Composer V1 for server stock recommendations.

Ты не просто ранжируешь. Ты presales composer: сначала выбери рациональную
общую компонентную базу CPU/RAM/SSD, затем сравни платформы, затем верни
proposal pool до proposal_pool_limit коммерчески сильных рекомендаций. Код
сам проверит pool и выберет финальные 1-3 пользовательские варианты:
1. оптимальный по цене;
2. технически более спокойный;
3. альтернативный бренд/платформа;
4. частичный fallback только если полный вариант объективно слабее или невозможен.

Use online/web search for selected candidates. Prefer official vendor sources first:
Dell, ASUS, Supermicro, Intel, AMD, KIOXIA, Samsung, Micron, Gooxi.

Return only a strict JSON object:
{
  "recommendations": [
    {
      "recommendation_id": "llm_rec_1",
      "recommendation_slot": "price_optimal|technical_clean|alternative|...",
      "source_type": "build_from_parts|partial_build|ready_server",
      "source_candidate_id": null,
      "component_candidate_ids": {
        "platform": "...",
        "cpu": "...",
        "ram": "...",
        "storage": "..."
      },
      "optional_component_candidate_ids": {
        "storage_controller": "...",
        "network_adapter": "..."
      },
      "quantities": {"platform": 2, "cpu": 4, "ram": 16, "storage": 4},
      "title": "...",
      "why_selected": "...",
      "why_selected_short": "...",
      "right_size_note": "...",
      "critical_checks": [],
      "evidence_summary": {
        "status": "confirmed|partially_confirmed|not_confirmed|error",
        "sources_count": 0,
        "confirmed_facts": [],
        "not_confirmed": [],
        "source_domains": [],
        "notes": "..."
      },
      "engineering_review_required": true,
      "confidence": "low|medium|high"
    }
  ],
  "no_recommendation": null,
  "general_notes": []
}

Hard rules:
- Use the provided normalized_requirements, ready_stock_candidates,
  rule_based_build_candidates and component_candidate_matrix.
- Candidate rows may be compact. Omitted facts are unknown, not satisfied. Use only
  component_candidate_id values from the matrix.
- Think as a presales/procurement engineer: choose the hard core component base, compare
  platforms, decide what goes into the commercial offer, then separate optional add-ons
  and engineer checks.
- Return a proposal pool, not the final user-visible shortlist. Aim to return
  proposal_pool_limit diverse proposals when the matrix allows it, not only 3 final
  cards; final 1-3
  user-visible recommendations are selected by application code.
- Use only component_candidate_id values already present in the input.
- Do not invent products, components, item IDs, part numbers, prices, currencies,
  stock, sources, source URLs, support lists, or facts.
- For build_from_parts and partial_build, component_candidate_ids are the source of truth.
- Use component_candidate_matrix role keys exactly. For product_group=network, use
  switch/router/firewall/access_point/etc. and do not use server roles such as
  platform/cpu/ram/storage. For product_group=storage, use storage_system for the base
  storage system and do not use platform.
- Core component_candidate_ids must include only hard requirements. Do not add
  storage_controller, network_adapter, cables, rails, or extra options to the mandatory
  BOM unless the request explicitly requires them or the platform cannot work without them.
- Put useful non-required add-ons into optional_component_candidate_ids or critical_checks
  as optional_or_engineer_check; they must not increase the mandatory minimum price.
- For ready_server, source_candidate_id is required.
- The application will materialize IDs, prices and stock; do not calculate totals.
- If online search did not find sources, set evidence_summary.status="not_confirmed",
  sources_count=0, empty confirmed_facts/source_domains, and explain briefly in notes.
- Do not claim confirmed compatibility without external sources.
- If sources confirm a socket, CPU generation, memory type, NVMe/backplane or hard
  platform mismatch, do not recommend that candidate.
- Missing official support list is an engineering check, not always fatal.
- Do not assert final engineering compatibility.
- Keep cost_minimal_fit: минимально достаточная конфигурация, без переплаты.
- If a candidate has visible stock shortage, wrong role, wrong platform family,
  socket mismatch, RAM type mismatch, or another hard incompatibility, do not recommend it.
- Set engineering_review_required=true for every recommendation.
- Do not return an empty recommendations list without structured no_recommendation.
- If no complete valid BOM exists, return recommendations=[] and no_recommendation with:
  summary, missing_roles, missing_required_capabilities, hard_mismatches,
  stock_shortages, role_analysis, considered_candidate_ids, explanation_ru.
- no_recommendation must cite role names and relevant component_candidate_id values
  where possible, and must explain which hard requirements cannot be satisfied by the
  provided matrix.
""".strip()

LLM_ONLINE_COMPOSER_EMPTY_RESPONSE_REPAIR_SYSTEM_PROMPT = """
You are Online Composer V1 empty-output repair for stock recommendations.

You returned no proposal and no structured no_recommendation. Using the same candidate
matrix, either produce one safe BOM or produce structured no_recommendation with exact
missing/mismatched requirements.

Do not invent products, component IDs, prices, stock, quantities, compatibility, source
facts, URLs, or support lists. Use only component_candidate_id values already present in
the provided component_candidate_matrix. Application code remains the source of truth for
prices, stock, quantities, totals, and final BOM materialization.

Valid outcomes are exactly one of:
1. A strict JSON object with recommendations containing one safe BOM proposal.
2. A strict JSON object with recommendations=[] and structured no_recommendation.

If returning no_recommendation, include:
{
  "recommendations": [],
  "no_recommendation": {
    "summary": "Short business summary.",
    "missing_roles": ["role_name"],
    "missing_required_capabilities": [
      {
        "role": "role_name",
        "capability_id": "capability id when provided",
        "requirement_text": "hard requirement text",
        "reason": "why the provided matrix cannot satisfy it"
      }
    ],
    "hard_mismatches": [
      {
        "role": "role_name",
        "component_candidate_id": "candidate id when relevant",
        "requirement": "hard requirement",
        "candidate_fact": "matrix fact that conflicts",
        "reason": "why this blocks a safe BOM"
      }
    ],
    "stock_shortages": [
      {
        "role": "role_name",
        "component_candidate_id": "candidate id when relevant",
        "required_quantity": 1,
        "available_quantity": 0,
        "reason": "why stock blocks the BOM"
      }
    ],
    "role_analysis": [
      {
        "role": "role_name",
        "status": "satisfied|missing|mismatch|stock_shortage|uncertain",
        "considered_candidate_ids": ["component_candidate_id"],
        "explanation": "matrix-level explanation"
      }
    ],
    "considered_candidate_ids": {
      "role_name": ["component_candidate_id"]
    },
    "explanation_ru": "Explain in Russian which hard requirements the matrix cannot satisfy."
  },
  "general_notes": []
}

No markdown. No prose. No code fences. No <think>. No chain-of-thought. No comments.
""".strip()

LLM_NO_RECOMMENDATION_COVERAGE_REPAIR_SYSTEM_PROMPT = f"""
You are no_recommendation coverage repair for stock recommendations.

{NO_RECOMMENDATION_COVERAGE_REPAIR_INSTRUCTION}

Do not invent products, component IDs, prices, stock, quantities, compatibility, source
facts, URLs, or support lists. Use only component_candidate_id values already present in
the provided component_candidate_matrix. Application code remains the source of truth for
prices, stock, quantities, totals, and final BOM materialization.

Valid outcomes are exactly one of:
1. A strict JSON object with recommendations containing one safe BOM proposal.
2. A strict JSON object with recommendations=[] and structured no_recommendation.

If returning no_recommendation, include role_analysis and considered_candidate_ids with
enough candidates per role to satisfy package.no_recommendation_coverage_thresholds.

No markdown. No prose. No code fences. No <think>. No chain-of-thought. No comments.
""".strip()

LLM_REQUIREMENT_CONTRACT_SYSTEM_PROMPT = """
You are the Requirement Contract pass for the V2.2 bounded Composer-first stock
pipeline.

Read only original_request_text and return strict JSON. Extract the commercial
procurement contract as an AI reasoning task, not as a regex parser.

Return:
- primary_object
- required_roles: purchasable hard BOM roles, using role keys from allowed_roles
- required_quantities_by_role: machine-readable quantities and useful facts per role
- hard_requirements
- optional_requirements
- primary_object_features
- purchasable_component_roles
- accessories
- services_support
- logistics_commercial_constraints
- fulfillment_expectations
- engineer_checks

Put prefer/preferred/if-possible vendor/model wording into optional_requirements unless
the request says must/only/exact. For storage, classify capacity/media/protocol/ports as
primary features, included features, or separate generic roles instead of stopping at
storage_system.

Do not invent products, component IDs, categories, prices, stock, or support facts.
Unknown facts must remain unknown/unverified, not satisfied.
Use "server_platform" for the server/chassis/platform role.
Use "ssd", "hdd", or "drive" for server drives when the request says storage.
Use storage roles from allowed_roles: storage_system, drive/ssd/hdd, controller,
controller_module, disk_shelf, host_port, protocol_module.
""".strip()

LLM_ROLE_EVALUATION_SYSTEM_PROMPT = """
You are the Role Evaluation pass for an AI multi-pass stock Composer.

Evaluate every candidate in the provided chunk for exactly one required role.
Return strict JSON:
{
  "role": "...",
  "considered_candidate_ids": ["..."],
  "best_candidate_ids": ["..."],
  "rejected_candidate_ids": [{"component_candidate_id": "...", "reason": "..."}],
  "uncertain_candidate_ids": ["..."],
  "missing_facts": ["..."],
  "role_specific_risks": ["..."],
  "cheapest_safe_candidates": ["..."],
  "exact_or_equivalent_candidates": ["..."]
}

Do not invent candidate IDs. Do not skip candidates in this chunk. If facts are
missing, mark uncertainty instead of assuming satisfaction.
""".strip()

LLM_MULTI_PASS_BOM_COMPOSER_SYSTEM_PROMPT = """
You are the Main Composer pass for the V2.2 bounded Composer-first stock cascade.

Use original_request_text, requirement_contract, the compact full candidate matrix,
stock/price/quantity facts, and optional role evaluation summaries to build one
complete, commercially useful BOM, or return structured no_recommendation.

Requirement contract checklist:
- You must use requirement_contract as a checklist.
- You must either select components and quantities for every hard required
  purchasable role, or return structured no_recommendation.
- optional_requirements are not hard blockers; explain unmet preferences as tradeoffs.
- You must not output a partial BOM as valid.
- If no candidate fits a required role, say so in no_recommendation.
- If required quantity cannot be met from provided stock facts, say so.
- If facts are unknown, mark the requirement unverified; unknown facts are not
  satisfied facts.
- Do not invent component_candidate_id values, products, prices, stock, quantities,
  compatibility, or source facts.
- Use only component_candidate_id values from component_candidate_matrix.

Return the same JSON shape as the normal Composer:
{
  "requirement_analysis": {},
  "fulfillment_decisions": [],
  "recommendations": [
    {
      "recommendation_id": "...",
      "proposal_role": "cheapest_fit|technical_clean|alternative_platform|partial_fallback",
      "source_type": "build_from_parts|partial_build|ready_server",
      "component_candidate_ids": {"platform": "..."},
      "quantities": {"platform": 1},
      "decision": "recommend|recommend_with_checks|do_not_use",
      "title": "...",
      "why_selected": "...",
      "assumptions": [],
      "commercial_tradeoff": null,
      "requirement_fulfillment_summary": [],
      "engineer_checks": [],
      "confidence": "low|medium|high"
    }
  ],
  "no_recommendation": null,
  "general_notes": []
}

Core component_candidate_ids must include all hard purchasable roles from the
contract unless the contract says the role is included in a selected primary object.
Do not emit selected_components, general_notes, tradeoffs, or
unverified_requirements inside a recommendation. Use canonical component_candidate_ids,
assumptions, commercial_tradeoff, requirement_fulfillment_summary, and
engineer_checks.
When role evaluation summaries are absent, reason directly over the compact full
candidate matrix.
If a hard role has no safe candidate, return recommendations=[] with structured
no_recommendation and concrete role-level reasons.
""".strip()

LLM_COMPLETENESS_CRITIC_SYSTEM_PROMPT = """
You are the Completeness Critic pass for an AI multi-pass stock Composer.

Compare original_request_text, requirement_contract, and proposed_bom.
Return strict JSON:
{
  "all_hard_requirements_covered": true,
  "missing_roles": [],
  "insufficient_quantities": [],
  "unverified_requirements": [],
  "hard_mismatch_risks": [],
  "recommended_repair_actions": []
}

Treat unknown facts as unverified, not satisfied. A BOM that omits any hard
purchasable role from the requirement contract is incomplete.
""".strip()

LLM_MULTI_PASS_REPAIR_SYSTEM_PROMPT = """
You are the Repair pass for an AI multi-pass stock Composer.

Repair the proposed BOM using the criticism, requirement contract, role
evaluations, and candidate facts. Return the normal Composer JSON shape.
If a complete safe BOM cannot be built, return recommendations=[] with structured
no_recommendation and concrete reasons. Do not invent IDs or facts.
Return either a complete canonical BOM recommendation or structured
no_recommendation. Do not return recommendations=[] without no_recommendation.
""".strip()

LLM_VALIDATION_AWARE_REPAIR_SYSTEM_PROMPT = """
You are the Validation-Aware Repair pass for the V2 bounded Composer-first stock
cascade.

The previous Composer output was parsed as a BOM but rejected by deterministic
application validation. Repair exactly once using only the supplied original
request, requirement_contract, component_candidate_matrix, rejected_bom, and
validator errors.

Hard rules:
- Return the normal canonical Composer JSON shape.
- Return exactly one of:
  1. corrected canonical recommendations with non-empty component_candidate_ids
     from the matrix for all hard selected roles;
  2. recommendations=[] with structured no_recommendation explaining why no safe
     repair exists;
  3. non-normalizable output, which the application will classify as
     repair_schema_failure. Avoid this by returning one of the two JSON shapes.
- Do not invent component_candidate_id values, products, prices, stock,
  quantities, compatibility, or source facts.
- Do not repeat any forbidden rejected component combination. Repeating the
  rejected BOM combination is forbidden.
- Treat validator_errors as authoritative. If they say platform/CPU, role,
  stock, price, quantity, missing role, or hard capability is invalid, fix it or
  return structured no_recommendation.
- Vendor-specific option kits must match platform vendor and compatibility.
  If no compatible platform/CPU or option-kit pair exists in the provided
  candidates, return structured no_recommendation that says no compatible pair
  was found in the provided candidates.
- Do not claim a BOM with empty selected_components, chosen_candidate_ids, or
  component_candidate_ids.
- Do not return a general_notes-only answer. Notes without a canonical BOM or
  structured no_recommendation are invalid repair output.
- Do not weaken requirements. Unknown facts are not satisfied facts.
- If no compatible safe BOM exists in the matrix, return structured
  no_recommendation explaining the validator reason and the roles/candidates
  considered.
""".strip()

LLM_EVIDENCE_REVIEW_SYSTEM_PROMPT = """
You are LLM Evidence Review for server configuration compatibility.

Return only a strict JSON object:
{
  "evidence_review": [
    {
      "recommendation_id": "...",
      "decision": "keep|downgrade_to_partial|reject",
      "evidence_confidence": "high|medium|low",
      "confirmed_facts": [],
      "missing_evidence": [],
      "fatal_concerns": [],
      "engineering_checks": [],
      "user_note": "..."
    }
  ],
  "general_notes": []
}

Hard rules:
- Do not invent products, component IDs, URLs, sources, support lists, or facts.
- Use only the provided evidence_pack and component matrix.
- Do not claim official support unless a provided source is official_vendor, cpu_vendor,
  memory_vendor, or storage_vendor for that exact component.
- If evidence is absent, say not confirmed.
- If explicit generation, socket, memory type, or drive bay mismatch is present,
  recommend reject and name the fatal concern.
- If no support list is found, keep with engineering check instead of making it fatal.
- Do not invent URLs.
- Engineer review remains mandatory.
""".strip()


class LlmRecommendationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(min_length=1)
    proposal_role: Literal[
        "cheapest_fit",
        "technical_clean",
        "alternative_platform",
        "partial_fallback",
        "fallback_partial",
        "explicit_tradeoff",
    ] | None = None
    recommendation_slot: Literal[
        "price_optimal",
        "technical_clean",
        "alternative",
        "alternative_vendor_or_platform",
        "exact_cpu_if_available",
        "lower_price_with_tradeoff",
        "partial_only_if_no_full",
        "partial_fallback",
    ] | None = None
    source_type: Literal["ready_server", "build_from_parts", "partial_build"]
    source_candidate_id: str | None = None
    selected_component_candidate_ids: dict[str, str | None] = Field(default_factory=dict)
    component_candidate_ids: dict[str, str | None] = Field(default_factory=dict)
    optional_component_candidate_ids: dict[str, str | None] = Field(default_factory=dict)
    engineer_check_component_candidate_ids: dict[str, str | None] = Field(
        default_factory=dict
    )
    decision: Literal["recommend", "recommend_with_checks", "do_not_use"] = "recommend"
    title: str = Field(min_length=1)
    display_name: str | None = None
    components: dict[str, str | None] = Field(default_factory=dict)
    quantities: dict[str, int] = Field(default_factory=dict)
    total_price_value: None = None
    total_price_currency: None = None
    price_note: str | None = None
    why_selected: str = Field(min_length=1)
    why_selected_short: str | None = None
    right_size_note: str | None = None
    commercial_tradeoff: str | None = None
    requirement_fulfillment_summary: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    what_is_missing: list[str] = Field(default_factory=list)
    critical_checks: list[str] = Field(default_factory=list)
    engineer_checks: list[str] = Field(default_factory=list)
    engineering_review_required: bool = True
    engineering_confidence: str | None = None
    confidence: Literal["low", "medium", "high"]
    evidence_summary: LlmRecommendationEvidenceSummaryPayload | None = None


class LlmRecommendationEvidenceSummaryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str = "not_confirmed"
    sources_count: int = 0
    confirmed_facts: list[str] = Field(default_factory=list)
    not_confirmed: list[str] = Field(default_factory=list)
    source_domains: list[str] = Field(default_factory=list)
    notes: str = ""


class LlmComposerResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_analysis: dict[str, Any] = Field(default_factory=dict)
    fulfillment_decisions: list[dict[str, Any]] = Field(default_factory=list)
    selected_components: list[dict[str, Any]] = Field(default_factory=list)
    quantities: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    engineer_checks: list[str] = Field(default_factory=list)
    hard_mismatch_risks: list[dict[str, Any]] = Field(default_factory=list)
    unverified_requirements: list[dict[str, Any]] = Field(default_factory=list)
    considered_candidate_count_by_role: dict[str, int] = Field(default_factory=dict)
    chosen_candidate_ids: list[str] = Field(default_factory=list)
    requirement_coverage_summary: dict[str, Any] = Field(default_factory=dict)
    source_fragments_covered: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[LlmRecommendationPayload] = Field(default_factory=list)
    no_recommendation: dict[str, Any] = Field(default_factory=dict)
    general_notes: list[str] = Field(default_factory=list)


class RequirementContractPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_object: str = ""
    required_roles: list[str] = Field(default_factory=list)
    required_quantities_by_role: dict[str, Any] = Field(default_factory=dict)
    hard_requirements: list[Any] = Field(default_factory=list)
    optional_requirements: list[Any] = Field(default_factory=list)
    primary_object_features: list[Any] = Field(default_factory=list)
    purchasable_component_roles: list[str] = Field(default_factory=list)
    accessories: list[Any] = Field(default_factory=list)
    services_support: list[Any] = Field(default_factory=list)
    logistics_commercial_constraints: list[Any] = Field(default_factory=list)
    fulfillment_expectations: list[Any] = Field(default_factory=list)
    engineer_checks: list[Any] = Field(default_factory=list)


class RoleEvaluationPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    considered_candidate_ids: list[str] = Field(default_factory=list)
    best_candidate_ids: list[str] = Field(default_factory=list)
    rejected_candidate_ids: list[Any] = Field(default_factory=list)
    uncertain_candidate_ids: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    role_specific_risks: list[str] = Field(default_factory=list)
    cheapest_safe_candidates: list[str] = Field(default_factory=list)
    exact_or_equivalent_candidates: list[str] = Field(default_factory=list)


class CompletenessCriticPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    all_hard_requirements_covered: bool = False
    missing_roles: list[str] = Field(default_factory=list)
    insufficient_quantities: list[Any] = Field(default_factory=list)
    unverified_requirements: list[Any] = Field(default_factory=list)
    hard_mismatch_risks: list[Any] = Field(default_factory=list)
    recommended_repair_actions: list[str] = Field(default_factory=list)


_LLM_RECOMMENDATION_PAYLOAD_ADAPTER = TypeAdapter(LlmRecommendationPayload)


class LlmEvidenceReviewItemPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recommendation_id: str = Field(min_length=1)
    decision: Literal["keep", "downgrade_to_partial", "reject"] = "keep"
    evidence_confidence: Literal["high", "medium", "low"] = "low"
    confirmed_facts: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    fatal_concerns: list[str] = Field(default_factory=list)
    engineering_checks: list[str] = Field(default_factory=list)
    user_note: str = ""


class LlmEvidenceReviewResponsePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    evidence_review: list[LlmEvidenceReviewItemPayload] = Field(default_factory=list)
    general_notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class LlmConfiguratorOutcome:
    enabled: bool
    used: bool = False
    output_mode: str = OUTPUT_MODE_SINGLE_BEST_COST_VALID
    recommended_builds: list[dict[str, Any]] = field(default_factory=list)
    primary_recommendation: dict[str, Any] = field(default_factory=dict)
    primary_recommendation_status: str = "not_available"
    no_recommendation_reason: dict[str, Any] = field(default_factory=dict)
    commercial_summary: dict[str, Any] = field(default_factory=dict)
    configuration_groups: list[dict[str, Any]] = field(default_factory=list)
    quote_recommendation: dict[str, Any] = field(default_factory=dict)
    grouped_presales_mode_used: bool = False
    selected_configuration_group_id: str | None = None
    selected_platform_option_id: str | None = None
    selected_platform_option_index: int | None = None
    general_notes: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    parse_diagnostics: dict[str, Any] = field(default_factory=dict)
    internal_warnings: list[str] = field(default_factory=list)
    proposal_count: int = 0
    valid_proposals_count: int = 0
    validation_rejected_count: int = 0
    selection_skipped_count: int = 0
    rejected_recommendations_count: int = 0
    validation_warnings: list[str] = field(default_factory=list)
    validation_summary: dict[str, int] = field(default_factory=dict)
    rejected_reasons_top: list[dict[str, Any]] = field(default_factory=list)
    rejected_recommendations_debug_safe: list[dict[str, Any]] = field(default_factory=list)
    evidence_pack: dict[str, Any] = field(default_factory=dict)
    evidence_review: dict[str, Any] = field(default_factory=dict)
    repair_used: bool = False
    repair_attempted: bool = False
    repair_success: bool = False
    repair_fallback_reason: str | None = None
    repair_critique_count: int = 0
    repair_critique_summary: list[str] = field(default_factory=list)
    repair_blocked_critique_count: int = 0
    repair_blocked_critique_summary: list[str] = field(default_factory=list)
    repair_savings_estimate: str | None = None
    repair_revised_proposals_count: int = 0
    repair_validation_summary: dict[str, int] = field(default_factory=dict)
    thinking_diagnostics: dict[str, Any] = field(default_factory=dict)
    package_diagnostics: dict[str, Any] = field(default_factory=dict)
    composer_attempt_decision: dict[str, Any] = field(default_factory=dict)
    composer_requirement_analysis: dict[str, Any] = field(default_factory=dict)
    composer_fulfillment_decisions: list[dict[str, Any]] = field(default_factory=list)
    composer_selected_components: list[dict[str, Any]] = field(default_factory=list)
    composer_quantities: dict[str, Any] = field(default_factory=dict)
    composer_assumptions: list[str] = field(default_factory=list)
    composer_engineer_checks: list[str] = field(default_factory=list)
    composer_hard_mismatch_risks: list[dict[str, Any]] = field(default_factory=list)
    composer_unverified_requirements: list[dict[str, Any]] = field(default_factory=list)
    composer_considered_candidate_count_by_role: dict[str, Any] = field(
        default_factory=dict
    )
    composer_chosen_candidate_ids: list[str] = field(default_factory=list)
    composer_source_coverage_summary: dict[str, Any] = field(default_factory=dict)
    validation_hard_mismatches: list[dict[str, Any]] = field(default_factory=list)
    validation_unverified_requirements: list[dict[str, Any]] = field(default_factory=list)
    final_status_source: str | None = None


@dataclass(frozen=True)
class _RejectedProposal:
    recommendation_id: str
    category: str
    message: str
    proposal_index: int | None = None
    debug_safe: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ValidatedRecommendationPool:
    recommendations: list[dict[str, Any]]
    accepted_recommendations: list[dict[str, Any]]
    configuration_groups: list[dict[str, Any]]
    quote_recommendation: dict[str, Any]
    selected_configuration_group_id: str | None
    selected_platform_option_id: str | None
    selected_platform_option_index: int | None
    warnings: list[str]
    proposal_count: int
    valid_count: int
    validation_rejected_count: int
    selection_skipped_count: int
    rejected_count: int
    validation_summary: dict[str, int]
    rejected_reasons_top: list[dict[str, Any]]
    rejected_debug_safe: list[dict[str, Any]]


@dataclass(frozen=True)
class _RepairCritique:
    facts: list[dict[str, Any]]
    alternatives_by_role: dict[str, list[dict[str, Any]]]
    savings_estimate: Decimal | None
    summary: list[str]
    blocked_facts: list[dict[str, Any]] = field(default_factory=list)
    blocked_summary: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _RepairPassResult:
    used: bool = False
    attempted: bool = False
    success: bool = False
    fallback_reason: str | None = None
    critique_count: int = 0
    critique_summary: list[str] = field(default_factory=list)
    blocked_critique_count: int = 0
    blocked_critique_summary: list[str] = field(default_factory=list)
    savings_estimate: str | None = None
    revised_proposals_count: int = 0
    validation_summary: dict[str, int] = field(default_factory=dict)
    validated_pool: _ValidatedRecommendationPool | None = None
    response: LlmComposerResponsePayload | None = None
    warnings: list[str] = field(default_factory=list)
    thinking_diagnostics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ComposerTimeoutFallbackResult:
    payload: dict[str, Any] | None
    diagnostics: dict[str, Any]
    error: Exception | None = None


@dataclass(frozen=True)
class _IndexedComponentCandidate:
    component_candidate_id: str
    prompt_role: str
    internal_role: str
    row: dict[str, Any]
    source: Mapping[str, Any]


@dataclass(frozen=True)
class _RoleEligibility:
    classification: str
    reason_codes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_equivalent(self) -> bool:
        return self.classification == "cheaper_equivalent" and not self.reason_codes


@dataclass(frozen=True)
class _RepairRamType:
    generation: str
    module_class: str
    is_server_memory: bool


@dataclass(frozen=True)
class _RepairAlternativeSearchResult:
    allowed: _IndexedComponentCandidate | None
    blocked: list[tuple[_IndexedComponentCandidate, _RoleEligibility]]


@dataclass(frozen=True)
class _IndexedStockCandidate:
    candidate_id: str
    candidate_type: str
    row: Mapping[str, Any]


@dataclass(frozen=True)
class _MaterializedBomQuantities:
    quantities: dict[str, int]
    quantity_details: dict[str, dict[str, Any]]
    warnings: list[str]
    error: str | None = None


@dataclass(frozen=True)
class _ComposerResponseResult:
    response: LlmComposerResponsePayload
    client: LlmClient
    owns_client: bool
    online_composer_used: bool
    online_diagnostics: dict[str, Any]
    schema_rejections: list[_RejectedProposal] = field(default_factory=list)
    proposal_indexes: list[int] = field(default_factory=list)
    proposal_count: int = 0


@dataclass(frozen=True)
class _MultiPassRoleEvaluation:
    role: str
    candidate_ids: list[str]
    summaries: list[dict[str, Any]]
    failed_chunks: list[dict[str, Any]]

    @property
    def considered_candidate_ids(self) -> list[str]:
        ids: list[str] = []
        for summary in self.summaries:
            ids.extend(_string_list(summary.get("considered_candidate_ids")))
        return _unique(ids)


@dataclass(frozen=True)
class _ParsedComposerPayload:
    response: LlmComposerResponsePayload
    schema_rejections: list[_RejectedProposal]
    proposal_indexes: list[int]
    proposal_count: int


@dataclass(frozen=True)
class _EmptyResponseRepairResult:
    attempted: bool = False
    success: bool = False
    response: LlmComposerResponsePayload | None = None
    schema_rejections: list[_RejectedProposal] = field(default_factory=list)
    proposal_indexes: list[int] = field(default_factory=list)
    proposal_count: int = 0
    error_type: str = ""
    parse_status: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _NoRecommendationCoverageRepairResult:
    gate_passed: bool = True
    repair_attempted: bool = False
    repair_success: bool = False
    coverage_rejected: bool = False
    coverage: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    repair_reason: str | None = None
    response: LlmComposerResponsePayload | None = None
    validated_pool: _ValidatedRecommendationPool | None = None
    warnings: list[str] = field(default_factory=list)
    revised_proposals_count: int = 0
    validation_summary: dict[str, int] = field(default_factory=dict)
    error_type: str = ""
    parse_status: str = ""
    thinking_diagnostics: dict[str, Any] = field(default_factory=dict)


def compose_llm_configurations(
    *,
    user_request: str | None,
    normalized_requirements: Any,
    ready_stock_candidates: list[Mapping[str, Any]],
    component_candidate_matrix: Mapping[str, Any],
    rule_based_build_candidates: list[Mapping[str, Any]],
    settings: LlmSettings | None = None,
    llm_client: LlmClient | None = None,
    web_evidence_settings: WebEvidenceSettings | None = None,
    web_search_provider: WebSearchProvider | None = None,
    evidence_cache: EvidenceSearchCache | None = None,
    llm_call_budget: LlmCallBudget | None = None,
) -> LlmConfiguratorOutcome:
    effective_settings = settings or get_llm_settings()
    effective_evidence_settings = web_evidence_settings or get_web_evidence_settings()
    effective_llm_client = budgeted_llm_client(llm_client, llm_call_budget)
    output_mode = _normalized_output_mode(effective_settings)
    enabled, disabled_reason = _enabled_state(effective_settings)
    if not enabled:
        return LlmConfiguratorOutcome(
            enabled=False,
            output_mode=output_mode,
            fallback_reason=disabled_reason,
            composer_attempt_decision=_composer_attempt_decision_without_package(
                enabled=False,
                output_mode=output_mode,
                provider_configured=_composer_provider_configured(
                    settings=effective_settings,
                    web_evidence_settings=effective_evidence_settings,
                    llm_client=effective_llm_client,
                ),
                blocked_by=[disabled_reason or "llm_configurator_disabled"],
            ),
        )

    if _single_best_output_mode(output_mode):
        limit = 1
        proposal_pool_limit = 1
    else:
        limit = max(
            1,
            min(
                effective_settings.llm_build_recommendations_limit,
                FINAL_SAFE_RECOMMENDATIONS_LIMIT,
            ),
        )
        proposal_pool_limit = effective_settings.llm_proposal_pool_limit
    candidates_per_role = effective_settings.llm_component_candidates_per_role
    package = build_llm_configurator_package(
        user_request=user_request,
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=ready_stock_candidates,
        component_candidate_matrix=component_candidate_matrix,
        rule_based_build_candidates=rule_based_build_candidates,
        candidates_per_role=candidates_per_role,
        proposal_pool_limit=proposal_pool_limit,
        final_display_limit=limit,
        output_mode=output_mode,
        max_package_chars=effective_settings.llm_configurator_max_package_chars,
    )
    package = prepare_v2_composer_package(
        package,
        max_package_chars=effective_settings.llm_configurator_max_package_chars,
    )
    attempt_decision = _composer_attempt_decision(
        package=package,
        settings=effective_settings,
        web_evidence_settings=effective_evidence_settings,
        llm_client=effective_llm_client,
        output_mode=output_mode,
        enabled=True,
    )

    def with_package_diagnostics(
        outcome: LlmConfiguratorOutcome,
    ) -> LlmConfiguratorOutcome:
        budget_diagnostics = llm_call_budget_diagnostics(llm_call_budget)
        parse_diagnostics = {
            **_safe_mapping(outcome.parse_diagnostics),
            **budget_diagnostics,
        }
        return _with_package_diagnostics(
            replace(
                outcome,
                parse_diagnostics=parse_diagnostics,
                composer_attempt_decision=(
                    outcome.composer_attempt_decision or attempt_decision
                ),
            ),
            package,
        )

    def not_attempted(
        outcome: LlmConfiguratorOutcome,
        reason: str,
    ) -> LlmConfiguratorOutcome:
        decision = _composer_attempt_decision_blocked(attempt_decision, reason)
        fallback_reason = (
            INCOMPLETE_MATRIX_EXPOSURE_REASON
            if reason == INCOMPLETE_MATRIX_EXPOSURE_REASON
            else _composer_not_attempted_reason(reason)
        )
        return with_package_diagnostics(
            replace(
                outcome,
                fallback_reason=fallback_reason,
                composer_attempt_decision=decision,
            )
        )

    if package.get("package_skipped_reason") == "package_over_budget_after_distillation":
        return not_attempted(
            _no_recommendation_outcome_from_package_skipped(
                output_mode=output_mode,
                package=package,
            ),
            "package_over_budget",
        )
    if package.get("package_candidate_exposure_incomplete"):
        return not_attempted(
            _no_recommendation_outcome_from_package_skipped(
                output_mode=output_mode,
                package=package,
            ),
            INCOMPLETE_MATRIX_EXPOSURE_REASON,
        )
    if package.get("llm_fallback_reason") == "llm_configurator_package_over_budget":
        return not_attempted(
            LlmConfiguratorOutcome(
                enabled=True,
                output_mode=output_mode,
                fallback_reason="llm_configurator_package_over_budget",
                internal_warnings=_string_list(package.get("package_budget_warnings")),
            ),
            "package_over_budget",
        )
    if package.get("package_skipped_reason"):
        skipped_reason = str(package.get("package_skipped_reason") or "package_skipped")
        blocked_reason = (
            INCOMPLETE_MATRIX_EXPOSURE_REASON
            if skipped_reason == INCOMPLETE_MATRIX_EXPOSURE_REASON
            else f"package_skipped:{skipped_reason}"
        )
        return not_attempted(
            _no_recommendation_outcome_from_package_skipped(
                output_mode=output_mode,
                package=package,
            ),
            blocked_reason,
        )
    if not attempt_decision["provider_configured"]:
        return not_attempted(
            LlmConfiguratorOutcome(
                enabled=True,
                output_mode=output_mode,
                fallback_reason="llm_provider_not_configured",
            ),
            "provider_not_configured",
        )
    if "product_group_unknown" in _string_list(attempt_decision.get("blocked_by")):
        reason = _no_recommendation_reason(
            fallback_reason="product_group_unknown",
            validated_pool=_empty_validated_recommendation_pool(),
            warnings=[],
            product_group="unknown",
            package_required_capabilities=_mapping_rows(
                package.get("required_capabilities")
            ),
            package_missing_required_capabilities=_mapping_rows(
                package.get("missing_required_capabilities")
            ),
        )
        return not_attempted(
            LlmConfiguratorOutcome(
                enabled=True,
                output_mode=output_mode,
                primary_recommendation_status="no_recommendation",
                no_recommendation_reason=reason,
                commercial_summary=_commercial_summary_from_no_recommendation(reason),
                fallback_reason="product_group_unknown",
            ),
            "product_group_unknown",
        )
    package_product_group = str(package.get("product_group") or "server")
    package_ready_candidate_ids = {
        _candidate_id(candidate)
        for candidate in _mapping_rows(package.get("ready_stock_candidates"))
        if _candidate_id(candidate)
    }
    ready_stock_candidates_for_validation = [
        candidate
        for candidate in ready_stock_candidates
        if _candidate_id(candidate) in package_ready_candidate_ids
    ]
    has_ready_server_candidate = (
        package_product_group == "server"
        and _has_ready_server_candidate(_mapping_rows(package.get("ready_stock_candidates")))
    )
    package_missing_required_roles = _string_list(
        package.get("missing_required_roles_before_llm")
        or package.get("missing_required_roles")
    )
    package_missing_required_capabilities = _mapping_rows(
        package.get("missing_required_capabilities_before_llm")
        or package.get("missing_required_capabilities")
    )
    package_missing_required_roles = _blocking_missing_required_roles(
        package_missing_required_roles,
        package=package,
    )
    package_missing_required_capabilities = _blocking_missing_required_capabilities(
        package_missing_required_capabilities
    )
    package_missing_required_capabilities = _unique_mapping_rows(
        [
            *package_missing_required_capabilities,
            *_role_coverage_missing_required_capabilities(package),
        ]
    )
    semantic_precomposer_blockers_are_diagnostics = (
        _semantic_precomposer_blockers_are_diagnostics(package)
    )
    if semantic_precomposer_blockers_are_diagnostics:
        (
            package_missing_required_roles,
            package_missing_required_capabilities,
        ) = _objective_precomposer_missing_after_semantic_downgrade(
            package,
            missing_roles=package_missing_required_roles,
            missing_capabilities=package_missing_required_capabilities,
        )
    if package_missing_required_roles and not package_missing_required_capabilities:
        package_missing_required_capabilities = _missing_capability_rows_for_roles(
            package_missing_required_roles,
            required_capabilities=_mapping_rows(package.get("required_capabilities")),
        )
    if (
        package_missing_required_roles or package_missing_required_capabilities
    ) and not has_ready_server_candidate:
        missing_outcome = _no_recommendation_outcome_from_package_missing(
            output_mode=output_mode,
            package=package,
            missing_required_capabilities=package_missing_required_capabilities,
        )
        return not_attempted(
            missing_outcome,
            missing_outcome.fallback_reason or "missing_required_roles_before_llm",
        )
    package_roles_without_candidates = (
        _required_roles_without_package_candidates(package)
        if package_product_group != "server"
        else []
    )
    if package_roles_without_candidates and not has_ready_server_candidate:
        missing_outcome = _no_recommendation_outcome_from_package_missing(
            output_mode=output_mode,
            package={
                **package,
                "missing_required_roles_before_llm": package_roles_without_candidates,
            },
            missing_required_capabilities=_missing_capability_rows_for_roles(
                package_roles_without_candidates,
                required_capabilities=_mapping_rows(
                    package.get("required_capabilities")
                ),
            ),
            fallback_reason="no_eligible_candidates_for_required_role",
        )
        return not_attempted(
            missing_outcome,
            missing_outcome.fallback_reason or "no_eligible_candidates_for_required_role",
        )
    component_index = _component_index_for_package(
        source_component_candidate_matrix=component_candidate_matrix,
        package=package,
    )
    stock_candidate_index = _stock_candidate_index(
        ready_stock_candidates=ready_stock_candidates_for_validation,
        rule_based_build_candidates=rule_based_build_candidates,
    )
    if not stock_candidate_index and not component_index:
        required_roles_without_candidates = _string_list(package.get("required_roles"))
        if package_product_group != "server" and required_roles_without_candidates:
            missing_required_capabilities = _mapping_rows(
                package.get("missing_required_capabilities")
            ) or _missing_capability_rows_for_roles(
                required_roles_without_candidates,
                required_capabilities=_mapping_rows(package.get("required_capabilities")),
            )
            missing_outcome = _no_recommendation_outcome_from_package_missing(
                output_mode=output_mode,
                package=package,
                missing_required_capabilities=missing_required_capabilities,
                fallback_reason="no_eligible_candidates_for_required_role",
            )
            return not_attempted(
                missing_outcome,
                missing_outcome.fallback_reason
                or "no_eligible_candidates_for_required_role",
            )
        return not_attempted(
            LlmConfiguratorOutcome(
                enabled=True,
                output_mode=output_mode,
                fallback_reason="llm_configurator_no_stock_candidates",
            ),
            "llm_configurator_no_stock_candidates",
        )
    if not component_index and not any(
        candidate.candidate_type == READY_SERVER_CANDIDATE_TYPE
        for candidate in stock_candidate_index.values()
    ):
        return not_attempted(
            LlmConfiguratorOutcome(
                enabled=True,
                output_mode=output_mode,
                fallback_reason="llm_configurator_no_component_candidates",
            ),
            "llm_configurator_no_component_candidates",
        )

    online_composer_requested = _online_composer_requested(effective_evidence_settings)
    if _multi_pass_composer_requested(package, effective_settings):
        composer_result = _generate_multi_pass_composer_response(
            package=package,
            settings=effective_settings,
            web_evidence_settings=effective_evidence_settings,
            llm_client=effective_llm_client,
            llm_call_budget=llm_call_budget,
            output_mode=output_mode,
        )
    else:
        composer_result = _generate_composer_response(
            package=package,
            settings=effective_settings,
            web_evidence_settings=effective_evidence_settings,
            llm_client=effective_llm_client,
            llm_call_budget=llm_call_budget,
            online_composer_requested=online_composer_requested,
            output_mode=output_mode,
        )
    if isinstance(composer_result, LlmConfiguratorOutcome):
        return with_package_diagnostics(composer_result)

    response = composer_result.response
    client = composer_result.client
    owns_client = composer_result.owns_client

    if online_composer_requested:
        evidence_pack = _online_composer_evidence_pack(
            recommendations=response.recommendations,
            settings=effective_evidence_settings,
            diagnostics=composer_result.online_diagnostics,
            online_composer_used=composer_result.online_composer_used,
        )
        evidence_review: dict[str, Any] = {}
        evidence_review_warnings: list[str] = []
    else:
        evidence_pack = _collect_evidence_for_response(
            recommendations=response.recommendations,
            component_index=component_index,
            stock_candidate_index=stock_candidate_index,
            settings=effective_evidence_settings,
            llm_settings=effective_settings,
            normalized_requirements=package["normalized_requirements"],
            provider=web_search_provider,
            cache=evidence_cache,
        )
        evidence_review, evidence_review_warnings = _run_evidence_review(
            client=client,
            recommendations=response.recommendations,
            normalized_requirements=package["normalized_requirements"],
            component_matrix=package["component_candidate_matrix"],
            evidence_pack=evidence_pack,
        )
        evidence_pack = _mark_composer_attempt_diagnostics(
            evidence_pack,
            settings=effective_evidence_settings,
            diagnostics=composer_result.online_diagnostics,
            online_composer_used=composer_result.online_composer_used,
        )
    validated_pool = _validate_recommendations(
        response.recommendations,
        stock_candidate_index=stock_candidate_index,
        component_index=component_index,
        user_request=package["user_request"],
        normalized_requirements=package["normalized_requirements"],
        limit=limit,
        evidence_pack=evidence_pack,
        evidence_review=evidence_review,
        use_recommendation_evidence=online_composer_requested,
        schema_rejections=composer_result.schema_rejections,
        proposal_indexes=composer_result.proposal_indexes,
        proposal_count=composer_result.proposal_count,
    )
    if online_composer_requested and _online_composer_needs_posthoc_relation_evidence(
        evidence_pack,
        validated_pool.recommendations,
    ):
        selected_recommendation_ids = _selected_recommendation_ids(
            validated_pool.recommendations
        )
        relation_pack = _collect_relation_evidence_for_selected_recommendations(
            recommendations=validated_pool.recommendations,
            component_index=component_index,
            settings=effective_evidence_settings,
            llm_settings=effective_settings,
            normalized_requirements=_safe_mapping(package["normalized_requirements"]),
            provider=web_search_provider,
            cache=evidence_cache,
        )
        evidence_pack = _merge_online_composer_relation_evidence_pack(
            evidence_pack,
            relation_pack,
            settings=effective_evidence_settings,
        )
        if selected_recommendation_ids and (
            _int_value(relation_pack.get("total_tasks")) or 0
        ) > 0:
            validated_pool = _validate_recommendations(
                response.recommendations,
                stock_candidate_index=stock_candidate_index,
                component_index=component_index,
                user_request=package["user_request"],
                normalized_requirements=package["normalized_requirements"],
                limit=limit,
                evidence_pack=evidence_pack,
                evidence_review=evidence_review,
                use_recommendation_evidence=False,
                selection_eligible_recommendation_ids=selected_recommendation_ids,
                schema_rejections=composer_result.schema_rejections,
                proposal_indexes=composer_result.proposal_indexes,
                proposal_count=composer_result.proposal_count,
            )
    bounded_v2_cascade = _is_v2_composer_package(package) and str(
        composer_result.online_diagnostics.get("composer_mode") or ""
    ) in {"composer_cascade", "deep_audit"}
    if bounded_v2_cascade:
        repair_result = _run_validation_aware_repair_if_needed(
            package=package,
            settings=effective_settings,
            client=client,
            primary_response=response,
            primary_validated_pool=validated_pool,
            stock_candidate_index=stock_candidate_index,
            component_index=component_index,
            limit=limit,
            evidence_pack=evidence_pack,
            evidence_review=evidence_review,
            use_recommendation_evidence=online_composer_requested,
        )
    else:
        repair_result = _run_repair_pass(
            package=package,
            settings=effective_settings,
            llm_client=effective_llm_client,
            llm_call_budget=llm_call_budget,
            primary_response=response,
            primary_validated_pool=validated_pool,
            stock_candidate_index=stock_candidate_index,
            component_index=component_index,
            limit=limit,
            evidence_pack=evidence_pack,
            evidence_review=evidence_review,
            use_recommendation_evidence=online_composer_requested,
            output_mode=output_mode,
        )
    if repair_result.validated_pool is not None and repair_result.response is not None:
        response = repair_result.response
        validated_pool = repair_result.validated_pool
    if repair_result.diagnostics:
        evidence_pack = _merge_evidence_pack_diagnostics(
            evidence_pack,
            repair_result.diagnostics,
        )
    deterministic_primary_role_repair_result = (
        _run_deterministic_primary_role_repair_if_needed(
            package=package,
            response=response,
            validated_pool=validated_pool,
            component_index=component_index,
            stock_candidate_index=stock_candidate_index,
            limit=limit,
            evidence_pack=evidence_pack,
            evidence_review=evidence_review,
            use_recommendation_evidence=online_composer_requested,
        )
    )
    if (
        deterministic_primary_role_repair_result.validated_pool is not None
        and deterministic_primary_role_repair_result.response is not None
    ):
        response = deterministic_primary_role_repair_result.response
        validated_pool = deterministic_primary_role_repair_result.validated_pool
    if deterministic_primary_role_repair_result.diagnostics:
        evidence_pack = _merge_evidence_pack_diagnostics(
            evidence_pack,
            deterministic_primary_role_repair_result.diagnostics,
        )
    if bounded_v2_cascade:
        coverage_repair_result = _NoRecommendationCoverageRepairResult(
            gate_passed=True,
            thresholds=_no_recommendation_coverage_thresholds(effective_settings),
        )
    else:
        coverage_repair_result = _run_no_recommendation_coverage_repair_if_needed(
            package=package,
            settings=effective_settings,
            client=client,
            response=response,
            validated_pool=validated_pool,
            stock_candidate_index=stock_candidate_index,
            component_index=component_index,
            limit=limit,
            evidence_pack=evidence_pack,
            evidence_review=evidence_review,
            use_recommendation_evidence=online_composer_requested,
        )
    if (
        coverage_repair_result.response is not None
        and coverage_repair_result.validated_pool is not None
    ):
        response = coverage_repair_result.response
        validated_pool = coverage_repair_result.validated_pool
    coverage_repair_diagnostics = _no_recommendation_coverage_repair_diagnostics(
        coverage_repair_result
    )
    if coverage_repair_diagnostics:
        evidence_pack = _merge_evidence_pack_diagnostics(
            evidence_pack,
            coverage_repair_diagnostics,
        )
    thinking_diagnostics = _merge_thinking_diagnostics(
        _client_thinking_diagnostics(client),
        repair_result.thinking_diagnostics,
        coverage_repair_result.thinking_diagnostics,
        settings=effective_settings,
    )
    _close_owned_client(client, owns_client)
    combined_general_notes = _clean_notes(
        [
            *response.general_notes,
            *_string_list(evidence_review.get("general_notes")),
        ]
    )
    combined_warnings = _unique(
        [
            *validated_pool.warnings,
            *evidence_review_warnings,
            *repair_result.warnings,
            *deterministic_primary_role_repair_result.warnings,
            *coverage_repair_result.warnings,
        ]
    )
    if not validated_pool.recommendations:
        structured_no_recommendation = _has_structured_no_recommendation(response)
        coverage_rejected = bool(coverage_repair_result.coverage_rejected)
        repair_empty_without_no_recommendation = (
            repair_result.fallback_reason
            == "validation_repair_empty_without_no_recommendation"
        )
        fallback_reason = (
            "validation_repair_empty_without_no_recommendation"
            if repair_empty_without_no_recommendation
            else COMPOSER_NO_SAFE_COMPLETE_BOM
            if coverage_rejected
            else COMPOSER_STRUCTURED_NO_RECOMMENDATION
            if structured_no_recommendation
            else _no_valid_recommendations_fallback_reason(
                response=response,
                validated_pool=validated_pool,
            )
        )
        if (
            not structured_no_recommendation
            and bool(
                composer_result.online_diagnostics.get(
                    "online_composer_empty_response_repair_attempted"
                )
            )
            and not bool(
                composer_result.online_diagnostics.get(
                    "online_composer_empty_response_repair_success"
                )
            )
            and fallback_reason == "llm_configurator_no_proposals"
        ):
            fallback_reason = "llm_configurator_no_proposals_after_repair"
        no_recommendation_reason = (
            _structured_no_recommendation_reason(
                no_recommendation=response.no_recommendation,
                fallback_reason=fallback_reason,
                validated_pool=validated_pool,
                warnings=combined_warnings,
                product_group=str(package.get("product_group") or "server"),
            )
            if structured_no_recommendation
            else _no_recommendation_reason(
                fallback_reason=fallback_reason,
                validated_pool=validated_pool,
                warnings=combined_warnings,
                product_group=str(package.get("product_group") or "server"),
                package_required_capabilities=_mapping_rows(
                    package.get("required_capabilities")
                ),
                package_missing_required_capabilities=_mapping_rows(
                    package.get("missing_required_capabilities")
                ),
            )
        )
        no_recommendation_reason = _with_no_recommendation_coverage(
            no_recommendation_reason,
            package=package,
            no_recommendation=response.no_recommendation if structured_no_recommendation else None,
            thresholds=coverage_repair_result.thresholds,
        )
        if coverage_rejected:
            no_recommendation_reason = _incomplete_matrix_coverage_reason(
                no_recommendation_reason,
                coverage=(
                    coverage_repair_result.coverage
                    or _safe_mapping(
                        no_recommendation_reason.get("no_recommendation_coverage")
                    )
                ),
                thresholds=coverage_repair_result.thresholds,
                repair_reason=coverage_repair_result.repair_reason,
                repair_attempted=coverage_repair_result.repair_attempted,
                repair_success=coverage_repair_result.repair_success,
            )
        if fallback_reason == "llm_configurator_no_proposals_after_repair":
            no_recommendation_reason = _composer_returned_no_proposal_twice_reason(
                no_recommendation_reason
            )
        composer_validation_fields = _composer_output_validation_fields(
            response,
            validated_pool,
        )
        normalized_result = normalize_composer_result(
            product_group=str(package.get("product_group") or "server"),
            primary_object=_package_primary_object(package),
            original_request_text=str(
                package.get("original_request_text") or package.get("user_request") or ""
            ),
            requirement_contract=_safe_mapping(package.get("requirement_contract")),
            role_evaluation_coverage_by_role=_safe_mapping(
                composer_result.online_diagnostics.get(
                    "role_evaluation_coverage_by_role"
                )
            ),
            bom_composer_output=_jsonable(response.model_dump()),
            completeness_critic_result=_safe_mapping(
                composer_result.online_diagnostics.get("completeness_critic_result")
            ),
            repair_composer_output=(
                _jsonable(response.model_dump())
                if repair_result.used
                or composer_result.online_diagnostics.get("repair_composer_used")
                else {}
            ),
            code_validation_result={
                "validation_hard_mismatches": composer_validation_fields[
                    "validation_hard_mismatches"
                ],
                "validation_unverified_requirements": composer_validation_fields[
                    "validation_unverified_requirements"
                ],
                "validation_summary": _jsonable(validated_pool.validation_summary),
                "rejected_recommendations": _jsonable(
                    validated_pool.rejected_debug_safe
                ),
            },
            final_status_source=(
                "composer_rejected_by_validation"
                if validated_pool.validation_rejected_count
                or validated_pool.rejected_count
                else "composer_no_recommendation"
            ),
            primary_recommendation_status="no_recommendation",
            llm_fallback_reason=fallback_reason,
            existing_no_recommendation_reason=no_recommendation_reason,
            warnings=combined_warnings,
        )
        fallback_reason = str(
            normalized_result.get("llm_fallback_reason") or fallback_reason
        )
        no_recommendation_reason = _safe_mapping(
            normalized_result.get("no_recommendation_reason")
        )
        return with_package_diagnostics(
            LlmConfiguratorOutcome(
                enabled=True,
                output_mode=output_mode,
                primary_recommendation_status="no_recommendation",
                no_recommendation_reason=no_recommendation_reason,
                commercial_summary=_commercial_summary_from_no_recommendation(
                    no_recommendation_reason
                ),
                fallback_reason=fallback_reason,
                general_notes=combined_general_notes,
                internal_warnings=combined_warnings,
                parse_diagnostics=_empty_response_parse_diagnostics(fallback_reason),
                proposal_count=validated_pool.proposal_count,
                valid_proposals_count=validated_pool.valid_count,
                validation_rejected_count=validated_pool.validation_rejected_count,
                selection_skipped_count=validated_pool.selection_skipped_count,
                rejected_recommendations_count=validated_pool.rejected_count,
                validation_warnings=combined_warnings,
                validation_summary=validated_pool.validation_summary,
                rejected_reasons_top=validated_pool.rejected_reasons_top,
                rejected_recommendations_debug_safe=validated_pool.rejected_debug_safe,
                evidence_pack=evidence_pack,
                evidence_review=evidence_review,
                repair_used=repair_result.used,
                repair_attempted=repair_result.attempted,
                repair_success=repair_result.success,
                repair_fallback_reason=repair_result.fallback_reason,
                repair_critique_count=repair_result.critique_count,
                repair_critique_summary=repair_result.critique_summary,
                repair_blocked_critique_count=repair_result.blocked_critique_count,
                repair_blocked_critique_summary=repair_result.blocked_critique_summary,
                repair_savings_estimate=repair_result.savings_estimate,
                repair_revised_proposals_count=repair_result.revised_proposals_count,
                repair_validation_summary=repair_result.validation_summary,
                thinking_diagnostics=thinking_diagnostics,
                **composer_validation_fields,
            )
        )

    if _single_best_output_mode(output_mode) and not _has_complete_recommendation(
        validated_pool.recommendations
    ):
        fallback_reason = "llm_configurator_no_complete_recommendation"
        no_recommendation_reason = _no_recommendation_reason(
            fallback_reason=fallback_reason,
            validated_pool=validated_pool,
            warnings=combined_warnings,
            product_group=str(package.get("product_group") or "server"),
            package_required_capabilities=_mapping_rows(
                package.get("required_capabilities")
            ),
            package_missing_required_capabilities=_mapping_rows(
                package.get("missing_required_capabilities")
            ),
        )
        no_recommendation_reason = _with_no_recommendation_coverage(
            no_recommendation_reason,
            package=package,
        )
        composer_validation_fields = _composer_output_validation_fields(
            response,
            validated_pool,
        )
        normalized_result = normalize_composer_result(
            product_group=str(package.get("product_group") or "server"),
            primary_object=_package_primary_object(package),
            original_request_text=str(
                package.get("original_request_text") or package.get("user_request") or ""
            ),
            requirement_contract=_safe_mapping(package.get("requirement_contract")),
            role_evaluation_coverage_by_role=_safe_mapping(
                composer_result.online_diagnostics.get(
                    "role_evaluation_coverage_by_role"
                )
            ),
            bom_composer_output=_jsonable(response.model_dump()),
            completeness_critic_result=_safe_mapping(
                composer_result.online_diagnostics.get("completeness_critic_result")
            ),
            repair_composer_output=(
                _jsonable(response.model_dump())
                if repair_result.used
                or composer_result.online_diagnostics.get("repair_composer_used")
                else {}
            ),
            code_validation_result={
                "validation_hard_mismatches": composer_validation_fields[
                    "validation_hard_mismatches"
                ],
                "validation_unverified_requirements": composer_validation_fields[
                    "validation_unverified_requirements"
                ],
                "validation_summary": _jsonable(validated_pool.validation_summary),
                "rejected_recommendations": _jsonable(
                    validated_pool.rejected_debug_safe
                ),
            },
            final_status_source=(
                "composer_rejected_by_validation"
                if validated_pool.validation_rejected_count
                or validated_pool.rejected_count
                else "composer_no_recommendation"
            ),
            primary_recommendation_status="no_recommendation",
            llm_fallback_reason=fallback_reason,
            existing_no_recommendation_reason=no_recommendation_reason,
            warnings=combined_warnings,
        )
        fallback_reason = str(
            normalized_result.get("llm_fallback_reason") or fallback_reason
        )
        no_recommendation_reason = _safe_mapping(
            normalized_result.get("no_recommendation_reason")
        )
        return with_package_diagnostics(
            LlmConfiguratorOutcome(
                enabled=True,
                output_mode=output_mode,
                primary_recommendation_status="no_recommendation",
                no_recommendation_reason=no_recommendation_reason,
                commercial_summary=_commercial_summary_from_no_recommendation(
                    no_recommendation_reason
                ),
                fallback_reason=fallback_reason,
                general_notes=combined_general_notes,
                internal_warnings=combined_warnings,
                proposal_count=validated_pool.proposal_count,
                valid_proposals_count=validated_pool.valid_count,
                validation_rejected_count=validated_pool.validation_rejected_count,
                selection_skipped_count=validated_pool.selection_skipped_count,
                rejected_recommendations_count=validated_pool.rejected_count,
                validation_warnings=combined_warnings,
                validation_summary=validated_pool.validation_summary,
                rejected_reasons_top=validated_pool.rejected_reasons_top,
                rejected_recommendations_debug_safe=validated_pool.rejected_debug_safe,
                evidence_pack=evidence_pack,
                evidence_review=evidence_review,
                repair_used=repair_result.used,
                repair_attempted=repair_result.attempted,
                repair_success=repair_result.success,
                repair_fallback_reason=repair_result.fallback_reason,
                repair_critique_count=repair_result.critique_count,
                repair_critique_summary=repair_result.critique_summary,
                repair_blocked_critique_count=repair_result.blocked_critique_count,
                repair_blocked_critique_summary=repair_result.blocked_critique_summary,
                repair_savings_estimate=repair_result.savings_estimate,
                repair_revised_proposals_count=repair_result.revised_proposals_count,
                repair_validation_summary=repair_result.validation_summary,
                thinking_diagnostics=thinking_diagnostics,
                **composer_validation_fields,
            )
        )

    primary_recommendation = _primary_recommendation_from_selected(
        validated_pool.recommendations[0]
    )
    commercial_summary = _commercial_summary_from_primary(primary_recommendation)
    grouped_enabled = _grouped_presales_output_mode(output_mode)
    return with_package_diagnostics(
        LlmConfiguratorOutcome(
            enabled=True,
            used=True,
            output_mode=output_mode,
            recommended_builds=validated_pool.recommendations,
            primary_recommendation=primary_recommendation,
            primary_recommendation_status="valid",
            commercial_summary=commercial_summary,
            configuration_groups=(
                validated_pool.configuration_groups if grouped_enabled else []
            ),
            quote_recommendation=(
                validated_pool.quote_recommendation if grouped_enabled else {}
            ),
            grouped_presales_mode_used=(
                grouped_enabled and bool(validated_pool.configuration_groups)
            ),
            selected_configuration_group_id=(
                validated_pool.selected_configuration_group_id
                if grouped_enabled
                else None
            ),
            selected_platform_option_id=(
                validated_pool.selected_platform_option_id if grouped_enabled else None
            ),
            selected_platform_option_index=(
                validated_pool.selected_platform_option_index
                if grouped_enabled
                else None
            ),
            general_notes=combined_general_notes,
            internal_warnings=combined_warnings,
            proposal_count=validated_pool.proposal_count,
            valid_proposals_count=validated_pool.valid_count,
            validation_rejected_count=validated_pool.validation_rejected_count,
            selection_skipped_count=validated_pool.selection_skipped_count,
            rejected_recommendations_count=validated_pool.rejected_count,
            validation_warnings=combined_warnings,
            validation_summary=validated_pool.validation_summary,
            rejected_reasons_top=validated_pool.rejected_reasons_top,
            rejected_recommendations_debug_safe=validated_pool.rejected_debug_safe,
            evidence_pack=evidence_pack,
            evidence_review=evidence_review,
            repair_used=repair_result.used,
            repair_attempted=repair_result.attempted,
            repair_success=repair_result.success,
            repair_fallback_reason=repair_result.fallback_reason,
            repair_critique_count=repair_result.critique_count,
            repair_critique_summary=repair_result.critique_summary,
            repair_blocked_critique_count=repair_result.blocked_critique_count,
            repair_blocked_critique_summary=repair_result.blocked_critique_summary,
            repair_savings_estimate=repair_result.savings_estimate,
            repair_revised_proposals_count=repair_result.revised_proposals_count,
            repair_validation_summary=repair_result.validation_summary,
            thinking_diagnostics=thinking_diagnostics,
            **_composer_output_validation_fields(response, validated_pool),
        )
    )


def build_llm_configurator_package(
    *,
    user_request: str | None,
    normalized_requirements: Any,
    ready_stock_candidates: list[Mapping[str, Any]],
    component_candidate_matrix: Mapping[str, Any],
    rule_based_build_candidates: list[Mapping[str, Any]],
    candidates_per_role: int = 30,
    proposal_pool_limit: int = 10,
    final_display_limit: int = 5,
    output_mode: str = OUTPUT_MODE_SINGLE_BEST_COST_VALID,
    max_package_chars: int | None = None,
) -> dict[str, Any]:
    normalized_output_mode = _normalize_output_mode(output_mode)
    package_requirements = _package_normalized_requirements(
        normalized_requirements,
        component_candidate_matrix,
    )
    role_plan = _safe_mapping(component_candidate_matrix.get("role_plan"))
    semantic_fields = _semantic_package_fields(component_candidate_matrix, role_plan)
    category_plan = _safe_mapping(component_candidate_matrix.get("category_plan"))
    category_planner_source = _text_or_none(
        component_candidate_matrix.get("category_planner_source")
    )
    category_plan_source = _text_or_none(
        component_candidate_matrix.get("category_plan_source")
    )
    has_candidate_matrix_rows = _has_package_candidate_rows(component_candidate_matrix)
    if (
        not semantic_fields.get("semantic_planner_source")
        and _looks_like_technical_request(user_request)
        and not role_plan
        and not package_requirements
        and not category_plan
        and not has_candidate_matrix_rows
    ):
        semantic_fields = {
            **semantic_fields,
            "semantic_planner_source": SEMANTIC_SOURCE_PLANNER_UNAVAILABLE,
            "semantic_planner_used": False,
            "semantic_planner_fallback_reason": SEMANTIC_SOURCE_PLANNER_UNAVAILABLE,
        }
    product_group = (
        component_candidate_matrix.get("product_group")
        or semantic_fields.get("primary_product_group")
        or package_requirements.get("product_group")
        or (
            "unknown"
            if semantic_fields.get("semantic_planner_source")
            == SEMANTIC_SOURCE_PLANNER_UNAVAILABLE
            else SERVER_PRODUCT_GROUP
        )
    )
    ready_candidates_policy = _ready_candidates_package_policy(
        user_request=user_request,
        product_group=product_group,
        category_plan=category_plan,
        component_candidate_matrix=component_candidate_matrix,
    )
    matrix = _compact_component_candidate_matrix(
        component_candidate_matrix,
        candidates_per_role=candidates_per_role,
        include_ready_server=ready_candidates_policy["include_ready_server"],
        ready_server_limit=READY_SERVER_CANDIDATES_LIMIT,
    )
    if not ready_candidates_policy["include_ready_server"]:
        matrix.pop(READY_SERVER_CANDIDATE_TYPE, None)
    if product_group != "server":
        matrix.pop(READY_SERVER_CANDIDATE_TYPE, None)
    required_capabilities = _required_capabilities_for_package(
        package_requirements,
        component_candidate_matrix,
    )
    optional_capabilities = _optional_capabilities_for_package(
        package_requirements,
        component_candidate_matrix,
    )
    required_roles = _package_required_roles(package_requirements, role_plan)
    required_roles = _required_roles_after_classification(required_roles, role_plan)
    package_requirements = _requirements_with_role_plan(
        package_requirements,
        role_plan=role_plan,
        required_capabilities=required_capabilities,
        optional_capabilities=optional_capabilities,
        required_roles=required_roles,
        semantic_fields=semantic_fields,
    )
    semantic_package_skipped_reason = _semantic_package_skipped_reason(
        user_request=user_request,
        product_group=product_group,
        semantic_fields=semantic_fields,
        required_capabilities=required_capabilities,
        category_plan=category_plan,
        role_coverage_summary=_safe_mapping(
            component_candidate_matrix.get("role_coverage_summary")
        ),
        has_candidate_matrix_rows=has_candidate_matrix_rows,
    )
    matrix_package_skipped_reason = _text_or_none(
        component_candidate_matrix.get("package_skipped_reason")
    )
    package_skipped_reason = (
        (
            None
            if _matrix_skip_reason_should_be_rechecked(
                matrix_package_skipped_reason,
                component_candidate_matrix,
            )
            else matrix_package_skipped_reason
        )
        or semantic_package_skipped_reason
    )
    proposal_pool_instruction = (
        "Return exactly one primary_recommendation. Do not return alternatives, "
        "configuration families, or fake diversity. The application validates the single "
        "recommendation and materializes quantities/prices/stock."
        if _single_best_output_mode(normalized_output_mode)
        else (
            "Return a diverse proposal pool larger than final_display_limit; "
            "application code validates each proposal and selects safe deterministic top results. "
            "Core BOM must include only hard requirements; non-required controller/NIC/extra "
            "options belong in optional_component_candidate_ids or engineer checks."
        )
    )
    if package_skipped_reason:
        return _skipped_llm_configurator_package(
            user_request=user_request,
            product_group=product_group,
            package_requirements=package_requirements,
            required_capabilities=required_capabilities,
            optional_capabilities=optional_capabilities,
            required_roles=required_roles,
            role_plan=role_plan,
            semantic_fields=semantic_fields,
            category_plan=category_plan,
            component_candidate_matrix=component_candidate_matrix,
            category_planner_source=category_planner_source,
            category_plan_source=category_plan_source,
            package_skipped_reason=package_skipped_reason,
            output_mode=normalized_output_mode,
            proposal_pool_limit=proposal_pool_limit,
            final_display_limit=final_display_limit,
            proposal_pool_instruction=proposal_pool_instruction,
            max_package_chars=max_package_chars,
        )
    compact_ready_stock_candidates = _compact_ready_stock_candidates(
        ready_stock_candidates
        if ready_candidates_policy["include_ready_server"] and product_group == "server"
        else [],
        limit=READY_SERVER_CANDIDATES_LIMIT,
    )
    original_request_text = (user_request or "").strip()
    package = {
        "original_request_text": original_request_text,
        "user_request": original_request_text,
        "product_group": product_group,
        "normalized_requirements": package_requirements,
        "required_capabilities": required_capabilities,
        "optional_capabilities": optional_capabilities,
        "unsupported_or_unmapped_requirements": _string_list(
            component_candidate_matrix.get("unsupported_or_unmapped_requirements")
            or package_requirements.get("unsupported_or_unmapped_requirements")
        ),
        "required_roles": required_roles,
        "missing_required_roles": _blocking_missing_required_roles(
            _string_list(component_candidate_matrix.get("missing_required_roles")),
            package={**semantic_fields, "role_plan": role_plan},
        ),
        "missing_required_roles_before_llm": _blocking_missing_required_roles(
            _string_list(
                component_candidate_matrix.get("missing_required_roles_before_llm")
                or component_candidate_matrix.get("missing_required_roles")
            ),
            package={**semantic_fields, "role_plan": role_plan},
        ),
        "missing_required_capabilities": _blocking_missing_required_capabilities(
            _mapping_rows(component_candidate_matrix.get("missing_required_capabilities"))
        ),
        "missing_required_capabilities_before_llm": _blocking_missing_required_capabilities(
            _mapping_rows(
                component_candidate_matrix.get("missing_required_capabilities_before_llm")
                or component_candidate_matrix.get("missing_required_capabilities")
            )
        ),
        "role_coverage_summary": _compact_role_coverage_summary(
            component_candidate_matrix.get("role_coverage_summary")
        ),
        "role_plan": _role_plan_for_package(role_plan),
        **semantic_fields,
        "category_plan": category_plan,
        "category_planner_source": category_planner_source,
        "category_plan_source": category_plan_source,
        "category_plan_entries": _mapping_rows(
            component_candidate_matrix.get("category_plan_entries")
        ),
        "category_catalog_summary": _safe_mapping(
            component_candidate_matrix.get("category_catalog_summary")
        ),
        "category_plan_warnings": _string_list(
            component_candidate_matrix.get("category_plan_warnings")
        ),
        "matrix_distiller_used": bool(component_candidate_matrix.get("matrix_distiller_used")),
        "matrix_distiller_source": _text_or_none(
            component_candidate_matrix.get("matrix_distiller_source")
        ),
        "matrix_distiller_diagnostics": _safe_mapping(
            component_candidate_matrix.get("matrix_distiller_diagnostics")
        ),
        "full_matrix_evaluation_used": bool(
            component_candidate_matrix.get("full_matrix_evaluation_used")
        ),
        "full_matrix_evaluation_fallback_reason": _text_or_none(
            component_candidate_matrix.get("full_matrix_evaluation_fallback_reason")
        ),
        "provider_error_type": _text_or_none(
            component_candidate_matrix.get("provider_error_type")
        ),
        "provider_context_limit": _safe_mapping(
            component_candidate_matrix.get("provider_context_limit")
        ),
        "role_chunk_count_by_role": _safe_mapping(
            component_candidate_matrix.get("role_chunk_count_by_role")
        ),
        "evaluated_candidate_count_by_role": _safe_mapping(
            component_candidate_matrix.get("evaluated_candidate_count_by_role")
        ),
        "selected_candidate_count_by_role": _safe_mapping(
            component_candidate_matrix.get("selected_candidate_count_by_role")
        ),
        "role_reducer_summary": _safe_mapping(
            component_candidate_matrix.get("role_reducer_summary")
        ),
        FULL_MATRIX_FAILED_CHUNKS_KEY: _mapping_rows(
            component_candidate_matrix.get(FULL_MATRIX_FAILED_CHUNKS_KEY)
        ),
        "no_recommendation_coverage": _safe_mapping(
            component_candidate_matrix.get("no_recommendation_coverage")
        ),
        "llm_cost_diagnostics": _safe_mapping(
            component_candidate_matrix.get("llm_cost_diagnostics")
        ),
        "broad_count_by_role": _safe_mapping(
            component_candidate_matrix.get("broad_count_by_role")
        ),
        "distilled_count_by_role": _safe_mapping(
            component_candidate_matrix.get("distilled_count_by_role")
        ),
        **_role_lifecycle_package_fields(component_candidate_matrix),
        "component_role_contract": _component_role_contract(
            product_group,
            required_roles=required_roles,
        ),
        "ready_stock_candidates": compact_ready_stock_candidates,
        "rule_based_build_candidates": _compact_rule_based_build_candidates(
            rule_based_build_candidates
        ),
        "component_candidate_matrix": matrix,
        "role_candidate_pools": _role_candidate_pools_for_package(
            matrix=matrix,
            component_candidate_matrix=component_candidate_matrix,
        ),
        "component_matrix_coverage_summary": _compact_component_matrix_coverage_summary(
            component_candidate_matrix.get("component_matrix_coverage_summary")
        ),
        "optimization_mode": OPTIMIZATION_MODE_COST_MINIMAL_FIT,
        "output_mode": normalized_output_mode,
        "proposal_pool_limit": proposal_pool_limit,
        "final_display_limit": final_display_limit,
        "proposal_pool_instruction": proposal_pool_instruction,
    }
    _attach_pre_composer_semantic_diagnostics(package)
    _attach_package_candidate_exposure_fields(
        package,
        source_component_candidate_matrix=component_candidate_matrix,
        required_roles=required_roles,
        include_ready_server=(
            ready_candidates_policy["include_ready_server"] and product_group == "server"
        ),
    )
    if _package_has_unresolved_required_category_plan_roles(package):
        package["package_skipped_reason"] = CATEGORY_PLAN_MISSING_REQUIRED_ROLES_REASON
        package["llm_fallback_reason"] = CATEGORY_PLAN_MISSING_REQUIRED_ROLES_REASON
        package["package_budget_warnings"] = _unique(
            [
                *_string_list(package.get("package_budget_warnings")),
                CATEGORY_PLAN_MISSING_REQUIRED_ROLES_REASON,
            ]
        )
    elif package.get("package_candidate_exposure_incomplete"):
        package["package_skipped_reason"] = INCOMPLETE_MATRIX_EXPOSURE_REASON
        package["llm_fallback_reason"] = INCOMPLETE_MATRIX_EXPOSURE_REASON
        package["package_budget_warnings"] = _unique(
            [
                *_string_list(package.get("package_budget_warnings")),
                INCOMPLETE_MATRIX_EXPOSURE_REASON,
            ]
        )
    if ready_candidates_policy["excluded_reason"]:
        package["ready_candidates_excluded_reason"] = ready_candidates_policy[
            "excluded_reason"
        ]
    if ready_candidates_policy["include_ready_server"]:
        package["ready_candidates_limit"] = READY_SERVER_CANDIDATES_LIMIT
    budget = max_package_chars or LlmSettings().llm_configurator_max_package_chars
    return _package_with_budget(package, max_chars=budget)


def _ready_candidates_package_policy(
    *,
    user_request: str | None,
    product_group: Any,
    category_plan: Mapping[str, Any],
    component_candidate_matrix: Mapping[str, Any],
) -> dict[str, Any]:
    if str(product_group or "").strip() != SERVER_PRODUCT_GROUP:
        return {"include_ready_server": False, "excluded_reason": None}
    if _category_plan_selects_ready_server(category_plan):
        return {"include_ready_server": True, "excluded_reason": None}
    if _request_asks_for_ready_server(user_request):
        return {"include_ready_server": True, "excluded_reason": None}
    if _ai_category_plan_matrix_path(category_plan, component_candidate_matrix):
        return {
            "include_ready_server": False,
            "excluded_reason": READY_CANDIDATES_EXCLUDED_AI_CATEGORY_PLAN_REASON,
        }
    return {"include_ready_server": True, "excluded_reason": None}


def _category_plan_selects_ready_server(category_plan: Mapping[str, Any]) -> bool:
    for role, category_ids in category_plan.items():
        if str(role or "").strip() != READY_SERVER_CANDIDATE_TYPE:
            continue
        if isinstance(category_ids, Sequence) and not isinstance(
            category_ids, (str, bytes)
        ):
            return bool([item for item in category_ids if str(item or "").strip()])
        return bool(str(category_ids or "").strip())
    return False


def _ai_category_plan_matrix_path(
    category_plan: Mapping[str, Any],
    component_candidate_matrix: Mapping[str, Any],
) -> bool:
    return bool(category_plan) and _has_non_ready_package_candidate_rows(
        component_candidate_matrix
    )


def _has_non_ready_package_candidate_rows(
    component_candidate_matrix: Mapping[str, Any],
) -> bool:
    for _, matrix_key, _ in MATRIX_KEYS:
        rows = component_candidate_matrix.get(matrix_key)
        if isinstance(rows, list) and rows:
            return True
    return False


def _attach_pre_composer_semantic_diagnostics(package: dict[str, Any]) -> None:
    package["pre_composer_requirement_classifier_status"] = _text_or_none(
        package.get("requirement_classifier_status")
    )
    package["pre_composer_requirement_source_coverage_percent"] = package.get(
        "requirement_source_coverage_percent"
    )
    package["pre_composer_unclassified_source_fragments"] = _string_list(
        package.get("unclassified_source_fragments")
    )
    package["pre_composer_semantic_diagnostics_are_blocking"] = False


def _request_asks_for_ready_server(user_request: str | None) -> bool:
    text = str(user_request or "").casefold()
    if not text:
        return False
    return bool(
        re.search(
            r"\b(?:ready|prebuilt|assembled|complete|turnkey)\s+servers?\b|"
            r"\bservers?\s+(?:in\s+assembly|assembled|prebuilt|ready)\b|"
            r"\u0433\u043e\u0442\u043e\u0432\w*\s+"
            r"\u0441\u0435\u0440\u0432\u0435\u0440\w*|"
            r"\u0441\u0435\u0440\u0432\u0435\u0440\w*\s+"
            r"\u0432\s+\u0441\u0431\u043e\u0440\u0435|"
            r"\u0441\u0431\u043e\u0440\w*\s+"
            r"\u0441\u0435\u0440\u0432\u0435\u0440\w*",
            text,
            re.I,
        )
    )


def _skipped_llm_configurator_package(
    *,
    user_request: str | None,
    product_group: Any,
    package_requirements: Mapping[str, Any],
    required_capabilities: Sequence[Mapping[str, Any]],
    optional_capabilities: Sequence[Mapping[str, Any]],
    required_roles: Sequence[str],
    role_plan: Mapping[str, Any],
    semantic_fields: Mapping[str, Any],
    category_plan: Mapping[str, Any],
    component_candidate_matrix: Mapping[str, Any],
    category_planner_source: str | None,
    category_plan_source: str | None,
    package_skipped_reason: str,
    output_mode: str,
    proposal_pool_limit: int,
    final_display_limit: int,
    proposal_pool_instruction: str,
    max_package_chars: int | None,
) -> dict[str, Any]:
    original_request_text = (user_request or "").strip()
    package = {
        "original_request_text": original_request_text,
        "user_request": original_request_text,
        "product_group": product_group or "unknown",
        "normalized_requirements": dict(package_requirements),
        "required_capabilities": [dict(row) for row in required_capabilities],
        "optional_capabilities": [dict(row) for row in optional_capabilities],
        "unsupported_or_unmapped_requirements": _string_list(
            role_plan.get("unsupported_or_unmapped_requirements")
            or package_requirements.get("unsupported_or_unmapped_requirements")
        ),
        "required_roles": _string_list(list(required_roles)),
        "missing_required_roles": [],
        "missing_required_roles_before_llm": _blocking_missing_required_roles(
            _string_list(
                component_candidate_matrix.get("missing_required_roles_before_llm")
                or component_candidate_matrix.get("missing_required_roles")
            ),
            package={**dict(semantic_fields), "role_plan": role_plan},
        ),
        "missing_required_capabilities": _blocking_missing_required_capabilities(
            _mapping_rows(component_candidate_matrix.get("missing_required_capabilities"))
        ),
        "missing_required_capabilities_before_llm": _blocking_missing_required_capabilities(
            _mapping_rows(
                component_candidate_matrix.get("missing_required_capabilities_before_llm")
                or component_candidate_matrix.get("missing_required_capabilities")
            )
        ),
        "role_coverage_summary": _compact_role_coverage_summary(
            component_candidate_matrix.get("role_coverage_summary")
        ),
        "role_plan": _role_plan_for_package(role_plan),
        **dict(semantic_fields),
        "category_plan": dict(category_plan),
        "category_planner_source": category_planner_source,
        "category_plan_source": category_plan_source,
        "category_plan_entries": _mapping_rows(
            component_candidate_matrix.get("category_plan_entries")
        ),
        "category_catalog_summary": _safe_mapping(
            component_candidate_matrix.get("category_catalog_summary")
        ),
        "category_plan_warnings": _string_list(
            component_candidate_matrix.get("category_plan_warnings")
        ),
        "matrix_distiller_used": bool(component_candidate_matrix.get("matrix_distiller_used")),
        "matrix_distiller_source": _text_or_none(
            component_candidate_matrix.get("matrix_distiller_source")
        ),
        "matrix_distiller_diagnostics": _safe_mapping(
            component_candidate_matrix.get("matrix_distiller_diagnostics")
        ),
        "full_matrix_evaluation_used": bool(
            component_candidate_matrix.get("full_matrix_evaluation_used")
        ),
        "full_matrix_evaluation_fallback_reason": _text_or_none(
            component_candidate_matrix.get("full_matrix_evaluation_fallback_reason")
        ),
        "provider_error_type": _text_or_none(
            component_candidate_matrix.get("provider_error_type")
        ),
        "provider_context_limit": _safe_mapping(
            component_candidate_matrix.get("provider_context_limit")
        ),
        "role_chunk_count_by_role": _safe_mapping(
            component_candidate_matrix.get("role_chunk_count_by_role")
        ),
        "evaluated_candidate_count_by_role": _safe_mapping(
            component_candidate_matrix.get("evaluated_candidate_count_by_role")
        ),
        "selected_candidate_count_by_role": _safe_mapping(
            component_candidate_matrix.get("selected_candidate_count_by_role")
        ),
        "role_reducer_summary": _safe_mapping(
            component_candidate_matrix.get("role_reducer_summary")
        ),
        FULL_MATRIX_FAILED_CHUNKS_KEY: _mapping_rows(
            component_candidate_matrix.get(FULL_MATRIX_FAILED_CHUNKS_KEY)
        ),
        "no_recommendation_coverage": _safe_mapping(
            component_candidate_matrix.get("no_recommendation_coverage")
        ),
        "llm_cost_diagnostics": _safe_mapping(
            component_candidate_matrix.get("llm_cost_diagnostics")
        ),
        "broad_count_by_role": _safe_mapping(
            component_candidate_matrix.get("broad_count_by_role")
        ),
        "distilled_count_by_role": _safe_mapping(
            component_candidate_matrix.get("distilled_count_by_role")
        ),
        **_role_lifecycle_package_fields(component_candidate_matrix),
        "component_role_contract": _component_role_contract(
            product_group,
            required_roles=required_roles,
        ),
        "ready_stock_candidates": [],
        "rule_based_build_candidates": [],
        "component_candidate_matrix": {},
        "role_candidate_pools": _role_candidate_pools_for_package(
            matrix={},
            component_candidate_matrix=component_candidate_matrix,
        ),
        "component_matrix_coverage_summary": _compact_component_matrix_coverage_summary(
            component_candidate_matrix.get("component_matrix_coverage_summary")
        ),
        "optimization_mode": OPTIMIZATION_MODE_COST_MINIMAL_FIT,
        "output_mode": output_mode,
        "proposal_pool_limit": proposal_pool_limit,
        "final_display_limit": final_display_limit,
        "proposal_pool_instruction": proposal_pool_instruction,
        "package_skipped_reason": package_skipped_reason,
        "llm_fallback_reason": package_skipped_reason,
        "package_budget_warnings": [f"llm_configurator_package_skipped:{package_skipped_reason}"],
    }
    _attach_pre_composer_semantic_diagnostics(package)
    _attach_package_candidate_exposure_fields(
        package,
        source_component_candidate_matrix=component_candidate_matrix,
        required_roles=required_roles,
        include_ready_server=False,
        package_skipped_reason=package_skipped_reason,
    )
    budget = max_package_chars or LlmSettings().llm_configurator_max_package_chars
    return _package_with_budget(package, max_chars=budget)


def _semantic_package_skipped_reason(
    *,
    user_request: str | None,
    product_group: Any,
    semantic_fields: Mapping[str, Any],
    required_capabilities: Sequence[Mapping[str, Any]],
    category_plan: Mapping[str, Any],
    role_coverage_summary: Mapping[str, Any],
    has_candidate_matrix_rows: bool,
) -> str | None:
    source = str(semantic_fields.get("semantic_planner_source") or "").strip()
    fallback_reason = str(
        semantic_fields.get("semantic_planner_fallback_reason") or ""
    ).strip()
    if source == SEMANTIC_COMPLEX_FALLBACK_REASON:
        return SEMANTIC_COMPLEX_FALLBACK_REASON
    if fallback_reason == SEMANTIC_COMPLEX_FALLBACK_REASON:
        return SEMANTIC_COMPLEX_FALLBACK_REASON
    if source == SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT:
        return SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
    if (
        fallback_reason == SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
        and str(product_group or "").strip() == "unknown"
    ):
        return SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
    if category_plan and not has_candidate_matrix_rows:
        if _role_coverage_has_selected_categories(role_coverage_summary):
            return "matrix_empty_after_category_plan"
        return "category_plan_materialization_failed"
    if required_capabilities or category_plan or has_candidate_matrix_rows:
        return None
    if source and source != "llm" and _looks_like_technical_request(user_request):
        return fallback_reason or source
    if _looks_like_technical_request(user_request) and str(product_group or "") == "unknown":
        return SEMANTIC_PACKAGE_SKIPPED_PLANNING_UNAVAILABLE
    return None


def _role_coverage_has_selected_categories(value: Mapping[str, Any]) -> bool:
    for row in value.values():
        if not isinstance(row, Mapping):
            continue
        if _string_list(row.get("category_ids")):
            return True
    return False


def _matrix_skip_reason_should_be_rechecked(
    reason: str | None,
    component_candidate_matrix: Mapping[str, Any],
) -> bool:
    if not reason or not _has_package_candidate_rows(component_candidate_matrix):
        return False
    if reason == DISTILLER_OVER_BUDGET_SKIP_REASON:
        return True
    if reason == DISTILLER_FAILED_SKIP_REASON:
        return _matrix_has_distiller_fallback_context(component_candidate_matrix)
    return False


def _component_matrix_has_unresolved_required_category_plan_roles(
    component_candidate_matrix: Mapping[str, Any],
) -> bool:
    if _string_list(
        component_candidate_matrix.get("category_planner_unresolved_required_roles")
    ):
        return True
    reason_by_role = _safe_mapping(
        component_candidate_matrix.get("roles_dropped_reason_by_role")
    )
    for role in _string_list(
        component_candidate_matrix.get("category_planner_missing_required_roles")
        or component_candidate_matrix.get("missing_category_roles")
    ):
        if reason_by_role.get(role) == "missing_category":
            return True
    return False


def _package_has_unresolved_required_category_plan_roles(
    package: Mapping[str, Any],
) -> bool:
    unresolved_roles = _string_list(
        package.get("category_planner_unresolved_required_roles")
    )
    if any(
        not _ready_server_satisfies_lifecycle_role(package, role)
        for role in unresolved_roles
    ):
        return True
    reason_by_role = _safe_mapping(package.get("roles_dropped_reason_by_role"))
    coverage_by_role = _safe_mapping(package.get("role_coverage_summary"))
    for role in _string_list(package.get("package_exposure_blocking_lifecycle_roles")):
        coverage = _safe_mapping(coverage_by_role.get(role))
        if coverage.get("missing_category") or reason_by_role.get(role) == "missing_category":
            return True
    return False


def _matrix_has_distiller_fallback_context(
    component_candidate_matrix: Mapping[str, Any],
) -> bool:
    source = str(component_candidate_matrix.get("matrix_distiller_source") or "").strip()
    diagnostics = _safe_mapping(
        component_candidate_matrix.get("matrix_distiller_diagnostics")
    )
    return (
        source in DISTILLER_FALLBACK_PACKAGE_SOURCES
        or bool(diagnostics.get("fallback_compaction_attempted"))
        or bool(diagnostics.get("package_budget_after_fallback"))
    )


def _has_package_candidate_rows(component_candidate_matrix: Mapping[str, Any]) -> bool:
    for key, value in component_candidate_matrix.items():
        if not str(key).endswith("_candidates"):
            continue
        if isinstance(value, list) and value:
            return True
    return False


def _looks_like_technical_request(text: Any) -> bool:
    value = str(text or "").casefold()
    return bool(
        re.search(
            r"\b(?:server|switch|router|firewall|cpu|xeon|epyc|ram|rdimm|ddr[345]|"
            r"ssd|hdd|nvme|sata|sas|raid|hba|nic|sfp\+?|qsfp|10gbe|25gbe|"
            r"c13|c14|psu|storage|nas|san)\b|"
            r"СЃРµСЂРІРµСЂ|РєРѕРјРјСѓС‚Р°С‚РѕСЂ|СЃС…Рґ|РїСЂРѕС†РµСЃСЃРѕСЂ|"
            r"РѕРїРµСЂР°С‚РёРІ|РґРёСЃРє|СЃРµС‚РµРІ|РїРёС‚Р°РЅ",
            value,
            re.I,
        )
    )


def _no_recommendation_outcome_from_package_missing(
    *,
    output_mode: str,
    package: Mapping[str, Any],
    missing_required_capabilities: Sequence[Mapping[str, Any]],
    fallback_reason: str | None = None,
) -> LlmConfiguratorOutcome:
    fallback_reason = fallback_reason or (
        "missing_required_roles_before_llm"
        if _string_list(
            package.get("missing_required_roles_before_llm")
            or package.get("missing_required_roles")
        )
        else "hard_capability_coverage_missing_before_llm"
    )
    no_recommendation_reason = _no_recommendation_reason(
        fallback_reason=fallback_reason,
        validated_pool=_empty_validated_recommendation_pool(),
        warnings=[],
        product_group=str(package.get("product_group") or "server"),
        package_required_capabilities=_mapping_rows(package.get("required_capabilities")),
        package_missing_required_capabilities=missing_required_capabilities,
    )
    return LlmConfiguratorOutcome(
        enabled=True,
        output_mode=output_mode,
        primary_recommendation_status="no_recommendation",
        no_recommendation_reason=no_recommendation_reason,
        commercial_summary=_commercial_summary_from_no_recommendation(
            no_recommendation_reason
        ),
        fallback_reason=fallback_reason,
    )


def _blocking_missing_required_roles(
    roles: Sequence[str],
    *,
    package: Mapping[str, Any],
) -> list[str]:
    blocking_unmapped = bool(_mapping_rows(package.get("unmapped_requirements_blocking")))
    result: list[str] = []
    for role in _string_list(list(roles)):
        if role == UNMAPPED_ROLE and not blocking_unmapped:
            continue
        result.append(role)
    return _unique(result)


def _blocking_missing_required_capabilities(
    capabilities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for capability in capabilities:
        classification = str(
            capability.get("requirement_classification")
            or capability.get("classification")
            or ""
        ).strip()
        fulfillment_mode = str(capability.get("fulfillment_mode") or "").strip()
        if fulfillment_mode in {
            FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
            FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
            FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
            FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
            FULFILLMENT_LOGISTICS_CONSTRAINT,
            FULFILLMENT_ENGINEERING_CHECK_ONLY,
            FULFILLMENT_NOT_APPLICABLE,
        }:
            continue
        if classification in {
            REQ_CLASS_PRIMARY_OBJECT_FEATURE,
            REQ_CLASS_ENGINEERING_CHECK,
            REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
            REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
        }:
            continue
        role = str(capability.get("role") or "").strip()
        if role == UNMAPPED_ROLE and classification != (
            REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE
        ):
            continue
        result.append(dict(capability))
    return result


def _no_recommendation_outcome_from_package_skipped(
    *,
    output_mode: str,
    package: Mapping[str, Any],
) -> LlmConfiguratorOutcome:
    reason_code = str(
        package.get("package_skipped_reason")
        or package.get("llm_fallback_reason")
        or SEMANTIC_PACKAGE_SKIPPED_PLANNING_UNAVAILABLE
    )
    summary = (
        "Planning unavailable for this technical request; normal LLM configurator package "
        "was skipped."
    )
    if reason_code in {
        "matrix_distiller_failed",
        "package_over_budget_after_distillation",
        "package_over_budget_after_full_matrix_failure",
        "full_matrix_evaluation_failed",
    }:
        summary = (
            "Не удалось безопасно сузить серверную матрицу компонентов до размера "
            "Composer package."
        )
    if reason_code == INCOMPLETE_MATRIX_EXPOSURE_REASON:
        summary = (
            "Composer package did not receive complete candidate-matrix exposure; "
            "recommendation is blocked until the full matrix or complete AI-reviewed "
            "role pools are available."
        )
    if reason_code in {
        CATEGORY_PLAN_MISSING_REQUIRED_ROLES_REASON,
        MISSING_REQUIRED_CATEGORY_BEFORE_COMPOSER_REASON,
    }:
        summary = (
            "Category plan is missing a required lifecycle role; Composer is blocked "
            "until the category planner selects a real distributor category or returns "
            "an explicit no-category diagnostic."
        )
    reason = {
        "product_group": package.get("product_group") or "unknown",
        "reason_code": reason_code,
        "summary": summary,
        "package_candidate_exposure_policy": _safe_mapping(
            package.get("package_candidate_exposure_policy")
        ),
        "package_candidate_exposure_incomplete_roles": _string_list(
            package.get("package_candidate_exposure_incomplete_roles")
        ),
        "semantic_planner_source": package.get("semantic_planner_source"),
        "semantic_planner_error_type": package.get("semantic_planner_error_type"),
        "semantic_planner_http_status": package.get("semantic_planner_http_status"),
        "semantic_planner_parse_status": package.get("semantic_planner_parse_status"),
        "semantic_planner_fallback_reason": package.get(
            "semantic_planner_fallback_reason"
        ),
    }
    return LlmConfiguratorOutcome(
        enabled=True,
        output_mode=output_mode,
        primary_recommendation_status="no_recommendation",
        no_recommendation_reason=reason,
        commercial_summary={
            "mode": output_mode,
            "status": "no_recommendation",
            "title": summary,
            "reasons": [summary],
            "copy_paste_text": summary,
            "lines": [summary],
        },
        fallback_reason=reason_code,
        internal_warnings=_string_list(package.get("package_budget_warnings")),
    )


def _component_role_contract(
    product_group: Any,
    *,
    required_roles: Sequence[str],
) -> dict[str, Any]:
    normalized = str(product_group or SERVER_PRODUCT_GROUP).strip() or SERVER_PRODUCT_GROUP
    required = _unique([role for role in _string_list(required_roles) if role])
    if normalized == NETWORK_PRODUCT_GROUP:
        return {
            "product_group": NETWORK_PRODUCT_GROUP,
            "allowed_component_role_keys": list(NETWORK_COMPOSER_ROLE_KEYS),
            "base_device_roles": sorted(NETWORK_BASE_DEVICE_ROLES),
            "required_role_keys": required,
            "generic_aliases_not_for_output": sorted(GENERIC_COMPONENT_ROLE_ALIASES),
            "primary_role_key_guidance": (
                "Use switch/router/firewall/access_point exactly as they appear in "
                "component_candidate_matrix. For a switch quote, use "
                "component_candidate_ids.switch, not platform."
            ),
            "forbidden_role_guidance": (
                "Do not use server roles such as platform/cpu/ram/storage for "
                "network recommendations."
            ),
        }
    if normalized == STORAGE_PRODUCT_GROUP:
        return {
            "product_group": STORAGE_PRODUCT_GROUP,
            "allowed_component_role_keys": list(STORAGE_COMPOSER_ROLE_KEYS),
            "base_device_roles": [STORAGE_SYSTEM_ROLE],
            "required_role_keys": required,
            "generic_aliases_not_for_output": sorted(GENERIC_COMPONENT_ROLE_ALIASES),
            "primary_role_key_guidance": (
                "Use component_candidate_ids.storage_system for the base storage "
                "system. Do not use platform for storage_system."
            ),
            "forbidden_role_guidance": (
                "Do not use server CPU/RAM/platform roles for storage recommendations."
            ),
        }
    return {
        "product_group": SERVER_PRODUCT_GROUP,
        "allowed_component_role_keys": list(SERVER_COMPOSER_ROLE_KEYS),
        "base_device_roles": [SERVER_PLATFORM_ROLE, "platform"],
        "required_role_keys": required,
        "generic_aliases_not_for_output": [],
        "primary_role_key_guidance": (
            "For server recommendations, platform is the legacy output key for "
            "server_platform and remains accepted."
        ),
        "forbidden_role_guidance": "",
    }


def _has_ready_server_candidate(candidates: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        str(candidate.get("candidate_type") or READY_SERVER_CANDIDATE_TYPE)
        == READY_SERVER_CANDIDATE_TYPE
        for candidate in candidates
    )


def _required_roles_without_package_candidates(package: Mapping[str, Any]) -> list[str]:
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    missing: list[str] = []
    for role in _string_list(package.get("required_roles")):
        prompt_role = PROMPT_ROLE_BY_INTERNAL_ROLE.get(role, role)
        candidates = matrix.get(prompt_role)
        if (
            not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or not candidates
        ):
            missing.append(role)
    return _unique(missing)


def _empty_validated_recommendation_pool() -> _ValidatedRecommendationPool:
    return _ValidatedRecommendationPool(
        recommendations=[],
        accepted_recommendations=[],
        configuration_groups=[],
        quote_recommendation={},
        selected_configuration_group_id=None,
        selected_platform_option_id=None,
        selected_platform_option_index=None,
        warnings=[],
        proposal_count=0,
        valid_count=0,
        validation_rejected_count=0,
        selection_skipped_count=0,
        rejected_count=0,
        validation_summary={},
        rejected_reasons_top=[],
        rejected_debug_safe=[],
    )


def _composer_output_validation_fields(
    response: LlmComposerResponsePayload,
    validated_pool: _ValidatedRecommendationPool,
) -> dict[str, Any]:
    requirement_analysis = _safe_mapping(response.requirement_analysis)
    fulfillment_decisions = _mapping_rows(
        requirement_analysis.get("fulfillment_decisions")
    ) or _mapping_rows(response.fulfillment_decisions)
    coverage_summary = _safe_mapping(response.requirement_coverage_summary)
    if not coverage_summary and response.source_fragments_covered:
        coverage_summary = {
            "source_fragments_covered": _mapping_rows(response.source_fragments_covered)
        }
    return {
        "composer_requirement_analysis": requirement_analysis,
        "composer_fulfillment_decisions": fulfillment_decisions,
        "composer_selected_components": _mapping_rows(response.selected_components),
        "composer_quantities": _safe_mapping(response.quantities),
        "composer_assumptions": _string_list(response.assumptions),
        "composer_engineer_checks": _string_list(response.engineer_checks),
        "composer_hard_mismatch_risks": _mapping_rows(response.hard_mismatch_risks),
        "composer_unverified_requirements": _mapping_rows(
            response.unverified_requirements
        ),
        "composer_considered_candidate_count_by_role": _safe_mapping(
            response.considered_candidate_count_by_role
        ),
        "composer_chosen_candidate_ids": _string_list(response.chosen_candidate_ids),
        "composer_source_coverage_summary": coverage_summary,
        "validation_hard_mismatches": _validation_rows_by_status(
            validated_pool,
            statuses={"hard_mismatch"},
        ),
        "validation_unverified_requirements": _validation_rows_by_status(
            validated_pool,
            statuses={"unverified_hard_requirement"},
        ),
    }


def _validation_rows_by_status(
    validated_pool: _ValidatedRecommendationPool,
    *,
    statuses: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources: list[Mapping[str, Any]] = [
        *validated_pool.recommendations,
        *validated_pool.accepted_recommendations,
        *validated_pool.rejected_debug_safe,
    ]
    for source in sources:
        for row in _mapping_rows(source.get("hard_capability_validation")):
            if str(row.get("status") or "").strip() in statuses:
                rows.append(dict(row))
        for key in ("validation_hard_mismatches", "validation_unverified_requirements"):
            for row in _mapping_rows(source.get(key)):
                if str(row.get("status") or "").strip() in statuses:
                    rows.append(dict(row))
    return _unique_mapping_rows(rows)


def _unique_mapping_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def _with_package_diagnostics(
    outcome: LlmConfiguratorOutcome,
    package: Mapping[str, Any],
) -> LlmConfiguratorOutcome:
    diagnostics = _package_report_diagnostics(package)
    parse_diagnostics = _safe_mapping(outcome.parse_diagnostics)
    for key in COMPACT_PACKAGE_DIAGNOSTIC_KEYS:
        if key in parse_diagnostics:
            diagnostics[key] = parse_diagnostics[key]
    for key in MULTI_PASS_DIAGNOSTIC_KEYS:
        if key in parse_diagnostics:
            diagnostics[key] = parse_diagnostics[key]
    if (
        outcome.fallback_reason == PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON
        or parse_diagnostics.get("provider_error_type")
        == PROVIDER_CONTEXT_LIMIT_ERROR_TYPE
    ):
        diagnostics["provider_error_type"] = PROVIDER_CONTEXT_LIMIT_ERROR_TYPE
        diagnostics["provider_context_limit"] = parse_diagnostics
    if outcome.composer_attempt_decision:
        diagnostics["composer_attempt_decision"] = outcome.composer_attempt_decision
    if outcome.no_recommendation_reason.get("no_recommendation_coverage"):
        diagnostics["no_recommendation_coverage"] = _safe_mapping(
            outcome.no_recommendation_reason.get("no_recommendation_coverage")
        )
    for key in NO_RECOMMENDATION_COVERAGE_DIAGNOSTIC_KEYS:
        if key in outcome.no_recommendation_reason:
            diagnostics[key] = outcome.no_recommendation_reason[key]
    evidence_diagnostics = _safe_mapping(
        _safe_mapping(outcome.evidence_pack).get("diagnostics")
    )
    for key in COMPACT_PACKAGE_DIAGNOSTIC_KEYS:
        if key in evidence_diagnostics:
            diagnostics[key] = evidence_diagnostics[key]
    for key in MULTI_PASS_DIAGNOSTIC_KEYS:
        if key in evidence_diagnostics:
            diagnostics[key] = evidence_diagnostics[key]
    for key in NO_RECOMMENDATION_COVERAGE_DIAGNOSTIC_KEYS:
        if key in evidence_diagnostics:
            diagnostics[key] = evidence_diagnostics[key]
    if FULL_MATRIX_FAILED_CHUNKS_KEY in evidence_diagnostics:
        diagnostics[FULL_MATRIX_FAILED_CHUNKS_KEY] = evidence_diagnostics[
            FULL_MATRIX_FAILED_CHUNKS_KEY
        ]
    if outcome.composer_requirement_analysis:
        diagnostics["composer_requirement_analysis"] = _safe_mapping(
            outcome.composer_requirement_analysis
        )
    if outcome.composer_fulfillment_decisions:
        diagnostics["composer_fulfillment_decisions"] = _mapping_rows(
            outcome.composer_fulfillment_decisions
        )
    if outcome.composer_source_coverage_summary:
        diagnostics["composer_source_coverage_summary"] = _safe_mapping(
            outcome.composer_source_coverage_summary
        )
    if outcome.validation_hard_mismatches:
        diagnostics["validation_hard_mismatches"] = _mapping_rows(
            outcome.validation_hard_mismatches
        )
    if outcome.validation_unverified_requirements:
        diagnostics["validation_unverified_requirements"] = _mapping_rows(
            outcome.validation_unverified_requirements
        )
    final_status_source = outcome.final_status_source or _infer_final_status_source(
        outcome,
        package=package,
    )
    if final_status_source:
        diagnostics["final_status_source"] = final_status_source
    diagnostics["bom_critic_used"] = bool(
        diagnostics.get("bom_critic_used")
        or outcome.repair_attempted
        or outcome.repair_used
        or diagnostics.get("no_recommendation_coverage_repair_attempted")
        or _safe_mapping(diagnostics.get("no_recommendation_coverage")).get(
            "coverage_incomplete"
        )
    )
    return replace(
        outcome,
        package_diagnostics=diagnostics,
        final_status_source=final_status_source,
    )


def _infer_final_status_source(
    outcome: LlmConfiguratorOutcome,
    *,
    package: Mapping[str, Any],
) -> str | None:
    if outcome.primary_recommendation_status == "valid":
        return "composer_validated"
    package_skipped_reason = _text_or_none(package.get("package_skipped_reason"))
    attempt_decision = _safe_mapping(outcome.composer_attempt_decision)
    composer_not_attempted = (
        bool(attempt_decision)
        and not bool(attempt_decision.get("should_attempt"))
        and not bool(outcome.used)
    )
    if (
        _package_over_budget(package)
        or "package_over_budget" in str(outcome.fallback_reason or "")
        or package_skipped_reason
        in {
            "package_over_budget_before_composer",
            "package_over_budget_after_distillation",
            "package_over_budget_after_full_matrix_failure",
            "full_matrix_evaluation_timeout_package_over_budget",
        }
    ):
        return "package_over_budget"
    if package_skipped_reason in {
        SEMANTIC_PACKAGE_SKIPPED_PLANNING_UNAVAILABLE,
        SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON,
        SEMANTIC_COMPLEX_FALLBACK_REASON,
    } or str(package.get("product_group") or "").strip() == "unknown":
        return "planner_unavailable"
    if package_skipped_reason in {
        CATEGORY_PLAN_MISSING_REQUIRED_ROLES_REASON,
        MISSING_REQUIRED_CATEGORY_BEFORE_COMPOSER_REASON,
    } or _package_has_unresolved_required_category_plan_roles(package):
        return "category_plan_incomplete"
    if package_skipped_reason == INCOMPLETE_MATRIX_EXPOSURE_REASON or bool(
        package.get("package_candidate_exposure_incomplete")
    ):
        return "matrix_exposure_incomplete"
    if composer_not_attempted:
        blocked_by = _string_list(attempt_decision.get("blocked_by"))
        if blocked_by:
            return "composer_not_attempted"
    if outcome.primary_recommendation_status == "no_recommendation":
        if outcome.validation_rejected_count or outcome.rejected_recommendations_count:
            return "composer_rejected"
        if (
            outcome.fallback_reason == "llm_configurator_structured_no_recommendation"
            or outcome.fallback_reason == COMPOSER_STRUCTURED_NO_RECOMMENDATION
            or outcome.no_recommendation_reason.get("structured_no_recommendation")
        ):
            return "composer_no_recommendation"
        return "composer_no_recommendation"
    return None


def _composer_not_attempted_reason(reason: str) -> str:
    normalized = str(reason or "unknown_guard").strip() or "unknown_guard"
    return f"llm_configurator_not_attempted:{normalized}"


def _composer_attempt_decision_without_package(
    *,
    enabled: bool,
    output_mode: str,
    provider_configured: bool,
    blocked_by: Sequence[str],
) -> dict[str, Any]:
    blocked = _unique([str(reason) for reason in blocked_by if str(reason).strip()])
    return {
        "enabled": bool(enabled),
        "package_present": False,
        "package_over_budget": False,
        "package_skipped_reason": None,
        "candidate_count_total": 0,
        "required_roles": [],
        "output_mode": _normalize_output_mode(output_mode),
        "provider_configured": bool(provider_configured),
        "should_attempt": False,
        "blocked_by": blocked or ["package_missing"],
    }


def _semantic_precomposer_blockers_are_diagnostics(package: Mapping[str, Any]) -> bool:
    if str(package.get("product_group") or "").strip() in {"", "unknown"}:
        return False
    if _package_over_budget(package):
        return False
    if _text_or_none(package.get("package_skipped_reason")) is not None:
        return False
    if bool(package.get("package_candidate_exposure_incomplete")):
        return False
    if _composer_candidate_count_total(package) <= 0:
        return False
    exposure_policy = _safe_mapping(package.get("package_candidate_exposure_policy"))
    if str(exposure_policy.get("mode") or "").strip() != FULL_BROAD_MATRIX_EXPOSURE_MODE:
        return False
    return bool(
        _safe_mapping(package.get("category_plan"))
        or _safe_mapping(package.get("broad_matrix_count_by_role"))
        or _safe_mapping(package.get("component_candidate_matrix"))
    )


def _objective_precomposer_missing_after_semantic_downgrade(
    package: Mapping[str, Any],
    *,
    missing_roles: Sequence[str],
    missing_capabilities: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    candidate_counts = _package_component_candidate_count_by_role(package)
    required_roles = {
        _coverage_role_key(role)
        for role in _string_list(package.get("required_roles"))
        if _coverage_role_key(role)
    }
    blocking_roles = [
        role_key
        for role_key in (
            _coverage_role_key(role) for role in _string_list(missing_roles)
        )
        if role_key
        and role_key in required_roles
        and int(candidate_counts.get(role_key, 0)) <= 0
    ]
    blocking_role_set = set(blocking_roles)
    blocking_capabilities: list[dict[str, Any]] = []
    for capability in _mapping_rows(missing_capabilities):
        role_key = _coverage_role_key(str(capability.get("role") or ""))
        if not role_key:
            continue
        if int(candidate_counts.get(role_key, 0)) > 0:
            continue
        status = str(capability.get("status") or "").strip()
        if (
            status in {"missing_candidates", "missing_category"}
            or role_key in required_roles
            or role_key in blocking_role_set
        ):
            blocking_capabilities.append(dict(capability))
    for capability in _role_coverage_missing_required_capabilities(package):
        role_key = _coverage_role_key(str(capability.get("role") or ""))
        if not role_key:
            continue
        if int(candidate_counts.get(role_key, 0)) > 0:
            continue
        blocking_role_set.add(role_key)
        blocking_capabilities.append(dict(capability))
    return _unique(blocking_roles), _unique_mapping_rows(blocking_capabilities)


def _role_coverage_missing_required_capabilities(
    package: Mapping[str, Any],
) -> list[dict[str, Any]]:
    coverage_by_role = _safe_mapping(package.get("role_coverage_summary"))
    if not coverage_by_role:
        return []
    missing: list[dict[str, Any]] = []
    for capability in _blocking_missing_required_capabilities(
        _mapping_rows(package.get("required_capabilities"))
    ):
        role_key = _coverage_role_key(str(capability.get("role") or ""))
        coverage = _safe_mapping(coverage_by_role.get(role_key))
        if not bool(coverage.get("missing")):
            continue
        if int(coverage.get("platform_satisfied_candidates_count") or 0) > 0:
            continue
        status = "missing"
        if coverage.get("missing_category"):
            status = "missing_category"
        elif coverage.get("missing_candidates"):
            status = "missing_candidates"
        missing.append(
            {
                **dict(capability),
                "role": role_key,
                "status": status,
                "satisfied_by": None,
                "component_role": None,
                "component_candidate_id": None,
            }
        )
    return _unique_mapping_rows(missing)


def _composer_attempt_decision(
    *,
    package: Mapping[str, Any],
    settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings,
    llm_client: LlmClient | None,
    output_mode: str,
    enabled: bool,
) -> dict[str, Any]:
    package_skipped_reason = _text_or_none(package.get("package_skipped_reason"))
    package_over_budget = (
        _package_over_budget(package)
        or package.get("llm_fallback_reason") == PACKAGE_OVER_BUDGET_FALLBACK_REASON
    )
    provider_configured = _composer_provider_configured(
        settings=settings,
        web_evidence_settings=web_evidence_settings,
        llm_client=llm_client,
    )
    candidate_count_total = _composer_candidate_count_total(package)
    candidate_count_by_role = _safe_mapping(
        package.get("composer_package_candidate_count_by_role")
    ) or _package_component_candidate_count_by_role(package)
    product_group = str(package.get("product_group") or "").strip()
    category_plan = _safe_mapping(package.get("category_plan"))
    has_broad_matrix = bool(
        _safe_mapping(package.get("broad_matrix_count_by_role"))
        or _safe_mapping(package.get("component_candidate_matrix"))
    )
    blocking_requirements = _composer_blocking_requirements(package)
    feature_requirements = _mapping_rows(
        package.get("primary_object_feature_requirements")
    )
    unmapped_non_blocking = _mapping_rows(
        package.get("unmapped_requirements_non_blocking")
    )
    blocking_unmapped = _mapping_rows(package.get("unmapped_requirements_blocking"))
    exposure_incomplete = bool(package.get("package_candidate_exposure_incomplete"))
    blocked_by: list[str] = []
    if not enabled:
        blocked_by.append("llm_configurator_disabled")
    if not package:
        blocked_by.append("package_missing")
    if product_group in {"", "unknown"}:
        blocked_by.append("product_group_unknown")
    if package_over_budget:
        blocked_by.append("package_over_budget")
    if package_skipped_reason and not package_over_budget:
        blocked_by.append(f"package_skipped:{package_skipped_reason}")
    if _package_has_unresolved_required_category_plan_roles(package):
        blocked_by.append(CATEGORY_PLAN_MISSING_REQUIRED_ROLES_REASON)
    if package and not category_plan and not has_broad_matrix:
        blocked_by.append("no_category_plan_or_broad_matrix")
    if exposure_incomplete:
        blocked_by.append(INCOMPLETE_MATRIX_EXPOSURE_REASON)
    if candidate_count_total <= 0 and not package_skipped_reason and not package_over_budget:
        blocked_by.append("no_selectable_component_candidates")
    if not provider_configured:
        blocked_by.append("provider_not_configured")
    return {
        "enabled": bool(enabled),
        "package_present": bool(package),
        "package_over_budget": bool(package_over_budget),
        "package_skipped_reason": package_skipped_reason,
        "candidate_count_total": candidate_count_total,
        "candidate_count_by_role": candidate_count_by_role,
        "required_roles": _string_list(package.get("required_roles")),
        "blocking_requirements": blocking_requirements,
        "non_blocking_feature_requirements_count": len(feature_requirements),
        "unmapped_non_blocking_count": len(unmapped_non_blocking),
        "blocking_unmapped_count": len(blocking_unmapped),
        "package_candidate_exposure_incomplete": exposure_incomplete,
        "package_candidate_exposure_policy": _safe_mapping(
            package.get("package_candidate_exposure_policy")
        ),
        "output_mode": _normalize_output_mode(output_mode),
        "provider_configured": bool(provider_configured),
        "should_attempt": not blocked_by,
        "blocked_by": _unique(blocked_by),
    }


def _composer_attempt_decision_blocked(
    decision: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    blocked_by = _unique(
        [
            *_string_list(decision.get("blocked_by")),
            str(reason or "unknown_guard").strip() or "unknown_guard",
        ]
    )
    return {
        **dict(decision),
        "should_attempt": False,
        "blocked_by": blocked_by,
    }


def _composer_blocking_requirements(package: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in _mapping_rows(package.get("classified_requirements")):
        if not _truthy(row.get("should_block_before_composer")):
            continue
        result.append(
            {
                "requirement_id": row.get("requirement_id"),
                "source_text": row.get("source_text"),
                "classification": row.get("classification"),
                "target_role": row.get("target_role"),
                "category_needed": row.get("category_needed"),
                "reason": row.get("reason"),
            }
        )
    return result


def _composer_provider_configured(
    *,
    settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings,
    llm_client: LlmClient | None,
) -> bool:
    if llm_client is not None:
        return True
    provider = settings.llm_provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        return False
    if not (
        settings.llm_base_url.strip()
        and settings.llm_api_key.strip()
        and settings.llm_model.strip()
    ):
        return False
    if not _online_composer_requested(web_evidence_settings):
        return True
    online_settings = _online_composer_llm_settings(settings, web_evidence_settings)
    return bool(
        online_settings.llm_base_url.strip()
        and online_settings.llm_api_key.strip()
        and online_settings.llm_model.strip()
    )


def _composer_candidate_count_total(package: Mapping[str, Any]) -> int:
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    total = 0
    for rows in matrix.values():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("component_candidate_id") or row.get("candidate_id") or "").strip():
                total += 1
    for key in ("ready_stock_candidates", "rule_based_build_candidates"):
        for row in _mapping_rows(package.get(key)):
            if _candidate_id(row):
                total += 1
    return total


def _package_component_candidate_count_by_role(
    package: Mapping[str, Any],
) -> dict[str, int]:
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    counts: dict[str, int] = {}
    for prompt_role, rows in matrix.items():
        if not isinstance(rows, list):
            continue
        internal_role = INTERNAL_ROLE_BY_PROMPT_ROLE.get(
            str(prompt_role),
            str(prompt_role),
        )
        role = _coverage_role_key(internal_role)
        count = 0
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("component_candidate_id") or row.get("candidate_id") or "").strip():
                count += 1
        if count:
            counts[role] = count
    return counts


def _package_report_diagnostics(package: Mapping[str, Any]) -> dict[str, Any]:
    package_size_chars = _package_size_chars(package)
    diagnostics = {
        "package_budget": _safe_mapping(package.get("package_budget")),
        "package_budget_warnings": _string_list(package.get("package_budget_warnings")),
        "package_skipped_reason": _text_or_none(package.get("package_skipped_reason")),
        "package_approximate_size": {
            "chars": package_size_chars,
            "tokens_estimate": max(1, package_size_chars // 4),
        },
        "ready_candidates_excluded_reason": _text_or_none(
            package.get("ready_candidates_excluded_reason")
        ),
        "full_matrix_evaluation_used": bool(package.get("full_matrix_evaluation_used")),
        "full_matrix_evaluation_fallback_reason": _text_or_none(
            package.get("full_matrix_evaluation_fallback_reason")
        ),
        "provider_error_type": _text_or_none(package.get("provider_error_type")),
        "provider_context_limit": _safe_mapping(package.get("provider_context_limit")),
        "role_chunk_count_by_role": _safe_mapping(
            package.get("role_chunk_count_by_role")
        ),
        "evaluated_candidate_count_by_role": _safe_mapping(
            package.get("evaluated_candidate_count_by_role")
        ),
        "selected_candidate_count_by_role": _safe_mapping(
            package.get("selected_candidate_count_by_role")
        ),
        "role_reducer_summary": _safe_mapping(package.get("role_reducer_summary")),
        FULL_MATRIX_FAILED_CHUNKS_KEY: _mapping_rows(
            package.get(FULL_MATRIX_FAILED_CHUNKS_KEY)
        ),
        "bom_critic_used": bool(
            package.get("bom_critic_used")
            or _safe_mapping(package.get("llm_cost_diagnostics")).get("bom_critic_used")
        ),
        "no_recommendation_coverage": _safe_mapping(
            package.get("no_recommendation_coverage")
        ),
        "llm_cost_diagnostics": _safe_mapping(package.get("llm_cost_diagnostics")),
        "count_by_role": _package_count_by_role(package),
        "broad_matrix_count_by_role": _safe_mapping(
            package.get("broad_matrix_count_by_role")
        ),
        "composer_package_candidate_count_by_role": _safe_mapping(
            package.get("composer_package_candidate_count_by_role")
        ),
        "composer_package_candidate_total": _int_value(
            package.get("composer_package_candidate_total")
        )
        or 0,
        "composer_package_candidate_ids_by_role": _safe_mapping(
            package.get("composer_package_candidate_ids_by_role")
        ),
        "dropped_before_composer_count_by_role": _safe_mapping(
            package.get("dropped_before_composer_count_by_role")
        ),
        "dropped_before_composer_reason_by_role": _safe_mapping(
            package.get("dropped_before_composer_reason_by_role")
        ),
        "package_candidate_exposure_ratio_by_role": _safe_mapping(
            package.get("package_candidate_exposure_ratio_by_role")
        ),
        "original_candidate_count_by_role": _safe_mapping(
            package.get("original_candidate_count_by_role")
        ),
        "fallback_candidate_count_by_role": _safe_mapping(
            package.get("fallback_candidate_count_by_role")
        ),
        "dropped_before_fallback_count_by_role": _safe_mapping(
            package.get("dropped_before_fallback_count_by_role")
        ),
        "dropped_before_fallback_reasons": _safe_mapping(
            package.get("dropped_before_fallback_reasons")
        ),
        "timeout_fallback_coverage_ratio_by_role": _safe_mapping(
            package.get("timeout_fallback_coverage_ratio_by_role")
        ),
        "package_candidate_exposure_policy": _safe_mapping(
            package.get("package_candidate_exposure_policy")
        ),
        "package_candidate_exposure_incomplete": bool(
            package.get("package_candidate_exposure_incomplete")
        ),
        "package_candidate_exposure_incomplete_roles": _string_list(
            package.get("package_candidate_exposure_incomplete_roles")
        ),
        "package_exposure_blocking_lifecycle_roles": _string_list(
            package.get("package_exposure_blocking_lifecycle_roles")
        ),
        "category_planner_missing_required_roles": _string_list(
            package.get("category_planner_missing_required_roles")
        ),
        "category_planner_repair_attempted": bool(
            package.get("category_planner_repair_attempted")
        ),
        "category_planner_repair_success": bool(
            package.get("category_planner_repair_success")
        ),
        "category_planner_repair_reason": _text_or_none(
            package.get("category_planner_repair_reason")
        ),
        "category_planner_repaired_roles": _string_list(
            package.get("category_planner_repaired_roles")
        ),
        "category_planner_unresolved_required_roles": _string_list(
            package.get("category_planner_unresolved_required_roles")
        ),
        "stage_a_broad_roles": _string_list(package.get("stage_a_broad_roles")),
        "semantic_matrix_blueprint_roles": _string_list(
            package.get("semantic_matrix_blueprint_roles")
        ),
        "requirement_classifier_roles": _string_list(
            package.get("requirement_classifier_roles")
        ),
        "effective_matrix_roles_before_category_planner": _string_list(
            package.get("effective_matrix_roles_before_category_planner")
        ),
        "category_planner_input_roles": _string_list(
            package.get("category_planner_input_roles")
        ),
        "category_planner_output_roles": _string_list(
            package.get("category_planner_output_roles")
        ),
        "validated_category_plan_roles": _string_list(
            package.get("validated_category_plan_roles")
        ),
        "materialized_matrix_roles": _string_list(
            package.get("materialized_matrix_roles")
        ),
        "composer_package_roles": _string_list(package.get("composer_package_roles")),
        "roles_dropped_after_stage_a": _string_list(
            package.get("roles_dropped_after_stage_a")
        ),
        "roles_dropped_before_category_planner": _string_list(
            package.get("roles_dropped_before_category_planner")
        ),
        "roles_dropped_after_category_planner": _string_list(
            package.get("roles_dropped_after_category_planner")
        ),
        "roles_dropped_during_materialization": _string_list(
            package.get("roles_dropped_during_materialization")
        ),
        "roles_dropped_reason_by_role": _safe_mapping(
            package.get("roles_dropped_reason_by_role")
        ),
        "role_source_by_role": _safe_mapping(package.get("role_source_by_role")),
        "role_lifecycle_trace": _mapping_rows(package.get("role_lifecycle_trace")),
        "component_role_indicators": _mapping_rows(
            package.get("component_role_indicators")
        ),
        "embedded_requirements": _mapping_rows(package.get("embedded_requirements")),
        "requirement_fulfillment_decision": _mapping_rows(
            package.get("requirement_fulfillment_decision")
        ),
        "role_fulfillment_diagnostics": _mapping_rows(
            package.get("role_fulfillment_diagnostics")
        ),
        "classified_requirements": _mapping_rows(
            package.get("classified_requirements")
        ),
        "purchasable_role_requirements": _mapping_rows(
            package.get("purchasable_role_requirements")
        ),
        "primary_object_feature_requirements": _mapping_rows(
            package.get("primary_object_feature_requirements")
        ),
        "accessory_or_consumable_requirements": _mapping_rows(
            package.get("accessory_or_consumable_requirements")
        ),
        "service_or_support_requirements": _mapping_rows(
            package.get("service_or_support_requirements")
        ),
        "logistics_or_commercial_constraints": _mapping_rows(
            package.get("logistics_or_commercial_constraints")
        ),
        "engineering_check_requirements": _mapping_rows(
            package.get("engineering_check_requirements")
        ),
        "requirement_classifier_incomplete_reason": _text_or_none(
            package.get("requirement_classifier_incomplete_reason")
        ),
        "requirement_source_coverage": _mapping_rows(
            package.get("requirement_source_coverage")
        ),
        "requirement_source_coverage_percent": package.get(
            "requirement_source_coverage_percent"
        ),
        "unclassified_source_fragments": _string_list(
            package.get("unclassified_source_fragments")
        ),
        "pre_composer_requirement_classifier_status": _text_or_none(
            package.get("pre_composer_requirement_classifier_status")
            or package.get("requirement_classifier_status")
        ),
        "pre_composer_requirement_source_coverage_percent": package.get(
            "pre_composer_requirement_source_coverage_percent"
            if "pre_composer_requirement_source_coverage_percent" in package
            else "requirement_source_coverage_percent"
        ),
        "pre_composer_unclassified_source_fragments": _string_list(
            package.get("pre_composer_unclassified_source_fragments")
            or package.get("unclassified_source_fragments")
        ),
        "pre_composer_semantic_diagnostics_are_blocking": bool(
            package.get("pre_composer_semantic_diagnostics_are_blocking", False)
        ),
        "synthetic_requirement_count": _int_value(
            package.get("synthetic_requirement_count")
        )
        or 0,
        "source_backed_requirement_count": _int_value(
            package.get("source_backed_requirement_count")
        )
        or 0,
        "requirement_classifier_repair_quality": _text_or_none(
            package.get("requirement_classifier_repair_quality")
        ),
        "requirement_classifier_repair_accepted": bool(
            package.get("requirement_classifier_repair_accepted")
        ),
        "unmapped_requirements_non_blocking": _mapping_rows(
            package.get("unmapped_requirements_non_blocking")
        ),
        "unmapped_requirements_blocking": _mapping_rows(
            package.get("unmapped_requirements_blocking")
        ),
        "requirement_role_mapping_decision": _mapping_rows(
            package.get("requirement_role_mapping_decision")
        ),
    }
    for key in COMPACT_PACKAGE_DIAGNOSTIC_KEYS:
        if key in package:
            diagnostics[key] = package[key]
    return diagnostics


def _package_count_by_role(package: Mapping[str, Any]) -> dict[str, int]:
    broad_count_by_role = _safe_mapping(package.get("broad_count_by_role"))
    counts = {
        str(role): int(count)
        for role, count in broad_count_by_role.items()
        if str(role).strip() and isinstance(count, int) and count > 0
    }
    if counts:
        return counts

    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    return {
        _coverage_role_key(str(role)): len(rows)
        for role, rows in matrix.items()
        if isinstance(rows, list) and rows
    }


def _no_valid_recommendations_fallback_reason(
    *,
    response: LlmComposerResponsePayload,
    validated_pool: _ValidatedRecommendationPool,
) -> str:
    if response.recommendations:
        return "llm_configurator_all_recommendations_rejected"
    if validated_pool.proposal_count == 0 and validated_pool.rejected_count == 0:
        return "llm_configurator_no_proposals"
    return "llm_configurator_no_valid_recommendations"


def _has_structured_no_recommendation(
    response: LlmComposerResponsePayload | None,
) -> bool:
    if response is None:
        return False
    return _is_structured_no_recommendation_payload(response.no_recommendation)


def _is_structured_no_recommendation_payload(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    structured_keys = {
        "reason",
        "summary",
        "missing_roles",
        "missing_required_capabilities",
        "hard_mismatches",
        "hard_mismatch_risks",
        "hard_requirements_failed",
        "hard_requirements_met",
        "failed_requirements",
        "stock_shortages",
        "role_analysis",
        "role_failures",
        "role_level_reasons",
        "failures_by_role",
        "considered_candidate_ids",
        "considered_candidate_count_by_role",
        "requirement_coverage_summary",
        "recommended_next_actions",
        "recommended_repair_actions",
        "unverified_requirements",
        "explanation_ru",
    }
    if not structured_keys.intersection(value.keys()):
        return False
    if str(
        value.get("reason")
        or value.get("summary")
        or value.get("explanation_ru")
        or ""
    ).strip():
        return True
    return any(
        bool(_string_list(value.get(key)))
        for key in (
            "missing_roles",
            "general_notes",
            "recommended_next_actions",
            "recommended_repair_actions",
        )
    ) or any(
        bool(_mapping_rows(value.get(key)))
        for key in (
            "missing_required_capabilities",
            "hard_mismatches",
            "hard_mismatch_risks",
            "hard_requirements_failed",
            "hard_requirements_met",
            "failed_requirements",
            "stock_shortages",
            "role_analysis",
            "role_failures",
            "role_level_reasons",
            "unverified_requirements",
        )
    ) or bool(_role_failure_rows_from_mapping(value.get("failures_by_role"))) or any(
        bool(_string_list(value.get(key)))
        for key in (
            "hard_requirements_failed",
            "hard_requirements_met",
            "failed_requirements",
            "unverified_requirements",
            "hard_mismatch_risks",
        )
    ) or bool(_normalize_considered_candidate_ids(value.get("considered_candidate_ids")))


def _structured_no_recommendation_reason(
    *,
    no_recommendation: Mapping[str, Any],
    fallback_reason: str,
    validated_pool: _ValidatedRecommendationPool,
    warnings: Sequence[str],
    product_group: str,
) -> dict[str, Any]:
    normalized = _normalize_no_recommendation_payload(no_recommendation)
    base_reason = _no_recommendation_reason(
        fallback_reason=fallback_reason,
        validated_pool=validated_pool,
        warnings=warnings,
        product_group=product_group,
    )
    missing_roles = _unique(
        [
            *_string_list(normalized.get("missing_roles")),
            *_string_list(base_reason.get("missing_roles")),
        ]
    )
    missing_required_capabilities = _unique_capability_rows(
        [
            *_mapping_rows(normalized.get("missing_required_capabilities")),
            *_mapping_rows(base_reason.get("missing_required_capabilities")),
        ]
    )
    stock_shortages = [
        *_mapping_rows(normalized.get("stock_shortages")),
        *_mapping_rows(base_reason.get("stock_shortages")),
    ]
    hard_mismatches = _mapping_rows(normalized.get("hard_mismatches"))
    hard_incompatibility = _unique(
        [
            *_string_list(base_reason.get("hard_incompatibility")),
            *[
                _safe_diagnostic_text(
                    mismatch.get("reason")
                    or mismatch.get("requirement")
                    or mismatch.get("candidate_fact"),
                    limit=200,
                )
                for mismatch in hard_mismatches
                if isinstance(mismatch, Mapping)
            ],
        ]
    )
    reason = {
        **base_reason,
        "summary": str(
            normalized.get("summary") or base_reason.get("summary") or ""
        ).strip(),
        "fallback_reason": fallback_reason,
        "product_group": product_group,
        "missing_roles": missing_roles,
        "missing_required_capabilities": missing_required_capabilities,
        "hard_mismatches": hard_mismatches,
        "stock_shortages": stock_shortages,
        "role_analysis": _mapping_rows(normalized.get("role_analysis")),
        "role_failures": _mapping_rows(normalized.get("role_failures")),
        "considered_candidate_ids": _normalize_considered_candidate_ids(
            normalized.get("considered_candidate_ids")
        ),
        "validator_rejection_summary": _safe_mapping(
            normalized.get("validator_rejection_summary")
        ),
        "validator_errors": _mapping_rows(normalized.get("validator_errors")),
        "rejected_component_candidate_ids": _string_list(
            normalized.get("rejected_component_candidate_ids")
        ),
        "explanation_ru": str(normalized.get("explanation_ru") or "").strip(),
        "hard_incompatibility": hard_incompatibility,
        "structured_no_recommendation": True,
    }
    return reason


def _composer_returned_no_proposal_twice_reason(
    reason: dict[str, Any],
) -> dict[str, Any]:
    notes = _string_list(reason.get("diagnostic_notes"))
    message = (
        "Composer returned no proposal twice: the parsed primary response and the "
        "parsed repair response were both empty, with no structured no_recommendation."
    )
    return {
        **reason,
        "summary": "Composer returned no proposal twice and no structured no_recommendation.",
        "diagnostic_notes": _unique([message, *notes]),
        "composer_no_proposal_attempts": 2,
    }


def _empty_response_parse_diagnostics(fallback_reason: str) -> dict[str, Any]:
    if fallback_reason not in {
        "llm_configurator_no_proposals",
        "llm_configurator_no_proposals_after_repair",
    }:
        return {}
    return {
        "llm_parse_stage": "composer_response",
        "llm_json_extract_status": "parsed",
        "llm_recommendations_count": 0,
    }


def _normalized_output_mode(settings: LlmSettings) -> str:
    return _normalize_output_mode(
        getattr(settings, "llm_configurator_output_mode", OUTPUT_MODE_SINGLE_BEST_COST_VALID)
    )


def _normalize_output_mode(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("-", "_")
    if text == "multi_option":
        return OUTPUT_MODE_GROUPED_PRESALES
    if text in SUPPORTED_OUTPUT_MODES:
        return text
    return OUTPUT_MODE_SINGLE_BEST_COST_VALID


def _single_best_output_mode(output_mode: Any) -> bool:
    return _normalize_output_mode(output_mode) == OUTPUT_MODE_SINGLE_BEST_COST_VALID


def _grouped_presales_output_mode(output_mode: Any) -> bool:
    return _normalize_output_mode(output_mode) in {
        OUTPUT_MODE_GROUPED_PRESALES,
        OUTPUT_MODE_LEGACY_MULTI_OPTION,
    }


def build_llm_configurator_user_prompt(package: Mapping[str, Any]) -> str:
    return json.dumps(package, ensure_ascii=False, sort_keys=True)


def _run_repair_pass(
    *,
    package: Mapping[str, Any],
    settings: LlmSettings,
    llm_client: LlmClient | None,
    llm_call_budget: LlmCallBudget | None,
    primary_response: LlmComposerResponsePayload,
    primary_validated_pool: _ValidatedRecommendationPool,
    stock_candidate_index: dict[str, _IndexedStockCandidate],
    component_index: dict[str, _IndexedComponentCandidate],
    limit: int,
    evidence_pack: Mapping[str, Any] | None,
    evidence_review: Mapping[str, Any] | None,
    use_recommendation_evidence: bool,
    output_mode: str,
) -> _RepairPassResult:
    repair_source_recommendations = (
        primary_validated_pool.recommendations
        if _single_best_output_mode(output_mode)
        else primary_validated_pool.accepted_recommendations
    )
    critique = _build_repair_critique(
        accepted_recommendations=repair_source_recommendations,
        component_index=component_index,
        normalized_requirements=package["normalized_requirements"],
        max_alternatives_per_role=(
            settings.llm_configurator_repair_max_alternatives_per_role
        ),
    )
    if not critique.facts:
        return _RepairPassResult(
            critique_count=0,
            critique_summary=[],
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=None,
        )
    savings_estimate = _json_decimal(critique.savings_estimate)
    if not settings.llm_configurator_repair_enabled:
        return _RepairPassResult(
            fallback_reason="llm_repair_disabled",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
        )

    repair_package = _build_repair_package(
        package=package,
        primary_response=primary_response,
        primary_validated_pool=primary_validated_pool,
        critique=critique,
        output_mode=output_mode,
        original_recommendations=repair_source_recommendations,
    )
    client = budgeted_llm_client(llm_client, llm_call_budget)
    owns_client = False
    if client is None:
        try:
            client = budgeted_llm_client(
                _build_repair_llm_client(settings),
                llm_call_budget,
            )
            owns_client = True
        except LlmError as exc:
            return _RepairPassResult(
                attempted=True,
                fallback_reason="llm_repair_client_unavailable",
                critique_count=len(critique.facts),
                critique_summary=critique.summary,
                blocked_critique_count=len(critique.blocked_facts),
                blocked_critique_summary=critique.blocked_summary,
                savings_estimate=savings_estimate,
                warnings=[f"llm_repair: {type(exc).__name__}"],
            )

    try:
        payload = client.generate_json(
            LLM_CONFIGURATOR_REPAIR_SYSTEM_PROMPT,
            build_llm_configurator_user_prompt(repair_package),
        )
    except LlmCallBudgetExceededError as exc:
        _close_owned_client(client, owns_client)
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_call_budget_exceeded",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
            warnings=[f"llm_repair: {type(exc).__name__}"],
            thinking_diagnostics=llm_call_budget_diagnostics(llm_call_budget),
        )
    except LlmReadTimeoutError as exc:
        _close_owned_client(client, owns_client)
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_repair_read_timeout",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
            warnings=[f"llm_repair: {type(exc).__name__}"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )
    except LlmInvalidJsonError as exc:
        _close_owned_client(client, owns_client)
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_repair_invalid_json",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
            warnings=[f"llm_repair: {type(exc).__name__}"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )
    except LlmError as exc:
        _close_owned_client(client, owns_client)
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_repair_request_failed",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
            warnings=[f"llm_repair: {type(exc).__name__}"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    normalized_payload = _normalize_composer_payload(payload, package=package)
    try:
        parsed_payload = _parse_composer_payload(normalized_payload)
    except ValidationError:
        _close_owned_client(client, owns_client)
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_repair_validation_failed",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
            revised_proposals_count=_raw_proposals_count(normalized_payload),
            warnings=["llm_repair: ValidationError"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )
    if not parsed_payload.response.recommendations:
        _close_owned_client(client, owns_client)
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_repair_no_recommendations",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
            revised_proposals_count=parsed_payload.proposal_count,
            warnings=["llm_repair: no valid proposal schema rows"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    allowed_component_ids = _repair_allowed_component_candidate_ids(repair_package)
    repair_recommendations, repair_guard_rejections, repair_proposal_indexes = (
        _filter_repair_recommendations_by_allowed_component_ids(
            parsed_payload.response.recommendations,
            parsed_payload.proposal_indexes,
            allowed_component_ids=allowed_component_ids,
        )
    )
    if not repair_recommendations:
        _close_owned_client(client, owns_client)
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_repair_revised_no_valid_recommendations",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
            revised_proposals_count=parsed_payload.proposal_count,
            validation_summary=_validation_summary(
                accepted=0,
                accepted_after_validation=0,
                rejected=[
                    *parsed_payload.schema_rejections,
                    *repair_guard_rejections,
                ],
            ),
            warnings=[
                "llm_repair: revised proposals used component IDs outside "
                "allowed alternatives"
            ],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    revised_pool = _validate_recommendations(
        repair_recommendations,
        stock_candidate_index=stock_candidate_index,
        component_index=component_index,
        user_request=package["user_request"],
        normalized_requirements=package["normalized_requirements"],
        limit=limit,
        evidence_pack=evidence_pack,
        evidence_review=evidence_review,
        use_recommendation_evidence=use_recommendation_evidence,
        schema_rejections=[
            *parsed_payload.schema_rejections,
            *repair_guard_rejections,
        ],
        proposal_indexes=repair_proposal_indexes,
        proposal_count=parsed_payload.proposal_count,
    )
    diagnostics = _client_thinking_diagnostics(client)
    _close_owned_client(client, owns_client)
    if not revised_pool.recommendations:
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_repair_revised_no_valid_recommendations",
            critique_count=len(critique.facts),
            critique_summary=critique.summary,
            blocked_critique_count=len(critique.blocked_facts),
            blocked_critique_summary=critique.blocked_summary,
            savings_estimate=savings_estimate,
            revised_proposals_count=revised_pool.proposal_count,
            validation_summary=revised_pool.validation_summary,
            warnings=["llm_repair: revised proposals were rejected by validator"],
            thinking_diagnostics=diagnostics,
        )

    return _RepairPassResult(
        used=True,
        attempted=True,
        success=True,
        critique_count=len(critique.facts),
        critique_summary=critique.summary,
        blocked_critique_count=len(critique.blocked_facts),
        blocked_critique_summary=critique.blocked_summary,
        savings_estimate=savings_estimate,
        revised_proposals_count=revised_pool.proposal_count,
        validation_summary=revised_pool.validation_summary,
        validated_pool=revised_pool,
        response=parsed_payload.response,
        warnings=[
            "llm_repair: revised proposal pool accepted by deterministic validator"
        ],
        thinking_diagnostics=diagnostics,
    )


def _run_validation_aware_repair_if_needed(
    *,
    package: Mapping[str, Any],
    settings: LlmSettings,
    client: LlmClient | None,
    primary_response: LlmComposerResponsePayload,
    primary_validated_pool: _ValidatedRecommendationPool,
    stock_candidate_index: dict[str, _IndexedStockCandidate],
    component_index: dict[str, _IndexedComponentCandidate],
    limit: int,
    evidence_pack: Mapping[str, Any] | None,
    evidence_review: Mapping[str, Any] | None,
    use_recommendation_evidence: bool,
) -> _RepairPassResult:
    if not _validation_repair_should_run(
        response=primary_response,
        validated_pool=primary_validated_pool,
    ):
        return _RepairPassResult()

    base_diagnostics = _validation_repair_base_diagnostics(
        primary_response=primary_response,
        primary_validated_pool=primary_validated_pool,
    )
    if _composer_repair_max_attempts(settings) <= 0:
        return _RepairPassResult(
            fallback_reason="validation_repair_budget_exhausted",
            diagnostics={
                **base_diagnostics,
                "validation_repair_fallback_reason": (
                    "validation_repair_budget_exhausted"
                ),
            },
        )
    if client is None:
        return _RepairPassResult(
            fallback_reason="validation_repair_client_unavailable",
            diagnostics={
                **base_diagnostics,
                "validation_repair_fallback_reason": (
                    "validation_repair_client_unavailable"
                ),
            },
        )

    repair_package = _validation_repair_package(
        package=package,
        primary_response=primary_response,
        primary_validated_pool=primary_validated_pool,
    )
    try:
        payload = client.generate_json(
            LLM_VALIDATION_AWARE_REPAIR_SYSTEM_PROMPT,
            build_llm_configurator_user_prompt(repair_package),
        )
    except LlmCallBudgetExceededError as exc:
        diagnostics = {
            **base_diagnostics,
            **llm_call_budget_diagnostics(getattr(client, "budget", None)),
            "validation_repair_attempted": True,
            "validation_repair_fallback_reason": "llm_call_budget_exceeded",
        }
        return _RepairPassResult(
            attempted=True,
            fallback_reason="llm_call_budget_exceeded",
            warnings=[f"validation_repair: {type(exc).__name__}"],
            diagnostics=diagnostics,
        )
    except LlmReadTimeoutError as exc:
        return _RepairPassResult(
            attempted=True,
            fallback_reason="validation_repair_read_timeout",
            warnings=[f"validation_repair: {type(exc).__name__}"],
            diagnostics={
                **base_diagnostics,
                "validation_repair_attempted": True,
                "validation_repair_fallback_reason": "validation_repair_read_timeout",
            },
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )
    except LlmInvalidJsonError as exc:
        return _RepairPassResult(
            attempted=True,
            fallback_reason="validation_repair_invalid_json",
            warnings=[f"validation_repair: {type(exc).__name__}"],
            diagnostics={
                **base_diagnostics,
                "validation_repair_attempted": True,
                "validation_repair_fallback_reason": "validation_repair_invalid_json",
            },
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )
    except LlmError as exc:
        return _RepairPassResult(
            attempted=True,
            fallback_reason="validation_repair_request_failed",
            warnings=[f"validation_repair: {type(exc).__name__}"],
            diagnostics={
                **base_diagnostics,
                "validation_repair_attempted": True,
                "validation_repair_fallback_reason": "validation_repair_request_failed",
            },
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    normalized_payload = _normalize_composer_payload(payload, package=package)
    try:
        parsed_payload = _parse_composer_payload(normalized_payload)
    except ValidationError:
        diagnostics = {
            **base_diagnostics,
            "validation_repair_attempted": True,
            "validation_repair_used": False,
            "validation_repair_success": False,
            "validation_repair_fallback_reason": "validation_repair_schema_failure",
            "validation_repair_failure_reason": "repair_schema_failure",
            "validation_repair_schema_failure": True,
        }
        return _RepairPassResult(
            attempted=True,
            fallback_reason="validation_repair_schema_failure",
            revised_proposals_count=_raw_proposals_count(normalized_payload),
            warnings=["validation_repair: repair_schema_failure"],
            diagnostics=diagnostics,
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    if _has_structured_no_recommendation(parsed_payload.response):
        response = _response_with_validation_repair_no_recommendation_context(
            parsed_payload.response,
            primary_response=primary_response,
            primary_validated_pool=primary_validated_pool,
        )
        diagnostics = {
            **base_diagnostics,
            "validation_repair_attempted": True,
            "validation_repair_used": True,
            "validation_repair_success": True,
            "validation_repair_returned_no_recommendation": True,
            "validation_repair_fallback_reason": COMPOSER_STRUCTURED_NO_RECOMMENDATION,
            "validation_repair_final_validation_summary": {},
            "final_bom_after_repair": _jsonable(response.model_dump()),
        }
        return _RepairPassResult(
            used=True,
            attempted=True,
            success=True,
            fallback_reason=COMPOSER_STRUCTURED_NO_RECOMMENDATION,
            revised_proposals_count=parsed_payload.proposal_count,
            validated_pool=_empty_validated_recommendation_pool(),
            response=response,
            warnings=[
                "validation_repair: structured no_recommendation returned by Composer"
            ],
            diagnostics=diagnostics,
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    if not _validation_repair_response_has_component_ids(parsed_payload.response):
        response = _response_with_validation_repair_empty_no_recommendation_context(
            parsed_payload.response,
            primary_response=primary_response,
            primary_validated_pool=primary_validated_pool,
        )
        diagnostics = {
            **base_diagnostics,
            "validation_repair_attempted": True,
            "validation_repair_used": True,
            "validation_repair_success": False,
            "validation_repair_fallback_reason": (
                "validation_repair_empty_without_no_recommendation"
            ),
            "validation_repair_failure_reason": "empty_without_no_recommendation",
            "validation_repair_empty_output": True,
            "validation_repair_final_validation_summary": {},
            "final_bom_after_repair": _jsonable(response.model_dump()),
        }
        return _RepairPassResult(
            used=True,
            attempted=True,
            fallback_reason="validation_repair_empty_without_no_recommendation",
            revised_proposals_count=parsed_payload.proposal_count,
            validated_pool=primary_validated_pool,
            response=response,
            warnings=["validation_repair: empty response without no_recommendation"],
            diagnostics=diagnostics,
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    repair_recommendations, guard_rejections, repair_proposal_indexes = (
        _filter_repeated_validation_repair_recommendations(
            parsed_payload.response.recommendations,
            parsed_payload.proposal_indexes,
            primary_response=primary_response,
        )
    )
    revised_pool = _validate_recommendations(
        repair_recommendations,
        stock_candidate_index=stock_candidate_index,
        component_index=component_index,
        user_request=package["user_request"],
        normalized_requirements=package["normalized_requirements"],
        limit=limit,
        evidence_pack=evidence_pack,
        evidence_review=evidence_review,
        use_recommendation_evidence=use_recommendation_evidence,
        schema_rejections=[
            *parsed_payload.schema_rejections,
            *guard_rejections,
        ],
        proposal_indexes=repair_proposal_indexes,
        proposal_count=parsed_payload.proposal_count,
    )
    diagnostics = {
        **base_diagnostics,
        "validation_repair_attempted": True,
        "validation_repair_used": True,
        "validation_repair_success": bool(revised_pool.recommendations),
        "validation_repair_returned_no_recommendation": False,
        "validation_repair_final_validation_summary": _jsonable(
            revised_pool.validation_summary
        ),
        "validation_repair_rejected_debug_safe": _validation_repair_debug_rows(
            revised_pool,
            attempt="repair",
        ),
        "final_bom_after_repair": _jsonable(parsed_payload.response.model_dump()),
    }
    if revised_pool.recommendations:
        return _RepairPassResult(
            used=True,
            attempted=True,
            success=True,
            revised_proposals_count=revised_pool.proposal_count,
            validation_summary=revised_pool.validation_summary,
            validated_pool=revised_pool,
            response=parsed_payload.response,
            warnings=[
                "validation_repair: repaired BOM accepted by deterministic validator"
            ],
            diagnostics=diagnostics,
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    combined_pool = _merge_validation_repair_failure_pools(
        primary_validated_pool,
        revised_pool,
    )
    return _RepairPassResult(
        used=True,
        attempted=True,
        success=False,
        fallback_reason="validation_repair_rejected_by_validation",
        revised_proposals_count=revised_pool.proposal_count,
        validation_summary=combined_pool.validation_summary,
        validated_pool=combined_pool,
        response=parsed_payload.response,
        warnings=["validation_repair: repaired BOM rejected by deterministic validator"],
        diagnostics={
            **diagnostics,
            "validation_repair_fallback_reason": (
                "validation_repair_rejected_by_validation"
            ),
        },
        thinking_diagnostics=_client_thinking_diagnostics(client),
    )


def _run_deterministic_primary_role_repair_if_needed(
    *,
    package: Mapping[str, Any],
    response: LlmComposerResponsePayload,
    validated_pool: _ValidatedRecommendationPool,
    component_index: dict[str, _IndexedComponentCandidate],
    stock_candidate_index: dict[str, _IndexedStockCandidate],
    limit: int,
    evidence_pack: Mapping[str, Any] | None,
    evidence_review: Mapping[str, Any] | None,
    use_recommendation_evidence: bool,
) -> _RepairPassResult:
    if validated_pool.recommendations or not _has_structured_no_recommendation(response):
        return _RepairPassResult()
    if str(package.get("product_group") or "").strip() != NETWORK_PRODUCT_GROUP:
        return _RepairPassResult()

    reason = _safe_mapping(response.no_recommendation)
    if (
        _mapping_rows(reason.get("hard_mismatches"))
        or _mapping_rows(reason.get("stock_shortages"))
        or _mapping_rows(reason.get("missing_required_capabilities"))
    ):
        return _RepairPassResult()

    primary_roles = {
        SWITCH_ROLE,
        ROUTER_ROLE,
        FIREWALL_ROLE,
        ACCESS_POINT_ROLE,
    }
    hard_roles = _string_list(package.get("hard_purchasable_bom_roles"))
    if not hard_roles:
        contract = _safe_mapping(package.get("requirement_contract"))
        hard_roles = _string_list(contract.get("required_roles"))
    required_primary_roles = [role for role in hard_roles if role in primary_roles]
    if len(required_primary_roles) != 1:
        return _RepairPassResult()
    role = required_primary_roles[0]

    missing_roles = set(_string_list(reason.get("missing_roles")))
    for row in _mapping_rows(reason.get("role_analysis")):
        row_role = str(row.get("role") or "").strip()
        row_status = str(row.get("status") or row.get("decision") or "").strip()
        if row_role and row_status in {"missing", "not_selected", "no_safe_choice"}:
            missing_roles.add(row_role)
    for row in _mapping_rows(reason.get("role_failures")):
        row_role = str(row.get("role") or "").strip()
        if row_role:
            missing_roles.add(row_role)
    if missing_roles and role not in missing_roles:
        return _RepairPassResult()

    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    rows = _mapping_rows(matrix.get(role))
    selectable_rows = _selectable_component_rows(rows) or rows
    if not selectable_rows:
        return _RepairPassResult()

    diagnostics_base = {
        "deterministic_primary_role_repair_attempted": True,
        "deterministic_primary_role_repair_role": role,
        "deterministic_primary_role_repair_reason": (
            "structured_no_recommendation_missing_role_with_candidates"
        ),
        "deterministic_primary_role_repair_candidate_count": len(selectable_rows),
    }
    last_pool: _ValidatedRecommendationPool | None = None
    skipped_by_text_requirement = 0
    for row in sorted(selectable_rows, key=_component_candidate_sort_key):
        component_id = str(row.get("component_candidate_id") or "").strip()
        if not component_id:
            continue
        component = component_index.get(component_id)
        if component is None:
            continue
        text_mismatch = _network_primary_role_text_mismatch(
            component,
            str(
                package.get("original_request_text")
                or package.get("user_request")
                or ""
            ),
        )
        if text_mismatch:
            skipped_by_text_requirement += 1
            continue
        repaired_response = LlmComposerResponsePayload(
            requirement_analysis=dict(response.requirement_analysis),
            fulfillment_decisions=list(response.fulfillment_decisions),
            recommendations=[
                LlmRecommendationPayload(
                    recommendation_id=f"deterministic_primary_{role}",
                    proposal_role="cheapest_fit",
                    recommendation_slot="price_optimal",
                    source_type=BUILD_CANDIDATE_TYPE,
                    component_candidate_ids={role: component_id},
                    quantities={role: 1},
                    decision="recommend_with_checks",
                    title=f"Deterministic {role} recovery",
                    why_selected=(
                        "Composer returned a missing-role no-recommendation despite "
                        "stocked primary-role candidates; application selected the "
                        "lowest-price candidate that passes deterministic validation."
                    ),
                    engineer_checks=[
                        "Verify final port/media/licensing/support details before quotation."
                    ],
                    confidence="medium",
                )
            ],
            general_notes=[
                *response.general_notes,
                "deterministic_primary_role_repair",
            ],
        )
        revised_pool = _validate_recommendations(
            repaired_response.recommendations,
            stock_candidate_index=stock_candidate_index,
            component_index=component_index,
            user_request=package.get("user_request"),
            normalized_requirements=package.get("normalized_requirements"),
            limit=limit,
            evidence_pack=evidence_pack,
            evidence_review=evidence_review,
            use_recommendation_evidence=use_recommendation_evidence,
            proposal_count=1,
        )
        last_pool = revised_pool
        if revised_pool.recommendations:
            return _RepairPassResult(
                used=True,
                attempted=True,
                success=True,
                revised_proposals_count=revised_pool.proposal_count,
                validation_summary=revised_pool.validation_summary,
                validated_pool=revised_pool,
                response=repaired_response,
                warnings=[
                    "deterministic_primary_role_repair: valid primary-role BOM accepted"
                ],
                diagnostics={
                    **diagnostics_base,
                    "deterministic_primary_role_repair_success": True,
                    "deterministic_primary_role_repair_candidate_id": component_id,
                    "deterministic_primary_role_repair_skipped_by_text_requirement": (
                        skipped_by_text_requirement
                    ),
                    "deterministic_primary_role_repair_validation_summary": _jsonable(
                        revised_pool.validation_summary
                    ),
                },
            )

    return _RepairPassResult(
        attempted=True,
        success=False,
        validation_summary=(
            last_pool.validation_summary if last_pool is not None else {}
        ),
        warnings=[
            "deterministic_primary_role_repair: no primary-role candidate passed validation"
        ],
        diagnostics={
            **diagnostics_base,
            "deterministic_primary_role_repair_success": False,
            "deterministic_primary_role_repair_skipped_by_text_requirement": (
                skipped_by_text_requirement
            ),
            "deterministic_primary_role_repair_validation_summary": _jsonable(
                last_pool.validation_summary if last_pool is not None else {}
            ),
        },
    )


def _network_primary_role_text_mismatch(
    candidate: _IndexedComponentCandidate,
    request_text: str,
) -> str:
    text = request_text.casefold()
    if not text:
        return ""
    port_match = re.search(
        r"\b(\d{2,3})\s*(?:x|×|х)?\s*"
        r"(?:(?:1\s*g(?:be)?)|(?:1000\s*base)|rj-?45|ports?|порт(?:а|ов)?)\b",
        text,
    )
    if port_match:
        required_ports = _int_value(port_match.group(1))
        actual_ports = _int_value(_fact(candidate, "port_count"))
        if required_ports is not None and (
            actual_ports is None or actual_ports < required_ports
        ):
            return "port_count"
    if re.search(r"\b1\s*g(?:be)?\b|\b1g(?:be)?\b", text) and "port" in text:
        speed = _decimal_value(_fact(candidate, "port_speed_gbps"))
        if speed is None or speed < Decimal("1"):
            return "port_speed"
    if "poe" in text and not _truthy(_fact(candidate, "poe_supported")):
        return "poe"
    if re.search(r"\bl3\b|layer\s*3", text) and not _truthy(
        _fact(candidate, "l3_supported")
    ):
        return "l3"
    uplink_match = re.search(r"\b(\d+)\s*x\s*10\s*g", text)
    if uplink_match and ("uplink" in text or "sfp" in text):
        required_uplinks = _int_value(uplink_match.group(1))
        actual_uplinks = _int_value(_fact(candidate, "uplink_count"))
        if required_uplinks is not None and (
            actual_uplinks is None or actual_uplinks < required_uplinks
        ):
            return "uplink_count"
        uplink_speed = _decimal_value(_fact(candidate, "uplink_speed_gbps"))
        if uplink_speed is None or uplink_speed < Decimal("10"):
            return "uplink_speed"
        if "sfp" in text:
            uplink_media = str(_fact(candidate, "uplink_media") or "").casefold()
            if "sfp" not in uplink_media:
                return "uplink_media"
    return ""


def _validation_repair_should_run(
    *,
    response: LlmComposerResponsePayload,
    validated_pool: _ValidatedRecommendationPool,
) -> bool:
    if _has_structured_no_recommendation(response):
        return False
    if not response.recommendations:
        return False
    if validated_pool.recommendations:
        return False
    return bool(
        validated_pool.validation_rejected_count
        or validated_pool.rejected_count
        or validated_pool.rejected_debug_safe
    )


def _validation_repair_package(
    *,
    package: Mapping[str, Any],
    primary_response: LlmComposerResponsePayload,
    primary_validated_pool: _ValidatedRecommendationPool,
) -> dict[str, Any]:
    contract = _safe_mapping(
        package.get("requirement_contract")
    ) or _fallback_requirement_contract_from_package(package)
    validation_errors = _validation_repair_error_rows(primary_validated_pool)
    rejected_ids = _validation_repair_rejected_candidate_ids(
        primary_response=primary_response,
        validation_errors=validation_errors,
    )
    forbidden_combinations = _validation_repair_forbidden_combinations(
        primary_response=primary_response,
        validation_errors=validation_errors,
    )
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    return {
        "multi_pass_stage": "validation_repair",
        "repair_attempt": 1,
        "repair_kind": "post_composer_validation",
        "original_request_text": package.get("original_request_text")
        or package.get("user_request"),
        "product_group": package.get("product_group"),
        "primary_object": package.get("primary_object"),
        "requirement_contract": _compact_requirement_contract_for_repair(
            contract,
            package=package,
        ),
        "component_role_contract": _safe_mapping(
            package.get("component_role_contract")
        ),
        "component_candidate_matrix": _jsonable(matrix),
        "candidate_facts_by_role": _jsonable(
            _validation_repair_candidate_facts_by_role(
                package=package,
                validation_errors=validation_errors,
                primary_response=primary_response,
            )
        ),
        "rejected_bom": _jsonable(primary_response.model_dump()),
        "validator_errors": validation_errors,
        "validation_summary": _jsonable(primary_validated_pool.validation_summary),
        "rejected_component_candidate_ids": rejected_ids,
        "forbidden_component_combinations": forbidden_combinations,
        "candidate_exposure_policy": {
            "mode": "validation_repair_full_matrix",
            "silent_trimming": False,
            "candidate_count_by_role": _safe_mapping(
                package.get("composer_package_candidate_count_by_role")
            )
            or {
                role: len(_mapping_rows(rows))
                for role, rows in matrix.items()
                if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
            },
        },
        "repair_instructions": [
            "Use validator_errors as authoritative rejection reasons.",
            "Do not repeat forbidden_component_combinations.",
            "Return corrected canonical BOM or structured no_recommendation.",
            "Use only component_candidate_id values from component_candidate_matrix.",
            "Do not change stock, price, compatibility, or requirement facts.",
        ],
    }


def _compact_requirement_contract_for_repair(
    contract: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "primary_object": contract.get("primary_object")
        or _package_primary_object(package),
        "required_roles": _contract_required_roles(contract, package=package),
        "required_quantities_by_role": _normalized_contract_quantities_by_role(
            contract,
            package=package,
        ),
        "hard_requirements": _jsonable(contract.get("hard_requirements")),
        "primary_object_features": _jsonable(contract.get("primary_object_features")),
        "fulfillment_expectations": _jsonable(
            contract.get("fulfillment_expectations")
        ),
        "logistics_commercial_constraints": _jsonable(
            contract.get("logistics_commercial_constraints")
        ),
        "engineer_checks": _jsonable(contract.get("engineer_checks")),
    }


def _validation_repair_base_diagnostics(
    *,
    primary_response: LlmComposerResponsePayload,
    primary_validated_pool: _ValidatedRecommendationPool,
) -> dict[str, Any]:
    errors = _validation_repair_error_rows(primary_validated_pool)
    return {
        "validation_repair_attempted": False,
        "validation_repair_used": False,
        "validation_repair_success": False,
        "validation_repair_returned_no_recommendation": False,
        "validation_repair_initial_validation_summary": _jsonable(
            primary_validated_pool.validation_summary
        ),
        "validation_repair_rejected_candidate_ids": (
            _validation_repair_rejected_candidate_ids(
                primary_response=primary_response,
                validation_errors=errors,
            )
        ),
        "validation_repair_concrete_errors": errors,
        "validation_repair_forbidden_component_combinations": (
            _validation_repair_forbidden_combinations(
                primary_response=primary_response,
                validation_errors=errors,
            )
        ),
    }


def _validation_repair_error_rows(
    validated_pool: _ValidatedRecommendationPool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _mapping_rows(validated_pool.rejected_debug_safe):
        concrete_reasons = _concrete_validation_reasons(row)
        rows.append(
            {
                "recommendation_id": row.get("recommendation_id"),
                "proposal_index": row.get("proposal_index"),
                "rejection_code": row.get("rejection_code"),
                "rejection_category": row.get("rejection_category"),
                "stage": row.get("stage"),
                "concrete_validation_reasons": concrete_reasons,
                "validation_errors": _jsonable(row.get("validation_errors")),
                "component_candidate_ids": _safe_component_id_map(
                    row.get("component_candidate_ids")
                ),
                "normalized_core_component_candidate_ids": _safe_component_id_map(
                    row.get("normalized_core_component_candidate_ids")
                ),
                "selected_component_candidate_ids": _safe_component_id_map(
                    row.get("selected_component_candidate_ids")
                ),
                "missing_roles": _string_list(row.get("missing_roles")),
                "stock_shortages": _mapping_rows(row.get("stock_shortages")),
                "role_mismatches": _mapping_rows(row.get("role_mismatches")),
                "unknown_component_ids": _jsonable(row.get("unknown_component_ids")),
                "validation_hard_mismatches": _mapping_rows(
                    row.get("validation_hard_mismatches")
                ),
                "missing_required_capabilities": _mapping_rows(
                    row.get("missing_required_capabilities")
                ),
            }
        )
    return rows


def _concrete_validation_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = [
        str(row.get("rejection_code") or "").strip(),
        str(row.get("rejection_category") or "").strip(),
        str(row.get("rejection_message_ru") or "").strip(),
    ]
    reasons.extend(_string_list(row.get("concrete_validation_reasons")))
    reasons.extend(_string_list(row.get("validation_errors")))
    reasons.extend(
        f"unknown_component_id:{component_id}"
        for component_id in _string_list(row.get("unknown_component_ids"))
    )
    reasons.extend(
        f"stock_shortage:{item.get('role')} "
        f"{item.get('required_quantity')} > {item.get('available_quantity')}"
        for item in _mapping_rows(row.get("stock_shortages"))
    )
    reasons.extend(
        _validation_row_reason(item)
        for item in _mapping_rows(row.get("validation_hard_mismatches"))
        if _validation_row_reason(item)
    )
    reasons.extend(
        _validation_row_reason(item)
        for item in _mapping_rows(row.get("hard_capability_validation"))
        if _validation_row_reason(item)
    )
    reasons.extend(
        f"missing_role:{role}" for role in _string_list(row.get("missing_roles"))
    )
    materialization_error = str(row.get("materialization_error") or "").strip()
    if materialization_error:
        reasons.append(f"materialization_error:{materialization_error}")
    return _unique([reason for reason in reasons if reason])


def _validation_repair_rejected_candidate_ids(
    *,
    primary_response: LlmComposerResponsePayload,
    validation_errors: Sequence[Mapping[str, Any]],
) -> list[str]:
    ids: list[str] = []
    for recommendation in primary_response.recommendations:
        ids.extend(_core_component_candidate_ids(recommendation).values())
        ids.extend(recommendation.optional_component_candidate_ids.values())
        ids.extend(recommendation.engineer_check_component_candidate_ids.values())
    for row in validation_errors:
        for key in (
            "component_candidate_ids",
            "normalized_core_component_candidate_ids",
            "selected_component_candidate_ids",
        ):
            ids.extend(_safe_component_id_map(row.get(key)).values())
        for shortage in _mapping_rows(row.get("stock_shortages")):
            ids.append(str(shortage.get("component_candidate_id") or "").strip())
        ids.extend(_string_list(row.get("unknown_component_ids")))
    return _unique([str(item).strip() for item in ids if str(item).strip()])


def _validation_repair_forbidden_combinations(
    *,
    primary_response: LlmComposerResponsePayload,
    validation_errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    errors_by_id = {
        str(row.get("recommendation_id") or "").strip(): _string_list(
            row.get("concrete_validation_reasons")
        )
        for row in validation_errors
        if str(row.get("recommendation_id") or "").strip()
    }
    combinations: list[dict[str, Any]] = []
    for recommendation in primary_response.recommendations:
        component_ids = _core_component_candidate_ids(recommendation)
        if not component_ids:
            continue
        combinations.append(
            {
                "recommendation_id": recommendation.recommendation_id,
                "component_candidate_ids": component_ids,
                "concrete_validation_reasons": errors_by_id.get(
                    recommendation.recommendation_id,
                    [],
                ),
            }
        )
    return combinations


def _validation_repair_candidate_facts_by_role(
    *,
    package: Mapping[str, Any],
    validation_errors: Sequence[Mapping[str, Any]],
    primary_response: LlmComposerResponsePayload,
) -> dict[str, Any]:
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    roles: list[str] = []
    for row in validation_errors:
        roles.extend(_string_list(row.get("missing_roles")))
        for shortage in _mapping_rows(row.get("stock_shortages")):
            roles.append(str(shortage.get("role") or "").strip())
        for mismatch in _mapping_rows(row.get("role_mismatches")):
            roles.append(str(mismatch.get("expected_role") or "").strip())
            roles.append(str(mismatch.get("actual_role") or "").strip())
        for mismatch in _mapping_rows(row.get("validation_hard_mismatches")):
            roles.append(str(mismatch.get("role") or "").strip())
    for recommendation in primary_response.recommendations:
        roles.extend(_core_component_candidate_ids(recommendation).keys())
    normalized_roles = _unique(
        [
            PROMPT_ROLE_BY_INTERNAL_ROLE.get(role, role)
            for role in roles
            if str(role or "").strip()
        ]
    )
    return {
        role: _jsonable(_mapping_rows(matrix.get(role)))
        for role in normalized_roles
        if _mapping_rows(matrix.get(role))
    }


def _filter_repeated_validation_repair_recommendations(
    recommendations: Sequence[LlmRecommendationPayload],
    proposal_indexes: Sequence[int],
    *,
    primary_response: LlmComposerResponsePayload,
) -> tuple[list[LlmRecommendationPayload], list[_RejectedProposal], list[int]]:
    forbidden_signatures = {
        _validation_repair_signature(recommendation)
        for recommendation in primary_response.recommendations
        if _validation_repair_signature(recommendation)
    }
    accepted: list[LlmRecommendationPayload] = []
    accepted_indexes: list[int] = []
    rejected: list[_RejectedProposal] = []
    for source_order, recommendation in enumerate(recommendations):
        proposal_index = (
            proposal_indexes[source_order]
            if source_order < len(proposal_indexes)
            else source_order
        )
        signature = _validation_repair_signature(recommendation)
        if signature and signature in forbidden_signatures:
            rejected.append(
                _validation_repair_repeated_combo_rejection(
                    recommendation,
                    proposal_index=proposal_index,
                )
            )
            continue
        accepted.append(recommendation)
        accepted_indexes.append(proposal_index)
    return accepted, rejected, accepted_indexes


def _validation_repair_signature(
    recommendation: LlmRecommendationPayload,
) -> tuple[tuple[str, str, int], ...]:
    component_ids = _core_component_candidate_ids(recommendation)
    return tuple(
        sorted(
            (
                str(role),
                str(component_id),
                _int_value(recommendation.quantities.get(role)) or 0,
            )
            for role, component_id in component_ids.items()
            if str(component_id).strip()
        )
    )


def _validation_repair_repeated_combo_rejection(
    recommendation: LlmRecommendationPayload,
    *,
    proposal_index: int,
) -> _RejectedProposal:
    component_ids = _core_component_candidate_ids(recommendation)
    message = "Validation repair repeated exact rejected BOM combination"
    return _RejectedProposal(
        recommendation_id=recommendation.recommendation_id,
        category="rejected_fatal",
        message=message,
        proposal_index=proposal_index,
        debug_safe={
            "proposal_index": proposal_index,
            "recommendation_id": recommendation.recommendation_id,
            "rejection_category": "fatal",
            "rejection_code": "repeated_rejected_component_combination",
            "rejection_message_ru": message,
            "validation_errors": [message],
            "concrete_validation_reasons": [
                "repeated_rejected_component_combination"
            ],
            "component_candidate_ids": component_ids,
            "normalized_core_component_candidate_ids": component_ids,
            "selected_component_candidate_ids": {},
            "missing_roles": [],
            "stock_shortages": [],
            "role_mismatches": [],
            "unknown_component_ids": [],
            "validation_hard_mismatches": [],
            "validation_unverified_requirements": [],
            "missing_required_capabilities": [],
            "stage": "validation_repair_guard",
        },
    )


def _validation_repair_response_has_component_ids(
    response: LlmComposerResponsePayload,
) -> bool:
    for recommendation in response.recommendations:
        if _core_component_candidate_ids(recommendation):
            return True
    if _string_list(response.chosen_candidate_ids):
        return True
    for row in _mapping_rows(response.selected_components):
        if str(
            row.get("component_candidate_id")
            or row.get("candidate_id")
            or row.get("id")
            or ""
        ).strip():
            return True
    return False


def _response_with_validation_repair_empty_no_recommendation_context(
    response: LlmComposerResponsePayload,
    *,
    primary_response: LlmComposerResponsePayload,
    primary_validated_pool: _ValidatedRecommendationPool,
) -> LlmComposerResponsePayload:
    errors = _validation_repair_error_rows(primary_validated_pool)
    rejected_ids = _validation_repair_rejected_candidate_ids(
        primary_response=primary_response,
        validation_errors=errors,
    )
    reason_rows = _validation_repair_reason_rows(errors)
    no_recommendation = {
        "summary": (
            "Validation repair returned no canonical BOM and no structured "
            "no_recommendation: empty_without_no_recommendation."
        ),
        "missing_roles": _unique(
            role
            for row in errors
            for role in _string_list(row.get("missing_roles"))
        ),
        "missing_required_capabilities": [
            capability
            for row in errors
            for capability in _mapping_rows(row.get("missing_required_capabilities"))
        ],
        "hard_mismatches": reason_rows,
        "stock_shortages": [
            shortage
            for row in errors
            for shortage in _mapping_rows(row.get("stock_shortages"))
        ],
        "role_analysis": _validation_repair_role_analysis_rows(errors),
        "considered_candidate_ids": {"rejected": rejected_ids},
        "validator_rejection_summary": _jsonable(
            primary_validated_pool.validation_summary
        ),
        "validator_errors": _jsonable(errors),
        "rejected_component_candidate_ids": _jsonable(rejected_ids),
        "repair_failure_reason": "empty_without_no_recommendation",
        "recommended_next_actions": [
            (
                "Return a canonical BOM with valid component_candidate_id values "
                "or escalate to engineer before quote."
            )
        ],
    }
    return response.model_copy(
        update={
            "recommendations": [],
            "no_recommendation": no_recommendation,
            "general_notes": [],
        }
    )


def _validation_repair_reason_rows(
    errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for error in errors:
        role = _validation_repair_error_role(error)
        for reason in _string_list(error.get("concrete_validation_reasons")):
            rows.append(
                {
                    "role": role,
                    "reason": reason,
                    "rejection_code": error.get("rejection_code"),
                    "recommendation_id": error.get("recommendation_id"),
                }
            )
    return _unique_mapping_rows(rows)


def _validation_repair_role_analysis_rows(
    errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for error in errors:
        rows.append(
            {
                "role": _validation_repair_error_role(error),
                "status": "rejected_by_validation",
                "reasons": _string_list(error.get("concrete_validation_reasons")),
                "component_candidate_ids": _safe_component_id_map(
                    error.get("component_candidate_ids")
                ),
            }
        )
    return _unique_mapping_rows(rows)


def _validation_repair_error_role(error: Mapping[str, Any]) -> str:
    roles = _string_list(error.get("missing_roles"))
    if roles:
        return roles[0]
    for mismatch in _mapping_rows(error.get("role_mismatches")):
        role = str(mismatch.get("expected_role") or mismatch.get("actual_role") or "")
        if role.strip():
            return role.strip()
    for mismatch in _mapping_rows(error.get("validation_hard_mismatches")):
        role = str(mismatch.get("role") or "")
        if role.strip():
            return role.strip()
    component_ids = _safe_component_id_map(error.get("component_candidate_ids"))
    if component_ids:
        return "/".join(component_ids.keys())
    return "validation"


def _response_with_validation_repair_no_recommendation_context(
    response: LlmComposerResponsePayload,
    *,
    primary_response: LlmComposerResponsePayload,
    primary_validated_pool: _ValidatedRecommendationPool,
) -> LlmComposerResponsePayload:
    no_recommendation = dict(response.no_recommendation or {})
    errors = _validation_repair_error_rows(primary_validated_pool)
    no_recommendation.setdefault(
        "summary",
        "No safe BOM after deterministic validation rejection.",
    )
    no_recommendation["validator_rejection_summary"] = _jsonable(
        primary_validated_pool.validation_summary
    )
    no_recommendation["validator_errors"] = _jsonable(errors)
    no_recommendation["rejected_component_candidate_ids"] = _jsonable(
        _validation_repair_rejected_candidate_ids(
            primary_response=primary_response,
            validation_errors=errors,
        )
    )
    return response.model_copy(update={"no_recommendation": no_recommendation})


def _merge_validation_repair_failure_pools(
    primary_pool: _ValidatedRecommendationPool,
    repair_pool: _ValidatedRecommendationPool,
) -> _ValidatedRecommendationPool:
    validation_summary = _merge_validation_summaries(
        primary_pool.validation_summary,
        repair_pool.validation_summary,
    )
    rejected_debug_safe = [
        *_validation_repair_debug_rows(primary_pool, attempt="initial"),
        *_validation_repair_debug_rows(repair_pool, attempt="repair"),
    ]
    return _ValidatedRecommendationPool(
        recommendations=[],
        accepted_recommendations=[],
        configuration_groups=[],
        quote_recommendation={},
        selected_configuration_group_id=None,
        selected_platform_option_id=None,
        selected_platform_option_index=None,
        warnings=_unique([*primary_pool.warnings, *repair_pool.warnings]),
        proposal_count=primary_pool.proposal_count + repair_pool.proposal_count,
        valid_count=primary_pool.valid_count + repair_pool.valid_count,
        validation_rejected_count=(
            primary_pool.validation_rejected_count
            + repair_pool.validation_rejected_count
        ),
        selection_skipped_count=(
            primary_pool.selection_skipped_count + repair_pool.selection_skipped_count
        ),
        rejected_count=primary_pool.rejected_count + repair_pool.rejected_count,
        validation_summary=validation_summary,
        rejected_reasons_top=_rejected_reasons_top_from_summary(validation_summary),
        rejected_debug_safe=rejected_debug_safe,
    )


def _validation_repair_debug_rows(
    pool: _ValidatedRecommendationPool,
    *,
    attempt: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _mapping_rows(pool.rejected_debug_safe):
        normalized = dict(row)
        normalized["validation_attempt"] = attempt
        normalized["concrete_validation_reasons"] = _concrete_validation_reasons(
            normalized
        )
        rows.append(normalized)
    return rows


def _merge_validation_summaries(
    *summaries: Mapping[str, Any],
) -> dict[str, int]:
    merged = {key: 0 for key in REJECTION_SUMMARY_KEYS}
    for summary in summaries:
        for key, value in _safe_mapping(summary).items():
            number = _int_value(value)
            if number is not None:
                merged[str(key)] = merged.get(str(key), 0) + number
    return merged


def _rejected_reasons_top_from_summary(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for category in REJECTION_REASON_ORDER:
        count = _int_value(summary.get(category)) or 0
        if count <= 0:
            continue
        ranked.append(
            {
                "reason": category.removeprefix("rejected_"),
                "count": count,
                "message": REJECTION_REASON_MESSAGES.get(category, category),
            }
        )
    return ranked[:5]


def _validation_row_reason(row: Mapping[str, Any]) -> str:
    return str(
        row.get("reason")
        or row.get("message")
        or row.get("user_message")
        or row.get("status")
        or row.get("type")
        or row.get("capability_id")
        or ""
    ).strip()


def _run_no_recommendation_coverage_repair_if_needed(
    *,
    package: Mapping[str, Any],
    settings: LlmSettings,
    client: LlmClient | None,
    response: LlmComposerResponsePayload,
    validated_pool: _ValidatedRecommendationPool,
    stock_candidate_index: dict[str, _IndexedStockCandidate],
    component_index: dict[str, _IndexedComponentCandidate],
    limit: int,
    evidence_pack: Mapping[str, Any] | None,
    evidence_review: Mapping[str, Any] | None,
    use_recommendation_evidence: bool,
) -> _NoRecommendationCoverageRepairResult:
    thresholds = _no_recommendation_coverage_thresholds(settings)
    if validated_pool.recommendations or not _has_structured_no_recommendation(response):
        return _NoRecommendationCoverageRepairResult(
            gate_passed=True,
            thresholds=thresholds,
        )

    coverage = _no_recommendation_coverage(
        package=package,
        no_recommendation=response.no_recommendation,
        thresholds=thresholds,
    )
    if not coverage or not bool(coverage.get("coverage_incomplete")):
        return _NoRecommendationCoverageRepairResult(
            gate_passed=True,
            coverage=coverage,
            thresholds=thresholds,
        )

    if not settings.llm_configurator_repair_enabled:
        return _NoRecommendationCoverageRepairResult(
            gate_passed=False,
            coverage_rejected=True,
            coverage=coverage,
            thresholds=thresholds,
            repair_reason="llm_repair_disabled",
            warnings=["no_recommendation_coverage_repair: disabled"],
        )
    if client is None:
        return _NoRecommendationCoverageRepairResult(
            gate_passed=False,
            coverage_rejected=True,
            coverage=coverage,
            thresholds=thresholds,
            repair_reason="llm_repair_client_unavailable",
            warnings=["no_recommendation_coverage_repair: client unavailable"],
        )

    repair_package = _no_recommendation_coverage_repair_package(
        package=package,
        primary_response=response,
        coverage=coverage,
        thresholds=thresholds,
    )
    try:
        payload = client.generate_json(
            LLM_NO_RECOMMENDATION_COVERAGE_REPAIR_SYSTEM_PROMPT,
            build_llm_configurator_user_prompt(repair_package),
        )
    except LlmReadTimeoutError as exc:
        return _NoRecommendationCoverageRepairResult(
            gate_passed=False,
            repair_attempted=True,
            coverage_rejected=True,
            coverage=coverage,
            thresholds=thresholds,
            repair_reason="llm_repair_read_timeout",
            error_type=type(exc).__name__,
            parse_status="request_failed",
            warnings=[f"no_recommendation_coverage_repair: {type(exc).__name__}"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )
    except LlmInvalidJsonError as exc:
        return _NoRecommendationCoverageRepairResult(
            gate_passed=False,
            repair_attempted=True,
            coverage_rejected=True,
            coverage=coverage,
            thresholds=thresholds,
            repair_reason="llm_repair_invalid_json",
            error_type=type(exc).__name__,
            parse_status="parse_error",
            warnings=[f"no_recommendation_coverage_repair: {type(exc).__name__}"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )
    except LlmError as exc:
        return _NoRecommendationCoverageRepairResult(
            gate_passed=False,
            repair_attempted=True,
            coverage_rejected=True,
            coverage=coverage,
            thresholds=thresholds,
            repair_reason="llm_repair_request_failed",
            error_type=type(exc).__name__,
            parse_status="request_failed",
            warnings=[f"no_recommendation_coverage_repair: {type(exc).__name__}"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    normalized_payload = _normalize_composer_payload(payload, package=package)
    try:
        parsed_payload = _parse_composer_payload(normalized_payload)
    except ValidationError:
        return _NoRecommendationCoverageRepairResult(
            gate_passed=False,
            repair_attempted=True,
            coverage_rejected=True,
            coverage=coverage,
            thresholds=thresholds,
            repair_reason="llm_repair_validation_failed",
            revised_proposals_count=_raw_proposals_count(normalized_payload),
            error_type="ValidationError",
            parse_status="validation_error",
            warnings=["no_recommendation_coverage_repair: ValidationError"],
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    if parsed_payload.response.recommendations:
        revised_pool = _validate_recommendations(
            parsed_payload.response.recommendations,
            stock_candidate_index=stock_candidate_index,
            component_index=component_index,
            user_request=package["user_request"],
            normalized_requirements=package["normalized_requirements"],
            limit=limit,
            evidence_pack=evidence_pack,
            evidence_review=evidence_review,
            use_recommendation_evidence=use_recommendation_evidence,
            schema_rejections=parsed_payload.schema_rejections,
            proposal_indexes=parsed_payload.proposal_indexes,
            proposal_count=parsed_payload.proposal_count,
        )
        if revised_pool.recommendations:
            return _NoRecommendationCoverageRepairResult(
                gate_passed=True,
                repair_attempted=True,
                repair_success=True,
                coverage=coverage,
                thresholds=thresholds,
                repair_reason="repair_returned_valid_bom",
                response=parsed_payload.response,
                validated_pool=revised_pool,
                revised_proposals_count=revised_pool.proposal_count,
                validation_summary=revised_pool.validation_summary,
                warnings=[
                    "no_recommendation_coverage_repair: valid BOM accepted by validator"
                ],
                parse_status="parsed",
                thinking_diagnostics=_client_thinking_diagnostics(client),
            )
        return _NoRecommendationCoverageRepairResult(
            gate_passed=False,
            repair_attempted=True,
            coverage_rejected=True,
            coverage=coverage,
            thresholds=thresholds,
            repair_reason="repair_bom_validation_failed",
            revised_proposals_count=revised_pool.proposal_count,
            validation_summary=revised_pool.validation_summary,
            warnings=["no_recommendation_coverage_repair: repaired BOM rejected"],
            parse_status="parsed",
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    if _has_structured_no_recommendation(parsed_payload.response):
        repaired_coverage = _no_recommendation_coverage(
            package=package,
            no_recommendation=parsed_payload.response.no_recommendation,
            thresholds=thresholds,
        )
        if repaired_coverage and not bool(repaired_coverage.get("coverage_incomplete")):
            return _NoRecommendationCoverageRepairResult(
                gate_passed=True,
                repair_attempted=True,
                repair_success=True,
                coverage=repaired_coverage,
                thresholds=thresholds,
                repair_reason="repair_returned_sufficient_no_recommendation",
                response=parsed_payload.response,
                validated_pool=validated_pool,
                revised_proposals_count=parsed_payload.proposal_count,
                parse_status="parsed",
                thinking_diagnostics=_client_thinking_diagnostics(client),
            )
        return _NoRecommendationCoverageRepairResult(
            gate_passed=False,
            repair_attempted=True,
            coverage_rejected=True,
            coverage=repaired_coverage or coverage,
            thresholds=thresholds,
            repair_reason="repair_no_recommendation_coverage_incomplete",
            response=parsed_payload.response,
            validated_pool=validated_pool,
            revised_proposals_count=parsed_payload.proposal_count,
            warnings=[
                "no_recommendation_coverage_repair: repaired no_recommendation "
                "still has incomplete matrix coverage"
            ],
            parse_status="parsed",
            thinking_diagnostics=_client_thinking_diagnostics(client),
        )

    return _NoRecommendationCoverageRepairResult(
        gate_passed=False,
        repair_attempted=True,
        coverage_rejected=True,
        coverage=coverage,
        thresholds=thresholds,
        repair_reason="repair_returned_empty_response",
        revised_proposals_count=parsed_payload.proposal_count,
        warnings=["no_recommendation_coverage_repair: empty response after repair"],
        parse_status="parsed",
        thinking_diagnostics=_client_thinking_diagnostics(client),
    )


def _no_recommendation_coverage_repair_package(
    *,
    package: Mapping[str, Any],
    primary_response: LlmComposerResponsePayload,
    coverage: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(package),
        "no_recommendation_coverage_repair_attempt": 1,
        "original_no_recommendation": _jsonable(primary_response.no_recommendation),
        "no_recommendation_coverage": _jsonable(coverage),
        "no_recommendation_coverage_thresholds": _jsonable(thresholds),
        "repair_instructions": [
            NO_RECOMMENDATION_COVERAGE_REPAIR_INSTRUCTION,
            "Use all candidates present in component_candidate_matrix for each role.",
            "Do not invent component_candidate_id values.",
            "Keep prices, stock, quantities, and totals from application code only.",
            "Do not rerun or alter semantic planner, category planner, or matrix builder output.",
        ],
    }


def _no_recommendation_coverage_repair_diagnostics(
    result: _NoRecommendationCoverageRepairResult,
) -> dict[str, Any]:
    diagnostics = {
        "no_recommendation_coverage_gate_passed": bool(result.gate_passed),
        "no_recommendation_coverage_repair_attempted": bool(result.repair_attempted),
        "no_recommendation_coverage_repair_success": bool(result.repair_success),
        "no_recommendation_coverage_rejected": bool(result.coverage_rejected),
        "no_recommendation_coverage_thresholds": _jsonable(result.thresholds),
    }
    if result.repair_reason:
        diagnostics["no_recommendation_coverage_repair_reason"] = result.repair_reason
    if result.error_type:
        diagnostics["no_recommendation_coverage_repair_error_type"] = result.error_type
    if result.parse_status:
        diagnostics["no_recommendation_coverage_repair_parse_status"] = result.parse_status
    if result.coverage:
        diagnostics["no_recommendation_coverage"] = _jsonable(result.coverage)
    if result.coverage_rejected:
        diagnostics["structured_no_recommendation_used"] = False
        diagnostics["structured_no_recommendation_coverage_rejected"] = True
    elif result.repair_success:
        diagnostics["structured_no_recommendation_coverage_rejected"] = False
    return diagnostics


def _merge_evidence_pack_diagnostics(
    evidence_pack: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    pack = dict(evidence_pack or {})
    existing = _safe_mapping(pack.get("diagnostics"))
    pack["diagnostics"] = {**existing, **_safe_mapping(diagnostics)}
    return pack


def _build_repair_package(
    *,
    package: Mapping[str, Any],
    primary_response: LlmComposerResponsePayload,
    primary_validated_pool: _ValidatedRecommendationPool,
    critique: _RepairCritique,
    output_mode: str,
    original_recommendations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    original_proposals = list(
        original_recommendations or primary_validated_pool.accepted_recommendations
    )
    single_mode = _single_best_output_mode(output_mode)
    repair_instructions = [
        "Revise the single primary recommendation only.",
        "Do not return alternatives.",
        "Use cheaper_equivalent facts if they improve cheapest valid quote.",
        "Do not use blocked/matrix_note candidates.",
        "Unknown hard compatibility is not equivalent.",
        "Keep one primary recommendation.",
        "JSON only.",
    ] if single_mode else [
        "Return a revised proposal pool in the same schema.",
        "Use only IDs from original_accepted_proposals or allowed_candidate_alternatives.",
        "Use only critique facts with classification=cheaper_equivalent.",
        "Do not choose matrix_note or engineer_check alternatives for cheapest_quote.",
        "Unknown hard compatibility is not equivalent.",
        "Keep cheapest valid quote first; keep branded-safe options as alternatives.",
        "Do not include raw reasoning, markdown, or <think>.",
    ]
    return {
        "repair_attempt": 1,
        "output_mode": _normalize_output_mode(output_mode),
        "user_request": package.get("user_request"),
        "normalized_requirements": package.get("normalized_requirements"),
        "optimization_mode": "cost_minimal_valid_fit",
        "proposal_pool_limit": 1 if single_mode else package.get("proposal_pool_limit"),
        "final_display_limit": 1 if single_mode else package.get("final_display_limit"),
        "original_accepted_proposals": [
            _repair_original_proposal(candidate)
            for candidate in original_proposals
        ],
        "original_general_notes": _string_list(primary_response.general_notes),
        "critique_facts": critique.facts,
        "allowed_candidate_alternatives": critique.alternatives_by_role,
        "repair_instructions": repair_instructions,
    }


def _repair_allowed_component_candidate_ids(
    repair_package: Mapping[str, Any],
) -> set[str]:
    allowed: set[str] = set()
    for proposal in _mapping_rows(repair_package.get("original_accepted_proposals")):
        allowed.update(_safe_component_id_map(proposal.get("component_candidate_ids")).values())
        for component in _mapping_rows(proposal.get("components")):
            component_id = str(component.get("component_candidate_id") or "").strip()
            if component_id:
                allowed.add(component_id)
    alternatives = repair_package.get("allowed_candidate_alternatives")
    if isinstance(alternatives, Mapping):
        for rows in alternatives.values():
            for row in _mapping_rows(rows):
                component_id = str(row.get("component_candidate_id") or "").strip()
                if component_id:
                    allowed.add(component_id)
    return {component_id for component_id in allowed if component_id}


def _filter_repair_recommendations_by_allowed_component_ids(
    recommendations: Sequence[LlmRecommendationPayload],
    proposal_indexes: Sequence[int],
    *,
    allowed_component_ids: set[str],
) -> tuple[list[LlmRecommendationPayload], list[_RejectedProposal], list[int]]:
    valid: list[LlmRecommendationPayload] = []
    rejected: list[_RejectedProposal] = []
    valid_indexes: list[int] = []
    for source_order, recommendation in enumerate(recommendations):
        proposal_index = (
            proposal_indexes[source_order]
            if source_order < len(proposal_indexes)
            else source_order
        )
        component_ids = _repair_recommendation_component_ids(recommendation)
        disallowed = [
            component_id
            for component_id in component_ids
            if component_id not in allowed_component_ids
        ]
        if disallowed:
            rejected.append(
                _RejectedProposal(
                    recommendation_id=recommendation.recommendation_id,
                    category="rejected_unknown_component",
                    message=(
                        f"{recommendation.recommendation_id}: repair used "
                        "component_candidate_id outside allowed alternatives"
                    ),
                    proposal_index=proposal_index,
                    debug_safe={
                        "proposal_index": proposal_index,
                        "recommendation_id": recommendation.recommendation_id,
                        "rejection_category": "repair_disallowed_component",
                        "rejection_code": "repair_disallowed_component",
                        "validation_errors": [
                            "repair used component_candidate_id outside allowed alternatives"
                        ],
                        "disallowed_component_ids_count": len(disallowed),
                    },
                )
            )
            continue
        valid.append(recommendation)
        valid_indexes.append(proposal_index)
    return valid, rejected, valid_indexes


def _repair_recommendation_component_ids(
    recommendation: LlmRecommendationPayload,
) -> list[str]:
    ids: list[str] = []
    for component_ids in (
        recommendation.component_candidate_ids,
        recommendation.selected_component_candidate_ids,
        recommendation.optional_component_candidate_ids,
        recommendation.engineer_check_component_candidate_ids,
    ):
        ids.extend(
            str(component_id or "").strip()
            for component_id in component_ids.values()
            if str(component_id or "").strip()
        )
    return _unique(ids)


def _repair_original_proposal(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "recommendation_id": str(
            candidate.get("recommendation_id") or candidate.get("candidate_id") or ""
        ),
        "proposal_role": candidate.get("proposal_role"),
        "recommendation_slot": candidate.get("recommendation_slot"),
        "source_type": candidate.get("source_type") or candidate.get("candidate_type"),
        "source_candidate_id": candidate.get("source_candidate_id"),
        "component_candidate_ids": _safe_component_id_map(
            candidate.get("component_candidate_ids")
        ),
        "quantities": _safe_mapping(candidate.get("quantities")),
        "total_price_value": candidate.get("total_price_value"),
        "total_price_currency": candidate.get("total_price_currency"),
        "components": [
            _repair_public_component(component)
            for component in _mapping_rows(candidate.get("components"))
            if component.get("component_candidate_id")
        ],
        "why_selected": _safe_diagnostic_text(candidate.get("why_selected"), limit=240),
        "commercial_tradeoff": _safe_diagnostic_text(
            candidate.get("commercial_tradeoff"),
            limit=240,
        ),
    }


def _repair_public_component(component: Mapping[str, Any]) -> dict[str, Any]:
    facts = component.get("facts") if isinstance(component.get("facts"), Mapping) else {}
    return {
        "role": component.get("role"),
        "component_candidate_id": component.get("component_candidate_id"),
        "producer": component.get("producer"),
        "part_number": component.get("part_number"),
        "name": component.get("item_name"),
        "quantity_required": component.get("quantity_required"),
        "available_quantity": component.get("available_quantity"),
        "price_value": component.get("price_value"),
        "price_currency": component.get("price_currency"),
        "line_total_value": component.get("line_total_value"),
        "line_total_currency": component.get("line_total_currency"),
        "facts": _public_facts(facts),
        "cpu_cores": component.get("cpu_cores"),
        "storage_capacity_tb": component.get("storage_capacity_tb"),
        "ram_module_capacity_gb": component.get("ram_module_capacity_gb"),
        "ram_total_gb_per_server": component.get("ram_total_gb_per_server"),
    }


def _build_repair_critique(
    *,
    accepted_recommendations: Sequence[Mapping[str, Any]],
    component_index: Mapping[str, _IndexedComponentCandidate],
    normalized_requirements: Any,
    max_alternatives_per_role: int,
) -> _RepairCritique:
    facts: list[dict[str, Any]] = []
    alternatives_by_role: dict[str, list[dict[str, Any]]] = {}
    savings: list[Decimal] = []
    summary: list[str] = []
    blocked_facts: list[dict[str, Any]] = []
    requirements = _first_requirements(normalized_requirements)

    for proposal in accepted_recommendations:
        selected = _proposal_selected_components(proposal, component_index)
        quantities = _proposal_component_quantities(proposal)
        for role in (RAM_ROLE, SSD_ROLE, HDD_ROLE):
            selected_candidate = selected.get(role)
            quantity = quantities.get(role)
            if selected_candidate is None or quantity is None:
                continue
            alternative = _cheaper_equivalent_component(
                role=role,
                selected_candidate=selected_candidate,
                selected=selected,
                quantity=quantity,
                component_index=component_index,
                requirements=requirements,
            )
            blocked_facts.extend(
                _repair_blocked_critique_facts(
                    proposal=proposal,
                    role=role,
                    selected_candidate=selected_candidate,
                    blocked=alternative.blocked,
                    quantity=quantity,
                )
            )
            if alternative.allowed is None:
                continue
            fact = _repair_critique_fact(
                proposal=proposal,
                role=role,
                issue="cheaper_equivalent_available",
                selected_candidate=selected_candidate,
                alternative_candidate=alternative.allowed,
                quantity=quantity,
                requirements=requirements,
            )
            if fact is not None:
                facts.append(fact)
                savings.append(_decimal_value(fact.get("saving_value")) or Decimal("0"))
                summary.append(str(fact["summary"]))
                _append_repair_alternative(
                    alternatives_by_role,
                    alternative.allowed,
                    role=role,
                    max_per_role=max_alternatives_per_role,
                )

        platform = selected.get(SERVER_PLATFORM_ROLE)
        platform_quantity = quantities.get(SERVER_PLATFORM_ROLE)
        if (
            platform is not None
            and platform_quantity is not None
            and _proposal_is_cost_sensitive(proposal)
        ):
            platform_alternative = _cheaper_compatible_platform(
                selected_platform=platform,
                selected=selected,
                quantity=platform_quantity,
                component_index=component_index,
                requirements=requirements,
            )
            blocked_facts.extend(
                _repair_blocked_critique_facts(
                    proposal=proposal,
                    role=SERVER_PLATFORM_ROLE,
                    selected_candidate=platform,
                    blocked=platform_alternative.blocked,
                    quantity=platform_quantity,
                )
            )
            if platform_alternative.allowed is not None:
                fact = _repair_critique_fact(
                    proposal=proposal,
                    role=SERVER_PLATFORM_ROLE,
                    issue="cheaper_equivalent_available",
                    selected_candidate=platform,
                    alternative_candidate=platform_alternative.allowed,
                    quantity=platform_quantity,
                    requirements=requirements,
                )
                if fact is not None:
                    facts.append(fact)
                    savings.append(
                        _decimal_value(fact.get("saving_value")) or Decimal("0")
                    )
                    summary.append(str(fact["summary"]))
                    _append_repair_alternative(
                        alternatives_by_role,
                        platform_alternative.allowed,
                        role=SERVER_PLATFORM_ROLE,
                        max_per_role=max_alternatives_per_role,
                    )

        cpu = selected.get(CPU_ROLE)
        cpu_quantity = quantities.get(CPU_ROLE)
        if cpu is not None and cpu_quantity is not None:
            cpu_alternative = _cheaper_cpu_over_requirement_alternative(
                selected_cpu=cpu,
                selected=selected,
                quantity=cpu_quantity,
                component_index=component_index,
                requirements=requirements,
            )
            blocked_facts.extend(
                _repair_blocked_critique_facts(
                    proposal=proposal,
                    role=CPU_ROLE,
                    selected_candidate=cpu,
                    blocked=cpu_alternative.blocked,
                    quantity=cpu_quantity,
                )
            )
            if cpu_alternative.allowed is not None:
                fact = _repair_critique_fact(
                    proposal=proposal,
                    role=CPU_ROLE,
                    issue="cheaper_equivalent_available",
                    selected_candidate=cpu,
                    alternative_candidate=cpu_alternative.allowed,
                    quantity=cpu_quantity,
                    requirements=requirements,
                )
                if fact is not None:
                    facts.append(fact)
                    savings.append(
                        _decimal_value(fact.get("saving_value")) or Decimal("0")
                    )
                    summary.append(str(fact["summary"]))
                    _append_repair_alternative(
                        alternatives_by_role,
                        cpu_alternative.allowed,
                        role=CPU_ROLE,
                        max_per_role=max_alternatives_per_role,
                    )

    total_savings = sum(savings, Decimal("0")) if savings else None
    blocked_summary = [
        str(fact["summary"])
        for fact in blocked_facts
        if fact.get("summary")
    ][:10]
    return _RepairCritique(
        facts=facts,
        alternatives_by_role=alternatives_by_role,
        savings_estimate=total_savings,
        summary=summary[:10],
        blocked_facts=blocked_facts,
        blocked_summary=blocked_summary,
    )


def _proposal_selected_components(
    proposal: Mapping[str, Any],
    component_index: Mapping[str, _IndexedComponentCandidate],
) -> dict[str, _IndexedComponentCandidate]:
    selected: dict[str, _IndexedComponentCandidate] = {}
    for component in _mapping_rows(proposal.get("components")):
        role = _normalize_role(component.get("role"))
        component_id = str(component.get("component_candidate_id") or "").strip()
        candidate = component_index.get(component_id)
        if role is not None and candidate is not None and candidate.internal_role == role:
            selected[role] = candidate
    return selected


def _proposal_component_quantities(proposal: Mapping[str, Any]) -> dict[str, int]:
    quantities: dict[str, int] = {}
    for component in _mapping_rows(proposal.get("components")):
        role = _normalize_role(component.get("role"))
        quantity = _int_value(component.get("quantity_required"))
        if role is not None and quantity is not None and quantity > 0:
            quantities[role] = quantity
    return quantities


def _cheaper_equivalent_component(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    component_index: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> _RepairAlternativeSearchResult:
    selected_line_price = _line_price(selected_candidate, quantity)
    if selected_line_price is None:
        return _RepairAlternativeSearchResult(allowed=None, blocked=[])
    alternatives: list[_IndexedComponentCandidate] = []
    blocked: list[tuple[_IndexedComponentCandidate, _RoleEligibility]] = []
    for candidate in component_index.values():
        if candidate.internal_role != role:
            continue
        if candidate.component_candidate_id == selected_candidate.component_candidate_id:
            continue
        if _candidate_incompatible_with_selection(
            role=role,
            candidate=candidate,
            selected=selected,
        ):
            continue
        line_price = _line_price(candidate, quantity)
        if line_price is None or line_price >= selected_line_price:
            continue
        eligibility = evaluate_role_eligibility(
            role=role,
            selected_candidate=selected_candidate,
            candidate=candidate,
            selected=selected,
            quantity=quantity,
            requirements=requirements,
        )
        if eligibility.is_equivalent:
            alternatives.append(candidate)
        else:
            blocked.append((candidate, eligibility))
    return _RepairAlternativeSearchResult(
        allowed=(
            min(alternatives, key=lambda item: _line_price(item, quantity))
            if alternatives
            else None
        ),
        blocked=blocked,
    )


def _same_role_capacity_and_type(
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    requirements: Mapping[str, Any],
) -> bool:
    if role == RAM_ROLE:
        selected_gb = _candidate_ram_module_gb(selected_candidate)
        candidate_gb = _candidate_ram_module_gb(candidate)
        if selected_gb is None or candidate_gb is None or selected_gb != candidate_gb:
            return False
        required_type = _normalize_ram_type(requirements.get("ram_type_preference"))
        selected_type = _normalized_repair_ram_type(selected_candidate)
        candidate_type = _normalized_repair_ram_type(candidate)
        if not _is_same_ram_generation(
            selected=selected_type,
            candidate=candidate_type,
            required_generation=required_type,
        ):
            return False
        return _is_server_ram_equivalent_class(selected_type, candidate_type)
    if role in {SSD_ROLE, HDD_ROLE}:
        selected_tb = _candidate_storage_tb(selected_candidate)
        candidate_tb = _candidate_storage_tb(candidate)
        if selected_tb is None or candidate_tb is None:
            return False
        if abs(selected_tb - candidate_tb) > 0.001:
            return False
        required_interface = str(
            requirements.get("storage_interface_preference") or ""
        ).strip()
        selected_interface = _fact(selected_candidate, "storage_interface")
        candidate_interface = _fact(candidate, "storage_interface")
        return _same_or_unknown_type(
            selected_interface,
            candidate_interface,
            required_interface,
        )
    return False


def _same_or_unknown_type(left: str, right: str, required: str) -> bool:
    if required and required != UNKNOWN_FACT:
        required_normalized = required.casefold()
        return left.casefold() == required_normalized and right.casefold() == required_normalized
    if left != UNKNOWN_FACT and right != UNKNOWN_FACT:
        return left.casefold() == right.casefold()
    return True


def _cheaper_compatible_platform(
    *,
    selected_platform: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    component_index: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> _RepairAlternativeSearchResult:
    selected_line_price = _line_price(selected_platform, quantity)
    if selected_line_price is None:
        return _RepairAlternativeSearchResult(allowed=None, blocked=[])
    alternatives: list[_IndexedComponentCandidate] = []
    blocked: list[tuple[_IndexedComponentCandidate, _RoleEligibility]] = []
    for candidate in component_index.values():
        if candidate.internal_role != SERVER_PLATFORM_ROLE:
            continue
        if candidate.component_candidate_id == selected_platform.component_candidate_id:
            continue
        line_price = _line_price(candidate, quantity)
        if line_price is None or line_price >= selected_line_price:
            continue
        eligibility = evaluate_role_eligibility(
            role=SERVER_PLATFORM_ROLE,
            selected_candidate=selected_platform,
            candidate=candidate,
            selected=selected,
            quantity=quantity,
            requirements=requirements,
        )
        if eligibility.is_equivalent:
            alternatives.append(candidate)
        else:
            blocked.append((candidate, eligibility))
    return _RepairAlternativeSearchResult(
        allowed=(
            min(alternatives, key=lambda item: _line_price(item, quantity))
            if alternatives
            else None
        ),
        blocked=blocked,
    )


def _same_architecture_family(
    selected_platform: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
) -> bool:
    cpu = selected.get(CPU_ROLE)
    socket = _fact(cpu, "cpu_socket") if cpu is not None else _fact(selected_platform, "cpu_socket")
    selected_socket = _fact(selected_platform, "cpu_socket")
    candidate_socket = _fact(candidate, "cpu_socket")
    if socket != UNKNOWN_FACT and candidate_socket != UNKNOWN_FACT and candidate_socket != socket:
        return False
    if (
        selected_socket != UNKNOWN_FACT
        and candidate_socket != UNKNOWN_FACT
        and candidate_socket != selected_socket
    ):
        return False
    for key in ("cpu_brand", "cpu_family"):
        selected_value = _fact(selected_platform, key)
        candidate_value = _fact(candidate, key)
        if (
            selected_value != UNKNOWN_FACT
            and candidate_value != UNKNOWN_FACT
            and selected_value != candidate_value
        ):
            return False
    selected_ram_type = _normalized_repair_ram_type(selected_platform).generation
    candidate_ram_type = _normalized_repair_ram_type(candidate).generation
    return not (
        selected_ram_type
        and candidate_ram_type
        and selected_ram_type != candidate_ram_type
    )


def _platform_compatible_with_selected(
    platform: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> bool:
    cpu = selected.get(CPU_ROLE)
    if cpu is not None and _fatal_platform_cpu_mismatch(platform, cpu):
        return False
    ram = selected.get(RAM_ROLE)
    if ram is not None:
        platform_ram_type = _normalized_repair_ram_type(platform).generation
        ram_type = _normalized_repair_ram_type(ram).generation
        if platform_ram_type and ram_type and platform_ram_type != ram_type:
            return False
    required_interface = str(
        requirements.get("storage_interface_preference") or ""
    ).strip().upper()
    return not (
        required_interface == "NVME"
        and _explicit_false_fact(platform, "nvme_support")
    )


def _explicit_false_fact(candidate: _IndexedComponentCandidate, key: str) -> bool:
    facts = candidate.row.get("extracted_facts")
    value = facts.get(key) if isinstance(facts, Mapping) else candidate.row.get(key)
    if isinstance(value, bool):
        return value is False
    if value is None or value == "":
        return False
    return str(value).strip().casefold() in {"false", "no", "0"}


def _cheaper_cpu_over_requirement_alternative(
    *,
    selected_cpu: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    component_index: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> _RepairAlternativeSearchResult:
    required_cores = _required_cpu_cores(requirements)
    selected_cores = _candidate_cpu_cores(selected_cpu)
    if (
        required_cores is None
        or selected_cores is None
        or selected_cores <= required_cores
    ):
        return _RepairAlternativeSearchResult(allowed=None, blocked=[])
    selected_line_price = _line_price(selected_cpu, quantity)
    if selected_line_price is None:
        return _RepairAlternativeSearchResult(allowed=None, blocked=[])
    alternatives: list[_IndexedComponentCandidate] = []
    blocked: list[tuple[_IndexedComponentCandidate, _RoleEligibility]] = []
    for candidate in component_index.values():
        if candidate.internal_role != CPU_ROLE:
            continue
        if candidate.component_candidate_id == selected_cpu.component_candidate_id:
            continue
        candidate_cores = _candidate_cpu_cores(candidate)
        if candidate_cores is None or candidate_cores < required_cores:
            continue
        if candidate_cores > selected_cores:
            continue
        line_price = _line_price(candidate, quantity)
        if line_price is None or line_price >= selected_line_price:
            continue
        eligibility = evaluate_role_eligibility(
            role=CPU_ROLE,
            selected_candidate=selected_cpu,
            candidate=candidate,
            selected=selected,
            quantity=quantity,
            requirements=requirements,
        )
        if eligibility.is_equivalent:
            alternatives.append(candidate)
        else:
            blocked.append((candidate, eligibility))
    return _RepairAlternativeSearchResult(
        allowed=(
            min(
                alternatives,
                key=lambda item: (
                    _candidate_cpu_cores(item) or 0,
                    _line_price(item, quantity),
                ),
            )
            if alternatives
            else None
        ),
        blocked=blocked,
    )


def is_role_equivalent_candidate(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    requirements: Mapping[str, Any],
) -> bool:
    return evaluate_role_eligibility(
        role=role,
        selected_candidate=selected_candidate,
        candidate=candidate,
        selected=selected,
        quantity=quantity,
        requirements=requirements,
    ).is_equivalent


def evaluate_role_eligibility(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    requirements: Mapping[str, Any],
) -> _RoleEligibility:
    if role == SERVER_PLATFORM_ROLE:
        return evaluate_platform_repair_eligibility(
            selected_platform=selected_candidate,
            candidate=candidate,
            selected=selected,
            quantity=quantity,
            requirements=requirements,
        )
    if role == RAM_ROLE:
        return evaluate_ram_repair_eligibility(
            selected_ram=selected_candidate,
            candidate=candidate,
            selected=selected,
            quantity=quantity,
            requirements=requirements,
        )
    if role in {SSD_ROLE, HDD_ROLE}:
        return evaluate_storage_repair_eligibility(
            role=role,
            selected_storage=selected_candidate,
            candidate=candidate,
            selected=selected,
            quantity=quantity,
            requirements=requirements,
        )
    if role == CPU_ROLE:
        return evaluate_cpu_repair_eligibility(
            selected_cpu=selected_candidate,
            candidate=candidate,
            selected=selected,
            quantity=quantity,
            requirements=requirements,
        )
    return _RoleEligibility(
        classification="not_equivalent_requires_engineering_review",
        reason_codes=["role_not_supported"],
    )


def evaluate_platform_repair_eligibility(
    *,
    selected_platform: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    requirements: Mapping[str, Any],
) -> _RoleEligibility:
    reason_codes: list[str] = []
    if candidate.internal_role != SERVER_PLATFORM_ROLE:
        reason_codes.append("role_mismatch")
    if not _has_sufficient_stock(candidate, quantity):
        reason_codes.append("stock_shortage")
    if not _same_architecture_family(selected_platform, candidate, selected):
        reason_codes.append("cpu_family_mismatch")
    if not _platform_compatible_with_selected(candidate, selected, requirements):
        reason_codes.append("cpu_family_mismatch")

    requested_form_factor = str(requirements.get("form_factor") or "").strip()
    if requested_form_factor and not _platform_form_factor_confirmed(
        candidate,
        requested_form_factor,
    ):
        reason_codes.append("platform_form_factor_unknown")

    required_sockets = _int_value(requirements.get("cpu_per_server"))
    if required_sockets is not None and required_sockets > 0:
        socket_count = _platform_cpu_socket_count(candidate)
        if socket_count is None or socket_count < required_sockets:
            reason_codes.append("platform_cpu_socket_unknown")

    selected_cpu = selected.get(CPU_ROLE)
    if selected_cpu is not None:
        selected_side = _cpu_component_side(selected_cpu)
        platform_side = _cpu_platform_side(candidate)
        if selected_side != UNKNOWN_FACT:
            if platform_side == UNKNOWN_FACT:
                reason_codes.append("platform_cpu_socket_unknown")
            elif platform_side != selected_side:
                reason_codes.append("cpu_family_mismatch")
        selected_socket = _cpu_socket(selected_cpu)
        platform_socket = _cpu_socket(candidate)
        if selected_socket != UNKNOWN_FACT:
            if platform_socket == UNKNOWN_FACT:
                reason_codes.append("platform_cpu_socket_unknown")
            elif platform_socket != selected_socket:
                reason_codes.append("cpu_family_mismatch")

    required_ram_type = _required_ram_type(requirements, selected.get(RAM_ROLE))
    platform_ram_type = _normalized_repair_ram_type(candidate).generation
    if required_ram_type:
        if not platform_ram_type:
            reason_codes.append("platform_ram_type_unknown")
        elif platform_ram_type != required_ram_type:
            reason_codes.append("ram_type_mismatch")

    required_storage_interface = _required_storage_interface(
        requirements,
        selected.get(SSD_ROLE) or selected.get(HDD_ROLE),
    )
    if required_storage_interface == "NVME":
        if _explicit_false_fact(candidate, "nvme_support"):
            reason_codes.append("storage_interface_mismatch")
        elif not _platform_nvme_confirmed(candidate):
            reason_codes.append("platform_storage_unknown")

    if _platform_incomplete_chassis(candidate):
        reason_codes.append("platform_incomplete_chassis")

    reason_codes.extend(
        _candidate_hard_warning_reason_codes(candidate, SERVER_PLATFORM_ROLE)
    )
    return _role_eligibility_from_reasons(reason_codes)


def evaluate_ram_repair_eligibility(
    *,
    selected_ram: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    requirements: Mapping[str, Any],
) -> _RoleEligibility:
    reason_codes: list[str] = []
    if candidate.internal_role != RAM_ROLE:
        reason_codes.append("role_mismatch")
    if not _has_sufficient_stock(candidate, quantity):
        reason_codes.append("stock_shortage")

    required_type = _required_ram_type(requirements, selected_ram)
    selected_type = _normalized_repair_ram_type(selected_ram)
    candidate_type = _normalized_repair_ram_type(candidate)
    if not selected_type.generation or not candidate_type.generation:
        reason_codes.append("ram_type_unknown")
    elif not _is_same_ram_generation(
        selected=selected_type,
        candidate=candidate_type,
        required_generation=required_type,
    ):
        reason_codes.append("ram_type_mismatch")
    class_reason = _ram_class_block_reason(selected_type, candidate_type)
    if class_reason:
        reason_codes.append(class_reason)

    selected_gb = _candidate_ram_module_gb(selected_ram)
    candidate_gb = _candidate_ram_module_gb(candidate)
    if selected_gb is None or candidate_gb is None:
        reason_codes.append("ram_capacity_unknown")
    elif candidate_gb != selected_gb:
        reason_codes.append("ram_capacity_mismatch")

    reason_codes.extend(_candidate_hard_warning_reason_codes(candidate, RAM_ROLE))
    return _role_eligibility_from_reasons(reason_codes)


def evaluate_storage_repair_eligibility(
    *,
    role: str,
    selected_storage: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    requirements: Mapping[str, Any],
) -> _RoleEligibility:
    reason_codes: list[str] = []
    if candidate.internal_role != role:
        reason_codes.append("role_mismatch")
    if not _has_sufficient_stock(candidate, quantity):
        reason_codes.append("stock_shortage")

    required_tb = _required_storage_tb(requirements)
    selected_tb = _candidate_storage_tb(selected_storage)
    candidate_tb = _candidate_storage_tb(candidate)
    min_tb = required_tb if required_tb is not None else selected_tb
    if candidate_tb is None or (min_tb is not None and candidate_tb < min_tb):
        reason_codes.append("storage_capacity_mismatch")

    required_interface = _required_storage_interface(requirements, selected_storage)
    candidate_interface = _normalize_storage_interface(
        _fact(candidate, "storage_interface")
    )
    if required_interface:
        if candidate_interface != required_interface:
            reason_codes.append("storage_interface_mismatch")
    elif not candidate_interface:
        reason_codes.append("storage_interface_mismatch")

    platform = selected.get(SERVER_PLATFORM_ROLE)
    if (
        required_interface == "NVME"
        and platform is not None
        and not _platform_nvme_confirmed(platform)
    ):
        reason_codes.append("platform_storage_unknown")

    reason_codes.extend(_candidate_hard_warning_reason_codes(candidate, role))
    return _role_eligibility_from_reasons(reason_codes)


def evaluate_cpu_repair_eligibility(
    *,
    selected_cpu: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantity: int,
    requirements: Mapping[str, Any],
) -> _RoleEligibility:
    reason_codes: list[str] = []
    required_cores = _required_cpu_cores(requirements)
    selected_cores = _candidate_cpu_cores(selected_cpu)
    candidate_cores = _candidate_cpu_cores(candidate)
    if candidate.internal_role != CPU_ROLE:
        reason_codes.append("role_mismatch")
    if not _has_sufficient_stock(candidate, quantity):
        reason_codes.append("stock_shortage")
    if (
        required_cores is None
        or selected_cores is None
        or selected_cores <= required_cores
        or candidate_cores is None
        or candidate_cores < required_cores
    ):
        reason_codes.append("cpu_hard_requirement_not_met")

    required_vendor = str(requirements.get("cpu_vendor_preference") or "").strip()
    if (
        required_vendor
        and required_vendor != UNKNOWN_FACT
        and _cpu_component_side(candidate) != required_vendor
    ):
        reason_codes.append("cpu_family_mismatch")
    required_family = str(requirements.get("cpu_family_preference") or "").strip()
    if (
        required_family
        and required_family != UNKNOWN_FACT
        and _fact(candidate, "cpu_family") != required_family
    ):
        reason_codes.append("cpu_family_mismatch")

    platform = selected.get(SERVER_PLATFORM_ROLE)
    if platform is not None:
        if _candidate_incompatible_with_selection(
            role=CPU_ROLE,
            candidate=candidate,
            selected=selected,
        ):
            reason_codes.append("cpu_family_mismatch")
        if _fatal_platform_cpu_mismatch(platform, candidate):
            reason_codes.append("cpu_family_mismatch")
        platform_socket = _cpu_socket(platform)
        candidate_socket = _cpu_socket(candidate)
        if platform_socket != UNKNOWN_FACT:
            if candidate_socket == UNKNOWN_FACT:
                reason_codes.append("platform_cpu_socket_unknown")
            elif candidate_socket != platform_socket:
                reason_codes.append("cpu_family_mismatch")

    selected_socket = _cpu_socket(selected_cpu)
    candidate_socket = _cpu_socket(candidate)
    if selected_socket != UNKNOWN_FACT:
        if candidate_socket == UNKNOWN_FACT:
            reason_codes.append("platform_cpu_socket_unknown")
        elif candidate_socket != selected_socket:
            reason_codes.append("cpu_family_mismatch")

    reason_codes.extend(_candidate_hard_warning_reason_codes(candidate, CPU_ROLE))
    return _role_eligibility_from_reasons(reason_codes)


def _role_eligibility_from_reasons(reason_codes: Sequence[str]) -> _RoleEligibility:
    unique_codes = _unique([code for code in reason_codes if code])
    if not unique_codes:
        return _RoleEligibility(classification="cheaper_equivalent")
    return _RoleEligibility(
        classification="not_equivalent_requires_engineering_review",
        reason_codes=unique_codes,
    )


def _repair_blocked_critique_facts(
    *,
    proposal: Mapping[str, Any],
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    blocked: Sequence[tuple[_IndexedComponentCandidate, _RoleEligibility]],
    quantity: int,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for candidate, eligibility in blocked:
        fact = _repair_blocked_critique_fact(
            proposal=proposal,
            role=role,
            selected_candidate=selected_candidate,
            blocked_candidate=candidate,
            eligibility=eligibility,
            quantity=quantity,
        )
        if fact is not None:
            facts.append(fact)
    return facts


def _repair_blocked_critique_fact(
    *,
    proposal: Mapping[str, Any],
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    blocked_candidate: _IndexedComponentCandidate,
    eligibility: _RoleEligibility,
    quantity: int,
) -> dict[str, Any] | None:
    selected_line = _line_price(selected_candidate, quantity)
    blocked_line = _line_price(blocked_candidate, quantity)
    if selected_line is None or blocked_line is None or blocked_line >= selected_line:
        return None
    currency = str(selected_candidate.row.get("price_currency") or "").strip()
    recommendation_id = str(
        proposal.get("recommendation_id") or proposal.get("candidate_id") or ""
    )
    role_label = "platform" if role == SERVER_PLATFORM_ROLE else role
    selected_summary = _repair_candidate_summary(selected_candidate, quantity)
    blocked_summary = _repair_candidate_summary(blocked_candidate, quantity)
    reason_codes = eligibility.reason_codes or ["requires_engineering_review"]
    summary = (
        f"{recommendation_id}: blocked cheaper {role_label} "
        f"{blocked_summary['producer']} {blocked_summary['part_number']} "
        f"total {_json_decimal(blocked_line)} {currency}; not cheaper_equivalent "
        f"because {', '.join(reason_codes)}."
    )
    return {
        "classification": eligibility.classification,
        "recommendation_id": recommendation_id,
        "proposal_role": proposal.get("proposal_role"),
        "recommendation_slot": proposal.get("recommendation_slot"),
        "role": role_label,
        "selected": selected_summary,
        "blocked_candidate": blocked_summary,
        "quantity": quantity,
        "reason_codes": reason_codes,
        "summary": summary,
    }


def _required_ram_type(
    requirements: Mapping[str, Any],
    selected_ram: _IndexedComponentCandidate | None,
) -> str:
    required = _normalize_ram_type(requirements.get("ram_type_preference"))
    if required:
        return required
    if selected_ram is not None:
        return _normalized_repair_ram_type(selected_ram).generation
    return ""


def _required_storage_interface(
    requirements: Mapping[str, Any],
    selected_storage: _IndexedComponentCandidate | None,
) -> str:
    required = _normalize_storage_interface(
        requirements.get("storage_interface_preference")
    )
    if required:
        return required
    if selected_storage is not None:
        return _normalize_storage_interface(
            _fact(selected_storage, "storage_interface")
        )
    return ""


def _normalize_ram_type(value: Any) -> str:
    return _detect_direct_ram_generation(str(value or ""))


def _normalized_repair_ram_type(candidate: _IndexedComponentCandidate) -> _RepairRamType:
    text = _repair_ram_search_text(candidate)
    generation = _detect_direct_ram_generation(text)
    if not generation and candidate.internal_role == SERVER_PLATFORM_ROLE:
        inferred = _detect_ram_type(text)
        generation = "" if inferred == UNKNOWN_FACT else inferred
    module_class = _detect_ram_module_class(text)
    return _RepairRamType(
        generation=generation,
        module_class=module_class,
        is_server_memory=_is_server_ram_text(text, module_class),
    )


def _repair_ram_search_text(candidate: _IndexedComponentCandidate) -> str:
    values: list[str] = []
    for source in (candidate.row, candidate.source):
        for key in (
            "producer",
            "normalized_vendor",
            "part_number",
            "name",
            "item_name",
            "category",
            "product_category",
            "item_category",
            "component_category",
            "ram_type",
            "memory_type",
            "memory_class",
            "module_type",
            "dimm_type",
            "form_factor",
        ):
            value = source.get(key)
            if value not in (None, ""):
                values.append(str(value))
        raw_facts = source.get("extracted_facts")
        facts = raw_facts if isinstance(raw_facts, Mapping) else {}
        values.extend(str(value) for value in facts.values() if value not in (None, "", []))
    return " ".join(values)


def _detect_direct_ram_generation(text: str) -> str:
    if not text:
        return ""
    if re.search(
        r"\b(?:DDR\s*5|DDR5|PC5)(?:[-\s]?\d{3,5}[A-Z]*)?\b",
        text,
        re.IGNORECASE,
    ):
        return "DDR5"
    if re.search(
        r"\b(?:DDR\s*4|DDR4|PC4)(?:[-\s]?\d{3,5}[A-Z]*)?\b",
        text,
        re.IGNORECASE,
    ):
        return "DDR4"
    return ""


def _detect_ram_module_class(text: str) -> str:
    if not text:
        return ""
    if re.search(r"\bSO[-\s]?DIMM\b|\bSODIMM\b", text, re.IGNORECASE):
        return "sodimm"
    if re.search(r"\bLR[-\s]?DIMM\b|\bLRDIMM\b|\bLOAD[-\s]?REDUCED\b", text, re.IGNORECASE):
        return "lrdimm"
    if re.search(
        r"\bR[-\s]?DIMM\b|\bRDIMM\b|\bREGISTERED\b|\bECC\s+REG(?:ISTERED)?\b|\bREG(?:ISTERED)?\s+ECC\b",
        text,
        re.IGNORECASE,
    ):
        return "rdimm"
    if re.search(r"\bU[-\s]?DIMM\b|\bUDIMM\b|\bUNBUFFERED\b", text, re.IGNORECASE):
        return "udimm"
    if re.search(r"\bDIMM\b", text, re.IGNORECASE):
        return "dimm"
    return ""


def _is_server_ram_text(text: str, module_class: str) -> bool:
    if module_class in {"rdimm", "lrdimm"}:
        return True
    return bool(
        re.search(
            r"\bserver\b|\bregistered\b|\brdimm\b|\blr\s*-?\s*dimm\b|\blrdimm\b",
            text,
            re.IGNORECASE,
        )
    )


def _is_same_ram_generation(
    *,
    selected: _RepairRamType,
    candidate: _RepairRamType,
    required_generation: str,
) -> bool:
    if not selected.generation or not candidate.generation:
        return False
    if required_generation and (
        selected.generation != required_generation
        or candidate.generation != required_generation
    ):
        return False
    return selected.generation == candidate.generation


def _is_server_ram_equivalent_class(
    selected: _RepairRamType,
    candidate: _RepairRamType,
) -> bool:
    if selected.module_class and candidate.module_class:
        if selected.module_class == candidate.module_class:
            return True
        if "sodimm" in {selected.module_class, candidate.module_class}:
            return False
        if "udimm" in {selected.module_class, candidate.module_class}:
            return False
    return (
        selected.generation == "DDR5"
        and candidate.generation == "DDR5"
        and selected.is_server_memory
        and candidate.is_server_memory
    )


def _ram_class_block_reason(
    selected: _RepairRamType,
    candidate: _RepairRamType,
) -> str | None:
    if _is_server_ram_equivalent_class(selected, candidate):
        return None
    if (
        selected.module_class
        and candidate.module_class
        and selected.module_class != candidate.module_class
    ):
        return "ram_type_mismatch"
    return "ram_type_unknown"


def _normalize_storage_interface(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text or text == UNKNOWN_FACT.upper():
        return ""
    if re.search(r"\bNVME\b|\bU\.?2\b|\bU\.?3\b", text):
        return "NVME"
    if "SAS" in text:
        return "SAS"
    if "SATA" in text:
        return "SATA"
    return text


def _platform_form_factor_confirmed(
    candidate: _IndexedComponentCandidate,
    requested_form_factor: str,
) -> bool:
    requested = requested_form_factor.strip().upper().replace(" ", "")
    if not requested:
        return True
    text = _candidate_search_text(candidate)
    facts = candidate.row.get("extracted_facts")
    fact_values: list[str] = []
    if isinstance(facts, Mapping):
        for key in ("form_factor", "form_factor_hints", "chassis_form_factor"):
            value = facts.get(key)
            if isinstance(value, list):
                fact_values.extend(str(item) for item in value)
            elif value not in (None, ""):
                fact_values.append(str(value))
    fact_text = " ".join([text, *fact_values])
    return bool(
        re.search(
            rf"\b{re.escape(requested)}\b|\b{re.escape(requested[:-1])}\s*U\b",
            fact_text,
            re.IGNORECASE,
        )
    )


def _platform_cpu_socket_count(candidate: _IndexedComponentCandidate) -> int | None:
    facts = candidate.row.get("extracted_facts")
    if isinstance(facts, Mapping):
        for key in (
            "cpu_sockets",
            "cpu_socket_count",
            "sockets_per_server",
            "cpu_per_server",
            "cpu_sockets_per_server",
        ):
            value = _int_value(facts.get(key))
            if value is not None:
                return value
    text = _candidate_search_text(candidate)
    if re.search(r"\b(?:dual|2\s*x|2x|two)\s*(?:cpu|socket|processor)", text, re.IGNORECASE):
        return 2
    match = re.search(r"\b([1-8])\s*(?:cpu|socket|processor)s?\b", text, re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    return None


def _platform_nvme_confirmed(candidate: _IndexedComponentCandidate) -> bool:
    if _bool_fact(candidate, "nvme_support"):
        return True
    interface = _normalize_storage_interface(_fact(candidate, "storage_interface"))
    if interface == "NVME":
        return True
    return _detect_nvme_support(_candidate_search_text(candidate))


def _platform_incomplete_chassis(candidate: _IndexedComponentCandidate) -> bool:
    text = _candidate_search_text(candidate)
    return bool(
        re.search(
            r"\b(?:no|without|less|w/o)\s+"
            r"(?:cpu|processor|memory|ram|hdd|ssd|disk|drive|psu|power\s*supply)",
            text,
            re.IGNORECASE,
        )
    )


def _same_ram_total_capacity_fit(
    selected_ram: _IndexedComponentCandidate,
    candidate: _IndexedComponentCandidate,
    platform: _IndexedComponentCandidate | None,
    requirements: Mapping[str, Any],
) -> bool:
    required_gb = _required_ram_gb_per_server(requirements)
    selected_gb = _candidate_ram_module_gb(selected_ram)
    candidate_gb = _candidate_ram_module_gb(candidate)
    if required_gb is None or selected_gb is None or candidate_gb is None:
        return False
    selected_modules = max(ceil(required_gb / selected_gb), 1)
    candidate_modules = max(ceil(required_gb / candidate_gb), 1)
    if selected_modules * selected_gb != candidate_modules * candidate_gb:
        return False
    slot_count = _platform_dimm_slot_count(platform)
    return slot_count is None or candidate_modules <= slot_count


def _platform_dimm_slot_count(
    platform: _IndexedComponentCandidate | None,
) -> int | None:
    if platform is None:
        return None
    facts = platform.row.get("extracted_facts")
    if isinstance(facts, Mapping):
        for key in ("dimm_slots", "ram_slots", "memory_slots"):
            value = _int_value(facts.get(key))
            if value is not None:
                return value
    text = _candidate_search_text(platform)
    match = re.search(r"\b([1-9]\d?)\s*(?:dimm|ram|memory)\s*slots?\b", text, re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    return None


def _candidate_hard_warning_reason_codes(
    candidate: _IndexedComponentCandidate,
    role: str,
) -> list[str]:
    reason_codes: list[str] = []
    warnings = [
        *_string_list(candidate.row.get("eligibility_warnings")),
        *_string_list(candidate.source.get("eligibility_warnings")),
    ]
    for warning in warnings:
        lowered = warning.casefold()
        hardish = any(
            marker in lowered
            for marker in (
                "fatal",
                "incompat",
                "mismatch",
                "not confirmed",
                "unknown",
                "check",
                "no ",
                "without",
                "не подтверж",
                "провер",
                "несовмест",
            )
        )
        if not hardish:
            continue
        if "stock" in lowered or "остат" in lowered:
            reason_codes.append("stock_shortage")
        if role == SERVER_PLATFORM_ROLE:
            if "form" in lowered or "форм" in lowered:
                reason_codes.append("platform_form_factor_unknown")
            if "cpu" in lowered or "processor" in lowered or "socket" in lowered:
                reason_codes.append("platform_cpu_socket_unknown")
            if "ram" in lowered or "memory" in lowered or "ddr" in lowered or "пам" in lowered:
                reason_codes.append("platform_ram_type_unknown")
            if (
                "storage" in lowered
                or "backplane" in lowered
                or "nvme" in lowered
                or "ssd" in lowered
                or "hdd" in lowered
            ):
                reason_codes.append("platform_storage_unknown")
            if "psu" in lowered or "power" in lowered or "комплект" in lowered:
                reason_codes.append("platform_incomplete_chassis")
            if "incompat" in lowered or "mismatch" in lowered or "несовмест" in lowered:
                reason_codes.append("cpu_family_mismatch")
        elif role == RAM_ROLE:
            has_ram_marker = any(
                marker in lowered
                for marker in ("ram", "memory", "ddr", "dimm", "РїР°Рј")
            )
            has_mismatch_marker = any(
                marker in lowered
                for marker in ("incompat", "mismatch", "РЅРµСЃРѕРІРјРµСЃС‚")
            )
            if has_mismatch_marker:
                reason_codes.append("ram_type_mismatch")
            elif has_ram_marker or "unknown" in lowered or "check" in lowered:
                reason_codes.append("ram_type_unknown")
        elif role in {SSD_ROLE, HDD_ROLE}:
            reason_codes.append("storage_interface_mismatch")
        elif role == CPU_ROLE:
            reason_codes.append("cpu_family_mismatch")
    return _unique(reason_codes)


def _repair_critique_fact(
    *,
    proposal: Mapping[str, Any],
    role: str,
    issue: str,
    selected_candidate: _IndexedComponentCandidate,
    alternative_candidate: _IndexedComponentCandidate,
    quantity: int,
    requirements: Mapping[str, Any],
) -> dict[str, Any] | None:
    selected_line = _line_price(selected_candidate, quantity)
    alternative_line = _line_price(alternative_candidate, quantity)
    if selected_line is None or alternative_line is None:
        return None
    saving = selected_line - alternative_line
    if saving <= 0:
        return None
    currency = str(selected_candidate.row.get("price_currency") or "").strip()
    recommendation_id = str(
        proposal.get("recommendation_id") or proposal.get("candidate_id") or ""
    )
    role_label = "platform" if role == SERVER_PLATFORM_ROLE else role
    selected_summary = _repair_candidate_summary(selected_candidate, quantity)
    alternative_summary = _repair_candidate_summary(alternative_candidate, quantity)
    summary = (
        f"{recommendation_id}: {role_label} selected "
        f"{selected_summary['producer']} {selected_summary['part_number']} "
        f"total {_json_decimal(selected_line)} {currency}; cheaper "
        f"{alternative_summary['producer']} {alternative_summary['part_number']} "
        f"total {_json_decimal(alternative_line)} {currency}; potential saving "
        f"{_json_decimal(saving)} {currency}."
    )
    selected_unit_price = _json_decimal(
        _decimal_value(selected_candidate.row.get("price_value"))
    )
    alternative_unit_price = _json_decimal(
        _decimal_value(alternative_candidate.row.get("price_value"))
    )
    facts = [
        f"Selected {role_label} unit price {selected_unit_price} {currency}.",
        f"Alternative {role_label} unit price {alternative_unit_price} {currency}.",
        f"Total quantity is {quantity}.",
        f"Alternative stock is {_int_value(alternative_candidate.row.get('available_quantity'))}.",
        f"Potential saving is {_json_decimal(saving)} {currency}.",
    ]
    if role == RAM_ROLE:
        required_ram = _required_ram_gb_per_server(requirements)
        module_gb = _candidate_ram_module_gb(selected_candidate)
        selected_ram_type = _normalized_repair_ram_type(selected_candidate)
        ram_type = selected_ram_type.generation or _fact(selected_candidate, "ram_type")
        facts.append(
            f"Both RAM candidates use {module_gb}GB modules and {ram_type}; "
            f"they satisfy {required_ram}GB per server when code materializes quantity."
        )
    if role in {SSD_ROLE, HDD_ROLE}:
        capacity = _candidate_storage_tb(selected_candidate)
        interface = _fact(selected_candidate, "storage_interface")
        facts.append(f"Both storage candidates are {capacity}TB {interface}.")
    if role == SERVER_PLATFORM_ROLE:
        facts.append("Both platforms stay in the same architecture family.")
    if role == CPU_ROLE:
        required_cores = _required_cpu_cores(requirements)
        facts.append(
            f"Alternative CPU still satisfies the hard core requirement of {required_cores} cores."
        )
    return {
        "critique_id": f"crit_{len(summary)}_{role_label}",
        "classification": "cheaper_equivalent",
        "recommendation_id": recommendation_id,
        "proposal_role": proposal.get("proposal_role"),
        "recommendation_slot": proposal.get("recommendation_slot"),
        "role": role_label,
        "issue": issue,
        "selected": selected_summary,
        "alternative": alternative_summary,
        "quantity": quantity,
        "saving_value": _json_decimal(saving),
        "saving_currency": currency,
        "summary": summary,
        "facts": facts,
        "guidance": (
            "For cheapest_quote/price_optimal, prefer the cheaper valid equivalent "
            "unless there is a concrete technical tradeoff."
        ),
    }


def _repair_candidate_summary(
    candidate: _IndexedComponentCandidate,
    quantity: int,
) -> dict[str, Any]:
    line_price = _line_price(candidate, quantity)
    return {
        "component_candidate_id": candidate.component_candidate_id,
        "role": candidate.prompt_role,
        "producer": candidate.row.get("producer"),
        "part_number": candidate.row.get("part_number"),
        "name": candidate.row.get("name"),
        "price_value": candidate.row.get("price_value"),
        "price_currency": candidate.row.get("price_currency"),
        "available_quantity": candidate.row.get("available_quantity"),
        "quantity_required": quantity,
        "line_total_value": _json_decimal(line_price),
        "line_total_currency": candidate.row.get("price_currency"),
        "facts": _public_facts(
            candidate.row.get("extracted_facts")
            if isinstance(candidate.row.get("extracted_facts"), Mapping)
            else {}
        ),
        "cpu_cores": _candidate_cpu_cores(candidate),
        "ram_module_capacity_gb": _candidate_ram_module_gb(candidate),
        "storage_capacity_tb": _candidate_storage_tb(candidate),
    }


def _append_repair_alternative(
    alternatives_by_role: dict[str, list[dict[str, Any]]],
    candidate: _IndexedComponentCandidate,
    *,
    role: str,
    max_per_role: int,
) -> None:
    key = "platform" if role == SERVER_PLATFORM_ROLE else role
    rows = alternatives_by_role.setdefault(key, [])
    if any(row.get("component_candidate_id") == candidate.component_candidate_id for row in rows):
        return
    if len(rows) >= max_per_role:
        return
    rows.append(_repair_candidate_summary(candidate, 1))


def _proposal_is_cost_sensitive(proposal: Mapping[str, Any]) -> bool:
    role = str(proposal.get("proposal_role") or "").strip()
    slot = str(proposal.get("recommendation_slot") or "").strip()
    title = str(proposal.get("title") or "").casefold()
    return (
        role in {"cheapest_fit", "budget_option"}
        or slot in {"price_optimal", "cheapest_quote"}
        or "price" in title
        or "cheapest" in title
    )


def _client_thinking_diagnostics(client: Any) -> dict[str, Any]:
    diagnostics_method = getattr(client, "safe_diagnostics", None)
    if callable(diagnostics_method):
        diagnostics = diagnostics_method()
        return _safe_mapping(diagnostics)
    return {}


def _merge_thinking_diagnostics(
    *diagnostics_items: Mapping[str, Any],
    settings: LlmSettings,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "llm_thinking_enabled": bool(settings.llm_configurator_thinking_enabled),
        "llm_thinking_budget_tokens": settings.llm_configurator_thinking_budget_tokens,
        "llm_thinking_fallback_reason": None,
    }
    for diagnostics in diagnostics_items:
        if not isinstance(diagnostics, Mapping):
            continue
        if diagnostics.get("llm_thinking_fallback_reason"):
            result["llm_thinking_fallback_reason"] = _safe_diagnostic_text(
                diagnostics.get("llm_thinking_fallback_reason"),
                limit=120,
            )
        if "llm_thinking_enabled" in diagnostics:
            result["llm_thinking_enabled"] = bool(diagnostics.get("llm_thinking_enabled"))
        if diagnostics.get("llm_thinking_budget_tokens") is not None:
            result["llm_thinking_budget_tokens"] = _int_value(
                diagnostics.get("llm_thinking_budget_tokens")
            )
    return result


def _multi_pass_composer_requested(
    package: Mapping[str, Any],
    settings: LlmSettings,
) -> bool:
    if not _is_v2_composer_package(package):
        return False
    mode = _v2_pipeline_mode(settings)
    if mode in {"single_pass", "legacy_single_pass"}:
        return False
    if mode in {"composer_cascade", "deep_audit", "multi_pass", "multipass"}:
        return True
    if _role_evaluation_requested(settings, mode=mode):
        return True
    if mode not in {"auto", "high_quality"}:
        return False
    threshold = _int_value(
        getattr(settings, "llm_composer_multi_pass_candidate_threshold", None)
    ) or 120
    return _composer_candidate_count_total(package) >= threshold or _looks_complex_for_multi_pass(
        package
    )


def _v2_pipeline_mode(settings: LlmSettings) -> str:
    mode = str(
        os.getenv("STOCK_MATCH_PIPELINE_V2_MODE")
        or getattr(settings, "stock_match_pipeline_v2_mode", "")
        or ""
    ).strip().casefold().replace("-", "_")
    return mode or "composer_cascade"


def _role_evaluation_requested(settings: LlmSettings, *, mode: str) -> bool:
    role_flag_value = os.getenv("LLM_ROLE_EVALUATION_ENABLED")
    role_flag = (
        _truthy(role_flag_value)
        if role_flag_value is not None
        else bool(getattr(settings, "llm_role_evaluation_enabled", False))
    )
    legacy_flag_value = os.getenv("LLM_COMPOSER_MULTI_PASS")
    legacy_flag = (
        _truthy(legacy_flag_value)
        if legacy_flag_value is not None
        else bool(getattr(settings, "llm_composer_multi_pass", False))
    )
    return role_flag or legacy_flag or mode in {"deep_audit", "multi_pass", "multipass"}


def _composer_critic_enabled(settings: LlmSettings) -> bool:
    return bool(getattr(settings, "llm_composer_critic_enabled", True))


def _composer_repair_max_attempts(settings: LlmSettings) -> int:
    return max(0, _int_value(getattr(settings, "llm_composer_repair_max_attempts", None)) or 0)


def _looks_complex_for_multi_pass(package: Mapping[str, Any]) -> bool:
    counts = _safe_mapping(package.get("composer_package_candidate_count_by_role"))
    populated_roles = [role for role, count in counts.items() if (_int_value(count) or 0) > 0]
    return len(populated_roles) >= 6


def _generate_multi_pass_composer_response(
    *,
    package: dict[str, Any],
    settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings,
    llm_client: LlmClient | None,
    llm_call_budget: LlmCallBudget | None,
    output_mode: str,
) -> _ComposerResponseResult | LlmConfiguratorOutcome:
    owns_client = False
    client = budgeted_llm_client(llm_client, llm_call_budget)
    if client is None:
        try:
            client = budgeted_llm_client(_build_llm_client(settings), llm_call_budget)
        except LlmError as exc:
            logger.info("Multi-pass Composer unavailable: %s", type(exc).__name__)
            return LlmConfiguratorOutcome(
                enabled=True,
                output_mode=_normalize_output_mode(package.get("output_mode")),
                fallback_reason=_client_unavailable_fallback_reason(exc),
                error_type=type(exc).__name__,
                http_status=_llm_http_status(exc),
                parse_diagnostics={"composer_mode": "multi_pass"},
            )
        owns_client = True

    pipeline_mode = _v2_pipeline_mode(settings)
    role_evaluation_enabled = _role_evaluation_requested(settings, mode=pipeline_mode)
    composer_mode = "deep_audit" if role_evaluation_enabled else "composer_cascade"
    diagnostics: dict[str, Any] = {
        "composer_mode": composer_mode,
        "requirement_contract_used": False,
        "main_composer_used": False,
        "role_evaluation_used": False,
        "role_evaluation_skipped_reason": (
            None if role_evaluation_enabled else "default_composer_cascade"
        ),
        "bom_composer_used": False,
        "critic_used": False,
        "completeness_critic_used": False,
        "repair_used": False,
        "repair_composer_used": False,
        "multi_pass_chunk_size": _multi_pass_chunk_size(settings),
        **llm_call_budget_diagnostics(llm_call_budget),
    }
    pass_count = 0
    try:
        contract_payload = client.generate_json(
            LLM_REQUIREMENT_CONTRACT_SYSTEM_PROMPT,
            build_llm_configurator_user_prompt(_requirement_contract_pass_package(package)),
        )
        pass_count += 1
        try:
            contract = _parse_requirement_contract_payload(contract_payload)
            contract_dict = contract.model_dump()
            diagnostics["requirement_contract_used"] = True
            diagnostics["requirement_contract_source"] = "llm_requirement_contract"
        except ValidationError as exc:
            contract_dict = _fallback_requirement_contract_from_package(package)
            diagnostics.update(
                {
                    "requirement_contract_fallback_used": True,
                    "requirement_contract_source": "fallback_from_v2_package",
                    "requirement_contract_error_stage": "requirement_contract",
                    "requirement_contract_error_type": type(exc).__name__,
                    "requirement_contract_validation_errors": _schema_validation_errors(
                        exc
                    ),
                }
            )
        _apply_requirement_contract_to_package(package, contract_dict)
        diagnostics["requirement_contract"] = _jsonable(contract_dict)

        roles = _contract_required_roles(contract_dict, package=package)
        diagnostics["multi_pass_role_order"] = roles
        if role_evaluation_enabled:
            evaluations = _run_role_evaluation_passes(
                client=client,
                package=package,
                contract=contract_dict,
                roles=roles,
                settings=settings,
            )
            pass_count += sum(
                len(item.summaries) + len(item.failed_chunks) for item in evaluations
            )
            diagnostics.update(_role_evaluation_diagnostics(evaluations))
            diagnostics["role_evaluation_skipped_reason"] = None
        else:
            evaluations = _virtual_role_evaluations_for_contract(
                package=package,
                roles=roles,
            )
            diagnostics.update(
                {
                    "role_evaluation_used": False,
                    "role_evaluation_count_by_role": {},
                    "role_evaluation_coverage_by_role": {},
                    "role_evaluation_failed_chunks": [],
                    "role_evaluation_skipped_reason": "default_composer_cascade",
                }
            )

        try:
            bom_payload = client.generate_json(
                LLM_MULTI_PASS_BOM_COMPOSER_SYSTEM_PROMPT,
                build_llm_configurator_user_prompt(
                    _bom_composition_pass_package(
                        package=package,
                        contract=contract_dict,
                        evaluations=evaluations,
                    )
                ),
            )
        except LlmReadTimeoutError as exc:
            fallback = _retry_bom_composer_after_timeout(
                client=client,
                package=package,
                contract=contract_dict,
                evaluations=evaluations,
                settings=settings,
                diagnostics=diagnostics,
            )
            diagnostics.update(fallback.diagnostics)
            if fallback.payload is None:
                _close_owned_client(client, owns_client)
                return _multi_pass_failure_result(
                    package=package,
                    diagnostics=diagnostics,
                    error=fallback.error or exc,
                    fallback_reason=COMPOSER_PROVIDER_TIMEOUT,
                    parse_status="request_failed",
                    request_attempted=True,
                )
            bom_payload = fallback.payload
        except LlmError as exc:
            if not _llm_error_is_context_limit(exc):
                raise
            if _composer_package_mode(package) == COMPACT_FULL_MATRIX_MODE:
                raise
            compact_package = prepare_v2_composer_package(
                package,
                max_package_chars=_int_value(
                    _safe_mapping(package.get("package_budget")).get("max_chars")
                )
                or settings.llm_configurator_max_package_chars,
                force_mode=COMPACT_FULL_MATRIX_MODE,
            )
            _apply_requirement_contract_to_package(compact_package, contract_dict)
            try:
                bom_payload = client.generate_json(
                    LLM_MULTI_PASS_BOM_COMPOSER_SYSTEM_PROMPT,
                    build_llm_configurator_user_prompt(
                        _bom_composition_pass_package(
                            package=compact_package,
                            contract=contract_dict,
                            evaluations=evaluations,
                        )
                    ),
                )
            except LlmError as retry_exc:
                if _llm_error_is_context_limit(retry_exc):
                    retry_diagnostics = _context_limit_retry_diagnostics(
                        package=package,
                        compact_package=compact_package,
                        retry_attempted=True,
                        retry_success=False,
                        after_compact=True,
                    )
                    _close_owned_client(client, owns_client)
                    return LlmConfiguratorOutcome(
                        enabled=True,
                        output_mode=_normalize_output_mode(package.get("output_mode")),
                        fallback_reason=COMPACT_FULL_MATRIX_CONTEXT_LIMIT_FALLBACK_REASON,
                        error_type=type(retry_exc).__name__,
                        http_status=_llm_http_status(retry_exc),
                        parse_diagnostics={
                            **diagnostics,
                            **retry_diagnostics,
                            "provider_error_type": PROVIDER_CONTEXT_LIMIT_ERROR_TYPE,
                        },
                        internal_warnings=[
                            COMPACT_FULL_MATRIX_CONTEXT_LIMIT_FALLBACK_REASON
                        ],
                        final_status_source="provider_context_limit",
                    )
                raise
            diagnostics.update(
                _context_limit_retry_diagnostics(
                    package=package,
                    compact_package=compact_package,
                    retry_attempted=True,
                    retry_success=True,
                    after_compact=False,
                )
            )
            package.update(compact_package)
        pass_count += 1
        diagnostics["bom_composer_used"] = True
        diagnostics["main_composer_used"] = True
        parsed_bom = _parse_composer_payload(
            _normalize_composer_payload(bom_payload, package=package)
        )
        response = parsed_bom.response
        schema_rejections = parsed_bom.schema_rejections
        proposal_indexes = parsed_bom.proposal_indexes
        proposal_count = parsed_bom.proposal_count

        code_completeness = _composer_code_completeness_check(
            package=package,
            contract=contract_dict,
            proposed_bom=response,
        )
        diagnostics["code_completeness_result"] = _jsonable(code_completeness)
        should_repair = bool(code_completeness.get("repair_required"))
        if (
            schema_rejections
            and not response.recommendations
            and not _has_structured_no_recommendation(response)
        ):
            should_repair = False
            diagnostics["schema_only_rejection_skipped_repair"] = True
        critic_dict: dict[str, Any] = _critic_from_code_completeness(code_completeness)
        if (
            should_repair
            and _composer_critic_enabled(settings)
            and _composer_repair_max_attempts(settings) > 0
        ):
            critic_payload = client.generate_json(
                LLM_COMPLETENESS_CRITIC_SYSTEM_PROMPT,
                build_llm_configurator_user_prompt(
                    _completeness_critic_pass_package(
                        package=package,
                        contract=contract_dict,
                        proposed_bom=response,
                        code_completeness=code_completeness,
                    )
                ),
            )
            pass_count += 1
            critic = _parse_completeness_critic_payload(critic_payload)
            llm_critic_dict = critic.model_dump()
            code_critic_dict = _critic_from_code_completeness(code_completeness)
            critic_dict = {
                **llm_critic_dict,
                **code_critic_dict,
                "recommended_repair_actions": _unique(
                    [
                        *_string_list(
                            llm_critic_dict.get("recommended_repair_actions")
                        ),
                        *_string_list(
                            code_critic_dict.get("recommended_repair_actions")
                        ),
                    ]
                ),
            }
            diagnostics["completeness_critic_used"] = True
            diagnostics["critic_used"] = True
            diagnostics["completeness_critic_result"] = _jsonable(critic_dict)
            should_repair = _critic_requires_repair(critic_dict)
        else:
            diagnostics["completeness_critic_result"] = _jsonable(critic_dict)

        if should_repair and _composer_repair_max_attempts(settings) > 0:
            diagnostics["repair_composer_used"] = True
            diagnostics["repair_used"] = True
            repair_payload = client.generate_json(
                LLM_MULTI_PASS_REPAIR_SYSTEM_PROMPT,
                build_llm_configurator_user_prompt(
                    _repair_pass_package(
                        package=package,
                        contract=contract_dict,
                        evaluations=evaluations,
                        proposed_bom=response,
                        criticism=critic_dict,
                    )
                ),
            )
            pass_count += 1
            parsed_repair = _parse_composer_payload(
                _normalize_composer_payload(repair_payload, package=package)
            )
            response = parsed_repair.response
            schema_rejections = parsed_repair.schema_rejections
            proposal_indexes = parsed_repair.proposal_indexes
            proposal_count = parsed_repair.proposal_count
            repaired_completeness = _composer_code_completeness_check(
                package=package,
                contract=contract_dict,
                proposed_bom=response,
            )
            diagnostics["code_completeness_after_repair"] = _jsonable(
                repaired_completeness
            )
            if bool(repaired_completeness.get("repair_required")):
                response = _composer_response_from_completeness_failure(
                    contract=contract_dict,
                    completeness=repaired_completeness,
                )
                schema_rejections = []
                proposal_indexes = []
                proposal_count = 0
        elif should_repair:
            response = _composer_response_from_completeness_failure(
                contract=contract_dict,
                completeness=code_completeness,
            )
            schema_rejections = []
            proposal_indexes = []
            proposal_count = 0
        diagnostics["final_bom_after_repair"] = _jsonable(response.model_dump())
    except LlmCallBudgetExceededError as exc:
        _close_owned_client(client, owns_client)
        diagnostics.update(llm_call_budget_diagnostics(llm_call_budget))
        return _llm_call_budget_exceeded_outcome(
            package=package,
            diagnostics=diagnostics,
            error=exc,
        )
    except LlmReadTimeoutError as exc:
        _close_owned_client(client, owns_client)
        return _multi_pass_failure_result(
            package=package,
            diagnostics=diagnostics,
            error=exc,
            fallback_reason=COMPOSER_PROVIDER_TIMEOUT,
            parse_status="request_failed",
            request_attempted=True,
        )
    except LlmInvalidJsonError as exc:
        _close_owned_client(client, owns_client)
        return _multi_pass_failure_result(
            package=package,
            diagnostics={**diagnostics, **_invalid_json_diagnostics(exc)},
            error=exc,
            fallback_reason="multi_pass_invalid_json",
            parse_status="parse_error",
            request_attempted=True,
        )
    except ValidationError as exc:
        _close_owned_client(client, owns_client)
        return _multi_pass_failure_result(
            package=package,
            diagnostics=diagnostics,
            error=exc,
            fallback_reason="multi_pass_validation_failed",
            parse_status="validation_error",
            request_attempted=True,
        )
    except LlmError as exc:
        _close_owned_client(client, owns_client)
        if _llm_error_is_context_limit(exc):
            parse_diagnostics = _provider_context_limit_diagnostics(exc)
            return LlmConfiguratorOutcome(
                enabled=True,
                output_mode=_normalize_output_mode(package.get("output_mode")),
                fallback_reason=PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON,
                error_type=type(exc).__name__,
                http_status=_llm_http_status(exc),
                parse_diagnostics={
                    **diagnostics,
                    **parse_diagnostics,
                    "provider_error_type": PROVIDER_CONTEXT_LIMIT_ERROR_TYPE,
                },
                internal_warnings=[PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON],
            )
        return _multi_pass_failure_result(
            package=package,
            diagnostics=diagnostics,
            error=exc,
            fallback_reason="multi_pass_request_failed",
            parse_status="request_failed",
            request_attempted=True,
        )

    diagnostics["multi_pass_pass_count"] = pass_count
    diagnostics.update(llm_call_budget_diagnostics(llm_call_budget))
    if not response.recommendations and schema_rejections:
        _close_owned_client(client, owns_client)
        outcome = _validation_failed_outcome_from_schema_rejections(
            schema_rejections,
            proposal_count=proposal_count,
        )
        return replace(
            outcome,
            parse_diagnostics={**_safe_mapping(outcome.parse_diagnostics), **diagnostics},
        )
    return _ComposerResponseResult(
        response=response,
        client=client,
        owns_client=owns_client,
        online_composer_used=True,
        online_diagnostics={
            **_online_composer_diagnostics(
                settings=web_evidence_settings,
                online_composer_used=True,
                parse_status="parsed",
            ),
            **diagnostics,
            "structured_no_recommendation_used": _has_structured_no_recommendation(
                response
            ),
        },
        schema_rejections=schema_rejections,
        proposal_indexes=proposal_indexes,
        proposal_count=proposal_count,
    )


def _multi_pass_failure_result(
    *,
    package: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    error: Exception,
    fallback_reason: str,
    parse_status: str,
    request_attempted: bool,
) -> LlmConfiguratorOutcome:
    del request_attempted
    failure_stage = _last_llm_call_stage(diagnostics) or "multi_pass"
    parse_diagnostics = {
        **dict(diagnostics),
        "parse_status": parse_status,
        "llm_parse_stage": failure_stage,
        "composer_failure_stage": failure_stage,
        "composer_failure_error_type": type(error).__name__,
    }
    exposure_policy = _safe_mapping(package.get("package_candidate_exposure_policy"))
    if exposure_policy:
        parse_diagnostics["package_candidate_exposure_policy"] = exposure_policy
    final_status_source = None
    if fallback_reason == COMPOSER_PROVIDER_TIMEOUT:
        parse_diagnostics["final_status_source"] = COMPOSER_PROVIDER_TIMEOUT
        final_status_source = COMPOSER_PROVIDER_TIMEOUT
    if isinstance(error, ValidationError):
        parse_diagnostics["llm_schema_validation_errors"] = _schema_validation_errors(
            error
        )
        final_status_source = COMPOSER_SCHEMA_VALIDATION_FAILED
    return LlmConfiguratorOutcome(
        enabled=True,
        output_mode=_normalize_output_mode(package.get("output_mode")),
        fallback_reason=fallback_reason,
        error_type=type(error).__name__,
        http_status=_llm_http_status(error),
        parse_diagnostics=parse_diagnostics,
        internal_warnings=[f"multi_pass_composer: {type(error).__name__}"],
        final_status_source=final_status_source,
    )


def _retry_bom_composer_after_timeout(
    *,
    client: LlmClient,
    package: dict[str, Any],
    contract: Mapping[str, Any],
    evaluations: Mapping[str, Sequence[Mapping[str, Any]]],
    settings: LlmSettings,
    diagnostics: Mapping[str, Any],
) -> _ComposerTimeoutFallbackResult:
    original_mode = _composer_package_mode(package)
    original_chars = _int_value(
        _safe_mapping(package.get("selected_context_size")).get("chars")
    ) or _int_value(_safe_mapping(package.get("composer_context_size")).get("chars"))
    base = {
        "composer_provider_timeout": True,
        "composer_timeout_original_fallback_reason": "multi_pass_read_timeout",
        "composer_timeout_original_package_mode": original_mode,
        "composer_timeout_original_context_chars": original_chars,
    }
    if original_mode == COMPACT_FULL_MATRIX_MODE:
        reduced = _retry_bom_composer_with_reduced_timeout_package(
            client=client,
            package=package,
            contract=contract,
            settings=settings,
            base_diagnostics=base,
            original_mode=original_mode,
            original_context_chars=original_chars,
            prior_fallback_type=None,
            prior_fallback_reason=None,
        )
        if reduced is not None:
            return reduced
        return _composer_timeout_no_fallback_result(
            package=package,
            base_diagnostics=base,
            reason="already_compact_full_matrix_no_reduced_package",
        )

    compact_package = prepare_v2_composer_package(
        package,
        max_package_chars=_int_value(
            _safe_mapping(package.get("package_budget")).get("max_chars")
        )
        or settings.llm_configurator_max_package_chars,
        force_mode=COMPACT_FULL_MATRIX_MODE,
    )
    _apply_requirement_contract_to_package(compact_package, contract)
    retry_chars = _int_value(
        _safe_mapping(compact_package.get("selected_context_size")).get("chars")
    ) or _int_value(
        _safe_mapping(compact_package.get("composer_context_size")).get("chars")
    )
    fallback_policy = {
        "attempted": True,
        "type": "compact_full_matrix_retry",
        "reason": "main_composer_read_timeout",
        "original_package_mode": original_mode,
        "retry_package_mode": _composer_package_mode(compact_package),
        "original_context_chars": original_chars,
        "retry_context_chars": retry_chars,
        "candidate_exposure_preserved": not bool(
            compact_package.get("package_candidate_loss")
        ),
        "silent_trimming": False,
    }
    _mark_timeout_fallback_policy(compact_package, fallback_policy)
    package.update(compact_package)
    retry_base = {
        **base,
        "composer_timeout_fallback_attempted": True,
        "composer_timeout_fallback_type": "compact_full_matrix_retry",
        "composer_timeout_retry_package_mode": _composer_package_mode(compact_package),
        "composer_timeout_retry_context_chars": retry_chars,
    }
    try:
        payload = client.generate_json(
            LLM_MULTI_PASS_BOM_COMPOSER_SYSTEM_PROMPT,
            build_llm_configurator_user_prompt(
                _bom_composition_pass_package(
                    package=package,
                    contract=contract,
                    evaluations=evaluations,
                )
            ),
        )
    except LlmReadTimeoutError as exc:
        reduced = _retry_bom_composer_with_reduced_timeout_package(
            client=client,
            package=package,
            contract=contract,
            settings=settings,
            base_diagnostics={
                **retry_base,
                "composer_timeout_prior_fallback_type": "compact_full_matrix_retry",
                "composer_timeout_prior_fallback_reason": "compact_retry_read_timeout",
            },
            original_mode=original_mode,
            original_context_chars=original_chars,
            prior_fallback_type="compact_full_matrix_retry",
            prior_fallback_reason="compact_retry_read_timeout",
        )
        if reduced is not None:
            return reduced
        _mark_timeout_fallback_policy(
            package,
            {**fallback_policy, "success": False, "failure": type(exc).__name__},
        )
        return _ComposerTimeoutFallbackResult(
            payload=None,
            error=exc,
            diagnostics={
                **retry_base,
                "composer_timeout_fallback_success": False,
                "composer_timeout_fallback_reason": "compact_retry_read_timeout",
            },
        )
    except LlmError as exc:
        _mark_timeout_fallback_policy(
            package,
            {**fallback_policy, "success": False, "failure": type(exc).__name__},
        )
        return _ComposerTimeoutFallbackResult(
            payload=None,
            error=exc,
            diagnostics={
                **retry_base,
                "composer_timeout_fallback_success": False,
                "composer_timeout_fallback_reason": (
                    f"compact_retry_failed:{type(exc).__name__}"
                ),
            },
        )
    _mark_timeout_fallback_policy(
        package,
        {**fallback_policy, "success": True},
    )
    return _ComposerTimeoutFallbackResult(
        payload=payload,
        diagnostics={
            **retry_base,
            "composer_timeout_fallback_success": True,
            "composer_timeout_fallback_reason": "compact_retry_succeeded",
        },
    )


def _composer_timeout_no_fallback_result(
    *,
    package: dict[str, Any],
    base_diagnostics: Mapping[str, Any],
    reason: str,
) -> _ComposerTimeoutFallbackResult:
    _mark_timeout_fallback_policy(
        package,
        {
            "attempted": False,
            "type": "none_available",
            "reason": reason,
            "silent_trimming": False,
        },
    )
    return _ComposerTimeoutFallbackResult(
        payload=None,
        error=None,
        diagnostics={
            **dict(base_diagnostics),
            "composer_timeout_fallback_attempted": False,
            "composer_timeout_fallback_type": "none_available",
            "composer_timeout_fallback_success": False,
            "composer_timeout_fallback_reason": reason,
        },
    )


def _retry_bom_composer_with_reduced_timeout_package(
    *,
    client: LlmClient,
    package: dict[str, Any],
    contract: Mapping[str, Any],
    settings: LlmSettings,
    base_diagnostics: Mapping[str, Any],
    original_mode: str,
    original_context_chars: int | None,
    prior_fallback_type: str | None,
    prior_fallback_reason: str | None,
) -> _ComposerTimeoutFallbackResult | None:
    reduced_package = _build_timeout_reduced_composer_package(
        package=package,
        contract=contract,
        settings=settings,
        original_mode=original_mode,
        original_context_chars=original_context_chars,
    )
    if reduced_package is None:
        return None

    _apply_requirement_contract_to_package(reduced_package, contract)
    retry_chars = _int_value(
        _safe_mapping(reduced_package.get("selected_context_size")).get("chars")
    ) or _int_value(
        _safe_mapping(reduced_package.get("composer_context_size")).get("chars")
    )
    fallback_policy = _safe_mapping(
        _safe_mapping(reduced_package.get("package_candidate_exposure_policy")).get(
            "timeout_fallback"
        )
    )
    fallback_policy = {
        **fallback_policy,
        "attempted": True,
        "type": TIMEOUT_FALLBACK_REDUCED_PACKAGE_TYPE,
        "reason": "main_composer_read_timeout",
        "original_package_mode": original_mode,
        "retry_package_mode": TIMEOUT_FALLBACK_REDUCED_PACKAGE_MODE,
        "original_context_chars": original_context_chars,
        "retry_context_chars": retry_chars,
        "prior_fallback_type": prior_fallback_type,
        "prior_fallback_reason": prior_fallback_reason,
        "silent_trimming": False,
    }
    _mark_timeout_fallback_policy(reduced_package, fallback_policy)
    package.update(reduced_package)
    fallback_evaluations = _virtual_role_evaluations_for_contract(
        package=package,
        roles=_contract_required_roles(contract, package=package),
    )
    retry_base = {
        **dict(base_diagnostics),
        "composer_timeout_fallback_attempted": True,
        "composer_timeout_fallback_type": TIMEOUT_FALLBACK_REDUCED_PACKAGE_TYPE,
        "composer_timeout_retry_package_mode": TIMEOUT_FALLBACK_REDUCED_PACKAGE_MODE,
        "composer_timeout_retry_context_chars": retry_chars,
        "composer_timeout_fallback_original_candidate_count_by_role": _safe_mapping(
            package.get("original_candidate_count_by_role")
        ),
        "composer_timeout_fallback_candidate_count_by_role": _safe_mapping(
            package.get("fallback_candidate_count_by_role")
        ),
        "composer_timeout_fallback_dropped_before_fallback_count_by_role": (
            _safe_mapping(package.get("dropped_before_fallback_count_by_role"))
        ),
        "composer_timeout_fallback_dropped_before_fallback_reasons": _safe_mapping(
            package.get("dropped_before_fallback_reasons")
        ),
        "composer_timeout_fallback_coverage_ratio_by_role": _safe_mapping(
            package.get("timeout_fallback_coverage_ratio_by_role")
        ),
        "original_candidate_count_by_role": _safe_mapping(
            package.get("original_candidate_count_by_role")
        ),
        "fallback_candidate_count_by_role": _safe_mapping(
            package.get("fallback_candidate_count_by_role")
        ),
        "dropped_before_fallback_count_by_role": _safe_mapping(
            package.get("dropped_before_fallback_count_by_role")
        ),
        "dropped_before_fallback_reasons": _safe_mapping(
            package.get("dropped_before_fallback_reasons")
        ),
        "timeout_fallback_coverage_ratio_by_role": _safe_mapping(
            package.get("timeout_fallback_coverage_ratio_by_role")
        ),
    }
    try:
        payload = client.generate_json(
            LLM_MULTI_PASS_BOM_COMPOSER_SYSTEM_PROMPT,
            build_llm_configurator_user_prompt(
                _bom_composition_pass_package(
                    package=package,
                    contract=contract,
                    evaluations=fallback_evaluations,
                )
            ),
        )
    except LlmReadTimeoutError as exc:
        _mark_timeout_fallback_policy(
            package,
            {**fallback_policy, "success": False, "failure": type(exc).__name__},
        )
        return _ComposerTimeoutFallbackResult(
            payload=None,
            error=exc,
            diagnostics={
                **retry_base,
                "composer_timeout_fallback_success": False,
                "composer_timeout_fallback_reason": (
                    "role_aware_reduced_package_read_timeout"
                ),
            },
        )
    except LlmError as exc:
        _mark_timeout_fallback_policy(
            package,
            {**fallback_policy, "success": False, "failure": type(exc).__name__},
        )
        return _ComposerTimeoutFallbackResult(
            payload=None,
            error=exc,
            diagnostics={
                **retry_base,
                "composer_timeout_fallback_success": False,
                "composer_timeout_fallback_reason": (
                    f"role_aware_reduced_package_failed:{type(exc).__name__}"
                ),
            },
        )
    _mark_timeout_fallback_policy(package, {**fallback_policy, "success": True})
    return _ComposerTimeoutFallbackResult(
        payload=payload,
        diagnostics={
            **retry_base,
            "composer_timeout_fallback_success": True,
            "composer_timeout_fallback_reason": (
                "role_aware_reduced_package_succeeded"
            ),
        },
    )


def _build_timeout_reduced_composer_package(
    *,
    package: Mapping[str, Any],
    contract: Mapping[str, Any],
    settings: LlmSettings,
    original_mode: str,
    original_context_chars: int | None,
) -> dict[str, Any] | None:
    source_package = prepare_v2_composer_package(
        package,
        max_package_chars=_int_value(
            _safe_mapping(package.get("package_budget")).get("max_chars")
        )
        or settings.llm_configurator_max_package_chars,
        force_mode=COMPACT_FULL_MATRIX_MODE,
    )
    _apply_requirement_contract_to_package(source_package, contract)
    if not _timeout_fallback_should_reduce(
        source_package,
        original_mode=original_mode,
        original_context_chars=original_context_chars,
    ):
        return None

    matrix = _safe_mapping(source_package.get("component_candidate_matrix"))
    required_roles = _timeout_fallback_required_roles(contract, package=source_package)
    required_role_keys = {_coverage_role_key(role) for role in required_roles if role}
    reduced_matrix: dict[str, list[dict[str, Any]]] = {}
    original_counts: dict[str, int] = {}
    fallback_counts: dict[str, int] = {}
    fallback_ids: dict[str, list[str]] = {}
    dropped_counts: dict[str, int] = {}
    dropped_reasons: dict[str, list[dict[str, Any]]] = {}
    selection_reasons_by_role: dict[str, dict[str, list[str]]] = {}

    for prompt_role, raw_rows in matrix.items():
        rows = _mapping_rows(raw_rows)
        if not rows:
            continue
        internal_role = INTERNAL_ROLE_BY_PROMPT_ROLE.get(
            str(prompt_role),
            str(prompt_role),
        )
        role = _coverage_role_key(internal_role)
        if not role:
            continue
        original_counts[role] = len(_candidate_ids_for_rows(rows))
        role_required = role in required_role_keys
        role_limit = _timeout_fallback_role_limit(
            settings,
            required=role_required,
            role_count=original_counts[role],
        )
        selected_rows, selection_reasons = _select_timeout_fallback_rows(
            rows,
            limit=role_limit,
            required=role_required,
        )
        if not selected_rows:
            continue
        reduced_matrix[str(prompt_role)] = selected_rows
        selected_ids = _candidate_ids_for_rows(selected_rows)
        fallback_counts[role] = len(selected_ids)
        fallback_ids[role] = selected_ids
        dropped = max(0, original_counts[role] - fallback_counts[role])
        dropped_counts[role] = dropped
        if selection_reasons:
            selection_reasons_by_role[role] = selection_reasons
        if dropped:
            dropped_reasons[role] = [
                {
                    "reason": "timeout_fallback_bounded_role_aware_reduction",
                    "count": dropped,
                    "role_limit": role_limit,
                    "required_role": role_required,
                    "selection_reason_counts": _reason_counts(selection_reasons),
                    "dropped_candidate_ids_sample": _dropped_candidate_ids_sample(
                        rows,
                        selected_ids,
                    ),
                }
            ]

    if not reduced_matrix:
        return None
    original_total = sum(original_counts.values())
    fallback_total = sum(fallback_counts.values())
    if fallback_total <= 0 or fallback_total >= original_total:
        return None

    ratios = {
        role: round(
            min(1.0, int(fallback_counts.get(role, 0) or 0) / max(1, count)),
            4,
        )
        for role, count in original_counts.items()
        if count > 0
    }
    insufficient_roles = [
        role
        for role in sorted(required_role_keys)
        if int(original_counts.get(role, 0) or 0) > 0
        and int(fallback_counts.get(role, 0) or 0) <= 0
    ]
    fallback = dict(source_package)
    fallback["component_candidate_matrix"] = reduced_matrix
    fallback["timeout_fallback_package_mode"] = TIMEOUT_FALLBACK_REDUCED_PACKAGE_MODE
    fallback["original_candidate_count_by_role"] = original_counts
    fallback["fallback_candidate_count_by_role"] = fallback_counts
    fallback["dropped_before_fallback_count_by_role"] = dropped_counts
    fallback["dropped_before_fallback_reasons"] = dropped_reasons
    fallback["timeout_fallback_coverage_ratio_by_role"] = ratios
    fallback["timeout_fallback_selection_reasons_by_role"] = selection_reasons_by_role
    fallback["composer_package_candidate_count_by_role"] = fallback_counts
    fallback["composer_package_candidate_total"] = fallback_total
    fallback["composer_package_candidate_ids_by_role"] = fallback_ids
    fallback["dropped_before_composer_count_by_role"] = dropped_counts
    fallback["dropped_before_composer_reason_by_role"] = {
        role: "timeout_fallback_bounded_role_aware_reduction"
        for role, count in dropped_counts.items()
        if count > 0
    }
    fallback["package_candidate_exposure_ratio_by_role"] = ratios
    fallback["package_candidate_exposure_incomplete"] = bool(insufficient_roles)
    fallback["package_candidate_exposure_incomplete_roles"] = insufficient_roles
    fallback["package_candidate_exposure_policy"] = {
        **_safe_mapping(fallback.get("package_candidate_exposure_policy")),
        "mode": TIMEOUT_FALLBACK_REDUCED_PACKAGE_MODE,
        "allow_incomplete": False,
        "required_roles": sorted(required_role_keys),
        "incomplete_roles": insufficient_roles,
        "candidate_matrix_trimming_allowed": False,
        "semantic_filtering_allowed": False,
        "top_n_allowed": False,
        "silent_trimming": False,
        "timeout_fallback": {
            "attempted": True,
            "type": TIMEOUT_FALLBACK_REDUCED_PACKAGE_TYPE,
            "reason": "main_composer_read_timeout",
            "silent_trimming": False,
            "original_package_mode": original_mode,
            "retry_package_mode": TIMEOUT_FALLBACK_REDUCED_PACKAGE_MODE,
            "original_context_chars": original_context_chars,
            "original_candidate_count_by_role": original_counts,
            "fallback_candidate_count_by_role": fallback_counts,
            "dropped_before_fallback_count_by_role": dropped_counts,
            "dropped_before_fallback_reasons": dropped_reasons,
            "coverage_ratio_by_role": ratios,
        },
    }
    fallback["package_candidate_loss"] = False
    fallback["compact_candidate_count_by_role"] = fallback_counts
    fallback["compact_candidate_total"] = fallback_total
    fallback["compact_candidate_ids_by_role"] = fallback_ids
    _finalize_timeout_fallback_package_size(
        fallback,
        max_chars=_int_value(_safe_mapping(fallback.get("package_budget")).get("max_chars"))
        or settings.llm_configurator_max_package_chars,
        original_context_chars=original_context_chars,
    )
    if _package_over_budget(fallback):
        return None
    return fallback


def _timeout_fallback_should_reduce(
    package: Mapping[str, Any],
    *,
    original_mode: str,
    original_context_chars: int | None,
) -> bool:
    candidate_total = _composer_candidate_count_total(package)
    context_chars = original_context_chars or _package_size_chars(package)
    return (
        original_mode == COMPACT_FULL_MATRIX_MODE
        or candidate_total >= COMPACT_FULL_MATRIX_AUTO_CANDIDATE_THRESHOLD
        or context_chars >= COMPACT_FULL_MATRIX_AUTO_CHAR_THRESHOLD
    )


def _timeout_fallback_required_roles(
    contract: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
) -> list[str]:
    roles = [
        *_contract_required_roles(contract, package=package),
        *_string_list(package.get("hard_purchasable_bom_roles")),
    ]
    normalized_roles: list[str] = []
    for role in roles:
        normalized = _normalize_contract_role(role, package=package)
        if normalized is not None:
            normalized_roles.append(normalized)
    return _unique(normalized_roles)


def _timeout_fallback_role_limit(
    settings: LlmSettings,
    *,
    required: bool,
    role_count: int,
) -> int:
    if role_count <= 0:
        return 0
    min_required = max(
        1,
        _int_value(
            getattr(
                settings,
                "llm_configurator_no_recommendation_min_large_role_candidates",
                None,
            )
        )
        or DEFAULT_TIMEOUT_FALLBACK_MIN_REQUIRED_ROLE_CANDIDATES,
    )
    configured = max(
        1,
        _int_value(getattr(settings, "llm_component_candidates_per_role", None))
        or DEFAULT_TIMEOUT_FALLBACK_MAX_REQUIRED_ROLE_CANDIDATES,
    )
    if required:
        return min(
            role_count,
            max(
                min_required,
                min(configured, DEFAULT_TIMEOUT_FALLBACK_MAX_REQUIRED_ROLE_CANDIDATES),
            ),
        )
    return min(
        role_count,
        min(
            max(4, min_required // 2),
            DEFAULT_TIMEOUT_FALLBACK_MAX_OPTIONAL_ROLE_CANDIDATES,
        ),
    )


def _select_timeout_fallback_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    required: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    indexed = [
        (index, dict(row), candidate_id)
        for index, row in enumerate(rows)
        if (candidate_id := str(row.get("component_candidate_id") or "").strip())
    ]
    if limit <= 0 or not indexed:
        return [], {}
    if len(indexed) <= limit:
        return [row for _, row, _ in indexed], {
            candidate_id: ["all_candidates_under_role_limit"]
            for _, _, candidate_id in indexed
        }

    selected: set[str] = set()
    reasons: dict[str, list[str]] = {}

    def add(
        candidates: Sequence[tuple[int, dict[str, Any], str]],
        reason: str,
        max_items: int | None = None,
    ) -> None:
        added = 0
        for _, _, candidate_id in candidates:
            if len(selected) >= limit:
                return
            if max_items is not None and added >= max_items:
                return
            if candidate_id in selected:
                continue
            selected.add(candidate_id)
            reasons.setdefault(candidate_id, []).append(reason)
            added += 1

    preferred_budget = max(2, limit // 3)
    possible_budget = max(2, limit // 4)
    if required:
        add(
            [item for item in indexed if _timeout_fallback_row_is_strong(item[1])],
            "strong_or_exact_fit",
            preferred_budget,
        )
        add(
            [item for item in indexed if _timeout_fallback_row_is_possible(item[1])],
            "possible_or_unknown_fit",
            possible_budget,
        )
    add(
        sorted(indexed, key=lambda item: (_timeout_fallback_price_sort(item[1]), item[0])),
        "cheapest_available",
        max(2, limit // 4),
    )
    add(
        sorted(
            indexed,
            key=lambda item: (-_timeout_fallback_available_quantity(item[1]), item[0]),
        ),
        "highest_stock",
        max(2, limit // 4),
    )
    add(
        _timeout_fallback_diversity_rows(indexed),
        "category_vendor_diversity",
        max(2, limit // 4),
    )
    add(
        _timeout_fallback_spread_rows(indexed, target=max(2, limit // 4)),
        "stable_broad_coverage",
    )
    add(indexed, "fill_to_role_minimum")

    selected_rows = [
        row
        for _, row, candidate_id in indexed
        if candidate_id in selected
    ][:limit]
    selected_ids = {str(row.get("component_candidate_id") or "") for row in selected_rows}
    return selected_rows, {
        candidate_id: value
        for candidate_id, value in reasons.items()
        if candidate_id in selected_ids
    }


def _timeout_fallback_row_is_strong(row: Mapping[str, Any]) -> bool:
    fit = str(row.get("fit_tier") or row.get("fit_label") or "").casefold()
    if fit in {FIT_TIER_STRONG, FIT_EXACT_OR_CLOSE, "strong", "exact"}:
        return True
    return bool(
        _string_list(row.get("fit_reasons"))
        or _string_list(row.get("matrix_distiller_matched_constraints"))
    )


def _timeout_fallback_row_is_possible(row: Mapping[str, Any]) -> bool:
    fit = str(row.get("fit_tier") or row.get("fit_label") or "").casefold()
    return fit in {
        FIT_TIER_POSSIBLE,
        FIT_TIER_FALLBACK_UNKNOWN,
        FIT_UNKNOWN,
        "possible",
        "unknown",
    }


def _timeout_fallback_price_sort(row: Mapping[str, Any]) -> tuple[int, Decimal]:
    price = _decimal_value(row.get("price_value"))
    return (1, Decimal("0")) if price is None else (0, price)


def _timeout_fallback_available_quantity(row: Mapping[str, Any]) -> int:
    return max(0, _int_value(row.get("available_quantity")) or 0)


def _timeout_fallback_diversity_rows(
    indexed: Sequence[tuple[int, dict[str, Any], str]],
) -> list[tuple[int, dict[str, Any], str]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[int, dict[str, Any], str]] = []
    for item in indexed:
        _, row, _ = item
        key = (
            str(row.get("category_id") or "").strip(),
            str(row.get("producer") or row.get("normalized_vendor") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _timeout_fallback_spread_rows(
    indexed: Sequence[tuple[int, dict[str, Any], str]],
    *,
    target: int,
) -> list[tuple[int, dict[str, Any], str]]:
    if not indexed or target <= 0:
        return []
    if len(indexed) <= target:
        return list(indexed)
    if target == 1:
        return [indexed[0]]
    last = len(indexed) - 1
    positions = sorted(
        {
            round(last * offset / max(1, target - 1))
            for offset in range(target)
        }
    )
    return [indexed[position] for position in positions]


def _reason_counts(reasons: Mapping[str, Sequence[str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for values in reasons.values():
        for reason in values:
            counts[str(reason)] += 1
    return dict(sorted(counts.items()))


def _dropped_candidate_ids_sample(
    rows: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
    *,
    limit: int = 20,
) -> list[str]:
    selected = set(selected_ids)
    dropped: list[str] = []
    for row in rows:
        candidate_id = str(row.get("component_candidate_id") or "").strip()
        if not candidate_id or candidate_id in selected:
            continue
        dropped.append(candidate_id)
        if len(dropped) >= limit:
            break
    return dropped


def _finalize_timeout_fallback_package_size(
    package: dict[str, Any],
    *,
    max_chars: int,
    original_context_chars: int | None,
) -> None:
    initial_chars = original_context_chars or _package_size_chars(package)
    for _ in range(3):
        _sync_package_budget(
            package,
            max_chars=max_chars,
            initial_chars=initial_chars,
            trimmed=True,
        )
        selected_chars = json_size_chars(package)
        package["selected_context_chars"] = selected_chars
        package["selected_context_size"] = {
            "chars": selected_chars,
            "tokens_estimate": max(1, selected_chars // 4),
        }
        package["composer_context_size"] = dict(package["selected_context_size"])
    _sync_package_budget(
        package,
        max_chars=max_chars,
        initial_chars=initial_chars,
        trimmed=True,
    )
    _sync_package_candidate_exposure_policy_budget(package)


def _mark_timeout_fallback_policy(
    package: dict[str, Any],
    fallback: Mapping[str, Any],
) -> None:
    policy = _safe_mapping(package.get("package_candidate_exposure_policy"))
    package["package_candidate_exposure_policy"] = {
        **policy,
        "timeout_fallback": _jsonable(fallback),
        "silent_trimming": False,
    }


def _last_llm_call_stage(diagnostics: Mapping[str, Any]) -> str:
    stages = _string_list(diagnostics.get("llm_call_stages"))
    return stages[-1] if stages else ""


def _llm_call_budget_exceeded_outcome(
    *,
    package: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    error: LlmCallBudgetExceededError,
) -> LlmConfiguratorOutcome:
    budget_diagnostics = {
        **_safe_mapping(diagnostics),
        "llm_call_budget_exceeded": True,
        "final_status_source": "llm_call_budget_exceeded",
    }
    reason = {
        "structured_no_recommendation": True,
        "summary": "LLM call budget was exceeded before a safe complete BOM was produced.",
        "fallback_reason": "llm_call_budget_exceeded",
        "diagnostic_notes": [
            "The bounded Composer cascade stopped instead of continuing hidden LLM calls."
        ],
        "missing_roles": [],
        "missing_required_capabilities": [],
        "hard_mismatches": [],
        "stock_shortages": [],
        "role_analysis": [],
        "considered_candidate_ids": {},
        "recommended_next_actions": [
            "Inspect llm_call_count and llm_call_stages, then rerun in deep_audit only if needed."
        ],
    }
    return LlmConfiguratorOutcome(
        enabled=True,
        output_mode=_normalize_output_mode(package.get("output_mode")),
        primary_recommendation_status="no_recommendation",
        no_recommendation_reason=reason,
        commercial_summary=_commercial_summary_from_no_recommendation(reason),
        fallback_reason="llm_call_budget_exceeded",
        error_type=type(error).__name__,
        parse_diagnostics=budget_diagnostics,
        internal_warnings=["llm_call_budget_exceeded"],
        final_status_source="llm_call_budget_exceeded",
    )


def _multi_pass_chunk_size(settings: LlmSettings) -> int:
    return max(1, _int_value(getattr(settings, "llm_composer_multi_pass_chunk_size", None)) or 80)


def _package_primary_object(package: Mapping[str, Any]) -> str:
    normalized = _safe_mapping(package.get("normalized_requirements"))
    contract = _safe_mapping(package.get("requirement_contract"))
    return str(
        package.get("primary_object")
        or normalized.get("primary_object")
        or contract.get("primary_object")
        or package.get("product_group")
        or ""
    ).strip()


def _requirement_contract_pass_package(package: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "multi_pass_stage": "requirement_contract",
        "pipeline_version": package.get("pipeline_version"),
        "original_request_text": package.get("original_request_text")
        or package.get("user_request"),
        "primary_product_group": package.get("product_group"),
        "primary_object": _safe_mapping(package.get("normalized_requirements")).get(
            "primary_object"
        ),
        "allowed_roles": [
            SERVER_PLATFORM_ROLE,
            CPU_ROLE,
            RAM_ROLE,
            SSD_ROLE,
            HDD_ROLE,
            DRIVE_ROLE,
            STORAGE_CONTROLLER_ROLE,
            NETWORK_ADAPTER_ROLE,
            POWER_SUPPLY_ROLE,
            CABLE_ROLE,
            RAIL_KIT_ROLE,
            GPU_ROLE,
            TRANSCEIVER_ROLE,
            DAC_CABLE_ROLE,
            LICENSE_ROLE,
            SUPPORT_ROLE,
            OTHER_ACCESSORY_ROLE,
            SWITCH_ROLE,
            ROUTER_ROLE,
            FIREWALL_ROLE,
            ACCESS_POINT_ROLE,
            STORAGE_SYSTEM_ROLE,
            STORAGE_ARRAY_CONTROLLER_ROLE,
            CONTROLLER_MODULE_ROLE,
            DISK_SHELF_ROLE,
            CACHE_ROLE,
            HOST_PORT_ROLE,
            PROTOCOL_MODULE_ROLE,
            STACKING_MODULE_ROLE,
        ],
        "planner_context": {
            "component_role_indicators": _mapping_rows(
                package.get("component_role_indicators")
            ),
            "embedded_requirements": _mapping_rows(
                package.get("embedded_requirements")
            ),
            "requirement_fulfillment_decision": _mapping_rows(
                package.get("requirement_fulfillment_decision")
            ),
            "role_fulfillment_diagnostics": _mapping_rows(
                package.get("role_fulfillment_diagnostics")
            ),
            "role_lifecycle_trace": _mapping_rows(
                package.get("role_lifecycle_trace")
            ),
            "roles_dropped_reason_by_role": _safe_mapping(
                package.get("roles_dropped_reason_by_role")
            ),
            "accessory_indicators": _string_list(package.get("accessory_indicators")),
            "service_support_indicators": _string_list(
                package.get("service_support_indicators")
            ),
            "logistics_commercial_constraints": _string_list(
                package.get("logistics_commercial_constraints")
            ),
            "broad_role_hints": _string_list(package.get("broad_role_hints")),
            "roles_sent_to_composer": _string_list(
                package.get("roles_sent_to_composer")
            ),
            "hard_purchasable_bom_roles": _string_list(
                package.get("hard_purchasable_bom_roles")
            ),
            "primary_object_feature_requirements": _mapping_rows(
                package.get("primary_object_feature_requirements")
            ),
            "optional_accessory_engineering_roles": _string_list(
                package.get("optional_accessory_engineering_roles")
            ),
            "optional_accessory_engineering_requirements": _mapping_rows(
                package.get("optional_accessory_engineering_requirements")
            ),
        },
    }


def _fallback_requirement_contract_from_package(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_requirements = _safe_mapping(package.get("normalized_requirements"))
    role_plan = _safe_mapping(
        package.get("role_plan") or normalized_requirements.get("role_plan")
    )
    roles = _unique(
        [
            *_string_list(package.get("hard_purchasable_bom_roles")),
            *_string_list(package.get("required_roles")),
            *_string_list(role_plan.get("required_roles")),
            *_string_list(package.get("composer_package_roles")),
            *_string_list(package.get("roles_sent_to_composer")),
        ]
    )
    quantities = _safe_mapping(
        normalized_requirements.get("required_quantities_by_role")
        or role_plan.get("requirements_by_role")
    )
    return {
        "primary_object": _package_primary_object(package),
        "required_roles": roles,
        "required_quantities_by_role": quantities,
        "hard_requirements": _mapping_rows(
            package.get("purchasable_role_requirements")
            or package.get("hard_purchasable_bom_role_requirements")
            or normalized_requirements.get("purchasable_role_requirements")
        ),
        "optional_requirements": _mapping_rows(
            package.get("optional_accessory_engineering_requirements")
            or normalized_requirements.get("optional_accessory_engineering_requirements")
        ),
        "primary_object_features": _mapping_rows(
            package.get("primary_object_feature_requirements")
            or normalized_requirements.get("primary_object_feature_requirements")
        ),
        "purchasable_component_roles": roles,
        "accessories": _mapping_rows(
            package.get("accessory_or_consumable_requirements")
            or normalized_requirements.get("accessory_or_consumable_requirements")
        ),
        "services_support": _string_list(package.get("service_support_indicators")),
        "logistics_commercial_constraints": _mapping_rows(
            package.get("logistics_or_commercial_constraint_requirements")
            or normalized_requirements.get("logistics_or_commercial_constraint_requirements")
        ),
        "fulfillment_expectations": _mapping_rows(
            package.get("requirement_fulfillment_decision")
        ),
        "engineer_checks": _unique(
            [
                *_string_list(package.get("engineering_check_requirements")),
                "Requirement contract fallback was synthesized from the validated v2 package.",
            ]
        ),
    }


def _parse_requirement_contract_payload(payload: Any) -> RequirementContractPayload:
    source = _safe_mapping(payload)
    contract = source.get("requirement_contract")
    if isinstance(contract, Mapping):
        source = dict(contract)
    return RequirementContractPayload.model_validate(source)


def _parse_completeness_critic_payload(payload: Any) -> CompletenessCriticPayload:
    source = _safe_mapping(payload)
    critic = source.get("completeness_critic_result") or source.get("critic")
    if isinstance(critic, Mapping):
        source = dict(critic)
    return CompletenessCriticPayload.model_validate(source)


def _apply_requirement_contract_to_package(
    package: dict[str, Any],
    contract: Mapping[str, Any],
) -> None:
    roles = _contract_required_roles(contract, package=package)
    normalized_requirements = dict(_safe_mapping(package.get("normalized_requirements")))
    role_plan = dict(_safe_mapping(normalized_requirements.get("role_plan")))
    role_plan.update(
        {
            "required_roles": roles,
            "requirement_contract": _jsonable(contract),
            "requirements_by_role": _normalized_contract_quantities_by_role(
                contract,
                package=package,
            ),
        }
    )
    normalized_requirements.update(
        {
            "required_roles": roles,
            "requirement_contract": _jsonable(contract),
            "required_quantities_by_role": _safe_mapping(
                contract.get("required_quantities_by_role")
            ),
            "role_plan": role_plan,
            **_requirements_from_contract_quantities(contract, roles=roles, package=package),
        }
    )
    package["requirement_contract"] = _jsonable(contract)
    package["required_roles"] = roles
    package["normalized_requirements"] = normalized_requirements
    package["role_plan"] = {
        **_safe_mapping(package.get("role_plan")),
        "required_roles": roles,
        "requirement_contract": _jsonable(contract),
        "requirements_by_role": role_plan["requirements_by_role"],
    }


def _contract_required_roles(
    contract: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
) -> list[str]:
    raw_roles = [
        *_string_list(contract.get("required_roles")),
        *_string_list(contract.get("purchasable_component_roles")),
        *[
            str(role)
            for role in _safe_mapping(contract.get("required_quantities_by_role"))
        ],
    ]
    roles: list[str] = []
    for role in raw_roles:
        normalized = _normalize_contract_role(role, package=package)
        if normalized is not None:
            roles.append(normalized)
    roles = _unique(roles)
    hard_bom_roles = _string_list(package.get("hard_purchasable_bom_roles"))
    if _is_v2_composer_package(package) and hard_bom_roles:
        hard_set = set(hard_bom_roles)
        return [role for role in roles if role in hard_set]
    return roles


def _normalize_contract_role(role: Any, *, package: Mapping[str, Any]) -> str | None:
    text = str(role or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "platform": SERVER_PLATFORM_ROLE,
        "server": SERVER_PLATFORM_ROLE,
        "server_chassis": SERVER_PLATFORM_ROLE,
        "server_platform": SERVER_PLATFORM_ROLE,
        "processor": CPU_ROLE,
        "processors": CPU_ROLE,
        "memory": RAM_ROLE,
        "storage_controller": STORAGE_CONTROLLER_ROLE,
        "hba": STORAGE_CONTROLLER_ROLE,
        "nic": NETWORK_ADAPTER_ROLE,
        "network": NETWORK_ADAPTER_ROLE,
        "network_adapter": NETWORK_ADAPTER_ROLE,
        "psu": POWER_SUPPLY_ROLE,
        "power": POWER_SUPPLY_ROLE,
        "power_supply": POWER_SUPPLY_ROLE,
        "power_cable": CABLE_ROLE,
        "power_cord": CABLE_ROLE,
        "cables": CABLE_ROLE,
        "services/support": SUPPORT_ROLE,
        "services_support": SUPPORT_ROLE,
    }
    text = aliases.get(text, text)
    if text == "storage":
        product_group = str(package.get("product_group") or "").strip()
        matrix = _safe_mapping(package.get("component_candidate_matrix"))
        if product_group == SERVER_PRODUCT_GROUP:
            for candidate_role in (SSD_ROLE, HDD_ROLE, DRIVE_ROLE):
                prompt_role = PROMPT_ROLE_BY_INTERNAL_ROLE.get(candidate_role, candidate_role)
                if _mapping_rows(matrix.get(prompt_role)):
                    return candidate_role
            return DRIVE_ROLE
        return STORAGE_SYSTEM_ROLE
    return INTERNAL_ROLE_BY_PROMPT_ROLE.get(text)


def _normalized_contract_quantities_by_role(
    contract: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_role, value in _safe_mapping(contract.get("required_quantities_by_role")).items():
        role = _normalize_contract_role(raw_role, package=package)
        if role is not None:
            result[role] = value
    return result


def _requirements_from_contract_quantities(
    contract: Mapping[str, Any],
    *,
    roles: Sequence[str],
    package: Mapping[str, Any],
) -> dict[str, Any]:
    quantities = _normalized_contract_quantities_by_role(contract, package=package)
    result: dict[str, Any] = {}
    server_quantity = _contract_role_count(quantities.get(SERVER_PLATFORM_ROLE)) or 1
    result["server_qty"] = server_quantity
    if CPU_ROLE in roles:
        cpu = _safe_mapping(quantities.get(CPU_ROLE))
        cpu_count = _contract_role_count(cpu)
        if cpu_count is not None:
            result["total_cpu_required"] = cpu_count
            result["cpu_per_server"] = max(1, ceil(cpu_count / server_quantity))
        cores = _first_int_from_keys(cpu, ("min_cores", "min_cores_per_cpu", "cores"))
        if cores is not None:
            result["cpu_min_cores_per_cpu"] = cores
        frequency = cpu.get("min_frequency_ghz") or cpu.get("frequency_ghz")
        if frequency not in (None, ""):
            result["cpu_min_frequency_ghz"] = frequency
    if RAM_ROLE in roles:
        ram = _safe_mapping(quantities.get(RAM_ROLE))
        module_count = _first_int_from_keys(
            ram,
            ("module_count", "modules", "count", "quantity", "total_quantity"),
        )
        module_gb = _first_int_from_keys(
            ram,
            ("module_capacity_gb", "capacity_per_module_gb", "capacity_gb"),
        )
        total_gb = _first_int_from_keys(
            ram,
            ("total_gb", "total_capacity_gb", "min_total_gb", "capacity_total_gb"),
        )
        if total_gb is None and module_count is not None and module_gb is not None:
            total_gb = module_count * module_gb
        if total_gb is not None:
            result["ram_gb_per_server"] = max(1, ceil(total_gb / server_quantity))
        if module_count is not None:
            result["ram_module_count_total"] = module_count
    storage_role = next((role for role in (SSD_ROLE, HDD_ROLE, DRIVE_ROLE) if role in roles), None)
    if storage_role is not None:
        storage = _safe_mapping(quantities.get(storage_role)) or _safe_mapping(
            _safe_mapping(contract.get("required_quantities_by_role")).get("storage")
        )
        drive_count = _contract_role_count(storage)
        if drive_count is not None:
            result["storage_qty_per_server"] = max(1, ceil(drive_count / server_quantity))
        result["storage_required"] = True
        result["storage_type_preference"] = storage_role
    if NETWORK_ADAPTER_ROLE in roles:
        network = _safe_mapping(quantities.get(NETWORK_ADAPTER_ROLE))
        result["network_required"] = True
        result["network_adapter_required"] = True
        min_ports = _first_int_from_keys(
            network,
            ("min_ports_per_server", "ports_per_adapter", "ports", "port_count"),
        )
        if min_ports is not None:
            result["network_min_ports_per_server"] = min_ports
        speed = network.get("speed") or network.get("speed_gbps")
        if speed not in (None, ""):
            result["network_speed"] = speed
        media = network.get("media") or network.get("connector")
        if media not in (None, ""):
            result["network_media"] = media
    return result


def _first_int_from_keys(source: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = _int_value(source.get(key))
        if value is not None and value > 0:
            return value
    return None


def _contract_role_count(value: Any) -> int | None:
    if isinstance(value, Mapping):
        direct = _first_int_from_keys(
            value,
            (
                "count",
                "quantity",
                "total_quantity",
                "module_count",
                "modules",
                "drive_count",
                "device_count",
                "psu_count",
                "cable_count",
            ),
        )
        if direct is not None:
            return direct
        groups = value.get("groups")
        if isinstance(groups, list):
            total = 0
            for group in groups:
                group_count = _contract_role_count(group)
                if group_count is not None:
                    total += group_count
            return total if total > 0 else None
        return None
    return _int_value(value)


def _run_role_evaluation_passes(
    *,
    client: LlmClient,
    package: Mapping[str, Any],
    contract: Mapping[str, Any],
    roles: Sequence[str],
    settings: LlmSettings,
) -> list[_MultiPassRoleEvaluation]:
    evaluations: list[_MultiPassRoleEvaluation] = []
    chunk_size = _multi_pass_chunk_size(settings)
    for role in roles:
        rows = _candidate_rows_for_contract_role(package, role)
        candidate_ids = _candidate_ids_for_rows(rows)
        summaries: list[dict[str, Any]] = []
        failed_chunks: list[dict[str, Any]] = []
        chunks = _chunks(rows, chunk_size) or [[]]
        for index, chunk in enumerate(chunks):
            chunk_ids = _candidate_ids_for_rows(chunk)
            try:
                payload = client.generate_json(
                    LLM_ROLE_EVALUATION_SYSTEM_PROMPT,
                    build_llm_configurator_user_prompt(
                        {
                            "multi_pass_stage": "role_evaluation",
                            "role": role,
                            "chunk_index": index,
                            "chunk_count": len(chunks),
                            "original_request_text": package.get("original_request_text")
                            or package.get("user_request"),
                            "requirement_contract": _jsonable(contract),
                            "candidate_count_for_role": len(candidate_ids),
                            "candidate_ids_for_chunk": chunk_ids,
                            "candidates": _jsonable(chunk),
                        }
                    ),
                )
                parsed = _parse_role_evaluation_payload(payload, role=role)
                summaries.append(parsed)
            except LlmCallBudgetExceededError:
                raise
            except (LlmError, ValidationError) as exc:
                failed_chunks.append(
                    {
                        "role": role,
                        "chunk_index": index,
                        "candidate_ids": chunk_ids,
                        "error_type": type(exc).__name__,
                    }
                )
        evaluations.append(
            _MultiPassRoleEvaluation(
                role=role,
                candidate_ids=candidate_ids,
                summaries=summaries,
                failed_chunks=failed_chunks,
            )
        )
    return evaluations


def _parse_role_evaluation_payload(payload: Any, *, role: str) -> dict[str, Any]:
    source = _safe_mapping(payload)
    row = source.get("role_evaluation")
    if isinstance(row, Mapping):
        source = dict(row)
    source.setdefault("role", role)
    parsed = RoleEvaluationPayload.model_validate(source)
    result = parsed.model_dump()
    result["rejected_candidate_ids"] = _normalize_rejected_candidate_ids(
        result.get("rejected_candidate_ids")
    )
    return result


def _normalize_rejected_candidate_ids(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    items = value if isinstance(value, list) else []
    for item in items:
        if isinstance(item, Mapping):
            candidate_id = str(
                item.get("component_candidate_id") or item.get("candidate_id") or ""
            ).strip()
            if candidate_id:
                rows.append(
                    {
                        "component_candidate_id": candidate_id,
                        "reason": str(item.get("reason") or "").strip(),
                    }
                )
            continue
        candidate_id = str(item or "").strip()
        if candidate_id:
            rows.append({"component_candidate_id": candidate_id, "reason": ""})
    return rows


def _role_evaluation_diagnostics(
    evaluations: Sequence[_MultiPassRoleEvaluation],
) -> dict[str, Any]:
    count_by_role: dict[str, int] = {}
    coverage_by_role: dict[str, Any] = {}
    failed_chunks: list[dict[str, Any]] = []
    for evaluation in evaluations:
        considered = evaluation.considered_candidate_ids
        missing = sorted(set(evaluation.candidate_ids).difference(considered))
        count_by_role[evaluation.role] = len(considered)
        coverage_by_role[evaluation.role] = {
            "candidate_count": len(evaluation.candidate_ids),
            "considered_count": len(considered),
            "all_candidates_considered": not missing and not evaluation.failed_chunks,
            "missing_candidate_ids": missing,
            "failed_chunk_count": len(evaluation.failed_chunks),
        }
        failed_chunks.extend(evaluation.failed_chunks)
    return {
        "role_evaluation_used": bool(evaluations),
        "role_evaluation_count_by_role": count_by_role,
        "role_evaluation_coverage_by_role": coverage_by_role,
        "role_evaluation_failed_chunks": failed_chunks,
    }


def _candidate_rows_for_contract_role(
    package: Mapping[str, Any],
    role: str,
) -> list[Mapping[str, Any]]:
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    prompt_role = PROMPT_ROLE_BY_INTERNAL_ROLE.get(role, role)
    rows = _mapping_rows(matrix.get(prompt_role))
    if rows:
        return rows
    if role == SERVER_PLATFORM_ROLE:
        return _mapping_rows(matrix.get("platform"))
    if role == DRIVE_ROLE:
        result: list[Mapping[str, Any]] = []
        for prompt in (DRIVE_ROLE, SSD_ROLE, HDD_ROLE, "storage"):
            result.extend(_mapping_rows(matrix.get(prompt)))
        return result
    return []


def _candidate_ids_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return _unique(
        [
            str(row.get("component_candidate_id") or row.get("candidate_id") or "").strip()
            for row in rows
            if str(row.get("component_candidate_id") or row.get("candidate_id") or "").strip()
        ]
    )


def _chunks(
    rows: Sequence[Mapping[str, Any]],
    chunk_size: int,
) -> list[list[Mapping[str, Any]]]:
    if not rows:
        return []
    return [list(rows[index : index + chunk_size]) for index in range(0, len(rows), chunk_size)]


def _bom_composition_pass_package(
    *,
    package: Mapping[str, Any],
    contract: Mapping[str, Any],
    evaluations: Sequence[_MultiPassRoleEvaluation],
) -> dict[str, Any]:
    diagnostics = _role_evaluation_diagnostics(evaluations)
    return {
        "multi_pass_stage": "bom_composition",
        "v2_package_mode": package.get("v2_package_mode")
        or package.get("selected_package_mode"),
        "selected_context_chars": package.get("selected_context_chars"),
        "original_request_text": package.get("original_request_text")
        or package.get("user_request"),
        "product_group": package.get("product_group"),
        "primary_object": package.get("primary_object"),
        "component_role_indicators": _mapping_rows(
            package.get("component_role_indicators")
        ),
        "embedded_requirements": _mapping_rows(package.get("embedded_requirements")),
        "requirement_fulfillment_decision": _mapping_rows(
            package.get("requirement_fulfillment_decision")
        ),
        "role_fulfillment_diagnostics": _mapping_rows(
            package.get("role_fulfillment_diagnostics")
        ),
        "roles_dropped_reason_by_role": _safe_mapping(
            package.get("roles_dropped_reason_by_role")
        ),
        "role_lifecycle_trace": _mapping_rows(package.get("role_lifecycle_trace")),
        "requirement_contract": _jsonable(contract),
        "component_candidate_matrix": _jsonable(
            package.get("component_candidate_matrix")
        ),
        "composer_contract_checklist": {
            "required_roles": _contract_required_roles(contract, package=package),
            "required_quantities_by_role": _normalized_contract_quantities_by_role(
                contract,
                package=package,
            ),
            "hard_requirements": _jsonable(contract.get("hard_requirements")),
            "primary_object_features": _jsonable(
                contract.get("primary_object_features")
            ),
        },
        "role_evaluation_summaries": _jsonable(
            {evaluation.role: evaluation.summaries for evaluation in evaluations}
        ),
        "role_evaluation_coverage_by_role": diagnostics[
            "role_evaluation_coverage_by_role"
        ],
        "failed_chunks": diagnostics["role_evaluation_failed_chunks"],
        "candidate_facts_by_role": _candidate_facts_for_bom_pass(
            package=package,
            evaluations=evaluations,
        ),
        "output_mode": package.get("output_mode"),
    }


def _candidate_facts_for_bom_pass(
    *,
    package: Mapping[str, Any],
    evaluations: Sequence[_MultiPassRoleEvaluation],
) -> dict[str, Any]:
    rows_by_id = _candidate_rows_by_id(package)
    result: dict[str, list[Mapping[str, Any]]] = {}
    for evaluation in evaluations:
        selected_ids = _role_evaluation_candidate_context_ids(evaluation)
        if not selected_ids:
            selected_ids = evaluation.candidate_ids
        result[evaluation.role] = [
            rows_by_id[candidate_id]
            for candidate_id in selected_ids
            if candidate_id in rows_by_id
        ]
    return _jsonable(result)


def _role_evaluation_candidate_context_ids(
    evaluation: _MultiPassRoleEvaluation,
) -> list[str]:
    ids: list[str] = []
    for summary in evaluation.summaries:
        ids.extend(_string_list(summary.get("best_candidate_ids")))
        ids.extend(_string_list(summary.get("exact_or_equivalent_candidates")))
        ids.extend(_string_list(summary.get("cheapest_safe_candidates")))
        ids.extend(_string_list(summary.get("uncertain_candidate_ids")))
    known = set(evaluation.candidate_ids)
    return _unique([candidate_id for candidate_id in ids if candidate_id in known])


def _candidate_rows_by_id(package: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    for rows in matrix.values():
        for row in _mapping_rows(rows):
            candidate_id = str(
                row.get("component_candidate_id") or row.get("candidate_id") or ""
            ).strip()
            if candidate_id:
                result[candidate_id] = row
    return result


def _virtual_role_evaluations_for_contract(
    *,
    package: Mapping[str, Any],
    roles: Sequence[str],
) -> list[_MultiPassRoleEvaluation]:
    evaluations: list[_MultiPassRoleEvaluation] = []
    for role in roles:
        candidate_ids = _candidate_ids_for_rows(_candidate_rows_for_contract_role(package, role))
        evaluations.append(
            _MultiPassRoleEvaluation(
                role=role,
                candidate_ids=candidate_ids,
                summaries=[],
                failed_chunks=[],
            )
        )
    return evaluations


def _composer_code_completeness_check(
    *,
    package: Mapping[str, Any],
    contract: Mapping[str, Any],
    proposed_bom: LlmComposerResponsePayload,
) -> dict[str, Any]:
    if _has_structured_no_recommendation(proposed_bom):
        return {
            "repair_required": False,
            "structured_no_recommendation": True,
            "missing_roles": [],
            "insufficient_quantities": [],
            "empty_requirement_analysis": False,
            "unknown_component_ids": [],
            "hard_mismatch_risks": [],
        }

    required_roles = _contract_required_roles(contract, package=package)
    selected = proposed_bom.recommendations[0] if proposed_bom.recommendations else None
    selected_roles: set[str] = set()
    selected_quantities: dict[str, int] = {}
    unknown_component_ids: list[dict[str, Any]] = []
    if selected is not None:
        rows_by_id = _candidate_rows_by_id(package)
        component_ids = {
            **selected.component_candidate_ids,
            **selected.selected_component_candidate_ids,
        }
        for role_key, raw_candidate_id in component_ids.items():
            candidate_id = str(raw_candidate_id or "").strip()
            if not candidate_id:
                continue
            candidate_row = rows_by_id.get(candidate_id)
            if candidate_row is None:
                unknown_component_ids.append(
                    {"role": role_key, "component_candidate_id": candidate_id}
                )
                normalized_role = _normalize_contract_role(role_key, package=package)
            else:
                normalized_role = _selected_role_from_candidate_row(
                    role_key,
                    candidate_row,
                    package=package,
                )
            if normalized_role is None:
                continue
            selected_roles.add(normalized_role)
            quantity = _selected_quantity_for_role(
                selected.quantities,
                role_key=role_key,
                normalized_role=normalized_role,
            )
            if quantity is not None:
                selected_quantities[normalized_role] = (
                    selected_quantities.get(normalized_role, 0) + quantity
                )

    missing_roles = [
        role
        for role in required_roles
        if not _required_role_satisfied_by_selection(role, selected_roles)
    ]
    insufficient_quantities = _composer_insufficient_quantities(
        contract=contract,
        package=package,
        selected_quantities=selected_quantities,
        selected_roles=selected_roles,
    )
    empty_requirement_analysis = not bool(proposed_bom.requirement_analysis)
    hard_mismatch_risks = _mapping_rows(proposed_bom.hard_mismatch_risks)
    empty_bom_without_no_recommendation = not proposed_bom.recommendations
    repair_required = bool(
        empty_bom_without_no_recommendation
        or (
            proposed_bom.recommendations
            and (
                missing_roles
                or insufficient_quantities
                or empty_requirement_analysis
                or unknown_component_ids
                or hard_mismatch_risks
            )
        )
    )
    return {
        "repair_required": repair_required,
        "structured_no_recommendation": False,
        "required_roles": required_roles,
        "selected_roles": sorted(selected_roles),
        "missing_roles": missing_roles,
        "selected_quantities_by_role": selected_quantities,
        "insufficient_quantities": insufficient_quantities,
        "empty_requirement_analysis": empty_requirement_analysis,
        "empty_response": empty_bom_without_no_recommendation,
        "unknown_component_ids": unknown_component_ids,
        "hard_mismatch_risks": hard_mismatch_risks,
    }


def _selected_role_from_candidate_row(
    role_key: str,
    candidate_row: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
) -> str | None:
    prompt_role = str(candidate_row.get("role") or "").strip()
    if prompt_role:
        return INTERNAL_ROLE_BY_PROMPT_ROLE.get(prompt_role, prompt_role)
    return _normalize_contract_role(role_key, package=package)


def _selected_quantity_for_role(
    quantities: Mapping[str, Any],
    *,
    role_key: str,
    normalized_role: str,
) -> int | None:
    prompt_role = PROMPT_ROLE_BY_INTERNAL_ROLE.get(normalized_role, normalized_role)
    for key in (role_key, normalized_role, prompt_role):
        quantity = _int_value(quantities.get(key))
        if quantity is not None:
            return quantity
    return None


def _required_role_satisfied_by_selection(
    required_role: str,
    selected_roles: set[str],
) -> bool:
    if required_role in selected_roles:
        return True
    if required_role == "storage":
        return bool(
            {STORAGE_SYSTEM_ROLE, DRIVE_ROLE, SSD_ROLE, HDD_ROLE}.intersection(
                selected_roles
            )
        )
    if required_role == DRIVE_ROLE:
        return bool({SSD_ROLE, HDD_ROLE, DRIVE_ROLE}.intersection(selected_roles))
    if required_role in {SSD_ROLE, HDD_ROLE}:
        return required_role in selected_roles or DRIVE_ROLE in selected_roles
    return False


def _composer_insufficient_quantities(
    *,
    contract: Mapping[str, Any],
    package: Mapping[str, Any],
    selected_quantities: Mapping[str, int],
    selected_roles: set[str],
) -> list[dict[str, Any]]:
    quantities = _normalized_contract_quantities_by_role(contract, package=package)
    result: list[dict[str, Any]] = []
    for role, value in quantities.items():
        if not _required_role_satisfied_by_selection(role, selected_roles):
            continue
        required = _contract_role_count(value)
        if required is None or required <= 1:
            continue
        selected = selected_quantities.get(role)
        if selected is None and role in {SSD_ROLE, HDD_ROLE, DRIVE_ROLE}:
            selected = max(
                selected_quantities.get(SSD_ROLE, 0),
                selected_quantities.get(HDD_ROLE, 0),
                selected_quantities.get(DRIVE_ROLE, 0),
            )
        if selected is None or selected < required:
            result.append(
                {
                    "role": role,
                    "required_quantity": required,
                    "selected_quantity": selected or 0,
                }
            )
    return result


def _critic_from_code_completeness(
    completeness: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "all_hard_requirements_covered": not bool(
            completeness.get("repair_required")
        ),
        "missing_roles": _string_list(completeness.get("missing_roles")),
        "insufficient_quantities": _mapping_rows(
            completeness.get("insufficient_quantities")
        ),
        "unverified_requirements": (
            [{"reason": "empty_requirement_analysis"}]
            if completeness.get("empty_requirement_analysis")
            else []
        ),
        "hard_mismatch_risks": [
            *_mapping_rows(completeness.get("hard_mismatch_risks")),
            *_mapping_rows(completeness.get("unknown_component_ids")),
        ],
        "recommended_repair_actions": [
            (
                "Return a complete BOM covering every hard required role, or return "
                "structured no_recommendation."
            )
        ],
    }


def _composer_response_from_completeness_failure(
    *,
    contract: Mapping[str, Any],
    completeness: Mapping[str, Any],
) -> LlmComposerResponsePayload:
    missing_roles = _string_list(completeness.get("missing_roles"))
    insufficient = _mapping_rows(completeness.get("insufficient_quantities"))
    hard_mismatches = [
        *_mapping_rows(completeness.get("hard_mismatch_risks")),
        *_mapping_rows(completeness.get("unknown_component_ids")),
    ]
    return LlmComposerResponsePayload.model_validate(
        {
            "requirement_analysis": {
                "code_completeness_failure": _jsonable(completeness),
                "requirement_contract_used": True,
            },
            "recommendations": [],
            "no_recommendation": {
                "summary": "Composer returned an incomplete or unsafe BOM.",
                "missing_roles": missing_roles,
                "missing_required_capabilities": [],
                "hard_mismatches": hard_mismatches,
                "stock_shortages": [],
                "role_analysis": [
                    {
                        "role": role,
                        "status": "missing",
                        "explanation": "Required by requirement_contract but absent from the BOM.",
                    }
                    for role in missing_roles
                ],
                "insufficient_quantities": insufficient,
                "considered_candidate_ids": {},
                "explanation_ru": (
                    "Composer вернул неполный BOM: не закрыты все обязательные "
                    "роли или количества из requirement_contract."
                ),
                "recommended_next_actions": [
                    "Re-run with deep audit or ask an engineer to review the contract and matrix."
                ],
            },
            "general_notes": [],
        }
    )


def _completeness_critic_pass_package(
    *,
    package: Mapping[str, Any],
    contract: Mapping[str, Any],
    proposed_bom: LlmComposerResponsePayload,
    code_completeness: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "multi_pass_stage": "completeness_critic",
        "original_request_text": package.get("original_request_text")
        or package.get("user_request"),
        "requirement_contract": _jsonable(contract),
        "proposed_bom": _jsonable(proposed_bom.model_dump()),
        "code_completeness": _jsonable(code_completeness),
    }


def _critic_requires_repair(critic: Mapping[str, Any]) -> bool:
    return not bool(critic.get("all_hard_requirements_covered")) or any(
        _mapping_rows(critic.get(key)) or _string_list(critic.get(key))
        for key in (
            "missing_roles",
            "insufficient_quantities",
            "unverified_requirements",
            "hard_mismatch_risks",
        )
    )


def _repair_pass_package(
    *,
    package: Mapping[str, Any],
    contract: Mapping[str, Any],
    evaluations: Sequence[_MultiPassRoleEvaluation],
    proposed_bom: LlmComposerResponsePayload,
    criticism: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "multi_pass_stage": "repair",
        "original_request_text": package.get("original_request_text")
        or package.get("user_request"),
        "requirement_contract": _jsonable(contract),
        "criticism": _jsonable(criticism),
        "proposed_bom": _jsonable(proposed_bom.model_dump()),
        "component_candidate_matrix": _jsonable(
            package.get("component_candidate_matrix")
        ),
        "role_evaluation_summaries": _jsonable(
            {evaluation.role: evaluation.summaries for evaluation in evaluations}
        ),
        "candidate_facts_by_role": _candidate_facts_for_bom_pass(
            package=package,
            evaluations=evaluations,
        ),
    }


def _generate_composer_response(
    *,
    package: Mapping[str, Any],
    settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings,
    llm_client: LlmClient | None,
    llm_call_budget: LlmCallBudget | None,
    online_composer_requested: bool,
    output_mode: str,
) -> _ComposerResponseResult | LlmConfiguratorOutcome:
    online_diagnostics = _online_composer_diagnostics(
        settings=web_evidence_settings,
        online_composer_used=False,
        parse_status="not_requested" if not online_composer_requested else "not_started",
    )
    online_diagnostics["composer_mode"] = "single_pass"
    if online_composer_requested:
        result = _try_composer_response(
            package=package,
            settings=settings,
            web_evidence_settings=web_evidence_settings,
            llm_client=llm_client,
            llm_call_budget=llm_call_budget,
            system_prompt=LLM_ONLINE_COMPOSER_SYSTEM_PROMPT,
            online_composer=True,
        )
        if not isinstance(result, _ComposerResponseResult):
            retry_result = _retry_context_limit_with_compact_package(
                result=result,
                package=package,
                settings=settings,
                web_evidence_settings=web_evidence_settings,
                llm_client=llm_client,
                llm_call_budget=llm_call_budget,
                system_prompt=LLM_ONLINE_COMPOSER_SYSTEM_PROMPT,
                online_composer=True,
            )
            if retry_result is not None:
                result = retry_result
        if isinstance(result, _ComposerResponseResult):
            repair_result = _run_online_empty_response_repair_if_needed(
                package=package,
                client=result.client,
                response_result=result,
            )
            if repair_result.attempted and repair_result.response is not None:
                result = _ComposerResponseResult(
                    response=repair_result.response,
                    client=result.client,
                    owns_client=result.owns_client,
                    online_composer_used=result.online_composer_used,
                    online_diagnostics=result.online_diagnostics,
                    schema_rejections=repair_result.schema_rejections,
                    proposal_indexes=repair_result.proposal_indexes,
                    proposal_count=repair_result.proposal_count,
                )
            return _ComposerResponseResult(
                response=result.response,
                client=result.client,
                owns_client=result.owns_client,
                online_composer_used=True,
                online_diagnostics={
                    **_online_composer_diagnostics(
                        settings=web_evidence_settings,
                        online_composer_used=True,
                        parse_status="parsed",
                    ),
                    **result.online_diagnostics,
                    "online_composer_empty_response_repair_attempted": (
                        repair_result.attempted
                    ),
                    "online_composer_empty_response_repair_success": (
                        repair_result.success
                    ),
                    "structured_no_recommendation_used": (
                        _has_structured_no_recommendation(result.response)
                    ),
                    **(
                        {
                            "online_composer_empty_response_repair_error_type": (
                                repair_result.error_type
                            )
                        }
                        if repair_result.error_type
                        else {}
                    ),
                    **(
                        {
                            "online_composer_empty_response_repair_parse_status": (
                                repair_result.parse_status
                            )
                        }
                        if repair_result.parse_status
                        else {}
                    ),
                },
                schema_rejections=result.schema_rejections,
                proposal_indexes=result.proposal_indexes,
                proposal_count=result.proposal_count,
            )

        if result.get("provider_error_type") == PROVIDER_CONTEXT_LIMIT_ERROR_TYPE:
            outcome = result.get("outcome")
            if isinstance(outcome, LlmConfiguratorOutcome):
                return outcome

        online_request_attempted = bool(result.get("request_attempted"))
        online_diagnostics = _online_composer_diagnostics(
            settings=web_evidence_settings,
            online_composer_used=online_request_attempted,
            parse_status=result.get("parse_status", "request_failed"),
            error_type=result.get("error_type", ""),
        )
        online_diagnostics["composer_mode"] = "single_pass"
        online_diagnostics.update(
            {
                key: result[key]
                for key in COMPACT_PACKAGE_DIAGNOSTIC_KEYS
                if key in result
            }
        )
        logger.info(
            "Online composer unavailable, falling back to normal composer: %s",
            result.get("error_type", ""),
        )

    normal_result = _try_composer_response(
        package=package,
        settings=settings,
        web_evidence_settings=web_evidence_settings,
        llm_client=llm_client if online_composer_requested else llm_client,
        llm_call_budget=llm_call_budget,
        system_prompt=_configurator_system_prompt(output_mode),
        online_composer=False,
    )
    if not isinstance(normal_result, _ComposerResponseResult):
        retry_result = _retry_context_limit_with_compact_package(
            result=normal_result,
            package=package,
            settings=settings,
            web_evidence_settings=web_evidence_settings,
            llm_client=llm_client if online_composer_requested else llm_client,
            llm_call_budget=llm_call_budget,
            system_prompt=_configurator_system_prompt(output_mode),
            online_composer=False,
        )
        if retry_result is not None:
            normal_result = retry_result
    if isinstance(normal_result, _ComposerResponseResult):
        normal_online_composer_used = (
            bool(online_diagnostics.get("online_composer_used"))
            or normal_result.online_composer_used
        )
        normal_diagnostics = online_diagnostics
        if not online_composer_requested:
            normal_diagnostics = _online_composer_diagnostics(
                settings=web_evidence_settings,
                online_composer_used=normal_online_composer_used,
                parse_status="parsed",
            )
            normal_diagnostics["composer_mode"] = "single_pass"
        normal_diagnostics = {
            **normal_diagnostics,
            **normal_result.online_diagnostics,
            "structured_no_recommendation_used": _has_structured_no_recommendation(
                normal_result.response
            ),
        }
        return _ComposerResponseResult(
            response=normal_result.response,
            client=normal_result.client,
            owns_client=normal_result.owns_client,
            online_composer_used=normal_online_composer_used,
            online_diagnostics=normal_diagnostics,
            schema_rejections=normal_result.schema_rejections,
            proposal_indexes=normal_result.proposal_indexes,
            proposal_count=normal_result.proposal_count,
        )

    outcome = normal_result["outcome"]
    if isinstance(outcome, LlmConfiguratorOutcome):
        request_attempted = bool(online_diagnostics.get("online_composer_used")) or bool(
            normal_result.get("request_attempted")
        )
        error_type = (
            online_diagnostics.get("online_composer_error_type")
            or normal_result.get("error_type", "")
        )
        parse_status = normal_result.get(
            "parse_status",
            "request_failed" if error_type else "not_applicable",
        )
        return _with_online_evidence_diagnostics(
            outcome,
            settings=web_evidence_settings,
            diagnostics={
                **_online_composer_diagnostics(
                    settings=web_evidence_settings,
                    online_composer_used=request_attempted,
                    parse_status=parse_status,
                    error_type=error_type,
                ),
                "composer_mode": "single_pass",
                **{
                    key: normal_result[key]
                    for key in COMPACT_PACKAGE_DIAGNOSTIC_KEYS
                    if key in normal_result
                },
            },
        )
    return outcome


def _retry_context_limit_with_compact_package(
    *,
    result: dict[str, Any],
    package: Mapping[str, Any],
    settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings,
    llm_client: LlmClient | None,
    llm_call_budget: LlmCallBudget | None,
    system_prompt: str,
    online_composer: bool,
) -> _ComposerResponseResult | dict[str, Any] | None:
    if not _composer_result_is_provider_context_limit(result):
        return None
    if not _is_v2_composer_package(package):
        return None

    mode = _composer_package_mode(package)
    if mode == COMPACT_FULL_MATRIX_MODE:
        return _compact_context_limit_result(
            result,
            package=package,
            compact_package=package,
            retry_attempted=False,
        )

    compact_package = prepare_v2_composer_package(
        package,
        max_package_chars=_int_value(
            _safe_mapping(package.get("package_budget")).get("max_chars")
        )
        or settings.llm_configurator_max_package_chars,
        force_mode=COMPACT_FULL_MATRIX_MODE,
    )
    retry_result = _try_composer_response(
        package=compact_package,
        settings=settings,
        web_evidence_settings=web_evidence_settings,
        llm_client=llm_client,
        llm_call_budget=llm_call_budget,
        system_prompt=system_prompt,
        online_composer=online_composer,
    )
    if isinstance(retry_result, _ComposerResponseResult):
        diagnostics = _context_limit_retry_diagnostics(
            package=package,
            compact_package=compact_package,
            retry_attempted=True,
            retry_success=True,
            after_compact=False,
        )
        return replace(
            retry_result,
            online_diagnostics={
                **retry_result.online_diagnostics,
                **diagnostics,
            },
        )
    if _composer_result_is_provider_context_limit(retry_result):
        return _compact_context_limit_result(
            retry_result,
            package=package,
            compact_package=compact_package,
            retry_attempted=True,
        )
    return _result_with_context_retry_diagnostics(
        retry_result,
        _context_limit_retry_diagnostics(
            package=package,
            compact_package=compact_package,
            retry_attempted=True,
            retry_success=False,
            after_compact=False,
        ),
    )


def _composer_result_is_provider_context_limit(result: Any) -> bool:
    return (
        isinstance(result, Mapping)
        and result.get("provider_error_type") == PROVIDER_CONTEXT_LIMIT_ERROR_TYPE
    )


def _compact_context_limit_result(
    result: dict[str, Any],
    *,
    package: Mapping[str, Any],
    compact_package: Mapping[str, Any],
    retry_attempted: bool,
) -> dict[str, Any]:
    diagnostics = _context_limit_retry_diagnostics(
        package=package,
        compact_package=compact_package,
        retry_attempted=retry_attempted,
        retry_success=False,
        after_compact=True,
    )
    outcome = result.get("outcome")
    if isinstance(outcome, LlmConfiguratorOutcome):
        parse_diagnostics = {
            **_safe_mapping(outcome.parse_diagnostics),
            **diagnostics,
        }
        outcome = replace(
            outcome,
            fallback_reason=COMPACT_FULL_MATRIX_CONTEXT_LIMIT_FALLBACK_REASON,
            parse_diagnostics=parse_diagnostics,
            internal_warnings=_unique(
                [
                    *outcome.internal_warnings,
                    COMPACT_FULL_MATRIX_CONTEXT_LIMIT_FALLBACK_REASON,
                ]
            ),
            final_status_source="provider_context_limit",
        )
    return {
        **result,
        **diagnostics,
        "outcome": outcome,
        "parse_status": "provider_context_limit",
    }


def _result_with_context_retry_diagnostics(
    result: dict[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    outcome = result.get("outcome")
    if isinstance(outcome, LlmConfiguratorOutcome):
        outcome = replace(
            outcome,
            parse_diagnostics={
                **_safe_mapping(outcome.parse_diagnostics),
                **diagnostics,
            },
            internal_warnings=_unique(
                [
                    *outcome.internal_warnings,
                    "provider_context_limit_retry_compact_failed",
                ]
            ),
        )
    return {**result, **dict(diagnostics), "outcome": outcome}


def _context_limit_retry_diagnostics(
    *,
    package: Mapping[str, Any],
    compact_package: Mapping[str, Any],
    retry_attempted: bool,
    retry_success: bool,
    after_compact: bool,
) -> dict[str, Any]:
    original_chars = _int_value(package.get("selected_context_chars")) or json_size_chars(
        package
    )
    compact_chars = _int_value(
        compact_package.get("selected_context_chars")
        or compact_package.get("compact_context_chars")
    ) or json_size_chars(compact_package)
    selected_chars = compact_chars if (retry_success or after_compact) else original_chars
    selected_mode = (
        COMPACT_FULL_MATRIX_MODE
        if retry_success or after_compact
        else _composer_package_mode(package)
    )
    return {
        "provider_context_limit_retry_compact_attempted": bool(retry_attempted),
        "provider_context_limit_retry_compact_success": bool(retry_success),
        "provider_context_limit_original_chars": original_chars,
        "provider_context_limit_compact_chars": compact_chars,
        "provider_context_limit_after_compact": bool(after_compact),
        "v2_package_mode": selected_mode,
        "selected_package_mode": selected_mode,
        "compact_context_chars": compact_chars,
        "selected_context_chars": selected_chars,
        "compact_context_size": {
            "chars": compact_chars,
            "tokens_estimate": max(1, compact_chars // 4),
        },
        "selected_context_size": {
            "chars": selected_chars,
            "tokens_estimate": max(1, selected_chars // 4),
        },
    }


def _run_online_empty_response_repair_if_needed(
    *,
    package: Mapping[str, Any],
    client: LlmClient,
    response_result: _ComposerResponseResult,
) -> _EmptyResponseRepairResult:
    response = response_result.response
    if (
        response.recommendations
        or response_result.proposal_count != 0
        or response_result.schema_rejections
        or _has_structured_no_recommendation(response)
    ):
        return _EmptyResponseRepairResult()

    repair_package = _empty_response_repair_package(package)
    try:
        payload = client.generate_json(
            LLM_ONLINE_COMPOSER_EMPTY_RESPONSE_REPAIR_SYSTEM_PROMPT,
            build_llm_configurator_user_prompt(repair_package),
        )
    except LlmInvalidJsonError as exc:
        return _EmptyResponseRepairResult(
            attempted=True,
            error_type=type(exc).__name__,
            parse_status="parse_error",
            warnings=[f"online_empty_response_repair: {type(exc).__name__}"],
        )
    except LlmError as exc:
        return _EmptyResponseRepairResult(
            attempted=True,
            error_type=type(exc).__name__,
            parse_status="request_failed",
            warnings=[f"online_empty_response_repair: {type(exc).__name__}"],
        )

    normalized_payload = _normalize_composer_payload(payload, package=package)
    try:
        parsed_payload = _parse_composer_payload(normalized_payload)
    except ValidationError:
        return _EmptyResponseRepairResult(
            attempted=True,
            error_type="ValidationError",
            parse_status="validation_error",
            proposal_count=_raw_proposals_count(normalized_payload),
            warnings=["online_empty_response_repair: ValidationError"],
        )

    success = bool(parsed_payload.response.recommendations) or _has_structured_no_recommendation(
        parsed_payload.response
    )
    return _EmptyResponseRepairResult(
        attempted=True,
        success=success,
        response=parsed_payload.response,
        schema_rejections=parsed_payload.schema_rejections,
        proposal_indexes=parsed_payload.proposal_indexes,
        proposal_count=parsed_payload.proposal_count,
        parse_status="parsed",
        warnings=(
            []
            if success
            else ["online_empty_response_repair: empty response after repair"]
        ),
    )


def _empty_response_repair_package(package: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(package),
        "empty_response_repair_attempt": 1,
        "repair_instructions": [
            (
                "You returned no proposal and no structured no_recommendation. "
                "Using the same candidate matrix, either produce one safe BOM or "
                "produce structured no_recommendation with exact missing/mismatched "
                "requirements."
            ),
            "Do not invent component_candidate_id values.",
            "Keep prices, stock, quantities, and totals from application code only.",
            "Do not rerun or alter semantic planner, category planner, or matrix builder output.",
        ],
    }


def _configurator_system_prompt(output_mode: str) -> str:
    if _single_best_output_mode(output_mode):
        return LLM_CONFIGURATOR_SINGLE_BEST_SYSTEM_PROMPT
    return LLM_CONFIGURATOR_SYSTEM_PROMPT


def _try_composer_response(
    *,
    package: Mapping[str, Any],
    settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings,
    llm_client: LlmClient | None,
    llm_call_budget: LlmCallBudget | None,
    system_prompt: str,
    online_composer: bool,
) -> _ComposerResponseResult | dict[str, Any]:
    owns_client = False
    client = budgeted_llm_client(llm_client, llm_call_budget)
    if client is None:
        try:
            raw_client = (
                _build_online_composer_client(settings, web_evidence_settings)
                if online_composer
                else _build_llm_client(settings)
            )
            client = budgeted_llm_client(raw_client, llm_call_budget)
        except LlmError as exc:
            logger.info("LLM configurator unavailable: %s", type(exc).__name__)
            return {
                "outcome": LlmConfiguratorOutcome(
                    enabled=True,
                    output_mode=_normalize_output_mode(package.get("output_mode")),
                    fallback_reason=_client_unavailable_fallback_reason(exc),
                    error_type=type(exc).__name__,
                    http_status=_llm_http_status(exc),
                ),
                "error_type": type(exc).__name__,
                "parse_status": "not_requested",
                "request_attempted": False,
            }
        owns_client = True

    payload: dict[str, Any]
    try:
        payload = client.generate_json(
            system_prompt,
            build_llm_configurator_user_prompt(package),
        )
    except LlmCallBudgetExceededError as exc:
        _close_owned_client(client, owns_client)
        return {
            "outcome": _llm_call_budget_exceeded_outcome(
                package=package,
                diagnostics=llm_call_budget_diagnostics(llm_call_budget),
                error=exc,
            ),
            "error_type": type(exc).__name__,
            "parse_status": "llm_call_budget_exceeded",
            "request_attempted": False,
        }
    except LlmReadTimeoutError as exc:
        _close_owned_client(client, owns_client)
        logger.info("LLM configurator read timed out without retry: %s", type(exc).__name__)
        return {
            "outcome": LlmConfiguratorOutcome(
                enabled=True,
                output_mode=_normalize_output_mode(package.get("output_mode")),
                fallback_reason="llm_configurator_read_timeout_not_retried",
                error_type=type(exc).__name__,
                http_status=_llm_http_status(exc),
            ),
            "error_type": type(exc).__name__,
            "parse_status": "request_failed",
            "request_attempted": True,
        }
    except LlmInvalidJsonError as exc:
        _close_owned_client(client, owns_client)
        logger.info("LLM configurator returned invalid JSON: %s", type(exc).__name__)
        parse_diagnostics = _invalid_json_diagnostics(exc)
        return {
            "outcome": LlmConfiguratorOutcome(
                enabled=True,
                output_mode=_normalize_output_mode(package.get("output_mode")),
                fallback_reason="llm_configurator_invalid_json",
                error_type=type(exc).__name__,
                http_status=_llm_http_status(exc),
                parse_diagnostics=parse_diagnostics,
            ),
            "error_type": type(exc).__name__,
            "parse_status": "parse_error",
            "parse_diagnostics": parse_diagnostics,
            "request_attempted": True,
        }
    except LlmError as exc:
        _close_owned_client(client, owns_client)
        if _llm_error_is_context_limit(exc):
            logger.info(
                "LLM configurator context limit reached, falling back to full matrix: %s",
                type(exc).__name__,
            )
            parse_diagnostics = _provider_context_limit_diagnostics(exc)
            return {
                "outcome": LlmConfiguratorOutcome(
                    enabled=True,
                    output_mode=_normalize_output_mode(package.get("output_mode")),
                    fallback_reason=PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON,
                    error_type=type(exc).__name__,
                    http_status=_llm_http_status(exc),
                    parse_diagnostics=parse_diagnostics,
                    internal_warnings=[PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON],
                ),
                "error_type": type(exc).__name__,
                "provider_error_type": PROVIDER_CONTEXT_LIMIT_ERROR_TYPE,
                "provider_context_limit": parse_diagnostics,
                "parse_status": "provider_context_limit",
                "parse_diagnostics": parse_diagnostics,
                "request_attempted": True,
            }
        logger.info("LLM configurator request failed: %s", type(exc).__name__)
        return {
            "outcome": LlmConfiguratorOutcome(
                enabled=True,
                output_mode=_normalize_output_mode(package.get("output_mode")),
                fallback_reason="llm_configurator_request_failed",
                error_type=type(exc).__name__,
                http_status=_llm_http_status(exc),
            ),
            "error_type": type(exc).__name__,
            "parse_status": "request_failed",
            "request_attempted": True,
        }

    normalized_payload = _normalize_composer_payload(payload, package=package)
    try:
        parsed_payload = _parse_composer_payload(normalized_payload)
    except ValidationError as exc:
        _close_owned_client(client, owns_client)
        return {
            "outcome": _validation_failed_outcome(normalized_payload, exc),
            "error_type": "ValidationError",
            "parse_status": "validation_error",
            "request_attempted": True,
        }
    if (
        not parsed_payload.response.recommendations
        and parsed_payload.schema_rejections
    ):
        _close_owned_client(client, owns_client)
        return {
            "outcome": _validation_failed_outcome_from_schema_rejections(
                parsed_payload.schema_rejections,
                proposal_count=parsed_payload.proposal_count,
            ),
            "error_type": "ValidationError",
            "parse_status": "validation_error",
            "request_attempted": True,
        }

    return _ComposerResponseResult(
        response=parsed_payload.response,
        client=client,
        owns_client=owns_client,
        online_composer_used=True,
        online_diagnostics={},
        schema_rejections=parsed_payload.schema_rejections,
        proposal_indexes=parsed_payload.proposal_indexes,
        proposal_count=parsed_payload.proposal_count,
    )


def _normalize_composer_payload(
    payload: Any,
    *,
    package: Mapping[str, Any] | None = None,
) -> Any:
    if isinstance(payload, list):
        return {
            "recommendations": [
                _normalize_proposal_payload(item, package=package) for item in payload
            ]
        }
    if not isinstance(payload, Mapping):
        return payload

    payload_dict = dict(payload)
    payload_dict = _normalize_root_composer_aliases(payload_dict)
    semantic_output = _composer_semantic_output_fields(payload_dict)
    primary = payload_dict.get("primary_recommendation")
    if isinstance(primary, Mapping):
        root_aliases = _composer_root_aliases_from_proposals([primary])
        return {
            **semantic_output,
            **root_aliases,
            "recommendations": [
                _normalize_primary_recommendation_payload(primary, package=package)
            ],
            "no_recommendation": {},
            "general_notes": _unique(
                [
                    *_string_list(payload_dict.get("general_notes")),
                    *_string_list(root_aliases.get("general_notes")),
                ]
            ),
        }
    no_recommendation = payload_dict.get("no_recommendation")
    if no_recommendation:
        notes = _string_list(payload_dict.get("general_notes"))
        if isinstance(no_recommendation, Mapping):
            notes.extend(_string_list(no_recommendation.get("reasons")))
        return {
            **semantic_output,
            "recommendations": [],
            "no_recommendation": _normalize_no_recommendation_payload(no_recommendation),
            "general_notes": _unique(notes),
        }
    proposal_list = _proposal_list_from_payload(payload_dict)
    if proposal_list is not None:
        root_aliases = _composer_root_aliases_from_proposals(proposal_list)
        _merge_root_composer_aliases(payload_dict, root_aliases)
        payload_dict["recommendations"] = [
            _normalize_proposal_payload(item, package=package) for item in proposal_list
        ]
        payload_dict["no_recommendation"] = _normalize_no_recommendation_payload(
            payload_dict.get("no_recommendation")
        )
        for alias in (
            "proposals",
            "proposal_pool",
            "ai_recommendations",
            "llm_recommendations",
        ):
            payload_dict.pop(alias, None)
        payload_dict.setdefault("general_notes", [])
        return payload_dict

    if _looks_like_single_proposal(payload_dict):
        root_aliases = _composer_root_aliases_from_proposals([payload_dict])
        return {
            **semantic_output,
            **root_aliases,
            "recommendations": [
                _normalize_proposal_payload(payload_dict, package=package)
            ],
            "no_recommendation": {},
            "general_notes": _string_list(root_aliases.get("general_notes")),
        }
    return payload_dict


def _normalize_root_composer_aliases(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    root_tradeoffs = _string_list(result.pop("tradeoffs", None))
    if root_tradeoffs:
        result["assumptions"] = _unique(
            [*_string_list(result.get("assumptions")), *root_tradeoffs]
        )
        result["general_notes"] = _unique(
            [*_string_list(result.get("general_notes")), *root_tradeoffs]
        )
    return result


def _composer_root_aliases_from_proposals(rows: Sequence[Any]) -> dict[str, Any]:
    assumptions: list[str] = []
    general_notes: list[str] = []
    engineer_checks: list[str] = []
    unverified: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        notes = _string_list(row.get("general_notes"))
        tradeoffs = _string_list(row.get("tradeoffs"))
        assumptions.extend(notes)
        assumptions.extend(tradeoffs)
        general_notes.extend(notes)
        unverified_rows = _mapping_rows(row.get("unverified_requirements"))
        unverified.extend(unverified_rows)
        for item in _string_list(row.get("unverified_requirements")):
            engineer_checks.append(item)
    result: dict[str, Any] = {}
    if assumptions:
        result["assumptions"] = _unique(assumptions)
    if general_notes:
        result["general_notes"] = _unique(general_notes)
    if engineer_checks:
        result["engineer_checks"] = _unique(engineer_checks)
    if unverified:
        result["unverified_requirements"] = _unique_mapping_rows(unverified)
    return result


def _merge_root_composer_aliases(
    payload: dict[str, Any],
    aliases: Mapping[str, Any],
) -> None:
    for key in ("assumptions", "engineer_checks", "general_notes"):
        values = _unique([*_string_list(payload.get(key)), *_string_list(aliases.get(key))])
        if values:
            payload[key] = values
    unverified = _unique_mapping_rows(
        [
            *_mapping_rows(payload.get("unverified_requirements")),
            *_mapping_rows(aliases.get("unverified_requirements")),
        ]
    )
    if unverified:
        payload["unverified_requirements"] = unverified


def _composer_semantic_output_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requirement_analysis": _safe_mapping(payload.get("requirement_analysis")),
        "requirement_coverage_summary": _safe_mapping(
            payload.get("requirement_coverage_summary")
            or payload.get("composer_source_coverage_summary")
        ),
        "source_fragments_covered": _mapping_rows(
            payload.get("source_fragments_covered")
        ),
    }
    return {key: value for key, value in result.items() if value not in ({}, [])}


def _normalize_no_recommendation_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = dict(value)
    for key in (
        "missing_roles",
        "manual_checks",
        "diagnostic_notes",
        "general_notes",
        "recommended_next_actions",
        "recommended_repair_actions",
        "engineer_checks",
        "engineering_checks",
    ):
        if key in result:
            result[key] = _string_list(result.get(key))
    for key in (
        "missing_required_capabilities",
        "hard_mismatches",
        "stock_shortages",
        "role_analysis",
        "role_failures",
        "role_level_reasons",
        "partial_available_components",
    ):
        if key in result:
            result[key] = _mapping_rows(result.get(key))
    if "failures_by_role" in result:
        result["role_failures"] = [
            *_mapping_rows(result.get("role_failures")),
            *_role_failure_rows_from_mapping(result.get("failures_by_role")),
        ]
    if "role_level_reasons" in result:
        result["role_failures"] = [
            *_mapping_rows(result.get("role_failures")),
            *_mapping_rows(result.get("role_level_reasons")),
        ]
    if "considered_candidate_ids" in result:
        result["considered_candidate_ids"] = _normalize_considered_candidate_ids(
            result.get("considered_candidate_ids")
        )
    return result


def _role_failure_rows_from_mapping(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for role, reason in value.items():
            role_text = str(role or "").strip()
            if not role_text:
                continue
            if isinstance(reason, Mapping):
                row = dict(reason)
                row.setdefault("role", role_text)
            else:
                row = {"role": role_text, "reason": reason}
            rows.append(row)
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _normalize_considered_candidate_ids(value: Any) -> dict[str, list[str]] | list[str]:
    if isinstance(value, Mapping):
        result: dict[str, list[str]] = {}
        for role, ids in value.items():
            role_text = str(role or "").strip()
            if role_text:
                result[role_text] = _string_list(ids)
        return result
    return _string_list(value)


def _normalize_primary_recommendation_payload(
    payload: Mapping[str, Any],
    *,
    package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proposal = dict(payload)
    candidate_type = str(
        proposal.pop("candidate_type", None)
        or proposal.get("source_type")
        or BUILD_CANDIDATE_TYPE
    ).strip()
    component_ids = _safe_component_id_map(proposal.get("component_candidate_ids"))
    if not component_ids:
        component_ids = _safe_component_id_map(
            proposal.get("selected_component_candidate_ids")
        )
    fulfillment_rows, fulfillment_notes = _normalize_requirement_fulfillment_summary(
        proposal.get("requirement_fulfillment_summary")
    )
    normalized = {
        "recommendation_id": str(
            proposal.get("recommendation_id") or "primary_recommendation"
        ),
        "proposal_role": proposal.get("proposal_role") or "cheapest_fit",
        "recommendation_slot": proposal.get("recommendation_slot") or "price_optimal",
        "source_type": candidate_type,
        "source_candidate_id": proposal.get("source_candidate_id"),
        "component_candidate_ids": component_ids,
        "optional_component_candidate_ids": _safe_component_id_map(
            proposal.get("optional_component_candidate_ids")
        ),
        "engineer_check_component_candidate_ids": _safe_component_id_map(
            proposal.get("engineer_check_component_candidate_ids")
        ),
        "quantities": _clean_quantities(_safe_mapping(proposal.get("quantities"))),
        "decision": proposal.get("decision") or "recommend",
        "title": proposal.get("title") or "Cheapest valid complete stock build",
        "why_selected": (
            proposal.get("why_selected")
            or "Cheapest complete stocked build that satisfies hard requirements."
        ),
        "why_selected_short": proposal.get("why_selected_short"),
        "right_size_note": proposal.get("right_size_note"),
        "commercial_tradeoff": proposal.get("commercial_tradeoff"),
        "requirement_fulfillment_summary": fulfillment_rows,
        "assumptions": _unique(
            [*_string_list(proposal.get("assumptions")), *fulfillment_notes]
        ),
        "what_is_missing": _string_list(
            proposal.get("what_is_missing") or proposal.get("missing_roles")
        ),
        "critical_checks": _string_list(proposal.get("critical_checks")),
        "engineer_checks": _string_list(proposal.get("engineer_checks")),
        "engineering_review_required": True,
        "engineering_confidence": (
            proposal.get("engineering_confidence")
            or "preliminary_requires_engineer_review"
        ),
        "confidence": proposal.get("confidence") or "medium",
    }
    return _normalize_proposal_payload(normalized, package=package)


def _selected_components_alias_maps(
    value: Any,
    *,
    package: Mapping[str, Any] | None,
) -> tuple[dict[str, str], dict[str, int]]:
    if not isinstance(value, list):
        return {}, {}
    rows = [row for row in value if isinstance(row, Mapping)]
    if not rows:
        return {}, {}
    allowed_ids = _component_candidate_ids_from_package(package or {})
    if not allowed_ids:
        return {}, {}

    component_ids: dict[str, str] = {}
    quantities: dict[str, int] = {}
    rows_by_id = _candidate_rows_by_id(package or {})
    for row in rows:
        component_id = _selected_component_alias_id(row)
        if not component_id or component_id not in allowed_ids:
            return {}, {}
        role_key = _selected_component_alias_role(row, component_id, rows_by_id)
        if not role_key:
            return {}, {}
        role_key = _unique_component_role_key(role_key, component_ids)
        component_ids[role_key] = component_id
        quantity = _int_value(
            row.get("quantity")
            or row.get("quantity_required")
            or row.get("count")
            or row.get("qty")
        )
        if quantity is not None and quantity > 0:
            quantities[role_key] = quantity
    return component_ids, quantities


def _selected_component_alias_id(row: Mapping[str, Any]) -> str:
    for key in ("component_candidate_id", "candidate_id", "id"):
        text = str(row.get(key) or "").strip()
        if text:
            return text
    return ""


def _selected_component_alias_role(
    row: Mapping[str, Any],
    component_id: str,
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    role_text = str(
        row.get("role")
        or row.get("component_role")
        or row.get("prompt_role")
        or ""
    ).strip()
    if role_text:
        return role_text
    candidate = rows_by_id.get(component_id)
    if candidate is not None:
        role = str(candidate.get("role") or "").strip()
        if role:
            return role
    return ""


def _unique_component_role_key(role_key: str, existing: Mapping[str, str]) -> str:
    role_key = str(role_key or "").strip()
    if not role_key or role_key not in existing:
        return role_key
    index = 2
    while f"{role_key}_{index}" in existing:
        index += 1
    return f"{role_key}_{index}"


def _normalize_requirement_fulfillment_summary(
    value: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    if not isinstance(value, list):
        return rows, notes
    for index, item in enumerate(value, start=1):
        if isinstance(item, Mapping):
            rows.append(dict(item))
            continue
        text = str(item or "").strip()
        if not text:
            continue
        rows.append(
            {
                "requirement_id": f"llm_summary_{index}",
                "source_text": text,
                "fulfillment_mode": FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
                "closed_by": "llm_string_summary",
                "normalization_note": (
                    "LLM returned requirement_fulfillment_summary as text; "
                    "preserved without making it a hard validation gate."
                ),
            }
        )
        notes.append(f"Requirement summary note: {text}")
    return rows, notes


def _normalize_product_group_source_type(
    proposal: dict[str, Any],
    *,
    package: Mapping[str, Any] | None,
) -> None:
    product_group = str((package or {}).get("product_group") or "").strip()
    if not product_group or product_group == SERVER_PRODUCT_GROUP:
        return
    source_type = str(proposal.get("source_type") or "").strip()
    if source_type != READY_SERVER_CANDIDATE_TYPE:
        return
    source_candidate_id = str(proposal.get("source_candidate_id") or "").strip()
    rows_by_id = _candidate_rows_by_id(package or {})
    if source_candidate_id:
        component_ids = _safe_component_id_map(
            proposal.get("component_candidate_ids"),
            package=package,
        )
        if source_candidate_id in rows_by_id and source_candidate_id not in set(
            component_ids.values()
        ):
            role_key = str(rows_by_id[source_candidate_id].get("role") or "").strip()
            component_ids[
                _unique_component_role_key(
                    role_key or _primary_component_role_for_package(package),
                    component_ids,
                )
            ] = source_candidate_id
        if component_ids:
            proposal["component_candidate_ids"] = component_ids
    proposal["source_type"] = BUILD_CANDIDATE_TYPE
    proposal["source_candidate_id"] = None


def _primary_component_role_for_package(package: Mapping[str, Any] | None) -> str:
    primary = str((package or {}).get("primary_object") or "").strip()
    if primary:
        return primary
    product_group = str((package or {}).get("product_group") or "").strip()
    if product_group == STORAGE_PRODUCT_GROUP:
        return STORAGE_SYSTEM_ROLE
    if product_group == NETWORK_PRODUCT_GROUP:
        return SWITCH_ROLE
    return "component_1"


def _safe_default_source_type(
    proposal: Mapping[str, Any],
    *,
    candidate_type: str,
    package: Mapping[str, Any] | None,
) -> str | None:
    candidate_type = candidate_type.strip()
    product_group = str((package or {}).get("product_group") or "").strip()
    if candidate_type == READY_SERVER_CANDIDATE_TYPE and (
        product_group and product_group != SERVER_PRODUCT_GROUP
    ):
        return BUILD_CANDIDATE_TYPE
    if candidate_type in {
        READY_SERVER_CANDIDATE_TYPE,
        BUILD_CANDIDATE_TYPE,
        PARTIAL_BUILD_CANDIDATE_TYPE,
    }:
        if candidate_type == READY_SERVER_CANDIDATE_TYPE:
            return candidate_type if proposal.get("source_candidate_id") else None
        if _proposal_component_ids_known(proposal, package=package):
            return candidate_type
        return None
    if not _proposal_component_ids_known(proposal, package=package):
        return None
    primary_object = str((package or {}).get("primary_object") or "").strip()
    if product_group or primary_object or proposal.get("component_candidate_ids"):
        return BUILD_CANDIDATE_TYPE
    return None


def _proposal_component_ids_known(
    proposal: Mapping[str, Any],
    *,
    package: Mapping[str, Any] | None,
) -> bool:
    allowed_ids = _component_candidate_ids_from_package(package or {})
    if not allowed_ids:
        return False
    component_ids = _component_ids_from_proposal(proposal)
    return bool(component_ids) and component_ids.issubset(allowed_ids)


def _component_ids_from_proposal(proposal: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in (
        "component_candidate_ids",
        "selected_component_candidate_ids",
        "optional_component_candidate_ids",
        "engineer_check_component_candidate_ids",
    ):
        result.update(_safe_component_id_map(proposal.get(key)).values())
    return {component_id for component_id in result if component_id}


def _parse_composer_payload(payload: Any) -> _ParsedComposerPayload:
    proposal_rows = _raw_proposal_rows(payload)
    try:
        response = LlmComposerResponsePayload.model_validate(payload)
    except ValidationError as root_exc:
        if not proposal_rows:
            raise root_exc
        return _parse_composer_payload_per_item(payload, proposal_rows)
    return _ParsedComposerPayload(
        response=response,
        schema_rejections=[],
        proposal_indexes=list(range(len(response.recommendations))),
        proposal_count=len(proposal_rows),
    )


def _parse_composer_payload_per_item(
    payload: Any,
    proposal_rows: list[Any],
) -> _ParsedComposerPayload:
    recommendations: list[LlmRecommendationPayload] = []
    proposal_indexes: list[int] = []
    schema_rejections: list[_RejectedProposal] = []

    for proposal_index, proposal in enumerate(proposal_rows):
        normalized_proposal = _normalize_proposal_payload(proposal)
        try:
            recommendation = _LLM_RECOMMENDATION_PAYLOAD_ADAPTER.validate_python(
                normalized_proposal
            )
        except ValidationError as exc:
            schema_rejections.append(
                _schema_rejected_proposal(
                    normalized_proposal,
                    proposal_index=proposal_index,
                    validation_errors=_schema_validation_errors(exc),
                )
            )
            continue
        recommendations.append(recommendation)
        proposal_indexes.append(proposal_index)

    return _ParsedComposerPayload(
        response=LlmComposerResponsePayload(
            requirement_analysis=_safe_mapping(
                _safe_mapping(payload).get("requirement_analysis")
            ),
            requirement_coverage_summary=_safe_mapping(
                _safe_mapping(payload).get("requirement_coverage_summary")
                or _safe_mapping(payload).get("composer_source_coverage_summary")
            ),
            source_fragments_covered=_mapping_rows(
                _safe_mapping(payload).get("source_fragments_covered")
            ),
            recommendations=recommendations,
            no_recommendation=_normalize_no_recommendation_payload(
                _safe_mapping(payload).get("no_recommendation")
            ),
            general_notes=_general_notes_from_payload(payload),
        ),
        schema_rejections=schema_rejections,
        proposal_indexes=proposal_indexes,
        proposal_count=len(proposal_rows),
    )


def _general_notes_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, Mapping):
        return _string_list(payload.get("general_notes"))
    return []


def _proposal_list_from_payload(payload: Mapping[str, Any]) -> list[Any] | None:
    for key in (
        "recommendations",
        "proposals",
        "proposal_pool",
        "ai_recommendations",
        "llm_recommendations",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _looks_like_single_proposal(payload: Mapping[str, Any]) -> bool:
    return any(
        key in payload
        for key in (
            "recommendation_id",
            "proposal_role",
            "candidate_type",
            "source_type",
            "selected_components",
            "component_candidate_ids",
            "selected_component_candidate_ids",
        )
    )


def _normalize_proposal_payload(
    payload: Any,
    *,
    package: Mapping[str, Any] | None = None,
) -> Any:
    if not isinstance(payload, Mapping):
        return payload
    proposal = dict(payload)
    candidate_type = str(proposal.pop("candidate_type", "") or "").strip()
    selected_component_ids, selected_quantities = _selected_components_alias_maps(
        proposal.get("selected_components"),
        package=package,
    )
    if selected_component_ids and not proposal.get("component_candidate_ids"):
        proposal["component_candidate_ids"] = selected_component_ids
    if selected_quantities:
        proposal["quantities"] = {
            **_clean_quantities(_safe_mapping(proposal.get("quantities"))),
            **selected_quantities,
        }
    proposal.pop("selected_components", None)
    if "components" in proposal and isinstance(proposal.get("components"), list):
        component_ids, component_quantities = _selected_components_alias_maps(
            proposal.get("components"),
            package=package,
        )
        if component_ids and not proposal.get("component_candidate_ids"):
            proposal["component_candidate_ids"] = component_ids
        if component_quantities:
            proposal["quantities"] = {
                **_clean_quantities(_safe_mapping(proposal.get("quantities"))),
                **component_quantities,
            }
        proposal.pop("components", None)
    if "component_candidate_ids" in proposal:
        proposal["component_candidate_ids"] = _safe_component_id_map(
            proposal.get("component_candidate_ids"),
            package=package,
        )
    if "selected_component_candidate_ids" in proposal:
        proposal["selected_component_candidate_ids"] = _safe_component_id_map(
            proposal.get("selected_component_candidate_ids"),
            package=package,
        )
    if "optional_component_candidate_ids" in proposal:
        proposal["optional_component_candidate_ids"] = _safe_component_id_map(
            proposal.get("optional_component_candidate_ids"),
            package=package,
        )
    if "engineer_check_component_candidate_ids" in proposal:
        proposal["engineer_check_component_candidate_ids"] = _safe_component_id_map(
            proposal.get("engineer_check_component_candidate_ids"),
            package=package,
        )
    fulfillment_rows, fulfillment_notes = _normalize_requirement_fulfillment_summary(
        proposal.get("requirement_fulfillment_summary")
    )
    if "requirement_fulfillment_summary" in proposal or fulfillment_rows:
        proposal["requirement_fulfillment_summary"] = fulfillment_rows
    if fulfillment_notes:
        proposal["assumptions"] = _unique(
            [*_string_list(proposal.get("assumptions")), *fulfillment_notes]
        )
    if "source_type" not in proposal or not str(proposal.get("source_type") or "").strip():
        source_type = _safe_default_source_type(
            proposal,
            candidate_type=candidate_type,
            package=package,
        )
        if source_type:
            proposal["source_type"] = source_type
    _normalize_product_group_source_type(proposal, package=package)
    tradeoffs = _string_list(proposal.pop("tradeoffs", None))
    notes = _string_list(proposal.pop("general_notes", None))
    if tradeoffs and not proposal.get("commercial_tradeoff"):
        proposal["commercial_tradeoff"] = "; ".join(tradeoffs[:3])
    if notes or tradeoffs:
        proposal["assumptions"] = _unique(
            [*_string_list(proposal.get("assumptions")), *notes, *tradeoffs]
        )
    unverified_rows = _mapping_rows(proposal.pop("unverified_requirements", None))
    if unverified_rows:
        proposal["requirement_fulfillment_summary"] = [
            *_normalize_requirement_fulfillment_summary(
                proposal.get("requirement_fulfillment_summary")
            )[0],
            *[
                {
                    **row,
                    "fulfillment_mode": row.get("fulfillment_mode")
                    or FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
                    "closed_by": row.get("closed_by") or "unverified",
                }
                for row in unverified_rows
            ],
        ]
    if "proposal_role" in proposal:
        proposal["proposal_role"] = _normalize_alias_value(
            proposal.get("proposal_role"),
            PROPOSAL_ROLE_ALIASES,
        )
    if "recommendation_slot" in proposal:
        proposal["recommendation_slot"] = _normalize_alias_value(
            proposal.get("recommendation_slot"),
            RECOMMENDATION_SLOT_ALIASES,
        )
    selected_ids = proposal.get("selected_component_candidate_ids")
    if isinstance(selected_ids, Mapping):
        proposal["selected_component_candidate_ids"] = (
            _normalize_selected_component_candidate_ids(selected_ids)
        )
    if (
        not proposal.get("component_candidate_ids")
        and proposal.get("selected_component_candidate_ids")
    ):
        proposal["component_candidate_ids"] = dict(
            proposal["selected_component_candidate_ids"]
        )
    if not proposal.get("recommendation_slot"):
        slot = _proposal_role_to_recommendation_slot(proposal.get("proposal_role"))
        if slot:
            proposal["recommendation_slot"] = slot
    return proposal


def _normalize_alias_value(value: Any, aliases: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return aliases.get(text.casefold(), text)


def _normalize_selected_component_candidate_ids(
    value: Mapping[str, str | None],
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for role, component_id in value.items():
        normalized_role = _normalize_selected_component_role_key(role, component_id)
        if (
            normalized_role in result
            and str(result.get(normalized_role) or "").strip()
        ):
            continue
        result[normalized_role] = component_id
    return result


def _normalize_selected_component_role_key(role: Any, component_id: Any) -> str:
    role_text = str(role or "").strip()
    if role_text != "storage":
        return role_text
    component_role = _component_role_from_id_prefix(component_id)
    return component_role if component_role in {SSD_ROLE, HDD_ROLE} else role_text


def _component_role_from_id_prefix(component_id: Any) -> str | None:
    text = str(component_id or "").strip().casefold()
    if text.startswith("ssd-"):
        return SSD_ROLE
    if text.startswith("hdd-"):
        return HDD_ROLE
    return None


def _proposal_role_to_recommendation_slot(value: Any) -> str | None:
    role = _normalize_alias_value(value, PROPOSAL_ROLE_ALIASES)
    return {
        "cheapest_fit": "price_optimal",
        "technical_clean": "technical_clean",
        "alternative_platform": "alternative",
        "partial_fallback": "partial_fallback",
        "fallback_partial": "partial_fallback",
        "explicit_tradeoff": "alternative",
    }.get(role)


def _invalid_json_diagnostics(exc: LlmInvalidJsonError) -> dict[str, Any]:
    return {
        "llm_parse_stage": _safe_diagnostic_text(exc.parse_stage),
        "llm_json_extract_status": _safe_diagnostic_text(exc.json_extract_status),
        "llm_invalid_json_reason": _safe_diagnostic_text(
            exc.invalid_json_reason,
            limit=200,
        ),
        "llm_invalid_json_preview_sanitized": _safe_diagnostic_text(
            exc.preview_sanitized,
            limit=500,
        ),
    }


def _safe_diagnostic_text(value: Any, *, limit: int = 100) -> str:
    text = str(value or "").replace("\x00", "")
    text = re.sub(
        r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+",
        "[redacted]",
        text,
    )
    text = re.sub(
        r"(?i)\b(api[_-]?key|authorization|token)\b\s*[:=]\s*['\"]?[^'\"\s,}]+",
        "[redacted]",
        text,
    )
    return " ".join(text.split())[:limit]


def _schema_failure_parse_diagnostics(
    *,
    stage: str,
    validation_errors: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stage_text = _safe_diagnostic_text(stage or "main_composer")
    return {
        "parse_status": "validation_error",
        "llm_parse_stage": stage_text,
        "composer_failure_stage": stage_text,
        "composer_failure_error_type": "ValidationError",
        "llm_schema_validation_errors": _mapping_rows(list(validation_errors)),
    }


def _validation_failed_outcome(
    payload: Any,
    exc: ValidationError | None = None,
    *,
    stage: str = "main_composer",
) -> LlmConfiguratorOutcome:
    proposal_count = _raw_proposals_count(payload)
    proposal_rows = _raw_proposal_rows(payload)
    validation_errors = _schema_validation_errors(exc)
    rejected: list[_RejectedProposal] = []
    if proposal_rows:
        for index, proposal in enumerate(proposal_rows):
            rejected.append(
                _RejectedProposal(
                    recommendation_id=_raw_recommendation_id(proposal, index),
                    category="rejected_invalid_schema",
                    message=REJECTION_REASON_MESSAGES["rejected_invalid_schema"],
                    proposal_index=index,
                    debug_safe=_schema_rejected_debug_safe(
                        proposal,
                        proposal_index=index,
                        validation_errors=validation_errors,
                    ),
                )
            )
    else:
        rejected = [
            _RejectedProposal(
                recommendation_id="response",
                category="rejected_invalid_schema",
                message=REJECTION_REASON_MESSAGES["rejected_invalid_schema"],
                proposal_index=index,
                debug_safe={
                    "proposal_index": index,
                    "rejection_category": "invalid_schema",
                    "rejection_code": "invalid_schema",
                    "rejection_message_ru": REJECTION_REASON_MESSAGES[
                        "rejected_invalid_schema"
                    ],
                    "validation_errors": validation_errors,
                    "validation_warnings": [],
                    "stage": "parse",
                },
            )
            for index in range(proposal_count)
        ]
    return LlmConfiguratorOutcome(
        enabled=True,
        output_mode=_normalize_output_mode(
            payload.get("output_mode") if isinstance(payload, Mapping) else None
        ),
        fallback_reason="llm_configurator_validation_failed",
        error_type="ValidationError",
        parse_diagnostics=_schema_failure_parse_diagnostics(
            stage=stage,
            validation_errors=validation_errors,
        ),
        proposal_count=proposal_count,
        valid_proposals_count=0,
        validation_rejected_count=proposal_count,
        selection_skipped_count=0,
        rejected_recommendations_count=proposal_count,
        validation_summary=_validation_summary(
            accepted=0,
            accepted_after_validation=0,
            rejected=rejected,
        ),
        rejected_reasons_top=_rejected_reasons_top(rejected),
        rejected_recommendations_debug_safe=_rejected_debug_safe(rejected),
        final_status_source=COMPOSER_SCHEMA_VALIDATION_FAILED,
    )


def _schema_rejected_proposal(
    proposal: Any,
    *,
    proposal_index: int,
    validation_errors: list[dict[str, Any]],
) -> _RejectedProposal:
    return _RejectedProposal(
        recommendation_id=_raw_recommendation_id(proposal, proposal_index),
        category="rejected_invalid_schema",
        message=REJECTION_REASON_MESSAGES["rejected_invalid_schema"],
        proposal_index=proposal_index,
        debug_safe=_schema_rejected_debug_safe(
            proposal,
            proposal_index=proposal_index,
            validation_errors=validation_errors,
        ),
    )


def _validation_failed_outcome_from_schema_rejections(
    rejected: list[_RejectedProposal],
    *,
    proposal_count: int,
    stage: str = "main_composer",
) -> LlmConfiguratorOutcome:
    validation_errors = [
        error
        for row in rejected
        for error in _mapping_rows(row.debug_safe.get("validation_errors"))
    ]
    return LlmConfiguratorOutcome(
        enabled=True,
        output_mode=OUTPUT_MODE_SINGLE_BEST_COST_VALID,
        fallback_reason="llm_configurator_validation_failed",
        error_type="ValidationError",
        parse_diagnostics=_schema_failure_parse_diagnostics(
            stage=stage,
            validation_errors=validation_errors,
        ),
        proposal_count=proposal_count,
        valid_proposals_count=0,
        validation_rejected_count=len(rejected),
        selection_skipped_count=0,
        rejected_recommendations_count=len(rejected),
        validation_summary=_validation_summary(
            accepted=0,
            accepted_after_validation=0,
            rejected=rejected,
        ),
        rejected_reasons_top=_rejected_reasons_top(rejected),
        rejected_recommendations_debug_safe=_rejected_debug_safe(rejected),
        final_status_source=COMPOSER_SCHEMA_VALIDATION_FAILED,
    )


def _enabled_state(settings: LlmSettings) -> tuple[bool, str | None]:
    if not settings.llm_configurator_enabled:
        return False, "llm_configurator_disabled"
    mode = settings.llm_configurator_mode.strip().lower()
    if mode not in SUPPORTED_MODES:
        return False, "llm_configurator_mode_disabled"
    return True, None


def _build_llm_client(settings: LlmSettings) -> OpenAICompatibleLlmClient:
    provider = settings.llm_provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise LlmConfigurationError(
            "LLM configurator requires LLM_PROVIDER=openai-compatible."
        )
    return OpenAICompatibleLlmClient(
        settings=settings,
        timeout_seconds=settings.llm_configurator_timeout_seconds,
        read_timeout_seconds=settings.llm_configurator_read_timeout_seconds,
        max_output_tokens=settings.llm_configurator_max_output_tokens,
        use_response_format=False,
        thinking_enabled=settings.llm_configurator_thinking_enabled,
        thinking_budget_tokens=settings.llm_configurator_thinking_budget_tokens,
    )


def _build_repair_llm_client(settings: LlmSettings) -> OpenAICompatibleLlmClient:
    provider = settings.llm_provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise LlmConfigurationError(
            "LLM configurator repair requires LLM_PROVIDER=openai-compatible."
        )
    timeout_seconds = settings.llm_configurator_repair_timeout_seconds
    return OpenAICompatibleLlmClient(
        settings=settings,
        timeout_seconds=timeout_seconds,
        read_timeout_seconds=timeout_seconds,
        max_output_tokens=settings.llm_configurator_max_output_tokens,
        use_response_format=False,
        thinking_enabled=settings.llm_configurator_thinking_enabled,
        thinking_budget_tokens=settings.llm_configurator_thinking_budget_tokens,
    )


def _build_online_composer_client(
    settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings,
) -> OpenAICompatibleLlmClient:
    online_settings = _online_composer_llm_settings(settings, web_evidence_settings)
    provider = online_settings.llm_provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise LlmConfigurationError(
            "Online composer requires LLM_PROVIDER=openai-compatible."
        )
    timeout_seconds = max(
        float(settings.llm_configurator_timeout_seconds or 0),
        float(web_evidence_settings.web_evidence_timeout_seconds or 0),
    )
    read_timeout_seconds = max(
        float(settings.llm_configurator_read_timeout_seconds or 0),
        float(web_evidence_settings.web_evidence_timeout_seconds or 0),
    )
    max_output_tokens = max(
        int(settings.llm_configurator_max_output_tokens or 0),
        int(web_evidence_settings.web_evidence_max_output_tokens or 0),
    )
    return OpenAICompatibleLlmClient(
        settings=online_settings,
        timeout_seconds=timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        max_output_tokens=max_output_tokens,
        use_response_format=False,
        thinking_enabled=online_settings.llm_configurator_thinking_enabled,
        thinking_budget_tokens=online_settings.llm_configurator_thinking_budget_tokens,
    )


def _online_composer_llm_settings(
    settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings,
) -> LlmSettings:
    base_url = (
        web_evidence_settings.web_evidence_base_url.strip()
        or settings.llm_base_url.strip()
    )
    api_key = (
        web_evidence_settings.web_evidence_api_key.strip()
        or settings.llm_api_key.strip()
    )
    model = (
        web_evidence_settings.web_evidence_model.strip()
        or settings.llm_model.strip()
    )
    return settings.model_copy(
        update={
            "llm_base_url": base_url,
            "llm_api_key": api_key,
            "llm_model": model,
        }
    )


def _online_composer_requested(settings: WebEvidenceSettings) -> bool:
    return (
        bool(settings.web_evidence_enabled)
        and _evidence_mode(settings) == "online_composer"
        and settings.web_evidence_provider.strip().lower() == "routerai"
    )


def _evidence_mode(settings: WebEvidenceSettings | Mapping[str, Any] | None) -> str:
    raw = ""
    if settings is None:
        raw = ""
    elif isinstance(settings, Mapping):
        raw = str(settings.get("web_evidence_mode") or settings.get("evidence_mode") or "")
    else:
        raw = str(getattr(settings, "web_evidence_mode", "") or "")
    mode = raw.strip().lower()
    return mode if mode in {"separate", "online_composer"} else "separate"


def _llm_http_status(exc: LlmError) -> int | None:
    if isinstance(exc, LlmHttpError):
        return exc.status_code
    return None


def _llm_error_is_context_limit(exc: LlmError) -> bool:
    text = str(exc).casefold()
    status = _llm_http_status(exc)
    context_markers = (
        "context length",
        "context_length",
        "maximum context",
        "max context",
        "prompt too long",
        "input too long",
        "too many tokens",
        "token limit",
        "maximum token",
        "payload too large",
        "request too large",
        "request entity too large",
    )
    if any(marker in text for marker in context_markers):
        return True
    return status in {413, 422} and any(
        marker in text for marker in ("large", "token", "context", "length")
    )


def _provider_context_limit_diagnostics(exc: LlmError) -> dict[str, Any]:
    message = _safe_diagnostic_text(exc, limit=500)
    diagnostics: dict[str, Any] = {
        "provider_error_type": PROVIDER_CONTEXT_LIMIT_ERROR_TYPE,
        "error_type": type(exc).__name__,
        "http_status": _llm_http_status(exc),
        "fallback_reason": PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON,
        "message_safe": message,
    }
    if context_window := _context_window_from_text(message):
        diagnostics["context_limit"] = context_window
    return diagnostics


def _context_window_from_text(text: str) -> int | None:
    normalized = str(text or "")
    patterns = (
        r"context(?:\s+window|\s+length)?\D{0,40}(\d{4,8})",
        r"maximum(?:\s+context|\s+token)?\D{0,40}(\d{4,8})",
        r"limit\D{0,40}(\d{4,8})",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, re.I)
        if not match:
            continue
        value = _int_value(match.group(1))
        if value is not None:
            return value
    return None


def _client_unavailable_fallback_reason(exc: LlmError) -> str:
    if isinstance(exc, LlmConfigurationError):
        return "llm_provider_not_configured"
    return "llm_configurator_client_unavailable"


def _close_owned_client(client: LlmClient | None, owns_client: bool) -> None:
    if owns_client and client is not None and hasattr(client, "close"):
        client.close()  # type: ignore[union-attr]


def _online_composer_evidence_pack(
    *,
    recommendations: list[LlmRecommendationPayload],
    settings: WebEvidenceSettings,
    diagnostics: Mapping[str, Any],
    online_composer_used: bool,
) -> dict[str, Any]:
    summaries = [
        summary
        for recommendation in recommendations
        if (summary := _coerce_llm_evidence_summary(recommendation.evidence_summary))
    ]
    source_count = sum(_int_value(summary.get("sources_count")) or 0 for summary in summaries)
    status_summary: dict[str, int] = {}
    for summary in summaries:
        status = str(summary.get("status") or "unknown").strip() or "unknown"
        status_summary[status] = status_summary.get(status, 0) + 1
    if not status_summary and recommendations:
        status_summary["unknown"] = len(recommendations)

    has_error = bool(str(diagnostics.get("online_composer_error_type") or "").strip())
    evidence_summary = (
        "online composer request failed; normal Composer fallback used"
        if has_error
        else f"online composer evidence sources: {source_count}"
        if online_composer_used
        else "online composer was not requested"
    )
    evidence_mode = _evidence_mode(settings)
    pack = {
        "enabled": bool(settings.web_evidence_enabled),
        "provider": settings.web_evidence_provider,
        "total_tasks": len(recommendations),
        "completed_tasks": len(summaries),
        "error_count": 1 if has_error else 0,
        "components": [],
        "evidence_summary": evidence_summary,
        "search_tasks": [],
        "diagnostics": {
            **diagnostics,
            "evidence_mode": evidence_mode,
            "online_composer_used": bool(online_composer_used),
            "evidence_used": source_count > 0,
            "evidence_sources_count": source_count,
            "evidence_status_summary": status_summary,
            "evidence_requests_count": 1 if online_composer_used else 0,
        },
    }
    pack["diagnostics"] = safe_evidence_diagnostics(
        pack,
        model=settings.web_evidence_model,
        raw_response_parse_status=str(
            pack["diagnostics"].get("online_composer_parse_status") or "not_applicable"
        ),
    )
    pack["diagnostics"] = {
        **pack["diagnostics"],
        **diagnostics,
        "evidence_mode": evidence_mode,
        "online_composer_used": bool(online_composer_used),
        "evidence_used": source_count > 0,
        "evidence_sources_count": source_count,
        "evidence_status_summary": status_summary,
        "online_composer_parse_status": str(
            diagnostics.get("online_composer_parse_status") or "not_applicable"
        ),
        "online_composer_error_type": str(
            diagnostics.get("online_composer_error_type") or ""
        ),
        "evidence_requests_count": 1 if online_composer_used else 0,
    }
    return pack


def _mark_composer_attempt_diagnostics(
    evidence_pack: Mapping[str, Any],
    *,
    settings: WebEvidenceSettings,
    diagnostics: Mapping[str, Any],
    online_composer_used: bool,
) -> dict[str, Any]:
    pack = dict(evidence_pack)
    existing = _safe_mapping(pack.get("diagnostics"))
    request_count = max(
        _int_value(existing.get("evidence_requests_count")) or 0,
        _int_value(diagnostics.get("evidence_requests_count")) or 0,
        1 if online_composer_used else 0,
    )
    pack["diagnostics"] = {
        **existing,
        **_safe_mapping(diagnostics),
        "evidence_mode": existing.get("evidence_mode") or _evidence_mode(settings),
        "online_composer_used": bool(online_composer_used),
        "online_composer_error_type": str(
            diagnostics.get("online_composer_error_type")
            or existing.get("online_composer_error_type")
            or ""
        ),
        "online_composer_parse_status": str(
            diagnostics.get("online_composer_parse_status")
            or existing.get("online_composer_parse_status")
            or ("parsed" if online_composer_used else "")
        ),
        "evidence_requests_count": request_count,
    }
    return pack


def _online_composer_diagnostics(
    *,
    settings: WebEvidenceSettings,
    online_composer_used: bool,
    parse_status: str,
    error_type: str = "",
) -> dict[str, Any]:
    return {
        "evidence_mode": _evidence_mode(settings),
        "online_composer_used": bool(online_composer_used),
        "evidence_used": bool(online_composer_used),
        "evidence_sources_count": 0,
        "evidence_status_summary": {},
        "online_composer_error_type": str(error_type or ""),
        "online_composer_parse_status": str(parse_status or "not_applicable"),
        "online_composer_empty_response_repair_attempted": False,
        "online_composer_empty_response_repair_success": False,
        "structured_no_recommendation_used": False,
        "evidence_requests_count": 1 if online_composer_used else 0,
    }


def _with_online_evidence_diagnostics(
    outcome: LlmConfiguratorOutcome,
    *,
    settings: WebEvidenceSettings,
    diagnostics: Mapping[str, Any],
) -> LlmConfiguratorOutcome:
    evidence_pack = _online_composer_evidence_pack(
        recommendations=[],
        settings=settings,
        diagnostics=diagnostics,
        online_composer_used=bool(diagnostics.get("online_composer_used")),
    )
    return replace(outcome, evidence_pack=evidence_pack)


def _evidence_component_rows(
    component_index: Mapping[str, _IndexedComponentCandidate],
) -> dict[str, dict[str, Any]]:
    return {
        component_id: {
            **candidate.row,
            "role": candidate.internal_role,
        }
        for component_id, candidate in component_index.items()
    }


def _evidence_stock_rows(
    stock_candidate_index: Mapping[str, _IndexedStockCandidate],
) -> dict[str, dict[str, Any]]:
    return {
        candidate_id: {
            **dict(candidate.row),
            "role": (
                READY_SERVER_CANDIDATE_TYPE
                if candidate.candidate_type == READY_SERVER_CANDIDATE_TYPE
                else candidate.candidate_type
            ),
            "component_candidate_id": candidate_id,
            "candidate_id": candidate_id,
        }
        for candidate_id, candidate in stock_candidate_index.items()
    }


def _collect_evidence_for_response(
    *,
    recommendations: list[LlmRecommendationPayload],
    component_index: Mapping[str, _IndexedComponentCandidate],
    stock_candidate_index: Mapping[str, _IndexedStockCandidate],
    settings: WebEvidenceSettings,
    llm_settings: LlmSettings,
    normalized_requirements: Mapping[str, Any],
    provider: WebSearchProvider | None,
    cache: EvidenceSearchCache | None,
) -> dict[str, Any]:
    component_rows = _evidence_component_rows(component_index)
    stock_rows = _evidence_stock_rows(stock_candidate_index)
    tasks = build_evidence_tasks_for_proposals(
        recommendations,
        component_rows_by_id=component_rows,
        stock_rows_by_id=stock_rows,
        max_queries=settings.web_evidence_max_queries,
        normalized_requirements=normalized_requirements,
    )
    try:
        evidence_pack = collect_web_evidence(
            tasks=tasks,
            settings=settings,
            provider=provider,
            cache=cache,
            normalized_requirements=normalized_requirements,
            llm_settings=llm_settings,
        )
    except Exception as exc:
        logger.info("Web evidence layer unavailable: %s", type(exc).__name__)
        evidence_pack = EvidencePack(
            enabled=settings.web_evidence_enabled,
            provider=settings.web_evidence_provider,
            total_tasks=len(tasks),
            error_count=1,
            evidence_summary="web evidence unavailable; Composer fallback used",
            search_tasks=[task.model_dump() for task in tasks],
            diagnostics=safe_evidence_diagnostics(
                {
                    "provider": settings.web_evidence_provider,
                    "total_tasks": len(tasks),
                    "completed_tasks": 0,
                    "error_count": 1,
                    "components": [],
                },
                model=settings.web_evidence_model,
                error_type=type(exc).__name__,
                raw_response_parse_status="provider_error",
            ),
        )
    pack = evidence_pack.model_dump()
    pack["diagnostics"] = safe_evidence_diagnostics(
        pack,
        model=settings.web_evidence_model,
        raw_response_parse_status=str(
            _safe_mapping(pack.get("diagnostics")).get("evidence_raw_response_parse_status")
            or "not_applicable"
        ),
    )
    return pack


def _collect_relation_evidence_for_selected_recommendations(
    *,
    recommendations: Sequence[Mapping[str, Any]],
    component_index: Mapping[str, _IndexedComponentCandidate],
    settings: WebEvidenceSettings,
    llm_settings: LlmSettings,
    normalized_requirements: Mapping[str, Any],
    provider: WebSearchProvider | None,
    cache: EvidenceSearchCache | None,
) -> dict[str, Any]:
    component_rows = _evidence_component_rows(component_index)
    tasks = build_relation_evidence_tasks_for_recommendations(
        recommendations,
        component_rows_by_id=component_rows,
        normalized_requirements=normalized_requirements,
    )
    try:
        evidence_pack = collect_web_evidence(
            tasks=tasks,
            settings=settings,
            provider=provider,
            cache=cache,
            normalized_requirements=normalized_requirements,
            llm_settings=llm_settings,
        )
    except Exception as exc:
        logger.info("Post-hoc relation evidence unavailable: %s", type(exc).__name__)
        evidence_pack = EvidencePack(
            enabled=settings.web_evidence_enabled,
            provider=settings.web_evidence_provider,
            total_tasks=len(tasks),
            error_count=1 if tasks else 0,
            relation_evidence=[
                {
                    "relation_type": _relation_type_from_task_dump(task.model_dump()),
                    "recommendation_id": task.recommendation_id,
                    "components": {},
                    "platform_name": task.platform_name,
                    "cpu_name": task.cpu_name,
                    "ram_name": task.ram_name,
                    "storage_name": task.storage_name,
                    "status": "error",
                    "confidence": "unknown",
                    "missing_evidence": [
                        "External relation evidence was unavailable for this selected build."
                    ],
                    "engineering_checks": [
                        "Run a final engineering compatibility review."
                    ],
                    "sources": [],
                    "warnings": [],
                }
                for task in tasks
            ],
            evidence_summary="post-hoc relation evidence unavailable",
            search_tasks=[task.model_dump() for task in tasks],
            diagnostics=safe_evidence_diagnostics(
                {
                    "provider": settings.web_evidence_provider,
                    "total_tasks": len(tasks),
                    "completed_tasks": 0,
                    "error_count": 1 if tasks else 0,
                    "components": [],
                    "relation_evidence": [],
                    "search_tasks": [task.model_dump() for task in tasks],
                },
                model=settings.web_evidence_model,
                error_type=type(exc).__name__,
                raw_response_parse_status="provider_error",
            ),
        )
    pack = evidence_pack.model_dump()
    pack["diagnostics"] = safe_evidence_diagnostics(
        pack,
        model=settings.web_evidence_model,
        raw_response_parse_status=str(
            _safe_mapping(pack.get("diagnostics")).get("evidence_raw_response_parse_status")
            or "not_applicable"
        ),
    )
    return pack


def _relation_type_from_task_dump(task: Mapping[str, Any]) -> str:
    role = str(task.get("role") or "").strip()
    if role in {"platform_cpu", "platform_ram", "platform_storage", "build_sanity"}:
        return role
    target_type = str(task.get("target_type") or "").strip()
    return {
        "relation_platform_cpu": "platform_cpu",
        "relation_platform_ram": "platform_ram",
        "relation_platform_storage": "platform_storage",
        "relation_build_sanity": "build_sanity",
    }.get(target_type, "build_sanity")


def _online_composer_needs_posthoc_relation_evidence(
    evidence_pack: Mapping[str, Any],
    selected_recommendations: Sequence[Mapping[str, Any]],
) -> bool:
    if not selected_recommendations:
        return False
    diagnostics = safe_evidence_diagnostics(evidence_pack)
    tasks_by_type = diagnostics.get("evidence_tasks_count_by_type")
    has_empty_tasks = not isinstance(tasks_by_type, Mapping) or not tasks_by_type
    if (_int_value(diagnostics.get("evidence_sources_count")) or 0) <= 0:
        return True
    if (_int_value(diagnostics.get("relation_evidence_count")) or 0) <= 0:
        return True
    if has_empty_tasks:
        return True
    for recommendation in selected_recommendations:
        summary = recommendation.get("evidence_summary")
        if not isinstance(summary, Mapping):
            return True
        status = str(summary.get("status") or summary.get("evidence_status") or "").strip()
        if status in {"", "unknown", "not_confirmed", "not_found", "disabled", "error"}:
            return True
        if (_int_value(summary.get("sources_count")) or 0) <= 0:
            return True
    return False


def _merge_online_composer_relation_evidence_pack(
    online_pack: Mapping[str, Any],
    relation_pack: Mapping[str, Any],
    *,
    settings: WebEvidenceSettings,
) -> dict[str, Any]:
    merged = dict(online_pack)
    existing_diagnostics = _safe_mapping(merged.get("diagnostics"))
    relation_diagnostics = safe_evidence_diagnostics(
        relation_pack,
        model=settings.web_evidence_model,
    )
    existing_sources_count = _int_value(existing_diagnostics.get("evidence_sources_count")) or 0
    relation_sources_count = _int_value(relation_diagnostics.get("evidence_sources_count")) or 0
    existing_requests = _int_value(existing_diagnostics.get("evidence_requests_count")) or 0
    relation_requests = 1 if (_int_value(relation_pack.get("total_tasks")) or 0) > 0 else 0

    components = [
        *_mapping_rows(merged.get("components")),
        *_mapping_rows(relation_pack.get("components")),
    ]
    relations = [
        *_mapping_rows(merged.get("relation_evidence")),
        *_mapping_rows(relation_pack.get("relation_evidence")),
    ]
    search_tasks = [
        *_mapping_rows(merged.get("search_tasks")),
        *_mapping_rows(relation_pack.get("search_tasks")),
    ]
    merged.update(
        {
            "components": components,
            "relation_evidence": relations,
            "search_tasks": search_tasks,
            "total_tasks": len(search_tasks)
            or (
                (_int_value(merged.get("total_tasks")) or 0)
                + (_int_value(relation_pack.get("total_tasks")) or 0)
            ),
            "completed_tasks": (_int_value(merged.get("completed_tasks")) or 0)
            + (_int_value(relation_pack.get("completed_tasks")) or 0),
            "error_count": (_int_value(merged.get("error_count")) or 0)
            + (_int_value(relation_pack.get("error_count")) or 0),
            "evidence_summary": _merged_evidence_summary_text(
                online_pack,
                relation_pack,
            ),
        }
    )
    merged["diagnostics"] = safe_evidence_diagnostics(
        merged,
        model=settings.web_evidence_model,
        raw_response_parse_status=str(
            existing_diagnostics.get("evidence_raw_response_parse_status")
            or relation_diagnostics.get("evidence_raw_response_parse_status")
            or "not_applicable"
        ),
    )
    merged["diagnostics"] = {
        **merged["diagnostics"],
        "evidence_mode": "online_composer",
        "online_composer_used": bool(existing_diagnostics.get("online_composer_used")),
        "evidence_used": bool(
            existing_diagnostics.get("evidence_used") or relation_sources_count > 0
        ),
        "evidence_sources_count": existing_sources_count + relation_sources_count,
        "evidence_requests_count": existing_requests + relation_requests,
        "online_composer_error_type": str(
            existing_diagnostics.get("online_composer_error_type") or ""
        ),
        "online_composer_parse_status": str(
            existing_diagnostics.get("online_composer_parse_status") or ""
        ),
        "posthoc_relation_evidence_used": relation_requests > 0,
    }
    return merged


def _merged_evidence_summary_text(
    online_pack: Mapping[str, Any],
    relation_pack: Mapping[str, Any],
) -> str:
    relation_sources = _int_value(
        safe_evidence_diagnostics(relation_pack).get("evidence_sources_count")
    ) or 0
    if relation_sources > 0:
        return f"post-hoc relation evidence sources: {relation_sources}"
    return str(online_pack.get("evidence_summary") or "post-hoc relation evidence checked")


def _selected_recommendation_ids(
    recommendations: Sequence[Mapping[str, Any]],
) -> set[str]:
    return {
        recommendation_id
        for recommendation in recommendations
        if (
            recommendation_id := str(
                recommendation.get("recommendation_id")
                or recommendation.get("candidate_id")
                or ""
            ).strip()
        )
    }


def _run_evidence_review(
    *,
    client: LlmClient,
    recommendations: list[LlmRecommendationPayload],
    normalized_requirements: Mapping[str, Any],
    component_matrix: Mapping[str, Any],
    evidence_pack: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not evidence_pack_has_found_sources(evidence_pack):
        return {}, []
    package = {
        "normalized_requirements": normalized_requirements,
        "recommendations": [recommendation.model_dump() for recommendation in recommendations],
        "component_candidate_matrix": component_matrix,
        "evidence_pack": evidence_pack,
    }
    try:
        payload = client.generate_json(
            LLM_EVIDENCE_REVIEW_SYSTEM_PROMPT,
            json.dumps(package, ensure_ascii=False, sort_keys=True),
        )
        response = LlmEvidenceReviewResponsePayload.model_validate(payload)
    except (LlmError, ValidationError) as exc:
        logger.info("LLM evidence review unavailable: %s", type(exc).__name__)
        return {}, [f"evidence_review_unavailable: {type(exc).__name__}"]
    return response.model_dump(), []


def _evidence_review_by_recommendation_id(
    evidence_review: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(evidence_review, Mapping):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for item in _mapping_rows(evidence_review.get("evidence_review")):
        recommendation_id = str(item.get("recommendation_id") or "").strip()
        if recommendation_id:
            result[recommendation_id] = item
    return result


def _raw_proposals_count(payload: Any) -> int:
    return len(_raw_proposal_rows(payload))


def _compact_component_candidate_matrix(
    component_candidate_matrix: Mapping[str, Any],
    *,
    candidates_per_role: int,
    include_ready_server: bool = True,
    ready_server_limit: int = READY_SERVER_CANDIDATES_LIMIT,
) -> dict[str, list[dict[str, Any]]]:
    matrix: dict[str, list[dict[str, Any]]] = {}
    for prompt_role, matrix_key, internal_role in PACKAGE_MATRIX_KEYS:
        if internal_role == READY_SERVER_CANDIDATE_TYPE and not include_ready_server:
            continue
        rows = _selectable_component_rows(
            _stable_component_rows(_mapping_rows(component_candidate_matrix.get(matrix_key)))
        )
        role_limit = len(rows)
        if internal_role == READY_SERVER_CANDIDATE_TYPE:
            role_limit = max(0, min(candidates_per_role, ready_server_limit))
        compact_rows = [
            _compact_component_candidate(row, prompt_role, internal_role)
            for row in rows[:role_limit]
        ]
        if compact_rows:
            matrix[prompt_role] = compact_rows
    return matrix


def _role_lifecycle_package_fields(
    component_candidate_matrix: Mapping[str, Any],
) -> dict[str, Any]:
    role_plan = _safe_mapping(component_candidate_matrix.get("role_plan"))
    fields = {
        "stage_a_broad_roles": _string_list(
            component_candidate_matrix.get("stage_a_broad_roles")
            or role_plan.get("stage_a_broad_roles")
        ),
        "semantic_matrix_blueprint_roles": _string_list(
            component_candidate_matrix.get("semantic_matrix_blueprint_roles")
            or role_plan.get("semantic_matrix_blueprint_roles")
        ),
        "requirement_classifier_roles": _string_list(
            component_candidate_matrix.get("requirement_classifier_roles")
            or role_plan.get("requirement_classifier_roles")
        ),
        "effective_matrix_roles_before_category_planner": _string_list(
            component_candidate_matrix.get(
                "effective_matrix_roles_before_category_planner"
            )
            or role_plan.get("effective_matrix_roles_before_category_planner")
        ),
        "category_planner_input_roles": _string_list(
            component_candidate_matrix.get("category_planner_input_roles")
            or role_plan.get("category_planner_input_roles")
        ),
        "category_planner_output_roles": _string_list(
            component_candidate_matrix.get("category_planner_output_roles")
            or role_plan.get("category_planner_output_roles")
        ),
        "category_planner_missing_required_roles": _string_list(
            component_candidate_matrix.get("category_planner_missing_required_roles")
            or role_plan.get("category_planner_missing_required_roles")
        ),
        "category_planner_repair_attempted": bool(
            component_candidate_matrix.get("category_planner_repair_attempted")
            or role_plan.get("category_planner_repair_attempted")
        ),
        "category_planner_repair_success": bool(
            component_candidate_matrix.get("category_planner_repair_success")
            or role_plan.get("category_planner_repair_success")
        ),
        "category_planner_repair_reason": _text_or_none(
            component_candidate_matrix.get("category_planner_repair_reason")
            or role_plan.get("category_planner_repair_reason")
        ),
        "category_planner_repaired_roles": _string_list(
            component_candidate_matrix.get("category_planner_repaired_roles")
            or role_plan.get("category_planner_repaired_roles")
        ),
        "category_planner_unresolved_required_roles": _string_list(
            component_candidate_matrix.get("category_planner_unresolved_required_roles")
            or role_plan.get("category_planner_unresolved_required_roles")
        ),
        "validated_category_plan_roles": _string_list(
            component_candidate_matrix.get("validated_category_plan_roles")
            or role_plan.get("validated_category_plan_roles")
        ),
        "materialized_matrix_roles": _string_list(
            component_candidate_matrix.get("materialized_matrix_roles")
            or role_plan.get("materialized_matrix_roles")
        ),
        "roles_dropped_after_stage_a": _string_list(
            component_candidate_matrix.get("roles_dropped_after_stage_a")
            or role_plan.get("roles_dropped_after_stage_a")
        ),
        "roles_dropped_before_category_planner": _string_list(
            component_candidate_matrix.get("roles_dropped_before_category_planner")
            or role_plan.get("roles_dropped_before_category_planner")
        ),
        "roles_dropped_after_category_planner": _string_list(
            component_candidate_matrix.get("roles_dropped_after_category_planner")
            or role_plan.get("roles_dropped_after_category_planner")
        ),
        "roles_dropped_during_materialization": _string_list(
            component_candidate_matrix.get("roles_dropped_during_materialization")
            or role_plan.get("roles_dropped_during_materialization")
        ),
        "roles_dropped_reason_by_role": _safe_mapping(
            component_candidate_matrix.get("roles_dropped_reason_by_role")
            or role_plan.get("roles_dropped_reason_by_role")
        ),
        "role_source_by_role": _safe_mapping(
            component_candidate_matrix.get("role_source_by_role")
            or role_plan.get("role_source_by_role")
        ),
        "role_lifecycle_trace": _mapping_rows(
            component_candidate_matrix.get("role_lifecycle_trace")
            or role_plan.get("role_lifecycle_trace")
        ),
        "roles_sent_to_composer": _string_list(
            component_candidate_matrix.get("roles_sent_to_composer")
            or role_plan.get("roles_sent_to_composer")
        ),
        "broad_reasoning_roles": _string_list(
            component_candidate_matrix.get("broad_reasoning_roles")
            or role_plan.get("broad_reasoning_roles")
        ),
        "hard_purchasable_bom_roles": _string_list(
            component_candidate_matrix.get("hard_purchasable_bom_roles")
            or role_plan.get("hard_purchasable_bom_roles")
        ),
        "hard_purchasable_bom_role_requirements": _mapping_rows(
            component_candidate_matrix.get("hard_purchasable_bom_role_requirements")
            or role_plan.get("hard_purchasable_bom_role_requirements")
        ),
        "optional_accessory_engineering_roles": _string_list(
            component_candidate_matrix.get("optional_accessory_engineering_roles")
            or role_plan.get("optional_accessory_engineering_roles")
        ),
        "optional_accessory_engineering_requirements": _mapping_rows(
            component_candidate_matrix.get(
                "optional_accessory_engineering_requirements"
            )
            or role_plan.get("optional_accessory_engineering_requirements")
        ),
    }
    return fields


def _attach_package_candidate_exposure_fields(
    package: dict[str, Any],
    *,
    source_component_candidate_matrix: Mapping[str, Any],
    required_roles: Sequence[str],
    include_ready_server: bool,
    package_skipped_reason: str | None = None,
) -> None:
    source_ids_by_role = _source_component_candidate_ids_by_role(
        source_component_candidate_matrix,
        include_ready_server=include_ready_server,
    )
    broad_counts = _source_component_candidate_count_by_role(
        source_component_candidate_matrix,
        source_ids_by_role=source_ids_by_role,
    )
    package_ids_by_role = _package_component_candidate_ids_by_role(
        _safe_mapping(package.get("component_candidate_matrix"))
    )
    package_counts = {
        role: len(ids)
        for role, ids in package_ids_by_role.items()
        if ids or broad_counts.get(role, 0) > 0
    }
    composer_package_roles = sorted(
        role for role, count in package_counts.items() if int(count or 0) > 0
    )
    package["composer_package_roles"] = composer_package_roles
    roles = _unique(
        [
            *broad_counts.keys(),
            *package_counts.keys(),
            *[_coverage_role_key(role) for role in _string_list(required_roles)],
        ]
    )
    full_matrix_complete = _component_matrix_full_evaluation_complete(
        source_component_candidate_matrix,
        matrix_counts=broad_counts,
    )
    full_matrix_used = bool(source_component_candidate_matrix.get("full_matrix_evaluation_used"))
    dropped_counts: dict[str, int] = {}
    dropped_reasons: dict[str, str] = {}
    ratios: dict[str, float] = {}
    for role in roles:
        broad_count = int(broad_counts.get(role, 0) or 0)
        package_count = int(package_counts.get(role, 0) or 0)
        if broad_count <= 0 and package_count <= 0:
            continue
        dropped = max(0, broad_count - package_count)
        dropped_counts[role] = dropped
        ratios[role] = (
            1.0
            if broad_count <= 0
            else round(min(1.0, package_count / max(1, broad_count)), 4)
        )
        if dropped:
            dropped_reasons[role] = _dropped_before_composer_reason(
                role=role,
                source_ids_by_role=source_ids_by_role,
                package_skipped_reason=package_skipped_reason,
                full_matrix_used=full_matrix_used,
                full_matrix_complete=full_matrix_complete,
            )

    required = _unique(
        [_coverage_role_key(role) for role in _string_list(required_roles)]
    )
    hard_lifecycle_roles = _unique(
        [
            _coverage_role_key(role)
            for role in _string_list(
                package.get("hard_purchasable_bom_roles")
                or _safe_mapping(package.get("role_plan")).get(
                    "hard_purchasable_bom_roles"
                )
            )
            if _coverage_role_key(role)
        ]
    )
    if hard_lifecycle_roles:
        required = hard_lifecycle_roles
    elif not required and not _is_v2_composer_package(package):
        required = [role for role, count in broad_counts.items() if int(count or 0) > 0]
    lifecycle_contract_roles = _unique(
        [
            *_string_list(package.get("stage_a_broad_roles")),
            *_string_list(package.get("effective_matrix_roles_before_category_planner")),
        ]
    )
    lifecycle_required = (
        hard_lifecycle_roles
        if hard_lifecycle_roles
        else (
            _unique(
                [
                    *lifecycle_contract_roles,
                    *_string_list(package.get("category_planner_input_roles")),
                    *_string_list(package.get("validated_category_plan_roles")),
                ]
            )
            if lifecycle_contract_roles
            else []
        )
    )
    blocking_lifecycle_roles: list[str] = []
    lifecycle_absence_reasons: dict[str, str] = {}
    for role in lifecycle_required:
        if _lifecycle_role_candidate_count(role, broad_counts, package_counts) > 0:
            continue
        absence_reason = _candidate_absence_reason_for_role(package, role)
        if _lifecycle_absence_reason_is_terminal_non_blocking(
            package,
            role,
            absence_reason,
        ):
            lifecycle_absence_reasons[role] = absence_reason
            continue
        blocking_lifecycle_roles.append(role)
        lifecycle_absence_reasons[role] = (
            absence_reason or "missing_from_composer_package"
        )
    incomplete_roles = [
        role
        for role in required
        if int(dropped_counts.get(role, 0) or 0) > 0 and not full_matrix_complete
    ]
    incomplete_roles = _unique(
        [*incomplete_roles, *blocking_lifecycle_roles]
    )
    exposure_incomplete = bool(incomplete_roles)
    if package_skipped_reason:
        policy_mode = "package_skipped"
    elif full_matrix_complete and any(dropped_counts.values()):
        policy_mode = "ai_reviewed_role_pools"
    elif exposure_incomplete:
        policy_mode = INCOMPLETE_MATRIX_EXPOSURE_REASON
    elif any(dropped_counts.values()):
        policy_mode = "partial_non_required_candidates"
    else:
        policy_mode = FULL_BROAD_MATRIX_EXPOSURE_MODE

    package["broad_matrix_count_by_role"] = {
        role: int(count)
        for role, count in broad_counts.items()
        if int(count or 0) > 0
    }
    package["composer_package_candidate_count_by_role"] = {
        role: int(package_counts.get(role, 0) or 0)
        for role in roles
        if int(package_counts.get(role, 0) or 0) > 0
        or int(broad_counts.get(role, 0) or 0) > 0
    }
    package["composer_package_candidate_total"] = sum(
        package["composer_package_candidate_count_by_role"].values()
    )
    package["composer_package_candidate_ids_by_role"] = package_ids_by_role
    package["dropped_before_composer_count_by_role"] = {
        role: int(count)
        for role, count in dropped_counts.items()
        if role in roles
    }
    package["dropped_before_composer_reason_by_role"] = dropped_reasons
    package["roles_dropped_reason_by_role"] = role_lifecycle.merge_drop_reasons(
        dropped_reasons,
        lifecycle_absence_reasons,
        existing=_safe_mapping(package.get("roles_dropped_reason_by_role")),
    )
    package["package_candidate_exposure_ratio_by_role"] = {
        role: ratios[role]
        for role in roles
        if role in ratios
    }
    package["package_candidate_exposure_incomplete"] = exposure_incomplete
    package["package_candidate_exposure_incomplete_roles"] = incomplete_roles
    package["package_exposure_blocking_lifecycle_roles"] = blocking_lifecycle_roles
    package["package_candidate_exposure_policy"] = {
        "mode": policy_mode,
        "allow_incomplete": False,
        "required_roles": required,
        "incomplete_roles": incomplete_roles,
        "lifecycle_required_roles": lifecycle_required,
        "blocking_lifecycle_roles": blocking_lifecycle_roles,
        "full_matrix_evaluation_used": full_matrix_used,
        "full_matrix_evaluation_complete": full_matrix_complete,
        "candidate_matrix_trimming_allowed": False,
    }
    package["role_lifecycle_trace"] = role_lifecycle.build_role_lifecycle_trace(
        lifecycle_required,
        role_source_by_role=_safe_mapping(package.get("role_source_by_role")),
        stage_a_roles=_string_list(package.get("stage_a_broad_roles")),
        semantic_matrix_blueprint_roles=_string_list(
            package.get("semantic_matrix_blueprint_roles")
        ),
        requirement_classifier_roles=_string_list(
            package.get("requirement_classifier_roles")
        ),
        before_category_planner_roles=_string_list(
            package.get("effective_matrix_roles_before_category_planner")
        ),
        category_planner_input_roles=_string_list(
            package.get("category_planner_input_roles")
        ),
        category_planner_output_roles=_string_list(
            package.get("category_planner_output_roles")
        ),
        validated_category_plan_roles=_string_list(
            package.get("validated_category_plan_roles")
        ),
        materialized_matrix_roles=_string_list(package.get("materialized_matrix_roles")),
        composer_package_roles=composer_package_roles,
        dropped_reason_by_role=_safe_mapping(
            package.get("roles_dropped_reason_by_role")
        ),
    )


def _lifecycle_role_candidate_count(
    role: str,
    broad_counts: Mapping[str, int],
    package_counts: Mapping[str, int],
) -> int:
    aliases = _lifecycle_role_aliases(role)
    return max(
        [
            int(broad_counts.get(alias, 0) or 0)
            + int(package_counts.get(alias, 0) or 0)
            for alias in aliases
        ]
        or [0]
    )


def _lifecycle_role_aliases(role: str) -> list[str]:
    normalized = _coverage_role_key(str(role or "").strip())
    aliases = [normalized]
    if normalized == "storage":
        aliases.extend(["drive", "ssd", "hdd"])
    elif normalized == "drive":
        aliases.extend(["ssd", "hdd", "storage"])
    elif normalized in {"ssd", "hdd"}:
        aliases.extend(["drive", "storage"])
    if normalized == "server_platform":
        aliases.append("platform")
    return _unique(aliases)


def _candidate_absence_reason_for_role(
    package: Mapping[str, Any],
    role: str,
) -> str | None:
    role_coverage = _safe_mapping(package.get("role_coverage_summary"))
    coverage = _safe_mapping(role_coverage.get(role))
    if not coverage and role == "storage":
        coverage = _safe_mapping(role_coverage.get("ssd") or role_coverage.get("drive"))
    if _ready_server_satisfies_lifecycle_role(package, role):
        return "satisfied_by_ready_server"
    if coverage:
        if (
            coverage.get("can_be_satisfied_by_platform")
            and (_int_value(coverage.get("platform_satisfied_candidates_count")) or 0) > 0
        ):
            return "satisfied_by_platform"
        if coverage.get("missing_category"):
            return "missing_category"
        if coverage.get("missing_candidates"):
            after_category = _int_value(coverage.get("after_category_count")) or 0
            after_eligibility = _int_value(coverage.get("after_eligibility_count")) or 0
            return (
                "missing_candidates:"
                f"after_category_count={after_category}:"
                f"after_eligibility_count={after_eligibility}"
            )
    reason_by_role = _safe_mapping(package.get("roles_dropped_reason_by_role"))
    reason = _text_or_none(reason_by_role.get(role))
    if reason and reason not in {
        "not_emitted_by_requirement_classifier_preserved_by_union",
    }:
        return reason
    return None


def _lifecycle_absence_reason_is_terminal_non_blocking(
    package: Mapping[str, Any],
    role: str,
    reason: str | None,
) -> bool:
    _ = package
    reason_text = str(reason or "").strip()
    if reason_text == "satisfied_by_platform":
        return True
    if reason_text == "satisfied_by_ready_server":
        return True
    if reason_text in {
        "included_in_primary_object",
        "included_in_selected_component",
        "optional_only",
        "engineering_check_only",
        "logistics_constraint",
    }:
        return True
    if reason_text.startswith("fulfilled_by_role:"):
        return True
    if reason_text.startswith("classifier_marked_not_applicable:"):
        return True
    return role == UNMAPPED_ROLE and reason_text in {
        "out_of_scope_or_unmapped_non_blocking",
        "not_applicable",
    }


def _ready_server_satisfies_lifecycle_role(
    package: Mapping[str, Any],
    role: str,
) -> bool:
    if str(package.get("product_group") or "").strip() != SERVER_PRODUCT_GROUP:
        return False
    if role not in {SERVER_PLATFORM_ROLE, CPU_ROLE, RAM_ROLE, "storage", SSD_ROLE, HDD_ROLE}:
        return False
    if _mapping_rows(package.get("ready_stock_candidates")):
        return True
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    return bool(
        _mapping_rows(matrix.get(READY_SERVER_CANDIDATE_TYPE))
        or _mapping_rows(matrix.get("ready_server_candidates"))
    )


def _source_component_candidate_ids_by_role(
    component_candidate_matrix: Mapping[str, Any],
    *,
    include_ready_server: bool,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for _, matrix_key, internal_role in PACKAGE_MATRIX_KEYS:
        role = _coverage_role_key(internal_role)
        if role == READY_SERVER_CANDIDATE_TYPE and not include_ready_server:
            continue
        rows = _selectable_component_rows(
            _stable_component_rows(_mapping_rows(component_candidate_matrix.get(matrix_key)))
        )
        ids = _component_candidate_ids_for_exposure(rows)
        if ids:
            result[role] = ids
    return result


def _source_component_candidate_count_by_role(
    component_candidate_matrix: Mapping[str, Any],
    *,
    source_ids_by_role: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    counts = {
        _coverage_role_key(role): len(ids)
        for role, ids in source_ids_by_role.items()
        if ids
    }
    for role, count in _safe_int_mapping(
        component_candidate_matrix.get("broad_count_by_role")
    ).items():
        if count > 0:
            counts[role] = max(int(count), int(counts.get(role, 0) or 0))
    return counts


def _package_component_candidate_ids_by_role(
    matrix: Mapping[str, Any],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for prompt_role, rows in matrix.items():
        if not isinstance(rows, list):
            continue
        internal_role = INTERNAL_ROLE_BY_PROMPT_ROLE.get(
            str(prompt_role),
            str(prompt_role),
        )
        role = _coverage_role_key(internal_role)
        ids = _component_candidate_ids_for_exposure(_mapping_rows(rows))
        if ids:
            result[role] = ids
    return result


def _component_candidate_ids_for_exposure(
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        component_id = str(
            row.get("component_candidate_id")
            or row.get("candidate_id")
            or _candidate_id(row)
            or ""
        ).strip()
        if not component_id or component_id in seen:
            continue
        seen.add(component_id)
        ids.append(component_id)
    return ids


def _component_matrix_full_evaluation_complete(
    component_candidate_matrix: Mapping[str, Any],
    *,
    matrix_counts: Mapping[str, int],
) -> bool:
    if not bool(component_candidate_matrix.get("full_matrix_evaluation_used")):
        return False
    if _mapping_rows(component_candidate_matrix.get(FULL_MATRIX_FAILED_CHUNKS_KEY)):
        return False
    evaluated_counts = _safe_int_mapping(
        component_candidate_matrix.get("evaluated_candidate_count_by_role")
    )
    if not evaluated_counts:
        return False
    return all(
        int(evaluated_counts.get(role, 0) or 0) >= int(total or 0)
        for role, total in matrix_counts.items()
        if int(total or 0) > 0
    )


def _dropped_before_composer_reason(
    *,
    role: str,
    source_ids_by_role: Mapping[str, Sequence[str]],
    package_skipped_reason: str | None,
    full_matrix_used: bool,
    full_matrix_complete: bool,
) -> str:
    if package_skipped_reason:
        return f"package_skipped:{package_skipped_reason}"
    if full_matrix_complete:
        return "ai_reviewed_role_pool_reduction"
    if full_matrix_used:
        return "incomplete_ai_role_pool_reduction"
    if not source_ids_by_role.get(role):
        return "broad_count_metadata_without_package_rows"
    return "pre_composer_candidate_drop"


def _role_candidate_pools_for_package(
    *,
    matrix: Mapping[str, Any],
    component_candidate_matrix: Mapping[str, Any],
) -> dict[str, Any]:
    reducer_summary = _safe_mapping(component_candidate_matrix.get("role_reducer_summary"))
    evaluated_counts = _safe_mapping(
        component_candidate_matrix.get("evaluated_candidate_count_by_role")
    )
    broad_counts = _safe_mapping(component_candidate_matrix.get("broad_count_by_role"))
    pools: dict[str, Any] = {}
    for role, rows in matrix.items():
        if not isinstance(rows, list) or not rows:
            continue
        role_key = SERVER_PLATFORM_ROLE if role == "platform" else str(role)
        reducer_row = _safe_mapping(reducer_summary.get(role_key))
        candidate_ids = [
            str(row.get("component_candidate_id") or row.get("candidate_id") or "").strip()
            for row in rows
            if isinstance(row, Mapping)
            and str(row.get("component_candidate_id") or row.get("candidate_id") or "").strip()
        ]
        pools[str(role)] = {
            "considered_count_total": _int_value(evaluated_counts.get(role_key))
            or _int_value(broad_counts.get(role_key))
            or len(candidate_ids),
            "matrix_count_total": _int_value(broad_counts.get(role_key)) or len(candidate_ids),
            "selected_count": len(candidate_ids),
            "candidate_ids": candidate_ids,
            "role_summary": _safe_diagnostic_text(
                reducer_row.get("role_summary"),
                limit=360,
            ),
            "no_viable_reason": _safe_diagnostic_text(
                reducer_row.get("no_viable_reason"),
                limit=240,
            ),
            "fit_tier_counts": _safe_mapping(reducer_row.get("fit_tier_counts")),
            "rejected_summary": _mapping_rows(reducer_row.get("rejected_summary"))[:8],
        }
    return pools


def _compact_role_coverage_summary(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return result
    for role, raw in value.items():
        if not isinstance(raw, Mapping):
            continue
        result[str(role)] = {
            "required": raw.get("required"),
            "category_ids": _string_list(raw.get("category_ids")),
            "category_count": raw.get("category_count"),
            "sent_to_llm_count": raw.get("sent_to_llm_count"),
            "raw_products_count": raw.get("raw_products_count"),
            "after_category_count": raw.get("after_category_count"),
            "after_eligibility_count": raw.get("after_eligibility_count"),
            "fit_tier_counts": _safe_mapping(raw.get("fit_tier_counts")),
            "missing_category": raw.get("missing_category"),
            "missing_candidates": raw.get("missing_candidates"),
            "missing": raw.get("missing"),
            "can_be_satisfied_by_platform": raw.get("can_be_satisfied_by_platform"),
            "filtered_reasons_top": _mapping_rows(raw.get("filtered_reasons_top"))[:3],
        }
    return result


def _compact_component_matrix_coverage_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key in (
        "total_products_by_role",
        "eligible_products_by_role",
        "sent_to_llm_by_role",
        "omitted_by_role",
        "limit_per_role",
        "bucket_summary_by_role",
        "fit_tier_summary_by_role",
        "selection_strategy",
    ):
        if key in value:
            result[key] = value[key]
    return _jsonable(result)


def _package_uses_distiller_budget_path(package: Mapping[str, Any]) -> bool:
    source = str(package.get("matrix_distiller_source") or "").strip()
    diagnostics = _safe_mapping(package.get("matrix_distiller_diagnostics"))
    return (
        bool(package.get("matrix_distiller_used"))
        or source in DISTILLER_FALLBACK_PACKAGE_SOURCES
        or bool(diagnostics.get("fallback_compaction_attempted"))
    )


def _clear_stale_distiller_budget_skip(package: dict[str, Any]) -> dict[str, Any]:
    budget = _safe_mapping(package.get("package_budget"))
    if budget.get("over_budget") is not False:
        return package
    skipped_reason = _text_or_none(package.get("package_skipped_reason"))
    fallback_reason = _text_or_none(package.get("llm_fallback_reason"))
    if (
        skipped_reason != DISTILLER_OVER_BUDGET_SKIP_REASON
        and fallback_reason != PACKAGE_OVER_BUDGET_FALLBACK_REASON
    ):
        return package
    if not _package_has_candidate_signals(package):
        return package

    result = dict(package)
    if skipped_reason == DISTILLER_OVER_BUDGET_SKIP_REASON:
        result.pop("package_skipped_reason", None)
    if fallback_reason in {
        DISTILLER_OVER_BUDGET_SKIP_REASON,
        PACKAGE_OVER_BUDGET_FALLBACK_REASON,
    }:
        result.pop("llm_fallback_reason", None)
    warnings = [
        warning
        for warning in _string_list(result.get("package_budget_warnings"))
        if warning
        not in {
            DISTILLER_OVER_BUDGET_SKIP_REASON,
            PACKAGE_OVER_BUDGET_FALLBACK_REASON,
            f"llm_configurator_package_skipped:{DISTILLER_OVER_BUDGET_SKIP_REASON}",
        }
        and not warning.startswith("llm_configurator_package_skipped:")
    ]
    if warnings:
        result["package_budget_warnings"] = _unique(warnings)
    else:
        result.pop("package_budget_warnings", None)
    return result


def _package_has_candidate_signals(package: Mapping[str, Any]) -> bool:
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    if any(isinstance(rows, list) and rows for rows in matrix.values()):
        return True
    for key in ("broad_count_by_role", "distilled_count_by_role"):
        counts = _safe_mapping(package.get(key))
        if any(isinstance(count, int) and count > 0 for count in counts.values()):
            return True
    return False


def _package_over_budget(package: Mapping[str, Any]) -> bool:
    return _safe_mapping(package.get("package_budget")).get("over_budget") is True


def _sync_package_budget(
    package: dict[str, Any],
    *,
    max_chars: int,
    initial_chars: int,
    trimmed: bool,
) -> None:
    for _ in range(12):
        measured_chars = _package_size_chars(package)
        budget = {
            **_safe_mapping(package.get("package_budget")),
            "max_chars": max_chars,
            "initial_chars": initial_chars,
            "final_chars": measured_chars,
            "trimmed": trimmed,
            "over_budget": measured_chars > max_chars,
        }
        if budget == package.get("package_budget"):
            return
        package["package_budget"] = budget
        _sync_package_candidate_exposure_policy_budget(package)
    measured_chars = _package_size_chars(package)
    package["package_budget"] = {
        **_safe_mapping(package.get("package_budget")),
        "max_chars": max_chars,
        "initial_chars": initial_chars,
        "final_chars": measured_chars,
        "trimmed": trimmed,
        "over_budget": measured_chars > max_chars,
    }
    _sync_package_candidate_exposure_policy_budget(package)


def _sync_package_candidate_exposure_policy_budget(package: dict[str, Any]) -> None:
    policy = _safe_mapping(package.get("package_candidate_exposure_policy"))
    if not policy:
        return
    budget = _safe_mapping(package.get("package_budget"))
    package["package_candidate_exposure_policy"] = {
        **policy,
        "package_budget_over_budget": budget.get("over_budget") is True,
        "package_budget_trimmed": bool(budget.get("trimmed")),
    }


def _replace_trimmed_budget_warning(
    package: dict[str, Any],
    *,
    initial_chars: int,
    max_chars: int,
) -> None:
    final_chars = _safe_mapping(package.get("package_budget")).get("final_chars")
    warnings = [
        warning
        for warning in _string_list(package.get("package_budget_warnings"))
        if not warning.startswith("llm_configurator_package_trimmed:")
    ]
    warnings.append(
        f"llm_configurator_package_trimmed:{initial_chars}:{final_chars}:{max_chars}"
    )
    package["package_budget_warnings"] = _unique(warnings)


def _trim_nonessential_package_sections(package: dict[str, Any]) -> dict[str, Any]:
    trimmed = dict(package)
    lifecycle_prompt_diagnostics = {
        "stage_a_broad_roles",
        "semantic_matrix_blueprint_roles",
        "requirement_classifier_roles",
        "effective_matrix_roles_before_category_planner",
        "category_planner_input_roles",
        "category_planner_output_roles",
        "category_planner_missing_required_roles",
        "category_planner_repair_attempted",
        "category_planner_repair_success",
        "category_planner_repair_reason",
        "category_planner_repaired_roles",
        "category_planner_unresolved_required_roles",
        "validated_category_plan_roles",
        "materialized_matrix_roles",
        "roles_dropped_after_stage_a",
        "roles_dropped_before_category_planner",
        "roles_dropped_after_category_planner",
        "roles_dropped_during_materialization",
        "role_source_by_role",
        "role_lifecycle_trace",
    }
    has_lifecycle_contract = any(
        package.get(key) not in (None, [], {})
        for key in (
            "stage_a_broad_roles",
            "effective_matrix_roles_before_category_planner",
        )
    )
    if has_lifecycle_contract:
        for key in lifecycle_prompt_diagnostics:
            if trimmed.get(key) not in (None, [], {}):
                trimmed.pop(key, None)
    requirements = _safe_mapping(trimmed.get("normalized_requirements"))
    requirement_role_plan = _safe_mapping(requirements.get("role_plan"))
    if requirement_role_plan:
        requirements = dict(requirements)
        requirements["role_plan"] = _role_plan_for_package(requirement_role_plan)
        trimmed["normalized_requirements"] = requirements
    if _package_uses_distiller_budget_path(package):
        trimmed["matrix_distiller_diagnostics"] = _compact_package_diagnostics(
            trimmed.get("matrix_distiller_diagnostics")
        )
    return trimmed if trimmed != package else package


def _compact_package_diagnostics(value: Any) -> dict[str, Any]:
    diagnostics = _safe_mapping(value)
    keep_keys = {
        "matrix_distiller_source",
        "reason",
        "error_type",
        "original_distiller_error_type",
        "fallback_compaction_attempted",
        "fallback_limit_per_role",
        "fallback_count_by_role",
        "broad_count_by_role",
        "full_matrix_evaluation_used",
        "full_matrix_evaluation_fallback_reason",
        "fallback_decision",
        "stage",
        "role",
        "chunk_index",
        "cause_error_type",
        FULL_MATRIX_FAILED_CHUNKS_KEY,
        "role_chunk_count_by_role",
        "evaluated_candidate_count_by_role",
        "selected_candidate_count_by_role",
        "role_reducer_summary",
        "no_recommendation_coverage",
        "no_recommendation_coverage_gate_passed",
        "no_recommendation_coverage_repair_attempted",
        "no_recommendation_coverage_repair_success",
        "no_recommendation_coverage_rejected",
        "no_recommendation_coverage_thresholds",
        "no_recommendation_coverage_repair_reason",
        "llm_cost_diagnostics",
        "package_budget_before_distillation",
        "package_budget_after_distillation",
        "package_budget_after_fallback",
        "package_budget_at_failure",
        "package_skipped_reason",
        "ocs_content",
    }
    compact = {key: diagnostics[key] for key in keep_keys if key in diagnostics}
    ocs_content = _safe_mapping(compact.get("ocs_content"))
    if ocs_content:
        compact["ocs_content"] = {
            key: ocs_content.get(key)
            for key in (
                "enabled",
                "available",
                "skipped_reason",
                "error_type",
                "http_status",
            )
            if key in ocs_content
        }
    return compact


def _attach_package_section_size_diagnostics(package: dict[str, Any]) -> None:
    package["package_budget"] = {
        **_safe_mapping(package.get("package_budget")),
        "section_size_diagnostics": _package_section_size_diagnostics(package),
    }


def _drop_section_size_diagnostics_if_over_budget(
    package: dict[str, Any],
    *,
    max_chars: int,
    initial_chars: int,
    trimmed: bool,
) -> None:
    if not _package_over_budget(package):
        return
    budget = dict(_safe_mapping(package.get("package_budget")))
    if "section_size_diagnostics" not in budget:
        return
    budget.pop("section_size_diagnostics", None)
    package["package_budget"] = budget
    _sync_package_budget(
        package,
        max_chars=max_chars,
        initial_chars=initial_chars,
        trimmed=trimmed,
    )


def _should_attach_package_section_size_diagnostics(
    package: Mapping[str, Any],
) -> bool:
    return (
        bool(package.get("ready_candidates_excluded_reason"))
        or package.get("ready_candidates_limit") is not None
        or _package_uses_distiller_budget_path(package)
        or _package_over_budget(package)
    )


def _package_section_size_diagnostics(package: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics_sections = {
        "matrix_distiller_diagnostics": package.get("matrix_distiller_diagnostics"),
        "role_coverage_summary": package.get("role_coverage_summary"),
        "component_matrix_coverage_summary": package.get(
            "component_matrix_coverage_summary"
        ),
        "category_catalog_summary": package.get("category_catalog_summary"),
        "category_plan_warnings": package.get("category_plan_warnings"),
        "package_budget_warnings": package.get("package_budget_warnings"),
    }
    sizes = {
        "normalized_requirements_chars": _json_size_chars(
            package.get("normalized_requirements")
        ),
        "role_plan_chars": _json_size_chars(package.get("role_plan")),
        "category_plan_chars": _json_size_chars(package.get("category_plan")),
        "matrix_chars": _json_size_chars(package.get("component_candidate_matrix")),
        "ready_candidates_chars": _json_size_chars(
            package.get("ready_stock_candidates")
        ),
        "diagnostics_chars": _json_size_chars(diagnostics_sections),
        "source_request_chars": _json_size_chars(package.get("user_request")),
    }
    if package.get("ready_candidates_excluded_reason"):
        sizes["ready_candidates_excluded_reason"] = package.get(
            "ready_candidates_excluded_reason"
        )
    if package.get("ready_candidates_limit") is not None:
        sizes["ready_candidates_limit"] = package.get("ready_candidates_limit")
    char_sizes = {
        key: chars
        for key, chars in sizes.items()
        if key.endswith("_chars") and isinstance(chars, int)
    }
    sizes["largest_sections"] = [
        {"section": key, "chars": chars}
        for key, chars in sorted(char_sizes.items(), key=lambda item: item[1], reverse=True)
    ][:7]
    return sizes


def _json_size_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _package_with_budget(package: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    package = dict(package)
    initial_size = _package_size_chars(package)
    package["package_budget"] = {
        "max_chars": max_chars,
        "initial_chars": initial_size,
        "final_chars": 0,
        "trimmed": False,
        "over_budget": False,
    }
    _sync_package_budget(
        package,
        max_chars=max_chars,
        initial_chars=initial_size,
        trimmed=False,
    )
    if not _package_over_budget(package):
        result = _clear_stale_distiller_budget_skip(package)
        _sync_package_budget(
            result,
            max_chars=max_chars,
            initial_chars=initial_size,
            trimmed=False,
        )
        if _should_attach_package_section_size_diagnostics(result):
            _attach_package_section_size_diagnostics(result)
            _sync_package_budget(
                result,
                max_chars=max_chars,
                initial_chars=initial_size,
                trimmed=False,
            )
            _drop_section_size_diagnostics_if_over_budget(
                result,
                max_chars=max_chars,
                initial_chars=initial_size,
                trimmed=False,
            )
        return result

    trimmed = _trim_nonessential_package_sections(package)
    diagnostics_trimmed = trimmed != package
    _sync_package_budget(
        trimmed,
        max_chars=max_chars,
        initial_chars=initial_size,
        trimmed=diagnostics_trimmed,
    )
    if diagnostics_trimmed:
        _replace_trimmed_budget_warning(
            trimmed,
            initial_chars=initial_size,
            max_chars=max_chars,
        )
        _sync_package_budget(
            trimmed,
            max_chars=max_chars,
            initial_chars=initial_size,
            trimmed=diagnostics_trimmed,
        )

    if _package_over_budget(trimmed):
        trimmed["llm_fallback_reason"] = PACKAGE_OVER_BUDGET_FALLBACK_REASON
        existing_skipped_reason = _text_or_none(trimmed.get("package_skipped_reason"))
        skipped_reason = existing_skipped_reason or (
            DISTILLER_OVER_BUDGET_SKIP_REASON
            if _package_uses_distiller_budget_path(trimmed)
            else PACKAGE_OVER_BUDGET_BEFORE_COMPOSER_REASON
        )
        trimmed["package_skipped_reason"] = skipped_reason
        trimmed["package_budget_warnings"] = _unique(
            [
                *_string_list(trimmed.get("package_budget_warnings")),
                PACKAGE_OVER_BUDGET_FALLBACK_REASON,
                skipped_reason,
            ]
        )
        if skipped_reason == DISTILLER_OVER_BUDGET_SKIP_REASON:
            _attach_package_section_size_diagnostics(trimmed)
            _drop_section_size_diagnostics_if_over_budget(
                trimmed,
                max_chars=max_chars,
                initial_chars=initial_size,
                trimmed=diagnostics_trimmed,
            )
        _sync_package_budget(
            trimmed,
            max_chars=max_chars,
            initial_chars=initial_size,
            trimmed=diagnostics_trimmed,
        )
    result = _clear_stale_distiller_budget_skip(trimmed)
    _sync_package_budget(
        result,
        max_chars=max_chars,
        initial_chars=initial_size,
        trimmed=bool(_safe_mapping(result.get("package_budget")).get("trimmed")),
    )
    if _should_attach_package_section_size_diagnostics(result):
        _attach_package_section_size_diagnostics(result)
        result_trimmed = bool(_safe_mapping(result.get("package_budget")).get("trimmed"))
        _sync_package_budget(
            result,
            max_chars=max_chars,
            initial_chars=initial_size,
            trimmed=result_trimmed,
        )
        _drop_section_size_diagnostics_if_over_budget(
            result,
            max_chars=max_chars,
            initial_chars=initial_size,
            trimmed=result_trimmed,
        )
    return result


def _package_size_chars(package: Mapping[str, Any]) -> int:
    return len(json.dumps(package, ensure_ascii=False, sort_keys=True, default=str))


def prepare_v2_composer_package(
    package: Mapping[str, Any],
    *,
    max_package_chars: int | None = None,
    force_mode: str | None = None,
) -> dict[str, Any]:
    """Select verbose or compact full-matrix serialization for v2 Composer."""

    verbose_package = dict(package)
    if not _is_v2_composer_package(verbose_package):
        return verbose_package

    compact_package = compact_composer_package(verbose_package)
    verbose_chars = json_size_chars(verbose_package)
    candidate_total = _composer_candidate_count_total(verbose_package)
    diagnostics_without_mode = composer_package_compaction_diagnostics(
        verbose_package,
        compact_package,
        selected_package=compact_package,
        selected_package_mode=COMPACT_FULL_MATRIX_MODE,
    )
    compact_has_loss = bool(diagnostics_without_mode.get("package_candidate_loss"))
    requested_mode = str(force_mode or "").strip()
    if requested_mode in {VERBOSE_FULL_MATRIX_MODE, COMPACT_FULL_MATRIX_MODE}:
        selected_mode = requested_mode
    elif (
        not compact_has_loss
        and (
            candidate_total >= COMPACT_FULL_MATRIX_AUTO_CANDIDATE_THRESHOLD
            or verbose_chars > COMPACT_FULL_MATRIX_AUTO_CHAR_THRESHOLD
            or _package_over_budget(verbose_package)
            or bool(verbose_package.get("package_skipped_reason"))
        )
    ):
        selected_mode = COMPACT_FULL_MATRIX_MODE
    else:
        selected_mode = VERBOSE_FULL_MATRIX_MODE

    if selected_mode == COMPACT_FULL_MATRIX_MODE and compact_has_loss:
        selected_mode = VERBOSE_FULL_MATRIX_MODE

    selected = (
        dict(compact_package)
        if selected_mode == COMPACT_FULL_MATRIX_MODE
        else dict(verbose_package)
    )
    diagnostics = composer_package_compaction_diagnostics(
        verbose_package,
        compact_package,
        selected_package=selected,
        selected_package_mode=selected_mode,
    )
    selected.update(diagnostics)
    selected["composer_package_full_matrix_used"] = (
        (
            bool(selected.get("composer_package_full_matrix_used"))
            or not bool(selected.get("package_candidate_exposure_incomplete"))
        )
        and not bool(selected.get("package_candidate_loss"))
    )
    selected["composer_package_full_matrix_policy"] = {
        **_safe_mapping(selected.get("composer_package_full_matrix_policy")),
        "candidate_loss_allowed": False,
        "technical_trimming_allowed": False,
        "semantic_filtering_allowed": False,
        "top_n_allowed": False,
        "chunking_only_when_over_limit": True,
    }
    if selected_mode == COMPACT_FULL_MATRIX_MODE:
        _clear_package_over_budget_skip_after_compaction(selected)

    budget = max_package_chars or _int_value(
        _safe_mapping(verbose_package.get("package_budget")).get("max_chars")
    )
    if budget is not None:
        _sync_package_budget(
            selected,
            max_chars=budget,
            initial_chars=verbose_chars,
            trimmed=selected_mode == COMPACT_FULL_MATRIX_MODE,
        )
        if selected_mode == COMPACT_FULL_MATRIX_MODE:
            _clear_package_over_budget_skip_after_compaction(selected)
            _sync_package_budget(
                selected,
                max_chars=budget,
                initial_chars=verbose_chars,
                trimmed=True,
            )
    _refresh_context_mode_diagnostics(
        selected,
        verbose_package=verbose_package,
        compact_package=compact_package,
        selected_mode=selected_mode,
    )
    if budget is not None:
        _sync_package_budget(
            selected,
            max_chars=budget,
            initial_chars=verbose_chars,
            trimmed=selected_mode == COMPACT_FULL_MATRIX_MODE,
        )
        _refresh_context_mode_diagnostics(
            selected,
            verbose_package=verbose_package,
            compact_package=compact_package,
            selected_mode=selected_mode,
        )
    return selected


def _refresh_context_mode_diagnostics(
    package: dict[str, Any],
    *,
    verbose_package: Mapping[str, Any],
    compact_package: Mapping[str, Any],
    selected_mode: str,
) -> None:
    for _ in range(4):
        diagnostics = composer_package_compaction_diagnostics(
            verbose_package,
            compact_package,
            selected_package=package,
            selected_package_mode=selected_mode,
        )
        changed = False
        for key, value in diagnostics.items():
            if package.get(key) != value:
                package[key] = value
                changed = True
        if not changed:
            break
    selected_chars = json_size_chars(package)
    package["selected_context_chars"] = selected_chars
    package["selected_context_size"] = {
        "chars": selected_chars,
        "tokens_estimate": max(1, selected_chars // 4),
    }


def _clear_package_over_budget_skip_after_compaction(package: dict[str, Any]) -> None:
    budget = _safe_mapping(package.get("package_budget"))
    if budget.get("over_budget") is True:
        return
    skipped_reason = _text_or_none(package.get("package_skipped_reason"))
    fallback_reason = _text_or_none(package.get("llm_fallback_reason"))
    removable = {
        PACKAGE_OVER_BUDGET_FALLBACK_REASON,
        PACKAGE_OVER_BUDGET_BEFORE_COMPOSER_REASON,
        DISTILLER_OVER_BUDGET_SKIP_REASON,
    }
    if skipped_reason in removable:
        package.pop("package_skipped_reason", None)
    if fallback_reason in removable:
        package.pop("llm_fallback_reason", None)
    warnings = [
        warning
        for warning in _string_list(package.get("package_budget_warnings"))
        if warning not in removable
        and not warning.startswith("llm_configurator_package_skipped:")
        and not warning.startswith("llm_configurator_package_trimmed:")
    ]
    if warnings:
        package["package_budget_warnings"] = _unique(warnings)
    else:
        package.pop("package_budget_warnings", None)


def _is_v2_composer_package(package: Mapping[str, Any]) -> bool:
    if str(package.get("pipeline_version") or "").strip() == "v2_composer_first":
        return True
    requirements = _safe_mapping(package.get("normalized_requirements"))
    return bool(requirements.get("composer_first")) or str(
        requirements.get("pipeline_version") or ""
    ).strip() == "v2_composer_first"


def _composer_package_mode(package: Mapping[str, Any]) -> str:
    mode = str(
        package.get("v2_package_mode") or package.get("selected_package_mode") or ""
    ).strip()
    if mode in {VERBOSE_FULL_MATRIX_MODE, COMPACT_FULL_MATRIX_MODE}:
        return mode
    return VERBOSE_FULL_MATRIX_MODE


def _component_index_for_package(
    *,
    source_component_candidate_matrix: Mapping[str, Any],
    package: Mapping[str, Any],
) -> dict[str, _IndexedComponentCandidate]:
    allowed_ids = _component_candidate_ids_from_package(package)
    if not allowed_ids:
        return {}
    source_index = _component_index(source_component_candidate_matrix)
    index = {
        component_id: candidate
        for component_id, candidate in source_index.items()
        if component_id in allowed_ids
    }
    missing_ids = allowed_ids.difference(index)
    if not missing_ids:
        return index

    package_matrix = _safe_mapping(package.get("component_candidate_matrix"))
    for prompt_role, rows in package_matrix.items():
        internal_role = INTERNAL_ROLE_BY_PROMPT_ROLE.get(str(prompt_role), str(prompt_role))
        if internal_role not in PROMPT_ROLE_BY_INTERNAL_ROLE:
            continue
        for row in _mapping_rows(rows):
            compact = dict(row)
            component_candidate_id = str(
                compact.get("component_candidate_id") or compact.get("candidate_id") or ""
            ).strip()
            if not component_candidate_id or component_candidate_id not in missing_ids:
                continue
            index[component_candidate_id] = _IndexedComponentCandidate(
                component_candidate_id=component_candidate_id,
                prompt_role=str(prompt_role),
                internal_role=internal_role,
                row=compact,
                source=compact,
            )
    return index


def _component_candidate_ids_from_package(package: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    for rows in matrix.values():
        for row in _mapping_rows(rows):
            component_id = str(
                row.get("component_candidate_id") or row.get("candidate_id") or ""
            ).strip()
            if component_id:
                result.add(component_id)
    return result


def _component_index(
    component_candidate_matrix: Mapping[str, Any],
) -> dict[str, _IndexedComponentCandidate]:
    index: dict[str, _IndexedComponentCandidate] = {}
    for prompt_role, matrix_key, internal_role in MATRIX_KEYS:
        rows = _selectable_component_rows(
            _stable_component_rows(_mapping_rows(component_candidate_matrix.get(matrix_key)))
        )
        for row in rows:
            compact = _compact_component_candidate(row, prompt_role, internal_role)
            component_candidate_id = str(compact.get("component_candidate_id") or "").strip()
            if not component_candidate_id:
                continue
            index[component_candidate_id] = _IndexedComponentCandidate(
                component_candidate_id=component_candidate_id,
                prompt_role=prompt_role,
                internal_role=internal_role,
                row=compact,
                source=row,
            )
    return index


def _stock_candidate_index(
    *,
    ready_stock_candidates: list[Mapping[str, Any]],
    rule_based_build_candidates: list[Mapping[str, Any]],
) -> dict[str, _IndexedStockCandidate]:
    index: dict[str, _IndexedStockCandidate] = {}
    for candidate in ready_stock_candidates:
        candidate_id = _candidate_id(candidate)
        if not candidate_id:
            continue
        index[candidate_id] = _IndexedStockCandidate(
            candidate_id=candidate_id,
            candidate_type=READY_SERVER_CANDIDATE_TYPE,
            row=candidate,
        )
    for candidate in rule_based_build_candidates:
        candidate_id = _candidate_id(candidate)
        if not candidate_id:
            continue
        candidate_type = str(candidate.get("candidate_type") or BUILD_CANDIDATE_TYPE)
        if candidate_type == BUILD_CANDIDATE_TYPE and _is_partial_build(candidate):
            candidate_type = PARTIAL_BUILD_CANDIDATE_TYPE
        index[candidate_id] = _IndexedStockCandidate(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            row=candidate,
        )
    return index


def _compact_component_candidate(
    row: Mapping[str, Any],
    prompt_role: str,
    internal_role: str,
) -> dict[str, Any]:
    facts = _safe_extracted_facts(row.get("extracted_facts"))
    normalized_vendor = row.get("normalized_vendor") or facts.get("normalized_vendor")
    return {
        "component_candidate_id": row.get("component_candidate_id") or row.get("candidate_id"),
        "role": prompt_role,
        "capability_id": row.get("capability_id"),
        "category_id": row.get("category_id"),
        "category_name": row.get("category_name"),
        "producer": row.get("producer"),
        "normalized_vendor": normalized_vendor,
        "part_number": row.get("part_number"),
        "item_name": _safe_diagnostic_text(
            row.get("item_name") or row.get("name"),
            limit=180,
        ),
        "item_name_rus": _safe_diagnostic_text(row.get("item_name_rus"), limit=180),
        "product_name": _safe_diagnostic_text(row.get("product_name"), limit=180),
        "product_description": _safe_diagnostic_text(
            row.get("product_description"),
            limit=240,
        ),
        "name": _safe_diagnostic_text(row.get("name") or row.get("item_name"), limit=180),
        "catalog_path": _compact_catalog_path(
            row.get("catalog_path") or row.get("catalog_path_json")
        ),
        "package_json": _compact_package_json(row.get("package_json")),
        "price_value": row.get("price_value"),
        "price_currency": row.get("price_currency"),
        "available_quantity": row.get("available_quantity"),
        "quantity_required": row.get("quantity_required"),
        "extracted_facts": facts,
        "cpu_cores": row.get("cpu_cores") or facts.get("cpu_cores"),
        "cpu_over_requirement": row.get("cpu_over_requirement"),
        "storage_capacity_tb": row.get("storage_capacity_tb")
        or facts.get("storage_capacity_tb"),
        "storage_over_requirement": row.get("storage_over_requirement"),
        "raw_capacity_tb": row.get("raw_capacity_tb") or facts.get("raw_capacity_tb"),
        "usable_capacity_tb": row.get("usable_capacity_tb")
        or facts.get("usable_capacity_tb"),
        "redundancy_level": row.get("redundancy_level")
        or facts.get("redundancy_level"),
        "controller_count": row.get("controller_count")
        or facts.get("controller_count"),
        "drive_count": row.get("drive_count") or facts.get("drive_count"),
        "drive_capacity_tb": row.get("drive_capacity_tb")
        or facts.get("drive_capacity_tb"),
        "drive_type": row.get("drive_type") or facts.get("drive_type"),
        "drive_interface": row.get("drive_interface")
        or facts.get("drive_interface"),
        "host_protocol": row.get("host_protocol") or facts.get("host_protocol"),
        "host_port_count": row.get("host_port_count")
        or facts.get("host_port_count"),
        "host_port_speed": row.get("host_port_speed")
        or facts.get("host_port_speed"),
        "host_port_speed_gbps": row.get("host_port_speed_gbps")
        or facts.get("host_port_speed_gbps"),
        "host_port_media": row.get("host_port_media") or facts.get("host_port_media"),
        "warranty_months": row.get("warranty_months") or facts.get("warranty_months"),
        "ram_module_capacity_gb": row.get("ram_module_capacity_gb")
        or facts.get("ram_capacity_gb"),
        "ram_over_requirement_gb": row.get("ram_over_requirement_gb"),
        "network_ports_count": row.get("network_ports_count")
        or facts.get("network_ports_count"),
        "network_speed": row.get("network_speed") or facts.get("network_speed"),
        "network_speed_gbps": row.get("network_speed_gbps")
        or facts.get("network_speed_gbps"),
        "network_media": row.get("network_media") or facts.get("network_media"),
        "network_interface": row.get("network_interface")
        or facts.get("network_interface"),
        "port_count": row.get("port_count") or facts.get("port_count"),
        "port_speed": row.get("port_speed") or facts.get("port_speed"),
        "port_speed_gbps": row.get("port_speed_gbps") or facts.get("port_speed_gbps"),
        "port_media": row.get("port_media") or facts.get("port_media"),
        "uplink_count": row.get("uplink_count") or facts.get("uplink_count"),
        "uplink_speed": row.get("uplink_speed") or facts.get("uplink_speed"),
        "uplink_speed_gbps": row.get("uplink_speed_gbps")
        or facts.get("uplink_speed_gbps"),
        "uplink_media": row.get("uplink_media") or facts.get("uplink_media"),
        "poe_supported": row.get("poe_supported") or facts.get("poe_supported"),
        "poe_budget_w": row.get("poe_budget_w") or facts.get("poe_budget_w"),
        "poe_standard": row.get("poe_standard") or facts.get("poe_standard"),
        "l2_supported": row.get("l2_supported") or facts.get("l2_supported"),
        "l3_supported": row.get("l3_supported") or facts.get("l3_supported"),
        "stacking_supported": row.get("stacking_supported")
        or facts.get("stacking_supported"),
        "managed_status": row.get("managed_status") or facts.get("managed_status"),
        "airflow": row.get("airflow") or facts.get("airflow"),
        "redundant_psu": row.get("redundant_psu") or facts.get("redundant_psu"),
        "transceiver_form_factor": row.get("transceiver_form_factor")
        or facts.get("transceiver_form_factor"),
        "fit_label": row.get("fit_label"),
        "fit_reason": row.get("fit_reason"),
        "fit_tier": _normalized_fit_tier(row.get("fit_tier")),
        "content_properties": _compact_content_properties(
            row.get("ocs_content_properties")
        ),
        "matrix_distiller_fit_tier": row.get("matrix_distiller_fit_tier"),
        "matrix_distiller_confidence": row.get("matrix_distiller_confidence"),
        "matrix_distiller_facts": _safe_extracted_facts(
            row.get("matrix_distiller_facts")
        ),
        "matrix_distiller_matched_constraints": _string_list(
            row.get("matrix_distiller_matched_constraints")
        )[:8],
        "matrix_distiller_missing_facts": _string_list(
            row.get("matrix_distiller_missing_facts")
        )[:8],
        "matrix_distiller_mismatch_reasons": _string_list(
            row.get("matrix_distiller_mismatch_reasons")
        )[:8],
        "matrix_distiller_price_stock_notes": _string_list(
            row.get("matrix_distiller_price_stock_notes")
        )[:8],
        "matrix_distiller_compatibility_assumptions": _string_list(
            row.get("matrix_distiller_compatibility_assumptions")
        )[:8],
        "matrix_distiller_engineer_checks": _string_list(
            row.get("matrix_distiller_engineer_checks")
        )[:8],
        "matrix_distiller_evidence": _safe_diagnostic_text(
            row.get("matrix_distiller_evidence"),
            limit=220,
        ),
        "over_requirement": row.get("over_requirement"),
        "eligibility_warnings": _string_list(row.get("eligibility_warnings")),
        "fit_reasons": _string_list(row.get("fit_reasons")),
        "score": row.get("score"),
        "selection_bucket": row.get("selection_bucket"),
        "bucket_priority": row.get("bucket_priority"),
    }


def _safe_extracted_facts(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if key_text in {"raw", "raw_json", "payload", "debug", "diagnostics"}:
            continue
        if isinstance(item, Mapping | list | tuple):
            continue
        result[key_text] = _jsonable(item)
    return result


def _compact_catalog_path(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            text = _safe_diagnostic_text(
                item.get("name") or item.get("category_name"),
                limit=80,
            )
        else:
            text = _safe_diagnostic_text(item, limit=80)
        if text:
            result.append(text)
        if len(result) >= 6:
            break
    return result


def _compact_content_properties(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _safe_diagnostic_text(item.get("name"), limit=80)
        property_value = _safe_diagnostic_text(item.get("value"), limit=120)
        if not name or not property_value:
            continue
        row = {
            "name": name,
            "value": property_value,
            "unit": _safe_diagnostic_text(item.get("unit"), limit=24),
        }
        rows.append({key: val for key, val in row.items() if val})
        if len(rows) >= 12:
            break
    return rows


def _compact_package_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        key_text = _safe_diagnostic_text(key, limit=60)
        if not key_text:
            continue
        if isinstance(item, str | int | float | bool) or item is None:
            result[key_text] = item
        elif isinstance(item, Mapping):
            nested = {
                _safe_diagnostic_text(nested_key, limit=60): nested_value
                for nested_key, nested_value in item.items()
                if isinstance(nested_value, str | int | float | bool)
                or nested_value is None
            }
            nested = {key: val for key, val in nested.items() if key}
            if nested:
                result[key_text] = nested
        elif isinstance(item, list | tuple):
            scalars = [
                nested_item
                for nested_item in item
                if isinstance(nested_item, str | int | float | bool)
                or nested_item is None
            ]
            if scalars:
                result[key_text] = scalars[:12]
        if len(result) >= 12:
            break
    return _jsonable(result)


def _compact_ready_stock_candidates(
    ready_stock_candidates: list[Mapping[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = sorted(ready_stock_candidates, key=_ready_stock_candidate_sort_key)
    if limit is not None:
        candidates = candidates[: max(0, limit)]
    for candidate in candidates:
        raw = candidate.get("raw") if isinstance(candidate.get("raw"), Mapping) else {}
        facts = _ready_candidate_facts(candidate)
        rows.append(
            {
                "candidate_id": _candidate_id(candidate),
                "candidate_type": READY_SERVER_CANDIDATE_TYPE,
                "producer": _safe_diagnostic_text(candidate.get("producer"), limit=120),
                "part_number": _safe_diagnostic_text(
                    candidate.get("part_number"),
                    limit=120,
                ),
                "name": _safe_diagnostic_text(
                    candidate.get("item_name") or candidate.get("name"),
                    limit=220,
                ),
                "price_value": candidate.get("price_value"),
                "price_currency": candidate.get("price_currency"),
                "available_quantity": candidate.get("available_quantity"),
                "quantity_required": raw.get("quantity_required"),
                "extracted_facts": facts,
                "gaps": _string_list(candidate.get("missing_requirements"))[:6],
                "risks": _string_list(candidate.get("risk_flags"))[:6],
                "score": candidate.get("score") or candidate.get("confidence_score"),
                "fit_reason": _safe_diagnostic_text(
                    _ready_fit_reason(candidate),
                    limit=260,
                ),
            }
        )
    return rows


def _compact_rule_based_build_candidates(
    build_candidates: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in sorted(build_candidates, key=_rule_based_build_candidate_sort_key):
        components = _mapping_rows(candidate.get("components"))
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "candidate_type": (
                    PARTIAL_BUILD_CANDIDATE_TYPE
                    if _is_partial_build(candidate)
                    else BUILD_CANDIDATE_TYPE
                ),
                "platform": _compact_build_platform(candidate),
                "components": _compact_build_components(components),
                "completeness_status": candidate.get("completeness_status"),
                "score": candidate.get("score"),
                "component_candidate_ids": _component_ids_by_prompt_role(
                    components
                ),
                "quantities": _quantities_by_prompt_role(
                    components
                ),
                "total_price_value": candidate.get("total_price_value"),
                "total_price_currency": candidate.get("total_price_currency"),
                "total_price_note": candidate.get("total_price_note"),
                "missing_component_roles": candidate.get("missing_component_roles", []),
                "compatibility_warnings": candidate.get("compatibility_warnings", []),
                "rank_reason": candidate.get("rank_reason", []),
                "requirement_fit": candidate.get("requirement_fit"),
                "right_size_note": candidate.get("right_size_note"),
                "cpu_over_requirement": candidate.get("cpu_over_requirement"),
                "storage_over_requirement": candidate.get("storage_over_requirement"),
                "ram_overage_gb": candidate.get("ram_overage_gb"),
                "overfit_reason": candidate.get("overfit_reason"),
            }
        )
    return rows


def _stable_component_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=_component_candidate_sort_key)


def _selectable_component_rows(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if _normalized_fit_tier(row.get("fit_tier")) in SELECTABLE_FIT_TIERS
    ]


def _component_candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    facts = row.get("extracted_facts") if isinstance(row.get("extracted_facts"), Mapping) else {}
    normalized_vendor = row.get("normalized_vendor") or facts.get("normalized_vendor")
    return (
        _fit_tier_rank(row.get("fit_tier")),
        _int_value(row.get("bucket_priority")) or 90,
        -(_int_value(row.get("score")) or 0),
        *_price_sort_key(row.get("price_value"), row.get("quantity_required")),
        *_over_requirement_sort_key(row),
        -(_int_value(row.get("available_quantity")) or -1),
        _stable_text(normalized_vendor or row.get("producer")),
        _stable_text(row.get("part_number")),
        _stable_text(
            row.get("item_id")
            or row.get("component_candidate_id")
            or row.get("candidate_id")
        ),
    )


def _normalized_fit_tier(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in FIT_TIER_RANK else FIT_TIER_POSSIBLE


def _fit_tier_rank(value: Any) -> int:
    return FIT_TIER_RANK.get(
        _normalized_fit_tier(value),
        FIT_TIER_RANK[FIT_TIER_FALLBACK_UNKNOWN],
    )


def _ready_stock_candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -(_int_value(candidate.get("score") or candidate.get("confidence_score")) or 0),
        -len(_string_list(candidate.get("matched_requirements"))),
        *_price_sort_key(candidate.get("price_value"), candidate.get("quantity_required")),
        _stable_text(candidate.get("producer")),
        _stable_text(candidate.get("part_number")),
        _stable_text(candidate.get("item_id") or _candidate_id(candidate)),
    )


def _rule_based_build_candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    missing_roles = _string_list(candidate.get("missing_component_roles"))
    if not missing_roles:
        missing_roles = _string_list(candidate.get("missing_components"))
    warnings = [
        *_string_list(candidate.get("compatibility_warnings")),
        *_string_list(candidate.get("risk_flags")),
        *_string_list(candidate.get("missing_requirements")),
    ]
    return (
        0 if str(candidate.get("completeness_status") or "") == "complete" else 1,
        _fatal_warning_count(warnings),
        len(missing_roles),
        *_price_sort_key(
            candidate.get("total_price_value") or candidate.get("price_value"),
            1,
        ),
        -(_int_value(candidate.get("score") or candidate.get("confidence_score")) or 0),
        _stable_text(_candidate_id(candidate)),
    )


def _price_sort_key(value: Any, quantity: Any) -> tuple[int, Decimal]:
    price = _decimal_value(value)
    if price is None:
        return (1, Decimal("Infinity"))
    quantity_value = _int_value(quantity)
    if quantity_value is not None and quantity_value > 0:
        price *= quantity_value
    return (0, price)


def _over_requirement_sort_key(row: Mapping[str, Any]) -> tuple[int, float]:
    value = (
        row.get("over_requirement")
        if row.get("over_requirement") not in (None, "")
        else row.get("cpu_over_requirement")
    )
    if value in (None, ""):
        value = row.get("storage_over_requirement")
    if value in (None, ""):
        value = row.get("ram_over_requirement_gb")
    numeric = _float_value(value)
    if numeric is None:
        return (1, float("inf"))
    return (0, numeric)


def _fatal_warning_count(values: list[str]) -> int:
    count = 0
    for value in values:
        lowered = str(value or "").casefold()
        if (
            "fatal" in lowered
            or "incompat" in lowered
            or "mismatch" in lowered
            or "несовмест" in lowered
        ):
            count += 1
    return count


def _stable_text(value: Any) -> str:
    return str(value or "").strip().casefold()


def _ready_candidate_facts(candidate: Mapping[str, Any]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    name = str(candidate.get("item_name") or candidate.get("name") or "")
    ram_gb = _parse_ready_ram_gb(name)
    if ram_gb is not None:
        facts["ram_gb"] = ram_gb
    if "ssd" in name.casefold():
        facts["storage_type"] = "SSD"
    if "hdd" in name.casefold():
        facts["storage_type"] = "HDD"
    return facts


def _parse_ready_ram_gb(text: str) -> int | None:
    matches = re.findall(r"\b(\d+)\s*(?:GB|ГБ)\b", text, flags=re.IGNORECASE)
    if not matches:
        return None
    return max(int(value) for value in matches)


def _ready_fit_reason(candidate: Mapping[str, Any]) -> str | None:
    matched = _string_list(candidate.get("matched_requirements"))
    if matched:
        return "; ".join(matched[:4])
    return None


def _compact_build_platform(candidate: Mapping[str, Any]) -> dict[str, Any]:
    platform = candidate.get("platform")
    if isinstance(platform, Mapping):
        return {
            "producer": platform.get("producer"),
            "part_number": platform.get("part_number"),
            "name": platform.get("item_name") or platform.get("name"),
        }
    for component in _mapping_rows(candidate.get("components")):
        if _normalize_role(component.get("role")) == SERVER_PLATFORM_ROLE:
            return {
                "producer": component.get("producer"),
                "part_number": component.get("part_number"),
                "name": component.get("item_name") or component.get("name"),
            }
    return {}


def _compact_build_components(components: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in components:
        internal_role = _normalize_role(component.get("role"))
        if internal_role is None:
            continue
        rows.append(
            {
                "role": PROMPT_ROLE_BY_INTERNAL_ROLE[internal_role],
                "component_candidate_id": component.get("component_candidate_id"),
                "producer": component.get("producer"),
                "part_number": component.get("part_number"),
                "name": component.get("item_name") or component.get("name"),
                "quantity_required": component.get("quantity_required"),
                "available_quantity": component.get("available_quantity"),
                "price_value": component.get("price_value"),
                "price_currency": component.get("price_currency"),
                "fit_label": component.get("fit_label"),
                "fit_reason": component.get("fit_reason"),
            }
        )
    return rows


def _component_ids_by_prompt_role(components: list[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for component in components:
        internal_role = _normalize_role(component.get("role"))
        component_candidate_id = str(component.get("component_candidate_id") or "").strip()
        if internal_role is None or not component_candidate_id:
            continue
        result[PROMPT_ROLE_BY_INTERNAL_ROLE[internal_role]] = component_candidate_id
    return result


def _quantities_by_prompt_role(components: list[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for component in components:
        internal_role = _normalize_role(component.get("role"))
        quantity = _int_value(component.get("quantity_required"))
        if internal_role is None or quantity is None:
            continue
        result[PROMPT_ROLE_BY_INTERNAL_ROLE[internal_role]] = quantity
    return result


def _package_normalized_requirements(
    normalized_requirements: Any,
    component_candidate_matrix: Mapping[str, Any],
) -> Any:
    if isinstance(normalized_requirements, list):
        rows = [row for row in normalized_requirements if isinstance(row, Mapping)]
        if len(rows) == 1:
            return dict(rows[0])
        if rows:
            return {"items": [dict(row) for row in rows]}
    if isinstance(normalized_requirements, Mapping):
        return dict(normalized_requirements)

    matrix_requirements = component_candidate_matrix.get("normalized_requirements")
    if isinstance(matrix_requirements, Mapping):
        return dict(matrix_requirements)
    return {}


def _requirements_with_role_plan(
    package_requirements: Mapping[str, Any],
    *,
    role_plan: Mapping[str, Any],
    required_capabilities: Sequence[Mapping[str, Any]],
    optional_capabilities: Sequence[Mapping[str, Any]],
    required_roles: Sequence[str],
    semantic_fields: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(package_requirements)
    if role_plan and not isinstance(result.get("role_plan"), Mapping):
        result["role_plan"] = _role_plan_for_package(role_plan)
    if required_capabilities:
        result["required_capabilities"] = [dict(row) for row in required_capabilities]
    if optional_capabilities:
        result["optional_capabilities"] = [dict(row) for row in optional_capabilities]
    if required_roles:
        result["required_roles"] = _string_list(list(required_roles))
    for key in (
        "classified_requirements",
        "primary_object_feature_requirements",
        "engineering_check_requirements",
        "logistics_or_commercial_constraints",
        "unclassified_source_fragments",
    ):
        value = semantic_fields.get(key)
        if isinstance(value, list) and value and not isinstance(result.get(key), list):
            result[key] = (
                [dict(row) for row in _mapping_rows(value)]
                if key != "unclassified_source_fragments"
                else _string_list(value)
            )
    return result


def _role_plan_for_package(role_plan: Mapping[str, Any]) -> dict[str, Any]:
    omitted_keys = {
        "stage_a_broad_roles",
        "semantic_matrix_blueprint_roles",
        "requirement_classifier_roles",
        "effective_matrix_roles_before_category_planner",
        "category_planner_input_roles",
        "category_planner_output_roles",
        "validated_category_plan_roles",
        "materialized_matrix_roles",
        "composer_package_roles",
        "roles_dropped_after_stage_a",
        "roles_dropped_before_category_planner",
        "roles_dropped_after_category_planner",
        "roles_dropped_during_materialization",
        "roles_dropped_reason_by_role",
        "role_source_by_role",
        "role_lifecycle_trace",
    }
    return {
        str(key): value
        for key, value in role_plan.items()
        if str(key) not in omitted_keys
    }


def _validate_recommendations(
    recommendations: list[LlmRecommendationPayload],
    *,
    stock_candidate_index: dict[str, _IndexedStockCandidate],
    component_index: dict[str, _IndexedComponentCandidate],
    user_request: str | None,
    normalized_requirements: Any,
    limit: int,
    evidence_pack: Mapping[str, Any] | None = None,
    evidence_review: Mapping[str, Any] | None = None,
    use_recommendation_evidence: bool = False,
    selection_eligible_recommendation_ids: set[str] | None = None,
    schema_rejections: Sequence[_RejectedProposal] = (),
    proposal_indexes: Sequence[int] = (),
    proposal_count: int | None = None,
) -> _ValidatedRecommendationPool:
    valid: list[dict[str, Any]] = []
    rejected: list[_RejectedProposal] = list(schema_rejections)
    warnings: list[str] = [_rejected_warning(item) for item in rejected]
    evidence_by_component_id = evidence_components_by_id(evidence_pack)
    relation_evidence_by_recommendation_id = evidence_relations_by_recommendation_id(
        evidence_pack
    )
    evidence_review_by_recommendation_id = _evidence_review_by_recommendation_id(
        evidence_review
    )
    for source_order, recommendation in enumerate(recommendations):
        proposal_index = (
            proposal_indexes[source_order]
            if source_order < len(proposal_indexes)
            else source_order
        )
        try:
            validated, recommendation_warnings = _validate_recommendation(
                recommendation,
                stock_candidate_index=stock_candidate_index,
                component_index=component_index,
                user_request=user_request,
                normalized_requirements=normalized_requirements,
                evidence_by_component_id=evidence_by_component_id,
                relation_evidence=relation_evidence_by_recommendation_id.get(
                    recommendation.recommendation_id,
                    [],
                ),
                evidence_review=evidence_review_by_recommendation_id.get(
                    recommendation.recommendation_id
                ),
                use_recommendation_evidence=use_recommendation_evidence,
            )
        except Exception as exc:  # pragma: no cover - exercised via safety regression
            validated = None
            recommendation_warnings = [
                (
                    f"{recommendation.recommendation_id}: validator exception "
                    f"{type(exc).__name__}: {_safe_diagnostic_text(exc, limit=200)}"
                )
            ]
            rejected.append(
                _rejected_proposal(
                    recommendation=recommendation,
                    proposal_index=proposal_index,
                    warnings=recommendation_warnings,
                    component_index=component_index,
                    normalized_requirements=normalized_requirements,
                    exception=exc,
                )
            )
            warnings.extend(recommendation_warnings)
            continue
        warnings.extend(recommendation_warnings)
        if validated is not None:
            validated["_llm_source_order"] = proposal_index
            valid.append(validated)
        else:
            rejected.append(
                _rejected_proposal(
                    recommendation=recommendation,
                    proposal_index=proposal_index,
                    warnings=recommendation_warnings,
                    component_index=component_index,
                    normalized_requirements=normalized_requirements,
                )
            )

    valid_count = len(valid)
    validation_rejected_count = len(rejected)
    selection_skipped: list[_RejectedProposal] = []

    representatives, duplicate_skips = _deduplicate_valid_recommendations(valid)
    selection_skipped.extend(duplicate_skips)
    warnings.extend(_rejected_warning(item) for item in duplicate_skips)

    if any(candidate.get("completeness_status") == "complete" for candidate in representatives):
        kept: list[dict[str, Any]] = []
        partial_skips: list[_RejectedProposal] = []
        for candidate in representatives:
            if candidate.get("completeness_status") != "incomplete":
                kept.append(candidate)
                continue
            candidate_id = str(
                candidate.get("recommendation_id") or candidate.get("candidate_id") or ""
            )
            partial_skips.append(
                _RejectedProposal(
                    recommendation_id=candidate_id,
                    category="selection_skipped_lower_ranked_alternative",
                    message=(
                        "partial_build selection skipped because a full safe "
                        "AI recommendation exists"
                    ),
                )
            )
        if partial_skips:
            selection_skipped.extend(partial_skips)
            warnings.extend(
                _rejected_warning(item)
                for item in partial_skips
            )
            representatives = kept

    if selection_eligible_recommendation_ids is not None:
        eligible: list[dict[str, Any]] = []
        ineligible_skips: list[_RejectedProposal] = []
        for candidate in representatives:
            candidate_id = str(
                candidate.get("recommendation_id") or candidate.get("candidate_id") or ""
            )
            if candidate_id in selection_eligible_recommendation_ids:
                eligible.append(candidate)
                continue
            ineligible_skips.append(
                _RejectedProposal(
                    recommendation_id=candidate_id,
                    category="selection_skipped_lower_ranked_alternative",
                    message="not selected before post-hoc relation evidence",
                )
            )
        if ineligible_skips:
            selection_skipped.extend(ineligible_skips)
            warnings.extend(_rejected_warning(item) for item in ineligible_skips)
            representatives = eligible

    selectable_representatives = [
        candidate for candidate in representatives if not candidate.get("_selection_skip_reason")
    ]
    selection_pool = selectable_representatives or representatives

    selected = _select_deterministic_safe_recommendations(selection_pool, limit=limit)
    if not selected and selection_pool:
        selected = [
            min(
                selection_pool,
                key=_price_optimal_sort_key,
            )
        ]
    selected_signatures = {_recommendation_identity(candidate) for candidate in selected}
    not_selected_skips = [
        _selection_skipped_proposal(candidate, selected)
        for candidate in representatives
        if _recommendation_identity(candidate) not in selected_signatures
    ]
    selection_skipped.extend(not_selected_skips)
    warnings.extend(_rejected_warning(item) for item in not_selected_skips)

    diagnostic_events = [*rejected, *selection_skipped]
    selection_skipped_count = len(selection_skipped)
    limited = selected
    for candidate in limited:
        candidate.pop("_llm_source_order", None)
        candidate.pop("_selection_skip_reason", None)
        candidate.pop("_selection_skip_message", None)
    grouped_presales = _build_grouped_presales_output(
        representatives,
        selected=limited,
        normalized_requirements=normalized_requirements,
    )
    total_proposal_count = (
        proposal_count if proposal_count is not None else len(recommendations)
    )
    summary = _validation_summary(
        accepted=len(limited),
        accepted_after_validation=valid_count,
        rejected=diagnostic_events,
    )
    return _ValidatedRecommendationPool(
        recommendations=limited,
        accepted_recommendations=representatives,
        configuration_groups=grouped_presales["configuration_groups"],
        quote_recommendation=grouped_presales["quote_recommendation"],
        selected_configuration_group_id=grouped_presales[
            "selected_configuration_group_id"
        ],
        selected_platform_option_id=grouped_presales["selected_platform_option_id"],
        selected_platform_option_index=grouped_presales[
            "selected_platform_option_index"
        ],
        warnings=_unique(warnings),
        proposal_count=total_proposal_count,
        valid_count=valid_count,
        validation_rejected_count=validation_rejected_count,
        selection_skipped_count=selection_skipped_count,
        rejected_count=max(0, total_proposal_count - len(limited)),
        validation_summary=summary,
        rejected_reasons_top=_rejected_reasons_top(diagnostic_events),
        rejected_debug_safe=_rejected_debug_safe(diagnostic_events),
    )


def _deduplicate_valid_recommendations(
    recommendations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[_RejectedProposal]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for recommendation in recommendations:
        groups.setdefault(_bom_signature(recommendation), []).append(recommendation)

    kept: list[dict[str, Any]] = []
    skipped: list[_RejectedProposal] = []
    for group in groups.values():
        representative = min(group, key=_duplicate_representative_sort_key)
        kept.append(representative)
        for recommendation in group:
            if recommendation is representative:
                continue
            skipped.append(
                _RejectedProposal(
                    recommendation_id=str(
                        recommendation.get("recommendation_id")
                        or recommendation.get("candidate_id")
                        or ""
                    ),
                    category="selection_skipped_duplicate",
                    message=(
                        "duplicate_same_core_bom_optional_peripherals selection skipped"
                        if _differs_only_by_optional_peripherals(
                            recommendation,
                            representative,
                        )
                        else "duplicate_same_bom selection skipped"
                    ),
                    proposal_index=_int_value(recommendation.get("_llm_source_order")),
                    debug_safe=_candidate_rejected_debug_safe(
                        recommendation,
                        category="selection_skipped_duplicate",
                        message=(
                            "duplicate_same_core_bom_optional_peripherals selection skipped"
                            if _differs_only_by_optional_peripherals(
                                recommendation,
                                representative,
                            )
                            else "duplicate_same_bom selection skipped"
                        ),
                    ),
                )
            )
    return kept, skipped


def _differs_only_by_optional_peripherals(
    candidate: Mapping[str, Any],
    representative: Mapping[str, Any],
) -> bool:
    candidate_roles = _role_identity_map(candidate)
    representative_roles = _role_identity_map(representative)
    all_roles = set(candidate_roles) | set(representative_roles)
    differing_roles = {
        role
        for role in all_roles
        if candidate_roles.get(role) != representative_roles.get(role)
    }
    return bool(differing_roles) and differing_roles.issubset(
        OPTIONAL_ENGINEER_CHECK_ROLES
    )


def _duplicate_representative_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if candidate.get("completeness_status") == "complete" else 1,
        *_price_sort_key(candidate.get("total_price_value"), 1),
        _technical_warning_count(candidate),
        _slot_priority(candidate.get("recommendation_slot")),
        _int_value(candidate.get("_llm_source_order")) or 0,
        _stable_text(candidate.get("recommendation_id") or candidate.get("candidate_id")),
    )


def _selection_skipped_proposal(
    candidate: Mapping[str, Any],
    selected: list[Mapping[str, Any]],
) -> _RejectedProposal:
    recommendation_id = str(
        candidate.get("recommendation_id") or candidate.get("candidate_id") or ""
    )
    category = _selection_skip_category(candidate, selected)
    message = str(candidate.get("_selection_skip_message") or "").strip()
    if not message:
        message = "not selected after deterministic safe top selection"
    return _RejectedProposal(
        recommendation_id=recommendation_id,
        category=category,
        message=message,
        proposal_index=_int_value(candidate.get("_llm_source_order")),
        debug_safe=_candidate_rejected_debug_safe(
            candidate,
            category=category,
            message=message,
        ),
    )


def _selection_skip_category(
    candidate: Mapping[str, Any],
    selected: list[Mapping[str, Any]],
) -> str:
    explicit_reason = str(candidate.get("_selection_skip_reason") or "").strip()
    if explicit_reason in {
        "dominated_by_cheaper_equivalent",
        "worse_by_price",
    }:
        return (
            "selection_skipped_dominated_by_cheaper_equivalent"
            if explicit_reason == "dominated_by_cheaper_equivalent"
            else "selection_skipped_worse_by_price"
        )
    if selected and not _meaningfully_different(candidate, selected):
        return "selection_skipped_same_platform_without_meaningful_difference"
    candidate_price = _decimal_value(candidate.get("total_price_value"))
    selected_prices = [
        price
        for item in selected
        if (price := _decimal_value(item.get("total_price_value"))) is not None
    ]
    if candidate_price is not None and selected_prices and candidate_price >= min(selected_prices):
        return "selection_skipped_worse_by_price"
    return "selection_skipped_lower_ranked_alternative"


def _select_deterministic_safe_recommendations(
    recommendations: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not recommendations:
        return []

    display_limit = max(1, min(limit, FINAL_SAFE_RECOMMENDATIONS_LIMIT))
    full = [
        recommendation
        for recommendation in recommendations
        if recommendation.get("completeness_status") == "complete"
    ]
    pool = full or recommendations
    selected: list[dict[str, Any]] = []

    price_candidate = min(pool, key=_price_optimal_sort_key)
    _append_selected_recommendation(
        selected,
        price_candidate,
        slot="price_optimal",
        title="Оптимальный по цене вариант",
    )
    if len(selected) >= display_limit:
        return selected

    technical_candidates = [
        candidate
        for candidate in pool
        if _recommendation_identity(candidate)
        not in {_recommendation_identity(item) for item in selected}
    ]
    if technical_candidates:
        technical_candidate = min(technical_candidates, key=_technical_clean_sort_key)
        _append_selected_recommendation(
            selected,
            technical_candidate,
            slot="technical_clean",
            title="Технически более чистый вариант",
        )
    if len(selected) >= display_limit:
        return selected

    selected_identities = {_recommendation_identity(item) for item in selected}
    alternative_candidates = [
        candidate
        for candidate in pool
        if _recommendation_identity(candidate) not in selected_identities
        and _meaningfully_different(candidate, selected)
    ]
    if alternative_candidates:
        alternative_candidate = min(
            alternative_candidates,
            key=lambda candidate: _alternative_sort_key(candidate, selected),
        )
        _append_selected_recommendation(
            selected,
            alternative_candidate,
            slot="alternative",
            title=_alternative_title(alternative_candidate, selected),
        )
    return selected[:display_limit]


def _has_complete_recommendation(recommendations: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        str(recommendation.get("completeness_status") or "") == "complete"
        for recommendation in recommendations
    )


def _primary_recommendation_from_selected(candidate: Mapping[str, Any]) -> dict[str, Any]:
    components = _mapping_rows(candidate.get("components"))
    product_group = str(candidate.get("product_group") or SERVER_PRODUCT_GROUP).strip()
    if product_group not in {NETWORK_PRODUCT_GROUP, STORAGE_PRODUCT_GROUP}:
        product_group = SERVER_PRODUCT_GROUP
    component_candidate_ids = _safe_component_id_map(
        candidate.get("component_candidate_ids")
    )
    if not component_candidate_ids and components:
        component_candidate_ids = {
            PROMPT_ROLE_BY_INTERNAL_ROLE.get(str(component.get("role") or ""), ""):
            str(component.get("component_candidate_id") or "").strip()
            for component in components
            if str(component.get("component_candidate_id") or "").strip()
        }
        component_candidate_ids = {
            role: component_id
            for role, component_id in component_candidate_ids.items()
            if role and component_id
        }
    engineering_confidence_code = str(
        candidate.get("engineering_confidence")
        or "preliminary_requires_engineer_review"
    ).strip()
    return {
        "candidate_type": candidate.get("candidate_type")
        or candidate.get("source_type")
        or BUILD_CANDIDATE_TYPE,
        "source_type": candidate.get("source_type")
        or candidate.get("candidate_type")
        or BUILD_CANDIDATE_TYPE,
        "product_group": product_group,
        "title": candidate.get("title") or "Рекомендуемый вариант для самого дешевого КП",
        "component_candidate_ids": component_candidate_ids,
        "why_selected": candidate.get("why_selected") or "",
        "requirement_fulfillment_summary": _mapping_rows(
            candidate.get("requirement_fulfillment_summary")
        ),
        "assumptions": _string_list(candidate.get("assumptions")),
        "engineer_checks": sanitize_engineer_checks_for_product_group(
            [
                *_string_list(candidate.get("engineer_checks")),
                *_string_list(candidate.get("critical_checks")),
                *_string_list(candidate.get("compatibility_warnings")),
            ],
            product_group=product_group,
        ),
        "components": _jsonable(components),
        "platform": _jsonable(candidate.get("platform"))
        if isinstance(candidate.get("platform"), Mapping)
        else {},
        "quantities": _jsonable(candidate.get("quantities")),
        "total_price_value": candidate.get("total_price_value"),
        "total_price_currency": candidate.get("total_price_currency"),
        "total_price_note": candidate.get("total_price_note") or candidate.get("price_note"),
        "hard_capability_validation": _jsonable(
            candidate.get("hard_capability_validation")
        ),
        "missing_required_capabilities": _mapping_rows(
            candidate.get("missing_required_capabilities")
        ),
        "completeness_status": candidate.get("completeness_status"),
        "engineer_review_required": True,
        "engineering_confidence_code": engineering_confidence_code,
        "engineering_confidence": human_engineering_confidence_label(
            engineering_confidence_code
        ),
        "commercial_fit_confidence": candidate.get("commercial_fit_confidence")
        or candidate.get("confidence"),
    }


def _commercial_summary_from_primary(primary: Mapping[str, Any]) -> dict[str, Any]:
    if not primary:
        return {}
    return build_primary_commercial_summary({}, primary, match_run_id=None) or {}


def _commercial_summary_from_no_recommendation(
    reason: Mapping[str, Any],
) -> dict[str, Any]:
    missing_roles = _string_list(reason.get("missing_roles"))
    missing_required_capabilities = _mapping_rows(
        reason.get("missing_required_capabilities")
    )
    stock_shortage_lines = [
        (
            f"{row.get('role') or 'component'}: нужно "
            f"{row.get('required_quantity')}, склад {row.get('available_quantity')}"
        )
        for row in _mapping_rows(reason.get("stock_shortages"))
    ]
    hard_incompatibility = [
        sanitize_user_facing_text(item)
        for item in _string_list(reason.get("hard_incompatibility"))
    ]
    manual_checks = [
        sanitize_user_facing_text(item) for item in _string_list(reason.get("manual_checks"))
    ]
    engineering_checks = [
        sanitize_user_facing_text(item)
        for item in _string_list(reason.get("engineering_checks"))
    ]
    missing_lines = _no_recommendation_missing_lines(missing_required_capabilities)
    clean_manual_checks = _unique(
        [item for item in [*manual_checks, *engineering_checks] if item]
    )
    lines = [
        "Безопасную складскую рекомендацию дать нельзя.",
        *missing_lines,
    ]
    if clean_manual_checks:
        lines.extend(["", "Проверить вручную:", *[f"- {item}" for item in clean_manual_checks]])
    return {
        "mode": OUTPUT_MODE_SINGLE_BEST_COST_VALID,
        "status": "no_recommendation",
        "title": "Безопасную складскую рекомендацию дать нельзя.",
        "summary": reason.get("summary")
        or "Безопасную складскую рекомендацию дать нельзя.",
        "missing_roles": missing_roles,
        "missing_required_capabilities": missing_required_capabilities,
        "stock_shortage_lines": stock_shortage_lines,
        "hard_incompatibility": [item for item in hard_incompatibility if item],
        "manual_checks": clean_manual_checks,
        "lines": lines,
        "copy_paste_text": "\n".join(lines),
    }


def _no_recommendation_missing_lines(
    missing_required_capabilities: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for capability in missing_required_capabilities:
        source_text = sanitize_user_facing_text(
            capability.get("source_text")
            or capability.get("requirement_text")
            or capability.get("capability_id")
            or capability.get("role")
        )
        reason = sanitize_user_facing_text(
            capability.get("user_message")
            or capability.get("reason")
            or capability.get("status")
        )
        if source_text:
            lines.append(f"Не закрыто требование: {source_text}.")
        if reason:
            lines.append(f"Причина: {reason}")
    return lines


def _primary_server_quantity(primary: Mapping[str, Any]) -> int | None:
    components = _mapping_rows(primary.get("components"))
    platform = _component_by_role(components, SERVER_PLATFORM_ROLE)
    if platform is not None:
        quantity = _int_value(platform.get("quantity_required"))
        if quantity is not None:
            return quantity
    for component in components:
        quantity = _int_value(component.get("server_quantity"))
        if quantity is not None:
            return quantity
    return None


def _no_recommendation_reason(
    *,
    fallback_reason: str,
    validated_pool: _ValidatedRecommendationPool,
    warnings: Sequence[str],
    product_group: str = "server",
    package_required_capabilities: Sequence[Mapping[str, Any]] = (),
    package_missing_required_capabilities: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    debug_rows = _mapping_rows(validated_pool.rejected_debug_safe)
    missing_roles = _unique(
        [
            *[
                str(capability.get("role") or "").strip()
                for capability in package_missing_required_capabilities
                if str(capability.get("role") or "").strip()
            ],
            *[
                role
                for row in debug_rows
                for role in _string_list(row.get("missing_roles"))
            ],
            *[
                role
                for recommendation in validated_pool.recommendations
                for role in _string_list(recommendation.get("missing_component_roles"))
            ],
        ]
    )
    stock_shortages = [
        {
            "role": shortage.get("role"),
            "required_quantity": shortage.get("required_quantity"),
            "available_quantity": shortage.get("available_quantity"),
        }
        for row in debug_rows
        for shortage in _mapping_rows(row.get("stock_shortages"))
    ]
    missing_required_capabilities = [
        dict(capability)
        for capability in package_missing_required_capabilities
    ]
    missing_required_capabilities.extend(
        dict(capability)
        for row in [
            *debug_rows,
            *validated_pool.recommendations,
        ]
        for capability in _mapping_rows(row.get("missing_required_capabilities"))
    )
    roles_with_specific_capability_rows = {
        str(row.get("role") or "").strip()
        for row in missing_required_capabilities
        if str(row.get("role") or "").strip()
    }
    missing_required_capabilities.extend(
        _missing_capability_rows_for_roles(
            [
                role
                for role in missing_roles
                if role not in roles_with_specific_capability_rows
            ],
            required_capabilities=package_required_capabilities,
        )
    )
    missing_required_capabilities = _unique_capability_rows(
        missing_required_capabilities
    )
    unverified_platform_requirements = [
        dict(row)
        for row in missing_required_capabilities
        if str(row.get("requirement_classification") or "").strip()
        == REQ_CLASS_PRIMARY_OBJECT_FEATURE
        and str(row.get("status") or "").strip()
        in {"unverified_hard_requirement", "not_applicable", "missing_component"}
    ]
    unmet_platform_feature_requirements = [
        dict(row)
        for row in missing_required_capabilities
        if str(row.get("requirement_classification") or "").strip()
        == REQ_CLASS_PRIMARY_OBJECT_FEATURE
        and str(row.get("status") or "").strip() == "hard_mismatch"
    ]
    engineering_checks = _unique(
        [
            *[
                str(row.get("suggested_engineer_check_ru") or "").strip()
                for row in missing_required_capabilities
                if str(row.get("suggested_engineer_check_ru") or "").strip()
            ],
            *[
                check
                for recommendation in validated_pool.recommendations
                for check in _string_list(recommendation.get("engineer_checks"))
            ],
        ]
    )
    hard_incompatibility = [
        str(row.get("rejection_message_ru") or row.get("rejection_code") or "").strip()
        for row in debug_rows
        if str(row.get("rejection_code") or "").strip()
        in {
            "fatal",
            "platform_cpu_mismatch",
            "role_mismatch",
            "optional_core_conflict",
        }
    ]
    return {
        "summary": "Безопасную складскую рекомендацию дать нельзя.",
        "fallback_reason": fallback_reason,
        "product_group": product_group,
        "missing_roles": missing_roles,
        "missing_required_capabilities": missing_required_capabilities,
        "unverified_platform_requirements": unverified_platform_requirements,
        "unmet_platform_feature_requirements": unmet_platform_feature_requirements,
        "engineering_checks": engineering_checks,
        "stock_shortages": stock_shortages,
        "hard_incompatibility": _unique(hard_incompatibility),
        "manual_checks": _no_recommendation_manual_checks(product_group),
        "validation_summary": _jsonable(validated_pool.validation_summary),
        "diagnostic_notes": [
            _safe_diagnostic_text(warning, limit=160) for warning in list(warnings)[:5]
        ],
    }


def _with_no_recommendation_coverage(
    reason: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
    no_recommendation: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = _no_recommendation_coverage(
        package=package,
        no_recommendation=no_recommendation,
        thresholds=thresholds,
    )
    if not coverage:
        return dict(reason)
    result = dict(reason)
    result["no_recommendation_coverage"] = coverage
    if coverage.get("coverage_incomplete"):
        diagnostic_notes = _string_list(result.get("diagnostic_notes"))
        diagnostic_notes.append(
            "structured no_recommendation did not demonstrate full role-matrix coverage"
        )
        result["diagnostic_notes"] = _unique(diagnostic_notes)
    return result


def _multi_pass_contract_roles_for_coverage(package: Mapping[str, Any]) -> set[str]:
    if not isinstance(package.get("requirement_contract"), Mapping):
        return set()
    roles = {
        str(role).strip()
        for role in _string_list(package.get("required_roles"))
        if str(role).strip()
    }
    contract = _safe_mapping(package.get("requirement_contract"))
    for role in _string_list(contract.get("required_roles")):
        normalized = _normalize_contract_role(role, package=package)
        if normalized is not None:
            roles.add(normalized)
    return roles


def _no_recommendation_coverage(
    *,
    package: Mapping[str, Any],
    no_recommendation: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = _normalize_no_recommendation_coverage_thresholds(thresholds)
    matrix_counts = _safe_int_mapping(package.get("broad_count_by_role"))
    if not matrix_counts:
        matrix_counts = _package_count_by_role(package)
    if not matrix_counts:
        matrix_counts = _safe_int_mapping(package.get("evaluated_candidate_count_by_role"))
    if not matrix_counts:
        return {}
    contract_roles = _multi_pass_contract_roles_for_coverage(package)
    if contract_roles:
        matrix_counts = {
            role: count
            for role, count in matrix_counts.items()
            if role in contract_roles
        }
        if not matrix_counts:
            return {}
    structured_considered_counts = _considered_counts_from_no_recommendation(
        no_recommendation or {}
    )
    considered_counts = dict(structured_considered_counts)
    full_matrix_complete = _full_matrix_evaluation_covers_matrix(
        package,
        matrix_counts=matrix_counts,
    )
    if full_matrix_complete:
        evaluated_counts = _safe_int_mapping(package.get("evaluated_candidate_count_by_role"))
        considered_counts = {
            role: max(
                int(considered_counts.get(role, 0) or 0),
                int(evaluated_counts.get(role, 0) or 0),
            )
            for role in matrix_counts
        }
    if not considered_counts:
        considered_counts = {
            role: len(_role_rows_from_package_matrix(package, role))
            for role in matrix_counts
        }
    coverage_percent_by_role: dict[str, float] = {}
    required_count_by_role: dict[str, int] = {}
    gate_passed_by_role: dict[str, bool] = {}
    policy_by_role: dict[str, str] = {}
    incomplete_roles: list[str] = []
    for role, matrix_count in matrix_counts.items():
        considered = int(considered_counts.get(role, 0) or 0)
        total = int(matrix_count or 0)
        coverage_percent_by_role[role] = round(
            (considered / total * 100) if total else 100.0,
            2,
        )
        required_count = _no_recommendation_required_coverage_count(
            total,
            thresholds=policy,
        )
        required_count_by_role[role] = required_count
        gate_passed = considered >= required_count
        gate_passed_by_role[role] = gate_passed
        policy_by_role[role] = (
            "all_candidates"
            if total <= int(policy["full_coverage_limit"])
            else "min_candidates_or_fraction"
        )
        if total > 0 and not gate_passed:
            incomplete_roles.append(role)
    return {
        "considered_count_by_role": considered_counts,
        "structured_considered_count_by_role": structured_considered_counts,
        "matrix_count_by_role": matrix_counts,
        "coverage_percent_by_role": coverage_percent_by_role,
        "required_count_by_role": required_count_by_role,
        "gate_passed_by_role": gate_passed_by_role,
        "coverage_policy_by_role": policy_by_role,
        "thresholds": policy,
        "coverage_incomplete": bool(incomplete_roles),
        "incomplete_roles": incomplete_roles,
        "full_matrix_evaluation_used": bool(package.get("full_matrix_evaluation_used")),
        "full_matrix_evaluation_complete": full_matrix_complete,
        "next_action": (
            "rerun with full-matrix evaluation or larger budget"
            if incomplete_roles
            else None
        ),
    }


def _no_recommendation_coverage_thresholds(
    settings: LlmSettings | None,
) -> dict[str, Any]:
    if settings is None:
        return _normalize_no_recommendation_coverage_thresholds(None)
    return _normalize_no_recommendation_coverage_thresholds(
        {
            "full_coverage_limit": getattr(
                settings,
                "llm_configurator_no_recommendation_full_coverage_limit",
                DEFAULT_NO_RECOMMENDATION_FULL_COVERAGE_LIMIT,
            ),
            "min_large_role_candidates": getattr(
                settings,
                "llm_configurator_no_recommendation_min_large_role_candidates",
                DEFAULT_NO_RECOMMENDATION_MIN_LARGE_ROLE_CANDIDATES,
            ),
            "min_large_role_fraction": getattr(
                settings,
                "llm_configurator_no_recommendation_min_large_role_fraction",
                DEFAULT_NO_RECOMMENDATION_MIN_LARGE_ROLE_FRACTION,
            ),
        }
    )


def _normalize_no_recommendation_coverage_thresholds(
    thresholds: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw = _safe_mapping(thresholds)
    full_coverage_limit = _int_value(raw.get("full_coverage_limit"))
    min_large_role_candidates = _int_value(raw.get("min_large_role_candidates"))
    min_large_role_fraction = _decimal_value(raw.get("min_large_role_fraction"))
    return {
        "full_coverage_limit": max(
            1,
            full_coverage_limit or DEFAULT_NO_RECOMMENDATION_FULL_COVERAGE_LIMIT,
        ),
        "min_large_role_candidates": max(
            1,
            min_large_role_candidates
            or DEFAULT_NO_RECOMMENDATION_MIN_LARGE_ROLE_CANDIDATES,
        ),
        "min_large_role_fraction": float(
            min(
                Decimal("1"),
                max(
                    Decimal("0"),
                    min_large_role_fraction
                    if min_large_role_fraction is not None
                    else Decimal(str(DEFAULT_NO_RECOMMENDATION_MIN_LARGE_ROLE_FRACTION)),
                ),
            )
        ),
    }


def _no_recommendation_required_coverage_count(
    total: int,
    *,
    thresholds: Mapping[str, Any],
) -> int:
    if total <= 0:
        return 0
    full_coverage_limit = int(
        thresholds.get("full_coverage_limit")
        or DEFAULT_NO_RECOMMENDATION_FULL_COVERAGE_LIMIT
    )
    if total <= full_coverage_limit:
        return total
    min_candidates = int(
        thresholds.get("min_large_role_candidates")
        or DEFAULT_NO_RECOMMENDATION_MIN_LARGE_ROLE_CANDIDATES
    )
    fraction = float(
        thresholds.get("min_large_role_fraction")
        or DEFAULT_NO_RECOMMENDATION_MIN_LARGE_ROLE_FRACTION
    )
    fraction_count = ceil(total * max(0.0, min(1.0, fraction)))
    return max(1, min(total, min(min_candidates, fraction_count)))


def _full_matrix_evaluation_covers_matrix(
    package: Mapping[str, Any],
    *,
    matrix_counts: Mapping[str, int],
) -> bool:
    if not bool(package.get("full_matrix_evaluation_used")):
        return False
    if _mapping_rows(package.get(FULL_MATRIX_FAILED_CHUNKS_KEY)):
        return False
    evaluated_counts = _safe_int_mapping(package.get("evaluated_candidate_count_by_role"))
    if not evaluated_counts:
        return False
    return all(
        int(evaluated_counts.get(role, 0) or 0) >= int(total or 0)
        for role, total in matrix_counts.items()
        if int(total or 0) > 0
    )


def _incomplete_matrix_coverage_reason(
    reason: Mapping[str, Any],
    *,
    coverage: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    repair_reason: str | None,
    repair_attempted: bool,
    repair_success: bool,
) -> dict[str, Any]:
    diagnostic_notes = _string_list(reason.get("diagnostic_notes"))
    diagnostic_notes.append(
        "structured no_recommendation was rejected because role-matrix coverage "
        "was below the configured threshold"
    )
    coverage_payload = {
        **_safe_mapping(coverage),
        "coverage_incomplete": True,
        "next_action": "rerun with full-matrix evaluation or larger budget",
    }
    return {
        **dict(reason),
        "summary": "AI не смог надежно оценить полную матрицу кандидатов.",
        "fallback_reason": INCOMPLETE_MATRIX_COVERAGE_FALLBACK_REASON,
        "structured_no_recommendation": False,
        "coverage_rejected": True,
        "no_recommendation_coverage": coverage_payload,
        "no_recommendation_coverage_gate_passed": False,
        "no_recommendation_coverage_repair_attempted": bool(repair_attempted),
        "no_recommendation_coverage_repair_success": bool(repair_success),
        "no_recommendation_coverage_rejected": True,
        "no_recommendation_coverage_thresholds": _jsonable(thresholds),
        "no_recommendation_coverage_repair_reason": (
            repair_reason or "coverage_incomplete"
        ),
        "diagnostic_notes": _unique(diagnostic_notes),
    }


def _considered_counts_from_no_recommendation(
    no_recommendation: Mapping[str, Any],
) -> dict[str, int]:
    result: dict[str, int] = {}
    considered = no_recommendation.get("considered_candidate_ids")
    if isinstance(considered, Mapping):
        for role, value in considered.items():
            ids = _string_list(value)
            if ids:
                result[_coverage_role_key(str(role))] = len(set(ids))
    for row in _mapping_rows(no_recommendation.get("role_analysis")):
        role = _coverage_role_key(str(row.get("role") or ""))
        if not role:
            continue
        ids = _string_list(row.get("considered_candidate_ids"))
        if ids:
            result[role] = max(result.get(role, 0), len(set(ids)))
        count = _int_value(row.get("considered_count") or row.get("considered_count_total"))
        if count is not None:
            result[role] = max(result.get(role, 0), count)
    return result


def _role_rows_from_package_matrix(
    package: Mapping[str, Any],
    role: str,
) -> list[Mapping[str, Any]]:
    matrix = _safe_mapping(package.get("component_candidate_matrix"))
    aliases = {role, "platform" if role == SERVER_PLATFORM_ROLE else role}
    rows: list[Mapping[str, Any]] = []
    for alias in aliases:
        rows.extend(_mapping_rows(matrix.get(alias)))
    return rows


def _coverage_role_key(role: str) -> str:
    return SERVER_PLATFORM_ROLE if role == "platform" else str(role or "").strip()


def _safe_int_mapping(value: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    if not isinstance(value, Mapping):
        return result
    for key, raw in value.items():
        count = _int_value(raw)
        if count is not None:
            result[_coverage_role_key(str(key))] = count
    return result


def _no_recommendation_manual_checks(product_group: str) -> list[str]:
    normalized = str(product_group or "server").strip().casefold()
    if normalized == "network":
        return [
            "проверить количество и тип портов",
            "проверить PoE budget и PoE standard",
            "проверить uplink SFP+/SFP28/QSFP modules или DAC, если они требуются",
            "проверить L2/L3 feature set",
            "проверить stacking compatibility, лицензии и кабели",
            "проверить поддержку, гарантию и срок поставки",
        ]
    if normalized == "storage":
        return [
            "проверить usable/raw capacity и модель RAID/избыточности",
            "проверить количество и резервирование контроллеров",
            "проверить тип, интерфейс и количество дисков",
            "проверить FC/iSCSI/NVMe-oF/SAS порты",
            "проверить лицензии, поддержку и гарантию",
            "проверить совместимость полок и кабелей",
        ]
    return [
        "проверить CPU support list / BIOS",
        "проверить QVL RAM и правила заполнения DIMM",
        "проверить NVMe/U.2/U.3 backplane",
        "проверить комплектацию БП, кулеров, реек и кабелей",
        "проверить гарантию и срок поставки",
    ]


def _unique_capability_rows(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for value in values:
        row = dict(value)
        key = (
            str(row.get("capability_id") or ""),
            str(row.get("role") or ""),
            str(row.get("status") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _missing_capability_rows_for_roles(
    missing_roles: Sequence[str],
    *,
    required_capabilities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean_roles = [str(role or "").strip() for role in missing_roles if str(role or "").strip()]
    for role in clean_roles:
        matched = False
        for capability in required_capabilities:
            if not capability.get("hard", True):
                continue
            if not _capability_matches_missing_role(capability, role):
                continue
            rows.append(_missing_component_capability_row(capability, role=role))
            matched = True
        if not matched:
            rows.append(
                {
                    "capability_id": f"{role}.required",
                    "role": role,
                    "status": "missing_component",
                    "satisfied_by": None,
                    "component_role": None,
                    "component_candidate_id": None,
                    "source_text": _role_label(role),
                    "reason": "Required BOM role was not selected by Composer.",
                    "user_message": (
                        "No selected component or platform feature closed this hard role."
                    ),
                }
            )
    return rows


def _capability_matches_missing_role(
    capability: Mapping[str, Any],
    role: str,
) -> bool:
    capability_role = str(capability.get("role") or "").strip()
    if capability_role == role:
        return True
    return capability_role == "storage" and role in {
        STORAGE_SYSTEM_ROLE,
        DRIVE_ROLE,
        SSD_ROLE,
        HDD_ROLE,
    }


def _missing_component_capability_row(
    capability: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    source_text = (
        capability.get("source_text")
        or capability.get("requirement_text")
        or capability.get("capability_id")
        or role
    )
    return {
        "capability_id": capability.get("capability_id") or f"{role}.required",
        "role": role,
        "status": "missing_component",
        "satisfied_by": None,
        "component_role": None,
        "component_candidate_id": None,
        "source_text": source_text,
        "reason": _missing_component_reason(role),
        "user_message": _missing_component_user_message(role),
    }


def _missing_component_reason(role: str) -> str:
    if role == NETWORK_ADAPTER_ROLE:
        return (
            "Selected platform does not satisfy the hard network requirement and "
            "Composer did not select a suitable network adapter."
        )
    if role == POWER_SUPPLY_ROLE:
        return (
            "Selected platform does not prove bundled PSU redundancy and Composer did "
            "not select a suitable power supply component."
        )
    return "Composer did not select a component or platform feature for this hard capability."


def _missing_component_user_message(role: str) -> str:
    if role == NETWORK_ADAPTER_ROLE:
        return (
            "Выбранная платформа не имеет onboard сети под запрошенные "
            "speed/media/ports, и подходящий сетевой адаптер не выбран."
        )
    if role == POWER_SUPPLY_ROLE:
        return (
            "Selected platform does not show bundled PSU redundancy, and no separate "
            "power supply component was selected."
        )
    return "No selected component or platform feature closed this hard requirement."


def _append_selected_recommendation(
    selected: list[dict[str, Any]],
    candidate: Mapping[str, Any],
    *,
    slot: str,
    title: str,
) -> None:
    row = dict(candidate)
    row["llm_recommendation_slot"] = row.get("recommendation_slot")
    row["recommendation_slot"] = slot
    row["title"] = title
    selected.append(row)


def _build_grouped_presales_output(
    validated_recommendations: list[dict[str, Any]],
    *,
    selected: list[dict[str, Any]],
    normalized_requirements: Any,
) -> dict[str, Any]:
    candidates = _grouping_visible_candidates(validated_recommendations)
    if not candidates:
        return {
            "configuration_groups": [],
            "quote_recommendation": {},
            "selected_configuration_group_id": None,
            "selected_platform_option_id": None,
            "selected_platform_option_index": None,
        }

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        signature = _component_base_signature(candidate)
        if not signature:
            continue
        grouped.setdefault(signature, []).append(candidate)
    if not grouped:
        return {
            "configuration_groups": [],
            "quote_recommendation": {},
            "selected_configuration_group_id": None,
            "selected_platform_option_id": None,
            "selected_platform_option_index": None,
        }

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            *_price_sort_key(_minimum_group_price(item[1]), 1),
            _stable_text(_family_label_from_candidate(item[1][0])),
        ),
    )
    selected_identities = {_recommendation_identity(candidate) for candidate in selected}
    groups: list[dict[str, Any]] = []
    group_contexts: list[dict[str, Any]] = []
    option_refs: list[dict[str, Any]] = []
    global_option_index = 1
    requirements = _first_requirements(normalized_requirements)

    for group_index, (signature, group_candidates) in enumerate(ordered_groups, start=1):
        group_id = f"cfg_group_{group_index}"
        ordered_options = sorted(group_candidates, key=_platform_option_sort_key)
        group_minimum_price = _minimum_group_price(ordered_options)
        group_cheapest = min(ordered_options, key=_price_optimal_sort_key)
        group_database = _database_preferred_candidate(ordered_options)
        family_label = _family_label_from_candidate(ordered_options[0])
        component_base = _component_base_from_candidate(ordered_options[0])
        platform_options: list[dict[str, Any]] = []

        for option_index, candidate in enumerate(ordered_options, start=1):
            option_id = f"platform_option_{group_index}_{option_index}"
            option_role = _platform_option_role(
                candidate,
                cheapest=group_cheapest,
                database_preferred=group_database,
            )
            option = _platform_option_from_candidate(
                candidate,
                option_id=option_id,
                option_index=global_option_index,
                option_role=option_role,
                selected_by_legacy_top=_recommendation_identity(candidate)
                in selected_identities,
            )
            platform_options.append(option)
            option_refs.append(
                {
                    "group_id": group_id,
                    "option_id": option_id,
                    "option_index": global_option_index,
                    "candidate": candidate,
                    "option": option,
                }
            )
            global_option_index += 1

        recommended_option = min(platform_options, key=_platform_option_price_sort_key)
        group = {
            "group_id": group_id,
            "group_title": family_label,
            "family_signature": _family_signature_label(
                signature,
                ordered_options[0],
            ),
            "architecture_summary": _safe_presales_text(
                _architecture_summary(ordered_options[0])
            ),
            "component_base": component_base,
            "component_base_notes": _component_base_notes(component_base),
            "platform_options": platform_options,
            "recommended_option_id": recommended_option["option_id"],
            "why_group_matters": _safe_presales_text(
                _why_group_matters(ordered_options[0])
            ),
        }
        groups.append(group)
        group_contexts.append(
            {
                "group": group,
                "family_label": family_label,
                "minimum_price": group_minimum_price,
                "component_base": component_base,
            }
        )

    _apply_group_titles(group_contexts, requirements=requirements)
    quote_recommendation = _quote_recommendation(groups, option_refs)
    return {
        "configuration_groups": groups,
        "quote_recommendation": quote_recommendation,
        "selected_configuration_group_id": quote_recommendation.get(
            "recommended_group_id"
        ),
        "selected_platform_option_id": quote_recommendation.get(
            "recommended_option_id"
        ),
        "selected_platform_option_index": quote_recommendation.get(
            "recommended_option_index"
        ),
    }


def _grouping_visible_candidates(
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [
        candidate
        for candidate in recommendations
        if str(candidate.get("source_type") or candidate.get("candidate_type") or "")
        != READY_SERVER_CANDIDATE_TYPE
    ]
    by_platform_cpu_ram: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_platform_cpu_ram.setdefault(
            _platform_cpu_ram_signature(candidate),
            [],
        ).append(candidate)

    visible: list[dict[str, Any]] = []
    for group in by_platform_cpu_ram.values():
        if len(group) == 1:
            visible.extend(group)
            continue
        explicit_tradeoffs = [
            candidate for candidate in group if _has_explicit_component_tradeoff(candidate)
        ]
        ordinary = [candidate for candidate in group if candidate not in explicit_tradeoffs]
        if ordinary:
            visible.append(min(ordinary, key=_price_optimal_sort_key))
        visible.extend(explicit_tradeoffs)
    return sorted(visible, key=_platform_option_sort_key)


def _platform_cpu_ram_signature(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    role_map = _role_identity_map(candidate)
    quantity_map = _role_quantity_map(candidate)
    return tuple(
        (role, role_map.get(role, ""), quantity_map.get(role, 0))
        for role in (SERVER_PLATFORM_ROLE, CPU_ROLE, RAM_ROLE)
    )


def _component_base_signature(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    role_map = _role_identity_map(candidate)
    quantity_map = _role_quantity_map(candidate)
    storage_roles = [
        (role, role_map.get(role, ""), quantity_map.get(role, 0))
        for role in (SSD_ROLE, HDD_ROLE)
        if role in role_map
    ]
    signature = [
        (role, role_map.get(role, ""), quantity_map.get(role, 0))
        for role in (CPU_ROLE, RAM_ROLE)
        if role in role_map
    ]
    if storage_roles:
        signature.append(("storage", tuple(sorted(storage_roles))))
    return tuple(signature)


def _has_explicit_component_tradeoff(candidate: Mapping[str, Any]) -> bool:
    role = str(candidate.get("proposal_role") or "").strip()
    slot = str(candidate.get("recommendation_slot") or "").strip()
    if role == "explicit_tradeoff" or slot == "lower_price_with_tradeoff":
        return True
    text = " ".join(
        str(candidate.get(key) or "")
        for key in (
            "commercial_tradeoff",
            "right_size_note",
            "why_selected",
            "why_selected_short",
            "title",
        )
    ).casefold()
    tradeoff_markers = (
        "tradeoff",
        "компромисс",
        "альтернатив",
        "меньше dimm",
        "dimm",
        "backplane",
        "psu",
        "tdp",
        "headroom",
        "слот",
        "корзин",
    )
    return any(marker in text for marker in tradeoff_markers)


def _platform_option_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    platform = _component_by_role(
        _mapping_rows(candidate.get("components")),
        SERVER_PLATFORM_ROLE,
    )
    return (
        *_price_sort_key(candidate.get("total_price_value"), 1),
        -_technical_confirmation_score(candidate),
        _stable_text(_component_article_text(platform)),
        _stable_text(candidate.get("recommendation_id") or candidate.get("candidate_id")),
    )


def _minimum_group_price(candidates: list[Mapping[str, Any]]) -> Decimal | None:
    prices = [
        price
        for candidate in candidates
        if (price := _decimal_value(candidate.get("total_price_value"))) is not None
    ]
    return min(prices) if prices else None


def _apply_group_titles(
    group_contexts: list[dict[str, Any]],
    *,
    requirements: Mapping[str, Any],
) -> None:
    prices = [
        price
        for context in group_contexts
        if (price := context.get("minimum_price")) is not None
    ]
    global_minimum_price = min(prices) if prices else None
    contexts_by_family: dict[str, list[dict[str, Any]]] = {}
    for context in group_contexts:
        contexts_by_family.setdefault(str(context.get("family_label") or ""), []).append(
            context
        )

    for family_contexts in contexts_by_family.values():
        family_prices = [
            price
            for context in family_contexts
            if (price := context.get("minimum_price")) is not None
        ]
        family_minimum_price = min(family_prices) if family_prices else None
        family_base_context = next(
            (
                context
                for context in family_contexts
                if _same_decimal(context.get("minimum_price"), family_minimum_price)
            ),
            family_contexts[0],
        )
        family_base_ram_module_gb = _ram_module_capacity_from_base(
            family_base_context.get("component_base")
        )
        ram_variants = {
            module_gb
            for context in family_contexts
            if (
                module_gb := _ram_module_capacity_from_base(
                    context.get("component_base")
                )
            )
            is not None
        }
        storage_capacities = [
            capacity
            for context in family_contexts
            if (
                capacity := _storage_capacity_from_base(
                    context.get("component_base")
                )
            )
            is not None
        ]
        family_minimum_storage = min(storage_capacities) if storage_capacities else None
        for context in family_contexts:
            group = context.get("group")
            if not isinstance(group, dict):
                continue
            family_label = sanitize_user_facing_text(context.get("family_label")) or (
                "архитектурная семья требует проверки"
            )
            suffixes: list[str] = []
            if _same_decimal(context.get("minimum_price"), global_minimum_price):
                suffixes.append("минимальная базовая конфигурация")

            ram_module_gb = _ram_module_capacity_from_base(context.get("component_base"))
            if ram_module_gb is not None and len(ram_variants) > 1:
                if ram_module_gb == family_base_ram_module_gb:
                    suffixes.append(f"база на {ram_module_gb} ГБ DIMM")
                else:
                    suffixes.append(f"вариант с {ram_module_gb} ГБ DIMM")
                    suffixes.append("компромисс по RAM/DIMM-слотам")

            storage_capacity = _storage_capacity_from_base(context.get("component_base"))
            if _storage_above_requirement_or_family_base(
                storage_capacity,
                family_minimum_storage=family_minimum_storage,
                requirements=requirements,
            ):
                suffixes.append(
                    f"вариант с SSD {_format_decimal_number(storage_capacity)} ТБ"
                )
                suffixes.append("компромисс по емкости накопителей")

            suffix_text = ", ".join(_unique(suffixes))
            group["group_title"] = (
                f"{family_label} - {suffix_text}" if suffix_text else family_label
            )


def _ram_module_capacity_from_base(component_base: Any) -> int | None:
    if not isinstance(component_base, Mapping):
        return None
    ram = component_base.get("ram")
    if not isinstance(ram, Mapping):
        return None
    return _int_value(ram.get("ram_module_capacity_gb"))


def _storage_capacity_from_base(component_base: Any) -> float | None:
    if not isinstance(component_base, Mapping):
        return None
    storage = component_base.get("storage")
    if not isinstance(storage, Mapping):
        return None
    return _float_value(storage.get("storage_capacity_tb"))


def _storage_above_requirement_or_family_base(
    storage_capacity: float | None,
    *,
    family_minimum_storage: float | None,
    requirements: Mapping[str, Any],
) -> bool:
    if storage_capacity is None:
        return False
    required_storage = _required_storage_tb(requirements)
    baseline = required_storage if required_storage is not None else family_minimum_storage
    if baseline is None:
        return False
    return storage_capacity > baseline + 0.001


def _same_decimal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return _decimal_value(left) == _decimal_value(right)


def _component_base_from_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    components = _mapping_rows(candidate.get("components"))
    base: dict[str, Any] = {}
    for key, role in (
        ("cpu", CPU_ROLE),
        ("ram", RAM_ROLE),
        ("storage", SSD_ROLE),
    ):
        component = _component_by_role(components, role)
        if component is None and role == SSD_ROLE:
            component = _component_by_role(components, HDD_ROLE)
        if component is not None:
            base[key] = _public_component_row(component)
    return base


def _platform_option_from_candidate(
    candidate: Mapping[str, Any],
    *,
    option_id: str,
    option_index: int,
    option_role: str,
    selected_by_legacy_top: bool,
) -> dict[str, Any]:
    components = _mapping_rows(candidate.get("components"))
    product_group = str(candidate.get("product_group") or SERVER_PRODUCT_GROUP).strip()
    platform = _component_by_role(components, SERVER_PLATFORM_ROLE)
    optional_components = [
        _public_component_row(component)
        for component in _mapping_rows(candidate.get("optional_components"))
    ]
    checks = _safe_engineer_checks(
        [
            *(_string_list(candidate.get("critical_checks"))),
            *(_string_list(candidate.get("engineer_checks"))),
            *(_string_list(candidate.get("compatibility_warnings"))),
        ],
        product_group=product_group,
    )
    tradeoffs = _safe_tradeoffs(candidate)
    why = (
        _safe_presales_text(candidate.get("why_selected_short"))
        or _safe_presales_text(candidate.get("why_selected"))
        or "Выбран по сочетанию цены, наличия и предварительного соответствия запросу."
    )
    engineering_confidence_code = str(
        candidate.get("engineering_confidence")
        or "preliminary_requires_engineer_review"
    ).strip()
    return {
        "option_id": option_id,
        "option_index": option_index,
        "role": option_role,
        "option_role": option_role,
        "platform": _public_component_row(platform) if platform is not None else {},
        "total_price_value": _json_decimal(_decimal_value(candidate.get("total_price_value"))),
        "total_price_currency": str(candidate.get("total_price_currency") or "").strip(),
        "stock_status": _platform_option_stock_status(candidate),
        "available_stock": _int_value(candidate.get("available_quantity")),
        "why_this_platform": why,
        "tradeoffs": tradeoffs,
        "engineer_checks": checks,
        "engineering_confidence_code": engineering_confidence_code,
        "engineering_confidence": human_engineering_confidence_label(
            engineering_confidence_code
        ),
        "selected_by_legacy_top": selected_by_legacy_top,
        "optional_components": optional_components,
        "optional_total_price_value": _json_decimal(
            _decimal_value(candidate.get("optional_total_price_value"))
        ),
        "optional_total_price_currency": str(
            candidate.get("optional_total_price_currency") or ""
        ).strip(),
    }


def _public_component_row(component: Mapping[str, Any] | None) -> dict[str, Any]:
    if component is None:
        return {}
    facts = component.get("facts") if isinstance(component.get("facts"), Mapping) else {}
    return {
        "role": str(component.get("role") or "").strip(),
        "role_ru": str(component.get("role_ru") or "").strip(),
        "producer": str(component.get("producer") or "").strip(),
        "part_number": str(component.get("part_number") or "").strip(),
        "item_name": str(component.get("item_name") or "").strip(),
        "display_name": _component_article_text(component),
        "quantity_required": _int_value(component.get("quantity_required")),
        "server_quantity": _int_value(component.get("server_quantity")),
        "per_server_quantity": _int_value(component.get("per_server_quantity")),
        "available_quantity": _int_value(component.get("available_quantity")),
        "price_value": _json_decimal(_decimal_value(component.get("price_value"))),
        "price_currency": str(component.get("price_currency") or "").strip(),
        "line_total_value": _json_decimal(_decimal_value(component.get("line_total_value"))),
        "line_total_currency": str(component.get("line_total_currency") or "").strip(),
        "facts": _public_facts(facts),
        "cpu_cores": _int_value(component.get("cpu_cores")),
        "storage_capacity_tb": _float_value(component.get("storage_capacity_tb")),
        "ram_module_capacity_gb": _int_value(component.get("ram_module_capacity_gb")),
        "ram_total_gb_per_server": _int_value(component.get("ram_total_gb_per_server")),
    }


def _public_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    safe_keys = {
        "normalized_vendor",
        "cpu_brand",
        "cpu_family",
        "cpu_generation",
        "cpu_socket",
        "cpu_cores",
        "ram_type",
        "ram_capacity_gb",
        "storage_capacity",
        "storage_capacity_tb",
        "storage_interface",
        "nvme_support",
    }
    return {
        str(key): value
        for key, value in facts.items()
        if key in safe_keys and value not in (None, "", [])
    }


def _component_base_notes(component_base: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    cpu = component_base.get("cpu")
    if isinstance(cpu, Mapping):
        notes.extend(_base_component_quantity_notes("CPU", cpu))
    ram = component_base.get("ram")
    if isinstance(ram, Mapping):
        notes.extend(_ram_component_quantity_notes(ram))
    storage = component_base.get("storage")
    if isinstance(storage, Mapping):
        label = "SSD" if storage.get("role") == SSD_ROLE else "HDD"
        notes.extend(_storage_component_quantity_notes(label, storage))
    return notes


def _base_component_quantity_notes(label: str, component: Mapping[str, Any]) -> list[str]:
    quantity = _int_value(component.get("quantity_required"))
    per_server = _int_value(component.get("per_server_quantity"))
    notes: list[str] = []
    if per_server is not None:
        notes.append(f"{label}: {per_server} шт. на сервер")
    if quantity is not None:
        notes.append(f"{label}: {quantity} шт. всего")
    return notes


def _ram_component_quantity_notes(component: Mapping[str, Any]) -> list[str]:
    quantity = _int_value(component.get("quantity_required"))
    per_server = _int_value(component.get("per_server_quantity"))
    module_gb = _int_value(component.get("ram_module_capacity_gb"))
    total_gb = _int_value(component.get("ram_total_gb_per_server"))
    notes: list[str] = []
    if per_server is not None and module_gb is not None:
        total_text = f" = {total_gb} ГБ" if total_gb is not None else ""
        notes.append(f"RAM: {per_server} x {module_gb} ГБ на сервер{total_text}")
    elif per_server is not None:
        notes.append(f"RAM: {per_server} шт. на сервер")
    if quantity is not None:
        notes.append(f"RAM: {quantity} шт. всего")
    return notes


def _storage_component_quantity_notes(label: str, component: Mapping[str, Any]) -> list[str]:
    quantity = _int_value(component.get("quantity_required"))
    per_server = _int_value(component.get("per_server_quantity"))
    capacity = _float_value(component.get("storage_capacity_tb"))
    interface = _fact_from_public_component(component, "storage_interface")
    notes: list[str] = []
    if per_server is not None and capacity is not None:
        interface_text = f" {interface}" if interface and interface != UNKNOWN_FACT else ""
        capacity_text = _format_decimal_number(capacity)
        notes.append(
            f"{label}: {per_server} x {capacity_text} ТБ{interface_text} "
            "на сервер"
        )
    elif per_server is not None:
        notes.append(f"{label}: {per_server} шт. на сервер")
    if quantity is not None:
        notes.append(f"{label}: {quantity} шт. всего")
    return notes


def _platform_option_role(
    candidate: Mapping[str, Any],
    *,
    cheapest: Mapping[str, Any],
    database_preferred: Mapping[str, Any],
) -> str:
    if _recommendation_identity(candidate) == _recommendation_identity(cheapest):
        return "cheapest_quote"
    if _recommendation_identity(candidate) == _recommendation_identity(database_preferred):
        return "preferred_for_database"
    if _is_branded_safe_platform(candidate):
        return "branded_safe"
    if _technical_confirmation_score(candidate) > 0:
        return "engineering_clear"
    return "alternative"


def _database_preferred_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda candidate: (
            _technical_confirmation_score(candidate),
            1 if _is_branded_safe_platform(candidate) else 0,
            -_technical_warning_count(candidate),
            -(_decimal_value(candidate.get("total_price_value")) or Decimal("0")),
        ),
    )


def _is_branded_safe_platform(candidate: Mapping[str, Any]) -> bool:
    platform = _component_by_role(
        _mapping_rows(candidate.get("components")),
        SERVER_PLATFORM_ROLE,
    )
    text = _component_article_text(platform).casefold()
    return any(vendor in text for vendor in ("asus", "supermicro"))


def _platform_option_price_sort_key(option: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*_price_sort_key(option.get("total_price_value"), 1), option.get("option_index") or 0)


def _platform_option_stock_status(candidate: Mapping[str, Any]) -> str:
    available = _int_value(candidate.get("available_quantity"))
    if available is None:
        return "остаток требует проверки"
    if available >= 1:
        return "достаточно для текущего запроса"
    return "остаток ниже текущего запроса"


def _safe_tradeoffs(candidate: Mapping[str, Any]) -> list[str]:
    values = [
        candidate.get("commercial_tradeoff"),
        candidate.get("right_size_note"),
    ]
    return _unique(
        text
        for value in values
        if (text := _safe_presales_text(value))
    )


def _safe_engineer_checks(
    values: Sequence[Any],
    *,
    product_group: str = SERVER_PRODUCT_GROUP,
) -> list[str]:
    if product_group in {NETWORK_PRODUCT_GROUP, STORAGE_PRODUCT_GROUP}:
        return sanitize_engineer_checks_for_product_group(
            values,
            product_group=product_group,
            max_items=8,
        )
    checks: list[str] = []
    for value in values:
        raw = str(value or "")
        raw_lowered = raw.casefold()
        if (
            "llm_rec_" in raw_lowered
            or "component_candidate_id" in raw_lowered
            or "raw json" in raw_lowered
            or "web evidence not found" in raw_lowered
            or "keep engineer" in raw_lowered
            or "fatal " in raw_lowered
        ):
            continue
        text = _safe_presales_text(value)
        if not text:
            continue
        lowered = text.casefold()
        if (
            "llm_rec_" in lowered
            or "component_candidate_id" in lowered
            or "raw json" in lowered
            or "служебные данные" in lowered
            or "web evidence not found" in lowered
            or "keep engineer" in lowered
            or "fatal " in lowered
        ):
            continue
        checks.append(text)
    return deduplicate_engineer_checks(checks, max_items=8)


def _safe_presales_text(value: Any) -> str:
    text = sanitize_user_facing_text(value)
    if not text:
        return ""
    text = re.sub(r"\bllm_rec_[\w-]+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcomponent_candidate_id\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .;:-")
    return text


def _quote_recommendation(
    groups: list[Mapping[str, Any]],
    option_refs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not option_refs:
        return {}
    cheapest = min(
        option_refs,
        key=lambda ref: _platform_option_price_sort_key(ref["option"]),
    )
    database = max(
        option_refs,
        key=lambda ref: (
            _technical_confirmation_score(ref["candidate"]),
            1 if _is_branded_safe_platform(ref["candidate"]) else 0,
            -_technical_warning_count(ref["candidate"]),
            -(_decimal_value(ref["candidate"].get("total_price_value")) or Decimal("0")),
        ),
    )
    engineering = max(
        option_refs,
        key=lambda ref: (
            1 if _is_branded_safe_platform(ref["candidate"]) else 0,
            _technical_confirmation_score(ref["candidate"]),
            -_technical_warning_count(ref["candidate"]),
        ),
    )
    result = {
        "recommended_group_id": cheapest["group_id"],
        "recommended_option_id": cheapest["option_id"],
        "recommended_option_index": cheapest["option_index"],
        "for_cheapest_quote": _quote_option_text(cheapest),
        "for_database_preferred": _quote_option_text(database),
        "for_engineering_clarity": _quote_option_text(engineering),
        "summary": _quote_recommendation_summary(groups),
    }
    return {
        key: _safe_presales_text(value) if isinstance(value, str) else value
        for key, value in result.items()
    }


def _quote_recommendation_summary(groups: list[Mapping[str, Any]]) -> str:
    summary = (
        "Для КП код выбирает самый дешевый прошедший проверки вариант по текущим ценам OCS; "
        "для более спокойного технического варианта стоит сравнить его с платформой "
        "с лучшими признаками по бренду, backplane/PSU/DIMM и инженерной проверяемости."
    )
    if len(groups) <= 1:
        return summary
    return (
        f"{summary} Остальные конфигурационные базы в отчете являются компромиссами "
        "по RAM/DIMM-слотам или емкости накопителей и не являются отдельной "
        "обязательной базой для КП."
    )


def _quote_option_text(ref: Mapping[str, Any]) -> str:
    option = ref.get("option")
    if not isinstance(option, Mapping):
        return ""
    platform = option.get("platform")
    platform_text = (
        _component_article_text(platform)
        if isinstance(platform, Mapping)
        else "платформа требует проверки"
    )
    amount = _format_quote_amount(option)
    return " - ".join(part for part in [platform_text, amount] if part)


def _format_quote_amount(option: Mapping[str, Any]) -> str:
    value = option.get("total_price_value")
    currency = str(option.get("total_price_currency") or "").strip()
    if value in (None, ""):
        return ""
    return " ".join(part for part in [str(value), currency] if part)


def _family_label_from_candidate(candidate: Mapping[str, Any]) -> str:
    components = _mapping_rows(candidate.get("components"))
    platform = _component_by_role(components, SERVER_PLATFORM_ROLE)
    cpu = _component_by_role(components, CPU_ROLE)
    ram = _component_by_role(components, RAM_ROLE)
    storage = _component_by_role(components, SSD_ROLE) or _component_by_role(
        components,
        HDD_ROLE,
    )
    cpu_side = _cpu_side_from_public_components(cpu, platform)
    socket = _first_known_fact(platform, cpu, key="cpu_socket")
    chipset = _detect_chipset(_component_combined_text(components, SERVER_PLATFORM_ROLE))
    ram_type = _first_known_fact(ram, platform, key="ram_type")
    storage_interface = _first_known_fact(storage, platform, key="storage_interface")
    if storage_interface == UNKNOWN_FACT and _component_fact(platform, "nvme_support") == "True":
        storage_interface = "NVMe"

    cpu_part = " ".join(
        part
        for part in [
            cpu_side if cpu_side != UNKNOWN_FACT else "",
            socket if socket != UNKNOWN_FACT else "",
            f"/{chipset}" if chipset else "",
        ]
        if part
    ).replace(" /", "/")
    parts = [
        cpu_part or "CPU family requires review",
        ram_type if ram_type != UNKNOWN_FACT else "RAM type requires review",
        storage_interface if storage_interface != UNKNOWN_FACT else "storage requires review",
    ]
    return " / ".join(parts)


def _family_signature_label(
    signature: tuple[Any, ...],
    candidate: Mapping[str, Any],
) -> str:
    return f"{_family_label_from_candidate(candidate)} | component base {len(signature)} roles"


def _architecture_summary(candidate: Mapping[str, Any]) -> str:
    return (
        f"Семейство {_family_label_from_candidate(candidate)}: компонентная база "
        "сгруппирована по CPU/RAM/storage, платформы сравниваются отдельно."
    )


def _why_group_matters(candidate: Mapping[str, Any]) -> str:
    label = _family_label_from_candidate(candidate)
    return (
        f"{label} показывает совместимую архитектурную семью; внутри нее можно сравнивать "
        "платформы без искусственной смены CPU/RAM/SSD."
    )


def _cpu_side_from_public_components(
    cpu: Mapping[str, Any] | None,
    platform: Mapping[str, Any] | None,
) -> str:
    for component in (cpu, platform):
        family = _component_fact(component, "cpu_family")
        socket = _component_fact(component, "cpu_socket")
        brand = _component_fact(component, "cpu_brand")
        if family == "EPYC" or socket in {"SP3", "SP5", "LGA4094", "LGA6096"}:
            return "AMD"
        if family == "Xeon" or socket in {"LGA3647", "LGA4189", "LGA4677", "LGA4710"}:
            return "Intel"
        if brand in {"Intel", "AMD"}:
            return brand
    return UNKNOWN_FACT


def _first_known_fact(
    *components: Mapping[str, Any] | None,
    key: str,
) -> str:
    for component in components:
        value = _component_fact(component, key)
        if value != UNKNOWN_FACT:
            return value
    return UNKNOWN_FACT


def _fact_from_public_component(component: Mapping[str, Any], key: str) -> str:
    facts = component.get("facts")
    if isinstance(facts, Mapping):
        value = facts.get(key)
        if value not in (None, ""):
            return str(value)
    value = component.get(key)
    if value not in (None, ""):
        return str(value)
    return UNKNOWN_FACT


def _detect_chipset(text: str) -> str:
    match = re.search(r"\b(C\d{3})\b", text, re.IGNORECASE)
    return match.group(1).upper() if match is not None else ""


def _component_article_text(component: Mapping[str, Any] | None) -> str:
    if component is None:
        return ""
    parts = [
        str(component.get("producer") or "").strip(),
        str(component.get("part_number") or "").strip(),
    ]
    display = " ".join(part for part in parts if part)
    if display:
        return display
    return str(
        component.get("display_name")
        or component.get("item_name")
        or component.get("name")
        or ""
    ).strip()


def _format_decimal_number(value: float) -> str:
    decimal = Decimal(str(value))
    if decimal == decimal.to_integral():
        return str(int(decimal))
    return format(decimal.normalize(), "f")


def _price_optimal_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if candidate.get("completeness_status") == "complete" else 1,
        _evidence_sort_rank(candidate),
        *_price_sort_key(candidate.get("total_price_value"), 1),
        _technical_warning_count(candidate),
        _stable_text(_recommendation_signature_text(candidate)),
        _int_value(candidate.get("_llm_source_order")) or 0,
    )


def _technical_clean_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _evidence_sort_rank(candidate),
        _technical_warning_count(candidate),
        -_technical_confirmation_score(candidate),
        0 if candidate.get("completeness_status") == "complete" else 1,
        *_price_sort_key(candidate.get("total_price_value"), 1),
        _stable_text(_recommendation_signature_text(candidate)),
        _int_value(candidate.get("_llm_source_order")) or 0,
    )


def _alternative_sort_key(
    candidate: Mapping[str, Any],
    selected: list[Mapping[str, Any]],
) -> tuple[Any, ...]:
    return (
        -_diversity_score(candidate, selected),
        _evidence_sort_rank(candidate),
        _technical_warning_count(candidate),
        *_price_sort_key(candidate.get("total_price_value"), 1),
        _stable_text(_recommendation_signature_text(candidate)),
        _int_value(candidate.get("_llm_source_order")) or 0,
    )


def _evidence_sort_rank(candidate: Mapping[str, Any]) -> int:
    summary = candidate.get("evidence_summary")
    if not isinstance(summary, Mapping):
        return 1
    if _string_list(summary.get("fatal_concerns")):
        return 4
    status = str(summary.get("status") or "").strip()
    source_count = _int_value(summary.get("sources_count")) or 0
    if status == "disabled":
        return 1
    if status in {"not_found", "error"} or source_count <= 0:
        return 3
    missing = _string_list(summary.get("missing"))
    return 0 if not missing else 1


def _meaningfully_different(
    candidate: Mapping[str, Any],
    selected: list[Mapping[str, Any]],
) -> bool:
    return _diversity_score(candidate, selected) > 0


def _diversity_score(
    candidate: Mapping[str, Any],
    selected: list[Mapping[str, Any]],
) -> int:
    candidate_roles = _role_identity_map(candidate)
    score = 0
    for selected_candidate in selected:
        selected_roles = _role_identity_map(selected_candidate)
        if candidate_roles.get(SERVER_PLATFORM_ROLE) != selected_roles.get(
            SERVER_PLATFORM_ROLE
        ):
            score = max(score, 6)
        for role, weight in (
            (CPU_ROLE, 4),
            (RAM_ROLE, 3),
            (SSD_ROLE, 3),
            (HDD_ROLE, 3),
        ):
            if candidate_roles.get(role) != selected_roles.get(role):
                score = max(score, weight)
    return score


def _alternative_title(
    candidate: Mapping[str, Any],
    selected: list[Mapping[str, Any]],
) -> str:
    candidate_roles = _role_identity_map(candidate)
    if any(
        candidate_roles.get(SERVER_PLATFORM_ROLE)
        != _role_identity_map(selected_candidate).get(SERVER_PLATFORM_ROLE)
        for selected_candidate in selected
    ):
        return "Альтернативный вариант"

    cpu_cores = _component_fact_int(candidate, CPU_ROLE, "cpu_cores")
    if cpu_cores is not None:
        return f"Вариант с CPU {_cores_text(cpu_cores)}"
    if any(
        candidate_roles.get(RAM_ROLE) != _role_identity_map(selected_candidate).get(RAM_ROLE)
        for selected_candidate in selected
    ):
        return "Альтернативный вариант с другой RAM"
    if any(
        candidate_roles.get(SSD_ROLE) != _role_identity_map(selected_candidate).get(SSD_ROLE)
        or candidate_roles.get(HDD_ROLE) != _role_identity_map(selected_candidate).get(HDD_ROLE)
        for selected_candidate in selected
    ):
        return "Альтернативный вариант с другим накопителем"
    return "Альтернативный вариант"


def _technical_warning_count(candidate: Mapping[str, Any]) -> int:
    warnings = [
        *(_string_list(candidate.get("critical_checks"))),
        *(_string_list(candidate.get("critical_risks"))),
        *(_string_list(candidate.get("compatibility_warnings"))),
        *(_string_list(candidate.get("risk_flags"))),
        *(_string_list(candidate.get("missing_component_roles"))),
        *(_string_list(candidate.get("missing_components"))),
    ]
    informational_markers = (
        "llm composer",
        "инженер",
        "support list",
        "список поддерживаем",
    )
    meaningful = [
        warning
        for warning in _unique(warnings)
        if not any(marker in warning.casefold() for marker in informational_markers)
    ]
    return len(meaningful)


def _technical_confirmation_score(candidate: Mapping[str, Any]) -> int:
    score = 0
    components = _mapping_rows(candidate.get("components"))
    platform_text = _component_combined_text(components, SERVER_PLATFORM_ROLE)
    ram_text = _component_combined_text(components, RAM_ROLE)
    storage_text = " ".join(
        _component_combined_text(components, role) for role in (SSD_ROLE, HDD_ROLE)
    )
    cpu = _component_by_role(components, CPU_ROLE)
    platform = _component_by_role(components, SERVER_PLATFORM_ROLE)
    ram = _component_by_role(components, RAM_ROLE)
    storage = _component_by_role(components, SSD_ROLE) or _component_by_role(components, HDD_ROLE)

    if re.search(r"\b2\s*u\b|\b2u\b", platform_text, re.IGNORECASE):
        score += 1
    psu_pattern = r"\b(?:2x|2\s*x|dual|redundant)\s*(?:psu|power)\b|2\s*бп"
    if re.search(psu_pattern, platform_text, re.IGNORECASE):
        score += 1
    if _component_fact(cpu, "cpu_socket") != UNKNOWN_FACT:
        score += 1
    if _component_fact(platform, "cpu_socket") != UNKNOWN_FACT:
        score += 1
    if _component_fact(ram, "ram_type") == "DDR5" or "ddr5" in ram_text.casefold():
        score += 1
    if _component_fact(storage, "storage_interface") == "NVMe" or "nvme" in storage_text.casefold():
        score += 1
    return score


def _bom_signature(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    source_type = str(candidate.get("source_type") or candidate.get("candidate_type") or "")
    if source_type == READY_SERVER_CANDIDATE_TYPE:
        return (
            READY_SERVER_CANDIDATE_TYPE,
            str(candidate.get("source_candidate_id") or candidate.get("item_id") or ""),
            _int_value(candidate.get("quantity_required")) or 1,
        )
    role_map = _role_identity_map(candidate)
    quantity_map = _role_quantity_map(candidate)
    optional_roles = _optional_signature_roles(candidate)
    return tuple(
        (role, role_map.get(role, ""), quantity_map.get(role, 0))
        for role in CORE_BOM_ROLE_ORDER
        if role in role_map and role not in optional_roles
    )


def _recommendation_identity(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(candidate.get("recommendation_id") or candidate.get("candidate_id") or ""),
        _bom_signature(candidate),
    )


def _recommendation_signature_text(candidate: Mapping[str, Any]) -> str:
    return "|".join(str(part) for part in _bom_signature(candidate))


def _optional_signature_roles(candidate: Mapping[str, Any]) -> set[str]:
    roles: set[str] = set()
    for key in (
        "optional_component_roles",
        "engineer_check_component_roles",
    ):
        roles.update(_string_list(candidate.get(key)))
    return {
        role
        for role in roles
        if role in OPTIONAL_ENGINEER_CHECK_ROLES
    }


def _role_identity_map(candidate: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    components = _mapping_rows(candidate.get("components"))
    for component in components:
        role = _normalize_role(component.get("role"))
        if role is None:
            continue
        result[role] = str(
            component.get("component_candidate_id")
            or component.get("part_number")
            or component.get("item_id")
            or component.get("item_name")
            or ""
        )
    if not result and isinstance(candidate.get("component_candidate_ids"), Mapping):
        for prompt_role, component_id in candidate["component_candidate_ids"].items():
            role = _normalize_role(prompt_role)
            if role is None and str(prompt_role) == "storage":
                role = SSD_ROLE
            if role is not None:
                result[role] = str(component_id or "")
    return result


def _role_quantity_map(candidate: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    components = _mapping_rows(candidate.get("components"))
    for component in components:
        role = _normalize_role(component.get("role"))
        if role is None:
            continue
        quantity = _int_value(component.get("quantity_required"))
        if quantity is not None:
            result[role] = quantity
    quantities = candidate.get("quantities")
    if isinstance(quantities, Mapping):
        for prompt_role, value in quantities.items():
            role = _normalize_role(prompt_role)
            if role is None and str(prompt_role) == "storage":
                role = SSD_ROLE
            quantity = _int_value(value)
            if role is not None and quantity is not None:
                result.setdefault(role, quantity)
    return result


def _component_by_role(
    components: list[Mapping[str, Any]],
    role: str,
) -> Mapping[str, Any] | None:
    for component in components:
        if _normalize_role(component.get("role")) == role:
            return component
    return None


def _component_combined_text(components: list[Mapping[str, Any]], role: str) -> str:
    component = _component_by_role(components, role)
    if component is None:
        return ""
    values: list[str] = []
    for key in ("producer", "part_number", "item_name", "name"):
        value = component.get(key)
        if value not in (None, ""):
            values.append(str(value))
    facts = component.get("facts")
    if isinstance(facts, Mapping):
        values.extend(str(value) for value in facts.values() if value not in (None, "", []))
    return " ".join(values)


def _component_fact(component: Mapping[str, Any] | None, key: str) -> str:
    if component is None:
        return UNKNOWN_FACT
    facts = component.get("facts")
    if isinstance(facts, Mapping):
        value = facts.get(key)
        if value not in (None, ""):
            return str(value)
    value = component.get(key)
    if value not in (None, ""):
        return str(value)
    text = _component_combined_text([component], str(component.get("role") or ""))
    if key in {"ram_type", "memory_type"}:
        return _detect_ram_type(text)
    if key in {"cpu_socket", "socket_family"}:
        return _detect_cpu_socket(text)
    if key == "storage_interface":
        return _detect_storage_interface(text)
    return UNKNOWN_FACT


def _component_fact_int(
    candidate: Mapping[str, Any],
    role: str,
    key: str,
) -> int | None:
    component = _component_by_role(_mapping_rows(candidate.get("components")), role)
    value = _component_fact(component, key)
    if value == UNKNOWN_FACT:
        return None
    return _int_value(value)


def _rejected_proposal(
    *,
    recommendation: LlmRecommendationPayload,
    proposal_index: int,
    warnings: list[str],
    component_index: Mapping[str, _IndexedComponentCandidate],
    normalized_requirements: Any,
    exception: Exception | None = None,
) -> _RejectedProposal:
    message = "; ".join(warnings) if warnings else "rejected by validator"
    category = _classify_rejection(message)
    debug_safe = _proposal_rejected_debug_safe(
        recommendation,
        proposal_index=proposal_index,
        category=category,
        warnings=warnings,
        component_index=component_index,
        normalized_requirements=normalized_requirements,
        exception=exception,
    )
    category = _refined_rejection_category(category, debug_safe)
    debug_safe["rejection_category"] = _reason_code(category)
    debug_safe["rejection_code"] = _reason_code(category)
    debug_safe["rejection_message_ru"] = REJECTION_REASON_MESSAGES.get(
        category,
        REJECTION_REASON_MESSAGES["rejected_other"],
    )
    return _RejectedProposal(
        recommendation_id=recommendation.recommendation_id,
        category=category,
        message=REJECTION_REASON_MESSAGES.get(
            category,
            REJECTION_REASON_MESSAGES["rejected_other"],
        ),
        proposal_index=proposal_index,
        debug_safe=debug_safe,
    )


def _classify_rejection(message: str) -> str:
    lowered = message.casefold()
    if "duplicate" in lowered or "дубликат" in lowered:
        return "rejected_duplicate"
    if "schema" in lowered or "validationerror" in lowered:
        return "rejected_invalid_schema"
    if "unknown component" in lowered or "unknown source" in lowered:
        return "rejected_unknown_component"
    if "component role mismatch" in lowered or "source type mismatch" in lowered:
        return "rejected_role_mismatch"
    if "source_candidate_id required" in lowered:
        return "rejected_invalid_candidate_type"
    if "optional" in lowered and "core" in lowered:
        return "rejected_optional_core_conflict"
    if "ram module capacity is unknown" in lowered:
        return "rejected_ram_capacity_unknown"
    if (
        "materializ" in lowered
        or "capacity is unknown" in lowered
        or "cores are unknown" in lowered
    ):
        return "rejected_quantity_materialization_failed"
    if "stock" in lowered or "остат" in lowered:
        return "rejected_stock_shortage"
    if "price" in lowered or "currency" in lowered:
        return "rejected_invalid_price_or_currency"
    if "right-size" in lowered or "overfit rejected" in lowered:
        return "rejected_right_size_rejected"
    if "amd epyc" in lowered or "intel xeon" in lowered or "socket mismatch" in lowered:
        return "rejected_platform_cpu_mismatch"
    if (
        "fatal" in lowered
        or "mismatch" in lowered
        or "incompat" in lowered
        or "serious gaps" in lowered
    ):
        return "rejected_fatal"
    if (
        "required" in lowered
        or "missing" in lowered
        or "quantity" in lowered
        or "price or currency" in lowered
    ):
        return "rejected_missing_required_role"
    return "rejected_other"


def _safe_rejection_message(message: str) -> str:
    category = _classify_rejection(message)
    return REJECTION_REASON_MESSAGES.get(category, REJECTION_REASON_MESSAGES["rejected_other"])


def _rejected_warning(rejected: _RejectedProposal) -> str:
    return f"{rejected.recommendation_id}: {rejected.message}"


def _validation_summary(
    *,
    accepted: int,
    accepted_after_validation: int,
    rejected: list[_RejectedProposal],
) -> dict[str, int]:
    summary = {key: 0 for key in REJECTION_SUMMARY_KEYS}
    summary["accepted"] = accepted
    summary["accepted_after_validation"] = accepted_after_validation
    for item in rejected:
        summary[item.category] = summary.get(item.category, 0) + 1
    summary["validation_rejected_count"] = sum(
        summary.get(key, 0) for key in VALIDATION_REJECTION_KEYS
    )
    summary["selection_skipped_count"] = sum(
        summary.get(key, 0) for key in SELECTION_SKIP_KEYS
    )
    summary["rejected_missing_required"] = summary.get(
        "rejected_missing_required_role",
        0,
    )
    summary["rejected_stock"] = summary.get("rejected_stock_shortage", 0)
    summary["rejected_fatal"] = summary.get("rejected_fatal", 0) + summary.get(
        "rejected_platform_cpu_mismatch",
        0,
    )
    summary["rejected_duplicate"] = summary.get("selection_skipped_duplicate", 0)
    summary["rejected_right_size"] = (
        summary.get("selection_skipped_dominated_by_cheaper_equivalent", 0)
        + summary.get("selection_skipped_worse_by_price", 0)
    )
    summary["rejected"] = len(rejected)
    return summary


def _rejected_reasons_top(rejected: list[_RejectedProposal]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in rejected:
        counts[item.category] = counts.get(item.category, 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (
            -item[1],
            REJECTION_REASON_ORDER.index(item[0])
            if item[0] in REJECTION_REASON_ORDER
            else len(REJECTION_REASON_ORDER),
        ),
    )
    return [
        {
            "reason": category.removeprefix("rejected_"),
            "count": count,
            "message": REJECTION_REASON_MESSAGES.get(category, category),
        }
        for category, count in ranked[:5]
    ]


def _rejected_debug_safe(rejected: list[_RejectedProposal]) -> list[dict[str, Any]]:
    return [
        item.debug_safe
        if item.debug_safe
        else {
            "proposal_index": item.proposal_index,
            "recommendation_id": item.recommendation_id,
            "rejection_category": _reason_code(item.category),
            "rejection_code": _reason_code(item.category),
            "rejection_message_ru": item.message,
            "validation_errors": [],
            "validation_warnings": [],
        }
        for item in rejected
    ]


def _proposal_rejected_debug_safe(
    recommendation: LlmRecommendationPayload,
    *,
    proposal_index: int,
    category: str,
    warnings: list[str],
    component_index: Mapping[str, _IndexedComponentCandidate],
    normalized_requirements: Any,
    exception: Exception | None = None,
) -> dict[str, Any]:
    selected, role_mismatches, unknown_ids = _inspect_llm_component_selection(
        recommendation,
        component_index,
        normalized_requirements=normalized_requirements,
    )
    optional_roles_seen = _optional_roles_seen(recommendation, component_index)
    core_roles_seen = [role for role in ROLE_ORDER if role in selected]
    missing_roles = _missing_mandatory_roles(
        mandatory_roles=_mandatory_roles(normalized_requirements),
        selected=selected,
        normalized_requirements=normalized_requirements,
    )
    normalized_quantities: dict[str, Any] = {}
    materialized_roles: list[str] = []
    stock_shortages: list[dict[str, Any]] = []
    hard_capability_validation: list[dict[str, Any]] = []
    materialization_error = ""
    if selected:
        try:
            optional_component_roles = [
                role for role in optional_roles_seen if role in selected
            ]
            materialized = _materialize_build_quantities(
                recommendation,
                selected=selected,
                quantities=_diagnostic_quantities(recommendation, selected),
                normalized_requirements=normalized_requirements,
                optional_component_roles=optional_component_roles,
            )
            materialization_error = materialized.error or ""
            normalized_quantities = {
                PROMPT_ROLE_BY_INTERNAL_ROLE.get(role, role): detail
                for role, detail in materialized.quantity_details.items()
            }
            materialized_roles = [
                role
                for role in ROLE_ORDER
                if role in materialized.quantities
            ]
            stock_shortages = _diagnostic_stock_shortages(
                selected,
                materialized.quantities,
            )
            hard_capability_validation = _hard_capability_validation(
                selected=selected,
                quantities=materialized.quantities,
                normalized_requirements=normalized_requirements,
            )
        except Exception as exc:
            materialization_error = (
                f"{type(exc).__name__}: {_safe_diagnostic_text(exc, limit=200)}"
            )

    missing_required_capabilities = [
        dict(row)
        for row in hard_capability_validation
        if str(row.get("status") or "").strip() != "satisfied"
    ]
    missing_roles = _unique(
        [
            *missing_roles,
            *[
                str(row.get("role") or row.get("capability_id") or "").strip()
                for row in missing_required_capabilities
                if str(row.get("role") or row.get("capability_id") or "").strip()
            ],
        ]
    )
    diagnostic = {
        "proposal_index": proposal_index,
        "recommendation_id": recommendation.recommendation_id,
        "recommendation_slot": _recommendation_slot_value(recommendation),
        "proposal_role": recommendation.proposal_role,
        "title": _safe_diagnostic_text(recommendation.title, limit=160),
        "candidate_type": recommendation.source_type,
        "source_type": recommendation.source_type,
        "source_candidate_id": recommendation.source_candidate_id,
        "selected_component_candidate_ids": dict(
            recommendation.selected_component_candidate_ids
        ),
        "component_candidate_ids": dict(recommendation.component_candidate_ids),
        "normalized_core_component_candidate_ids": _core_component_candidate_ids(
            recommendation,
            component_index=component_index,
            normalized_requirements=normalized_requirements,
        ),
        "selected_component_candidate_ids_alias_used": bool(
            recommendation.selected_component_candidate_ids
            and not recommendation.component_candidate_ids
        ),
        "materialized_component_roles": materialized_roles,
        "rejection_category": _reason_code(category),
        "rejection_code": _reason_code(category),
        "rejection_message_ru": REJECTION_REASON_MESSAGES.get(
            category,
            REJECTION_REASON_MESSAGES["rejected_other"],
        ),
        "validation_errors": [_safe_diagnostic_text(item, limit=300) for item in warnings],
        "validation_warnings": [],
        "missing_roles": missing_roles,
        "stock_shortages": stock_shortages,
        "role_mismatches": role_mismatches,
        "unknown_component_ids": unknown_ids,
        "normalized_quantities": normalized_quantities,
        "hard_capability_validation": _jsonable(hard_capability_validation),
        "validation_hard_mismatches": [
            row
            for row in hard_capability_validation
            if row.get("status") == "hard_mismatch"
        ],
        "validation_unverified_requirements": [
            row
            for row in hard_capability_validation
            if row.get("status") == "unverified_hard_requirement"
        ],
        "missing_required_capabilities": _jsonable(missing_required_capabilities),
        "core_roles_seen": core_roles_seen,
        "optional_roles_seen": optional_roles_seen,
        "optional_core_conflicts": [
            role
            for role in core_roles_seen
            if role in OPTIONAL_ENGINEER_CHECK_ROLES
            and role not in set(_mandatory_roles(normalized_requirements))
        ],
        "stage": _rejection_stage(category),
    }
    if materialization_error:
        diagnostic["materialization_error"] = _safe_diagnostic_text(
            materialization_error,
            limit=300,
        )
    if exception is not None:
        diagnostic["exception_type"] = type(exception).__name__
        diagnostic["exception_message_sanitized"] = _safe_diagnostic_text(
            exception,
            limit=300,
        )
        diagnostic["stage"] = "validate_exception"
    return diagnostic


def _schema_rejected_debug_safe(
    proposal: Any,
    *,
    proposal_index: int,
    validation_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    row = proposal if isinstance(proposal, Mapping) else {}
    return {
        "proposal_index": proposal_index,
        "recommendation_id": _raw_recommendation_id(proposal, proposal_index),
        "recommendation_slot": _safe_diagnostic_text(
            row.get("recommendation_slot"),
            limit=80,
        ),
        "proposal_role": _safe_diagnostic_text(row.get("proposal_role"), limit=80),
        "title": _safe_diagnostic_text(row.get("title"), limit=160),
        "candidate_type": _safe_diagnostic_text(row.get("source_type"), limit=80),
        "source_candidate_id": _safe_diagnostic_text(
            row.get("source_candidate_id"),
            limit=120,
        ),
        "selected_component_candidate_ids": _safe_component_id_map(
            row.get("selected_component_candidate_ids")
        ),
        "component_candidate_ids": _safe_component_id_map(
            row.get("component_candidate_ids")
        ),
        "materialized_component_roles": [],
        "rejection_category": "invalid_schema",
        "rejection_code": "invalid_schema",
        "rejection_message_ru": REJECTION_REASON_MESSAGES["rejected_invalid_schema"],
        "validation_errors": validation_errors,
        "validation_warnings": [],
        "missing_roles": [],
        "stock_shortages": [],
        "role_mismatches": [],
        "unknown_component_ids": [],
        "normalized_quantities": {},
        "core_roles_seen": [],
        "optional_roles_seen": [],
        "stage": "parse",
    }


def _candidate_rejected_debug_safe(
    candidate: Mapping[str, Any],
    *,
    category: str,
    message: str,
) -> dict[str, Any]:
    return {
        "proposal_index": _int_value(candidate.get("_llm_source_order")),
        "recommendation_id": str(
            candidate.get("recommendation_id") or candidate.get("candidate_id") or ""
        ),
        "recommendation_slot": candidate.get("recommendation_slot"),
        "proposal_role": candidate.get("proposal_role"),
        "title": _safe_diagnostic_text(candidate.get("title"), limit=160),
        "candidate_type": candidate.get("candidate_type") or candidate.get("source_type"),
        "source_candidate_id": candidate.get("source_candidate_id"),
        "selected_component_candidate_ids": {},
        "component_candidate_ids": _safe_component_id_map(
            candidate.get("component_candidate_ids")
        ),
        "materialized_component_roles": _string_list(
            candidate.get("included_component_roles")
        ),
        "rejection_category": _reason_code(category),
        "rejection_code": _reason_code(category),
        "rejection_message_ru": REJECTION_REASON_MESSAGES.get(category, message),
        "validation_errors": [_safe_diagnostic_text(message, limit=300)],
        "validation_warnings": _string_list(candidate.get("compatibility_warnings")),
        "missing_roles": _string_list(candidate.get("missing_component_roles")),
        "stock_shortages": [],
        "role_mismatches": [],
        "unknown_component_ids": [],
        "normalized_quantities": _safe_mapping(
            candidate.get("normalized_bom_quantities")
        ),
        "core_roles_seen": _string_list(candidate.get("included_component_roles")),
        "optional_roles_seen": _string_list(candidate.get("optional_component_roles")),
        "stage": "select",
    }


def _inspect_llm_component_selection(
    recommendation: LlmRecommendationPayload,
    component_index: Mapping[str, _IndexedComponentCandidate],
    *,
    normalized_requirements: Any,
) -> tuple[dict[str, _IndexedComponentCandidate], list[dict[str, Any]], list[str]]:
    selected: dict[str, _IndexedComponentCandidate] = {}
    role_mismatches: list[dict[str, Any]] = []
    unknown_ids: list[str] = []
    for prompt_role, component_id in _core_component_candidate_ids(recommendation).items():
        candidate = component_index.get(component_id)
        if candidate is None:
            unknown_ids.append(component_id)
            continue
        expected_role = _diagnostic_expected_role(
            prompt_role,
            candidate,
            normalized_requirements=normalized_requirements,
        )
        if expected_role is None:
            role_mismatches.append(
                {
                    "prompt_role": str(prompt_role),
                    "component_candidate_id": component_id,
                    "expected_role": None,
                    "actual_role": candidate.internal_role,
                }
            )
            continue
        if candidate.internal_role != expected_role:
            role_mismatches.append(
                {
                    "prompt_role": str(prompt_role),
                    "component_candidate_id": component_id,
                    "expected_role": expected_role,
                    "actual_role": candidate.internal_role,
                }
            )
            continue
        selection_key = _selection_key_for_component(
            prompt_role,
            expected_role=expected_role,
            selected=selected,
        )
        selected[selection_key] = candidate
    return selected, role_mismatches, _unique(unknown_ids)


def _core_component_candidate_ids(
    recommendation: LlmRecommendationPayload,
    *,
    component_index: Mapping[str, _IndexedComponentCandidate] | None = None,
    normalized_requirements: Any = None,
) -> dict[str, str]:
    source = (
        recommendation.component_candidate_ids
        or recommendation.selected_component_candidate_ids
    )
    raw = {
        str(role): str(component_id).strip()
        for role, component_id in source.items()
        if str(component_id or "").strip()
    }
    if component_index is None:
        return raw
    return _normalized_component_candidate_id_map(
        raw,
        component_index=component_index,
        normalized_requirements=normalized_requirements,
    )


def _normalized_component_candidate_id_map(
    component_candidate_ids: Mapping[str, str],
    *,
    component_index: Mapping[str, _IndexedComponentCandidate],
    normalized_requirements: Any,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for prompt_role, component_id in component_candidate_ids.items():
        candidate = component_index.get(component_id)
        if candidate is None:
            result[str(prompt_role)] = component_id
            continue
        expected_role = _role_from_component_id_key(
            prompt_role,
            candidate,
            normalized_requirements=normalized_requirements,
        )
        if expected_role is not None and candidate.internal_role == expected_role:
            result[PROMPT_ROLE_BY_INTERNAL_ROLE.get(expected_role, expected_role)] = component_id
            continue
        result[str(prompt_role)] = component_id
    return result


def _optional_roles_seen(
    recommendation: LlmRecommendationPayload,
    component_index: Mapping[str, _IndexedComponentCandidate],
) -> list[str]:
    roles: list[str] = []
    for source in (
        recommendation.optional_component_candidate_ids,
        recommendation.engineer_check_component_candidate_ids,
    ):
        for component_id_value in source.values():
            component_id = str(component_id_value or "").strip()
            candidate = component_index.get(component_id)
            if candidate is not None:
                roles.append(candidate.internal_role)
    return _unique(roles)


def _diagnostic_expected_role(
    prompt_role: str,
    candidate: _IndexedComponentCandidate,
    *,
    normalized_requirements: Any,
) -> str | None:
    expected_role = _role_from_component_id_key(
        prompt_role,
        candidate,
        normalized_requirements=normalized_requirements,
    )
    if expected_role is None and prompt_role == "storage":
        return (
            candidate.internal_role
            if candidate.internal_role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE, STORAGE_SYSTEM_ROLE}
            else None
        )
    return expected_role


def _diagnostic_missing_roles(
    core_roles_seen: Sequence[str],
    normalized_requirements: Any,
) -> list[str]:
    seen = set(core_roles_seen)
    missing: list[str] = []
    for role in _mandatory_roles(normalized_requirements):
        if role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE} and (
            {DRIVE_ROLE, SSD_ROLE, HDD_ROLE} & seen
        ):
            continue
        if role not in seen:
            missing.append(role)
    return missing


def _diagnostic_quantities(
    recommendation: LlmRecommendationPayload,
    selected: Mapping[str, _IndexedComponentCandidate],
) -> dict[str, int]:
    quantities: dict[str, int] = {}
    for role, candidate in selected.items():
        quantity = _llm_quantity_for_internal_role(recommendation.quantities, role)
        if quantity is None:
            quantity = _int_value(candidate.row.get("quantity_required"))
        quantities[role] = quantity if quantity is not None and quantity > 0 else 1
    return quantities


def _diagnostic_stock_shortages(
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
) -> list[dict[str, Any]]:
    shortages: list[dict[str, Any]] = []
    for role, candidate in selected.items():
        required = _int_value(quantities.get(role))
        available = _int_value(candidate.row.get("available_quantity"))
        if required is None:
            continue
        if available is None or available < required:
            shortages.append(
                {
                    "role": role,
                    "component_candidate_id": candidate.component_candidate_id,
                    "required_quantity": required,
                    "available_quantity": available,
                }
            )
    return shortages


def _refined_rejection_category(category: str, diagnostic: Mapping[str, Any]) -> str:
    if diagnostic.get("unknown_component_ids"):
        return "rejected_unknown_component"
    if diagnostic.get("role_mismatches"):
        return "rejected_role_mismatch"
    if diagnostic.get("stock_shortages"):
        return "rejected_stock_shortage"
    if diagnostic.get("optional_core_conflicts"):
        return "rejected_optional_core_conflict"
    if diagnostic.get("materialization_error") and category == "rejected_other":
        return "rejected_quantity_materialization_failed"
    if diagnostic.get("missing_roles") and category == "rejected_other":
        return "rejected_missing_required_role"
    return category


def _rejection_stage(category: str) -> str:
    return {
        "rejected_invalid_schema": "parse",
        "rejected_unknown_component": "validate_roles",
        "rejected_role_mismatch": "validate_roles",
        "rejected_missing_required_role": "validate_roles",
        "rejected_stock_shortage": "validate_stock",
        "rejected_platform_cpu_mismatch": "validate_compatibility",
        "rejected_ram_capacity_unknown": "materialize",
        "rejected_quantity_materialization_failed": "materialize",
        "rejected_invalid_price_or_currency": "validate_price",
        "rejected_right_size_rejected": "validate_right_size",
        "rejected_optional_core_conflict": "validate_roles",
    }.get(category, "validate")


def _reason_code(category: str) -> str:
    if category.startswith("rejected_"):
        return category.removeprefix("rejected_")
    if category.startswith("selection_skipped_"):
        return "duplicate_or_selection_skip"
    return category


def _raw_proposal_rows(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        rows = _proposal_list_from_payload(payload)
        return list(rows) if rows is not None else []
    return []


def _raw_recommendation_id(proposal: Any, index: int) -> str:
    if isinstance(proposal, Mapping):
        value = proposal.get("recommendation_id") or proposal.get("candidate_id")
        if value:
            return _safe_diagnostic_text(value, limit=120)
    return f"proposal_{index + 1}"


def _schema_validation_errors(exc: ValidationError | None) -> list[dict[str, Any]]:
    if exc is None:
        return []
    errors: list[dict[str, Any]] = []
    for error in exc.errors()[:20]:
        errors.append(
            {
                "loc": [str(item) for item in error.get("loc", [])],
                "type": _safe_diagnostic_text(error.get("type"), limit=80),
                "message": _safe_diagnostic_text(error.get("msg"), limit=200),
            }
        )
    return errors


def _safe_component_id_map(
    value: Any,
    *,
    package: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            component_id = _component_id_from_alias_value(item)
            text = str(component_id or "").strip()
            if text:
                result[str(key)] = _safe_diagnostic_text(text, limit=160)
        return result
    if not isinstance(value, list):
        return {}

    rows_by_id = _candidate_rows_by_id(package or {})
    for index, item in enumerate(value, start=1):
        if isinstance(item, Mapping):
            component_id = _selected_component_alias_id(item)
            if not component_id:
                component_id = _component_id_from_alias_value(item)
            role_key = _selected_component_alias_role(item, component_id, rows_by_id)
        else:
            component_id = str(item or "").strip()
            candidate_row = rows_by_id.get(component_id)
            role_key = str(candidate_row.get("role") or "").strip() if candidate_row else ""
        text = str(component_id or "").strip()
        if text:
            result[_unique_component_role_key(role_key or f"component_{index}", result)] = (
                _safe_diagnostic_text(text, limit=160)
            )
    return result


def _component_id_from_alias_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(
            value.get("component_candidate_id")
            or value.get("candidate_id")
            or value.get("id")
            or ""
        ).strip()
    return str(value or "").strip()


def _validate_recommendation(
    recommendation: LlmRecommendationPayload,
    *,
    stock_candidate_index: dict[str, _IndexedStockCandidate],
    component_index: dict[str, _IndexedComponentCandidate],
    user_request: str | None,
    normalized_requirements: Any,
    evidence_by_component_id: Mapping[str, Mapping[str, Any]],
    relation_evidence: Sequence[Mapping[str, Any]],
    evidence_review: Mapping[str, Any] | None,
    use_recommendation_evidence: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    if recommendation.decision == "do_not_use":
        return None, [f"{recommendation.recommendation_id}: do_not_use skipped"]

    if recommendation.source_type == READY_SERVER_CANDIDATE_TYPE:
        if not recommendation.source_candidate_id:
            return None, [
                f"{recommendation.recommendation_id}: ready_server source_candidate_id required"
            ]
        source = stock_candidate_index.get(recommendation.source_candidate_id)
        if source is None:
            return None, [f"{recommendation.recommendation_id}: unknown source_candidate_id"]
        if source.candidate_type != READY_SERVER_CANDIDATE_TYPE:
            return None, [f"{recommendation.recommendation_id}: source type mismatch"]
        return _validate_ready_server_recommendation(
            recommendation,
            source=source,
            normalized_requirements=normalized_requirements,
            evidence_by_component_id=evidence_by_component_id,
            relation_evidence=relation_evidence,
            evidence_review=evidence_review,
            use_recommendation_evidence=use_recommendation_evidence,
        )

    source = None
    if recommendation.source_candidate_id:
        source = stock_candidate_index.get(recommendation.source_candidate_id)
        if source is None:
            return None, [f"{recommendation.recommendation_id}: unknown source_candidate_id"]
        if source.candidate_type not in {BUILD_CANDIDATE_TYPE, PARTIAL_BUILD_CANDIDATE_TYPE}:
            return None, [f"{recommendation.recommendation_id}: source type mismatch"]
    return _validate_build_recommendation(
        recommendation,
        source=source,
        component_index=component_index,
        user_request=user_request,
        normalized_requirements=normalized_requirements,
        evidence_by_component_id=evidence_by_component_id,
        relation_evidence=relation_evidence,
        evidence_review=evidence_review,
        use_recommendation_evidence=use_recommendation_evidence,
    )


def _validate_ready_server_recommendation(
    recommendation: LlmRecommendationPayload,
    *,
    source: _IndexedStockCandidate,
    normalized_requirements: Any,
    evidence_by_component_id: Mapping[str, Mapping[str, Any]],
    relation_evidence: Sequence[Mapping[str, Any]],
    evidence_review: Mapping[str, Any] | None,
    use_recommendation_evidence: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    source_row = source.row
    missing = _string_list(source_row.get("missing_requirements"))
    risks = _string_list(source_row.get("risk_flags"))
    fatal_warning = _fatal_warning_text(
        [
            *missing,
            *risks,
            *_string_list(source_row.get("compatibility_warnings")),
        ]
    )
    if fatal_warning:
        return None, [f"{recommendation.recommendation_id}: {fatal_warning}"]
    if _has_serious_ready_gap(missing):
        return None, [f"{recommendation.recommendation_id}: ready_server has serious gaps"]
    hard_capability_warning = _validate_ready_server_hard_capabilities(
        source_row,
        normalized_requirements=normalized_requirements,
        recommendation_id=recommendation.recommendation_id,
    )
    if hard_capability_warning:
        return None, [hard_capability_warning]

    quantity = _ready_server_quantity(source_row, normalized_requirements)
    if quantity is None or quantity <= 0:
        return None, [f"{recommendation.recommendation_id}: ready_server quantity missing"]

    available = _int_value(source_row.get("available_quantity"))
    if available is None or available < quantity:
        return None, [f"{recommendation.recommendation_id}: ready_server stock is insufficient"]

    unit_price = _decimal_value(source_row.get("price_value"))
    currency = str(source_row.get("price_currency") or "").strip()
    if unit_price is None or not currency:
        return None, [f"{recommendation.recommendation_id}: ready_server price is missing"]

    total_price = unit_price * quantity
    confidence_score = _confidence_score(recommendation.confidence)
    checks = _critical_checks(
        [
            *recommendation.critical_checks,
            *recommendation.engineer_checks,
            *risks,
            "Проверить фактическую комплектацию готового сервера перед КП.",
        ]
    )
    display_name = _stock_candidate_display_name(source_row)
    component_summary = _recommendation_component_summary(recommendation, source_row)
    if not any(component_summary.values()):
        component_summary["platform"] = display_name
    source_evidence = evidence_by_component_id.get(source.candidate_id)
    if source_evidence is None and not evidence_by_component_id:
        source_evidence = {"evidence_status": "disabled", "confidence": "unknown"}
    evidence_summary = _ready_server_evidence_summary(
        source_evidence,
        relation_evidence=relation_evidence,
        evidence_review=evidence_review,
    )
    if use_recommendation_evidence:
        evidence_summary = _coerce_llm_evidence_summary(
            recommendation.evidence_summary
        ) or _default_online_evidence_summary()
    evidence_penalty = _evidence_confidence_penalty(evidence_summary)
    adjusted_confidence_score = max(1, confidence_score - evidence_penalty)
    checks = _critical_checks(
        [
            *checks,
            *_string_list(evidence_summary.get("engineering_checks")),
        ]
    )

    row = {
        "candidate_id": recommendation.recommendation_id,
        "recommendation_id": recommendation.recommendation_id,
        "source_candidate_id": source.candidate_id,
        "source_type": READY_SERVER_CANDIDATE_TYPE,
        "candidate_type": READY_SERVER_CANDIDATE_TYPE,
        "distributor_code": source_row.get("distributor_code") or "ocs",
        "item_id": source_row.get("item_id") or source.candidate_id,
        "product_key": source_row.get("product_key"),
        "part_number": source_row.get("part_number"),
        "producer": source_row.get("producer"),
        "category_id": source_row.get("category_id"),
        "item_name": display_name,
        "display_name": display_name,
        "confidence_score": adjusted_confidence_score,
        "price_value": _json_decimal(unit_price),
        "price_currency": currency,
        "available_quantity": available,
        "quantity_required": quantity,
        "reservable_locations": source_row.get("reservable_locations") or 0,
        "matched_requirements": _string_list(source_row.get("matched_requirements")),
        "missing_requirements": missing,
        "risk_flags": checks,
        "platform": {},
        "components": [],
        "component_summary": component_summary,
        "total_price_value": _json_decimal(total_price),
        "total_price_currency": currency,
        "price_note": None,
        "total_price_note": None,
        "missing_components": [],
        "compatibility_warnings": checks,
        "engineer_review_required": True,
        "completeness_status": "complete",
        "completeness_label": "Готовый складской вариант, требуется инженерная проверка",
        "included_component_roles": [],
        "missing_component_roles": [],
        "excluded_from_total_roles": [],
        "score": adjusted_confidence_score,
        "rank_reason": [recommendation.why_selected],
        "llm_configurator": True,
        "decision": recommendation.decision,
        "proposal_role": recommendation.proposal_role,
        "recommendation_slot": _recommendation_slot_value(recommendation),
        "title": recommendation.title,
        "quantities": _clean_quantities(recommendation.quantities),
        "why_selected": recommendation.why_selected,
        "why_selected_short": recommendation.why_selected_short,
        "commercial_tradeoff": recommendation.commercial_tradeoff,
        "requirement_fulfillment_summary": _jsonable(
            recommendation.requirement_fulfillment_summary
        ),
        "assumptions": _string_list(recommendation.assumptions),
        "what_is_missing": _string_list(recommendation.what_is_missing),
        "critical_checks": checks,
        "critical_risks": checks,
        "confidence": _adjusted_confidence_label(
            recommendation.confidence,
            penalty=evidence_penalty,
        ),
        "commercial_fit_confidence": _normalized_confidence_label(
            recommendation.confidence
        ),
        "engineering_confidence": _engineering_confidence_label(evidence_summary),
        "displayed_confidence": _displayed_confidence_text(
            recommendation.confidence,
            evidence_summary,
        ),
        "optimization_mode": OPTIMIZATION_MODE_COST_MINIMAL_FIT,
        "requirement_fit": "ready_stock_fit",
        "right_size_note": recommendation.right_size_note
        or "Подбор: готовый складской вариант с проверками",
        "evidence_summary": evidence_summary,
        "evidence_review": dict(evidence_review or {}),
    }
    return row, []


def _validate_build_recommendation(
    recommendation: LlmRecommendationPayload,
    *,
    source: _IndexedStockCandidate | None,
    component_index: dict[str, _IndexedComponentCandidate],
    user_request: str | None,
    normalized_requirements: Any,
    evidence_by_component_id: Mapping[str, Mapping[str, Any]],
    relation_evidence: Sequence[Mapping[str, Any]],
    evidence_review: Mapping[str, Any] | None,
    use_recommendation_evidence: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    selected: dict[str, _IndexedComponentCandidate] = {}
    quantities: dict[str, int] = {}
    source_fatal_warning = None
    if source is not None:
        source_fatal_warning = _fatal_warning_text(
            [
                *_string_list(source.row.get("compatibility_warnings")),
                *_string_list(source.row.get("risk_flags")),
                *_string_list(source.row.get("missing_requirements")),
            ]
        )
    if source_fatal_warning:
        return None, [f"{recommendation.recommendation_id}: {source_fatal_warning}"]

    selection_error = _select_components_from_llm_payload(
        recommendation,
        component_index=component_index,
        selected=selected,
        quantities=quantities,
        normalized_requirements=normalized_requirements,
    )
    if selection_error is not None:
        return None, [selection_error]

    if not selected and source is not None:
        selection_error = _select_components_from_source_candidate(
            recommendation,
            source=source,
            component_index=component_index,
            selected=selected,
            quantities=quantities,
        )
        if selection_error is not None:
            return None, [selection_error]

    if not selected:
        return None, [f"{recommendation.recommendation_id}: component_candidate_ids required"]

    requirements = _first_requirements(normalized_requirements)
    product_group = _product_group_from_requirements(requirements)
    if (
        product_group == SERVER_PRODUCT_GROUP
        and _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE) is None
    ):
        return None, [f"{recommendation.recommendation_id}: platform is required"]

    source_missing_roles = (
        _string_list(source.row.get("missing_component_roles")) if source is not None else []
    )
    optional_component_roles = _optional_component_roles(
        selected=selected,
        user_request=user_request,
        normalized_requirements=normalized_requirements,
        source=source,
    )
    materialized_quantities = _materialize_build_quantities(
        recommendation,
        selected=selected,
        quantities=quantities,
        normalized_requirements=normalized_requirements,
        optional_component_roles=optional_component_roles,
    )
    warnings.extend(materialized_quantities.warnings)
    if materialized_quantities.error is not None:
        return None, [materialized_quantities.error]
    quantities = materialized_quantities.quantities
    quantity_details = materialized_quantities.quantity_details
    core_selected = {
        selection_key: candidate
        for selection_key, candidate in selected.items()
        if candidate.internal_role not in optional_component_roles
    }
    core_quantities = {
        selection_key: quantity
        for selection_key, quantity in quantities.items()
        if _selection_internal_role(selected, selection_key) not in optional_component_roles
    }
    if source_missing_roles and recommendation.source_type != PARTIAL_BUILD_CANDIDATE_TYPE:
        warnings.append(
            f"{recommendation.recommendation_id}: incomplete build downgraded to partial_build"
        )

    stock_warning = _validate_stock(
        core_selected,
        core_quantities,
        recommendation.recommendation_id,
    )
    if stock_warning:
        return None, [stock_warning]

    compatibility_warning = _validate_vendor_compatibility(
        selected,
        recommendation.recommendation_id,
    )
    if compatibility_warning:
        return None, [compatibility_warning]

    evidence_warning, evidence_warnings = _validate_evidence_compatibility(
        selected,
        recommendation.recommendation_id,
        normalized_requirements=normalized_requirements,
        evidence_by_component_id=evidence_by_component_id,
        relation_evidence=relation_evidence,
    )
    warnings.extend(evidence_warnings)
    if evidence_warning:
        return None, [evidence_warning]

    total_price, currency, price_warning = _calculate_total(
        selected,
        quantities,
        excluded_roles=optional_component_roles,
    )
    if price_warning:
        return None, [f"{recommendation.recommendation_id}: {price_warning}"]
    optional_total_price, optional_currency, optional_price_warning = (
        _calculate_optional_total(
            selected,
            quantities,
            optional_component_roles,
        )
    )
    if optional_price_warning:
        warnings.append(
            f"{recommendation.recommendation_id}: optional total not calculated: "
            f"{optional_price_warning}"
        )

    right_size_summary, right_size_warnings, right_size_selection_reason = _validate_right_size(
        selected=selected,
        quantities=quantities,
        component_index=component_index,
        normalized_requirements=normalized_requirements,
        recommendation_id=recommendation.recommendation_id,
        selection_text=" ".join(
            [
                recommendation.why_selected,
                recommendation.why_selected_short or "",
                recommendation.right_size_note or "",
            ]
        ),
    )
    warnings.extend(right_size_warnings)
    if right_size_selection_reason:
        warnings.append(f"{recommendation.recommendation_id}: {right_size_selection_reason}")

    mandatory_roles = _mandatory_roles(normalized_requirements)
    network_validation_warning = _validate_network_requirement(
        selected=selected,
        quantities=quantities,
        normalized_requirements=normalized_requirements,
        recommendation_id=recommendation.recommendation_id,
    )
    if network_validation_warning:
        return None, [network_validation_warning]
    hard_capability_validation = _hard_capability_validation(
        selected=selected,
        quantities=quantities,
        normalized_requirements=normalized_requirements,
    )
    hard_capability_validation.extend(
        _request_text_evidence_gate_validation_rows(
            selected=selected,
            user_request=user_request,
            normalized_requirements=normalized_requirements,
        )
    )
    hard_mismatch_rows = [
        row
        for row in hard_capability_validation
        if str(row.get("status") or "").strip() == "hard_mismatch"
    ]
    if hard_mismatch_rows:
        mismatch_reasons = _unique(
            [
                str(row.get("reason") or row.get("user_message") or "").strip()
                for row in hard_mismatch_rows
                if str(row.get("reason") or row.get("user_message") or "").strip()
            ]
        )
        detail = "; ".join(mismatch_reasons[:3]) or "hard capability mismatch"
        return None, [f"{recommendation.recommendation_id}: {detail}"]
    missing_capability_roles = [
        str(row.get("role") or row.get("capability_id") or "unknown")
        for row in hard_capability_validation
        if row.get("status") != "satisfied"
    ]
    missing_roles = _unique(
        [
            *_missing_mandatory_roles(
                mandatory_roles=mandatory_roles,
                selected=selected,
                normalized_requirements=normalized_requirements,
            ),
            *missing_capability_roles,
            *source_missing_roles,
            *_normalized_missing_roles(recommendation.what_is_missing),
        ]
    )
    included_component_keys = _selection_keys_in_role_order(selected)
    included_roles = _unique(
        [selected[selection_key].internal_role for selection_key in included_component_keys]
    )
    components = [
        _component_report_row(
            selected[selection_key],
            quantities[selection_key],
            quantity_detail=quantity_details.get(selection_key),
            evidence=evidence_by_component_id.get(
                selected[selection_key].component_candidate_id
            ),
            pricing_scope=(
                "optional_engineer_check"
                if selected[selection_key].internal_role in optional_component_roles
                else "core"
            ),
        )
        for selection_key in included_component_keys
    ]
    primary_component = components[0]
    core_components = [
        component
        for component in components
        if component.get("role") not in optional_component_roles
    ]
    available_quantity = _build_available_quantity(core_components)
    evidence_summary = _build_evidence_summary(
        selected,
        evidence_by_component_id=evidence_by_component_id,
        relation_evidence=relation_evidence,
        evidence_review=evidence_review,
    )
    if use_recommendation_evidence:
        evidence_summary = _coerce_llm_evidence_summary(
            recommendation.evidence_summary
        ) or _default_online_evidence_summary()
    checks = _critical_checks(
        [
            *recommendation.critical_checks,
            *recommendation.engineer_checks,
            *(
                _string_list(source.row.get("compatibility_warnings"))
                if source is not None
                else []
            ),
            *right_size_summary.get("compatibility_warnings", []),
            *materialized_quantities.warnings,
            *evidence_warnings,
            *_string_list(evidence_summary.get("engineering_checks")),
            *_optional_component_checks(optional_component_roles),
            *_classified_requirement_engineer_checks(requirements),
            *_platform_feature_validation_checks(hard_capability_validation),
            "LLM Composer не является финальной инженерной проверкой совместимости.",
        ],
        product_group=product_group,
    )
    completeness_status = "incomplete" if missing_roles else "complete"
    candidate_type = (
        PARTIAL_BUILD_CANDIDATE_TYPE
        if (
            completeness_status == "incomplete"
            or recommendation.source_type == PARTIAL_BUILD_CANDIDATE_TYPE
        )
        else BUILD_CANDIDATE_TYPE
    )
    completeness_label = _completeness_label(completeness_status, missing_roles)
    total_price_note = _total_price_note(
        missing_roles,
        optional_roles=optional_component_roles,
    )
    confidence_score = _confidence_score(recommendation.confidence)
    evidence_penalty = _evidence_confidence_penalty(evidence_summary)
    adjusted_score = max(
        1,
        confidence_score
        - int(right_size_summary["overfit_penalty"])
        - evidence_penalty,
    )
    primary_key = _selection_key_for_internal_role(
        selected,
        SERVER_PLATFORM_ROLE,
    ) or included_component_keys[0]
    display_name = _candidate_name(selected[primary_key])

    row = {
        "candidate_id": recommendation.recommendation_id,
        "recommendation_id": recommendation.recommendation_id,
        "product_group": product_group,
        "source_candidate_id": source.candidate_id if source is not None else None,
        "source_type": candidate_type,
        "candidate_type": candidate_type,
        "distributor_code": (source.row.get("distributor_code") if source is not None else None)
        or "llm",
        "item_id": (source.row.get("item_id") if source is not None else None)
        or recommendation.recommendation_id,
        "product_key": (source.row.get("product_key") if source is not None else None)
        or (source.candidate_id if source is not None else recommendation.recommendation_id),
        "part_number": primary_component.get("part_number"),
        "producer": primary_component.get("producer"),
        "category_id": primary_component.get("category_id"),
        "item_name": display_name,
        "display_name": display_name,
        "confidence_score": adjusted_score,
        "price_value": _json_decimal(total_price),
        "price_currency": currency,
        "available_quantity": available_quantity,
        "reservable_locations": (
            source.row.get("reservable_locations") if source is not None else 0
        )
        or 0,
        "matched_requirements": [recommendation.why_selected],
        "missing_requirements": missing_roles,
        "risk_flags": checks,
        "platform": primary_component,
        "components": components,
        "optional_components": [
            component
            for component in components
            if component.get("role") in optional_component_roles
        ],
        "optional_total_price_value": _json_decimal(optional_total_price),
        "optional_total_price_currency": optional_currency,
        "optional_component_roles": optional_component_roles,
        "engineer_check_component_roles": optional_component_roles,
        "component_summary": _materialized_component_summary(selected),
        "normalized_bom_quantities": {
            _prompt_role_for_selection_key(selection_key, selected[selection_key]): details
            for selection_key, details in quantity_details.items()
        },
        "total_price_value": _json_decimal(total_price),
        "total_price_currency": currency,
        "price_note": total_price_note,
        "missing_components": missing_roles,
        "hard_capability_validation": _jsonable(hard_capability_validation),
        "validation_hard_mismatches": [
            row
            for row in hard_capability_validation
            if row.get("status") == "hard_mismatch"
        ],
        "validation_unverified_requirements": [
            row
            for row in hard_capability_validation
            if row.get("status") == "unverified_hard_requirement"
        ],
        "missing_required_capabilities": [
            row for row in hard_capability_validation if row.get("status") != "satisfied"
        ],
        "compatibility_warnings": checks,
        "engineer_review_required": True,
        "completeness_status": completeness_status,
        "completeness_label": completeness_label,
        "included_component_roles": included_roles,
        "missing_component_roles": missing_roles,
        "excluded_from_total_roles": _unique([*missing_roles, *optional_component_roles]),
        "total_price_note": total_price_note,
        "score": adjusted_score,
        "rank_reason": [recommendation.why_selected],
        "llm_configurator": True,
        "decision": recommendation.decision,
        "proposal_role": recommendation.proposal_role,
        "recommendation_slot": _recommendation_slot_value(recommendation),
        "title": recommendation.title,
        "component_candidate_ids": {
            _prompt_role_for_selection_key(selection_key, selected[selection_key]): selected[
                selection_key
            ].component_candidate_id
            for selection_key in included_component_keys
        },
        "quantities": {
            _prompt_role_for_selection_key(
                selection_key,
                selected[selection_key],
            ): quantities[selection_key]
            for selection_key in included_component_keys
        },
        "why_selected": recommendation.why_selected,
        "why_selected_short": recommendation.why_selected_short,
        "commercial_tradeoff": recommendation.commercial_tradeoff,
        "requirement_fulfillment_summary": _jsonable(
            recommendation.requirement_fulfillment_summary
        ),
        "assumptions": _string_list(recommendation.assumptions),
        "what_is_missing": _string_list(recommendation.what_is_missing) or missing_roles,
        "overfit_reason": right_size_summary.get("overfit_reason"),
        "critical_checks": checks,
        "critical_risks": checks,
        "confidence": _adjusted_confidence_label(
            recommendation.confidence,
            penalty=evidence_penalty,
        ),
        "commercial_fit_confidence": _normalized_confidence_label(
            recommendation.confidence
        ),
        "engineering_confidence": _engineering_confidence_label(evidence_summary),
        "displayed_confidence": _displayed_confidence_text(
            recommendation.confidence,
            evidence_summary,
        ),
        "optimization_mode": right_size_summary["optimization_mode"],
        "requirement_fit": right_size_summary["requirement_fit"],
        "right_size_note": _validated_right_size_note(
            recommendation.right_size_note,
            right_size_summary,
        ),
        "cpu_over_requirement": right_size_summary.get("cpu_over_requirement"),
        "storage_over_requirement": right_size_summary.get("storage_over_requirement"),
        "ram_overage_gb": right_size_summary.get("ram_overage_gb"),
        "evidence_summary": evidence_summary,
        "evidence_review": dict(evidence_review or {}),
    }
    if right_size_selection_reason:
        row["_selection_skip_reason"] = "dominated_by_cheaper_equivalent"
        row["_selection_skip_message"] = right_size_selection_reason
    return row, warnings


def _validate_ready_server_hard_capabilities(
    source_row: Mapping[str, Any],
    *,
    normalized_requirements: Any,
    recommendation_id: str,
) -> str | None:
    requirements = _first_requirements(normalized_requirements)
    unsupported = _string_list(requirements.get("unsupported_or_unmapped_requirements"))
    if unsupported:
        return f"{recommendation_id}: unsupported hard requirement"
    components = _mapping_rows(source_row.get("components"))
    component_roles = {
        _normalize_role(component.get("role"))
        for component in components
        if _normalize_role(component.get("role")) is not None
    }
    for capability in _hard_capabilities(normalized_requirements):
        role = _role_from_capability(capability)
        if role in {None, SERVER_PLATFORM_ROLE, CPU_ROLE, RAM_ROLE, "storage"}:
            continue
        if role == NETWORK_ADAPTER_ROLE:
            facts = _ready_candidate_facts(source_row)
            if not network_facts_satisfy_requirement(facts, _network_requirement(requirements)):
                return f"{recommendation_id}: ready_server does not satisfy hard network capability"
            continue
        if role == POWER_SUPPLY_ROLE:
            if platform_power_bundle_satisfies(
                _ready_candidate_search_text(source_row),
                required_psu_count=_psu_count_requirement(requirements),
                raw_json=source_row.get("raw"),
            ):
                continue
            return f"{recommendation_id}: ready_server does not expose hard role {role}"
        if role not in component_roles:
            return f"{recommendation_id}: ready_server does not expose hard role {role}"
    return None


def _select_components_from_llm_payload(
    recommendation: LlmRecommendationPayload,
    *,
    component_index: dict[str, _IndexedComponentCandidate],
    selected: dict[str, _IndexedComponentCandidate],
    quantities: dict[str, int],
    normalized_requirements: Any,
) -> str | None:
    core_component_candidate_ids = (
        recommendation.component_candidate_ids
        or recommendation.selected_component_candidate_ids
    )
    selection_groups = [
        (core_component_candidate_ids, False),
        (recommendation.optional_component_candidate_ids, True),
        (recommendation.engineer_check_component_candidate_ids, True),
    ]
    for component_candidate_ids, allow_empty in selection_groups:
        selection_error = _select_components_from_candidate_id_map(
            recommendation,
            component_candidate_ids=component_candidate_ids,
            component_index=component_index,
            selected=selected,
            quantities=quantities,
            allow_empty=allow_empty,
            normalized_requirements=normalized_requirements,
        )
        if selection_error is not None:
            return selection_error
    return None


def _select_components_from_candidate_id_map(
    recommendation: LlmRecommendationPayload,
    *,
    component_candidate_ids: Mapping[str, str | None],
    component_index: dict[str, _IndexedComponentCandidate],
    selected: dict[str, _IndexedComponentCandidate],
    quantities: dict[str, int],
    allow_empty: bool,
    normalized_requirements: Any,
) -> str | None:
    for prompt_role, component_candidate_id_value in component_candidate_ids.items():
        component_candidate_id = str(component_candidate_id_value or "").strip()
        if not component_candidate_id:
            if allow_empty:
                continue
            return f"{recommendation.recommendation_id}: component_candidate_id missing"
        candidate = component_index.get(component_candidate_id)
        if candidate is None:
            return f"{recommendation.recommendation_id}: unknown component_candidate_id"

        expected_role = _role_from_component_id_key(
            prompt_role,
            candidate,
            normalized_requirements=normalized_requirements,
        )
        if expected_role is None:
            if _is_generic_component_role_alias(prompt_role):
                return f"{recommendation.recommendation_id}: component role mismatch"
            return f"{recommendation.recommendation_id}: unknown component role"
        if candidate.internal_role != expected_role:
            return f"{recommendation.recommendation_id}: component role mismatch"

        quantity = _quantity_for_role(
            recommendation.quantities,
            str(prompt_role),
            expected_role,
        )
        if quantity is None:
            quantity = _quantity_for_role(
                recommendation.quantities,
                PROMPT_ROLE_BY_INTERNAL_ROLE[expected_role],
                expected_role,
            )
        if quantity is None:
            quantity = _int_value(candidate.row.get("quantity_required"))
        validation_error = _validate_selected_quantity(
            recommendation_id=recommendation.recommendation_id,
            candidate=candidate,
            quantity=quantity,
        )
        if validation_error is not None:
            return validation_error
        selection_key = _selection_key_for_component(
            prompt_role,
            expected_role=expected_role,
            selected=selected,
        )
        selected[selection_key] = candidate
        quantities[selection_key] = quantity
    return None


def _select_components_from_source_candidate(
    recommendation: LlmRecommendationPayload,
    *,
    source: _IndexedStockCandidate,
    component_index: dict[str, _IndexedComponentCandidate],
    selected: dict[str, _IndexedComponentCandidate],
    quantities: dict[str, int],
) -> str | None:
    for component in _mapping_rows(source.row.get("components")):
        internal_role = _normalize_role(component.get("role"))
        if internal_role is None:
            return f"{recommendation.recommendation_id}: unknown component role"
        component_candidate_id = str(component.get("component_candidate_id") or "").strip()
        if not component_candidate_id:
            return f"{recommendation.recommendation_id}: component_candidate_id missing"
        candidate = component_index.get(component_candidate_id)
        if candidate is None:
            return f"{recommendation.recommendation_id}: unknown component_candidate_id"
        if candidate.internal_role != internal_role:
            return f"{recommendation.recommendation_id}: component role mismatch"
        quantity = _quantity_for_role(
            recommendation.quantities,
            PROMPT_ROLE_BY_INTERNAL_ROLE[internal_role],
            internal_role,
        )
        if quantity is None:
            quantity = _int_value(component.get("quantity_required"))
        validation_error = _validate_selected_quantity(
            recommendation_id=recommendation.recommendation_id,
            candidate=candidate,
            quantity=quantity,
        )
        if validation_error is not None:
            return validation_error
        selected[internal_role] = candidate
        quantities[internal_role] = quantity
    return None


def _selection_key_for_component(
    prompt_role: Any,
    *,
    expected_role: str,
    selected: Mapping[str, _IndexedComponentCandidate],
) -> str:
    role_key = str(prompt_role or "").strip()
    canonical_prompt_role = PROMPT_ROLE_BY_INTERNAL_ROLE.get(expected_role, expected_role)
    instance_role = _role_from_instance_key(role_key)
    if instance_role == expected_role and role_key not in {
        expected_role,
        canonical_prompt_role,
    }:
        return _unique_component_role_key(role_key, selected)
    if expected_role not in selected:
        return expected_role
    fallback = role_key if role_key else expected_role
    return _unique_component_role_key(fallback, selected)


def _selection_internal_role(
    selected: Mapping[str, _IndexedComponentCandidate],
    selection_key: str,
) -> str:
    candidate = selected.get(selection_key)
    return candidate.internal_role if candidate is not None else selection_key


def _selection_keys_for_role(
    selected: Mapping[str, _IndexedComponentCandidate],
    role: str,
) -> list[str]:
    return [
        selection_key
        for selection_key, candidate in selected.items()
        if candidate.internal_role == role
    ]


def _selected_candidate_for_role(
    selected: Mapping[str, _IndexedComponentCandidate],
    role: str,
) -> _IndexedComponentCandidate | None:
    exact = selected.get(role)
    if exact is not None and exact.internal_role == role:
        return exact
    for candidate in selected.values():
        if candidate.internal_role == role:
            return candidate
    return None


def _selected_candidates_for_role(
    selected: Mapping[str, _IndexedComponentCandidate],
    role: str,
) -> list[_IndexedComponentCandidate]:
    return [
        candidate for candidate in selected.values() if candidate.internal_role == role
    ]


def _selected_quantity_total_for_role(
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
    role: str,
) -> int | None:
    total = 0
    found = False
    for selection_key in _selection_keys_for_role(selected, role):
        quantity = _int_value(quantities.get(selection_key))
        if quantity is None:
            continue
        total += quantity
        found = True
    return total if found else None


def _selection_key_for_internal_role(
    selected: Mapping[str, _IndexedComponentCandidate],
    role: str,
) -> str | None:
    for selection_key, candidate in selected.items():
        if candidate.internal_role == role:
            return selection_key
    return None


def _selection_keys_in_role_order(
    selected: Mapping[str, _IndexedComponentCandidate],
) -> list[str]:
    positions = {selection_key: index for index, selection_key in enumerate(selected)}
    role_order = {role: index for index, role in enumerate(ROLE_ORDER)}
    return sorted(
        selected,
        key=lambda selection_key: (
            role_order.get(selected[selection_key].internal_role, len(role_order)),
            positions[selection_key],
        ),
    )


def _prompt_role_for_selection_key(
    selection_key: str,
    candidate: _IndexedComponentCandidate,
) -> str:
    canonical_prompt_role = PROMPT_ROLE_BY_INTERNAL_ROLE.get(
        candidate.internal_role,
        candidate.internal_role,
    )
    if selection_key in {candidate.internal_role, canonical_prompt_role}:
        return canonical_prompt_role
    if _role_from_instance_key(selection_key) == candidate.internal_role:
        return selection_key
    return selection_key


def _roles_with_explicit_split_quantities(
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
) -> set[str]:
    result: set[str] = set()
    for role in {candidate.internal_role for candidate in selected.values()}:
        selection_keys = _selection_keys_for_role(selected, role)
        if len(selection_keys) < 2:
            continue
        if all(
            (_int_value(quantities.get(selection_key)) or 0) > 0
            for selection_key in selection_keys
        ):
            result.add(role)
    return result


def _llm_quantity_for_selection(
    quantities: Mapping[str, int],
    selection_key: str,
    internal_role: str,
) -> int | None:
    value = _quantity_for_role(quantities, selection_key, internal_role)
    if value is None:
        value = _llm_quantity_for_internal_role(quantities, internal_role)
    return value


def _materialize_build_quantities(
    recommendation: LlmRecommendationPayload,
    *,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
    normalized_requirements: Any,
    optional_component_roles: Sequence[str],
) -> _MaterializedBomQuantities:
    requirements = _first_requirements(normalized_requirements)
    product_group = _product_group_from_requirements(requirements)
    server_quantity = _required_server_quantity(requirements)
    hard_roles = _expand_hard_quantity_roles(set(_mandatory_roles(normalized_requirements)))
    if product_group != STORAGE_PRODUCT_GROUP and requirements.get("storage_required") and not any(
        role in hard_roles for role in (SSD_ROLE, HDD_ROLE)
    ):
        hard_roles.update(
            role
            for role in (SSD_ROLE, HDD_ROLE)
            if _selected_candidate_for_role(selected, role) is not None
        )

    optional_roles = set(optional_component_roles)
    split_quantity_roles = _roles_with_explicit_split_quantities(selected, quantities)
    materialized: dict[str, int] = {}
    details: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for selection_key, candidate in selected.items():
        role = candidate.internal_role
        candidate_required = _int_value(candidate.row.get("quantity_required"))
        current_quantity = _int_value(quantities.get(selection_key))
        fallback_quantity = current_quantity or candidate_required or 1
        quantity = fallback_quantity
        per_server_quantity: int | None = None
        split_quantity_role = role in split_quantity_roles
        detail: dict[str, Any] = {
            "server_quantity": server_quantity,
            "quantity_source": "code_materialized",
            "hard_requirement": role in hard_roles and role not in optional_roles,
        }

        if role == SERVER_PLATFORM_ROLE:
            quantity = server_quantity
            per_server_quantity = 1
        elif role == CPU_ROLE and role in hard_roles:
            required_total = _required_cpu_total(requirements, server_quantity)
            if required_total is not None:
                quantity = required_total
                per_server_quantity = _required_cpu_per_server(
                    requirements,
                    server_quantity,
                    required_total,
                )
            required_cores = _required_cpu_cores(requirements)
            selected_cores = _candidate_cpu_cores(candidate)
            if required_cores is not None:
                if selected_cores is None:
                    return _MaterializedBomQuantities(
                        materialized,
                        details,
                        warnings,
                        error=(
                            f"{recommendation.recommendation_id}: CPU cores are unknown "
                            "for selected component"
                        ),
                    )
                if selected_cores < required_cores:
                    return _MaterializedBomQuantities(
                        materialized,
                        details,
                        warnings,
                        error=(
                            f"{recommendation.recommendation_id}: CPU cores are below "
                            "hard requirement"
                        ),
                    )
                detail["cpu_cores_per_cpu"] = selected_cores
        elif role == RAM_ROLE and role in hard_roles:
            contract_quantity = _required_contract_quantity_for_role(
                requirements,
                role,
                server_quantity=server_quantity,
            )
            if contract_quantity is not None and not split_quantity_role:
                quantity = contract_quantity
                per_server_quantity = max(1, ceil(quantity / server_quantity))
            required_ram_gb = _required_ram_gb_per_server(requirements)
            if required_ram_gb is not None:
                module_gb = _candidate_ram_module_gb(candidate)
                if module_gb is None:
                    return _MaterializedBomQuantities(
                        materialized,
                        details,
                        warnings,
                        error=(
                            f"{recommendation.recommendation_id}: RAM module capacity "
                            "is unknown"
                        ),
                    )
                if contract_quantity is None:
                    modules_per_server = max(ceil(required_ram_gb / module_gb), 1)
                    quantity = server_quantity * modules_per_server
                    per_server_quantity = modules_per_server
                detail["ram_module_capacity_gb"] = module_gb
                detail["ram_total_gb_per_server"] = module_gb * (
                    per_server_quantity or max(ceil(required_ram_gb / module_gb), 1)
                )
                detail["ram_required_gb_per_server"] = required_ram_gb
        elif (
            product_group != STORAGE_PRODUCT_GROUP
            and role in {SSD_ROLE, HDD_ROLE}
            and role in hard_roles
        ):
            contract_quantity = _required_contract_quantity_for_role(
                requirements,
                role,
                server_quantity=server_quantity,
            )
            if contract_quantity is not None and not split_quantity_role:
                quantity = contract_quantity
                per_server_quantity = max(1, ceil(quantity / server_quantity))
            else:
                if not split_quantity_role:
                    drives_per_server = _required_storage_qty_per_server(requirements)
                    quantity = server_quantity * drives_per_server
                    per_server_quantity = drives_per_server
            required_tb = _required_storage_tb(requirements)
            selected_tb = _candidate_storage_tb(candidate)
            if required_tb is not None:
                if selected_tb is None:
                    return _MaterializedBomQuantities(
                        materialized,
                        details,
                        warnings,
                        error=(
                            f"{recommendation.recommendation_id}: storage capacity "
                            "is unknown for selected drive"
                        ),
                    )
                if selected_tb < required_tb:
                    return _MaterializedBomQuantities(
                        materialized,
                        details,
                        warnings,
                        error=(
                            f"{recommendation.recommendation_id}: storage capacity "
                            "is below hard requirement"
                        ),
                    )
                detail["storage_capacity_tb"] = selected_tb
                detail["storage_required_capacity_tb"] = required_tb
        elif role == NETWORK_ADAPTER_ROLE and role in hard_roles:
            facts = _network_facts_for_candidate(candidate)
            required_quantity = required_network_adapter_quantity(
                facts,
                _network_requirement(requirements),
                server_quantity=server_quantity,
            )
            if required_quantity is None:
                return _MaterializedBomQuantities(
                    materialized,
                    details,
                    warnings,
                    error=(
                        f"{recommendation.recommendation_id}: network_adapter "
                        "port count is unknown"
                    ),
                )
            quantity = required_quantity
            per_server_quantity = max(1, ceil(quantity / server_quantity))
            detail["network_ports_per_adapter"] = _int_value(facts.get("ports_count"))
            detail["network_required_ports_per_server"] = _int_value(
                _network_requirement(requirements).get("min_ports_per_server")
            )
            detail["network_speed"] = facts.get("speed")
            detail["network_media"] = facts.get("media")
        elif _is_storage_product_role(role) and role in hard_roles:
            quantity = _storage_product_role_quantity(
                role=role,
                candidate=candidate,
                requirements=requirements,
                system_quantity=server_quantity,
            )
            if role == STORAGE_SYSTEM_ROLE:
                per_server_quantity = 1
            detail["storage_role"] = role
        elif (
            product_group == NETWORK_PRODUCT_GROUP
            and _is_network_product_role(role)
            and role in hard_roles
        ):
            quantity = _network_product_role_quantity(
                role=role,
                candidate=candidate,
                requirements=requirements,
                server_quantity=server_quantity,
            )
            if role in {SWITCH_ROLE, ROUTER_ROLE, FIREWALL_ROLE, ACCESS_POINT_ROLE}:
                per_server_quantity = 1
            detail["network_role"] = role
        elif role in {STORAGE_CONTROLLER_ROLE, NETWORK_ADAPTER_ROLE}:
            quantity = candidate_required or server_quantity
            per_server_quantity = max(1, quantity // server_quantity)
        elif role == POWER_SUPPLY_ROLE and role in hard_roles:
            contract_quantity = _required_contract_quantity_for_role(
                requirements,
                role,
                server_quantity=server_quantity,
            )
            if contract_quantity is not None and not split_quantity_role:
                quantity = contract_quantity
                per_server_quantity = max(1, ceil(quantity / server_quantity))
            else:
                psu_per_server = _psu_count_requirement(requirements) or 1
                if not split_quantity_role:
                    quantity = server_quantity * psu_per_server
                    per_server_quantity = psu_per_server
        elif role == CABLE_ROLE and role in hard_roles:
            contract_quantity = _required_contract_quantity_for_role(
                requirements,
                role,
                server_quantity=server_quantity,
            )
            if contract_quantity is not None and not split_quantity_role:
                quantity = contract_quantity
                per_server_quantity = max(1, ceil(quantity / server_quantity))
            else:
                quantity = candidate_required or fallback_quantity
        else:
            contract_quantity = (
                _required_contract_quantity_for_role(
                    requirements,
                    role,
                    server_quantity=server_quantity,
                )
                if role in hard_roles
                else None
            )
            quantity = contract_quantity or candidate_required or fallback_quantity
            if quantity == server_quantity:
                per_server_quantity = 1

        required_total = _required_contract_quantity_for_role(
            requirements,
            role,
            server_quantity=server_quantity,
        )
        if role in hard_roles and required_total is not None and split_quantity_role:
            selected_total = _selected_quantity_total_for_role(
                selected,
                quantities,
                role,
            )
            if selected_total is not None and selected_total < required_total:
                return _MaterializedBomQuantities(
                    materialized,
                    details,
                    warnings,
                    error=(
                        f"{recommendation.recommendation_id}: {role} quantity below "
                        f"hard requirement ({selected_total} < {required_total})"
                    ),
                )

        llm_quantity = _llm_quantity_for_selection(
            recommendation.quantities,
            selection_key,
            role,
        )
        if llm_quantity is not None and llm_quantity != quantity:
            warnings.append(
                f"Количество {_role_label(role)} нормализовано до "
                f"минимально достаточного: {quantity} шт. вместо {llm_quantity}."
            )
            detail["llm_quantity"] = llm_quantity
            detail["quantity_normalized"] = True

        materialized[selection_key] = quantity
        detail["total_quantity"] = quantity
        if per_server_quantity is not None:
            detail["per_server_quantity"] = per_server_quantity
        details[selection_key] = detail

    return _MaterializedBomQuantities(materialized, details, warnings)


def _is_network_product_role(role: str) -> bool:
    return role in {
        SWITCH_ROLE,
        ROUTER_ROLE,
        FIREWALL_ROLE,
        ACCESS_POINT_ROLE,
        TRANSCEIVER_ROLE,
        DAC_CABLE_ROLE,
        CABLE_ROLE,
        LICENSE_ROLE,
        SUPPORT_ROLE,
        POWER_SUPPLY_ROLE,
        STACKING_MODULE_ROLE,
        OTHER_ACCESSORY_ROLE,
    }


def _is_storage_product_role(role: str) -> bool:
    return role in {
        STORAGE_SYSTEM_ROLE,
        STORAGE_ARRAY_CONTROLLER_ROLE,
        CONTROLLER_MODULE_ROLE,
        DISK_SHELF_ROLE,
        DRIVE_ROLE,
        SSD_ROLE,
        HDD_ROLE,
        CACHE_ROLE,
        HOST_PORT_ROLE,
        PROTOCOL_MODULE_ROLE,
        TRANSCEIVER_ROLE,
        CABLE_ROLE,
        LICENSE_ROLE,
        SUPPORT_ROLE,
        POWER_SUPPLY_ROLE,
        RAIL_KIT_ROLE,
        OTHER_ACCESSORY_ROLE,
    }


def _storage_product_role_quantity(
    *,
    role: str,
    candidate: _IndexedComponentCandidate,
    requirements: Mapping[str, Any],
    system_quantity: int,
) -> int:
    parsed = _role_parsed_requirements(requirements, role)
    quantity = _int_value(
        parsed.get("count")
        or parsed.get("quantity")
        or parsed.get("device_count")
        or parsed.get("drive_count")
    )
    if quantity is not None and quantity > 0:
        return quantity
    candidate_required = _int_value(candidate.row.get("quantity_required"))
    if candidate_required is not None and candidate_required > 0:
        return candidate_required
    if role == STORAGE_SYSTEM_ROLE:
        return system_quantity
    if role in {LICENSE_ROLE, SUPPORT_ROLE, RAIL_KIT_ROLE}:
        return system_quantity
    if role in {STORAGE_ARRAY_CONTROLLER_ROLE, CONTROLLER_MODULE_ROLE}:
        return _int_value(requirements.get("controller_count")) or system_quantity
    if role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}:
        return _int_value(requirements.get("drive_count")) or system_quantity
    if role == DISK_SHELF_ROLE:
        return _int_value(requirements.get("shelf_count")) or system_quantity
    if role in {HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE, TRANSCEIVER_ROLE, CABLE_ROLE}:
        return _int_value(requirements.get("host_port_count")) or system_quantity
    return 1


def _network_product_role_quantity(
    *,
    role: str,
    candidate: _IndexedComponentCandidate,
    requirements: Mapping[str, Any],
    server_quantity: int,
) -> int:
    parsed = _role_parsed_requirements(requirements, role)
    quantity = _int_value(
        parsed.get("count")
        or parsed.get("quantity")
        or parsed.get("device_count")
    )
    if quantity is not None and quantity > 0:
        return quantity
    candidate_required = _int_value(candidate.row.get("quantity_required"))
    if candidate_required is not None and candidate_required > 0:
        return candidate_required
    if role in {
        SWITCH_ROLE,
        ROUTER_ROLE,
        FIREWALL_ROLE,
        ACCESS_POINT_ROLE,
        LICENSE_ROLE,
        SUPPORT_ROLE,
        STACKING_MODULE_ROLE,
    }:
        return server_quantity
    if role == TRANSCEIVER_ROLE:
        for device_role in (SWITCH_ROLE, ROUTER_ROLE, FIREWALL_ROLE, ACCESS_POINT_ROLE):
            device_requirements = _role_parsed_requirements(requirements, device_role)
            uplink_count = _int_value(device_requirements.get("uplink_count"))
            if uplink_count is not None and uplink_count > 0:
                return uplink_count * server_quantity
    return 1


def _role_parsed_requirements(
    requirements: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any]:
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        by_role = role_plan.get("requirements_by_role")
        if isinstance(by_role, Mapping):
            row = by_role.get(role)
            if isinstance(row, Mapping):
                return row
    for capability in _mapping_rows(requirements.get("required_capabilities")):
        if str(capability.get("role") or "").strip() != role:
            continue
        parsed = capability.get("parsed_requirements")
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def _required_server_quantity(requirements: Mapping[str, Any]) -> int:
    quantity = _int_value(
        requirements.get("server_qty")
        or requirements.get("device_qty")
        or requirements.get("system_qty")
    )
    return quantity if quantity is not None and quantity > 0 else 1


def _required_cpu_total(
    requirements: Mapping[str, Any],
    server_quantity: int,
) -> int | None:
    total = _int_value(requirements.get("total_cpu_required"))
    if total is not None and total > 0:
        return total
    per_server = _int_value(requirements.get("cpu_per_server"))
    if per_server is not None and per_server > 0:
        return server_quantity * per_server
    return None


def _required_cpu_per_server(
    requirements: Mapping[str, Any],
    server_quantity: int,
    total_cpu_required: int,
) -> int:
    per_server = _int_value(requirements.get("cpu_per_server"))
    if per_server is not None and per_server > 0:
        return per_server
    return max(1, ceil(total_cpu_required / server_quantity))


def _required_ram_gb_per_server(requirements: Mapping[str, Any]) -> int | None:
    return _int_value(
        requirements.get("ram_gb_per_server")
        or requirements.get("ram_min_gb_per_server")
        or requirements.get("ram_min_gb")
    )


def _candidate_ram_module_gb(candidate: _IndexedComponentCandidate) -> int | None:
    value = candidate.row.get("ram_module_capacity_gb")
    if value in (None, ""):
        value = _fact(candidate, "ram_capacity_gb")
    return _int_value(value)


def _required_storage_qty_per_server(requirements: Mapping[str, Any]) -> int:
    quantity = _int_value(
        requirements.get("storage_qty_per_server")
        or requirements.get("storage_quantity_per_server")
        or requirements.get("storage_count_per_server")
    )
    return quantity if quantity is not None and quantity > 0 else 1


def _required_contract_quantity_for_role(
    requirements: Mapping[str, Any],
    role: str,
    *,
    server_quantity: int,
) -> int | None:
    by_role = _contract_quantities_from_requirements(requirements)
    value = by_role.get(role)
    if value is None and role in {SSD_ROLE, HDD_ROLE, DRIVE_ROLE}:
        value = by_role.get("storage")
    quantity = _contract_role_count(value)
    if quantity is not None and quantity > 0:
        return quantity
    if isinstance(value, Mapping):
        per_server = _int_value(
            value.get("count_per_server")
            or value.get("quantity_per_server")
            or value.get("per_server")
        )
        if per_server is not None and per_server > 0:
            return server_quantity * per_server
    if role in {SSD_ROLE, HDD_ROLE, DRIVE_ROLE} and (
        requirements.get("storage_required")
        or requirements.get("storage_qty_per_server")
        or requirements.get("storage_quantity_per_server")
        or requirements.get("storage_count_per_server")
    ):
        return server_quantity * _required_storage_qty_per_server(requirements)
    return None


def _expand_hard_quantity_roles(hard_roles: set[str]) -> set[str]:
    result = set(hard_roles)
    if "storage" in result or DRIVE_ROLE in result:
        result.update({DRIVE_ROLE, SSD_ROLE, HDD_ROLE})
    elif SSD_ROLE in result or HDD_ROLE in result:
        result.add(DRIVE_ROLE)
    return result


def _contract_quantities_from_requirements(
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        str(role): value
        for role, value in _safe_mapping(
            requirements.get("required_quantities_by_role")
        ).items()
    }
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        for role, value in _safe_mapping(role_plan.get("requirements_by_role")).items():
            result.setdefault(str(role), value)
    return result


def _llm_quantity_for_internal_role(
    quantities: Mapping[str, int],
    internal_role: str,
) -> int | None:
    prompt_role = PROMPT_ROLE_BY_INTERNAL_ROLE.get(internal_role, internal_role)
    value = _quantity_for_role(quantities, prompt_role, internal_role)
    if value is None and internal_role in {
        STORAGE_SYSTEM_ROLE,
        DRIVE_ROLE,
        SSD_ROLE,
        HDD_ROLE,
    }:
        value = _quantity_for_role(quantities, "storage", internal_role)
    return value


def _role_from_component_id_key(
    prompt_role: Any,
    candidate: _IndexedComponentCandidate,
    *,
    normalized_requirements: Any = None,
) -> str | None:
    role_text = str(prompt_role or "").strip()
    if role_text == "storage":
        if candidate.internal_role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE, STORAGE_SYSTEM_ROLE}:
            return candidate.internal_role
        return None
    if _is_generic_component_role_alias(role_text):
        return _generic_alias_role_from_matrix_candidate(
            candidate,
            normalized_requirements=normalized_requirements,
        )
    normalized_role = _normalize_role(role_text)
    if normalized_role is not None:
        return normalized_role
    return _role_from_instance_key(role_text)


def _role_from_instance_key(role_text: Any) -> str | None:
    text = str(role_text or "").strip()
    if "_" not in text and "-" not in text:
        return None
    normalized_text = text.replace("-", "_").casefold()
    aliases: list[tuple[str, str]] = []
    for prompt_role, internal_role in INTERNAL_ROLE_BY_PROMPT_ROLE.items():
        aliases.append((str(prompt_role).replace("-", "_").casefold(), internal_role))
    for internal_role, prompt_role in PROMPT_ROLE_BY_INTERNAL_ROLE.items():
        aliases.append((str(internal_role).replace("-", "_").casefold(), internal_role))
        aliases.append((str(prompt_role).replace("-", "_").casefold(), internal_role))
    aliases = sorted(set(aliases), key=lambda item: len(item[0]), reverse=True)
    for alias, internal_role in aliases:
        if alias and normalized_text.startswith(f"{alias}_"):
            return internal_role
    return None


def _is_generic_component_role_alias(value: Any) -> bool:
    return str(value or "").strip().casefold() in GENERIC_COMPONENT_ROLE_ALIASES


def _generic_alias_role_from_matrix_candidate(
    candidate: _IndexedComponentCandidate,
    *,
    normalized_requirements: Any,
) -> str | None:
    requirements = _first_requirements(normalized_requirements)
    product_group = _product_group_from_requirements(requirements)
    actual_role = candidate.internal_role

    if product_group == SERVER_PRODUCT_GROUP:
        return SERVER_PLATFORM_ROLE if actual_role == SERVER_PLATFORM_ROLE else None
    if product_group == NETWORK_PRODUCT_GROUP:
        allowed_base_roles = NETWORK_BASE_DEVICE_ROLES
    elif product_group == STORAGE_PRODUCT_GROUP:
        allowed_base_roles = {STORAGE_SYSTEM_ROLE}
    else:
        return None

    if actual_role not in allowed_base_roles:
        return None
    requested_roles = _requested_roles_for_generic_aliases(normalized_requirements)
    if requested_roles and actual_role not in requested_roles:
        return None
    return actual_role


def _requested_roles_for_generic_aliases(normalized_requirements: Any) -> set[str]:
    requirements = _first_requirements(normalized_requirements)
    roles = set(_mandatory_roles(normalized_requirements))
    roles.update(_string_list(requirements.get("required_roles")))
    roles.update(_string_list(requirements.get("optional_roles")))
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        roles.update(_string_list(role_plan.get("required_roles")))
        roles.update(_string_list(role_plan.get("optional_roles")))
        optional_capabilities = _mapping_rows(role_plan.get("optional_capabilities"))
    else:
        optional_capabilities = []
    optional_capabilities.extend(_mapping_rows(requirements.get("optional_capabilities")))
    for capability in optional_capabilities:
        role = _role_from_capability(capability)
        if role and role != "storage":
            roles.add(role)
    return {role for role in roles if role}


def _validate_selected_quantity(
    *,
    recommendation_id: str,
    candidate: _IndexedComponentCandidate,
    quantity: int | None,
) -> str | None:
    if quantity is None or quantity <= 0:
        return f"{recommendation_id}: quantity must be positive"
    return None


def _materialized_component_summary(
    selected: Mapping[str, _IndexedComponentCandidate],
) -> dict[str, str]:
    summary = {
        "platform": _candidate_name(_selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE))
        if _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE) is not None
        else "",
        "cpu": _candidate_name(_selected_candidate_for_role(selected, CPU_ROLE))
        if _selected_candidate_for_role(selected, CPU_ROLE) is not None
        else "",
        "ram": _candidate_name(_selected_candidate_for_role(selected, RAM_ROLE))
        if _selected_candidate_for_role(selected, RAM_ROLE) is not None
        else "",
        "storage": " / ".join(
            _candidate_name(candidate)
            for role in (STORAGE_SYSTEM_ROLE, DRIVE_ROLE, SSD_ROLE, HDD_ROLE)
            for candidate in _selected_candidates_for_role(selected, role)
        ),
        "network": _candidate_name(
            _selected_candidate_for_role(selected, NETWORK_ADAPTER_ROLE)
        )
        if _selected_candidate_for_role(selected, NETWORK_ADAPTER_ROLE) is not None
        else "",
    }
    for role in (
        SWITCH_ROLE,
        ROUTER_ROLE,
        FIREWALL_ROLE,
        ACCESS_POINT_ROLE,
        STORAGE_ARRAY_CONTROLLER_ROLE,
        CONTROLLER_MODULE_ROLE,
        DISK_SHELF_ROLE,
        CACHE_ROLE,
        HOST_PORT_ROLE,
        PROTOCOL_MODULE_ROLE,
        STORAGE_CONTROLLER_ROLE,
        GPU_ROLE,
        TRANSCEIVER_ROLE,
        DAC_CABLE_ROLE,
        CABLE_ROLE,
        POWER_SUPPLY_ROLE,
        RAIL_KIT_ROLE,
        LICENSE_ROLE,
        SUPPORT_ROLE,
        STACKING_MODULE_ROLE,
        OTHER_ACCESSORY_ROLE,
    ):
        candidates = _selected_candidates_for_role(selected, role)
        if candidates:
            summary[role] = " / ".join(_candidate_name(candidate) for candidate in candidates)
    return summary


def _validate_right_size(
    *,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
    component_index: Mapping[str, _IndexedComponentCandidate],
    normalized_requirements: Any,
    recommendation_id: str,
    selection_text: str,
) -> tuple[dict[str, Any], list[str], str | None]:
    requirements = _first_requirements(normalized_requirements)
    optimization_mode = str(
        requirements.get("optimization_mode") or OPTIMIZATION_MODE_COST_MINIMAL_FIT
    )
    warnings: list[str] = []
    overfit_reasons: list[str] = []
    exact_fit_notes: list[str] = []
    overfit_penalty = 0
    compatibility_warnings: list[str] = []
    selection_reason: str | None = None

    for role in (CPU_ROLE, RAM_ROLE, DRIVE_ROLE, SSD_ROLE, HDD_ROLE):
        for selection_key in _selection_keys_for_role(selected, role):
            candidate = selected[selection_key]
            fit_label = str(candidate.row.get("fit_label") or FIT_UNKNOWN)
            selected_quantity = quantities[selection_key]
            if fit_label == FIT_EXCESSIVE_OVERFIT:
                reject_reason = _right_sized_alternative_reject_reason(
                    role=role,
                    selected_candidate=candidate,
                    selected_quantity=selected_quantity,
                    component_index=component_index,
                    selected=selected,
                    requirements=requirements,
                )
                if reject_reason:
                    selection_reason = selection_reason or reject_reason
                overfit_penalty += 18
                overfit_reasons.append(
                    _evidence_based_over_requirement_reason(
                        role=role,
                        selected_candidate=candidate,
                        selected_quantity=selected_quantity,
                        component_index=component_index,
                        selected=selected,
                        requirements=requirements,
                    )
                )
            elif fit_label == FIT_ACCEPTABLE_OVERFIT:
                reject_reason = _right_sized_alternative_reject_reason(
                    role=role,
                    selected_candidate=candidate,
                    selected_quantity=selected_quantity,
                    component_index=component_index,
                    selected=selected,
                    requirements=requirements,
                )
                if reject_reason:
                    selection_reason = selection_reason or reject_reason
                overfit_penalty += 5
                overfit_reasons.append(
                    _evidence_based_over_requirement_reason(
                        role=role,
                        selected_candidate=candidate,
                        selected_quantity=selected_quantity,
                        component_index=component_index,
                        selected=selected,
                        requirements=requirements,
                    )
                )
            elif fit_label == FIT_EXACT_OR_CLOSE:
                exact_fit_note = _exact_requirement_note(
                    role=role,
                    selected_candidate=candidate,
                    requirements=requirements,
                )
                if exact_fit_note:
                    exact_fit_notes.append(exact_fit_note)
            elif fit_label == FIT_UNKNOWN:
                overfit_penalty += 3

    if overfit_reasons and not _text_has_overfit_reason(selection_text):
        warnings.append(
            f"{recommendation_id}: right-size reason was added by deterministic validator"
        )
        compatibility_warnings.append(
            "Компонент выше требования; перед КП проверьте альтернативы в матрице компонентов."
        )

    overfit_reasons = _unique([reason for reason in overfit_reasons if reason])
    if overfit_reasons:
        overfit_reason = "; ".join(overfit_reasons)
        right_size_note = f"Подбор: {overfit_reason}"
        requirement_fit = "overfit_with_reason"
    elif exact_fit_notes:
        overfit_reason = None
        right_size_note = "Подбор: " + " ".join(_unique(exact_fit_notes))
        requirement_fit = "minimal_fit"
    else:
        overfit_reason = None
        right_size_note = "Подбор: минимально подходящий по требованиям"
        requirement_fit = "minimal_fit"

    summary = {
        "optimization_mode": optimization_mode,
        "requirement_fit": requirement_fit,
        "right_size_note": right_size_note,
        "overfit_reason": overfit_reason,
        "overfit_penalty": overfit_penalty,
        "compatibility_warnings": compatibility_warnings,
        "cpu_over_requirement": _selected_int(selected, CPU_ROLE, "cpu_over_requirement"),
        "storage_over_requirement": _selected_float(
            _selected_candidate_for_role(selected, DRIVE_ROLE)
            or _selected_candidate_for_role(selected, SSD_ROLE)
            or _selected_candidate_for_role(selected, HDD_ROLE),
            "storage_over_requirement",
        ),
        "ram_overage_gb": _selected_int(selected, RAM_ROLE, "ram_over_requirement_gb"),
    }
    return summary, warnings, selection_reason


def _right_sized_alternative_reject_reason(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    selected_quantity: int,
    component_index: Mapping[str, _IndexedComponentCandidate],
    selected: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> str | None:
    selected_line_price = _line_price(selected_candidate, selected_quantity)
    selected_name = _candidate_name(selected_candidate)
    if selected_line_price is None:
        return None
    alternatives = _closer_right_size_alternatives(
        role=role,
        selected_candidate=selected_candidate,
        component_index=component_index,
        selected=selected,
        requirements=requirements,
    )
    for candidate in alternatives.valid:
        if not _has_sufficient_stock(candidate, selected_quantity):
            continue
        alternative_line_price = _line_price(candidate, selected_quantity)
        if alternative_line_price is None:
            continue
        if alternative_line_price <= selected_line_price:
            return (
                "excessive overfit rejected: closer cheaper stocked alternative exists "
                f"for {role} ({_candidate_name(candidate)} instead of {selected_name})"
            )
    return None


def _evidence_based_over_requirement_reason(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    selected_quantity: int,
    component_index: Mapping[str, _IndexedComponentCandidate],
    selected: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> str:
    if role == CPU_ROLE:
        reason = _cpu_over_requirement_reason(
            selected_candidate=selected_candidate,
            selected_quantity=selected_quantity,
            component_index=component_index,
            selected=selected,
            requirements=requirements,
        )
        if reason:
            return reason
    if role in {SSD_ROLE, HDD_ROLE}:
        reason = _storage_over_requirement_reason(
            role=role,
            selected_candidate=selected_candidate,
            selected_quantity=selected_quantity,
            component_index=component_index,
            selected=selected,
            requirements=requirements,
        )
        if reason:
            return reason
    return _allowed_overfit_reason(
        role=role,
        selected_candidate=selected_candidate,
        selected_quantity=selected_quantity,
        component_index=component_index,
    )


def _allowed_overfit_reason(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    selected_quantity: int,
    component_index: Mapping[str, _IndexedComponentCandidate],
) -> str:
    selected_line_price = _line_price(selected_candidate, selected_quantity)
    closer_candidates = [
        candidate
        for candidate in component_index.values()
        if candidate.internal_role == role
        and candidate.component_candidate_id != selected_candidate.component_candidate_id
        and str(candidate.row.get("fit_label") or "") in {
            FIT_EXACT_OR_CLOSE,
            FIT_ACCEPTABLE_OVERFIT,
        }
    ]
    stocked_closer = [
        candidate
        for candidate in closer_candidates
        if _has_sufficient_stock(candidate, selected_quantity)
    ]
    if not closer_candidates:
        return "нет более близкого подходящего варианта в матрице компонентов"
    if not stocked_closer:
        return "у более близких компонентов недостаточный или неизвестный складской остаток"
    if selected_line_price is not None:
        closer_prices = [
            price
            for candidate in stocked_closer
            if (price := _line_price(candidate, selected_quantity)) is not None
        ]
        if closer_prices and selected_line_price < min(closer_prices):
            return "выбранный компонент дешевле более близких альтернатив"
    return _candidate_fit_reason(selected_candidate)


def _cpu_over_requirement_reason(
    *,
    selected_candidate: _IndexedComponentCandidate,
    selected_quantity: int,
    component_index: Mapping[str, _IndexedComponentCandidate],
    selected: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> str:
    required_cores = _required_cpu_cores(requirements)
    selected_cores = _candidate_cpu_cores(selected_candidate)
    if required_cores is None or selected_cores is None or selected_cores <= required_cores:
        return ""
    base = (
        "CPU выше минимального требования: "
        f"{_cores_text(selected_cores)} вместо {_cores_text(required_cores)}."
    )
    alternatives = _closer_right_size_alternatives(
        role=CPU_ROLE,
        selected_candidate=selected_candidate,
        component_index=component_index,
        selected=selected,
        requirements=requirements,
    )
    no_valid_text = _cpu_no_closer_option_text(
        requirements=requirements,
        required_cores=required_cores,
        selected_quantity=selected_quantity,
    )
    return _evidence_reason_from_alternatives(
        base=base,
        selected_candidate=selected_candidate,
        selected_quantity=selected_quantity,
        alternatives=alternatives,
        no_valid_text=no_valid_text,
        incompatible_text="",
        insufficient_stock_text=(
            "Более близкий по ядрам вариант не закрывает требуемое количество на складе."
        ),
        cheaper_selected_text=(
            "Более близкий по ядрам вариант есть, но он дороже "
            "по текущим складским ценам."
        ),
        unknown_text="Перед КП проверьте альтернативы CPU в матрице компонентов.",
    )


def _cpu_no_closer_option_text(
    *,
    requirements: Mapping[str, Any],
    required_cores: int,
    selected_quantity: int,
) -> str:
    vendor = str(requirements.get("cpu_vendor_preference") or "").strip()
    family = str(requirements.get("cpu_family_preference") or "").strip()
    family_text = " ".join(
        item
        for item in [
            vendor if vendor != UNKNOWN_FACT else "",
            family if family != UNKNOWN_FACT else "",
        ]
        if item
    )
    prefix = f"Среди складских {family_text}" if family_text else "В матрице"
    platform_text = " для этой платформы" if family_text else ""
    return (
        f"{prefix}{platform_text} не найден более близкий к {required_cores} ядрам "
        f"вариант с остатком на {selected_quantity} CPU и без явных конфликтов совместимости."
    )


def _storage_over_requirement_reason(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    selected_quantity: int,
    component_index: Mapping[str, _IndexedComponentCandidate],
    selected: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> str:
    required_tb = _required_storage_tb(requirements)
    selected_tb = _candidate_storage_tb(selected_candidate)
    if required_tb is None or selected_tb is None or selected_tb <= required_tb:
        return ""
    role_label = "SSD" if role == SSD_ROLE else "HDD"
    base = (
        f"{role_label} выше минимального требования: "
        f"{_tb_text(selected_tb)} вместо {_tb_text(required_tb)}."
    )
    alternatives = _closer_right_size_alternatives(
        role=role,
        selected_candidate=selected_candidate,
        component_index=component_index,
        selected=selected,
        requirements=requirements,
    )
    return _evidence_reason_from_alternatives(
        base=base,
        selected_candidate=selected_candidate,
        selected_quantity=selected_quantity,
        alternatives=alternatives,
        no_valid_text=(
            f"В матрице не найден более близкий вариант {role_label} "
            "с достаточным остатком и без явных конфликтов с платформой."
        ),
        incompatible_text="",
        insufficient_stock_text=(
            f"Более близкий вариант {role_label} не закрывает требуемое количество на складе."
        ),
        cheaper_selected_text=(
            f"Более близкий вариант {role_label} есть, но он дороже "
            "по текущим складским ценам."
        ),
        unknown_text=f"Перед КП проверьте альтернативы {role_label} в матрице компонентов.",
    )


def _evidence_reason_from_alternatives(
    *,
    base: str,
    selected_candidate: _IndexedComponentCandidate,
    selected_quantity: int,
    alternatives: _CloserAlternatives,
    no_valid_text: str,
    incompatible_text: str,
    insufficient_stock_text: str,
    cheaper_selected_text: str,
    unknown_text: str,
) -> str:
    if alternatives.incompatible and not alternatives.valid:
        return f"{base} {incompatible_text or no_valid_text}"
    stocked = [
        candidate
        for candidate in alternatives.valid
        if _has_sufficient_stock(candidate, selected_quantity)
    ]
    if not stocked:
        if alternatives.valid:
            return f"{base} {insufficient_stock_text}"
        return f"{base} {no_valid_text}"
    selected_line_price = _line_price(selected_candidate, selected_quantity)
    stocked_prices = [
        price
        for candidate in stocked
        if (price := _line_price(candidate, selected_quantity)) is not None
    ]
    if selected_line_price is not None and stocked_prices:
        if selected_line_price < min(stocked_prices):
            return f"{base} {cheaper_selected_text}"
        if any(price <= selected_line_price for price in stocked_prices):
            return f"{base} {unknown_text}"
    return f"{base} {unknown_text}"


@dataclass(frozen=True)
class _CloserAlternatives:
    valid: list[_IndexedComponentCandidate]
    incompatible: list[_IndexedComponentCandidate]


def _closer_right_size_alternatives(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    component_index: Mapping[str, _IndexedComponentCandidate],
    selected: Mapping[str, _IndexedComponentCandidate],
    requirements: Mapping[str, Any],
) -> _CloserAlternatives:
    selected_distance = _right_size_distance(role, selected_candidate, requirements)
    if selected_distance is None or selected_distance <= 0:
        return _CloserAlternatives(valid=[], incompatible=[])

    valid: list[_IndexedComponentCandidate] = []
    incompatible: list[_IndexedComponentCandidate] = []
    for candidate in component_index.values():
        if candidate.internal_role != role:
            continue
        if candidate.component_candidate_id == selected_candidate.component_candidate_id:
            continue
        distance = _right_size_distance(role, candidate, requirements)
        if distance is None or distance >= selected_distance:
            continue
        if role == CPU_ROLE and not _cpu_matches_preferences(candidate, requirements):
            continue
        if role in {SSD_ROLE, HDD_ROLE} and not _storage_matches_preferences(
            candidate,
            requirements,
        ):
            continue
        if _candidate_incompatible_with_selection(
            role=role,
            candidate=candidate,
            selected=selected,
        ):
            incompatible.append(candidate)
            continue
        valid.append(candidate)
    return _CloserAlternatives(valid=valid, incompatible=incompatible)


def _candidate_incompatible_with_selection(
    *,
    role: str,
    candidate: _IndexedComponentCandidate,
    selected: Mapping[str, _IndexedComponentCandidate],
) -> bool:
    platform = _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE)
    if platform is None:
        return False
    if role == CPU_ROLE:
        if _fatal_platform_cpu_mismatch(platform, candidate):
            return True
        platform_vendor = _fact(platform, "normalized_vendor")
        option_vendor = _fact(candidate, "option_kit_vendor")
        return bool(
            _bool_fact(candidate, "is_vendor_option_kit")
            and option_vendor != UNKNOWN_FACT
            and platform_vendor != option_vendor
        )
    return False


def _right_size_distance(
    role: str,
    candidate: _IndexedComponentCandidate,
    requirements: Mapping[str, Any],
) -> float | None:
    if role == CPU_ROLE:
        required_cores = _required_cpu_cores(requirements)
        cpu_cores = _candidate_cpu_cores(candidate)
        if required_cores is None or cpu_cores is None or cpu_cores < required_cores:
            return None
        return float(cpu_cores - required_cores)
    if role in {SSD_ROLE, HDD_ROLE}:
        required_tb = _required_storage_tb(requirements)
        capacity_tb = _candidate_storage_tb(candidate)
        if required_tb is None or capacity_tb is None or capacity_tb < required_tb:
            return None
        return float(capacity_tb - required_tb)
    return None


def _exact_requirement_note(
    *,
    role: str,
    selected_candidate: _IndexedComponentCandidate,
    requirements: Mapping[str, Any],
) -> str:
    if role == CPU_ROLE:
        required_cores = _required_cpu_cores(requirements)
        selected_cores = _candidate_cpu_cores(selected_candidate)
        if (
            required_cores is not None
            and selected_cores is not None
            and selected_cores >= required_cores
        ):
            return f"CPU соответствует минимальному требованию {_cores_text(required_cores)}."
        return ""
    if role not in {SSD_ROLE, HDD_ROLE}:
        return ""
    required_tb = _required_storage_tb(requirements)
    selected_tb = _candidate_storage_tb(selected_candidate)
    if required_tb is None or selected_tb is None:
        return ""
    if abs(selected_tb - required_tb) > 0.001:
        return ""
    role_label = "SSD" if role == SSD_ROLE else "HDD"
    return f"{role_label} соответствует минимальному требованию {_tb_text(required_tb)}."


def _required_cpu_cores(requirements: Mapping[str, Any]) -> int | None:
    return _int_value(
        requirements.get("cpu_min_cores_per_cpu")
        or requirements.get("min_cores_per_cpu")
    )


def _candidate_cpu_cores(candidate: _IndexedComponentCandidate) -> int | None:
    return _int_value(candidate.row.get("cpu_cores") or _fact(candidate, "cpu_cores"))


def _required_storage_tb(requirements: Mapping[str, Any]) -> float | None:
    value = requirements.get("storage_min_capacity_tb")
    if value in (None, ""):
        value = requirements.get("storage_min_capacity")
    if value in (None, ""):
        return None
    if isinstance(value, int | float | Decimal):
        return float(value)
    text = str(value).replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return None
    amount = float(match.group(1))
    if re.search(r"\b(?:gb|гб)\b", text, re.IGNORECASE):
        return amount / 1024
    return amount


def _candidate_storage_tb(candidate: _IndexedComponentCandidate) -> float | None:
    for value in (
        candidate.row.get("drive_capacity_tb"),
        _fact(candidate, "drive_capacity_tb"),
        candidate.row.get("storage_capacity_tb"),
        _fact(candidate, "storage_capacity_tb"),
    ):
        parsed = _float_value(value)
        if parsed is not None:
            return parsed
    return None


def _cpu_matches_preferences(
    candidate: _IndexedComponentCandidate,
    requirements: Mapping[str, Any],
) -> bool:
    vendor = str(requirements.get("cpu_vendor_preference") or "").strip()
    if vendor and vendor != UNKNOWN_FACT and _cpu_component_side(candidate) != vendor:
        return False
    family = str(requirements.get("cpu_family_preference") or "").strip()
    return not (family and family != UNKNOWN_FACT and _fact(candidate, "cpu_family") != family)


def _storage_matches_preferences(
    candidate: _IndexedComponentCandidate,
    requirements: Mapping[str, Any],
) -> bool:
    interface = str(requirements.get("storage_interface_preference") or "").strip()
    return not (
        interface
        and interface != UNKNOWN_FACT
        and _fact(candidate, "storage_interface") != interface
        and _fact(candidate, "drive_interface") != interface
    )


def _cores_text(value: int) -> str:
    if value % 10 == 1 and value % 100 != 11:
        word = "ядро"
    elif value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        word = "ядра"
    else:
        word = "ядер"
    return f"{value} {word}"


def _tb_text(value: float) -> str:
    amount = Decimal(str(value))
    text = (
        str(int(amount))
        if amount == amount.to_integral()
        else format(amount.normalize(), "f")
    )
    return f"{text} ТБ"


def _candidate_fit_reason(candidate: _IndexedComponentCandidate) -> str:
    reason = str(candidate.row.get("fit_reason") or "").strip()
    if reason:
        return reason
    reasons = _string_list(candidate.row.get("fit_reasons"))
    return reasons[0] if reasons else "выбран как доступный складской компромисс"


def _has_sufficient_stock(candidate: _IndexedComponentCandidate, quantity: int) -> bool:
    available = _int_value(candidate.row.get("available_quantity"))
    return available is not None and available >= quantity


def _line_price(
    candidate: _IndexedComponentCandidate,
    quantity: int,
) -> Decimal | None:
    price = _decimal_value(candidate.row.get("price_value"))
    if price is None:
        return None
    return price * quantity


def _text_has_overfit_reason(text: str) -> bool:
    lowered = text.casefold()
    markers = (
        "нет",
        "отсутств",
        "дешев",
        "совмест",
        "остат",
        "stock",
        "cheaper",
        "compat",
        "availability",
        "only",
    )
    return any(marker in lowered for marker in markers)


def _selected_int(
    selected: Mapping[str, _IndexedComponentCandidate],
    role: str,
    key: str,
) -> int | None:
    candidate = _selected_candidate_for_role(selected, role)
    if candidate is None:
        return None
    return _int_value(candidate.row.get(key))


def _selected_float(
    candidate: _IndexedComponentCandidate | None,
    key: str,
) -> float | None:
    if candidate is None:
        return None
    return _float_value(candidate.row.get(key))


def _recommendation_slot_value(recommendation: LlmRecommendationPayload) -> str | None:
    if recommendation.recommendation_slot:
        return recommendation.recommendation_slot
    slot = _proposal_role_to_recommendation_slot(recommendation.proposal_role)
    if slot:
        return slot
    title = recommendation.title.casefold()
    if "цен" in title or "price" in title:
        return "price_optimal"
    if "техничес" in title or "clean" in title:
        return "technical_clean"
    if "альтернатив" in title or "alternative" in title:
        return "alternative"
    return None


def _slot_priority(value: Any) -> int:
    priorities = {
        "price_optimal": 0,
        "technical_clean": 1,
        "alternative": 2,
        "alternative_vendor_or_platform": 2,
        "exact_cpu_if_available": 3,
        "lower_price_with_tradeoff": 4,
        "partial_only_if_no_full": 5,
        "partial_fallback": 5,
    }
    return priorities.get(str(value or "").strip(), 9)


def _llm_recommendation_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    platform = candidate.get("platform")
    platform_part_number = ""
    if isinstance(platform, Mapping):
        platform_part_number = str(platform.get("part_number") or "").strip()
    display_part_number = str(
        candidate.get("part_number") or platform_part_number or candidate.get("display_name") or ""
    ).strip()
    return (
        _int_value(candidate.get("_llm_source_order")) or 0,
        _slot_priority(candidate.get("recommendation_slot")),
        *_price_sort_key(candidate.get("total_price_value"), 1),
        _stable_text(display_part_number),
        _stable_text(candidate.get("recommendation_id") or candidate.get("candidate_id")),
    )


def _quantity_for_role(
    quantities: Mapping[str, int],
    prompt_role: str,
    internal_role: str,
) -> int | None:
    value = quantities.get(prompt_role)
    if value is None:
        value = quantities.get(internal_role)
    return _int_value(value)


def _validate_stock(
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
    build_id: str,
) -> str | None:
    for role, candidate in selected.items():
        available = _int_value(candidate.row.get("available_quantity"))
        if available is None:
            return f"{build_id}: stock is unknown for selected component"
        if available < quantities[role]:
            return f"{build_id}: stock is below required quantity"
    return None


def _validate_vendor_compatibility(
    selected: Mapping[str, _IndexedComponentCandidate],
    build_id: str,
) -> str | None:
    platform = _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE)
    if platform is None:
        return None

    platform_vendor = _fact(platform, "normalized_vendor")
    cpu = _selected_candidate_for_role(selected, CPU_ROLE)
    if cpu is not None:
        fatal_mismatch = _fatal_platform_cpu_mismatch(platform, cpu)
        if fatal_mismatch:
            return f"{build_id}: {fatal_mismatch}"
    if cpu is not None and _bool_fact(cpu, "is_vendor_option_kit"):
        option_vendor = _fact(cpu, "option_kit_vendor")
        if option_vendor != UNKNOWN_FACT and platform_vendor != option_vendor:
            return f"{build_id}: vendor-specific CPU kit does not match platform vendor"

    ram = _selected_candidate_for_role(selected, RAM_ROLE)
    if ram is not None:
        platform_ram_type = _normalized_repair_ram_type(platform).generation
        ram_type = _normalized_repair_ram_type(ram).generation
        if platform_ram_type and ram_type and platform_ram_type != ram_type:
            return f"{build_id}: RAM type does not match platform RAM type"
    return None


def _validate_evidence_compatibility(
    selected: Mapping[str, _IndexedComponentCandidate],
    build_id: str,
    *,
    normalized_requirements: Any,
    evidence_by_component_id: Mapping[str, Mapping[str, Any]],
    relation_evidence: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[str]]:
    if not evidence_by_component_id and not relation_evidence:
        return None, []
    warnings: list[str] = []
    if evidence_by_component_id:
        for candidate in selected.values():
            evidence = evidence_by_component_id.get(candidate.component_candidate_id)
            if not evidence or evidence.get("evidence_status") != "found":
                warnings.append(_missing_evidence_warning(candidate.internal_role))

    relation_fatal, relation_warnings = _validate_relation_evidence(
        relation_evidence,
        build_id=build_id,
    )
    warnings.extend(relation_warnings)
    if relation_fatal:
        return relation_fatal, warnings

    platform = _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE)
    cpu = _selected_candidate_for_role(selected, CPU_ROLE)
    if platform is not None and cpu is not None:
        platform_evidence = evidence_by_component_id.get(platform.component_candidate_id)
        cpu_evidence = evidence_by_component_id.get(cpu.component_candidate_id)
        fatal = _evidence_platform_cpu_mismatch(platform_evidence, cpu_evidence)
        if fatal:
            return f"{build_id}: {fatal}", warnings

    ram = _selected_candidate_for_role(selected, RAM_ROLE)
    if platform is not None and ram is not None:
        platform_evidence = evidence_by_component_id.get(platform.component_candidate_id)
        ram_evidence = evidence_by_component_id.get(ram.component_candidate_id)
        fatal = _evidence_platform_required_ram_mismatch(
            platform_evidence,
            normalized_requirements=normalized_requirements,
        )
        if fatal:
            return f"{build_id}: {fatal}", warnings
        fatal = _evidence_platform_ram_mismatch(platform_evidence, ram_evidence)
        if fatal:
            return f"{build_id}: {fatal}", warnings

    storage = _selected_candidate_for_role(selected, SSD_ROLE) or _selected_candidate_for_role(
        selected,
        HDD_ROLE,
    )
    if platform is not None and storage is not None:
        platform_evidence = evidence_by_component_id.get(platform.component_candidate_id)
        storage_evidence = evidence_by_component_id.get(storage.component_candidate_id)
        fatal = _evidence_platform_required_storage_mismatch(
            platform_evidence,
            normalized_requirements=normalized_requirements,
        )
        if fatal:
            return f"{build_id}: {fatal}", warnings
        fatal = _evidence_platform_storage_mismatch(platform_evidence, storage_evidence)
        if fatal:
            return f"{build_id}: {fatal}", warnings
    return None, warnings


def _validate_relation_evidence(
    relation_evidence: Sequence[Mapping[str, Any]],
    *,
    build_id: str,
) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    for relation in relation_evidence:
        relation_type = str(relation.get("relation_type") or "").strip()
        status = str(relation.get("status") or "").strip()
        if status == "mismatch":
            facts = _string_list(relation.get("mismatch_facts"))
            detail = "; ".join(facts[:3]) if facts else "relation evidence mismatch"
            if relation_type == "platform_cpu":
                return f"{build_id}: relation_platform_cpu mismatch: {detail}", warnings
            if relation_type == "platform_ram":
                return f"{build_id}: relation_platform_ram mismatch: {detail}", warnings
            if relation_type == "platform_storage":
                return f"{build_id}: relation_platform_storage mismatch: {detail}", warnings
            return f"{build_id}: relation evidence mismatch: {detail}", warnings
        if status in {"not_confirmed", "error"}:
            warnings.append(_relation_missing_warning(relation_type))
        elif status == "partially_confirmed":
            warnings.append(_relation_partial_warning(relation_type))
    return None, _unique(warnings)


def _relation_missing_warning(relation_type: str) -> str:
    if relation_type == "platform_cpu":
        return (
            "Доказательная проверка: support list CPU для платформы не найден; "
            "нужна инженерная сверка."
        )
    if relation_type == "platform_ram":
        return "Доказательная проверка: QVL/support list RAM не найден; нужна инженерная сверка."
    if relation_type == "platform_storage":
        return (
            "Доказательная проверка: backplane/NVMe support list не найден; "
            "нужна инженерная сверка."
        )
    return "Доказательная проверка: полная совместимость сборки не подтверждена источниками."


def _relation_partial_warning(relation_type: str) -> str:
    if relation_type == "platform_cpu":
        return (
            "Доказательная проверка: поколение/socket CPU частично подтверждены, "
            "CPU support list нужно сверить инженеру."
        )
    if relation_type == "platform_ram":
        return (
            "Доказательная проверка: DDR5/RDIMM частично подтверждены, "
            "QVL нужно сверить инженеру."
        )
    if relation_type == "platform_storage":
        return (
            "Доказательная проверка: NVMe/backplane частично подтверждены, "
            "совместимость накопителя нужно сверить инженеру."
        )
    return (
        "Доказательная проверка: часть совместимости подтверждена внешними "
        "источниками, финальная проверка инженером обязательна."
    )


def _missing_evidence_warning(role: str) -> str:
    if role == SERVER_PLATFORM_ROLE:
        return (
            "Внешние источники не подтвердили совместимость платформы и компонентов; "
            "требуется инженерная проверка."
        )
    return (
        "Доказательная проверка не нашла достаточных источников; "
        "требуется инженерная проверка."
    )


def _evidence_platform_required_ram_mismatch(
    platform_evidence: Mapping[str, Any] | None,
    *,
    normalized_requirements: Any,
) -> str | None:
    if not _evidence_can_reject(platform_evidence):
        return None
    requirements = _first_requirements(normalized_requirements)
    required_memory = _normalize_memory_type(requirements.get("ram_type_preference"))
    if not required_memory:
        return None
    platform_memory = _normalize_memory_type(
        _evidence_facts(platform_evidence).get("memory_type")
    )
    if platform_memory and platform_memory != required_memory:
        return (
            "fatal evidence memory mismatch: "
            f"request requires {required_memory}, platform evidence says {platform_memory}"
        )
    return None


def _evidence_platform_required_storage_mismatch(
    platform_evidence: Mapping[str, Any] | None,
    *,
    normalized_requirements: Any,
) -> str | None:
    if not _evidence_can_reject(platform_evidence):
        return None
    requirements = _first_requirements(normalized_requirements)
    required_interface = str(
        requirements.get("storage_interface_preference") or ""
    ).strip().upper()
    if required_interface != "NVME":
        return None
    if _evidence_facts(platform_evidence).get("nvme_support") is False:
        return (
            "fatal evidence storage mismatch: request requires NVMe, "
            "platform evidence says NVMe is not supported"
        )
    return None


def _evidence_platform_cpu_mismatch(
    platform_evidence: Mapping[str, Any] | None,
    cpu_evidence: Mapping[str, Any] | None,
) -> str | None:
    if not _evidence_can_reject(platform_evidence) or not _evidence_can_reject(cpu_evidence):
        return None
    platform_facts = _evidence_facts(platform_evidence)
    cpu_facts = _evidence_facts(cpu_evidence)
    platform_socket = _normalize_evidence_socket(platform_facts.get("socket_family"))
    cpu_socket = _normalize_evidence_socket(cpu_facts.get("socket_family"))
    if platform_socket and cpu_socket and platform_socket != cpu_socket:
        return (
            "fatal evidence socket mismatch: "
            f"platform {platform_socket} cannot use CPU {cpu_socket}"
        )
    platform_generations = _cpu_generation_tokens(
        platform_facts.get("supported_cpu_generation")
        or platform_facts.get("cpu_generation")
    )
    cpu_generations = _cpu_generation_tokens(cpu_facts.get("cpu_generation"))
    if platform_generations and cpu_generations and platform_generations.isdisjoint(
        cpu_generations
    ):
        return (
            "fatal evidence generation mismatch: "
            f"platform supports {sorted(platform_generations)} but CPU is {sorted(cpu_generations)}"
        )
    return None


def _evidence_platform_ram_mismatch(
    platform_evidence: Mapping[str, Any] | None,
    ram_evidence: Mapping[str, Any] | None,
) -> str | None:
    if not _evidence_can_reject(platform_evidence) or not _evidence_can_reject(ram_evidence):
        return None
    platform_memory = _normalize_memory_type(_evidence_facts(platform_evidence).get("memory_type"))
    ram_memory = _normalize_memory_type(_evidence_facts(ram_evidence).get("memory_type"))
    if platform_memory and ram_memory and platform_memory != ram_memory:
        return (
            "fatal evidence memory mismatch: "
            f"platform {platform_memory} cannot use RAM {ram_memory}"
        )
    return None


def _evidence_platform_storage_mismatch(
    platform_evidence: Mapping[str, Any] | None,
    storage_evidence: Mapping[str, Any] | None,
) -> str | None:
    if not _evidence_can_reject(platform_evidence) or not _evidence_can_reject(storage_evidence):
        return None
    platform_facts = _evidence_facts(platform_evidence)
    storage_facts = _evidence_facts(storage_evidence)
    storage_interface = str(storage_facts.get("storage_interface") or "").strip().upper()
    if storage_interface == "NVME" and platform_facts.get("nvme_support") is False:
        return "fatal evidence storage mismatch: platform evidence says NVMe is not supported"
    return None


def _evidence_can_reject(evidence: Mapping[str, Any] | None) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    if evidence.get("evidence_status") != "found":
        return False
    confidence = str(evidence.get("confidence") or "").strip()
    if confidence not in {"high", "medium"}:
        return False
    return bool(_mapping_rows(evidence.get("sources")))


def _evidence_facts(evidence: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(evidence, Mapping):
        return {}
    facts = evidence.get("facts")
    return facts if isinstance(facts, Mapping) else {}


def _normalize_evidence_socket(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    text = text.replace("FCLGA", "LGA")
    if text in {"SP3", "SP5", "LGA3647", "LGA4189", "LGA4677", "LGA4710", "LGA4094", "LGA6096"}:
        return text
    match = re.search(r"(?:FC)?LGA(3647|4189|4677|4710|4094|6096)", text)
    if match is not None:
        return f"LGA{match.group(1)}"
    match = re.search(r"SP([35])", text)
    if match is not None:
        return f"SP{match.group(1)}"
    return ""


def _normalize_memory_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    if re.search(r"DDR\s*5", text):
        return "DDR5"
    if re.search(r"DDR\s*4", text):
        return "DDR4"
    return ""


def _cpu_generation_tokens(value: Any) -> set[str]:
    text = str(value or "").casefold()
    tokens: set[str] = set()
    if "xeon 6" in text or re.search(r"\b65\d{2}p\b|\b6[57]\d{2}p\b", text):
        tokens.add("xeon6")
    if re.search(r"\b4th\s*/\s*5th\b|\b4th\s+or\s+5th\b", text):
        tokens.update({"xeon4", "xeon5"})
    if "5th gen" in text or "emerald rapids" in text:
        tokens.add("xeon5")
    if "4th gen" in text or "sapphire rapids" in text or "4th_gen_xeon" in text:
        tokens.add("xeon4")
    if "3rd gen" in text or "ice lake" in text or "3rd_gen_xeon" in text:
        tokens.add("xeon3")
    if "2nd gen" in text or "cascade lake" in text:
        tokens.add("xeon2")
    if "epyc 9004" in text or "epyc_9004" in text or "genoa" in text:
        tokens.add("epyc9004")
    if "epyc 7003" in text or "epyc_700x" in text or "milan" in text:
        tokens.add("epyc7003")
    return tokens


def _fatal_platform_cpu_mismatch(
    platform: _IndexedComponentCandidate,
    cpu: _IndexedComponentCandidate,
) -> str | None:
    platform_side = _cpu_platform_side(platform)
    cpu_side = _cpu_component_side(cpu)
    if platform_side == "AMD" and cpu_side == "Intel":
        return "fatal compatibility mismatch: AMD EPYC/SP platform with Intel Xeon CPU"
    if platform_side == "Intel" and cpu_side == "AMD":
        return "fatal compatibility mismatch: Intel Xeon platform with AMD EPYC CPU"

    platform_socket = _cpu_socket(platform)
    cpu_socket = _cpu_socket(cpu)
    if (
        platform_socket != UNKNOWN_FACT
        and cpu_socket != UNKNOWN_FACT
        and platform_socket != cpu_socket
    ):
        return (
            "fatal socket mismatch: "
            f"platform {platform_socket} cannot use CPU {cpu_socket}"
        )
    return None


def _cpu_platform_side(candidate: _IndexedComponentCandidate) -> str:
    family = _fact(candidate, "cpu_family")
    brand = _fact(candidate, "cpu_brand")
    socket = _cpu_socket(candidate)
    text = _candidate_search_text(candidate)
    if family == "EPYC" or socket in {"SP3", "SP5", "LGA4094", "LGA6096"}:
        return "AMD"
    if family == "Xeon" or socket in {"LGA3647", "LGA4189", "LGA4677", "LGA4710"}:
        return "Intel"
    if brand in {"Intel", "AMD"}:
        return brand
    if _has_amd_platform_marker(text):
        return "AMD"
    if _has_intel_platform_marker(text):
        return "Intel"
    return UNKNOWN_FACT


def _cpu_component_side(candidate: _IndexedComponentCandidate) -> str:
    family = _fact(candidate, "cpu_family")
    brand = _fact(candidate, "cpu_brand")
    socket = _cpu_socket(candidate)
    text = _candidate_search_text(candidate)
    if family == "Xeon" or socket in {"LGA3647", "LGA4189", "LGA4677", "LGA4710"}:
        return "Intel"
    if family == "EPYC" or socket in {"SP3", "SP5", "LGA4094", "LGA6096"}:
        return "AMD"
    if brand in {"Intel", "AMD"}:
        return brand
    if re.search(r"\b(?:intel|xeon)\b", text, re.IGNORECASE):
        return "Intel"
    if re.search(r"\b(?:amd|epyc)\b", text, re.IGNORECASE):
        return "AMD"
    return UNKNOWN_FACT


def _cpu_socket(candidate: _IndexedComponentCandidate) -> str:
    socket = _fact(candidate, "cpu_socket")
    if socket != UNKNOWN_FACT:
        return socket.upper().replace(" ", "")
    return _detect_cpu_socket(_candidate_search_text(candidate))


def _inferred_fact_from_name(candidate: _IndexedComponentCandidate, key: str) -> str:
    text = _candidate_search_text(candidate)
    if key in {"ram_type", "memory_type"}:
        return _detect_ram_type(text)
    if key in {"cpu_socket", "socket_family"}:
        return _detect_cpu_socket(text)
    if key in {"cpu_generation", "supported_cpu_generation"}:
        return _detect_cpu_generation(text)
    if key == "storage_interface":
        return _detect_storage_interface(text)
    return UNKNOWN_FACT


def _detect_cpu_socket(text: str) -> str:
    lga_match = re.search(r"\b(?:FC)?LGA\s*(3647|4189|4677|4710|4094|6096)\b", text, re.IGNORECASE)
    if lga_match is not None:
        return f"LGA{lga_match.group(1)}"
    bare_lga_match = re.search(r"\b(3647|4189|4677|4710|4094|6096)\b", text, re.IGNORECASE)
    if bare_lga_match is not None and re.search(r"\blga\b", text, re.IGNORECASE):
        return f"LGA{bare_lga_match.group(1)}"
    sp_match = re.search(r"\bSP\s*([35])\b", text, re.IGNORECASE)
    if sp_match is not None:
        return f"SP{sp_match.group(1)}"
    lowered = text.casefold()
    if re.search(r"\bice\s*lake\b|\b3rd\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b", lowered):
        return "LGA4189"
    if re.search(
        r"\bsapphire\s+rapids\b|\brapid\s+sapphire\b|\b4th\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b",
        lowered,
    ):
        return "LGA4677"
    if re.search(r"\b(?:epyc\s*)?(?:7001|7002|7003)\b|\bsp3\b", lowered):
        return "SP3"
    if re.search(r"\b(?:epyc\s*)?9004\b|\bsp5\b|\bgenoa\b", lowered):
        return "SP5"
    return UNKNOWN_FACT


def _detect_ram_type(text: str) -> str:
    if re.search(r"\bDDR\s*5\b", text, re.IGNORECASE):
        return "DDR5"
    if re.search(r"\bDDR\s*4\b", text, re.IGNORECASE):
        return "DDR4"
    lowered = text.casefold()
    if re.search(r"\bice\s*lake\b|\b3rd\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b", lowered):
        return "DDR4"
    if re.search(
        r"\bsapphire\s+rapids\b|\brapid\s+sapphire\b|\b4th\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b|\blga\s*4677\b",
        lowered,
    ):
        return "DDR5"
    if re.search(r"\b(?:epyc\s*)?(?:7001|7002|7003)\b|\bsp3\b", lowered):
        return "DDR4"
    if re.search(r"\b(?:epyc\s*)?9004\b|\bsp5\b|\bgenoa\b", lowered):
        return "DDR5"
    return UNKNOWN_FACT


def _detect_cpu_generation(text: str) -> str:
    lowered = text.casefold()
    if re.search(r"\bice\s*lake\b|\b3rd\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b", lowered):
        return "3rd_gen_xeon"
    if re.search(
        r"\bsapphire\s+rapids\b|\brapid\s+sapphire\b|\b4th\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b",
        lowered,
    ):
        return "4th_gen_xeon"
    if re.search(r"\b(?:epyc\s*)?(?:7001|7002|7003)\b|\bsp3\b", lowered):
        return "epyc_700x"
    if re.search(r"\b(?:epyc\s*)?9004\b|\bsp5\b|\bgenoa\b", lowered):
        return "epyc_9004"
    return UNKNOWN_FACT


def _detect_storage_interface(text: str) -> str:
    if re.search(
        r"\bnvme\b|\bnvme\s*bp\b|\bnvme\s*backplane\b|\bu\.?2\b|\bu\.?3\b",
        text,
        re.IGNORECASE,
    ):
        return "NVMe"
    if re.search(r"\bsas\b", text, re.IGNORECASE):
        return "SAS"
    if re.search(r"\bsata\b", text, re.IGNORECASE):
        return "SATA"
    return UNKNOWN_FACT


def _detect_nvme_support(text: str) -> bool:
    return bool(
        re.search(
            r"\bnvme\b|\bnvme\s*bp\b|\bnvme\s*backplane\b|\bu\.?2\b|\bu\.?3\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_amd_platform_marker(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:amd|epyc|sp\s*[35]|lga\s*(?:4094|6096)|7001|7002|7003|9004)\b",
            text,
            re.IGNORECASE,
        )
    )


def _has_intel_platform_marker(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:intel|xeon|ice\s*lake|sapphire\s+rapids|rapid\s+sapphire|c621|c741|lga\s*(?:3647|4189|4677|4710))\b",
            text,
            re.IGNORECASE,
        )
    )


def _candidate_search_text(candidate: _IndexedComponentCandidate) -> str:
    values: list[str] = []
    for source in (candidate.row, candidate.source):
        for key in ("producer", "normalized_vendor", "part_number", "name", "item_name"):
            value = source.get(key)
            if value not in (None, ""):
                values.append(str(value))
        raw_facts = source.get("extracted_facts")
        facts = raw_facts if isinstance(raw_facts, Mapping) else {}
        if isinstance(facts, Mapping):
            values.extend(str(value) for value in facts.values() if value not in (None, "", []))
    return " ".join(values)


def _fatal_warning_text(values: list[str]) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.casefold()
        if (
            "fatal" in lowered
            or "incompat" in lowered
            or "mismatch" in lowered
            or "несовмест" in lowered
        ):
            return text
    return None


def _calculate_total(
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
    *,
    excluded_roles: Sequence[str] | None = None,
) -> tuple[Decimal | None, str | None, str | None]:
    excluded = set(excluded_roles or [])
    currencies: set[str] = set()
    total = Decimal("0")
    for selection_key, candidate in selected.items():
        if candidate.internal_role in excluded:
            continue
        price = _decimal_value(candidate.row.get("price_value"))
        currency = str(candidate.row.get("price_currency") or "").strip()
        if price is None or not currency:
            return None, None, "price or currency is missing"
        currencies.add(currency)
        total += price * quantities[selection_key]
    if not currencies:
        return None, None, "price or currency is missing"
    if len(currencies) != 1:
        return None, None, "currencies are mixed"
    return total, next(iter(currencies)), None


def _calculate_optional_total(
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
    optional_roles: Sequence[str],
) -> tuple[Decimal | None, str | None, str | None]:
    optional_selected = {
        selection_key: candidate
        for selection_key, candidate in selected.items()
        if candidate.internal_role in set(optional_roles)
    }
    if not optional_selected:
        return None, None, None
    return _calculate_total(optional_selected, quantities)


def _mandatory_roles(normalized_requirements: Any) -> list[str]:
    requirements = _first_requirements(normalized_requirements)
    product_group = _product_group_from_requirements(requirements)
    mandatory = [SERVER_PLATFORM_ROLE] if product_group == SERVER_PRODUCT_GROUP else []
    if product_group == STORAGE_PRODUCT_GROUP:
        _append_unique_role(mandatory, STORAGE_SYSTEM_ROLE)
    for capability in _hard_capabilities(normalized_requirements):
        role = _role_from_capability(capability)
        if role is None:
            continue
        if role == "storage" and product_group != STORAGE_PRODUCT_GROUP:
            storage_role = _storage_role_for_preference(
                requirements.get("storage_type_preference")
            )
            if storage_role is not None:
                _append_unique_role(mandatory, storage_role)
            continue
        _append_unique_role(mandatory, role)
    for role in _required_roles_for_package(requirements):
        if role == "server_platform":
            _append_unique_role(mandatory, SERVER_PLATFORM_ROLE)
        elif role in {
            SWITCH_ROLE,
            ROUTER_ROLE,
            FIREWALL_ROLE,
            ACCESS_POINT_ROLE,
            STORAGE_SYSTEM_ROLE,
            STORAGE_ARRAY_CONTROLLER_ROLE,
            CONTROLLER_MODULE_ROLE,
            DISK_SHELF_ROLE,
            DRIVE_ROLE,
            CACHE_ROLE,
            HOST_PORT_ROLE,
            PROTOCOL_MODULE_ROLE,
            CPU_ROLE,
            RAM_ROLE,
            STORAGE_CONTROLLER_ROLE,
            NETWORK_ADAPTER_ROLE,
            GPU_ROLE,
            TRANSCEIVER_ROLE,
            DAC_CABLE_ROLE,
            CABLE_ROLE,
            POWER_SUPPLY_ROLE,
            RAIL_KIT_ROLE,
            LICENSE_ROLE,
            SUPPORT_ROLE,
            STACKING_MODULE_ROLE,
            OTHER_ACCESSORY_ROLE,
            UNMAPPED_ROLE,
        }:
            _append_unique_role(mandatory, role)
        elif role == "storage":
            if product_group == STORAGE_PRODUCT_GROUP:
                _append_unique_role(mandatory, STORAGE_SYSTEM_ROLE)
            else:
                storage_role = _storage_role_for_preference(
                    requirements.get("storage_type_preference")
                )
                if storage_role is not None:
                    _append_unique_role(mandatory, storage_role)
    if requirements.get("total_cpu_required") or requirements.get("cpu_per_server"):
        _append_unique_role(mandatory, CPU_ROLE)
    if requirements.get("ram_gb_per_server"):
        _append_unique_role(mandatory, RAM_ROLE)
    if product_group != STORAGE_PRODUCT_GROUP and requirements.get("storage_required"):
        storage_role = _storage_role_for_preference(requirements.get("storage_type_preference"))
        if storage_role is not None:
            _append_unique_role(mandatory, storage_role)
    if _network_required(requirements):
        _append_unique_role(mandatory, NETWORK_ADAPTER_ROLE)
    return _unique(mandatory)


def _append_unique_role(roles: list[str], role: str) -> None:
    if role not in roles:
        roles.append(role)


def _product_group_from_requirements(requirements: Mapping[str, Any]) -> str:
    product_group = str(requirements.get("product_group") or "").strip()
    if product_group:
        return product_group
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        product_group = str(role_plan.get("product_group") or "").strip()
        if product_group:
            return product_group
    return SERVER_PRODUCT_GROUP


def _required_roles_for_package(normalized_requirements: Any) -> list[str]:
    requirements = _first_requirements(normalized_requirements)
    roles = _string_list(requirements.get("required_roles"))
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        roles.extend(_string_list(role_plan.get("required_roles")))
    if not roles and isinstance(normalized_requirements, Mapping):
        roles.extend(_string_list(normalized_requirements.get("required_roles")))
    return _unique(roles)


def _package_required_roles(
    normalized_requirements: Any,
    role_plan: Mapping[str, Any],
) -> list[str]:
    roles = [
        *_required_roles_for_package(normalized_requirements),
        *_string_list(role_plan.get("required_roles")),
        *_mandatory_roles(normalized_requirements),
    ]
    return _unique([_coverage_role_key(role) for role in roles if str(role).strip()])


def _required_roles_after_classification(
    required_roles: Sequence[str],
    role_plan: Mapping[str, Any],
) -> list[str]:
    roles = _string_list(list(required_roles))
    non_bom_roles = {
        _coverage_role_key(str(row.get("target_role") or row.get("fulfillment_target_role")))
        for row in _mapping_rows(role_plan.get("classified_requirements"))
        if str(row.get("target_role") or row.get("fulfillment_target_role") or "").strip()
        and str(row.get("classification") or "").strip()
        != REQ_CLASS_PRIMARY_OBJECT_FEATURE
        and not _truthy(row.get("should_create_bom_role"))
        and str(row.get("fulfillment_mode") or "").strip()
        in {
            FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
            FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
            FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
            FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
            FULFILLMENT_NOT_APPLICABLE,
        }
    }
    bom_roles = {
        _coverage_role_key(str(row.get("target_role") or row.get("fulfillment_target_role")))
        for row in _mapping_rows(role_plan.get("classified_requirements"))
        if str(row.get("target_role") or row.get("fulfillment_target_role") or "").strip()
        and _truthy(row.get("should_create_bom_role"))
    }
    if non_bom_roles:
        roles = [role for role in roles if role not in non_bom_roles or role in bom_roles]
    if UNMAPPED_ROLE not in roles:
        return roles
    if _mapping_rows(role_plan.get("unmapped_requirements_blocking")):
        return roles
    if any(
        str(row.get("classification") or "").strip()
        == REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE
        for row in _mapping_rows(role_plan.get("classified_requirements"))
    ):
        return roles
    return [role for role in roles if role != UNMAPPED_ROLE]


def _required_capabilities_for_package(
    normalized_requirements: Any,
    component_candidate_matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    capabilities = _mapping_rows(component_candidate_matrix.get("required_capabilities"))
    requirements = _first_requirements(normalized_requirements)
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        capabilities.extend(_mapping_rows(role_plan.get("required_capabilities")))
    capabilities.extend(_mapping_rows(requirements.get("required_capabilities")))
    return _unique_capabilities(capabilities)


def _semantic_package_fields(
    component_candidate_matrix: Mapping[str, Any],
    role_plan: Mapping[str, Any],
) -> dict[str, Any]:
    is_v2_context = str(
        component_candidate_matrix.get("pipeline_version")
        or role_plan.get("pipeline_version")
        or ""
    ).strip() == "v2_composer_first"

    def field(name: str, default: Any = None) -> Any:
        matrix_value = component_candidate_matrix.get(name)
        if matrix_value not in (None, "", [], {}):
            return matrix_value
        return role_plan.get(name, default)

    def text_or_none(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    return {
        "primary_product_group": text_or_none(field("primary_product_group")),
        "primary_object": text_or_none(field("primary_object")),
        "semantic_planner_source": text_or_none(field("semantic_planner_source")),
        "semantic_planner_used": bool(
            field("semantic_planner_used", False)
            or text_or_none(field("semantic_planner_source"))
            in {"llm", "llm_repaired", "llm_minimal_fallback"}
        ),
        "semantic_planner_confidence": text_or_none(
            field("semantic_planner_confidence")
        ),
        "semantic_planner_error_type": text_or_none(
            field("semantic_planner_error_type")
        ),
        "semantic_planner_http_status": field("semantic_planner_http_status"),
        "semantic_planner_parse_status": text_or_none(
            field("semantic_planner_parse_status")
        ),
        "semantic_planner_fallback_reason": text_or_none(
            field("semantic_planner_fallback_reason")
        ),
        "semantic_planner_attempts": field("semantic_planner_attempts", [])
        if isinstance(field("semantic_planner_attempts", []), list)
        else [],
        "semantic_planner_stage": text_or_none(field("semantic_planner_stage")),
        "semantic_planner_stage_timeouts": field("semantic_planner_stage_timeouts", [])
        if isinstance(field("semantic_planner_stage_timeouts", []), list)
        else [],
        "semantic_planner_timeout_reason": text_or_none(
            field("semantic_planner_timeout_reason")
        ),
        "semantic_planner_timeout_seconds": field(
            "semantic_planner_timeout_seconds"
        ),
        "semantic_planner_elapsed_ms": field("semantic_planner_elapsed_ms"),
        "semantic_planner_repair_attempted": bool(
            field("semantic_planner_repair_attempted", False)
        ),
        "semantic_planner_repair_success": bool(
            field("semantic_planner_repair_success", False)
        ),
        "semantic_planner_minimal_router_used": bool(
            field("semantic_planner_minimal_router_used", False)
        ),
        "semantic_planner_minimal_fallback_used": bool(
            field("semantic_planner_minimal_fallback_used", False)
        ),
        "requirement_classifier_status": text_or_none(
            field("requirement_classifier_status")
        ),
        "requirement_classifier_error_type": text_or_none(
            field("requirement_classifier_error_type")
        ),
        "requirement_classifier_parse_status": text_or_none(
            field("requirement_classifier_parse_status")
        ),
        "requirement_classifier_incomplete_reason": text_or_none(
            field("requirement_classifier_incomplete_reason")
        ),
        "requirement_source_coverage_percent": field(
            "requirement_source_coverage_percent"
        ),
        "unclassified_source_fragments": _string_list(
            field("unclassified_source_fragments", [])
        ),
        "synthetic_requirement_count": _int_value(
            field("synthetic_requirement_count")
        )
        or 0,
        "source_backed_requirement_count": _int_value(
            field("source_backed_requirement_count")
        )
        or 0,
        "requirement_classifier_repair_quality": text_or_none(
            field("requirement_classifier_repair_quality")
        ),
        "requirement_classifier_repair_accepted": bool(
            field("requirement_classifier_repair_accepted", False)
        ),
        "semantic_planner_model": text_or_none(field("semantic_planner_model")),
        "semantic_planner_provider": text_or_none(field("semantic_planner_provider")),
        "selected_product_group_reason": text_or_none(
            field("selected_product_group_reason")
        ),
        "deterministic_product_group_hint": text_or_none(
            field("deterministic_product_group_hint")
        ),
        "semantic_planner_disagreement": bool(
            field("semantic_planner_disagreement", False)
        ),
        "matrix_blueprint": _safe_mapping(field("matrix_blueprint", {})),
        "matrix_blueprint_roles": _string_list(field("matrix_blueprint_roles", [])),
        "roles_sent_to_composer": _string_list(field("roles_sent_to_composer", [])),
        "broad_reasoning_roles": _string_list(field("broad_reasoning_roles", [])),
        "hard_purchasable_bom_roles": _string_list(
            field("hard_purchasable_bom_roles", [])
        ),
        "hard_purchasable_bom_role_requirements": field(
            "hard_purchasable_bom_role_requirements",
            [],
        )
        if isinstance(field("hard_purchasable_bom_role_requirements", []), list)
        else [],
        "component_role_indicators": field("component_role_indicators", [])
        if is_v2_context and isinstance(field("component_role_indicators", []), list)
        else [],
        "embedded_requirements": field("embedded_requirements", [])
        if isinstance(field("embedded_requirements", []), list)
        else [],
        "classified_requirements": _normalize_package_classified_requirements(
            field("classified_requirements", [])
        ),
        "purchasable_role_requirements": field("purchasable_role_requirements", [])
        if isinstance(field("purchasable_role_requirements", []), list)
        else [],
        "primary_object_feature_requirements": field(
            "primary_object_feature_requirements",
            [],
        )
        if isinstance(field("primary_object_feature_requirements", []), list)
        else [],
        "accessory_or_consumable_requirements": field(
            "accessory_or_consumable_requirements",
            [],
        )
        if isinstance(field("accessory_or_consumable_requirements", []), list)
        else [],
        "service_or_support_requirements": field(
            "service_or_support_requirements",
            [],
        )
        if isinstance(field("service_or_support_requirements", []), list)
        else [],
        "logistics_or_commercial_constraints": field(
            "logistics_or_commercial_constraints",
            [],
        )
        if isinstance(field("logistics_or_commercial_constraints", []), list)
        else [],
        "engineering_check_requirements": field("engineering_check_requirements", [])
        if isinstance(field("engineering_check_requirements", []), list)
        else [],
        "optional_accessory_engineering_roles": _string_list(
            field("optional_accessory_engineering_roles", [])
        ),
        "optional_accessory_engineering_requirements": field(
            "optional_accessory_engineering_requirements",
            [],
        )
        if isinstance(field("optional_accessory_engineering_requirements", []), list)
        else [],
        "logistics_or_commercial_constraint_requirements": field(
            "logistics_or_commercial_constraint_requirements",
            [],
        )
        if isinstance(
            field("logistics_or_commercial_constraint_requirements", []),
            list,
        )
        else [],
        "unmapped_requirements_non_blocking": field(
            "unmapped_requirements_non_blocking",
            [],
        )
        if isinstance(field("unmapped_requirements_non_blocking", []), list)
        else [],
        "unmapped_requirements_blocking": field(
            "unmapped_requirements_blocking",
            [],
        )
        if isinstance(field("unmapped_requirements_blocking", []), list)
        else [],
        "requirement_role_mapping_decision": field(
            "requirement_role_mapping_decision",
            [],
        )
        if isinstance(field("requirement_role_mapping_decision", []), list)
        else [],
        "requirement_fulfillment_decision": field(
            "requirement_fulfillment_decision",
            [],
        )
        if isinstance(field("requirement_fulfillment_decision", []), list)
        else [],
        "role_fulfillment_diagnostics": field(
            "role_fulfillment_diagnostics",
            [],
        )
        if is_v2_context and isinstance(field("role_fulfillment_diagnostics", []), list)
        else [],
        "not_primary_product_groups": field("not_primary_product_groups", [])
        if isinstance(field("not_primary_product_groups", []), list)
        else [],
    }


def _normalize_package_classified_requirements(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(_mapping_rows(value), start=1):
        classification = _normalize_requirement_classification(
            row.get("requirement_classification") or row.get("classification")
        )
        mode = _normalize_package_fulfillment_mode(
            row.get("fulfillment_mode"),
            classification=classification,
        )
        if classification == REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT:
            mode = FULFILLMENT_LOGISTICS_CONSTRAINT
        evidence_text = str(row.get("evidence_text") or "").strip()
        if (
            mode in FULFILLMENT_INCLUDED_MODES
            and not evidence_text
            and not _package_explicit_assumption(row)
            and not (
                classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE
                and mode == FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT
            )
        ):
            mode = FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION
        should_create = _package_should_create_bom_role(row, classification, mode)
        should_validate = _package_should_validate(row, mode)
        engineer_check = str(
            row.get("engineer_check_ru")
            or row.get("suggested_engineer_check_ru")
            or row.get("source_text")
            or ""
        ).strip()
        normalized = {
            **dict(row),
            "requirement_id": str(
                row.get("requirement_id") or row.get("id") or f"req_{index}"
            ).strip(),
            "classification": classification,
            "fulfillment_mode": mode,
            "fulfillment_target_role": str(
                row.get("fulfillment_target_role")
                or row.get("target_role")
                or row.get("role")
                or ""
            ).strip(),
            "fulfillment_target_component_candidate_id": str(
                row.get("fulfillment_target_component_candidate_id") or ""
            ).strip(),
            "evidence_source": str(row.get("evidence_source") or "").strip(),
            "evidence_text": evidence_text,
            "should_create_bom_role": should_create,
            "should_validate_after_composer": should_validate,
            "should_be_validated_after_composer": should_validate,
            "engineer_check_ru": engineer_check,
            "suggested_engineer_check_ru": engineer_check,
        }
        rows.append(normalized)
    return rows


def _normalize_requirement_classification(value: Any) -> str:
    text = str(value or "").strip()
    normalized = text.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "commercial": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "commercial_constraint": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "budget": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "budget_constraint": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "price_constraint": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "logistics_constraint": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "logistic_constraint": REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
    }
    return aliases.get(normalized, text)


def _normalize_package_fulfillment_mode(value: Any, *, classification: str) -> str:
    text = str(value or "").strip()
    if text in {
        FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
        FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
        FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        FULFILLMENT_SERVICE_OR_SUPPORT,
        FULFILLMENT_LOGISTICS_CONSTRAINT,
        FULFILLMENT_ENGINEERING_CHECK_ONLY,
        FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
        FULFILLMENT_NOT_APPLICABLE,
    }:
        return text
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


def _package_explicit_assumption(row: Mapping[str, Any]) -> bool:
    return _truthy(row.get("explicit_assumption")) or bool(
        str(row.get("assumption") or "").strip()
    )


def _package_should_create_bom_role(
    row: Mapping[str, Any],
    classification: str,
    fulfillment_mode: str,
) -> bool:
    if "should_create_bom_role" in row:
        return _truthy(row.get("should_create_bom_role"))
    return fulfillment_mode in {
        FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        FULFILLMENT_SERVICE_OR_SUPPORT,
    } and classification in {
        REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
        REQ_CLASS_SERVICE_OR_SUPPORT,
        REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE,
    }


def _package_should_validate(row: Mapping[str, Any], fulfillment_mode: str) -> bool:
    if "should_validate_after_composer" in row:
        return _truthy(row.get("should_validate_after_composer"))
    if "should_be_validated_after_composer" in row:
        return _truthy(row.get("should_be_validated_after_composer"))
    return fulfillment_mode in {
        FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
        FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
        FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
        FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
        FULFILLMENT_ENGINEERING_CHECK_ONLY,
    }


def _optional_capabilities_for_package(
    normalized_requirements: Any,
    component_candidate_matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    capabilities = _mapping_rows(component_candidate_matrix.get("optional_capabilities"))
    requirements = _first_requirements(normalized_requirements)
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        capabilities.extend(_mapping_rows(role_plan.get("optional_capabilities")))
    capabilities.extend(_mapping_rows(requirements.get("optional_capabilities")))
    return _unique_capabilities(capabilities)


def _unique_capabilities(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values:
        capability_id = str(value.get("capability_id") or "").strip()
        role = str(value.get("role") or "").strip()
        key = (capability_id, role)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(value))
    return result


def _hard_capabilities(normalized_requirements: Any) -> list[dict[str, Any]]:
    requirements = _first_requirements(normalized_requirements)
    capabilities: list[Mapping[str, Any]] = []
    capabilities.extend(_mapping_rows(requirements.get("required_capabilities")))
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        capabilities.extend(_mapping_rows(role_plan.get("required_capabilities")))
    if isinstance(normalized_requirements, Mapping):
        capabilities.extend(_mapping_rows(normalized_requirements.get("required_capabilities")))
    return [
        dict(capability)
        for capability in _unique_capabilities(capabilities)
        if capability.get("hard", True)
    ]


def _role_from_capability(capability: Mapping[str, Any]) -> str | None:
    role = str(capability.get("role") or "").strip()
    if role == "platform":
        return SERVER_PLATFORM_ROLE
    if role == "storage":
        return "storage"
    if role == "license/support":
        return SUPPORT_ROLE
    if role == "unknown":
        return UNMAPPED_ROLE
    if role in {"storage_array", "system"}:
        return STORAGE_SYSTEM_ROLE
    if role in {"shelf", "drive_shelf", "expansion_shelf"}:
        return DISK_SHELF_ROLE
    if role in {"drives", "disks"}:
        return DRIVE_ROLE
    if role in {"host_ports", "ports", "host_interface"}:
        return HOST_PORT_ROLE
    if role in {"protocol", "protocol_adapter", "interface_module"}:
        return PROTOCOL_MODULE_ROLE
    if role in {
        SWITCH_ROLE,
        ROUTER_ROLE,
        FIREWALL_ROLE,
        ACCESS_POINT_ROLE,
        STORAGE_SYSTEM_ROLE,
        STORAGE_ARRAY_CONTROLLER_ROLE,
        CONTROLLER_MODULE_ROLE,
        DISK_SHELF_ROLE,
        DRIVE_ROLE,
        CACHE_ROLE,
        HOST_PORT_ROLE,
        PROTOCOL_MODULE_ROLE,
        DAC_CABLE_ROLE,
        STACKING_MODULE_ROLE,
    }:
        return role
    return INTERNAL_ROLE_BY_PROMPT_ROLE.get(role)


def _missing_mandatory_roles(
    *,
    mandatory_roles: Sequence[str],
    selected: Mapping[str, _IndexedComponentCandidate],
    normalized_requirements: Any,
) -> list[str]:
    requirements = _first_requirements(normalized_requirements)
    selected_roles = {candidate.internal_role for candidate in selected.values()}
    missing: list[str] = []
    for role in mandatory_roles:
        if _required_role_satisfied_by_selection(role, selected_roles):
            continue
        if role == NETWORK_ADAPTER_ROLE and _platform_onboard_network_satisfies(
            _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE),
            requirements,
        ):
            continue
        if (
            _product_group_from_requirements(requirements) == SERVER_PRODUCT_GROUP
            and role == POWER_SUPPLY_ROLE
            and _platform_power_bundle_satisfies(
                _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE),
                requirements,
            )
        ):
            continue
        missing.append(role)
    return missing


def _validate_network_requirement(
    *,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
    normalized_requirements: Any,
    recommendation_id: str,
) -> str | None:
    requirements = _first_requirements(normalized_requirements)
    if not _network_required(requirements):
        return None
    if _platform_onboard_network_satisfies(
        _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE),
        requirements,
    ):
        return None
    adapter = _selected_candidate_for_role(selected, NETWORK_ADAPTER_ROLE)
    capability_label = _network_capability_label(requirements)
    if adapter is None:
        return (
            f"{recommendation_id}: network_adapter required by hard "
            f"network requirement {capability_label}"
        )
    requirement = _network_requirement(requirements)
    facts = _network_facts_for_candidate(adapter)
    if not network_adapter_facts_satisfy_requirement(facts, requirement):
        return (
            f"{recommendation_id}: network_adapter does not satisfy requested "
            f"port speed/media requirement {capability_label}"
        )
    required_quantity = required_network_adapter_quantity(
        facts,
        requirement,
        server_quantity=_required_server_quantity(requirements),
    )
    selected_quantity = _selected_quantity_total_for_role(
        selected,
        quantities,
        NETWORK_ADAPTER_ROLE,
    )
    if required_quantity is None:
        return f"{recommendation_id}: network_adapter port count is unknown for {capability_label}"
    if selected_quantity is None or selected_quantity < required_quantity:
        return (
            f"{recommendation_id}: network_adapter quantity below hard "
            f"requirement {capability_label} ({selected_quantity} < {required_quantity})"
        )
    return None


def _hard_capability_validation(
    *,
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
    normalized_requirements: Any,
) -> list[dict[str, Any]]:
    requirements = _first_requirements(normalized_requirements)
    rows: list[dict[str, Any]] = []
    for requirement in _string_list(requirements.get("unsupported_or_unmapped_requirements")):
        rows.append(
            {
                "capability_id": "unsupported",
                "role": "unsupported",
                "status": "unsupported",
                "satisfied_by": None,
                "component_role": None,
                "component_candidate_id": None,
                "source_text": requirement,
                "evidence_source": "requirement_planner",
                "reason": requirement,
                "user_message": requirement,
            }
        )
    for capability in _hard_capabilities(normalized_requirements):
        role = _role_from_capability(capability)
        capability_id = str(capability.get("capability_id") or role or "unknown")
        if role is None:
            rows.append(
                {
                    "capability_id": capability_id,
                    "role": str(capability.get("role") or "unknown"),
                    "status": "unsupported_unvalidated",
                    "satisfied_by": None,
                    "component_role": None,
                    "component_candidate_id": None,
                    "source_text": _capability_source_text(capability),
                    "evidence_source": "role_catalog",
                    "reason": "Hard capability role is not supported by validator.",
                    "user_message": "Hard capability is preserved but has no validator role.",
                }
            )
            continue
        if _capability_is_primary_object_feature(capability):
            rows.append(
                _primary_object_feature_validation_row(
                    capability_id=capability_id,
                    role=role,
                    capability=capability,
                    selected=selected,
                )
            )
            continue
        if (
            _product_group_from_requirements(requirements) == STORAGE_PRODUCT_GROUP
            and _is_storage_product_role(role)
        ):
            candidate = _selected_candidate_for_role(selected, role)
            if candidate is None and role != STORAGE_SYSTEM_ROLE:
                bundle = _selected_candidate_for_role(selected, STORAGE_SYSTEM_ROLE)
                if bundle is not None and not _storage_product_capability_mismatch(
                    bundle,
                    capability,
                    selected,
                    quantities,
                ):
                    rows.append(
                        _capability_satisfied_row(
                            capability_id,
                            role,
                            bundle,
                            source="component_candidate_matrix_bundle",
                            source_text=_capability_source_text(capability),
                        )
                    )
                    continue
            if candidate is None:
                rows.append(_capability_missing_row(capability_id, role, capability))
                continue
            mismatch = _storage_product_capability_mismatch(
                candidate,
                capability,
                selected,
                quantities,
            )
            if mismatch:
                rows.append(
                    {
                        "capability_id": capability_id,
                        "role": role,
                        "status": "hard_mismatch",
                        "satisfied_by": "component",
                        "component_role": role,
                        "component_candidate_id": candidate.component_candidate_id,
                        "source_text": _capability_source_text(capability),
                        "evidence_source": "local_extracted_facts",
                        "reason": mismatch,
                        "user_message": mismatch,
                    }
                )
                continue
            rows.append(
                _capability_satisfied_row(
                    capability_id,
                    role,
                    candidate,
                    source="component_candidate_matrix",
                    source_text=_capability_source_text(capability),
                )
            )
            continue
        if (
            _product_group_from_requirements(requirements) == NETWORK_PRODUCT_GROUP
            and _is_network_product_role(role)
        ):
            candidate = _selected_candidate_for_role(selected, role)
            if candidate is None:
                rows.append(_capability_missing_row(capability_id, role, capability))
                continue
            mismatch = _network_product_capability_mismatch(candidate, capability)
            if mismatch:
                rows.append(
                    {
                        "capability_id": capability_id,
                        "role": role,
                        "status": "hard_mismatch",
                        "satisfied_by": "component",
                        "component_role": role,
                        "component_candidate_id": candidate.component_candidate_id,
                        "source_text": _capability_source_text(capability),
                        "evidence_source": "local_extracted_facts",
                        "reason": mismatch,
                        "user_message": mismatch,
                    }
                )
                continue
            rows.append(
                _capability_satisfied_row(
                    capability_id,
                    role,
                    candidate,
                    source="component_candidate_matrix",
                    source_text=_capability_source_text(capability),
                )
            )
            continue
        if role == "storage":
            if _selected_candidate_for_role(selected, SSD_ROLE) is not None:
                selected_role = SSD_ROLE
            elif _selected_candidate_for_role(selected, HDD_ROLE) is not None:
                selected_role = HDD_ROLE
            else:
                selected_role = None
            if selected_role is None:
                rows.append(_capability_missing_row(capability_id, role, capability))
            else:
                rows.append(
                    _capability_satisfied_row(
                        capability_id,
                        role,
                        _selected_candidate_for_role(selected, selected_role),
                        source="component_candidate_matrix",
                        source_text=_capability_source_text(capability),
                    )
            )
            continue
        if role == NETWORK_ADAPTER_ROLE:
            if _platform_onboard_network_satisfies(
                _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE),
                requirements,
            ):
                platform = _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE)
                rows.append(
                    {
                        "capability_id": capability_id,
                        "role": role,
                        "status": "satisfied",
                        "satisfied_by": "platform_onboard",
                        "component_role": SERVER_PLATFORM_ROLE,
                        "component_candidate_id": (
                            platform.component_candidate_id if platform is not None else None
                        ),
                        "source_text": _capability_source_text(capability),
                        "evidence_source": "local_extracted_facts",
                        "reason": "Platform onboard network satisfies requested network facts.",
                        "user_message": (
                            "Platform onboard network satisfies the requested "
                            "network requirement."
                        ),
                    }
                )
                continue
            adapter = _selected_candidate_for_role(selected, NETWORK_ADAPTER_ROLE)
            if adapter is None:
                rows.append(_capability_missing_row(capability_id, role, capability))
                continue
            facts = _network_facts_for_candidate(adapter)
            if not network_adapter_facts_satisfy_requirement(
                facts,
                _network_requirement(requirements),
            ):
                rows.append(
                    {
                        "capability_id": capability_id,
                        "role": role,
                        "status": "hard_mismatch",
                        "satisfied_by": "component",
                        "component_role": NETWORK_ADAPTER_ROLE,
                        "component_candidate_id": adapter.component_candidate_id,
                        "source_text": _capability_source_text(capability),
                        "evidence_source": "local_extracted_facts",
                        "reason": "Network adapter facts do not satisfy speed/media/ports.",
                        "user_message": (
                            "Selected network adapter does not satisfy the requested "
                            "speed/media/ports."
                        ),
                    }
                )
                continue
            required_quantity = required_network_adapter_quantity(
                facts,
                _network_requirement(requirements),
                server_quantity=_required_server_quantity(requirements),
            )
            selected_quantity = _selected_quantity_total_for_role(
                selected,
                quantities,
                NETWORK_ADAPTER_ROLE,
            )
            if (
                required_quantity is None
                or selected_quantity is None
                or selected_quantity < required_quantity
            ):
                rows.append(
                    {
                        "capability_id": capability_id,
                        "role": role,
                        "status": "hard_mismatch",
                        "satisfied_by": "component",
                        "component_role": NETWORK_ADAPTER_ROLE,
                        "component_candidate_id": adapter.component_candidate_id,
                        "source_text": _capability_source_text(capability),
                        "evidence_source": "local_extracted_facts",
                        "reason": "Network adapter quantity does not cover requested ports.",
                        "user_message": "Network adapter quantity does not cover requested ports.",
                    }
                )
                continue
            rows.append(
                _capability_satisfied_row(
                    capability_id,
                    role,
                    adapter,
                    source="local_extracted_facts",
                    source_text=_capability_source_text(capability),
                )
            )
            continue
        if role == POWER_SUPPLY_ROLE and _platform_power_bundle_satisfies(
            _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE),
            requirements,
        ):
            platform = _selected_candidate_for_role(selected, SERVER_PLATFORM_ROLE)
            if platform is None:
                rows.append(_capability_missing_row(capability_id, role, capability))
                continue
            rows.append(
                {
                    "capability_id": capability_id,
                    "role": role,
                    "status": "satisfied",
                    "satisfied_by": "platform_bundle",
                    "component_role": SERVER_PLATFORM_ROLE,
                    "component_candidate_id": platform.component_candidate_id,
                    "source_text": _capability_source_text(capability),
                    "evidence_source": "local_extracted_facts",
                    "reason": "Selected platform bundle includes required PSU redundancy.",
                    "user_message": "Selected platform includes the required PSU redundancy.",
                }
            )
            continue
        candidate = _selected_candidate_for_role(selected, role)
        if candidate is None:
            rows.append(_capability_missing_row(capability_id, role, capability))
            continue
        available = _int_value(candidate.row.get("available_quantity"))
        required_quantity = _selected_quantity_total_for_role(selected, quantities, role)
        if (
            available is not None
            and required_quantity is not None
            and available < required_quantity
        ):
            rows.append(
                {
                    "capability_id": capability_id,
                    "role": role,
                    "status": "blocked_by_stock",
                    "satisfied_by": "component",
                    "component_role": role,
                    "component_candidate_id": candidate.component_candidate_id,
                    "source_text": _capability_source_text(capability),
                    "evidence_source": "local_stock",
                    "reason": "Selected component stock is below required quantity.",
                    "user_message": "Selected component stock is below required quantity.",
                }
            )
            continue
        rows.append(
            _capability_satisfied_row(
                capability_id,
                role,
                candidate,
                source="component_candidate_matrix",
                source_text=_capability_source_text(capability),
            )
        )
    rows.extend(
        _classified_fulfillment_validation_rows(
            selected=selected,
            normalized_requirements=normalized_requirements,
        )
    )
    return rows


def _request_text_evidence_gate_validation_rows(
    *,
    selected: Mapping[str, _IndexedComponentCandidate],
    user_request: str | None,
    normalized_requirements: Any,
) -> list[dict[str, Any]]:
    requirements = _first_requirements(normalized_requirements)
    if _product_group_from_requirements(requirements) != NETWORK_PRODUCT_GROUP:
        return []
    request_text = str(user_request or "").strip()
    if not request_text:
        return []

    rows: list[dict[str, Any]] = []
    for role in (SWITCH_ROLE, ROUTER_ROLE, FIREWALL_ROLE, ACCESS_POINT_ROLE):
        candidate = _selected_candidate_for_role(selected, role)
        if candidate is None:
            continue
        mismatch = _network_primary_role_text_mismatch(candidate, request_text)
        if not mismatch:
            continue
        rows.append(
            {
                "capability_id": f"{role}.{mismatch}.request_text",
                "role": role,
                "status": "hard_mismatch",
                "satisfied_by": "component",
                "component_role": role,
                "component_candidate_id": candidate.component_candidate_id,
                "source_text": request_text,
                "evidence_source": "original_request_text",
                "reason": (
                    "Selected network primary component does not prove hard "
                    f"request-text fact: {mismatch}."
                ),
                "user_message": (
                    "Selected network primary component does not prove a hard "
                    "request-text fact."
                ),
            }
        )
    return rows


def _capability_missing_row(
    capability_id: str,
    role: str,
    capability: Mapping[str, Any],
) -> dict[str, Any]:
    reason = "No selected component or onboard platform feature covers this hard capability."
    if role == NETWORK_ADAPTER_ROLE:
        reason = (
            "No selected network_adapter or onboard network covers hard "
            f"network capability {capability_id}."
        )
    user_message = reason
    if role == NETWORK_ADAPTER_ROLE:
        user_message = (
            "Выбранная платформа не имеет onboard сети под запрошенные "
            "speed/media/ports, и подходящий сетевой адаптер не выбран."
        )
    if role == POWER_SUPPLY_ROLE:
        user_message = (
            "Selected platform does not expose the requested PSU redundancy and no "
            "separate PSU component is selected."
        )
    return {
        "capability_id": capability_id,
        "role": role,
        "status": "missing_component",
        "satisfied_by": None,
        "component_role": None,
        "component_candidate_id": None,
        "source_text": _capability_source_text(capability),
        "evidence_source": "component_candidate_matrix",
        "reason": reason,
        "user_message": user_message,
    }


def _classified_fulfillment_validation_rows(
    *,
    selected: Mapping[str, _IndexedComponentCandidate],
    normalized_requirements: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for requirement in _classified_requirements_for_validation(normalized_requirements):
        if str(requirement.get("hard_or_optional") or REQ_HARD).strip() != REQ_HARD:
            continue
        if not _classified_requirement_should_validate(requirement):
            continue
        classification = _normalize_requirement_classification(
            requirement.get("classification")
        )
        mode = str(requirement.get("fulfillment_mode") or "").strip()
        if classification == REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT:
            continue
        if (
            classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE
            and mode == FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT
        ):
            continue
        if mode == FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION:
            rows.append(
                _classified_unverified_row(
                    requirement,
                    selected=selected,
                    reason=(
                        "Requirement fulfillment was not proven by package/card/"
                        "content evidence."
                    ),
                )
            )
            continue
        if mode not in FULFILLMENT_INCLUDED_MODES:
            continue
        target = _classified_fulfillment_target(requirement, selected)
        if target is None:
            rows.append(_classified_fulfillment_missing_target_row(requirement))
            continue
        if not _classified_has_evidence(requirement):
            rows.append(
                _classified_unverified_row(
                    requirement,
                    selected=selected,
                    target=target,
                    reason=(
                        "Included fulfillment mode requires evidence_text or an explicit "
                        "assumption, but none was provided."
                    ),
                )
            )
            continue
        rows.append(_classified_fulfillment_satisfied_row(requirement, target))
    return rows


def _classified_requirements_for_validation(
    normalized_requirements: Any,
) -> list[dict[str, Any]]:
    requirements = _first_requirements(normalized_requirements)
    rows: list[Mapping[str, Any]] = []
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        rows.extend(_mapping_rows(role_plan.get("classified_requirements")))
    rows.extend(_mapping_rows(requirements.get("classified_requirements")))
    if isinstance(normalized_requirements, Mapping):
        rows.extend(_mapping_rows(normalized_requirements.get("classified_requirements")))
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        requirement_id = str(row.get("requirement_id") or f"req_{index}").strip()
        source_text = str(row.get("source_text") or "").strip()
        key = (requirement_id, source_text.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def _classified_requirement_should_validate(requirement: Mapping[str, Any]) -> bool:
    if "should_validate_after_composer" in requirement:
        return _truthy(requirement.get("should_validate_after_composer"))
    if "should_be_validated_after_composer" in requirement:
        return _truthy(requirement.get("should_be_validated_after_composer"))
    return True


def _classified_has_evidence(requirement: Mapping[str, Any]) -> bool:
    evidence_text = str(requirement.get("evidence_text") or "").strip()
    if evidence_text:
        return True
    if _truthy(requirement.get("explicit_assumption")):
        return True
    return bool(str(requirement.get("assumption") or "").strip())


def _classified_fulfillment_target(
    requirement: Mapping[str, Any],
    selected: Mapping[str, _IndexedComponentCandidate],
) -> _IndexedComponentCandidate | None:
    target_id = str(
        requirement.get("fulfillment_target_component_candidate_id") or ""
    ).strip()
    if target_id:
        return next(
            (
                candidate
                for candidate in selected.values()
                if candidate.component_candidate_id == target_id
            ),
            None,
        )
    role = _normalize_role(
        requirement.get("fulfillment_target_role")
        or requirement.get("target_role")
        or requirement.get("role")
    )
    if role is None:
        return None
    return _selected_candidate_for_role(selected, role)


def _classified_unverified_row(
    requirement: Mapping[str, Any],
    *,
    selected: Mapping[str, _IndexedComponentCandidate],
    reason: str,
    target: _IndexedComponentCandidate | None = None,
) -> dict[str, Any]:
    target = target or _classified_fulfillment_target(requirement, selected)
    role = _classified_requirement_role(requirement)
    return {
        **_classified_validation_base(requirement, role=role),
        "status": "unverified_hard_requirement",
        "satisfied_by": None,
        "component_role": target.internal_role if target is not None else None,
        "component_candidate_id": (
            target.component_candidate_id if target is not None else None
        ),
        "evidence_source": str(requirement.get("evidence_source") or "none"),
        "reason": reason,
        "user_message": (
            _classified_engineer_check(requirement)
            or "Requirement fulfillment needs engineer confirmation."
        ),
    }


def _classified_fulfillment_missing_target_row(
    requirement: Mapping[str, Any],
) -> dict[str, Any]:
    role = _classified_requirement_role(requirement)
    return {
        **_classified_validation_base(requirement, role=role),
        "status": "missing_component",
        "satisfied_by": None,
        "component_role": None,
        "component_candidate_id": None,
        "evidence_source": str(requirement.get("evidence_source") or "none"),
        "reason": "Selected BOM does not contain the component targeted by fulfillment.",
        "user_message": (
            _classified_engineer_check(requirement)
            or "Selected BOM does not prove the included requirement."
        ),
    }


def _classified_fulfillment_satisfied_row(
    requirement: Mapping[str, Any],
    target: _IndexedComponentCandidate,
) -> dict[str, Any]:
    mode = str(requirement.get("fulfillment_mode") or "")
    satisfied_by = {
        FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT: "primary_object",
        FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT: "selected_component",
        FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT: "bundle_or_kit",
    }.get(mode, "component")
    return {
        **_classified_validation_base(
            requirement,
            role=_classified_requirement_role(requirement),
        ),
        "status": "satisfied",
        "satisfied_by": satisfied_by,
        "component_role": target.internal_role,
        "component_candidate_id": target.component_candidate_id,
        "evidence_source": str(requirement.get("evidence_source") or "requirement"),
        "reason": str(requirement.get("evidence_text") or "Evidence provided."),
        "user_message": "Requirement fulfillment has explicit evidence.",
    }


def _classified_validation_base(
    requirement: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    return {
        "capability_id": str(
            requirement.get("capability_id")
            or requirement.get("requirement_id")
            or f"{role}.classified"
        ),
        "role": role,
        "source_text": str(requirement.get("source_text") or "").strip(),
        "requirement_classification": requirement.get("classification"),
        "fulfillment_mode": requirement.get("fulfillment_mode"),
        "fulfillment_target_role": requirement.get("fulfillment_target_role"),
        "fulfillment_target_component_candidate_id": requirement.get(
            "fulfillment_target_component_candidate_id"
        ),
        "evidence_text": str(requirement.get("evidence_text") or "").strip(),
        "engineer_check_ru": _classified_engineer_check(requirement),
        "suggested_engineer_check_ru": _classified_engineer_check(requirement),
        "should_create_bom_role": bool(requirement.get("should_create_bom_role")),
        "should_validate_after_composer": _classified_requirement_should_validate(
            requirement
        ),
    }


def _classified_requirement_role(requirement: Mapping[str, Any]) -> str:
    role = _normalize_role(
        requirement.get("fulfillment_target_role")
        or requirement.get("target_role")
        or requirement.get("role")
    )
    return role or "classified_requirement"


def _classified_engineer_check(requirement: Mapping[str, Any]) -> str:
    return str(
        requirement.get("engineer_check_ru")
        or requirement.get("suggested_engineer_check_ru")
        or requirement.get("source_text")
        or ""
    ).strip()


def _capability_is_primary_object_feature(capability: Mapping[str, Any]) -> bool:
    return (
        str(capability.get("requirement_classification") or "").strip()
        == REQ_CLASS_PRIMARY_OBJECT_FEATURE
    )


def _primary_object_feature_validation_row(
    *,
    capability_id: str,
    role: str,
    capability: Mapping[str, Any],
    selected: Mapping[str, _IndexedComponentCandidate],
) -> dict[str, Any]:
    candidate = _selected_candidate_for_role(selected, role)
    source_text = _capability_source_text(capability)
    base = {
        "capability_id": capability_id,
        "role": role,
        "source_text": source_text,
        "requirement_classification": REQ_CLASS_PRIMARY_OBJECT_FEATURE,
        "target_primary_object": capability.get("target_primary_object"),
        "suggested_engineer_check_ru": capability.get("suggested_engineer_check_ru"),
    }
    if candidate is None:
        return {
            **base,
            "status": "not_applicable",
            "satisfied_by": None,
            "component_role": None,
            "component_candidate_id": None,
            "evidence_source": "component_candidate_matrix",
            "reason": "No selected primary object is tied to this feature requirement.",
            "user_message": "No selected primary object is tied to this feature requirement.",
        }
    parsed = {
        str(key): value
        for key, value in _safe_mapping(capability.get("parsed_requirements")).items()
        if str(key).strip() not in {"required", "included"}
    }
    if not parsed:
        return {
            **base,
            "status": "unverified_hard_requirement",
            "satisfied_by": "component",
            "component_role": role,
            "component_candidate_id": candidate.component_candidate_id,
            "evidence_source": "local_extracted_facts",
            "reason": "No machine-verifiable facts were supplied for this platform feature.",
            "user_message": "Platform feature requirement needs engineer verification.",
        }
    missing: list[str] = []
    mismatches: list[str] = []
    for key, expected in parsed.items():
        status = _primary_feature_fact_status(candidate, key, expected)
        if status == "unknown":
            missing.append(key)
        elif status == "mismatch":
            mismatches.append(key)
    if mismatches:
        return {
            **base,
            "status": "hard_mismatch",
            "satisfied_by": "component",
            "component_role": role,
            "component_candidate_id": candidate.component_candidate_id,
            "evidence_source": "local_extracted_facts",
            "reason": (
                "Selected primary object contradicts feature requirement fields: "
                + ", ".join(mismatches)
            ),
            "user_message": (
                "Selected primary object contradicts a hard platform feature requirement."
            ),
        }
    if missing:
        return {
            **base,
            "status": "unverified_hard_requirement",
            "satisfied_by": "component",
            "component_role": role,
            "component_candidate_id": candidate.component_candidate_id,
            "evidence_source": "local_extracted_facts",
            "reason": (
                "Selected primary object lacks machine-verifiable facts for: "
                + ", ".join(missing)
            ),
            "user_message": (
                "Selected primary object needs engineer verification for a hard feature."
            ),
        }
    return _capability_satisfied_row(
        capability_id,
        role,
        candidate,
        source="local_extracted_facts",
        source_text=source_text,
    )


def _primary_feature_fact_status(
    candidate: _IndexedComponentCandidate,
    key: str,
    expected: Any,
) -> str:
    if expected in (None, "", UNKNOWN_FACT):
        return "satisfied"
    if isinstance(expected, Mapping | list | tuple):
        return "unknown"
    actual = _fact(candidate, key)
    if actual in (None, "", UNKNOWN_FACT):
        return "unknown"
    if isinstance(expected, bool):
        return "satisfied" if _truthy(actual) == expected else "mismatch"
    expected_int = _int_value(expected)
    if expected_int is not None:
        actual_int = _int_value(actual)
        if actual_int is None:
            return "unknown"
        return "satisfied" if actual_int >= expected_int else "mismatch"
    expected_float = _float_value(expected)
    if expected_float is not None:
        actual_float = _float_value(actual)
        if actual_float is None:
            return "unknown"
        return "satisfied" if actual_float >= expected_float else "mismatch"
    expected_text = str(expected).strip()
    actual_text = str(actual).strip()
    if not expected_text:
        return "satisfied"
    if _normalized_token_matches(actual_text, expected_text):
        return "satisfied"
    if expected_text.casefold() in actual_text.casefold():
        return "satisfied"
    return "mismatch"


def _network_product_capability_mismatch(
    candidate: _IndexedComponentCandidate,
    capability: Mapping[str, Any],
) -> str | None:
    parsed = capability.get("parsed_requirements")
    parsed = parsed if isinstance(parsed, Mapping) else {}
    checks = (
        ("port_count", "port_count", _int_at_least, "количество портов"),
        ("uplink_count", "uplink_count", _int_at_least, "количество uplink-портов"),
        ("port_speed", "port_speed_gbps", _speed_at_least, "скорость портов"),
        ("uplink_speed", "uplink_speed_gbps", _speed_at_least, "скорость uplink"),
        ("poe_budget_w", "poe_budget_w", _int_at_least, "PoE budget"),
    )
    for requirement_key, fact_key, predicate, label in checks:
        required = parsed.get(requirement_key)
        if required in (None, "", UNKNOWN_FACT):
            continue
        actual = _fact(candidate, fact_key)
        if not predicate(actual, required):
            return f"Не подтверждено требование: {label} {required}."

    media_checks = (
        ("port_media", "port_media", "тип портов"),
        ("uplink_media", "uplink_media", "тип uplink"),
        ("transceiver_form_factor", "transceiver_form_factor", "форм-фактор трансивера"),
        ("airflow", "airflow", "airflow"),
    )
    for requirement_key, fact_key, label in media_checks:
        required = str(parsed.get(requirement_key) or "").strip()
        if not required or required == UNKNOWN_FACT:
            continue
        actual = str(_fact(candidate, fact_key) or "").strip()
        if not _normalized_token_matches(actual, required):
            return f"Не подтверждено требование: {label} {required}."

    bool_checks = (
        ("poe_required", "poe_supported", "PoE"),
        ("l2_required", "l2_supported", "L2"),
        ("l3_required", "l3_supported", "L3"),
        ("stacking_required", "stacking_supported", "stacking"),
        ("redundant_psu", "redundant_psu", "резервирование БП"),
    )
    for requirement_key, fact_key, label in bool_checks:
        if not _truthy(parsed.get(requirement_key)):
            continue
        if not _truthy(_fact(candidate, fact_key)):
            return f"Не подтверждено требование: {label}."

    required_standard = str(parsed.get("poe_standard") or "").strip()
    if required_standard:
        actual_standard = str(_fact(candidate, "poe_standard") or "").strip()
        if not _poe_standard_satisfies(actual_standard, required_standard):
            return f"Не подтверждено требование: {required_standard}."
    return None


def _storage_product_capability_mismatch(
    candidate: _IndexedComponentCandidate,
    capability: Mapping[str, Any],
    selected: Mapping[str, _IndexedComponentCandidate],
    quantities: Mapping[str, int],
) -> str | None:
    parsed = capability.get("parsed_requirements")
    parsed = parsed if isinstance(parsed, Mapping) else {}
    role = _role_from_capability(capability)
    capacity_checks = (
        ("usable_capacity_tb", "usable_capacity_tb", "полезная емкость"),
        ("raw_capacity_tb", "raw_capacity_tb", "raw емкость"),
    )
    for requirement_key, fact_key, label in capacity_checks:
        required = _float_value(parsed.get(requirement_key))
        if required is None:
            continue
        actual = _float_value(_fact(candidate, fact_key))
        if actual is None or actual < required:
            return f"Не подтверждено требование: {label} {required:g} TB."

    controller_count = _int_value(parsed.get("controller_count"))
    if controller_count is not None:
        actual = _int_value(_fact(candidate, "controller_count"))
        if actual is None or actual < controller_count:
            return f"Не подтверждено требование: контроллеры {controller_count} шт."

    drive_count = _int_value(parsed.get("drive_count"))
    if drive_count is not None:
        selected_quantity = (
            _selected_quantity_total_for_role(selected, quantities, role)
            if role
            else None
        )
        actual_count = _int_value(_fact(candidate, "drive_count")) or selected_quantity
        if actual_count is None or actual_count < drive_count:
            return f"Не подтверждено требование: диски {drive_count} шт."

    drive_capacity = _float_value(parsed.get("drive_capacity_tb"))
    if drive_capacity is not None:
        actual = _float_value(_fact(candidate, "drive_capacity_tb"))
        actual = actual or _candidate_storage_tb(candidate)
        if actual is None or actual < drive_capacity:
            return f"Не подтверждено требование: емкость диска {drive_capacity:g} TB."

    drive_type = str(parsed.get("drive_type") or "").strip()
    if drive_type and drive_type != UNKNOWN_FACT:
        actual = str(_fact(candidate, "drive_type") or "").strip()
        if not _normalized_token_matches(actual, drive_type):
            return f"Не подтверждено требование: тип диска {drive_type}."

    drive_interface = str(parsed.get("drive_interface") or "").strip()
    if drive_interface and drive_interface != UNKNOWN_FACT:
        actual = str(_fact(candidate, "drive_interface") or "").strip()
        fallback = str(_fact(candidate, "storage_interface") or "").strip()
        if not (
            _normalized_token_matches(actual, drive_interface)
            or _normalized_token_matches(fallback, drive_interface)
        ):
            return f"Не подтверждено требование: интерфейс диска {drive_interface}."

    host_protocol = str(parsed.get("host_protocol") or "").strip()
    if host_protocol and host_protocol != UNKNOWN_FACT:
        actual = str(_fact(candidate, "host_protocol") or "").strip()
        if not _normalized_token_matches(actual, host_protocol):
            return f"Не подтверждено требование: протокол {host_protocol}."

    host_port_count = _int_value(parsed.get("host_port_count"))
    if host_port_count is not None:
        actual = _int_value(_fact(candidate, "host_port_count"))
        selected_quantity = (
            _selected_quantity_total_for_role(selected, quantities, role)
            if role
            else None
        )
        actual = actual or selected_quantity
        if actual is None or actual < host_port_count:
            return f"Не подтверждено требование: host-порты {host_port_count} шт."

    host_port_speed = parsed.get("host_port_speed")
    if host_port_speed not in (None, "", UNKNOWN_FACT):
        actual = _fact(candidate, "host_port_speed_gbps")
        if actual == UNKNOWN_FACT:
            actual = _fact(candidate, "host_port_speed")
        if not _speed_at_least(actual, host_port_speed):
            return f"Не подтверждено требование: скорость host-портов {host_port_speed}."

    host_port_media = str(parsed.get("host_port_media") or "").strip()
    if host_port_media and host_port_media != UNKNOWN_FACT:
        actual = str(_fact(candidate, "host_port_media") or "").strip()
        if not _normalized_token_matches(actual, host_port_media):
            return f"Не подтверждено требование: media host-портов {host_port_media}."

    warranty_months = _int_value(parsed.get("warranty_months"))
    if warranty_months is not None:
        actual = _int_value(_fact(candidate, "warranty_months"))
        if actual is None or actual < warranty_months:
            return f"Не подтверждено требование: поддержка {warranty_months} месяцев."

    if role == LICENSE_ROLE and _truthy(parsed.get("license_required")):
        return _missing_linked_role(selected, LICENSE_ROLE, "лицензия")
    if role == SUPPORT_ROLE and _truthy(parsed.get("support_required")):
        return _missing_linked_role(selected, SUPPORT_ROLE, "поддержка")
    return None


def _missing_linked_role(
    selected: Mapping[str, _IndexedComponentCandidate],
    role: str,
    label: str,
) -> str | None:
    if (
        _selected_candidate_for_role(selected, role) is not None
        or _selected_candidate_for_role(selected, STORAGE_SYSTEM_ROLE) is not None
    ):
        return None
    return f"Не подтверждено требование: {label}."


def _int_at_least(actual: Any, required: Any) -> bool:
    actual_int = _int_value(actual)
    required_int = _int_value(required)
    return actual_int is not None and required_int is not None and actual_int >= required_int


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    return text in {"1", "true", "yes", "y", "да", "required", "poe", "poe+", "poe++"}


def _speed_at_least(actual: Any, required: Any) -> bool:
    actual_speed = _speed_to_gbps(actual)
    required_speed = _speed_to_gbps(required)
    return (
        actual_speed is not None
        and required_speed is not None
        and actual_speed >= required_speed
    )


def _speed_to_gbps(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        result = int(value)
        return result if result > 0 else None
    match = re.search(r"\b(1|2\.5|5|10|16|25|32|40|64|100|200|400)", str(value), re.I)
    if not match:
        return None
    return int(float(match.group(1)))


def _normalized_token_matches(actual: str, required: str) -> bool:
    actual_norm = actual.replace(" ", "").replace("-", "").casefold()
    required_norm = required.replace(" ", "").replace("-", "").casefold()
    if not actual_norm or actual_norm == UNKNOWN_FACT:
        return False
    if required_norm in {"sfp", "qsfp"}:
        return actual_norm.startswith(required_norm)
    return actual_norm == required_norm


def _poe_standard_satisfies(actual: str, required: str) -> bool:
    order = {"poe": 1, "poe+": 2, "poe++": 3}
    actual_rank = order.get(actual.replace(" ", "").casefold(), 0)
    required_rank = order.get(required.replace(" ", "").casefold(), 0)
    return actual_rank >= required_rank and required_rank > 0


def _capability_satisfied_row(
    capability_id: str,
    role: str,
    candidate: _IndexedComponentCandidate,
    *,
    source: str,
    source_text: str,
) -> dict[str, Any]:
    return {
        "capability_id": capability_id,
        "role": role,
        "status": "satisfied",
        "satisfied_by": "component",
        "component_role": role,
        "component_candidate_id": candidate.component_candidate_id,
        "source_text": source_text,
        "evidence_source": source,
        "reason": "Selected component covers the hard capability.",
        "user_message": "Selected component covers the hard capability.",
    }


def _capability_source_text(capability: Mapping[str, Any]) -> str:
    return str(
        capability.get("source_text")
        or capability.get("requirement_text")
        or capability.get("capability_id")
        or capability.get("role")
        or ""
    ).strip()


def _platform_onboard_network_satisfies(
    platform: _IndexedComponentCandidate | None,
    requirements: Mapping[str, Any],
) -> bool:
    if platform is None or not _network_required(requirements):
        return False
    return network_facts_satisfy_requirement(
        _network_facts_for_candidate(platform),
        _network_requirement(requirements),
    )


def _platform_power_bundle_satisfies(
    platform: _IndexedComponentCandidate | None,
    requirements: Mapping[str, Any],
) -> bool:
    if platform is None:
        return False
    psu_count = _psu_count_requirement(requirements)
    if psu_count is None:
        return False
    return platform_power_bundle_satisfies(
        _candidate_search_text(platform),
        required_psu_count=psu_count,
        raw_json=platform.source.get("raw") if isinstance(platform.source, Mapping) else None,
    )


def _psu_count_requirement(requirements: Mapping[str, Any]) -> int | None:
    for key in ("psu_count_per_server", "power_supply_min_count", "power_supply_count"):
        value = _int_value(requirements.get(key))
        if value is not None:
            return value
    role_plan = requirements.get("role_plan")
    role_requirements = {}
    if isinstance(role_plan, Mapping):
        role_requirements = role_plan.get("requirements_by_role")
    if isinstance(role_requirements, Mapping):
        power = role_requirements.get(POWER_SUPPLY_ROLE)
        if isinstance(power, Mapping):
            for key in ("psu_count_per_server", "count_per_server", "min_count"):
                value = _int_value(power.get(key))
                if value is not None:
                    return value
    required_capability_pack = {
        "required_capabilities": requirements.get("required_capabilities")
    }
    for capability in _hard_capabilities(required_capability_pack):
        if _role_from_capability(capability) != POWER_SUPPLY_ROLE:
            continue
        parsed = capability.get("parsed_requirements")
        if not isinstance(parsed, Mapping):
            continue
        for key in ("psu_count_per_server", "count_per_server", "min_count", "count"):
            value = _int_value(parsed.get(key))
            if value is not None:
                return value
    return None


def _network_requirement(requirements: Mapping[str, Any]) -> Mapping[str, Any]:
    requirement = requirements.get("network_requirement")
    if isinstance(requirement, Mapping):
        return requirement
    return {
        "required": _network_required(requirements),
        "min_ports_per_server": requirements.get("network_min_ports_per_server"),
        "speed": requirements.get("network_speed"),
        "media": requirements.get("network_media"),
        "interface": requirements.get("network_interface"),
    }


def _network_capability_label(requirements: Mapping[str, Any]) -> str:
    for capability in _mapping_rows(requirements.get("required_capabilities")):
        if str(capability.get("role") or "").strip() == NETWORK_ADAPTER_ROLE:
            return str(capability.get("capability_id") or NETWORK_ADAPTER_ROLE).strip()
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        for capability in _mapping_rows(role_plan.get("required_capabilities")):
            if str(capability.get("role") or "").strip() == NETWORK_ADAPTER_ROLE:
                return str(capability.get("capability_id") or NETWORK_ADAPTER_ROLE).strip()
    return NETWORK_ADAPTER_ROLE


def _network_facts_for_candidate(candidate: _IndexedComponentCandidate) -> dict[str, Any]:
    return {
        "ports_count": candidate.row.get("network_ports_count")
        or _fact(candidate, "network_ports_count"),
        "speed": candidate.row.get("network_speed") or _fact(candidate, "network_speed"),
        "speed_gbps": candidate.row.get("network_speed_gbps")
        or _fact(candidate, "network_speed_gbps"),
        "media": candidate.row.get("network_media") or _fact(candidate, "network_media"),
        "interface": candidate.row.get("network_interface")
        or _fact(candidate, "network_interface"),
    }


def _optional_component_roles(
    *,
    selected: Mapping[str, _IndexedComponentCandidate],
    user_request: str | None,
    normalized_requirements: Any,
    source: _IndexedStockCandidate | None,
) -> list[str]:
    selected_optional_roles = [
        role
        for role in ROLE_ORDER
        if _selected_candidate_for_role(selected, role) is not None
        and role in OPTIONAL_ENGINEER_CHECK_ROLES
    ]
    if not selected_optional_roles:
        return []
    mandatory_optional_roles = _requested_optional_component_roles(
        user_request=user_request,
        normalized_requirements=normalized_requirements,
    )
    mandatory_optional_roles.update(_mandatory_roles(normalized_requirements))
    mandatory_optional_roles.update(_source_optional_component_roles(source))
    return _unique(
        [
            role
            for role in selected_optional_roles
            if role not in mandatory_optional_roles
        ]
    )


def _requested_optional_component_roles(
    *,
    user_request: str | None,
    normalized_requirements: Any,
) -> set[str]:
    requirements = _first_requirements(normalized_requirements)
    text = " ".join(
        part
        for part in [
            user_request or "",
            str(requirements.get("storage") or ""),
            str(requirements.get("network") or ""),
            str(requirements.get("storage_controller") or ""),
            str(requirements.get("network_adapter") or ""),
        ]
        if part
    )
    roles: set[str] = set()
    if any(
        requirements.get(key) not in (None, "", False, UNKNOWN_FACT)
        for key in (
            "storage_controller_required",
            "storage_controller",
            "raid_controller_required",
            "hba_required",
        )
    ) or re.search(r"\b(?:raid|hba)\b|контроллер", text, re.IGNORECASE):
        roles.add(STORAGE_CONTROLLER_ROLE)
    if any(
        requirements.get(key) not in (None, "", False, UNKNOWN_FACT)
        for key in (
            "network_required",
            "network",
            "network_adapter_required",
            "network_adapter",
        )
    ) or re.search(
        r"\b(?:nic|ethernet|lan|10g|25g|40g|100g)\b|сетев|порт",
        text,
        re.IGNORECASE,
    ):
        roles.add(NETWORK_ADAPTER_ROLE)
    return roles


def _source_optional_component_roles(source: _IndexedStockCandidate | None) -> set[str]:
    if source is None:
        return set()
    roles: set[str] = set()
    for component in _mapping_rows(source.row.get("components")):
        role = _normalize_role(component.get("role"))
        if role in OPTIONAL_ENGINEER_CHECK_ROLES:
            roles.add(role)
    return roles


def _optional_component_checks(optional_roles: Sequence[str]) -> list[str]:
    if not optional_roles:
        return []
    role_text = ", ".join(_role_label(role) for role in optional_roles)
    return [
        (
            f"{role_text} добавлены как optional_or_engineer_check: "
            "они не входят в минимальный обязательный расчет без явного подтверждения инженером."
        )
    ]


def _classified_requirement_engineer_checks(
    requirements: Mapping[str, Any],
) -> list[str]:
    role_plan = requirements.get("role_plan")
    rows = []
    if isinstance(role_plan, Mapping):
        rows.extend(_mapping_rows(role_plan.get("classified_requirements")))
    rows.extend(_mapping_rows(requirements.get("classified_requirements")))
    checks: list[str] = []
    for row in rows:
        classification = str(row.get("classification") or "").strip()
        mode = str(row.get("fulfillment_mode") or "").strip()
        if (
            classification
            not in {
                REQ_CLASS_PRIMARY_OBJECT_FEATURE,
                REQ_CLASS_ENGINEERING_CHECK,
            }
            and mode != FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION
        ):
            continue
        check = str(
            row.get("engineer_check_ru")
            or row.get("suggested_engineer_check_ru")
            or row.get("source_text")
            or ""
        ).strip()
        if check:
            checks.append(check)
    return _unique(checks)


def _platform_feature_validation_checks(
    validation_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    checks: list[str] = []
    for row in validation_rows:
        if (
            str(row.get("requirement_classification") or "").strip()
            != REQ_CLASS_PRIMARY_OBJECT_FEATURE
        ):
            continue
        if str(row.get("status") or "").strip() == "satisfied":
            continue
        check = str(
            row.get("suggested_engineer_check_ru")
            or row.get("user_message")
            or row.get("reason")
            or row.get("source_text")
            or ""
        ).strip()
        if check:
            checks.append(check)
    return _unique(checks)


def _first_requirements(normalized_requirements: Any) -> Mapping[str, Any]:
    if isinstance(normalized_requirements, Mapping):
        items = normalized_requirements.get("items")
        if isinstance(items, list):
            for item in items:
                if isinstance(item, Mapping):
                    return item
        return normalized_requirements
    return {}


def _storage_role_for_preference(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    if text in {"SSD", "NVME", "SAS", "SATA"}:
        return SSD_ROLE
    if text == "HDD":
        return HDD_ROLE
    return None


def _network_required(requirements: Mapping[str, Any]) -> bool:
    if bool(requirements.get("network_required")):
        return True
    requirement = requirements.get("network_requirement")
    if isinstance(requirement, Mapping) and bool(requirement.get("required")):
        return True
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        return NETWORK_ADAPTER_ROLE in _string_list(role_plan.get("required_roles"))
    return NETWORK_ADAPTER_ROLE in _string_list(requirements.get("required_roles"))


def _build_evidence_summary(
    selected: Mapping[str, _IndexedComponentCandidate],
    *,
    evidence_by_component_id: Mapping[str, Mapping[str, Any]],
    relation_evidence: Sequence[Mapping[str, Any]],
    evidence_review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not evidence_by_component_id and not relation_evidence:
        return {
            "status": "disabled",
            "evidence_status": "disabled",
            "status_text": "Внешние источники не использовались.",
            "confidence": "unknown",
            "sources_count": 0,
            "component_evidence": [],
            "relation_evidence": [],
            "relation_evidence_count": 0,
            "confirmed": [],
            "confirmed_facts": [],
            "missing": [],
            "not_confirmed": [],
            "fatal_concerns": [],
            "engineering_checks": [],
            "review_decision": "",
            "user_note": "",
        }
    component_summaries = [
        _component_evidence_summary(
            evidence_by_component_id.get(candidate.component_candidate_id)
        )
        for candidate in selected.values()
    ]
    source_count = sum(
        _int_value(summary.get("sources_count")) or 0
        for summary in component_summaries
    )
    relation_summaries = [_relation_evidence_summary(row) for row in relation_evidence]
    source_count += sum(
        _int_value(summary.get("sources_count")) or 0 for summary in relation_summaries
    )
    missing = _unique(
        item
        for summary in component_summaries
        for item in _string_list(summary.get("missing"))
    )
    confirmed = _unique(
        item
        for summary in component_summaries
        for item in _string_list(summary.get("confirmed"))
    )
    relation_missing = _unique(
        item
        for summary in relation_summaries
        for item in _string_list(summary.get("missing"))
    )
    relation_confirmed = _unique(
        item
        for summary in relation_summaries
        for item in _string_list(summary.get("confirmed"))
    )
    relation_checks = _unique(
        item
        for summary in relation_summaries
        for item in _string_list(summary.get("engineering_checks"))
    )
    relation_mismatch = _unique(
        item
        for summary in relation_summaries
        for item in _string_list(summary.get("mismatch_facts"))
    )
    missing = _unique([*missing, *relation_missing])
    confirmed = _unique([*confirmed, *relation_confirmed])
    if source_count and not missing:
        status_text = "Явных конфликтов по найденным источникам не выявлено."
    elif source_count:
        status_text = "Часть совместимости не подтверждена источниками."
    else:
        status_text = "Внешние подтверждения не найдены."
    review = evidence_review if isinstance(evidence_review, Mapping) else {}
    missing.extend(_string_list(review.get("missing_evidence")))
    confirmed.extend(_string_list(review.get("confirmed_facts")))
    status = _recommendation_evidence_status(
        source_count=source_count,
        component_summaries=component_summaries,
        relation_summaries=relation_summaries,
        missing=missing,
        mismatch=relation_mismatch,
    )
    status_text = _recommendation_evidence_status_text(status)
    engineering_checks = _unique(
        [
            *_string_list(review.get("engineering_checks")),
            *relation_checks,
            *(
                ["Проверить CPU support list, правила DIMM-слотов и backplane/контроллер."]
                if source_count
                else []
            ),
        ]
    )
    fatal_concerns = _unique([*_string_list(review.get("fatal_concerns")), *relation_mismatch])
    source_domains = _unique(
        domain
        for summary in [*component_summaries, *relation_summaries]
        for domain in _string_list(summary.get("source_domains"))
    )
    return {
        "status": status,
        "evidence_status": status,
        "status_text": status_text,
        "confidence": _combined_evidence_confidence(
            [*component_summaries, *relation_summaries],
            review,
        ),
        "sources_count": source_count,
        "component_evidence": component_summaries,
        "relation_evidence": relation_summaries,
        "relation_evidence_count": len(relation_summaries),
        "source_domains": source_domains[:8],
        "confirmed": _unique(confirmed)[:8],
        "confirmed_facts": _unique(confirmed)[:8],
        "missing": _unique(missing)[:8],
        "not_confirmed": _unique(missing)[:8],
        "fatal_concerns": fatal_concerns,
        "engineering_checks": engineering_checks,
        "review_decision": str(review.get("decision") or "").strip(),
        "user_note": str(review.get("user_note") or "").strip(),
    }


def _relation_evidence_summary(relation: Mapping[str, Any]) -> dict[str, Any]:
    sources = _mapping_rows(relation.get("sources"))
    status = str(relation.get("status") or "not_confirmed").strip() or "not_confirmed"
    relation_type = str(relation.get("relation_type") or "").strip()
    confirmed = _string_list(relation.get("confirmed_facts"))
    missing = _string_list(relation.get("missing_evidence"))
    mismatch = _string_list(relation.get("mismatch_facts"))
    checks = _string_list(relation.get("engineering_checks"))
    if status == "partially_confirmed" and not missing:
        missing = [_relation_missing_warning(relation_type)]
    if status in {"not_confirmed", "error"} and not missing:
        missing = [_relation_missing_warning(relation_type)]
    return {
        "relation_type": relation_type,
        "recommendation_id": str(relation.get("recommendation_id") or "").strip(),
        "status": status,
        "confidence": str(relation.get("confidence") or "unknown").strip() or "unknown",
        "sources_count": len(sources),
        "confirmed": confirmed,
        "missing": missing,
        "not_confirmed": missing,
        "mismatch_facts": mismatch,
        "engineering_checks": checks,
        "source_domains": _unique(
            str(source.get("domain") or "").strip()
            for source in sources
            if str(source.get("domain") or "").strip()
        ),
    }


def _recommendation_evidence_status(
    *,
    source_count: int,
    component_summaries: Sequence[Mapping[str, Any]],
    relation_summaries: Sequence[Mapping[str, Any]],
    missing: Sequence[str],
    mismatch: Sequence[str],
) -> str:
    if mismatch:
        return "mismatch"
    if source_count <= 0:
        return "not_confirmed"
    relation_statuses = {
        str(summary.get("status") or "").strip() for summary in relation_summaries
    }
    component_statuses = {
        str(summary.get("status") or "").strip() for summary in component_summaries
    }
    if (
        relation_summaries
        and relation_statuses <= {"confirmed"}
        and component_statuses <= {"found"}
        and not missing
    ):
        return "confirmed"
    if relation_statuses & {"partially_confirmed", "not_confirmed", "error"}:
        return "partially_confirmed"
    if missing:
        return "partially_confirmed"
    return "partially_confirmed"


def _recommendation_evidence_status_text(status: str) -> str:
    if status == "confirmed":
        return "Совместимость подтверждена найденными внешними источниками."
    if status == "mismatch":
        return "Источники нашли конфликт совместимости."
    if status == "not_confirmed":
        return "Внешние источники не подтвердили совместимость."
    if status == "partially_confirmed":
        return "Часть совместимости подтверждена, support list/QVL нужно сверить инженеру."
    return "Проверка источников завершилась ошибкой."


def _coerce_llm_evidence_summary(
    value: LlmRecommendationEvidenceSummaryPayload | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, LlmRecommendationEvidenceSummaryPayload):
        source = value.model_dump()
    elif isinstance(value, Mapping):
        source = dict(value)
    else:
        return {}

    raw_status = str(source.get("status") or "").strip().lower()
    status = {
        "found": "confirmed",
        "not_found": "not_confirmed",
        "disabled": "not_confirmed",
        "incompatible": "mismatch",
        "not_compatible": "mismatch",
        "unsupported": "mismatch",
    }.get(raw_status, raw_status)
    if status not in {
        "confirmed",
        "partially_confirmed",
        "not_confirmed",
        "mismatch",
        "error",
    }:
        status = "not_confirmed"

    source_count = max(0, _int_value(source.get("sources_count")) or 0)
    confirmed = _string_list(source.get("confirmed_facts")) or _string_list(
        source.get("confirmed")
    )
    missing = _string_list(source.get("not_confirmed")) or _string_list(
        source.get("missing")
    )
    source_domains = _unique(_string_list(source.get("source_domains")))[:8]
    notes = str(source.get("notes") or source.get("user_note") or "").strip()
    if source_count <= 0 and status in {"confirmed", "partially_confirmed", "error"}:
        status = "not_confirmed"
    elif status == "error" and confirmed:
        status = "partially_confirmed"
    elif status == "error":
        status = "not_confirmed"
    if status == "confirmed":
        status_text = "Совместимость подтверждена найденными источниками."
    elif status == "partially_confirmed":
        status_text = "Часть совместимости не подтверждена источниками."
    elif status == "mismatch":
        status_text = "Источники нашли конфликт совместимости."
    elif status == "error":
        status_text = "Проверка источников завершилась ошибкой."
    else:
        status_text = "Внешние подтверждения не найдены."
    return {
        "status": status,
        "status_text": status_text,
        "confidence": (
            "high"
            if status == "confirmed" and source_count > 0
            else ("medium" if status == "partially_confirmed" and source_count > 0 else "low")
        ),
        "sources_count": source_count,
        "confirmed": _unique(confirmed)[:8],
        "missing": _unique(missing)[:8],
        "not_confirmed": _unique(missing)[:8],
        "source_domains": source_domains,
        "fatal_concerns": [],
        "engineering_checks": [],
        "review_decision": "",
        "user_note": notes,
        "notes": notes,
    }


def _default_online_evidence_summary() -> dict[str, Any]:
    return {
        "status": "not_confirmed",
        "status_text": "Внешние подтверждения не найдены.",
        "confidence": "low",
        "sources_count": 0,
        "confirmed": [],
        "missing": [],
        "not_confirmed": [],
        "source_domains": [],
        "fatal_concerns": [],
        "engineering_checks": [],
        "review_decision": "",
        "user_note": "",
        "notes": "",
    }


def _ready_server_evidence_summary(
    evidence: Mapping[str, Any] | None,
    *,
    relation_evidence: Sequence[Mapping[str, Any]],
    evidence_review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    summary = _component_evidence_summary(evidence)
    relation_summaries = [_relation_evidence_summary(row) for row in relation_evidence]
    review = evidence_review if isinstance(evidence_review, Mapping) else {}
    missing = _unique(
        [
            *_string_list(summary.get("missing")),
            *[
                item
                for relation in relation_summaries
                for item in _string_list(relation.get("missing"))
            ],
            *_string_list(review.get("missing_evidence")),
        ]
    )
    confirmed = _unique(
        [
            *_string_list(summary.get("confirmed")),
            *[
                item
                for relation in relation_summaries
                for item in _string_list(relation.get("confirmed"))
            ],
            *_string_list(review.get("confirmed_facts")),
        ]
    )
    source_count = _int_value(summary.get("sources_count")) or 0
    source_count += sum(
        _int_value(relation.get("sources_count")) or 0 for relation in relation_summaries
    )
    mismatch = _unique(
        item
        for relation in relation_summaries
        for item in _string_list(relation.get("mismatch_facts"))
    )
    status = _recommendation_evidence_status(
        source_count=source_count,
        component_summaries=[summary],
        relation_summaries=relation_summaries,
        missing=missing,
        mismatch=mismatch,
    )
    source_domains = _unique(
        domain
        for item in [summary, *relation_summaries]
        for domain in _string_list(item.get("source_domains"))
    )
    return {
        "status": status,
        "evidence_status": status,
        "status_text": (
            "Явных конфликтов по найденным источникам не выявлено."
            if source_count and not missing
            else (
                "Часть совместимости не подтверждена источниками."
                if source_count
                else "Внешние подтверждения не найдены."
            )
        ),
        "confidence": _combined_evidence_confidence([summary, *relation_summaries], review),
        "sources_count": source_count,
        "component_evidence": [summary],
        "relation_evidence": relation_summaries,
        "relation_evidence_count": len(relation_summaries),
        "source_domains": source_domains[:8],
        "confirmed": confirmed[:8],
        "confirmed_facts": confirmed[:8],
        "missing": missing[:8],
        "not_confirmed": missing[:8],
        "fatal_concerns": _unique([*_string_list(review.get("fatal_concerns")), *mismatch]),
        "engineering_checks": _unique(
            [
                *_string_list(review.get("engineering_checks")),
                *[
                    item
                    for relation in relation_summaries
                    for item in _string_list(relation.get("engineering_checks"))
                ],
            ]
        ),
        "review_decision": str(review.get("decision") or "").strip(),
        "user_note": str(review.get("user_note") or "").strip(),
    }


def _component_evidence_summary(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        return {
            "status": "not_found",
            "confidence": "unknown",
            "sources_count": 0,
            "source_domains": [],
            "confirmed": [],
            "missing": ["источник не найден"],
        }
    facts = _evidence_facts(evidence)
    sources = _mapping_rows(evidence.get("sources"))
    confirmed = _evidence_confirmed_fact_texts(facts)
    missing = _evidence_missing_fact_texts(str(evidence.get("role") or ""), facts)
    return {
        "status": evidence.get("evidence_status") or "unknown",
        "confidence": evidence.get("confidence") or "unknown",
        "sources_count": len(sources),
        "source_domains": _unique(
            str(source.get("domain") or "").strip()
            for source in sources
            if str(source.get("domain") or "").strip()
        ),
        "confirmed": confirmed,
        "missing": missing,
    }


def _evidence_confirmed_fact_texts(facts: Mapping[str, Any]) -> list[str]:
    labels = {
        "vendor": "производитель",
        "platform_family": "семейство платформы",
        "cpu_generation": "поколение CPU",
        "socket_family": "socket",
        "supported_cpu_generation": "поддерживаемое поколение CPU",
        "memory_type": "тип памяти",
        "dimm_slots": "DIMM-слоты",
        "drive_bays": "отсеки накопителей",
        "nvme_support": "NVMe",
        "form_factor": "форм-фактор",
        "psu_info": "PSU",
        "storage_interface": "интерфейс накопителя",
        "capacity": "емкость",
    }
    rows: list[str] = []
    for key, label in labels.items():
        value = facts.get(key)
        if value in (None, "", [], UNKNOWN_FACT):
            continue
        rows.append(f"{label}: {value}")
    notes = str(facts.get("notes") or "").strip()
    if notes:
        rows.append(f"заметки: {notes}")
    return rows


def _evidence_missing_fact_texts(role: str, facts: Mapping[str, Any]) -> list[str]:
    normalized_role = _normalize_role(role) or role
    expected_by_role = {
        SERVER_PLATFORM_ROLE: ["supported_cpu_generation", "socket_family", "memory_type"],
        "platform": ["supported_cpu_generation", "socket_family", "memory_type"],
        CPU_ROLE: ["cpu_generation", "socket_family"],
        RAM_ROLE: ["memory_type", "capacity"],
        SSD_ROLE: ["storage_interface", "capacity"],
        HDD_ROLE: ["storage_interface", "capacity"],
        "storage": ["storage_interface", "capacity"],
        READY_SERVER_CANDIDATE_TYPE: ["platform_family"],
    }
    expected = expected_by_role.get(normalized_role, [])
    missing = [key for key in expected if facts.get(key) in (None, "", [], UNKNOWN_FACT)]
    labels = {
        "supported_cpu_generation": "поддержка поколения CPU",
        "socket_family": "socket",
        "memory_type": "тип памяти",
        "cpu_generation": "поколение CPU",
        "capacity": "емкость",
        "storage_interface": "интерфейс накопителя",
        "platform_family": "семейство платформы",
    }
    return [labels.get(key, key) for key in missing]


def _combined_evidence_confidence(
    summaries: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
) -> str:
    review_confidence = str(review.get("evidence_confidence") or "").strip()
    if review_confidence in {"high", "medium", "low"}:
        return review_confidence
    confidences = {str(summary.get("confidence") or "") for summary in summaries}
    if "high" in confidences:
        return "high"
    if "medium" in confidences:
        return "medium"
    if "low" in confidences:
        return "low"
    return "unknown"


def _component_report_row(
    candidate: _IndexedComponentCandidate,
    quantity: int,
    *,
    quantity_detail: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    pricing_scope: str = "core",
) -> dict[str, Any]:
    price = _decimal_value(candidate.row.get("price_value"))
    line_total = price * quantity if price is not None and pricing_scope == "core" else None
    source = candidate.source
    evidence_summary = _component_evidence_summary(evidence)
    quantity_detail = quantity_detail or {}
    return {
        "role": candidate.internal_role,
        "role_ru": _role_label(candidate.internal_role),
        "component_candidate_id": candidate.component_candidate_id,
        "category_id": source.get("category_id"),
        "producer": candidate.row.get("producer"),
        "part_number": candidate.row.get("part_number"),
        "item_name": candidate.row.get("name"),
        "quantity_required": quantity,
        "server_quantity": quantity_detail.get("server_quantity"),
        "per_server_quantity": quantity_detail.get("per_server_quantity"),
        "quantity_source": quantity_detail.get("quantity_source"),
        "quantity_normalized": quantity_detail.get("quantity_normalized", False),
        "llm_quantity": quantity_detail.get("llm_quantity"),
        "pricing_scope": pricing_scope,
        "optional_component": pricing_scope != "core",
        "available_quantity": candidate.row.get("available_quantity"),
        "reservable_locations": source.get("reservable_locations"),
        "price_value": candidate.row.get("price_value"),
        "price_currency": candidate.row.get("price_currency"),
        "line_total_value": _json_decimal(line_total),
        "line_total_currency": (
            candidate.row.get("price_currency") if line_total is not None else None
        ),
        "facts": candidate.row.get("extracted_facts"),
        "cpu_cores": candidate.row.get("cpu_cores"),
        "cpu_over_requirement": candidate.row.get("cpu_over_requirement"),
        "storage_capacity_tb": candidate.row.get("storage_capacity_tb")
        or quantity_detail.get("storage_capacity_tb"),
        "storage_required_capacity_tb": quantity_detail.get("storage_required_capacity_tb"),
        "storage_over_requirement": candidate.row.get("storage_over_requirement"),
        "raw_capacity_tb": candidate.row.get("raw_capacity_tb")
        or _fact(candidate, "raw_capacity_tb"),
        "usable_capacity_tb": candidate.row.get("usable_capacity_tb")
        or _fact(candidate, "usable_capacity_tb"),
        "redundancy_level": candidate.row.get("redundancy_level")
        or _fact(candidate, "redundancy_level"),
        "controller_count": candidate.row.get("controller_count")
        or _fact(candidate, "controller_count"),
        "drive_count": candidate.row.get("drive_count") or _fact(candidate, "drive_count"),
        "drive_capacity_tb": candidate.row.get("drive_capacity_tb")
        or _fact(candidate, "drive_capacity_tb"),
        "drive_type": candidate.row.get("drive_type") or _fact(candidate, "drive_type"),
        "drive_interface": candidate.row.get("drive_interface")
        or _fact(candidate, "drive_interface"),
        "host_protocol": candidate.row.get("host_protocol")
        or _fact(candidate, "host_protocol"),
        "host_port_count": candidate.row.get("host_port_count")
        or _fact(candidate, "host_port_count"),
        "host_port_speed": candidate.row.get("host_port_speed")
        or _fact(candidate, "host_port_speed"),
        "host_port_speed_gbps": candidate.row.get("host_port_speed_gbps")
        or _fact(candidate, "host_port_speed_gbps"),
        "host_port_media": candidate.row.get("host_port_media")
        or _fact(candidate, "host_port_media"),
        "warranty_months": candidate.row.get("warranty_months")
        or _fact(candidate, "warranty_months"),
        "ram_module_capacity_gb": candidate.row.get("ram_module_capacity_gb")
        or quantity_detail.get("ram_module_capacity_gb"),
        "ram_required_gb_per_server": quantity_detail.get("ram_required_gb_per_server"),
        "ram_total_gb_per_server": quantity_detail.get("ram_total_gb_per_server"),
        "ram_over_requirement_gb": candidate.row.get("ram_over_requirement_gb"),
        "network_ports_count": candidate.row.get("network_ports_count")
        or _fact(candidate, "network_ports_count"),
        "network_speed": candidate.row.get("network_speed") or _fact(candidate, "network_speed"),
        "network_speed_gbps": candidate.row.get("network_speed_gbps")
        or _fact(candidate, "network_speed_gbps"),
        "network_media": candidate.row.get("network_media") or _fact(candidate, "network_media"),
        "network_interface": candidate.row.get("network_interface")
        or _fact(candidate, "network_interface"),
        "network_required_ports_per_server": quantity_detail.get(
            "network_required_ports_per_server"
        ),
        "network_ports_per_adapter": quantity_detail.get("network_ports_per_adapter"),
        "port_count": candidate.row.get("port_count") or _fact(candidate, "port_count"),
        "port_speed": candidate.row.get("port_speed") or _fact(candidate, "port_speed"),
        "port_speed_gbps": candidate.row.get("port_speed_gbps")
        or _fact(candidate, "port_speed_gbps"),
        "port_media": candidate.row.get("port_media") or _fact(candidate, "port_media"),
        "uplink_count": candidate.row.get("uplink_count") or _fact(candidate, "uplink_count"),
        "uplink_speed": candidate.row.get("uplink_speed") or _fact(candidate, "uplink_speed"),
        "uplink_speed_gbps": candidate.row.get("uplink_speed_gbps")
        or _fact(candidate, "uplink_speed_gbps"),
        "uplink_media": candidate.row.get("uplink_media") or _fact(candidate, "uplink_media"),
        "poe_supported": candidate.row.get("poe_supported") or _fact(candidate, "poe_supported"),
        "poe_budget_w": candidate.row.get("poe_budget_w") or _fact(candidate, "poe_budget_w"),
        "poe_standard": candidate.row.get("poe_standard") or _fact(candidate, "poe_standard"),
        "l2_supported": candidate.row.get("l2_supported") or _fact(candidate, "l2_supported"),
        "l3_supported": candidate.row.get("l3_supported") or _fact(candidate, "l3_supported"),
        "stacking_supported": candidate.row.get("stacking_supported")
        or _fact(candidate, "stacking_supported"),
        "airflow": candidate.row.get("airflow") or _fact(candidate, "airflow"),
        "redundant_psu": candidate.row.get("redundant_psu") or _fact(candidate, "redundant_psu"),
        "transceiver_form_factor": candidate.row.get("transceiver_form_factor")
        or _fact(candidate, "transceiver_form_factor"),
        "fit_label": candidate.row.get("fit_label"),
        "fit_reason": candidate.row.get("fit_reason"),
        "evidence": dict(evidence or {}),
        "evidence_status": evidence_summary.get("status"),
        "evidence_confidence": evidence_summary.get("confidence"),
        "evidence_source_count": evidence_summary.get("sources_count"),
        "evidence_confirmed": evidence_summary.get("confirmed"),
        "evidence_missing": evidence_summary.get("missing"),
    }


def _build_available_quantity(components: list[Mapping[str, Any]]) -> int | None:
    quantities: list[int] = []
    for component in components:
        available = _int_value(component.get("available_quantity"))
        required = _int_value(component.get("quantity_required"))
        if available is None or required is None or required <= 0:
            return None
        quantities.append(available // required)
    return min(quantities) if quantities else None


def _role_label(role: str) -> str:
    labels = {
        SERVER_PLATFORM_ROLE: "Платформа",
        CPU_ROLE: "CPU",
        RAM_ROLE: "RAM",
        SSD_ROLE: "SSD",
        HDD_ROLE: "HDD",
        STORAGE_CONTROLLER_ROLE: "Контроллеры",
        NETWORK_ADAPTER_ROLE: "Сетевые адаптеры",
    }
    labels.update(
        {
            SWITCH_ROLE: "Коммутаторы",
        ROUTER_ROLE: "Маршрутизаторы",
        FIREWALL_ROLE: "Межсетевые экраны",
        ACCESS_POINT_ROLE: "Точки доступа",
        STORAGE_SYSTEM_ROLE: "СХД",
        STORAGE_ARRAY_CONTROLLER_ROLE: "Контроллеры СХД",
        CONTROLLER_MODULE_ROLE: "Модули контроллера",
        DISK_SHELF_ROLE: "Дисковые полки",
        DRIVE_ROLE: "Диски",
        CACHE_ROLE: "Кэш",
        HOST_PORT_ROLE: "Host-порты",
        PROTOCOL_MODULE_ROLE: "Протокольные модули",
        GPU_ROLE: "GPU",
            TRANSCEIVER_ROLE: "Трансиверы",
            DAC_CABLE_ROLE: "DAC-кабели",
            CABLE_ROLE: "Кабели",
            POWER_SUPPLY_ROLE: "Блоки питания",
            RAIL_KIT_ROLE: "Рельсы",
            LICENSE_ROLE: "Лицензии",
            SUPPORT_ROLE: "Поддержка",
            STACKING_MODULE_ROLE: "Модули стекирования",
            OTHER_ACCESSORY_ROLE: "Аксессуары",
        }
    )
    return labels.get(role, role)


def _completeness_label(status: str, missing_roles: list[str]) -> str:
    if status == "complete":
        return "Компоненты подобраны, требуется инженерная проверка"
    missing = ", ".join(_role_label(role) for role in missing_roles)
    return f"Неполная сборка: не подобраны {missing}"


def _total_price_note(
    missing_roles: Sequence[str],
    *,
    optional_roles: Sequence[str] | None = None,
) -> str | None:
    parts: list[str] = []
    if missing_roles:
        parts.append("без " + ", ".join(_role_label(role) for role in missing_roles))
    if optional_roles:
        parts.append(
            "опциональные "
            + ", ".join(_role_label(role) for role in optional_roles)
            + " не входят в минимальный расчет"
        )
    return "; ".join(parts) if parts else None


def _critical_checks(
    values: list[str],
    *,
    product_group: str = SERVER_PRODUCT_GROUP,
) -> list[str]:
    if product_group in {NETWORK_PRODUCT_GROUP, STORAGE_PRODUCT_GROUP}:
        return sanitize_engineer_checks_for_product_group(
            values,
            product_group=product_group,
        )
    defaults = [
        "Проверить список поддерживаемых CPU платформы",
        "Проверить совместимость RAM",
        "Проверить комплектацию 2 БП",
    ]
    return _unique([*_string_list(values), *defaults])


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("candidate_id") or "").strip()


def _is_partial_build(candidate: Mapping[str, Any]) -> bool:
    status = str(candidate.get("completeness_status") or "").strip()
    if status == "incomplete":
        return True
    return bool(_string_list(candidate.get("missing_component_roles")))


def _stock_candidate_display_name(candidate: Mapping[str, Any]) -> str:
    producer = str(candidate.get("producer") or "").strip()
    part_number = str(candidate.get("part_number") or "").strip()
    item_name = str(candidate.get("item_name") or candidate.get("name") or "").strip()
    platform = candidate.get("platform")
    if isinstance(platform, Mapping):
        platform_producer = str(platform.get("producer") or "").strip()
        platform_part_number = str(platform.get("part_number") or "").strip()
        platform_name = str(platform.get("name") or platform.get("item_name") or "").strip()
        if platform_producer and platform_part_number:
            return f"{platform_producer} {platform_part_number}"
        if platform_part_number:
            return platform_part_number
        if platform_name:
            return platform_name
    if producer and part_number:
        return f"{producer} {part_number}"
    if part_number:
        return part_number
    if item_name:
        return item_name
    return str(candidate.get("item_id") or candidate.get("candidate_id") or "вариант")


def _ready_candidate_search_text(candidate: Mapping[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key) or "").strip()
        for key in (
            "producer",
            "part_number",
            "item_name",
            "name",
            "display_name",
            "product_name",
            "product_description",
            "product_notes",
        )
        if str(candidate.get(key) or "").strip()
    )


def _has_serious_ready_gap(missing: list[str]) -> bool:
    markers = (
        "ram",
        "оператив",
        "cpu",
        "процесс",
        "ssd",
        "hdd",
        "накоп",
        "остаток ниже",
        "остаток не найден",
        "stock",
    )
    return any(any(marker in item.casefold() for marker in markers) for item in missing)


def _ready_server_quantity(
    candidate: Mapping[str, Any],
    normalized_requirements: Any,
) -> int | None:
    raw = candidate.get("raw") if isinstance(candidate.get("raw"), Mapping) else {}
    quantity = _int_value(raw.get("quantity_required"))
    if quantity is not None:
        return quantity
    requirements = _first_requirements(normalized_requirements)
    quantity = _int_value(requirements.get("server_qty"))
    if quantity is not None:
        return quantity
    return 1


def _recommendation_component_summary(
    recommendation: LlmRecommendationPayload,
    source_candidate: Mapping[str, Any],
) -> dict[str, str]:
    summary = {
        "platform": "",
        "cpu": "",
        "ram": "",
        "storage": "",
    }

    components = _mapping_rows(source_candidate.get("components"))
    summary["platform"] = _component_summary_text(components, SERVER_PLATFORM_ROLE)
    summary["cpu"] = _component_summary_text(components, CPU_ROLE)
    summary["ram"] = _component_summary_text(components, RAM_ROLE)
    summary["storage"] = " / ".join(
        item
        for item in [
            _component_summary_text(components, SSD_ROLE),
            _component_summary_text(components, HDD_ROLE),
        ]
        if item
    )
    if any(summary.values()):
        return summary

    facts = source_candidate.get("extracted_facts")
    if not isinstance(facts, Mapping):
        facts = _ready_candidate_facts(source_candidate)
    display_name = _stock_candidate_display_name(source_candidate)
    summary["platform"] = display_name
    ram_gb = _int_value(facts.get("ram_gb")) if isinstance(facts, Mapping) else None
    if ram_gb is not None:
        summary["ram"] = f"{ram_gb} ГБ по данным наименования"
    storage_type = (
        str(facts.get("storage_type") or "").strip()
        if isinstance(facts, Mapping)
        else ""
    )
    if storage_type:
        summary["storage"] = storage_type
    if any(summary.values()):
        return summary
    return summary


def _component_summary_text(components: list[Mapping[str, Any]], role: str) -> str:
    for component in components:
        if _normalize_role(component.get("role")) != role:
            continue
        parts = [
            str(component.get("producer") or "").strip(),
            str(component.get("part_number") or "").strip(),
        ]
        display = " ".join(part for part in parts if part)
        if display:
            return display
        return str(component.get("item_name") or component.get("name") or "").strip()
    return ""


def _clean_quantities(values: Mapping[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in values.items():
        quantity = _int_value(value)
        if quantity is not None and quantity > 0:
            result[str(key)] = quantity
    return result


def _normalized_missing_roles(values: list[str]) -> list[str]:
    roles: list[str] = []
    for value in values:
        role = _normalize_role(value)
        if role is not None:
            roles.append(role)
    return _unique(roles)


def _validated_right_size_note(
    llm_note: str | None,
    right_size_summary: Mapping[str, Any],
) -> str:
    deterministic_note = str(right_size_summary.get("right_size_note") or "").strip()
    if right_size_summary.get("requirement_fit") == "overfit_with_reason":
        return deterministic_note
    clean_llm_note = str(llm_note or "").strip()
    if (
        deterministic_note
        and deterministic_note != "Подбор: минимально подходящий по требованиям"
        and (not clean_llm_note or _is_generic_right_size_note(clean_llm_note))
    ):
        return deterministic_note
    return clean_llm_note or deterministic_note


def _is_generic_right_size_note(value: str) -> bool:
    lowered = value.casefold()
    return (
        "минимально" in lowered
        or "minimal" in lowered
        or "закрывает требования" in lowered
    )


def _confidence_score(confidence: str) -> int:
    return {"low": 60, "medium": 75, "high": 85}.get(confidence, 60)


def _normalized_confidence_label(confidence: str) -> str:
    return confidence if confidence in {"low", "medium", "high"} else "low"


def _evidence_confidence_penalty(summary: Mapping[str, Any]) -> int:
    if not isinstance(summary, Mapping):
        return 0
    if _string_list(summary.get("fatal_concerns")):
        return 30
    status = str(summary.get("status") or "").strip()
    if status == "mismatch":
        return 30
    if status == "disabled":
        return 6
    source_count = _int_value(summary.get("sources_count")) or 0
    if status in {"not_found", "not_confirmed", "error"} or source_count <= 0:
        return 18
    if status == "partially_confirmed" or _string_list(summary.get("missing")):
        return 6
    return 0


def _adjusted_confidence_label(confidence: str, *, penalty: int) -> str:
    if penalty >= 15:
        return "low"
    if penalty > 0 and confidence == "high":
        return "medium"
    return confidence if confidence in {"low", "medium", "high"} else "low"


def _engineering_confidence_label(summary: Mapping[str, Any]) -> str:
    if not isinstance(summary, Mapping):
        return "preliminary_requires_engineer_review"
    if _string_list(summary.get("fatal_concerns")):
        return "not_confirmed_requires_engineer_review"
    status = str(summary.get("status") or "").strip()
    source_count = _int_value(summary.get("sources_count")) or 0
    if status in {"disabled", "not_found", "not_confirmed", "error"} or source_count <= 0:
        return "preliminary_requires_engineer_review"
    if status == "mismatch":
        return "not_confirmed_requires_engineer_review"
    if status == "partially_confirmed" or _string_list(summary.get("missing")):
        return "partially_source_checked_requires_engineer_review"
    return "source_checked_requires_engineer_review"


def _displayed_confidence_text(
    commercial_confidence: str,
    evidence_summary: Mapping[str, Any],
) -> str:
    commercial_label = {
        "high": "высокое",
        "medium": "среднее",
        "low": "низкое",
    }.get(_normalized_confidence_label(commercial_confidence), "низкое")
    engineering_label = {
        "preliminary_requires_engineer_review": "предварительно, требуется проверка",
        "not_confirmed_requires_engineer_review": "не подтверждено, требуется проверка",
        "partially_source_checked_requires_engineer_review": (
            "частично проверено источниками, требуется проверка"
        ),
        "source_checked_requires_engineer_review": (
            "проверено по источникам, требуется финальная проверка"
        ),
    }[_engineering_confidence_label(evidence_summary)]
    return (
        f"Коммерческое соответствие: {commercial_label}; "
        f"инженерная подтвержденность: {engineering_label}."
    )


def _fact(candidate: _IndexedComponentCandidate, key: str) -> str:
    facts = candidate.row.get("extracted_facts")
    if isinstance(facts, Mapping):
        value = facts.get(key)
        if value not in (None, ""):
            return str(value)
    value = candidate.row.get(key)
    if value not in (None, ""):
        return str(value)
    inferred = _inferred_fact_from_name(candidate, key)
    if inferred != UNKNOWN_FACT:
        return inferred
    return UNKNOWN_FACT


def _bool_fact(candidate: _IndexedComponentCandidate, key: str) -> bool:
    facts = candidate.row.get("extracted_facts")
    if isinstance(facts, Mapping):
        value = facts.get(key)
        if value not in (None, ""):
            return bool(value)
    if key == "nvme_support":
        text = _candidate_search_text(candidate)
        if _detect_nvme_support(text):
            return True
    return False


def _normalize_role(value: Any) -> str | None:
    text = str(value or "").strip()
    if text == "unknown":
        return UNMAPPED_ROLE
    return INTERNAL_ROLE_BY_PROMPT_ROLE.get(text)


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _clean_notes(value: list[str]) -> list[str]:
    return _unique([item.strip() for item in value if item and item.strip()])


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None


def _decimal_value(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value).replace(" ", ""))
    except (InvalidOperation, ValueError):
        return None


def _candidate_name(candidate: _IndexedComponentCandidate) -> str:
    parts = [
        str(candidate.row.get("producer") or "").strip(),
        str(candidate.row.get("part_number") or "").strip(),
    ]
    display = " ".join(part for part in parts if part)
    if display:
        return display
    return str(candidate.row.get("name") or candidate.component_candidate_id)


def _json_decimal(value: Decimal | None) -> str | None:
    if value is None:
        return None
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _json_decimal(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _unique(values: list[str] | Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
