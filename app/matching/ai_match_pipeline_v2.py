from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.ocs_anchor_categories import OcsAnchorCategory, load_ocs_anchor_categories
from app.core.config import LlmSettings, WebEvidenceSettings, get_llm_settings
from app.db.models import DistributorCategory, DistributorProduct, DistributorStockPrice
from app.evidence.web_evidence import EvidenceSearchCache, WebSearchProvider
from app.llm.base import LlmClient, LlmError
from app.llm.configuration_composer import (
    FULL_BROAD_MATRIX_EXPOSURE_MODE,
    LlmCallBudget,
    LlmConfiguratorOutcome,
    budgeted_llm_client,
    build_llm_configurator_package,
    compose_llm_configurations,
    llm_call_budget_diagnostics,
    prepare_v2_composer_package,
)
from app.llm.openai_compatible import OpenAICompatibleLlmClient
from app.matching.match_engine import (
    ACCESS_POINT_ROLE,
    CABLE_ROLE,
    CACHE_ROLE,
    CONTROLLER_MODULE_ROLE,
    CPU_ROLE,
    DAC_CABLE_ROLE,
    DISK_SHELF_ROLE,
    DRIVE_ROLE,
    FIREWALL_ROLE,
    GPU_ROLE,
    HDD_ROLE,
    HOST_PORT_ROLE,
    LICENSE_ROLE,
    NETWORK_ADAPTER_ROLE,
    NETWORK_PRODUCT_GROUP,
    OTHER_ACCESSORY_ROLE,
    POWER_SUPPLY_ROLE,
    PROTOCOL_MODULE_ROLE,
    RAIL_KIT_ROLE,
    RAM_ROLE,
    READY_SERVER_CANDIDATE_TYPE,
    ROUTER_ROLE,
    SERVER_PLATFORM_ROLE,
    SERVER_PRODUCT_GROUP,
    SSD_ROLE,
    STACKING_MODULE_ROLE,
    STATUS_NO_STOCK_MATCH,
    STATUS_PARTIAL_STOCK_MATCHED,
    STATUS_STOCK_MATCHED,
    STORAGE_ARRAY_CONTROLLER_ROLE,
    STORAGE_CONTROLLER_ROLE,
    STORAGE_PRODUCT_GROUP,
    STORAGE_SYSTEM_ROLE,
    SUPPORT_ROLE,
    SWITCH_ROLE,
    TRANSCEIVER_ROLE,
    MatchResult,
    _available_quantity,
    _ComponentCandidate,
    _extract_product_facts,
    _jsonable,
    _reservable_locations,
    _select_price,
    _unique,
)
from app.matching.requirement_execution_contract import (
    build_execution_contract,
    selected_components_by_role_from_recommendation,
)
from app.matching.spec_schema import StockSpec
from app.planning import role_lifecycle
from app.policies.product_group_policy import get_product_group_profile
from app.reports.composer_result_normalizer import (
    COMPOSER_NO_SAFE_COMPLETE_BOM,
    normalize_composer_result,
)

PIPELINE_VERSION = "v2_composer_first"
COMPOSER_REJECTED_BY_VALIDATION = "composer_rejected_by_validation"
COMPOSER_SCHEMA_VALIDATION_FAILED = "composer_schema_validation_failed"
CANDIDATE_UNIVERSE_PLANNER_MODE = "llm_candidate_universe_planner_v2"
CANDIDATE_UNIVERSE_PLANNER_REPAIRED_MODE = "llm_candidate_universe_planner_v2_repaired"
CANDIDATE_UNIVERSE_PLANNER_FALLBACK_MODE = "diagnostic_fallback"
CANDIDATE_UNIVERSE_PLANNER_SOURCE = "candidate_universe_planner_v2"
_ALLOWED_PRODUCT_GROUPS = {
    SERVER_PRODUCT_GROUP,
    NETWORK_PRODUCT_GROUP,
    STORAGE_PRODUCT_GROUP,
    "unknown",
}
_ALLOWED_CONFIDENCE = {"high", "medium", "low"}
_V2_COMPACT_PACKAGE_DIAGNOSTIC_KEYS = (
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
    "compact_candidate_count_by_role",
    "compact_candidate_total",
    "compact_candidate_ids_by_role",
    "compact_candidate_ids_hash",
    "compact_package_full_matrix_used",
    "package_candidate_loss",
    "provider_context_limit_retry_compact_attempted",
    "provider_context_limit_retry_compact_success",
    "provider_context_limit_original_chars",
    "provider_context_limit_compact_chars",
    "provider_context_limit_after_compact",
)

_ROLE_MATRIX_KEY = {
    READY_SERVER_CANDIDATE_TYPE: "ready_server_candidates",
    SERVER_PLATFORM_ROLE: "platform_candidates",
    SWITCH_ROLE: "switch_candidates",
    ROUTER_ROLE: "router_candidates",
    FIREWALL_ROLE: "firewall_candidates",
    ACCESS_POINT_ROLE: "access_point_candidates",
    CPU_ROLE: "cpu_candidates",
    RAM_ROLE: "ram_candidates",
    DRIVE_ROLE: "drive_candidates",
    SSD_ROLE: "ssd_candidates",
    HDD_ROLE: "hdd_candidates",
    STORAGE_CONTROLLER_ROLE: "storage_controller_candidates",
    NETWORK_ADAPTER_ROLE: "network_adapter_candidates",
    GPU_ROLE: "gpu_candidates",
    TRANSCEIVER_ROLE: "transceiver_candidates",
    DAC_CABLE_ROLE: "dac_cable_candidates",
    CABLE_ROLE: "cable_candidates",
    POWER_SUPPLY_ROLE: "power_supply_candidates",
    RAIL_KIT_ROLE: "rail_kit_candidates",
    LICENSE_ROLE: "license_candidates",
    SUPPORT_ROLE: "support_candidates",
    STACKING_MODULE_ROLE: "stacking_module_candidates",
    OTHER_ACCESSORY_ROLE: "other_accessory_candidates",
    STORAGE_SYSTEM_ROLE: "storage_system_candidates",
    STORAGE_ARRAY_CONTROLLER_ROLE: "controller_candidates",
    CONTROLLER_MODULE_ROLE: "controller_module_candidates",
    DISK_SHELF_ROLE: "disk_shelf_candidates",
    CACHE_ROLE: "cache_candidates",
    HOST_PORT_ROLE: "host_port_candidates",
    PROTOCOL_MODULE_ROLE: "protocol_module_candidates",
}

_PROMPT_ROLE_BY_INTERNAL_ROLE = {
    READY_SERVER_CANDIDATE_TYPE: READY_SERVER_CANDIDATE_TYPE,
    SERVER_PLATFORM_ROLE: "platform",
    **{
        role: role
        for role in (
            SWITCH_ROLE,
            ROUTER_ROLE,
            FIREWALL_ROLE,
            ACCESS_POINT_ROLE,
            CPU_ROLE,
            RAM_ROLE,
            DRIVE_ROLE,
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
            STORAGE_SYSTEM_ROLE,
            STORAGE_ARRAY_CONTROLLER_ROLE,
            CONTROLLER_MODULE_ROLE,
            DISK_SHELF_ROLE,
            CACHE_ROLE,
            HOST_PORT_ROLE,
            PROTOCOL_MODULE_ROLE,
        )
    },
}

_SERVER_BASELINE_ROLES = (
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
    OTHER_ACCESSORY_ROLE,
)
_NETWORK_BASELINE_ROLES = (
    SWITCH_ROLE,
    ROUTER_ROLE,
    FIREWALL_ROLE,
    ACCESS_POINT_ROLE,
    TRANSCEIVER_ROLE,
    DAC_CABLE_ROLE,
    CABLE_ROLE,
    POWER_SUPPLY_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
    STACKING_MODULE_ROLE,
)
_STORAGE_BASELINE_ROLES = (
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
    STORAGE_CONTROLLER_ROLE,
    CABLE_ROLE,
    POWER_SUPPLY_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
)

_ROLE_KEYWORDS = {
    SERVER_PLATFORM_ROLE: (
        "server platform",
        "barebone",
        "chassis",
        "server system",
        "rack server",
        "1u",
        "2u",
    ),
    READY_SERVER_CANDIDATE_TYPE: ("ready server", "assembled server", "turnkey server"),
    CPU_ROLE: ("cpu", "processor", "xeon", "epyc"),
    RAM_ROLE: ("ram", "memory", "rdimm", "lrdimm", "dimm", "ddr4", "ddr5"),
    SSD_ROLE: ("ssd", "solid state", "nvme", "sata ssd", "sas ssd"),
    HDD_ROLE: ("hdd", "hard drive", "hard disk"),
    DRIVE_ROLE: ("drive", "disk", "storage drive"),
    STORAGE_CONTROLLER_ROLE: (
        "storage controller",
        "raid",
        "hba",
        "lsi",
        "sas controller",
        "tri-mode",
    ),
    NETWORK_ADAPTER_ROLE: (
        "network adapter",
        "nic",
        "ethernet adapter",
        "x710",
        "sfp",
        "sfp+",
        "sfp28",
        "10gbe",
        "25gbe",
    ),
    POWER_SUPPLY_ROLE: ("power supply", "psu", "power module"),
    CABLE_ROLE: ("cable", "cord", "c13", "c14", "schuko", "dac"),
    DAC_CABLE_ROLE: ("dac", "direct attach"),
    TRANSCEIVER_ROLE: ("transceiver", "optic", "sfp", "qsfp"),
    RAIL_KIT_ROLE: ("rail", "rail kit"),
    LICENSE_ROLE: ("license", "subscription"),
    SUPPORT_ROLE: ("support", "service", "warranty"),
    OTHER_ACCESSORY_ROLE: ("accessory", "fan", "cooling", "kit"),
    GPU_ROLE: ("gpu", "accelerator", "nvidia"),
    SWITCH_ROLE: ("switch", "poe", "uplink", "stacking", "l2", "l3"),
    ROUTER_ROLE: ("router",),
    FIREWALL_ROLE: ("firewall", "ngfw", "utm"),
    ACCESS_POINT_ROLE: ("access point", "wifi", "wi-fi"),
    STACKING_MODULE_ROLE: ("stack", "stacking"),
    STORAGE_SYSTEM_ROLE: ("storage system", "storage array", "san", "nas", "disk array"),
    STORAGE_ARRAY_CONTROLLER_ROLE: ("storage controller", "controller", "dual controller"),
    CONTROLLER_MODULE_ROLE: ("controller module", "controller canister"),
    DISK_SHELF_ROLE: ("disk shelf", "drive shelf", "expansion shelf"),
    CACHE_ROLE: ("cache", "flash cache"),
    HOST_PORT_ROLE: ("host port", "fc port", "iscsi port", "nvme-of", "host interface"),
    PROTOCOL_MODULE_ROLE: ("protocol module", "fc module", "iscsi module", "nvme-of module"),
}

_REQ_CLASS_PURCHASABLE_COMPONENT_ROLE = "purchasable_component_role"
_REQ_CLASS_PRIMARY_OBJECT_FEATURE = "primary_object_feature"
_REQ_CLASS_ACCESSORY_OR_CONSUMABLE = "accessory_or_consumable"
_REQ_CLASS_SERVICE_OR_SUPPORT = "service_or_support"
_REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT = "logistics_or_commercial_constraint"
_REQ_CLASS_ENGINEERING_CHECK = "engineering_check"
_REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING = (
    "out_of_scope_or_unmapped_non_blocking"
)
_REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE = (
    "blocking_unmapped_purchasable_role"
)

_FULFILLMENT_SEPARATE_COMPONENT_REQUIRED = "separate_component_required"
_FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT = "included_in_primary_object"
_FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT = "included_in_selected_component"
_FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT = "included_in_bundle_or_kit"
_FULFILLMENT_SERVICE_OR_SUPPORT = "service_or_support"
_FULFILLMENT_LOGISTICS_CONSTRAINT = "logistics_constraint"
_FULFILLMENT_ENGINEERING_CHECK_ONLY = "engineering_check_only"
_FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION = "unverified_requires_confirmation"
_FULFILLMENT_NOT_APPLICABLE = "not_applicable"

_NON_BLOCKING_REQUIREMENT_CLASSES = {
    _REQ_CLASS_PRIMARY_OBJECT_FEATURE,
    _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
    _REQ_CLASS_ENGINEERING_CHECK,
    _REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
}
_NON_BLOCKING_FULFILLMENT_MODES = {
    _FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT,
    _FULFILLMENT_INCLUDED_IN_SELECTED_COMPONENT,
    _FULFILLMENT_INCLUDED_IN_BUNDLE_OR_KIT,
    _FULFILLMENT_LOGISTICS_CONSTRAINT,
    _FULFILLMENT_ENGINEERING_CHECK_ONLY,
    _FULFILLMENT_UNVERIFIED_REQUIRES_CONFIRMATION,
    _FULFILLMENT_NOT_APPLICABLE,
}
_NETWORK_PRIMARY_DEVICE_ROLES = {
    SWITCH_ROLE,
    ROUTER_ROLE,
    FIREWALL_ROLE,
    ACCESS_POINT_ROLE,
}
_NETWORK_ACCESSORY_ENGINEERING_ROLES = {
    TRANSCEIVER_ROLE,
    DAC_CABLE_ROLE,
    CABLE_ROLE,
    STACKING_MODULE_ROLE,
    LICENSE_ROLE,
    SUPPORT_ROLE,
    POWER_SUPPLY_ROLE,
    OTHER_ACCESSORY_ROLE,
}
_STORAGE_DRIVE_LIKE_ROLES = {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}
_PACKAGE_SIZE_SKIP_REASONS = {
    "package_over_budget_before_composer",
    "package_over_budget_after_distillation",
    "package_over_budget_after_full_matrix_failure",
    "full_matrix_evaluation_timeout_package_over_budget",
    "matrix_distiller_failed",
}


@dataclass(frozen=True)
class CategorySummary:
    category_id: str
    distributor_code: str = ""
    name: str = ""
    path: list[str] = field(default_factory=list)
    parent_category_id: str | None = None
    product_count: int = 0
    stocked_count: int = 0
    priced_count: int = 0
    sample_product_names: list[str] = field(default_factory=list)
    sample_producers: list[str] = field(default_factory=list)
    sample_part_numbers: list[str] = field(default_factory=list)
    product_group_contexts: list[str] = field(default_factory=list)
    allowed_roles: list[str] = field(default_factory=list)
    suggested_roles: list[str] = field(default_factory=list)
    category_kind: str = "unknown"
    metadata_source: str = "inferred"
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def search_text(self) -> str:
        return " ".join(
            part
            for part in [
                self.category_id,
                self.distributor_code,
                self.name,
                *self.path,
                *self.sample_product_names,
                *self.sample_producers,
                *self.sample_part_numbers,
                *self.product_group_contexts,
                *self.allowed_roles,
                *self.suggested_roles,
                self.category_kind,
                self.metadata_source,
            ]
            if part
        ).casefold()


@dataclass(frozen=True)
class CandidateUniversePlan:
    product_group: str
    primary_object: str
    broad_category_plan: dict[str, list[str]]
    broad_role_hints: list[str]
    included_category_ids: list[str]
    planner_reasoning: str
    confidence: str
    category_plan_entries: list[dict[str, Any]]
    procurement_intent: str = ""
    selected_group_reason: str = ""
    competing_product_groups: list[dict[str, Any]] = field(default_factory=list)
    primary_object_indicators: list[Any] = field(default_factory=list)
    component_role_indicators: list[dict[str, Any]] = field(default_factory=list)
    embedded_requirements: list[dict[str, Any]] = field(default_factory=list)
    requirement_fulfillment_decision: list[dict[str, Any]] = field(default_factory=list)
    accessory_indicators: list[Any] = field(default_factory=list)
    service_support_indicators: list[Any] = field(default_factory=list)
    logistics_commercial_constraints: list[Any] = field(default_factory=list)
    excluded_category_groups: list[dict[str, Any]] = field(default_factory=list)
    needs_repair: bool = False
    candidate_universe_planner_mode: str = CANDIDATE_UNIVERSE_PLANNER_FALLBACK_MODE
    planner_repair_attempted: bool = False
    planner_repair_success: bool = False
    planner_suspicion_reasons: list[str] = field(default_factory=list)
    planner_warnings: list[str] = field(default_factory=list)
    rejected_category_reasons: list[dict[str, Any]] = field(default_factory=list)
    category_catalog_total: int = 0
    category_catalog_sent_to_ai_count: int = 0
    category_catalog_truncated: bool = False
    planner_error_type: str | None = None

    def to_report_json(self) -> dict[str, Any]:
        return {
            "candidate_universe_planner_mode": self.candidate_universe_planner_mode,
            "product_group": self.product_group,
            "primary_product_group": self.product_group,
            "primary_object": self.primary_object,
            "confidence": self.confidence,
            "procurement_intent": self.procurement_intent,
            "selected_group_reason": self.selected_group_reason,
            "competing_product_groups": self.competing_product_groups,
            "primary_object_indicators": self.primary_object_indicators,
            "component_role_indicators": self.component_role_indicators,
            "embedded_requirements": self.embedded_requirements,
            "requirement_fulfillment_decision": self.requirement_fulfillment_decision,
            "accessory_indicators": self.accessory_indicators,
            "service_support_indicators": self.service_support_indicators,
            "logistics_commercial_constraints": self.logistics_commercial_constraints,
            "broad_category_plan": self.broad_category_plan,
            "broad_role_hints": self.broad_role_hints,
            "included_category_ids": self.included_category_ids,
            "category_plan_entries": self.category_plan_entries,
            "excluded_category_groups": self.excluded_category_groups,
            "needs_repair": self.needs_repair,
            "planner_reasoning": self.planner_reasoning,
            "planner_repair_attempted": self.planner_repair_attempted,
            "planner_repair_success": self.planner_repair_success,
            "planner_suspicion_reasons": self.planner_suspicion_reasons,
            "planner_warnings": self.planner_warnings,
            "rejected_category_count": len(self.rejected_category_reasons),
            "rejected_category_reasons": self.rejected_category_reasons,
            "category_catalog_total": self.category_catalog_total,
            "category_catalog_sent_to_ai_count": self.category_catalog_sent_to_ai_count,
            "category_catalog_truncated": self.category_catalog_truncated,
            "planner_error_type": self.planner_error_type,
        }


@dataclass(frozen=True)
class AiMatchPipelineV2Result:
    match_result: MatchResult
    package: dict[str, Any]
    report_fields: dict[str, Any]


async def run_ai_match_pipeline_v2(
    spec: StockSpec,
    session: AsyncSession,
    *,
    distributor_code: str = "ocs",
    preview_only: bool = False,
    llm_settings: LlmSettings | None = None,
    llm_configurator_client: LlmClient | None = None,
    web_evidence_settings: WebEvidenceSettings | None = None,
    web_search_provider: WebSearchProvider | None = None,
    evidence_cache: EvidenceSearchCache | None = None,
) -> AiMatchPipelineV2Result:
    """Run the Composer-first v2 matching pipeline without v1 semantic gates."""

    settings = llm_settings or get_llm_settings()
    llm_call_budget = LlmCallBudget(max_calls=settings.llm_max_calls_per_match)
    budgeted_client = budgeted_llm_client(llm_configurator_client, llm_call_budget)
    request_text = str(spec.source_text or "").strip()
    products = await _load_distributor_products(session, distributor_code=distributor_code)
    categories = await _load_distributor_categories(
        session,
        distributor_code=distributor_code,
    )
    stock_rows_by_key = await _load_latest_stock_rows(
        session,
        products,
        distributor_code=distributor_code,
    )
    category_summaries = _category_summaries(
        categories=categories,
        products=products,
        stock_rows_by_key=stock_rows_by_key,
        distributor_code=distributor_code,
    )
    universe_plan = _plan_candidate_universe(
        spec,
        original_request_text=request_text,
        distributor_code=distributor_code,
        categories=category_summaries,
        llm_settings=settings,
        llm_client=budgeted_client,
        llm_call_budget=llm_call_budget,
    )
    matrix = _build_full_candidate_matrix(
        universe_plan=universe_plan,
        products=products,
        stock_rows_by_key=stock_rows_by_key,
        categories=category_summaries,
        distributor_code=distributor_code,
    )
    normalized_requirements = _v2_normalized_requirements(
        spec,
        universe_plan=universe_plan,
    )
    package = _build_v2_composer_package(
        original_request_text=request_text,
        normalized_requirements=normalized_requirements,
        component_candidate_matrix=matrix,
        settings=settings,
    )
    attempt_decision = _v2_composer_attempt_decision(
        universe_plan=universe_plan,
        matrix=matrix,
        package=package,
        settings=settings,
        llm_client=budgeted_client,
        preview_only=preview_only,
        llm_call_budget=llm_call_budget,
    )

    if preview_only or not attempt_decision.get("should_attempt"):
        llm_outcome = _not_attempted_v2_outcome(
            settings=settings,
            package=package,
            attempt_decision=attempt_decision,
        )
    else:
        llm_outcome = compose_llm_configurations(
            user_request=request_text,
            normalized_requirements=normalized_requirements,
            ready_stock_candidates=[],
            component_candidate_matrix=matrix,
            rule_based_build_candidates=[],
            settings=settings,
            llm_client=budgeted_client,
            web_evidence_settings=web_evidence_settings,
            web_search_provider=web_search_provider,
            evidence_cache=evidence_cache,
            llm_call_budget=llm_call_budget,
        )
        llm_outcome = _apply_v2_validation_overrides(llm_outcome)
        llm_outcome = _with_v2_final_status(llm_outcome)
        attempt_decision = _merge_attempt_decision(
            attempt_decision,
            llm_outcome.composer_attempt_decision,
        )
    llm_outcome = _ensure_v2_safe_no_recommendation(
        llm_outcome,
        package=package,
        matrix=matrix,
        attempt_decision=attempt_decision,
    )
    llm_outcome = _with_v2_execution_contract_status(
        llm_outcome,
        package=package,
        matrix=matrix,
        attempt_decision=attempt_decision,
    )

    package_diagnostics = _v2_package_diagnostics(
        package=package,
        matrix=matrix,
        universe_plan=universe_plan,
        attempt_decision=attempt_decision,
        llm_outcome=llm_outcome,
        llm_call_budget=llm_call_budget,
    )
    matrix = {**matrix, **_matrix_report_fields_from_diagnostics(package_diagnostics)}
    status = _status_from_v2_outcome(llm_outcome, matrix)
    match_result = MatchResult(
        spec=spec,
        status=status,
        engineer_review_required=True,
        total_candidates=int(package_diagnostics.get("composer_package_candidate_total") or 0),
        matched_items=1 if status == STATUS_STOCK_MATCHED else 0,
        missing_requirements=_v2_missing_requirements(llm_outcome),
        risk_flags=_v2_risk_flags(llm_outcome),
        candidates=[],
        component_candidate_matrix=matrix,
        normalized_requirements=[normalized_requirements],
        llm_configurator_enabled=llm_outcome.enabled,
        llm_configurator_used=llm_outcome.used,
        output_mode=llm_outcome.output_mode,
        llm_recommended_build_candidates=llm_outcome.recommended_builds,
        primary_recommendation=llm_outcome.primary_recommendation,
        primary_recommendation_status=llm_outcome.primary_recommendation_status,
        no_recommendation_reason=llm_outcome.no_recommendation_reason,
        commercial_summary=llm_outcome.commercial_summary,
        configuration_groups=llm_outcome.configuration_groups,
        quote_recommendation=llm_outcome.quote_recommendation,
        grouped_presales_mode_used=llm_outcome.grouped_presales_mode_used,
        selected_configuration_group_id=llm_outcome.selected_configuration_group_id,
        selected_platform_option_id=llm_outcome.selected_platform_option_id,
        selected_platform_option_index=llm_outcome.selected_platform_option_index,
        llm_general_notes=llm_outcome.general_notes,
        llm_fallback_reason=llm_outcome.fallback_reason,
        llm_error_type=llm_outcome.error_type,
        llm_http_status=llm_outcome.http_status,
        llm_parse_diagnostics=llm_outcome.parse_diagnostics,
        llm_internal_warnings=llm_outcome.internal_warnings,
        llm_proposals_count=llm_outcome.proposal_count,
        valid_proposals_count=llm_outcome.valid_proposals_count,
        validation_rejected_count=llm_outcome.validation_rejected_count,
        selection_skipped_count=llm_outcome.selection_skipped_count,
        rejected_ai_recommendations_count=llm_outcome.rejected_recommendations_count,
        ai_recommendations_validation_warnings=llm_outcome.validation_warnings,
        ai_validation_summary=llm_outcome.validation_summary,
        rejected_reasons_top=llm_outcome.rejected_reasons_top,
        rejected_ai_recommendations_debug_safe=(
            llm_outcome.rejected_recommendations_debug_safe
        ),
        web_evidence_pack=llm_outcome.evidence_pack,
        llm_evidence_review=llm_outcome.evidence_review,
        llm_repair_used=llm_outcome.repair_used,
        llm_repair_attempted=llm_outcome.repair_attempted,
        llm_repair_success=llm_outcome.repair_success,
        llm_repair_fallback_reason=llm_outcome.repair_fallback_reason,
        llm_repair_critique_count=llm_outcome.repair_critique_count,
        llm_repair_critique_summary=llm_outcome.repair_critique_summary,
        llm_repair_blocked_critique_count=llm_outcome.repair_blocked_critique_count,
        llm_repair_blocked_critique_summary=llm_outcome.repair_blocked_critique_summary,
        llm_repair_savings_estimate=llm_outcome.repair_savings_estimate,
        llm_repair_revised_proposals_count=llm_outcome.repair_revised_proposals_count,
        llm_repair_validation_summary=llm_outcome.repair_validation_summary,
        llm_thinking_diagnostics=llm_outcome.thinking_diagnostics,
        llm_package_diagnostics=package_diagnostics,
        composer_attempt_decision=attempt_decision,
        composer_requirement_analysis=llm_outcome.composer_requirement_analysis,
        composer_fulfillment_decisions=llm_outcome.composer_fulfillment_decisions,
        composer_source_coverage_summary=llm_outcome.composer_source_coverage_summary,
        composer_assumptions=llm_outcome.composer_assumptions,
        composer_engineer_checks=llm_outcome.composer_engineer_checks,
        composer_hard_mismatch_risks=llm_outcome.composer_hard_mismatch_risks,
        composer_unverified_requirements=llm_outcome.composer_unverified_requirements,
        composer_considered_candidate_count_by_role=(
            llm_outcome.composer_considered_candidate_count_by_role
        ),
        composer_chosen_candidate_ids=llm_outcome.composer_chosen_candidate_ids,
        validation_hard_mismatches=llm_outcome.validation_hard_mismatches,
        validation_unverified_requirements=llm_outcome.validation_unverified_requirements,
        final_status_source=llm_outcome.final_status_source,
        product_group=universe_plan.product_group,
        role_plan=_v2_role_plan(universe_plan),
        category_plan=universe_plan.broad_category_plan,
        category_plan_entries=universe_plan.category_plan_entries,
        category_catalog_summary=_category_catalog_summary(category_summaries),
        category_planner_source=CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        category_plan_source=CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        required_capabilities=[],
        optional_capabilities=[],
        unsupported_or_unmapped_requirements=[],
        required_roles=[],
        missing_required_roles=[],
        missing_category_roles=[],
        missing_required_capabilities=[],
        role_coverage_summary=_v2_role_coverage_summary(matrix),
        category_plan_warnings=universe_plan.planner_warnings,
    )
    report_fields = _v2_report_fields(
        package=package,
        matrix=matrix,
        universe_plan=universe_plan,
        package_diagnostics=package_diagnostics,
        attempt_decision=attempt_decision,
    )
    package = {**package, **report_fields}
    return AiMatchPipelineV2Result(
        match_result=match_result,
        package=package,
        report_fields=report_fields,
    )


def _plan_candidate_universe(
    spec: StockSpec,
    *,
    original_request_text: str,
    distributor_code: str,
    categories: Sequence[CategorySummary],
    llm_settings: LlmSettings,
    llm_client: LlmClient | None,
    llm_call_budget: LlmCallBudget | None = None,
) -> CandidateUniversePlan:
    del spec
    planner_client, owns_client, client_error = _candidate_universe_planner_client(
        llm_settings=llm_settings,
        llm_client=llm_client,
    )
    catalog_ids = _category_ids_in_planner_catalog(categories)
    if planner_client is None:
        return _diagnostic_candidate_universe_plan(
            distributor_code=distributor_code,
            reason=client_error or "llm_planner_unavailable",
            category_catalog_total=len(categories),
            category_catalog_sent_to_ai_count=len(
                _candidate_universe_prompt_categories(categories)
            ),
        )
    planner_client = budgeted_llm_client(planner_client, llm_call_budget)

    response: Mapping[str, Any] | None = None
    planner_error_type: str | None = None
    try:
        response = planner_client.generate_json(
            _candidate_universe_planner_system_prompt(),
            _candidate_universe_planner_user_prompt(
                original_request_text=original_request_text,
                distributor_code=distributor_code,
                categories=categories,
            ),
        )
    except (LlmError, ValueError, TypeError) as exc:
        planner_error_type = type(exc).__name__
    finally:
        if owns_client:
            close = getattr(planner_client, "close", None)
            if callable(close):
                close()

    if not isinstance(response, Mapping):
        return _diagnostic_candidate_universe_plan(
            distributor_code=distributor_code,
            reason="llm_planner_failed",
            planner_error_type=planner_error_type,
            category_catalog_total=len(categories),
            category_catalog_sent_to_ai_count=len(
                _candidate_universe_prompt_categories(categories)
            ),
        )

    plan = _coerce_candidate_universe_plan(
        response,
        distributor_code=distributor_code,
        categories=categories,
        mode=CANDIDATE_UNIVERSE_PLANNER_MODE,
    )
    initial_repair_reasons = _candidate_universe_repair_reasons(plan, catalog_ids)
    if not initial_repair_reasons:
        return plan

    repair_response: Mapping[str, Any] | None = None
    repair_error_type: str | None = None
    repair_client, owns_repair_client, repair_client_error = (
        _candidate_universe_planner_client(
            llm_settings=llm_settings,
            llm_client=llm_client,
        )
    )
    if repair_client is not None:
        repair_client = budgeted_llm_client(repair_client, llm_call_budget)
        try:
            repair_response = repair_client.generate_json(
                _candidate_universe_planner_repair_system_prompt(),
                _candidate_universe_planner_repair_user_prompt(
                    original_request_text=original_request_text,
                    distributor_code=distributor_code,
                    categories=categories,
                    previous_output=response,
                    repair_reasons=initial_repair_reasons,
                ),
            )
        except (LlmError, ValueError, TypeError) as exc:
            repair_error_type = type(exc).__name__
        finally:
            if owns_repair_client:
                close = getattr(repair_client, "close", None)
                if callable(close):
                    close()

    if isinstance(repair_response, Mapping):
        repaired = _coerce_candidate_universe_plan(
            repair_response,
            distributor_code=distributor_code,
            categories=categories,
            mode=CANDIDATE_UNIVERSE_PLANNER_REPAIRED_MODE,
            repair_attempted=True,
        )
        repaired_reasons = _candidate_universe_repair_reasons(repaired, catalog_ids)
        repaired = replace(
            repaired,
            planner_repair_attempted=True,
            planner_repair_success=not _blocking_candidate_universe_reasons(
                repaired_reasons
            ),
            planner_suspicion_reasons=_unique(
                [
                    *initial_repair_reasons,
                    *repaired.planner_suspicion_reasons,
                    *repaired_reasons,
                ]
            ),
            planner_warnings=_unique([*plan.planner_warnings, *repaired.planner_warnings]),
            rejected_category_reasons=_unique_mapping_rows(
                [
                    *plan.rejected_category_reasons,
                    *repaired.rejected_category_reasons,
                ]
            ),
            needs_repair=bool(repaired_reasons),
            planner_error_type=repair_error_type,
        )
        if not _blocking_candidate_universe_reasons(repaired_reasons):
            return repaired
        return _fail_closed_suspicious_candidate_universe_plan(
            repaired,
            distributor_code=distributor_code,
            reasons=repaired_reasons,
        )

    failed_reasons = [
        *initial_repair_reasons,
        repair_client_error or repair_error_type or "repair_llm_failed",
    ]
    plan = replace(
        plan,
        planner_repair_attempted=True,
        planner_repair_success=False,
        planner_suspicion_reasons=_unique(
            [
                *plan.planner_suspicion_reasons,
                *failed_reasons,
            ]
        ),
        needs_repair=True,
        planner_error_type=repair_error_type,
    )
    if _blocking_candidate_universe_reasons(initial_repair_reasons):
        return _fail_closed_suspicious_candidate_universe_plan(
            plan,
            distributor_code=distributor_code,
            reasons=failed_reasons,
        )
    return plan


def _candidate_universe_planner_client(
    *,
    llm_settings: LlmSettings,
    llm_client: LlmClient | None,
) -> tuple[LlmClient | None, bool, str | None]:
    if llm_client is not None:
        return llm_client, False, None
    provider = llm_settings.llm_provider.strip().lower()
    if provider == "disabled":
        return None, False, "llm_provider_disabled"
    if provider not in {"openai", "openai-compatible", "openai_compatible"}:
        return None, False, "llm_provider_unsupported"
    if not (
        llm_settings.llm_base_url.strip()
        and llm_settings.llm_api_key.strip()
        and llm_settings.llm_model.strip()
    ):
        return None, False, "llm_settings_incomplete"
    try:
        return (
            OpenAICompatibleLlmClient(
                settings=llm_settings,
                timeout_seconds=min(
                    llm_settings.llm_timeout_seconds,
                    llm_settings.llm_semantic_planner_stage_timeout_seconds,
                ),
                max_output_tokens=min(
                    llm_settings.llm_configurator_max_output_tokens,
                    8192,
                ),
                use_response_format=True,
                thinking_enabled=False,
                max_retries=0,
            ),
            True,
            None,
        )
    except LlmError as exc:
        return None, False, f"llm_client_build_failed:{type(exc).__name__}"


def _candidate_universe_planner_system_prompt() -> str:
    return """
You are the v2 Candidate Universe Planner for a Composer-first procurement pipeline.

Your job is to read the whole procurement request and choose the broad distributor
category universe that the Composer should inspect. Do not classify the primary
product group from isolated keywords. Terms like Ethernet, SFP+, QSFP, X710, RAID,
CPU, RAM, PSU, support, and cables can be component roles inside a larger BOM.

Planning order:
1. Decide primary_product_group, primary_object, procurement_intent, confidence,
   competing_product_groups, and why the selected primary group wins.
2. Separate source terms into primary_object_indicators, component_role_indicators,
   accessory_indicators, service_support_indicators, and
   logistics_commercial_constraints.
   Preserve embedded requirements such as capacity, drive media/interface, protocol,
   support, and brand preferences with hardness/optional markers. Phrases like
   "prefer", "preferred", "nice to have", "желательно", "лучше", and
   "по возможности" are optional/desirable preferences, not hard blockers.
3. Select a broad category universe from the supplied real distributor catalog.
   Use category_id values only from category_catalog. The universe must be broad
   enough for likely BOM roles but bounded to the selected primary product group
   and its component/accessory/support roles. Do not overselect unrelated global
   switch/router/access_point/server/storage categories. Treat category_id as a
   distributor fact, not as business policy. Use product_group metadata,
   allowed_roles, category_kind, stock/priced counts, and samples only as catalog
   evidence.
4. For storage/NAS requests, do not stop at storage_system when the request
   includes capacity, media/interface (SSD/HDD/NVMe/SAS/SATA), protocol, shelves,
   ports, or explicit completeness. Add the corresponding generic roles such as
   drive/ssd/hdd/host_port/protocol_module/support when a separate purchasable
   item may be needed; otherwise record fulfillment as included_in_primary_object
   or unverified_requires_confirmation.
5. Do not select products, component_candidate_id values, prices, or stock rows at
   this planning stage. Return only category planning and reasons.

Examples:
A. Request includes "1U, sockets, CPU, RAM, SSD, controller, NIC, PSU, cooling".
   Even if it mentions "10GbE SFP+ X710", the primary_product_group is server,
   primary_object is server. Network terms are component role network_adapter.
B. Request includes "48 ports 1G PoE, uplink SFP+, L3, stacking". The primary
   product_group is network and primary_object is switch. SFP+ uplinks are switch
   features or optional transceiver/DAC roles, not server roles.
C. Request includes "storage array, usable capacity, RAID, controllers, FC ports".
   The primary_product_group is storage and primary_object is storage_system.

Return only one JSON object with this exact contract:
{
  "primary_product_group": "server|network|storage|unknown",
  "primary_object": "...",
  "confidence": "high|medium|low",
  "procurement_intent": "...",
  "selected_group_reason": "...",
  "competing_product_groups": [
    {"product_group": "...", "reason": "...", "why_not_primary": "..."}
  ],
  "primary_object_indicators": ["..."],
  "component_role_indicators": [
    {
      "role": "...",
      "source_text": "...",
      "reason": "...",
      "hardness": "hard|optional",
      "fulfillment_mode": "separate_component_required|included_in_primary_object|..."
    }
  ],
  "embedded_requirements": [
    {
      "source_text": "...",
      "target_role": "...",
      "requirement_classification": "primary_object_feature|purchasable_component_role|...",
      "hardness": "hard|optional",
      "fulfillment_mode": "separate_component_required|included_in_primary_object|..."
    }
  ],
  "requirement_fulfillment_decision": [
    {
      "source_text": "...",
      "target_role": "...",
      "hardness": "hard|optional",
      "fulfillment_mode": "...",
      "reason": "..."
    }
  ],
  "accessory_indicators": ["..."],
  "service_support_indicators": ["..."],
  "logistics_commercial_constraints": ["..."],
  "broad_role_hints": ["..."],
  "category_plan_entries": [
    {
      "role": "...",
      "selected_category_ids": ["..."],
      "purpose": "candidate_universe",
      "reason": "...",
      "confidence": "high|medium|low"
    }
  ],
  "excluded_category_groups": [
    {"category_id_or_group": "...", "reason": "..."}
  ],
  "needs_repair": false
}
""".strip()


def _candidate_universe_planner_repair_system_prompt() -> str:
    return (
        _candidate_universe_planner_system_prompt()
        + "\n\nThis is a repair pass. The previous JSON failed validation or looked "
        "internally inconsistent. Reconsider the whole procurement object from the "
        "original request. Distinguish primary object indicators from component "
        "roles. Use only category_id values from the supplied category_catalog. "
        "Return the same JSON contract."
    )


def _candidate_universe_planner_user_prompt(
    *,
    original_request_text: str,
    distributor_code: str,
    categories: Sequence[CategorySummary],
) -> str:
    return json.dumps(
        {
            "original_request_text": original_request_text,
            "distributor_code": distributor_code,
            "product_group_policy": _candidate_universe_product_group_policy(),
            "category_catalog": _candidate_universe_category_catalog(categories),
            "category_catalog_summary": _category_catalog_summary(categories),
            "catalog_rule": (
                "Every selected_category_ids value must be one of the category_id "
                "values in category_catalog."
            ),
        },
        ensure_ascii=False,
    )


def _candidate_universe_planner_repair_user_prompt(
    *,
    original_request_text: str,
    distributor_code: str,
    categories: Sequence[CategorySummary],
    previous_output: Mapping[str, Any],
    repair_reasons: Sequence[str],
) -> str:
    return json.dumps(
        {
            "original_request_text": original_request_text,
            "distributor_code": distributor_code,
            "repair_reasons": list(repair_reasons),
            "previous_planner_output": _jsonable(previous_output),
            "product_group_policy": _candidate_universe_product_group_policy(),
            "category_catalog": _candidate_universe_category_catalog(categories),
            "category_catalog_summary": _category_catalog_summary(categories),
            "repair_instruction": (
                "If the previous primary_product_group was chosen from component "
                "keywords, select the true primary procurement object from the "
                "whole request. Keep network/storage/server terms as component "
                "roles when they are embedded in another primary BOM."
            ),
        },
        ensure_ascii=False,
    )


def _candidate_universe_product_group_policy() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in (SERVER_PRODUCT_GROUP, NETWORK_PRODUCT_GROUP, STORAGE_PRODUCT_GROUP):
        profile = get_product_group_profile(group)
        if profile is None:
            continue
        rows.append(
            {
                "product_group": group,
                "primary_object_default": _primary_object_for_group(group),
                "roles": list(profile.roles),
                "required_roles": list(profile.required_roles),
                "optional_roles": list(profile.optional_roles),
                "role_guidance": [
                    {
                        "role": role_id,
                        "behavior": entry.behavior,
                        "quantity_rule": entry.quantity_rule,
                        "synonyms": list(entry.synonyms),
                    }
                    for role_id, entry in profile.role_catalog.items()
                ],
            }
        )
    return rows


def _candidate_universe_category_catalog(
    categories: Sequence[CategorySummary],
) -> list[dict[str, Any]]:
    prompt_categories = _candidate_universe_prompt_categories(categories)
    return [
        {
            "distributor": category.distributor_code,
            "distributor_code": category.distributor_code,
            "category_id": category.category_id,
            "category_name": category.name,
            "category_path": category.path,
            "parent_category_id": category.parent_category_id,
            "product_group_metadata": {
                "contexts": category.product_group_contexts,
                "source": category.metadata_source,
            },
            "allowed_roles": category.allowed_roles,
            "suggested_roles": category.suggested_roles,
            "category_kind": category.category_kind,
            "product_count": category.product_count,
            "stocked_count": category.stocked_count,
            "priced_count": category.priced_count,
            "sample_product_names": category.sample_product_names[:3],
            "sample_producers": category.sample_producers[:3],
            "sample_part_numbers": category.sample_part_numbers[:3],
            "notes": category.notes[:3],
            "warnings": category.warnings[:3],
        }
        for category in prompt_categories
    ]


def _candidate_universe_prompt_categories(
    categories: Sequence[CategorySummary],
) -> list[CategorySummary]:
    return list(categories)


def _category_ids_in_planner_catalog(
    categories: Sequence[CategorySummary],
) -> set[str]:
    return {
        category.category_id
        for category in _candidate_universe_prompt_categories(categories)
        if category.category_id
    }


def _coerce_candidate_universe_plan(
    payload: Mapping[str, Any],
    *,
    distributor_code: str,
    categories: Sequence[CategorySummary],
    mode: str,
    repair_attempted: bool = False,
) -> CandidateUniversePlan:
    warnings: list[str] = []
    categories_by_id = {
        category.category_id: category
        for category in _candidate_universe_prompt_categories(categories)
        if category.category_id
    }
    product_group = _normalize_product_group(
        payload.get("primary_product_group") or payload.get("product_group")
    )
    if product_group == "unknown":
        warnings.append("primary_product_group_unknown")
    confidence = str(payload.get("confidence") or "low").strip().casefold()
    if confidence not in _ALLOWED_CONFIDENCE:
        confidence = "low"
        warnings.append("invalid_confidence_normalized_to_low")
    entries, rejected_category_reasons = _coerce_candidate_category_plan_entries(
        payload,
        categories_by_id=categories_by_id,
        distributor_code=distributor_code,
        product_group=product_group,
    )
    invalid_category_ids = [
        str(row.get("category_id") or "")
        for row in rejected_category_reasons
        if row.get("reason") == "not_in_supplied_catalog"
    ]
    if invalid_category_ids:
        warnings.append(
            "invalid_category_ids:" + ",".join(sorted(set(invalid_category_ids)))
        )
    warnings.extend(
        _category_plan_rejection_warning(row) for row in rejected_category_reasons
    )
    broad_category_plan: dict[str, list[str]] = {}
    for entry in entries:
        role = str(entry.get("role") or "").strip()
        category_ids = _string_list(entry.get("selected_category_ids"))
        if not role or not category_ids:
            continue
        broad_category_plan[role] = _unique(
            [*broad_category_plan.get(role, []), *category_ids]
        )
    broad_role_hints = _unique(
        [
            *[
                role
                for role in (
                    _normalize_plan_role(role) for role in _string_list(
                        payload.get("broad_role_hints")
                    )
                )
                if role
            ],
            *broad_category_plan,
        ]
    )
    included_ids = _unique(
        [
            category_id
            for category_ids in broad_category_plan.values()
            for category_id in category_ids
        ]
    )
    primary_object = str(
        payload.get("primary_object") or _primary_object_for_group(product_group)
    ).strip()
    plan = CandidateUniversePlan(
        product_group=product_group,
        primary_object=primary_object or _primary_object_for_group(product_group),
        broad_category_plan=broad_category_plan,
        broad_role_hints=broad_role_hints,
        included_category_ids=included_ids,
        planner_reasoning=str(payload.get("selected_group_reason") or "").strip()
        or (
            f"LLM selected {product_group or 'unknown'} as the primary product group "
            f"for distributor {distributor_code}."
        ),
        confidence=confidence,
        category_plan_entries=entries,
        procurement_intent=str(payload.get("procurement_intent") or "").strip(),
        selected_group_reason=str(payload.get("selected_group_reason") or "").strip(),
        competing_product_groups=_mapping_rows(
            payload.get("competing_product_groups")
        ),
        primary_object_indicators=_list_or_string_values(
            payload.get("primary_object_indicators")
        ),
        component_role_indicators=_coerce_component_role_indicators(
            payload.get("component_role_indicators")
        ),
        embedded_requirements=_mapping_rows(payload.get("embedded_requirements")),
        requirement_fulfillment_decision=_mapping_rows(
            payload.get("requirement_fulfillment_decision")
        ),
        accessory_indicators=_list_or_string_values(payload.get("accessory_indicators")),
        service_support_indicators=_list_or_string_values(
            payload.get("service_support_indicators")
        ),
        logistics_commercial_constraints=_list_or_string_values(
            payload.get("logistics_commercial_constraints")
        ),
        excluded_category_groups=_mapping_rows(
            payload.get("excluded_category_groups")
        ),
        needs_repair=bool(payload.get("needs_repair")) or bool(warnings),
        candidate_universe_planner_mode=mode,
        planner_repair_attempted=repair_attempted,
        planner_repair_success=False,
        planner_warnings=warnings,
        rejected_category_reasons=rejected_category_reasons,
        category_catalog_total=len(categories),
        category_catalog_sent_to_ai_count=len(categories_by_id),
        category_catalog_truncated=False,
    )
    return replace(
        plan,
        planner_suspicion_reasons=_candidate_universe_suspicion_reasons(plan),
    )


def _coerce_candidate_category_plan_entries(
    payload: Mapping[str, Any],
    *,
    categories_by_id: Mapping[str, CategorySummary],
    distributor_code: str,
    product_group: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_entries: list[Mapping[str, Any]] = []
    if isinstance(payload.get("category_plan_entries"), list):
        raw_entries.extend(
            row
            for row in payload["category_plan_entries"]
            if isinstance(row, Mapping)
        )
    for key in ("category_plan", "broad_category_plan"):
        raw_plan = payload.get(key)
        if not isinstance(raw_plan, Mapping):
            continue
        for role, category_ids in raw_plan.items():
            raw_entries.append(
                {
                    "role": role,
                    "selected_category_ids": _string_list(category_ids),
                    "purpose": "candidate_universe",
                    "reason": f"Imported from {key}.",
                    "confidence": payload.get("confidence") or "medium",
                }
            )

    entries: list[dict[str, Any]] = []
    rejected_category_reasons: list[dict[str, Any]] = []
    for raw in raw_entries:
        role = _normalize_plan_role(raw.get("role"))
        category_ids = _string_list(raw.get("selected_category_ids"))
        valid_ids: list[str] = []
        for category_id in category_ids:
            category = categories_by_id.get(category_id)
            if category is None:
                rejected_category_reasons.append(
                    _rejected_category_reason(
                        category_id=category_id,
                        role=role,
                        reason="not_in_supplied_catalog",
                        detail="AI selected a category_id absent from the supplied catalog.",
                    )
                )
                continue
            rejection_reason = _category_plan_category_rejection(
                category=category,
                role=role,
                product_group=product_group,
                distributor_code=distributor_code,
                purpose=str(raw.get("purpose") or "").strip(),
            )
            if rejection_reason:
                rejected_category_reasons.append(
                    _rejected_category_reason(
                        category_id=category_id,
                        role=role,
                        reason=rejection_reason["reason"],
                        detail=rejection_reason["detail"],
                    )
                )
                continue
            valid_ids.append(category_id)
        if not role or not valid_ids:
            continue
        confidence = str(raw.get("confidence") or "medium").strip().casefold()
        if confidence not in _ALLOWED_CONFIDENCE:
            confidence = "medium"
        entries.append(
            {
                "role": role,
                "selected_category_ids": _unique(valid_ids),
                "purpose": "candidate_universe",
                "reason": str(raw.get("reason") or "").strip(),
                "confidence": confidence,
            }
        )
    return entries, _unique_mapping_rows(rejected_category_reasons)


def _category_plan_category_rejection(
    *,
    category: CategorySummary,
    role: str | None,
    product_group: str,
    distributor_code: str,
    purpose: str,
) -> dict[str, str] | None:
    if category.distributor_code and category.distributor_code != distributor_code:
        return {
            "reason": "wrong_distributor",
            "detail": (
                f"category distributor {category.distributor_code!r} does not match "
                f"{distributor_code!r}."
            ),
        }
    if not role:
        return None
    if not _role_compatible_with_product_group(role, product_group):
        return {
            "reason": "role_not_compatible_with_product_group",
            "detail": f"role {role!r} is not in product group {product_group!r}.",
        }
    known_contexts = [
        context
        for context in category.product_group_contexts
        if context
        and context
        not in {"unknown"}
    ]
    compatible_contexts = {
        product_group,
        "shared",
        "support_license",
        "accessory",
    }
    if known_contexts and not set(known_contexts).intersection(compatible_contexts):
        return {
            "reason": "category_context_not_compatible_with_product_group",
            "detail": (
                f"category contexts {known_contexts!r} do not include "
                f"primary product group {product_group!r}."
            ),
        }
    if category.allowed_roles and not _category_roles_allow_role(
        category.allowed_roles,
        role,
    ):
        return {
            "reason": "role_not_allowed_by_category_metadata",
            "detail": f"role {role!r} is not in allowed category roles.",
        }
    kind = str(category.category_kind or "unknown").strip().casefold()
    purpose = str(purpose or "").strip().casefold()
    if (
        role in _BASE_DEVICE_ROLES_V2
        and (not purpose or purpose == "base_device")
        and kind
        in {
            "accessory",
            "support",
            "license",
            "drive",
            "cable",
            "transceiver",
            "module",
        }
    ):
        return {
            "reason": "category_kind_not_base_device",
            "detail": f"category_kind {kind!r} cannot satisfy base role {role!r}.",
        }
    if role == SUPPORT_ROLE and kind not in {"support", "license", "mixed", "unknown"}:
        return {
            "reason": "category_kind_not_support",
            "detail": f"category_kind {kind!r} cannot satisfy support.",
        }
    if role == LICENSE_ROLE and kind not in {"license", "support", "mixed", "unknown"}:
        return {
            "reason": "category_kind_not_license",
            "detail": f"category_kind {kind!r} cannot satisfy license.",
        }
    return None


def _category_roles_allow_role(allowed_roles: Sequence[str], role: str) -> bool:
    normalized = {
        _normalize_plan_role(value) or str(value).strip()
        for value in allowed_roles
    }
    role_aliases = _role_metadata_aliases(role)
    return bool(normalized.intersection(role_aliases))


def _role_compatible_with_product_group(role: str, product_group: str) -> bool:
    if product_group in {"", "unknown"}:
        return True
    profile = get_product_group_profile(product_group)
    profile_roles = set(profile.role_catalog) if profile is not None else set()
    allowed_roles = profile_roles.union(_baseline_roles_for_group(product_group))
    return bool(allowed_roles.intersection(_role_metadata_aliases(role)))


def _role_metadata_aliases(role: str) -> set[str]:
    aliases = {role}
    if role == DRIVE_ROLE:
        aliases.update({SSD_ROLE, HDD_ROLE, "storage"})
    elif role in {SSD_ROLE, HDD_ROLE}:
        aliases.update({DRIVE_ROLE, "storage"})
    elif role == DAC_CABLE_ROLE:
        aliases.add(CABLE_ROLE)
    elif role == CABLE_ROLE:
        aliases.add(DAC_CABLE_ROLE)
    elif role == SERVER_PLATFORM_ROLE:
        aliases.add("platform")
    elif role == STORAGE_ARRAY_CONTROLLER_ROLE:
        aliases.add("controller")
    return aliases


def _rejected_category_reason(
    *,
    category_id: str,
    role: str | None,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "category_id": str(category_id or "").strip(),
        "role": role or "",
        "reason": reason,
        "detail": detail,
    }


def _category_plan_rejection_warning(row: Mapping[str, Any]) -> str:
    category_id = str(row.get("category_id") or "").strip()
    role = str(row.get("role") or "").strip()
    reason = str(row.get("reason") or "unknown").strip()
    return f"category_plan_rejected:{category_id}:{role}:{reason}"


_BASE_DEVICE_ROLES_V2 = {
    READY_SERVER_CANDIDATE_TYPE,
    SERVER_PLATFORM_ROLE,
    SWITCH_ROLE,
    ROUTER_ROLE,
    FIREWALL_ROLE,
    ACCESS_POINT_ROLE,
    STORAGE_SYSTEM_ROLE,
}


def _coerce_component_role_indicators(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if isinstance(item, Mapping):
            row = dict(item)
            role = _normalize_plan_role(row.get("role")) or str(
                row.get("role") or ""
            ).strip()
            if role:
                row["role"] = role
            hardness = _requirement_hardness(row)
            if hardness:
                row["hardness"] = hardness
            rows.append(row)
        elif str(item or "").strip():
            rows.append({"role": "", "source_text": str(item), "reason": ""})
    return rows


def _candidate_universe_repair_reasons(
    plan: CandidateUniversePlan,
    catalog_ids: set[str],
) -> list[str]:
    reasons: list[str] = []
    del catalog_ids
    if plan.product_group in {"", "unknown"}:
        reasons.append("primary_product_group_unknown")
    if plan.confidence == "low":
        reasons.append("primary_product_group_low_confidence")
    if not plan.included_category_ids:
        reasons.append("no_category_universe")
    reasons.extend(
        warning
        for warning in plan.planner_warnings
        if warning.startswith("invalid_category_ids")
    )
    reasons.extend(_candidate_universe_missing_role_reasons(plan))
    reasons.extend(plan.planner_suspicion_reasons)
    return _unique(reasons)


def _candidate_universe_missing_role_reasons(
    plan: CandidateUniversePlan,
) -> list[str]:
    """Ask the planner to repair roles it noticed but did not map to categories."""

    planned_roles = {
        _normalize_plan_role(role)
        for role in plan.broad_category_plan
        if _normalize_plan_role(role)
    }
    reasons: list[str] = []
    for row in _v2_requested_role_rows(plan):
        role = row["role"]
        if role in planned_roles:
            continue
        reason = row.get("absence_reason") or _v2_missing_role_reason(plan, role, row)
        if reason in {
            "included_in_primary_object",
            "included_in_selected_component",
            "role_not_purchasable",
            "optional_only",
        }:
            continue
        reasons.append(f"{reason}:{role}")
    return _unique(reasons)


def _blocking_candidate_universe_reasons(reasons: Sequence[str]) -> list[str]:
    return [
        reason
        for reason in reasons
        if reason.startswith("invalid_category_ids")
        or reason.startswith("suspicious_primary_group_mismatch")
        or reason == "primary_product_group_unknown"
    ]


def _candidate_universe_suspicion_reasons(
    plan: CandidateUniversePlan,
) -> list[str]:
    product_group = plan.product_group
    if product_group in {"", "unknown"}:
        return []
    votes = _candidate_universe_group_votes(plan)
    selected_votes = set(votes.get(product_group, []))
    reasons: list[str] = []
    for group, group_votes in votes.items():
        if group == product_group:
            continue
        unique_votes = set(group_votes)
        if len(unique_votes) >= 2 and len(unique_votes) >= len(selected_votes) + 2:
            reasons.append(
                "suspicious_primary_group_mismatch:"
                f"selected={product_group};indicated={group};"
                f"evidence={','.join(sorted(unique_votes)[:8])}"
            )
    return reasons


def _candidate_universe_group_votes(
    plan: CandidateUniversePlan,
) -> dict[str, list[str]]:
    votes: dict[str, list[str]] = defaultdict(list)
    for row in plan.component_role_indicators:
        role = _normalize_plan_role(row.get("role"))
        if not role:
            continue
        owner_groups = _unique_role_owner_groups(role)
        if len(owner_groups) == 1:
            votes[owner_groups[0]].append(role)
    for text in _indicator_texts(plan.primary_object_indicators):
        for group, role_hits in _product_group_role_hits(text).items():
            votes[group].extend(role_hits)
    return votes


def _product_group_role_hits(text: str) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = defaultdict(list)
    lowered = text.casefold()
    for group in (SERVER_PRODUCT_GROUP, NETWORK_PRODUCT_GROUP, STORAGE_PRODUCT_GROUP):
        primary_object_term = _primary_object_for_group(group).replace("_", " ")
        if (
            primary_object_term
            and primary_object_term != "network device"
            and re.search(rf"\b{re.escape(primary_object_term)}\b", lowered)
        ):
            hits[group].append(primary_object_term.replace(" ", "_"))
        profile = get_product_group_profile(group)
        if profile is None:
            continue
        for role_id, entry in profile.role_catalog.items():
            role = _normalize_plan_role(role_id) or role_id
            owner_groups = _unique_role_owner_groups(role)
            if len(owner_groups) != 1 or owner_groups[0] != group:
                continue
            terms = [role_id.replace("_", " "), *_string_list(entry.synonyms)]
            if _text_matches_any(lowered, terms):
                hits[group].append(role)
    return hits


def _unique_role_owner_groups(role: str) -> list[str]:
    owners: list[str] = []
    role = _normalize_plan_role(role) or role
    group_roles = {
        SERVER_PRODUCT_GROUP: set(_SERVER_BASELINE_ROLES),
        NETWORK_PRODUCT_GROUP: set(_NETWORK_BASELINE_ROLES),
        STORAGE_PRODUCT_GROUP: set(_STORAGE_BASELINE_ROLES),
    }
    for group, roles in group_roles.items():
        profile = get_product_group_profile(group)
        profile_roles = set(profile.role_catalog) if profile is not None else set()
        normalized_profile_roles = {
            normalized
            for normalized in (_normalize_plan_role(item) for item in profile_roles)
            if normalized
        }
        if role in roles or role in normalized_profile_roles:
            owners.append(group)
    return _unique(owners)


def _normalize_product_group(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("-", "_")
    aliases = {
        "servers": SERVER_PRODUCT_GROUP,
        "server_bom": SERVER_PRODUCT_GROUP,
        "networking": NETWORK_PRODUCT_GROUP,
        "network_device": NETWORK_PRODUCT_GROUP,
        "storage_system": STORAGE_PRODUCT_GROUP,
        "storage_array": STORAGE_PRODUCT_GROUP,
    }
    text = aliases.get(text, text)
    return text if text in _ALLOWED_PRODUCT_GROUPS else "unknown"


def _normalize_plan_role(value: Any) -> str | None:
    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "server": SERVER_PLATFORM_ROLE,
        "server_platforms": SERVER_PLATFORM_ROLE,
        "storage": DRIVE_ROLE,
        "storage_array": STORAGE_SYSTEM_ROLE,
        "storage_drive": DRIVE_ROLE,
        "drives": DRIVE_ROLE,
        "disk": DRIVE_ROLE,
        "disks": DRIVE_ROLE,
        "controller": STORAGE_ARRAY_CONTROLLER_ROLE,
        "storage_array_controller": STORAGE_ARRAY_CONTROLLER_ROLE,
        "shelf": DISK_SHELF_ROLE,
        "drive_shelf": DISK_SHELF_ROLE,
        "host_ports": HOST_PORT_ROLE,
        "protocol": PROTOCOL_MODULE_ROLE,
        "network_interface": NETWORK_ADAPTER_ROLE,
        "nic": NETWORK_ADAPTER_ROLE,
        "psu": POWER_SUPPLY_ROLE,
        "platform": SERVER_PLATFORM_ROLE,
        "license_support": SUPPORT_ROLE,
        "service": SUPPORT_ROLE,
        "services": SUPPORT_ROLE,
    }
    return _normalize_role(aliases.get(text, text))


def _indicator_texts(value: Sequence[Any]) -> list[str]:
    texts: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            texts.extend(
                str(part)
                for part in (
                    item.get("source_text"),
                    item.get("text"),
                    item.get("indicator"),
                    item.get("reason"),
                )
                if str(part or "").strip()
            )
        elif str(item or "").strip():
            texts.append(str(item))
    return texts


def _list_or_string_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [value]


def _v2_requested_role_rows(plan: CandidateUniversePlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, values in (
        ("broad_role_hints", plan.broad_role_hints),
        ("component_role_indicators", plan.component_role_indicators),
        ("embedded_requirements", plan.embedded_requirements),
        ("requirement_fulfillment_decision", plan.requirement_fulfillment_decision),
        ("accessory_indicators", plan.accessory_indicators),
        ("service_support_indicators", plan.service_support_indicators),
    ):
        for value in values:
            row = dict(value) if isinstance(value, Mapping) else {"source_text": value}
            role = _normalize_plan_role(
                row.get("role")
                or row.get("target_role")
                or row.get("role_id")
                or row.get("component_role")
                or (value if source == "broad_role_hints" else None)
            )
            if not role:
                raw_role = str(
                    row.get("role")
                    or row.get("target_role")
                    or row.get("role_id")
                    or row.get("component_role")
                    or ""
                ).strip()
                role = (
                    raw_role.casefold().replace("-", "_").replace(" ", "_")
                    if raw_role
                    else None
                )
            if not _normalize_plan_role(role):
                role = _v2_role_from_indicator_text(plan, row) or role
            if not role:
                continue
            role = _normalize_plan_role(role) or role
            row["role"] = role
            row["source"] = source
            hardness = _requirement_hardness(row)
            if hardness:
                row["hardness"] = hardness
            rows.append(row)
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("role") or ""),
            str(row.get("source_text") or row.get("requirement_text") or ""),
            str(row.get("source") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def _v2_role_from_indicator_text(
    plan: CandidateUniversePlan,
    row: Mapping[str, Any],
) -> str | None:
    text = _v2_row_text(row).casefold()
    if not text:
        return None
    profile = get_product_group_profile(plan.product_group)
    candidate_roles = list(profile.role_catalog) if profile is not None else list(_ROLE_KEYWORDS)
    for role in candidate_roles:
        normalized = _normalize_plan_role(role)
        if not normalized:
            continue
        terms = _ROLE_KEYWORDS.get(normalized, (normalized.replace("_", " "),))
        if _text_matches_any(text, terms):
            return normalized
    return None


def _requirement_hardness(row: Mapping[str, Any]) -> str | None:
    explicit = str(
        row.get("hardness")
        or row.get("requirement_hardness")
        or row.get("priority")
        or ""
    ).strip().casefold()
    if explicit in {"hard", "required", "mandatory", "must"}:
        return "hard"
    if explicit in {"optional", "desirable", "preferred", "nice_to_have"}:
        return "optional"
    if row.get("required") is False or row.get("optional") is True:
        return "optional"
    text = " ".join(
        str(part or "")
        for part in (
            row.get("source_text"),
            row.get("requirement_text"),
            row.get("text"),
            row.get("reason"),
        )
    ).casefold()
    optional_markers = (
        "желательно",
        "по возможности",
        "лучше",
        "предпочтительно",
        "необязательно",
        "optional",
        "prefer",
        "preferred",
        "desirable",
        "nice to have",
        "if possible",
    )
    if any(marker in text for marker in optional_markers):
        return "optional"
    return None


def _v2_row_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(part or "")
        for part in (
            row.get("source_text"),
            row.get("requirement_text"),
            row.get("text"),
            row.get("reason"),
            row.get("indicator"),
        )
        if str(part or "").strip()
    ).strip()


def _v2_row_classification(row: Mapping[str, Any]) -> str:
    return _normalize_requirement_classification(
        row.get("requirement_classification") or row.get("classification") or ""
    )


def _normalize_requirement_classification(value: Any) -> str:
    text = str(value or "").strip()
    normalized = text.casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "commercial": _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "commercial_constraint": _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "budget": _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "budget_constraint": _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "price_constraint": _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "logistics_constraint": _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
        "logistic_constraint": _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
    }
    return aliases.get(normalized, text)


def _v2_row_fulfillment_mode(row: Mapping[str, Any]) -> str:
    return str(
        row.get("fulfillment_mode")
        or row.get("fulfillment")
        or row.get("closed_by")
        or ""
    ).strip()


def _v2_role_is_profile_purchasable(plan: CandidateUniversePlan, role: str) -> bool:
    profile = get_product_group_profile(plan.product_group)
    if role not in _ROLE_MATRIX_KEY:
        return False
    if profile is None:
        return True
    profile_roles = set(profile.role_catalog)
    if role in profile_roles:
        return True
    return role in _STORAGE_DRIVE_LIKE_ROLES and (
        "storage" in profile_roles or DRIVE_ROLE in profile_roles
    )


def _v2_primary_required_role(plan: CandidateUniversePlan) -> str | None:
    product_group = plan.product_group
    primary_object = _normalize_plan_role(plan.primary_object)
    if product_group == NETWORK_PRODUCT_GROUP:
        if primary_object in _NETWORK_PRIMARY_DEVICE_ROLES:
            return primary_object
        for role in [*plan.broad_role_hints, *plan.broad_category_plan]:
            normalized = _normalize_plan_role(role)
            if normalized in _NETWORK_PRIMARY_DEVICE_ROLES:
                return normalized
        return SWITCH_ROLE
    if product_group == STORAGE_PRODUCT_GROUP:
        return STORAGE_SYSTEM_ROLE
    if product_group == SERVER_PRODUCT_GROUP:
        return SERVER_PLATFORM_ROLE
    return None


def _v2_explicit_network_accessory_request(row: Mapping[str, Any], role: str) -> bool:
    text = " ".join(
        str(part or "")
        for part in (
            row.get("source_text"),
            row.get("requirement_text"),
            row.get("text"),
        )
        if str(part or "").strip()
    ).casefold()
    if not text:
        return False
    explicit_terms_by_role = {
        TRANSCEIVER_ROLE: (
            "transceiver",
            "optic",
            "optics",
            "optical module",
            "sfp module",
            "sfp+ module",
            "qsfp module",
        ),
        DAC_CABLE_ROLE: ("dac", "direct attach"),
        CABLE_ROLE: ("cable", "cord", "patch cord", "patch cable"),
        STACKING_MODULE_ROLE: (
            "stacking module",
            "stack module",
            "stacking cable",
            "stack cable",
        ),
        LICENSE_ROLE: ("license", "subscription"),
        SUPPORT_ROLE: ("support", "service", "warranty"),
        POWER_SUPPLY_ROLE: ("power supply", "psu", "spare psu", "redundant psu"),
        OTHER_ACCESSORY_ROLE: ("accessory", "kit", "module"),
    }
    return any(term in text for term in explicit_terms_by_role.get(role, ()))


def _v2_explicit_storage_drive_request(row: Mapping[str, Any]) -> bool:
    text = _v2_row_text(row).casefold()
    if not text:
        return False
    if re.search(r"\b\d+\s*x\s*(?:ssd|hdd|drive|drives|disk|disks)\b", text):
        return True
    if re.search(r"\b\d+\s+(?:ssd|hdd|drive|drives|disk|disks)\b", text):
        return True
    explicit_terms = (
        "separate drive",
        "separate disk",
        "drive module",
        "disk module",
        "replacement drive",
        "spare drive",
        "add drives",
        "with drives as separate",
    )
    return any(term in text for term in explicit_terms)


def _v2_storage_drive_is_primary_object_feature(
    plan: CandidateUniversePlan,
    row: Mapping[str, Any],
    role: str,
) -> bool:
    if plan.product_group != STORAGE_PRODUCT_GROUP or role not in _STORAGE_DRIVE_LIKE_ROLES:
        return False
    classification = _v2_row_classification(row)
    fulfillment_mode = _v2_row_fulfillment_mode(row)
    if (
        classification == _REQ_CLASS_PURCHASABLE_COMPONENT_ROLE
        or fulfillment_mode == _FULFILLMENT_SEPARATE_COMPONENT_REQUIRED
    ):
        return False
    if classification == _REQ_CLASS_PRIMARY_OBJECT_FEATURE:
        return True
    if fulfillment_mode == _FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT:
        return True
    combined_text = " ".join(
        [
            _v2_row_text(row),
            " ".join(str(item) for item in plan.primary_object_indicators),
            plan.procurement_intent,
            plan.primary_object,
        ]
    ).casefold()
    if "nas" not in combined_text:
        return False
    return not _v2_explicit_storage_drive_request(row)


def _v2_role_is_hard_purchasable_bom(
    plan: CandidateUniversePlan,
    row: Mapping[str, Any],
) -> bool:
    role = str(row.get("role") or "").strip()
    if not role or not _v2_role_is_profile_purchasable(plan, role):
        return False
    if _requirement_hardness(row) == "optional":
        return False
    classification = _v2_row_classification(row)
    fulfillment_mode = _v2_row_fulfillment_mode(row)
    if classification in _NON_BLOCKING_REQUIREMENT_CLASSES:
        return False
    if fulfillment_mode in _NON_BLOCKING_FULFILLMENT_MODES:
        return False
    if _v2_storage_drive_is_primary_object_feature(plan, row, role):
        return False

    primary_role = _v2_primary_required_role(plan)
    if role == primary_role:
        return True
    if plan.product_group == NETWORK_PRODUCT_GROUP:
        if role in _NETWORK_PRIMARY_DEVICE_ROLES:
            return role == primary_role
        if role in _NETWORK_ACCESSORY_ENGINEERING_ROLES:
            return (
                classification
                in {
                    _REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
                    _REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
                    _REQ_CLASS_SERVICE_OR_SUPPORT,
                }
                or fulfillment_mode
                in {
                    _FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
                    _FULFILLMENT_SERVICE_OR_SUPPORT,
                }
            ) and _v2_explicit_network_accessory_request(row, role)
    if str(row.get("source") or "") == "broad_role_hints":
        return False
    if (
        plan.product_group == SERVER_PRODUCT_GROUP
        and str(row.get("source") or "")
        in {"accessory_indicators", "service_support_indicators"}
    ):
        return role in {
            CABLE_ROLE,
            TRANSCEIVER_ROLE,
            RAIL_KIT_ROLE,
            LICENSE_ROLE,
            SUPPORT_ROLE,
            POWER_SUPPLY_ROLE,
        }
    if classification in {
        _REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        _REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
        _REQ_CLASS_SERVICE_OR_SUPPORT,
        _REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE,
    }:
        return True
    if fulfillment_mode in {
        _FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
        _FULFILLMENT_SERVICE_OR_SUPPORT,
    }:
        return True
    return bool(row.get("source") == "component_role_indicators")


def _v2_requirement_bucket(
    plan: CandidateUniversePlan,
    row: Mapping[str, Any],
) -> str:
    classification = _v2_row_classification(row)
    fulfillment_mode = _v2_row_fulfillment_mode(row)
    role = str(row.get("role") or "").strip()
    if _v2_role_is_hard_purchasable_bom(plan, row):
        return "hard_purchasable_bom_role"
    if classification == _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT:
        return "logistics_or_commercial_constraint"
    if classification == _REQ_CLASS_ENGINEERING_CHECK:
        return "engineering_check"
    if (
        classification == _REQ_CLASS_PRIMARY_OBJECT_FEATURE
        or fulfillment_mode == _FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT
        or _v2_storage_drive_is_primary_object_feature(plan, row, role)
    ):
        return "primary_object_feature"
    if (
        _requirement_hardness(row) == "optional"
        or classification
        in {
            _REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
            _REQ_CLASS_SERVICE_OR_SUPPORT,
            _REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
        }
        or fulfillment_mode
        in {
            "optional",
            "optional_preference",
            _FULFILLMENT_SERVICE_OR_SUPPORT,
            _FULFILLMENT_ENGINEERING_CHECK_ONLY,
            _FULFILLMENT_NOT_APPLICABLE,
        }
        or (
            plan.product_group == NETWORK_PRODUCT_GROUP
            and role in _NETWORK_ACCESSORY_ENGINEERING_ROLES
        )
        or str(row.get("source") or "") == "broad_role_hints"
    ):
        return "optional_accessory_or_engineering"
    if not _v2_role_is_profile_purchasable(plan, role):
        return "non_purchasable_non_blocking"
    return "primary_object_feature"


def _v2_requirement_row_for_report(row: Mapping[str, Any], *, bucket: str) -> dict[str, Any]:
    role = str(row.get("role") or "").strip()
    default_classification = {
        "hard_purchasable_bom_role": _REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
        "primary_object_feature": _REQ_CLASS_PRIMARY_OBJECT_FEATURE,
        "optional_accessory_or_engineering": _REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
        "engineering_check": _REQ_CLASS_ENGINEERING_CHECK,
        "logistics_or_commercial_constraint": (
            _REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT
        ),
    }.get(bucket, bucket)
    default_fulfillment = (
        _FULFILLMENT_SEPARATE_COMPONENT_REQUIRED
        if bucket == "hard_purchasable_bom_role"
        else _FULFILLMENT_INCLUDED_IN_PRIMARY_OBJECT
        if bucket == "primary_object_feature"
        else _FULFILLMENT_ENGINEERING_CHECK_ONLY
        if bucket == "engineering_check"
        else _FULFILLMENT_LOGISTICS_CONSTRAINT
        if bucket == "logistics_or_commercial_constraint"
        else _FULFILLMENT_NOT_APPLICABLE
    )
    return {
        "role": role,
        "target_role": role,
        "source_text": _v2_row_text(row),
        "classification": _v2_row_classification(row) or default_classification,
        "fulfillment_mode": _v2_row_fulfillment_mode(row) or default_fulfillment,
        "hardness": _requirement_hardness(row) or "hard",
        "source": row.get("source"),
        "bucket": bucket,
        "reason": row.get("reason"),
    }


def _v2_role_requirement_summary(plan: CandidateUniversePlan) -> dict[str, Any]:
    rows = _v2_requested_role_rows(plan)
    broad_roles = _unique(
        [
            *[
                role
                for role in (_normalize_plan_role(role) for role in plan.broad_role_hints)
                if role
            ],
            *[
                role
                for role in (_normalize_plan_role(role) for role in plan.broad_category_plan)
                if role
            ],
            *[row["role"] for row in rows],
        ]
    )
    hard_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    optional_rows: list[dict[str, Any]] = []
    engineering_rows: list[dict[str, Any]] = []
    logistics_rows: list[dict[str, Any]] = []
    for row in rows:
        bucket = _v2_requirement_bucket(plan, row)
        report_row = _v2_requirement_row_for_report(row, bucket=bucket)
        if bucket == "hard_purchasable_bom_role":
            hard_rows.append(report_row)
        elif bucket == "primary_object_feature":
            feature_rows.append(report_row)
        elif bucket == "logistics_or_commercial_constraint":
            logistics_rows.append(report_row)
        elif bucket == "engineering_check":
            engineering_rows.append(report_row)
        else:
            optional_rows.append(report_row)

    primary_role = _v2_primary_required_role(plan)
    hard_roles = _unique([row["role"] for row in hard_rows if row.get("role")])
    if primary_role and primary_role not in hard_roles:
        hard_roles = _unique([primary_role, *hard_roles])
        hard_rows.insert(
            0,
            {
                "role": primary_role,
                "target_role": primary_role,
                "source_text": str(plan.primary_object or primary_role),
                "classification": _REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
                "fulfillment_mode": _FULFILLMENT_SEPARATE_COMPONENT_REQUIRED,
                "hardness": "hard",
                "source": "primary_object",
                "bucket": "hard_purchasable_bom_role",
                "reason": "Primary purchasable object for the selected product group.",
            },
        )
    return {
        "broad_reasoning_roles": broad_roles,
        "roles_sent_to_composer": broad_roles,
        "hard_purchasable_bom_roles": hard_roles,
        "hard_purchasable_bom_role_requirements": _unique_mapping_rows(hard_rows),
        "purchasable_role_requirements": _unique_mapping_rows(hard_rows),
        "primary_object_feature_requirements": _unique_mapping_rows(feature_rows),
        "optional_accessory_engineering_roles": _unique(
            [row["role"] for row in optional_rows + engineering_rows if row.get("role")]
        ),
        "optional_accessory_engineering_requirements": _unique_mapping_rows(
            [*optional_rows, *engineering_rows]
        ),
        "accessory_or_consumable_requirements": _unique_mapping_rows(optional_rows),
        "engineering_check_requirements": _unique_mapping_rows(engineering_rows),
        "logistics_or_commercial_constraint_requirements": _unique_mapping_rows(
            logistics_rows
        ),
        "classified_requirements": _unique_mapping_rows(
            [*hard_rows, *feature_rows, *optional_rows, *engineering_rows, *logistics_rows]
        ),
    }


def _v2_missing_role_reason(
    plan: CandidateUniversePlan,
    role: str,
    row: Mapping[str, Any] | None = None,
    *,
    has_category: bool = False,
    has_candidates: bool = False,
) -> str:
    row = row or {}
    bucket = _v2_requirement_bucket(plan, {**dict(row), "role": role})
    if bucket == "primary_object_feature":
        return "included_in_primary_object"
    if bucket == "optional_accessory_or_engineering":
        return "optional_only"
    if bucket == "engineering_check":
        return "engineering_check_only"
    if bucket == "logistics_or_commercial_constraint":
        return "logistics_constraint"
    mode = str(
        row.get("fulfillment_mode")
        or row.get("fulfillment")
        or row.get("closed_by")
        or ""
    ).strip()
    if mode in {"included_in_primary_object", "primary_object"}:
        return "included_in_primary_object"
    if mode in {"optional_preference", "optional"}:
        return "optional_only"
    if mode in {
        "included_in_selected_component",
        "included_in_bundle_or_kit",
        "selected_component",
        "bundle_or_kit",
    }:
        return "included_in_selected_component"
    classification = str(
        row.get("requirement_classification") or row.get("classification") or ""
    ).strip()
    if classification in {
        "primary_object_feature",
        "logistics_or_commercial_constraint",
        "engineering_check",
        "out_of_scope_or_unmapped_non_blocking",
    }:
        return "included_in_primary_object"
    if _requirement_hardness(row) == "optional":
        return "optional_only"
    profile = get_product_group_profile(plan.product_group)
    profile_roles = set(profile.role_catalog) if profile is not None else set()
    if role not in _ROLE_MATRIX_KEY or (profile is not None and role not in profile_roles):
        return "role_not_purchasable"
    if has_candidates:
        return "sent_to_composer"
    if has_category:
        return "no_stock_candidates"
    return "no_category_found"


def _diagnostic_candidate_universe_plan(
    *,
    distributor_code: str,
    reason: str,
    planner_error_type: str | None = None,
    category_catalog_total: int = 0,
    category_catalog_sent_to_ai_count: int = 0,
) -> CandidateUniversePlan:
    return CandidateUniversePlan(
        product_group="unknown",
        primary_object="unknown",
        broad_category_plan={},
        broad_role_hints=[],
        included_category_ids=[],
        planner_reasoning=(
            "LLM candidate universe planner was unavailable or failed; v2 did not "
            f"choose a product group by keyword fallback for distributor {distributor_code}."
        ),
        confidence="low",
        category_plan_entries=[],
        selected_group_reason="Planner unavailable; deterministic hints are diagnostic only.",
        needs_repair=True,
        candidate_universe_planner_mode=CANDIDATE_UNIVERSE_PLANNER_FALLBACK_MODE,
        planner_warnings=[reason],
        category_catalog_total=category_catalog_total,
        category_catalog_sent_to_ai_count=category_catalog_sent_to_ai_count,
        category_catalog_truncated=False,
        planner_error_type=planner_error_type,
    )


def _fail_closed_suspicious_candidate_universe_plan(
    plan: CandidateUniversePlan,
    *,
    distributor_code: str,
    reasons: Sequence[str],
) -> CandidateUniversePlan:
    return CandidateUniversePlan(
        product_group="unknown",
        primary_object="unknown",
        broad_category_plan={},
        broad_role_hints=plan.broad_role_hints,
        included_category_ids=[],
        planner_reasoning=(
            "Candidate universe planner output was rejected after repair because "
            "the primary object/category contract remained unsafe."
        ),
        confidence="low",
        category_plan_entries=[],
        procurement_intent=plan.procurement_intent,
        selected_group_reason=plan.selected_group_reason,
        competing_product_groups=plan.competing_product_groups,
        primary_object_indicators=plan.primary_object_indicators,
        component_role_indicators=plan.component_role_indicators,
        embedded_requirements=plan.embedded_requirements,
        requirement_fulfillment_decision=plan.requirement_fulfillment_decision,
        accessory_indicators=plan.accessory_indicators,
        service_support_indicators=plan.service_support_indicators,
        logistics_commercial_constraints=plan.logistics_commercial_constraints,
        excluded_category_groups=plan.excluded_category_groups,
        needs_repair=True,
        candidate_universe_planner_mode=plan.candidate_universe_planner_mode,
        planner_repair_attempted=plan.planner_repair_attempted,
        planner_repair_success=False,
        planner_suspicion_reasons=_unique(
            [*plan.planner_suspicion_reasons, *reasons]
        ),
        planner_warnings=_unique(
            [
                *plan.planner_warnings,
                f"candidate_universe_failed_closed:{distributor_code}",
            ]
        ),
        rejected_category_reasons=plan.rejected_category_reasons,
        category_catalog_total=plan.category_catalog_total,
        category_catalog_sent_to_ai_count=plan.category_catalog_sent_to_ai_count,
        category_catalog_truncated=plan.category_catalog_truncated,
        planner_error_type=plan.planner_error_type,
    )


def _build_full_candidate_matrix(
    *,
    universe_plan: CandidateUniversePlan,
    products: Sequence[DistributorProduct],
    stock_rows_by_key: Mapping[tuple[str, str], list[DistributorStockPrice]],
    categories: Sequence[CategorySummary],
    distributor_code: str,
) -> dict[str, Any]:
    category_by_id = {category.category_id: category for category in categories}
    role_by_category_id = _roles_by_category_id(universe_plan.broad_category_plan)
    requirement_summary = _v2_role_requirement_summary(universe_plan)
    rows_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diagnostics = {
        "distributor_code": distributor_code,
        "input_products": len(products),
        "selected_category_count": len(universe_plan.included_category_ids),
        "rejected_category_count": len(universe_plan.rejected_category_reasons),
        "rejected_category_reasons": universe_plan.rejected_category_reasons,
        "objective_reject_policy": [
            "wrong_distributor",
            "category_not_selected",
            "no_stock",
            "no_price",
            "broken_row",
            "objectively_wrong_role",
        ],
        "excluded_unselected_category": 0,
        "excluded_no_stock": 0,
        "excluded_no_price": 0,
        "excluded_broken_row": 0,
        "wrong_distributor": 0,
        "category_not_selected": 0,
        "no_stock": 0,
        "no_price": 0,
        "broken_row": 0,
        "objectively_wrong_role": 0,
    }

    selected_category_ids = set(universe_plan.included_category_ids)
    for product in products:
        if product.distributor_code != distributor_code:
            diagnostics["wrong_distributor"] += 1
            continue
        category_id = str(product.category_id or "").strip()
        if not category_id:
            diagnostics["excluded_broken_row"] += 1
            diagnostics["broken_row"] += 1
            continue
        if category_id not in selected_category_ids:
            diagnostics["excluded_unselected_category"] += 1
            diagnostics["category_not_selected"] += 1
            continue
        stock_rows = stock_rows_by_key.get((product.distributor_code, product.item_id), [])
        available_quantity = _available_quantity(stock_rows)
        if available_quantity is None or available_quantity <= 0:
            diagnostics["excluded_no_stock"] += 1
            diagnostics["no_stock"] += 1
            continue
        price_value, price_currency = _select_price(stock_rows)
        if price_value is None or not str(price_currency or "").strip():
            diagnostics["excluded_no_price"] += 1
            diagnostics["no_price"] += 1
            continue
        role = _role_for_product(
            product,
            planned_roles=role_by_category_id.get(category_id, []),
            product_group=universe_plan.product_group,
        )
        if role is None:
            diagnostics["excluded_broken_row"] += 1
            diagnostics["objectively_wrong_role"] += 1
            continue
        candidate = _candidate_row_for_product(
            product=product,
            role=role,
            category=category_by_id.get(category_id),
            stock_rows=stock_rows,
            available_quantity=available_quantity,
            price_value=price_value,
            price_currency=str(price_currency),
        )
        rows_by_role[role].append(candidate)

    matrix: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "product_group": universe_plan.product_group,
        "primary_product_group": universe_plan.product_group,
        "primary_object": universe_plan.primary_object,
        "candidate_universe_planner_mode": (
            universe_plan.candidate_universe_planner_mode
        ),
        "semantic_planner_source": CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        "semantic_planner_used": universe_plan.candidate_universe_planner_mode
        != CANDIDATE_UNIVERSE_PLANNER_FALLBACK_MODE,
        "semantic_planner_confidence": universe_plan.confidence,
        "selected_product_group_reason": universe_plan.selected_group_reason,
        "competing_product_groups": universe_plan.competing_product_groups,
        "primary_object_indicators": universe_plan.primary_object_indicators,
        "component_role_indicators": universe_plan.component_role_indicators,
        "embedded_requirements": universe_plan.embedded_requirements,
        "requirement_fulfillment_decision": universe_plan.requirement_fulfillment_decision,
        "excluded_category_groups": universe_plan.excluded_category_groups,
        "planner_repair_attempted": universe_plan.planner_repair_attempted,
        "planner_repair_success": universe_plan.planner_repair_success,
        "planner_suspicion_reasons": universe_plan.planner_suspicion_reasons,
        "category_planner_source": CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        "category_plan_source": CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        "candidate_universe_planner_output": universe_plan.to_report_json(),
        "candidate_universe_category_plan": universe_plan.broad_category_plan,
        "category_plan": universe_plan.broad_category_plan,
        "category_plan_entries": universe_plan.category_plan_entries,
        "selected_category_count": len(universe_plan.included_category_ids),
        "rejected_category_count": len(universe_plan.rejected_category_reasons),
        "rejected_category_reasons": universe_plan.rejected_category_reasons,
        "category_catalog_total": universe_plan.category_catalog_total,
        "category_catalog_sent_to_ai_count": (
            universe_plan.category_catalog_sent_to_ai_count
        ),
        "category_catalog_truncated": universe_plan.category_catalog_truncated,
        "category_catalog_summary": _category_catalog_summary(categories),
        "role_plan": _v2_role_plan(universe_plan),
        "roles_sent_to_composer": requirement_summary["roles_sent_to_composer"],
        "broad_reasoning_roles": requirement_summary["broad_reasoning_roles"],
        "hard_purchasable_bom_roles": requirement_summary[
            "hard_purchasable_bom_roles"
        ],
        "hard_purchasable_bom_role_requirements": requirement_summary[
            "hard_purchasable_bom_role_requirements"
        ],
        "purchasable_role_requirements": requirement_summary[
            "purchasable_role_requirements"
        ],
        "primary_object_feature_requirements": requirement_summary[
            "primary_object_feature_requirements"
        ],
        "optional_accessory_engineering_roles": requirement_summary[
            "optional_accessory_engineering_roles"
        ],
        "optional_accessory_engineering_requirements": requirement_summary[
            "optional_accessory_engineering_requirements"
        ],
        "accessory_or_consumable_requirements": requirement_summary[
            "accessory_or_consumable_requirements"
        ],
        "engineering_check_requirements": requirement_summary[
            "engineering_check_requirements"
        ],
        "logistics_or_commercial_constraint_requirements": requirement_summary[
            "logistics_or_commercial_constraint_requirements"
        ],
        "classified_requirements": requirement_summary["classified_requirements"],
        "required_roles": requirement_summary["hard_purchasable_bom_roles"],
        "required_capabilities": [],
        "optional_capabilities": [],
        "component_matrix_coverage_summary": {},
        "role_coverage_summary": {},
        "matrix_source_diagnostics": diagnostics,
    }
    for role, rows in rows_by_role.items():
        rows.sort(key=_candidate_sort_key)
        matrix_key = _ROLE_MATRIX_KEY.get(role)
        if matrix_key:
            matrix[matrix_key] = rows

    count_by_role = _count_by_role_from_matrix(matrix)
    count_by_category = _count_by_category_from_rows(rows_by_role.values())
    matrix["full_candidate_matrix_count_by_role"] = count_by_role
    matrix["full_candidate_matrix_count_by_category"] = count_by_category
    matrix["matrix_materialized_count_by_role"] = count_by_role
    matrix["broad_matrix_count_by_role"] = count_by_role
    matrix["broad_count_by_role"] = count_by_role
    matrix["count_by_role"] = count_by_role
    matrix["role_coverage_summary"] = _v2_role_coverage_summary(matrix)
    matrix["component_matrix_coverage_summary"] = _v2_component_coverage_summary(
        matrix,
        universe_plan=universe_plan,
    )
    matrix.update(_v2_role_lifecycle_fields(universe_plan=universe_plan, matrix=matrix))
    return matrix


def _v2_role_candidate_count_with_alias(
    role: str,
    matrix_counts: Mapping[str, Any],
) -> tuple[int, str | None]:
    normalized = _normalize_plan_role(role) or str(role or "").strip()
    aliases = [normalized]
    if normalized == DRIVE_ROLE:
        aliases.extend([SSD_ROLE, HDD_ROLE])
    elif normalized in {SSD_ROLE, HDD_ROLE}:
        aliases.append(DRIVE_ROLE)
    elif normalized == "storage":
        aliases.extend([DRIVE_ROLE, SSD_ROLE, HDD_ROLE])
    for alias in _unique(aliases):
        count = int(matrix_counts.get(alias, 0) or 0)
        if count > 0:
            return count, alias if alias != normalized else None
    return 0, None


def _v2_preferred_lifecycle_row(
    plan: CandidateUniversePlan,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    for row in rows:
        if _v2_role_is_hard_purchasable_bom(plan, row):
            return row
    for row in rows:
        if str(row.get("source") or "") != "broad_role_hints":
            return row
    return rows[0] if rows else {}


def _v2_role_lifecycle_fields(
    *,
    universe_plan: CandidateUniversePlan,
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    requirement_summary = _v2_role_requirement_summary(universe_plan)
    requested_rows = _v2_requested_role_rows(universe_plan)
    requested_roles = _unique(
        requirement_summary["broad_reasoning_roles"]
        or [row["role"] for row in requested_rows]
    )
    hard_purchasable_bom_roles = _string_list(
        requirement_summary["hard_purchasable_bom_roles"]
    )
    category_plan_roles = _unique(
        [
            role
            for role in (_normalize_plan_role(role) for role in universe_plan.broad_category_plan)
            if role
        ]
    )
    matrix_counts = _safe_mapping(matrix.get("full_candidate_matrix_count_by_role"))
    materialized_roles = _unique(
        [role for role, count in matrix_counts.items() if int(count or 0) > 0]
    )
    category_plan = {
        role: list(category_ids)
        for role, category_ids in universe_plan.broad_category_plan.items()
    }
    rows_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in requested_rows:
        rows_by_role[row["role"]].append(row)

    all_roles = _unique([*requested_roles, *category_plan_roles, *materialized_roles])
    role_fulfillment_diagnostics: list[dict[str, Any]] = []
    dropped_reason_by_role: dict[str, str] = {}
    coverage = _safe_mapping(matrix.get("role_coverage_summary"))
    for role in all_roles:
        role_rows = rows_by_role.get(role, [])
        lifecycle_row = _v2_preferred_lifecycle_row(universe_plan, role_rows)
        selected_category_ids = _string_list(category_plan.get(role))
        has_category = bool(selected_category_ids)
        candidate_count, fulfilled_by_role = _v2_role_candidate_count_with_alias(
            role,
            matrix_counts,
        )
        has_candidates = candidate_count > 0
        reason = (
            "sent_to_composer"
            if has_candidates
            else _v2_missing_role_reason(
                universe_plan,
                role,
                lifecycle_row,
                has_category=has_category,
                has_candidates=has_candidates,
            )
        )
        if reason != "sent_to_composer":
            dropped_reason_by_role[role] = reason
        role_fulfillment_diagnostics.append(
            {
                "role": role,
                "lifecycle_reason": reason,
                "selected_category_ids": selected_category_ids,
                "candidate_count": candidate_count,
                "fulfilled_by_role": fulfilled_by_role,
                "alias_fulfilled": bool(fulfilled_by_role and fulfilled_by_role != role),
                "source_texts": _unique(
                    [
                        str(
                            row.get("source_text")
                            or row.get("requirement_text")
                            or row.get("text")
                            or ""
                        ).strip()
                        for row in role_rows
                        if str(
                            row.get("source_text")
                            or row.get("requirement_text")
                            or row.get("text")
                            or ""
                        ).strip()
                    ]
                ),
                "hardness": (
                    "hard"
                    if any(
                        _v2_role_is_hard_purchasable_bom(universe_plan, row)
                        for row in role_rows
                    )
                    else
                    "optional"
                    if any(_requirement_hardness(row) == "optional" for row in role_rows)
                    else "hard"
                    if role_rows
                    else None
                ),
                "fulfillment_modes": _unique(
                    [
                        str(row.get("fulfillment_mode") or "").strip()
                        for row in role_rows
                        if str(row.get("fulfillment_mode") or "").strip()
                    ]
                ),
            }
        )
        if role not in coverage:
            coverage[role] = {
                "required": False,
                "missing": not has_candidates,
                "candidate_count": candidate_count,
                "after_category_count": candidate_count,
                "after_eligibility_count": candidate_count,
                "coverage_source": "v2_role_lifecycle",
                "category_ids": selected_category_ids,
                "lifecycle_reason": reason,
            }
        else:
            coverage[role] = {**_safe_mapping(coverage[role]), "lifecycle_reason": reason}

    role_source_by_role = role_lifecycle.merge_role_sources(
        ("stage_a", requested_roles),
        ("category_planner", category_plan_roles),
    )
    lifecycle_roles = _unique([*requested_roles, *category_plan_roles])
    trace = role_lifecycle.build_role_lifecycle_trace(
        lifecycle_roles,
        role_source_by_role=role_source_by_role,
        stage_a_roles=requested_roles,
        semantic_matrix_blueprint_roles=requested_roles,
        before_category_planner_roles=requested_roles,
        category_planner_input_roles=requested_roles,
        category_planner_output_roles=category_plan_roles,
        validated_category_plan_roles=category_plan_roles,
        materialized_matrix_roles=materialized_roles,
        composer_package_roles=materialized_roles,
        dropped_reason_by_role=dropped_reason_by_role,
    )
    return {
        "stage_a_broad_roles": requested_roles,
        "semantic_matrix_blueprint_roles": requested_roles,
        "requirement_classifier_roles": [],
        "hard_purchasable_bom_roles": hard_purchasable_bom_roles,
        "hard_purchasable_bom_role_requirements": requirement_summary[
            "hard_purchasable_bom_role_requirements"
        ],
        "primary_object_feature_requirements": requirement_summary[
            "primary_object_feature_requirements"
        ],
        "optional_accessory_engineering_roles": requirement_summary[
            "optional_accessory_engineering_roles"
        ],
        "optional_accessory_engineering_requirements": requirement_summary[
            "optional_accessory_engineering_requirements"
        ],
        "accessory_or_consumable_requirements": requirement_summary[
            "accessory_or_consumable_requirements"
        ],
        "engineering_check_requirements": requirement_summary[
            "engineering_check_requirements"
        ],
        "classified_requirements": requirement_summary["classified_requirements"],
        "effective_matrix_roles_before_category_planner": requested_roles,
        "category_planner_input_roles": requested_roles,
        "category_planner_output_roles": category_plan_roles,
        "validated_category_plan_roles": category_plan_roles,
        "materialized_matrix_roles": materialized_roles,
        "composer_package_roles": materialized_roles,
        "roles_dropped_after_stage_a": [],
        "roles_dropped_before_category_planner": [],
        "roles_dropped_after_category_planner": role_lifecycle.dropped_roles(
            requested_roles,
            category_plan_roles,
        ),
        "roles_dropped_during_materialization": role_lifecycle.dropped_roles(
            category_plan_roles,
            materialized_roles,
        ),
        "roles_dropped_reason_by_role": dropped_reason_by_role,
        "role_source_by_role": role_source_by_role,
        "role_lifecycle_trace": trace,
        "role_fulfillment_diagnostics": role_fulfillment_diagnostics,
        "requirement_fulfillment_decision": _unique_mapping_rows(
            [
                *universe_plan.requirement_fulfillment_decision,
                *role_fulfillment_diagnostics,
            ]
        ),
        "role_coverage_summary": coverage,
    }


def _build_v2_composer_package(
    *,
    original_request_text: str,
    normalized_requirements: Mapping[str, Any],
    component_candidate_matrix: Mapping[str, Any],
    settings: LlmSettings,
) -> dict[str, Any]:
    package = build_llm_configurator_package(
        user_request=original_request_text,
        normalized_requirements=dict(normalized_requirements),
        ready_stock_candidates=[],
        component_candidate_matrix=component_candidate_matrix,
        rule_based_build_candidates=[],
        candidates_per_role=max(
            1,
            _total_candidate_rows(component_candidate_matrix),
        ),
        proposal_pool_limit=1
        if settings.llm_configurator_output_mode == "single_best_cost_valid"
        else settings.llm_proposal_pool_limit,
        output_mode=settings.llm_configurator_output_mode,
        max_package_chars=settings.llm_configurator_max_package_chars,
    )
    package = dict(package)
    package["pipeline_version"] = PIPELINE_VERSION
    package["candidate_universe_planner_output"] = component_candidate_matrix.get(
        "candidate_universe_planner_output",
        {},
    )
    package["candidate_universe_category_plan"] = component_candidate_matrix.get(
        "candidate_universe_category_plan",
        {},
    )
    for key in (
        "candidate_universe_planner_mode",
        "primary_product_group",
        "selected_product_group_reason",
        "competing_product_groups",
        "primary_object_indicators",
        "component_role_indicators",
        "embedded_requirements",
        "requirement_fulfillment_decision",
        "role_fulfillment_diagnostics",
        "roles_sent_to_composer",
        "broad_reasoning_roles",
        "hard_purchasable_bom_roles",
        "hard_purchasable_bom_role_requirements",
        "purchasable_role_requirements",
        "primary_object_feature_requirements",
        "optional_accessory_engineering_roles",
        "optional_accessory_engineering_requirements",
        "accessory_or_consumable_requirements",
        "engineering_check_requirements",
        "logistics_or_commercial_constraint_requirements",
        "classified_requirements",
        "stage_a_broad_roles",
        "semantic_matrix_blueprint_roles",
        "effective_matrix_roles_before_category_planner",
        "category_planner_input_roles",
        "category_planner_output_roles",
        "validated_category_plan_roles",
        "materialized_matrix_roles",
        "composer_package_roles",
        "roles_dropped_after_category_planner",
        "roles_dropped_during_materialization",
        "roles_dropped_reason_by_role",
        "role_source_by_role",
        "role_lifecycle_trace",
        "excluded_category_groups",
        "rejected_category_count",
        "rejected_category_reasons",
        "category_catalog_total",
        "category_catalog_sent_to_ai_count",
        "category_catalog_truncated",
        "planner_repair_attempted",
        "planner_repair_success",
        "planner_suspicion_reasons",
        "matrix_materialized_count_by_role",
    ):
        package[key] = component_candidate_matrix.get(key)
    package["full_candidate_matrix_count_by_role"] = component_candidate_matrix.get(
        "full_candidate_matrix_count_by_role",
        {},
    )
    package["full_candidate_matrix_count_by_category"] = component_candidate_matrix.get(
        "full_candidate_matrix_count_by_category",
        {},
    )
    package["matrix_source_diagnostics"] = component_candidate_matrix.get(
        "matrix_source_diagnostics",
        {},
    )
    package["category_catalog_summary"] = component_candidate_matrix.get(
        "category_catalog_summary",
        {},
    )
    package["composer_package_full_matrix_used"] = _package_uses_full_matrix(
        package,
        component_candidate_matrix,
    )
    package["composer_package_full_matrix_policy"] = {
        "candidate_loss_allowed": False,
        "technical_trimming_allowed": False,
        "chunking_only_when_over_limit": True,
    }
    package = prepare_v2_composer_package(
        package,
        max_package_chars=settings.llm_configurator_max_package_chars,
    )
    package["composer_context_size"] = _package_context_size(package)
    return package


def _v2_effective_package_limit(
    package: Mapping[str, Any],
    settings: LlmSettings,
) -> int:
    budget_limit = _int_or_none(_safe_mapping(package.get("package_budget")).get("max_chars"))
    return int(budget_limit or settings.llm_configurator_max_package_chars)


def _v2_selected_context_chars(package: Mapping[str, Any]) -> int:
    selected = _int_or_none(package.get("selected_context_chars"))
    if selected is not None:
        return selected
    selected_size = _safe_mapping(package.get("selected_context_size"))
    selected = _int_or_none(selected_size.get("chars"))
    if selected is not None:
        return selected
    budget = _safe_mapping(package.get("package_budget"))
    final_chars = _int_or_none(budget.get("final_chars"))
    if final_chars is not None:
        return final_chars
    return _package_context_size(package)["chars"]


def _v2_package_build_blocker(package: Mapping[str, Any]) -> str | None:
    skipped_reason = str(package.get("package_skipped_reason") or "").strip()
    if skipped_reason in _PACKAGE_SIZE_SKIP_REASONS:
        return skipped_reason
    return None


def _v2_composer_attempt_decision(
    *,
    universe_plan: CandidateUniversePlan,
    matrix: Mapping[str, Any],
    package: Mapping[str, Any],
    settings: LlmSettings,
    llm_client: LlmClient | None,
    preview_only: bool,
    llm_call_budget: LlmCallBudget | None = None,
) -> dict[str, Any]:
    provider_configured = _provider_configured(settings, llm_client=llm_client)
    package_budget = _safe_mapping(package.get("package_budget"))
    effective_limit = _v2_effective_package_limit(package, settings)
    selected_context_chars = _v2_selected_context_chars(package)
    package_over_budget = selected_context_chars > effective_limit
    package_build_blocker = _v2_package_build_blocker(package)
    blocked_by: list[str] = []
    if universe_plan.product_group in {"", "unknown"}:
        blocked_by.append("product_group_unknown")
    if not universe_plan.included_category_ids:
        blocked_by.append("no_category_universe")
    if _total_candidate_rows(matrix) <= 0:
        blocked_by.append("no_candidates")
    if package_over_budget:
        blocked_by.append("package_over_budget")
    elif package_build_blocker:
        blocked_by.append(f"package_build_failed:{package_build_blocker}")
    if not provider_configured:
        blocked_by.append("provider_not_configured")
    if not preview_only and not settings.llm_configurator_enabled:
        blocked_by.append("llm_configurator_disabled")
    budget_diagnostics = llm_call_budget_diagnostics(llm_call_budget)
    if budget_diagnostics.get("llm_call_budget_exceeded"):
        blocked_by.append("llm_call_budget_exceeded")
    return {
        "enabled": bool(settings.llm_configurator_enabled),
        "pipeline_version": PIPELINE_VERSION,
        "expected_composer_mode": _v2_expected_composer_mode(settings),
        "role_evaluation_would_run": _v2_role_evaluation_would_run(settings),
        "package_mode": package.get("v2_package_mode")
        or package.get("selected_package_mode"),
        "package_size": _safe_mapping(package.get("selected_context_size"))
        or _safe_mapping(package.get("composer_context_size")),
        **budget_diagnostics,
        "package_present": bool(package),
        "v2_package_mode": package.get("v2_package_mode")
        or package.get("selected_package_mode"),
        "selected_context_chars": selected_context_chars,
        "effective_max_package_chars": effective_limit,
        "package_budget_max_chars": package_budget.get("max_chars"),
        "package_budget_final_chars": package_budget.get("final_chars"),
        "package_budget_over_budget": package_budget.get("over_budget") is True,
        "verbose_context_chars": package.get("verbose_context_chars"),
        "compact_context_chars": package.get("compact_context_chars"),
        "package_over_budget": package_over_budget,
        "package_skipped_reason": package.get("package_skipped_reason"),
        "package_build_blocker": package_build_blocker,
        "candidate_count_total": _composer_candidate_total_from_package(package),
        "candidate_count_by_role": _safe_mapping(
            package.get("composer_package_candidate_count_by_role")
        ),
        "required_roles": [],
        "provider_configured": provider_configured,
        "preview_only": preview_only,
        "attempt_skipped_for_preview": preview_only,
        "should_attempt": not blocked_by,
        "blocked_by": _unique(blocked_by),
        "allowed_blockers": [
            "product_group_unknown",
            "no_category_universe",
            "no_candidates",
            "package_over_budget",
            "package_build_failed",
            "provider_not_configured",
            "llm_configurator_disabled",
            "llm_call_budget_exceeded",
        ],
    }


def _v2_pipeline_mode(settings: LlmSettings) -> str:
    mode = str(settings.stock_match_pipeline_v2_mode or "").strip().casefold()
    return mode.replace("-", "_") or "composer_cascade"


def _v2_role_evaluation_would_run(settings: LlmSettings) -> bool:
    mode = _v2_pipeline_mode(settings)
    return bool(settings.llm_role_evaluation_enabled) or bool(
        settings.llm_composer_multi_pass
    ) or mode in {"deep_audit", "multi_pass", "multipass"}


def _v2_expected_composer_mode(settings: LlmSettings) -> str:
    return "deep_audit" if _v2_role_evaluation_would_run(settings) else "composer_cascade"


def _v2_expected_composer_mode_from_attempt(attempt_decision: Mapping[str, Any]) -> str:
    value = str(attempt_decision.get("expected_composer_mode") or "").strip()
    return value or "composer_cascade"


def _not_attempted_v2_outcome(
    *,
    settings: LlmSettings,
    package: Mapping[str, Any],
    attempt_decision: Mapping[str, Any],
) -> LlmConfiguratorOutcome:
    output_mode = str(
        package.get("output_mode")
        or settings.llm_configurator_output_mode
        or "single_best_cost_valid"
    )
    blocked_by = _string_list(attempt_decision.get("blocked_by"))
    fallback_reason = blocked_by[0] if blocked_by else "preview_only"
    if fallback_reason == "llm_call_budget_exceeded":
        reason = {
            "structured_no_recommendation": True,
            "summary": "LLM call budget was exceeded before Composer could safely run.",
            "fallback_reason": fallback_reason,
            "diagnostic_notes": [
                "The v2.2 bounded pipeline stopped instead of continuing hidden LLM calls."
            ],
            "missing_roles": [],
            "missing_required_capabilities": [],
            "hard_mismatches": [],
            "stock_shortages": [],
            "role_analysis": [],
            "considered_candidate_ids": {},
        }
        return LlmConfiguratorOutcome(
            enabled=bool(settings.llm_configurator_enabled),
            output_mode=output_mode,
            primary_recommendation_status="no_recommendation",
            no_recommendation_reason=reason,
            commercial_summary={"status": "no_recommendation"},
            fallback_reason=fallback_reason,
            composer_attempt_decision=dict(attempt_decision),
            package_diagnostics={
                "composer_attempt_decision": dict(attempt_decision),
                "final_status_source": "llm_call_budget_exceeded",
                **{
                    key: attempt_decision[key]
                    for key in (
                        "llm_call_count",
                        "llm_call_stages",
                        "llm_call_budget_exceeded",
                        "llm_call_budget_exceeded_stage",
                        "max_llm_calls_per_match",
                    )
                    if key in attempt_decision
                },
            },
            final_status_source="llm_call_budget_exceeded",
        )
    return LlmConfiguratorOutcome(
        enabled=bool(settings.llm_configurator_enabled),
        output_mode=output_mode,
        fallback_reason=fallback_reason,
        composer_attempt_decision=dict(attempt_decision),
        package_diagnostics={
            "composer_attempt_decision": dict(attempt_decision),
            "final_status_source": "composer_not_attempted",
        },
        final_status_source="composer_not_attempted",
    )


def _v2_execution_contract(
    *,
    package: Mapping[str, Any],
    matrix: Mapping[str, Any],
    attempt_decision: Mapping[str, Any],
    outcome: LlmConfiguratorOutcome,
    fallback_reason: str | None = None,
    final_status_source: str | None = None,
) -> dict[str, Any]:
    outcome_diagnostics = _safe_mapping(outcome.package_diagnostics)
    role_candidate_count = (
        _safe_mapping(package.get("composer_package_candidate_count_by_role"))
        or _safe_mapping(outcome_diagnostics.get("composer_package_candidate_count_by_role"))
        or _safe_mapping(matrix.get("full_candidate_matrix_count_by_role"))
        or _safe_mapping(matrix.get("count_by_role"))
    )
    selected_components = selected_components_by_role_from_recommendation(
        outcome.primary_recommendation
    )
    return build_execution_contract(
        product_group=package.get("product_group") or matrix.get("product_group"),
        primary_object=package.get("primary_object") or matrix.get("primary_object"),
        classified_requirements=(
            package.get("classified_requirements")
            or matrix.get("classified_requirements")
            or outcome_diagnostics.get("classified_requirements")
        ),
        hard_roles=(
            package.get("hard_purchasable_bom_roles")
            or matrix.get("hard_purchasable_bom_roles")
            or package.get("required_roles")
            or attempt_decision.get("required_roles")
        ),
        role_candidate_count=role_candidate_count,
        roles_sent_to_composer=(
            package.get("composer_package_roles")
            or matrix.get("composer_package_roles")
            or package.get("roles_sent_to_composer")
            or matrix.get("roles_sent_to_composer")
        ),
        role_lifecycle_trace=(
            package.get("role_lifecycle_trace")
            or matrix.get("role_lifecycle_trace")
            or outcome_diagnostics.get("role_lifecycle_trace")
        ),
        role_fulfillment_diagnostics=(
            matrix.get("role_fulfillment_diagnostics")
            or package.get("role_fulfillment_diagnostics")
            or outcome_diagnostics.get("role_fulfillment_diagnostics")
        ),
        roles_dropped_reason_by_role=(
            package.get("roles_dropped_reason_by_role")
            or matrix.get("roles_dropped_reason_by_role")
            or outcome_diagnostics.get("roles_dropped_reason_by_role")
        ),
        attempt_decision=attempt_decision,
        selected_components_by_role=selected_components,
        primary_recommendation_status=outcome.primary_recommendation_status,
        primary_recommendation=outcome.primary_recommendation,
        recommended_builds=outcome.recommended_builds,
        no_recommendation_reason=outcome.no_recommendation_reason,
        fallback_reason=fallback_reason or outcome.fallback_reason,
        error_type=outcome.error_type,
        parse_diagnostics=outcome.parse_diagnostics,
        proposal_count=outcome.proposal_count,
        valid_proposals_count=outcome.valid_proposals_count,
        final_status_source=final_status_source or outcome.final_status_source,
        llm_call_stages=_unique(
            [
                *_string_list(attempt_decision.get("llm_call_stages")),
                *_string_list(outcome_diagnostics.get("llm_call_stages")),
                *_string_list(
                    _safe_mapping(outcome.parse_diagnostics).get("llm_call_stages")
                ),
            ]
        ),
        validation_hard_mismatches=(
            outcome.validation_hard_mismatches
            or outcome_diagnostics.get("validation_hard_mismatches")
        ),
        validation_unverified_requirements=(
            outcome.validation_unverified_requirements
            or outcome_diagnostics.get("validation_unverified_requirements")
        ),
        validation_summary=outcome.validation_summary,
        rejected_recommendations_debug_safe=outcome.rejected_recommendations_debug_safe,
        original_request_text=(
            package.get("original_request_text")
            or package.get("user_request")
            or matrix.get("original_request_text")
        ),
    )


def _with_v2_execution_contract_status(
    outcome: LlmConfiguratorOutcome,
    *,
    package: Mapping[str, Any],
    matrix: Mapping[str, Any],
    attempt_decision: Mapping[str, Any],
) -> LlmConfiguratorOutcome:
    contract = _v2_execution_contract(
        package=package,
        matrix=matrix,
        attempt_decision=attempt_decision,
        outcome=outcome,
    )
    state = _safe_mapping(contract.get("composer_execution_state"))
    final_status_source = str(state.get("final_status_source") or "").strip()
    if not final_status_source or final_status_source == outcome.final_status_source:
        return outcome
    package_diagnostics = {
        **_safe_mapping(outcome.package_diagnostics),
        "final_status_source": final_status_source,
        **{
            key: contract[key]
            for key in (
                "requirement_graph",
                "candidate_universe_ledger",
                "composer_execution_state",
                "coverage_evidence",
                "validation_ledger",
                "execution_ledger",
            )
            if key in contract
        },
    }
    reason = _safe_mapping(outcome.no_recommendation_reason)
    if reason:
        reason = {
            **reason,
            "final_status_source": final_status_source,
            "diagnostics": {
                **_safe_mapping(reason.get("diagnostics")),
                "composer_execution_state": state,
            },
        }
    return replace(
        outcome,
        final_status_source=final_status_source,
        package_diagnostics=package_diagnostics,
        no_recommendation_reason=reason,
    )


def _ensure_v2_safe_no_recommendation(
    outcome: LlmConfiguratorOutcome,
    *,
    package: Mapping[str, Any],
    matrix: Mapping[str, Any],
    attempt_decision: Mapping[str, Any],
) -> LlmConfiguratorOutcome:
    if outcome.primary_recommendation_status == "valid":
        return outcome
    if (
        outcome.primary_recommendation_status == "no_recommendation"
        and outcome.no_recommendation_reason
        and outcome.final_status_source
    ):
        return outcome
    fallback_reason = str(
        outcome.fallback_reason
        or outcome.error_type
        or package.get("llm_fallback_reason")
        or package.get("package_skipped_reason")
        or COMPOSER_NO_SAFE_COMPLETE_BOM
    ).strip()
    final_status_source = outcome.final_status_source or _v2_failure_final_status_source(
        fallback_reason
    )
    reason = _v2_safe_no_recommendation_reason(
        fallback_reason=fallback_reason,
        final_status_source=final_status_source,
        package=package,
        matrix=matrix,
        attempt_decision=attempt_decision,
        outcome=outcome,
    )
    package_diagnostics = {
        **_safe_mapping(outcome.package_diagnostics),
        "final_status_source": final_status_source,
        "no_recommendation_reason": reason,
        "structured_no_recommendation_used": True,
    }
    return replace(
        outcome,
        primary_recommendation_status="no_recommendation",
        no_recommendation_reason=reason,
        commercial_summary={"status": "no_recommendation"},
        fallback_reason=fallback_reason,
        package_diagnostics=package_diagnostics,
        final_status_source=final_status_source,
    )


def _v2_failure_final_status_source(fallback_reason: str) -> str:
    reason = str(fallback_reason or "").strip()
    if reason == "llm_call_budget_exceeded":
        return "llm_call_budget_exceeded"
    if reason in {
        "composer_provider_timeout",
        "multi_pass_read_timeout",
        "llm_configurator_read_timeout_not_retried",
    }:
        return "composer_provider_timeout"
    if reason in {"llm_configurator_validation_failed", "multi_pass_validation_failed"}:
        return COMPOSER_SCHEMA_VALIDATION_FAILED
    if "context_limit" in reason:
        return "provider_context_limit"
    if reason.startswith("llm_configurator_not_attempted"):
        return "composer_not_attempted"
    return "composer_failure_safe_no_recommendation"


def _v2_hard_roles_from_context(
    *,
    package: Mapping[str, Any],
    matrix: Mapping[str, Any],
    attempt_decision: Mapping[str, Any],
) -> list[str]:
    roles = _string_list(
        package.get("hard_purchasable_bom_roles")
        or matrix.get("hard_purchasable_bom_roles")
        or attempt_decision.get("required_roles")
        or package.get("required_roles")
    )
    if roles:
        return roles
    product_group = str(package.get("product_group") or matrix.get("product_group") or "")
    primary_object = _normalize_plan_role(
        package.get("primary_object") or matrix.get("primary_object")
    )
    if product_group == NETWORK_PRODUCT_GROUP and primary_object in _NETWORK_PRIMARY_DEVICE_ROLES:
        return [primary_object]
    if product_group == STORAGE_PRODUCT_GROUP:
        return [STORAGE_SYSTEM_ROLE]
    if product_group == SERVER_PRODUCT_GROUP:
        return [SERVER_PLATFORM_ROLE]
    return []


def _v2_lifecycle_reason_is_non_blocking(reason: str) -> bool:
    reason_text = str(reason or "").strip()
    return (
        reason_text
        in {
            "sent_to_composer",
            "included_in_primary_object",
            "included_in_selected_component",
            "optional_only",
            "engineering_check_only",
            "logistics_constraint",
            "satisfied_by_platform",
            "satisfied_by_ready_server",
        }
        or reason_text.startswith("fulfilled_by_role:")
    )


def _v2_safe_no_recommendation_reason(
    *,
    fallback_reason: str,
    final_status_source: str,
    package: Mapping[str, Any],
    matrix: Mapping[str, Any],
    attempt_decision: Mapping[str, Any],
    outcome: LlmConfiguratorOutcome,
) -> dict[str, Any]:
    contract = _v2_execution_contract(
        package=package,
        matrix=matrix,
        attempt_decision=attempt_decision,
        outcome=outcome,
        fallback_reason=fallback_reason,
        final_status_source=final_status_source,
    )
    reason = _safe_mapping(contract.get("safe_no_recommendation"))
    diagnostics = {
        **_safe_mapping(reason.get("diagnostics")),
        "final_status_source": _safe_mapping(
            contract.get("composer_execution_state")
        ).get("final_status_source")
        or final_status_source,
        "composer_attempt_decision": dict(attempt_decision),
        "package_mode": package.get("v2_package_mode")
        or package.get("selected_package_mode"),
        "composer_package_candidate_count_by_role": _safe_mapping(
            package.get("composer_package_candidate_count_by_role")
        )
        or _safe_mapping(matrix.get("full_candidate_matrix_count_by_role")),
        "role_fulfillment_diagnostics": _mapping_rows(
            matrix.get("role_fulfillment_diagnostics")
        ),
        "role_lifecycle_trace": _mapping_rows(matrix.get("role_lifecycle_trace")),
        "llm_error_type": outcome.error_type,
        "llm_parse_diagnostics": _safe_mapping(outcome.parse_diagnostics),
    }
    reason.update(
        {
            "fallback_reason": fallback_reason,
            "final_status_source": diagnostics["final_status_source"],
            "original_request_text": package.get("original_request_text")
            or package.get("user_request"),
            "unverified_requirements": _mapping_rows(
                outcome.validation_unverified_requirements
            ),
            "diagnostics": diagnostics,
        }
    )
    return reason


def _with_v2_final_status(outcome: LlmConfiguratorOutcome) -> LlmConfiguratorOutcome:
    final_status = outcome.final_status_source
    validation_rejected_count = int(
        outcome.validation_summary.get("validation_rejected_count", 0) or 0
    )
    if _v2_composer_schema_validation_failed(outcome):
        final_status = COMPOSER_SCHEMA_VALIDATION_FAILED
    elif (
        outcome.validation_rejected_count > 0
        or outcome.rejected_recommendations_count > 0
        or validation_rejected_count > 0
    ):
        final_status = COMPOSER_REJECTED_BY_VALIDATION
    elif (
        outcome.primary_recommendation_status == "valid"
        or outcome.valid_proposals_count > 0
        or bool(outcome.primary_recommendation)
    ):
        final_status = "composer_validated"
    return replace(outcome, final_status_source=final_status)


def _v2_composer_schema_validation_failed(outcome: LlmConfiguratorOutcome) -> bool:
    diagnostics = _safe_mapping(outcome.parse_diagnostics)
    fallback_reason = str(outcome.fallback_reason or "").strip()
    final_status = str(outcome.final_status_source or "").strip()
    if final_status == COMPOSER_SCHEMA_VALIDATION_FAILED:
        return True
    if outcome.error_type != "ValidationError":
        return False
    if fallback_reason in {
        "llm_configurator_validation_failed",
        "multi_pass_validation_failed",
    }:
        return True
    return str(diagnostics.get("parse_status") or "").strip() == "validation_error"


def _apply_v2_validation_overrides(
    outcome: LlmConfiguratorOutcome,
) -> LlmConfiguratorOutcome:
    stock_shortages = _v2_llm_quantity_stock_shortages(outcome)
    if not stock_shortages:
        return outcome
    validation_summary = dict(outcome.validation_summary)
    for key in (
        "validation_rejected_count",
        "rejected",
        "rejected_stock_shortage",
        "rejected_stock",
        "rejected_fatal",
    ):
        validation_summary[key] = int(validation_summary.get(key, 0) or 0) + 1
    validation_hard_mismatches = [
        *outcome.validation_hard_mismatches,
        *[
            {
                "type": "stock_quantity_mismatch",
                "role": row.get("role"),
                "component_candidate_id": row.get("component_candidate_id"),
                "required_quantity": row.get("required_quantity"),
                "available_quantity": row.get("available_quantity"),
                "message": "Composer requested a quantity greater than available stock.",
            }
            for row in stock_shortages
        ],
    ]
    rejected_debug = [
        *outcome.rejected_recommendations_debug_safe,
        {
            "recommendation_id": _primary_recommendation_id(outcome),
            "rejection_category": "stock_shortage",
            "rejection_code": "stock_quantity_mismatch",
            "rejection_message_ru": (
                "Composer requested a quantity greater than available stock"
            ),
            "stock_shortages": stock_shortages,
            "validation_hard_mismatches": validation_hard_mismatches,
            "stage": "v2_post_composer_validation",
        },
    ]
    no_recommendation_reason = {
        "structured_no_recommendation": True,
        "summary": "Composer BOM was rejected by v2 validation.",
        "stock_shortages": stock_shortages,
        "hard_mismatches": validation_hard_mismatches,
        "missing_roles": [],
        "missing_required_capabilities": [],
        "role_analysis": [],
        "recommended_next_actions": ["Escalate to an engineer before preparing a quote."],
    }
    normalized_result = normalize_composer_result(
        product_group=str(
            _safe_mapping(outcome.primary_recommendation).get("product_group") or ""
        ),
        primary_object=str(
            _safe_mapping(outcome.primary_recommendation).get("primary_object") or ""
        ),
        bom_composer_output=_safe_mapping(outcome.primary_recommendation),
        code_validation_result={
            "validation_hard_mismatches": validation_hard_mismatches,
            "validation_summary": validation_summary,
            "rejected_recommendations": rejected_debug,
        },
        final_status_source=COMPOSER_REJECTED_BY_VALIDATION,
        primary_recommendation_status="no_recommendation",
        llm_fallback_reason=COMPOSER_NO_SAFE_COMPLETE_BOM,
        existing_no_recommendation_reason=no_recommendation_reason,
    )
    no_recommendation_reason = _safe_mapping(
        normalized_result.get("no_recommendation_reason")
    )
    return replace(
        outcome,
        recommended_builds=[],
        primary_recommendation={},
        primary_recommendation_status="no_recommendation",
        no_recommendation_reason=no_recommendation_reason,
        fallback_reason=str(
            normalized_result.get("llm_fallback_reason")
            or COMPOSER_NO_SAFE_COMPLETE_BOM
        ),
        valid_proposals_count=0,
        validation_rejected_count=outcome.validation_rejected_count + 1,
        rejected_recommendations_count=outcome.rejected_recommendations_count + 1,
        validation_summary=validation_summary,
        rejected_recommendations_debug_safe=rejected_debug,
        validation_hard_mismatches=validation_hard_mismatches,
        final_status_source=COMPOSER_REJECTED_BY_VALIDATION,
    )


def _v2_llm_quantity_stock_shortages(
    outcome: LlmConfiguratorOutcome,
) -> list[dict[str, Any]]:
    shortages: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    recommendations = [
        outcome.primary_recommendation,
        *outcome.recommended_builds,
    ]
    for recommendation in recommendations:
        if not isinstance(recommendation, Mapping):
            continue
        recommendation_id = str(
            recommendation.get("recommendation_id")
            or recommendation.get("candidate_id")
            or recommendation.get("title")
            or "composer_recommendation"
        )
        for component in _mapping_rows(recommendation.get("components")):
            component_id = str(component.get("component_candidate_id") or "").strip()
            role = str(component.get("role") or "").strip()
            pricing_scope = str(component.get("pricing_scope") or "").strip()
            if bool(component.get("optional_component")) or (
                pricing_scope and pricing_scope != "core"
            ):
                continue
            llm_quantity = _int_or_none(component.get("llm_quantity"))
            required_quantity = (
                llm_quantity
                if llm_quantity is not None
                else _int_or_none(component.get("quantity_required"))
            )
            available_quantity = _int_or_none(component.get("available_quantity"))
            if (
                not component_id
                or required_quantity is None
                or required_quantity <= 0
                or available_quantity is None
                or required_quantity <= available_quantity
            ):
                continue
            key = (recommendation_id, component_id)
            if key in seen:
                continue
            seen.add(key)
            shortages.append(
                {
                    "recommendation_id": recommendation_id,
                    "role": role,
                    "component_candidate_id": component_id,
                    "required_quantity": required_quantity,
                    "available_quantity": available_quantity,
                    "quantity_source": "composer_quantity",
                }
            )
    return shortages


def _primary_recommendation_id(outcome: LlmConfiguratorOutcome) -> str:
    primary = _safe_mapping(outcome.primary_recommendation)
    return str(
        primary.get("recommendation_id")
        or primary.get("candidate_id")
        or primary.get("title")
        or "composer_recommendation"
    )


def _v2_package_diagnostics(
    *,
    package: Mapping[str, Any],
    matrix: Mapping[str, Any],
    universe_plan: CandidateUniversePlan,
    attempt_decision: Mapping[str, Any],
    llm_outcome: LlmConfiguratorOutcome,
    llm_call_budget: LlmCallBudget | None,
) -> dict[str, Any]:
    outcome_diagnostics = _safe_mapping(llm_outcome.package_diagnostics)
    budget_diagnostics = llm_call_budget_diagnostics(llm_call_budget)
    final_status_source = llm_outcome.final_status_source or outcome_diagnostics.get(
        "final_status_source"
    )
    validation_hard_mismatches = _mapping_rows(
        llm_outcome.validation_hard_mismatches
        or outcome_diagnostics.get("validation_hard_mismatches")
    )
    validation_unverified = _mapping_rows(
        llm_outcome.validation_unverified_requirements
        or outcome_diagnostics.get("validation_unverified_requirements")
    )
    diagnostics = {
        **outcome_diagnostics,
        "pipeline_version": PIPELINE_VERSION,
        "composer_mode": outcome_diagnostics.get("composer_mode")
        or attempt_decision.get("expected_composer_mode")
        or _v2_expected_composer_mode_from_attempt(attempt_decision),
        "expected_composer_mode": attempt_decision.get("expected_composer_mode"),
        "llm_call_count": budget_diagnostics.get("llm_call_count"),
        "llm_call_stages": budget_diagnostics.get("llm_call_stages"),
        "llm_call_budget_exceeded": budget_diagnostics.get(
            "llm_call_budget_exceeded"
        ),
        "llm_call_budget_exceeded_stage": budget_diagnostics.get(
            "llm_call_budget_exceeded_stage"
        ),
        "max_llm_calls_per_match": budget_diagnostics.get("max_llm_calls_per_match"),
        "requirement_contract_used": bool(
            outcome_diagnostics.get("requirement_contract_used")
        ),
        "main_composer_used": bool(outcome_diagnostics.get("main_composer_used")),
        "critic_used": bool(
            outcome_diagnostics.get("critic_used")
            or outcome_diagnostics.get("completeness_critic_used")
        ),
        "repair_used": bool(
            outcome_diagnostics.get("repair_used")
            or outcome_diagnostics.get("repair_composer_used")
        ),
        "role_evaluation_used": bool(outcome_diagnostics.get("role_evaluation_used")),
        "role_evaluation_skipped_reason": outcome_diagnostics.get(
            "role_evaluation_skipped_reason"
        )
        or (
            "default_composer_cascade"
            if not bool(outcome_diagnostics.get("role_evaluation_used"))
            else None
        ),
        "candidate_universe_planner_mode": (
            universe_plan.candidate_universe_planner_mode
        ),
        "category_catalog_total": universe_plan.category_catalog_total,
        "category_catalog_sent_to_ai_count": (
            universe_plan.category_catalog_sent_to_ai_count
        ),
        "category_catalog_truncated": universe_plan.category_catalog_truncated,
        "category_catalog_summary": _safe_mapping(
            package.get("category_catalog_summary")
            or matrix.get("category_catalog_summary")
        ),
        "primary_product_group": universe_plan.product_group,
        "primary_object": universe_plan.primary_object,
        "selected_group_reason": universe_plan.selected_group_reason,
        "selected_product_group_reason": universe_plan.selected_group_reason,
        "competing_product_groups": universe_plan.competing_product_groups,
        "primary_object_indicators": universe_plan.primary_object_indicators,
        "component_role_indicators": universe_plan.component_role_indicators,
        "embedded_requirements": universe_plan.embedded_requirements,
        "roles_sent_to_composer": _string_list(
            package.get("roles_sent_to_composer")
            or matrix.get("roles_sent_to_composer")
        ),
        "broad_reasoning_roles": _string_list(
            package.get("broad_reasoning_roles")
            or matrix.get("broad_reasoning_roles")
        ),
        "hard_purchasable_bom_roles": _string_list(
            package.get("hard_purchasable_bom_roles")
            or matrix.get("hard_purchasable_bom_roles")
        ),
        "hard_purchasable_bom_role_requirements": _mapping_rows(
            package.get("hard_purchasable_bom_role_requirements")
            or matrix.get("hard_purchasable_bom_role_requirements")
        ),
        "purchasable_role_requirements": _mapping_rows(
            package.get("purchasable_role_requirements")
            or matrix.get("purchasable_role_requirements")
        ),
        "primary_object_feature_requirements": _mapping_rows(
            package.get("primary_object_feature_requirements")
            or matrix.get("primary_object_feature_requirements")
        ),
        "optional_accessory_engineering_roles": _string_list(
            package.get("optional_accessory_engineering_roles")
            or matrix.get("optional_accessory_engineering_roles")
        ),
        "optional_accessory_engineering_requirements": _mapping_rows(
            package.get("optional_accessory_engineering_requirements")
            or matrix.get("optional_accessory_engineering_requirements")
        ),
        "accessory_or_consumable_requirements": _mapping_rows(
            package.get("accessory_or_consumable_requirements")
            or matrix.get("accessory_or_consumable_requirements")
        ),
        "engineering_check_requirements": _mapping_rows(
            package.get("engineering_check_requirements")
            or matrix.get("engineering_check_requirements")
        ),
        "logistics_or_commercial_constraint_requirements": _mapping_rows(
            package.get("logistics_or_commercial_constraint_requirements")
            or matrix.get("logistics_or_commercial_constraint_requirements")
        ),
        "classified_requirements": _mapping_rows(
            package.get("classified_requirements")
            or matrix.get("classified_requirements")
        ),
        "requirement_fulfillment_decision": _mapping_rows(
            matrix.get("requirement_fulfillment_decision")
        )
        or universe_plan.requirement_fulfillment_decision,
        "role_fulfillment_diagnostics": _mapping_rows(
            matrix.get("role_fulfillment_diagnostics")
        ),
        "accessory_indicators": universe_plan.accessory_indicators,
        "service_support_indicators": universe_plan.service_support_indicators,
        "logistics_commercial_constraints": (
            universe_plan.logistics_commercial_constraints
        ),
        "category_plan_entries": universe_plan.category_plan_entries,
        "selected_category_count": len(universe_plan.included_category_ids),
        "rejected_category_count": len(universe_plan.rejected_category_reasons),
        "rejected_category_reasons": universe_plan.rejected_category_reasons,
        "excluded_category_groups": universe_plan.excluded_category_groups,
        "planner_repair_attempted": universe_plan.planner_repair_attempted,
        "planner_repair_success": universe_plan.planner_repair_success,
        "planner_suspicion_reasons": universe_plan.planner_suspicion_reasons,
        "candidate_universe_planner_output": universe_plan.to_report_json(),
        "candidate_universe_category_plan": universe_plan.broad_category_plan,
        "stage_a_broad_roles": _string_list(matrix.get("stage_a_broad_roles")),
        "semantic_matrix_blueprint_roles": _string_list(
            matrix.get("semantic_matrix_blueprint_roles")
        ),
        "effective_matrix_roles_before_category_planner": _string_list(
            matrix.get("effective_matrix_roles_before_category_planner")
        ),
        "category_planner_input_roles": _string_list(
            matrix.get("category_planner_input_roles")
        ),
        "category_planner_output_roles": _string_list(
            matrix.get("category_planner_output_roles")
        ),
        "validated_category_plan_roles": _string_list(
            matrix.get("validated_category_plan_roles")
        ),
        "materialized_matrix_roles": _string_list(
            matrix.get("materialized_matrix_roles")
        ),
        "composer_package_roles": _string_list(package.get("composer_package_roles"))
        or _string_list(matrix.get("composer_package_roles")),
        "roles_dropped_after_category_planner": _string_list(
            matrix.get("roles_dropped_after_category_planner")
        ),
        "roles_dropped_during_materialization": _string_list(
            matrix.get("roles_dropped_during_materialization")
        ),
        "roles_dropped_reason_by_role": _safe_mapping(
            package.get("roles_dropped_reason_by_role")
        )
        or _safe_mapping(matrix.get("roles_dropped_reason_by_role")),
        "role_source_by_role": _safe_mapping(matrix.get("role_source_by_role")),
        "role_lifecycle_trace": _mapping_rows(package.get("role_lifecycle_trace"))
        or _mapping_rows(matrix.get("role_lifecycle_trace")),
        "full_candidate_matrix_count_by_role": _safe_mapping(
            matrix.get("full_candidate_matrix_count_by_role")
        ),
        "matrix_materialized_count_by_role": _safe_mapping(
            matrix.get("matrix_materialized_count_by_role")
        ),
        "full_candidate_matrix_count_by_category": _safe_mapping(
            matrix.get("full_candidate_matrix_count_by_category")
        ),
        "matrix_source_diagnostics": _safe_mapping(
            matrix.get("matrix_source_diagnostics")
        ),
        "count_by_role": _safe_mapping(matrix.get("count_by_role")),
        "broad_matrix_count_by_role": _safe_mapping(
            package.get("broad_matrix_count_by_role")
            or matrix.get("broad_matrix_count_by_role")
        ),
        "composer_package_candidate_count_by_role": _safe_mapping(
            package.get("composer_package_candidate_count_by_role")
        ),
        "composer_package_candidate_total": _composer_candidate_total_from_package(
            package
        ),
        "composer_package_candidate_ids_by_role": _safe_mapping(
            package.get("composer_package_candidate_ids_by_role")
        ),
        "composer_package_full_matrix_used": bool(
            package.get("composer_package_full_matrix_used")
        ),
        "composer_context_size": _safe_mapping(package.get("composer_context_size")),
        "composer_attempt_decision": dict(attempt_decision),
        "package_strategy_decision": {
            "pipeline_version": PIPELINE_VERSION,
            "strategy": (
                "full_broad_package"
                if not package.get("package_skipped_reason")
                else "package_blocked_before_composer"
            ),
            "candidate_exposure_mode": FULL_BROAD_MATRIX_EXPOSURE_MODE,
            "silent_top_n_trimming": False,
            "package_skipped_reason": package.get("package_skipped_reason"),
            "package_over_budget": attempt_decision.get("package_over_budget"),
        },
        "composer_requirement_analysis": _safe_mapping(
            llm_outcome.composer_requirement_analysis
            or outcome_diagnostics.get("composer_requirement_analysis")
        ),
        "composer_fulfillment_decisions": _mapping_rows(
            llm_outcome.composer_fulfillment_decisions
            or outcome_diagnostics.get("composer_fulfillment_decisions")
        ),
        "composer_selected_components": _mapping_rows(
            llm_outcome.composer_selected_components
            or outcome_diagnostics.get("composer_selected_components")
        ),
        "composer_quantities": _safe_mapping(
            llm_outcome.composer_quantities
            or outcome_diagnostics.get("composer_quantities")
        ),
        "composer_assumptions": _string_list(
            llm_outcome.composer_assumptions
            or outcome_diagnostics.get("composer_assumptions")
        ),
        "composer_engineer_checks": _string_list(
            llm_outcome.composer_engineer_checks
            or outcome_diagnostics.get("composer_engineer_checks")
        ),
        "composer_hard_mismatch_risks": _mapping_rows(
            llm_outcome.composer_hard_mismatch_risks
            or outcome_diagnostics.get("composer_hard_mismatch_risks")
        ),
        "composer_unverified_requirements": _mapping_rows(
            llm_outcome.composer_unverified_requirements
            or outcome_diagnostics.get("composer_unverified_requirements")
        ),
        "composer_considered_candidate_count_by_role": _safe_mapping(
            llm_outcome.composer_considered_candidate_count_by_role
            or outcome_diagnostics.get("composer_considered_candidate_count_by_role")
        ),
        "composer_chosen_candidate_ids": _string_list(
            llm_outcome.composer_chosen_candidate_ids
            or outcome_diagnostics.get("composer_chosen_candidate_ids")
        ),
        "validation_hard_mismatches": validation_hard_mismatches,
        "validation_unverified_requirements": validation_unverified,
        "final_status_source": final_status_source,
        "package_budget": _safe_mapping(package.get("package_budget")),
        "effective_max_package_chars": attempt_decision.get(
            "effective_max_package_chars"
        ),
        "package_budget_selected_context_chars": attempt_decision.get(
            "selected_context_chars"
        ),
        "package_budget_over_budget": attempt_decision.get("package_over_budget"),
        "package_budget_warnings": _string_list(package.get("package_budget_warnings")),
        "package_skipped_reason": package.get("package_skipped_reason"),
        "package_candidate_exposure_policy": {
            "mode": FULL_BROAD_MATRIX_EXPOSURE_MODE,
            "allow_incomplete": False,
            "candidate_matrix_trimming_allowed": False,
            "pre_composer_semantic_gates_enabled": False,
            **_safe_mapping(package.get("package_candidate_exposure_policy")),
            **_safe_mapping(
                outcome_diagnostics.get("package_candidate_exposure_policy")
            ),
        },
        "package_candidate_exposure_incomplete": False,
        "package_candidate_exposure_incomplete_roles": [],
        "package_exposure_blocking_lifecycle_roles": [],
        "dropped_before_composer_count_by_role": _zero_dropped_counts(package),
        "dropped_before_composer_count_by_reason": (
            _dropped_before_composer_count_by_reason(matrix)
        ),
        "dropped_before_composer_reason_by_role": {},
        "pre_composer_requirement_classifier_status": None,
        "pre_composer_requirement_source_coverage_percent": None,
        "pre_composer_unclassified_source_fragments": [],
        "pre_composer_semantic_diagnostics_are_blocking": False,
    }
    execution_contract = _v2_execution_contract(
        package=package,
        matrix=matrix,
        attempt_decision=attempt_decision,
        outcome=llm_outcome,
    )
    execution_state = _safe_mapping(
        execution_contract.get("composer_execution_state")
    )
    diagnostics.update(
        {
            "requirement_graph": _safe_mapping(
                execution_contract.get("requirement_graph")
            ),
            "candidate_universe_ledger": _safe_mapping(
                execution_contract.get("candidate_universe_ledger")
            ),
            "composer_execution_state": execution_state,
            "coverage_evidence": _mapping_rows(
                execution_contract.get("coverage_evidence")
            ),
            "validation_ledger": _safe_mapping(
                execution_contract.get("validation_ledger")
            ),
            "execution_ledger": _safe_mapping(
                execution_contract.get("execution_ledger")
            ),
            "safe_no_recommendation": _safe_mapping(
                execution_contract.get("safe_no_recommendation")
            ),
            "final_status_source": (
                execution_state.get("final_status_source") or final_status_source
            ),
        }
    )
    if llm_outcome.no_recommendation_reason:
        diagnostics["no_recommendation_reason"] = _safe_mapping(
            llm_outcome.no_recommendation_reason
        )
    for key in _V2_COMPACT_PACKAGE_DIAGNOSTIC_KEYS:
        if key in package:
            diagnostics[key] = package[key]
        elif key in outcome_diagnostics:
            diagnostics[key] = outcome_diagnostics[key]
    if outcome_diagnostics.get("provider_context_limit_retry_compact_attempted"):
        for key in _V2_COMPACT_PACKAGE_DIAGNOSTIC_KEYS:
            if key in outcome_diagnostics:
                diagnostics[key] = outcome_diagnostics[key]
    return diagnostics


async def _load_distributor_products(
    session: AsyncSession,
    *,
    distributor_code: str,
) -> list[DistributorProduct]:
    result = await session.execute(
        select(DistributorProduct)
        .where(DistributorProduct.distributor_code == distributor_code)
        .order_by(
            DistributorProduct.synced_at.desc(),
            DistributorProduct.id.desc(),
        )
    )
    return list(result.scalars().all())


async def _load_distributor_categories(
    session: AsyncSession,
    *,
    distributor_code: str,
) -> list[DistributorCategory]:
    result = await session.execute(
        select(DistributorCategory)
        .where(DistributorCategory.distributor_code == distributor_code)
        .order_by(DistributorCategory.level, DistributorCategory.name)
    )
    return list(result.scalars().all())


async def _load_latest_stock_rows(
    session: AsyncSession,
    products: Sequence[DistributorProduct],
    *,
    distributor_code: str,
) -> dict[tuple[str, str], list[DistributorStockPrice]]:
    item_ids = sorted(
        {
            product.item_id
            for product in products
            if product.distributor_code == distributor_code
        }
    )
    if not item_ids:
        return {}
    latest_per_item = (
        select(
            DistributorStockPrice.distributor_code.label("distributor_code"),
            DistributorStockPrice.item_id.label("item_id"),
            func.max(DistributorStockPrice.synced_at).label("synced_at"),
        )
        .where(
            DistributorStockPrice.distributor_code == distributor_code,
            DistributorStockPrice.item_id.in_(item_ids),
        )
        .group_by(
            DistributorStockPrice.distributor_code,
            DistributorStockPrice.item_id,
        )
        .subquery()
    )
    result = await session.execute(
        select(DistributorStockPrice)
        .join(
            latest_per_item,
            (
                DistributorStockPrice.distributor_code
                == latest_per_item.c.distributor_code
            )
            & (DistributorStockPrice.item_id == latest_per_item.c.item_id)
            & (DistributorStockPrice.synced_at == latest_per_item.c.synced_at),
        )
        .order_by(
            DistributorStockPrice.item_id,
            DistributorStockPrice.location,
            DistributorStockPrice.id,
        )
    )
    rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]] = {}
    for row in result.scalars().all():
        rows_by_key.setdefault((row.distributor_code, row.item_id), []).append(row)
    return rows_by_key


def _category_summaries(
    *,
    categories: Sequence[DistributorCategory],
    products: Sequence[DistributorProduct],
    stock_rows_by_key: Mapping[tuple[str, str], list[DistributorStockPrice]],
    distributor_code: str,
) -> list[CategorySummary]:
    product_count_by_category: Counter[str] = Counter()
    stocked_count_by_category: Counter[str] = Counter()
    priced_count_by_category: Counter[str] = Counter()
    product_names_by_category: dict[str, list[str]] = defaultdict(list)
    producers_by_category: dict[str, list[str]] = defaultdict(list)
    part_numbers_by_category: dict[str, list[str]] = defaultdict(list)
    for product in products:
        category_id = str(product.category_id or "").strip()
        if not category_id:
            continue
        product_count_by_category[category_id] += 1
        name = _product_text(product)
        if name and len(product_names_by_category[category_id]) < 12:
            product_names_by_category[category_id].append(_short_catalog_sample(name))
        producer = _short_catalog_sample(str(product.producer or ""))
        if producer and len(producers_by_category[category_id]) < 12:
            producers_by_category[category_id].append(producer)
        part_number = _short_catalog_sample(str(product.part_number or ""))
        if part_number and len(part_numbers_by_category[category_id]) < 12:
            part_numbers_by_category[category_id].append(part_number)
        stock_rows = stock_rows_by_key.get((product.distributor_code, product.item_id), [])
        available_quantity = _available_quantity(stock_rows)
        if available_quantity is not None and available_quantity > 0:
            stocked_count_by_category[category_id] += 1
        price_value, price_currency = _select_price(stock_rows)
        if price_value is not None and str(price_currency or "").strip():
            priced_count_by_category[category_id] += 1

    summaries: dict[str, CategorySummary] = {}
    anchors_by_category_id = _ocs_anchors_by_category_id(distributor_code)
    for category in categories:
        category_id = str(category.category_id or "").strip()
        if not category_id:
            continue
        path = [
            str(item.get("name") or item.get("category_name") or "").strip()
            for item in category.path_json
            if isinstance(item, Mapping)
        ]
        path = [item for item in path if item]
        sample_names = product_names_by_category.get(category_id, [])
        metadata = _category_summary_metadata(
            category_id=category_id,
            category_name=category.name,
            path=path,
            sample_product_names=sample_names,
            anchors=anchors_by_category_id.get(category_id, ()),
        )
        summaries[category_id] = CategorySummary(
            category_id=category_id,
            distributor_code=str(category.distributor_code or distributor_code),
            name=category.name,
            path=path,
            parent_category_id=category.parent_category_id,
            product_count=product_count_by_category[category_id],
            stocked_count=stocked_count_by_category[category_id],
            priced_count=priced_count_by_category[category_id],
            sample_product_names=sample_names,
            sample_producers=_unique(producers_by_category.get(category_id, []))[:12],
            sample_part_numbers=_unique(part_numbers_by_category.get(category_id, []))[:12],
            product_group_contexts=metadata["product_group_contexts"],
            allowed_roles=metadata["allowed_roles"],
            suggested_roles=metadata["suggested_roles"],
            category_kind=metadata["category_kind"],
            metadata_source=metadata["metadata_source"],
            notes=metadata["notes"],
            warnings=metadata["warnings"],
        )
    for category_id, sample_names in product_names_by_category.items():
        if category_id in summaries:
            continue
        metadata = _category_summary_metadata(
            category_id=category_id,
            category_name=category_id,
            path=[],
            sample_product_names=sample_names,
            anchors=anchors_by_category_id.get(category_id, ()),
        )
        summaries[category_id] = CategorySummary(
            category_id=category_id,
            distributor_code=distributor_code,
            name=category_id,
            path=[],
            product_count=product_count_by_category[category_id],
            stocked_count=stocked_count_by_category[category_id],
            priced_count=priced_count_by_category[category_id],
            sample_product_names=sample_names,
            sample_producers=_unique(producers_by_category.get(category_id, []))[:12],
            sample_part_numbers=_unique(part_numbers_by_category.get(category_id, []))[:12],
            product_group_contexts=metadata["product_group_contexts"],
            allowed_roles=metadata["allowed_roles"],
            suggested_roles=metadata["suggested_roles"],
            category_kind=metadata["category_kind"],
            metadata_source=metadata["metadata_source"],
            notes=metadata["notes"],
            warnings=metadata["warnings"],
        )
    return sorted(summaries.values(), key=lambda item: (item.name, item.category_id))


def _ocs_anchors_by_category_id(
    distributor_code: str,
) -> dict[str, tuple[OcsAnchorCategory, ...]]:
    if str(distributor_code or "").strip().casefold() != "ocs":
        return {}
    anchors_by_id: dict[str, list[OcsAnchorCategory]] = defaultdict(list)
    for anchor in load_ocs_anchor_categories():
        if anchor.review_status == "rejected":
            continue
        anchors_by_id[anchor.category_id].append(anchor)
    return {category_id: tuple(anchors) for category_id, anchors in anchors_by_id.items()}


def _category_summary_metadata(
    *,
    category_id: str,
    category_name: str,
    path: Sequence[str],
    sample_product_names: Sequence[str],
    anchors: Sequence[OcsAnchorCategory],
) -> dict[str, Any]:
    warnings = ["category_id_is_distributor_fact_not_business_policy"]
    if anchors:
        contexts = _unique([anchor.group for anchor in anchors if anchor.group])
        allowed_roles = _unique(
            [
                role
                for anchor in anchors
                for role in (
                    *_string_list(anchor.allowed_roles),
                    anchor.role,
                )
                if role
            ]
        )
        suggested_roles = _unique([anchor.role for anchor in anchors if anchor.role])
        kinds = _unique(
            [
                str(anchor.category_kind or "").strip()
                for anchor in anchors
                if str(anchor.category_kind or "").strip()
            ]
        )
        notes = _unique(
            [
                _short_catalog_sample(text)
                for anchor in anchors
                for text in (anchor.comment, anchor.notes)
                if text
            ]
        )[:3]
        review_warnings = [
            f"anchor_review_status:{anchor.review_status}"
            for anchor in anchors
            if anchor.review_status != "approved"
        ]
        return {
            "product_group_contexts": contexts or ["unknown"],
            "allowed_roles": _normalize_role_list(allowed_roles),
            "suggested_roles": _normalize_role_list(suggested_roles),
            "category_kind": kinds[0] if len(kinds) == 1 else "mixed",
            "metadata_source": "ocs_anchor_categories",
            "notes": notes,
            "warnings": _unique([*warnings, *review_warnings]),
        }

    inferred = _infer_category_summary_metadata(
        category_id=category_id,
        category_name=category_name,
        path=path,
        sample_product_names=sample_product_names,
    )
    inferred["warnings"] = _unique([*warnings, *inferred["warnings"]])
    return inferred


def _infer_category_summary_metadata(
    *,
    category_id: str,
    category_name: str,
    path: Sequence[str],
    sample_product_names: Sequence[str],
) -> dict[str, Any]:
    haystack = " ".join(
        [
            category_id,
            category_name,
            *path,
            *sample_product_names[:5],
        ]
    ).casefold()
    matches: list[tuple[list[str], list[str], str]] = []
    for needles, contexts, roles, kind in _CATEGORY_METADATA_PATTERNS:
        if _metadata_text_matches_any(haystack, needles):
            matches.append((contexts, roles, kind))
    if not matches:
        return {
            "product_group_contexts": ["unknown"],
            "allowed_roles": [],
            "suggested_roles": [],
            "category_kind": "unknown",
            "metadata_source": "inferred_from_catalog_text",
            "notes": [],
            "warnings": [],
        }
    contexts = _unique([context for row in matches for context in row[0]])
    roles = _normalize_role_list([role for row in matches for role in row[1]])
    kinds = _unique([row[2] for row in matches if row[2]])
    return {
        "product_group_contexts": contexts or ["unknown"],
        "allowed_roles": roles,
        "suggested_roles": roles[:3],
        "category_kind": kinds[0] if len(kinds) == 1 else "mixed",
        "metadata_source": "inferred_from_catalog_text",
        "notes": [],
        "warnings": [],
    }


_CATEGORY_METADATA_PATTERNS: tuple[tuple[tuple[str, ...], list[str], list[str], str], ...] = (
    (("switch", "ethernet switch"), ["network"], [SWITCH_ROLE], "base_device"),
    (("router",), ["network"], [ROUTER_ROLE], "base_device"),
    (("firewall", "ngfw", "utm"), ["network"], [FIREWALL_ROLE], "base_device"),
    (("access point", "wi-fi", "wifi"), ["network"], [ACCESS_POINT_ROLE], "base_device"),
    (
        ("transceiver", "optic", "sfp module", "qsfp module"),
        ["network"],
        [TRANSCEIVER_ROLE],
        "transceiver",
    ),
    (
        ("network adapter", "nic", "ethernet adapter"),
        ["server"],
        [NETWORK_ADAPTER_ROLE],
        "component",
    ),
    (
        ("server platform", "barebone", "chassis", "server system"),
        ["server"],
        [SERVER_PLATFORM_ROLE],
        "base_device",
    ),
    (("cpu", "processor", "xeon", "epyc"), ["server"], [CPU_ROLE], "component"),
    (("ram", "memory", "rdimm", "lrdimm", "dimm"), ["server"], [RAM_ROLE], "component"),
    (
        ("raid", "hba", "storage controller", "tri-mode"),
        ["server"],
        [STORAGE_CONTROLLER_ROLE],
        "component",
    ),
    (
        ("storage system", "storage array", "nas", "san"),
        ["storage"],
        [STORAGE_SYSTEM_ROLE],
        "base_device",
    ),
    (("ssd", "nvme", "solid state"), ["server", "storage"], [DRIVE_ROLE, SSD_ROLE], "drive"),
    (("hdd", "hard drive", "hard disk"), ["server", "storage"], [DRIVE_ROLE, HDD_ROLE], "drive"),
    (("drive", "disk"), ["server", "storage"], [DRIVE_ROLE], "drive"),
    (("power supply", "psu"), ["server"], [POWER_SUPPLY_ROLE], "component"),
    (("dac", "aoc"), ["network", "accessory"], [DAC_CABLE_ROLE, CABLE_ROLE], "cable"),
    (("cable", "c13", "c14"), ["accessory"], [CABLE_ROLE, OTHER_ACCESSORY_ROLE], "cable"),
    (("rail", "rails"), ["accessory"], [RAIL_KIT_ROLE, OTHER_ACCESSORY_ROLE], "accessory"),
    (("fan", "accessory", "option kit"), ["accessory"], [OTHER_ACCESSORY_ROLE], "accessory"),
    (("support", "warranty", "service"), ["support_license"], [SUPPORT_ROLE], "support"),
    (("license", "licence", "subscription"), ["support_license"], [LICENSE_ROLE], "license"),
)


def _normalize_role_list(values: Sequence[Any]) -> list[str]:
    return _unique(
        [
            normalized
            for normalized in (_normalize_plan_role(value) for value in values)
            if normalized
        ]
    )


def _metadata_text_matches_any(text: str, needles: Sequence[str]) -> bool:
    for needle in needles:
        normalized = str(needle or "").strip().casefold()
        if not normalized:
            continue
        if " " in normalized:
            if normalized in text:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text):
            return True
    return False


def _short_catalog_sample(value: str, *, max_chars: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _role_for_product(
    product: DistributorProduct,
    *,
    planned_roles: Sequence[str],
    product_group: str,
) -> str | None:
    roles = [_normalize_role(role) for role in planned_roles if _normalize_role(role)]
    text = _product_text(product).casefold()
    if product_group == SERVER_PRODUCT_GROUP and set(roles).intersection(
        {SSD_ROLE, HDD_ROLE, DRIVE_ROLE}
    ):
        if _text_matches_any(text, _ROLE_KEYWORDS[SSD_ROLE]):
            return SSD_ROLE
        if _text_matches_any(text, _ROLE_KEYWORDS[HDD_ROLE]):
            return HDD_ROLE
        return DRIVE_ROLE
    for role in roles:
        if _text_matches_any(text, _ROLE_KEYWORDS.get(role, (role.replace("_", " "),))):
            return role
    if roles:
        return roles[0]
    return _role_from_product_text(text, product_group=product_group)


def _candidate_row_for_product(
    *,
    product: DistributorProduct,
    role: str,
    category: CategorySummary | None,
    stock_rows: list[DistributorStockPrice],
    available_quantity: int,
    price_value: Decimal,
    price_currency: str,
) -> dict[str, Any]:
    facts = _extract_product_facts(product, role)
    candidate = _ComponentCandidate(
        role=role,
        product=product,
        facts=facts,
        quantity_required=1,
        available_quantity=available_quantity,
        reservable_locations=_reservable_locations(stock_rows),
        price_value=price_value,
        price_currency=price_currency,
        eligibility_status="v2_objective_valid",
        eligibility_warnings=[],
        fit_reasons=["v2 broad candidate; Composer decides compatibility"],
        score=50,
        fit_label="possible",
        fit_reason="Included by v2 broad category universe.",
        fit_tier="possible_fit",
        selection_bucket="v2_full_matrix",
        bucket_priority=50,
    ).to_report_json()
    candidate["category_name"] = category.name if category else None
    candidate["category_kind"] = category.category_kind if category else "unknown"
    candidate["category_allowed_roles"] = category.allowed_roles if category else []
    candidate["category_product_group_contexts"] = (
        category.product_group_contexts if category else []
    )
    candidate["category_metadata_source"] = category.metadata_source if category else ""
    candidate["catalog_path"] = _jsonable(product.catalog_path_json)
    candidate["stock_locations"] = [
        {
            "shipment_city": row.shipment_city,
            "location": row.location,
            "quantity_value": row.quantity_value,
            "can_reserve": row.can_reserve,
        }
        for row in stock_rows
    ]
    candidate["pipeline_version"] = PIPELINE_VERSION
    return candidate


def _v2_normalized_requirements(
    spec: StockSpec,
    *,
    universe_plan: CandidateUniversePlan,
) -> dict[str, Any]:
    requirement_summary = _v2_role_requirement_summary(universe_plan)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "product_group": universe_plan.product_group,
        "primary_product_group": universe_plan.product_group,
        "primary_object": universe_plan.primary_object,
        "original_request_text": spec.source_text,
        "broad_role_hints": universe_plan.broad_role_hints,
        "procurement_intent": universe_plan.procurement_intent,
        "selected_group_reason": universe_plan.selected_group_reason,
        "primary_object_indicators": universe_plan.primary_object_indicators,
        "component_role_indicators": universe_plan.component_role_indicators,
        "embedded_requirements": universe_plan.embedded_requirements,
        "requirement_fulfillment_decision": (
            universe_plan.requirement_fulfillment_decision
        ),
        "roles_sent_to_composer": requirement_summary["roles_sent_to_composer"],
        "broad_reasoning_roles": requirement_summary["broad_reasoning_roles"],
        "hard_purchasable_bom_roles": requirement_summary[
            "hard_purchasable_bom_roles"
        ],
        "hard_purchasable_bom_role_requirements": requirement_summary[
            "hard_purchasable_bom_role_requirements"
        ],
        "purchasable_role_requirements": requirement_summary[
            "purchasable_role_requirements"
        ],
        "primary_object_feature_requirements": requirement_summary[
            "primary_object_feature_requirements"
        ],
        "optional_accessory_engineering_roles": requirement_summary[
            "optional_accessory_engineering_roles"
        ],
        "optional_accessory_engineering_requirements": requirement_summary[
            "optional_accessory_engineering_requirements"
        ],
        "accessory_or_consumable_requirements": requirement_summary[
            "accessory_or_consumable_requirements"
        ],
        "engineering_check_requirements": requirement_summary[
            "engineering_check_requirements"
        ],
        "logistics_or_commercial_constraint_requirements": requirement_summary[
            "logistics_or_commercial_constraint_requirements"
        ],
        "classified_requirements": requirement_summary["classified_requirements"],
        "required_roles": requirement_summary["hard_purchasable_bom_roles"],
        "required_capabilities": [],
        "optional_capabilities": [],
        "composer_first": True,
    }


def _v2_role_plan(universe_plan: CandidateUniversePlan) -> dict[str, Any]:
    requirement_summary = _v2_role_requirement_summary(universe_plan)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "product_group": universe_plan.product_group,
        "primary_product_group": universe_plan.product_group,
        "primary_object": universe_plan.primary_object,
        "candidate_universe_planner_mode": (
            universe_plan.candidate_universe_planner_mode
        ),
        "procurement_intent": universe_plan.procurement_intent,
        "selected_product_group_reason": universe_plan.selected_group_reason,
        "competing_product_groups": universe_plan.competing_product_groups,
        "primary_object_indicators": universe_plan.primary_object_indicators,
        "component_role_indicators": universe_plan.component_role_indicators,
        "embedded_requirements": universe_plan.embedded_requirements,
        "requirement_fulfillment_decision": (
            universe_plan.requirement_fulfillment_decision
        ),
        "accessory_indicators": universe_plan.accessory_indicators,
        "service_support_indicators": universe_plan.service_support_indicators,
        "logistics_commercial_constraints": (
            universe_plan.logistics_commercial_constraints
        ),
        "category_plan_entries": universe_plan.category_plan_entries,
        "excluded_category_groups": universe_plan.excluded_category_groups,
        "planner_repair_attempted": universe_plan.planner_repair_attempted,
        "planner_repair_success": universe_plan.planner_repair_success,
        "planner_suspicion_reasons": universe_plan.planner_suspicion_reasons,
        "broad_role_hints": universe_plan.broad_role_hints,
        "broad_category_plan": universe_plan.broad_category_plan,
        "roles_sent_to_composer": requirement_summary["roles_sent_to_composer"],
        "broad_reasoning_roles": requirement_summary["broad_reasoning_roles"],
        "hard_purchasable_bom_roles": requirement_summary[
            "hard_purchasable_bom_roles"
        ],
        "hard_purchasable_bom_role_requirements": requirement_summary[
            "hard_purchasable_bom_role_requirements"
        ],
        "purchasable_role_requirements": requirement_summary[
            "purchasable_role_requirements"
        ],
        "primary_object_feature_requirements": requirement_summary[
            "primary_object_feature_requirements"
        ],
        "optional_accessory_engineering_roles": requirement_summary[
            "optional_accessory_engineering_roles"
        ],
        "optional_accessory_engineering_requirements": requirement_summary[
            "optional_accessory_engineering_requirements"
        ],
        "accessory_or_consumable_requirements": requirement_summary[
            "accessory_or_consumable_requirements"
        ],
        "engineering_check_requirements": requirement_summary[
            "engineering_check_requirements"
        ],
        "logistics_or_commercial_constraint_requirements": requirement_summary[
            "logistics_or_commercial_constraint_requirements"
        ],
        "classified_requirements": requirement_summary["classified_requirements"],
        "required_roles": requirement_summary["hard_purchasable_bom_roles"],
        "required_capabilities": [],
        "optional_capabilities": [],
        "semantic_planner_source": CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        "semantic_planner_used": universe_plan.candidate_universe_planner_mode
        != CANDIDATE_UNIVERSE_PLANNER_FALLBACK_MODE,
        "semantic_planner_confidence": universe_plan.confidence,
    }


def _v2_report_fields(
    *,
    package: Mapping[str, Any],
    matrix: Mapping[str, Any],
    universe_plan: CandidateUniversePlan,
    package_diagnostics: Mapping[str, Any],
    attempt_decision: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "product_group": universe_plan.product_group,
        "primary_product_group": universe_plan.product_group,
        "primary_object": universe_plan.primary_object,
        "candidate_universe_planner_mode": (
            universe_plan.candidate_universe_planner_mode
        ),
        "semantic_planner_source": CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        "semantic_planner_used": universe_plan.candidate_universe_planner_mode
        != CANDIDATE_UNIVERSE_PLANNER_FALLBACK_MODE,
        "semantic_planner_confidence": universe_plan.confidence,
        "procurement_intent": universe_plan.procurement_intent,
        "selected_group_reason": universe_plan.selected_group_reason,
        "selected_product_group_reason": universe_plan.selected_group_reason,
        "competing_product_groups": universe_plan.competing_product_groups,
        "primary_object_indicators": universe_plan.primary_object_indicators,
        "component_role_indicators": universe_plan.component_role_indicators,
        "embedded_requirements": universe_plan.embedded_requirements,
        "requirement_fulfillment_decision": _mapping_rows(
            package_diagnostics.get("requirement_fulfillment_decision")
        )
        or universe_plan.requirement_fulfillment_decision,
        "role_fulfillment_diagnostics": _mapping_rows(
            package_diagnostics.get("role_fulfillment_diagnostics")
        ),
        "accessory_indicators": universe_plan.accessory_indicators,
        "service_support_indicators": universe_plan.service_support_indicators,
        "logistics_commercial_constraints": (
            universe_plan.logistics_commercial_constraints
        ),
        "excluded_category_groups": universe_plan.excluded_category_groups,
        "planner_repair_attempted": universe_plan.planner_repair_attempted,
        "planner_repair_success": universe_plan.planner_repair_success,
        "planner_suspicion_reasons": universe_plan.planner_suspicion_reasons,
        "category_planner_source": CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        "category_plan_source": CANDIDATE_UNIVERSE_PLANNER_SOURCE,
        "candidate_universe_planner_output": universe_plan.to_report_json(),
        "candidate_universe_category_plan": universe_plan.broad_category_plan,
        "category_plan": universe_plan.broad_category_plan,
        "category_plan_entries": universe_plan.category_plan_entries,
        "stage_a_broad_roles": _string_list(
            package_diagnostics.get("stage_a_broad_roles")
        ),
        "semantic_matrix_blueprint_roles": _string_list(
            package_diagnostics.get("semantic_matrix_blueprint_roles")
        ),
        "effective_matrix_roles_before_category_planner": _string_list(
            package_diagnostics.get("effective_matrix_roles_before_category_planner")
        ),
        "category_planner_input_roles": _string_list(
            package_diagnostics.get("category_planner_input_roles")
        ),
        "category_planner_output_roles": _string_list(
            package_diagnostics.get("category_planner_output_roles")
        ),
        "validated_category_plan_roles": _string_list(
            package_diagnostics.get("validated_category_plan_roles")
        ),
        "materialized_matrix_roles": _string_list(
            package_diagnostics.get("materialized_matrix_roles")
        ),
        "composer_package_roles": _string_list(
            package_diagnostics.get("composer_package_roles")
        ),
        "roles_dropped_after_category_planner": _string_list(
            package_diagnostics.get("roles_dropped_after_category_planner")
        ),
        "roles_dropped_during_materialization": _string_list(
            package_diagnostics.get("roles_dropped_during_materialization")
        ),
        "roles_dropped_reason_by_role": _safe_mapping(
            package_diagnostics.get("roles_dropped_reason_by_role")
        ),
        "role_source_by_role": _safe_mapping(
            package_diagnostics.get("role_source_by_role")
        ),
        "role_lifecycle_trace": _mapping_rows(
            package_diagnostics.get("role_lifecycle_trace")
        ),
        "full_candidate_matrix_count_by_role": _safe_mapping(
            matrix.get("full_candidate_matrix_count_by_role")
        ),
        "full_candidate_matrix_count_by_category": _safe_mapping(
            matrix.get("full_candidate_matrix_count_by_category")
        ),
        "matrix_materialized_count_by_role": _safe_mapping(
            matrix.get("matrix_materialized_count_by_role")
        ),
        "dropped_before_composer_count_by_reason": _safe_mapping(
            package_diagnostics.get("dropped_before_composer_count_by_reason")
        ),
        "matrix_source_diagnostics": _safe_mapping(matrix.get("matrix_source_diagnostics")),
        "composer_package_full_matrix_used": bool(
            package_diagnostics.get("composer_package_full_matrix_used")
        ),
        "composer_context_size": _safe_mapping(
            package_diagnostics.get("composer_context_size")
        ),
        "composer_attempt_decision": dict(attempt_decision),
        "composer_requirement_analysis": _safe_mapping(
            package_diagnostics.get("composer_requirement_analysis")
        ),
        "composer_fulfillment_decisions": _mapping_rows(
            package_diagnostics.get("composer_fulfillment_decisions")
        ),
        "composer_assumptions": _string_list(
            package_diagnostics.get("composer_assumptions")
        ),
        "composer_engineer_checks": _string_list(
            package_diagnostics.get("composer_engineer_checks")
        ),
        "composer_hard_mismatch_risks": _mapping_rows(
            package_diagnostics.get("composer_hard_mismatch_risks")
        ),
        "composer_unverified_requirements": _mapping_rows(
            package_diagnostics.get("composer_unverified_requirements")
        ),
        "composer_considered_candidate_count_by_role": _safe_mapping(
            package_diagnostics.get("composer_considered_candidate_count_by_role")
        ),
        "composer_chosen_candidate_ids": _string_list(
            package_diagnostics.get("composer_chosen_candidate_ids")
        ),
        "validation_hard_mismatches": _mapping_rows(
            package_diagnostics.get("validation_hard_mismatches")
        ),
        "validation_unverified_requirements": _mapping_rows(
            package_diagnostics.get("validation_unverified_requirements")
        ),
        "final_status_source": package_diagnostics.get("final_status_source"),
        "diagnostics": {
            "pipeline_version": PIPELINE_VERSION,
            "v2_bypassed_pre_composer_gates": [
                "requirement_classifier_status",
                "requirement_source_coverage_percent",
                "unclassified_source_fragments",
                "role_lifecycle",
                "unmapped_feature",
                "category_repair",
                "no_recommendation_coverage",
            ],
        },
        **dict(package_diagnostics),
    }


def _matrix_report_fields_from_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: diagnostics[key]
        for key in (
            "composer_package_full_matrix_used",
            "composer_context_size",
            "composer_attempt_decision",
            "composer_mode",
            "expected_composer_mode",
            "llm_call_count",
            "llm_call_stages",
            "llm_call_budget_exceeded",
            "max_llm_calls_per_match",
            "validation_hard_mismatches",
            "validation_unverified_requirements",
            "final_status_source",
            "category_catalog_total",
            "category_catalog_sent_to_ai_count",
            "category_catalog_truncated",
            "selected_category_count",
            "rejected_category_count",
            "rejected_category_reasons",
            "matrix_materialized_count_by_role",
            "dropped_before_composer_count_by_reason",
            "package_strategy_decision",
            "requirement_graph",
            "candidate_universe_ledger",
            "composer_execution_state",
            "coverage_evidence",
            "validation_ledger",
            "execution_ledger",
            "safe_no_recommendation",
            "stage_a_broad_roles",
            "semantic_matrix_blueprint_roles",
            "effective_matrix_roles_before_category_planner",
            "category_planner_input_roles",
            "category_planner_output_roles",
            "validated_category_plan_roles",
            "materialized_matrix_roles",
            "composer_package_roles",
            "roles_dropped_after_category_planner",
            "roles_dropped_during_materialization",
            "roles_dropped_reason_by_role",
            "role_source_by_role",
            "role_lifecycle_trace",
            "role_fulfillment_diagnostics",
            "roles_sent_to_composer",
            "broad_reasoning_roles",
            "hard_purchasable_bom_roles",
            "hard_purchasable_bom_role_requirements",
            "purchasable_role_requirements",
            "primary_object_feature_requirements",
            "optional_accessory_engineering_roles",
            "optional_accessory_engineering_requirements",
            "accessory_or_consumable_requirements",
            "engineering_check_requirements",
            "logistics_or_commercial_constraint_requirements",
            "classified_requirements",
            "requirement_fulfillment_decision",
            *_V2_COMPACT_PACKAGE_DIAGNOSTIC_KEYS,
        )
        if key in diagnostics
    }


def _category_catalog_summary(categories: Sequence[CategorySummary]) -> dict[str, Any]:
    prompt_categories = _candidate_universe_prompt_categories(categories)
    return {
        "category_count": len(categories),
        "category_catalog_total": len(categories),
        "category_catalog_sent_to_ai_count": len(prompt_categories),
        "category_catalog_truncated": False,
        "categories_with_products": sum(1 for category in categories if category.product_count),
        "categories_with_stock": sum(1 for category in categories if category.stocked_count),
        "categories_with_price": sum(1 for category in categories if category.priced_count),
        "categories_with_metadata": sum(
            1
            for category in categories
            if category.allowed_roles or category.product_group_contexts != ["unknown"]
        ),
        "source": "distributor_categories_and_products",
    }


def _v2_role_coverage_summary(matrix: Mapping[str, Any]) -> dict[str, Any]:
    counts = _safe_mapping(matrix.get("full_candidate_matrix_count_by_role")) or (
        _count_by_role_from_matrix(matrix)
    )
    return {
        role: {
            "required": False,
            "missing": False,
            "candidate_count": count,
            "after_category_count": count,
            "after_eligibility_count": count,
            "coverage_source": "v2_full_matrix",
        }
        for role, count in counts.items()
    }


def _v2_component_coverage_summary(
    matrix: Mapping[str, Any],
    *,
    universe_plan: CandidateUniversePlan,
) -> dict[str, Any]:
    counts = _safe_mapping(matrix.get("full_candidate_matrix_count_by_role"))
    return {
        "pipeline_version": PIPELINE_VERSION,
        "role_hints": universe_plan.broad_role_hints,
        "count_by_role": counts,
        "coverage_policy": "all_stocked_priced_candidates_from_selected_categories",
    }


def _roles_by_category_id(category_plan: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    roles_by_category: dict[str, list[str]] = defaultdict(list)
    for role, category_ids in category_plan.items():
        normalized_role = _normalize_role(role)
        if not normalized_role:
            continue
        for category_id in category_ids:
            text = str(category_id or "").strip()
            if text and normalized_role not in roles_by_category[text]:
                roles_by_category[text].append(normalized_role)
    return dict(roles_by_category)


def _normalize_role(role: Any) -> str | None:
    text = str(role or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "platform": SERVER_PLATFORM_ROLE,
        "storage": DRIVE_ROLE,
        "disk": DRIVE_ROLE,
        "disks": DRIVE_ROLE,
        "drives": DRIVE_ROLE,
        "power_cable": CABLE_ROLE,
        "power_cord": CABLE_ROLE,
        "network": SWITCH_ROLE,
    }
    text = aliases.get(text, text)
    return text if text in _ROLE_MATRIX_KEY else None


def _role_from_product_text(text: str, *, product_group: str) -> str | None:
    for role in _baseline_roles_for_group(product_group):
        if _text_matches_any(text, _ROLE_KEYWORDS.get(role, (role.replace("_", " "),))):
            return role
    return OTHER_ACCESSORY_ROLE if product_group == SERVER_PRODUCT_GROUP else None


def _baseline_roles_for_group(product_group: str) -> tuple[str, ...]:
    if product_group == NETWORK_PRODUCT_GROUP:
        return _NETWORK_BASELINE_ROLES
    if product_group == STORAGE_PRODUCT_GROUP:
        return _STORAGE_BASELINE_ROLES
    if product_group == SERVER_PRODUCT_GROUP:
        return _SERVER_BASELINE_ROLES
    return ()


def _primary_object_for_group(product_group: str) -> str:
    if product_group == NETWORK_PRODUCT_GROUP:
        return "network_device"
    if product_group == STORAGE_PRODUCT_GROUP:
        return STORAGE_SYSTEM_ROLE
    if product_group == SERVER_PRODUCT_GROUP:
        return "server"
    return "unknown"


def _product_text(product: DistributorProduct) -> str:
    return " ".join(
        part
        for part in (
            product.producer,
            product.part_number,
            product.item_name,
            product.item_name_rus,
            product.product_name,
            product.product_description,
            product.product_notes,
        )
        if part
    )


def _text_matches_any(text: str, needles: Sequence[str]) -> bool:
    lowered = text.casefold()
    return any(needle.casefold() in lowered for needle in needles)


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _decimal_sort_value(row.get("price_value")),
        -(int(row.get("available_quantity") or 0)),
        str(row.get("producer") or ""),
        str(row.get("part_number") or ""),
        str(row.get("item_id") or ""),
    )


def _decimal_sort_value(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("999999999999")


def _count_by_role_from_matrix(matrix: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for role, key in _ROLE_MATRIX_KEY.items():
        rows = matrix.get(key)
        if isinstance(rows, list) and rows:
            counts[role] = len(rows)
    return counts


def _count_by_category_from_rows(
    role_rows: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rows in role_rows:
        for row in rows:
            category_id = str(row.get("category_id") or "").strip()
            if category_id:
                counts[category_id] += 1
    return dict(sorted(counts.items()))


def _total_candidate_rows(matrix: Mapping[str, Any]) -> int:
    return sum(_count_by_role_from_matrix(matrix).values())


def _package_uses_full_matrix(
    package: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> bool:
    if package.get("package_skipped_reason"):
        return False
    if _safe_mapping(package.get("package_budget")).get("over_budget") is True:
        return False
    package_counts = _safe_mapping(package.get("composer_package_candidate_count_by_role"))
    source_counts = _safe_mapping(matrix.get("full_candidate_matrix_count_by_role"))
    return {
        str(role): int(count)
        for role, count in package_counts.items()
        if int(count or 0) > 0
    } == {
        str(role): int(count)
        for role, count in source_counts.items()
        if int(count or 0) > 0
    }


def _package_context_size(package: Mapping[str, Any]) -> dict[str, Any]:
    chars = len(json.dumps(package, ensure_ascii=False, sort_keys=True, default=str))
    budget = _safe_mapping(package.get("package_budget"))
    return {
        "chars": chars,
        "tokens_estimate": max(1, chars // 4),
        "max_chars": budget.get("max_chars"),
        "over_budget": budget.get("over_budget"),
    }


def _provider_configured(settings: LlmSettings, *, llm_client: LlmClient | None) -> bool:
    if llm_client is not None:
        return True
    provider = settings.llm_provider.strip().lower()
    return (
        provider in {"openai", "openai-compatible", "openai_compatible"}
        and bool(settings.llm_base_url.strip())
        and bool(settings.llm_api_key.strip())
        and bool(settings.llm_model.strip())
    )


def _merge_attempt_decision(
    v2_decision: Mapping[str, Any],
    composer_decision: Mapping[str, Any],
) -> dict[str, Any]:
    if not composer_decision:
        return dict(v2_decision)
    blocked_by = _unique(
        [
            *_string_list(v2_decision.get("blocked_by")),
            *_string_list(composer_decision.get("blocked_by")),
        ]
    )
    return {
        **dict(v2_decision),
        **dict(composer_decision),
        "pipeline_version": PIPELINE_VERSION,
        "should_attempt": bool(v2_decision.get("should_attempt"))
        and not blocked_by,
        "blocked_by": blocked_by,
        "v2_allowed_blockers": v2_decision.get("allowed_blockers", []),
    }


def _composer_candidate_total_from_package(package: Mapping[str, Any]) -> int:
    value = package.get("composer_package_candidate_total")
    if isinstance(value, int):
        return value
    counts = _safe_mapping(package.get("composer_package_candidate_count_by_role"))
    return sum(int(count or 0) for count in counts.values())


def _zero_dropped_counts(package: Mapping[str, Any]) -> dict[str, int]:
    counts = _safe_mapping(package.get("composer_package_candidate_count_by_role"))
    return {str(role): 0 for role in counts}


def _dropped_before_composer_count_by_reason(
    matrix: Mapping[str, Any],
) -> dict[str, int]:
    diagnostics = _safe_mapping(matrix.get("matrix_source_diagnostics"))
    reasons = (
        "wrong_distributor",
        "category_not_selected",
        "no_stock",
        "no_price",
        "broken_row",
        "objectively_wrong_role",
    )
    return {
        reason: int(diagnostics.get(reason) or 0)
        for reason in reasons
        if int(diagnostics.get(reason) or 0) > 0
    }


def _status_from_v2_outcome(
    outcome: LlmConfiguratorOutcome,
    matrix: Mapping[str, Any],
) -> str:
    if outcome.primary_recommendation_status == "valid":
        return STATUS_STOCK_MATCHED
    if _total_candidate_rows(matrix) > 0:
        return STATUS_PARTIAL_STOCK_MATCHED
    return STATUS_NO_STOCK_MATCH


def _v2_missing_requirements(outcome: LlmConfiguratorOutcome) -> list[str]:
    reason = _safe_mapping(outcome.no_recommendation_reason)
    missing = _string_list(reason.get("missing_roles"))
    missing.extend(
        str(row.get("source_text") or row.get("capability_id") or row.get("role") or "")
        for row in _mapping_rows(reason.get("missing_required_capabilities"))
    )
    return _unique([item for item in missing if item])


def _v2_risk_flags(outcome: LlmConfiguratorOutcome) -> list[str]:
    flags = _string_list(outcome.internal_warnings)
    if outcome.fallback_reason:
        flags.append(str(outcome.fallback_reason))
    if outcome.final_status_source == COMPOSER_REJECTED_BY_VALIDATION:
        flags.append(COMPOSER_REJECTED_BY_VALIDATION)
    return _unique(flags)


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _unique_mapping_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        clean = {key: value for key, value in dict(row).items() if value not in ("", [], {})}
        key = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item or "").strip()]
    return [str(value)]


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
