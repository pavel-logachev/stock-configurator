from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha1
from math import ceil
from typing import Any

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.ocs_server_categories import (
    server_category_ids_for_role,
    server_category_role,
)
from app.core.config import (
    LlmSettings,
    WebEvidenceSettings,
    get_llm_settings,
    get_web_evidence_settings,
)
from app.core.database import get_session_factory
from app.db.models import DistributorCategory, DistributorProduct, DistributorStockPrice
from app.distributors.ocs.content_enrichment import enrich_matrix_with_ocs_content
from app.evidence.web_evidence import (
    EvidenceSearchCache,
    WebSearchProvider,
    safe_evidence_diagnostics,
)
from app.llm.base import LlmClient, LlmError
from app.llm.configuration_composer import (
    HIGH_QUALITY_BROAD_PACKAGE_UNDER_LIMIT_REASON,
    INCOMPLETE_MATRIX_EXPOSURE_REASON,
    PROVIDER_CONTEXT_LIMIT_ERROR_TYPE,
    PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON,
    SKIPPED_FULL_BROAD_PACKAGE_UNDER_HIGH_QUALITY_LIMIT_REASON,
    LlmConfiguratorOutcome,
    build_llm_configurator_package,
    compose_llm_configurations,
)
from app.llm.matrix_distiller import (
    MatrixDistillerError,
    MatrixDistillerTimeoutError,
    distill_component_candidate_matrix,
)
from app.llm.openai_compatible import OpenAICompatibleLlmClient
from app.llm.stock_spec_extractor import extract_stock_spec_from_text
from app.matching.candidate_inclusion_policy import (
    POLICY_NAME as CANDIDATE_INCLUSION_POLICY_NAME,
)
from app.matching.candidate_inclusion_policy import (
    BroadPreLlmDecision,
    broad_pre_llm_for_ai_reasoning,
    objective_role_reject_reason,
)
from app.matching.spec_schema import StockSpec, StockSpecExtractionResult, StockSpecItem
from app.planning import role_lifecycle
from app.planning.category_planner import (
    CategoryPlanResult,
    build_compact_category_catalog,
    plan_distributor_categories,
)
from app.planning.generic_constraints import constraints_by_role_from_role_plan
from app.planning.network_facts import (
    extract_network_facts,
    network_adapter_facts_satisfy_requirement,
    network_facts_satisfy_requirement,
    network_requirement_from_sources,
    required_network_adapter_quantity,
)
from app.planning.power_facts import platform_power_bundle_satisfies
from app.planning.requirement_planner import (
    SEMANTIC_COMPLEX_FALLBACK_REASON,
    SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON,
    SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT,
)
from app.planning.role_planner import plan_semantic_matrix_roles
from app.policies.product_group_policy import get_product_group_profile

READY_SERVER_ROLE = "ready_server"
BUILD_CANDIDATE_TYPE = "build_from_parts"
READY_SERVER_CANDIDATE_TYPE = "ready_server"
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
SWITCH_ROLE = "switch"
ROUTER_ROLE = "router"
FIREWALL_ROLE = "firewall"
ACCESS_POINT_ROLE = "access_point"
DAC_CABLE_ROLE = "dac_cable"
STACKING_MODULE_ROLE = "stacking_module"
NETWORK_PRODUCT_GROUP = "network"
SERVER_PRODUCT_GROUP = "server"
STORAGE_PRODUCT_GROUP = "storage"
STORAGE_SYSTEM_ROLE = "storage_system"
STORAGE_ARRAY_CONTROLLER_ROLE = "controller"
CONTROLLER_MODULE_ROLE = "controller_module"
DISK_SHELF_ROLE = "disk_shelf"
DRIVE_ROLE = "drive"
CACHE_ROLE = "cache"
HOST_PORT_ROLE = "host_port"
PROTOCOL_MODULE_ROLE = "protocol_module"
READY_SERVER_CATEGORY_IDS = server_category_ids_for_role(READY_SERVER_ROLE)
SERVER_CATEGORY_ID = READY_SERVER_CATEGORY_IDS[0]
STATUS_STOCK_MATCHED = "stock_matched"
STATUS_PARTIAL_STOCK_MATCHED = "partial_stock_matched"
STATUS_NO_STOCK_MATCH = "no_stock_match"
UNKNOWN_FACT = "unknown"
MAX_BUILD_CANDIDATES = 10
MAX_API_BUILD_CANDIDATES = 5
MAX_INTERNAL_BUILD_CANDIDATES = 50
MAX_PLATFORM_CANDIDATES = 20
MAX_COMPONENT_CANDIDATES_PER_ROLE = 10
DEFAULT_MATRIX_CANDIDATES_PER_ROLE = 100
MAX_MATRIX_CANDIDATES_PER_ROLE = 150
OPTIMIZATION_MODE_COST_MINIMAL_FIT = "cost_minimal_fit"
SEMANTIC_PLANNER_COMPLEX_FALLBACK_REASON = SEMANTIC_COMPLEX_FALLBACK_REASON
SEMANTIC_PLANNER_TIMEOUT_REASON = SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
SEMANTIC_PLANNER_TIMEOUT_SOURCE = SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT
SEMANTIC_PLANNER_SOURCE_PLANNER_UNAVAILABLE = "planner_unavailable"
SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE = (
    "Не удалось безопасно разобрать сложный запрос без AI semantic planner. "
    "Повторите позже или проверьте настройки LLM."
)
SEMANTIC_PLANNER_SUPPORTED_PROVIDERS = {"openai", "openai-compatible", "openai_compatible"}
FULL_MATRIX_FAILED_BUT_PACKAGE_UNDER_BUDGET_REASON = (
    "full_matrix_evaluation_failed_but_package_under_budget"
)
FULL_MATRIX_UNAVAILABLE_BUT_PACKAGE_UNDER_BUDGET_REASON = (
    "full_matrix_evaluation_unavailable_but_package_under_budget"
)
FULL_MATRIX_OVER_BUDGET_FAILURE_SKIP_REASON = (
    "package_over_budget_after_full_matrix_failure"
)
FULL_MATRIX_TIMEOUT_BUT_PACKAGE_UNDER_BUDGET_REASON = (
    "full_matrix_evaluation_timeout_but_package_under_budget"
)
FULL_MATRIX_TIMEOUT_PACKAGE_OVER_BUDGET_REASON = (
    "full_matrix_evaluation_timeout_package_over_budget"
)
SUPPORTED_OPTIMIZATION_MODES = {
    OPTIMIZATION_MODE_COST_MINIMAL_FIT,
    "balanced",
    "performance",
}
FIT_EXACT_OR_CLOSE = "exact_or_close_fit"
FIT_ACCEPTABLE_OVERFIT = "acceptable_overfit"
FIT_EXCESSIVE_OVERFIT = "excessive_overfit"
FIT_UNKNOWN = "unknown_fit"
FIT_TIER_STRONG = "strong_fit"
FIT_TIER_POSSIBLE = "possible_fit"
FIT_TIER_FALLBACK_UNKNOWN = "fallback_unknown"
FIT_TIER_EXPLICIT_MISMATCH = "explicit_mismatch"
FIT_TIER_WRONG_ROLE = "wrong_role"
ACTIVE_FIT_TIERS = {FIT_TIER_STRONG, FIT_TIER_POSSIBLE, FIT_TIER_FALLBACK_UNKNOWN}
FIT_TIER_RANK = {
    FIT_TIER_STRONG: 0,
    FIT_TIER_POSSIBLE: 1,
    FIT_TIER_FALLBACK_UNKNOWN: 2,
    FIT_TIER_EXPLICIT_MISMATCH: 98,
    FIT_TIER_WRONG_ROLE: 99,
}
RIGHT_SIZE_COMPONENT_ROLES = {CPU_ROLE, RAM_ROLE, SSD_ROLE, HDD_ROLE}
MATRIX_ROLE_ORDER = [
    READY_SERVER_ROLE,
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
MATRIX_PROMPT_ROLE_BY_INTERNAL_ROLE = {
    READY_SERVER_ROLE: READY_SERVER_ROLE,
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
GENERIC_COMPONENT_ROLES = (
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
)
CPU_CORE_BUCKETS = (16, 20, 24, 32, 48, 64)
RAM_MODULE_BUCKETS_GB = (32, 64, 128)
STORAGE_CAPACITY_BUCKETS_TB = (3.84, 7.68, 15.36)
BUCKET_SAMPLE_SIZE = 8

_OEM_VENDOR_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("HPE", (r"\bhpe\b", r"hewlett[\s-]*packard\s+enterprise", r"\bhp\s+enterprise\b")),
    ("Dell", (r"\bdell\b", r"\bpoweredge\b")),
    ("Lenovo", (r"\blenovo\b", r"\bthinksystem\b")),
    ("ASUS", (r"\basus\b",)),
    ("Supermicro", (r"\bsuper\s*micro\b", r"\bsupermicro\b")),
    ("Fujitsu", (r"\bfujitsu\b",)),
    ("Cisco", (r"\bcisco\b", r"\bucs\b")),
)
_CPU_VENDOR_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Intel", (r"\bintel\b", r"\bxeon\b")),
    ("AMD", (r"\bamd\b", r"\bepyc\b")),
)


@dataclass(frozen=True)
class MatchCandidateResult:
    distributor_code: str
    item_id: str
    product_key: str | None
    part_number: str | None
    producer: str | None
    category_id: str | None
    item_name: str | None
    confidence_score: int
    price_value: Decimal | None
    price_currency: str | None
    available_quantity: int | None
    reservable_locations: int
    matched_requirements: list[str]
    missing_requirements: list[str]
    risk_flags: list[str]
    raw: dict[str, Any]
    candidate_type: str = READY_SERVER_CANDIDATE_TYPE
    components: list[dict[str, Any]] = field(default_factory=list)
    total_price_value: Decimal | None = None
    total_price_currency: str | None = None
    missing_components: list[str] = field(default_factory=list)
    compatibility_warnings: list[str] = field(default_factory=list)
    engineer_review_required: bool = True
    completeness_status: str = "complete"
    completeness_label: str = "Предварительная сборка"
    included_component_roles: list[str] = field(default_factory=list)
    missing_component_roles: list[str] = field(default_factory=list)
    excluded_from_total_roles: list[str] = field(default_factory=list)
    cpu_per_server: int | None = None
    total_cpu_required: int | None = None
    total_price_note: str | None = None
    platform: dict[str, Any] = field(default_factory=dict)
    score: int | None = None
    rank_reason: list[str] = field(default_factory=list)
    candidate_id: str | None = None

    @property
    def is_full_match(self) -> bool:
        return (
            self.candidate_type == READY_SERVER_CANDIDATE_TYPE
            and self.confidence_score >= 70
            and not self.missing_requirements
        )

    def to_report_json(self) -> dict[str, Any]:
        return {
            "candidate_type": self.candidate_type,
            "distributor_code": self.distributor_code,
            "item_id": self.item_id,
            "product_key": self.product_key,
            "part_number": self.part_number,
            "producer": self.producer,
            "category_id": self.category_id,
            "item_name": self.item_name,
            "confidence_score": self.confidence_score,
            "price_value": _jsonable(self.price_value),
            "price_currency": self.price_currency,
            "available_quantity": self.available_quantity,
            "reservable_locations": self.reservable_locations,
            "matched_requirements": self.matched_requirements,
            "missing_requirements": self.missing_requirements,
            "risk_flags": self.risk_flags,
            "components": _jsonable(self.components),
            "total_price_value": _jsonable(self.total_price_value),
            "total_price_currency": self.total_price_currency,
            "missing_components": self.missing_components,
            "compatibility_warnings": self.compatibility_warnings,
            "engineer_review_required": self.engineer_review_required,
            "completeness_status": self.completeness_status,
            "completeness_label": self.completeness_label,
            "included_component_roles": self.included_component_roles,
            "missing_component_roles": self.missing_component_roles,
            "excluded_from_total_roles": self.excluded_from_total_roles,
            "cpu_per_server": self.cpu_per_server,
            "total_cpu_required": self.total_cpu_required,
            "total_price_note": self.total_price_note,
            "platform": _jsonable(self.platform),
            "score": self.score if self.score is not None else self.confidence_score,
            "rank_reason": self.rank_reason,
            "candidate_id": self.candidate_id,
            "optimization_mode": self.raw.get("optimization_mode"),
            "requirement_fit": self.raw.get("requirement_fit"),
            "right_size_note": self.raw.get("right_size_note"),
            "cpu_over_requirement": self.raw.get("cpu_over_requirement"),
            "storage_over_requirement": self.raw.get("storage_over_requirement"),
            "ram_overage_gb": self.raw.get("ram_overage_gb"),
            "overfit_reason": self.raw.get("overfit_reason"),
            "raw": _jsonable(self.raw),
        }


@dataclass(frozen=True)
class _ProductFacts:
    normalized_vendor: str = UNKNOWN_FACT
    is_vendor_option_kit: bool = False
    option_kit_vendor: str = UNKNOWN_FACT
    cpu_brand: str = UNKNOWN_FACT
    cpu_family: str = UNKNOWN_FACT
    cpu_generation: str = UNKNOWN_FACT
    cpu_socket: str = UNKNOWN_FACT
    cpu_cores: int | None = None
    ram_capacity_gb: int | None = None
    ram_type: str = UNKNOWN_FACT
    storage_capacity: str = UNKNOWN_FACT
    storage_capacity_tb: float | None = None
    storage_interface: str = UNKNOWN_FACT
    nvme_support: bool | None = None
    raw_capacity_tb: float | None = None
    usable_capacity_tb: float | None = None
    redundancy_level: str = UNKNOWN_FACT
    controller_count: int | None = None
    drive_count: int | None = None
    drive_capacity_tb: float | None = None
    drive_type: str = UNKNOWN_FACT
    drive_interface: str = UNKNOWN_FACT
    host_protocol: str = UNKNOWN_FACT
    host_port_count: int | None = None
    host_port_speed: str = UNKNOWN_FACT
    host_port_speed_gbps: int | None = None
    host_port_media: str = UNKNOWN_FACT
    warranty_months: int | None = None
    network_ports_count: int | None = None
    network_speed: str = UNKNOWN_FACT
    network_speed_gbps: int | None = None
    network_media: str = UNKNOWN_FACT
    network_interface: str = UNKNOWN_FACT
    port_count: int | None = None
    port_speed: str = UNKNOWN_FACT
    port_speed_gbps: float | None = None
    port_media: str = UNKNOWN_FACT
    uplink_count: int | None = None
    uplink_speed: str = UNKNOWN_FACT
    uplink_speed_gbps: float | None = None
    uplink_media: str = UNKNOWN_FACT
    poe_supported: bool | None = None
    poe_budget_w: int | None = None
    poe_standard: str = UNKNOWN_FACT
    l2_supported: bool | None = None
    l3_supported: bool | None = None
    stacking_supported: bool | None = None
    managed_status: str = UNKNOWN_FACT
    airflow: str = UNKNOWN_FACT
    redundant_psu: bool | None = None
    transceiver_form_factor: str = UNKNOWN_FACT
    form_factor_hints: list[str] = field(default_factory=list)

    def to_report_json(self) -> dict[str, Any]:
        return {
            "normalized_vendor": self.normalized_vendor,
            "is_vendor_option_kit": self.is_vendor_option_kit,
            "option_kit_vendor": self.option_kit_vendor,
            "cpu_brand": self.cpu_brand,
            "cpu_family": self.cpu_family,
            "cpu_generation": self.cpu_generation,
            "cpu_socket": self.cpu_socket,
            "cpu_cores": self.cpu_cores,
            "ram_capacity_gb": self.ram_capacity_gb,
            "ram_type": self.ram_type,
            "storage_capacity": self.storage_capacity,
            "storage_capacity_tb": self.storage_capacity_tb,
            "storage_interface": self.storage_interface,
            "nvme_support": self.nvme_support,
            "raw_capacity_tb": self.raw_capacity_tb,
            "usable_capacity_tb": self.usable_capacity_tb,
            "redundancy_level": self.redundancy_level,
            "controller_count": self.controller_count,
            "drive_count": self.drive_count,
            "drive_capacity_tb": self.drive_capacity_tb,
            "drive_type": self.drive_type,
            "drive_interface": self.drive_interface,
            "host_protocol": self.host_protocol,
            "host_port_count": self.host_port_count,
            "host_port_speed": self.host_port_speed,
            "host_port_speed_gbps": self.host_port_speed_gbps,
            "host_port_media": self.host_port_media,
            "warranty_months": self.warranty_months,
            "network_ports_count": self.network_ports_count,
            "network_speed": self.network_speed,
            "network_speed_gbps": self.network_speed_gbps,
            "network_media": self.network_media,
            "network_interface": self.network_interface,
            "port_count": self.port_count,
            "port_speed": self.port_speed,
            "port_speed_gbps": self.port_speed_gbps,
            "port_media": self.port_media,
            "uplink_count": self.uplink_count,
            "uplink_speed": self.uplink_speed,
            "uplink_speed_gbps": self.uplink_speed_gbps,
            "uplink_media": self.uplink_media,
            "poe_supported": self.poe_supported,
            "poe_budget_w": self.poe_budget_w,
            "poe_standard": self.poe_standard,
            "l2_supported": self.l2_supported,
            "l3_supported": self.l3_supported,
            "stacking_supported": self.stacking_supported,
            "managed_status": self.managed_status,
            "airflow": self.airflow,
            "redundant_psu": self.redundant_psu,
            "transceiver_form_factor": self.transceiver_form_factor,
            "form_factor_hints": self.form_factor_hints,
        }


@dataclass(frozen=True)
class MatchResult:
    spec: StockSpec
    status: str
    engineer_review_required: bool
    total_candidates: int
    matched_items: int
    missing_requirements: list[str]
    risk_flags: list[str]
    candidates: list[MatchCandidateResult]
    component_candidate_matrix: dict[str, Any] = field(default_factory=dict)
    normalized_requirements: list[dict[str, Any]] = field(default_factory=list)
    llm_configurator_enabled: bool = False
    llm_configurator_used: bool = False
    output_mode: str = "single_best_cost_valid"
    llm_recommended_build_candidates: list[dict[str, Any]] = field(default_factory=list)
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
    llm_general_notes: list[str] = field(default_factory=list)
    llm_fallback_reason: str | None = None
    llm_error_type: str | None = None
    llm_http_status: int | None = None
    llm_parse_diagnostics: dict[str, Any] = field(default_factory=dict)
    llm_internal_warnings: list[str] = field(default_factory=list)
    llm_proposals_count: int = 0
    valid_proposals_count: int = 0
    validation_rejected_count: int = 0
    selection_skipped_count: int = 0
    rejected_ai_recommendations_count: int = 0
    ai_recommendations_validation_warnings: list[str] = field(default_factory=list)
    ai_validation_summary: dict[str, int] = field(default_factory=dict)
    rejected_reasons_top: list[dict[str, Any]] = field(default_factory=list)
    rejected_ai_recommendations_debug_safe: list[dict[str, Any]] = field(
        default_factory=list
    )
    web_evidence_pack: dict[str, Any] = field(default_factory=dict)
    llm_evidence_review: dict[str, Any] = field(default_factory=dict)
    llm_repair_used: bool = False
    llm_repair_attempted: bool = False
    llm_repair_success: bool = False
    llm_repair_fallback_reason: str | None = None
    llm_repair_critique_count: int = 0
    llm_repair_critique_summary: list[str] = field(default_factory=list)
    llm_repair_blocked_critique_count: int = 0
    llm_repair_blocked_critique_summary: list[str] = field(default_factory=list)
    llm_repair_savings_estimate: str | None = None
    llm_repair_revised_proposals_count: int = 0
    llm_repair_validation_summary: dict[str, int] = field(default_factory=dict)
    llm_thinking_diagnostics: dict[str, Any] = field(default_factory=dict)
    llm_package_diagnostics: dict[str, Any] = field(default_factory=dict)
    composer_attempt_decision: dict[str, Any] = field(default_factory=dict)
    composer_requirement_analysis: dict[str, Any] = field(default_factory=dict)
    composer_fulfillment_decisions: list[dict[str, Any]] = field(default_factory=list)
    composer_source_coverage_summary: dict[str, Any] = field(default_factory=dict)
    composer_assumptions: list[str] = field(default_factory=list)
    composer_engineer_checks: list[str] = field(default_factory=list)
    composer_hard_mismatch_risks: list[dict[str, Any]] = field(default_factory=list)
    composer_unverified_requirements: list[dict[str, Any]] = field(default_factory=list)
    composer_considered_candidate_count_by_role: dict[str, Any] = field(
        default_factory=dict
    )
    composer_chosen_candidate_ids: list[str] = field(default_factory=list)
    validation_hard_mismatches: list[dict[str, Any]] = field(default_factory=list)
    validation_unverified_requirements: list[dict[str, Any]] = field(default_factory=list)
    final_status_source: str | None = None
    product_group: str = "unknown"
    role_plan: dict[str, Any] = field(default_factory=dict)
    category_plan: dict[str, list[str]] = field(default_factory=dict)
    category_plan_entries: list[dict[str, Any]] = field(default_factory=list)
    category_catalog_summary: dict[str, Any] = field(default_factory=dict)
    category_planner_source: str = "none"
    category_plan_source: str = "none"
    category_planner_missing_required_roles: list[str] = field(default_factory=list)
    category_planner_repair_attempted: bool = False
    category_planner_repair_success: bool = False
    category_planner_repair_reason: str | None = None
    category_planner_repaired_roles: list[str] = field(default_factory=list)
    category_planner_unresolved_required_roles: list[str] = field(default_factory=list)
    required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    optional_capabilities: list[dict[str, Any]] = field(default_factory=list)
    unsupported_or_unmapped_requirements: list[str] = field(default_factory=list)
    required_roles: list[str] = field(default_factory=list)
    missing_required_roles: list[str] = field(default_factory=list)
    missing_category_roles: list[str] = field(default_factory=list)
    missing_required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    role_coverage_summary: dict[str, Any] = field(default_factory=dict)
    category_plan_warnings: list[str] = field(default_factory=list)
    stage_a_broad_roles: list[str] = field(default_factory=list)
    semantic_matrix_blueprint_roles: list[str] = field(default_factory=list)
    requirement_classifier_roles: list[str] = field(default_factory=list)
    effective_matrix_roles_before_category_planner: list[str] = field(default_factory=list)
    category_planner_input_roles: list[str] = field(default_factory=list)
    category_planner_output_roles: list[str] = field(default_factory=list)
    validated_category_plan_roles: list[str] = field(default_factory=list)
    materialized_matrix_roles: list[str] = field(default_factory=list)
    roles_dropped_after_stage_a: list[str] = field(default_factory=list)
    roles_dropped_before_category_planner: list[str] = field(default_factory=list)
    roles_dropped_after_category_planner: list[str] = field(default_factory=list)
    roles_dropped_during_materialization: list[str] = field(default_factory=list)
    roles_dropped_reason_by_role: dict[str, str] = field(default_factory=dict)
    role_source_by_role: dict[str, list[str]] = field(default_factory=dict)
    role_lifecycle_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_report_json(self) -> dict[str, Any]:
        ready_stock_candidates = [
            candidate.to_report_json()
            for candidate in self.candidates
            if candidate.candidate_type == READY_SERVER_CANDIDATE_TYPE
        ]
        build_candidates = [
            candidate.to_report_json()
            for candidate in self.candidates
            if candidate.candidate_type == BUILD_CANDIDATE_TYPE
        ]
        ai_mode = _ai_recommendation_mode(self)
        ai_recommendations_count = len(self.llm_recommended_build_candidates)
        configuration_groups = _jsonable(self.configuration_groups)
        grouped_presales_mode_used = bool(
            self.grouped_presales_mode_used and configuration_groups
        )
        matrix_coverage_summary = {}
        if isinstance(self.component_candidate_matrix, dict):
            matrix_coverage_summary = self.component_candidate_matrix.get(
                "component_matrix_coverage_summary",
                {},
            )
        package_diagnostics = _match_result_package_diagnostics(
            self,
            ready_stock_candidates=ready_stock_candidates,
            build_candidates=build_candidates,
        )
        evidence_diagnostics = safe_evidence_diagnostics(self.web_evidence_pack)
        evidence_mode = str(evidence_diagnostics.get("evidence_mode") or "separate")
        evidence_sources_count = int(evidence_diagnostics.get("evidence_sources_count") or 0)
        return {
            "status": self.status,
            "engineer_review_required": self.engineer_review_required,
            "total_candidates": self.total_candidates,
            "matched_items": self.matched_items,
            "missing_requirements": self.missing_requirements,
            "risk_flags": self.risk_flags,
            "spec": self.spec.model_dump(mode="json", exclude_none=True),
            "normalized_requirements": _jsonable(self.normalized_requirements),
            "product_group": self.product_group,
            "requirements": _jsonable(self.role_plan.get("requirements", [])),
            "required_capabilities": _jsonable(self.required_capabilities),
            "optional_capabilities": _jsonable(self.optional_capabilities),
            "workload_context": _jsonable(self.role_plan.get("workload_context", [])),
            "logistics_constraints": _jsonable(
                self.role_plan.get("logistics_constraints", {})
            ),
            "commercial_instructions": _jsonable(
                self.role_plan.get("commercial_instructions", [])
            ),
            "response_instructions": _jsonable(
                self.role_plan.get("response_instructions", [])
            ),
            "unsupported_or_unmapped_requirements": _jsonable(
                self.unsupported_or_unmapped_requirements
            ),
            "role_plan": _jsonable(self.role_plan),
            "candidate_inclusion_policy": CANDIDATE_INCLUSION_POLICY_NAME,
            **_semantic_report_fields(self.role_plan),
            "category_plan": _jsonable(self.category_plan),
            "category_plan_entries": _jsonable(self.category_plan_entries),
            "category_catalog_summary": _jsonable(self.category_catalog_summary),
            "category_planner_source": self.category_planner_source,
            "category_plan_source": self.category_plan_source,
            "category_planner_missing_required_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "category_planner_missing_required_roles",
                    self.category_planner_missing_required_roles,
                )
            ),
            "category_planner_repair_attempted": bool(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "category_planner_repair_attempted",
                    self.category_planner_repair_attempted,
                )
            ),
            "category_planner_repair_success": bool(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "category_planner_repair_success",
                    self.category_planner_repair_success,
                )
            ),
            "category_planner_repair_reason": _package_or_matrix_value(
                package_diagnostics,
                self.component_candidate_matrix,
                "category_planner_repair_reason",
                self.category_planner_repair_reason,
            ),
            "category_planner_repaired_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "category_planner_repaired_roles",
                    self.category_planner_repaired_roles,
                )
            ),
            "category_planner_unresolved_required_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "category_planner_unresolved_required_roles",
                    self.category_planner_unresolved_required_roles,
                )
            ),
            "stage_a_broad_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "stage_a_broad_roles",
                    [],
                )
            ),
            "semantic_matrix_blueprint_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "semantic_matrix_blueprint_roles",
                    [],
                )
            ),
            "requirement_classifier_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "requirement_classifier_roles",
                    [],
                )
            ),
            "effective_matrix_roles_before_category_planner": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "effective_matrix_roles_before_category_planner",
                    [],
                )
            ),
            "category_planner_input_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "category_planner_input_roles",
                    [],
                )
            ),
            "category_planner_output_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "category_planner_output_roles",
                    [],
                )
            ),
            "validated_category_plan_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "validated_category_plan_roles",
                    [],
                )
            ),
            "materialized_matrix_roles": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "materialized_matrix_roles",
                    [],
                )
            ),
            "composer_package_roles": _jsonable(
                package_diagnostics.get("composer_package_roles", [])
            ),
            "package_exposure_blocking_lifecycle_roles": _jsonable(
                package_diagnostics.get("package_exposure_blocking_lifecycle_roles", [])
            ),
            "roles_dropped_after_stage_a": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "roles_dropped_after_stage_a",
                    [],
                )
            ),
            "roles_dropped_before_category_planner": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "roles_dropped_before_category_planner",
                    [],
                )
            ),
            "roles_dropped_after_category_planner": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "roles_dropped_after_category_planner",
                    [],
                )
            ),
            "roles_dropped_during_materialization": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "roles_dropped_during_materialization",
                    [],
                )
            ),
            "roles_dropped_reason_by_role": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "roles_dropped_reason_by_role",
                    {},
                )
            ),
            "role_source_by_role": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "role_source_by_role",
                    {},
                )
            ),
            "role_lifecycle_trace": _jsonable(
                _package_or_matrix_value(
                    package_diagnostics,
                    self.component_candidate_matrix,
                    "role_lifecycle_trace",
                    [],
                )
            ),
            "required_roles": _jsonable(self.required_roles),
            "missing_required_roles": _jsonable(self.missing_required_roles),
            "missing_category_roles": _jsonable(self.missing_category_roles),
            "missing_required_capabilities": _jsonable(
                self.missing_required_capabilities
            ),
            "role_coverage_summary": _jsonable(self.role_coverage_summary),
            "category_plan_warnings": _jsonable(self.category_plan_warnings),
            "matrix_distiller_used": bool(
                self.component_candidate_matrix.get("matrix_distiller_used")
            )
            if isinstance(self.component_candidate_matrix, dict)
            else False,
            "matrix_distiller_source": _jsonable(
                self.component_candidate_matrix.get("matrix_distiller_source")
            )
            if isinstance(self.component_candidate_matrix, dict)
            else None,
            "matrix_distiller_diagnostics": _jsonable(
                self.component_candidate_matrix.get("matrix_distiller_diagnostics", {})
            )
            if isinstance(self.component_candidate_matrix, dict)
            else {},
            "broad_count_by_role": _jsonable(
                self.component_candidate_matrix.get("broad_count_by_role", {})
            )
            if isinstance(self.component_candidate_matrix, dict)
            else {},
            "distilled_count_by_role": _jsonable(
                self.component_candidate_matrix.get("distilled_count_by_role", {})
            )
            if isinstance(self.component_candidate_matrix, dict)
            else {},
            "full_matrix_evaluation_used": bool(
                self.component_candidate_matrix.get("full_matrix_evaluation_used")
            )
            if isinstance(self.component_candidate_matrix, dict)
            else bool(package_diagnostics.get("full_matrix_evaluation_used")),
            "full_matrix_evaluation_fallback_reason": _jsonable(
                self.component_candidate_matrix.get(
                    "full_matrix_evaluation_fallback_reason",
                    package_diagnostics.get("full_matrix_evaluation_fallback_reason"),
                )
            )
            if isinstance(self.component_candidate_matrix, dict)
            else _jsonable(
                package_diagnostics.get("full_matrix_evaluation_fallback_reason")
            ),
            "provider_error_type": _jsonable(
                package_diagnostics.get("provider_error_type")
                or (
                    self.component_candidate_matrix.get("provider_error_type")
                    if isinstance(self.component_candidate_matrix, dict)
                    else None
                )
            ),
            "provider_context_limit": _jsonable(
                package_diagnostics.get("provider_context_limit")
                or (
                    self.component_candidate_matrix.get("provider_context_limit", {})
                    if isinstance(self.component_candidate_matrix, dict)
                    else {}
                )
            ),
            "role_chunk_count_by_role": _jsonable(
                self.component_candidate_matrix.get(
                    "role_chunk_count_by_role",
                    package_diagnostics.get("role_chunk_count_by_role", {}),
                )
            )
            if isinstance(self.component_candidate_matrix, dict)
            else _jsonable(package_diagnostics.get("role_chunk_count_by_role", {})),
            "evaluated_candidate_count_by_role": _jsonable(
                self.component_candidate_matrix.get(
                    "evaluated_candidate_count_by_role",
                    package_diagnostics.get("evaluated_candidate_count_by_role", {}),
                )
            )
            if isinstance(self.component_candidate_matrix, dict)
            else _jsonable(package_diagnostics.get("evaluated_candidate_count_by_role", {})),
            "selected_candidate_count_by_role": _jsonable(
                self.component_candidate_matrix.get(
                    "selected_candidate_count_by_role",
                    package_diagnostics.get("selected_candidate_count_by_role", {}),
                )
            )
            if isinstance(self.component_candidate_matrix, dict)
            else _jsonable(package_diagnostics.get("selected_candidate_count_by_role", {})),
            "role_reducer_summary": _jsonable(
                self.component_candidate_matrix.get(
                    "role_reducer_summary",
                    package_diagnostics.get("role_reducer_summary", {}),
                )
            )
            if isinstance(self.component_candidate_matrix, dict)
            else _jsonable(package_diagnostics.get("role_reducer_summary", {})),
            "full_matrix_failed_chunks": _jsonable(
                self.component_candidate_matrix.get(
                    "full_matrix_failed_chunks",
                    package_diagnostics.get("full_matrix_failed_chunks", []),
                )
            )
            if isinstance(self.component_candidate_matrix, dict)
            else _jsonable(package_diagnostics.get("full_matrix_failed_chunks", [])),
            "bom_critic_used": bool(
                package_diagnostics.get("bom_critic_used")
                or self.llm_repair_attempted
                or self.llm_repair_used
                or package_diagnostics.get("no_recommendation_coverage_repair_attempted")
            ),
            "no_recommendation_coverage": _jsonable(
                self.no_recommendation_reason.get("no_recommendation_coverage")
                or self.component_candidate_matrix.get("no_recommendation_coverage", {})
                if isinstance(self.component_candidate_matrix, dict)
                else package_diagnostics.get("no_recommendation_coverage", {})
            ),
            "no_recommendation_coverage_gate_passed": bool(
                package_diagnostics.get("no_recommendation_coverage_gate_passed")
            ),
            "no_recommendation_coverage_repair_attempted": bool(
                package_diagnostics.get("no_recommendation_coverage_repair_attempted")
            ),
            "no_recommendation_coverage_repair_success": bool(
                package_diagnostics.get("no_recommendation_coverage_repair_success")
            ),
            "no_recommendation_coverage_rejected": bool(
                package_diagnostics.get("no_recommendation_coverage_rejected")
            ),
            "no_recommendation_coverage_thresholds": _jsonable(
                package_diagnostics.get("no_recommendation_coverage_thresholds", {})
            ),
            "no_recommendation_coverage_repair_reason": _jsonable(
                package_diagnostics.get("no_recommendation_coverage_repair_reason")
            ),
            "llm_cost_diagnostics": _jsonable(
                self.component_candidate_matrix.get(
                    "llm_cost_diagnostics",
                    package_diagnostics.get("llm_cost_diagnostics", {}),
                )
            )
            if isinstance(self.component_candidate_matrix, dict)
            else _jsonable(package_diagnostics.get("llm_cost_diagnostics", {})),
            "count_by_role": _jsonable(package_diagnostics.get("count_by_role", {})),
            "broad_matrix_count_by_role": _jsonable(
                package_diagnostics.get("broad_matrix_count_by_role", {})
            ),
            "composer_package_candidate_count_by_role": _jsonable(
                package_diagnostics.get("composer_package_candidate_count_by_role", {})
            ),
            "composer_package_candidate_total": int(
                package_diagnostics.get("composer_package_candidate_total") or 0
            ),
            "composer_package_candidate_ids_by_role": _jsonable(
                package_diagnostics.get("composer_package_candidate_ids_by_role", {})
            ),
            "v2_package_mode": _jsonable(package_diagnostics.get("v2_package_mode")),
            "selected_package_mode": _jsonable(
                package_diagnostics.get("selected_package_mode")
            ),
            "verbose_context_chars": _jsonable(
                package_diagnostics.get("verbose_context_chars")
            ),
            "compact_context_chars": _jsonable(
                package_diagnostics.get("compact_context_chars")
            ),
            "selected_context_chars": _jsonable(
                package_diagnostics.get("selected_context_chars")
            ),
            "verbose_context_size": _jsonable(
                package_diagnostics.get("verbose_context_size", {})
            ),
            "compact_context_size": _jsonable(
                package_diagnostics.get("compact_context_size", {})
            ),
            "selected_context_size": _jsonable(
                package_diagnostics.get("selected_context_size", {})
            ),
            "chars_by_section": _jsonable(
                package_diagnostics.get("chars_by_section", {})
            ),
            "avg_chars_per_candidate_by_role": _jsonable(
                package_diagnostics.get("avg_chars_per_candidate_by_role", {})
            ),
            "removed_verbose_fields": _jsonable(
                package_diagnostics.get("removed_verbose_fields", [])
            ),
            "removed_verbose_field_counts": _jsonable(
                package_diagnostics.get("removed_verbose_field_counts", {})
            ),
            "compact_candidate_count_by_role": _jsonable(
                package_diagnostics.get("compact_candidate_count_by_role", {})
            ),
            "compact_candidate_total": int(
                package_diagnostics.get("compact_candidate_total") or 0
            ),
            "compact_candidate_ids_hash": _jsonable(
                package_diagnostics.get("compact_candidate_ids_hash")
            ),
            "compact_package_full_matrix_used": bool(
                package_diagnostics.get("compact_package_full_matrix_used")
            ),
            "package_candidate_loss": bool(
                package_diagnostics.get("package_candidate_loss")
            ),
            "provider_context_limit_retry_compact_attempted": bool(
                package_diagnostics.get(
                    "provider_context_limit_retry_compact_attempted"
                )
            ),
            "provider_context_limit_retry_compact_success": bool(
                package_diagnostics.get("provider_context_limit_retry_compact_success")
            ),
            "provider_context_limit_original_chars": _jsonable(
                package_diagnostics.get("provider_context_limit_original_chars")
            ),
            "provider_context_limit_compact_chars": _jsonable(
                package_diagnostics.get("provider_context_limit_compact_chars")
            ),
            "provider_context_limit_after_compact": bool(
                package_diagnostics.get("provider_context_limit_after_compact")
            ),
            "dropped_before_composer_count_by_role": _jsonable(
                package_diagnostics.get("dropped_before_composer_count_by_role", {})
            ),
            "dropped_before_composer_reason_by_role": _jsonable(
                package_diagnostics.get("dropped_before_composer_reason_by_role", {})
            ),
            "package_candidate_exposure_ratio_by_role": _jsonable(
                package_diagnostics.get("package_candidate_exposure_ratio_by_role", {})
            ),
            "package_candidate_exposure_policy": _jsonable(
                package_diagnostics.get("package_candidate_exposure_policy", {})
            ),
            "package_candidate_exposure_incomplete": bool(
                package_diagnostics.get("package_candidate_exposure_incomplete")
            ),
            "package_candidate_exposure_incomplete_roles": _jsonable(
                package_diagnostics.get("package_candidate_exposure_incomplete_roles", [])
            ),
            "package_budget": _jsonable(package_diagnostics.get("package_budget", {})),
            "package_budget_warnings": _jsonable(
                package_diagnostics.get("package_budget_warnings", [])
            ),
            "package_approximate_size": _jsonable(
                package_diagnostics.get("package_approximate_size", {})
            ),
            "package_skipped_reason": _jsonable(
                package_diagnostics.get("package_skipped_reason")
                if package_diagnostics.get("package_skipped_reason") is not None
                else self.component_candidate_matrix.get("package_skipped_reason")
            )
            if isinstance(self.component_candidate_matrix, dict)
            else None,
            "ready_candidates_excluded_reason": _jsonable(
                package_diagnostics.get("ready_candidates_excluded_reason")
            ),
            "composer_attempt_decision": _jsonable(
                self.composer_attempt_decision
                or package_diagnostics.get("composer_attempt_decision")
                or {}
            ),
            "pre_composer_requirement_classifier_status": _jsonable(
                package_diagnostics.get("pre_composer_requirement_classifier_status")
                or package_diagnostics.get("requirement_classifier_status")
            ),
            "pre_composer_requirement_source_coverage_percent": _jsonable(
                package_diagnostics.get(
                    "pre_composer_requirement_source_coverage_percent"
                )
                if package_diagnostics.get(
                    "pre_composer_requirement_source_coverage_percent"
                )
                is not None
                else package_diagnostics.get("requirement_source_coverage_percent")
            ),
            "pre_composer_unclassified_source_fragments": _jsonable(
                package_diagnostics.get("pre_composer_unclassified_source_fragments")
                or package_diagnostics.get("unclassified_source_fragments", [])
            ),
            "pre_composer_semantic_diagnostics_are_blocking": bool(
                package_diagnostics.get(
                    "pre_composer_semantic_diagnostics_are_blocking",
                    False,
                )
            ),
            "composer_requirement_analysis": _jsonable(
                self.composer_requirement_analysis
                or package_diagnostics.get("composer_requirement_analysis", {})
            ),
            "composer_fulfillment_decisions": _jsonable(
                self.composer_fulfillment_decisions
                or package_diagnostics.get("composer_fulfillment_decisions", [])
            ),
            "composer_source_coverage_summary": _jsonable(
                self.composer_source_coverage_summary
                or package_diagnostics.get("composer_source_coverage_summary", {})
            ),
            "composer_assumptions": _jsonable(
                self.composer_assumptions
                or package_diagnostics.get("composer_assumptions", [])
            ),
            "composer_engineer_checks": _jsonable(
                self.composer_engineer_checks
                or package_diagnostics.get("composer_engineer_checks", [])
            ),
            "composer_hard_mismatch_risks": _jsonable(
                self.composer_hard_mismatch_risks
                or package_diagnostics.get("composer_hard_mismatch_risks", [])
            ),
            "composer_unverified_requirements": _jsonable(
                self.composer_unverified_requirements
                or package_diagnostics.get("composer_unverified_requirements", [])
            ),
            "composer_considered_candidate_count_by_role": _jsonable(
                self.composer_considered_candidate_count_by_role
                or package_diagnostics.get(
                    "composer_considered_candidate_count_by_role", {}
                )
            ),
            "composer_chosen_candidate_ids": _jsonable(
                self.composer_chosen_candidate_ids
                or package_diagnostics.get("composer_chosen_candidate_ids", [])
            ),
            "validation_hard_mismatches": _jsonable(
                self.validation_hard_mismatches
                or package_diagnostics.get("validation_hard_mismatches", [])
            ),
            "validation_unverified_requirements": _jsonable(
                self.validation_unverified_requirements
                or package_diagnostics.get("validation_unverified_requirements", [])
            ),
            "final_status_source": _jsonable(
                self.final_status_source
                or package_diagnostics.get("final_status_source")
            ),
            "component_candidate_matrix": _jsonable(self.component_candidate_matrix),
            "component_matrix_coverage_summary": _jsonable(matrix_coverage_summary),
            "shortlist_for_llm": _jsonable(
                _shortlist_for_llm(build_candidates, self.component_candidate_matrix)
            ),
            "llm_configurator_enabled": self.llm_configurator_enabled,
            "llm_configurator_used": self.llm_configurator_used,
            "output_mode": self.output_mode,
            "llm_configurator_output_mode": self.output_mode,
            "ai_recommendation_mode": ai_mode,
            "ai_recommendations_count": ai_recommendations_count,
            "ai_recommendations": _jsonable(self.llm_recommended_build_candidates),
            "llm_recommendations": _jsonable(self.llm_recommended_build_candidates),
            "llm_recommended_build_candidates": _jsonable(
                self.llm_recommended_build_candidates
            ),
            "primary_recommendation": _jsonable(self.primary_recommendation),
            "primary_recommendation_status": self.primary_recommendation_status,
            "no_recommendation_reason": _jsonable(self.no_recommendation_reason),
            "partial_available_components": _jsonable(
                self.no_recommendation_reason.get("partial_available_components", [])
            ),
            "failed_requirements": _jsonable(
                self.no_recommendation_reason.get("failed_requirements", [])
            ),
            "role_failures": _jsonable(
                self.no_recommendation_reason.get("role_failures", [])
            ),
            "unverified_requirements": _jsonable(
                self.no_recommendation_reason.get("unverified_requirements", [])
            ),
            "hard_mismatch_risks": _jsonable(
                self.no_recommendation_reason.get("hard_mismatch_risks", [])
            ),
            "recommended_next_actions": _jsonable(
                self.no_recommendation_reason.get("recommended_next_actions", [])
            ),
            "engineer_checks": _jsonable(
                self.no_recommendation_reason.get("engineer_checks")
                or self.no_recommendation_reason.get("engineering_checks", [])
            ),
            "composer_summary_ru": _jsonable(
                self.no_recommendation_reason.get("composer_summary_ru")
            ),
            "customer_safe_summary_ru": _jsonable(
                self.no_recommendation_reason.get("customer_safe_summary_ru")
            ),
            "commercial_summary": _jsonable(self.commercial_summary),
            "grouped_presales_mode_used": grouped_presales_mode_used,
            "configuration_groups": configuration_groups,
            "configuration_groups_count": len(configuration_groups)
            if isinstance(configuration_groups, list)
            else 0,
            "quote_recommendation": _jsonable(self.quote_recommendation),
            "selected_configuration_group_id": self.selected_configuration_group_id,
            "selected_platform_option_id": self.selected_platform_option_id,
            "selected_platform_option_index": self.selected_platform_option_index,
            "llm_general_notes": self.llm_general_notes,
            "llm_fallback_reason": self.llm_fallback_reason,
            "llm_error_type": self.llm_error_type,
            "llm_http_status": self.llm_http_status,
            "llm_parse_diagnostics": _jsonable(self.llm_parse_diagnostics),
            "llm_parse_stage": str(
                self.llm_parse_diagnostics.get("llm_parse_stage") or ""
            ),
            "llm_json_extract_status": str(
                self.llm_parse_diagnostics.get("llm_json_extract_status") or ""
            ),
            "llm_invalid_json_reason": str(
                self.llm_parse_diagnostics.get("llm_invalid_json_reason") or ""
            ),
            "llm_invalid_json_preview_sanitized": str(
                self.llm_parse_diagnostics.get(
                    "llm_invalid_json_preview_sanitized"
                )
                or ""
            ),
            "llm_internal_warnings": self.llm_internal_warnings,
            "llm_repair_used": self.llm_repair_used,
            "llm_repair_attempted": self.llm_repair_attempted,
            "llm_repair_success": self.llm_repair_success,
            "llm_repair_fallback_reason": self.llm_repair_fallback_reason,
            "llm_repair_critique_count": self.llm_repair_critique_count,
            "llm_repair_critique_summary": _jsonable(
                self.llm_repair_critique_summary
            ),
            "llm_repair_blocked_critique_count": (
                self.llm_repair_blocked_critique_count
            ),
            "llm_repair_blocked_critique_summary": _jsonable(
                self.llm_repair_blocked_critique_summary
            ),
            "llm_repair_savings_estimate": self.llm_repair_savings_estimate,
            "llm_repair_revised_proposals_count": (
                self.llm_repair_revised_proposals_count
            ),
            "llm_repair_validation_summary": _jsonable(
                self.llm_repair_validation_summary
            ),
            "llm_thinking_diagnostics": _jsonable(self.llm_thinking_diagnostics),
            "llm_thinking_enabled": bool(
                self.llm_thinking_diagnostics.get("llm_thinking_enabled")
            ),
            "llm_thinking_budget_tokens": self.llm_thinking_diagnostics.get(
                "llm_thinking_budget_tokens"
            ),
            "llm_thinking_fallback_reason": self.llm_thinking_diagnostics.get(
                "llm_thinking_fallback_reason"
            ),
            "llm_proposals_count": self.llm_proposals_count,
            "valid_proposals_count": self.valid_proposals_count,
            "validation_rejected_count": self.validation_rejected_count,
            "selection_skipped_count": self.selection_skipped_count,
            "rejected_ai_recommendations_count": self.rejected_ai_recommendations_count,
            "ai_recommendations_validation_warnings": self.ai_recommendations_validation_warnings,
            "ai_validation_summary": {
                "mode": ai_mode,
                "accepted": ai_recommendations_count,
                "accepted_after_validation": self.valid_proposals_count,
                **self.ai_validation_summary,
                "rejected": self.rejected_ai_recommendations_count,
                "warnings": self.ai_recommendations_validation_warnings,
            },
            "rejected_reasons_top": _jsonable(self.rejected_reasons_top),
            "rejected_ai_recommendations_debug_safe": _jsonable(
                self.rejected_ai_recommendations_debug_safe
            ),
            "rejected_ai_recommendations": _jsonable(
                self.rejected_ai_recommendations_debug_safe
            ),
            "web_evidence_pack": _jsonable(self.web_evidence_pack),
            "web_evidence_diagnostics": _jsonable(evidence_diagnostics),
            "evidence_mode": evidence_mode,
            "online_composer_used": bool(
                evidence_diagnostics.get("online_composer_used")
            ),
            "evidence_used": bool(evidence_diagnostics.get("evidence_used")),
            "evidence_sources_count": evidence_sources_count,
            "evidence_status_summary": _jsonable(
                evidence_diagnostics.get("evidence_status_summary") or {}
            ),
            "online_composer_error_type": str(
                evidence_diagnostics.get("online_composer_error_type") or ""
            ),
            "online_composer_parse_status": str(
                evidence_diagnostics.get("online_composer_parse_status") or ""
            ),
            "online_composer_empty_response_repair_attempted": bool(
                evidence_diagnostics.get(
                    "online_composer_empty_response_repair_attempted"
                )
            ),
            "online_composer_empty_response_repair_success": bool(
                evidence_diagnostics.get("online_composer_empty_response_repair_success")
            ),
            "structured_no_recommendation_used": bool(
                evidence_diagnostics.get("structured_no_recommendation_used")
            ),
            "evidence_requests_count": int(
                evidence_diagnostics.get("evidence_requests_count") or 0
            ),
            "llm_evidence_review": _jsonable(self.llm_evidence_review),
            "candidates": [candidate.to_report_json() for candidate in self.candidates],
            "ready_stock_candidates": ready_stock_candidates,
            "build_candidates": build_candidates,
        }


@dataclass(frozen=True)
class PlannedMatchContext:
    spec: StockSpec
    product_group: str
    products: list[DistributorProduct]
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]]
    role_plan: dict[str, Any]
    category_plan_result: CategoryPlanResult
    build_candidates: list[MatchCandidateResult]
    build_missing: list[str]
    build_risks: list[str]
    component_candidate_matrix: dict[str, Any]
    normalized_requirements: list[dict[str, Any]]

    def to_report_json(self) -> dict[str, Any]:
        build_rows = [candidate.to_report_json() for candidate in self.build_candidates]
        package = build_llm_configurator_package(
            user_request=self.spec.source_text,
            normalized_requirements=self.normalized_requirements,
            ready_stock_candidates=[],
            component_candidate_matrix=self.component_candidate_matrix,
            rule_based_build_candidates=build_rows,
        )
        return {
            "normalized_requirements": _jsonable(self.normalized_requirements),
            "product_group": self.product_group,
            "role_plan": _jsonable(self.role_plan),
            **_semantic_report_fields(self.role_plan),
            "required_capabilities": _jsonable(
                self.role_plan.get("required_capabilities", [])
            ),
            "optional_capabilities": _jsonable(
                self.role_plan.get("optional_capabilities", [])
            ),
            "unsupported_or_unmapped_requirements": _jsonable(
                self.role_plan.get("unsupported_or_unmapped_requirements", [])
            ),
            "required_roles": _jsonable(self.role_plan.get("required_roles", [])),
            "category_catalog_summary": _jsonable(
                self.category_plan_result.category_catalog_summary
            ),
            "category_plan": _jsonable(self.category_plan_result.category_plan),
            "category_plan_entries": _jsonable(
                self.category_plan_result.category_plan_entries
            ),
            "category_planner_source": self.category_plan_result.category_planner_source,
            "category_plan_source": self.category_plan_result.category_plan_source,
            "category_planner_missing_required_roles": _jsonable(
                self.category_plan_result.category_planner_missing_required_roles
            ),
            "category_planner_repair_attempted": (
                self.category_plan_result.category_planner_repair_attempted
            ),
            "category_planner_repair_success": (
                self.category_plan_result.category_planner_repair_success
            ),
            "category_planner_repair_reason": (
                self.category_plan_result.category_planner_repair_reason
            ),
            "category_planner_repaired_roles": _jsonable(
                self.category_plan_result.category_planner_repaired_roles
            ),
            "category_planner_unresolved_required_roles": _jsonable(
                self.category_plan_result.category_planner_unresolved_required_roles
            ),
            "category_plan_warnings": _jsonable(
                self.category_plan_result.category_plan_warnings
            ),
            "category_planner_input_roles": _jsonable(
                self.category_plan_result.category_planner_input_roles
            ),
            "category_planner_output_roles": _jsonable(
                self.category_plan_result.category_planner_output_roles
            ),
            "validated_category_plan_roles": _jsonable(
                self.category_plan_result.validated_category_plan_roles
            ),
            "roles_dropped_before_category_planner": _jsonable(
                self.category_plan_result.roles_dropped_before_category_planner
            ),
            "roles_dropped_after_category_planner": _jsonable(
                self.category_plan_result.roles_dropped_after_category_planner
            ),
            "roles_dropped_reason_by_role": _jsonable(
                self.category_plan_result.roles_dropped_reason_by_role
            ),
            "role_source_by_role": _jsonable(
                self.category_plan_result.role_source_by_role
            ),
            "role_lifecycle_trace": _jsonable(
                self.component_candidate_matrix.get("role_lifecycle_trace")
                or self.category_plan_result.role_lifecycle_trace
            ),
            "missing_category_roles": _jsonable(
                self.category_plan_result.missing_category_roles
            ),
            "component_candidate_matrix": _jsonable(self.component_candidate_matrix),
            "shortlist_for_llm": _jsonable(
                _shortlist_for_llm(build_rows, self.component_candidate_matrix)
            ),
            "role_coverage_summary": _jsonable(
                self.component_candidate_matrix.get("role_coverage_summary", {})
            ),
            "missing_required_roles_before_llm": _jsonable(
                self.component_candidate_matrix.get("missing_required_roles_before_llm")
                or self.component_candidate_matrix.get("missing_required_roles")
                or []
            ),
            "missing_required_capabilities_before_llm": _jsonable(
                self.component_candidate_matrix.get(
                    "missing_required_capabilities_before_llm"
                )
                or self.component_candidate_matrix.get("missing_required_capabilities")
                or []
            ),
            "package_budget": _jsonable(package.get("package_budget", {})),
            "package_budget_warnings": _jsonable(
                package.get("package_budget_warnings", [])
            ),
        }


@dataclass(frozen=True)
class _PlannerClientState:
    client: LlmClient | None = None
    provider: str | None = None
    model: str | None = None
    unavailable_reason: str | None = None
    error_type: str | None = None


def _ai_recommendation_mode(result: MatchResult) -> str:
    if not result.llm_configurator_enabled:
        return "llm_disabled"
    if result.llm_configurator_used:
        has_complete = any(
            recommendation.get("completeness_status") == "complete"
            for recommendation in result.llm_recommended_build_candidates
        )
        return "ai_success" if has_complete else "ai_partial_success"
    if result.llm_fallback_reason in {
        "llm_configurator_all_recommendations_rejected",
        "llm_configurator_no_valid_recommendations",
        "llm_configurator_no_proposals",
        "llm_configurator_no_proposals_after_repair",
        "llm_configurator_structured_no_recommendation",
        "llm_configurator_no_complete_recommendation",
        "composer_structured_no_recommendation",
        "composer_no_safe_complete_bom",
    }:
        return "ai_no_safe_recommendations"
    return "ai_unavailable"


def _match_result_package_diagnostics(
    result: MatchResult,
    *,
    ready_stock_candidates: list[dict[str, Any]],
    build_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if result.llm_package_diagnostics:
        return dict(result.llm_package_diagnostics)
    if not isinstance(result.component_candidate_matrix, dict):
        return {}

    settings = get_llm_settings()
    package = build_llm_configurator_package(
        user_request=result.spec.source_text,
        normalized_requirements=result.normalized_requirements,
        ready_stock_candidates=ready_stock_candidates,
        component_candidate_matrix=result.component_candidate_matrix,
        rule_based_build_candidates=build_candidates,
        candidates_per_role=_matrix_candidates_per_role(settings),
        output_mode=result.output_mode,
        max_package_chars=settings.llm_configurator_max_package_chars,
    )
    package_size_chars = len(
        json.dumps(package, ensure_ascii=False, sort_keys=True, default=str)
    )
    return {
        "package_budget": _dict_or_empty(package.get("package_budget")),
        "package_budget_warnings": _string_list(package.get("package_budget_warnings")),
        "package_skipped_reason": package.get("package_skipped_reason"),
        "package_approximate_size": {
            "chars": package_size_chars,
            "tokens_estimate": max(1, package_size_chars // 4),
        },
        "ready_candidates_excluded_reason": package.get(
            "ready_candidates_excluded_reason"
        ),
        "full_matrix_evaluation_used": bool(package.get("full_matrix_evaluation_used")),
        "full_matrix_evaluation_fallback_reason": package.get(
            "full_matrix_evaluation_fallback_reason"
        ),
        "provider_error_type": package.get("provider_error_type"),
        "provider_context_limit": _dict_or_empty(package.get("provider_context_limit")),
        "role_chunk_count_by_role": _dict_or_empty(
            package.get("role_chunk_count_by_role")
        ),
        "evaluated_candidate_count_by_role": _dict_or_empty(
            package.get("evaluated_candidate_count_by_role")
        ),
        "selected_candidate_count_by_role": _dict_or_empty(
            package.get("selected_candidate_count_by_role")
        ),
        "role_reducer_summary": _dict_or_empty(package.get("role_reducer_summary")),
        "full_matrix_failed_chunks": _mapping_list(
            package.get("full_matrix_failed_chunks")
        ),
        "no_recommendation_coverage": _dict_or_empty(
            package.get("no_recommendation_coverage")
        ),
        "llm_cost_diagnostics": _dict_or_empty(package.get("llm_cost_diagnostics")),
        "count_by_role": _package_count_by_role_for_report(package),
        "broad_matrix_count_by_role": _dict_or_empty(
            package.get("broad_matrix_count_by_role")
        ),
        "composer_package_candidate_count_by_role": _dict_or_empty(
            package.get("composer_package_candidate_count_by_role")
        ),
        "composer_package_candidate_total": int(
            package.get("composer_package_candidate_total") or 0
        ),
        "composer_package_candidate_ids_by_role": _dict_or_empty(
            package.get("composer_package_candidate_ids_by_role")
        ),
        "dropped_before_composer_count_by_role": _dict_or_empty(
            package.get("dropped_before_composer_count_by_role")
        ),
        "dropped_before_composer_reason_by_role": _dict_or_empty(
            package.get("dropped_before_composer_reason_by_role")
        ),
        "package_candidate_exposure_ratio_by_role": _dict_or_empty(
            package.get("package_candidate_exposure_ratio_by_role")
        ),
        "package_candidate_exposure_policy": _dict_or_empty(
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
        "materialized_matrix_roles": _string_list(package.get("materialized_matrix_roles")),
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
        "roles_dropped_reason_by_role": _dict_or_empty(
            package.get("roles_dropped_reason_by_role")
        ),
        "role_source_by_role": _dict_or_empty(package.get("role_source_by_role")),
        "role_lifecycle_trace": _mapping_list(package.get("role_lifecycle_trace")),
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
    }


def _package_count_by_role_for_report(package: Mapping[str, Any]) -> dict[str, int]:
    broad_count = _dict_or_empty(package.get("broad_count_by_role"))
    result = {
        str(role): int(count)
        for role, count in broad_count.items()
        if str(role).strip() and isinstance(count, int) and count > 0
    }
    if result:
        return result
    matrix = _dict_or_empty(package.get("component_candidate_matrix"))
    return {
        str(role): len(rows)
        for role, rows in matrix.items()
        if isinstance(rows, list) and rows
    }


@dataclass(frozen=True)
class _NormalizedServerRequirements:
    server_qty: int
    product_group: str = SERVER_PRODUCT_GROUP
    network_device_role: str | None = None
    form_factor: str | None = None
    cpu_per_server: int | None = None
    total_cpu_required: int | None = None
    cpu_vendor_preference: str = UNKNOWN_FACT
    cpu_family_preference: str = UNKNOWN_FACT
    cpu_min_cores_per_cpu: int | None = None
    cpu_generation_or_model_hint: str | None = None
    ram_gb_per_server: int | None = None
    ram_type_preference: str = UNKNOWN_FACT
    storage_required: bool = False
    storage_type_preference: str = UNKNOWN_FACT
    storage_interface_preference: str = UNKNOWN_FACT
    storage_min_capacity: str | None = None
    storage_min_capacity_tb: float | None = None
    storage_qty_per_server: int | None = None
    raw_capacity_tb: float | None = None
    usable_capacity_tb: float | None = None
    redundancy_level: str = UNKNOWN_FACT
    controller_count: int | None = None
    shelf_count: int | None = None
    drive_count: int | None = None
    drive_capacity_tb: float | None = None
    drive_type: str = UNKNOWN_FACT
    drive_interface: str = UNKNOWN_FACT
    host_protocol: str = UNKNOWN_FACT
    host_port_count: int | None = None
    host_port_speed: str = UNKNOWN_FACT
    host_port_speed_gbps: int | None = None
    host_port_media: str = UNKNOWN_FACT
    license_required: bool = False
    support_required: bool = False
    warranty_months: int | None = None
    network_required: bool = False
    network_min_ports_per_server: int | None = None
    network_speed: str = UNKNOWN_FACT
    network_media: str = UNKNOWN_FACT
    network_interface: str = UNKNOWN_FACT
    network_requirement: dict[str, Any] = field(default_factory=dict)
    psu_count_per_server: int | None = None
    location: str | None = None
    optimization_mode: str = OPTIMIZATION_MODE_COST_MINIMAL_FIT
    role_plan: dict[str, Any] = field(default_factory=dict)
    required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    optional_capabilities: list[dict[str, Any]] = field(default_factory=list)
    unsupported_or_unmapped_requirements: list[str] = field(default_factory=list)
    required_roles: list[str] = field(default_factory=list)
    category_plan: dict[str, list[str]] = field(default_factory=dict)

    def to_report_json(self) -> dict[str, Any]:
        return {
            "product_group": self.product_group,
            "server_qty": self.server_qty,
            "device_qty": self.server_qty if self.product_group == NETWORK_PRODUCT_GROUP else None,
            "system_qty": self.server_qty
            if self.product_group == STORAGE_PRODUCT_GROUP
            else None,
            "network_device_role": self.network_device_role,
            "form_factor": self.form_factor,
            "cpu_per_server": self.cpu_per_server,
            "total_cpu_required": self.total_cpu_required,
            "cpu_vendor_preference": self.cpu_vendor_preference,
            "cpu_family_preference": self.cpu_family_preference,
            "cpu_min_cores_per_cpu": self.cpu_min_cores_per_cpu,
            "cpu_generation_or_model_hint": self.cpu_generation_or_model_hint,
            "ram_gb_per_server": self.ram_gb_per_server,
            "ram_type_preference": self.ram_type_preference,
            "storage_required": self.storage_required,
            "storage_type_preference": self.storage_type_preference,
            "storage_interface_preference": self.storage_interface_preference,
            "storage_min_capacity": self.storage_min_capacity,
            "storage_min_capacity_tb": self.storage_min_capacity_tb,
            "storage_qty_per_server": self.storage_qty_per_server,
            "raw_capacity_tb": self.raw_capacity_tb,
            "usable_capacity_tb": self.usable_capacity_tb,
            "redundancy_level": self.redundancy_level,
            "controller_count": self.controller_count,
            "shelf_count": self.shelf_count,
            "drive_count": self.drive_count,
            "drive_capacity_tb": self.drive_capacity_tb,
            "drive_type": self.drive_type,
            "drive_interface": self.drive_interface,
            "host_protocol": self.host_protocol,
            "host_port_count": self.host_port_count,
            "host_port_speed": self.host_port_speed,
            "host_port_speed_gbps": self.host_port_speed_gbps,
            "host_port_media": self.host_port_media,
            "license_required": self.license_required,
            "support_required": self.support_required,
            "warranty_months": self.warranty_months,
            "network_required": self.network_required,
            "network_min_ports_per_server": self.network_min_ports_per_server,
            "network_speed": self.network_speed,
            "network_media": self.network_media,
            "network_interface": self.network_interface,
            "network_requirement": self.network_requirement,
            "psu_count_per_server": self.psu_count_per_server,
            "location": self.location,
            "optimization_mode": self.optimization_mode,
            "role_plan": _jsonable(self.role_plan),
            "required_capabilities": _jsonable(self.required_capabilities),
            "optional_capabilities": _jsonable(self.optional_capabilities),
            "unsupported_or_unmapped_requirements": _jsonable(
                self.unsupported_or_unmapped_requirements
            ),
            "required_roles": self.required_roles,
            "category_plan": _jsonable(self.category_plan),
        }


@dataclass(frozen=True)
class _ComponentCandidate:
    role: str
    product: DistributorProduct
    facts: _ProductFacts
    quantity_required: int
    available_quantity: int | None
    reservable_locations: int
    price_value: Decimal | None
    price_currency: str | None
    eligibility_status: str
    eligibility_warnings: list[str]
    fit_reasons: list[str]
    score: int
    fit_label: str = FIT_UNKNOWN
    fit_reason: str = ""
    fit_tier: str = FIT_TIER_POSSIBLE
    match_warnings: list[str] = field(default_factory=list)
    uncertainty_reasons: list[str] = field(default_factory=list)
    objective_reject_reason: str | None = None
    evidence_summary: str = ""
    inclusion_policy: str = CANDIDATE_INCLUSION_POLICY_NAME
    cpu_over_requirement: int | None = None
    storage_over_requirement: float | None = None
    ram_over_requirement_gb: int | None = None
    selection_bucket: str = "score"
    bucket_priority: int = 90
    capability_id: str | None = None

    @property
    def candidate_id(self) -> str:
        return _stable_candidate_id(self.role, self.product)

    def to_report_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "component_candidate_id": self.candidate_id,
            "role": self.role,
            "capability_id": self.capability_id,
            "distributor_code": self.product.distributor_code,
            "item_id": self.product.item_id,
            "product_key": self.product.product_key,
            "producer": self.product.producer,
            "normalized_vendor": self.facts.normalized_vendor,
            "part_number": self.product.part_number,
            "name": self.product.item_name,
            "item_name": self.product.item_name,
            "item_name_rus": self.product.item_name_rus,
            "product_name": self.product.product_name,
            "product_description": self.product.product_description,
            "category_id": self.product.category_id,
            "catalog_path": _jsonable(self.product.catalog_path_json),
            "package_json": _compact_package_json(self.product.package_json),
            "price_value": _jsonable(self.price_value),
            "price_currency": self.price_currency,
            "available_quantity": self.available_quantity,
            "quantity_required": self.quantity_required,
            "reservable_locations": self.reservable_locations,
            "extracted_facts": self.facts.to_report_json(),
            "cpu_cores": self.facts.cpu_cores,
            "cpu_over_requirement": self.cpu_over_requirement,
            "storage_capacity_tb": self.facts.storage_capacity_tb,
            "storage_over_requirement": self.storage_over_requirement,
            "raw_capacity_tb": self.facts.raw_capacity_tb,
            "usable_capacity_tb": self.facts.usable_capacity_tb,
            "redundancy_level": self.facts.redundancy_level,
            "controller_count": self.facts.controller_count,
            "drive_count": self.facts.drive_count,
            "drive_capacity_tb": self.facts.drive_capacity_tb,
            "drive_type": self.facts.drive_type,
            "drive_interface": self.facts.drive_interface,
            "host_protocol": self.facts.host_protocol,
            "host_port_count": self.facts.host_port_count,
            "host_port_speed": self.facts.host_port_speed,
            "host_port_speed_gbps": self.facts.host_port_speed_gbps,
            "host_port_media": self.facts.host_port_media,
            "warranty_months": self.facts.warranty_months,
            "ram_module_capacity_gb": self.facts.ram_capacity_gb,
            "ram_over_requirement_gb": self.ram_over_requirement_gb,
            "network_ports_count": self.facts.network_ports_count,
            "network_speed": self.facts.network_speed,
            "network_media": self.facts.network_media,
            "network_interface": self.facts.network_interface,
            "port_count": self.facts.port_count,
            "port_speed": self.facts.port_speed,
            "port_speed_gbps": self.facts.port_speed_gbps,
            "port_media": self.facts.port_media,
            "uplink_count": self.facts.uplink_count,
            "uplink_speed": self.facts.uplink_speed,
            "uplink_speed_gbps": self.facts.uplink_speed_gbps,
            "uplink_media": self.facts.uplink_media,
            "poe_supported": self.facts.poe_supported,
            "poe_budget_w": self.facts.poe_budget_w,
            "poe_standard": self.facts.poe_standard,
            "l2_supported": self.facts.l2_supported,
            "l3_supported": self.facts.l3_supported,
            "stacking_supported": self.facts.stacking_supported,
            "airflow": self.facts.airflow,
            "redundant_psu": self.facts.redundant_psu,
            "transceiver_form_factor": self.facts.transceiver_form_factor,
            "fit_label": self.fit_label,
            "fit_reason": self.fit_reason,
            "fit_tier": self.fit_tier,
            "match_warnings": self.match_warnings,
            "uncertainty_reasons": self.uncertainty_reasons,
            "objective_reject_reason": self.objective_reject_reason,
            "evidence_summary": self.evidence_summary,
            "inclusion_policy": self.inclusion_policy,
            "over_requirement": _component_over_requirement_value(self),
            "eligibility_status": self.eligibility_status,
            "eligibility_warnings": self.eligibility_warnings,
            "fit_reasons": self.fit_reasons,
            "score": self.score,
            "selection_bucket": self.selection_bucket,
            "bucket_priority": self.bucket_priority,
        }


@dataclass(frozen=True)
class _CandidateMatrix:
    normalized_requirements: _NormalizedServerRequirements
    ready_server_candidates: list[_ComponentCandidate] = field(default_factory=list)
    platform_candidates: list[_ComponentCandidate] = field(default_factory=list)
    cpu_candidates: list[_ComponentCandidate] = field(default_factory=list)
    ram_candidates: list[_ComponentCandidate] = field(default_factory=list)
    drive_candidates: list[_ComponentCandidate] = field(default_factory=list)
    ssd_candidates: list[_ComponentCandidate] = field(default_factory=list)
    hdd_candidates: list[_ComponentCandidate] = field(default_factory=list)
    storage_controller_candidates: list[_ComponentCandidate] = field(default_factory=list)
    network_adapter_candidates: list[_ComponentCandidate] = field(default_factory=list)
    generic_role_candidates: dict[str, list[_ComponentCandidate]] = field(default_factory=dict)
    coverage_summary: dict[str, Any] = field(default_factory=dict)
    role_plan: dict[str, Any] = field(default_factory=dict)
    category_plan: dict[str, list[str]] = field(default_factory=dict)
    category_plan_entries: list[dict[str, Any]] = field(default_factory=list)
    category_catalog_summary: dict[str, Any] = field(default_factory=dict)
    category_planner_source: str = "none"
    category_plan_source: str = "none"
    category_planner_missing_required_roles: list[str] = field(default_factory=list)
    category_planner_repair_attempted: bool = False
    category_planner_repair_success: bool = False
    category_planner_repair_reason: str | None = None
    category_planner_repaired_roles: list[str] = field(default_factory=list)
    category_planner_unresolved_required_roles: list[str] = field(default_factory=list)
    required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    optional_capabilities: list[dict[str, Any]] = field(default_factory=list)
    unsupported_or_unmapped_requirements: list[str] = field(default_factory=list)
    required_roles: list[str] = field(default_factory=list)
    missing_required_roles: list[str] = field(default_factory=list)
    missing_category_roles: list[str] = field(default_factory=list)
    missing_required_capabilities: list[dict[str, Any]] = field(default_factory=list)
    role_coverage_summary: dict[str, Any] = field(default_factory=dict)
    category_plan_warnings: list[str] = field(default_factory=list)
    stage_a_broad_roles: list[str] = field(default_factory=list)
    semantic_matrix_blueprint_roles: list[str] = field(default_factory=list)
    requirement_classifier_roles: list[str] = field(default_factory=list)
    effective_matrix_roles_before_category_planner: list[str] = field(default_factory=list)
    category_planner_input_roles: list[str] = field(default_factory=list)
    category_planner_output_roles: list[str] = field(default_factory=list)
    validated_category_plan_roles: list[str] = field(default_factory=list)
    materialized_matrix_roles: list[str] = field(default_factory=list)
    roles_dropped_after_stage_a: list[str] = field(default_factory=list)
    roles_dropped_before_category_planner: list[str] = field(default_factory=list)
    roles_dropped_after_category_planner: list[str] = field(default_factory=list)
    roles_dropped_during_materialization: list[str] = field(default_factory=list)
    roles_dropped_reason_by_role: dict[str, str] = field(default_factory=dict)
    role_source_by_role: dict[str, list[str]] = field(default_factory=dict)
    role_lifecycle_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_report_json(self) -> dict[str, Any]:
        dynamic_rows = {
            f"{role}_candidates": _candidate_rows(candidates)
            for role, candidates in self.generic_role_candidates.items()
        }
        return {
            "product_group": self.normalized_requirements.product_group,
            "normalized_requirements": self.normalized_requirements.to_report_json(),
            "role_plan": _jsonable(self.role_plan),
            **_semantic_report_fields(self.role_plan),
            "category_plan": _jsonable(self.category_plan),
            "category_plan_entries": _jsonable(self.category_plan_entries),
            "category_catalog_summary": _jsonable(self.category_catalog_summary),
            "category_planner_source": self.category_planner_source,
            "category_plan_source": self.category_plan_source,
            "category_planner_missing_required_roles": (
                self.category_planner_missing_required_roles
            ),
            "category_planner_repair_attempted": self.category_planner_repair_attempted,
            "category_planner_repair_success": self.category_planner_repair_success,
            "category_planner_repair_reason": self.category_planner_repair_reason,
            "category_planner_repaired_roles": self.category_planner_repaired_roles,
            "category_planner_unresolved_required_roles": (
                self.category_planner_unresolved_required_roles
            ),
            "required_capabilities": _jsonable(self.required_capabilities),
            "optional_capabilities": _jsonable(self.optional_capabilities),
            "unsupported_or_unmapped_requirements": _jsonable(
                self.unsupported_or_unmapped_requirements
            ),
            "required_roles": self.required_roles,
            "missing_required_roles": self.missing_required_roles,
            "missing_required_roles_before_llm": self.missing_required_roles,
            "missing_category_roles": self.missing_category_roles,
            "missing_required_capabilities": _jsonable(
                self.missing_required_capabilities
            ),
            "missing_required_capabilities_before_llm": _jsonable(
                self.missing_required_capabilities
            ),
            "role_coverage_summary": _jsonable(self.role_coverage_summary),
            "category_plan_warnings": self.category_plan_warnings,
            "stage_a_broad_roles": self.stage_a_broad_roles,
            "semantic_matrix_blueprint_roles": self.semantic_matrix_blueprint_roles,
            "requirement_classifier_roles": self.requirement_classifier_roles,
            "effective_matrix_roles_before_category_planner": (
                self.effective_matrix_roles_before_category_planner
            ),
            "category_planner_input_roles": self.category_planner_input_roles,
            "category_planner_output_roles": self.category_planner_output_roles,
            "validated_category_plan_roles": self.validated_category_plan_roles,
            "materialized_matrix_roles": self.materialized_matrix_roles,
            "roles_dropped_after_stage_a": self.roles_dropped_after_stage_a,
            "roles_dropped_before_category_planner": (
                self.roles_dropped_before_category_planner
            ),
            "roles_dropped_after_category_planner": (
                self.roles_dropped_after_category_planner
            ),
            "roles_dropped_during_materialization": (
                self.roles_dropped_during_materialization
            ),
            "roles_dropped_reason_by_role": self.roles_dropped_reason_by_role,
            "role_source_by_role": self.role_source_by_role,
            "role_lifecycle_trace": _jsonable(self.role_lifecycle_trace),
            "ready_server_candidates": _candidate_rows(self.ready_server_candidates),
            "platform_candidates": _candidate_rows(self.platform_candidates),
            "cpu_candidates": _candidate_rows(self.cpu_candidates),
            "ram_candidates": _candidate_rows(self.ram_candidates),
            "drive_candidates": _candidate_rows(self.drive_candidates),
            "ssd_candidates": _candidate_rows(self.ssd_candidates),
            "hdd_candidates": _candidate_rows(self.hdd_candidates),
            "storage_controller_candidates": _candidate_rows(
                self.storage_controller_candidates
            ),
            "network_adapter_candidates": _candidate_rows(self.network_adapter_candidates),
            **dynamic_rows,
            "component_matrix_coverage_summary": _jsonable(self.coverage_summary),
            "note": (
                "Broad bucketed matrix for LLM Composer; application code "
                "validates selected IDs."
            ),
        }


@dataclass(frozen=True)
class _BuildComponent:
    role: str
    role_ru: str
    product: DistributorProduct
    quantity_required: int
    available_quantity: int | None
    reservable_locations: int
    price_value: Decimal | None
    price_currency: str | None
    facts: _ProductFacts | None = None
    component_candidate_id: str | None = None
    fit_label: str = FIT_UNKNOWN
    fit_reason: str = ""
    cpu_over_requirement: int | None = None
    storage_over_requirement: float | None = None
    ram_over_requirement_gb: int | None = None

    def to_report_json(self) -> dict[str, Any]:
        line_total = None
        if self.price_value is not None:
            line_total = self.price_value * self.quantity_required

        row = {
            "role": self.role,
            "role_ru": self.role_ru,
            "component_candidate_id": self.component_candidate_id,
            "category_id": self.product.category_id,
            "distributor_code": self.product.distributor_code,
            "item_id": self.product.item_id,
            "product_key": self.product.product_key,
            "part_number": self.product.part_number,
            "producer": self.product.producer,
            "item_name": self.product.item_name,
            "quantity_required": self.quantity_required,
            "available_quantity": self.available_quantity,
            "reservable_locations": self.reservable_locations,
            "price_value": _jsonable(self.price_value),
            "price_currency": self.price_currency,
            "line_total_value": _jsonable(line_total),
            "line_total_currency": self.price_currency if line_total is not None else None,
            "fit_label": self.fit_label,
            "fit_reason": self.fit_reason,
            "cpu_over_requirement": self.cpu_over_requirement,
            "storage_over_requirement": self.storage_over_requirement,
            "ram_over_requirement_gb": self.ram_over_requirement_gb,
        }
        if self.facts is not None:
            row["facts"] = self.facts.to_report_json()
            row["cpu_cores"] = self.facts.cpu_cores
            row["storage_capacity_tb"] = self.facts.storage_capacity_tb
            row["ram_module_capacity_gb"] = self.facts.ram_capacity_gb
            row["network_ports_count"] = self.facts.network_ports_count
            row["network_speed"] = self.facts.network_speed
            row["network_media"] = self.facts.network_media
            row["network_interface"] = self.facts.network_interface
        return row


async def match_stock_spec(
    spec: StockSpec,
    session: AsyncSession | None = None,
    *,
    llm_configurator_client: LlmClient | None = None,
    llm_settings: LlmSettings | None = None,
    web_evidence_settings: WebEvidenceSettings | None = None,
    web_search_provider: WebSearchProvider | None = None,
    evidence_cache: EvidenceSearchCache | None = None,
) -> MatchResult:
    if session is not None:
        return await _match_stock_spec_with_session(
            spec,
            session,
            llm_configurator_client=llm_configurator_client,
            llm_settings=llm_settings,
            web_evidence_settings=web_evidence_settings,
            web_search_provider=web_search_provider,
            evidence_cache=evidence_cache,
        )

    session_factory = get_session_factory()
    async with session_factory() as owned_session:
        return await _match_stock_spec_with_session(
            spec,
            owned_session,
            llm_configurator_client=llm_configurator_client,
            llm_settings=llm_settings,
            web_evidence_settings=web_evidence_settings,
            web_search_provider=web_search_provider,
            evidence_cache=evidence_cache,
        )


def extract_stock_spec_for_text_match(text: str) -> StockSpecExtractionResult:
    """Build the deterministic StockSpec seed used by the semantic planner."""

    extraction = extract_stock_spec_from_text(
        text,
        settings=LlmSettings(llm_provider="disabled"),
    )
    spec = extraction.spec_json
    if spec.source_text != text:
        spec = spec.model_copy(update={"source_text": text})
    return extraction.model_copy(update={"spec_json": spec})


def plan_semantic_matrix_for_text(
    text: str,
    *,
    llm_settings: LlmSettings | None = None,
    distributor_code: str = "ocs",
) -> dict[str, Any]:
    """Run only the semantic matrix planner for a text request."""

    extraction = extract_stock_spec_for_text_match(text)
    spec = extraction.spec_json
    effective_llm_settings = llm_settings or get_llm_settings()
    planner_state = _build_semantic_planner_llm_client(
        llm_settings=effective_llm_settings,
        llm_configurator_client=None,
    )
    try:
        role_plan = plan_semantic_matrix_roles(
            spec,
            distributor_code=distributor_code,
            planner_client=planner_state.client,
            deterministic_product_group_hint=_detect_spec_product_group(spec),
            semantic_planner_max_seconds=(
                effective_llm_settings.llm_semantic_planner_max_seconds
            ),
            semantic_planner_stage_timeout_seconds=(
                effective_llm_settings.llm_semantic_planner_stage_timeout_seconds
            ),
        )
        return _with_semantic_runtime_diagnostics(role_plan, planner_state)
    finally:
        close = getattr(planner_state.client, "close", None)
        if callable(close):
            close()


async def prepare_match_planning_context(
    spec: StockSpec,
    session: AsyncSession,
    *,
    llm_configurator_client: LlmClient | None = None,
    llm_settings: LlmSettings | None = None,
) -> PlannedMatchContext:
    products = await _list_candidate_products(session, spec)
    role_plan, category_plan_result = await _build_product_group_plans(
        session,
        spec=spec,
        products=products,
        llm_settings=llm_settings,
        llm_configurator_client=llm_configurator_client,
    )
    stock_rows_by_key = await _load_latest_stock_rows(session, products)
    product_group = str(role_plan.get("product_group") or _detect_spec_product_group(spec))
    (
        build_candidates,
        build_missing,
        build_risks,
        component_candidate_matrix,
        normalized_requirements,
    ) = _build_configuration_candidates(
        spec=spec,
        products=products,
        stock_rows_by_key=stock_rows_by_key,
        matrix_candidates_per_role=_matrix_candidates_per_role(llm_settings),
        role_plan=role_plan,
        category_plan_result=category_plan_result,
    )
    component_candidate_matrix = await _distill_component_matrix_if_needed(
        session=session,
        spec=spec,
        products=products,
        component_candidate_matrix=component_candidate_matrix,
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        rule_based_build_candidates=[candidate.to_report_json() for candidate in build_candidates],
        role_plan=role_plan,
        llm_settings=llm_settings,
    )
    return PlannedMatchContext(
        spec=spec,
        product_group=product_group,
        products=products,
        stock_rows_by_key=stock_rows_by_key,
        role_plan=role_plan,
        category_plan_result=category_plan_result,
        build_candidates=build_candidates,
        build_missing=build_missing,
        build_risks=build_risks,
        component_candidate_matrix=component_candidate_matrix,
        normalized_requirements=normalized_requirements,
    )


async def build_llm_configurator_package_from_text(
    text: str,
    session: AsyncSession,
    *,
    candidates_per_role: int | None = None,
    llm_settings: LlmSettings | None = None,
) -> dict[str, Any]:
    extraction = extract_stock_spec_for_text_match(text)
    match_result = await match_stock_spec(
        extraction.spec_json,
        session,
        llm_settings=llm_settings,
    )
    report_json = match_result.to_report_json()
    return build_llm_configurator_package_from_report_json(
        report_json,
        user_request=text,
        candidates_per_role=candidates_per_role,
        llm_settings=llm_settings,
    )


def build_llm_configurator_package_from_report_json(
    report_json: Mapping[str, Any],
    *,
    user_request: str | None = None,
    candidates_per_role: int | None = None,
    llm_settings: LlmSettings | None = None,
) -> dict[str, Any]:
    """Build the Composer package from the persisted/report JSON shape."""

    effective_llm_settings = llm_settings or get_llm_settings()
    component_candidate_matrix = _component_matrix_for_configurator_package(report_json)
    ready_stock_candidates = report_json.get("ready_stock_candidates", [])
    rule_based_build_candidates = report_json.get("build_candidates", [])
    if _normal_configurator_package_must_be_skipped(report_json):
        ready_stock_candidates = []
        rule_based_build_candidates = []
    return build_llm_configurator_package(
        user_request=user_request or _text_or_none(report_json.get("source_text")),
        normalized_requirements=report_json.get("normalized_requirements"),
        ready_stock_candidates=ready_stock_candidates,
        component_candidate_matrix=component_candidate_matrix,
        rule_based_build_candidates=rule_based_build_candidates,
        candidates_per_role=candidates_per_role
        or _matrix_candidates_per_role(effective_llm_settings),
        max_package_chars=effective_llm_settings.llm_configurator_max_package_chars,
    )


def _component_matrix_for_configurator_package(
    report_json: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = _dict_or_empty(report_json.get("component_candidate_matrix"))
    role_plan = _dict_or_empty(report_json.get("role_plan"))
    if role_plan and not matrix.get("role_plan"):
        matrix["role_plan"] = role_plan
    semantic_fields = _semantic_report_fields(role_plan)
    for key, value in {
        **semantic_fields,
        "product_group": report_json.get("product_group"),
        "required_capabilities": report_json.get("required_capabilities"),
        "optional_capabilities": report_json.get("optional_capabilities"),
        "unsupported_or_unmapped_requirements": report_json.get(
            "unsupported_or_unmapped_requirements"
        ),
        "classified_requirements": report_json.get("classified_requirements"),
        "primary_object_feature_requirements": report_json.get(
            "primary_object_feature_requirements"
        ),
        "requirement_classifier_incomplete_reason": report_json.get(
            "requirement_classifier_incomplete_reason"
        ),
        "requirement_source_coverage": report_json.get("requirement_source_coverage"),
        "requirement_source_coverage_percent": report_json.get(
            "requirement_source_coverage_percent"
        ),
        "unclassified_source_fragments": report_json.get(
            "unclassified_source_fragments"
        ),
        "synthetic_requirement_count": report_json.get("synthetic_requirement_count"),
        "source_backed_requirement_count": report_json.get(
            "source_backed_requirement_count"
        ),
        "requirement_classifier_repair_quality": report_json.get(
            "requirement_classifier_repair_quality"
        ),
        "requirement_classifier_repair_accepted": report_json.get(
            "requirement_classifier_repair_accepted"
        ),
        "engineering_check_requirements": report_json.get("engineering_check_requirements"),
        "logistics_or_commercial_constraints": report_json.get(
            "logistics_or_commercial_constraints"
        ),
        "required_roles": report_json.get("required_roles"),
        "category_plan": report_json.get("category_plan"),
        "category_plan_entries": report_json.get("category_plan_entries"),
        "category_catalog_summary": report_json.get("category_catalog_summary"),
        "category_planner_source": report_json.get("category_planner_source"),
        "category_plan_source": report_json.get("category_plan_source"),
        "category_planner_missing_required_roles": report_json.get(
            "category_planner_missing_required_roles"
        ),
        "category_planner_repair_attempted": report_json.get(
            "category_planner_repair_attempted"
        ),
        "category_planner_repair_success": report_json.get(
            "category_planner_repair_success"
        ),
        "category_planner_repair_reason": report_json.get(
            "category_planner_repair_reason"
        ),
        "category_planner_repaired_roles": report_json.get(
            "category_planner_repaired_roles"
        ),
        "category_planner_unresolved_required_roles": report_json.get(
            "category_planner_unresolved_required_roles"
        ),
        "category_plan_warnings": report_json.get("category_plan_warnings"),
        "missing_required_roles": report_json.get("missing_required_roles"),
        "missing_category_roles": report_json.get("missing_category_roles"),
        "missing_required_capabilities": report_json.get("missing_required_capabilities"),
        "provider_error_type": report_json.get("provider_error_type"),
        "provider_context_limit": report_json.get("provider_context_limit"),
        "full_matrix_evaluation_fallback_reason": report_json.get(
            "full_matrix_evaluation_fallback_reason"
        ),
    }.items():
        if matrix.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
            matrix[key] = value
    return matrix


def _normal_configurator_package_must_be_skipped(
    report_json: Mapping[str, Any],
) -> bool:
    source = str(report_json.get("semantic_planner_source") or "").strip()
    fallback_reason = str(
        report_json.get("semantic_planner_fallback_reason") or ""
    ).strip()
    product_group = str(report_json.get("product_group") or "").strip()
    if (
        source
        in (
            SEMANTIC_PLANNER_COMPLEX_FALLBACK_REASON,
            SEMANTIC_PLANNER_TIMEOUT_SOURCE,
        )
        or fallback_reason == SEMANTIC_PLANNER_COMPLEX_FALLBACK_REASON
        or (
            fallback_reason == SEMANTIC_PLANNER_TIMEOUT_REASON
            and product_group == "unknown"
        )
    ):
        return True
    required_capabilities = _mapping_list(report_json.get("required_capabilities"))
    category_plan = _dict_or_empty(report_json.get("category_plan"))
    if required_capabilities or category_plan:
        return False
    return _looks_like_technical_text(
        report_json.get("spec"),
        report_json.get("normalized_requirements"),
    )


def _looks_like_technical_text(*values: Any) -> bool:
    text = " ".join(
        str(value)
        for value in values
        if value not in (None, "", [], {})
    ).casefold()
    return bool(
        re.search(
            r"\b(?:server|switch|router|firewall|cpu|xeon|epyc|ram|rdimm|ddr[345]|"
            r"ssd|hdd|nvme|sata|sas|raid|hba|nic|sfp\+?|qsfp|10gbe|25gbe|"
            r"c13|c14|psu|storage|nas|san)\b|"
            r"СЃРµСЂРІРµСЂ|РєРѕРјРјСѓС‚Р°С‚РѕСЂ|СЃС…Рґ|РїСЂРѕС†РµСЃСЃРѕСЂ|"
            r"РѕРїРµСЂР°С‚РёРІ|РґРёСЃРє|СЃРµС‚РµРІ|РїРёС‚Р°РЅ",
            text,
            re.I,
        )
    )


def _log_full_matrix_progress(event: str, fields: Mapping[str, Any] | None = None) -> None:
    parts = [str(event)]
    for key, value in (fields or {}).items():
        if value in (None, "", [], {}):
            continue
        text = str(value).replace("\n", " ").replace("\r", " ")
        parts.append(f"{key}={text}")
    print(" ".join(parts), file=sys.stderr, flush=True)


async def _distill_component_matrix_if_needed(
    *,
    session: AsyncSession,
    spec: StockSpec,
    products: list[DistributorProduct],
    component_candidate_matrix: dict[str, Any],
    normalized_requirements: list[dict[str, Any]],
    ready_stock_candidates: list[Mapping[str, Any]],
    rule_based_build_candidates: list[Mapping[str, Any]],
    role_plan: Mapping[str, Any],
    llm_settings: LlmSettings | None,
    full_matrix_trigger_reason: str | None = None,
    allow_broad_under_budget_fallback: bool = True,
) -> dict[str, Any]:
    if not component_candidate_matrix:
        return component_candidate_matrix
    product_group = str(
        component_candidate_matrix.get("product_group")
        or role_plan.get("product_group")
        or ""
    ).strip()
    if product_group != SERVER_PRODUCT_GROUP:
        return _matrix_with_distiller_metadata(
            component_candidate_matrix,
            used=False,
            source="skipped",
            diagnostics={
                "reason": "non_server_product_group",
                "broad_count_by_role": _server_broad_count_by_role(component_candidate_matrix),
            },
        )

    settings = llm_settings or get_llm_settings()
    budget = settings.llm_configurator_max_package_chars
    force_full_matrix = bool(getattr(settings, "llm_full_matrix_force", False))
    high_quality_full_matrix_by_default = bool(
        getattr(settings, "high_quality_full_matrix_by_default", True)
    )
    estimate = build_llm_configurator_package(
        user_request=spec.source_text,
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=ready_stock_candidates,
        component_candidate_matrix=component_candidate_matrix,
        rule_based_build_candidates=rule_based_build_candidates,
        candidates_per_role=_matrix_candidates_per_role(settings),
        max_package_chars=budget,
    )
    estimate_budget = _dict_or_empty(estimate.get("package_budget"))
    broad_package_under_budget = estimate_budget.get("over_budget") is False
    _log_full_matrix_progress(
        "package_budget",
        {
            "stage": "broad_package",
            "over_budget": estimate_budget.get("over_budget"),
            "trimmed": estimate_budget.get("trimmed"),
            "final_chars": estimate_budget.get("final_chars"),
            "max_chars": estimate_budget.get("max_chars"),
            "package_skipped_reason": estimate.get("package_skipped_reason"),
            "force_full_matrix": force_full_matrix,
            "high_quality_full_matrix_by_default": high_quality_full_matrix_by_default,
            "full_matrix_trigger_reason": full_matrix_trigger_reason,
        },
    )
    if (
        broad_package_under_budget
        and not estimate.get("package_skipped_reason")
        and not force_full_matrix
    ):
        return _matrix_with_distiller_metadata(
            component_candidate_matrix,
            used=False,
            source="skipped",
            diagnostics={
                "reason": HIGH_QUALITY_BROAD_PACKAGE_UNDER_LIMIT_REASON,
                "full_matrix_evaluation_used": False,
                "full_matrix_evaluation_fallback_reason": (
                    SKIPPED_FULL_BROAD_PACKAGE_UNDER_HIGH_QUALITY_LIMIT_REASON
                ),
                "broad_count_by_role": _server_broad_count_by_role(
                    component_candidate_matrix
                ),
                "package_budget": estimate_budget,
                "package_skipped_reason": None,
                "force_full_matrix": False,
                "high_quality_full_matrix_by_default": high_quality_full_matrix_by_default,
            },
        )
    if not force_full_matrix and estimate_budget.get("over_budget") is not True:
        return _matrix_with_distiller_metadata(
            component_candidate_matrix,
            used=False,
            source="skipped",
            diagnostics={
                "reason": "package_within_budget_or_not_distillable",
                "broad_count_by_role": _server_broad_count_by_role(component_candidate_matrix),
                "package_budget": estimate_budget,
                "package_skipped_reason": estimate.get("package_skipped_reason"),
            },
        )

    _log_full_matrix_progress(
        "full_matrix_start",
        {
            "product_group": product_group,
            "max_seconds": settings.llm_full_matrix_max_seconds,
            "chunk_timeout_seconds": settings.llm_full_matrix_chunk_timeout_seconds,
            "broad_package_under_budget": broad_package_under_budget,
            "force_full_matrix": force_full_matrix,
            "full_matrix_trigger_reason": full_matrix_trigger_reason,
        },
    )
    enriched_matrix = dict(component_candidate_matrix)
    content_diagnostics = await enrich_matrix_with_ocs_content(
        session=session,
        component_candidate_matrix=enriched_matrix,
        products=products,
    )
    distiller_state = _build_matrix_distiller_llm_client(settings)
    if distiller_state.client is None:
        diagnostics = {
            "matrix_distiller_source": "skipped",
            "reason": distiller_state.unavailable_reason or "llm_unavailable",
            "error_type": distiller_state.error_type,
            "stage": "full_matrix_evaluation_client",
            "role": None,
            "chunk_index": None,
            "broad_count_by_role": _server_broad_count_by_role(enriched_matrix),
            "package_budget_before_distillation": estimate_budget,
            "package_budget_at_failure": estimate_budget,
            "ocs_content": content_diagnostics,
        }
        if broad_package_under_budget and allow_broad_under_budget_fallback:
            _log_full_matrix_progress(
                "fallback_to_broad_package",
                {
                    "reason": FULL_MATRIX_UNAVAILABLE_BUT_PACKAGE_UNDER_BUDGET_REASON,
                    "package_budget_over_budget": estimate_budget.get("over_budget"),
                },
            )
            return _matrix_distiller_preferred_fallback_to_broad_package(
                enriched_matrix,
                diagnostics={
                    **diagnostics,
                    "fallback_decision": "use_full_broad_package_under_high_quality_limit",
                },
                fallback_reason=FULL_MATRIX_UNAVAILABLE_BUT_PACKAGE_UNDER_BUDGET_REASON,
            )
        return _matrix_distiller_fail_closed(
            enriched_matrix,
            reason=INCOMPLETE_MATRIX_EXPOSURE_REASON,
            diagnostics={
                **diagnostics,
                "incomplete_matrix_exposure_reason": "full_matrix_evaluation_unavailable",
                "fallback_decision": "block_composer_package_over_budget",
            },
        )

    try:
        distilled = distill_component_candidate_matrix(
            product_group=product_group,
            component_candidate_matrix=enriched_matrix,
            constraints_by_role=constraints_by_role_from_role_plan(role_plan),
            llm_client=distiller_state.client,
            max_seconds=settings.llm_full_matrix_max_seconds,
            chunk_timeout_seconds=settings.llm_full_matrix_chunk_timeout_seconds,
            progress_callback=_log_full_matrix_progress,
        )
    except (MatrixDistillerError, LlmError) as exc:
        diagnostics = _matrix_distiller_failure_diagnostics(
            exc,
            matrix=enriched_matrix,
            estimate_budget=estimate_budget,
            content_diagnostics=content_diagnostics,
        )
        is_timeout = isinstance(exc, MatrixDistillerTimeoutError)
        if broad_package_under_budget and allow_broad_under_budget_fallback:
            fallback_reason = (
                FULL_MATRIX_TIMEOUT_BUT_PACKAGE_UNDER_BUDGET_REASON
                if is_timeout
                else FULL_MATRIX_FAILED_BUT_PACKAGE_UNDER_BUDGET_REASON
            )
            _log_full_matrix_progress(
                "fallback_to_broad_package",
                {
                    "reason": fallback_reason,
                    "package_budget_over_budget": estimate_budget.get("over_budget"),
                },
            )
            return _matrix_distiller_preferred_fallback_to_broad_package(
                enriched_matrix,
                diagnostics={
                    **diagnostics,
                    "fallback_decision": "use_full_broad_package_under_high_quality_limit",
                },
                fallback_reason=fallback_reason,
            )
        closed_reason = (
            FULL_MATRIX_TIMEOUT_PACKAGE_OVER_BUDGET_REASON
            if is_timeout
            else FULL_MATRIX_OVER_BUDGET_FAILURE_SKIP_REASON
        )
        return _matrix_distiller_fail_closed(
            enriched_matrix,
            reason=INCOMPLETE_MATRIX_EXPOSURE_REASON,
            diagnostics={
                **diagnostics,
                "incomplete_matrix_exposure_reason": closed_reason,
                "fallback_decision": "block_composer_package_over_budget",
            },
        )
    finally:
        close = getattr(distiller_state.client, "close", None)
        if callable(close):
            close()

    distilled_matrix = dict(distilled.component_candidate_matrix)
    diagnostics = {
        **distilled.diagnostics,
        "package_budget_before_distillation": estimate_budget,
        "ocs_content": content_diagnostics,
    }
    distilled_matrix["matrix_distiller_diagnostics"] = diagnostics
    final_estimate = build_llm_configurator_package(
        user_request=spec.source_text,
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=ready_stock_candidates,
        component_candidate_matrix=distilled_matrix,
        rule_based_build_candidates=rule_based_build_candidates,
        candidates_per_role=_matrix_candidates_per_role(settings),
        max_package_chars=budget,
    )
    _log_full_matrix_progress(
        "package_budget",
        {
            "stage": "after_full_matrix",
            "over_budget": (final_estimate.get("package_budget") or {}).get("over_budget"),
            "trimmed": (final_estimate.get("package_budget") or {}).get("trimmed"),
            "final_chars": (final_estimate.get("package_budget") or {}).get("final_chars"),
            "max_chars": (final_estimate.get("package_budget") or {}).get("max_chars"),
            "package_skipped_reason": final_estimate.get("package_skipped_reason"),
        },
    )
    if (final_estimate.get("package_budget") or {}).get("over_budget"):
        distilled_matrix["package_skipped_reason"] = "package_over_budget_after_distillation"
        distilled_matrix["llm_fallback_reason"] = "package_over_budget_after_distillation"
        distilled_matrix["matrix_distiller_diagnostics"] = {
            **diagnostics,
            "package_budget_after_distillation": final_estimate.get("package_budget") or {},
            "package_skipped_reason": "package_over_budget_after_distillation",
        }
    return distilled_matrix


def _build_matrix_distiller_llm_client(settings: LlmSettings) -> _PlannerClientState:
    provider = settings.llm_provider.strip().lower()
    model = settings.llm_model.strip() or None
    if provider == "disabled":
        return _PlannerClientState(
            provider=provider,
            model=model,
            unavailable_reason="llm_provider_disabled",
        )
    if provider not in SEMANTIC_PLANNER_SUPPORTED_PROVIDERS:
        return _PlannerClientState(
            provider=provider or None,
            model=model,
            unavailable_reason="llm_provider_unsupported",
        )
    if not (settings.llm_base_url.strip() and settings.llm_api_key.strip() and model):
        return _PlannerClientState(
            provider=provider,
            model=model,
            unavailable_reason="llm_settings_incomplete",
        )
    full_matrix_chunk_timeout = max(
        0.001,
        float(settings.llm_full_matrix_chunk_timeout_seconds or 0),
    )
    try:
        return _PlannerClientState(
            client=OpenAICompatibleLlmClient(
                settings=settings,
                timeout_seconds=max(
                    0.001,
                    min(
                        float(settings.llm_configurator_timeout_seconds or 0),
                        full_matrix_chunk_timeout,
                    ),
                ),
                read_timeout_seconds=max(
                    0.001,
                    min(
                        float(settings.llm_configurator_read_timeout_seconds or 0),
                        full_matrix_chunk_timeout,
                    ),
                ),
                max_output_tokens=min(settings.llm_configurator_max_output_tokens, 4096),
                use_response_format=True,
                thinking_enabled=False,
                max_retries=0,
            ),
            provider=provider,
            model=model,
        )
    except LlmError as exc:
        return _PlannerClientState(
            provider=provider,
            model=model,
            unavailable_reason="llm_client_build_failed",
            error_type=type(exc).__name__,
        )


def _matrix_distiller_failure_diagnostics(
    exc: LlmError,
    *,
    matrix: Mapping[str, Any],
    estimate_budget: Mapping[str, Any],
    content_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostics = {
        "matrix_distiller_source": "error",
        "error_type": type(exc).__name__,
        "stage": str(getattr(exc, "stage", None) or "full_matrix_evaluation"),
        "role": getattr(exc, "role", None),
        "chunk_index": getattr(exc, "chunk_index", None),
        "parse_status": getattr(exc, "parse_status", None),
        "http_status": getattr(exc, "http_status", None),
        "broad_count_by_role": _server_broad_count_by_role(matrix),
        "package_budget_before_distillation": dict(estimate_budget),
        "package_budget_at_failure": dict(estimate_budget),
        "ocs_content": dict(content_diagnostics),
    }
    cause_error_type = str(getattr(exc, "cause_error_type", "") or "").strip()
    if cause_error_type:
        diagnostics["cause_error_type"] = cause_error_type
    timeout_kind = str(getattr(exc, "timeout_kind", "") or "").strip()
    if timeout_kind:
        diagnostics["timeout_kind"] = timeout_kind
    timeout_seconds = getattr(exc, "timeout_seconds", None)
    if timeout_seconds is not None:
        diagnostics["timeout_seconds"] = timeout_seconds
    deadline_seconds = getattr(exc, "deadline_seconds", None)
    if deadline_seconds is not None:
        diagnostics["deadline_seconds"] = deadline_seconds
    elapsed_seconds = getattr(exc, "elapsed_seconds", None)
    if elapsed_seconds is not None:
        diagnostics["elapsed_seconds"] = elapsed_seconds
    failed_chunks = getattr(exc, "failed_chunks", None)
    if failed_chunks:
        diagnostics["full_matrix_failed_chunks"] = [
            dict(row) for row in failed_chunks if isinstance(row, Mapping)
        ]
    return diagnostics


def _matrix_distiller_preferred_fallback_to_broad_package(
    matrix: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Any],
    fallback_reason: str,
) -> dict[str, Any]:
    result = _matrix_with_distiller_metadata(
        matrix,
        used=False,
        source=str(diagnostics.get("matrix_distiller_source") or "error"),
        diagnostics={
            **dict(diagnostics),
            "full_matrix_evaluation_used": False,
            "full_matrix_evaluation_fallback_reason": fallback_reason,
        },
    )
    result["full_matrix_evaluation_used"] = False
    result["full_matrix_evaluation_fallback_reason"] = fallback_reason
    result["full_matrix_failed_chunks"] = _mapping_list(
        diagnostics.get("full_matrix_failed_chunks")
    )
    result.pop("package_skipped_reason", None)
    result.pop("llm_fallback_reason", None)
    result["broad_count_by_role"] = _server_broad_count_by_role(matrix)
    result.setdefault("distilled_count_by_role", {})
    return result


def _matrix_with_distiller_metadata(
    matrix: Mapping[str, Any],
    *,
    used: bool,
    source: str,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(matrix)
    result["matrix_distiller_used"] = used
    result["matrix_distiller_source"] = source
    result["matrix_distiller_diagnostics"] = dict(diagnostics)
    result["full_matrix_evaluation_used"] = bool(
        diagnostics.get("full_matrix_evaluation_used") or used
    )
    fallback_reason = _text_or_none(
        diagnostics.get("full_matrix_evaluation_fallback_reason")
    )
    if fallback_reason:
        result["full_matrix_evaluation_fallback_reason"] = fallback_reason
    failed_chunks = _mapping_list(diagnostics.get("full_matrix_failed_chunks"))
    if failed_chunks:
        result["full_matrix_failed_chunks"] = failed_chunks
    result.setdefault("broad_count_by_role", _server_broad_count_by_role(matrix))
    if used:
        result.setdefault("distilled_count_by_role", _server_broad_count_by_role(result))
    return result


def _matrix_distiller_fail_closed(
    matrix: Mapping[str, Any],
    *,
    reason: str,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    result = _strip_candidate_rows(matrix)
    result.update(
        {
            "matrix_distiller_used": False,
            "matrix_distiller_source": str(diagnostics.get("matrix_distiller_source") or "error"),
            "matrix_distiller_diagnostics": dict(diagnostics),
            "full_matrix_evaluation_used": False,
            "full_matrix_evaluation_fallback_reason": reason,
            "full_matrix_failed_chunks": _mapping_list(
                diagnostics.get("full_matrix_failed_chunks")
            ),
            "broad_count_by_role": _server_broad_count_by_role(matrix),
            "distilled_count_by_role": {},
            "package_skipped_reason": reason,
            "llm_fallback_reason": reason,
        }
    )
    return result


def _matrix_distiller_compact_fallback_package(
    *,
    matrix: Mapping[str, Any],
    spec: StockSpec,
    normalized_requirements: list[dict[str, Any]],
    ready_stock_candidates: list[Mapping[str, Any]],
    rule_based_build_candidates: list[Mapping[str, Any]],
    settings: LlmSettings,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    _ = (spec, normalized_requirements, ready_stock_candidates, rule_based_build_candidates)
    _ = settings
    base_diagnostics = dict(diagnostics)
    original_distiller_error_type = str(base_diagnostics.get("error_type") or "").strip()
    if original_distiller_error_type:
        base_diagnostics["original_distiller_error_type"] = original_distiller_error_type
    return _matrix_distiller_fail_closed(
        matrix,
        reason=INCOMPLETE_MATRIX_EXPOSURE_REASON,
        diagnostics={
            **base_diagnostics,
            "fallback_compaction_attempted": False,
            "fallback_compaction_disabled_reason": "no_silent_top_n",
            "incomplete_matrix_exposure_reason": "matrix_distiller_failed_no_compact_fallback",
            "fallback_decision": "block_incomplete_matrix_exposure",
        },
    )


def _strip_candidate_rows(matrix: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in matrix.items():
        if str(key).endswith("_candidates") and isinstance(value, list):
            continue
        result[key] = value
    return result


def _server_broad_count_by_role(matrix: Mapping[str, Any]) -> dict[str, int]:
    role_keys = {
        "ready_server": "ready_server_candidates",
        "server_platform": "platform_candidates",
        "cpu": "cpu_candidates",
        "ram": "ram_candidates",
        "drive": "drive_candidates",
        "ssd": "ssd_candidates",
        "hdd": "hdd_candidates",
        "storage_controller": "storage_controller_candidates",
        "network_adapter": "network_adapter_candidates",
        "power_supply": "power_supply_candidates",
        "cable": "cable_candidates",
        "other_accessory": "other_accessory_candidates",
        "gpu": "gpu_candidates",
        "transceiver": "transceiver_candidates",
        "rail_kit": "rail_kit_candidates",
        "license": "license_candidates",
        "support": "support_candidates",
    }
    result: dict[str, int] = {}
    for role, key in role_keys.items():
        rows = matrix.get(key)
        if isinstance(rows, list) and rows:
            result[role] = len(rows)
    return result


async def _match_stock_spec_with_session(
    spec: StockSpec,
    session: AsyncSession,
    *,
    llm_configurator_client: LlmClient | None = None,
    llm_settings: LlmSettings | None = None,
    web_evidence_settings: WebEvidenceSettings | None = None,
    web_search_provider: WebSearchProvider | None = None,
    evidence_cache: EvidenceSearchCache | None = None,
) -> MatchResult:
    planning_context = await prepare_match_planning_context(
        spec=spec,
        session=session,
        llm_settings=llm_settings,
        llm_configurator_client=llm_configurator_client,
    )
    products = planning_context.products
    role_plan = planning_context.role_plan
    category_plan_result = planning_context.category_plan_result
    stock_rows_by_key = planning_context.stock_rows_by_key
    product_group = planning_context.product_group

    candidates: list[MatchCandidateResult] = []
    for item_index, item in enumerate(spec.items):
        for product in products:
            stock_rows = stock_rows_by_key.get((product.distributor_code, product.item_id), [])
            candidate = _evaluate_candidate(
                spec=spec,
                item=item,
                item_index=item_index,
                product=product,
                stock_rows=stock_rows,
                product_group=product_group,
            )
            if candidate is not None:
                candidates.append(candidate)

    build_candidates = planning_context.build_candidates
    build_missing = planning_context.build_missing
    build_risks = planning_context.build_risks
    component_candidate_matrix = planning_context.component_candidate_matrix
    normalized_requirements = planning_context.normalized_requirements
    llm_outcome = _compose_llm_configurations(
        spec=spec,
        ready_stock_candidates=candidates,
        build_candidates=build_candidates,
        component_candidate_matrix=component_candidate_matrix,
        normalized_requirements=normalized_requirements,
        llm_configurator_client=llm_configurator_client,
        llm_settings=llm_settings,
        web_evidence_settings=web_evidence_settings,
        web_search_provider=web_search_provider,
        evidence_cache=evidence_cache,
    )
    if _composer_context_limit_fallback_needed(llm_outcome):
        fallback_settings = (llm_settings or get_llm_settings()).model_copy(
            update={"llm_full_matrix_force": True}
        )
        component_candidate_matrix = await _distill_component_matrix_if_needed(
            session=session,
            spec=spec,
            products=products,
            component_candidate_matrix=component_candidate_matrix,
            normalized_requirements=normalized_requirements,
            ready_stock_candidates=[
                candidate.to_report_json() for candidate in candidates
            ],
            rule_based_build_candidates=[
                candidate.to_report_json() for candidate in build_candidates
            ],
            role_plan=role_plan,
            llm_settings=fallback_settings,
            full_matrix_trigger_reason=PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON,
            allow_broad_under_budget_fallback=False,
        )
        component_candidate_matrix = _with_provider_context_limit_fallback_metadata(
            component_candidate_matrix,
            llm_outcome,
        )
        llm_outcome = _compose_llm_configurations(
            spec=spec,
            ready_stock_candidates=candidates,
            build_candidates=build_candidates,
            component_candidate_matrix=component_candidate_matrix,
            normalized_requirements=normalized_requirements,
            llm_configurator_client=llm_configurator_client,
            llm_settings=llm_settings,
            web_evidence_settings=web_evidence_settings,
            web_search_provider=web_search_provider,
            evidence_cache=evidence_cache,
        )
    candidates.extend(build_candidates)

    candidates.sort(
        key=lambda candidate: (
            candidate.confidence_score,
            candidate.available_quantity or 0,
            candidate.reservable_locations,
        ),
        reverse=True,
    )

    missing_requirements = _unique(
        requirement
        for candidate in candidates
        for requirement in candidate.missing_requirements
    )
    missing_requirements.extend(build_missing)
    risk_flags = _unique(flag for candidate in candidates for flag in candidate.risk_flags)
    risk_flags.extend(build_risks)

    if not spec.items:
        missing_requirements.append("Запрос не содержит позиций для подбора.")
        risk_flags.append("Нужна ручная проверка: в запросе нет позиций.")

    matched_item_indexes = {
        int(candidate.raw["spec_item_index"])
        for candidate in candidates
        if candidate.is_full_match
    }
    matched_items = len(matched_item_indexes)

    if not candidates:
        status = STATUS_NO_STOCK_MATCH
        if spec.items:
            missing_requirements.append("Складские варианты не найдены.")
            risk_flags.append("Складской подбор не нашел товаров.")
    else:
        full_matches = [candidate for candidate in candidates if candidate.is_full_match]
        status = STATUS_STOCK_MATCHED if full_matches else STATUS_PARTIAL_STOCK_MATCHED
        if not full_matches:
            risk_flags.append(
                "Нет варианта, который полностью закрывает запрос по правилам текущей версии."
            )

    missing_required_capabilities = _merge_missing_capability_rows(
        _mapping_list(component_candidate_matrix.get("missing_required_capabilities")),
        _mapping_list(llm_outcome.no_recommendation_reason.get("missing_required_capabilities")),
        _mapping_list(llm_outcome.commercial_summary.get("missing_required_capabilities")),
    )
    no_recommendation_reason = _dict_or_empty(llm_outcome.no_recommendation_reason)
    commercial_summary = _dict_or_empty(llm_outcome.commercial_summary)
    if (
        llm_outcome.primary_recommendation_status == "no_recommendation"
        and missing_required_capabilities
    ):
        no_recommendation_reason = _no_recommendation_reason_with_missing_capabilities(
            no_recommendation_reason,
            missing_required_capabilities,
        )
        commercial_summary = _commercial_summary_with_missing_capabilities(
            commercial_summary,
            missing_required_capabilities,
        )
    primary_recommendation = llm_outcome.primary_recommendation
    primary_recommendation_status = llm_outcome.primary_recommendation_status
    if _semantic_planner_failed_closed(role_plan):
        missing_requirements.append(SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE)
        risk_flags.append(SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE)
        primary_recommendation = {}
        primary_recommendation_status = "no_recommendation"
        no_recommendation_reason = _semantic_planner_unavailable_reason(role_plan)
        commercial_summary = _semantic_planner_unavailable_commercial_summary()

    return MatchResult(
        spec=spec,
        status=status,
        engineer_review_required=True,
        total_candidates=len(candidates),
        matched_items=matched_items,
        missing_requirements=_unique(missing_requirements),
        risk_flags=_unique(risk_flags),
        candidates=candidates,
        component_candidate_matrix=component_candidate_matrix,
        normalized_requirements=normalized_requirements,
        llm_configurator_enabled=llm_outcome.enabled,
        llm_configurator_used=llm_outcome.used,
        output_mode=llm_outcome.output_mode,
        llm_recommended_build_candidates=llm_outcome.recommended_builds,
        primary_recommendation=primary_recommendation,
        primary_recommendation_status=primary_recommendation_status,
        no_recommendation_reason=no_recommendation_reason,
        commercial_summary=commercial_summary,
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
        llm_repair_blocked_critique_count=(
            llm_outcome.repair_blocked_critique_count
        ),
        llm_repair_blocked_critique_summary=(
            llm_outcome.repair_blocked_critique_summary
        ),
        llm_repair_savings_estimate=llm_outcome.repair_savings_estimate,
        llm_repair_revised_proposals_count=(
            llm_outcome.repair_revised_proposals_count
        ),
        llm_repair_validation_summary=llm_outcome.repair_validation_summary,
        llm_thinking_diagnostics=llm_outcome.thinking_diagnostics,
        llm_package_diagnostics=llm_outcome.package_diagnostics,
        composer_attempt_decision=llm_outcome.composer_attempt_decision,
        composer_requirement_analysis=llm_outcome.composer_requirement_analysis,
        composer_fulfillment_decisions=llm_outcome.composer_fulfillment_decisions,
        composer_source_coverage_summary=llm_outcome.composer_source_coverage_summary,
        validation_hard_mismatches=llm_outcome.validation_hard_mismatches,
        validation_unverified_requirements=llm_outcome.validation_unverified_requirements,
        final_status_source=llm_outcome.final_status_source,
        product_group=str(role_plan.get("product_group") or "unknown"),
        role_plan=role_plan,
        category_plan=category_plan_result.category_plan,
        category_plan_entries=category_plan_result.category_plan_entries,
        category_catalog_summary=category_plan_result.category_catalog_summary,
        category_planner_source=category_plan_result.category_planner_source,
        category_plan_source=category_plan_result.category_plan_source,
        category_planner_missing_required_roles=(
            category_plan_result.category_planner_missing_required_roles
        ),
        category_planner_repair_attempted=(
            category_plan_result.category_planner_repair_attempted
        ),
        category_planner_repair_success=(
            category_plan_result.category_planner_repair_success
        ),
        category_planner_repair_reason=(
            category_plan_result.category_planner_repair_reason
        ),
        category_planner_repaired_roles=(
            category_plan_result.category_planner_repaired_roles
        ),
        category_planner_unresolved_required_roles=(
            category_plan_result.category_planner_unresolved_required_roles
        ),
        required_capabilities=_mapping_list(role_plan.get("required_capabilities")),
        optional_capabilities=_mapping_list(role_plan.get("optional_capabilities")),
        unsupported_or_unmapped_requirements=_string_list(
            role_plan.get("unsupported_or_unmapped_requirements")
        ),
        required_roles=_string_list(role_plan.get("required_roles")),
        missing_required_roles=_string_list(
            component_candidate_matrix.get("missing_required_roles")
        ),
        missing_category_roles=category_plan_result.missing_category_roles,
        missing_required_capabilities=missing_required_capabilities,
        role_coverage_summary=_dict_or_empty(
            component_candidate_matrix.get("role_coverage_summary")
        ),
        category_plan_warnings=category_plan_result.category_plan_warnings,
    )


async def _list_candidate_products(
    session: AsyncSession,
    spec: StockSpec,
) -> list[DistributorProduct]:
    has_server_item = any(item.item_type.casefold() == "server" for item in spec.items)

    if has_server_item:
        category_priority = case(
            (DistributorProduct.category_id.in_(READY_SERVER_CATEGORY_IDS), 0),
            else_=1,
        )
        statement = select(DistributorProduct).order_by(
            category_priority,
            DistributorProduct.synced_at.desc(),
            DistributorProduct.id.desc(),
        )
    else:
        statement = select(DistributorProduct).order_by(
            DistributorProduct.synced_at.desc(),
            DistributorProduct.id.desc(),
        )

    result = await session.execute(statement)
    return list(result.scalars().all())


async def _build_product_group_plans(
    session: AsyncSession,
    *,
    spec: StockSpec,
    products: list[DistributorProduct],
    llm_settings: LlmSettings | None = None,
    llm_configurator_client: LlmClient | None = None,
) -> tuple[dict[str, Any], CategoryPlanResult]:
    deterministic_product_group_hint = _detect_spec_product_group(spec)
    distributor_code = _primary_distributor_code(products) or "ocs"
    effective_llm_settings = llm_settings or get_llm_settings()
    planner_state = _build_semantic_planner_llm_client(
        llm_settings=effective_llm_settings,
        llm_configurator_client=llm_configurator_client,
    )
    try:
        role_plan = plan_semantic_matrix_roles(
            spec,
            distributor_code=distributor_code,
            planner_client=planner_state.client,
            deterministic_product_group_hint=deterministic_product_group_hint,
            semantic_planner_max_seconds=(
                effective_llm_settings.llm_semantic_planner_max_seconds
            ),
            semantic_planner_stage_timeout_seconds=(
                effective_llm_settings.llm_semantic_planner_stage_timeout_seconds
            ),
        )
        role_plan = _with_semantic_runtime_diagnostics(role_plan, planner_state)
        product_group = str(role_plan.get("product_group") or "unknown").strip()
        profile = get_product_group_profile(product_group)
        if profile is None:
            return role_plan, CategoryPlanResult(category_plan={})

        category_rows = await _load_category_rows_for_catalog(
            session,
            distributor_code=distributor_code,
            products=products,
        )
        compact_catalog = build_compact_category_catalog(
            distributor_code=distributor_code,
            category_rows=category_rows,
            product_rows=products,
            product_group=product_group,
            matrix_roles=[
                *_string_list(
                    role_plan.get("effective_matrix_roles_before_category_planner")
                ),
                *_string_list(role_plan.get("category_planner_input_roles")),
                *_string_list(role_plan.get("matrix_blueprint_roles")),
                *_string_list(role_plan.get("required_roles")),
                *_string_list(role_plan.get("optional_roles")),
            ],
        )
        category_plan_result = plan_distributor_categories(
            distributor_code=distributor_code,
            product_group=product_group,
            role_plan=role_plan,
            compact_catalog=compact_catalog,
            llm_client=(
                planner_state.client
                if _semantic_plan_is_llm_authoritative(role_plan)
                else None
            ),
        )
        role_plan = {
            **role_plan,
            "category_planner_input_roles": (
                category_plan_result.category_planner_input_roles
            ),
            "category_planner_output_roles": (
                category_plan_result.category_planner_output_roles
            ),
            "validated_category_plan_roles": (
                category_plan_result.validated_category_plan_roles
            ),
            "category_planner_missing_required_roles": (
                category_plan_result.category_planner_missing_required_roles
            ),
            "category_planner_repair_attempted": (
                category_plan_result.category_planner_repair_attempted
            ),
            "category_planner_repair_success": (
                category_plan_result.category_planner_repair_success
            ),
            "category_planner_repair_reason": (
                category_plan_result.category_planner_repair_reason
            ),
            "category_planner_repaired_roles": (
                category_plan_result.category_planner_repaired_roles
            ),
            "category_planner_unresolved_required_roles": (
                category_plan_result.category_planner_unresolved_required_roles
            ),
            "roles_dropped_before_category_planner": (
                category_plan_result.roles_dropped_before_category_planner
            ),
            "roles_dropped_after_category_planner": (
                category_plan_result.roles_dropped_after_category_planner
            ),
            "roles_dropped_reason_by_role": (
                category_plan_result.roles_dropped_reason_by_role
            ),
            "role_source_by_role": category_plan_result.role_source_by_role,
            "role_lifecycle_trace": category_plan_result.role_lifecycle_trace,
        }
        return role_plan, category_plan_result
    finally:
        close = getattr(planner_state.client, "close", None)
        if callable(close):
            close()


def _build_semantic_planner_llm_client(
    *,
    llm_settings: LlmSettings | None,
    llm_configurator_client: LlmClient | None,
) -> _PlannerClientState:
    if llm_configurator_client is not None:
        return _PlannerClientState(unavailable_reason="external_llm_client_supplied")
    settings = llm_settings or get_llm_settings()
    provider = settings.llm_provider.strip().lower()
    model = settings.llm_model.strip() or None
    if provider == "disabled":
        return _PlannerClientState(
            provider=provider,
            model=model,
            unavailable_reason="llm_provider_disabled",
        )
    if provider not in SEMANTIC_PLANNER_SUPPORTED_PROVIDERS:
        return _PlannerClientState(
            provider=provider or None,
            model=model,
            unavailable_reason="llm_provider_unsupported",
        )
    if not (settings.llm_base_url.strip() and settings.llm_api_key.strip() and model):
        return _PlannerClientState(
            provider=provider,
            model=model,
            unavailable_reason="llm_settings_incomplete",
        )
    try:
        return _PlannerClientState(
            client=OpenAICompatibleLlmClient(
                settings=settings,
                timeout_seconds=min(
                    settings.llm_timeout_seconds,
                    settings.llm_semantic_planner_stage_timeout_seconds,
                ),
                max_output_tokens=min(settings.llm_configurator_max_output_tokens, 4096),
                use_response_format=False,
                thinking_enabled=False,
                max_retries=0,
            ),
            provider=provider,
            model=model,
        )
    except LlmError as exc:
        return _PlannerClientState(
            provider=provider,
            model=model,
            unavailable_reason="llm_client_build_failed",
            error_type=type(exc).__name__,
        )


def _with_semantic_runtime_diagnostics(
    role_plan: Mapping[str, Any],
    planner_state: _PlannerClientState,
) -> dict[str, Any]:
    result = dict(role_plan)
    source = str(result.get("semantic_planner_source") or "").strip()
    if not source:
        source = (
            SEMANTIC_PLANNER_SOURCE_PLANNER_UNAVAILABLE
            if planner_state.client is None
            else "fallback_after_llm_error"
        )
        result["semantic_planner_source"] = source
    result["semantic_planner_used"] = source in {
        "llm",
        "llm_repaired",
        "llm_minimal_fallback",
    }
    if planner_state.provider:
        result["semantic_planner_provider"] = planner_state.provider
    if planner_state.model:
        result["semantic_planner_model"] = planner_state.model
    if planner_state.error_type and not result.get("semantic_planner_error_type"):
        result["semantic_planner_error_type"] = planner_state.error_type
    if (
        source == SEMANTIC_PLANNER_COMPLEX_FALLBACK_REASON
        and not result.get("semantic_planner_fallback_reason")
    ):
        result["semantic_planner_fallback_reason"] = (
            SEMANTIC_PLANNER_COMPLEX_FALLBACK_REASON
        )
    if (
        planner_state.unavailable_reason
        and source != "llm"
        and not result.get("semantic_planner_fallback_reason")
    ):
        result["semantic_planner_fallback_reason"] = planner_state.unavailable_reason
    return result


def _detect_spec_product_group(spec: StockSpec) -> str:
    item_types = {item.item_type.strip().casefold() for item in spec.items}
    storage_item_types = {
        STORAGE_PRODUCT_GROUP,
        "storage_array",
        "storage_system",
        "san",
        "nas",
        "схд",
        STORAGE_SYSTEM_ROLE,
        STORAGE_ARRAY_CONTROLLER_ROLE,
        CONTROLLER_MODULE_ROLE,
        DISK_SHELF_ROLE,
        DRIVE_ROLE,
        HOST_PORT_ROLE,
        PROTOCOL_MODULE_ROLE,
    }
    network_item_types = {
        "network",
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
    if item_types.intersection(storage_item_types):
        return STORAGE_PRODUCT_GROUP
    if item_types.intersection(network_item_types):
        return NETWORK_PRODUCT_GROUP
    if "server" in item_types:
        return SERVER_PRODUCT_GROUP
    source_text = " ".join(
        part
        for part in [
            spec.source_text,
            " ".join(str(item.name or item.category or "") for item in spec.items),
            str(spec.requirements or ""),
        ]
        if part
    )
    if _looks_like_network_request(source_text):
        return NETWORK_PRODUCT_GROUP
    if _looks_like_storage_request(source_text):
        return STORAGE_PRODUCT_GROUP
    return SERVER_PRODUCT_GROUP if spec.items else "unknown"


def _looks_like_storage_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:storage\s+(?:system|array)|disk\s+array|san|nas|usable\s+capacity|"
            r"raw\s+capacity|nvme-?of|fc\s*(?:16|32|64)\s*g|iscsi|raid\s*[05610]|"
            r"drive\s+shelf|disk\s+shelf|expansion\s+shelf)\b|"
            r"схд|система\s+хранения|полк[аи]|полезн\w+\s+емкост|сыр\w+\s+емкост|"
            r"fc\s*(?:16|32|64)\s*g|nvme-?of|iscsi",
            text,
            re.I,
        )
    )


def _looks_like_network_request(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:switch|router|firewall|ngfw|utm|access\s*point|wi-?fi|"
            r"transceiver|optic|sfp\+?|sfp28|qsfp\+?|qsfp28|dac|uplink|poe|"
            r"stacking|stack)\b|коммутатор|свитч|маршрутизатор|роутер|"
            r"межсетев|фаервол|точк[аи]\s+доступа|трансивер|аплинк|стек|стекир|\bL[23]\b",
            text,
            re.I,
        )
    )


async def _load_category_rows_for_catalog(
    session: AsyncSession,
    *,
    distributor_code: str,
    products: list[DistributorProduct],
) -> list[DistributorCategory]:
    category_ids = sorted(
        {
            str(product.category_id).strip()
            for product in products
            if str(product.category_id or "").strip()
        }
    )
    if not category_ids:
        return []
    result = await session.execute(
        select(DistributorCategory).where(
            DistributorCategory.distributor_code == distributor_code,
            DistributorCategory.category_id.in_(category_ids),
        )
    )
    return list(result.scalars().all())


def _primary_distributor_code(products: list[DistributorProduct]) -> str | None:
    for product in products:
        if product.distributor_code:
            return product.distributor_code
    return None


async def _load_latest_stock_rows(
    session: AsyncSession,
    products: list[DistributorProduct],
) -> dict[tuple[str, str], list[DistributorStockPrice]]:
    item_ids = sorted({product.item_id for product in products})
    if not item_ids:
        return {}

    latest_result = await session.execute(
        select(
            DistributorStockPrice.distributor_code,
            func.max(DistributorStockPrice.synced_at),
        ).group_by(DistributorStockPrice.distributor_code)
    )
    latest_by_distributor = {
        distributor_code: synced_at
        for distributor_code, synced_at in latest_result.all()
        if synced_at is not None
    }
    if not latest_by_distributor:
        return {}

    latest_conditions = [
        (DistributorStockPrice.distributor_code == distributor_code)
        & (DistributorStockPrice.synced_at == synced_at)
        for distributor_code, synced_at in latest_by_distributor.items()
    ]
    result = await session.execute(
        select(DistributorStockPrice)
        .where(
            DistributorStockPrice.item_id.in_(item_ids),
            or_(*latest_conditions),
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


def _build_configuration_candidates(
    *,
    spec: StockSpec,
    products: list[DistributorProduct],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
    matrix_candidates_per_role: int = DEFAULT_MATRIX_CANDIDATES_PER_ROLE,
    role_plan: Mapping[str, Any] | None = None,
    category_plan_result: CategoryPlanResult | None = None,
) -> tuple[list[MatchCandidateResult], list[str], list[str], dict[str, Any], list[dict[str, Any]]]:
    role_plan = dict(role_plan or {})
    product_group = str(role_plan.get("product_group") or _detect_spec_product_group(spec))
    planned_items = [
        (item_index, item)
        for item_index, item in enumerate(spec.items)
        if _item_belongs_to_product_group(item, product_group)
    ]
    if not planned_items:
        synthetic_item = _synthetic_product_group_item_from_role_plan(
            spec,
            product_group=product_group,
            role_plan=role_plan,
        )
        if synthetic_item is not None:
            planned_items = [(0, synthetic_item)]
    if not planned_items:
        return [], [], [], {}, []

    category_plan_result = category_plan_result or CategoryPlanResult(category_plan={})
    products_by_role = _products_by_server_role(
        products,
        category_plan=category_plan_result.category_plan,
        product_group=product_group,
    )
    platform_products = products_by_role.get(SERVER_PLATFORM_ROLE, [])
    component_products = [
        product
        for role, role_products in products_by_role.items()
        if role != READY_SERVER_ROLE
        for product in role_products
    ]

    build_missing: list[str] = []
    build_risks: list[str] = []
    if not component_products:
        message = (
            "Складская спецификация пока не предложена - нет достаточных складских данных "
            "по выбранным ролям товарной группы."
        )
        build_missing.append(message)
        build_risks.append(message)

    if product_group == SERVER_PRODUCT_GROUP and not platform_products:
        message = (
            "Сборка из комплектующих невозможна: в локальной OCS DB нет товаров категории "
            "серверных платформ."
        )
        build_missing.append(message)
        build_risks.append(message)

    facts_by_key = _facts_by_product(products_by_role)
    candidates: list[MatchCandidateResult] = []
    matrix_json: dict[str, Any] = {}
    normalized_requirements: list[dict[str, Any]] = []
    for item_index, item in planned_items:
        requirements = _normalize_product_group_requirements(
            spec,
            item,
            product_group=product_group,
            role_plan=role_plan,
            category_plan=category_plan_result.category_plan,
        )
        normalized_requirements.append(requirements.to_report_json())
        matrix = _build_candidate_matrix(
            requirements=requirements,
            products_by_role=products_by_role,
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
            limit_per_role=matrix_candidates_per_role,
            role_plan=role_plan,
            category_plan=category_plan_result.category_plan,
            category_plan_entries=category_plan_result.category_plan_entries,
            category_catalog_summary=category_plan_result.category_catalog_summary,
            category_planner_source=category_plan_result.category_planner_source,
            category_plan_source=category_plan_result.category_plan_source,
            category_planner_missing_required_roles=(
                category_plan_result.category_planner_missing_required_roles
            ),
            category_planner_repair_attempted=(
                category_plan_result.category_planner_repair_attempted
            ),
            category_planner_repair_success=(
                category_plan_result.category_planner_repair_success
            ),
            category_planner_repair_reason=(
                category_plan_result.category_planner_repair_reason
            ),
            category_planner_repaired_roles=(
                category_plan_result.category_planner_repaired_roles
            ),
            category_planner_unresolved_required_roles=(
                category_plan_result.category_planner_unresolved_required_roles
            ),
            missing_category_roles=category_plan_result.missing_category_roles,
            category_plan_warnings=category_plan_result.category_plan_warnings,
            planner_role_coverage=category_plan_result.role_coverage_summary,
        )
        if not matrix_json:
            matrix_json = matrix.to_report_json()
        candidates.extend(
            _generate_build_candidates(
                spec=spec,
                item=item,
                item_index=item_index,
                matrix=matrix,
            )
            if product_group == SERVER_PRODUCT_GROUP
            else []
        )

    candidates.sort(key=_build_candidate_sort_key)
    selected = _select_diverse_build_candidates(candidates)
    risks = _limited_alternatives_warnings(selected, matrix_json)
    return selected, build_missing, [*build_risks, *risks], matrix_json, normalized_requirements


def _candidate_rows(candidates: list[_ComponentCandidate]) -> list[dict[str, Any]]:
    return [candidate.to_report_json() for candidate in candidates]


def _matrix_candidates_per_role(llm_settings: LlmSettings | None) -> int:
    if llm_settings is not None:
        configured = llm_settings.llm_component_candidates_per_role
    else:
        configured = get_llm_settings().llm_component_candidates_per_role
    return max(
        1,
        min(
            configured or DEFAULT_MATRIX_CANDIDATES_PER_ROLE,
            MAX_MATRIX_CANDIDATES_PER_ROLE,
        ),
    )


def _compose_llm_configurations(
    *,
    spec: StockSpec,
    ready_stock_candidates: list[MatchCandidateResult],
    build_candidates: list[MatchCandidateResult],
    component_candidate_matrix: dict[str, Any],
    normalized_requirements: list[dict[str, Any]],
    llm_configurator_client: LlmClient | None,
    llm_settings: LlmSettings | None,
    web_evidence_settings: WebEvidenceSettings | None,
    web_search_provider: WebSearchProvider | None,
    evidence_cache: EvidenceSearchCache | None,
) -> LlmConfiguratorOutcome:
    ready_candidates = [candidate.to_report_json() for candidate in ready_stock_candidates]
    rule_based_build_candidates = [candidate.to_report_json() for candidate in build_candidates]
    return compose_llm_configurations(
        user_request=spec.source_text,
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=ready_candidates,
        component_candidate_matrix=component_candidate_matrix,
        rule_based_build_candidates=rule_based_build_candidates,
        settings=llm_settings,
        llm_client=llm_configurator_client,
        web_evidence_settings=web_evidence_settings or get_web_evidence_settings(),
        web_search_provider=web_search_provider,
        evidence_cache=evidence_cache,
    )


def _composer_context_limit_fallback_needed(outcome: LlmConfiguratorOutcome) -> bool:
    diagnostics = _dict_or_empty(outcome.package_diagnostics)
    parse_diagnostics = _dict_or_empty(outcome.parse_diagnostics)
    return (
        outcome.fallback_reason == PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON
        or diagnostics.get("provider_error_type") == PROVIDER_CONTEXT_LIMIT_ERROR_TYPE
        or parse_diagnostics.get("provider_error_type") == PROVIDER_CONTEXT_LIMIT_ERROR_TYPE
    )


def _with_provider_context_limit_fallback_metadata(
    matrix: Mapping[str, Any],
    outcome: LlmConfiguratorOutcome,
) -> dict[str, Any]:
    result = dict(matrix)
    context_limit = (
        _dict_or_empty(outcome.package_diagnostics.get("provider_context_limit"))
        or _dict_or_empty(outcome.parse_diagnostics)
    )
    provider_diagnostics = {
        "provider_error_type": PROVIDER_CONTEXT_LIMIT_ERROR_TYPE,
        "provider_context_limit": context_limit,
        "fallback_reason": PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON,
    }
    result["provider_error_type"] = PROVIDER_CONTEXT_LIMIT_ERROR_TYPE
    result["provider_context_limit"] = context_limit
    diagnostics = _dict_or_empty(result.get("matrix_distiller_diagnostics"))
    result["matrix_distiller_diagnostics"] = {
        **diagnostics,
        **provider_diagnostics,
        "full_matrix_trigger_reason": PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON,
    }
    if bool(result.get("full_matrix_evaluation_used")):
        result["full_matrix_evaluation_fallback_reason"] = (
            PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON
        )
    return result


def _shortlist_for_llm(
    build_candidates: list[dict[str, Any]],
    component_candidate_matrix: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in build_candidates[:MAX_API_BUILD_CANDIDATES]:
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "score": candidate.get("score"),
                "platform": candidate.get("platform"),
                "components": candidate.get("components"),
                "price": {
                    "value": candidate.get("total_price_value"),
                    "currency": candidate.get("total_price_currency"),
                    "note": candidate.get("total_price_note"),
                },
                "stock": {"available_quantity": candidate.get("available_quantity")},
                "warnings": candidate.get("compatibility_warnings", []),
                "missing_roles": candidate.get("missing_component_roles", []),
                "rank_reason": candidate.get("rank_reason", []),
            }
        )
    if rows:
        return rows
    return _component_matrix_shortlist_for_llm(component_candidate_matrix or {})


def _component_matrix_shortlist_for_llm(
    component_candidate_matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in MATRIX_ROLE_ORDER:
        candidates = component_candidate_matrix.get(f"{role}_candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        for candidate in candidates[:MAX_API_BUILD_CANDIDATES]:
            if not isinstance(candidate, Mapping):
                continue
            rows.append(
                {
                    "role": role,
                    "component_candidate_id": candidate.get("component_candidate_id"),
                    "category_id": candidate.get("category_id"),
                    "distributor_code": candidate.get("distributor_code"),
                    "item_id": candidate.get("item_id"),
                    "part_number": candidate.get("part_number"),
                    "producer": candidate.get("producer"),
                    "name": candidate.get("name") or candidate.get("item_name"),
                    "price": {
                        "value": candidate.get("price_value"),
                        "currency": candidate.get("price_currency"),
                    },
                    "stock": {
                        "available_quantity": candidate.get("available_quantity"),
                    },
                    "fit_tier": candidate.get("fit_tier"),
                    "score": candidate.get("score"),
                }
            )
    return rows[:MAX_API_BUILD_CANDIDATES]


def _item_belongs_to_product_group(item: StockSpecItem, product_group: str) -> bool:
    item_type = item.item_type.strip().casefold()
    if product_group == SERVER_PRODUCT_GROUP:
        return item_type == "server"
    if product_group == NETWORK_PRODUCT_GROUP:
        return item_type in {
            "network",
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
            "unknown",
        }
    if product_group == STORAGE_PRODUCT_GROUP:
        return item_type in {
            STORAGE_PRODUCT_GROUP,
            "storage_array",
            "storage_system",
            "san",
            "nas",
            "схд",
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
            "unknown",
        }
    return False


def _synthetic_product_group_item_from_role_plan(
    spec: StockSpec,
    *,
    product_group: str,
    role_plan: Mapping[str, Any],
) -> StockSpecItem | None:
    required_roles = _string_list(role_plan.get("required_roles"))
    if not required_roles:
        return None
    role_requirements = _dict_or_empty(role_plan.get("requirements_by_role"))
    if product_group == SERVER_PRODUCT_GROUP:
        parsed = _dict_or_empty(role_requirements.get(SERVER_PLATFORM_ROLE))
        quantity = (
            _as_int(parsed.get("server_count"))
            or _as_int(parsed.get("device_count"))
            or _as_int(parsed.get("count"))
            or _as_int(parsed.get("quantity"))
            or _as_int(_dict_or_empty(role_plan.get("logistics_constraints")).get("quantity"))
            or 1
        )
        return StockSpecItem(
            item_type=SERVER_PRODUCT_GROUP,
            quantity=max(1, quantity),
            name=SERVER_PRODUCT_GROUP,
            location=spec.location or spec.shipment_city,
            requirements={"role": SERVER_PRODUCT_GROUP},
        )
    if product_group == NETWORK_PRODUCT_GROUP:
        item_type = _primary_network_role(required_roles) or required_roles[0]
    elif product_group == STORAGE_PRODUCT_GROUP:
        item_type = _primary_storage_role(required_roles) or required_roles[0]
    else:
        return None
    parsed = _dict_or_empty(role_requirements.get(item_type))
    quantity = (
        _as_int(parsed.get("device_count"))
        or _as_int(parsed.get("system_count"))
        or _as_int(parsed.get("count"))
        or _as_int(parsed.get("quantity"))
        or 1
    )
    return StockSpecItem(
        item_type=item_type,
        quantity=max(1, quantity),
        name=item_type,
        location=spec.location or spec.shipment_city,
        requirements={"role": item_type},
    )


def _primary_network_role(required_roles: Sequence[str]) -> str | None:
    for role in (SWITCH_ROLE, ROUTER_ROLE, FIREWALL_ROLE, ACCESS_POINT_ROLE):
        if role in required_roles:
            return role
    return None


def _primary_storage_role(required_roles: Sequence[str]) -> str | None:
    for role in (
        STORAGE_SYSTEM_ROLE,
        STORAGE_ARRAY_CONTROLLER_ROLE,
        CONTROLLER_MODULE_ROLE,
        DISK_SHELF_ROLE,
        DRIVE_ROLE,
        HOST_PORT_ROLE,
        PROTOCOL_MODULE_ROLE,
    ):
        if role in required_roles:
            return role
    return None


def _normalize_product_group_requirements(
    spec: StockSpec,
    item: StockSpecItem,
    *,
    product_group: str,
    role_plan: Mapping[str, Any] | None = None,
    category_plan: Mapping[str, list[str]] | None = None,
) -> _NormalizedServerRequirements:
    if product_group == NETWORK_PRODUCT_GROUP:
        return _normalize_network_requirements(
            spec,
            item,
            role_plan=role_plan,
            category_plan=category_plan,
        )
    if product_group == STORAGE_PRODUCT_GROUP:
        return _normalize_storage_requirements(
            spec,
            item,
            role_plan=role_plan,
            category_plan=category_plan,
        )
    return _normalize_server_requirements(
        spec,
        item,
        role_plan=role_plan,
        category_plan=category_plan,
    )


def _normalize_network_requirements(
    spec: StockSpec,
    item: StockSpecItem,
    *,
    role_plan: Mapping[str, Any] | None = None,
    category_plan: Mapping[str, list[str]] | None = None,
) -> _NormalizedServerRequirements:
    role_plan_mapping = _dict_or_empty(role_plan)
    role_requirements = _dict_or_empty(role_plan_mapping.get("requirements_by_role"))
    required_roles = _string_list(role_plan_mapping.get("required_roles"))
    primary_role = _network_primary_device_role(required_roles, item)
    parsed = _dict_or_empty(role_requirements.get(primary_role or ""))
    quantity = (
        _as_int(parsed.get("device_count"))
        or _as_int(parsed.get("count"))
        or _as_int(parsed.get("quantity"))
        or item.quantity
        or 1
    )
    logistics_constraints = _dict_or_empty(role_plan_mapping.get("logistics_constraints"))
    location = _text_or_none(
        _explicit_field(spec, item, "location")
        or spec.shipment_city
        or logistics_constraints.get("shipment_city")
        or logistics_constraints.get("city")
        or logistics_constraints.get("warehouse_city")
    )
    return _NormalizedServerRequirements(
        server_qty=max(1, quantity),
        product_group=NETWORK_PRODUCT_GROUP,
        network_device_role=primary_role,
        location=location,
        optimization_mode=_normalize_optimization_mode(
            _explicit_field(spec, item, "optimization_mode")
            or _requirement(spec, item, "optimization_mode")
            or _commercial_optimization_goal(role_plan_mapping)
        ),
        role_plan=dict(role_plan_mapping),
        required_capabilities=_mapping_list(
            role_plan_mapping.get("required_capabilities")
        ),
        optional_capabilities=_mapping_list(
            role_plan_mapping.get("optional_capabilities")
        ),
        unsupported_or_unmapped_requirements=_string_list(
            role_plan_mapping.get("unsupported_or_unmapped_requirements")
        ),
        required_roles=required_roles,
        category_plan={
            str(role): list(ids)
            for role, ids in (category_plan or {}).items()
            if isinstance(ids, list)
        },
    )


def _network_primary_device_role(
    required_roles: Sequence[str],
    item: StockSpecItem,
) -> str | None:
    device_roles = (SWITCH_ROLE, ROUTER_ROLE, FIREWALL_ROLE, ACCESS_POINT_ROLE)
    for role in required_roles:
        if role in device_roles:
            return role
    item_type = item.item_type.strip().casefold()
    if item_type in device_roles:
        return item_type
    return None


def _normalize_storage_requirements(
    spec: StockSpec,
    item: StockSpecItem,
    *,
    role_plan: Mapping[str, Any] | None = None,
    category_plan: Mapping[str, list[str]] | None = None,
) -> _NormalizedServerRequirements:
    role_plan_mapping = _dict_or_empty(role_plan)
    role_requirements = _dict_or_empty(role_plan_mapping.get("requirements_by_role"))
    system_requirements = _dict_or_empty(role_requirements.get(STORAGE_SYSTEM_ROLE))
    drive_requirements = _dict_or_empty(role_requirements.get(DRIVE_ROLE))
    if not drive_requirements:
        drive_requirements = _dict_or_empty(role_requirements.get(SSD_ROLE))
    if not drive_requirements:
        drive_requirements = _dict_or_empty(role_requirements.get(HDD_ROLE))
    controller_requirements = _dict_or_empty(
        role_requirements.get(STORAGE_ARRAY_CONTROLLER_ROLE)
    )
    support_requirements = _dict_or_empty(role_requirements.get(SUPPORT_ROLE))
    license_requirements = _dict_or_empty(role_requirements.get(LICENSE_ROLE))
    host_requirements = _dict_or_empty(role_requirements.get(HOST_PORT_ROLE))
    if not host_requirements:
        host_requirements = _dict_or_empty(role_requirements.get(PROTOCOL_MODULE_ROLE))
    shelf_requirements = _dict_or_empty(role_requirements.get(DISK_SHELF_ROLE))
    logistics_constraints = _dict_or_empty(role_plan_mapping.get("logistics_constraints"))

    system_count = (
        _as_int(system_requirements.get("system_count"))
        or _as_int(system_requirements.get("device_count"))
        or _as_int(system_requirements.get("count"))
        or _as_int(system_requirements.get("quantity"))
        or item.quantity
        or 1
    )
    raw_capacity_tb = _first_float_requirement(
        role_plan_mapping,
        "raw_capacity_tb",
        preferred_roles=(STORAGE_SYSTEM_ROLE,),
    )
    usable_capacity_tb = _first_float_requirement(
        role_plan_mapping,
        "usable_capacity_tb",
        preferred_roles=(STORAGE_SYSTEM_ROLE,),
    )
    storage_min_capacity_tb = usable_capacity_tb or raw_capacity_tb
    storage_min_capacity = (
        f"{_format_number(storage_min_capacity_tb)} TB"
        if storage_min_capacity_tb is not None
        else None
    )
    drive_type = _normalize_storage_type(
        _first_text_requirement(
            role_plan_mapping,
            "drive_type",
            preferred_roles=(DRIVE_ROLE, SSD_ROLE, HDD_ROLE),
        )
        or drive_requirements.get("type")
    )
    drive_interface = _normalize_storage_interface(
        _first_text_requirement(
            role_plan_mapping,
            "drive_interface",
            preferred_roles=(DRIVE_ROLE, SSD_ROLE, HDD_ROLE),
        )
        or drive_requirements.get("interface")
    )
    host_protocol = _normalize_storage_protocol(
        _first_text_requirement(
            role_plan_mapping,
            "host_protocol",
            preferred_roles=(HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE, STORAGE_SYSTEM_ROLE),
        )
        or host_requirements.get("protocol")
    )
    host_port_speed = str(
        _first_text_requirement(
            role_plan_mapping,
            "host_port_speed",
            preferred_roles=(HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE, STORAGE_SYSTEM_ROLE),
        )
        or host_requirements.get("speed")
        or UNKNOWN_FACT
    )
    return _NormalizedServerRequirements(
        server_qty=max(1, system_count),
        product_group=STORAGE_PRODUCT_GROUP,
        storage_required=True,
        storage_type_preference=drive_type,
        storage_interface_preference=drive_interface,
        storage_min_capacity=storage_min_capacity,
        storage_min_capacity_tb=storage_min_capacity_tb,
        storage_qty_per_server=_as_int(drive_requirements.get("drives_per_system"))
        or _as_int(drive_requirements.get("count"))
        or _as_int(drive_requirements.get("quantity")),
        raw_capacity_tb=raw_capacity_tb,
        usable_capacity_tb=usable_capacity_tb,
        redundancy_level=str(
            _first_text_requirement(role_plan_mapping, "redundancy_level")
            or system_requirements.get("redundancy_level")
            or UNKNOWN_FACT
        ),
        controller_count=_as_int(controller_requirements.get("controller_count"))
        or _first_int_requirement(
            role_plan_mapping,
            "controller_count",
            preferred_roles=(STORAGE_ARRAY_CONTROLLER_ROLE, STORAGE_SYSTEM_ROLE),
        ),
        shelf_count=_as_int(shelf_requirements.get("shelf_count"))
        or _as_int(shelf_requirements.get("count")),
        drive_count=_first_int_requirement(
            role_plan_mapping,
            "drive_count",
            preferred_roles=(DRIVE_ROLE, SSD_ROLE, HDD_ROLE),
        )
        or _as_int(drive_requirements.get("drive_count"))
        or _as_int(drive_requirements.get("count")),
        drive_capacity_tb=_first_float_requirement(
            role_plan_mapping,
            "drive_capacity_tb",
            preferred_roles=(DRIVE_ROLE, SSD_ROLE, HDD_ROLE),
        ),
        drive_type=drive_type,
        drive_interface=drive_interface,
        host_protocol=host_protocol,
        host_port_count=_first_int_requirement(
            role_plan_mapping,
            "host_port_count",
            preferred_roles=(HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE, STORAGE_SYSTEM_ROLE),
        )
        or _as_int(host_requirements.get("port_count")),
        host_port_speed=host_port_speed,
        host_port_speed_gbps=_network_speed_requirement_gbps(host_port_speed),
        host_port_media=str(
            _first_text_requirement(
                role_plan_mapping,
                "host_port_media",
                preferred_roles=(HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE, STORAGE_SYSTEM_ROLE),
            )
            or host_requirements.get("media")
            or UNKNOWN_FACT
        ),
        license_required=LICENSE_ROLE in _string_list(role_plan_mapping.get("required_roles"))
        or bool(license_requirements.get("license_required"))
        or bool(system_requirements.get("license_required")),
        support_required=SUPPORT_ROLE in _string_list(role_plan_mapping.get("required_roles"))
        or bool(support_requirements.get("support_required"))
        or bool(system_requirements.get("support_required")),
        warranty_months=_first_int_requirement(role_plan_mapping, "warranty_months")
        or _as_int(support_requirements.get("warranty_months")),
        location=_text_or_none(
            _explicit_field(spec, item, "location")
            or spec.shipment_city
            or logistics_constraints.get("shipment_city")
            or logistics_constraints.get("city")
            or logistics_constraints.get("warehouse_city")
        ),
        optimization_mode=_normalize_optimization_mode(
            _explicit_field(spec, item, "optimization_mode")
            or _requirement(spec, item, "optimization_mode")
            or _commercial_optimization_goal(role_plan_mapping)
        ),
        role_plan=dict(role_plan_mapping),
        required_capabilities=_mapping_list(
            role_plan_mapping.get("required_capabilities")
        ),
        optional_capabilities=_mapping_list(
            role_plan_mapping.get("optional_capabilities")
        ),
        unsupported_or_unmapped_requirements=_string_list(
            role_plan_mapping.get("unsupported_or_unmapped_requirements")
        ),
        required_roles=_string_list(role_plan_mapping.get("required_roles")),
        category_plan={
            str(role): list(ids)
            for role, ids in (category_plan or {}).items()
            if isinstance(ids, list)
        },
    )


def _normalize_server_requirements(
    spec: StockSpec,
    item: StockSpecItem,
    *,
    role_plan: Mapping[str, Any] | None = None,
    category_plan: Mapping[str, list[str]] | None = None,
) -> _NormalizedServerRequirements:
    server_qty = _as_int(_explicit_field(spec, item, "server_qty")) or item.quantity
    form_factor = _text_or_none(
        _explicit_field(spec, item, "form_factor") or _requirement(spec, item, "form_factor")
    )
    cpu_per_server = _as_int(_explicit_field(spec, item, "cpu_per_server"))
    if cpu_per_server is None:
        cpu_per_server = _cpu_per_server(spec, item)

    total_cpu_required = _as_int(_explicit_field(spec, item, "total_cpu_required"))
    if total_cpu_required is None and cpu_per_server is not None:
        total_cpu_required = cpu_per_server * server_qty

    request_text = _request_text(spec, item)
    role_plan_mapping = _dict_or_empty(role_plan)
    role_requirements = _dict_or_empty(role_plan_mapping.get("requirements_by_role"))
    cpu_role_requirements = _dict_or_empty(role_requirements.get(CPU_ROLE))
    ram_role_requirements = _dict_or_empty(role_requirements.get(RAM_ROLE))
    storage_role_requirements = _dict_or_empty(role_requirements.get("storage"))
    platform_role_requirements = _dict_or_empty(
        role_requirements.get(SERVER_PLATFORM_ROLE)
    )
    psu_role_requirements = _dict_or_empty(role_requirements.get(POWER_SUPPLY_ROLE))
    if cpu_per_server is None:
        cpu_per_server = (
            _as_int(cpu_role_requirements.get("cpu_per_server"))
            or _as_int(cpu_role_requirements.get("per_server"))
            or _as_int(cpu_role_requirements.get("count_per_server"))
        )
    if total_cpu_required is None and cpu_per_server is not None:
        total_cpu_required = cpu_per_server * server_qty
    cpu_vendor_preference = _normalize_cpu_vendor(
        _explicit_field(spec, item, "cpu_vendor_preference")
        or _requirement(spec, item, "cpu", "vendor_preference")
        or _requirement(spec, item, "cpu", "vendor")
        or cpu_role_requirements.get("vendor_preference")
        or cpu_role_requirements.get("vendor")
        or _detect_requested_cpu_vendor(request_text)
    )
    cpu_family_preference = _normalize_cpu_family(
        _explicit_field(spec, item, "cpu_family_preference")
        or _requirement(spec, item, "cpu", "family_preference")
        or _requirement(spec, item, "cpu", "family")
        or cpu_role_requirements.get("family_preference")
        or cpu_role_requirements.get("family")
        or _detect_requested_cpu_family(request_text)
    )
    cpu_min_cores_per_cpu = (
        _as_int(_explicit_field(spec, item, "cpu_min_cores_per_cpu"))
        or _as_int(_requirement(spec, item, "cpu", "min_cores_per_cpu"))
        or _as_int(_requirement(spec, item, "cpu", "cores_per_cpu"))
        or _as_int(_requirement(spec, item, "cpu", "cores"))
        or _as_int(cpu_role_requirements.get("min_cores_per_cpu"))
        or _as_int(cpu_role_requirements.get("cores_per_cpu"))
        or _as_int(cpu_role_requirements.get("cores"))
        or _detect_requested_cpu_cores(request_text)
    )
    cpu_generation_or_model_hint = _text_or_none(
        _explicit_field(spec, item, "cpu_generation_or_model_hint")
        or _requirement(spec, item, "cpu", "generation_or_model_hint")
        or _requirement(spec, item, "cpu", "generation")
        or _requirement(spec, item, "cpu", "model")
        or cpu_role_requirements.get("generation_or_model_hint")
        or cpu_role_requirements.get("generation")
        or cpu_role_requirements.get("model")
    )

    ram_gb_per_server = (
        _as_int(_explicit_field(spec, item, "ram_gb_per_server"))
        or _as_int(_requirement(spec, item, "ram", "gb_per_server"))
        or _as_int(_requirement(spec, item, "ram", "min_gb"))
        or _as_int(ram_role_requirements.get("gb_per_server"))
        or _as_int(ram_role_requirements.get("min_gb_per_server"))
        or _as_int(ram_role_requirements.get("min_gb"))
        or _detect_requested_ram_gb(request_text)
    )
    ram_type_preference = _normalize_ram_type(
        _explicit_field(spec, item, "ram_type_preference")
        or _requirement(spec, item, "ram", "type")
        or _requirement(spec, item, "ram", "generation")
        or ram_role_requirements.get("type")
        or ram_role_requirements.get("generation")
        or _detect_requested_ram_type(request_text)
    )
    if (
        ram_gb_per_server is not None
        and ram_type_preference == UNKNOWN_FACT
        and _ram_per_server_phrase(request_text)
    ):
        ram_type_preference = "DDR5"

    explicit_storage_required = _explicit_field(spec, item, "storage_required")
    storage_preference_source = (
        _explicit_field(spec, item, "storage_type_preference")
        or _requirement(spec, item, "storage", "type")
        or _requirement(spec, item, "storage", "interface")
        or storage_role_requirements.get("type")
        or storage_role_requirements.get("interface")
        or _detect_requested_storage_preference(request_text)
    )
    normalized_storage_preference = _normalize_storage_preference(storage_preference_source)
    storage_type_preference = _normalize_storage_type(storage_preference_source)
    storage_interface_preference = _normalize_storage_interface(
        _explicit_field(spec, item, "storage_interface_preference")
        or _requirement(spec, item, "storage", "interface")
        or storage_role_requirements.get("interface")
        or (
            normalized_storage_preference
            if normalized_storage_preference in {"NVMe", "SAS", "SATA"}
            else None
        )
        or _detect_requested_storage_interface(request_text)
    )
    storage_required = (
        bool(explicit_storage_required)
        if explicit_storage_required is not None
        else storage_type_preference != UNKNOWN_FACT
        or storage_interface_preference != UNKNOWN_FACT
        or _requirement(spec, item, "storage") is not None
    )
    raw_storage_min_capacity = (
        _explicit_field(spec, item, "storage_min_capacity")
        or _requirement(spec, item, "storage", "min_capacity")
        or _requirement(spec, item, "storage", "capacity")
        or storage_role_requirements.get("min_capacity")
        or storage_role_requirements.get("capacity")
        or _detect_requested_storage_capacity(request_text)
    )
    storage_min_capacity_tb = _float_value(
        _explicit_field(spec, item, "storage_min_capacity_tb")
        or _requirement(spec, item, "storage", "min_capacity_tb")
        or storage_role_requirements.get("min_capacity_tb")
    )
    if storage_min_capacity_tb is None:
        storage_min_capacity_tb = _capacity_to_tb(raw_storage_min_capacity)
    storage_min_capacity = _text_or_none(raw_storage_min_capacity)
    if storage_min_capacity is None and storage_min_capacity_tb is not None:
        storage_min_capacity = f"{_format_number(storage_min_capacity_tb)} TB"

    storage_qty_per_server = (
        _as_int(_explicit_field(spec, item, "storage_qty_per_server"))
        or _as_int(_requirement(spec, item, "storage", "qty_per_server"))
        or _as_int(_requirement(spec, item, "storage", "quantity_per_server"))
        or _as_int(_requirement(spec, item, "storage", "count_per_server"))
        or _as_int(_requirement(spec, item, "storage", "qty"))
        or _as_int(_requirement(spec, item, "storage", "quantity"))
        or _as_int(_requirement(spec, item, "storage", "count"))
        or _as_int(storage_role_requirements.get("qty_per_server"))
        or _as_int(storage_role_requirements.get("quantity_per_server"))
        or _as_int(storage_role_requirements.get("count_per_server"))
        or _as_int(storage_role_requirements.get("drives_per_server"))
        or _detect_requested_storage_qty(request_text)
    )
    if storage_required and storage_qty_per_server is None:
        storage_qty_per_server = 1

    required_roles = _string_list(role_plan_mapping.get("required_roles"))
    semantic_authoritative = _semantic_plan_is_llm_authoritative(role_plan_mapping)
    network_explicit = (
        _requirement(spec, item, "network")
        or _requirement(spec, item, "network_adapter")
        or role_requirements.get(NETWORK_ADAPTER_ROLE)
    )
    if semantic_authoritative and NETWORK_ADAPTER_ROLE not in required_roles:
        network_requirement = network_requirement_from_sources(text="", explicit={})
    else:
        network_requirement = network_requirement_from_sources(
            text=request_text,
            explicit=_dict_or_empty(network_explicit),
        )
    network_required = (
        bool(network_requirement.get("required"))
        or NETWORK_ADAPTER_ROLE in required_roles
    )
    if network_required:
        network_requirement = {**network_requirement, "required": True}

    psu_count_per_server = (
        _as_int(_explicit_field(spec, item, "psu_count_per_server"))
        or _as_int(_requirement(spec, item, "power", "psu_count"))
        or _as_int(platform_role_requirements.get("psu_count_per_server"))
        or _as_int(platform_role_requirements.get("psu_count"))
        or _as_int(psu_role_requirements.get("psu_count_per_server"))
        or _as_int(psu_role_requirements.get("count_per_server"))
    )
    if psu_count_per_server is None and _requirement(spec, item, "power", "redundant_psu") is True:
        psu_count_per_server = 2

    logistics_constraints = _dict_or_empty(role_plan_mapping.get("logistics_constraints"))
    location = _text_or_none(
        _explicit_field(spec, item, "location")
        or spec.shipment_city
        or logistics_constraints.get("shipment_city")
        or logistics_constraints.get("city")
        or logistics_constraints.get("warehouse_city")
    )
    optimization_mode = _normalize_optimization_mode(
        _explicit_field(spec, item, "optimization_mode")
        or _requirement(spec, item, "optimization_mode")
        or _commercial_optimization_goal(role_plan_mapping)
    )

    return _NormalizedServerRequirements(
        server_qty=server_qty,
        form_factor=form_factor,
        cpu_per_server=cpu_per_server,
        total_cpu_required=total_cpu_required,
        cpu_vendor_preference=cpu_vendor_preference,
        cpu_family_preference=cpu_family_preference,
        cpu_min_cores_per_cpu=cpu_min_cores_per_cpu,
        cpu_generation_or_model_hint=cpu_generation_or_model_hint,
        ram_gb_per_server=ram_gb_per_server,
        ram_type_preference=ram_type_preference,
        storage_required=storage_required,
        storage_type_preference=storage_type_preference,
        storage_interface_preference=storage_interface_preference,
        storage_min_capacity=storage_min_capacity,
        storage_min_capacity_tb=storage_min_capacity_tb,
        storage_qty_per_server=storage_qty_per_server,
        network_required=network_required,
        network_min_ports_per_server=_as_int(
            network_requirement.get("min_ports_per_server")
        ),
        network_speed=str(network_requirement.get("speed") or UNKNOWN_FACT),
        network_media=str(network_requirement.get("media") or UNKNOWN_FACT),
        network_interface=str(network_requirement.get("interface") or UNKNOWN_FACT),
        network_requirement=dict(network_requirement),
        psu_count_per_server=psu_count_per_server,
        location=location,
        optimization_mode=optimization_mode,
        role_plan=dict(role_plan_mapping),
        required_capabilities=_mapping_list(
            role_plan_mapping.get("required_capabilities")
        ),
        optional_capabilities=_mapping_list(
            role_plan_mapping.get("optional_capabilities")
        ),
        unsupported_or_unmapped_requirements=_string_list(
            role_plan_mapping.get("unsupported_or_unmapped_requirements")
        ),
        required_roles=required_roles,
        category_plan={
            str(role): list(ids)
            for role, ids in (category_plan or {}).items()
            if isinstance(ids, list)
        },
    )


def _build_candidate_matrix(
    *,
    requirements: _NormalizedServerRequirements,
    products_by_role: dict[str, list[DistributorProduct]],
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
    limit_per_role: int,
    role_plan: Mapping[str, Any],
    category_plan: Mapping[str, list[str]],
    category_plan_entries: list[dict[str, Any]],
    category_catalog_summary: Mapping[str, Any],
    category_planner_source: str,
    category_plan_source: str,
    category_planner_missing_required_roles: list[str],
    category_planner_repair_attempted: bool,
    category_planner_repair_success: bool,
    category_planner_repair_reason: str | None,
    category_planner_repaired_roles: list[str],
    category_planner_unresolved_required_roles: list[str],
    missing_category_roles: list[str],
    category_plan_warnings: list[str],
    planner_role_coverage: Mapping[str, Any],
) -> _CandidateMatrix:
    limit_per_role = max(1, min(limit_per_role, MAX_MATRIX_CANDIDATES_PER_ROLE))
    if requirements.product_group == SERVER_PRODUCT_GROUP:
        ready_server_all = _score_simple_role_candidates(
            role=READY_SERVER_ROLE,
            products=products_by_role.get(READY_SERVER_ROLE, []),
            requirements=requirements,
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
        )
        ready_server_candidates = _select_broad_component_candidates(
            ready_server_all,
            role=READY_SERVER_ROLE,
            requirements=requirements,
            limit=limit_per_role,
        )
    else:
        ready_server_all = []
        ready_server_candidates = []
    platform_all = [
        candidate
        for product in products_by_role.get(SERVER_PLATFORM_ROLE, [])
        if (
            candidate := _score_platform_candidate(
                product=product,
                requirements=requirements,
                facts_by_key=facts_by_key,
                stock_rows_by_key=stock_rows_by_key,
            )
        )
        is not None
    ]
    platform_candidates = _select_broad_component_candidates(
        platform_all,
        role=SERVER_PLATFORM_ROLE,
        requirements=requirements,
        limit=limit_per_role,
    )
    platform_vendors = {
        candidate.facts.normalized_vendor
        for candidate in platform_candidates
        if candidate.facts.normalized_vendor != UNKNOWN_FACT
    }

    cpu_all = [
        candidate
        for product in products_by_role.get(CPU_ROLE, [])
        if (
            candidate := _score_cpu_candidate(
                product=product,
                requirements=requirements,
                platform_vendors=platform_vendors,
                facts_by_key=facts_by_key,
                stock_rows_by_key=stock_rows_by_key,
            )
        )
        is not None
    ]
    cpu_candidates = _select_broad_component_candidates(
        cpu_all,
        role=CPU_ROLE,
        requirements=requirements,
        limit=limit_per_role,
    )
    ram_all = [
        candidate
        for product in products_by_role.get(RAM_ROLE, [])
        if (
            candidate := _score_ram_candidate(
                product=product,
                requirements=requirements,
                facts_by_key=facts_by_key,
                stock_rows_by_key=stock_rows_by_key,
            )
        )
        is not None
    ]
    ram_candidates = _select_broad_component_candidates(
        ram_all,
        role=RAM_ROLE,
        requirements=requirements,
        limit=limit_per_role,
    )
    drive_all = [
        candidate
        for product in products_by_role.get(DRIVE_ROLE, [])
        if (
            candidate := _score_storage_candidate(
                role=DRIVE_ROLE,
                product=product,
                requirements=requirements,
                facts_by_key=facts_by_key,
                stock_rows_by_key=stock_rows_by_key,
            )
        )
        is not None
    ]
    drive_candidates = _select_broad_component_candidates(
        drive_all,
        role=DRIVE_ROLE,
        requirements=requirements,
        limit=limit_per_role,
    )
    ssd_all = [
        candidate
        for product in products_by_role.get(SSD_ROLE, [])
        if (
            candidate := _score_storage_candidate(
                role=SSD_ROLE,
                product=product,
                requirements=requirements,
                facts_by_key=facts_by_key,
                stock_rows_by_key=stock_rows_by_key,
            )
        )
        is not None
    ]
    ssd_candidates = _select_broad_component_candidates(
        ssd_all,
        role=SSD_ROLE,
        requirements=requirements,
        limit=limit_per_role,
    )
    hdd_all = [
        candidate
        for product in products_by_role.get(HDD_ROLE, [])
        if (
            candidate := _score_storage_candidate(
                role=HDD_ROLE,
                product=product,
                requirements=requirements,
                facts_by_key=facts_by_key,
                stock_rows_by_key=stock_rows_by_key,
            )
        )
        is not None
    ]
    hdd_candidates = _select_broad_component_candidates(
        hdd_all,
        role=HDD_ROLE,
        requirements=requirements,
        limit=limit_per_role,
    )
    controller_all = _score_simple_role_candidates(
        role=STORAGE_CONTROLLER_ROLE,
        products=products_by_role.get(STORAGE_CONTROLLER_ROLE, []),
        requirements=requirements,
        facts_by_key=facts_by_key,
        stock_rows_by_key=stock_rows_by_key,
    )
    controller_candidates = _select_broad_component_candidates(
        controller_all,
        role=STORAGE_CONTROLLER_ROLE,
        requirements=requirements,
        limit=limit_per_role,
    )
    network_all = _score_simple_role_candidates(
        role=NETWORK_ADAPTER_ROLE,
        products=products_by_role.get(NETWORK_ADAPTER_ROLE, []),
        requirements=requirements,
        facts_by_key=facts_by_key,
        stock_rows_by_key=stock_rows_by_key,
    )
    network_candidates = _select_broad_component_candidates(
        network_all,
        role=NETWORK_ADAPTER_ROLE,
        requirements=requirements,
        limit=limit_per_role,
    )
    generic_all_by_role: dict[str, list[_ComponentCandidate]] = {}
    generic_candidates_by_role: dict[str, list[_ComponentCandidate]] = {}
    for role in GENERIC_COMPONENT_ROLES:
        generic_all = _score_simple_role_candidates(
            role=role,
            products=products_by_role.get(role, []),
            requirements=requirements,
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
        )
        generic_all_by_role[role] = generic_all
        generic_candidates_by_role[role] = _select_broad_component_candidates(
            generic_all,
            role=role,
            requirements=requirements,
            limit=limit_per_role,
        )
    eligible_by_role = {
        READY_SERVER_ROLE: len(ready_server_all),
        SERVER_PLATFORM_ROLE: len(platform_all),
        CPU_ROLE: len(cpu_all),
        RAM_ROLE: len(ram_all),
        DRIVE_ROLE: len(drive_all),
        SSD_ROLE: len(ssd_all),
        HDD_ROLE: len(hdd_all),
        STORAGE_CONTROLLER_ROLE: len(controller_all),
        NETWORK_ADAPTER_ROLE: len(network_all),
        **{role: len(rows) for role, rows in generic_all_by_role.items()},
    }
    selected_by_role = {
        READY_SERVER_ROLE: ready_server_candidates,
        SERVER_PLATFORM_ROLE: platform_candidates,
        CPU_ROLE: cpu_candidates,
        RAM_ROLE: ram_candidates,
        DRIVE_ROLE: drive_candidates,
        SSD_ROLE: ssd_candidates,
        HDD_ROLE: hdd_candidates,
        STORAGE_CONTROLLER_ROLE: controller_candidates,
        NETWORK_ADAPTER_ROLE: network_candidates,
        **generic_candidates_by_role,
    }
    capability_id_by_role = _capability_id_by_role(role_plan)
    selected_by_role = {
        role: _with_capability_id(rows, capability_id_by_role.get(role))
        for role, rows in selected_by_role.items()
    }
    ready_server_candidates = selected_by_role[READY_SERVER_ROLE]
    platform_candidates = selected_by_role[SERVER_PLATFORM_ROLE]
    cpu_candidates = selected_by_role[CPU_ROLE]
    ram_candidates = selected_by_role[RAM_ROLE]
    drive_candidates = selected_by_role[DRIVE_ROLE]
    ssd_candidates = selected_by_role[SSD_ROLE]
    hdd_candidates = selected_by_role[HDD_ROLE]
    controller_candidates = selected_by_role[STORAGE_CONTROLLER_ROLE]
    network_candidates = selected_by_role[NETWORK_ADAPTER_ROLE]
    generic_candidates_by_role = {
        role: selected_by_role.get(role, []) for role in GENERIC_COMPONENT_ROLES
    }
    coverage_summary = _component_matrix_coverage_summary(
        products_by_role=products_by_role,
        eligible_by_role=eligible_by_role,
        selected_by_role=selected_by_role,
        role_filter_diagnostics=_role_filter_diagnostics(
            products_by_role=products_by_role,
            eligible_by_role=eligible_by_role,
            requirements=requirements,
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
        ),
        limit_per_role=limit_per_role,
    )
    required_roles = _string_list(role_plan.get("required_roles"))
    required_capabilities = _mapping_list(role_plan.get("required_capabilities"))
    role_coverage_summary = _role_coverage_summary(
        required_roles=required_roles,
        required_capabilities=required_capabilities,
        category_plan=category_plan,
        selected_by_role=selected_by_role,
        planner_role_coverage=planner_role_coverage,
        role_filter_diagnostics=coverage_summary.get("role_filter_diagnostics", {}),
        platform_satisfaction_counts=_platform_satisfaction_counts(
            platform_candidates,
            requirements,
        ),
    )
    missing_required_roles = [
        role
        for role, coverage in role_coverage_summary.items()
        if coverage.get("required") and coverage.get("missing")
    ]
    missing_required_capabilities = _missing_required_capabilities(
        required_capabilities=required_capabilities,
        role_coverage_summary=role_coverage_summary,
        unsupported_or_unmapped=_string_list(
            role_plan.get("unsupported_or_unmapped_requirements")
        ),
    )
    materialized_matrix_roles = _materialized_matrix_roles_from_coverage(
        role_coverage_summary
    )
    validated_category_plan_roles = role_lifecycle.roles_from_category_plan(
        category_plan,
        product_group=requirements.product_group,
    )
    roles_dropped_during_materialization = _roles_dropped_during_materialization(
        validated_category_plan_roles=validated_category_plan_roles,
        materialized_matrix_roles=materialized_matrix_roles,
        role_coverage_summary=role_coverage_summary,
    )
    role_source_by_role = role_lifecycle.merge_role_sources(
        existing=role_plan.get("role_source_by_role")
        if isinstance(role_plan.get("role_source_by_role"), Mapping)
        else {},
    )
    roles_dropped_reason_by_role = role_lifecycle.merge_drop_reasons(
        _materialization_drop_reasons(
            roles_dropped_during_materialization,
            role_coverage_summary,
        ),
        existing=role_plan.get("roles_dropped_reason_by_role")
        if isinstance(role_plan.get("roles_dropped_reason_by_role"), Mapping)
        else {},
    )
    role_lifecycle_trace = role_lifecycle.build_role_lifecycle_trace(
        [
            *required_roles,
            *validated_category_plan_roles,
            *materialized_matrix_roles,
        ],
        role_source_by_role=role_source_by_role,
        stage_a_roles=_string_list(role_plan.get("stage_a_broad_roles")),
        semantic_matrix_blueprint_roles=_string_list(
            role_plan.get("semantic_matrix_blueprint_roles")
        ),
        requirement_classifier_roles=_string_list(
            role_plan.get("requirement_classifier_roles")
        ),
        before_category_planner_roles=_string_list(
            role_plan.get("effective_matrix_roles_before_category_planner")
        ),
        category_planner_input_roles=_string_list(
            role_plan.get("category_planner_input_roles")
        ),
        category_planner_output_roles=_string_list(
            role_plan.get("category_planner_output_roles")
        ),
        validated_category_plan_roles=validated_category_plan_roles,
        materialized_matrix_roles=materialized_matrix_roles,
        dropped_reason_by_role=roles_dropped_reason_by_role,
    )

    return _CandidateMatrix(
        normalized_requirements=requirements,
        ready_server_candidates=ready_server_candidates,
        platform_candidates=platform_candidates,
        cpu_candidates=cpu_candidates,
        ram_candidates=ram_candidates,
        drive_candidates=drive_candidates,
        ssd_candidates=ssd_candidates,
        hdd_candidates=hdd_candidates,
        storage_controller_candidates=controller_candidates,
        network_adapter_candidates=network_candidates,
        generic_role_candidates=generic_candidates_by_role,
        coverage_summary=coverage_summary,
        role_plan=dict(role_plan),
        category_plan={
            str(role): list(ids)
            for role, ids in category_plan.items()
            if isinstance(ids, list)
        },
        category_plan_entries=category_plan_entries,
        category_catalog_summary=dict(category_catalog_summary),
        category_planner_source=category_planner_source,
        category_plan_source=category_plan_source,
        category_planner_missing_required_roles=_string_list(
            category_planner_missing_required_roles
        ),
        category_planner_repair_attempted=bool(category_planner_repair_attempted),
        category_planner_repair_success=bool(category_planner_repair_success),
        category_planner_repair_reason=category_planner_repair_reason,
        category_planner_repaired_roles=_string_list(category_planner_repaired_roles),
        category_planner_unresolved_required_roles=_string_list(
            category_planner_unresolved_required_roles
        ),
        required_capabilities=_mapping_list(role_plan.get("required_capabilities")),
        optional_capabilities=_mapping_list(role_plan.get("optional_capabilities")),
        unsupported_or_unmapped_requirements=_string_list(
            role_plan.get("unsupported_or_unmapped_requirements")
        ),
        required_roles=required_roles,
        missing_required_roles=missing_required_roles,
        missing_category_roles=missing_category_roles,
        missing_required_capabilities=missing_required_capabilities,
        role_coverage_summary=role_coverage_summary,
        category_plan_warnings=category_plan_warnings,
        stage_a_broad_roles=_string_list(role_plan.get("stage_a_broad_roles")),
        semantic_matrix_blueprint_roles=_string_list(
            role_plan.get("semantic_matrix_blueprint_roles")
        ),
        requirement_classifier_roles=_string_list(
            role_plan.get("requirement_classifier_roles")
        ),
        effective_matrix_roles_before_category_planner=_string_list(
            role_plan.get("effective_matrix_roles_before_category_planner")
        ),
        category_planner_input_roles=_string_list(
            role_plan.get("category_planner_input_roles")
        ),
        category_planner_output_roles=_string_list(
            role_plan.get("category_planner_output_roles")
        ),
        validated_category_plan_roles=validated_category_plan_roles,
        materialized_matrix_roles=materialized_matrix_roles,
        roles_dropped_after_stage_a=_string_list(
            role_plan.get("roles_dropped_after_stage_a")
        ),
        roles_dropped_before_category_planner=_string_list(
            role_plan.get("roles_dropped_before_category_planner")
        ),
        roles_dropped_after_category_planner=_string_list(
            role_plan.get("roles_dropped_after_category_planner")
        ),
        roles_dropped_during_materialization=roles_dropped_during_materialization,
        roles_dropped_reason_by_role=roles_dropped_reason_by_role,
        role_source_by_role=role_source_by_role,
        role_lifecycle_trace=role_lifecycle_trace,
    )


def _score_simple_role_candidates(
    *,
    role: str,
    products: list[DistributorProduct],
    requirements: _NormalizedServerRequirements,
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> list[_ComponentCandidate]:
    candidates: list[_ComponentCandidate] = []
    for product in products:
        candidate = _score_simple_candidate(
            role=role,
            product=product,
            requirements=requirements,
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _select_broad_component_candidates(
    candidates: list[_ComponentCandidate],
    *,
    role: str,
    requirements: _NormalizedServerRequirements,
    limit: int,
) -> list[_ComponentCandidate]:
    candidates = [
        candidate for candidate in candidates if candidate.fit_tier in ACTIVE_FIT_TIERS
    ]
    if not candidates:
        return []

    selected: dict[str, _ComponentCandidate] = {}
    for bucket_name, bucket_candidates in _candidate_buckets(
        candidates,
        role=role,
        requirements=requirements,
    ):
        for candidate in sorted(bucket_candidates, key=_bucket_sort_key(bucket_name))[
            :BUCKET_SAMPLE_SIZE
        ]:
            _select_bucket_candidate(selected, candidate, bucket_name)

    for candidate in sorted(candidates, key=_bucket_member_sort_key):
        if len(selected) >= limit:
            break
        _select_bucket_candidate(selected, candidate, "score_fill")

    return _limit_active_fit_tiers(
        sorted(selected.values(), key=_matrix_candidate_sort_key),
        role=role,
        limit=limit,
    )


def _select_bucket_candidate(
    selected: dict[str, _ComponentCandidate],
    candidate: _ComponentCandidate,
    bucket_name: str,
) -> None:
    current = selected.get(candidate.candidate_id)
    bucket_priority = _bucket_priority(candidate.role, bucket_name)
    if current is not None and current.bucket_priority <= bucket_priority:
        return
    selected[candidate.candidate_id] = replace(
        candidate,
        selection_bucket=bucket_name,
        bucket_priority=bucket_priority,
    )


def _candidate_buckets(
    candidates: list[_ComponentCandidate],
    *,
    role: str,
    requirements: _NormalizedServerRequirements,
) -> list[tuple[str, list[_ComponentCandidate]]]:
    buckets: list[tuple[str, list[_ComponentCandidate]]] = []
    if role == CPU_ROLE:
        buckets.extend(_cpu_candidate_buckets(candidates, requirements))
    elif role == RAM_ROLE:
        buckets.extend(_ram_candidate_buckets(candidates, requirements))
    elif role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}:
        buckets.extend(_storage_candidate_buckets(candidates, requirements))
    elif role == SERVER_PLATFORM_ROLE:
        buckets.extend(_platform_candidate_buckets(candidates))
    else:
        buckets.extend(_generic_candidate_buckets(candidates))
    buckets.append(
        (
            "cheapest",
            [candidate for candidate in candidates if candidate.price_value is not None],
        )
    )
    buckets.append(
        (
            "stock_enough",
            [
                candidate
                for candidate in candidates
                if _candidate_has_enough_stock(candidate)
            ],
        )
    )
    buckets.append(("score", candidates))
    return [(name, rows) for name, rows in buckets if rows]


def _cpu_candidate_buckets(
    candidates: list[_ComponentCandidate],
    requirements: _NormalizedServerRequirements,
) -> list[tuple[str, list[_ComponentCandidate]]]:
    buckets: list[tuple[str, list[_ComponentCandidate]]] = []
    if requirements.cpu_min_cores_per_cpu is not None:
        buckets.append(
            (
                "closest_to_min_cores",
                [
                    candidate
                    for candidate in candidates
                    if candidate.facts.cpu_cores is not None
                ],
            )
        )
    for core_count in CPU_CORE_BUCKETS:
        buckets.append(
            (
                f"cpu_{core_count}_cores",
                [
                    candidate
                    for candidate in candidates
                    if candidate.facts.cpu_cores is not None
                    and _cpu_core_bucket(candidate.facts.cpu_cores) == core_count
                ],
            )
        )
    if (
        requirements.cpu_vendor_preference == "Intel"
        or requirements.cpu_family_preference == "Xeon"
    ):
        buckets.append(
            (
                "intel_xeon",
                [
                    candidate
                    for candidate in candidates
                    if candidate.facts.cpu_brand == "Intel" or candidate.facts.cpu_family == "Xeon"
                ],
            )
        )
    if requirements.cpu_vendor_preference == "AMD" or requirements.cpu_family_preference == "EPYC":
        buckets.append(
            (
                "amd_epyc",
                [
                    candidate
                    for candidate in candidates
                    if candidate.facts.cpu_brand == "AMD" or candidate.facts.cpu_family == "EPYC"
                ],
            )
        )
    buckets.append(
        (
            "unknown_cores_good_stock",
            [
                candidate
                for candidate in candidates
                if candidate.facts.cpu_cores is None
                and (candidate.price_value is not None or _candidate_has_enough_stock(candidate))
            ],
        )
    )
    return buckets


def _ram_candidate_buckets(
    candidates: list[_ComponentCandidate],
    requirements: _NormalizedServerRequirements,
) -> list[tuple[str, list[_ComponentCandidate]]]:
    buckets: list[tuple[str, list[_ComponentCandidate]]] = []
    for ram_type in ("DDR4", "DDR5"):
        buckets.append(
            (
                f"ram_{ram_type.lower()}",
                [candidate for candidate in candidates if candidate.facts.ram_type == ram_type],
            )
        )
    for capacity_gb in RAM_MODULE_BUCKETS_GB:
        buckets.append(
            (
                f"ram_{capacity_gb}gb_modules",
                [
                    candidate
                    for candidate in candidates
                    if candidate.facts.ram_capacity_gb == capacity_gb
                ],
            )
        )
    if requirements.ram_gb_per_server is not None:
        buckets.append(
            (
                "closest_to_required_ram_total",
                [
                    candidate
                    for candidate in candidates
                    if candidate.ram_over_requirement_gb is not None
                ],
            )
        )
    return buckets


def _storage_candidate_buckets(
    candidates: list[_ComponentCandidate],
    requirements: _NormalizedServerRequirements,
) -> list[tuple[str, list[_ComponentCandidate]]]:
    buckets: list[tuple[str, list[_ComponentCandidate]]] = []
    for interface in ("NVMe", "SAS", "SATA"):
        buckets.append(
            (
                f"storage_{interface.lower()}",
                [
                    candidate
                    for candidate in candidates
                    if candidate.facts.storage_interface == interface
                ],
            )
        )
    for capacity_tb in STORAGE_CAPACITY_BUCKETS_TB:
        buckets.append(
            (
                f"storage_{_bucket_number(capacity_tb)}tb",
                [
                    candidate
                    for candidate in candidates
                    if candidate.facts.storage_capacity_tb is not None
                    and abs(candidate.facts.storage_capacity_tb - capacity_tb) <= 0.01
                ],
            )
        )
    if requirements.storage_min_capacity_tb is not None:
        buckets.append(
            (
                "closest_to_required_storage",
                [
                    candidate
                    for candidate in candidates
                    if candidate.storage_over_requirement is not None
                ],
            )
        )
    return buckets


def _platform_candidate_buckets(
    candidates: list[_ComponentCandidate],
) -> list[tuple[str, list[_ComponentCandidate]]]:
    buckets: list[tuple[str, list[_ComponentCandidate]]] = []
    vendors = sorted(
        {
            candidate.facts.normalized_vendor
            for candidate in candidates
            if candidate.facts.normalized_vendor != UNKNOWN_FACT
        }
    )
    for vendor in vendors:
        buckets.append(
            (
                f"platform_vendor_{vendor.casefold()}",
                [
                    candidate
                    for candidate in candidates
                    if candidate.facts.normalized_vendor == vendor
                ],
            )
        )
    buckets.append(
        (
            "platform_2u",
            [
                candidate
                for candidate in candidates
                if "2U" in {hint.upper() for hint in candidate.facts.form_factor_hints}
            ],
        )
    )
    buckets.append(
        (
            "platform_ddr5",
            [candidate for candidate in candidates if candidate.facts.ram_type == "DDR5"],
        )
    )
    buckets.append(
        (
            "technical_clean",
            [
                candidate
                for candidate in candidates
                if not _fatal_warning_text(candidate.eligibility_warnings)
            ],
        )
    )
    return buckets


def _generic_candidate_buckets(
    candidates: list[_ComponentCandidate],
) -> list[tuple[str, list[_ComponentCandidate]]]:
    return [
        (
            "technical_clean",
            [
                candidate
                for candidate in candidates
                if not _fatal_warning_text(candidate.eligibility_warnings)
            ],
        )
    ]


def _bucket_member_sort_key(candidate: _ComponentCandidate) -> tuple[Any, ...]:
    return (
        _fit_tier_rank(candidate.fit_tier),
        *_component_over_requirement_sort_key(candidate),
        *_component_price_sort_key(candidate),
        -candidate.score,
        -(_stock_sort_value(candidate.available_quantity)),
        _stable_text(candidate.facts.normalized_vendor or candidate.product.producer),
        _stable_text(candidate.product.part_number),
        _stable_text(candidate.candidate_id),
    )


def _bucket_price_sort_key(candidate: _ComponentCandidate) -> tuple[Any, ...]:
    return (
        _fit_tier_rank(candidate.fit_tier),
        *_component_price_sort_key(candidate),
        *_component_over_requirement_sort_key(candidate),
        -candidate.score,
        -(_stock_sort_value(candidate.available_quantity)),
        _stable_text(candidate.facts.normalized_vendor or candidate.product.producer),
        _stable_text(candidate.product.part_number),
        _stable_text(candidate.candidate_id),
    )


def _bucket_sort_key(bucket_name: str) -> Any:
    if bucket_name == "cheapest":
        return _bucket_price_sort_key
    return _bucket_member_sort_key


def _matrix_candidate_sort_key(candidate: _ComponentCandidate) -> tuple[Any, ...]:
    return (
        _fit_tier_rank(candidate.fit_tier),
        candidate.bucket_priority,
        -candidate.score,
        *_component_price_sort_key(candidate),
        *_component_over_requirement_sort_key(candidate),
        -(_stock_sort_value(candidate.available_quantity)),
        _stable_text(candidate.facts.normalized_vendor or candidate.product.producer),
        _stable_text(candidate.product.part_number),
        _stable_text(candidate.candidate_id),
    )


def _bucket_priority(role: str, bucket_name: str) -> int:
    priority_by_name = {
        "closest_to_min_cores": 6,
        "closest_to_required_ram_total": 6,
        "closest_to_required_storage": 6,
        "storage_3.84tb": 1,
        "cpu_16_cores": 1,
        "cpu_20_cores": 2,
        "cpu_24_cores": 3,
        "cpu_32_cores": 7,
        "cpu_48_cores": 8,
        "cpu_64_cores": 9,
        "ram_ddr5": 4,
        "ram_ddr4": 5,
        "storage_nvme": 4,
        "storage_sas": 5,
        "storage_sata": 6,
        "intel_xeon": 10,
        "amd_epyc": 10,
        "stock_enough": 18,
        "cheapest": 20,
        "technical_clean": 24,
        "unknown_cores_good_stock": 30,
        "score": 40,
        "score_fill": 50,
    }
    if bucket_name.startswith(("platform_vendor_", "ram_", "storage_")):
        return priority_by_name.get(bucket_name, 12)
    if role == READY_SERVER_ROLE and bucket_name == "score":
        return 15
    return priority_by_name.get(bucket_name, 35)


def _fit_tier_rank(fit_tier: str | None) -> int:
    return FIT_TIER_RANK.get(str(fit_tier or ""), FIT_TIER_RANK[FIT_TIER_FALLBACK_UNKNOWN])


def _limit_active_fit_tiers(
    candidates: list[_ComponentCandidate],
    *,
    role: str,
    limit: int,
) -> list[_ComponentCandidate]:
    primary = [
        candidate
        for candidate in candidates
        if candidate.fit_tier in {FIT_TIER_STRONG, FIT_TIER_POSSIBLE}
    ]
    fallback = [
        candidate
        for candidate in candidates
        if candidate.fit_tier == FIT_TIER_FALLBACK_UNKNOWN
    ]
    fallback_limit = _fallback_unknown_limit(role, limit, has_primary=bool(primary))
    return [*primary, *fallback[:fallback_limit]][:limit]


def _fallback_unknown_limit(role: str, limit: int, *, has_primary: bool) -> int:
    if not has_primary:
        return min(limit, 3)
    if role in {
        SWITCH_ROLE,
        ROUTER_ROLE,
        FIREWALL_ROLE,
        ACCESS_POINT_ROLE,
        STORAGE_SYSTEM_ROLE,
        HOST_PORT_ROLE,
        PROTOCOL_MODULE_ROLE,
    }:
        return min(3, max(1, limit // 10))
    return min(5, max(1, limit // 5))


def _cpu_core_bucket(core_count: int) -> int:
    for bucket in CPU_CORE_BUCKETS:
        if core_count <= bucket:
            return bucket
    return CPU_CORE_BUCKETS[-1]


def _bucket_number(value: float) -> str:
    text = format(Decimal(str(value)).normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _candidate_has_enough_stock(candidate: _ComponentCandidate) -> bool:
    return (
        candidate.available_quantity is not None
        and candidate.available_quantity >= candidate.quantity_required
    )


def _stock_sort_value(value: int | None) -> int:
    return value if value is not None else -1


def _component_matrix_coverage_summary(
    *,
    products_by_role: dict[str, list[DistributorProduct]],
    eligible_by_role: dict[str, int],
    selected_by_role: dict[str, list[_ComponentCandidate]],
    role_filter_diagnostics: Mapping[str, Any],
    limit_per_role: int,
) -> dict[str, Any]:
    total_products_by_role: dict[str, int] = {}
    eligible_products_by_role: dict[str, int] = {}
    sent_to_llm_by_role: dict[str, int] = {}
    omitted_by_role: dict[str, int] = {}
    bucket_summary_by_role: dict[str, dict[str, int]] = {}
    fit_tier_summary_by_role: dict[str, dict[str, int]] = {}
    for role in MATRIX_ROLE_ORDER:
        prompt_role = MATRIX_PROMPT_ROLE_BY_INTERNAL_ROLE[role]
        total = len(products_by_role.get(role, []))
        eligible = eligible_by_role.get(role, 0)
        selected = selected_by_role.get(role, [])
        total_products_by_role[prompt_role] = total
        eligible_products_by_role[prompt_role] = eligible
        sent_to_llm_by_role[prompt_role] = len(selected)
        omitted_by_role[prompt_role] = max(0, eligible - len(selected))
        bucket_summary_by_role[prompt_role] = _bucket_counts(selected)
        fit_tier_summary_by_role[prompt_role] = _fit_tier_counts(selected)

    return {
        "total_products_by_role": total_products_by_role,
        "eligible_products_by_role": eligible_products_by_role,
        "sent_to_llm_by_role": sent_to_llm_by_role,
        "omitted_by_role": omitted_by_role,
        "limit_per_role": limit_per_role,
        "bucket_summary_by_role": bucket_summary_by_role,
        "fit_tier_summary_by_role": fit_tier_summary_by_role,
        "role_filter_diagnostics": _jsonable(dict(role_filter_diagnostics)),
        "selection_strategy": "bucketed_broad_matrix_v3",
    }


def _role_coverage_summary(
    *,
    required_roles: list[str],
    required_capabilities: list[dict[str, Any]],
    category_plan: Mapping[str, list[str]],
    selected_by_role: dict[str, list[_ComponentCandidate]],
    planner_role_coverage: Mapping[str, Any],
    role_filter_diagnostics: Mapping[str, Any],
    platform_satisfaction_counts: Mapping[str, int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for role in required_roles:
        category_ids = _category_ids_for_coverage_role(role, category_plan)
        candidate_count = _candidate_count_for_planned_role(role, selected_by_role)
        candidate_fit_tiers = _fit_tier_counts_for_planned_role(role, selected_by_role)
        planner_row = planner_role_coverage.get(role)
        planner_mapping = planner_row if isinstance(planner_row, Mapping) else {}
        platform_satisfied_count = int(platform_satisfaction_counts.get(role, 0) or 0)
        platform_satisfied = platform_satisfied_count > 0
        can_be_satisfied_by_platform = bool(
            planner_mapping.get("can_be_satisfied_by_platform")
        ) or _role_can_be_satisfied_by_platform(required_capabilities, role)
        missing_category = (
            bool(planner_mapping.get("missing_category")) or not category_ids
        ) and not platform_satisfied
        missing_candidates = candidate_count == 0 and not platform_satisfied
        filter_diagnostics = _combined_filter_diagnostics_for_planned_role(
            role,
            role_filter_diagnostics,
        )
        summary[role] = {
            "required": True,
            "category_ids": category_ids,
            "category_count": len(category_ids),
            "sent_to_llm_count": candidate_count,
            "fit_tier_counts": candidate_fit_tiers,
            "raw_products_count": _as_int(filter_diagnostics.get("raw_products_count")) or 0,
            "after_category_count": _as_int(filter_diagnostics.get("after_category_count")) or 0,
            "after_fact_extraction_count": _as_int(
                filter_diagnostics.get("after_fact_extraction_count")
            )
            or 0,
            "after_eligibility_count": _as_int(
                filter_diagnostics.get("after_eligibility_count")
            )
            or 0,
            "filtered_reasons_top": _mapping_list(
                filter_diagnostics.get("filtered_reasons_top")
            ),
            "sample_filtered_products": _mapping_list(
                filter_diagnostics.get("sample_filtered_products")
            ),
            "missing_category": missing_category,
            "missing_candidates": missing_candidates,
            "missing": missing_category or missing_candidates,
            "can_be_satisfied_by_platform": can_be_satisfied_by_platform,
            "platform_satisfied_candidates_count": platform_satisfied_count,
            "filter_diagnostics": _jsonable(filter_diagnostics),
        }
    return summary


def _materialized_matrix_roles_from_coverage(
    role_coverage_summary: Mapping[str, Any],
) -> list[str]:
    roles: list[str] = []
    for role, coverage in role_coverage_summary.items():
        if not isinstance(coverage, Mapping):
            continue
        if (_as_int(coverage.get("sent_to_llm_count")) or 0) > 0:
            roles.append(str(role))
    return _unique(roles)


def _roles_dropped_during_materialization(
    *,
    validated_category_plan_roles: Sequence[str],
    materialized_matrix_roles: Sequence[str],
    role_coverage_summary: Mapping[str, Any],
) -> list[str]:
    materialized = set(materialized_matrix_roles)
    result: list[str] = []
    for role in validated_category_plan_roles:
        if role in materialized:
            continue
        coverage = role_coverage_summary.get(role)
        if not isinstance(coverage, Mapping):
            result.append(role)
            continue
        if coverage.get("missing_category"):
            continue
        if coverage.get("missing_candidates") or (
            (_as_int(coverage.get("category_count")) or 0) > 0
            and (_as_int(coverage.get("sent_to_llm_count")) or 0) <= 0
        ):
            result.append(role)
    return _unique(result)


def _materialization_drop_reasons(
    roles: Sequence[str],
    role_coverage_summary: Mapping[str, Any],
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for role in roles:
        coverage = role_coverage_summary.get(role)
        if not isinstance(coverage, Mapping):
            reasons[role] = "not_materialized"
            continue
        after_category = _as_int(coverage.get("after_category_count")) or 0
        after_eligibility = _as_int(coverage.get("after_eligibility_count")) or 0
        if after_category <= 0:
            reason = "no_products_after_category_materialization"
        elif after_eligibility <= 0:
            reason = "no_products_after_eligibility_filter"
        else:
            reason = "no_selectable_candidates_after_materialization"
        reasons[role] = (
            f"{reason}:after_category_count={after_category}:"
            f"after_eligibility_count={after_eligibility}"
        )
    return reasons


def _category_ids_for_coverage_role(
    role: str,
    category_plan: Mapping[str, list[str]],
) -> list[str]:
    role_aliases = {
        role,
        *_coverage_category_plan_aliases(role),
    }
    result: list[str] = []
    for alias in role_aliases:
        ids = category_plan.get(alias, [])
        if not isinstance(ids, list):
            continue
        for category_id in ids:
            text = str(category_id or "").strip()
            if text and text not in result:
                result.append(text)
    return result


def _coverage_category_plan_aliases(role: str) -> list[str]:
    if role == SERVER_PLATFORM_ROLE:
        return ["platform"]
    if role == "storage":
        return [DRIVE_ROLE, SSD_ROLE, HDD_ROLE]
    if role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}:
        return ["storage"]
    if role == CABLE_ROLE:
        return ["power_cable", "power_cord"]
    return []


def _combined_filter_diagnostics_for_planned_role(
    role: str,
    role_filter_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    internal_roles = _internal_roles_for_planned_role(role)
    prompt_roles = [
        MATRIX_PROMPT_ROLE_BY_INTERNAL_ROLE.get(internal_role, internal_role)
        for internal_role in internal_roles
    ] or [MATRIX_PROMPT_ROLE_BY_INTERNAL_ROLE.get(role, role)]
    rows = [
        _dict_or_empty(role_filter_diagnostics.get(prompt_role, {}))
        for prompt_role in prompt_roles
    ]
    rows = [row for row in rows if row]
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]
    reason_counts: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    for row in rows:
        for reason_row in _mapping_list(row.get("filtered_reasons_top")):
            reason = str(reason_row.get("reason") or "").strip()
            if not reason:
                continue
            reason_counts[reason] = reason_counts.get(reason, 0) + (
                _as_int(reason_row.get("count")) or 0
            )
        for sample in _mapping_list(row.get("sample_filtered_products")):
            if len(samples) < 5:
                samples.append(sample)
    return {
        "raw_products_count": sum(_as_int(row.get("raw_products_count")) or 0 for row in rows),
        "after_category_count": sum(
            _as_int(row.get("after_category_count")) or 0 for row in rows
        ),
        "after_fact_extraction_count": sum(
            _as_int(row.get("after_fact_extraction_count")) or 0 for row in rows
        ),
        "after_eligibility_count": sum(
            _as_int(row.get("after_eligibility_count")) or 0 for row in rows
        ),
        "filtered_reasons_top": [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                reason_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
        ],
        "sample_filtered_products": samples,
    }


def _candidate_count_for_planned_role(
    role: str,
    selected_by_role: dict[str, list[_ComponentCandidate]],
) -> int:
    if role == "storage":
        return (
            len(selected_by_role.get(STORAGE_SYSTEM_ROLE, []))
            + len(selected_by_role.get(DRIVE_ROLE, []))
            + len(selected_by_role.get(SSD_ROLE, []))
            + len(selected_by_role.get(HDD_ROLE, []))
        )
    role_map = {
        "switch": SWITCH_ROLE,
        "router": ROUTER_ROLE,
        "firewall": FIREWALL_ROLE,
        "access_point": ACCESS_POINT_ROLE,
        "storage_system": STORAGE_SYSTEM_ROLE,
        "storage_array": STORAGE_SYSTEM_ROLE,
        "controller": STORAGE_ARRAY_CONTROLLER_ROLE,
        "controller_module": CONTROLLER_MODULE_ROLE,
        "disk_shelf": DISK_SHELF_ROLE,
        "shelf": DISK_SHELF_ROLE,
        "drive": DRIVE_ROLE,
        "drives": DRIVE_ROLE,
        "ssd": SSD_ROLE,
        "hdd": HDD_ROLE,
        "cache": CACHE_ROLE,
        "host_port": HOST_PORT_ROLE,
        "host_ports": HOST_PORT_ROLE,
        "protocol_module": PROTOCOL_MODULE_ROLE,
        "server_platform": SERVER_PLATFORM_ROLE,
        "cpu": CPU_ROLE,
        "ram": RAM_ROLE,
        "storage_controller": STORAGE_CONTROLLER_ROLE,
        "network_adapter": NETWORK_ADAPTER_ROLE,
        "gpu": GPU_ROLE,
        "transceiver": TRANSCEIVER_ROLE,
        "dac_cable": DAC_CABLE_ROLE,
        "cable": CABLE_ROLE,
        "power_supply": POWER_SUPPLY_ROLE,
        "rail_kit": RAIL_KIT_ROLE,
        "license": LICENSE_ROLE,
        "support": SUPPORT_ROLE,
        "stacking_module": STACKING_MODULE_ROLE,
        "other_accessory": OTHER_ACCESSORY_ROLE,
        "unmapped": UNMAPPED_ROLE,
    }
    internal_role = role_map.get(role)
    if internal_role is None:
        return 0
    return len(selected_by_role.get(internal_role, []))


def _fit_tier_counts_for_planned_role(
    role: str,
    selected_by_role: dict[str, list[_ComponentCandidate]],
) -> dict[str, int]:
    roles = _internal_roles_for_planned_role(role)
    counts: dict[str, int] = {}
    for internal_role in roles:
        for candidate in selected_by_role.get(internal_role, []):
            counts[candidate.fit_tier] = counts.get(candidate.fit_tier, 0) + 1
    return dict(sorted(counts.items()))


def _internal_roles_for_planned_role(role: str) -> list[str]:
    if role == "storage":
        return [STORAGE_SYSTEM_ROLE, DRIVE_ROLE, SSD_ROLE, HDD_ROLE]
    role_map = {
        "switch": SWITCH_ROLE,
        "router": ROUTER_ROLE,
        "firewall": FIREWALL_ROLE,
        "access_point": ACCESS_POINT_ROLE,
        "storage_system": STORAGE_SYSTEM_ROLE,
        "storage_array": STORAGE_SYSTEM_ROLE,
        "controller": STORAGE_ARRAY_CONTROLLER_ROLE,
        "controller_module": CONTROLLER_MODULE_ROLE,
        "disk_shelf": DISK_SHELF_ROLE,
        "shelf": DISK_SHELF_ROLE,
        "drive": DRIVE_ROLE,
        "drives": DRIVE_ROLE,
        "ssd": SSD_ROLE,
        "hdd": HDD_ROLE,
        "cache": CACHE_ROLE,
        "host_port": HOST_PORT_ROLE,
        "host_ports": HOST_PORT_ROLE,
        "protocol_module": PROTOCOL_MODULE_ROLE,
        "server_platform": SERVER_PLATFORM_ROLE,
        "cpu": CPU_ROLE,
        "ram": RAM_ROLE,
        "storage_controller": STORAGE_CONTROLLER_ROLE,
        "network_adapter": NETWORK_ADAPTER_ROLE,
        "gpu": GPU_ROLE,
        "transceiver": TRANSCEIVER_ROLE,
        "dac_cable": DAC_CABLE_ROLE,
        "cable": CABLE_ROLE,
        "power_supply": POWER_SUPPLY_ROLE,
        "rail_kit": RAIL_KIT_ROLE,
        "license": LICENSE_ROLE,
        "support": SUPPORT_ROLE,
        "stacking_module": STACKING_MODULE_ROLE,
        "other_accessory": OTHER_ACCESSORY_ROLE,
        "unmapped": UNMAPPED_ROLE,
    }
    internal_role = role_map.get(role)
    return [internal_role] if internal_role else []


def _platform_satisfaction_counts(
    platform_candidates: Sequence[_ComponentCandidate],
    requirements: _NormalizedServerRequirements,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if requirements.network_required:
        counts[NETWORK_ADAPTER_ROLE] = sum(
            1
            for candidate in platform_candidates
            if _platform_onboard_network_satisfies(candidate, requirements)
        )
    if requirements.psu_count_per_server is not None:
        counts[POWER_SUPPLY_ROLE] = sum(
            1
            for candidate in platform_candidates
            if _platform_power_bundle_satisfies(candidate, requirements)
        )
    return counts


def _platform_power_bundle_satisfies(
    platform_candidate: _ComponentCandidate,
    requirements: _NormalizedServerRequirements,
) -> bool:
    if requirements.psu_count_per_server is None:
        return False
    return platform_power_bundle_satisfies(
        _product_search_text(platform_candidate.product),
        required_psu_count=requirements.psu_count_per_server,
        raw_json=platform_candidate.product.raw_json,
    )


def _role_can_be_satisfied_by_platform(
    required_capabilities: Sequence[Mapping[str, Any]],
    role: str,
) -> bool:
    matching = [
        capability
        for capability in required_capabilities
        if str(capability.get("role") or "").strip() == role
        and capability.get("hard", True)
    ]
    if not matching:
        return False
    return all(bool(capability.get("can_be_satisfied_by_platform")) for capability in matching)


def _missing_required_capabilities(
    *,
    required_capabilities: list[dict[str, Any]],
    role_coverage_summary: Mapping[str, Any],
    unsupported_or_unmapped: list[str],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = [
        {
            "capability_id": f"unsupported.{index + 1}",
            "role": "unsupported",
            "status": "unsupported",
            "satisfied_by": None,
            "component_role": None,
            "component_candidate_id": None,
            "source_text": requirement,
            "reason": requirement,
            "user_message": requirement,
        }
        for index, requirement in enumerate(unsupported_or_unmapped)
    ]
    for capability in required_capabilities:
        fulfillment_mode = str(capability.get("fulfillment_mode") or "").strip()
        if fulfillment_mode in {
            "included_in_primary_object",
            "included_in_selected_component",
            "included_in_bundle_or_kit",
            "unverified_requires_confirmation",
            "logistics_constraint",
            "engineering_check_only",
            "not_applicable",
        }:
            continue
        role = str(capability.get("role") or "").strip()
        coverage = role_coverage_summary.get(role)
        if not isinstance(coverage, Mapping) or not coverage.get("missing"):
            continue
        status = "missing"
        if coverage.get("missing_category"):
            status = "missing_category"
        elif coverage.get("missing_candidates"):
            status = "missing_candidates"
        missing.append(
            {
                "capability_id": capability.get("capability_id") or f"{role}.required",
                "role": role,
                "status": status,
                "satisfied_by": None,
                "component_role": None,
                "component_candidate_id": None,
                "source_text": capability.get("source_text")
                or capability.get("requirement_text")
                or role,
                "reason": _missing_capability_reason(capability, role, status),
                "user_message": _missing_capability_user_message(capability, status),
            }
        )
    return missing


def _missing_capability_reason(
    capability: Mapping[str, Any],
    role: str,
    status: str,
) -> str:
    if role == NETWORK_ADAPTER_ROLE:
        if status == "missing_candidates":
            label = _network_capability_label_from_capability(capability)
            return (
                f"Category was found, but no product candidate satisfied {label} "
                "with enough confirmed ports and stock."
            )
        return (
            "No onboard platform network or selected network adapter candidate covers "
            "the requested speed/media/ports."
        )
    if status == "missing_category":
        return "No valid distributor category was selected for this hard capability."
    if status == "missing_candidates":
        return "Selected distributor categories produced no eligible product candidates."
    return "No valid catalog category or product candidate for hard capability."


def _missing_capability_user_message(
    capability: Mapping[str, Any],
    status: str,
) -> str:
    role = str(capability.get("role") or "").strip()
    source_text = str(
        capability.get("source_text")
        or capability.get("requirement_text")
        or capability.get("capability_id")
        or role
    ).strip()
    if role == NETWORK_ADAPTER_ROLE:
        label = _network_capability_label_from_capability(capability)
        return (
            f"Не найден сетевой адаптер {label} с достаточным остатком или "
            f"подтвержденными портами; выбранная платформа не имеет onboard {label}."
        )
    if status == "missing_category":
        return "No distributor category was found for this hard requirement."
    if status == "missing_candidates":
        return (
            "Distributor category was selected, but no product passed local facts "
            "and eligibility checks."
        )
    return source_text


def _network_capability_label_from_capability(capability: Mapping[str, Any]) -> str:
    parsed = capability.get("parsed_requirements")
    parsed = parsed if isinstance(parsed, Mapping) else {}
    speed = str(parsed.get("speed") or "").strip()
    media = str(parsed.get("media") or "").strip()
    parts = [part for part in (speed, media) if part and part != UNKNOWN_FACT]
    return " ".join(parts) if parts else "с запрошенными speed/media/ports"


def _merge_missing_capability_rows(
    *sources: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for rows in sources:
        for row in rows:
            capability_id = str(row.get("capability_id") or "").strip()
            role = str(row.get("role") or "").strip()
            status = str(row.get("status") or "").strip()
            key = (capability_id, role, status)
            if key in seen:
                continue
            seen.add(key)
            result.append(dict(row))
    return result


def _no_recommendation_reason_with_missing_capabilities(
    reason: Mapping[str, Any],
    missing_required_capabilities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    merged_missing = _merge_missing_capability_rows(
        _mapping_list(reason.get("missing_required_capabilities")),
        missing_required_capabilities,
    )
    missing_roles = _unique(
        [
            *_string_list(reason.get("missing_roles")),
            *[
                str(row.get("role") or "").strip()
                for row in merged_missing
                if str(row.get("role") or "").strip()
            ],
        ]
    )
    return {
        **dict(reason),
        "summary": reason.get("summary") or "Безопасную складскую рекомендацию дать нельзя.",
        "missing_roles": missing_roles,
        "missing_required_capabilities": merged_missing,
    }


def _commercial_summary_with_missing_capabilities(
    summary: Mapping[str, Any],
    missing_required_capabilities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    merged_missing = _merge_missing_capability_rows(
        _mapping_list(summary.get("missing_required_capabilities")),
        missing_required_capabilities,
    )
    lines = _string_list(summary.get("lines"))
    if not lines:
        copy_text = str(summary.get("copy_paste_text") or "").strip()
        lines = [line for line in copy_text.splitlines() if line] if copy_text else []
    if not lines:
        lines = ["Безопасную складскую рекомендацию дать нельзя."]
    for line in _missing_capability_commercial_lines(merged_missing):
        if line not in lines:
            lines.append(line)
    return {
        **dict(summary),
        "mode": summary.get("mode") or "single_best_cost_valid",
        "status": "no_recommendation",
        "title": summary.get("title") or "Безопасную складскую рекомендацию дать нельзя.",
        "missing_required_capabilities": merged_missing,
        "lines": lines,
        "copy_paste_text": "\n".join(lines),
    }


def _missing_capability_commercial_lines(
    missing_required_capabilities: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for capability in missing_required_capabilities:
        source_text = str(
            capability.get("source_text")
            or capability.get("requirement_text")
            or capability.get("capability_id")
            or capability.get("role")
            or ""
        ).strip()
        reason = str(
            capability.get("user_message")
            or capability.get("reason")
            or capability.get("status")
            or ""
        ).strip()
        if source_text:
            lines.append(f"Не закрыто требование: {source_text}.")
        if reason:
            lines.append(f"Причина: {reason}")
    return lines


def _capability_id_by_role(role_plan: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for capability in _mapping_list(role_plan.get("required_capabilities")):
        role = str(capability.get("role") or "").strip()
        capability_id = str(capability.get("capability_id") or "").strip()
        if role and capability_id:
            result.setdefault(role, capability_id)
    return result


def _with_capability_id(
    candidates: list[_ComponentCandidate],
    capability_id: str | None,
) -> list[_ComponentCandidate]:
    if not capability_id:
        return candidates
    return [replace(candidate, capability_id=capability_id) for candidate in candidates]


def _bucket_counts(candidates: list[_ComponentCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.selection_bucket] = counts.get(candidate.selection_bucket, 0) + 1
    return dict(sorted(counts.items()))


def _fit_tier_counts(candidates: list[_ComponentCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.fit_tier] = counts.get(candidate.fit_tier, 0) + 1
    return dict(sorted(counts.items()))


def _role_filter_diagnostics(
    *,
    products_by_role: dict[str, list[DistributorProduct]],
    eligible_by_role: Mapping[str, int],
    requirements: _NormalizedServerRequirements,
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: Mapping[tuple[str, str], list[DistributorStockPrice]],
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for role in MATRIX_ROLE_ORDER:
        prompt_role = MATRIX_PROMPT_ROLE_BY_INTERNAL_ROLE[role]
        products = products_by_role.get(role, [])
        reason_counts: dict[str, int] = {}
        sample_filtered_products: list[dict[str, Any]] = []
        for product in products:
            facts = facts_by_key.get(_product_identity(product))
            reason = _role_filter_reason(
                role=role,
                product=product,
                requirements=requirements,
                facts=facts,
                stock_rows=stock_rows_by_key.get(_product_identity(product), []),
            )
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                if len(sample_filtered_products) < 5:
                    sample_filtered_products.append(
                        _sample_filtered_product(
                            product,
                            reason,
                            role=role,
                            facts=facts,
                        )
                    )
        diagnostics[prompt_role] = {
            "raw_products_count": len(products),
            "after_category_count": len(products),
            "after_fact_extraction_count": _fact_extraction_count(
                role,
                products,
                facts_by_key,
            ),
            "after_eligibility_count": int(eligible_by_role.get(role, 0) or 0),
            "filtered_reasons_top": [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    reason_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:5]
            ],
            "sample_filtered_products": sample_filtered_products,
        }
    return diagnostics


def _objective_candidate_policy_decision(
    *,
    product: DistributorProduct,
    stock_rows: Sequence[DistributorStockPrice],
    product_group: str,
    role: str,
    objective_role_reason: str | None = None,
    technical_reject_reason: str | None = None,
    technical_warnings: Sequence[str] = (),
    uncertainty_reasons: Sequence[str] = (),
    default_fit_tier: str = FIT_TIER_POSSIBLE,
) -> BroadPreLlmDecision:
    return broad_pre_llm_for_ai_reasoning(
        product_group=product_group,
        role=role,
        distributor_code=product.distributor_code,
        category_id=product.category_id,
        has_stock=_has_positive_stock(stock_rows),
        has_price=_has_price(stock_rows),
        broken_row=_is_broken_product_row(product),
        objective_role_reject_reason=objective_role_reason,
        technical_reject_reason=technical_reject_reason,
        technical_warnings=technical_warnings,
        uncertainty_reasons=uncertainty_reasons,
        default_fit_tier=default_fit_tier,
    )


def _objective_candidate_reject_reason(
    *,
    product: DistributorProduct,
    stock_rows: Sequence[DistributorStockPrice],
    product_group: str,
    role: str,
    objective_role_reason: str | None = None,
) -> str | None:
    decision = _objective_candidate_policy_decision(
        product=product,
        stock_rows=stock_rows,
        product_group=product_group,
        role=role,
        objective_role_reason=objective_role_reason,
    )
    return decision.objective_reject_reason if not decision.include else None


def _has_positive_stock(rows: Sequence[DistributorStockPrice]) -> bool:
    available_quantity = _available_quantity(list(rows))
    return available_quantity is not None and available_quantity > 0


def _has_price(rows: Sequence[DistributorStockPrice]) -> bool:
    price_value, _ = _select_price(list(rows))
    return price_value is not None


def _is_broken_product_row(product: DistributorProduct) -> bool:
    if not str(product.distributor_code or "").strip():
        return True
    if not str(product.item_id or "").strip():
        return True
    if not str(product.category_id or "").strip():
        return True
    return not _product_search_text(product).strip()


def _sample_filtered_product(
    product: DistributorProduct,
    reason: str,
    *,
    role: str,
    facts: _ProductFacts | None,
) -> dict[str, Any]:
    row = {
        "producer": _safe_diagnostic_value(product.producer, limit=80),
        "part_number": _safe_diagnostic_value(product.part_number, limit=80),
        "short_name": _safe_diagnostic_value(
            product.item_name or product.item_name_rus or product.product_name,
            limit=140,
        ),
        "reason": reason,
    }
    if role == NETWORK_ADAPTER_ROLE and facts is not None:
        row.update(
            {
                "network_ports_count": facts.network_ports_count,
                "network_speed": facts.network_speed,
                "network_media": facts.network_media,
            }
        )
    return row


def _safe_diagnostic_value(value: Any, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _fact_extraction_count(
    role: str,
    products: Sequence[DistributorProduct],
    facts_by_key: Mapping[tuple[str, str], _ProductFacts],
) -> int:
    count = 0
    for product in products:
        facts = facts_by_key.get(_product_identity(product))
        if facts is None:
            continue
        if role == NETWORK_ADAPTER_ROLE:
            if (
                facts.network_ports_count is not None
                or facts.network_speed != UNKNOWN_FACT
                or facts.network_media != UNKNOWN_FACT
            ):
                count += 1
            continue
        count += 1
    return count


def _role_filter_reason(
    *,
    role: str,
    product: DistributorProduct,
    requirements: _NormalizedServerRequirements,
    facts: _ProductFacts | None,
    stock_rows: Sequence[DistributorStockPrice],
) -> str | None:
    objective_reject = _objective_candidate_reject_reason(
        product=product,
        stock_rows=stock_rows,
        product_group=requirements.product_group,
        role=role,
    )
    if objective_reject:
        return objective_reject
    if facts is None:
        return "facts_not_extracted"
    eligibility_rejection = _role_product_eligibility_rejection(
        role=role,
        product=product,
        facts=facts,
        requirements=requirements,
    )
    if role == NETWORK_ADAPTER_ROLE:
        objective_network_reject = _network_adapter_objective_wrong_role_reason(product)
        if objective_network_reject:
            return objective_network_reject
    if objective_role_reject_reason(eligibility_rejection):
        return eligibility_rejection
    if role == NETWORK_ADAPTER_ROLE and requirements.network_required:
        return None
    if (
        role == CPU_ROLE
        and requirements.cpu_vendor_preference != UNKNOWN_FACT
        and facts.cpu_brand not in {UNKNOWN_FACT, requirements.cpu_vendor_preference}
    ):
        return "cpu_vendor_mismatch"
    if (
        role == CPU_ROLE
        and requirements.cpu_min_cores_per_cpu is not None
        and facts.cpu_cores is not None
        and facts.cpu_cores < requirements.cpu_min_cores_per_cpu
    ):
        return "cpu_cores_below_requirement"
    if (
        role == RAM_ROLE
        and requirements.ram_type_preference != UNKNOWN_FACT
        and facts.ram_type not in {UNKNOWN_FACT, requirements.ram_type_preference}
    ):
        return "ram_type_mismatch"
    if role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}:
        fact_interface = (
            facts.drive_interface
            if facts.drive_interface != UNKNOWN_FACT
            else facts.storage_interface
        )
        if (
            requirements.storage_interface_preference in {"NVMe", "SAS", "SATA"}
            and fact_interface
            not in {
                UNKNOWN_FACT,
                requirements.storage_interface_preference,
            }
        ):
            return "storage_interface_mismatch"
        fact_capacity = facts.drive_capacity_tb or facts.storage_capacity_tb
        required_capacity = (
            requirements.drive_capacity_tb
            if requirements.product_group == STORAGE_PRODUCT_GROUP
            else requirements.storage_min_capacity_tb
        )
        if (
            required_capacity is not None
            and fact_capacity is not None
            and fact_capacity < required_capacity
        ):
            return "storage_capacity_below_requirement"
    return None


def _role_product_eligibility_rejection(
    *,
    role: str,
    product: DistributorProduct,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    text = _product_search_text(product)
    if requirements.product_group == NETWORK_PRODUCT_GROUP:
        return _network_role_product_rejection(role, text, facts, requirements)
    if requirements.product_group == STORAGE_PRODUCT_GROUP:
        return _storage_role_product_rejection(role, text, facts, requirements)
    if role in {SUPPORT_ROLE, LICENSE_ROLE}:
        return _support_license_rejection(role, text)
    return None


def _network_role_product_rejection(
    role: str,
    text: str,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    if role == SWITCH_ROLE:
        if not _looks_like_switch_product(text):
            return "network_switch_not_base_switch"
        if _looks_like_network_accessory(text):
            return "network_switch_accessory_or_cable"
        if text_rejection := _network_device_text_hard_contradiction(role, text, requirements):
            return text_rejection
        return _network_device_hard_contradiction(role, facts, requirements)
    if role == ROUTER_ROLE:
        if not _looks_like_router_product(text):
            return "network_router_not_base_router"
        if _looks_like_network_accessory(text):
            return "network_router_accessory_or_cable"
        if text_rejection := _network_device_text_hard_contradiction(role, text, requirements):
            return text_rejection
        return _network_device_hard_contradiction(role, facts, requirements)
    if role == ACCESS_POINT_ROLE:
        if not _looks_like_access_point_product(text):
            return "network_ap_not_access_point"
        if _looks_like_controller_only(text):
            return "network_ap_controller_only"
        return None
    if role == TRANSCEIVER_ROLE:
        if not _looks_like_transceiver_product(text) or _looks_like_dac_cable_product(text):
            return "network_transceiver_role_mismatch"
        return None
    if role in {DAC_CABLE_ROLE, CABLE_ROLE}:
        if not _looks_like_cable_product(text):
            return "network_cable_role_mismatch"
        if _looks_like_base_network_device(text) or _looks_like_storage_system_product(text):
            return "network_cable_base_device_mismatch"
        return None
    if role in {SUPPORT_ROLE, LICENSE_ROLE}:
        return _support_license_rejection(role, text)
    return None


def _network_device_text_hard_contradiction(
    role: str,
    text: str,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    parsed = _role_parsed_requirements(requirements, role)
    l3_or_stacking_required = bool(
        parsed.get("l3_required") or parsed.get("stacking_required")
    )
    if l3_or_stacking_required and _managed_status_from_text(text) == "unmanaged":
        return "network_unmanaged_l3_stacking_mismatch"
    required_ports = _as_int(parsed.get("port_count"))
    if l3_or_stacking_required and required_ports and required_ports >= 24:
        port_count, _ = _network_access_port_segment(text)
        if port_count is not None and port_count <= 8:
            return "network_tiny_desktop_l3_stacking_mismatch"
    if parsed.get("poe_required") and re.search(r"\bnon[- ]?poe\b|without\s+poe", text, re.I):
        return "network_poe_contradiction"
    return None


def _network_device_hard_contradiction(
    role: str,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    parsed = _role_parsed_requirements(requirements, role)
    required_ports = _as_int(parsed.get("port_count"))
    observed_ports = facts.port_count or facts.network_ports_count
    if (
        required_ports is not None
        and observed_ports is not None
        and observed_ports < required_ports
    ):
        return "network_port_count_below_requirement"
    required_speed = _network_speed_requirement_gbps(parsed.get("port_speed"))
    if (
        required_speed is not None
        and facts.port_speed_gbps is not None
        and facts.port_speed_gbps < required_speed
    ):
        return "network_port_speed_below_requirement"
    required_uplinks = _as_int(parsed.get("uplink_count"))
    if (
        required_uplinks is not None
        and facts.uplink_count is not None
        and facts.uplink_count < required_uplinks
    ):
        return "network_uplink_count_below_requirement"
    required_uplink_speed = _network_speed_requirement_gbps(parsed.get("uplink_speed"))
    if (
        required_uplink_speed is not None
        and facts.uplink_speed_gbps is not None
        and facts.uplink_speed_gbps < required_uplink_speed
    ):
        return "network_uplink_speed_below_requirement"
    if parsed.get("poe_required") and facts.poe_supported is False:
        return "network_poe_contradiction"
    if parsed.get("l3_required") and facts.l3_supported is False:
        return "network_l3_contradiction"
    if parsed.get("stacking_required") and facts.stacking_supported is False:
        return "network_stacking_contradiction"
    if (
        (parsed.get("l3_required") or parsed.get("stacking_required"))
        and observed_ports is not None
        and observed_ports <= 8
        and (_as_int(parsed.get("port_count")) or 0) >= 24
    ):
        return "network_tiny_desktop_l3_stacking_mismatch"
    return None


def _storage_role_product_rejection(
    role: str,
    text: str,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    if role == STORAGE_SYSTEM_ROLE:
        if not _looks_like_storage_system_product(text):
            return "storage_system_role_mismatch"
        if _looks_like_storage_accessory(text):
            return "storage_system_accessory_or_component"
        return _storage_system_hard_contradiction(facts, requirements)
    if role == DISK_SHELF_ROLE:
        if not _looks_like_disk_shelf_product(text):
            return "storage_shelf_role_mismatch"
        if _looks_like_storage_accessory(text):
            return "storage_shelf_accessory_or_component"
        return None
    if role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}:
        if not _looks_like_drive_product(text):
            return "storage_drive_role_mismatch"
        if _looks_like_storage_system_product(text) or _looks_like_drive_accessory(text):
            return "storage_drive_not_standalone_drive"
        if role == SSD_ROLE and facts.drive_type not in {UNKNOWN_FACT, "SSD"}:
            return "storage_drive_type_mismatch"
        if role == HDD_ROLE and facts.drive_type not in {UNKNOWN_FACT, "HDD"}:
            return "storage_drive_type_mismatch"
        return None
    if role in {HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE}:
        if _looks_like_wifi_or_ap_controller(text):
            return "storage_host_port_wifi_controller"
        if _looks_like_drive_product(text) and not re.search(
            r"\b(?:hba|host\s+port|interface\s+module|protocol\s+module|storage\s+nic)\b",
            text,
            re.I,
        ):
            return "storage_host_port_drive_mismatch"
        if not _looks_like_storage_connectivity_product(text):
            return "storage_host_port_role_mismatch"
        return None
    if role in {SUPPORT_ROLE, LICENSE_ROLE}:
        return _support_license_rejection(role, text)
    if role in {TRANSCEIVER_ROLE, CABLE_ROLE}:
        if role == TRANSCEIVER_ROLE and not _looks_like_transceiver_product(text):
            return "storage_transceiver_role_mismatch"
        if role == CABLE_ROLE and not _looks_like_cable_product(text):
            return "storage_cable_role_mismatch"
    return None


def _storage_system_hard_contradiction(
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    if (
        requirements.usable_capacity_tb is not None
        and facts.usable_capacity_tb is not None
        and facts.usable_capacity_tb < requirements.usable_capacity_tb
    ):
        return "storage_usable_capacity_below_requirement"
    if (
        requirements.raw_capacity_tb is not None
        and facts.raw_capacity_tb is not None
        and facts.raw_capacity_tb < requirements.raw_capacity_tb
    ):
        return "storage_raw_capacity_below_requirement"
    if (
        requirements.controller_count is not None
        and facts.controller_count is not None
        and facts.controller_count < requirements.controller_count
    ):
        return "storage_controller_count_below_requirement"
    if (
        requirements.drive_type in {"SSD", "HDD"}
        and facts.drive_type not in {UNKNOWN_FACT, requirements.drive_type}
    ):
        return "storage_drive_type_mismatch"
    if (
        requirements.host_protocol != UNKNOWN_FACT
        and facts.host_protocol not in {UNKNOWN_FACT, requirements.host_protocol}
    ):
        return "storage_host_protocol_mismatch"
    if (
        requirements.host_port_speed_gbps is not None
        and facts.host_port_speed_gbps is not None
        and facts.host_port_speed_gbps < requirements.host_port_speed_gbps
    ):
        return "storage_host_port_speed_below_requirement"
    return None


def _support_license_rejection(role: str, text: str) -> str | None:
    if _looks_like_unrelated_support_hardware(text):
        return "support_license_hardware_mismatch"
    if role == SUPPORT_ROLE:
        if _looks_like_support_product(text):
            return None
        return "support_role_mismatch"
    if role == LICENSE_ROLE:
        if _looks_like_license_product(text):
            return None
        return "license_role_mismatch"
    return None


def _looks_like_switch_product(text: str) -> bool:
    return bool(re.search(r"\b(?:ethernet\s+)?switch(?:es)?\b|коммутатор|свитч", text, re.I))


def _looks_like_router_product(text: str) -> bool:
    return bool(re.search(r"\b(?:router|gateway)\b|маршрутизатор|роутер", text, re.I))


def _looks_like_access_point_product(text: str) -> bool:
    return bool(re.search(r"\b(?:access\s*point|wi-?fi\s+ap)\b|точк[аи]\s+доступ", text, re.I))


def _looks_like_base_network_device(text: str) -> bool:
    return (
        _looks_like_switch_product(text)
        or _looks_like_router_product(text)
        or _looks_like_access_point_product(text)
    )


def _looks_like_network_accessory(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:cable|dac|twinax|transceiver|module|injector|adapter|"
            r"kvm|mount|bracket|rail|accessory)\b|кабель|креплен|инжектор|адаптер",
            text,
            re.I,
        )
    )


def _looks_like_controller_only(text: str) -> bool:
    return bool(re.search(r"\bcontroller\b|контроллер", text, re.I))


def _looks_like_wifi_or_ap_controller(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:wi-?fi|wireless|access\s*point|ap)\b.*controller|контроллер.*wi",
            text,
            re.I,
        )
    )


def _looks_like_transceiver_product(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:transceiver|optic(?:al)?\s+module|sfp\+?|sfp28|qsfp\+?|"
            r"qsfp28)\b|трансивер|оптик",
            text,
            re.I,
        )
    )


def _looks_like_dac_cable_product(text: str) -> bool:
    return bool(re.search(r"\b(?:dac|direct\s+attach|twinax)\b", text, re.I))


def _looks_like_cable_product(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:cable|dac|direct\s+attach|twinax|aoc|patch\s*cord)\b|кабель",
            text,
            re.I,
        )
    )


def _looks_like_storage_system_product(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:storage\s+(?:system|array|enclosure)|disk\s+array|nas|san|jbod)\b|"
            r"схд|сетевое\s+хранилище|система\s+хранения|дисков(?:ый|ая)\s+массив",
            text,
            re.I,
        )
    )


def _looks_like_storage_accessory(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:cable|dac|twinax|ram|memory|adapter\s+card|expansion\s+card|"
            r"rail|psu|power\s+supply|license|support|warranty)\b|"
            r"кабель|память|адаптер|блок\s+питания|лиценз|поддержк",
            text,
            re.I,
        )
    )


def _looks_like_drive_accessory(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:cable|dac|twinax|ram|memory|adapter\s+card|expansion\s+card|"
            r"rail|psu|power\s+supply|license|support|warranty)\b|"
            r"кабель|память|адаптер|блок\s+питания|лиценз|поддержк",
            text,
            re.I,
        )
    )


def _looks_like_disk_shelf_product(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:disk|drive|expansion)\s+(?:shelf|enclosure)|jbod\b|"
            r"полк[аи]|enclosure",
            text,
            re.I,
        )
    )


def _looks_like_drive_product(text: str) -> bool:
    return bool(re.search(r"\b(?:ssd|hdd|drive|disk|nvme|sas|sata)\b|накопител|диск", text, re.I))


def _looks_like_storage_connectivity_product(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:fc|fibre\s+channel|iscsi|nvme-?of|sas|hba|host\s+port|"
            r"interface\s+module|protocol\s+module|storage\s+nic)\b",
            text,
            re.I,
        )
    )


def _looks_like_support_product(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:support|service|warranty|maintenance|care\s*pack)\b|"
            r"поддержк|гарант|сервис",
            text,
            re.I,
        )
    )


def _looks_like_license_product(text: str) -> bool:
    return bool(re.search(r"\b(?:license|licence|subscription)\b|лиценз|подписк", text, re.I))


def _looks_like_unrelated_support_hardware(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:cable|dac|twinax|psu|power\s+supply|transceiver|adapter|"
            r"rail|mount|bracket)\b|кабель|блок\s+питания|адаптер|креплен",
            text,
            re.I,
        )
    )


def _network_filter_reason(
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
    stock_rows: Sequence[DistributorStockPrice],
) -> str | None:
    requirement = requirements.network_requirement
    facts_json = facts.to_report_json()
    if network_adapter_facts_satisfy_requirement(facts_json, requirement):
        required_quantity = required_network_adapter_quantity(
            facts_json,
            requirement,
            server_quantity=requirements.server_qty,
        )
        if required_quantity is None:
            return "network_ports_unknown"
        available_quantity = _available_quantity(list(stock_rows))
        if available_quantity is None:
            return "network_stock_unknown"
        if available_quantity < required_quantity:
            return "network_stock_below_requirement"
        return None
    if facts.network_ports_count is None:
        return "network_ports_unknown"
    if facts.network_ports_count <= 0:
        return "network_ports_below_requirement"
    required_speed = _network_speed_requirement_gbps(requirement.get("speed"))
    if required_speed is not None:
        if facts.network_speed_gbps is None:
            return "network_speed_unknown"
        if facts.network_speed_gbps < required_speed:
            return "network_speed_below_requirement"
    required_media = str(requirement.get("media") or UNKNOWN_FACT).strip()
    if required_media != UNKNOWN_FACT and facts.network_media != required_media:
        return "network_media_mismatch"
    return "network_requirement_mismatch"


def _network_adapter_ai_reasoning_decision(
    *,
    product: DistributorProduct,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
    stock_rows: Sequence[DistributorStockPrice],
) -> tuple[BroadPreLlmDecision, int | None]:
    objective_role_reason = _network_adapter_objective_wrong_role_reason(product)
    if objective_role_reason:
        decision = _objective_candidate_policy_decision(
            product=product,
            stock_rows=stock_rows,
            product_group=requirements.product_group,
            role=NETWORK_ADAPTER_ROLE,
            objective_role_reason=objective_role_reason,
        )
        return decision, None

    if not requirements.network_required:
        decision = _objective_candidate_policy_decision(
            product=product,
            stock_rows=stock_rows,
            product_group=requirements.product_group,
            role=NETWORK_ADAPTER_ROLE,
            default_fit_tier=(
                FIT_TIER_POSSIBLE
                if _network_adapter_has_role_evidence(facts, product)
                else FIT_TIER_FALLBACK_UNKNOWN
            ),
        )
        return decision, None

    facts_json = facts.to_report_json()
    required_quantity = required_network_adapter_quantity(
        facts_json,
        requirements.network_requirement,
        server_quantity=requirements.server_qty,
    )
    warnings: list[str] = []
    uncertainties: list[str] = []
    technical_reject_reason: str | None = None
    default_fit_tier = FIT_TIER_STRONG

    if not network_adapter_facts_satisfy_requirement(
        facts_json,
        requirements.network_requirement,
    ):
        technical_reject_reason = _network_adapter_technical_gap_reason(
            facts,
            requirements,
        )
        if technical_reject_reason and "unknown" in technical_reject_reason:
            uncertainties.append(technical_reject_reason)
            technical_reject_reason = None
        elif technical_reject_reason:
            warnings.append(technical_reject_reason)
        default_fit_tier = FIT_TIER_POSSIBLE

    media_warning = _network_adapter_media_warning(facts, requirements)
    if media_warning:
        warnings.append(media_warning)
        if media_warning.endswith("_unknown"):
            uncertainties.append(media_warning)
        if default_fit_tier == FIT_TIER_STRONG:
            default_fit_tier = FIT_TIER_POSSIBLE

    interface_warning = _network_adapter_interface_warning(facts, requirements)
    if interface_warning:
        warnings.append(interface_warning)
        default_fit_tier = FIT_TIER_POSSIBLE

    if required_quantity is None:
        uncertainties.append("network_adapter_ports_count_unknown")
        default_fit_tier = FIT_TIER_FALLBACK_UNKNOWN
    else:
        available_quantity = _available_quantity(list(stock_rows))
        if available_quantity is not None and available_quantity < required_quantity:
            warnings.append("network_adapter_stock_below_calculated_quantity")
            default_fit_tier = FIT_TIER_POSSIBLE

    if (
        technical_reject_reason == "network_media_family_compatibility_check"
        and _network_adapter_is_media_family_compatible(facts, requirements)
    ):
        technical_reject_reason = None
        default_fit_tier = FIT_TIER_POSSIBLE

    if not _network_adapter_has_role_evidence(facts, product):
        uncertainties.append("network_adapter_role_evidence_incomplete")
        default_fit_tier = FIT_TIER_FALLBACK_UNKNOWN

    decision = _objective_candidate_policy_decision(
        product=product,
        stock_rows=stock_rows,
        product_group=requirements.product_group,
        role=NETWORK_ADAPTER_ROLE,
        technical_reject_reason=technical_reject_reason,
        technical_warnings=warnings,
        uncertainty_reasons=uncertainties,
        default_fit_tier=default_fit_tier,
    )
    return decision, required_quantity


def _network_adapter_objective_wrong_role_reason(product: DistributorProduct) -> str | None:
    text = _product_search_text(product)
    lowered = text.casefold()
    ethernet_markers = (
        r"\bethernet\b",
        r"\bnetwork\b",
        r"\bnic\b",
        r"\b\d+\s*gbe\b",
        r"\bbase\s*-?\s*t\b",
        r"\brj\s*-?\s*45\b",
    )
    has_ethernet_marker = any(re.search(pattern, lowered, re.I) for pattern in ethernet_markers)
    if (
        re.search(r"\b(?:fibre|fiber)\s+channel\b|\bfc\s+hba\b|\bhba\b.*\bfc\b", lowered)
        and not has_ethernet_marker
    ):
        return "wrong_role_objective"
    if (
        re.search(r"\b(?:sas|sata|raid)\s+hba\b|\bstorage\s+controller\b", lowered)
        and not has_ethernet_marker
    ):
        return "wrong_role_objective"
    return None


def _network_adapter_technical_gap_reason(
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    required_ports = _as_int(requirements.network_requirement.get("min_ports_per_server")) or 1
    if facts.network_ports_count is None:
        return "network_ports_unknown"
    if facts.network_ports_count <= 0:
        return "network_ports_below_requirement"
    required_speed = _network_speed_requirement_gbps(
        requirements.network_requirement.get("speed")
    )
    if required_speed is not None:
        if facts.network_speed_gbps is None:
            return "network_speed_unknown"
        if facts.network_speed_gbps < required_speed:
            return "network_speed_below_requirement"
    if required_ports > 0 and facts.network_ports_count < required_ports:
        return "network_ports_below_request_single_adapter_quantity_may_compensate"
    required_media = str(requirements.network_requirement.get("media") or UNKNOWN_FACT).strip()
    if required_media != UNKNOWN_FACT and facts.network_media == UNKNOWN_FACT:
        return "network_media_unknown"
    if required_media != UNKNOWN_FACT and facts.network_media != required_media:
        if _network_media_same_pluggable_family(facts.network_media, required_media):
            return "network_media_family_compatibility_check"
        return "network_media_mismatch"
    return "network_requirement_uncertain"


def _network_adapter_media_warning(
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    required_media = str(requirements.network_requirement.get("media") or UNKNOWN_FACT).strip()
    if required_media == UNKNOWN_FACT:
        return None
    actual_media = str(facts.network_media or UNKNOWN_FACT).strip()
    if actual_media == UNKNOWN_FACT:
        return "network_media_unknown"
    if actual_media == required_media:
        return None
    if _network_media_same_pluggable_family(actual_media, required_media):
        return "network_media_family_compatibility_check"
    return "network_media_mismatch"


def _network_adapter_interface_warning(
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str | None:
    required_interface = str(
        requirements.network_requirement.get("interface") or UNKNOWN_FACT
    ).strip()
    if required_interface == UNKNOWN_FACT:
        return None
    actual_interface = str(facts.network_interface or UNKNOWN_FACT).strip()
    if actual_interface == UNKNOWN_FACT:
        return "network_interface_unknown"
    if actual_interface == required_interface:
        return None
    return "network_interface_engineer_check"


def _network_adapter_is_media_family_compatible(
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> bool:
    required_media = str(requirements.network_requirement.get("media") or UNKNOWN_FACT).strip()
    actual_media = str(facts.network_media or UNKNOWN_FACT).strip()
    return _network_media_same_pluggable_family(actual_media, required_media)


def _network_media_same_pluggable_family(actual_media: str, required_media: str) -> bool:
    actual = actual_media.strip().upper().replace(" ", "")
    required = required_media.strip().upper().replace(" ", "")
    if UNKNOWN_FACT.upper() in {actual, required}:
        return False
    if actual == required:
        return True
    if actual.startswith("SFP") and required.startswith("SFP"):
        return True
    return actual.startswith("QSFP") and required.startswith("QSFP")


def _network_adapter_has_role_evidence(
    facts: _ProductFacts,
    product: DistributorProduct,
) -> bool:
    text = _product_search_text(product)
    return bool(
        facts.network_ports_count is not None
        or facts.network_speed_gbps is not None
        or facts.network_media != UNKNOWN_FACT
        or re.search(r"\b(?:ethernet|network|nic|gbe|sfp|qsfp|rj\s*-?\s*45)\b", text, re.I)
    )


def _network_speed_requirement_gbps(value: Any) -> float | None:
    text = str(value or "")
    if re.search(r"\b100\s*m(?:b(?:it)?(?:e|/s|ps)?|bit)?\b|\bfast\s+ethernet\b", text, re.I):
        return 0.1
    if re.search(r"\b100\s*base\s*-?\s*t[x]?\b", text, re.I):
        return 0.1
    if re.search(r"\b1000\s*base\s*-?\s*t[x]?\b|\bgigabit\b|гигабит", text, re.I):
        return 1
    match = re.search(
        r"\b(1|2\.5|5|10|16|25|32|40|56|64|100|200|400)(?=\D|$)",
        text,
        re.I,
    )
    return float(match.group(1)) if match else None


def _score_platform_candidate(
    *,
    product: DistributorProduct,
    requirements: _NormalizedServerRequirements,
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> _ComponentCandidate | None:
    facts = _facts_for_product(product, facts_by_key=facts_by_key, role=SERVER_PLATFORM_ROLE)
    stock_rows = stock_rows_by_key.get(_product_identity(product), [])
    objective_decision = _objective_candidate_policy_decision(
        product=product,
        stock_rows=stock_rows,
        product_group=requirements.product_group,
        role=SERVER_PLATFORM_ROLE,
    )
    if not objective_decision.include:
        return None
    search_text = _product_search_text(product)
    score = 35
    warnings: list[str] = []
    reasons = ["Платформа находится в серверной категории OCS."]

    if requirements.form_factor:
        known_u_hints = {hint.upper() for hint in facts.form_factor_hints if hint.endswith("U")}
        expected = requirements.form_factor.upper()
        if expected in known_u_hints or _matches_form_factor(search_text, requirements.form_factor):
            score += 20
            reasons.append(f"Форм-фактор {requirements.form_factor} подтвержден по данным товара.")
        elif known_u_hints:
            return None
        else:
            warnings.append(
                f"Форм-фактор {requirements.form_factor} не подтвержден по данным платформы."
            )

    if requirements.cpu_per_server is not None:
        if _matches_cpu_socket_requirement(search_text, requirements.cpu_per_server):
            score += 15
            reasons.append(
                f"{requirements.cpu_per_server} CPU/socket подтверждены по данным платформы."
            )
        else:
            warnings.append(
                f"{requirements.cpu_per_server} процессорных сокета не подтверждены "
                "по данным платформы."
            )

    if requirements.psu_count_per_server is not None:
        if _matches_psu_requirement(search_text, requirements.psu_count_per_server):
            score += 10
            reasons.append(
                f"{requirements.psu_count_per_server} БП подтверждены по данным платформы."
            )
        else:
            warnings.append(
                f"{requirements.psu_count_per_server} БП не подтверждены по данным "
                "платформы; требуется проверить комплектацию."
            )

    if facts.normalized_vendor != UNKNOWN_FACT:
        score += 5
        reasons.append(f"Vendor платформы распознан: {facts.normalized_vendor}.")

    if requirements.ram_type_preference != UNKNOWN_FACT:
        if facts.ram_type == requirements.ram_type_preference:
            score += 12
            reasons.append(
                f"Тип памяти платформы соответствует запросу: {facts.ram_type}."
            )
        elif facts.ram_type != UNKNOWN_FACT:
            return None
        else:
            warnings.append(
                "Тип памяти платформы не подтвержден по наименованию; "
                "требуется инженерная проверка."
            )

    if requirements.storage_interface_preference == "NVMe":
        if facts.nvme_support is True:
            score += 8
            reasons.append("NVMe/backplane поддержка распознана по данным платформы.")
        elif _looks_like_3_5_only_platform(facts):
            warnings.append(
                "Платформа выглядит как 3.5-дюймовая корзина без подтвержденного NVMe; "
                "проверить поддержку 2.5/U.2/U.3 NVMe SSD."
            )

    return _make_component_candidate(
        role=SERVER_PLATFORM_ROLE,
        product=product,
        facts=facts,
        requirements=requirements,
        stock_rows=stock_rows,
        quantity_required=requirements.server_qty,
        score=score,
        eligibility_status="eligible",
        warnings=warnings,
        reasons=reasons,
        match_warnings=objective_decision.match_warnings,
        uncertainty_reasons=objective_decision.uncertainty_reasons,
        evidence_summary=objective_decision.evidence_summary,
    )


def _score_cpu_candidate(
    *,
    product: DistributorProduct,
    requirements: _NormalizedServerRequirements,
    platform_vendors: set[str],
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> _ComponentCandidate | None:
    facts = _facts_for_product(product, facts_by_key=facts_by_key, role=CPU_ROLE)
    stock_rows = stock_rows_by_key.get(_product_identity(product), [])
    objective_decision = _objective_candidate_policy_decision(
        product=product,
        stock_rows=stock_rows,
        product_group=requirements.product_group,
        role=CPU_ROLE,
    )
    if not objective_decision.include:
        return None
    quantity_required = requirements.total_cpu_required or requirements.server_qty
    score = 40
    warnings: list[str] = []
    reasons: list[str] = []
    eligibility_status = "eligible"
    fit_label = FIT_UNKNOWN
    fit_reason = "Минимум ядер CPU в запросе не указан."
    cpu_over_requirement: int | None = None

    if facts.is_vendor_option_kit:
        if facts.option_kit_vendor not in platform_vendors:
            return None
        score = 70
        eligibility_status = "eligible_with_same_vendor_platform"
        warnings.append(
            "Проверить список поддерживаемых CPU для платформы "
            "того же производителя."
        )
        reasons.append(
            "Vendor-specific CPU kit оставлен только для платформ того же производителя."
        )
    elif facts.cpu_brand in {"Intel", "AMD"} or facts.cpu_family in {"Xeon", "EPYC"}:
        score = 65
        warnings.append(
            "Проверить совместимость CPU с платформой по списку поддерживаемых CPU "
            "платформы."
        )
        reasons.append("Bare Intel/AMD CPU подходит как preliminary-вариант.")
    else:
        warnings.append("CPU без надежно извлеченного vendor/семейства требует ручной проверки.")
        reasons.append("CPU находится в серверной категории процессоров OCS.")

    if requirements.cpu_vendor_preference != UNKNOWN_FACT:
        if facts.cpu_brand == requirements.cpu_vendor_preference:
            score += 15
            reasons.append(f"CPU соответствует vendor preference: {facts.cpu_brand}.")
        elif facts.cpu_brand in {"Intel", "AMD"}:
            return None
        else:
            warnings.append("Производитель CPU не распознан; требуется ручная проверка требования.")

    if requirements.cpu_family_preference != UNKNOWN_FACT:
        if facts.cpu_family == requirements.cpu_family_preference:
            score += 10
            reasons.append(f"CPU соответствует family preference: {facts.cpu_family}.")
        elif facts.cpu_family in {"Xeon", "EPYC"}:
            return None
        else:
            warnings.append("Семейство CPU не распознано; требуется ручная проверка preference.")

    if requirements.cpu_min_cores_per_cpu is not None:
        if facts.cpu_cores is None:
            warnings.append("Количество ядер CPU не распознано; требуется ручная проверка.")
            score -= 12
        elif facts.cpu_cores < requirements.cpu_min_cores_per_cpu:
            return None
        else:
            fit_label, fit_reason, cpu_over_requirement, fit_score = _cpu_fit_fields(
                facts.cpu_cores,
                requirements.cpu_min_cores_per_cpu,
            )
            score += fit_score
            reasons.append(fit_reason)
    if requirements.cpu_generation_or_model_hint:
        search_text = _product_search_text(product)
        if requirements.cpu_generation_or_model_hint.casefold() in search_text.casefold():
            score += 5
            reasons.append("CPU model/generation hint найден в карточке товара.")
        else:
            warnings.append("CPU model/generation hint не подтвержден по карточке товара.")

    return _make_component_candidate(
        role=CPU_ROLE,
        product=product,
        facts=facts,
        requirements=requirements,
        stock_rows=stock_rows,
        quantity_required=quantity_required,
        score=score,
        eligibility_status=eligibility_status,
        warnings=warnings,
        reasons=reasons,
        fit_label=fit_label,
        fit_reason=fit_reason,
        match_warnings=objective_decision.match_warnings,
        uncertainty_reasons=objective_decision.uncertainty_reasons,
        evidence_summary=objective_decision.evidence_summary,
        cpu_over_requirement=cpu_over_requirement,
    )


def _score_ram_candidate(
    *,
    product: DistributorProduct,
    requirements: _NormalizedServerRequirements,
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> _ComponentCandidate | None:
    facts = _facts_for_product(product, facts_by_key=facts_by_key, role=RAM_ROLE)
    stock_rows = stock_rows_by_key.get(_product_identity(product), [])
    objective_decision = _objective_candidate_policy_decision(
        product=product,
        stock_rows=stock_rows,
        product_group=requirements.product_group,
        role=RAM_ROLE,
    )
    if not objective_decision.include:
        return None
    warnings: list[str] = []
    reasons = ["RAM находится в серверной категории памяти OCS."]
    score = 35

    if facts.ram_capacity_gb is None:
        quantity_required = requirements.server_qty
        warnings.append("Емкость модуля RAM не распознана; требуется ручная проверка.")
        fit_label = FIT_UNKNOWN
        fit_reason = "Емкость модуля RAM не распознана."
        ram_over_requirement_gb = None
        score -= 8
    else:
        score += 15
        reasons.append(f"Емкость модуля RAM распознана: {facts.ram_capacity_gb} ГБ.")
        if requirements.ram_gb_per_server is None:
            quantity_required = requirements.server_qty
            fit_label = FIT_UNKNOWN
            fit_reason = "Требуемый объем RAM на сервер не указан."
            ram_over_requirement_gb = None
        else:
            modules_per_server = max(
                ceil(requirements.ram_gb_per_server / facts.ram_capacity_gb),
                1,
            )
            quantity_required = modules_per_server * requirements.server_qty
            (
                fit_label,
                fit_reason,
                ram_over_requirement_gb,
                fit_score,
            ) = _ram_fit_fields(
                module_gb=facts.ram_capacity_gb,
                required_gb=requirements.ram_gb_per_server,
                modules_per_server=modules_per_server,
            )
            score += fit_score
            reasons.append(fit_reason)
            if modules_per_server >= 16:
                warnings.append(
                    "RAM требует 16 или более модулей на сервер; проверить количество "
                    "доступных слотов и правила заполнения."
                )

    if requirements.ram_type_preference != UNKNOWN_FACT:
        if facts.ram_type == requirements.ram_type_preference:
            score += 8
            reasons.append(f"RAM соответствует требуемому типу: {facts.ram_type}.")
        elif facts.ram_type != UNKNOWN_FACT:
            return None
        else:
            warnings.append("Тип RAM не распознан; требуется ручная проверка требования.")

    if (
        facts.ram_type == UNKNOWN_FACT
        and requirements.ram_type_preference == UNKNOWN_FACT
    ):
        warnings.append("Тип RAM не указан; требуется проверка совместимости RAM с платформой.")
    else:
        score += 5
        if facts.ram_type != UNKNOWN_FACT:
            reasons.append(f"Тип RAM распознан: {facts.ram_type}.")

    return _make_component_candidate(
        role=RAM_ROLE,
        product=product,
        facts=facts,
        requirements=requirements,
        stock_rows=stock_rows,
        quantity_required=quantity_required,
        score=score,
        eligibility_status="eligible",
        warnings=warnings,
        reasons=reasons,
        fit_label=fit_label,
        fit_reason=fit_reason,
        match_warnings=objective_decision.match_warnings,
        uncertainty_reasons=objective_decision.uncertainty_reasons,
        evidence_summary=objective_decision.evidence_summary,
        ram_over_requirement_gb=ram_over_requirement_gb,
    )


def _score_storage_candidate(
    *,
    role: str,
    product: DistributorProduct,
    requirements: _NormalizedServerRequirements,
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> _ComponentCandidate | None:
    facts = _facts_for_product(product, facts_by_key=facts_by_key, role=role)
    stock_rows = stock_rows_by_key.get(_product_identity(product), [])
    eligibility_rejection = _role_product_eligibility_rejection(
        role=role,
        product=product,
        facts=facts,
        requirements=requirements,
    )
    objective_role_reason = objective_role_reject_reason(eligibility_rejection)
    objective_decision = _objective_candidate_policy_decision(
        product=product,
        stock_rows=stock_rows,
        product_group=requirements.product_group,
        role=role,
        objective_role_reason=objective_role_reason,
        technical_reject_reason=(
            eligibility_rejection if not objective_role_reason else None
        ),
    )
    if not objective_decision.include:
        return None
    warnings: list[str] = []
    reasons = [f"{_role_ru(role)} находится в нужной серверной категории OCS."]
    score = 45
    storage_role = _storage_role_for_requirements(requirements)
    fit_label = FIT_UNKNOWN
    fit_reason = "Минимальный объем накопителя в запросе не указан."
    storage_over_requirement: float | None = None
    facts_interface = (
        facts.drive_interface if facts.drive_interface != UNKNOWN_FACT else facts.storage_interface
    )
    facts_capacity_tb = facts.drive_capacity_tb or facts.storage_capacity_tb
    required_capacity_tb = (
        requirements.drive_capacity_tb
        if requirements.product_group == STORAGE_PRODUCT_GROUP
        else requirements.storage_min_capacity_tb
    )

    if storage_role == role:
        score += 20
        reasons.append(f"Категория соответствует требованию: {_role_ru(role)}.")

    if requirements.storage_interface_preference in {"NVMe", "SAS", "SATA"}:
        if facts_interface == requirements.storage_interface_preference:
            score += 12
            reasons.append(f"Интерфейс накопителя распознан: {facts_interface}.")
        elif facts_interface != UNKNOWN_FACT:
            return None
        else:
            warnings.append("Интерфейс накопителя не распознан; требуется ручная проверка.")
    elif facts_interface != UNKNOWN_FACT:
        score += 5
        reasons.append(f"Интерфейс накопителя распознан: {facts_interface}.")

    if facts.storage_capacity != UNKNOWN_FACT or facts_capacity_tb is not None:
        score += 5
        reasons.append(f"Объем накопителя распознан: {facts.storage_capacity}.")
        if required_capacity_tb is not None and facts_capacity_tb is not None:
            if facts_capacity_tb < required_capacity_tb:
                return None
            (
                fit_label,
                fit_reason,
                storage_over_requirement,
                fit_score,
            ) = _storage_fit_fields(
                capacity_tb=facts_capacity_tb,
                required_tb=required_capacity_tb,
            )
            score += fit_score
            reasons.append(fit_reason)
        elif required_capacity_tb is not None:
            fit_label = FIT_UNKNOWN
            fit_reason = "Объем накопителя не распознан для проверки минимальной емкости."
            storage_over_requirement = None
            score -= 10
            warnings.append("Объем накопителя не распознан; требуется ручная проверка.")
        else:
            fit_label = FIT_UNKNOWN
            fit_reason = "Минимальный объем накопителя в запросе не указан."
            storage_over_requirement = None
    elif (
        role == SSD_ROLE
        and requirements.storage_required
        and not requirements.storage_min_capacity
    ):
        warnings.append(
            "Объем SSD в запросе не указан; накопитель выбран предварительно по наличию."
        )
        fit_label = FIT_UNKNOWN
        fit_reason = "Минимальный объем накопителя в запросе не указан."
        storage_over_requirement = None
    else:
        warnings.append("Объем накопителя не распознан; требуется ручная проверка.")
        fit_label = FIT_UNKNOWN
        fit_reason = "Объем накопителя не распознан."
        storage_over_requirement = None
        score -= 10

    return _make_component_candidate(
        role=role,
        product=product,
        facts=facts,
        requirements=requirements,
        stock_rows=stock_rows,
        quantity_required=_storage_drive_quantity_required(role, requirements),
        score=score,
        eligibility_status="eligible",
        warnings=warnings,
        reasons=reasons,
        fit_label=fit_label,
        fit_reason=fit_reason,
        match_warnings=objective_decision.match_warnings,
        uncertainty_reasons=objective_decision.uncertainty_reasons,
        evidence_summary=objective_decision.evidence_summary,
        storage_over_requirement=storage_over_requirement,
    )


def _score_simple_candidate(
    *,
    role: str,
    product: DistributorProduct,
    requirements: _NormalizedServerRequirements,
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> _ComponentCandidate | None:
    facts = _facts_for_product(product, facts_by_key=facts_by_key, role=role)
    stock_rows = stock_rows_by_key.get(_product_identity(product), [])
    eligibility_rejection = _role_product_eligibility_rejection(
        role=role,
        product=product,
        facts=facts,
        requirements=requirements,
    )
    objective_role_reason = objective_role_reject_reason(eligibility_rejection)
    if role == NETWORK_ADAPTER_ROLE:
        objective_role_reason = (
            objective_role_reason or _network_adapter_objective_wrong_role_reason(product)
        )
    objective_decision = _objective_candidate_policy_decision(
        product=product,
        stock_rows=stock_rows,
        product_group=requirements.product_group,
        role=role,
        objective_role_reason=objective_role_reason,
        technical_reject_reason=(
            eligibility_rejection if not objective_role_reason else None
        ),
    )
    if not objective_decision.include:
        return None
    quantity_required = _simple_role_quantity_required(role, requirements)
    score = 45
    score += _role_fact_fit_score(role, facts, requirements)
    warnings = [_role_review_warning(role, requirements)]
    component_fit_tier: str | None = None
    reasons = [f"{_role_ru(role)} находится в выбранной категории."]
    if role == NETWORK_ADAPTER_ROLE and requirements.network_required:
        decision, network_quantity_required = _network_adapter_ai_reasoning_decision(
            product=product,
            facts=facts,
            requirements=requirements,
            stock_rows=stock_rows,
        )
        if not decision.include:
            return None
        if network_quantity_required is not None:
            quantity_required = network_quantity_required
        if decision.fit_tier == FIT_TIER_STRONG:
            score += 25
            reasons.append("Network adapter satisfies requested port speed/media requirement.")
        else:
            score += 8
            warnings.extend(decision.match_warnings)
            if decision.uncertainty_reasons:
                warnings.extend(decision.uncertainty_reasons)
            reasons.append("Network adapter kept for AI reasoning under broad pre-LLM policy.")
        objective_decision = decision
        component_fit_tier = decision.fit_tier
    return _make_component_candidate(
        role=role,
        product=product,
        facts=facts,
        requirements=requirements,
        stock_rows=stock_rows,
        quantity_required=quantity_required,
        score=score,
        eligibility_status="eligible",
        warnings=warnings,
        reasons=reasons,
        fit_tier=component_fit_tier,
        match_warnings=objective_decision.match_warnings,
        uncertainty_reasons=objective_decision.uncertainty_reasons,
        objective_reject_reason=objective_decision.objective_reject_reason,
        evidence_summary=objective_decision.evidence_summary,
    )


def _role_fact_fit_score(
    role: str,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> int:
    score = 0
    parsed = _role_parsed_requirements(requirements, role)
    if role in {SWITCH_ROLE, ROUTER_ROLE, FIREWALL_ROLE, ACCESS_POINT_ROLE}:
        if facts.port_count is not None:
            score += 8
        if facts.uplink_count is not None:
            score += 6
        if parsed.get("poe_required") and facts.poe_supported is True:
            score += 8
        if parsed.get("l3_required") and facts.l3_supported is True:
            score += 8
        if parsed.get("stacking_required") and facts.stacking_supported is True:
            score += 8
    if role == STORAGE_SYSTEM_ROLE:
        if facts.raw_capacity_tb is not None or facts.usable_capacity_tb is not None:
            score += 8
        if facts.controller_count is not None:
            score += 6
        if facts.host_protocol != UNKNOWN_FACT:
            score += 6
        if facts.host_port_speed_gbps is not None:
            score += 6
    if role in {HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE}:
        if facts.host_protocol != UNKNOWN_FACT:
            score += 10
        if facts.host_port_speed_gbps is not None:
            score += 8
    if role in {SUPPORT_ROLE, LICENSE_ROLE} and facts.warranty_months is not None:
        score += 6
    return score


def _candidate_fit_tier(
    *,
    role: str,
    product: DistributorProduct,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str:
    rejection = _role_product_eligibility_rejection(
        role=role,
        product=product,
        facts=facts,
        requirements=requirements,
    )
    if rejection:
        if _explicit_mismatch_reason(rejection):
            return FIT_TIER_EXPLICIT_MISMATCH
        return FIT_TIER_WRONG_ROLE
    if role in {SWITCH_ROLE, ROUTER_ROLE, FIREWALL_ROLE}:
        return _network_device_fit_tier(role, facts, requirements)
    if role == STORAGE_SYSTEM_ROLE:
        return _storage_system_fit_tier(facts, requirements)
    if role in {HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE}:
        return _storage_connectivity_fit_tier(facts, requirements)
    if role in {SUPPORT_ROLE, LICENSE_ROLE}:
        return FIT_TIER_STRONG if facts.warranty_months is not None else FIT_TIER_POSSIBLE
    if _role_has_any_known_fact(role, facts):
        return FIT_TIER_POSSIBLE
    return FIT_TIER_FALLBACK_UNKNOWN


def _explicit_mismatch_reason(reason: str) -> bool:
    return any(
        marker in reason
        for marker in (
            "below_requirement",
            "contradiction",
            "mismatch",
            "unmanaged_l3_stacking",
            "tiny_desktop",
        )
    )


def _network_device_fit_tier(
    role: str,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str:
    parsed = _role_parsed_requirements(requirements, role)
    if _network_device_hard_contradiction(role, facts, requirements):
        return FIT_TIER_EXPLICIT_MISMATCH
    has_port_count_fact = (
        facts.port_count is not None or facts.network_ports_count is not None
    )
    if parsed.get("port_count") is not None and not has_port_count_fact:
        return FIT_TIER_FALLBACK_UNKNOWN
    required_checks: list[bool | None] = []
    if parsed.get("port_count") is not None:
        required_checks.append(has_port_count_fact)
    if parsed.get("port_speed"):
        required_checks.append(facts.port_speed_gbps is not None)
    if parsed.get("uplink_count") is not None:
        required_checks.append(facts.uplink_count is not None)
    if parsed.get("uplink_speed"):
        required_checks.append(facts.uplink_speed_gbps is not None)
    if parsed.get("uplink_media"):
        required_checks.append(facts.uplink_media != UNKNOWN_FACT)
    if parsed.get("poe_required"):
        required_checks.append(facts.poe_supported is not None)
    if parsed.get("l3_required"):
        required_checks.append(facts.l3_supported is not None)
    if parsed.get("stacking_required"):
        required_checks.append(facts.stacking_supported is not None)
    if required_checks and all(required_checks):
        return FIT_TIER_STRONG
    known_count = sum(
        1
        for value in (
            facts.port_count,
            facts.port_speed_gbps,
            facts.uplink_count,
            facts.uplink_speed_gbps,
            facts.poe_supported,
            facts.l3_supported,
            facts.stacking_supported,
        )
        if value is not None
    )
    if known_count >= 2:
        return FIT_TIER_POSSIBLE
    return FIT_TIER_FALLBACK_UNKNOWN


def _storage_system_fit_tier(
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str:
    if _storage_system_hard_contradiction(facts, requirements):
        return FIT_TIER_EXPLICIT_MISMATCH
    checks: list[bool] = []
    if requirements.usable_capacity_tb is not None:
        checks.append(facts.usable_capacity_tb is not None)
    if requirements.raw_capacity_tb is not None:
        checks.append(facts.raw_capacity_tb is not None)
    if requirements.controller_count is not None:
        checks.append(facts.controller_count is not None)
    if requirements.drive_type in {"SSD", "HDD"}:
        checks.append(facts.drive_type != UNKNOWN_FACT)
    if requirements.host_protocol != UNKNOWN_FACT:
        checks.append(facts.host_protocol != UNKNOWN_FACT)
    if requirements.host_port_speed_gbps is not None:
        checks.append(facts.host_port_speed_gbps is not None)
    if checks and all(checks):
        return FIT_TIER_STRONG
    known_count = sum(
        1
        for value in (
            facts.raw_capacity_tb,
            facts.usable_capacity_tb,
            facts.controller_count,
            None if facts.drive_type == UNKNOWN_FACT else facts.drive_type,
            None if facts.host_protocol == UNKNOWN_FACT else facts.host_protocol,
            facts.host_port_speed_gbps,
        )
        if value is not None
    )
    if known_count >= 2:
        return FIT_TIER_POSSIBLE
    return FIT_TIER_FALLBACK_UNKNOWN


def _storage_connectivity_fit_tier(
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
) -> str:
    if (
        requirements.host_protocol != UNKNOWN_FACT
        and facts.host_protocol not in {UNKNOWN_FACT, requirements.host_protocol}
    ):
        return FIT_TIER_EXPLICIT_MISMATCH
    if (
        requirements.host_port_speed_gbps is not None
        and facts.host_port_speed_gbps is not None
        and facts.host_port_speed_gbps < requirements.host_port_speed_gbps
    ):
        return FIT_TIER_EXPLICIT_MISMATCH
    known_protocol = facts.host_protocol != UNKNOWN_FACT
    known_speed = facts.host_port_speed_gbps is not None
    if known_protocol and (
        requirements.host_port_speed_gbps is None or known_speed
    ):
        return FIT_TIER_STRONG
    if known_protocol or known_speed:
        return FIT_TIER_POSSIBLE
    return FIT_TIER_FALLBACK_UNKNOWN


def _role_has_any_known_fact(role: str, facts: _ProductFacts) -> bool:
    if role in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}:
        return bool(
            facts.drive_capacity_tb is not None
            or facts.storage_capacity_tb is not None
            or facts.drive_type != UNKNOWN_FACT
            or facts.drive_interface != UNKNOWN_FACT
        )
    if role in {TRANSCEIVER_ROLE, DAC_CABLE_ROLE, CABLE_ROLE}:
        return facts.transceiver_form_factor != UNKNOWN_FACT or facts.port_speed_gbps is not None
    return True


def _simple_role_quantity_required(
    role: str,
    requirements: _NormalizedServerRequirements,
) -> int:
    role_requirements = _role_parsed_requirements(requirements, role)
    quantity = (
        _as_int(role_requirements.get("count"))
        or _as_int(role_requirements.get("quantity"))
        or _as_int(role_requirements.get("device_count"))
    )
    if quantity is not None and quantity > 0:
        return quantity
    if requirements.product_group == NETWORK_PRODUCT_GROUP and role in {
        LICENSE_ROLE,
        SUPPORT_ROLE,
        STACKING_MODULE_ROLE,
    }:
        return requirements.server_qty
    if requirements.product_group == NETWORK_PRODUCT_GROUP and role in {
        SWITCH_ROLE,
        ROUTER_ROLE,
        FIREWALL_ROLE,
        ACCESS_POINT_ROLE,
    }:
        return requirements.server_qty
    if requirements.product_group == NETWORK_PRODUCT_GROUP and role == TRANSCEIVER_ROLE:
        primary = next(
            (
                item
                for item in (SWITCH_ROLE, ROUTER_ROLE, FIREWALL_ROLE, ACCESS_POINT_ROLE)
                if item in requirements.required_roles
            ),
            None,
        )
        primary_requirements = _role_parsed_requirements(requirements, primary or "")
        uplink_count = _as_int(primary_requirements.get("uplink_count"))
        if uplink_count is not None:
            return uplink_count * requirements.server_qty
    if requirements.product_group == STORAGE_PRODUCT_GROUP:
        if role == STORAGE_SYSTEM_ROLE:
            return requirements.server_qty
        if role in {LICENSE_ROLE, SUPPORT_ROLE}:
            return requirements.server_qty
        if role in {STORAGE_ARRAY_CONTROLLER_ROLE, CONTROLLER_MODULE_ROLE}:
            return requirements.controller_count or requirements.server_qty
        if role == DISK_SHELF_ROLE:
            return requirements.shelf_count or requirements.server_qty
        if role in {HOST_PORT_ROLE, PROTOCOL_MODULE_ROLE}:
            return requirements.host_port_count or requirements.server_qty
        if role in {TRANSCEIVER_ROLE, CABLE_ROLE}:
            return requirements.host_port_count or requirements.server_qty
        if role in {POWER_SUPPLY_ROLE, RAIL_KIT_ROLE}:
            return quantity or requirements.server_qty
    return requirements.server_qty


def _storage_drive_quantity_required(
    role: str,
    requirements: _NormalizedServerRequirements,
) -> int:
    if role not in {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}:
        return requirements.server_qty
    role_requirements = _role_parsed_requirements(requirements, role)
    quantity = (
        _as_int(role_requirements.get("count"))
        or _as_int(role_requirements.get("quantity"))
        or _as_int(role_requirements.get("drive_count"))
        or requirements.drive_count
    )
    if quantity is not None and quantity > 0:
        return quantity
    if requirements.product_group == STORAGE_PRODUCT_GROUP:
        return requirements.server_qty
    return requirements.server_qty * (requirements.storage_qty_per_server or 1)


def _role_parsed_requirements(
    requirements: _NormalizedServerRequirements,
    role: str,
) -> Mapping[str, Any]:
    role_plan = _dict_or_empty(requirements.role_plan)
    role_requirements = _dict_or_empty(role_plan.get("requirements_by_role"))
    return _dict_or_empty(role_requirements.get(role))


def _first_text_requirement(
    role_plan: Mapping[str, Any],
    key: str,
    *,
    preferred_roles: Sequence[str] = (),
) -> str | None:
    value = _first_requirement_value(role_plan, key, preferred_roles=preferred_roles)
    text = str(value or "").strip()
    return text or None


def _first_int_requirement(
    role_plan: Mapping[str, Any],
    key: str,
    *,
    preferred_roles: Sequence[str] = (),
) -> int | None:
    return _as_int(_first_requirement_value(role_plan, key, preferred_roles=preferred_roles))


def _first_float_requirement(
    role_plan: Mapping[str, Any],
    key: str,
    *,
    preferred_roles: Sequence[str] = (),
) -> float | None:
    value = _first_requirement_value(role_plan, key, preferred_roles=preferred_roles)
    return _float_value(value)


def _first_requirement_value(
    role_plan: Mapping[str, Any],
    key: str,
    *,
    preferred_roles: Sequence[str] = (),
) -> Any:
    role_requirements = _dict_or_empty(role_plan.get("requirements_by_role"))
    for role in preferred_roles:
        parsed = _dict_or_empty(role_requirements.get(role))
        value = parsed.get(key)
        if value not in (None, ""):
            return value
    for capability in _mapping_list(role_plan.get("required_capabilities")):
        role = str(capability.get("role") or "").strip()
        if preferred_roles and role not in set(preferred_roles):
            continue
        parsed = _dict_or_empty(capability.get("parsed_requirements"))
        value = parsed.get(key)
        if value not in (None, ""):
            return value
    for parsed in role_requirements.values():
        row = _dict_or_empty(parsed)
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _role_review_warning(role: str, requirements: _NormalizedServerRequirements) -> str:
    if requirements.product_group == NETWORK_PRODUCT_GROUP:
        return "Требуется инженерная проверка сетевых характеристик и комплектности перед КП."
    if requirements.product_group == STORAGE_PRODUCT_GROUP:
        return (
            "Требуется инженерная проверка емкости, протоколов, лицензий "
            "и комплектности СХД перед КП."
        )
    return "Требуется проверка совместимости компонента с платформой."


def _make_component_candidate(
    *,
    role: str,
    product: DistributorProduct,
    facts: _ProductFacts,
    requirements: _NormalizedServerRequirements,
    stock_rows: list[DistributorStockPrice],
    quantity_required: int,
    score: int,
    eligibility_status: str,
    warnings: list[str],
    reasons: list[str],
    fit_label: str = FIT_UNKNOWN,
    fit_reason: str = "",
    fit_tier: str | None = None,
    match_warnings: Sequence[str] = (),
    uncertainty_reasons: Sequence[str] = (),
    objective_reject_reason: str | None = None,
    evidence_summary: str = "",
    cpu_over_requirement: int | None = None,
    storage_over_requirement: float | None = None,
    ram_over_requirement_gb: int | None = None,
) -> _ComponentCandidate:
    price_value, price_currency = _select_price(stock_rows)
    available_quantity = _available_quantity(stock_rows)
    if available_quantity is None:
        warnings = [*warnings, f"{_role_ru(role)}: остаток не найден."]
    elif available_quantity < quantity_required:
        warnings = [
            *warnings,
            (
                f"{_role_ru(role)}: остаток ниже требования, доступно "
                f"{available_quantity} шт., требуется {quantity_required} шт."
            ),
        ]
        score += 2
    else:
        score += 10
        reasons = [*reasons, "Остаток закрывает требуемое количество."]

    return _ComponentCandidate(
        role=role,
        product=product,
        facts=facts,
        quantity_required=quantity_required,
        available_quantity=available_quantity,
        reservable_locations=_reservable_locations(stock_rows),
        price_value=price_value,
        price_currency=price_currency,
        eligibility_status=eligibility_status,
        eligibility_warnings=_unique(warnings),
        fit_reasons=_unique(reasons),
        score=max(1, min(score, 100)),
        fit_label=fit_label,
        fit_reason=fit_reason,
        fit_tier=fit_tier
        or _candidate_fit_tier(
            role=role,
            product=product,
            facts=facts,
            requirements=requirements,
        ),
        match_warnings=_unique([str(item) for item in match_warnings if str(item).strip()]),
        uncertainty_reasons=_unique(
            [str(item) for item in uncertainty_reasons if str(item).strip()]
        ),
        objective_reject_reason=objective_reject_reason,
        evidence_summary=evidence_summary,
        cpu_over_requirement=cpu_over_requirement,
        storage_over_requirement=storage_over_requirement,
        ram_over_requirement_gb=ram_over_requirement_gb,
    )


def _rank_component_candidates(
    candidates: list[_ComponentCandidate],
    *,
    limit: int,
) -> list[_ComponentCandidate]:
    return sorted(candidates, key=_component_candidate_sort_key)[:limit]


def _component_candidate_sort_key(candidate: _ComponentCandidate) -> tuple[Any, ...]:
    return (
        _fit_tier_rank(candidate.fit_tier),
        candidate.bucket_priority,
        -candidate.score,
        *_component_price_sort_key(candidate),
        *_component_over_requirement_sort_key(candidate),
        -(_stock_sort_value(candidate.available_quantity)),
        _stable_text(candidate.facts.normalized_vendor or candidate.product.producer),
        _stable_text(candidate.product.part_number),
        _stable_text(candidate.candidate_id),
    )


def _component_price_sort_key(candidate: _ComponentCandidate) -> tuple[int, Decimal]:
    if candidate.price_value is None:
        return (1, Decimal("Infinity"))
    return (0, candidate.price_value * candidate.quantity_required)


def _component_over_requirement_sort_key(candidate: _ComponentCandidate) -> tuple[int, float]:
    value = _component_over_requirement_value(candidate)
    if value in (None, ""):
        return (1, float("inf"))
    return (0, float(value))


def _stable_text(value: Any) -> str:
    return str(value or "").strip().casefold()


def _right_sized_component_sort_key(
    candidate: _ComponentCandidate,
) -> tuple[int, int, float, int, int, int, int]:
    return (
        _candidate_stock_fit_score(candidate),
        0 if candidate.fit_label == FIT_UNKNOWN else 1,
        _candidate_total_lower_price_rank(candidate),
        _fit_label_rank(candidate.fit_label),
        candidate.score,
        candidate.available_quantity or 0,
        candidate.reservable_locations,
    )


def _generate_build_candidates(
    *,
    spec: StockSpec,
    item: StockSpecItem,
    item_index: int,
    matrix: _CandidateMatrix,
) -> list[MatchCandidateResult]:
    requirements = matrix.normalized_requirements
    storage_role = _storage_role_for_requirements(requirements)
    candidates: list[MatchCandidateResult] = []
    seen_candidate_ids: set[str] = set()

    for platform_index, platform_candidate in enumerate(
        matrix.platform_candidates[:MAX_PLATFORM_CANDIDATES]
    ):
        cpu_pool = [
            candidate
            for candidate in matrix.cpu_candidates[:MAX_COMPONENT_CANDIDATES_PER_ROLE]
            if _cpu_candidate_allowed_for_platform(candidate, platform_candidate)
        ]
        ram_pool = [
            candidate
            for candidate in matrix.ram_candidates[:MAX_COMPONENT_CANDIDATES_PER_ROLE]
            if _ram_candidate_allowed_for_platform(candidate, platform_candidate)
        ]
        storage_pool = _storage_candidates_for_role(matrix, storage_role)

        cpu_options = (
            cpu_pool or [None]
            if requirements.total_cpu_required is not None
            else []
        )
        ram_options = (
            ram_pool or [None]
            if requirements.ram_gb_per_server is not None
            else []
        )
        storage_options = (
            storage_pool or [None]
            if requirements.storage_required and storage_role is not None
            else []
        )
        platform_satisfies_network = _platform_onboard_network_satisfies(
            platform_candidate,
            requirements,
        )
        network_pool = matrix.network_adapter_candidates[:MAX_COMPONENT_CANDIDATES_PER_ROLE]
        network_options = (
            network_pool or [None]
            if requirements.network_required and not platform_satisfies_network
            else []
        )
        variants = min(
            max(
                len(cpu_options) or 1,
                len(ram_options) or 1,
                len(storage_options) or 1,
                len(network_options) or 1,
            ),
            MAX_COMPONENT_CANDIDATES_PER_ROLE,
        )

        for variant_index in range(variants):
            cpu_candidate = _pick_candidate(cpu_options, variant_index + platform_index)
            ram_candidate = _pick_candidate(ram_options, variant_index + platform_index)
            storage_candidate = _pick_candidate(storage_options, variant_index + platform_index)
            network_candidate = _pick_candidate(network_options, variant_index + platform_index)
            candidate = _build_configuration_candidate_from_candidates(
                spec=spec,
                item=item,
                item_index=item_index,
                requirements=requirements,
                platform_candidate=platform_candidate,
                cpu_candidate=cpu_candidate,
                ram_candidate=ram_candidate,
                storage_candidate=storage_candidate,
                network_candidate=network_candidate,
                storage_role=storage_role,
            )
            if candidate.candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate.candidate_id or candidate.item_id)
            candidates.append(candidate)
            if len(candidates) >= MAX_INTERNAL_BUILD_CANDIDATES:
                return candidates

    return candidates


def _build_configuration_candidate_from_candidates(
    *,
    spec: StockSpec,
    item: StockSpecItem,
    item_index: int,
    requirements: _NormalizedServerRequirements,
    platform_candidate: _ComponentCandidate,
    cpu_candidate: _ComponentCandidate | None,
    ram_candidate: _ComponentCandidate | None,
    storage_candidate: _ComponentCandidate | None,
    network_candidate: _ComponentCandidate | None,
    storage_role: str | None,
) -> MatchCandidateResult:
    components = [_build_component_from_candidate(platform_candidate)]
    matched_requirements = ["Платформа выбрана из candidate matrix серверных платформ."]
    missing_components: list[str] = []
    missing_component_roles: list[str] = []
    excluded_from_total_roles: list[str] = []
    compatibility_warnings = [
        *platform_candidate.eligibility_warnings,
        "Требуется инженерная проверка совместимости платформы, RAM, накопителей и адаптеров.",
    ]
    rank_reason = list(platform_candidate.fit_reasons)

    if requirements.total_cpu_required is not None:
        if cpu_candidate is None:
            missing_component_roles.append(CPU_ROLE)
            excluded_from_total_roles.append(CPU_ROLE)
            missing_components.append("Неполная сборка - требуется подбор CPU.")
            rank_reason.append("Совместимый CPU в candidate matrix для платформы не найден.")
        else:
            cpu_component = _build_component_from_candidate(cpu_candidate)
            components.append(cpu_component)
            matched_requirements.append(_component_match_text(cpu_component))
            compatibility_warnings.extend(cpu_candidate.eligibility_warnings)
            rank_reason.extend(cpu_candidate.fit_reasons)

    if requirements.ram_gb_per_server is not None:
        if ram_candidate is None:
            missing_component_roles.append(RAM_ROLE)
            excluded_from_total_roles.append(RAM_ROLE)
            missing_components.append("RAM: не подобраны совместимые модули памяти.")
            rank_reason.append("RAM в candidate matrix для платформы не найдена.")
        else:
            ram_component = _build_component_from_candidate(ram_candidate)
            components.append(ram_component)
            matched_requirements.append(_component_match_text(ram_component))
            compatibility_warnings.extend(ram_candidate.eligibility_warnings)
            rank_reason.extend(ram_candidate.fit_reasons)

    if requirements.storage_required and storage_role is not None:
        if storage_candidate is None:
            missing_component_roles.append(storage_role)
            excluded_from_total_roles.append(storage_role)
            missing_components.append(f"{_role_ru(storage_role)}: не подобраны.")
            rank_reason.append(f"{_role_ru(storage_role)} в candidate matrix не найдены.")
        else:
            storage_component = _build_component_from_candidate(storage_candidate)
            components.append(storage_component)
            matched_requirements.append(_component_match_text(storage_component))
            compatibility_warnings.extend(storage_candidate.eligibility_warnings)
            rank_reason.extend(storage_candidate.fit_reasons)

    if requirements.network_required:
        if _platform_onboard_network_satisfies(platform_candidate, requirements):
            matched_requirements.append(
                "Platform onboard network satisfies requested port speed/media requirement."
            )
            rank_reason.append("Onboard network covers the requested network role.")
        elif network_candidate is None:
            missing_component_roles.append(NETWORK_ADAPTER_ROLE)
            excluded_from_total_roles.append(NETWORK_ADAPTER_ROLE)
            missing_components.append(
                "Network adapter: no candidate satisfies requested port speed/media requirement."
            )
            rank_reason.append("Network adapter is required but absent from candidate matrix.")
        else:
            network_component = _build_component_from_candidate(network_candidate)
            components.append(network_component)
            matched_requirements.append(_component_match_text(network_component))
            compatibility_warnings.extend(network_candidate.eligibility_warnings)
            rank_reason.extend(network_candidate.fit_reasons)

    compatibility_warnings.extend(_standard_build_warnings_for_requirements(requirements))
    total_price_value, total_price_currency, price_warnings = _build_total_price(components)
    compatibility_warnings.extend(price_warnings)
    missing_components.extend(_component_stock_warnings(components))
    missing_requirements = _unique(missing_components)
    compatibility_warnings = _unique(compatibility_warnings)
    risk_flags = _unique([*compatibility_warnings, *missing_requirements])
    component_rows = [component.to_report_json() for component in components]
    platform_row = component_rows[0]
    included_component_roles = _unique(component.role for component in components)
    missing_component_roles = _unique(missing_component_roles)
    excluded_from_total_roles = _unique(excluded_from_total_roles)
    completeness_status = (
        "incomplete"
        if missing_requirements or missing_component_roles
        else "complete"
    )
    completeness_label = _build_completeness_label(
        completeness_status,
        missing_component_roles=missing_component_roles,
    )
    total_price_note = _total_price_note(excluded_from_total_roles)
    available_quantity = _build_available_quantity(components)
    platform_name = _component_display_name(components[0])
    build_title = (
        "Неполная сборка"
        if completeness_status == "incomplete"
        else "Предварительная сборка"
    )
    right_size_summary = _build_right_size_summary(
        requirements=requirements,
        cpu_candidate=cpu_candidate,
        ram_candidate=ram_candidate,
        storage_candidate=storage_candidate,
    )
    score = _score_build_candidate(
        requirements=requirements,
        platform_candidate=platform_candidate,
        cpu_candidate=cpu_candidate,
        ram_candidate=ram_candidate,
        storage_candidate=storage_candidate,
        network_candidate=network_candidate,
        completeness_status=completeness_status,
        missing_component_roles=missing_component_roles,
        compatibility_warnings=compatibility_warnings,
        total_price_value=total_price_value,
    )
    if completeness_status == "complete":
        rank_reason.append("Все обязательные роли для предварительной сборки подобраны.")
    else:
        rank_reason.append("Сборка неполная; сумма считается без отсутствующих ролей.")

    candidate_id = _build_candidate_id(
        item_index,
        platform_candidate,
        cpu_candidate,
        ram_candidate,
        storage_candidate,
        network_candidate,
    )
    raw = {
        "spec_item_index": item_index,
        "quantity_required": requirements.server_qty,
        "candidate_type": BUILD_CANDIDATE_TYPE,
        "candidate_id": candidate_id,
        "normalized_requirements": requirements.to_report_json(),
        "platform": _jsonable(platform_row),
        "components": _jsonable(component_rows),
        "total_price_value": _jsonable(total_price_value),
        "total_price_currency": total_price_currency,
        "missing_components": missing_requirements,
        "compatibility_warnings": compatibility_warnings,
        "engineer_review_required": True,
        "completeness_status": completeness_status,
        "completeness_label": completeness_label,
        "included_component_roles": included_component_roles,
        "missing_component_roles": missing_component_roles,
        "excluded_from_total_roles": excluded_from_total_roles,
        "cpu_per_server": requirements.cpu_per_server,
        "total_cpu_required": requirements.total_cpu_required,
        "total_price_note": total_price_note,
        "score": score,
        "rank_reason": rank_reason,
        **right_size_summary,
    }

    return MatchCandidateResult(
        distributor_code=platform_candidate.product.distributor_code,
        item_id=candidate_id,
        product_key=platform_candidate.product.product_key,
        part_number=platform_candidate.product.part_number,
        producer=platform_candidate.product.producer,
        category_id=platform_candidate.product.category_id,
        item_name=f"{build_title} на платформе {platform_name}",
        confidence_score=score,
        price_value=total_price_value,
        price_currency=total_price_currency,
        available_quantity=available_quantity,
        reservable_locations=components[0].reservable_locations,
        matched_requirements=_unique(matched_requirements),
        missing_requirements=missing_requirements,
        risk_flags=risk_flags,
        raw=raw,
        candidate_type=BUILD_CANDIDATE_TYPE,
        components=component_rows,
        total_price_value=total_price_value,
        total_price_currency=total_price_currency,
        missing_components=missing_requirements,
        compatibility_warnings=compatibility_warnings,
        engineer_review_required=True,
        completeness_status=completeness_status,
        completeness_label=completeness_label,
        included_component_roles=included_component_roles,
        missing_component_roles=missing_component_roles,
        excluded_from_total_roles=excluded_from_total_roles,
        cpu_per_server=requirements.cpu_per_server,
        total_cpu_required=requirements.total_cpu_required,
        total_price_note=total_price_note,
        platform=platform_row,
        score=score,
        rank_reason=rank_reason,
        candidate_id=candidate_id,
    )


def _storage_candidates_for_role(
    matrix: _CandidateMatrix,
    role: str | None,
) -> list[_ComponentCandidate]:
    if role == DRIVE_ROLE:
        return matrix.drive_candidates[:MAX_COMPONENT_CANDIDATES_PER_ROLE]
    if role == SSD_ROLE:
        return matrix.ssd_candidates[:MAX_COMPONENT_CANDIDATES_PER_ROLE]
    if role == HDD_ROLE:
        return matrix.hdd_candidates[:MAX_COMPONENT_CANDIDATES_PER_ROLE]
    return []


def _storage_role_for_requirements(
    requirements: _NormalizedServerRequirements,
) -> str | None:
    if requirements.product_group == STORAGE_PRODUCT_GROUP:
        if DRIVE_ROLE in requirements.required_roles:
            return DRIVE_ROLE
        if (
            requirements.storage_type_preference == "SSD"
            and SSD_ROLE in requirements.required_roles
        ):
            return SSD_ROLE
        if (
            requirements.storage_type_preference == "HDD"
            and HDD_ROLE in requirements.required_roles
        ):
            return HDD_ROLE
        return DRIVE_ROLE
    if (
        requirements.storage_type_preference == "SSD"
        or requirements.storage_interface_preference in {"NVMe", "SAS", "SATA"}
    ):
        return SSD_ROLE
    if requirements.storage_type_preference == "HDD":
        return HDD_ROLE
    return None


def _platform_onboard_network_satisfies(
    platform_candidate: _ComponentCandidate,
    requirements: _NormalizedServerRequirements,
) -> bool:
    if not requirements.network_required:
        return True
    return network_facts_satisfy_requirement(
        platform_candidate.facts.to_report_json(),
        requirements.network_requirement,
    )


def _pick_candidate(
    candidates: list[_ComponentCandidate | None],
    index: int,
) -> _ComponentCandidate | None:
    if not candidates:
        return None
    return candidates[index % len(candidates)]


def _cpu_candidate_allowed_for_platform(
    candidate: _ComponentCandidate,
    platform_candidate: _ComponentCandidate,
) -> bool:
    if _fatal_platform_cpu_mismatch(platform_candidate.facts, candidate.facts):
        return False
    if not candidate.facts.is_vendor_option_kit:
        return True
    return (
        candidate.facts.option_kit_vendor != UNKNOWN_FACT
        and candidate.facts.option_kit_vendor == platform_candidate.facts.normalized_vendor
    )


def _ram_candidate_allowed_for_platform(
    candidate: _ComponentCandidate,
    platform_candidate: _ComponentCandidate,
) -> bool:
    platform_ram_type = platform_candidate.facts.ram_type
    ram_type = candidate.facts.ram_type
    if platform_ram_type == UNKNOWN_FACT or ram_type == UNKNOWN_FACT:
        return True
    return platform_ram_type == ram_type


def _looks_like_3_5_only_platform(facts: _ProductFacts) -> bool:
    hints = {hint.upper() for hint in facts.form_factor_hints}
    return "LFF/3.5" in hints and facts.nvme_support is not True


def _fatal_platform_cpu_mismatch(
    platform_facts: _ProductFacts,
    cpu_facts: _ProductFacts,
) -> str | None:
    platform_side = _cpu_platform_side(platform_facts)
    cpu_side = _cpu_component_side(cpu_facts)
    if platform_side == "AMD" and cpu_side == "Intel":
        return "fatal compatibility mismatch: AMD EPYC/SP platform with Intel Xeon CPU"
    if platform_side == "Intel" and cpu_side == "AMD":
        return "fatal compatibility mismatch: Intel Xeon platform with AMD EPYC CPU"

    platform_socket = platform_facts.cpu_socket
    cpu_socket = cpu_facts.cpu_socket
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


def _cpu_platform_side(facts: _ProductFacts) -> str:
    if facts.cpu_family == "EPYC" or facts.cpu_socket in {"SP3", "SP5", "LGA4094", "LGA6096"}:
        return "AMD"
    if facts.cpu_family == "Xeon" or facts.cpu_socket in {"LGA3647", "LGA4189", "LGA4677"}:
        return "Intel"
    return facts.cpu_brand if facts.cpu_brand in {"Intel", "AMD"} else UNKNOWN_FACT


def _cpu_component_side(facts: _ProductFacts) -> str:
    if facts.cpu_family == "Xeon" or facts.cpu_socket in {"LGA3647", "LGA4189", "LGA4677"}:
        return "Intel"
    if facts.cpu_family == "EPYC" or facts.cpu_socket in {"SP3", "SP5", "LGA4094", "LGA6096"}:
        return "AMD"
    return facts.cpu_brand if facts.cpu_brand in {"Intel", "AMD"} else UNKNOWN_FACT


def _score_build_candidate(
    *,
    requirements: _NormalizedServerRequirements,
    platform_candidate: _ComponentCandidate,
    cpu_candidate: _ComponentCandidate | None,
    ram_candidate: _ComponentCandidate | None,
    storage_candidate: _ComponentCandidate | None,
    network_candidate: _ComponentCandidate | None,
    completeness_status: str,
    missing_component_roles: list[str],
    compatibility_warnings: list[str],
    total_price_value: Decimal | None,
) -> int:
    selected = [
        candidate
        for candidate in [
            platform_candidate,
            cpu_candidate,
            ram_candidate,
            storage_candidate,
            network_candidate,
        ]
        if candidate is not None
    ]
    score = 25 + sum(candidate.score for candidate in selected) // max(len(selected) * 4, 1)
    score += sum(_candidate_stock_fit_score(candidate) for candidate in selected) // 2
    if completeness_status == "complete":
        score += 18
    score -= len(missing_component_roles) * 14
    score -= min(len(compatibility_warnings), 15)
    score -= _build_overfit_penalty(
        requirements=requirements,
        cpu_candidate=cpu_candidate,
        ram_candidate=ram_candidate,
        storage_candidate=storage_candidate,
    )
    if total_price_value is not None:
        score += 2
    return max(1, min(score, 95))


def _build_candidate_id(
    item_index: int,
    *candidates: _ComponentCandidate | None,
) -> str:
    raw_key = "|".join(
        _candidate_identity_key(candidate) if candidate is not None else "missing"
        for candidate in candidates
    )
    digest = sha1(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"build-v03-{item_index + 1}-{digest}"


def _stable_candidate_id(role: str, product: DistributorProduct) -> str:
    raw_key = f"{role}|{product.distributor_code}|{product.item_id}|{product.part_number or ''}"
    digest = sha1(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"{role}-{digest}"


def _candidate_identity_key(candidate: MatchCandidateResult | _ComponentCandidate) -> str:
    if isinstance(candidate, MatchCandidateResult):
        component_key = candidate.item_id
        platform = candidate.platform if isinstance(candidate.platform, dict) else {}
        if platform:
            component_key = "::".join(
                [
                    str(platform.get("producer") or ""),
                    str(platform.get("part_number") or ""),
                    str(platform.get("item_id") or component_key),
                ]
            )
        return component_key
    product = candidate.product
    return "::".join(
        [
            candidate.role,
            product.distributor_code,
            product.item_id,
            product.part_number or "",
            product.producer or "",
        ]
    )


def _select_diverse_build_candidates(
    candidates: list[MatchCandidateResult],
) -> list[MatchCandidateResult]:
    remaining = list(candidates[:MAX_INTERNAL_BUILD_CANDIDATES])
    selected: list[MatchCandidateResult] = []
    platform_counts: dict[str, int] = {}
    component_counts: dict[tuple[str, str], int] = {}

    while remaining and len(selected) < MAX_BUILD_CANDIDATES:
        allowed = [
            candidate
            for candidate in remaining
            if platform_counts.get(_platform_dedupe_key(candidate), 0) < 2
            and all(
                component_counts.get((role, key), 0) < 3
                for role, key in _selected_component_keys(candidate).items()
                if role in {CPU_ROLE, RAM_ROLE, SSD_ROLE, HDD_ROLE}
            )
        ]
        if not allowed:
            allowed = remaining

        best = max(
            allowed,
            key=lambda candidate: _diversity_sort_key(
                candidate,
                platform_counts=platform_counts,
                component_counts=component_counts,
            ),
        )
        remaining.remove(best)

        diversity_penalty = _diversity_penalty(
            best,
            platform_counts=platform_counts,
            component_counts=component_counts,
        )
        selected_candidate = _with_adjusted_build_score(best, diversity_penalty)
        selected.append(selected_candidate)
        platform_counts[_platform_dedupe_key(best)] = (
            platform_counts.get(_platform_dedupe_key(best), 0) + 1
        )
        for role, key in _selected_component_keys(best).items():
            component_counts[(role, key)] = component_counts.get((role, key), 0) + 1

    return selected


def _diversity_sort_key(
    candidate: MatchCandidateResult,
    *,
    platform_counts: dict[str, int],
    component_counts: dict[tuple[str, str], int],
) -> tuple[int, float, int, int, int]:
    adjusted_score = (candidate.score or candidate.confidence_score) - _diversity_penalty(
        candidate,
        platform_counts=platform_counts,
        component_counts=component_counts,
    )
    return (
        1 if candidate.completeness_status == "complete" else 0,
        -float(candidate.total_price_value) if candidate.total_price_value is not None else 0.0,
        adjusted_score,
        candidate.available_quantity or 0,
        -len(candidate.compatibility_warnings),
    )


def _diversity_penalty(
    candidate: MatchCandidateResult,
    *,
    platform_counts: dict[str, int],
    component_counts: dict[tuple[str, str], int],
) -> int:
    penalty = platform_counts.get(_platform_dedupe_key(candidate), 0) * 8
    for role, key in _selected_component_keys(candidate).items():
        if role in {CPU_ROLE, RAM_ROLE, SSD_ROLE, HDD_ROLE}:
            penalty += component_counts.get((role, key), 0) * 5
    return penalty


def _with_adjusted_build_score(
    candidate: MatchCandidateResult,
    diversity_penalty: int,
) -> MatchCandidateResult:
    if diversity_penalty <= 0:
        return candidate
    score = max(1, (candidate.score or candidate.confidence_score) - diversity_penalty)
    rank_reason = [
        *candidate.rank_reason,
        "Score снижен за повтор платформы или компонентов в уже выбранном top.",
    ]
    raw = {**candidate.raw, "score": score, "rank_reason": rank_reason}
    return replace(
        candidate,
        confidence_score=score,
        score=score,
        rank_reason=rank_reason,
        raw=raw,
    )


def _selected_component_keys(candidate: MatchCandidateResult) -> dict[str, str]:
    keys: dict[str, str] = {}
    for component in candidate.components:
        role = str(component.get("role") or "")
        if not role:
            continue
        keys[role] = "::".join(
            [
                str(component.get("producer") or ""),
                str(component.get("part_number") or ""),
                str(component.get("item_id") or ""),
            ]
        ).casefold()
    return keys


def _limited_alternatives_warnings(
    selected: list[MatchCandidateResult],
    matrix_json: dict[str, Any],
) -> list[str]:
    if len(selected) < 2:
        return []
    roles = {
        CPU_ROLE: "cpu_candidates",
        RAM_ROLE: "ram_candidates",
        SSD_ROLE: "ssd_candidates",
    }
    same_roles: list[str] = []
    for role, matrix_key in roles.items():
        role_keys = {
            key
            for candidate in selected
            if (key := _selected_component_keys(candidate).get(role))
        }
        if not role_keys:
            continue
        matrix_candidates = matrix_json.get(matrix_key)
        matrix_size = len(matrix_candidates) if isinstance(matrix_candidates, list) else 0
        if len(role_keys) <= 1 and matrix_size <= 1:
            same_roles.append(role)
    if {CPU_ROLE, RAM_ROLE, SSD_ROLE}.issubset(set(same_roles)):
        return [
            "Альтернатив по CPU/RAM/SSD на складе недостаточно; варианты отличаются "
            "в основном платформой."
        ]
    return []


def _standard_build_warnings_for_requirements(
    requirements: _NormalizedServerRequirements,
) -> list[str]:
    warnings = [
        "Совместимость RAM с платформой требуется проверить инженеру.",
        "Совместимость SSD/HDD/контроллера требуется проверить инженеру.",
        "Достаточность корзин и слотов платформы не подтверждена локальными данными OCS.",
        "Гарантию и срок поставки требуется проверить по OCS/поставщику.",
        "Характеристики в OCS могут быть неполными; требуется ручная проверка.",
    ]
    if requirements.ram_type_preference == UNKNOWN_FACT:
        warnings.append("Тип RAM не указан; требуется проверка совместимости RAM с платформой.")
    if (
        requirements.cpu_per_server is not None
        and requirements.cpu_vendor_preference == UNKNOWN_FACT
        and requirements.cpu_family_preference == UNKNOWN_FACT
        and requirements.cpu_min_cores_per_cpu is None
        and requirements.cpu_generation_or_model_hint is None
    ):
        warnings.append(
            "Характеристики CPU в запросе не указаны; CPU выбран предварительно "
            "по наличию и базовым признакам."
        )
    if (
        requirements.storage_required
        and requirements.storage_type_preference in {"SSD", "NVMe", "SAS", "SATA"}
        and not requirements.storage_min_capacity
    ):
        warnings.append(
            "Объем SSD в запросе не указан; накопитель выбран предварительно по наличию."
        )
    return warnings


def _build_configuration_candidate(
    *,
    spec: StockSpec,
    item: StockSpecItem,
    item_index: int,
    platform: DistributorProduct,
    products_by_role: dict[str, list[DistributorProduct]],
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> MatchCandidateResult:
    server_quantity = item.quantity
    cpu_per_server = _cpu_per_server(spec, item)
    total_cpu_required = (
        cpu_per_server * server_quantity
        if cpu_per_server is not None
        else None
    )
    platform_facts = _facts_for_product(
        platform,
        facts_by_key=facts_by_key,
        role=SERVER_PLATFORM_ROLE,
    )
    components: list[_BuildComponent] = [
        _component_from_product(
            SERVER_PLATFORM_ROLE,
            platform,
            stock_rows_by_key,
            quantity_required=server_quantity,
            facts=platform_facts,
        )
    ]
    matched_requirements = ["Платформа выбрана из складской категории серверных платформ."]
    missing_components: list[str] = []
    missing_component_roles: list[str] = []
    excluded_from_total_roles: list[str] = []
    compatibility_warnings = _platform_compatibility_warnings(spec, item, platform)
    rank_reason: list[str] = ["Платформа выбрана из candidate pool серверных платформ."]
    score = 25

    if components[0].available_quantity is None:
        missing_components.append(
            f"Платформа: остаток не найден, требуется {server_quantity} шт."
        )
    elif components[0].available_quantity < server_quantity:
        missing_components.append(
            "Платформа: остаток ниже требования, "
            f"доступно {components[0].available_quantity} шт., требуется {server_quantity} шт."
        )
    else:
        matched_requirements.append(
            f"Платформа: доступно {components[0].available_quantity} шт."
        )
        score += 10
        rank_reason.append("Остаток платформы закрывает требуемое количество серверов.")

    if total_cpu_required is not None:
        cpu_component, cpu_missing, cpu_warnings, cpu_rank_reason = _select_cpu_component(
            platform_facts=platform_facts,
            products=products_by_role.get(CPU_ROLE, []),
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
            quantity_required=total_cpu_required,
        )
        compatibility_warnings.extend(cpu_warnings)
        rank_reason.extend(cpu_rank_reason)
        if cpu_component is None:
            missing_component_roles.append(CPU_ROLE)
            excluded_from_total_roles.append(CPU_ROLE)
            missing_components.extend(cpu_missing)
            missing_components.append("Неполная сборка - требуется подбор CPU.")
        else:
            components.append(cpu_component)
            matched_requirements.append(_component_match_text(cpu_component))
            score += 15
            missing_components.extend(cpu_missing)

    ram_component, ram_missing, ram_warnings = _select_ram_component(
        spec=spec,
        item=item,
        products=products_by_role.get(RAM_ROLE, []),
        platform_facts=platform_facts,
        facts_by_key=facts_by_key,
        stock_rows_by_key=stock_rows_by_key,
    )
    if ram_component is not None:
        components.append(ram_component)
        matched_requirements.append(_component_match_text(ram_component))
        score += 15
        rank_reason.append("RAM подобрана из candidate pool серверной памяти.")
    elif ram_missing:
        missing_component_roles.append(RAM_ROLE)
    missing_components.extend(ram_missing)
    compatibility_warnings.extend(ram_warnings)

    storage_role = _required_storage_role(spec, item)
    if storage_role is not None:
        storage_requirements = _normalize_server_requirements(spec, item)
        storage_quantity_required = (
            server_quantity * (storage_requirements.storage_qty_per_server or 1)
        )
        storage_component, storage_missing, storage_warnings = _select_storage_component(
            role=storage_role,
            products=products_by_role.get(storage_role, []),
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
            quantity_required=storage_quantity_required,
        )
        compatibility_warnings.extend(storage_warnings)
        if storage_component is None:
            missing_component_roles.append(storage_role)
            missing_components.extend(storage_missing)
        else:
            components.append(storage_component)
            matched_requirements.append(_component_match_text(storage_component))
            score += 10
            rank_reason.append(f"{_role_ru(storage_role)} подобраны из нужной серверной категории.")

    if _requires_storage_controller(spec, item):
        controller_component, controller_missing = _select_simple_component(
            role=STORAGE_CONTROLLER_ROLE,
            products=products_by_role.get(STORAGE_CONTROLLER_ROLE, []),
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
            quantity_required=server_quantity,
        )
        if controller_component is None:
            missing_component_roles.append(STORAGE_CONTROLLER_ROLE)
            missing_components.extend(controller_missing)
        else:
            components.append(controller_component)
            matched_requirements.append(_component_match_text(controller_component))
            score += 10

    if _requires_network_adapter(spec, item):
        network_component, network_missing = _select_simple_component(
            role=NETWORK_ADAPTER_ROLE,
            products=products_by_role.get(NETWORK_ADAPTER_ROLE, []),
            facts_by_key=facts_by_key,
            stock_rows_by_key=stock_rows_by_key,
            quantity_required=server_quantity,
        )
        if network_component is None:
            missing_component_roles.append(NETWORK_ADAPTER_ROLE)
            missing_components.extend(network_missing)
        else:
            components.append(network_component)
            matched_requirements.append(_component_match_text(network_component))
            score += 10

    compatibility_warnings.extend(_standard_build_warnings(spec, item))
    total_price_value, total_price_currency, price_warnings = _build_total_price(components)
    compatibility_warnings.extend(price_warnings)
    missing_components.extend(_component_stock_warnings(components))
    missing_requirements = _unique(missing_components)
    compatibility_warnings = _unique(compatibility_warnings)
    risk_flags = _unique([*compatibility_warnings, *missing_requirements])
    component_rows = [component.to_report_json() for component in components]
    platform_row = component_rows[0] if component_rows else {}
    included_component_roles = _unique(component.role for component in components)
    missing_component_roles = _unique(missing_component_roles)
    excluded_from_total_roles = _unique(excluded_from_total_roles)
    completeness_status = (
        "incomplete"
        if missing_requirements or missing_component_roles
        else "complete"
    )
    completeness_label = _build_completeness_label(
        completeness_status,
        missing_component_roles=missing_component_roles,
    )
    total_price_note = _total_price_note(excluded_from_total_roles)
    available_quantity = _build_available_quantity(components)
    platform_name = _component_display_name(components[0])
    item_id = f"build-{item_index + 1}-{platform.item_id}"
    build_title = (
        "Неполная сборка"
        if completeness_status == "incomplete"
        else "Предварительная сборка"
    )
    score = _adjust_build_score(
        score,
        completeness_status=completeness_status,
        missing_component_roles=missing_component_roles,
        compatibility_warnings=compatibility_warnings,
    )
    if completeness_status == "complete":
        rank_reason.append("Все обязательные роли для предварительной сборки подобраны.")
    else:
        rank_reason.append("Сборка неполная; сумма считается без отсутствующих ролей.")

    return MatchCandidateResult(
        distributor_code=platform.distributor_code,
        item_id=item_id,
        product_key=platform.product_key,
        part_number=platform.part_number,
        producer=platform.producer,
        category_id=platform.category_id,
        item_name=f"{build_title} на платформе {platform_name}",
        confidence_score=score,
        price_value=total_price_value,
        price_currency=total_price_currency,
        available_quantity=available_quantity,
        reservable_locations=components[0].reservable_locations,
        matched_requirements=_unique(matched_requirements),
        missing_requirements=missing_requirements,
        risk_flags=risk_flags,
        raw={
            "spec_item_index": item_index,
            "quantity_required": server_quantity,
            "candidate_type": BUILD_CANDIDATE_TYPE,
            "platform": _jsonable(platform_row),
            "components": _jsonable(component_rows),
            "candidate_pool_sizes": {
                role: len(products_by_role.get(role, []))
                for role in (
                    SERVER_PLATFORM_ROLE,
                    CPU_ROLE,
                    RAM_ROLE,
                    SSD_ROLE,
                    HDD_ROLE,
                    STORAGE_CONTROLLER_ROLE,
                    NETWORK_ADAPTER_ROLE,
                )
            },
            "total_price_value": _jsonable(total_price_value),
            "total_price_currency": total_price_currency,
            "missing_components": missing_requirements,
            "compatibility_warnings": compatibility_warnings,
            "engineer_review_required": True,
            "completeness_status": completeness_status,
            "completeness_label": completeness_label,
            "included_component_roles": included_component_roles,
            "missing_component_roles": missing_component_roles,
            "excluded_from_total_roles": excluded_from_total_roles,
            "cpu_per_server": cpu_per_server,
            "total_cpu_required": total_cpu_required,
            "total_price_note": total_price_note,
            "score": score,
            "rank_reason": rank_reason,
        },
        candidate_type=BUILD_CANDIDATE_TYPE,
        components=component_rows,
        total_price_value=total_price_value,
        total_price_currency=total_price_currency,
        missing_components=missing_requirements,
        compatibility_warnings=compatibility_warnings,
        engineer_review_required=True,
        completeness_status=completeness_status,
        completeness_label=completeness_label,
        included_component_roles=included_component_roles,
        missing_component_roles=missing_component_roles,
        excluded_from_total_roles=excluded_from_total_roles,
        cpu_per_server=cpu_per_server,
        total_cpu_required=total_cpu_required,
        total_price_note=total_price_note,
        platform=platform_row,
        score=score,
        rank_reason=rank_reason,
    )


def _products_by_server_role(
    products: list[DistributorProduct],
    *,
    category_plan: Mapping[str, list[str]] | None = None,
    product_group: str = SERVER_PRODUCT_GROUP,
) -> dict[str, list[DistributorProduct]]:
    result: dict[str, list[DistributorProduct]] = {}
    planned_role_by_category_id = _planned_role_by_category_id(category_plan or {})
    if not planned_role_by_category_id:
        return result
    for product in products:
        role = planned_role_by_category_id.get(str(product.category_id or ""))
        if role is None:
            continue
        if not _role_belongs_to_product_group(role, product_group):
            continue
        result.setdefault(role, []).append(product)
    return result


def _role_belongs_to_product_group(role: str, product_group: str) -> bool:
    if product_group == SERVER_PRODUCT_GROUP:
        return role in {
            READY_SERVER_ROLE,
            SERVER_PLATFORM_ROLE,
            CPU_ROLE,
            RAM_ROLE,
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
            UNMAPPED_ROLE,
        }
    if product_group == NETWORK_PRODUCT_GROUP:
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
            UNMAPPED_ROLE,
        }
    if product_group == STORAGE_PRODUCT_GROUP:
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
            UNMAPPED_ROLE,
        }
    return False


def _planned_role_by_category_id(category_plan: Mapping[str, list[str]]) -> dict[str, str]:
    role_by_category_id: dict[str, str] = {}
    for planned_role, category_ids in category_plan.items():
        if not isinstance(category_ids, list):
            continue
        for category_id_value in category_ids:
            category_id = str(category_id_value or "").strip()
            if not category_id:
                continue
            internal_role = _materialization_role_for_planned_category(
                planned_role,
                category_id,
            )
            if internal_role is not None:
                role_by_category_id[category_id] = internal_role
    return role_by_category_id


def _materialization_role_for_planned_category(
    planned_role: str,
    category_id: str,
) -> str | None:
    normalized_role = planned_role.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized_role == "storage":
        category_role = server_category_role(category_id)
        if category_role in {SSD_ROLE, HDD_ROLE, DRIVE_ROLE}:
            return category_role
        return DRIVE_ROLE
    aliases = {
        "platform": SERVER_PLATFORM_ROLE,
        "server_platform": SERVER_PLATFORM_ROLE,
        "storage_array": STORAGE_SYSTEM_ROLE,
        "system": STORAGE_SYSTEM_ROLE,
        "shelf": DISK_SHELF_ROLE,
        "drive_shelf": DISK_SHELF_ROLE,
        "expansion_shelf": DISK_SHELF_ROLE,
        "drives": DRIVE_ROLE,
        "disks": DRIVE_ROLE,
        "host_ports": HOST_PORT_ROLE,
        "ports": HOST_PORT_ROLE,
        "host_interface": HOST_PORT_ROLE,
        "protocol": PROTOCOL_MODULE_ROLE,
        "protocol_adapter": PROTOCOL_MODULE_ROLE,
        "interface_module": PROTOCOL_MODULE_ROLE,
        "power_cable": CABLE_ROLE,
        "power_cord": CABLE_ROLE,
        "c13_c14": CABLE_ROLE,
        "c13_schuko": CABLE_ROLE,
    }
    if normalized_role in aliases:
        return aliases[normalized_role]
    if normalized_role in {
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
        READY_SERVER_ROLE,
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
        return normalized_role
    return None


def _facts_by_product(
    products_by_role: dict[str, list[DistributorProduct]],
) -> dict[tuple[str, str], _ProductFacts]:
    facts: dict[tuple[str, str], _ProductFacts] = {}
    for role, products in products_by_role.items():
        for product in products:
            facts[_product_identity(product)] = _extract_product_facts(product, role)
    return facts


def _facts_for_product(
    product: DistributorProduct,
    *,
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    role: str,
) -> _ProductFacts:
    return facts_by_key.get(_product_identity(product)) or _extract_product_facts(product, role)


def _product_identity(product: DistributorProduct) -> tuple[str, str]:
    return product.distributor_code, product.item_id


def _extract_product_facts(product: DistributorProduct, role: str | None) -> _ProductFacts:
    raw_texts = _raw_characteristic_texts(product.raw_json)
    raw_texts.extend(_raw_characteristic_texts(product.catalog_path_json))
    raw_texts.extend(_raw_characteristic_texts(product.package_json))
    manufacturer_text = " ".join(
        part
        for part in [
            product.producer,
            *_raw_named_values(
                product.raw_json,
                {
                    "producer",
                    "manufacturer",
                    "manufacture",
                    "vendor",
                    "brand",
                    "brandname",
                    "vendorname",
                },
            ),
        ]
        if part
    )
    full_text = " ".join(
        part
        for part in [
            manufacturer_text,
            product.part_number,
            product.category_id,
            product.item_name,
            product.item_name_rus,
            product.product_name,
            product.product_description,
            product.product_notes,
            *raw_texts,
        ]
        if part
    )

    normalized_vendor = _detect_vendor(manufacturer_text) or _detect_vendor(full_text)
    option_kit_vendor = _detect_oem_vendor(full_text) or UNKNOWN_FACT
    cpu_brand = _detect_cpu_brand(full_text)
    cpu_family = _detect_cpu_family(full_text)
    cpu_generation = _detect_cpu_generation(full_text)
    cpu_socket = _detect_cpu_socket(full_text)
    cpu_cores = _detect_cpu_cores(full_text)
    ram_type = _detect_ram_type(full_text)
    storage_drive_roles = {DRIVE_ROLE, SSD_ROLE, HDD_ROLE}
    storage_capacity = (
        _detect_storage_capacity(full_text) if role in storage_drive_roles else UNKNOWN_FACT
    )
    storage_interface = (
        _detect_storage_interface(full_text) if role in storage_drive_roles else UNKNOWN_FACT
    )
    storage_capacity_tb = _capacity_to_tb(storage_capacity)
    storage_product_facts = _extract_storage_product_facts(full_text, role)
    network_facts = extract_network_facts(full_text, product.raw_json)
    network_device_facts = _extract_network_device_facts(full_text)
    is_vendor_option_kit = _is_vendor_option_kit(
        role=role,
        normalized_vendor=normalized_vendor,
        option_kit_vendor=option_kit_vendor,
        cpu_brand=cpu_brand,
        text=full_text,
    )

    return _ProductFacts(
        normalized_vendor=normalized_vendor or UNKNOWN_FACT,
        is_vendor_option_kit=is_vendor_option_kit,
        option_kit_vendor=option_kit_vendor,
        cpu_brand=cpu_brand,
        cpu_family=cpu_family,
        cpu_generation=cpu_generation,
        cpu_socket=cpu_socket,
        cpu_cores=cpu_cores,
        ram_capacity_gb=parse_memory_module_gb(full_text, product.raw_json)
        if role == RAM_ROLE
        else None,
        ram_type=ram_type,
        storage_capacity=storage_capacity,
        storage_capacity_tb=storage_capacity_tb,
        storage_interface=storage_interface,
        nvme_support=_detect_nvme_support(full_text),
        raw_capacity_tb=_float_value(storage_product_facts.get("raw_capacity_tb")),
        usable_capacity_tb=_float_value(storage_product_facts.get("usable_capacity_tb")),
        redundancy_level=str(storage_product_facts.get("redundancy_level") or UNKNOWN_FACT),
        controller_count=_as_int(storage_product_facts.get("controller_count")),
        drive_count=_as_int(storage_product_facts.get("drive_count")),
        drive_capacity_tb=_float_value(storage_product_facts.get("drive_capacity_tb"))
        or storage_capacity_tb,
        drive_type=str(storage_product_facts.get("drive_type") or UNKNOWN_FACT),
        drive_interface=str(storage_product_facts.get("drive_interface") or storage_interface),
        host_protocol=str(storage_product_facts.get("host_protocol") or UNKNOWN_FACT),
        host_port_count=_as_int(storage_product_facts.get("host_port_count")),
        host_port_speed=str(storage_product_facts.get("host_port_speed") or UNKNOWN_FACT),
        host_port_speed_gbps=_as_int(storage_product_facts.get("host_port_speed_gbps")),
        host_port_media=str(storage_product_facts.get("host_port_media") or UNKNOWN_FACT),
        warranty_months=_as_int(storage_product_facts.get("warranty_months")),
        network_ports_count=_as_int(network_facts.get("ports_count")),
        network_speed=str(network_facts.get("speed") or UNKNOWN_FACT),
        network_speed_gbps=_as_int(network_facts.get("speed_gbps")),
        network_media=str(network_facts.get("media") or UNKNOWN_FACT),
        network_interface=str(network_facts.get("interface") or UNKNOWN_FACT),
        port_count=_as_int(network_device_facts.get("port_count")),
        port_speed=str(network_device_facts.get("port_speed") or UNKNOWN_FACT),
        port_speed_gbps=_float_value(network_device_facts.get("port_speed_gbps")),
        port_media=str(network_device_facts.get("port_media") or UNKNOWN_FACT),
        uplink_count=_as_int(network_device_facts.get("uplink_count")),
        uplink_speed=str(network_device_facts.get("uplink_speed") or UNKNOWN_FACT),
        uplink_speed_gbps=_float_value(network_device_facts.get("uplink_speed_gbps")),
        uplink_media=str(network_device_facts.get("uplink_media") or UNKNOWN_FACT),
        poe_supported=network_device_facts.get("poe_supported"),
        poe_budget_w=_as_int(network_device_facts.get("poe_budget_w")),
        poe_standard=str(network_device_facts.get("poe_standard") or UNKNOWN_FACT),
        l2_supported=network_device_facts.get("l2_supported"),
        l3_supported=network_device_facts.get("l3_supported"),
        stacking_supported=network_device_facts.get("stacking_supported"),
        managed_status=str(network_device_facts.get("managed_status") or UNKNOWN_FACT),
        airflow=str(network_device_facts.get("airflow") or UNKNOWN_FACT),
        redundant_psu=network_device_facts.get("redundant_psu"),
        transceiver_form_factor=str(
            network_device_facts.get("transceiver_form_factor") or UNKNOWN_FACT
        ),
        form_factor_hints=_detect_form_factor_hints(full_text),
    )


def _extract_network_device_facts(text: str) -> dict[str, Any]:
    port_count, port_segment = _network_access_port_segment(text)
    uplink_count, uplink_segment = _network_uplink_segment(text)
    port_speed_gbps = _speed_gbps_from_text(port_segment)
    if port_speed_gbps is None and port_count is None:
        port_speed_gbps = _speed_gbps_from_text(text)
    uplink_speed_gbps = _speed_gbps_from_text(uplink_segment)
    transceiver_form_factor = _media_from_text(text)
    managed_status = _managed_status_from_text(text)
    unmanaged = managed_status == "unmanaged"
    return {
        "port_count": port_count,
        "port_speed": _speed_label_from_gbps(port_speed_gbps),
        "port_speed_gbps": port_speed_gbps,
        "port_media": _access_media_from_text(port_segment) or UNKNOWN_FACT,
        "uplink_count": uplink_count,
        "uplink_speed": _speed_label_from_gbps(uplink_speed_gbps),
        "uplink_speed_gbps": uplink_speed_gbps,
        "uplink_media": _media_from_text(uplink_segment) or UNKNOWN_FACT,
        "poe_supported": True
        if re.search(r"\bPoE(?:\+\+|\+)?(?=\W|$)|802\.3(?:af|at|bt)", text, re.I)
        else None,
        "poe_budget_w": _poe_budget_from_text(text),
        "poe_standard": _poe_standard_from_text(text) or UNKNOWN_FACT,
        "l2_supported": True if re.search(r"\bL2\b|layer\s*2", text, re.I) else None,
        "l3_supported": (
            False
            if unmanaged
            else True if re.search(r"\bL3\b|layer\s*3|routing", text, re.I) else None
        ),
        "stacking_supported": True
        if re.search(r"\b(?:stacking|stackable|stack)\b|стек|стекир", text, re.I)
        else False if unmanaged else None,
        "managed_status": managed_status,
        "airflow": _airflow_from_text(text) or UNKNOWN_FACT,
        "redundant_psu": True
        if re.search(
            r"\b(?:redundant|1\+1|dual)\s+(?:psu|power\s+supply)\b|резерв\w+\s+бп",
            text,
            re.I,
        )
        else None,
        "transceiver_form_factor": transceiver_form_factor or UNKNOWN_FACT,
    }


def _extract_storage_product_facts(text: str, role: str | None) -> dict[str, Any]:
    drive_type = _normalize_storage_type(text)
    drive_interface = _normalize_storage_interface(text)
    host_protocol = _normalize_storage_protocol(text)
    host_speed_gbps = _network_speed_requirement_gbps(text)
    drive_capacity_tb = _capacity_to_tb(_detect_storage_capacity(text))
    if role == STORAGE_SYSTEM_ROLE:
        raw_capacity_tb = _capacity_after_marker(text, ("raw", "сыр", "raw capacity"))
        usable_capacity_tb = _capacity_after_marker(
            text,
            ("usable", "полезн", "effective"),
        )
    else:
        raw_capacity_tb = None
        usable_capacity_tb = None
    if raw_capacity_tb is None and role == STORAGE_SYSTEM_ROLE:
        raw_capacity_tb = _capacity_after_marker(text, ("capacity", "емкост"))
    return {
        "raw_capacity_tb": raw_capacity_tb,
        "usable_capacity_tb": usable_capacity_tb,
        "redundancy_level": _redundancy_level_from_text(text),
        "controller_count": _controller_count_from_text(text),
        "drive_count": _drive_count_from_text(text),
        "drive_capacity_tb": drive_capacity_tb,
        "drive_type": drive_type,
        "drive_interface": drive_interface,
        "host_protocol": host_protocol,
        "host_port_count": _host_port_count_from_text(text),
        "host_port_speed": f"{host_speed_gbps}G" if host_speed_gbps is not None else UNKNOWN_FACT,
        "host_port_speed_gbps": host_speed_gbps,
        "host_port_media": _media_from_text(text) or UNKNOWN_FACT,
        "warranty_months": _warranty_months_from_text(text),
    }


def _capacity_after_marker(text: str, markers: Sequence[str]) -> float | None:
    for marker in markers:
        pattern = (
            rf"{re.escape(marker)}[^\d]{{0,30}}(\d+(?:[.,]\d+)?)\s*"
            r"(tb|тб|gb|гб|pb|пб)?"
        )
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1).replace(",", "."))
            unit = (match.group(2) or "tb").casefold()
            if unit in {"gb", "гб"}:
                return amount / 1024
            if unit in {"pb", "пб"}:
                return amount * 1024
            return amount
    return None


def _redundancy_level_from_text(text: str) -> str:
    match = re.search(r"\b(?:raid|erasure)\s*[- ]?\s*([05610]+)\b", text, re.IGNORECASE)
    if match:
        return f"RAID {match.group(1)}"
    return UNKNOWN_FACT


def _controller_count_from_text(text: str) -> int | None:
    for pattern in (
        r"(\d{1,2})\s*(?:controllers?|контроллер)",
        r"(?:controllers?|контроллер)[^\d]{0,20}(\d{1,2})",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 16:
                return value
    if re.search(r"\bdual\s+controller\b|два\s+контроллер", text, re.IGNORECASE):
        return 2
    return None


def _drive_count_from_text(text: str) -> int | None:
    for pattern in (
        r"(\d{1,3})\s*(?:x|×)\s*\d+(?:[.,]\d+)?\s*(?:tb|тб)",
        r"(\d{1,3})\s*(?:drives?|disks?|ssd|hdd|диск|накопител)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 1000:
                return value
    return None


def _host_port_count_from_text(text: str) -> int | None:
    for pattern in (
        r"(\d{1,3})\s*(?:x|×)?\s*(?:fc|iscsi|sas|nvme-?of)\s*(?:ports?|порт)",
        r"(\d{1,3})\s*(?:x|×)\s*(?:16|32|64|10|25|100)\s*g",
        r"(?:ports?|порт)[^\d]{0,20}(\d{1,3})",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 512:
                return value
    return None


def _warranty_months_from_text(text: str) -> int | None:
    match = re.search(r"(\d{1,2})\s*(?:years?|года|лет)", text, re.IGNORECASE)
    if match:
        years = int(match.group(1))
        if 1 <= years <= 10:
            return years * 12
    match = re.search(r"(\d{1,3})\s*(?:months?|мес)", text, re.IGNORECASE)
    if match:
        months = int(match.group(1))
        if 1 <= months <= 120:
            return months
    return None


_NETWORK_PORT_MULTIPLIER_RE = r"[xх×*]"
_NETWORK_BASE_RATE_RE = r"(?:1000|100|(?:1|2\.5|5|10|25|40|100)\s*g?)"
_NETWORK_G_RATE_RE = r"(?:1|2\.5|5|10|25|40|100)"
_NETWORK_UPLINK_MEDIA_RE = r"(?:sfp\+?|sfp28|qsfp\+?|qsfp28)"
_NETWORK_SPEED_PREFIX_RE = r"(?:\b|(?<=[xх×*]))"


def _network_access_port_segment(text: str) -> tuple[int | None, str]:
    for pattern in (
        r"(\d{1,3})\s*[- ]?\s*ports?\b[^\n,;]{0,80}",
        rf"(\d{{1,3}})\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*"
        rf"{_NETWORK_BASE_RATE_RE}\s*base\s*-?\s*t[x]?\b[^\n,;]{{0,50}}",
        rf"(\d{{1,3}})\s*(?:{_NETWORK_PORT_MULTIPLIER_RE})?\s*"
        r"(?:порт\w*|ports?)(?!\s*(?:uplink|аплинк))[^\n,;]{0,80}",
        rf"(\d{{1,3}})\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*"
        rf"{_NETWORK_G_RATE_RE}\s*g(?:b(?:it)?(?:e|/s|ps)?)?"
        r"(?!\s*(?:base\s*-?\s*x|sfp|qsfp|uplink|аплинк))[^\n,;]{0,50}",
        rf"(\d{{1,3}})\s*{_NETWORK_PORT_MULTIPLIER_RE}\s*"
        r"(?:gigabit|гигабит)[^\n,;]{0,50}",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 512:
                return value, match.group(0)
    return None, ""


def _network_uplink_segment(text: str) -> tuple[int | None, str]:
    for pattern in (
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
    ):
        match = re.search(pattern, text, re.I)
        if match:
            value = int(match.group(1))
            if 1 <= value <= 128:
                return value, match.group(0)
    return None, ""


def _speed_gbps_from_text(text: str) -> float | None:
    if re.search(
        rf"{_NETWORK_SPEED_PREFIX_RE}100\s*base\s*-?\s*t[x]?\b"
        r"|\bfast\s+ethernet\b",
        text,
        re.I,
    ):
        return 0.1
    if re.search(
        rf"{_NETWORK_SPEED_PREFIX_RE}1000\s*base\s*-?\s*t[x]?\b"
        rf"|{_NETWORK_SPEED_PREFIX_RE}(?:gigabit|гигабит)",
        text,
        re.I,
    ):
        return 1
    match = re.search(
        rf"{_NETWORK_SPEED_PREFIX_RE}(1|2\.5|5|10|25|40|100|200|400)\s*g",
        text,
        re.I,
    )
    if match:
        return float(match.group(1))
    if re.search(r"\bsfp\s*\+(?=\W|$)", text, re.I):
        return 10
    if re.search(r"\bsfp\s*28\b|\bsfp28\b", text, re.I):
        return 25
    if re.search(r"\bqsfp\s*28\b|\bqsfp28\b", text, re.I):
        return 100
    return None


def _speed_label_from_gbps(value: Any) -> str:
    speed = _float_value(value)
    if speed is None:
        return UNKNOWN_FACT
    if abs(speed - 0.1) < 0.001:
        return "100MbE"
    if float(speed).is_integer():
        return f"{int(speed)}GbE"
    return f"{speed:g}GbE"


def _managed_status_from_text(text: str) -> str:
    if re.search(r"\bunmanaged\b|неуправляем", text, re.I):
        return "unmanaged"
    if re.search(r"\b(?:managed|smart|web[- ]?smart)\b|управляем", text, re.I):
        return "managed"
    return UNKNOWN_FACT


def _media_from_text(text: str) -> str | None:
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


def _access_media_from_text(text: str) -> str | None:
    if re.search(r"\bRJ\s*-?45\b|BASE\s*-?T(?:X)?\b|PoE", text, re.I):
        return "RJ45"
    return _media_from_text(text)


def _poe_budget_from_text(text: str) -> int | None:
    for pattern in (
        r"\bPoE[^\n,;]{0,40}?(?<!\d)(\d{2,5})\s*(?:W|Вт)\b",
        r"(?<!\d)(\d{2,5})\s*(?:W|Вт)\b",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            value = int(match.group(1))
            return value if 1 <= value <= 100000 else None
    return None


def _poe_standard_from_text(text: str) -> str | None:
    if re.search(r"\bPoE\+\+(?=\W|$)|802\.3bt", text, re.I):
        return "PoE++"
    if re.search(r"\bPoE\+(?=\W|$)|802\.3at", text, re.I):
        return "PoE+"
    if re.search(r"\bPoE\b|802\.3af", text, re.I):
        return "PoE"
    return None


def _airflow_from_text(text: str) -> str | None:
    lowered = text.casefold()
    if re.search(r"front\s*[- ]?to\s*[- ]?back", lowered):
        return "front-to-back"
    if re.search(r"back\s*[- ]?to\s*[- ]?front", lowered):
        return "back-to-front"
    if "port-side intake" in lowered:
        return "port-side-intake"
    if "port-side exhaust" in lowered:
        return "port-side-exhaust"
    return None


def _cpu_compatibility_decision(
    platform_facts: _ProductFacts,
    cpu_facts: _ProductFacts,
) -> dict[str, object]:
    if _fatal_platform_cpu_mismatch(platform_facts, cpu_facts):
        return {
            "allowed": False,
            "score": 0,
            "warnings": [],
            "rank_reason": [],
        }

    platform_vendor = platform_facts.normalized_vendor

    if cpu_facts.is_vendor_option_kit:
        cpu_vendor = cpu_facts.option_kit_vendor
        if (
            cpu_vendor != UNKNOWN_FACT
            and platform_vendor != UNKNOWN_FACT
            and cpu_vendor == platform_vendor
        ):
            return {
                "allowed": True,
                "score": 80,
                "warnings": [
            "Проверить список поддерживаемых CPU для комплекта CPU и платформы "
            "того же производителя."
                ],
                "rank_reason": [
                    "Vendor-specific CPU kit оставлен только потому, что vendor платформы совпал."
                ],
            }
        return {
            "allowed": False,
            "score": 0,
            "warnings": [],
            "rank_reason": [],
        }

    if cpu_facts.cpu_brand in {"Intel", "AMD"} or cpu_facts.cpu_family in {"Xeon", "EPYC"}:
        return {
            "allowed": True,
            "score": 65,
            "warnings": [
                "Проверить совместимость CPU с платформой по списку поддерживаемых CPU."
            ],
            "rank_reason": ["Bare Intel/AMD CPU рассмотрен как preliminary-вариант."],
        }

    return {
        "allowed": True,
        "score": 45,
        "warnings": [
            "Проверить совместимость CPU с платформой по списку поддерживаемых CPU."
        ],
        "rank_reason": [
            "CPU без надежно извлеченного vendor/семейства рассмотрен "
            "только как preliminary-вариант."
        ],
    }


def _build_candidate_sort_key(
    candidate: MatchCandidateResult,
) -> tuple[Any, ...]:
    return (
        0 if candidate.completeness_status == "complete" else 1,
        _fatal_warning_count(candidate.compatibility_warnings),
        len(candidate.missing_component_roles),
        *_build_total_price_sort_key(candidate.total_price_value),
        -(candidate.score or candidate.confidence_score),
        _stable_text(candidate.candidate_id or candidate.item_id),
    )


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


def _fatal_warning_text(values: list[str]) -> str | None:
    for value in values:
        text = str(value or "").strip()
        lowered = text.casefold()
        if (
            "fatal" in lowered
            or "incompat" in lowered
            or "mismatch" in lowered
            or "несовмест" in lowered
            or "РЅРµСЃРѕРІРјРµСЃС‚" in lowered
        ):
            return text
    return None


def _build_total_price_sort_key(value: Decimal | None) -> tuple[int, Decimal]:
    if value is None:
        return (1, Decimal("Infinity"))
    return (0, value)


def _dedupe_build_candidates_by_platform(
    candidates: list[MatchCandidateResult],
) -> list[MatchCandidateResult]:
    seen: set[str] = set()
    result: list[MatchCandidateResult] = []
    for candidate in candidates:
        key = _platform_dedupe_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _platform_dedupe_key(candidate: MatchCandidateResult) -> str:
    platform = candidate.platform if isinstance(candidate.platform, dict) else {}
    part_number = str(platform.get("part_number") or candidate.part_number or "").strip()
    producer = str(platform.get("producer") or candidate.producer or "").strip()
    if part_number:
        return f"{producer.casefold()}::{part_number.casefold()}"
    return candidate.item_id


def _adjust_build_score(
    score: int,
    *,
    completeness_status: str,
    missing_component_roles: list[str],
    compatibility_warnings: list[str],
) -> int:
    adjusted = score
    if completeness_status == "complete":
        adjusted += 10
    adjusted -= len(missing_component_roles) * 12
    adjusted -= min(len(compatibility_warnings), 12)
    return max(1, min(adjusted, 90))


def _rank_component_products(
    products: list[DistributorProduct],
    *,
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> list[DistributorProduct]:
    return sorted(
        products,
        key=lambda product: (
            _available_quantity(
                stock_rows_by_key.get((product.distributor_code, product.item_id), [])
            )
            or 0,
            product.synced_at,
            product.id,
        ),
        reverse=True,
    )


def _stock_fit_rank(
    stock_rows: list[DistributorStockPrice],
    *,
    quantity_required: int,
) -> int:
    available_quantity = _available_quantity(stock_rows)
    if available_quantity is None:
        return 0
    if available_quantity >= quantity_required:
        return 2
    return 1


def _component_stock_fit_score(component: _BuildComponent) -> int:
    if component.available_quantity is None:
        return 0
    if component.available_quantity >= component.quantity_required:
        return 10
    return 2


def _lower_price_rank(stock_rows: list[DistributorStockPrice]) -> float:
    price_value, _ = _select_price(stock_rows)
    if price_value is None:
        return -1_000_000_000.0
    return -float(price_value)


def _component_lower_price_rank(component: _BuildComponent) -> float:
    if component.price_value is None:
        return -1_000_000_000.0
    return -float(component.price_value)


def _candidate_stock_fit_score(candidate: _ComponentCandidate) -> int:
    if candidate.available_quantity is None:
        return 0
    if candidate.available_quantity >= candidate.quantity_required:
        return 10
    return 2


def _candidate_lower_price_rank(candidate: _ComponentCandidate) -> float:
    if candidate.price_value is None:
        return -1_000_000_000.0
    return -float(candidate.price_value)


def _candidate_total_lower_price_rank(candidate: _ComponentCandidate) -> float:
    if candidate.price_value is None:
        return -1_000_000_000.0
    return -float(candidate.price_value * candidate.quantity_required)


def _fit_label_rank(fit_label: str) -> int:
    ranks = {
        FIT_EXACT_OR_CLOSE: 3,
        FIT_ACCEPTABLE_OVERFIT: 2,
        FIT_EXCESSIVE_OVERFIT: 1,
        FIT_UNKNOWN: 0,
    }
    return ranks.get(fit_label, 0)


def _cpu_fit_fields(
    cpu_cores: int,
    required_cores: int,
) -> tuple[str, str, int, int]:
    over_requirement = max(cpu_cores - required_cores, 0)
    ratio = cpu_cores / required_cores if required_cores else 1
    if ratio <= 1.5:
        return (
            FIT_EXACT_OR_CLOSE,
            (
                f"CPU близок к минимальному требованию: {_cores_text(cpu_cores)} "
                f"при минимуме {_cores_text(required_cores)}."
            ),
            over_requirement,
            20,
        )
    if ratio <= 2:
        return (
            FIT_ACCEPTABLE_OVERFIT,
            (
                f"CPU выше минимального требования: {_cores_text(cpu_cores)} вместо "
                f"{_cores_text(required_cores)}. "
                "альтернативы CPU нужно сверить в матрице компонентов."
            ),
            over_requirement,
            5,
        )
    return (
        FIT_EXCESSIVE_OVERFIT,
        (
            f"CPU существенно выше требования: {_cores_text(cpu_cores)} вместо "
            f"{_cores_text(required_cores)}. Требуется проверить альтернативы "
            "в матрице компонентов."
        ),
        over_requirement,
        -18,
    )


def _ram_fit_fields(
    *,
    module_gb: int,
    required_gb: int,
    modules_per_server: int,
) -> tuple[str, str, int, int]:
    total_gb = module_gb * modules_per_server
    over_requirement = max(total_gb - required_gb, 0)
    if over_requirement <= max(32, int(required_gb * 0.10)):
        return (
            FIT_EXACT_OR_CLOSE,
            (
                f"RAM закрывает {required_gb} ГБ на сервер ближайшей комбинацией: "
                f"{modules_per_server} x {module_gb} ГБ."
            ),
            over_requirement,
            20,
        )
    if over_requirement <= max(128, int(required_gb * 0.50)):
        return (
            FIT_ACCEPTABLE_OVERFIT,
            (
                f"RAM дает {total_gb} ГБ на сервер при требовании {required_gb} ГБ; "
                "избыточность умеренная."
            ),
            over_requirement,
            8,
        )
    return (
        FIT_EXCESSIVE_OVERFIT,
        (
            f"RAM дает {total_gb} ГБ на сервер при требовании {required_gb} ГБ; "
            "избыточность существенная."
        ),
        over_requirement,
        -14,
    )


def _storage_fit_fields(
    *,
    capacity_tb: float,
    required_tb: float,
) -> tuple[str, str, float, int]:
    ratio = capacity_tb / required_tb if required_tb else 1.0
    rounded_ratio = round(ratio, 2)
    if abs(capacity_tb - required_tb) <= 0.001:
        return (
            FIT_EXACT_OR_CLOSE,
            f"SSD соответствует минимальному требованию {_format_number(required_tb)} ТБ.",
            rounded_ratio,
            22,
        )
    if ratio <= 1.25:
        return (
            FIT_EXACT_OR_CLOSE,
            (
                "Накопитель близок к минимуму: "
                f"{_format_number(capacity_tb)} ТБ при требовании "
                f"{_format_number(required_tb)} ТБ."
            ),
            rounded_ratio,
            22,
        )
    if ratio <= 2.0:
        return (
            FIT_ACCEPTABLE_OVERFIT,
            (
                "Накопитель выше минимума, но в допустимом диапазоне: "
                f"{_format_number(capacity_tb)} ТБ против "
                f"{_format_number(required_tb)} ТБ."
            ),
            rounded_ratio,
            8,
        )
    return (
        FIT_EXCESSIVE_OVERFIT,
        (
            "Накопитель существенно выше требования: "
            f"{_format_number(capacity_tb)} ТБ против "
            f"{_format_number(required_tb)} ТБ."
        ),
        rounded_ratio,
        -22,
    )


def _build_overfit_penalty(
    *,
    requirements: _NormalizedServerRequirements,
    cpu_candidate: _ComponentCandidate | None,
    ram_candidate: _ComponentCandidate | None,
    storage_candidate: _ComponentCandidate | None,
) -> int:
    if requirements.optimization_mode != OPTIMIZATION_MODE_COST_MINIMAL_FIT:
        return 0
    penalty = 0
    for candidate in (cpu_candidate, ram_candidate, storage_candidate):
        if candidate is None:
            continue
        if candidate.fit_label == FIT_ACCEPTABLE_OVERFIT:
            penalty += 3
        elif candidate.fit_label == FIT_EXCESSIVE_OVERFIT:
            penalty += 12
        elif candidate.fit_label == FIT_UNKNOWN and candidate.role in RIGHT_SIZE_COMPONENT_ROLES:
            penalty += 4
    return penalty


def _build_right_size_summary(
    *,
    requirements: _NormalizedServerRequirements,
    cpu_candidate: _ComponentCandidate | None,
    ram_candidate: _ComponentCandidate | None,
    storage_candidate: _ComponentCandidate | None,
) -> dict[str, Any]:
    selected = [
        candidate
        for candidate in (cpu_candidate, ram_candidate, storage_candidate)
        if candidate is not None
    ]
    overfit_reasons = [
        candidate.fit_reason
        for candidate in selected
        if candidate.fit_label in {FIT_ACCEPTABLE_OVERFIT, FIT_EXCESSIVE_OVERFIT}
        and candidate.fit_reason
    ]
    if overfit_reasons:
        overfit_reason = "; ".join(overfit_reasons)
        right_size_note = f"Подбор: {overfit_reason}"
        requirement_fit = "overfit_with_reason"
    else:
        overfit_reason = None
        right_size_note = "Подбор: минимально подходящий по требованиям"
        requirement_fit = "minimal_fit"

    return {
        "optimization_mode": requirements.optimization_mode,
        "requirement_fit": requirement_fit,
        "right_size_note": right_size_note,
        "cpu_over_requirement": (
            cpu_candidate.cpu_over_requirement if cpu_candidate is not None else None
        ),
        "storage_over_requirement": (
            storage_candidate.storage_over_requirement
            if storage_candidate is not None
            else None
        ),
        "ram_overage_gb": (
            ram_candidate.ram_over_requirement_gb if ram_candidate is not None else None
        ),
        "overfit_reason": overfit_reason,
    }


def _component_over_requirement_value(candidate: _ComponentCandidate) -> Any:
    if candidate.role == CPU_ROLE:
        return candidate.cpu_over_requirement
    if candidate.role in {SSD_ROLE, HDD_ROLE}:
        return candidate.storage_over_requirement
    if candidate.role == RAM_ROLE:
        return candidate.ram_over_requirement_gb
    return None


def _cores_text(value: int) -> str:
    if value % 10 == 1 and value % 100 != 11:
        word = "ядро"
    elif value % 10 in {2, 3, 4} and value % 100 not in {12, 13, 14}:
        word = "ядра"
    else:
        word = "ядер"
    return f"{value} {word}"


def _build_component_from_candidate(candidate: _ComponentCandidate) -> _BuildComponent:
    return _BuildComponent(
        role=candidate.role,
        role_ru=_role_ru(candidate.role),
        product=candidate.product,
        quantity_required=candidate.quantity_required,
        available_quantity=candidate.available_quantity,
        reservable_locations=candidate.reservable_locations,
        price_value=candidate.price_value,
        price_currency=candidate.price_currency,
        facts=candidate.facts,
        component_candidate_id=candidate.candidate_id,
        fit_label=candidate.fit_label,
        fit_reason=candidate.fit_reason,
        cpu_over_requirement=candidate.cpu_over_requirement,
        storage_over_requirement=candidate.storage_over_requirement,
        ram_over_requirement_gb=candidate.ram_over_requirement_gb,
    )


def _component_from_product(
    role: str,
    product: DistributorProduct,
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
    *,
    quantity_required: int,
    facts: _ProductFacts | None = None,
) -> _BuildComponent:
    stock_rows = stock_rows_by_key.get((product.distributor_code, product.item_id), [])
    price_value, price_currency = _select_price(stock_rows)
    return _BuildComponent(
        role=role,
        role_ru=_role_ru(role),
        product=product,
        quantity_required=quantity_required,
        available_quantity=_available_quantity(stock_rows),
        reservable_locations=_reservable_locations(stock_rows),
        price_value=price_value,
        price_currency=price_currency,
        facts=facts,
        component_candidate_id=_stable_candidate_id(role, product),
    )


def _select_ram_component(
    *,
    spec: StockSpec,
    item: StockSpecItem,
    products: list[DistributorProduct],
    platform_facts: _ProductFacts,
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
) -> tuple[_BuildComponent | None, list[str], list[str]]:
    ram_min_gb = _as_int(_requirement(spec, item, "ram", "min_gb"))
    if ram_min_gb is None:
        return None, [], []

    if not products:
        return (
            None,
            ["RAM: в локальной OCS DB нет товаров категории серверной памяти."],
            [],
        )

    parsed: list[tuple[DistributorProduct, int, _ProductFacts]] = []
    excluded_by_type = 0
    unknown_type_seen = False
    for product in products:
        facts = _facts_for_product(product, facts_by_key=facts_by_key, role=RAM_ROLE)
        module_gb = facts.ram_capacity_gb
        if module_gb is not None:
            if (
                platform_facts.ram_type != UNKNOWN_FACT
                and facts.ram_type != UNKNOWN_FACT
                and platform_facts.ram_type != facts.ram_type
            ):
                excluded_by_type += 1
                continue
            if platform_facts.ram_type == UNKNOWN_FACT or facts.ram_type == UNKNOWN_FACT:
                unknown_type_seen = True
            parsed.append((product, module_gb, facts))

    if not parsed:
        if excluded_by_type:
            return (
                None,
                ["RAM: не подобраны совместимые по DDR-поколению модули."],
                [
                    "Несовместимые по DDR-поколению модули RAM скрыты из сборки; "
                    "требуется проверить тип памяти платформы."
                ],
            )
        return (
            None,
            ["RAM: не удалось надежно извлечь объем модуля из названий или свойств OCS."],
            ["Характеристики RAM в OCS неполные; требуется ручная проверка объема модулей."],
        )

    parsed.sort(
        key=lambda row: (
            _stock_fit_rank(
                stock_rows_by_key.get((row[0].distributor_code, row[0].item_id), []),
                quantity_required=max(ceil(ram_min_gb / row[1]), 1) * item.quantity,
            ),
            1 if row[2].ram_type != UNKNOWN_FACT and platform_facts.ram_type != UNKNOWN_FACT else 0,
            row[1],
            _available_quantity(
                stock_rows_by_key.get((row[0].distributor_code, row[0].item_id), [])
            )
            or 0,
            _lower_price_rank(
                stock_rows_by_key.get((row[0].distributor_code, row[0].item_id), [])
            ),
        ),
        reverse=True,
    )
    product, module_gb, facts = parsed[0]
    modules_per_server = max(ceil(ram_min_gb / module_gb), 1)
    quantity_required = modules_per_server * item.quantity
    component = _component_from_product(
        RAM_ROLE,
        product,
        stock_rows_by_key,
        quantity_required=quantity_required,
        facts=facts,
    )

    missing: list[str] = []
    warnings: list[str] = []
    if component.available_quantity is None:
        missing.append(
            f"RAM: остаток не найден, требуется {quantity_required} модулей по {module_gb} ГБ."
        )
    elif component.available_quantity < quantity_required:
        missing.append(
            "RAM: недостаточный остаток, "
            f"доступно {component.available_quantity} модулей, "
            f"требуется {quantity_required} модулей по {module_gb} ГБ."
        )
    if unknown_type_seen:
        warnings.append("Проверить тип и совместимость RAM с платформой.")

    return component, missing, warnings


def _select_cpu_component(
    *,
    platform_facts: _ProductFacts,
    products: list[DistributorProduct],
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
    quantity_required: int,
) -> tuple[_BuildComponent | None, list[str], list[str], list[str]]:
    if not products:
        return (
            None,
            ["CPU: в локальной OCS DB нет товаров категории серверных процессоров."],
            [],
            ["CPU не найден в candidate pool серверных процессоров."],
        )

    compatible: list[tuple[int, _BuildComponent, list[str], list[str]]] = []
    for product in _rank_component_products(products, stock_rows_by_key=stock_rows_by_key):
        facts = _facts_for_product(product, facts_by_key=facts_by_key, role=CPU_ROLE)
        decision = _cpu_compatibility_decision(platform_facts, facts)
        if not decision["allowed"]:
            continue

        component = _component_from_product(
            CPU_ROLE,
            product,
            stock_rows_by_key,
            quantity_required=quantity_required,
            facts=facts,
        )
        score = int(decision["score"])
        score += _component_stock_fit_score(component)
        if facts.cpu_family != UNKNOWN_FACT:
            score += 3
        if (
            facts.option_kit_vendor != UNKNOWN_FACT
            and facts.option_kit_vendor == platform_facts.normalized_vendor
        ):
            score += 5
        compatible.append(
            (
                score,
                component,
                list(decision["warnings"]),
                list(decision["rank_reason"]),
            )
        )

    if not compatible:
        return (
            None,
            ["CPU: не подобраны после проверки совместимости с платформой."],
            [],
            ["CPU candidate pool просмотрен, совместимых CPU для платформы не найдено."],
        )

    compatible.sort(
        key=lambda row: (
            row[0],
            _component_stock_fit_score(row[1]),
            row[1].available_quantity or 0,
            row[1].reservable_locations,
            _component_lower_price_rank(row[1]),
        ),
        reverse=True,
    )
    _, component, warnings, rank_reason = compatible[0]

    missing: list[str] = []
    if component.available_quantity is None:
        missing.append(f"CPU: остаток не найден, требуется {quantity_required} шт.")
    elif component.available_quantity < quantity_required:
        missing.append(
            f"CPU: недостаточный остаток, доступно {component.available_quantity} шт., "
            f"требуется {quantity_required} шт."
        )

    return component, missing, warnings, rank_reason


def _select_storage_component(
    *,
    role: str,
    products: list[DistributorProduct],
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
    quantity_required: int,
) -> tuple[_BuildComponent | None, list[str], list[str]]:
    component, missing = _select_simple_component(
        role=role,
        products=products,
        facts_by_key=facts_by_key,
        stock_rows_by_key=stock_rows_by_key,
        quantity_required=quantity_required,
    )
    warnings: list[str] = []
    if component is not None and (
        component.facts is None or component.facts.storage_interface == UNKNOWN_FACT
    ):
        warnings.append(
            "Проверить интерфейс и совместимость накопителей с платформой/backplane."
        )
    return component, missing, warnings


def _select_simple_component(
    *,
    role: str,
    products: list[DistributorProduct],
    facts_by_key: dict[tuple[str, str], _ProductFacts],
    stock_rows_by_key: dict[tuple[str, str], list[DistributorStockPrice]],
    quantity_required: int,
) -> tuple[_BuildComponent | None, list[str]]:
    role_name = _role_ru(role)
    if not products:
        return None, [f"{role_name}: в локальной OCS DB нет товаров нужной категории."]

    product = _rank_component_products(products, stock_rows_by_key=stock_rows_by_key)[0]
    component = _component_from_product(
        role,
        product,
        stock_rows_by_key,
        quantity_required=quantity_required,
        facts=_facts_for_product(product, facts_by_key=facts_by_key, role=role),
    )

    missing: list[str] = []
    if component.available_quantity is None:
        missing.append(f"{role_name}: остаток не найден, требуется {quantity_required} шт.")
    elif component.available_quantity < quantity_required:
        missing.append(
            f"{role_name}: недостаточный остаток, доступно {component.available_quantity} шт., "
            f"требуется {quantity_required} шт."
        )

    return component, missing


def _cpu_per_server(spec: StockSpec, item: StockSpecItem) -> int | None:
    cpu_requirement = _requirement(spec, item, "cpu")
    if isinstance(cpu_requirement, dict):
        for key in ("per_server", "count", "quantity", "processors", "sockets"):
            value = _as_int(cpu_requirement.get(key))
            if value is not None and value > 0:
                return value

    direct_value = _as_int(cpu_requirement)
    if direct_value is not None and direct_value > 0:
        return direct_value

    text = _request_text(spec, item)
    match = re.search(
        r"\b(\d+)\s*(?:cpu|processors?|процессор(?:а|ов)?|проца|сокет(?:а|ов)?)\b",
        text,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1))
    return None


def _build_completeness_label(
    completeness_status: str,
    *,
    missing_component_roles: list[str],
) -> str:
    if completeness_status == "complete":
        return "Предварительная сборка"
    if CPU_ROLE in missing_component_roles:
        return "Неполная сборка - требуется подбор CPU."
    return "Неполная сборка"


def _total_price_note(excluded_from_total_roles: list[str]) -> str | None:
    if not excluded_from_total_roles:
        return None
    return "без " + ", ".join(
        _excluded_from_total_role_label(role)
        for role in excluded_from_total_roles
    )


def _excluded_from_total_role_label(role: str) -> str:
    labels = {
        CPU_ROLE: "CPU",
        RAM_ROLE: "RAM",
        SSD_ROLE: "SSD",
        HDD_ROLE: "HDD",
        STORAGE_CONTROLLER_ROLE: "контроллеров",
        NETWORK_ADAPTER_ROLE: "сетевых адаптеров",
    }
    return labels.get(role, _role_ru(role))


def _platform_compatibility_warnings(
    spec: StockSpec,
    item: StockSpecItem,
    platform: DistributorProduct,
) -> list[str]:
    warnings: list[str] = [
        "Требуется инженерная проверка совместимости платформы, RAM, накопителей и адаптеров.",
    ]
    search_text = _product_search_text(platform)

    form_factor = _requirement(spec, item, "form_factor")
    if (
        isinstance(form_factor, str)
        and form_factor
        and not _matches_form_factor(search_text, form_factor)
    ):
        warnings.append(
            f"Форм-фактор {form_factor} не подтвержден по данным платформы."
        )

    cpu_sockets = _as_int(_requirement(spec, item, "cpu", "sockets"))
    if cpu_sockets is not None and not _matches_cpu_socket_requirement(search_text, cpu_sockets):
        warnings.append(
            f"{cpu_sockets} процессорных сокета не подтверждены по данным платформы."
        )

    psu_count = _as_int(_requirement(spec, item, "power", "psu_count"))
    redundant_psu = _requirement(spec, item, "power", "redundant_psu")
    if psu_count is not None and not _matches_psu_requirement(search_text, psu_count):
        warnings.append(
            f"{psu_count} БП не подтверждены по данным платформы; требуется проверить комплектацию."
        )
    elif psu_count is None and redundant_psu is True:
        warnings.append(
            "БП не подтверждены по данным платформы; требуется проверить комплектацию."
        )

    return warnings


def _standard_build_warnings(spec: StockSpec, item: StockSpecItem) -> list[str]:
    warnings = [
        "Совместимость RAM с платформой требуется проверить инженеру.",
        "Совместимость SSD/HDD/контроллера требуется проверить инженеру.",
        "Достаточность корзин и слотов платформы не подтверждена локальными данными OCS.",
        "Гарантию и срок поставки требуется проверить по OCS/поставщику.",
        "Характеристики в OCS могут быть неполными; требуется ручная проверка.",
    ]
    return warnings


def _required_storage_role(spec: StockSpec, item: StockSpecItem) -> str | None:
    storage_type = _requirement(spec, item, "storage", "type")
    if not isinstance(storage_type, str):
        return None
    normalized = storage_type.strip().casefold()
    if normalized == "ssd":
        return SSD_ROLE
    if normalized == "hdd":
        return HDD_ROLE
    return None


def _requires_storage_controller(spec: StockSpec, item: StockSpecItem) -> bool:
    requirements = _requirement(spec, item, "storage", "controller")
    if requirements is not None:
        return True
    text = _request_text(spec, item)
    return bool(re.search(r"\b(?:raid|hba)\b|контроллер", text, re.IGNORECASE))


def _requires_network_adapter(spec: StockSpec, item: StockSpecItem) -> bool:
    request_text = _request_text(spec, item)
    requirement = network_requirement_from_sources(
        text=request_text,
        explicit=_dict_or_empty(_requirement(spec, item, "network")),
    )
    return bool(requirement.get("required"))


def _request_text(spec: StockSpec, item: StockSpecItem) -> str:
    parts = [
        spec.source_text,
        item.name,
        str(spec.requirements) if spec.requirements else None,
        str(item.requirements) if item.requirements else None,
    ]
    return " ".join(part for part in parts if part)


def _build_total_price(
    components: list[_BuildComponent],
) -> tuple[Decimal | None, str | None, list[str]]:
    warnings: list[str] = []
    currencies = {
        component.price_currency
        for component in components
        if component.price_value is not None and component.price_currency
    }
    if not components:
        return None, None, []
    if len(currencies) != 1:
        warnings.append(
            "Ориентировочная сумма не рассчитана: цены отсутствуют или указаны в разных валютах."
        )
        return None, None, warnings
    if any(component.price_value is None for component in components):
        warnings.append(
            "Ориентировочная сумма не рассчитана: не у всех комплектующих есть цена."
        )
        return None, None, warnings

    currency = next(iter(currencies))
    total = sum(
        component.price_value * component.quantity_required
        for component in components
        if component.price_value is not None
    )
    return total, currency, []


def _component_stock_warnings(components: list[_BuildComponent]) -> list[str]:
    warnings: list[str] = []
    for component in components:
        if component.available_quantity is None:
            continue
        if component.available_quantity < component.quantity_required:
            warnings.append(
                f"{component.role_ru}: остаток ниже требования, "
                f"доступно {component.available_quantity} шт., "
                f"требуется {component.quantity_required} шт."
            )
    return warnings


def _build_available_quantity(components: list[_BuildComponent]) -> int | None:
    if not components:
        return None
    quantities: list[int] = []
    for component in components:
        if component.available_quantity is None or component.quantity_required <= 0:
            return None
        quantities.append(component.available_quantity // component.quantity_required)
    if not quantities:
        return None
    return min(quantities)


def _component_match_text(component: _BuildComponent) -> str:
    display_name = _component_display_name(component)
    return (
        f"{component.role_ru}: {display_name}, "
        f"требуется {component.quantity_required} шт., "
        f"остаток {_stock_count_text(component.available_quantity)}."
    )


def _component_display_name(component: _BuildComponent) -> str:
    parts = [component.product.producer, component.product.part_number]
    article = " ".join(part for part in parts if part)
    if article:
        return article
    return component.product.item_name or component.product.item_id


def _stock_count_text(value: int | None) -> str:
    if value is None:
        return "не найден"
    return f"{value} шт."


def _role_ru(role: str) -> str:
    labels = {
        SERVER_PLATFORM_ROLE: "Платформа",
        SWITCH_ROLE: "Коммутатор",
        ROUTER_ROLE: "Маршрутизатор",
        FIREWALL_ROLE: "Межсетевой экран",
        ACCESS_POINT_ROLE: "Точка доступа",
        STORAGE_SYSTEM_ROLE: "СХД",
        STORAGE_ARRAY_CONTROLLER_ROLE: "Контроллеры СХД",
        CONTROLLER_MODULE_ROLE: "Модули контроллера",
        DISK_SHELF_ROLE: "Дисковые полки",
        DRIVE_ROLE: "Диски",
        CACHE_ROLE: "Кэш",
        HOST_PORT_ROLE: "Host-порты",
        PROTOCOL_MODULE_ROLE: "Протокольные модули",
        CPU_ROLE: "CPU",
        RAM_ROLE: "RAM",
        SSD_ROLE: "SSD",
        HDD_ROLE: "HDD",
        STORAGE_CONTROLLER_ROLE: "Контроллеры",
        NETWORK_ADAPTER_ROLE: "Сетевые адаптеры",
    }
    labels.update(
        {
            GPU_ROLE: "GPU",
            TRANSCEIVER_ROLE: "Трансиверы",
            DAC_CABLE_ROLE: "DAC-кабели",
            CABLE_ROLE: "Кабели",
            POWER_SUPPLY_ROLE: "Блоки питания",
            RAIL_KIT_ROLE: "Rail kits",
            LICENSE_ROLE: "Лицензии",
            SUPPORT_ROLE: "Поддержка",
            STACKING_MODULE_ROLE: "Модули стекирования",
            OTHER_ACCESSORY_ROLE: "Аксессуары",
            UNMAPPED_ROLE: "Unmapped hard capability",
        }
    )
    return labels.get(role, role)


def _matches_form_factor(text: str, form_factor: str) -> bool:
    match = re.fullmatch(r"([1-9]\d?)U", form_factor.strip(), re.IGNORECASE)
    if match is None:
        return form_factor.casefold() in text.casefold()
    return bool(re.search(rf"\b{re.escape(match.group(1))}\s*u\b", text, re.IGNORECASE))


def _matches_cpu_socket_requirement(text: str, sockets: int) -> bool:
    if sockets == 2 and re.search(r"\bdual\b|\b2\s*x\b", text, re.IGNORECASE):
        return True
    return bool(
        re.search(
            rf"\b{sockets}\s*(?:cpu|socket|sockets|процессор|процессора|сокет)",
            text,
            re.IGNORECASE,
        )
    )


def _matches_psu_requirement(text: str, psu_count: int) -> bool:
    if psu_count <= 0:
        return False
    if platform_power_bundle_satisfies(text, required_psu_count=psu_count):
        return True
    return bool(
        re.search(
            rf"\b{psu_count}\s*(?:x\s*)?(?:psu|power|бп|блока?\s+питания|блоков\s+питания)\b",
            text,
            re.IGNORECASE,
        )
    )


def _evaluate_candidate(
    *,
    spec: StockSpec,
    item: StockSpecItem,
    item_index: int,
    product: DistributorProduct,
    stock_rows: list[DistributorStockPrice],
    product_group: str = SERVER_PRODUCT_GROUP,
) -> MatchCandidateResult | None:
    if (
        product_group != SERVER_PRODUCT_GROUP
        and server_category_role(product.category_id) == READY_SERVER_ROLE
    ):
        return None
    if (
        item.item_type.casefold() == "server"
        and server_category_role(product.category_id) != READY_SERVER_ROLE
    ):
        return None

    search_text = _product_search_text(product)
    search_text_casefolded = search_text.casefold()
    score = 0
    matched_requirements: list[str] = []
    missing_requirements: list[str] = []
    risk_flags: list[str] = []

    if product.category_id in READY_SERVER_CATEGORY_IDS:
        score += 20
        matched_requirements.append("Категория OCS V1100")

    form_factor = _requirement(spec, item, "form_factor")
    if form_factor == "2U" and re.search(r"\b2\s*u\b", search_text, re.IGNORECASE):
        score += 20
        matched_requirements.append("Форм-фактор 2U")

    cpu_sockets = _requirement(spec, item, "cpu", "sockets")
    if _as_int(cpu_sockets) == 2 and re.search(r"\b2\s*x\b", search_text, re.IGNORECASE):
        score += 20
        matched_requirements.append("2 процессорных сокета по названию товара")

    storage_type = _requirement(spec, item, "storage", "type")
    if (
        isinstance(storage_type, str)
        and storage_type.casefold() == "ssd"
        and "ssd" in search_text_casefolded
    ):
        score += 20
        matched_requirements.append("Накопители SSD")

    redundant_psu = _requirement(spec, item, "power", "redundant_psu")
    has_2x = re.search(r"\b2\s*x\b", search_text, re.IGNORECASE)
    has_power_text = re.search(r"\b(?:power|psu)\b", search_text, re.IGNORECASE)
    if redundant_psu is True and has_2x and has_power_text:
        score += 10
        matched_requirements.append("Резервируемое питание по названию товара")

    available_quantity = _available_quantity(stock_rows)
    reservable_locations = _reservable_locations(stock_rows)
    if reservable_locations > 0:
        score += 10
        matched_requirements.append("Есть резервируемый склад")

    if score <= 0:
        return None

    _evaluate_ram_requirement(
        spec=spec,
        item=item,
        search_text=search_text,
        matched_requirements=matched_requirements,
        missing_requirements=missing_requirements,
        risk_flags=risk_flags,
    )
    _evaluate_quantity_requirement(
        item=item,
        available_quantity=available_quantity,
        stock_rows=stock_rows,
        matched_requirements=matched_requirements,
        missing_requirements=missing_requirements,
        risk_flags=risk_flags,
    )
    _evaluate_warranty_risk(
        spec=spec,
        item=item,
        product=product,
        risk_flags=risk_flags,
    )

    price_value, price_currency = _select_price(stock_rows)
    candidate_id = _stable_candidate_id(READY_SERVER_ROLE, product)
    return MatchCandidateResult(
        distributor_code=product.distributor_code,
        item_id=product.item_id,
        product_key=product.product_key,
        part_number=product.part_number,
        producer=product.producer,
        category_id=product.category_id,
        item_name=product.item_name,
        confidence_score=score,
        price_value=price_value,
        price_currency=price_currency,
        available_quantity=available_quantity,
        reservable_locations=reservable_locations,
        matched_requirements=_unique(matched_requirements),
        missing_requirements=_unique(missing_requirements),
        risk_flags=_unique(risk_flags),
        raw={
            "spec_item_index": item_index,
            "candidate_type": READY_SERVER_CANDIDATE_TYPE,
            "candidate_id": candidate_id,
            "quantity_required": item.quantity,
            "quantity_closed": (
                available_quantity is not None and available_quantity >= item.quantity
            ),
            "product": _jsonable(product.raw_json),
            "stock_rows": [_stock_row_raw(row) for row in stock_rows],
        },
        candidate_id=candidate_id,
    )


def _evaluate_ram_requirement(
    *,
    spec: StockSpec,
    item: StockSpecItem,
    search_text: str,
    matched_requirements: list[str],
    missing_requirements: list[str],
    risk_flags: list[str],
) -> None:
    ram_min_gb = _as_int(_requirement(spec, item, "ram", "min_gb"))
    if ram_min_gb is None:
        return

    parsed_ram_gb = parse_total_ram_gb(search_text)
    if parsed_ram_gb is None:
        risk_flags.append(
            "Оперативная память не распознана в наименовании товара; "
            "проверьте конфигурацию."
        )
        return

    if parsed_ram_gb < ram_min_gb:
        missing_requirements.append(
            "Оперативная память ниже требования: "
            f"найдено {parsed_ram_gb} ГБ, требуется {ram_min_gb} ГБ."
        )
        return

    matched_requirements.append(f"Оперативная память: найдено {parsed_ram_gb} ГБ")


def _evaluate_quantity_requirement(
    *,
    item: StockSpecItem,
    available_quantity: int | None,
    stock_rows: list[DistributorStockPrice],
    matched_requirements: list[str],
    missing_requirements: list[str],
    risk_flags: list[str],
) -> None:
    if available_quantity is None:
        missing_requirements.append(f"Остаток не найден: требуется {item.quantity} шт.")
        if not stock_rows:
            risk_flags.append("Нет актуального складского снимка для товара.")
        return

    if available_quantity < item.quantity:
        missing_requirements.append(
            "Остаток ниже требования: "
            f"доступно {available_quantity} шт., требуется {item.quantity} шт."
        )
        risk_flags.append(
            f"По одному варианту не хватает остатка: доступно {available_quantity} шт., "
            f"требуется {item.quantity} шт."
        )
        return

    matched_requirements.append(
        f"Количество закрыто: доступно {available_quantity} шт. из {item.quantity} шт."
    )


def _evaluate_warranty_risk(
    *,
    spec: StockSpec,
    item: StockSpecItem,
    product: DistributorProduct,
    risk_flags: list[str],
) -> None:
    if _requirement(spec, item, "warranty") is not None:
        return

    warranty = product.warranty or ""
    if re.search(r"\b12\s*(?:мес|месяц|months?|mo\.?)\b", warranty, re.IGNORECASE):
        risk_flags.append(
            "Гарантия у OCS указана 12 месяцев, нужно сверить с требованиями."
        )


def parse_total_ram_gb(text: str) -> int | None:
    multiplier_matches = list(
        re.finditer(
            r"\b(\d+)\s*x\s*(?:DDR[345]\s*)?(\d+)\s*GB\b",
            text,
            re.IGNORECASE,
        )
    )
    if multiplier_matches:
        return sum(int(match.group(1)) * int(match.group(2)) for match in multiplier_matches)

    direct_matches = [
        int(match.group(1))
        for match in re.finditer(
            r"\b(\d+)\s*GB\s*(?:RAM|Memory|DDR[345]|ОЗУ)?\b",
            text,
            re.IGNORECASE,
        )
    ]
    if direct_matches:
        return max(direct_matches)
    return None


def parse_memory_module_gb(text: str, raw_json: dict[str, Any] | None = None) -> int | None:
    candidates: list[int] = []
    search_texts = [text]
    if raw_json:
        search_texts.extend(_raw_characteristic_texts(raw_json))

    for source in search_texts:
        for match in re.finditer(
            r"\b(\d+)\s*(?:GB|ГБ)\b",
            source,
            re.IGNORECASE,
        ):
            value = int(match.group(1))
            if 1 <= value <= 2048:
                candidates.append(value)

    if not candidates:
        return None
    return max(candidates)


def _raw_characteristic_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str | int | float | Decimal):
                texts.append(f"{key}: {item}")
            else:
                texts.extend(_raw_characteristic_texts(item))
    elif isinstance(value, list):
        for item in value:
            texts.extend(_raw_characteristic_texts(item))
    elif isinstance(value, str):
        texts.append(value)
    return texts


def _compact_package_json(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(item, str | int | float | bool) or item is None:
            result[key_text] = item
        elif isinstance(item, Mapping):
            nested = {
                str(nested_key): nested_value
                for nested_key, nested_value in item.items()
                if isinstance(nested_value, str | int | float | bool)
                or nested_value is None
            }
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


def _raw_named_values(value: Any, names: set[str]) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).replace("_", "").replace("-", "").casefold()
            if normalized_key in names and isinstance(item, str | int | float | Decimal):
                values.append(str(item))
            elif isinstance(item, dict | list):
                values.extend(_raw_named_values(item, names))
    elif isinstance(value, list):
        for item in value:
            values.extend(_raw_named_values(item, names))
    return values


def _detect_vendor(text: str) -> str | None:
    for vendor, patterns in (*_OEM_VENDOR_ALIASES, *_CPU_VENDOR_ALIASES):
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return vendor
    return None


def _detect_oem_vendor(text: str) -> str | None:
    for vendor, patterns in _OEM_VENDOR_ALIASES:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return vendor
    return None


def _detect_cpu_brand(text: str) -> str:
    if _has_amd_platform_marker(text):
        return "AMD"
    if _has_intel_platform_marker(text):
        return "Intel"
    for brand, patterns in _CPU_VENDOR_ALIASES:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return brand
    return UNKNOWN_FACT


def _detect_cpu_family(text: str) -> str:
    if re.search(r"\bxeon\b", text, re.IGNORECASE):
        return "Xeon"
    if re.search(r"\bepyc\b", text, re.IGNORECASE):
        return "EPYC"
    if _has_amd_platform_marker(text):
        return "EPYC"
    if _has_intel_platform_marker(text):
        return "Xeon"
    return UNKNOWN_FACT


def _detect_cpu_socket(text: str) -> str:
    lga_match = re.search(r"\bLGA\s*(3647|4189|4677|4094|6096)\b", text, re.IGNORECASE)
    if lga_match is not None:
        return f"LGA{lga_match.group(1)}"
    bare_lga_match = re.search(r"\b(3647|4189|4677|4094|6096)\b", text, re.IGNORECASE)
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
            r"\b(?:intel|xeon|ice\s*lake|sapphire\s+rapids|rapid\s+sapphire|c621|c741|lga\s*(?:3647|4189|4677))\b",
            text,
            re.IGNORECASE,
        )
    )


def _detect_cpu_cores(text: str) -> int | None:
    matches: list[int] = []
    for pattern in (
        r"\b(\d{1,3})\s*(?:core|cores|ядер|ядра)\b",
        r"\b(\d{1,3})\s*c\b",
    ):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = int(match.group(1))
            if 1 <= value <= 256:
                matches.append(value)
    if not matches:
        return None
    return max(matches)


def _detect_ram_type(text: str) -> str:
    match = re.search(r"\bDDR\s*([345])\b", text, re.IGNORECASE)
    if match is not None:
        return f"DDR{match.group(1)}"
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


def _detect_storage_capacity(text: str) -> str:
    match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(TB|ТБ|GB|ГБ)\b", text, re.IGNORECASE)
    if match is None:
        return UNKNOWN_FACT
    value = match.group(1).replace(",", ".")
    unit = match.group(2).upper().replace("ТБ", "TB").replace("ГБ", "GB")
    return f"{value} {unit}"


def _detect_storage_interface(text: str) -> str:
    if re.search(
        r"\bnvme\b|\bnvme\s*bp\b|\bnvme\s*backplane\b|\bu\.?2\b|\bu\.?3\b|\bm\.?2\b",
        text,
        re.IGNORECASE,
    ):
        return "NVMe"
    if re.search(r"\bsas\b", text, re.IGNORECASE):
        return "SAS"
    if re.search(r"\bsata\b", text, re.IGNORECASE):
        return "SATA"
    return UNKNOWN_FACT


def _detect_nvme_support(text: str) -> bool | None:
    if re.search(r"\bno\s+nvme\b|\bwithout\s+nvme\b", text, re.IGNORECASE):
        return False
    if re.search(
        r"\bnvme\b|\bnvme\s*bp\b|\bnvme\s*backplane\b|\bu\.?2\b|\bu\.?3\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return None


def _detect_form_factor_hints(text: str) -> list[str]:
    hints: list[str] = []
    for match in re.finditer(r"\b([1-9]\d?)\s*u\b", text, re.IGNORECASE):
        hints.append(f"{match.group(1)}U")
    if re.search(r"\bLFF\b|\b3\.5", text, re.IGNORECASE):
        hints.append("LFF/3.5")
    if re.search(r"\bSFF\b|\b2\.5", text, re.IGNORECASE):
        hints.append("SFF/2.5")
    return _unique(hints)


def _is_vendor_option_kit(
    *,
    role: str | None,
    normalized_vendor: str | None,
    option_kit_vendor: str,
    cpu_brand: str,
    text: str,
) -> bool:
    if role != CPU_ROLE:
        return False
    if option_kit_vendor == UNKNOWN_FACT:
        return False
    if option_kit_vendor in {"Intel", "AMD"}:
        return False
    if normalized_vendor == option_kit_vendor:
        return True
    return bool(
        cpu_brand != UNKNOWN_FACT
        and re.search(
            r"\b(?:kit|option|spare|upgrade|for)\b|для\s+\S+",
            text,
            re.IGNORECASE,
        )
    )


def _explicit_field(spec: StockSpec, item: StockSpecItem, field_name: str) -> Any:
    item_value = getattr(item, field_name, None)
    if item_value is not None:
        return item_value
    return getattr(spec, field_name, None)


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_cpu_vendor(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return UNKNOWN_FACT
    detected = _detect_cpu_brand(text)
    return detected if detected != UNKNOWN_FACT else UNKNOWN_FACT


def _normalize_cpu_family(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return UNKNOWN_FACT
    detected = _detect_cpu_family(text)
    return detected if detected != UNKNOWN_FACT else UNKNOWN_FACT


def _normalize_ram_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return UNKNOWN_FACT
    detected = _detect_ram_type(text)
    return detected if detected != UNKNOWN_FACT else UNKNOWN_FACT


def _normalize_storage_preference(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return UNKNOWN_FACT
    if re.search(r"\bnvme\b", text, re.IGNORECASE):
        return "NVMe"
    if re.search(r"\bsas\b", text, re.IGNORECASE):
        return "SAS"
    if re.search(r"\bsata\b", text, re.IGNORECASE):
        return "SATA"
    if re.search(r"\bssd\b|ссд", text, re.IGNORECASE):
        return "SSD"
    if re.search(r"\bhdd\b|жестк(?:ий|ие|их)\s+диск", text, re.IGNORECASE):
        return "HDD"
    return UNKNOWN_FACT


def _normalize_storage_type(value: Any) -> str:
    preference = _normalize_storage_preference(value)
    if preference in {"NVMe", "SAS", "SATA"}:
        return "SSD"
    return preference


def _normalize_storage_interface(value: Any) -> str:
    preference = _normalize_storage_preference(value)
    if preference in {"NVMe", "SAS", "SATA"}:
        return preference
    return UNKNOWN_FACT


def _normalize_storage_protocol(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return UNKNOWN_FACT
    if re.search(r"\bnvme-?of\b|nvme\s+over\s+fabrics", text, re.IGNORECASE):
        return "NVMe-oF"
    if re.search(r"\bfc\b|fibre\s+channel", text, re.IGNORECASE):
        return "FC"
    if re.search(r"\biscsi\b", text, re.IGNORECASE):
        return "iSCSI"
    if re.search(r"\bsas\b", text, re.IGNORECASE):
        return "SAS"
    return UNKNOWN_FACT


def _normalize_optimization_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in SUPPORTED_OPTIMIZATION_MODES:
        return text
    return OPTIMIZATION_MODE_COST_MINIMAL_FIT


def _detect_requested_cpu_vendor(text: str) -> str | None:
    detected = _detect_cpu_brand(text)
    return detected if detected != UNKNOWN_FACT else None


def _detect_requested_cpu_family(text: str) -> str | None:
    detected = _detect_cpu_family(text)
    return detected if detected != UNKNOWN_FACT else None


def _detect_requested_cpu_cores(text: str) -> int | None:
    return _detect_cpu_cores(text)


def _detect_requested_storage_preference(text: str) -> str | None:
    detected = _normalize_storage_preference(text)
    return detected if detected != UNKNOWN_FACT else None


def _detect_requested_ram_type(text: str) -> str | None:
    detected = _detect_ram_type(text)
    return detected if detected != UNKNOWN_FACT else None


def _detect_requested_ram_gb(text: str) -> int | None:
    ram_marker = r"(?:ram|memory|озу|оператив(?:ной)?\s+памяти|оперативк[аи])"
    patterns = (
        rf"\b(?:по\s+)?(\d+)\s*(?:гб|gb)\s*{ram_marker}\b",
        rf"\b{ram_marker}\s*(?:на\s+сервер)?\s*(\d+)\s*(?:гб|gb)\b",
        r"\b(?:по\s+)?(\d+)\s*(?:гб|gb)\s*(?=ddr\s*[345]\b)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is not None:
            return int(match.group(1))

    for match in re.finditer(r"\b(?:по\s+)?(\d+)\s*(?:гб|gb)\b", text, re.IGNORECASE):
        segment = text[max(0, match.start() - 40) : match.end() + 40]
        if re.search(r"\b(?:ssd|hdd|nvme|sas|sata|накопител|диск)\b", segment, re.IGNORECASE):
            continue
        if re.search(rf"{ram_marker}|ddr\s*[345]|\bпамят", segment, re.IGNORECASE):
            return int(match.group(1))
    return None


def _ram_per_server_phrase(text: str) -> bool:
    ram_marker = r"(?:ram|memory|озу|оператив(?:ной)?\s+памяти|оперативк[аи])"
    return bool(
        re.search(
            rf"\bпо\s+\d+\s*(?:гб|gb)\s*{ram_marker}\s+на\s+сервер\b",
            text,
            re.IGNORECASE,
        )
    )


def _detect_requested_storage_interface(text: str) -> str | None:
    detected = _normalize_storage_interface(text)
    return detected if detected != UNKNOWN_FACT else None


def _detect_requested_storage_capacity(text: str) -> str | None:
    storage_markers = r"(?:ssd|hdd|nvme|sas|sata|накопител\w*|диск\w*)"
    capacity = r"(\d+(?:[.,]\d+)?)\s*(TB|ТБ|GB|ГБ)"
    patterns = (
        rf"{storage_markers}[^,;\n]{{0,60}}?{capacity}",
        rf"{capacity}[^,;\n]{{0,60}}?{storage_markers}",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            segment = match.group(0)
            if re.search(r"\b(?:ram|memory|озу|памят)", segment, re.IGNORECASE):
                continue
            value = match.group(1).replace(",", ".")
            unit = match.group(2).upper().replace("ТБ", "TB").replace("ГБ", "GB")
            return f"{value} {unit}"
    return None


def _detect_requested_storage_qty(text: str) -> int | None:
    match = re.search(
        r"\b(\d{1,2})\s*(?:x|х|шт\.?)?\s*"
        r"(?:ssd|hdd|nvme|sas|sata|накопител\w*|диск\w*)\b",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    value = int(match.group(1))
    if 1 <= value <= 64:
        return value
    return None


def _requirement(spec: StockSpec, item: StockSpecItem, *path: str) -> Any:
    item_value = _nested_get(item.requirements, *path)
    if item_value is not None:
        return item_value
    return _nested_get(spec.requirements, *path)


def _nested_get(source: dict[str, Any], *path: str) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(str(value).replace(",", ".").strip())
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _capacity_to_tb(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float | Decimal):
        numeric = float(value)
        if numeric <= 0:
            return None
        return numeric / 1000 if numeric > 100 else numeric

    text = str(value).strip()
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(TB|ТБ|GB|ГБ)?",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    numeric = float(match.group(1).replace(",", "."))
    unit = (match.group(2) or "TB").upper().replace("ТБ", "TB").replace("ГБ", "GB")
    if numeric <= 0:
        return None
    if unit == "GB":
        return numeric / 1000
    return numeric


def _format_number(value: float | int | Decimal) -> str:
    decimal_value = Decimal(str(value)).normalize()
    if decimal_value == decimal_value.to_integral():
        return str(decimal_value.quantize(Decimal("1")))
    return format(decimal_value, "f").rstrip("0").rstrip(".")


def _product_search_text(product: DistributorProduct) -> str:
    return " ".join(
        part
        for part in [
            product.item_name,
            product.item_name_rus,
            product.product_name,
            product.product_description,
            product.product_notes,
            product.part_number,
        ]
        if part
    )


def _available_quantity(rows: list[DistributorStockPrice]) -> int | None:
    quantities = [row.quantity_value for row in rows if row.quantity_value is not None]
    if not quantities:
        return None
    return sum(quantities)


def _reservable_locations(rows: list[DistributorStockPrice]) -> int:
    return sum(
        1
        for row in rows
        if row.can_reserve is True and (row.quantity_value is None or row.quantity_value > 0)
    )


def _select_price(rows: list[DistributorStockPrice]) -> tuple[Decimal | None, str | None]:
    for row in rows:
        if row.price_order_value is not None:
            return row.price_order_value, row.price_order_currency
    for row in rows:
        if row.price_list_value is not None:
            return row.price_list_value, row.price_list_currency
    for row in rows:
        if row.end_user_value is not None:
            return row.end_user_value, row.end_user_currency
    return None, None


def _stock_row_raw(row: DistributorStockPrice) -> dict[str, Any]:
    return {
        "id": row.id,
        "shipment_city": row.shipment_city,
        "location": row.location,
        "location_description": row.location_description,
        "location_type": row.location_type,
        "quantity_value": row.quantity_value,
        "quantity_is_greater_than": row.quantity_is_greater_than,
        "can_reserve": row.can_reserve,
        "price_order_value": _jsonable(row.price_order_value),
        "price_order_currency": row.price_order_currency,
        "price_list_value": _jsonable(row.price_list_value),
        "price_list_currency": row.price_list_currency,
        "end_user_value": _jsonable(row.end_user_value),
        "end_user_currency": row.end_user_currency,
        "synced_at": _jsonable(row.synced_at),
        "raw_json": _jsonable(row.raw_json),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            return str(normalized.quantize(Decimal("1")))
        return format(normalized, "f")
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _unique(values: list[str] | Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _commercial_optimization_goal(role_plan: Mapping[str, Any]) -> str | None:
    for instruction in _mapping_list(role_plan.get("commercial_instructions")):
        parsed = _dict_or_empty(instruction.get("parsed_requirements"))
        goal = str(
            parsed.get("optimization_goal")
            or instruction.get("optimization_goal")
            or ""
        ).strip()
        if goal in {"cheapest_valid_stock_quote", "cheapest_stock_quote"}:
            return OPTIMIZATION_MODE_COST_MINIMAL_FIT
        if goal:
            return goal
    return None


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _package_or_matrix_value(
    package_diagnostics: Mapping[str, Any],
    component_candidate_matrix: Any,
    key: str,
    default: Any,
) -> Any:
    package_value = package_diagnostics.get(key)
    if package_value not in (None, [], {}):
        return package_value
    if isinstance(component_candidate_matrix, Mapping):
        matrix_value = component_candidate_matrix.get(key)
        if matrix_value not in (None, [], {}):
            return matrix_value
    return default


def _semantic_report_fields(role_plan: Mapping[str, Any]) -> dict[str, Any]:
    role_plan_mapping = _dict_or_empty(role_plan)
    return {
        "primary_product_group": _text_or_none(
            role_plan_mapping.get("primary_product_group")
        ),
        "primary_object": _text_or_none(role_plan_mapping.get("primary_object")),
        "semantic_planner_source": _text_or_none(
            role_plan_mapping.get("semantic_planner_source")
        ),
        "semantic_planner_used": bool(
            role_plan_mapping.get("semantic_planner_used")
            or str(role_plan_mapping.get("semantic_planner_source") or "").strip()
            in {"llm", "llm_repaired", "llm_minimal_fallback"}
        ),
        "semantic_planner_confidence": _text_or_none(
            role_plan_mapping.get("semantic_planner_confidence")
        ),
        "semantic_planner_error_type": _text_or_none(
            role_plan_mapping.get("semantic_planner_error_type")
        ),
        "semantic_planner_http_status": (
            _as_int(role_plan_mapping.get("semantic_planner_http_status"))
        ),
        "semantic_planner_parse_status": _text_or_none(
            role_plan_mapping.get("semantic_planner_parse_status")
        ),
        "semantic_planner_fallback_reason": _text_or_none(
            role_plan_mapping.get("semantic_planner_fallback_reason")
        ),
        "semantic_planner_attempts": _jsonable(
            role_plan_mapping.get("semantic_planner_attempts")
            if isinstance(role_plan_mapping.get("semantic_planner_attempts"), list)
            else []
        ),
        "semantic_planner_stage": _text_or_none(
            role_plan_mapping.get("semantic_planner_stage")
        ),
        "semantic_planner_stage_timeouts": _jsonable(
            role_plan_mapping.get("semantic_planner_stage_timeouts")
            if isinstance(role_plan_mapping.get("semantic_planner_stage_timeouts"), list)
            else []
        ),
        "semantic_planner_timeout_reason": _text_or_none(
            role_plan_mapping.get("semantic_planner_timeout_reason")
        ),
        "semantic_planner_timeout_seconds": (
            role_plan_mapping.get("semantic_planner_timeout_seconds")
        ),
        "semantic_planner_elapsed_ms": _as_int(
            role_plan_mapping.get("semantic_planner_elapsed_ms")
        ),
        "semantic_planner_repair_attempted": bool(
            role_plan_mapping.get("semantic_planner_repair_attempted")
        ),
        "semantic_planner_repair_success": bool(
            role_plan_mapping.get("semantic_planner_repair_success")
        ),
        "semantic_planner_minimal_router_used": bool(
            role_plan_mapping.get("semantic_planner_minimal_router_used")
        ),
        "semantic_planner_minimal_fallback_used": bool(
            role_plan_mapping.get("semantic_planner_minimal_fallback_used")
        ),
        "semantic_planner_empty_response_count": _as_int(
            role_plan_mapping.get("semantic_planner_empty_response_count")
        )
        or 0,
        "semantic_planner_empty_response_reason": _text_or_none(
            role_plan_mapping.get("semantic_planner_empty_response_reason")
        ),
        "requirement_classifier_status": _text_or_none(
            role_plan_mapping.get("requirement_classifier_status")
        ),
        "requirement_classifier_error_type": _text_or_none(
            role_plan_mapping.get("requirement_classifier_error_type")
        ),
        "requirement_classifier_parse_status": _text_or_none(
            role_plan_mapping.get("requirement_classifier_parse_status")
        ),
        "requirement_classifier_incomplete_reason": _text_or_none(
            role_plan_mapping.get("requirement_classifier_incomplete_reason")
        ),
        "requirement_source_coverage": _jsonable(
            role_plan_mapping.get("requirement_source_coverage")
            if isinstance(role_plan_mapping.get("requirement_source_coverage"), list)
            else []
        ),
        "requirement_source_coverage_percent": role_plan_mapping.get(
            "requirement_source_coverage_percent"
        ),
        "unclassified_source_fragments": _jsonable(
            _string_list(role_plan_mapping.get("unclassified_source_fragments"))
        ),
        "synthetic_requirement_count": _as_int(
            role_plan_mapping.get("synthetic_requirement_count")
        )
        or 0,
        "source_backed_requirement_count": _as_int(
            role_plan_mapping.get("source_backed_requirement_count")
        )
        or 0,
        "requirement_classifier_repair_quality": _text_or_none(
            role_plan_mapping.get("requirement_classifier_repair_quality")
        ),
        "requirement_classifier_repair_accepted": bool(
            role_plan_mapping.get("requirement_classifier_repair_accepted")
        ),
        "semantic_planner_model": _text_or_none(
            role_plan_mapping.get("semantic_planner_model")
        ),
        "semantic_planner_provider": _text_or_none(
            role_plan_mapping.get("semantic_planner_provider")
        ),
        "selected_product_group_reason": _text_or_none(
            role_plan_mapping.get("selected_product_group_reason")
        ),
        "deterministic_product_group_hint": _text_or_none(
            role_plan_mapping.get("deterministic_product_group_hint")
        ),
        "semantic_planner_disagreement": bool(
            role_plan_mapping.get("semantic_planner_disagreement")
        ),
        "matrix_blueprint": _jsonable(
            _dict_or_empty(role_plan_mapping.get("matrix_blueprint"))
        ),
        "matrix_blueprint_roles": _jsonable(
            _string_list(role_plan_mapping.get("matrix_blueprint_roles"))
        ),
        "embedded_requirements": _jsonable(
            role_plan_mapping.get("embedded_requirements")
            if isinstance(role_plan_mapping.get("embedded_requirements"), list)
            else []
        ),
        "classified_requirements": _jsonable(
            role_plan_mapping.get("classified_requirements")
            if isinstance(role_plan_mapping.get("classified_requirements"), list)
            else []
        ),
        "purchasable_role_requirements": _jsonable(
            role_plan_mapping.get("purchasable_role_requirements")
            if isinstance(role_plan_mapping.get("purchasable_role_requirements"), list)
            else []
        ),
        "primary_object_feature_requirements": _jsonable(
            role_plan_mapping.get("primary_object_feature_requirements")
            if isinstance(
                role_plan_mapping.get("primary_object_feature_requirements"), list
            )
            else []
        ),
        "accessory_or_consumable_requirements": _jsonable(
            role_plan_mapping.get("accessory_or_consumable_requirements")
            if isinstance(
                role_plan_mapping.get("accessory_or_consumable_requirements"), list
            )
            else []
        ),
        "service_or_support_requirements": _jsonable(
            role_plan_mapping.get("service_or_support_requirements")
            if isinstance(role_plan_mapping.get("service_or_support_requirements"), list)
            else []
        ),
        "logistics_or_commercial_constraints": _jsonable(
            role_plan_mapping.get("logistics_or_commercial_constraints")
            if isinstance(
                role_plan_mapping.get("logistics_or_commercial_constraints"), list
            )
            else []
        ),
        "engineering_check_requirements": _jsonable(
            role_plan_mapping.get("engineering_check_requirements")
            if isinstance(role_plan_mapping.get("engineering_check_requirements"), list)
            else []
        ),
        "unmapped_requirements_non_blocking": _jsonable(
            role_plan_mapping.get("unmapped_requirements_non_blocking")
            if isinstance(role_plan_mapping.get("unmapped_requirements_non_blocking"), list)
            else []
        ),
        "unmapped_requirements_blocking": _jsonable(
            role_plan_mapping.get("unmapped_requirements_blocking")
            if isinstance(role_plan_mapping.get("unmapped_requirements_blocking"), list)
            else []
        ),
        "requirement_role_mapping_decision": _jsonable(
            role_plan_mapping.get("requirement_role_mapping_decision")
            if isinstance(role_plan_mapping.get("requirement_role_mapping_decision"), list)
            else []
        ),
        "requirement_fulfillment_decision": _jsonable(
            role_plan_mapping.get("requirement_fulfillment_decision")
            if isinstance(
                role_plan_mapping.get("requirement_fulfillment_decision"), list
            )
            else []
        ),
        "not_primary_product_groups": _jsonable(
            role_plan_mapping.get("not_primary_product_groups")
            if isinstance(role_plan_mapping.get("not_primary_product_groups"), list)
            else []
        ),
    }


def _semantic_plan_is_llm_authoritative(role_plan: Mapping[str, Any]) -> bool:
    return str(role_plan.get("semantic_planner_source") or "").strip() in {
        "llm",
        "llm_repaired",
        "llm_minimal_fallback",
    }


def _semantic_planner_failed_closed(role_plan: Mapping[str, Any]) -> bool:
    fallback_reason = str(role_plan.get("semantic_planner_fallback_reason") or "").strip()
    source = str(role_plan.get("semantic_planner_source") or "").strip()
    product_group = str(role_plan.get("product_group") or "").strip()
    return (
        fallback_reason == SEMANTIC_PLANNER_COMPLEX_FALLBACK_REASON
        or source == SEMANTIC_PLANNER_TIMEOUT_SOURCE
        or (
            fallback_reason == SEMANTIC_PLANNER_TIMEOUT_REASON
            and product_group == "unknown"
        )
    )


def _semantic_planner_unavailable_reason(role_plan: Mapping[str, Any]) -> dict[str, Any]:
    reason_code = (
        str(role_plan.get("semantic_planner_fallback_reason") or "").strip()
        or SEMANTIC_PLANNER_COMPLEX_FALLBACK_REASON
    )
    return {
        "product_group": "unknown",
        "reason_code": reason_code,
        "summary": SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE,
        "manual_checks": [
            "Проверить доступность LLM provider в stock-api.",
            "Повторить запрос после восстановления AI semantic planner.",
        ],
        "semantic_planner_source": role_plan.get("semantic_planner_source"),
        "semantic_planner_error_type": role_plan.get("semantic_planner_error_type"),
        "semantic_planner_http_status": role_plan.get("semantic_planner_http_status"),
        "semantic_planner_parse_status": role_plan.get("semantic_planner_parse_status"),
        "semantic_planner_fallback_reason": role_plan.get(
            "semantic_planner_fallback_reason"
        ),
    }


def _semantic_planner_unavailable_commercial_summary() -> dict[str, Any]:
    return {
        "mode": "single_best_cost_valid",
        "status": "no_recommendation",
        "title": "Безопасную складскую рекомендацию дать нельзя.",
        "reasons": [SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE],
        "copy_paste_text": SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE,
        "lines": [SEMANTIC_PLANNER_UNAVAILABLE_MESSAGE],
    }
