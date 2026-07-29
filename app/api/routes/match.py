from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.database import get_session
from app.db.models import MatchCandidate, MatchRun
from app.llm.base import LlmError
from app.matching.ai_match_orchestrator import (
    AiMatchOrchestratorRequest,
    run_ai_match_orchestrator,
)
from app.matching.match_engine import (
    MAX_API_BUILD_CANDIDATES,
    MatchCandidateResult,
    extract_stock_spec_for_text_match,
    match_stock_spec,
)
from app.matching.match_repository import MatchCandidateCreate, MatchRepository, MatchRunCreate
from app.matching.simple_stock_quote_service import run_simple_stock_quote
from app.matching.spec_schema import StockSpec
from app.matching.v3_full_category_quote_service import (
    V3FullCategoryQuoteResult,
    run_v3_full_category_quote,
)
from app.reports.composer_result_normalizer import normalize_composer_report_json
from app.reports.excel_report import EXCEL_MEDIA_TYPE, build_match_excel_report
from app.reports.match_report import build_match_markdown_report
from app.reports.v3_full_category_report import build_v3_full_category_markdown_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/match", tags=["match"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
LimitQuery = Annotated[int, Query(ge=1, le=100)]


class MatchRequest(BaseModel):
    text: str | None = None
    spec: StockSpec | None = None
    pipeline_v2: bool | None = None


MatchRequestBody = Annotated[MatchRequest | None, Body()]


class V3FullCategoryQuoteRequest(BaseModel):
    text: str
    profile: str | None = None
    category_ids: list[str] = Field(default_factory=list)
    distributor_code: str = "ocs"


class V3FullCategoryQuoteResponse(BaseModel):
    match_run_id: int | None = None
    profile: str | None = None
    category_ids: list[str] = Field(default_factory=list)
    distributor_code: str
    result_state: str
    pipeline_version: str | None = None
    llm_configurator_used: bool = False
    primary_recommendation_status: str | None = None
    final_status_source: str | None = None
    engineering_review_required: bool = False
    validated_quote: dict[str, Any] = Field(default_factory=dict)
    no_recommendation_reason: dict[str, Any] = Field(default_factory=dict)
    validation_failure_reason: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    validation_error_details: list[dict[str, Any]] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    llm_error_type: str | None = None
    llm_http_status: int | None = None
    report_url: str | None = None
    report_xlsx_url: str | None = None
    report_json: dict[str, Any] = Field(default_factory=dict)


class MatchSummaryCandidate(BaseModel):
    candidate_id: str | None = None
    candidate_type: str = "ready_server"
    distributor_code: str
    part_number: str | None
    item_id: str
    producer: str | None
    category_id: str | None
    item_name: str | None
    confidence_score: int
    price_value: str | None
    price_currency: str | None
    available_quantity: int | None
    platform: dict[str, Any] = Field(default_factory=dict)
    components: list[dict[str, Any]] = Field(default_factory=list)
    total_price_value: str | None = None
    total_price_currency: str | None = None
    missing_components: list[str] = Field(default_factory=list)
    compatibility_warnings: list[str] = Field(default_factory=list)
    engineer_review_required: bool = True
    completeness_status: str | None = None
    completeness_label: str | None = None
    included_component_roles: list[str] = Field(default_factory=list)
    missing_component_roles: list[str] = Field(default_factory=list)
    excluded_from_total_roles: list[str] = Field(default_factory=list)
    cpu_per_server: int | None = None
    total_cpu_required: int | None = None
    total_price_note: str | None = None
    score: int | None = None
    rank_reason: list[str] = Field(default_factory=list)
    optimization_mode: str | None = None
    requirement_fit: str | None = None
    right_size_note: str | None = None
    cpu_over_requirement: int | None = None
    storage_over_requirement: float | None = None
    ram_overage_gb: int | None = None
    overfit_reason: str | None = None


class MatchSummaryResponse(BaseModel):
    match_run_id: int
    status: str
    engineer_review_required: bool
    total_candidates: int
    matched_items: int
    confirmation_text: str | None
    risk_flags: list[str]
    missing_requirements: list[str]
    candidates: list[MatchSummaryCandidate]
    ready_stock_candidates: list[MatchSummaryCandidate]
    build_candidates: list[MatchSummaryCandidate]
    product_group: str | None = None
    primary_object: str | None = None
    semantic_planner_source: str | None = None
    semantic_planner_used: bool = False
    semantic_planner_confidence: str | None = None
    semantic_planner_error_type: str | None = None
    semantic_planner_http_status: int | None = None
    semantic_planner_parse_status: str | None = None
    semantic_planner_fallback_reason: str | None = None
    semantic_planner_attempts: list[dict[str, Any]] = Field(default_factory=list)
    semantic_planner_stage: str | None = None
    semantic_planner_stage_timeouts: list[dict[str, Any]] = Field(default_factory=list)
    semantic_planner_timeout_reason: str | None = None
    semantic_planner_timeout_seconds: float | None = None
    semantic_planner_elapsed_ms: int | None = None
    semantic_planner_repair_attempted: bool = False
    semantic_planner_repair_success: bool = False
    semantic_planner_minimal_router_used: bool = False
    semantic_planner_minimal_fallback_used: bool = False
    semantic_planner_empty_response_count: int = 0
    semantic_planner_empty_response_reason: str | None = None
    requirement_classifier_status: str | None = None
    requirement_classifier_error_type: str | None = None
    requirement_classifier_parse_status: str | None = None
    requirement_classifier_incomplete_reason: str | None = None
    requirement_source_coverage: list[dict[str, Any]] = Field(default_factory=list)
    requirement_source_coverage_percent: float | None = None
    unclassified_source_fragments: list[str] = Field(default_factory=list)
    synthetic_requirement_count: int = 0
    source_backed_requirement_count: int = 0
    requirement_classifier_repair_quality: str | None = None
    requirement_classifier_repair_accepted: bool = False
    semantic_planner_model: str | None = None
    semantic_planner_provider: str | None = None
    candidate_universe_planner_mode: str | None = None
    primary_product_group: str | None = None
    procurement_intent: str | None = None
    selected_group_reason: str | None = None
    selected_product_group_reason: str | None = None
    competing_product_groups: list[dict[str, Any]] = Field(default_factory=list)
    primary_object_indicators: list[Any] = Field(default_factory=list)
    component_role_indicators: list[dict[str, Any]] = Field(default_factory=list)
    excluded_category_groups: list[dict[str, Any]] = Field(default_factory=list)
    planner_repair_attempted: bool = False
    planner_repair_success: bool = False
    planner_suspicion_reasons: list[str] = Field(default_factory=list)
    deterministic_product_group_hint: str | None = None
    semantic_planner_disagreement: bool = False
    matrix_blueprint: dict[str, Any] = Field(default_factory=dict)
    matrix_blueprint_roles: list[str] = Field(default_factory=list)
    stage_a_broad_roles: list[str] = Field(default_factory=list)
    semantic_matrix_blueprint_roles: list[str] = Field(default_factory=list)
    requirement_classifier_roles: list[str] = Field(default_factory=list)
    effective_matrix_roles_before_category_planner: list[str] = Field(default_factory=list)
    category_planner_input_roles: list[str] = Field(default_factory=list)
    category_planner_output_roles: list[str] = Field(default_factory=list)
    validated_category_plan_roles: list[str] = Field(default_factory=list)
    materialized_matrix_roles: list[str] = Field(default_factory=list)
    composer_package_roles: list[str] = Field(default_factory=list)
    roles_dropped_after_stage_a: list[str] = Field(default_factory=list)
    roles_dropped_before_category_planner: list[str] = Field(default_factory=list)
    roles_dropped_after_category_planner: list[str] = Field(default_factory=list)
    roles_dropped_during_materialization: list[str] = Field(default_factory=list)
    roles_dropped_reason_by_role: dict[str, Any] = Field(default_factory=dict)
    role_source_by_role: dict[str, Any] = Field(default_factory=dict)
    role_lifecycle_trace: list[dict[str, Any]] = Field(default_factory=list)
    embedded_requirements: list[Any] = Field(default_factory=list)
    classified_requirements: list[dict[str, Any]] = Field(default_factory=list)
    purchasable_role_requirements: list[dict[str, Any]] = Field(default_factory=list)
    primary_object_feature_requirements: list[dict[str, Any]] = Field(default_factory=list)
    accessory_or_consumable_requirements: list[dict[str, Any]] = Field(default_factory=list)
    service_or_support_requirements: list[dict[str, Any]] = Field(default_factory=list)
    logistics_or_commercial_constraints: list[dict[str, Any]] = Field(default_factory=list)
    engineering_check_requirements: list[dict[str, Any]] = Field(default_factory=list)
    unmapped_requirements_non_blocking: list[dict[str, Any]] = Field(default_factory=list)
    unmapped_requirements_blocking: list[dict[str, Any]] = Field(default_factory=list)
    requirement_role_mapping_decision: list[dict[str, Any]] = Field(default_factory=list)
    requirement_fulfillment_decision: list[dict[str, Any]] = Field(default_factory=list)
    not_primary_product_groups: list[Any] = Field(default_factory=list)
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    required_capabilities: list[dict[str, Any]] = Field(default_factory=list)
    optional_capabilities: list[dict[str, Any]] = Field(default_factory=list)
    workload_context: list[str] = Field(default_factory=list)
    logistics_constraints: dict[str, Any] = Field(default_factory=dict)
    commercial_instructions: list[dict[str, Any]] = Field(default_factory=list)
    response_instructions: list[str] = Field(default_factory=list)
    unsupported_or_unmapped_requirements: list[str] = Field(default_factory=list)
    role_plan: dict[str, Any] = Field(default_factory=dict)
    category_plan: dict[str, Any] = Field(default_factory=dict)
    category_plan_entries: list[dict[str, Any]] = Field(default_factory=list)
    category_catalog_summary: dict[str, Any] = Field(default_factory=dict)
    category_planner_source: str | None = None
    category_plan_source: str | None = None
    category_planner_missing_required_roles: list[str] = Field(default_factory=list)
    category_planner_repair_attempted: bool = False
    category_planner_repair_success: bool = False
    category_planner_repair_reason: str | None = None
    category_planner_repaired_roles: list[str] = Field(default_factory=list)
    category_planner_unresolved_required_roles: list[str] = Field(default_factory=list)
    required_roles: list[str] = Field(default_factory=list)
    missing_required_roles: list[str] = Field(default_factory=list)
    missing_category_roles: list[str] = Field(default_factory=list)
    missing_required_capabilities: list[dict[str, Any]] = Field(default_factory=list)
    role_coverage_summary: dict[str, Any] = Field(default_factory=dict)
    category_plan_warnings: list[str] = Field(default_factory=list)
    matrix_distiller_used: bool = False
    matrix_distiller_source: str | None = None
    matrix_distiller_diagnostics: dict[str, Any] = Field(default_factory=dict)
    broad_count_by_role: dict[str, Any] = Field(default_factory=dict)
    distilled_count_by_role: dict[str, Any] = Field(default_factory=dict)
    full_matrix_evaluation_used: bool = False
    full_matrix_evaluation_fallback_reason: str | None = None
    provider_error_type: str | None = None
    provider_context_limit: dict[str, Any] = Field(default_factory=dict)
    role_chunk_count_by_role: dict[str, Any] = Field(default_factory=dict)
    evaluated_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    selected_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    role_reducer_summary: dict[str, Any] = Field(default_factory=dict)
    full_matrix_failed_chunks: list[dict[str, Any]] = Field(default_factory=list)
    bom_critic_used: bool = False
    no_recommendation_coverage: dict[str, Any] = Field(default_factory=dict)
    no_recommendation_coverage_gate_passed: bool = False
    no_recommendation_coverage_repair_attempted: bool = False
    no_recommendation_coverage_repair_success: bool = False
    no_recommendation_coverage_rejected: bool = False
    no_recommendation_coverage_thresholds: dict[str, Any] = Field(default_factory=dict)
    no_recommendation_coverage_repair_reason: str | None = None
    llm_cost_diagnostics: dict[str, Any] = Field(default_factory=dict)
    count_by_role: dict[str, Any] = Field(default_factory=dict)
    broad_matrix_count_by_role: dict[str, Any] = Field(default_factory=dict)
    composer_package_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    composer_package_candidate_total: int = 0
    composer_package_candidate_ids_by_role: dict[str, Any] = Field(default_factory=dict)
    v2_package_mode: str | None = None
    selected_package_mode: str | None = None
    verbose_context_chars: int | None = None
    compact_context_chars: int | None = None
    selected_context_chars: int | None = None
    verbose_context_size: dict[str, Any] = Field(default_factory=dict)
    compact_context_size: dict[str, Any] = Field(default_factory=dict)
    selected_context_size: dict[str, Any] = Field(default_factory=dict)
    chars_by_section: dict[str, Any] = Field(default_factory=dict)
    avg_chars_per_candidate_by_role: dict[str, Any] = Field(default_factory=dict)
    removed_verbose_fields: list[str] = Field(default_factory=list)
    removed_verbose_field_counts: dict[str, Any] = Field(default_factory=dict)
    compact_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    compact_candidate_total: int = 0
    compact_candidate_ids_hash: str | None = None
    compact_package_full_matrix_used: bool = False
    package_candidate_loss: bool = False
    provider_context_limit_retry_compact_attempted: bool = False
    provider_context_limit_retry_compact_success: bool = False
    provider_context_limit_original_chars: int | None = None
    provider_context_limit_compact_chars: int | None = None
    provider_context_limit_after_compact: bool = False
    dropped_before_composer_count_by_role: dict[str, Any] = Field(default_factory=dict)
    dropped_before_composer_reason_by_role: dict[str, Any] = Field(default_factory=dict)
    package_candidate_exposure_ratio_by_role: dict[str, Any] = Field(default_factory=dict)
    original_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    fallback_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    dropped_before_fallback_count_by_role: dict[str, Any] = Field(default_factory=dict)
    dropped_before_fallback_reasons: dict[str, Any] = Field(default_factory=dict)
    timeout_fallback_coverage_ratio_by_role: dict[str, Any] = Field(default_factory=dict)
    package_candidate_exposure_policy: dict[str, Any] = Field(default_factory=dict)
    package_candidate_exposure_incomplete: bool = False
    package_candidate_exposure_incomplete_roles: list[str] = Field(default_factory=list)
    package_exposure_blocking_lifecycle_roles: list[str] = Field(default_factory=list)
    package_budget: dict[str, Any] = Field(default_factory=dict)
    package_budget_warnings: list[str] = Field(default_factory=list)
    package_approximate_size: dict[str, Any] = Field(default_factory=dict)
    package_skipped_reason: str | None = None
    ready_candidates_excluded_reason: str | None = None
    composer_attempt_decision: dict[str, Any] = Field(default_factory=dict)
    composer_mode: str | None = None
    expected_composer_mode: str | None = None
    llm_call_count: int = 0
    llm_call_stages: list[str] = Field(default_factory=list)
    llm_call_budget_exceeded: bool = False
    max_llm_calls_per_match: int | None = None
    requirement_contract: dict[str, Any] = Field(default_factory=dict)
    requirement_contract_used: bool = False
    main_composer_used: bool = False
    critic_used: bool = False
    repair_used: bool = False
    role_evaluation_used: bool = False
    role_evaluation_skipped_reason: str | None = None
    role_evaluation_count_by_role: dict[str, Any] = Field(default_factory=dict)
    role_evaluation_coverage_by_role: dict[str, Any] = Field(default_factory=dict)
    role_evaluation_failed_chunks: list[dict[str, Any]] = Field(default_factory=list)
    bom_composer_used: bool = False
    completeness_critic_used: bool = False
    completeness_critic_result: dict[str, Any] = Field(default_factory=dict)
    repair_composer_used: bool = False
    final_bom_after_repair: dict[str, Any] = Field(default_factory=dict)
    pre_composer_requirement_classifier_status: str | None = None
    pre_composer_requirement_source_coverage_percent: float | None = None
    pre_composer_unclassified_source_fragments: list[str] = Field(default_factory=list)
    pre_composer_semantic_diagnostics_are_blocking: bool = False
    composer_requirement_analysis: dict[str, Any] = Field(default_factory=dict)
    composer_fulfillment_decisions: list[dict[str, Any]] = Field(default_factory=list)
    composer_source_coverage_summary: dict[str, Any] = Field(default_factory=dict)
    composer_assumptions: list[str] = Field(default_factory=list)
    composer_engineer_checks: list[str] = Field(default_factory=list)
    composer_hard_mismatch_risks: list[dict[str, Any]] = Field(default_factory=list)
    composer_unverified_requirements: list[dict[str, Any]] = Field(default_factory=list)
    composer_considered_candidate_count_by_role: dict[str, Any] = Field(
        default_factory=dict
    )
    composer_chosen_candidate_ids: list[str] = Field(default_factory=list)
    validation_hard_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    validation_unverified_requirements: list[dict[str, Any]] = Field(default_factory=list)
    final_status_source: str | None = None
    package_strategy_decision: dict[str, Any] = Field(default_factory=dict)
    match_trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    component_candidate_matrix: dict[str, Any] = Field(default_factory=dict)
    shortlist_for_llm: list[dict[str, Any]] = Field(default_factory=list)
    llm_configurator_enabled: bool = False
    llm_configurator_used: bool = False
    output_mode: str | None = None
    llm_configurator_output_mode: str | None = None
    ai_recommendation_mode: str | None = None
    ai_recommendations_count: int = 0
    ai_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    llm_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    llm_recommended_build_candidates: list[MatchSummaryCandidate] = Field(default_factory=list)
    primary_recommendation: dict[str, Any] = Field(default_factory=dict)
    primary_recommendation_status: str | None = None
    no_recommendation_reason: dict[str, Any] = Field(default_factory=dict)
    partial_available_components: list[dict[str, Any]] = Field(default_factory=list)
    failed_requirements: list[Any] = Field(default_factory=list)
    role_failures: list[dict[str, Any]] = Field(default_factory=list)
    unverified_requirements: list[Any] = Field(default_factory=list)
    hard_mismatch_risks: list[Any] = Field(default_factory=list)
    recommended_next_actions: list[str] = Field(default_factory=list)
    engineer_checks: list[str] = Field(default_factory=list)
    composer_summary_ru: str | None = None
    customer_safe_summary_ru: str | None = None
    commercial_summary: dict[str, Any] = Field(default_factory=dict)
    grouped_presales_mode_used: bool = False
    configuration_groups: list[dict[str, Any]] = Field(default_factory=list)
    configuration_groups_count: int = 0
    quote_recommendation: dict[str, Any] = Field(default_factory=dict)
    selected_configuration_group_id: str | None = None
    selected_platform_option_id: str | None = None
    selected_platform_option_index: int | None = None
    llm_general_notes: list[str] = Field(default_factory=list)
    llm_fallback_reason: str | None = None
    llm_error_type: str | None = None
    llm_http_status: int | None = None
    llm_repair_used: bool = False
    llm_repair_attempted: bool = False
    llm_repair_success: bool = False
    llm_repair_fallback_reason: str | None = None
    llm_repair_critique_count: int = 0
    llm_repair_critique_summary: list[str] = Field(default_factory=list)
    llm_repair_blocked_critique_count: int = 0
    llm_repair_blocked_critique_summary: list[str] = Field(default_factory=list)
    llm_repair_savings_estimate: str | None = None
    llm_repair_revised_proposals_count: int = 0
    llm_repair_validation_summary: dict[str, Any] = Field(default_factory=dict)
    llm_thinking_diagnostics: dict[str, Any] = Field(default_factory=dict)
    llm_thinking_enabled: bool = False
    llm_thinking_budget_tokens: int | None = None
    llm_thinking_fallback_reason: str | None = None
    llm_proposals_count: int = 0
    valid_proposals_count: int = 0
    validation_rejected_count: int = 0
    selection_skipped_count: int = 0
    rejected_ai_recommendations_count: int = 0
    ai_recommendations_validation_warnings: list[str] = Field(default_factory=list)
    ai_validation_summary: dict[str, Any] = Field(default_factory=dict)
    rejected_reasons_top: list[dict[str, Any]] = Field(default_factory=list)
    rejected_ai_recommendations_debug_safe: list[dict[str, Any]] = Field(
        default_factory=list
    )
    rejected_ai_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    web_evidence_pack: dict[str, Any] = Field(default_factory=dict)
    web_evidence_diagnostics: dict[str, Any] = Field(default_factory=dict)
    evidence_mode: str | None = None
    online_composer_used: bool = False
    evidence_used: bool = False
    evidence_sources_count: int = 0
    evidence_status_summary: dict[str, Any] = Field(default_factory=dict)
    online_composer_error_type: str | None = None
    online_composer_parse_status: str | None = None
    online_composer_empty_response_repair_attempted: bool = False
    online_composer_empty_response_repair_success: bool = False
    structured_no_recommendation_used: bool = False
    evidence_requests_count: int = 0
    llm_evidence_review: dict[str, Any] = Field(default_factory=dict)
    report_markdown: str
    report_url: str
    report_xlsx_url: str


class MatchCandidateBrief(BaseModel):
    id: int
    candidate_id: str | None = None
    candidate_type: str = "ready_server"
    distributor_code: str
    item_id: str
    product_key: str | None
    part_number: str | None
    producer: str | None
    category_id: str | None
    item_name: str | None
    confidence_score: int
    price_value: str | None
    price_currency: str | None
    available_quantity: int | None
    reservable_locations: int
    matched_requirements: list[str]
    missing_requirements: list[str]
    risk_flags: list[str]
    platform: dict[str, Any] = Field(default_factory=dict)
    components: list[dict[str, Any]] = Field(default_factory=list)
    total_price_value: str | None = None
    total_price_currency: str | None = None
    missing_components: list[str] = Field(default_factory=list)
    compatibility_warnings: list[str] = Field(default_factory=list)
    engineer_review_required: bool = True
    completeness_status: str | None = None
    completeness_label: str | None = None
    included_component_roles: list[str] = Field(default_factory=list)
    missing_component_roles: list[str] = Field(default_factory=list)
    excluded_from_total_roles: list[str] = Field(default_factory=list)
    cpu_per_server: int | None = None
    total_cpu_required: int | None = None
    total_price_note: str | None = None
    score: int | None = None
    rank_reason: list[str] = Field(default_factory=list)
    optimization_mode: str | None = None
    requirement_fit: str | None = None
    right_size_note: str | None = None
    cpu_over_requirement: int | None = None
    storage_over_requirement: float | None = None
    ram_overage_gb: int | None = None
    overfit_reason: str | None = None


class MatchRunResponse(BaseModel):
    id: int
    status: str
    engineer_review_required: bool
    total_candidates: int
    matched_items: int
    spec_json: dict[str, Any]
    report_json: dict[str, Any]
    risk_flags: list[str]
    missing_requirements: list[str]
    candidates: list[MatchCandidateBrief]
    ready_stock_candidates: list[MatchCandidateBrief]
    build_candidates: list[MatchCandidateBrief]
    product_group: str | None = None
    primary_object: str | None = None
    semantic_planner_source: str | None = None
    semantic_planner_used: bool = False
    semantic_planner_confidence: str | None = None
    semantic_planner_error_type: str | None = None
    semantic_planner_http_status: int | None = None
    semantic_planner_parse_status: str | None = None
    semantic_planner_fallback_reason: str | None = None
    semantic_planner_attempts: list[dict[str, Any]] = Field(default_factory=list)
    semantic_planner_stage: str | None = None
    semantic_planner_stage_timeouts: list[dict[str, Any]] = Field(default_factory=list)
    semantic_planner_timeout_reason: str | None = None
    semantic_planner_timeout_seconds: float | None = None
    semantic_planner_elapsed_ms: int | None = None
    semantic_planner_repair_attempted: bool = False
    semantic_planner_repair_success: bool = False
    semantic_planner_minimal_router_used: bool = False
    semantic_planner_minimal_fallback_used: bool = False
    semantic_planner_empty_response_count: int = 0
    semantic_planner_empty_response_reason: str | None = None
    requirement_classifier_status: str | None = None
    requirement_classifier_error_type: str | None = None
    requirement_classifier_parse_status: str | None = None
    requirement_classifier_incomplete_reason: str | None = None
    requirement_source_coverage: list[dict[str, Any]] = Field(default_factory=list)
    requirement_source_coverage_percent: float | None = None
    unclassified_source_fragments: list[str] = Field(default_factory=list)
    synthetic_requirement_count: int = 0
    source_backed_requirement_count: int = 0
    requirement_classifier_repair_quality: str | None = None
    requirement_classifier_repair_accepted: bool = False
    semantic_planner_model: str | None = None
    semantic_planner_provider: str | None = None
    candidate_universe_planner_mode: str | None = None
    primary_product_group: str | None = None
    procurement_intent: str | None = None
    selected_group_reason: str | None = None
    selected_product_group_reason: str | None = None
    competing_product_groups: list[dict[str, Any]] = Field(default_factory=list)
    primary_object_indicators: list[Any] = Field(default_factory=list)
    component_role_indicators: list[dict[str, Any]] = Field(default_factory=list)
    excluded_category_groups: list[dict[str, Any]] = Field(default_factory=list)
    planner_repair_attempted: bool = False
    planner_repair_success: bool = False
    planner_suspicion_reasons: list[str] = Field(default_factory=list)
    deterministic_product_group_hint: str | None = None
    semantic_planner_disagreement: bool = False
    matrix_blueprint: dict[str, Any] = Field(default_factory=dict)
    matrix_blueprint_roles: list[str] = Field(default_factory=list)
    stage_a_broad_roles: list[str] = Field(default_factory=list)
    semantic_matrix_blueprint_roles: list[str] = Field(default_factory=list)
    requirement_classifier_roles: list[str] = Field(default_factory=list)
    effective_matrix_roles_before_category_planner: list[str] = Field(default_factory=list)
    category_planner_input_roles: list[str] = Field(default_factory=list)
    category_planner_output_roles: list[str] = Field(default_factory=list)
    validated_category_plan_roles: list[str] = Field(default_factory=list)
    materialized_matrix_roles: list[str] = Field(default_factory=list)
    composer_package_roles: list[str] = Field(default_factory=list)
    roles_dropped_after_stage_a: list[str] = Field(default_factory=list)
    roles_dropped_before_category_planner: list[str] = Field(default_factory=list)
    roles_dropped_after_category_planner: list[str] = Field(default_factory=list)
    roles_dropped_during_materialization: list[str] = Field(default_factory=list)
    roles_dropped_reason_by_role: dict[str, Any] = Field(default_factory=dict)
    role_source_by_role: dict[str, Any] = Field(default_factory=dict)
    role_lifecycle_trace: list[dict[str, Any]] = Field(default_factory=list)
    embedded_requirements: list[Any] = Field(default_factory=list)
    classified_requirements: list[dict[str, Any]] = Field(default_factory=list)
    purchasable_role_requirements: list[dict[str, Any]] = Field(default_factory=list)
    primary_object_feature_requirements: list[dict[str, Any]] = Field(default_factory=list)
    accessory_or_consumable_requirements: list[dict[str, Any]] = Field(default_factory=list)
    service_or_support_requirements: list[dict[str, Any]] = Field(default_factory=list)
    logistics_or_commercial_constraints: list[dict[str, Any]] = Field(default_factory=list)
    engineering_check_requirements: list[dict[str, Any]] = Field(default_factory=list)
    unmapped_requirements_non_blocking: list[dict[str, Any]] = Field(default_factory=list)
    unmapped_requirements_blocking: list[dict[str, Any]] = Field(default_factory=list)
    requirement_role_mapping_decision: list[dict[str, Any]] = Field(default_factory=list)
    requirement_fulfillment_decision: list[dict[str, Any]] = Field(default_factory=list)
    not_primary_product_groups: list[Any] = Field(default_factory=list)
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    required_capabilities: list[dict[str, Any]] = Field(default_factory=list)
    optional_capabilities: list[dict[str, Any]] = Field(default_factory=list)
    workload_context: list[str] = Field(default_factory=list)
    logistics_constraints: dict[str, Any] = Field(default_factory=dict)
    commercial_instructions: list[dict[str, Any]] = Field(default_factory=list)
    response_instructions: list[str] = Field(default_factory=list)
    unsupported_or_unmapped_requirements: list[str] = Field(default_factory=list)
    role_plan: dict[str, Any] = Field(default_factory=dict)
    category_plan: dict[str, Any] = Field(default_factory=dict)
    category_plan_entries: list[dict[str, Any]] = Field(default_factory=list)
    category_catalog_summary: dict[str, Any] = Field(default_factory=dict)
    category_planner_source: str | None = None
    category_plan_source: str | None = None
    category_planner_missing_required_roles: list[str] = Field(default_factory=list)
    category_planner_repair_attempted: bool = False
    category_planner_repair_success: bool = False
    category_planner_repair_reason: str | None = None
    category_planner_repaired_roles: list[str] = Field(default_factory=list)
    category_planner_unresolved_required_roles: list[str] = Field(default_factory=list)
    required_roles: list[str] = Field(default_factory=list)
    missing_required_roles: list[str] = Field(default_factory=list)
    missing_category_roles: list[str] = Field(default_factory=list)
    missing_required_capabilities: list[dict[str, Any]] = Field(default_factory=list)
    role_coverage_summary: dict[str, Any] = Field(default_factory=dict)
    category_plan_warnings: list[str] = Field(default_factory=list)
    matrix_distiller_used: bool = False
    matrix_distiller_source: str | None = None
    matrix_distiller_diagnostics: dict[str, Any] = Field(default_factory=dict)
    broad_count_by_role: dict[str, Any] = Field(default_factory=dict)
    distilled_count_by_role: dict[str, Any] = Field(default_factory=dict)
    full_matrix_evaluation_used: bool = False
    full_matrix_evaluation_fallback_reason: str | None = None
    provider_error_type: str | None = None
    provider_context_limit: dict[str, Any] = Field(default_factory=dict)
    role_chunk_count_by_role: dict[str, Any] = Field(default_factory=dict)
    evaluated_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    selected_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    role_reducer_summary: dict[str, Any] = Field(default_factory=dict)
    full_matrix_failed_chunks: list[dict[str, Any]] = Field(default_factory=list)
    bom_critic_used: bool = False
    no_recommendation_coverage: dict[str, Any] = Field(default_factory=dict)
    no_recommendation_coverage_gate_passed: bool = False
    no_recommendation_coverage_repair_attempted: bool = False
    no_recommendation_coverage_repair_success: bool = False
    no_recommendation_coverage_rejected: bool = False
    no_recommendation_coverage_thresholds: dict[str, Any] = Field(default_factory=dict)
    no_recommendation_coverage_repair_reason: str | None = None
    llm_cost_diagnostics: dict[str, Any] = Field(default_factory=dict)
    count_by_role: dict[str, Any] = Field(default_factory=dict)
    broad_matrix_count_by_role: dict[str, Any] = Field(default_factory=dict)
    composer_package_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    composer_package_candidate_total: int = 0
    composer_package_candidate_ids_by_role: dict[str, Any] = Field(default_factory=dict)
    v2_package_mode: str | None = None
    selected_package_mode: str | None = None
    verbose_context_chars: int | None = None
    compact_context_chars: int | None = None
    selected_context_chars: int | None = None
    verbose_context_size: dict[str, Any] = Field(default_factory=dict)
    compact_context_size: dict[str, Any] = Field(default_factory=dict)
    selected_context_size: dict[str, Any] = Field(default_factory=dict)
    chars_by_section: dict[str, Any] = Field(default_factory=dict)
    avg_chars_per_candidate_by_role: dict[str, Any] = Field(default_factory=dict)
    removed_verbose_fields: list[str] = Field(default_factory=list)
    removed_verbose_field_counts: dict[str, Any] = Field(default_factory=dict)
    compact_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    compact_candidate_total: int = 0
    compact_candidate_ids_hash: str | None = None
    compact_package_full_matrix_used: bool = False
    package_candidate_loss: bool = False
    provider_context_limit_retry_compact_attempted: bool = False
    provider_context_limit_retry_compact_success: bool = False
    provider_context_limit_original_chars: int | None = None
    provider_context_limit_compact_chars: int | None = None
    provider_context_limit_after_compact: bool = False
    dropped_before_composer_count_by_role: dict[str, Any] = Field(default_factory=dict)
    dropped_before_composer_reason_by_role: dict[str, Any] = Field(default_factory=dict)
    package_candidate_exposure_ratio_by_role: dict[str, Any] = Field(default_factory=dict)
    original_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    fallback_candidate_count_by_role: dict[str, Any] = Field(default_factory=dict)
    dropped_before_fallback_count_by_role: dict[str, Any] = Field(default_factory=dict)
    dropped_before_fallback_reasons: dict[str, Any] = Field(default_factory=dict)
    timeout_fallback_coverage_ratio_by_role: dict[str, Any] = Field(default_factory=dict)
    package_candidate_exposure_policy: dict[str, Any] = Field(default_factory=dict)
    package_candidate_exposure_incomplete: bool = False
    package_candidate_exposure_incomplete_roles: list[str] = Field(default_factory=list)
    package_exposure_blocking_lifecycle_roles: list[str] = Field(default_factory=list)
    package_budget: dict[str, Any] = Field(default_factory=dict)
    package_budget_warnings: list[str] = Field(default_factory=list)
    package_approximate_size: dict[str, Any] = Field(default_factory=dict)
    package_skipped_reason: str | None = None
    ready_candidates_excluded_reason: str | None = None
    composer_attempt_decision: dict[str, Any] = Field(default_factory=dict)
    composer_mode: str | None = None
    expected_composer_mode: str | None = None
    llm_call_count: int = 0
    llm_call_stages: list[str] = Field(default_factory=list)
    llm_call_budget_exceeded: bool = False
    max_llm_calls_per_match: int | None = None
    requirement_contract: dict[str, Any] = Field(default_factory=dict)
    requirement_contract_used: bool = False
    main_composer_used: bool = False
    critic_used: bool = False
    repair_used: bool = False
    role_evaluation_used: bool = False
    role_evaluation_skipped_reason: str | None = None
    role_evaluation_count_by_role: dict[str, Any] = Field(default_factory=dict)
    role_evaluation_coverage_by_role: dict[str, Any] = Field(default_factory=dict)
    role_evaluation_failed_chunks: list[dict[str, Any]] = Field(default_factory=list)
    pre_composer_requirement_classifier_status: str | None = None
    pre_composer_requirement_source_coverage_percent: float | None = None
    pre_composer_unclassified_source_fragments: list[str] = Field(default_factory=list)
    pre_composer_semantic_diagnostics_are_blocking: bool = False
    composer_requirement_analysis: dict[str, Any] = Field(default_factory=dict)
    composer_fulfillment_decisions: list[dict[str, Any]] = Field(default_factory=list)
    composer_source_coverage_summary: dict[str, Any] = Field(default_factory=dict)
    validation_hard_mismatches: list[dict[str, Any]] = Field(default_factory=list)
    validation_unverified_requirements: list[dict[str, Any]] = Field(default_factory=list)
    final_status_source: str | None = None
    package_strategy_decision: dict[str, Any] = Field(default_factory=dict)
    match_trace: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    component_candidate_matrix: dict[str, Any] = Field(default_factory=dict)
    shortlist_for_llm: list[dict[str, Any]] = Field(default_factory=list)
    llm_configurator_enabled: bool = False
    llm_configurator_used: bool = False
    output_mode: str | None = None
    llm_configurator_output_mode: str | None = None
    ai_recommendation_mode: str | None = None
    ai_recommendations_count: int = 0
    ai_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    llm_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    llm_recommended_build_candidates: list[MatchSummaryCandidate] = Field(default_factory=list)
    primary_recommendation: dict[str, Any] = Field(default_factory=dict)
    primary_recommendation_status: str | None = None
    no_recommendation_reason: dict[str, Any] = Field(default_factory=dict)
    commercial_summary: dict[str, Any] = Field(default_factory=dict)
    grouped_presales_mode_used: bool = False
    configuration_groups: list[dict[str, Any]] = Field(default_factory=list)
    configuration_groups_count: int = 0
    quote_recommendation: dict[str, Any] = Field(default_factory=dict)
    selected_configuration_group_id: str | None = None
    selected_platform_option_id: str | None = None
    selected_platform_option_index: int | None = None
    llm_general_notes: list[str] = Field(default_factory=list)
    llm_fallback_reason: str | None = None
    llm_error_type: str | None = None
    llm_http_status: int | None = None
    llm_repair_used: bool = False
    llm_repair_attempted: bool = False
    llm_repair_success: bool = False
    llm_repair_fallback_reason: str | None = None
    llm_repair_critique_count: int = 0
    llm_repair_critique_summary: list[str] = Field(default_factory=list)
    llm_repair_blocked_critique_count: int = 0
    llm_repair_blocked_critique_summary: list[str] = Field(default_factory=list)
    llm_repair_savings_estimate: str | None = None
    llm_repair_revised_proposals_count: int = 0
    llm_repair_validation_summary: dict[str, Any] = Field(default_factory=dict)
    llm_thinking_diagnostics: dict[str, Any] = Field(default_factory=dict)
    llm_thinking_enabled: bool = False
    llm_thinking_budget_tokens: int | None = None
    llm_thinking_fallback_reason: str | None = None
    llm_proposals_count: int = 0
    valid_proposals_count: int = 0
    validation_rejected_count: int = 0
    selection_skipped_count: int = 0
    rejected_ai_recommendations_count: int = 0
    ai_recommendations_validation_warnings: list[str] = Field(default_factory=list)
    ai_validation_summary: dict[str, Any] = Field(default_factory=dict)
    rejected_reasons_top: list[dict[str, Any]] = Field(default_factory=list)
    rejected_ai_recommendations_debug_safe: list[dict[str, Any]] = Field(
        default_factory=list
    )
    rejected_ai_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    web_evidence_pack: dict[str, Any] = Field(default_factory=dict)
    web_evidence_diagnostics: dict[str, Any] = Field(default_factory=dict)
    evidence_mode: str | None = None
    online_composer_used: bool = False
    evidence_used: bool = False
    evidence_sources_count: int = 0
    evidence_status_summary: dict[str, Any] = Field(default_factory=dict)
    online_composer_error_type: str | None = None
    online_composer_parse_status: str | None = None
    online_composer_empty_response_repair_attempted: bool = False
    online_composer_empty_response_repair_success: bool = False
    structured_no_recommendation_used: bool = False
    evidence_requests_count: int = 0
    llm_evidence_review: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class MatchRunListItem(BaseModel):
    id: int
    status: str
    total_candidates: int
    matched_items: int
    created_at: datetime
    source_text: str | None


class MatchRunListResponse(BaseModel):
    items: list[MatchRunListItem]


@router.post("/v3/full-category", response_model=V3FullCategoryQuoteResponse)
async def create_v3_full_category_quote(
    session: SessionDep,
    payload: V3FullCategoryQuoteRequest,
) -> V3FullCategoryQuoteResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request text is empty.",
        )

    distributor_code = payload.distributor_code.strip() or "ocs"
    try:
        result = await run_v3_full_category_quote(
            text=text,
            session=session,
            profile=payload.profile,
            category_ids=payload.category_ids,
            distributor_code=distributor_code,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        await _persist_v3_full_category_quote_result(
            session=session,
            source_text=text,
            result=result,
        )
    except Exception as exc:
        logger.exception("V3 full-category result persistence failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save v3 full-category result.",
        ) from exc

    return _v3_full_category_quote_response(result)


@router.post("/v3/simple-stock-quote", response_model=V3FullCategoryQuoteResponse)
async def create_simple_stock_quote(
    session: SessionDep,
    payload: V3FullCategoryQuoteRequest,
) -> V3FullCategoryQuoteResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request text is empty.",
        )

    distributor_code = payload.distributor_code.strip() or "ocs"
    try:
        result = await run_simple_stock_quote(
            text=text,
            session=session,
            profile=payload.profile,
            category_ids=payload.category_ids,
            distributor_code=distributor_code,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    try:
        await _persist_v3_full_category_quote_result(
            session=session,
            source_text=text,
            result=result,
        )
    except Exception as exc:
        logger.exception("Simple stock quote result persistence failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save simple stock quote result.",
        ) from exc

    return _v3_full_category_quote_response(result)


@router.post("", response_model=MatchSummaryResponse, status_code=status.HTTP_201_CREATED)
async def create_match(
    session: SessionDep,
    payload: MatchRequestBody = None,
    pipeline_v2: Annotated[bool | None, Query()] = None,
) -> MatchSummaryResponse:
    try:
        spec, source, source_text, confirmation_text = _resolve_match_request(payload)
        orchestrator_result = await run_ai_match_orchestrator(
            AiMatchOrchestratorRequest(
                text=source_text,
                spec=spec,
                pipeline_v2=(
                    pipeline_v2
                    if pipeline_v2 is not None
                    else (payload.pipeline_v2 if payload is not None else None)
                ),
            ),
            session,
            match_func=match_stock_spec,
        )
        match_result = orchestrator_result.match_result
        report_markdown = build_match_markdown_report(match_result)
        report_json = normalize_composer_report_json(orchestrator_result.report_json)

        repository = MatchRepository(session)
        match_run = await repository.create_match_run(
            MatchRunCreate(
                source=source,
                source_text=source_text,
                status=match_result.status,
                engineer_review_required=match_result.engineer_review_required,
                total_candidates=match_result.total_candidates,
                matched_items=match_result.matched_items,
                missing_requirements_json=match_result.missing_requirements,
                risk_flags_json=match_result.risk_flags,
                spec_json=spec.model_dump(mode="json", exclude_none=True),
                report_json=report_json,
                report_markdown=report_markdown,
                candidates=[
                    _candidate_to_create(candidate) for candidate in match_result.candidates
                ],
            )
        )
        report_json = {"match_run_id": match_run.id, **report_json}
        commercial_summary = report_json.get("commercial_summary")
        if isinstance(commercial_summary, dict):
            commercial_summary["match_run_id"] = match_run.id
        match_run.report_json = report_json
        await session.flush()
        await session.commit()
    except HTTPException:
        raise
    except LlmError as exc:
        await _rollback_safely(session)
        logger.info("Stock Spec extraction failed in match API: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Stock Spec extraction failed.",
        ) from exc
    except Exception as exc:
        await _rollback_safely(session)
        logger.exception("Match API request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Match request failed.",
        ) from exc

    return MatchSummaryResponse(
        match_run_id=match_run.id,
        status=match_result.status,
        engineer_review_required=match_result.engineer_review_required,
        total_candidates=match_result.total_candidates,
        matched_items=match_result.matched_items,
        confirmation_text=confirmation_text,
        risk_flags=match_result.risk_flags,
        missing_requirements=match_result.missing_requirements,
        candidates=[
            _candidate_result_to_summary(candidate) for candidate in match_result.candidates[:3]
        ],
        ready_stock_candidates=[
            _candidate_result_to_summary(candidate)
            for candidate in match_result.candidates
            if candidate.candidate_type == "ready_server"
        ][:3],
        build_candidates=[
            _candidate_result_to_summary(candidate)
            for candidate in match_result.candidates
            if candidate.candidate_type == "build_from_parts"
        ][:MAX_API_BUILD_CANDIDATES],
        product_group=_string_or_none(report_json.get("product_group")),
        primary_object=_string_or_none(report_json.get("primary_object")),
        semantic_planner_source=_string_or_none(
            report_json.get("semantic_planner_source")
        ),
        semantic_planner_used=bool(report_json.get("semantic_planner_used")),
        semantic_planner_confidence=_string_or_none(
            report_json.get("semantic_planner_confidence")
        ),
        semantic_planner_error_type=_string_or_none(
            report_json.get("semantic_planner_error_type")
        ),
        semantic_planner_http_status=_int_or_none(
            report_json.get("semantic_planner_http_status")
        ),
        semantic_planner_parse_status=_string_or_none(
            report_json.get("semantic_planner_parse_status")
        ),
        semantic_planner_fallback_reason=_string_or_none(
            report_json.get("semantic_planner_fallback_reason")
        ),
        semantic_planner_attempts=_dict_list(
            report_json.get("semantic_planner_attempts")
        ),
        semantic_planner_stage=_string_or_none(
            report_json.get("semantic_planner_stage")
        ),
        semantic_planner_stage_timeouts=_dict_list(
            report_json.get("semantic_planner_stage_timeouts")
        ),
        semantic_planner_timeout_reason=_string_or_none(
            report_json.get("semantic_planner_timeout_reason")
        ),
        semantic_planner_timeout_seconds=_float_or_none(
            report_json.get("semantic_planner_timeout_seconds")
        ),
        semantic_planner_elapsed_ms=_int_or_none(
            report_json.get("semantic_planner_elapsed_ms")
        ),
        semantic_planner_repair_attempted=bool(
            report_json.get("semantic_planner_repair_attempted")
        ),
        semantic_planner_repair_success=bool(
            report_json.get("semantic_planner_repair_success")
        ),
        semantic_planner_minimal_router_used=bool(
            report_json.get("semantic_planner_minimal_router_used")
        ),
        semantic_planner_minimal_fallback_used=bool(
            report_json.get("semantic_planner_minimal_fallback_used")
        ),
        semantic_planner_empty_response_count=_int_or_none(
            report_json.get("semantic_planner_empty_response_count")
        )
        or 0,
        semantic_planner_empty_response_reason=_string_or_none(
            report_json.get("semantic_planner_empty_response_reason")
        ),
        requirement_classifier_status=_string_or_none(
            report_json.get("requirement_classifier_status")
        ),
        requirement_classifier_error_type=_string_or_none(
            report_json.get("requirement_classifier_error_type")
        ),
        requirement_classifier_parse_status=_string_or_none(
            report_json.get("requirement_classifier_parse_status")
        ),
        requirement_classifier_incomplete_reason=_string_or_none(
            report_json.get("requirement_classifier_incomplete_reason")
        ),
        requirement_source_coverage=_dict_list(
            report_json.get("requirement_source_coverage")
        ),
        requirement_source_coverage_percent=_float_or_none(
            report_json.get("requirement_source_coverage_percent")
        ),
        unclassified_source_fragments=_string_list(
            report_json.get("unclassified_source_fragments")
        ),
        synthetic_requirement_count=_int_or_none(
            report_json.get("synthetic_requirement_count")
        )
        or 0,
        source_backed_requirement_count=_int_or_none(
            report_json.get("source_backed_requirement_count")
        )
        or 0,
        requirement_classifier_repair_quality=_string_or_none(
            report_json.get("requirement_classifier_repair_quality")
        ),
        requirement_classifier_repair_accepted=bool(
            report_json.get("requirement_classifier_repair_accepted")
        ),
        semantic_planner_model=_string_or_none(report_json.get("semantic_planner_model")),
        semantic_planner_provider=_string_or_none(
            report_json.get("semantic_planner_provider")
        ),
        candidate_universe_planner_mode=_string_or_none(
            report_json.get("candidate_universe_planner_mode")
        ),
        primary_product_group=_string_or_none(
            report_json.get("primary_product_group")
        ),
        procurement_intent=_string_or_none(report_json.get("procurement_intent")),
        selected_group_reason=_string_or_none(
            report_json.get("selected_group_reason")
        ),
        selected_product_group_reason=_string_or_none(
            report_json.get("selected_product_group_reason")
        ),
        competing_product_groups=_dict_list(
            report_json.get("competing_product_groups")
        ),
        primary_object_indicators=_list_or_empty(
            report_json.get("primary_object_indicators")
        ),
        component_role_indicators=_dict_list(
            report_json.get("component_role_indicators")
        ),
        excluded_category_groups=_dict_list(
            report_json.get("excluded_category_groups")
        ),
        planner_repair_attempted=bool(report_json.get("planner_repair_attempted")),
        planner_repair_success=bool(report_json.get("planner_repair_success")),
        planner_suspicion_reasons=_string_list(
            report_json.get("planner_suspicion_reasons")
        ),
        deterministic_product_group_hint=_string_or_none(
            report_json.get("deterministic_product_group_hint")
        ),
        semantic_planner_disagreement=bool(
            report_json.get("semantic_planner_disagreement")
        ),
        matrix_blueprint=_dict_or_empty(report_json.get("matrix_blueprint")),
        matrix_blueprint_roles=_string_list(report_json.get("matrix_blueprint_roles")),
        stage_a_broad_roles=_string_list(report_json.get("stage_a_broad_roles")),
        semantic_matrix_blueprint_roles=_string_list(
            report_json.get("semantic_matrix_blueprint_roles")
        ),
        requirement_classifier_roles=_string_list(
            report_json.get("requirement_classifier_roles")
        ),
        effective_matrix_roles_before_category_planner=_string_list(
            report_json.get("effective_matrix_roles_before_category_planner")
        ),
        category_planner_input_roles=_string_list(
            report_json.get("category_planner_input_roles")
        ),
        category_planner_output_roles=_string_list(
            report_json.get("category_planner_output_roles")
        ),
        validated_category_plan_roles=_string_list(
            report_json.get("validated_category_plan_roles")
        ),
        materialized_matrix_roles=_string_list(
            report_json.get("materialized_matrix_roles")
        ),
        composer_package_roles=_string_list(report_json.get("composer_package_roles")),
        roles_dropped_after_stage_a=_string_list(
            report_json.get("roles_dropped_after_stage_a")
        ),
        roles_dropped_before_category_planner=_string_list(
            report_json.get("roles_dropped_before_category_planner")
        ),
        roles_dropped_after_category_planner=_string_list(
            report_json.get("roles_dropped_after_category_planner")
        ),
        roles_dropped_during_materialization=_string_list(
            report_json.get("roles_dropped_during_materialization")
        ),
        roles_dropped_reason_by_role=_dict_or_empty(
            report_json.get("roles_dropped_reason_by_role")
        ),
        role_source_by_role=_dict_or_empty(report_json.get("role_source_by_role")),
        role_lifecycle_trace=_dict_list(report_json.get("role_lifecycle_trace")),
        embedded_requirements=_list_or_empty(report_json.get("embedded_requirements")),
        classified_requirements=_dict_list(report_json.get("classified_requirements")),
        purchasable_role_requirements=_dict_list(
            report_json.get("purchasable_role_requirements")
        ),
        primary_object_feature_requirements=_dict_list(
            report_json.get("primary_object_feature_requirements")
        ),
        accessory_or_consumable_requirements=_dict_list(
            report_json.get("accessory_or_consumable_requirements")
        ),
        service_or_support_requirements=_dict_list(
            report_json.get("service_or_support_requirements")
        ),
        logistics_or_commercial_constraints=_dict_list(
            report_json.get("logistics_or_commercial_constraints")
        ),
        engineering_check_requirements=_dict_list(
            report_json.get("engineering_check_requirements")
        ),
        unmapped_requirements_non_blocking=_dict_list(
            report_json.get("unmapped_requirements_non_blocking")
        ),
        unmapped_requirements_blocking=_dict_list(
            report_json.get("unmapped_requirements_blocking")
        ),
        requirement_role_mapping_decision=_dict_list(
            report_json.get("requirement_role_mapping_decision")
        ),
        requirement_fulfillment_decision=_dict_list(
            report_json.get("requirement_fulfillment_decision")
        ),
        not_primary_product_groups=_list_or_empty(
            report_json.get("not_primary_product_groups")
        ),
        requirements=_dict_list(report_json.get("requirements")),
        required_capabilities=_dict_list(report_json.get("required_capabilities")),
        optional_capabilities=_dict_list(report_json.get("optional_capabilities")),
        workload_context=_string_list(report_json.get("workload_context")),
        logistics_constraints=_dict_or_empty(report_json.get("logistics_constraints")),
        commercial_instructions=_dict_list(report_json.get("commercial_instructions")),
        response_instructions=_string_list(report_json.get("response_instructions")),
        unsupported_or_unmapped_requirements=_string_list(
            report_json.get("unsupported_or_unmapped_requirements")
        ),
        role_plan=_dict_or_empty(report_json.get("role_plan")),
        category_plan=_dict_or_empty(report_json.get("category_plan")),
        category_plan_entries=_dict_list(report_json.get("category_plan_entries")),
        category_catalog_summary=_dict_or_empty(
            report_json.get("category_catalog_summary")
        ),
        category_planner_source=_string_or_none(
            report_json.get("category_planner_source")
        ),
        category_plan_source=_string_or_none(report_json.get("category_plan_source")),
        category_planner_missing_required_roles=_string_list(
            report_json.get("category_planner_missing_required_roles")
        ),
        category_planner_repair_attempted=bool(
            report_json.get("category_planner_repair_attempted")
        ),
        category_planner_repair_success=bool(
            report_json.get("category_planner_repair_success")
        ),
        category_planner_repair_reason=_string_or_none(
            report_json.get("category_planner_repair_reason")
        ),
        category_planner_repaired_roles=_string_list(
            report_json.get("category_planner_repaired_roles")
        ),
        category_planner_unresolved_required_roles=_string_list(
            report_json.get("category_planner_unresolved_required_roles")
        ),
        required_roles=_string_list(report_json.get("required_roles")),
        missing_required_roles=_string_list(report_json.get("missing_required_roles")),
        missing_category_roles=_string_list(report_json.get("missing_category_roles")),
        missing_required_capabilities=_dict_list(
            report_json.get("missing_required_capabilities")
        ),
        role_coverage_summary=_dict_or_empty(report_json.get("role_coverage_summary")),
        category_plan_warnings=_string_list(report_json.get("category_plan_warnings")),
        matrix_distiller_used=bool(report_json.get("matrix_distiller_used")),
        matrix_distiller_source=_string_or_none(report_json.get("matrix_distiller_source")),
        matrix_distiller_diagnostics=_dict_or_empty(
            report_json.get("matrix_distiller_diagnostics")
        ),
        broad_count_by_role=_dict_or_empty(report_json.get("broad_count_by_role")),
        distilled_count_by_role=_dict_or_empty(report_json.get("distilled_count_by_role")),
        full_matrix_evaluation_used=bool(
            report_json.get("full_matrix_evaluation_used")
        ),
        full_matrix_evaluation_fallback_reason=_string_or_none(
            report_json.get("full_matrix_evaluation_fallback_reason")
        ),
        provider_error_type=_string_or_none(report_json.get("provider_error_type")),
        provider_context_limit=_dict_or_empty(
            report_json.get("provider_context_limit")
        ),
        role_chunk_count_by_role=_dict_or_empty(
            report_json.get("role_chunk_count_by_role")
        ),
        evaluated_candidate_count_by_role=_dict_or_empty(
            report_json.get("evaluated_candidate_count_by_role")
        ),
        selected_candidate_count_by_role=_dict_or_empty(
            report_json.get("selected_candidate_count_by_role")
        ),
        role_reducer_summary=_dict_or_empty(report_json.get("role_reducer_summary")),
        full_matrix_failed_chunks=_dict_list(
            report_json.get("full_matrix_failed_chunks")
        ),
        bom_critic_used=bool(report_json.get("bom_critic_used")),
        no_recommendation_coverage=_dict_or_empty(
            report_json.get("no_recommendation_coverage")
        ),
        no_recommendation_coverage_gate_passed=bool(
            report_json.get("no_recommendation_coverage_gate_passed")
        ),
        no_recommendation_coverage_repair_attempted=bool(
            report_json.get("no_recommendation_coverage_repair_attempted")
        ),
        no_recommendation_coverage_repair_success=bool(
            report_json.get("no_recommendation_coverage_repair_success")
        ),
        no_recommendation_coverage_rejected=bool(
            report_json.get("no_recommendation_coverage_rejected")
        ),
        no_recommendation_coverage_thresholds=_dict_or_empty(
            report_json.get("no_recommendation_coverage_thresholds")
        ),
        no_recommendation_coverage_repair_reason=_string_or_none(
            report_json.get("no_recommendation_coverage_repair_reason")
        ),
        llm_cost_diagnostics=_dict_or_empty(report_json.get("llm_cost_diagnostics")),
        count_by_role=_dict_or_empty(report_json.get("count_by_role")),
        broad_matrix_count_by_role=_dict_or_empty(
            report_json.get("broad_matrix_count_by_role")
        ),
        composer_package_candidate_count_by_role=_dict_or_empty(
            report_json.get("composer_package_candidate_count_by_role")
        ),
        composer_package_candidate_total=_int_or_none(
            report_json.get("composer_package_candidate_total")
        )
        or 0,
        composer_package_candidate_ids_by_role=_dict_or_empty(
            report_json.get("composer_package_candidate_ids_by_role")
        ),
        v2_package_mode=_string_or_none(report_json.get("v2_package_mode")),
        selected_package_mode=_string_or_none(report_json.get("selected_package_mode")),
        verbose_context_chars=_int_or_none(report_json.get("verbose_context_chars")),
        compact_context_chars=_int_or_none(report_json.get("compact_context_chars")),
        selected_context_chars=_int_or_none(report_json.get("selected_context_chars")),
        verbose_context_size=_dict_or_empty(report_json.get("verbose_context_size")),
        compact_context_size=_dict_or_empty(report_json.get("compact_context_size")),
        selected_context_size=_dict_or_empty(report_json.get("selected_context_size")),
        chars_by_section=_dict_or_empty(report_json.get("chars_by_section")),
        avg_chars_per_candidate_by_role=_dict_or_empty(
            report_json.get("avg_chars_per_candidate_by_role")
        ),
        removed_verbose_fields=_string_list(report_json.get("removed_verbose_fields")),
        removed_verbose_field_counts=_dict_or_empty(
            report_json.get("removed_verbose_field_counts")
        ),
        compact_candidate_count_by_role=_dict_or_empty(
            report_json.get("compact_candidate_count_by_role")
        ),
        compact_candidate_total=_int_or_none(
            report_json.get("compact_candidate_total")
        )
        or 0,
        compact_candidate_ids_hash=_string_or_none(
            report_json.get("compact_candidate_ids_hash")
        ),
        compact_package_full_matrix_used=bool(
            report_json.get("compact_package_full_matrix_used")
        ),
        package_candidate_loss=bool(report_json.get("package_candidate_loss")),
        provider_context_limit_retry_compact_attempted=bool(
            report_json.get("provider_context_limit_retry_compact_attempted")
        ),
        provider_context_limit_retry_compact_success=bool(
            report_json.get("provider_context_limit_retry_compact_success")
        ),
        provider_context_limit_original_chars=_int_or_none(
            report_json.get("provider_context_limit_original_chars")
        ),
        provider_context_limit_compact_chars=_int_or_none(
            report_json.get("provider_context_limit_compact_chars")
        ),
        provider_context_limit_after_compact=bool(
            report_json.get("provider_context_limit_after_compact")
        ),
        dropped_before_composer_count_by_role=_dict_or_empty(
            report_json.get("dropped_before_composer_count_by_role")
        ),
        dropped_before_composer_reason_by_role=_dict_or_empty(
            report_json.get("dropped_before_composer_reason_by_role")
        ),
        package_candidate_exposure_ratio_by_role=_dict_or_empty(
            report_json.get("package_candidate_exposure_ratio_by_role")
        ),
        original_candidate_count_by_role=_dict_or_empty(
            report_json.get("original_candidate_count_by_role")
        ),
        fallback_candidate_count_by_role=_dict_or_empty(
            report_json.get("fallback_candidate_count_by_role")
        ),
        dropped_before_fallback_count_by_role=_dict_or_empty(
            report_json.get("dropped_before_fallback_count_by_role")
        ),
        dropped_before_fallback_reasons=_dict_or_empty(
            report_json.get("dropped_before_fallback_reasons")
        ),
        timeout_fallback_coverage_ratio_by_role=_dict_or_empty(
            report_json.get("timeout_fallback_coverage_ratio_by_role")
        ),
        package_candidate_exposure_policy=_dict_or_empty(
            report_json.get("package_candidate_exposure_policy")
        ),
        package_candidate_exposure_incomplete=bool(
            report_json.get("package_candidate_exposure_incomplete")
        ),
        package_candidate_exposure_incomplete_roles=_string_list(
            report_json.get("package_candidate_exposure_incomplete_roles")
        ),
        package_exposure_blocking_lifecycle_roles=_string_list(
            report_json.get("package_exposure_blocking_lifecycle_roles")
        ),
        package_budget=_dict_or_empty(report_json.get("package_budget")),
        package_budget_warnings=_string_list(report_json.get("package_budget_warnings")),
        package_approximate_size=_dict_or_empty(
            report_json.get("package_approximate_size")
        ),
        package_skipped_reason=_string_or_none(report_json.get("package_skipped_reason")),
        ready_candidates_excluded_reason=_string_or_none(
            report_json.get("ready_candidates_excluded_reason")
        ),
        composer_attempt_decision=_dict_or_empty(
            report_json.get("composer_attempt_decision")
        ),
        composer_mode=_string_or_none(report_json.get("composer_mode")),
        expected_composer_mode=_string_or_none(
            report_json.get("expected_composer_mode")
        ),
        llm_call_count=_int_or_none(report_json.get("llm_call_count")) or 0,
        llm_call_stages=_string_list(report_json.get("llm_call_stages")),
        llm_call_budget_exceeded=bool(report_json.get("llm_call_budget_exceeded")),
        max_llm_calls_per_match=_int_or_none(
            report_json.get("max_llm_calls_per_match")
        ),
        requirement_contract=_dict_or_empty(report_json.get("requirement_contract")),
        requirement_contract_used=bool(report_json.get("requirement_contract_used")),
        main_composer_used=bool(report_json.get("main_composer_used")),
        critic_used=bool(report_json.get("critic_used")),
        repair_used=bool(report_json.get("repair_used")),
        role_evaluation_used=bool(report_json.get("role_evaluation_used")),
        role_evaluation_skipped_reason=_string_or_none(
            report_json.get("role_evaluation_skipped_reason")
        ),
        role_evaluation_count_by_role=_dict_or_empty(
            report_json.get("role_evaluation_count_by_role")
        ),
        role_evaluation_coverage_by_role=_dict_or_empty(
            report_json.get("role_evaluation_coverage_by_role")
        ),
        role_evaluation_failed_chunks=_dict_list(
            report_json.get("role_evaluation_failed_chunks")
        ),
        bom_composer_used=bool(report_json.get("bom_composer_used")),
        completeness_critic_used=bool(report_json.get("completeness_critic_used")),
        completeness_critic_result=_dict_or_empty(
            report_json.get("completeness_critic_result")
        ),
        repair_composer_used=bool(report_json.get("repair_composer_used")),
        final_bom_after_repair=_dict_or_empty(report_json.get("final_bom_after_repair")),
        pre_composer_requirement_classifier_status=_string_or_none(
            report_json.get("pre_composer_requirement_classifier_status")
        ),
        pre_composer_requirement_source_coverage_percent=_float_or_none(
            report_json.get("pre_composer_requirement_source_coverage_percent")
        ),
        pre_composer_unclassified_source_fragments=_string_list(
            report_json.get("pre_composer_unclassified_source_fragments")
        ),
        pre_composer_semantic_diagnostics_are_blocking=bool(
            report_json.get("pre_composer_semantic_diagnostics_are_blocking")
        ),
        composer_requirement_analysis=_dict_or_empty(
            report_json.get("composer_requirement_analysis")
        ),
        composer_fulfillment_decisions=_dict_list(
            report_json.get("composer_fulfillment_decisions")
        ),
        composer_source_coverage_summary=_dict_or_empty(
            report_json.get("composer_source_coverage_summary")
        ),
        validation_hard_mismatches=_dict_list(
            report_json.get("validation_hard_mismatches")
        ),
        validation_unverified_requirements=_dict_list(
            report_json.get("validation_unverified_requirements")
        ),
        final_status_source=_string_or_none(report_json.get("final_status_source")),
        package_strategy_decision=_dict_or_empty(
            report_json.get("package_strategy_decision")
        ),
        match_trace=_dict_list(report_json.get("match_trace")),
        diagnostics=_dict_or_empty(report_json.get("diagnostics")),
        component_candidate_matrix=report_json.get("component_candidate_matrix", {}),
        shortlist_for_llm=report_json.get("shortlist_for_llm", []),
        llm_configurator_enabled=bool(report_json.get("llm_configurator_enabled")),
        llm_configurator_used=bool(report_json.get("llm_configurator_used")),
        output_mode=_string_or_none(report_json.get("output_mode")),
        llm_configurator_output_mode=_string_or_none(
            report_json.get("llm_configurator_output_mode")
        ),
        ai_recommendation_mode=_string_or_none(report_json.get("ai_recommendation_mode")),
        ai_recommendations_count=_int_or_none(report_json.get("ai_recommendations_count")) or 0,
        ai_recommendations=_dict_list(report_json.get("ai_recommendations")),
        llm_recommendations=_dict_list(report_json.get("llm_recommendations")),
        llm_recommended_build_candidates=[
            _llm_candidate_to_summary(candidate)
            for candidate in _dict_list(report_json.get("llm_recommended_build_candidates"))
        ],
        primary_recommendation=_dict_or_empty(report_json.get("primary_recommendation")),
        primary_recommendation_status=_string_or_none(
            report_json.get("primary_recommendation_status")
        ),
        no_recommendation_reason=_dict_or_empty(
            report_json.get("no_recommendation_reason")
        ),
        commercial_summary=_dict_or_empty(report_json.get("commercial_summary")),
        grouped_presales_mode_used=bool(report_json.get("grouped_presales_mode_used")),
        configuration_groups=_dict_list(report_json.get("configuration_groups")),
        configuration_groups_count=_int_or_none(
            report_json.get("configuration_groups_count")
        )
        or 0,
        quote_recommendation=_dict_or_empty(report_json.get("quote_recommendation")),
        selected_configuration_group_id=_string_or_none(
            report_json.get("selected_configuration_group_id")
        ),
        selected_platform_option_id=_string_or_none(
            report_json.get("selected_platform_option_id")
        ),
        selected_platform_option_index=_int_or_none(
            report_json.get("selected_platform_option_index")
        ),
        llm_general_notes=_string_list(report_json.get("llm_general_notes")),
        llm_fallback_reason=_string_or_none(report_json.get("llm_fallback_reason")),
        llm_error_type=_string_or_none(report_json.get("llm_error_type")),
        llm_http_status=_int_or_none(report_json.get("llm_http_status")),
        llm_repair_used=bool(report_json.get("llm_repair_used")),
        llm_repair_attempted=bool(report_json.get("llm_repair_attempted")),
        llm_repair_success=bool(report_json.get("llm_repair_success")),
        llm_repair_fallback_reason=_string_or_none(
            report_json.get("llm_repair_fallback_reason")
        ),
        llm_repair_critique_count=_int_or_none(
            report_json.get("llm_repair_critique_count")
        )
        or 0,
        llm_repair_critique_summary=_string_list(
            report_json.get("llm_repair_critique_summary")
        ),
        llm_repair_blocked_critique_count=_int_or_none(
            report_json.get("llm_repair_blocked_critique_count")
        )
        or 0,
        llm_repair_blocked_critique_summary=_string_list(
            report_json.get("llm_repair_blocked_critique_summary")
        ),
        llm_repair_savings_estimate=_string_or_none(
            report_json.get("llm_repair_savings_estimate")
        ),
        llm_repair_revised_proposals_count=_int_or_none(
            report_json.get("llm_repair_revised_proposals_count")
        )
        or 0,
        llm_repair_validation_summary=_dict_or_empty(
            report_json.get("llm_repair_validation_summary")
        ),
        llm_thinking_diagnostics=_dict_or_empty(
            report_json.get("llm_thinking_diagnostics")
        ),
        llm_thinking_enabled=bool(report_json.get("llm_thinking_enabled")),
        llm_thinking_budget_tokens=_int_or_none(
            report_json.get("llm_thinking_budget_tokens")
        ),
        llm_thinking_fallback_reason=_string_or_none(
            report_json.get("llm_thinking_fallback_reason")
        ),
        llm_proposals_count=_int_or_none(report_json.get("llm_proposals_count")) or 0,
        valid_proposals_count=_int_or_none(report_json.get("valid_proposals_count")) or 0,
        validation_rejected_count=_int_or_none(
            report_json.get("validation_rejected_count")
        )
        or 0,
        selection_skipped_count=_int_or_none(report_json.get("selection_skipped_count"))
        or 0,
        rejected_ai_recommendations_count=_int_or_none(
            report_json.get("rejected_ai_recommendations_count")
        )
        or 0,
        ai_recommendations_validation_warnings=_string_list(
            report_json.get("ai_recommendations_validation_warnings")
        ),
        ai_validation_summary=_dict_or_empty(report_json.get("ai_validation_summary")),
        rejected_reasons_top=_dict_list(report_json.get("rejected_reasons_top")),
        rejected_ai_recommendations_debug_safe=_dict_list(
            report_json.get("rejected_ai_recommendations_debug_safe")
        ),
        rejected_ai_recommendations=_dict_list(
            report_json.get("rejected_ai_recommendations")
        ),
        web_evidence_pack=_dict_or_empty(report_json.get("web_evidence_pack")),
        web_evidence_diagnostics=_dict_or_empty(
            report_json.get("web_evidence_diagnostics")
        ),
        evidence_mode=_string_or_none(report_json.get("evidence_mode")),
        online_composer_used=bool(report_json.get("online_composer_used")),
        evidence_used=bool(report_json.get("evidence_used")),
        evidence_sources_count=_int_or_none(report_json.get("evidence_sources_count")) or 0,
        evidence_status_summary=_dict_or_empty(
            report_json.get("evidence_status_summary")
        ),
        online_composer_error_type=_string_or_none(
            report_json.get("online_composer_error_type")
        ),
        online_composer_parse_status=_string_or_none(
            report_json.get("online_composer_parse_status")
        ),
        online_composer_empty_response_repair_attempted=bool(
            report_json.get("online_composer_empty_response_repair_attempted")
        ),
        online_composer_empty_response_repair_success=bool(
            report_json.get("online_composer_empty_response_repair_success")
        ),
        structured_no_recommendation_used=bool(
            report_json.get("structured_no_recommendation_used")
        ),
        evidence_requests_count=_int_or_none(report_json.get("evidence_requests_count")) or 0,
        llm_evidence_review=_dict_or_empty(report_json.get("llm_evidence_review")),
        report_markdown=report_markdown,
        report_url=f"/api/v1/match/{match_run.id}/report.md",
        report_xlsx_url=f"/api/v1/match/{match_run.id}/report.xlsx",
    )


@router.get("", response_model=MatchRunListResponse)
async def list_matches(
    session: SessionDep,
    limit: LimitQuery = 10,
) -> MatchRunListResponse:
    try:
        repository = MatchRepository(session)
        match_runs = await repository.list_match_runs(limit=limit)
    except Exception as exc:
        logger.exception("Match run list request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not list match runs.",
        ) from exc

    return MatchRunListResponse(
        items=[
            MatchRunListItem(
                id=match_run.id,
                status=match_run.status,
                total_candidates=match_run.total_candidates,
                matched_items=match_run.matched_items,
                created_at=match_run.created_at,
                source_text=_brief(match_run.source_text),
            )
            for match_run in match_runs
        ]
    )


@router.get("/{match_run_id}", response_model=MatchRunResponse)
async def get_match(
    match_run_id: int,
    session: SessionDep,
) -> MatchRunResponse:
    try:
        repository = MatchRepository(session)
        match_run = await repository.get_match_run(match_run_id)
    except Exception as exc:
        logger.exception("Match run detail request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not load match run.",
        ) from exc

    if match_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match run not found.")

    return _match_run_to_response(match_run)


@router.get("/{match_run_id}/report.md", response_class=PlainTextResponse)
async def get_match_report_markdown(
    match_run_id: int,
    session: SessionDep,
) -> PlainTextResponse:
    try:
        repository = MatchRepository(session)
        report_markdown = await repository.get_match_report_markdown(match_run_id)
    except Exception as exc:
        logger.exception("Match report markdown request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not load match report.",
        ) from exc

    if report_markdown is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match run not found.")

    return PlainTextResponse(report_markdown, media_type="text/markdown")


@router.get("/{match_run_id}/report.xlsx")
async def get_match_report_excel(
    match_run_id: int,
    session: SessionDep,
) -> Response:
    try:
        repository = MatchRepository(session)
        match_run = await repository.get_match_run(match_run_id)
    except Exception as exc:
        logger.exception("Match report Excel request failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not build match Excel report.",
        ) from exc

    if match_run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Match run not found.")

    filename = f"stock_match_{match_run.id}.xlsx"
    return Response(
        content=build_match_excel_report(match_run),
        media_type=EXCEL_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _persist_v3_full_category_quote_result(
    *,
    session: AsyncSession,
    source_text: str,
    result: V3FullCategoryQuoteResult,
) -> MatchRun:
    report_json = result.report_json
    validated_quote = _dict_or_empty(report_json.get("validated_quote"))
    no_recommendation = _dict_or_empty(report_json.get("no_recommendation_reason"))
    diagnostics = _dict_or_empty(report_json.get("diagnostics"))
    quote_lines = _dict_list(validated_quote.get("lines"))
    validation_errors = _string_list(report_json.get("v3_validation_errors"))
    validation_warnings = _string_list(report_json.get("v3_validation_warnings"))
    failed_requirements = _string_list(no_recommendation.get("failed_requirements"))
    engineering_review_required = bool(
        validated_quote.get(
            "engineering_review_required",
            result.result_state == "quote_draft_review_required",
        )
    )
    risk_flags = []
    if engineering_review_required:
        risk_flags.append("engineering_review_required")
    if result.result_state:
        risk_flags.append(result.result_state)
    risk_flags.extend(validation_warnings)

    repository = MatchRepository(session)
    match_run = await repository.create_match_run(
        MatchRunCreate(
            source="v3_full_category_text",
            source_text=source_text,
            status=result.result_state,
            engineer_review_required=engineering_review_required,
            total_candidates=(
                _int_or_none(diagnostics.get("matrix_component_count"))
                or _int_or_none(diagnostics.get("matrix_row_count"))
                or 0
            ),
            matched_items=len(quote_lines),
            missing_requirements_json=[*failed_requirements, *validation_errors],
            risk_flags_json=risk_flags,
            spec_json={
                "pipeline_version": report_json.get("pipeline_version"),
                "source_text": source_text,
                "profile": result.profile,
                "category_ids": result.category_ids,
                "distributor_code": result.distributor_code,
                "resolved_request": report_json.get("resolved_request") or {},
            },
            report_json=report_json,
            report_markdown=build_v3_full_category_markdown_report(report_json),
            candidates=[],
        )
    )
    report_json = {
        **report_json,
        "match_run_id": match_run.id,
        "report_url": f"/api/v1/match/{match_run.id}/report.md",
        "report_xlsx_url": f"/api/v1/match/{match_run.id}/report.xlsx",
    }
    match_run.report_json = report_json
    flag_modified(match_run, "report_json")
    match_run.report_markdown = build_v3_full_category_markdown_report(
        report_json,
        match_run_id=match_run.id,
    )
    result.report_json.clear()
    result.report_json.update(report_json)
    await session.flush()
    await session.commit()
    return match_run


def _v3_full_category_quote_response(
    result: V3FullCategoryQuoteResult,
) -> V3FullCategoryQuoteResponse:
    report_json = result.report_json
    validated_quote = _dict_or_empty(report_json.get("validated_quote"))
    engineering_review_required = bool(
        validated_quote.get(
            "engineering_review_required",
            result.result_state == "quote_draft_review_required",
        )
    )
    return V3FullCategoryQuoteResponse(
        match_run_id=_int_or_none(report_json.get("match_run_id")),
        profile=result.profile,
        category_ids=result.category_ids,
        distributor_code=result.distributor_code,
        result_state=result.result_state,
        pipeline_version=_string_or_none(report_json.get("pipeline_version")),
        llm_configurator_used=bool(report_json.get("llm_configurator_used")),
        primary_recommendation_status=_string_or_none(
            report_json.get("primary_recommendation_status")
        ),
        final_status_source=_string_or_none(report_json.get("final_status_source")),
        engineering_review_required=engineering_review_required,
        validated_quote=validated_quote,
        no_recommendation_reason=_dict_or_empty(
            report_json.get("no_recommendation_reason")
        ),
        validation_failure_reason=_dict_or_empty(
            report_json.get("validation_failure_reason")
        ),
        diagnostics=_dict_or_empty(report_json.get("diagnostics")),
        validation_errors=_string_list(report_json.get("v3_validation_errors")),
        validation_error_details=_dict_list(
            report_json.get("v3_validation_error_details")
        ),
        validation_warnings=_string_list(report_json.get("v3_validation_warnings")),
        llm_error_type=_string_or_none(report_json.get("llm_error_type")),
        llm_http_status=_int_or_none(report_json.get("llm_http_status")),
        report_url=_string_or_none(report_json.get("report_url")),
        report_xlsx_url=_string_or_none(report_json.get("report_xlsx_url")),
        report_json=report_json,
    )


def _resolve_match_request(
    payload: MatchRequest | None,
) -> tuple[StockSpec, str, str | None, str | None]:
    if payload is None:
        raise HTTPException(
            status_code=422,
            detail="Request body must include non-empty 'text' or 'spec'.",
        )

    text = payload.text.strip() if payload.text else None
    if text:
        extraction = extract_stock_spec_for_text_match(text)
        return extraction.spec_json, "text", text, extraction.confirmation_text or None

    if payload.spec is not None:
        return payload.spec, "spec", payload.spec.source_text, None

    raise HTTPException(
        status_code=422,
        detail="Request body must include non-empty 'text' or 'spec'.",
    )


def _candidate_to_create(candidate: MatchCandidateResult) -> MatchCandidateCreate:
    return MatchCandidateCreate(
        distributor_code=candidate.distributor_code,
        item_id=candidate.item_id,
        product_key=candidate.product_key,
        part_number=candidate.part_number,
        producer=candidate.producer,
        category_id=candidate.category_id,
        item_name=candidate.item_name,
        confidence_score=candidate.confidence_score,
        price_value=candidate.price_value,
        price_currency=candidate.price_currency,
        available_quantity=candidate.available_quantity,
        reservable_locations=candidate.reservable_locations,
        matched_requirements_json=candidate.matched_requirements,
        missing_requirements_json=candidate.missing_requirements,
        risk_flags_json=candidate.risk_flags,
        raw_json=candidate.raw,
    )


def _candidate_result_to_summary(candidate: MatchCandidateResult) -> MatchSummaryCandidate:
    return MatchSummaryCandidate(
        candidate_id=candidate.candidate_id,
        candidate_type=candidate.candidate_type,
        distributor_code=candidate.distributor_code,
        part_number=candidate.part_number,
        item_id=candidate.item_id,
        producer=candidate.producer,
        category_id=candidate.category_id,
        item_name=candidate.item_name,
        confidence_score=candidate.confidence_score,
        price_value=_decimal_to_str(candidate.price_value),
        price_currency=candidate.price_currency,
        available_quantity=candidate.available_quantity,
        platform=candidate.platform,
        components=candidate.components,
        total_price_value=_decimal_to_str(candidate.total_price_value),
        total_price_currency=candidate.total_price_currency,
        missing_components=candidate.missing_components,
        compatibility_warnings=candidate.compatibility_warnings,
        engineer_review_required=candidate.engineer_review_required,
        completeness_status=candidate.completeness_status,
        completeness_label=candidate.completeness_label,
        included_component_roles=candidate.included_component_roles,
        missing_component_roles=candidate.missing_component_roles,
        excluded_from_total_roles=candidate.excluded_from_total_roles,
        cpu_per_server=candidate.cpu_per_server,
        total_cpu_required=candidate.total_cpu_required,
        total_price_note=candidate.total_price_note,
        score=candidate.score if candidate.score is not None else candidate.confidence_score,
        rank_reason=candidate.rank_reason,
        optimization_mode=_string_or_none(candidate.raw.get("optimization_mode")),
        requirement_fit=_string_or_none(candidate.raw.get("requirement_fit")),
        right_size_note=_string_or_none(candidate.raw.get("right_size_note")),
        cpu_over_requirement=_int_or_none(candidate.raw.get("cpu_over_requirement")),
        storage_over_requirement=_float_or_none(candidate.raw.get("storage_over_requirement")),
        ram_overage_gb=_int_or_none(candidate.raw.get("ram_overage_gb")),
        overfit_reason=_string_or_none(candidate.raw.get("overfit_reason")),
    )


def _llm_candidate_to_summary(candidate: dict[str, Any]) -> MatchSummaryCandidate:
    return MatchSummaryCandidate(
        candidate_id=_string_or_none(candidate.get("candidate_id")),
        candidate_type=_string_or_none(candidate.get("source_type"))
        or _string_or_none(candidate.get("candidate_type"))
        or "build_from_parts",
        distributor_code=_string_or_none(candidate.get("distributor_code")) or "llm",
        part_number=_string_or_none(candidate.get("part_number")),
        item_id=_string_or_none(candidate.get("item_id"))
        or _string_or_none(candidate.get("candidate_id"))
        or "llm_build",
        producer=_string_or_none(candidate.get("producer")),
        category_id=_string_or_none(candidate.get("category_id")),
        item_name=_string_or_none(candidate.get("display_name"))
        or _string_or_none(candidate.get("item_name"))
        or _string_or_none(candidate.get("title")),
        confidence_score=_int_or_none(candidate.get("confidence_score"))
        or _int_or_none(candidate.get("score"))
        or 0,
        price_value=_string_or_none(candidate.get("price_value"))
        or _string_or_none(candidate.get("total_price_value")),
        price_currency=_string_or_none(candidate.get("price_currency"))
        or _string_or_none(candidate.get("total_price_currency")),
        available_quantity=_int_or_none(candidate.get("available_quantity")),
        platform=_dict_or_empty(candidate.get("platform")),
        components=_dict_list(candidate.get("components")),
        total_price_value=_string_or_none(candidate.get("total_price_value")),
        total_price_currency=_string_or_none(candidate.get("total_price_currency")),
        missing_components=_string_list(candidate.get("missing_components")),
        compatibility_warnings=_string_list(candidate.get("compatibility_warnings")),
        engineer_review_required=bool(candidate.get("engineer_review_required", True)),
        completeness_status=_string_or_none(candidate.get("completeness_status")),
        completeness_label=_string_or_none(candidate.get("completeness_label")),
        included_component_roles=_string_list(candidate.get("included_component_roles")),
        missing_component_roles=_string_list(candidate.get("missing_component_roles")),
        excluded_from_total_roles=_string_list(candidate.get("excluded_from_total_roles")),
        cpu_per_server=_int_or_none(candidate.get("cpu_per_server")),
        total_cpu_required=_int_or_none(candidate.get("total_cpu_required")),
        total_price_note=_string_or_none(candidate.get("total_price_note")),
        score=_int_or_none(candidate.get("score")),
        rank_reason=_string_list(candidate.get("rank_reason")),
        optimization_mode=_string_or_none(candidate.get("optimization_mode")),
        requirement_fit=_string_or_none(candidate.get("requirement_fit")),
        right_size_note=_string_or_none(candidate.get("right_size_note")),
        cpu_over_requirement=_int_or_none(candidate.get("cpu_over_requirement")),
        storage_over_requirement=_float_or_none(candidate.get("storage_over_requirement")),
        ram_overage_gb=_int_or_none(candidate.get("ram_overage_gb")),
        overfit_reason=_string_or_none(candidate.get("overfit_reason")),
    )


def _match_run_to_response(match_run: MatchRun) -> MatchRunResponse:
    candidates = sorted(
        match_run.candidates,
        key=lambda candidate: (
            candidate.confidence_score,
            candidate.available_quantity or 0,
            candidate.reservable_locations,
        ),
        reverse=True,
    )
    return MatchRunResponse(
        id=match_run.id,
        status=match_run.status,
        engineer_review_required=match_run.engineer_review_required,
        total_candidates=match_run.total_candidates,
        matched_items=match_run.matched_items,
        spec_json=match_run.spec_json,
        report_json=match_run.report_json,
        risk_flags=list(match_run.risk_flags_json),
        missing_requirements=list(match_run.missing_requirements_json),
        candidates=[_candidate_to_brief(candidate) for candidate in candidates],
        ready_stock_candidates=[
            _candidate_to_brief(candidate)
            for candidate in candidates
            if _candidate_type_from_raw(candidate) == "ready_server"
        ],
        build_candidates=[
            _candidate_to_brief(candidate)
            for candidate in candidates
            if _candidate_type_from_raw(candidate) == "build_from_parts"
        ],
        product_group=_string_or_none(match_run.report_json.get("product_group")),
        primary_object=_string_or_none(match_run.report_json.get("primary_object")),
        semantic_planner_source=_string_or_none(
            match_run.report_json.get("semantic_planner_source")
        ),
        semantic_planner_used=bool(match_run.report_json.get("semantic_planner_used")),
        semantic_planner_confidence=_string_or_none(
            match_run.report_json.get("semantic_planner_confidence")
        ),
        semantic_planner_error_type=_string_or_none(
            match_run.report_json.get("semantic_planner_error_type")
        ),
        semantic_planner_http_status=_int_or_none(
            match_run.report_json.get("semantic_planner_http_status")
        ),
        semantic_planner_parse_status=_string_or_none(
            match_run.report_json.get("semantic_planner_parse_status")
        ),
        semantic_planner_fallback_reason=_string_or_none(
            match_run.report_json.get("semantic_planner_fallback_reason")
        ),
        semantic_planner_attempts=_dict_list(
            match_run.report_json.get("semantic_planner_attempts")
        ),
        semantic_planner_stage=_string_or_none(
            match_run.report_json.get("semantic_planner_stage")
        ),
        semantic_planner_stage_timeouts=_dict_list(
            match_run.report_json.get("semantic_planner_stage_timeouts")
        ),
        semantic_planner_timeout_reason=_string_or_none(
            match_run.report_json.get("semantic_planner_timeout_reason")
        ),
        semantic_planner_timeout_seconds=_float_or_none(
            match_run.report_json.get("semantic_planner_timeout_seconds")
        ),
        semantic_planner_elapsed_ms=_int_or_none(
            match_run.report_json.get("semantic_planner_elapsed_ms")
        ),
        semantic_planner_repair_attempted=bool(
            match_run.report_json.get("semantic_planner_repair_attempted")
        ),
        semantic_planner_repair_success=bool(
            match_run.report_json.get("semantic_planner_repair_success")
        ),
        semantic_planner_minimal_router_used=bool(
            match_run.report_json.get("semantic_planner_minimal_router_used")
        ),
        semantic_planner_minimal_fallback_used=bool(
            match_run.report_json.get("semantic_planner_minimal_fallback_used")
        ),
        semantic_planner_empty_response_count=_int_or_none(
            match_run.report_json.get("semantic_planner_empty_response_count")
        )
        or 0,
        semantic_planner_empty_response_reason=_string_or_none(
            match_run.report_json.get("semantic_planner_empty_response_reason")
        ),
        requirement_classifier_status=_string_or_none(
            match_run.report_json.get("requirement_classifier_status")
        ),
        requirement_classifier_error_type=_string_or_none(
            match_run.report_json.get("requirement_classifier_error_type")
        ),
        requirement_classifier_parse_status=_string_or_none(
            match_run.report_json.get("requirement_classifier_parse_status")
        ),
        requirement_classifier_incomplete_reason=_string_or_none(
            match_run.report_json.get("requirement_classifier_incomplete_reason")
        ),
        requirement_source_coverage=_dict_list(
            match_run.report_json.get("requirement_source_coverage")
        ),
        requirement_source_coverage_percent=_float_or_none(
            match_run.report_json.get("requirement_source_coverage_percent")
        ),
        unclassified_source_fragments=_string_list(
            match_run.report_json.get("unclassified_source_fragments")
        ),
        synthetic_requirement_count=_int_or_none(
            match_run.report_json.get("synthetic_requirement_count")
        )
        or 0,
        source_backed_requirement_count=_int_or_none(
            match_run.report_json.get("source_backed_requirement_count")
        )
        or 0,
        requirement_classifier_repair_quality=_string_or_none(
            match_run.report_json.get("requirement_classifier_repair_quality")
        ),
        requirement_classifier_repair_accepted=bool(
            match_run.report_json.get("requirement_classifier_repair_accepted")
        ),
        semantic_planner_model=_string_or_none(
            match_run.report_json.get("semantic_planner_model")
        ),
        semantic_planner_provider=_string_or_none(
            match_run.report_json.get("semantic_planner_provider")
        ),
        candidate_universe_planner_mode=_string_or_none(
            match_run.report_json.get("candidate_universe_planner_mode")
        ),
        primary_product_group=_string_or_none(
            match_run.report_json.get("primary_product_group")
        ),
        procurement_intent=_string_or_none(
            match_run.report_json.get("procurement_intent")
        ),
        selected_group_reason=_string_or_none(
            match_run.report_json.get("selected_group_reason")
        ),
        selected_product_group_reason=_string_or_none(
            match_run.report_json.get("selected_product_group_reason")
        ),
        competing_product_groups=_dict_list(
            match_run.report_json.get("competing_product_groups")
        ),
        primary_object_indicators=_list_or_empty(
            match_run.report_json.get("primary_object_indicators")
        ),
        component_role_indicators=_dict_list(
            match_run.report_json.get("component_role_indicators")
        ),
        excluded_category_groups=_dict_list(
            match_run.report_json.get("excluded_category_groups")
        ),
        planner_repair_attempted=bool(
            match_run.report_json.get("planner_repair_attempted")
        ),
        planner_repair_success=bool(
            match_run.report_json.get("planner_repair_success")
        ),
        planner_suspicion_reasons=_string_list(
            match_run.report_json.get("planner_suspicion_reasons")
        ),
        deterministic_product_group_hint=_string_or_none(
            match_run.report_json.get("deterministic_product_group_hint")
        ),
        semantic_planner_disagreement=bool(
            match_run.report_json.get("semantic_planner_disagreement")
        ),
        matrix_blueprint=_dict_or_empty(match_run.report_json.get("matrix_blueprint")),
        matrix_blueprint_roles=_string_list(
            match_run.report_json.get("matrix_blueprint_roles")
        ),
        stage_a_broad_roles=_string_list(
            match_run.report_json.get("stage_a_broad_roles")
        ),
        semantic_matrix_blueprint_roles=_string_list(
            match_run.report_json.get("semantic_matrix_blueprint_roles")
        ),
        requirement_classifier_roles=_string_list(
            match_run.report_json.get("requirement_classifier_roles")
        ),
        effective_matrix_roles_before_category_planner=_string_list(
            match_run.report_json.get("effective_matrix_roles_before_category_planner")
        ),
        category_planner_input_roles=_string_list(
            match_run.report_json.get("category_planner_input_roles")
        ),
        category_planner_output_roles=_string_list(
            match_run.report_json.get("category_planner_output_roles")
        ),
        validated_category_plan_roles=_string_list(
            match_run.report_json.get("validated_category_plan_roles")
        ),
        materialized_matrix_roles=_string_list(
            match_run.report_json.get("materialized_matrix_roles")
        ),
        composer_package_roles=_string_list(
            match_run.report_json.get("composer_package_roles")
        ),
        roles_dropped_after_stage_a=_string_list(
            match_run.report_json.get("roles_dropped_after_stage_a")
        ),
        roles_dropped_before_category_planner=_string_list(
            match_run.report_json.get("roles_dropped_before_category_planner")
        ),
        roles_dropped_after_category_planner=_string_list(
            match_run.report_json.get("roles_dropped_after_category_planner")
        ),
        roles_dropped_during_materialization=_string_list(
            match_run.report_json.get("roles_dropped_during_materialization")
        ),
        roles_dropped_reason_by_role=_dict_or_empty(
            match_run.report_json.get("roles_dropped_reason_by_role")
        ),
        role_source_by_role=_dict_or_empty(
            match_run.report_json.get("role_source_by_role")
        ),
        role_lifecycle_trace=_dict_list(
            match_run.report_json.get("role_lifecycle_trace")
        ),
        embedded_requirements=_list_or_empty(
            match_run.report_json.get("embedded_requirements")
        ),
        classified_requirements=_dict_list(
            match_run.report_json.get("classified_requirements")
        ),
        purchasable_role_requirements=_dict_list(
            match_run.report_json.get("purchasable_role_requirements")
        ),
        primary_object_feature_requirements=_dict_list(
            match_run.report_json.get("primary_object_feature_requirements")
        ),
        accessory_or_consumable_requirements=_dict_list(
            match_run.report_json.get("accessory_or_consumable_requirements")
        ),
        service_or_support_requirements=_dict_list(
            match_run.report_json.get("service_or_support_requirements")
        ),
        logistics_or_commercial_constraints=_dict_list(
            match_run.report_json.get("logistics_or_commercial_constraints")
        ),
        engineering_check_requirements=_dict_list(
            match_run.report_json.get("engineering_check_requirements")
        ),
        unmapped_requirements_non_blocking=_dict_list(
            match_run.report_json.get("unmapped_requirements_non_blocking")
        ),
        unmapped_requirements_blocking=_dict_list(
            match_run.report_json.get("unmapped_requirements_blocking")
        ),
        requirement_role_mapping_decision=_dict_list(
            match_run.report_json.get("requirement_role_mapping_decision")
        ),
        requirement_fulfillment_decision=_dict_list(
            match_run.report_json.get("requirement_fulfillment_decision")
        ),
        not_primary_product_groups=_list_or_empty(
            match_run.report_json.get("not_primary_product_groups")
        ),
        requirements=_dict_list(match_run.report_json.get("requirements")),
        required_capabilities=_dict_list(
            match_run.report_json.get("required_capabilities")
        ),
        optional_capabilities=_dict_list(
            match_run.report_json.get("optional_capabilities")
        ),
        workload_context=_string_list(match_run.report_json.get("workload_context")),
        logistics_constraints=_dict_or_empty(
            match_run.report_json.get("logistics_constraints")
        ),
        commercial_instructions=_dict_list(
            match_run.report_json.get("commercial_instructions")
        ),
        response_instructions=_string_list(
            match_run.report_json.get("response_instructions")
        ),
        unsupported_or_unmapped_requirements=_string_list(
            match_run.report_json.get("unsupported_or_unmapped_requirements")
        ),
        role_plan=_dict_or_empty(match_run.report_json.get("role_plan")),
        category_plan=_dict_or_empty(match_run.report_json.get("category_plan")),
        category_plan_entries=_dict_list(
            match_run.report_json.get("category_plan_entries")
        ),
        category_catalog_summary=_dict_or_empty(
            match_run.report_json.get("category_catalog_summary")
        ),
        category_planner_source=_string_or_none(
            match_run.report_json.get("category_planner_source")
        ),
        category_plan_source=_string_or_none(
            match_run.report_json.get("category_plan_source")
        ),
        category_planner_missing_required_roles=_string_list(
            match_run.report_json.get("category_planner_missing_required_roles")
        ),
        category_planner_repair_attempted=bool(
            match_run.report_json.get("category_planner_repair_attempted")
        ),
        category_planner_repair_success=bool(
            match_run.report_json.get("category_planner_repair_success")
        ),
        category_planner_repair_reason=_string_or_none(
            match_run.report_json.get("category_planner_repair_reason")
        ),
        category_planner_repaired_roles=_string_list(
            match_run.report_json.get("category_planner_repaired_roles")
        ),
        category_planner_unresolved_required_roles=_string_list(
            match_run.report_json.get("category_planner_unresolved_required_roles")
        ),
        required_roles=_string_list(match_run.report_json.get("required_roles")),
        missing_required_roles=_string_list(
            match_run.report_json.get("missing_required_roles")
        ),
        missing_category_roles=_string_list(
            match_run.report_json.get("missing_category_roles")
        ),
        missing_required_capabilities=_dict_list(
            match_run.report_json.get("missing_required_capabilities")
        ),
        role_coverage_summary=_dict_or_empty(
            match_run.report_json.get("role_coverage_summary")
        ),
        category_plan_warnings=_string_list(
            match_run.report_json.get("category_plan_warnings")
        ),
        matrix_distiller_used=bool(match_run.report_json.get("matrix_distiller_used")),
        matrix_distiller_source=_string_or_none(
            match_run.report_json.get("matrix_distiller_source")
        ),
        matrix_distiller_diagnostics=_dict_or_empty(
            match_run.report_json.get("matrix_distiller_diagnostics")
        ),
        broad_count_by_role=_dict_or_empty(match_run.report_json.get("broad_count_by_role")),
        distilled_count_by_role=_dict_or_empty(
            match_run.report_json.get("distilled_count_by_role")
        ),
        full_matrix_evaluation_used=bool(
            match_run.report_json.get("full_matrix_evaluation_used")
        ),
        full_matrix_evaluation_fallback_reason=_string_or_none(
            match_run.report_json.get("full_matrix_evaluation_fallback_reason")
        ),
        provider_error_type=_string_or_none(
            match_run.report_json.get("provider_error_type")
        ),
        provider_context_limit=_dict_or_empty(
            match_run.report_json.get("provider_context_limit")
        ),
        role_chunk_count_by_role=_dict_or_empty(
            match_run.report_json.get("role_chunk_count_by_role")
        ),
        evaluated_candidate_count_by_role=_dict_or_empty(
            match_run.report_json.get("evaluated_candidate_count_by_role")
        ),
        selected_candidate_count_by_role=_dict_or_empty(
            match_run.report_json.get("selected_candidate_count_by_role")
        ),
        role_reducer_summary=_dict_or_empty(
            match_run.report_json.get("role_reducer_summary")
        ),
        full_matrix_failed_chunks=_dict_list(
            match_run.report_json.get("full_matrix_failed_chunks")
        ),
        bom_critic_used=bool(match_run.report_json.get("bom_critic_used")),
        no_recommendation_coverage=_dict_or_empty(
            match_run.report_json.get("no_recommendation_coverage")
        ),
        no_recommendation_coverage_gate_passed=bool(
            match_run.report_json.get("no_recommendation_coverage_gate_passed")
        ),
        no_recommendation_coverage_repair_attempted=bool(
            match_run.report_json.get("no_recommendation_coverage_repair_attempted")
        ),
        no_recommendation_coverage_repair_success=bool(
            match_run.report_json.get("no_recommendation_coverage_repair_success")
        ),
        no_recommendation_coverage_rejected=bool(
            match_run.report_json.get("no_recommendation_coverage_rejected")
        ),
        no_recommendation_coverage_thresholds=_dict_or_empty(
            match_run.report_json.get("no_recommendation_coverage_thresholds")
        ),
        no_recommendation_coverage_repair_reason=_string_or_none(
            match_run.report_json.get("no_recommendation_coverage_repair_reason")
        ),
        llm_cost_diagnostics=_dict_or_empty(
            match_run.report_json.get("llm_cost_diagnostics")
        ),
        count_by_role=_dict_or_empty(match_run.report_json.get("count_by_role")),
        broad_matrix_count_by_role=_dict_or_empty(
            match_run.report_json.get("broad_matrix_count_by_role")
        ),
        composer_package_candidate_count_by_role=_dict_or_empty(
            match_run.report_json.get("composer_package_candidate_count_by_role")
        ),
        composer_package_candidate_total=_int_or_none(
            match_run.report_json.get("composer_package_candidate_total")
        )
        or 0,
        composer_package_candidate_ids_by_role=_dict_or_empty(
            match_run.report_json.get("composer_package_candidate_ids_by_role")
        ),
        v2_package_mode=_string_or_none(match_run.report_json.get("v2_package_mode")),
        selected_package_mode=_string_or_none(
            match_run.report_json.get("selected_package_mode")
        ),
        verbose_context_chars=_int_or_none(
            match_run.report_json.get("verbose_context_chars")
        ),
        compact_context_chars=_int_or_none(
            match_run.report_json.get("compact_context_chars")
        ),
        selected_context_chars=_int_or_none(
            match_run.report_json.get("selected_context_chars")
        ),
        verbose_context_size=_dict_or_empty(
            match_run.report_json.get("verbose_context_size")
        ),
        compact_context_size=_dict_or_empty(
            match_run.report_json.get("compact_context_size")
        ),
        selected_context_size=_dict_or_empty(
            match_run.report_json.get("selected_context_size")
        ),
        chars_by_section=_dict_or_empty(match_run.report_json.get("chars_by_section")),
        avg_chars_per_candidate_by_role=_dict_or_empty(
            match_run.report_json.get("avg_chars_per_candidate_by_role")
        ),
        removed_verbose_fields=_string_list(
            match_run.report_json.get("removed_verbose_fields")
        ),
        removed_verbose_field_counts=_dict_or_empty(
            match_run.report_json.get("removed_verbose_field_counts")
        ),
        compact_candidate_count_by_role=_dict_or_empty(
            match_run.report_json.get("compact_candidate_count_by_role")
        ),
        compact_candidate_total=_int_or_none(
            match_run.report_json.get("compact_candidate_total")
        )
        or 0,
        compact_candidate_ids_hash=_string_or_none(
            match_run.report_json.get("compact_candidate_ids_hash")
        ),
        compact_package_full_matrix_used=bool(
            match_run.report_json.get("compact_package_full_matrix_used")
        ),
        package_candidate_loss=bool(
            match_run.report_json.get("package_candidate_loss")
        ),
        provider_context_limit_retry_compact_attempted=bool(
            match_run.report_json.get("provider_context_limit_retry_compact_attempted")
        ),
        provider_context_limit_retry_compact_success=bool(
            match_run.report_json.get("provider_context_limit_retry_compact_success")
        ),
        provider_context_limit_original_chars=_int_or_none(
            match_run.report_json.get("provider_context_limit_original_chars")
        ),
        provider_context_limit_compact_chars=_int_or_none(
            match_run.report_json.get("provider_context_limit_compact_chars")
        ),
        provider_context_limit_after_compact=bool(
            match_run.report_json.get("provider_context_limit_after_compact")
        ),
        dropped_before_composer_count_by_role=_dict_or_empty(
            match_run.report_json.get("dropped_before_composer_count_by_role")
        ),
        dropped_before_composer_reason_by_role=_dict_or_empty(
            match_run.report_json.get("dropped_before_composer_reason_by_role")
        ),
        package_candidate_exposure_ratio_by_role=_dict_or_empty(
            match_run.report_json.get("package_candidate_exposure_ratio_by_role")
        ),
        original_candidate_count_by_role=_dict_or_empty(
            match_run.report_json.get("original_candidate_count_by_role")
        ),
        fallback_candidate_count_by_role=_dict_or_empty(
            match_run.report_json.get("fallback_candidate_count_by_role")
        ),
        dropped_before_fallback_count_by_role=_dict_or_empty(
            match_run.report_json.get("dropped_before_fallback_count_by_role")
        ),
        dropped_before_fallback_reasons=_dict_or_empty(
            match_run.report_json.get("dropped_before_fallback_reasons")
        ),
        timeout_fallback_coverage_ratio_by_role=_dict_or_empty(
            match_run.report_json.get("timeout_fallback_coverage_ratio_by_role")
        ),
        package_candidate_exposure_policy=_dict_or_empty(
            match_run.report_json.get("package_candidate_exposure_policy")
        ),
        package_candidate_exposure_incomplete=bool(
            match_run.report_json.get("package_candidate_exposure_incomplete")
        ),
        package_candidate_exposure_incomplete_roles=_string_list(
            match_run.report_json.get("package_candidate_exposure_incomplete_roles")
        ),
        package_exposure_blocking_lifecycle_roles=_string_list(
            match_run.report_json.get("package_exposure_blocking_lifecycle_roles")
        ),
        package_budget=_dict_or_empty(match_run.report_json.get("package_budget")),
        package_budget_warnings=_string_list(
            match_run.report_json.get("package_budget_warnings")
        ),
        package_approximate_size=_dict_or_empty(
            match_run.report_json.get("package_approximate_size")
        ),
        package_skipped_reason=_string_or_none(
            match_run.report_json.get("package_skipped_reason")
        ),
        ready_candidates_excluded_reason=_string_or_none(
            match_run.report_json.get("ready_candidates_excluded_reason")
        ),
        composer_attempt_decision=_dict_or_empty(
            match_run.report_json.get("composer_attempt_decision")
        ),
        composer_mode=_string_or_none(match_run.report_json.get("composer_mode")),
        expected_composer_mode=_string_or_none(
            match_run.report_json.get("expected_composer_mode")
        ),
        llm_call_count=_int_or_none(match_run.report_json.get("llm_call_count")) or 0,
        llm_call_stages=_string_list(match_run.report_json.get("llm_call_stages")),
        llm_call_budget_exceeded=bool(
            match_run.report_json.get("llm_call_budget_exceeded")
        ),
        max_llm_calls_per_match=_int_or_none(
            match_run.report_json.get("max_llm_calls_per_match")
        ),
        requirement_contract=_dict_or_empty(
            match_run.report_json.get("requirement_contract")
        ),
        requirement_contract_used=bool(
            match_run.report_json.get("requirement_contract_used")
        ),
        main_composer_used=bool(match_run.report_json.get("main_composer_used")),
        critic_used=bool(match_run.report_json.get("critic_used")),
        repair_used=bool(match_run.report_json.get("repair_used")),
        role_evaluation_used=bool(match_run.report_json.get("role_evaluation_used")),
        role_evaluation_skipped_reason=_string_or_none(
            match_run.report_json.get("role_evaluation_skipped_reason")
        ),
        role_evaluation_count_by_role=_dict_or_empty(
            match_run.report_json.get("role_evaluation_count_by_role")
        ),
        role_evaluation_coverage_by_role=_dict_or_empty(
            match_run.report_json.get("role_evaluation_coverage_by_role")
        ),
        role_evaluation_failed_chunks=_dict_list(
            match_run.report_json.get("role_evaluation_failed_chunks")
        ),
        bom_composer_used=bool(match_run.report_json.get("bom_composer_used")),
        completeness_critic_used=bool(
            match_run.report_json.get("completeness_critic_used")
        ),
        completeness_critic_result=_dict_or_empty(
            match_run.report_json.get("completeness_critic_result")
        ),
        repair_composer_used=bool(match_run.report_json.get("repair_composer_used")),
        final_bom_after_repair=_dict_or_empty(
            match_run.report_json.get("final_bom_after_repair")
        ),
        pre_composer_requirement_classifier_status=_string_or_none(
            match_run.report_json.get("pre_composer_requirement_classifier_status")
        ),
        pre_composer_requirement_source_coverage_percent=_float_or_none(
            match_run.report_json.get(
                "pre_composer_requirement_source_coverage_percent"
            )
        ),
        pre_composer_unclassified_source_fragments=_string_list(
            match_run.report_json.get("pre_composer_unclassified_source_fragments")
        ),
        pre_composer_semantic_diagnostics_are_blocking=bool(
            match_run.report_json.get("pre_composer_semantic_diagnostics_are_blocking")
        ),
        composer_requirement_analysis=_dict_or_empty(
            match_run.report_json.get("composer_requirement_analysis")
        ),
        composer_fulfillment_decisions=_dict_list(
            match_run.report_json.get("composer_fulfillment_decisions")
        ),
        composer_source_coverage_summary=_dict_or_empty(
            match_run.report_json.get("composer_source_coverage_summary")
        ),
        composer_assumptions=_string_list(
            match_run.report_json.get("composer_assumptions")
        ),
        composer_engineer_checks=_string_list(
            match_run.report_json.get("composer_engineer_checks")
        ),
        composer_hard_mismatch_risks=_dict_list(
            match_run.report_json.get("composer_hard_mismatch_risks")
        ),
        composer_unverified_requirements=_dict_list(
            match_run.report_json.get("composer_unverified_requirements")
        ),
        composer_considered_candidate_count_by_role=_dict_or_empty(
            match_run.report_json.get("composer_considered_candidate_count_by_role")
        ),
        composer_chosen_candidate_ids=_string_list(
            match_run.report_json.get("composer_chosen_candidate_ids")
        ),
        validation_hard_mismatches=_dict_list(
            match_run.report_json.get("validation_hard_mismatches")
        ),
        validation_unverified_requirements=_dict_list(
            match_run.report_json.get("validation_unverified_requirements")
        ),
        final_status_source=_string_or_none(
            match_run.report_json.get("final_status_source")
        ),
        package_strategy_decision=_dict_or_empty(
            match_run.report_json.get("package_strategy_decision")
        ),
        match_trace=_dict_list(match_run.report_json.get("match_trace")),
        diagnostics=_dict_or_empty(match_run.report_json.get("diagnostics")),
        component_candidate_matrix=_dict_or_empty(
            match_run.report_json.get("component_candidate_matrix")
        ),
        shortlist_for_llm=_dict_list(match_run.report_json.get("shortlist_for_llm")),
        llm_configurator_enabled=bool(match_run.report_json.get("llm_configurator_enabled")),
        llm_configurator_used=bool(match_run.report_json.get("llm_configurator_used")),
        output_mode=_string_or_none(match_run.report_json.get("output_mode")),
        llm_configurator_output_mode=_string_or_none(
            match_run.report_json.get("llm_configurator_output_mode")
        ),
        ai_recommendation_mode=_string_or_none(
            match_run.report_json.get("ai_recommendation_mode")
        ),
        ai_recommendations_count=_int_or_none(
            match_run.report_json.get("ai_recommendations_count")
        )
        or 0,
        ai_recommendations=_dict_list(match_run.report_json.get("ai_recommendations")),
        llm_recommendations=_dict_list(match_run.report_json.get("llm_recommendations")),
        llm_recommended_build_candidates=[
            _llm_candidate_to_summary(candidate)
            for candidate in _dict_list(
                match_run.report_json.get("llm_recommended_build_candidates")
            )
        ],
        primary_recommendation=_dict_or_empty(
            match_run.report_json.get("primary_recommendation")
        ),
        primary_recommendation_status=_string_or_none(
            match_run.report_json.get("primary_recommendation_status")
        ),
        no_recommendation_reason=_dict_or_empty(
            match_run.report_json.get("no_recommendation_reason")
        ),
        partial_available_components=_dict_list(
            match_run.report_json.get("partial_available_components")
        ),
        failed_requirements=_list_or_empty(
            match_run.report_json.get("failed_requirements")
        ),
        role_failures=_dict_list(match_run.report_json.get("role_failures")),
        unverified_requirements=_list_or_empty(
            match_run.report_json.get("unverified_requirements")
        ),
        hard_mismatch_risks=_list_or_empty(
            match_run.report_json.get("hard_mismatch_risks")
        ),
        recommended_next_actions=_string_list(
            match_run.report_json.get("recommended_next_actions")
        ),
        engineer_checks=_string_list(match_run.report_json.get("engineer_checks")),
        composer_summary_ru=_string_or_none(
            match_run.report_json.get("composer_summary_ru")
        ),
        customer_safe_summary_ru=_string_or_none(
            match_run.report_json.get("customer_safe_summary_ru")
        ),
        commercial_summary=_dict_or_empty(match_run.report_json.get("commercial_summary")),
        grouped_presales_mode_used=bool(
            match_run.report_json.get("grouped_presales_mode_used")
        ),
        configuration_groups=_dict_list(
            match_run.report_json.get("configuration_groups")
        ),
        configuration_groups_count=_int_or_none(
            match_run.report_json.get("configuration_groups_count")
        )
        or 0,
        quote_recommendation=_dict_or_empty(
            match_run.report_json.get("quote_recommendation")
        ),
        selected_configuration_group_id=_string_or_none(
            match_run.report_json.get("selected_configuration_group_id")
        ),
        selected_platform_option_id=_string_or_none(
            match_run.report_json.get("selected_platform_option_id")
        ),
        selected_platform_option_index=_int_or_none(
            match_run.report_json.get("selected_platform_option_index")
        ),
        llm_general_notes=_string_list(match_run.report_json.get("llm_general_notes")),
        llm_fallback_reason=_string_or_none(match_run.report_json.get("llm_fallback_reason")),
        llm_error_type=_string_or_none(match_run.report_json.get("llm_error_type")),
        llm_http_status=_int_or_none(match_run.report_json.get("llm_http_status")),
        llm_repair_used=bool(match_run.report_json.get("llm_repair_used")),
        llm_repair_attempted=bool(match_run.report_json.get("llm_repair_attempted")),
        llm_repair_success=bool(match_run.report_json.get("llm_repair_success")),
        llm_repair_fallback_reason=_string_or_none(
            match_run.report_json.get("llm_repair_fallback_reason")
        ),
        llm_repair_critique_count=_int_or_none(
            match_run.report_json.get("llm_repair_critique_count")
        )
        or 0,
        llm_repair_critique_summary=_string_list(
            match_run.report_json.get("llm_repair_critique_summary")
        ),
        llm_repair_blocked_critique_count=_int_or_none(
            match_run.report_json.get("llm_repair_blocked_critique_count")
        )
        or 0,
        llm_repair_blocked_critique_summary=_string_list(
            match_run.report_json.get("llm_repair_blocked_critique_summary")
        ),
        llm_repair_savings_estimate=_string_or_none(
            match_run.report_json.get("llm_repair_savings_estimate")
        ),
        llm_repair_revised_proposals_count=_int_or_none(
            match_run.report_json.get("llm_repair_revised_proposals_count")
        )
        or 0,
        llm_repair_validation_summary=_dict_or_empty(
            match_run.report_json.get("llm_repair_validation_summary")
        ),
        llm_thinking_diagnostics=_dict_or_empty(
            match_run.report_json.get("llm_thinking_diagnostics")
        ),
        llm_thinking_enabled=bool(match_run.report_json.get("llm_thinking_enabled")),
        llm_thinking_budget_tokens=_int_or_none(
            match_run.report_json.get("llm_thinking_budget_tokens")
        ),
        llm_thinking_fallback_reason=_string_or_none(
            match_run.report_json.get("llm_thinking_fallback_reason")
        ),
        llm_proposals_count=_int_or_none(
            match_run.report_json.get("llm_proposals_count")
        )
        or 0,
        valid_proposals_count=_int_or_none(
            match_run.report_json.get("valid_proposals_count")
        )
        or 0,
        validation_rejected_count=_int_or_none(
            match_run.report_json.get("validation_rejected_count")
        )
        or 0,
        selection_skipped_count=_int_or_none(
            match_run.report_json.get("selection_skipped_count")
        )
        or 0,
        rejected_ai_recommendations_count=_int_or_none(
            match_run.report_json.get("rejected_ai_recommendations_count")
        )
        or 0,
        ai_recommendations_validation_warnings=_string_list(
            match_run.report_json.get("ai_recommendations_validation_warnings")
        ),
        ai_validation_summary=_dict_or_empty(
            match_run.report_json.get("ai_validation_summary")
        ),
        rejected_reasons_top=_dict_list(match_run.report_json.get("rejected_reasons_top")),
        rejected_ai_recommendations_debug_safe=_dict_list(
            match_run.report_json.get("rejected_ai_recommendations_debug_safe")
        ),
        rejected_ai_recommendations=_dict_list(
            match_run.report_json.get("rejected_ai_recommendations")
        ),
        web_evidence_pack=_dict_or_empty(match_run.report_json.get("web_evidence_pack")),
        web_evidence_diagnostics=_dict_or_empty(
            match_run.report_json.get("web_evidence_diagnostics")
        ),
        evidence_mode=_string_or_none(match_run.report_json.get("evidence_mode")),
        online_composer_used=bool(match_run.report_json.get("online_composer_used")),
        evidence_used=bool(match_run.report_json.get("evidence_used")),
        evidence_sources_count=_int_or_none(
            match_run.report_json.get("evidence_sources_count")
        )
        or 0,
        evidence_status_summary=_dict_or_empty(
            match_run.report_json.get("evidence_status_summary")
        ),
        online_composer_error_type=_string_or_none(
            match_run.report_json.get("online_composer_error_type")
        ),
        online_composer_parse_status=_string_or_none(
            match_run.report_json.get("online_composer_parse_status")
        ),
        online_composer_empty_response_repair_attempted=bool(
            match_run.report_json.get("online_composer_empty_response_repair_attempted")
        ),
        online_composer_empty_response_repair_success=bool(
            match_run.report_json.get("online_composer_empty_response_repair_success")
        ),
        structured_no_recommendation_used=bool(
            match_run.report_json.get("structured_no_recommendation_used")
        ),
        evidence_requests_count=_int_or_none(
            match_run.report_json.get("evidence_requests_count")
        )
        or 0,
        llm_evidence_review=_dict_or_empty(match_run.report_json.get("llm_evidence_review")),
        created_at=match_run.created_at,
    )


def _candidate_to_brief(candidate: MatchCandidate) -> MatchCandidateBrief:
    raw = candidate.raw_json if isinstance(candidate.raw_json, dict) else {}
    return MatchCandidateBrief(
        id=candidate.id,
        candidate_id=_string_or_none(raw.get("candidate_id")),
        candidate_type=_candidate_type_from_raw(candidate),
        distributor_code=candidate.distributor_code,
        item_id=candidate.item_id,
        product_key=candidate.product_key,
        part_number=candidate.part_number,
        producer=candidate.producer,
        category_id=candidate.category_id,
        item_name=candidate.item_name,
        confidence_score=candidate.confidence_score,
        price_value=_decimal_to_str(candidate.price_value),
        price_currency=candidate.price_currency,
        available_quantity=candidate.available_quantity,
        reservable_locations=candidate.reservable_locations,
        matched_requirements=list(candidate.matched_requirements_json),
        missing_requirements=list(candidate.missing_requirements_json),
        risk_flags=list(candidate.risk_flags_json),
        platform=_dict_or_empty(raw.get("platform")),
        components=_dict_list(raw.get("components")),
        total_price_value=_string_or_none(raw.get("total_price_value")),
        total_price_currency=_string_or_none(raw.get("total_price_currency")),
        missing_components=_string_list(raw.get("missing_components")),
        compatibility_warnings=_string_list(raw.get("compatibility_warnings")),
        engineer_review_required=bool(raw.get("engineer_review_required", True)),
        completeness_status=_string_or_none(raw.get("completeness_status")),
        completeness_label=_string_or_none(raw.get("completeness_label")),
        included_component_roles=_string_list(raw.get("included_component_roles")),
        missing_component_roles=_string_list(raw.get("missing_component_roles")),
        excluded_from_total_roles=_string_list(raw.get("excluded_from_total_roles")),
        cpu_per_server=_int_or_none(raw.get("cpu_per_server")),
        total_cpu_required=_int_or_none(raw.get("total_cpu_required")),
        total_price_note=_string_or_none(raw.get("total_price_note")),
        score=_int_or_none(raw.get("score")) or candidate.confidence_score,
        rank_reason=_string_list(raw.get("rank_reason")),
        optimization_mode=_string_or_none(raw.get("optimization_mode")),
        requirement_fit=_string_or_none(raw.get("requirement_fit")),
        right_size_note=_string_or_none(raw.get("right_size_note")),
        cpu_over_requirement=_int_or_none(raw.get("cpu_over_requirement")),
        storage_over_requirement=_float_or_none(raw.get("storage_over_requirement")),
        ram_overage_gb=_int_or_none(raw.get("ram_overage_gb")),
        overfit_reason=_string_or_none(raw.get("overfit_reason")),
    )


def _candidate_type_from_raw(candidate: MatchCandidate) -> str:
    raw = candidate.raw_json if isinstance(candidate.raw_json, dict) else {}
    candidate_type = raw.get("candidate_type")
    if isinstance(candidate_type, str) and candidate_type:
        return candidate_type
    return "ready_server"


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dict_or_empty(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _list_or_empty(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _decimal_to_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f")


def _brief(value: str | None, *, limit: int = 120) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


async def _rollback_safely(session: AsyncSession | None) -> None:
    if session is None:
        return
    try:
        await session.rollback()
    except Exception:
        logger.warning("Could not roll back match API session", exc_info=True)
