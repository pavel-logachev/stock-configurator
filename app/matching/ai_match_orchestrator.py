from __future__ import annotations

import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import LlmSettings, WebEvidenceSettings, get_llm_settings
from app.evidence.web_evidence import EvidenceSearchCache, WebSearchProvider
from app.llm.base import LlmClient
from app.llm.configuration_composer import (
    HIGH_QUALITY_BROAD_PACKAGE_UNDER_LIMIT_REASON,
    INCOMPLETE_MATRIX_EXPOSURE_REASON,
    PROVIDER_CONTEXT_LIMIT_ERROR_TYPE,
    PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON,
)
from app.matching.match_engine import (
    MatchResult,
    build_llm_configurator_package_from_report_json,
    extract_stock_spec_for_text_match,
    match_stock_spec,
)
from app.matching.spec_schema import StockSpec
from app.planning.requirement_planner import (
    SEMANTIC_COMPLEX_FALLBACK_REASON,
    SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON,
    SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT,
)
from app.reports.composer_result_normalizer import normalize_composer_report_json

StageStatus = str
MatchFunc = Callable[..., Awaitable[MatchResult]]


@dataclass(frozen=True)
class StageResult:
    stage_name: str
    status: StageStatus
    started_at: str | None = None
    finished_at: str | None = None
    duration_ms: int | None = None
    blocking: bool = False
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message_safe: str | None = None
    fallback_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_report_json(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "blocking": self.blocking,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "warnings": self.warnings,
            "error_type": self.error_type,
            "error_message_safe": self.error_message_safe,
            "fallback_reason": self.fallback_reason,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class AiMatchOrchestratorRequest:
    text: str | None = None
    spec: StockSpec | None = None
    distributor_code: str = "ocs"
    warehouse_constraints: dict[str, Any] = field(default_factory=dict)
    output_mode: str | None = None
    preview_only: bool = False
    allow_llm: bool = True
    allow_web_evidence: bool = True
    force_full_matrix: bool | None = None
    package_limit: int | None = None
    candidates_per_role: int | None = None
    pipeline_v2: bool | None = None


@dataclass(frozen=True)
class AiMatchOrchestratorResult:
    match_result: MatchResult
    report_json: dict[str, Any]
    package: dict[str, Any] = field(default_factory=dict)
    match_trace: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


async def run_ai_match_orchestrator(
    request: AiMatchOrchestratorRequest,
    session: AsyncSession,
    *,
    llm_configurator_client: LlmClient | None = None,
    llm_settings: LlmSettings | None = None,
    web_evidence_settings: WebEvidenceSettings | None = None,
    web_search_provider: WebSearchProvider | None = None,
    evidence_cache: EvidenceSearchCache | None = None,
    match_func: MatchFunc = match_stock_spec,
) -> AiMatchOrchestratorResult:
    """Run the canonical AI-first match pipeline and return persisted report JSON."""

    trace_builder = _TraceBuilder()
    spec = _resolve_spec(request, trace_builder)
    effective_llm_settings = _effective_llm_settings(request, llm_settings)
    effective_web_settings = _effective_web_settings(request, web_evidence_settings)

    started = time.perf_counter()
    stage_started = _utc_now()
    if _use_pipeline_v2(request, effective_llm_settings):
        from app.matching.ai_match_pipeline_v2 import run_ai_match_pipeline_v2

        v2_result = await run_ai_match_pipeline_v2(
            spec,
            session,
            distributor_code=request.distributor_code,
            preview_only=request.preview_only,
            llm_configurator_client=llm_configurator_client,
            llm_settings=effective_llm_settings,
            web_evidence_settings=effective_web_settings,
            web_search_provider=web_search_provider,
            evidence_cache=evidence_cache,
        )
        match_result = v2_result.match_result
        trace_builder.add(
            StageResult(
                stage_name="composer_first_match_engine_v2",
                status="success",
                started_at=stage_started,
                finished_at=_utc_now(),
                duration_ms=_duration_ms(started),
                blocking=True,
                input_summary={
                    "preview_only": request.preview_only,
                    "allow_llm": request.allow_llm,
                    "allow_web_evidence": request.allow_web_evidence,
                },
                output_summary={
                    "status": match_result.status,
                    "total_candidates": match_result.total_candidates,
                    "matched_items": match_result.matched_items,
                    "pipeline_version": "v2_composer_first",
                },
            )
        )
        package = dict(v2_result.package) if request.preview_only else {}
        report_json = {
            **match_result.to_report_json(),
            **v2_result.report_fields,
        }
        report_json = _enrich_v2_report_json(
            report_json,
            request=request,
            llm_settings=effective_llm_settings,
            trace=trace_builder,
            package=package,
        )
        return AiMatchOrchestratorResult(
            match_result=match_result,
            report_json=report_json,
            package=package,
            match_trace=list(report_json.get("match_trace") or []),
            diagnostics=dict(report_json.get("diagnostics") or {}),
        )

    match_result = await _call_match_func(
        match_func,
        spec,
        session,
        llm_configurator_client=llm_configurator_client,
        llm_settings=effective_llm_settings,
        web_evidence_settings=effective_web_settings,
        web_search_provider=web_search_provider,
        evidence_cache=evidence_cache,
        pass_runtime_kwargs=_request_needs_runtime_kwargs(
            request,
            llm_settings=llm_settings,
            web_evidence_settings=web_evidence_settings,
            llm_configurator_client=llm_configurator_client,
            web_search_provider=web_search_provider,
            evidence_cache=evidence_cache,
        ),
    )
    trace_builder.add(
        StageResult(
            stage_name="unified_match_engine",
            status="success",
            started_at=stage_started,
            finished_at=_utc_now(),
            duration_ms=_duration_ms(started),
            blocking=True,
            input_summary={
                "preview_only": request.preview_only,
                "allow_llm": request.allow_llm,
                "allow_web_evidence": request.allow_web_evidence,
            },
            output_summary={
                "status": match_result.status,
                "total_candidates": match_result.total_candidates,
                "matched_items": match_result.matched_items,
            },
        )
    )

    report_json = match_result.to_report_json()
    package: dict[str, Any] = {}
    if request.preview_only:
        package = build_llm_configurator_package_from_report_json(
            report_json,
            user_request=spec.source_text or request.text,
            candidates_per_role=request.candidates_per_role,
            llm_settings=effective_llm_settings,
        )
        trace_builder.add(_package_stage_from_package(package, preview_only=True))

    report_json = _enrich_report_json(
        report_json,
        request=request,
        llm_settings=effective_llm_settings,
        trace=trace_builder,
        package=package,
    )
    return AiMatchOrchestratorResult(
        match_result=match_result,
        report_json=report_json,
        package=package,
        match_trace=list(report_json.get("match_trace") or []),
        diagnostics=dict(report_json.get("diagnostics") or {}),
    )


async def preview_llm_configurator_package_from_text(
    text: str,
    session: AsyncSession,
    *,
    candidates_per_role: int | None = None,
    llm_settings: LlmSettings | None = None,
    force_full_matrix: bool | None = None,
    pipeline_v2: bool | None = None,
) -> dict[str, Any]:
    result = await run_ai_match_orchestrator(
        AiMatchOrchestratorRequest(
            text=text,
            preview_only=True,
            force_full_matrix=force_full_matrix,
            candidates_per_role=candidates_per_role,
            pipeline_v2=pipeline_v2,
        ),
        session,
        llm_settings=llm_settings,
    )
    package = dict(result.package)
    package.setdefault("match_trace", result.report_json.get("match_trace", []))
    package.setdefault(
        "package_strategy_decision",
        result.report_json.get("package_strategy_decision", {}),
    )
    package.setdefault("diagnostics", result.report_json.get("diagnostics", {}))
    return package


class _TraceBuilder:
    def __init__(self) -> None:
        self._items: list[StageResult] = []

    def add(self, result: StageResult) -> None:
        self._items.append(result)

    def extend_from_report(self, report_json: Mapping[str, Any]) -> None:
        self.add(_stage_from_semantic(report_json))
        self.add(_stage_from_category(report_json))
        self.add(_stage_from_matrix(report_json))
        self.add(_stage_from_package_strategy(report_json))
        self.add(_stage_from_full_matrix(report_json))
        self.add(_stage_from_composer(report_json))
        self.add(_stage_from_critic_repair(report_json))
        self.add(_stage_from_code_validation(report_json))
        self.add(_stage_from_report_json(report_json))

    def to_report_json(self) -> list[dict[str, Any]]:
        return [item.to_report_json() for item in self._items]


def _resolve_spec(
    request: AiMatchOrchestratorRequest,
    trace_builder: _TraceBuilder,
) -> StockSpec:
    started = time.perf_counter()
    started_at = _utc_now()
    if request.spec is not None:
        spec = request.spec
        source = "spec"
    else:
        text = str(request.text or "").strip()
        if not text:
            raise ValueError("AI match request requires text or spec.")
        spec = extract_stock_spec_for_text_match(text).spec_json
        source = "text"
    trace_builder.add(
        StageResult(
            stage_name="request_intake",
            status="success",
            started_at=started_at,
            finished_at=_utc_now(),
            duration_ms=_duration_ms(started),
            blocking=True,
            input_summary={
                "source": source,
                "distributor_code": request.distributor_code,
                "warehouse_constraints_present": bool(request.warehouse_constraints),
            },
            output_summary={
                "items_count": len(spec.items),
                "has_source_text": bool(spec.source_text),
            },
        )
    )
    return spec


def _effective_llm_settings(
    request: AiMatchOrchestratorRequest,
    settings: LlmSettings | None,
) -> LlmSettings:
    base = settings or get_llm_settings()
    updates: dict[str, Any] = {}
    if request.output_mode:
        updates["llm_configurator_output_mode"] = request.output_mode
    if request.package_limit is not None:
        updates["llm_configurator_max_package_chars"] = request.package_limit
    if request.force_full_matrix is not None:
        updates["llm_full_matrix_force"] = bool(request.force_full_matrix)
    if request.candidates_per_role is not None:
        updates["llm_component_candidates_per_role"] = request.candidates_per_role
    if request.pipeline_v2 is not None:
        updates["stock_match_pipeline_v2"] = bool(request.pipeline_v2)
    if request.preview_only:
        updates["llm_configurator_enabled"] = False
        updates["llm_configurator_mode"] = "disabled"
    if not request.allow_llm:
        updates.update(
            {
                "llm_provider": "disabled",
                "llm_configurator_enabled": False,
                "llm_configurator_mode": "disabled",
            }
        )
    return base.model_copy(update=updates) if updates else base


def _effective_web_settings(
    request: AiMatchOrchestratorRequest,
    settings: WebEvidenceSettings | None,
) -> WebEvidenceSettings | None:
    if request.allow_web_evidence:
        return settings
    effective = settings or WebEvidenceSettings()
    return effective.model_copy(
        update={
            "web_evidence_enabled": False,
            "web_evidence_provider": "disabled",
            "web_evidence_mode": "separate",
        }
    )


def _request_needs_runtime_kwargs(
    request: AiMatchOrchestratorRequest,
    *,
    llm_settings: LlmSettings | None,
    web_evidence_settings: WebEvidenceSettings | None,
    llm_configurator_client: LlmClient | None,
    web_search_provider: WebSearchProvider | None,
    evidence_cache: EvidenceSearchCache | None,
) -> bool:
    return any(
        [
            request.preview_only,
            not request.allow_llm,
            not request.allow_web_evidence,
            request.output_mode,
            request.force_full_matrix is not None,
            request.package_limit is not None,
            request.candidates_per_role is not None,
            request.pipeline_v2 is not None,
            llm_settings is not None,
            web_evidence_settings is not None,
            llm_configurator_client is not None,
            web_search_provider is not None,
            evidence_cache is not None,
        ]
    )


async def _call_match_func(
    match_func: MatchFunc,
    spec: StockSpec,
    session: AsyncSession,
    *,
    llm_configurator_client: LlmClient | None,
    llm_settings: LlmSettings,
    web_evidence_settings: WebEvidenceSettings | None,
    web_search_provider: WebSearchProvider | None,
    evidence_cache: EvidenceSearchCache | None,
    pass_runtime_kwargs: bool,
) -> MatchResult:
    if not pass_runtime_kwargs:
        return await match_func(spec, session)
    return await match_func(
        spec,
        session,
        llm_configurator_client=llm_configurator_client,
        llm_settings=llm_settings,
        web_evidence_settings=web_evidence_settings,
        web_search_provider=web_search_provider,
        evidence_cache=evidence_cache,
    )


def _use_pipeline_v2(
    request: AiMatchOrchestratorRequest,
    llm_settings: LlmSettings,
) -> bool:
    if request.pipeline_v2 is not None:
        return bool(request.pipeline_v2)
    return bool(getattr(llm_settings, "stock_match_pipeline_v2", False))


def _enrich_report_json(
    report_json: Mapping[str, Any],
    *,
    request: AiMatchOrchestratorRequest,
    llm_settings: LlmSettings,
    trace: _TraceBuilder,
    package: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(report_json)
    if package:
        result.setdefault("package_budget", package.get("package_budget") or {})
        result.setdefault("package_skipped_reason", package.get("package_skipped_reason"))
        result.setdefault(
            "full_matrix_evaluation_fallback_reason",
            package.get("full_matrix_evaluation_fallback_reason"),
        )
        for key in (
            "broad_matrix_count_by_role",
            "composer_package_candidate_count_by_role",
            "composer_package_candidate_total",
            "composer_package_candidate_ids_by_role",
            "dropped_before_composer_count_by_role",
            "dropped_before_composer_reason_by_role",
            "package_candidate_exposure_ratio_by_role",
            "package_candidate_exposure_policy",
            "package_candidate_exposure_incomplete",
            "package_candidate_exposure_incomplete_roles",
            "package_exposure_blocking_lifecycle_roles",
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
            "composer_package_roles",
            "roles_dropped_after_stage_a",
            "roles_dropped_before_category_planner",
            "roles_dropped_after_category_planner",
            "roles_dropped_during_materialization",
            "roles_dropped_reason_by_role",
            "role_source_by_role",
            "role_lifecycle_trace",
            "provider_error_type",
            "provider_context_limit",
        ):
            if key in package:
                result.setdefault(key, package.get(key))
    result["package_strategy_decision"] = _package_strategy_decision(
        result,
        request,
        llm_settings=llm_settings,
    )
    _log_package_strategy_decision(result["package_strategy_decision"])
    result["composer_attempt_decision"] = _audited_composer_attempt_decision(result)
    result["diagnostics"] = {
        **_safe_mapping(result.get("diagnostics")),
        "ai_match_orchestrator": {
            "version": "v1",
            "preview_only": request.preview_only,
            "allow_llm": request.allow_llm,
            "allow_web_evidence": request.allow_web_evidence,
            "force_full_matrix": bool(request.force_full_matrix),
            "package_limit": request.package_limit,
            "candidates_per_role": request.candidates_per_role,
        },
    }
    trace.extend_from_report(result)
    result["match_trace"] = trace.to_report_json()
    return result


def _enrich_v2_report_json(
    report_json: Mapping[str, Any],
    *,
    request: AiMatchOrchestratorRequest,
    llm_settings: LlmSettings,
    trace: _TraceBuilder,
    package: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(report_json)
    if package:
        for key in (
            "package_budget",
            "package_budget_warnings",
            "package_skipped_reason",
            "composer_package_candidate_count_by_role",
            "composer_package_candidate_total",
            "composer_package_candidate_ids_by_role",
            "composer_package_full_matrix_used",
            "composer_context_size",
            "candidate_universe_planner_mode",
            "candidate_universe_planner_output",
            "candidate_universe_category_plan",
            "primary_product_group",
            "procurement_intent",
            "selected_group_reason",
            "selected_product_group_reason",
            "competing_product_groups",
            "primary_object_indicators",
            "component_role_indicators",
            "excluded_category_groups",
            "planner_repair_attempted",
            "planner_repair_success",
            "planner_suspicion_reasons",
            "full_candidate_matrix_count_by_role",
            "full_candidate_matrix_count_by_category",
            "matrix_source_diagnostics",
            "composer_attempt_decision",
        ):
            if key in package:
                result[key] = package.get(key)
    result.setdefault(
        "package_strategy_decision",
        {
            "strategy": "composer_first_full_matrix",
            "decision": "use_full_matrix_if_under_limit",
            "reason": "pipeline_v2_composer_first",
            "full_matrix_evaluation_used": False,
            "package_over_budget": bool(
                _safe_mapping(result.get("package_budget")).get("over_budget")
            ),
            "candidate_count_total": int(
                result.get("composer_package_candidate_total") or 0
            ),
        },
    )
    result["pipeline_version"] = "v2_composer_first"
    result["diagnostics"] = {
        **_safe_mapping(result.get("diagnostics")),
        "pipeline_version": "v2_composer_first",
        "ai_match_orchestrator": {
            "version": "v2_composer_first",
            "preview_only": request.preview_only,
            "allow_llm": request.allow_llm,
            "allow_web_evidence": request.allow_web_evidence,
            "force_full_matrix": bool(request.force_full_matrix),
            "package_limit": request.package_limit,
            "candidates_per_role": request.candidates_per_role,
            "stock_match_pipeline_v2": bool(
                getattr(llm_settings, "stock_match_pipeline_v2", False)
            ),
        },
    }
    result["match_trace"] = trace.to_report_json()
    return normalize_composer_report_json(result)


def _package_strategy_decision(
    report_json: Mapping[str, Any],
    request: AiMatchOrchestratorRequest,
    *,
    llm_settings: LlmSettings,
) -> dict[str, Any]:
    budget = _safe_mapping(report_json.get("package_budget"))
    distiller_diagnostics = _safe_mapping(report_json.get("matrix_distiller_diagnostics"))
    broad_budget = _safe_mapping(
        distiller_diagnostics.get("package_budget_before_distillation")
    ) or _safe_mapping(distiller_diagnostics.get("package_budget")) or budget
    fallback_reason = _text_or_none(report_json.get("full_matrix_evaluation_fallback_reason"))
    package_skipped_reason = _text_or_none(report_json.get("package_skipped_reason"))
    semantic_fallback_reason = _text_or_none(
        report_json.get("semantic_planner_fallback_reason")
    )
    semantic_source = _text_or_none(report_json.get("semantic_planner_source"))
    provider_error_type = _text_or_none(report_json.get("provider_error_type"))
    provider_context_limit = _safe_mapping(report_json.get("provider_context_limit"))
    full_matrix_used = bool(report_json.get("full_matrix_evaluation_used"))
    exposure_incomplete = bool(report_json.get("package_candidate_exposure_incomplete"))
    composer_attempt = _safe_mapping(report_json.get("composer_attempt_decision"))
    candidate_count_total = int(
        composer_attempt.get("candidate_count_total")
        or report_json.get("composer_package_candidate_total")
        or 0
    )
    force_full_matrix = (
        bool(request.force_full_matrix)
        if request.force_full_matrix is not None
        else bool(getattr(llm_settings, "llm_full_matrix_force", False))
    )
    package_over_budget = budget.get("over_budget") is True
    broad_package_over_budget = broad_budget.get("over_budget") is True
    broad_package_under_budget = broad_budget.get("over_budget") is False
    full_matrix_required = bool(force_full_matrix or broad_package_over_budget)
    if _package_strategy_planner_unavailable(
        package_skipped_reason=package_skipped_reason,
        semantic_fallback_reason=semantic_fallback_reason,
        semantic_source=semantic_source,
        report_json=report_json,
    ):
        decision = "planner_unavailable"
        reason = (
            semantic_fallback_reason
            or package_skipped_reason
            or SEMANTIC_COMPLEX_FALLBACK_REASON
        )
        strategy = "fail_closed"
        package_over_budget = False
    elif exposure_incomplete:
        decision = "incomplete_matrix_exposure"
        reason = INCOMPLETE_MATRIX_EXPOSURE_REASON
        strategy = "fail_closed"
    elif package_skipped_reason:
        decision = "fail_over_budget"
        reason = package_skipped_reason
        strategy = "fail_closed"
    elif full_matrix_required or full_matrix_used:
        decision = "run_full_matrix"
        if (
            fallback_reason == PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON
            or provider_error_type == PROVIDER_CONTEXT_LIMIT_ERROR_TYPE
        ):
            reason = PROVIDER_CONTEXT_LIMIT_FALLBACK_REASON
        elif force_full_matrix:
            reason = "force_full_matrix"
        elif broad_package_over_budget:
            reason = "broad_package_over_budget"
        else:
            reason = "full_matrix_evaluation_used"
        strategy = (
            "full_matrix_reduced_package"
            if full_matrix_used
            else "full_matrix_or_fail_closed_required"
        )
    elif broad_package_under_budget:
        decision = "use_full_broad_package"
        reason = HIGH_QUALITY_BROAD_PACKAGE_UNDER_LIMIT_REASON
        strategy = "full_broad_package_direct_to_composer"
    elif package_over_budget:
        decision = "fail_over_budget"
        reason = "package_over_budget"
        strategy = "full_matrix_or_fail_closed_required"
    else:
        decision = "fail_over_budget"
        reason = "package_budget_unknown"
        strategy = "package_budget_unknown"
    return {
        "strategy": strategy,
        "broad_package_under_budget": bool(broad_package_under_budget),
        "force_full_matrix": force_full_matrix,
        "full_matrix_required": full_matrix_required,
        "decision": decision,
        "reason": reason,
        "full_matrix_evaluation_used": full_matrix_used,
        "full_matrix_evaluation_fallback_reason": fallback_reason,
        "package_over_budget": package_over_budget,
        "package_skipped_reason": package_skipped_reason,
        "max_package_chars": broad_budget.get("max_chars") or budget.get("max_chars"),
        "final_package_chars": broad_budget.get("final_chars") or budget.get("final_chars"),
        "candidate_count_total": candidate_count_total,
        "composer_package_candidate_count_by_role": _safe_mapping(
            report_json.get("composer_package_candidate_count_by_role")
        ),
        "broad_matrix_count_by_role": _safe_mapping(
            report_json.get("broad_matrix_count_by_role")
        ),
        "package_candidate_exposure_incomplete": exposure_incomplete,
        "package_candidate_exposure_policy": _safe_mapping(
            report_json.get("package_candidate_exposure_policy")
        ),
        "provider_error_type": provider_error_type,
        "provider_context_limit": provider_context_limit,
    }


def _package_strategy_planner_unavailable(
    *,
    package_skipped_reason: str | None,
    semantic_fallback_reason: str | None,
    semantic_source: str | None,
    report_json: Mapping[str, Any],
) -> bool:
    product_group = str(report_json.get("product_group") or "").strip()
    if package_skipped_reason == SEMANTIC_COMPLEX_FALLBACK_REASON:
        return True
    if semantic_fallback_reason == SEMANTIC_COMPLEX_FALLBACK_REASON:
        return True
    if semantic_source == SEMANTIC_SOURCE_FALLBACK_AFTER_LLM_TIMEOUT:
        return True
    return (
        semantic_fallback_reason == SEMANTIC_PLANNER_TIMEOUT_FALLBACK_REASON
        and product_group == "unknown"
    )


def _log_package_strategy_decision(decision: Mapping[str, Any]) -> None:
    fields = {
        "decision": decision.get("decision"),
        "reason": decision.get("reason"),
        "broad_package_under_budget": decision.get("broad_package_under_budget"),
        "force_full_matrix": decision.get("force_full_matrix"),
        "full_matrix_required": decision.get("full_matrix_required"),
        "package_over_budget": decision.get("package_over_budget"),
        "package_skipped_reason": decision.get("package_skipped_reason"),
        "provider_error_type": decision.get("provider_error_type"),
    }
    parts = ["package_strategy_decision"]
    for key, value in fields.items():
        if value in (None, "", [], {}):
            continue
        text = str(value).replace("\n", " ").replace("\r", " ")
        parts.append(f"{key}={text}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _audited_composer_attempt_decision(report_json: Mapping[str, Any]) -> dict[str, Any]:
    decision = _safe_mapping(report_json.get("composer_attempt_decision"))
    if not decision:
        return decision
    blocked_by = _string_list(decision.get("blocked_by"))
    invariant_applies = (
        bool(report_json.get("llm_configurator_enabled"))
        and bool(decision.get("package_present"))
        and not bool(decision.get("package_over_budget"))
        and _text_or_none(decision.get("package_skipped_reason")) is None
        and bool(decision.get("provider_configured"))
        and int(decision.get("candidate_count_total") or 0) > 0
    )
    composer_attempted = bool(report_json.get("online_composer_used"))
    if invariant_applies and not composer_attempted and not blocked_by:
        blocked_by = ["composer_attempt_audit_missing_attempt_diagnostics"]
        decision = {
            **decision,
            "should_attempt": True,
            "blocked_by": blocked_by,
            "audit_warning": (
                "Composer attempt invariant was not satisfied by report diagnostics."
            ),
        }
    return decision


def _package_stage_from_package(
    package: Mapping[str, Any],
    *,
    preview_only: bool,
) -> StageResult:
    return StageResult(
        stage_name="preview_package_build",
        status="success" if package else "skipped",
        blocking=False,
        input_summary={"preview_only": preview_only},
        output_summary={
            "package_present": bool(package),
            "package_budget": _safe_mapping(package.get("package_budget")),
            "package_skipped_reason": package.get("package_skipped_reason"),
        },
    )


def _stage_from_semantic(report_json: Mapping[str, Any]) -> StageResult:
    source = _text_or_none(report_json.get("semantic_planner_source"))
    fallback = _text_or_none(report_json.get("semantic_planner_fallback_reason"))
    status = (
        "success"
        if source in {"llm", "llm_repaired", "llm_minimal_fallback"}
        else "fallback"
        if fallback
        else "skipped"
    )
    return StageResult(
        stage_name="ai_semantic_planner",
        status=status,
        blocking=True,
        output_summary={
            "source": source,
            "product_group": report_json.get("product_group"),
            "primary_object": report_json.get("primary_object"),
            "required_roles": report_json.get("required_roles") or [],
        },
        error_type=_text_or_none(report_json.get("semantic_planner_error_type")),
        fallback_reason=fallback,
        diagnostics={
            "http_status": report_json.get("semantic_planner_http_status"),
            "parse_status": report_json.get("semantic_planner_parse_status"),
            "stage": report_json.get("semantic_planner_stage"),
            "stage_timeouts": report_json.get("semantic_planner_stage_timeouts"),
            "timeout_reason": report_json.get("semantic_planner_timeout_reason"),
            "timeout_seconds": report_json.get("semantic_planner_timeout_seconds"),
            "elapsed_ms": report_json.get("semantic_planner_elapsed_ms"),
            "repair_attempted": report_json.get("semantic_planner_repair_attempted"),
            "repair_success": report_json.get("semantic_planner_repair_success"),
            "minimal_router_used": report_json.get(
                "semantic_planner_minimal_router_used"
            ),
            "minimal_fallback_used": report_json.get(
                "semantic_planner_minimal_fallback_used"
            ),
            "empty_response_count": report_json.get(
                "semantic_planner_empty_response_count"
            ),
            "requirement_classifier_status": report_json.get(
                "requirement_classifier_status"
            ),
            "model": report_json.get("semantic_planner_model"),
            "provider": report_json.get("semantic_planner_provider"),
        },
    )


def _stage_from_category(report_json: Mapping[str, Any]) -> StageResult:
    source = _text_or_none(report_json.get("category_planner_source"))
    warnings = _string_list(report_json.get("category_plan_warnings"))
    status = "success" if source else "skipped"
    return StageResult(
        stage_name="ai_category_planner",
        status=status,
        blocking=True,
        output_summary={
            "source": source,
            "category_plan_source": report_json.get("category_plan_source"),
            "roles": sorted(_safe_mapping(report_json.get("category_plan")).keys()),
        },
        warnings=warnings,
        fallback_reason=None if source else "category_planner_not_used",
    )


def _stage_from_matrix(report_json: Mapping[str, Any]) -> StageResult:
    coverage = _safe_mapping(report_json.get("role_coverage_summary"))
    missing = _string_list(report_json.get("missing_required_roles"))
    return StageResult(
        stage_name="broad_matrix_builder",
        status=(
            "success"
            if coverage or report_json.get("component_candidate_matrix")
            else "skipped"
        ),
        blocking=True,
        output_summary={
            "count_by_role": report_json.get("count_by_role") or {},
            "missing_required_roles": missing,
            "role_coverage_summary": coverage,
        },
        warnings=[f"missing_required_role:{role}" for role in missing],
        fallback_reason="missing_required_roles_before_llm" if missing else None,
        diagnostics={
            "candidate_inclusion_policy": report_json.get("candidate_inclusion_policy"),
        },
    )


def _stage_from_package_strategy(report_json: Mapping[str, Any]) -> StageResult:
    decision = _safe_mapping(report_json.get("package_strategy_decision"))
    package_skipped_reason = _text_or_none(report_json.get("package_skipped_reason"))
    status = "failed" if package_skipped_reason else "success"
    return StageResult(
        stage_name="package_strategy",
        status=status,
        blocking=bool(package_skipped_reason),
        output_summary=decision,
        warnings=_string_list(report_json.get("package_budget_warnings")),
        fallback_reason=decision.get("full_matrix_evaluation_fallback_reason"),
    )


def _stage_from_full_matrix(report_json: Mapping[str, Any]) -> StageResult:
    used = bool(report_json.get("full_matrix_evaluation_used"))
    fallback = _text_or_none(report_json.get("full_matrix_evaluation_fallback_reason"))
    failed_chunks = report_json.get("full_matrix_failed_chunks") or []
    status = "success" if used and not failed_chunks else "fallback" if fallback else "skipped"
    return StageResult(
        stage_name="optional_full_matrix_ai_evaluation",
        status=status,
        blocking=False,
        output_summary={
            "used": used,
            "fallback_reason": fallback,
            "failed_chunks_count": len(failed_chunks) if isinstance(failed_chunks, list) else 0,
        },
        fallback_reason=fallback,
        diagnostics={
            "role_chunk_count_by_role": report_json.get("role_chunk_count_by_role") or {},
            "evaluated_candidate_count_by_role": (
                report_json.get("evaluated_candidate_count_by_role") or {}
            ),
            "selected_candidate_count_by_role": (
                report_json.get("selected_candidate_count_by_role") or {}
            ),
        },
    )


def _stage_from_composer(report_json: Mapping[str, Any]) -> StageResult:
    decision = _safe_mapping(report_json.get("composer_attempt_decision"))
    attempted = bool(report_json.get("online_composer_used"))
    status = "success" if attempted else "skipped"
    if report_json.get("llm_error_type"):
        status = "failed"
    elif decision.get("blocked_by"):
        status = "skipped"
    return StageResult(
        stage_name="ai_composer",
        status=status,
        blocking=True,
        output_summary={
            "attempted": attempted,
            "used": bool(report_json.get("llm_configurator_used")),
            "proposal_count": report_json.get("llm_proposals_count"),
            "primary_recommendation_status": report_json.get(
                "primary_recommendation_status"
            ),
            "composer_attempt_decision": decision,
        },
        error_type=_text_or_none(report_json.get("llm_error_type")),
        fallback_reason=_text_or_none(report_json.get("llm_fallback_reason")),
        diagnostics={
            "parse_diagnostics": report_json.get("llm_parse_diagnostics") or {},
            "http_status": report_json.get("llm_http_status"),
        },
    )


def _stage_from_critic_repair(report_json: Mapping[str, Any]) -> StageResult:
    repair_attempted = bool(
        report_json.get("llm_repair_attempted")
        or report_json.get("no_recommendation_coverage_repair_attempted")
    )
    coverage_rejected = bool(report_json.get("no_recommendation_coverage_rejected"))
    status = "failed" if coverage_rejected else "success" if repair_attempted else "skipped"
    return StageResult(
        stage_name="ai_critic_repair_coverage_gate",
        status=status,
        blocking=coverage_rejected,
        output_summary={
            "repair_attempted": repair_attempted,
            "repair_success": bool(
                report_json.get("llm_repair_success")
                or report_json.get("no_recommendation_coverage_repair_success")
            ),
            "coverage_gate_passed": bool(
                report_json.get("no_recommendation_coverage_gate_passed")
            ),
            "coverage_rejected": coverage_rejected,
        },
        fallback_reason=_text_or_none(
            report_json.get("no_recommendation_coverage_repair_reason")
        ),
        diagnostics={
            "no_recommendation_coverage": report_json.get("no_recommendation_coverage")
            or {},
            "thresholds": report_json.get("no_recommendation_coverage_thresholds") or {},
        },
    )


def _stage_from_code_validation(report_json: Mapping[str, Any]) -> StageResult:
    rejected = int(report_json.get("validation_rejected_count") or 0)
    valid = int(report_json.get("valid_proposals_count") or 0)
    return StageResult(
        stage_name="code_validation",
        status="success" if valid or not rejected else "failed",
        blocking=True,
        output_summary={
            "valid_proposals_count": valid,
            "validation_rejected_count": rejected,
            "selection_skipped_count": report_json.get("selection_skipped_count"),
        },
        warnings=_string_list(report_json.get("ai_recommendations_validation_warnings")),
        diagnostics={
            "ai_validation_summary": report_json.get("ai_validation_summary") or {},
            "rejected_reasons_top": report_json.get("rejected_reasons_top") or [],
        },
    )


def _stage_from_report_json(report_json: Mapping[str, Any]) -> StageResult:
    return StageResult(
        stage_name="report_json_excel_telegram_source",
        status="success",
        blocking=False,
        output_summary={
            "has_report_json": True,
            "primary_recommendation_status": report_json.get(
                "primary_recommendation_status"
            ),
            "excel_source": "persisted_report_json",
            "telegram_result_source": "api_get_persisted_report_json",
        },
    )


def _duration_ms(started: float) -> int:
    return int(max(0.0, time.perf_counter() - started) * 1000)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
