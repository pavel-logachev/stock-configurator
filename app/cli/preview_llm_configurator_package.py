from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import gettempdir

from pydantic import ValidationError

from app.core.config import LlmSettings, get_llm_settings, get_web_evidence_settings
from app.core.database import get_session_factory
from app.evidence.web_evidence import (
    DisabledWebSearchProvider,
    build_evidence_tasks_from_component_matrix,
    collect_web_evidence,
)
from app.matching.ai_match_orchestrator import (
    preview_llm_configurator_package_from_text as _preview_llm_configurator_package_from_text,
)
from app.matching.match_engine import (
    plan_semantic_matrix_for_text,
)

PREVIEW_CANDIDATES_PER_ROLE = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the safe LLM Configuration Composer package summary."
    )
    parser.add_argument("--text", required=True, help="Free-form user request text.")
    parser.add_argument(
        "--component-candidates-per-role",
        type=int,
        default=LlmSettings().llm_component_candidates_per_role,
        help="Maximum component candidates per role in the preview package.",
    )
    parser.add_argument(
        "--with-evidence-preview",
        action="store_true",
        help="Include Web Evidence V0 tasks and coverage in the preview.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the preview summary as JSON. Default output is concise human text.",
    )
    parser.add_argument(
        "--show-candidates",
        type=int,
        default=PREVIEW_CANDIDATES_PER_ROLE,
        help="How many first candidates to show per active role.",
    )
    parser.add_argument(
        "--show-dropped-categories",
        action="store_true",
        help="Show category planner rejected/dropped category warnings.",
    )
    parser.add_argument(
        "--show-filter-diagnostics",
        action="store_true",
        help="Show role filter diagnostics for active roles.",
    )
    parser.add_argument(
        "--no-llm-semantic-planner",
        "--deterministic-only",
        dest="deterministic_only",
        action="store_true",
        help=(
            "Skip the LLM semantic planner and run the deterministic preview path only."
        ),
    )
    parser.add_argument(
        "--with-prompt-package-size",
        action="store_true",
        default=True,
        help="Include prompt package size and budget. This is enabled by default.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        default=True,
        help="Do not call a real web search provider. This is the default.",
    )
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Run only the Semantic Planner and print safe diagnostics/plan JSON.",
    )
    parser.add_argument(
        "--force-full-matrix",
        action="store_true",
        help="Force bounded full-matrix AI evaluation before Composer package preview.",
    )
    parser.add_argument(
        "--pipeline-v2",
        action="store_true",
        help="Use the Composer-first v2 pipeline preview.",
    )
    parser.add_argument(
        "--dump-composer-package",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Write a safe final Composer package dump to PATH, or to /tmp when PATH "
            "is omitted."
        ),
    )
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        llm_settings = _preview_llm_settings(deterministic_only=args.deterministic_only)
        if args.semantic_only:
            preview = _semantic_only_preview(args.text, llm_settings=llm_settings)
            if args.deterministic_only:
                _mark_deterministic_preview_only(preview)
            if args.json:
                print(json.dumps(preview, ensure_ascii=False, indent=2))
            else:
                print(_human_semantic_only_preview(preview))
            return 0
        session_factory = get_session_factory()
        async with session_factory() as session:
            package = await build_llm_configurator_package_from_text(
                args.text,
                session,
                llm_settings=llm_settings,
                candidates_per_role=args.component_candidates_per_role,
                force_full_matrix=args.force_full_matrix,
                pipeline_v2=args.pipeline_v2,
            )
        preview = _preview_summary(
            package,
            show_candidates=args.show_candidates,
            show_filter_diagnostics=args.show_filter_diagnostics,
            show_dropped_categories=args.show_dropped_categories,
        )
        if args.deterministic_only:
            _mark_deterministic_preview_only(preview)
        if args.with_evidence_preview:
            preview["web_evidence_preview"] = _evidence_preview_summary(
                package,
                no_network=args.no_network,
            )
        if args.dump_composer_package is not None:
            dump_path = _dump_composer_package(
                package,
                path_arg=args.dump_composer_package,
            )
            preview["composer_package_dump_path"] = str(dump_path)
            print(f"Composer package dump written to {dump_path}", file=sys.stderr)
    except ValidationError as exc:
        print(f"Stock Spec validation failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Could not build LLM configurator package: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    else:
        print(_human_preview(preview))
    return 0


async def build_llm_configurator_package_from_text(
    text: str,
    session: object,
    *,
    candidates_per_role: int | None = None,
    llm_settings: LlmSettings | None = None,
    force_full_matrix: bool | None = None,
    pipeline_v2: bool | None = None,
) -> dict[str, object]:
    return await _preview_llm_configurator_package_from_text(
        text,
        session,  # type: ignore[arg-type]
        candidates_per_role=candidates_per_role,
        llm_settings=llm_settings,
        force_full_matrix=force_full_matrix,
        pipeline_v2=pipeline_v2,
    )


def _preview_llm_settings(*, deterministic_only: bool) -> LlmSettings:
    if deterministic_only:
        return LlmSettings(
            llm_provider="disabled",
            llm_configurator_enabled=False,
            llm_configurator_mode="disabled",
        )
    settings = get_llm_settings()
    return settings.model_copy(
        update={
            "llm_configurator_enabled": False,
            "llm_configurator_mode": "disabled",
        }
    )


def _mark_deterministic_preview_only(preview: dict[str, object]) -> None:
    source = str(preview.get("semantic_planner_source") or "").strip()
    if source != "llm":
        preview["semantic_planner_source"] = "deterministic_preview_only"
        preview["semantic_planner_used"] = False
        preview["selected_product_group_reason"] = (
            "Preview was run without LLM semantic planner"
        )
        preview["semantic_planner_fallback_reason"] = "deterministic_preview_only"


def _dump_composer_package(
    package: dict[str, object],
    *,
    path_arg: str,
) -> Path:
    path = _composer_package_dump_path(path_arg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "package_candidate_summary": {
            "broad_matrix_count_by_role": package.get("broad_matrix_count_by_role")
            or {},
            "composer_package_candidate_count_by_role": package.get(
                "composer_package_candidate_count_by_role"
            )
            or {},
            "composer_package_candidate_total": package.get(
                "composer_package_candidate_total"
            )
            or 0,
            "composer_package_candidate_ids_by_role": package.get(
                "composer_package_candidate_ids_by_role"
            )
            or {},
            "dropped_before_composer_count_by_role": package.get(
                "dropped_before_composer_count_by_role"
            )
            or {},
            "dropped_before_composer_reason_by_role": package.get(
                "dropped_before_composer_reason_by_role"
            )
            or {},
            "package_candidate_exposure_policy": package.get(
                "package_candidate_exposure_policy"
            )
            or {},
            "package_candidate_exposure_incomplete": bool(
                package.get("package_candidate_exposure_incomplete")
            ),
        },
        "composer_package": package,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _composer_package_dump_path(path_arg: str) -> Path:
    if path_arg and path_arg != "auto":
        return Path(path_arg)
    tmp_dir = Path("/tmp")
    if not tmp_dir.exists():
        tmp_dir = Path(gettempdir())
    return tmp_dir / "stock_configurator_composer_package.json"


def _semantic_only_preview(
    text: str,
    *,
    llm_settings: LlmSettings,
) -> dict[str, object]:
    plan = plan_semantic_matrix_for_text(text, llm_settings=llm_settings)
    return {
        "semantic_only": True,
        "product_group": plan.get("product_group") or plan.get("primary_product_group"),
        "primary_object": plan.get("primary_object"),
        "semantic_planner_source": plan.get("semantic_planner_source"),
        "semantic_planner_used": plan.get("semantic_planner_used"),
        "semantic_planner_confidence": plan.get("semantic_planner_confidence"),
        "semantic_planner_error_type": plan.get("semantic_planner_error_type"),
        "semantic_planner_http_status": plan.get("semantic_planner_http_status"),
        "semantic_planner_parse_status": plan.get("semantic_planner_parse_status"),
        "semantic_planner_fallback_reason": plan.get("semantic_planner_fallback_reason"),
        "semantic_planner_attempts": plan.get("semantic_planner_attempts") or [],
        "semantic_planner_stage": plan.get("semantic_planner_stage"),
        "semantic_planner_stage_timeouts": plan.get(
            "semantic_planner_stage_timeouts"
        )
        or [],
        "semantic_planner_timeout_reason": plan.get(
            "semantic_planner_timeout_reason"
        ),
        "semantic_planner_timeout_seconds": plan.get(
            "semantic_planner_timeout_seconds"
        ),
        "semantic_planner_elapsed_ms": plan.get("semantic_planner_elapsed_ms"),
        "semantic_planner_repair_attempted": plan.get(
            "semantic_planner_repair_attempted"
        ),
        "semantic_planner_repair_success": plan.get("semantic_planner_repair_success"),
        "semantic_planner_minimal_router_used": plan.get(
            "semantic_planner_minimal_router_used"
        ),
        "semantic_planner_minimal_fallback_used": plan.get(
            "semantic_planner_minimal_fallback_used"
        ),
        "semantic_planner_empty_response_count": plan.get(
            "semantic_planner_empty_response_count"
        ),
        "semantic_planner_empty_response_reason": plan.get(
            "semantic_planner_empty_response_reason"
        ),
        "requirement_classifier_status": plan.get("requirement_classifier_status"),
        "requirement_classifier_error_type": plan.get(
            "requirement_classifier_error_type"
        ),
        "requirement_classifier_parse_status": plan.get(
            "requirement_classifier_parse_status"
        ),
        "requirement_classifier_incomplete_reason": plan.get(
            "requirement_classifier_incomplete_reason"
        ),
        "requirement_source_coverage": plan.get("requirement_source_coverage") or [],
        "requirement_source_coverage_percent": plan.get(
            "requirement_source_coverage_percent"
        ),
        "unclassified_source_fragments": plan.get("unclassified_source_fragments")
        or [],
        "synthetic_requirement_count": plan.get("synthetic_requirement_count") or 0,
        "source_backed_requirement_count": plan.get(
            "source_backed_requirement_count"
        )
        or 0,
        "requirement_classifier_repair_quality": plan.get(
            "requirement_classifier_repair_quality"
        ),
        "requirement_classifier_repair_accepted": plan.get(
            "requirement_classifier_repair_accepted"
        ),
        "semantic_planner_model": plan.get("semantic_planner_model"),
        "semantic_planner_provider": plan.get("semantic_planner_provider"),
        "selected_product_group_reason": plan.get("selected_product_group_reason"),
        "deterministic_product_group_hint": plan.get("deterministic_product_group_hint"),
        "semantic_planner_disagreement": plan.get("semantic_planner_disagreement"),
        "matrix_blueprint": plan.get("matrix_blueprint") or {},
        "matrix_blueprint_roles": plan.get("matrix_blueprint_roles") or [],
        "classified_requirements": plan.get("classified_requirements") or [],
        "purchasable_role_requirements": plan.get("purchasable_role_requirements")
        or [],
        "primary_object_feature_requirements": plan.get(
            "primary_object_feature_requirements"
        )
        or [],
        "accessory_or_consumable_requirements": plan.get(
            "accessory_or_consumable_requirements"
        )
        or [],
        "engineering_check_requirements": plan.get("engineering_check_requirements")
        or [],
        "unmapped_requirements_blocking": plan.get("unmapped_requirements_blocking")
        or [],
        "embedded_requirements": plan.get("embedded_requirements") or [],
        "not_primary_product_groups": plan.get("not_primary_product_groups") or [],
        "required_capabilities": plan.get("required_capabilities") or [],
        "optional_capabilities": plan.get("optional_capabilities") or [],
        "unsupported_or_unmapped_requirements": plan.get(
            "unsupported_or_unmapped_requirements"
        )
        or [],
        "planner_warnings": plan.get("planner_warnings") or [],
    }


def _preview_summary(
    package: dict[str, object],
    *,
    show_candidates: int,
    show_filter_diagnostics: bool,
    show_dropped_categories: bool,
) -> dict[str, object]:
    matrix = package.get("component_candidate_matrix")
    matrix_by_role = matrix if isinstance(matrix, dict) else {}
    package_json = json.dumps(package, ensure_ascii=False, sort_keys=True)
    package_budget = package.get("package_budget") if isinstance(package, dict) else {}
    package_chars = (
        int(package_budget.get("final_chars"))
        if isinstance(package_budget, dict)
        and isinstance(package_budget.get("final_chars"), int)
        else len(package_json)
    )
    coverage = package.get("component_matrix_coverage_summary")
    coverage_by_role = coverage if isinstance(coverage, dict) else {}
    count_by_role = {
        role: len(rows) if isinstance(rows, list) else 0
        for role, rows in matrix_by_role.items()
        if isinstance(rows, list) and rows
    }
    broad_count_by_role = package.get("broad_count_by_role")
    if isinstance(broad_count_by_role, dict) and broad_count_by_role:
        count_by_role = {
            str(role): count
            for role, count in broad_count_by_role.items()
            if isinstance(count, int) and count > 0
        }
    warnings = list(package.get("category_plan_warnings") or [])
    if not show_dropped_categories:
        warnings = [
            warning
            for warning in warnings
            if not str(warning).startswith("category_plan_category_incompatible")
        ]
    return {
        "pipeline_version": package.get("pipeline_version"),
        "product_group": package.get("product_group"),
        "primary_object": package.get("primary_object"),
        "semantic_planner_source": package.get("semantic_planner_source"),
        "semantic_planner_used": package.get("semantic_planner_used"),
        "semantic_planner_confidence": package.get("semantic_planner_confidence"),
        "semantic_planner_error_type": package.get("semantic_planner_error_type"),
        "semantic_planner_http_status": package.get("semantic_planner_http_status"),
        "semantic_planner_parse_status": package.get("semantic_planner_parse_status"),
        "semantic_planner_fallback_reason": package.get(
            "semantic_planner_fallback_reason"
        ),
        "semantic_planner_attempts": package.get("semantic_planner_attempts") or [],
        "semantic_planner_stage": package.get("semantic_planner_stage"),
        "semantic_planner_stage_timeouts": package.get(
            "semantic_planner_stage_timeouts"
        )
        or [],
        "semantic_planner_timeout_reason": package.get(
            "semantic_planner_timeout_reason"
        ),
        "semantic_planner_timeout_seconds": package.get(
            "semantic_planner_timeout_seconds"
        ),
        "semantic_planner_elapsed_ms": package.get("semantic_planner_elapsed_ms"),
        "semantic_planner_repair_attempted": package.get(
            "semantic_planner_repair_attempted"
        ),
        "semantic_planner_repair_success": package.get(
            "semantic_planner_repair_success"
        ),
        "semantic_planner_minimal_router_used": package.get(
            "semantic_planner_minimal_router_used"
        ),
        "semantic_planner_minimal_fallback_used": package.get(
            "semantic_planner_minimal_fallback_used"
        ),
        "semantic_planner_empty_response_count": package.get(
            "semantic_planner_empty_response_count"
        ),
        "semantic_planner_empty_response_reason": package.get(
            "semantic_planner_empty_response_reason"
        ),
        "requirement_classifier_status": package.get("requirement_classifier_status"),
        "requirement_classifier_error_type": package.get(
            "requirement_classifier_error_type"
        ),
        "requirement_classifier_parse_status": package.get(
            "requirement_classifier_parse_status"
        ),
        "requirement_classifier_incomplete_reason": package.get(
            "requirement_classifier_incomplete_reason"
        ),
        "requirement_source_coverage": package.get("requirement_source_coverage")
        or [],
        "requirement_source_coverage_percent": package.get(
            "requirement_source_coverage_percent"
        ),
        "unclassified_source_fragments": package.get(
            "unclassified_source_fragments"
        )
        or [],
        "pre_composer_requirement_classifier_status": package.get(
            "pre_composer_requirement_classifier_status"
        ),
        "pre_composer_requirement_source_coverage_percent": package.get(
            "pre_composer_requirement_source_coverage_percent"
        ),
        "pre_composer_unclassified_source_fragments": package.get(
            "pre_composer_unclassified_source_fragments"
        )
        or [],
        "pre_composer_semantic_diagnostics_are_blocking": bool(
            package.get("pre_composer_semantic_diagnostics_are_blocking")
        ),
        "synthetic_requirement_count": package.get("synthetic_requirement_count") or 0,
        "source_backed_requirement_count": package.get(
            "source_backed_requirement_count"
        )
        or 0,
        "requirement_classifier_repair_quality": package.get(
            "requirement_classifier_repair_quality"
        ),
        "requirement_classifier_repair_accepted": package.get(
            "requirement_classifier_repair_accepted"
        ),
        "semantic_planner_model": package.get("semantic_planner_model"),
        "semantic_planner_provider": package.get("semantic_planner_provider"),
        "candidate_universe_planner_mode": package.get(
            "candidate_universe_planner_mode"
        ),
        "primary_product_group": package.get("primary_product_group"),
        "procurement_intent": package.get("procurement_intent"),
        "selected_group_reason": package.get("selected_group_reason"),
        "selected_product_group_reason": package.get("selected_product_group_reason"),
        "competing_product_groups": package.get("competing_product_groups") or [],
        "primary_object_indicators": package.get("primary_object_indicators") or [],
        "component_role_indicators": package.get("component_role_indicators") or [],
        "excluded_category_groups": package.get("excluded_category_groups") or [],
        "planner_repair_attempted": bool(package.get("planner_repair_attempted")),
        "planner_repair_success": bool(package.get("planner_repair_success")),
        "planner_suspicion_reasons": package.get("planner_suspicion_reasons") or [],
        "deterministic_product_group_hint": package.get(
            "deterministic_product_group_hint"
        ),
        "semantic_planner_disagreement": package.get("semantic_planner_disagreement"),
        "matrix_blueprint": package.get("matrix_blueprint") or {},
        "matrix_blueprint_roles": package.get("matrix_blueprint_roles") or [],
        "classified_requirements": package.get("classified_requirements") or [],
        "purchasable_role_requirements": package.get("purchasable_role_requirements")
        or [],
        "primary_object_feature_requirements": package.get(
            "primary_object_feature_requirements"
        )
        or [],
        "accessory_or_consumable_requirements": package.get(
            "accessory_or_consumable_requirements"
        )
        or [],
        "engineering_check_requirements": package.get("engineering_check_requirements")
        or [],
        "unmapped_requirements_blocking": package.get("unmapped_requirements_blocking")
        or [],
        "embedded_requirements": package.get("embedded_requirements") or [],
        "not_primary_product_groups": package.get("not_primary_product_groups") or [],
        "required_capabilities": package.get("required_capabilities") or [],
        "optional_capabilities": package.get("optional_capabilities") or [],
        "category_catalog_summary": package.get("category_catalog_summary") or {},
        "category_planner_source": package.get("category_planner_source"),
        "category_plan_source": package.get("category_plan_source"),
        "category_plan": package.get("category_plan") or {},
        "category_plan_entries": package.get("category_plan_entries") or [],
        "candidate_universe_planner_output": package.get(
            "candidate_universe_planner_output"
        )
        or {},
        "candidate_universe_category_plan": package.get(
            "candidate_universe_category_plan"
        )
        or {},
        "category_planner_missing_required_roles": package.get(
            "category_planner_missing_required_roles"
        )
        or [],
        "category_planner_repair_attempted": bool(
            package.get("category_planner_repair_attempted")
        ),
        "category_planner_repair_success": bool(
            package.get("category_planner_repair_success")
        ),
        "category_planner_repair_reason": package.get(
            "category_planner_repair_reason"
        ),
        "category_planner_repaired_roles": package.get(
            "category_planner_repaired_roles"
        )
        or [],
        "category_planner_unresolved_required_roles": package.get(
            "category_planner_unresolved_required_roles"
        )
        or [],
        "category_plan_warnings": warnings,
        "stage_a_broad_roles": package.get("stage_a_broad_roles") or [],
        "semantic_matrix_blueprint_roles": package.get(
            "semantic_matrix_blueprint_roles"
        )
        or [],
        "requirement_classifier_roles": package.get("requirement_classifier_roles")
        or [],
        "effective_matrix_roles_before_category_planner": package.get(
            "effective_matrix_roles_before_category_planner"
        )
        or [],
        "category_planner_input_roles": package.get("category_planner_input_roles")
        or [],
        "category_planner_output_roles": package.get("category_planner_output_roles")
        or [],
        "validated_category_plan_roles": package.get("validated_category_plan_roles")
        or [],
        "materialized_matrix_roles": package.get("materialized_matrix_roles") or [],
        "composer_package_roles": package.get("composer_package_roles") or [],
        "roles_dropped_after_stage_a": package.get("roles_dropped_after_stage_a")
        or [],
        "roles_dropped_before_category_planner": package.get(
            "roles_dropped_before_category_planner"
        )
        or [],
        "roles_dropped_after_category_planner": package.get(
            "roles_dropped_after_category_planner"
        )
        or [],
        "roles_dropped_during_materialization": package.get(
            "roles_dropped_during_materialization"
        )
        or [],
        "roles_dropped_reason_by_role": package.get("roles_dropped_reason_by_role")
        or {},
        "role_source_by_role": package.get("role_source_by_role") or {},
        "role_lifecycle_trace": package.get("role_lifecycle_trace") or [],
        "missing_required_roles": package.get("missing_required_roles") or [],
        "missing_required_roles_before_llm": package.get(
            "missing_required_roles_before_llm"
        )
        or package.get("missing_required_roles")
        or [],
        "missing_required_capabilities": package.get("missing_required_capabilities") or [],
        "missing_required_capabilities_before_llm": package.get(
            "missing_required_capabilities_before_llm"
        )
        or package.get("missing_required_capabilities")
        or [],
        "normalized_requirements": package.get("normalized_requirements"),
        "count_by_role": count_by_role,
        "full_candidate_matrix_count_by_role": package.get(
            "full_candidate_matrix_count_by_role"
        )
        or {},
        "full_candidate_matrix_count_by_category": package.get(
            "full_candidate_matrix_count_by_category"
        )
        or {},
        "broad_matrix_count_by_role": package.get("broad_matrix_count_by_role") or {},
        "composer_package_candidate_count_by_role": package.get(
            "composer_package_candidate_count_by_role"
        )
        or {},
        "composer_package_candidate_total": package.get(
            "composer_package_candidate_total"
        )
        or 0,
        "composer_package_candidate_ids_by_role": package.get(
            "composer_package_candidate_ids_by_role"
        )
        or {},
        "dropped_before_composer_count_by_role": package.get(
            "dropped_before_composer_count_by_role"
        )
        or {},
        "dropped_before_composer_reason_by_role": package.get(
            "dropped_before_composer_reason_by_role"
        )
        or {},
        "package_candidate_exposure_ratio_by_role": package.get(
            "package_candidate_exposure_ratio_by_role"
        )
        or {},
        "package_candidate_exposure_policy": package.get(
            "package_candidate_exposure_policy"
        )
        or {},
        "package_candidate_exposure_incomplete": bool(
            package.get("package_candidate_exposure_incomplete")
        ),
        "package_candidate_exposure_incomplete_roles": package.get(
            "package_candidate_exposure_incomplete_roles"
        )
        or [],
        "composer_package_full_matrix_used": bool(
            package.get("composer_package_full_matrix_used")
        ),
        "composer_context_size": package.get("composer_context_size") or {},
        "verbose_context_size": package.get("verbose_context_size") or {},
        "compact_context_size": package.get("compact_context_size") or {},
        "selected_context_size": package.get("selected_context_size") or {},
        "verbose_context_chars": package.get("verbose_context_chars"),
        "compact_context_chars": package.get("compact_context_chars"),
        "selected_context_chars": package.get("selected_context_chars"),
        "selected_package_mode": package.get("selected_package_mode"),
        "v2_package_mode": package.get("v2_package_mode"),
        "chars_by_section": package.get("chars_by_section") or {},
        "avg_chars_per_candidate_by_role": package.get(
            "avg_chars_per_candidate_by_role"
        )
        or {},
        "removed_verbose_fields": package.get("removed_verbose_fields") or [],
        "removed_verbose_field_counts": package.get("removed_verbose_field_counts")
        or {},
        "compact_candidate_total": package.get("compact_candidate_total") or 0,
        "compact_candidate_count_by_role": package.get(
            "compact_candidate_count_by_role"
        )
        or {},
        "compact_candidate_ids_hash": package.get("compact_candidate_ids_hash"),
        "compact_package_full_matrix_used": bool(
            package.get("compact_package_full_matrix_used")
        ),
        "package_candidate_loss": bool(package.get("package_candidate_loss")),
        "provider_context_limit_retry_compact_attempted": bool(
            package.get("provider_context_limit_retry_compact_attempted")
        ),
        "provider_context_limit_retry_compact_success": bool(
            package.get("provider_context_limit_retry_compact_success")
        ),
        "provider_context_limit_original_chars": package.get(
            "provider_context_limit_original_chars"
        ),
        "provider_context_limit_compact_chars": package.get(
            "provider_context_limit_compact_chars"
        ),
        "provider_context_limit_after_compact": bool(
            package.get("provider_context_limit_after_compact")
        ),
        "composer_attempt_decision": package.get("composer_attempt_decision") or {},
        "expected_composer_mode": package.get("expected_composer_mode")
        or (package.get("composer_attempt_decision") or {}).get(
            "expected_composer_mode"
        ),
        "role_evaluation_would_run": bool(
            (package.get("composer_attempt_decision") or {}).get(
                "role_evaluation_would_run"
            )
        ),
        "package_mode": package.get("v2_package_mode")
        or package.get("selected_package_mode")
        or (package.get("composer_attempt_decision") or {}).get("package_mode"),
        "package_size": package.get("selected_context_size")
        or package.get("composer_context_size")
        or (package.get("composer_attempt_decision") or {}).get("package_size")
        or {},
        "llm_call_count": package.get("llm_call_count"),
        "llm_call_stages": package.get("llm_call_stages") or [],
        "llm_call_budget_exceeded": bool(package.get("llm_call_budget_exceeded")),
        "max_llm_calls_per_match": package.get("max_llm_calls_per_match"),
        "matrix_source_diagnostics": package.get("matrix_source_diagnostics") or {},
        "package_exposure_blocking_lifecycle_roles": package.get(
            "package_exposure_blocking_lifecycle_roles"
        )
        or [],
        "role_coverage_summary": package.get("role_coverage_summary") or {},
        "matrix_coverage_by_role": coverage_by_role,
        "matrix_distiller_used": bool(package.get("matrix_distiller_used")),
        "matrix_distiller_source": package.get("matrix_distiller_source") or "skipped",
        "matrix_distiller_diagnostics": package.get("matrix_distiller_diagnostics") or {},
        "full_matrix_evaluation_used": bool(
            package.get("full_matrix_evaluation_used")
        ),
        "full_matrix_evaluation_fallback_reason": package.get(
            "full_matrix_evaluation_fallback_reason"
        ),
        "provider_error_type": package.get("provider_error_type"),
        "provider_context_limit": package.get("provider_context_limit") or {},
        "full_matrix_failed_chunks": package.get("full_matrix_failed_chunks") or [],
        "distilled_count_by_role": package.get("distilled_count_by_role")
        or {
            role: len(rows) if isinstance(rows, list) else 0
            for role, rows in matrix_by_role.items()
            if isinstance(rows, list) and rows
        },
        "first_candidates_by_role": {
            role: _first_candidates(rows, limit=show_candidates)
            for role, rows in matrix_by_role.items()
            if isinstance(rows, list) and rows
        },
        "package_approximate_size": {
            "chars": package_chars,
            "tokens_estimate": max(1, package_chars // 4),
        },
        "package_budget": package.get("package_budget") or {},
        "package_budget_warnings": package.get("package_budget_warnings") or [],
        "package_skipped_reason": package.get("package_skipped_reason"),
        "package_strategy_decision": package.get("package_strategy_decision") or {},
        "match_trace": package.get("match_trace") or [],
        "diagnostics": package.get("diagnostics") or {},
        "llm_fallback_reason": package.get("llm_fallback_reason"),
        "ready_candidates_excluded_reason": package.get(
            "ready_candidates_excluded_reason"
        ),
        "ready_candidates_limit": package.get("ready_candidates_limit"),
        "filter_diagnostics": _filter_diagnostics_for_active_roles(package)
        if show_filter_diagnostics
        else {},
    }


def _human_preview(preview: dict[str, object]) -> str:
    lines: list[str] = []
    if preview.get("pipeline_version"):
        lines.append(f"pipeline_version: {preview.get('pipeline_version')}")
    lines.append(f"product_group: {preview.get('product_group')}")
    if preview.get("primary_object"):
        lines.append(f"primary_object: {preview.get('primary_object')}")
    if preview.get("semantic_planner_source"):
        lines.append(
            "semantic_planner: "
            f"{preview.get('semantic_planner_source')} "
            f"({preview.get('semantic_planner_confidence')})"
        )
    if preview.get("semantic_planner_provider") or preview.get("semantic_planner_model"):
        lines.append(
            "semantic_planner_provider: "
            f"{preview.get('semantic_planner_provider')} "
            f"{preview.get('semantic_planner_model')}"
        )
    if preview.get("candidate_universe_planner_mode"):
        lines.append(
            "candidate_universe_planner_mode: "
            f"{preview.get('candidate_universe_planner_mode')}"
        )
    if preview.get("semantic_planner_fallback_reason"):
        lines.append(
            "semantic_planner_fallback_reason: "
            f"{preview.get('semantic_planner_fallback_reason')}"
        )
    if preview.get("semantic_planner_error_type"):
        error_parts = [
            str(part)
            for part in (
                preview.get("semantic_planner_error_type"),
                preview.get("semantic_planner_http_status"),
                preview.get("semantic_planner_parse_status"),
            )
            if part not in (None, "")
        ]
        lines.append(
            "semantic_planner_error: "
            + " ".join(error_parts)
        )
    if preview.get("selected_product_group_reason"):
        lines.append(
            "selected_product_group_reason: "
            f"{preview.get('selected_product_group_reason')}"
        )
    if preview.get("competing_product_groups"):
        lines.append(
            "competing_product_groups: "
            + json.dumps(preview.get("competing_product_groups"), ensure_ascii=False)
        )
    if preview.get("primary_object_indicators"):
        lines.append(
            "primary_object_indicators: "
            + json.dumps(preview.get("primary_object_indicators"), ensure_ascii=False)
        )
    if preview.get("component_role_indicators"):
        lines.append(
            "component_role_indicators: "
            + json.dumps(preview.get("component_role_indicators"), ensure_ascii=False)
        )
    if preview.get("planner_repair_attempted"):
        lines.append(
            "candidate_universe_planner_repair: "
            + json.dumps(
                {
                    "success": preview.get("planner_repair_success"),
                    "suspicion_reasons": preview.get("planner_suspicion_reasons"),
                },
                ensure_ascii=False,
            )
        )
    if preview.get("deterministic_product_group_hint"):
        lines.append(
            "deterministic_product_group_hint: "
            f"{preview.get('deterministic_product_group_hint')}"
        )
    if preview.get("semantic_planner_disagreement"):
        lines.append("semantic_planner_disagreement: true")
    if preview.get("matrix_blueprint_roles"):
        lines.append(
            "matrix_blueprint_roles: "
            + json.dumps(preview.get("matrix_blueprint_roles"), ensure_ascii=False)
        )
    if preview.get("embedded_requirements"):
        lines.append(
            "embedded_requirements: "
            + json.dumps(preview.get("embedded_requirements"), ensure_ascii=False)
        )
    if preview.get("not_primary_product_groups"):
        lines.append(
            "not_primary_product_groups: "
            + json.dumps(preview.get("not_primary_product_groups"), ensure_ascii=False)
        )
    lines.append(
        f"required_capabilities: {_capability_summary(preview.get('required_capabilities'))}"
    )
    lines.append(
        f"optional_capabilities: {_capability_summary(preview.get('optional_capabilities'))}"
    )
    lines.append(
        "category_plan accepted: "
        f"{json.dumps(preview.get('category_plan'), ensure_ascii=False)}"
    )
    if preview.get("category_planner_source"):
        lines.append(f"category_planner_source: {preview.get('category_planner_source')}")
    if preview.get("category_planner_missing_required_roles"):
        lines.append(
            "category_planner_missing_required_roles: "
            + json.dumps(
                preview.get("category_planner_missing_required_roles"),
                ensure_ascii=False,
            )
        )
    if preview.get("category_planner_repair_attempted"):
        lines.append(
            "category_planner_repair: "
            + json.dumps(
                {
                    "success": preview.get("category_planner_repair_success"),
                    "reason": preview.get("category_planner_repair_reason"),
                    "repaired_roles": preview.get("category_planner_repaired_roles"),
                    "unresolved_required_roles": preview.get(
                        "category_planner_unresolved_required_roles"
                    ),
                },
                ensure_ascii=False,
            )
        )
    warnings = preview.get("category_plan_warnings")
    if warnings:
        lines.append("category_plan warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    lines.append(f"count_by_role: {json.dumps(preview.get('count_by_role'), ensure_ascii=False)}")
    if preview.get("full_candidate_matrix_count_by_role"):
        lines.append(
            "full_candidate_matrix_count_by_role: "
            + json.dumps(
                preview.get("full_candidate_matrix_count_by_role"),
                ensure_ascii=False,
            )
        )
    lines.append(
        "composer_package_candidate_count_by_role: "
        + json.dumps(
            preview.get("composer_package_candidate_count_by_role"),
            ensure_ascii=False,
        )
    )
    lines.append(
        f"composer_package_candidate_total: {preview.get('composer_package_candidate_total')}"
    )
    if preview.get("package_candidate_exposure_incomplete"):
        lines.append(
            "package_candidate_exposure_incomplete: "
            + json.dumps(
                preview.get("package_candidate_exposure_incomplete_roles"),
                ensure_ascii=False,
            )
        )
    if preview.get("package_exposure_blocking_lifecycle_roles"):
        lines.append(
            "package_exposure_blocking_lifecycle_roles: "
            + json.dumps(
                preview.get("package_exposure_blocking_lifecycle_roles"),
                ensure_ascii=False,
            )
        )
    if preview.get("composer_package_full_matrix_used"):
        lines.append("composer_package_full_matrix_used: true")
    if preview.get("composer_context_size"):
        lines.append(
            "composer_context_size: "
            + json.dumps(preview.get("composer_context_size"), ensure_ascii=False)
        )
    if preview.get("v2_package_mode"):
        lines.append(f"v2_package_mode: {preview.get('v2_package_mode')}")
    if preview.get("expected_composer_mode"):
        lines.append(
            "expected_composer_mode: "
            f"{preview.get('expected_composer_mode')} "
            f"role_evaluation_would_run={preview.get('role_evaluation_would_run')}"
        )
    if preview.get("max_llm_calls_per_match") is not None:
        lines.append(
            "llm_call_budget: "
            f"{preview.get('llm_call_count')}/"
            f"{preview.get('max_llm_calls_per_match')}"
        )
    if preview.get("selected_package_mode"):
        lines.append(
            f"selected_package_mode: {preview.get('selected_package_mode')}"
        )
    if preview.get("verbose_context_size"):
        lines.append(
            "verbose_context_size: "
            + json.dumps(preview.get("verbose_context_size"), ensure_ascii=False)
        )
    if preview.get("compact_context_size"):
        lines.append(
            "compact_context_size: "
            + json.dumps(preview.get("compact_context_size"), ensure_ascii=False)
        )
    if preview.get("compact_candidate_count_by_role"):
        lines.append(
            "compact_candidate_count_by_role: "
            + json.dumps(
                preview.get("compact_candidate_count_by_role"),
                ensure_ascii=False,
            )
        )
    if preview.get("compact_candidate_total") is not None:
        lines.append(f"compact_candidate_total: {preview.get('compact_candidate_total')}")
    lines.append(
        "package_candidate_loss: "
        f"{preview.get('package_candidate_loss')}"
    )
    if preview.get("removed_verbose_fields"):
        lines.append(
            "removed_verbose_fields: "
            + json.dumps(preview.get("removed_verbose_fields"), ensure_ascii=False)
        )
    if preview.get("chars_by_section"):
        lines.append(
            "chars_by_section: "
            + json.dumps(preview.get("chars_by_section"), ensure_ascii=False)
        )
    if preview.get("avg_chars_per_candidate_by_role"):
        lines.append(
            "avg_chars_per_candidate_by_role: "
            + json.dumps(
                preview.get("avg_chars_per_candidate_by_role"),
                ensure_ascii=False,
            )
        )
    if preview.get("composer_attempt_decision"):
        lines.append(
            "composer_attempt_decision: "
            + json.dumps(
                preview.get("composer_attempt_decision"),
                ensure_ascii=False,
            )
        )
    if preview.get("dropped_before_composer_reason_by_role"):
        lines.append(
            "dropped_before_composer_reason_by_role: "
            + json.dumps(
                preview.get("dropped_before_composer_reason_by_role"),
                ensure_ascii=False,
            )
        )
    lines.append(
        "matrix_distiller: "
        f"{preview.get('matrix_distiller_source')} "
        f"used={preview.get('matrix_distiller_used')}"
    )
    if preview.get("full_matrix_evaluation_fallback_reason"):
        lines.append(
            "full_matrix_evaluation_fallback_reason: "
            f"{preview.get('full_matrix_evaluation_fallback_reason')}"
        )
    if preview.get("distilled_count_by_role"):
        lines.append(
            "distilled_count_by_role: "
            + json.dumps(preview.get("distilled_count_by_role"), ensure_ascii=False)
        )
    if preview.get("pre_composer_requirement_classifier_status"):
        lines.append(
            "pre_composer_requirement_classifier_status: "
            f"{preview.get('pre_composer_requirement_classifier_status')}"
        )
    if preview.get("pre_composer_requirement_source_coverage_percent") is not None:
        lines.append(
            "pre_composer_requirement_source_coverage_percent: "
            f"{preview.get('pre_composer_requirement_source_coverage_percent')}"
        )
    if preview.get("pre_composer_unclassified_source_fragments"):
        lines.append(
            "pre_composer_unclassified_source_fragments: "
            + json.dumps(
                preview.get("pre_composer_unclassified_source_fragments"),
                ensure_ascii=False,
            )
        )
    lines.append(
        "pre_composer_semantic_diagnostics_are_blocking: "
        f"{preview.get('pre_composer_semantic_diagnostics_are_blocking')}"
    )
    if preview.get("missing_required_roles_before_llm"):
        lines.append(
            "missing_required_roles_before_llm: "
            + json.dumps(
                preview.get("missing_required_roles_before_llm"),
                ensure_ascii=False,
            )
        )
    if preview.get("missing_required_capabilities_before_llm"):
        lines.append(
            "missing_required_capabilities_before_llm: "
            + json.dumps(
                preview.get("missing_required_capabilities_before_llm"),
                ensure_ascii=False,
            )
        )
    candidates = preview.get("first_candidates_by_role")
    if isinstance(candidates, dict) and candidates:
        lines.append("first candidates by active role:")
        for role, rows in candidates.items():
            lines.append(f"- {role}:")
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "  "
                    + " | ".join(
                        str(part)
                        for part in (
                            row.get("component_candidate_id"),
                            row.get("producer"),
                            row.get("part_number"),
                            row.get("fit_tier"),
                            row.get("name"),
                            row.get("price_value"),
                            row.get("available_quantity"),
                        )
                        if part not in (None, "")
                    )
                )
    lines.append(
        "package size: "
        + json.dumps(preview.get("package_approximate_size"), ensure_ascii=False)
    )
    budget = preview.get("package_budget")
    if budget:
        lines.append("package budget: " + json.dumps(budget, ensure_ascii=False))
    strategy = preview.get("package_strategy_decision")
    if strategy:
        lines.append("package strategy: " + json.dumps(strategy, ensure_ascii=False))
    if preview.get("llm_fallback_reason"):
        lines.append(f"llm_fallback_reason: {preview.get('llm_fallback_reason')}")
    if preview.get("package_skipped_reason"):
        lines.append(f"package_skipped_reason: {preview.get('package_skipped_reason')}")
    if preview.get("ready_candidates_excluded_reason"):
        lines.append(
            "ready_candidates_excluded_reason: "
            f"{preview.get('ready_candidates_excluded_reason')}"
        )
    if preview.get("ready_candidates_limit"):
        lines.append(f"ready_candidates_limit: {preview.get('ready_candidates_limit')}")
    if preview.get("package_budget_warnings"):
        lines.append(
            "package budget warnings: "
            + json.dumps(preview.get("package_budget_warnings"), ensure_ascii=False)
        )
    return "\n".join(lines)


def _human_semantic_only_preview(preview: dict[str, object]) -> str:
    lines = ["semantic_only: true"]
    lines.append(f"product_group: {preview.get('product_group')}")
    if preview.get("primary_object"):
        lines.append(f"primary_object: {preview.get('primary_object')}")
    lines.append(
        "semantic_planner: "
        f"{preview.get('semantic_planner_source')} "
        f"({preview.get('semantic_planner_confidence')})"
    )
    if preview.get("semantic_planner_provider") or preview.get("semantic_planner_model"):
        lines.append(
            "semantic_planner_provider: "
            f"{preview.get('semantic_planner_provider')} "
            f"{preview.get('semantic_planner_model')}"
        )
    if preview.get("semantic_planner_fallback_reason"):
        lines.append(
            "semantic_planner_fallback_reason: "
            f"{preview.get('semantic_planner_fallback_reason')}"
        )
    if preview.get("semantic_planner_error_type"):
        lines.append(
            "semantic_planner_error: "
            + " ".join(
                str(part)
                for part in (
                    preview.get("semantic_planner_error_type"),
                    preview.get("semantic_planner_http_status"),
                    preview.get("semantic_planner_parse_status"),
                )
                if part not in (None, "")
            )
        )
    if preview.get("requirement_source_coverage_percent") is not None:
        lines.append(
            "requirement_source_coverage_percent: "
            f"{preview.get('requirement_source_coverage_percent')}"
        )
    if preview.get("requirement_classifier_incomplete_reason"):
        lines.append(
            "requirement_classifier_incomplete_reason: "
            f"{preview.get('requirement_classifier_incomplete_reason')}"
        )
    if preview.get("matrix_blueprint_roles"):
        lines.append(
            "matrix_blueprint_roles: "
            + json.dumps(preview.get("matrix_blueprint_roles"), ensure_ascii=False)
        )
    lines.append(
        f"required_capabilities: {_capability_summary(preview.get('required_capabilities'))}"
    )
    if preview.get("planner_warnings"):
        lines.append(
            "planner_warnings: "
            + json.dumps(preview.get("planner_warnings"), ensure_ascii=False)
        )
    return "\n".join(lines)


def _capability_summary(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "[]"
    rows: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        capability_id = item.get("capability_id")
        parsed = item.get("parsed_requirements")
        rows.append(
            f"{role}:{capability_id}"
            + (f" {json.dumps(parsed, ensure_ascii=False)}" if parsed else "")
        )
    return "; ".join(rows) if rows else "[]"


def _filter_diagnostics_for_active_roles(package: dict[str, object]) -> dict[str, object]:
    coverage = package.get("role_coverage_summary")
    if not isinstance(coverage, dict):
        return {}
    result: dict[str, object] = {}
    for role, row in coverage.items():
        if isinstance(row, dict) and (
            row.get("raw_products_count") or row.get("after_eligibility_count")
        ):
            result[str(role)] = {
                "raw_products_count": row.get("raw_products_count"),
                "after_category_count": row.get("after_category_count"),
                "after_eligibility_count": row.get("after_eligibility_count"),
                "filtered_reasons_top": row.get("filtered_reasons_top"),
            }
    return result


def _evidence_preview_summary(
    package: dict[str, object],
    *,
    no_network: bool,
) -> dict[str, object]:
    settings = get_web_evidence_settings()
    matrix = package.get("component_candidate_matrix")
    matrix_by_role = matrix if isinstance(matrix, dict) else {}
    tasks = build_evidence_tasks_from_component_matrix(
        matrix_by_role,
        max_queries=settings.web_evidence_max_queries,
    )
    if not settings.web_evidence_enabled:
        return {
            "status": "web evidence disabled",
            "provider": settings.web_evidence_provider,
            "mode": settings.web_evidence_mode,
            "model": settings.web_evidence_model
            if settings.web_evidence_provider.strip().lower() == "routerai"
            else None,
            "planned_tasks_count": len(tasks),
            "tasks": [task.model_dump() for task in tasks],
            "coverage": {"total_tasks": len(tasks), "completed_tasks": 0},
        }
    if no_network:
        if settings.web_evidence_provider.strip().lower() == "routerai":
            return {
                "status": "web evidence planned; no network request sent",
                "provider": "routerai",
                "mode": settings.web_evidence_mode,
                "model": settings.web_evidence_model,
                "max_queries": settings.web_evidence_max_queries,
                "planned_tasks_count": len(tasks),
                "tasks": [task.model_dump() for task in tasks],
                "coverage": {"total_tasks": len(tasks), "completed_tasks": 0},
            }
        preview_settings = settings.model_copy(
            update={"web_evidence_enabled": True, "web_evidence_provider": "disabled"}
        )
        pack = collect_web_evidence(
            tasks=tasks,
            settings=preview_settings,
            provider=DisabledWebSearchProvider(),
            llm_settings=LlmSettings(),
        )
    else:
        pack = collect_web_evidence(
            tasks=tasks,
            settings=settings,
            llm_settings=LlmSettings(),
        )
    return {
        "status": pack.evidence_summary,
        "provider": pack.provider,
        "mode": settings.web_evidence_mode,
        "model": settings.web_evidence_model if pack.provider == "routerai" else None,
        "planned_tasks_count": len(tasks),
        "tasks": [task.model_dump() for task in tasks],
        "coverage": {
            "total_tasks": pack.total_tasks,
            "completed_tasks": pack.completed_tasks,
            "error_count": pack.error_count,
        },
    }


def _first_candidates(rows: list[object], *, limit: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows[: max(0, limit)]:
        if not isinstance(row, dict):
            continue
        result.append(
            {
                "component_candidate_id": row.get("component_candidate_id"),
                "category_id": row.get("category_id"),
                "producer": row.get("producer"),
                "part_number": row.get("part_number"),
                "name": row.get("name"),
                "price_value": row.get("price_value"),
                "price_currency": row.get("price_currency"),
                "available_quantity": row.get("available_quantity"),
                "fit_tier": row.get("fit_tier"),
                "score": row.get("score"),
                "selection_bucket": row.get("selection_bucket"),
            }
        )
    return result


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
