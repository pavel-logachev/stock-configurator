from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.user_facing_text import sanitize_user_facing_text

COMPOSER_STRUCTURED_NO_RECOMMENDATION = "composer_structured_no_recommendation"
COMPOSER_NO_SAFE_COMPLETE_BOM = "composer_no_safe_complete_bom"
SAFE_NO_RECOMMENDATION_SUMMARY_RU = (
    "Безопасную складскую рекомендацию дать нельзя."
)

_SUCCESS_STATUSES = {
    "available",
    "closed",
    "covered",
    "met",
    "ok",
    "pass",
    "passed",
    "satisfied",
    "success",
    "verified",
}


def normalize_composer_result(
    *,
    product_group: str | None = None,
    primary_object: str | None = None,
    original_request_text: str | None = None,
    requirement_contract: Mapping[str, Any] | None = None,
    role_evaluation_summaries: Mapping[str, Any] | None = None,
    role_evaluation_coverage_by_role: Mapping[str, Any] | None = None,
    bom_composer_output: Mapping[str, Any] | None = None,
    completeness_critic_result: Mapping[str, Any] | None = None,
    repair_composer_output: Mapping[str, Any] | None = None,
    code_validation_result: Mapping[str, Any] | None = None,
    final_status_source: str | None = None,
    primary_recommendation_status: str | None = None,
    llm_fallback_reason: str | None = None,
    existing_no_recommendation_reason: Mapping[str, Any] | None = None,
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Normalize Composer/critic/validator output into one user-facing shape."""

    existing_reason = _safe_mapping(existing_no_recommendation_reason)
    bom_output = _safe_mapping(bom_composer_output)
    repair_output = _safe_mapping(repair_composer_output)
    no_recommendation = _first_mapping(
        _safe_mapping(repair_output.get("no_recommendation")),
        _safe_mapping(bom_output.get("no_recommendation")),
        existing_reason if existing_reason.get("structured_no_recommendation") else {},
    )
    source_output = repair_output if _safe_mapping(
        repair_output.get("no_recommendation")
    ) else bom_output
    contract = _safe_mapping(requirement_contract)
    critic = _safe_mapping(completeness_critic_result)
    validation = _safe_mapping(code_validation_result)
    role_evaluation_coverage = _safe_mapping(role_evaluation_coverage_by_role)
    source_requirement_analysis = _safe_mapping(
        source_output.get("requirement_analysis")
    )
    coverage_diagnostics = _coverage_diagnostics(role_evaluation_coverage)

    coverage_rejected = bool(
        existing_reason.get("coverage_rejected")
        or existing_reason.get("no_recommendation_coverage_rejected")
    )
    structured_no_recommendation = (
        _is_structured_no_recommendation(no_recommendation)
        and not coverage_rejected
    )
    validation_hard_mismatches = _mapping_rows(
        validation.get("validation_hard_mismatches")
        or validation.get("hard_mismatches")
        or validation.get("hard_mismatch_risks")
    )
    validation_unverified = _mapping_rows(
        validation.get("validation_unverified_requirements")
        or validation.get("unverified_requirements")
    )
    validation_rejected = _validation_rejected(validation, final_status_source)
    input_status = _clean_scalar(primary_recommendation_status)
    valid_without_no_recommendation = (
        input_status == "valid"
        and not no_recommendation
        and not existing_reason
        and not validation_rejected
    )
    input_fallback_reason = llm_fallback_reason or _text_or_none(
        existing_reason.get("fallback_reason")
    )
    reason_fallback = _normalized_reason_fallback(
        structured_no_recommendation=structured_no_recommendation,
        validation_rejected=validation_rejected,
        fallback_reason=input_fallback_reason,
    )
    output_fallback = (
        _clean_scalar(llm_fallback_reason)
        if valid_without_no_recommendation
        else _normalized_output_fallback(
            structured_no_recommendation=structured_no_recommendation,
            reason_fallback=reason_fallback,
            fallback_reason=input_fallback_reason,
        )
    )

    hard_requirements_met = _unique_items(
        [
            *_generic_items(existing_reason.get("hard_requirements_met")),
            *_generic_items(no_recommendation.get("hard_requirements_met")),
            *_generic_items(source_requirement_analysis.get("hard_requirements_met")),
        ]
    )
    hard_requirements_failed = _unique_items(
        [
            *_generic_items(existing_reason.get("hard_requirements_failed")),
            *_generic_items(no_recommendation.get("hard_requirements_failed")),
            *validation_hard_mismatches,
        ]
    )
    failed_requirements = _unique_items(
        [
            *_generic_items(existing_reason.get("failed_requirements")),
            *_generic_items(no_recommendation.get("failed_requirements")),
            *_generic_items(no_recommendation.get("hard_requirements_failed")),
            *_mapping_rows(existing_reason.get("missing_required_capabilities")),
            *_mapping_rows(no_recommendation.get("missing_required_capabilities")),
            *_critic_failed_requirements(critic),
            *validation_hard_mismatches,
        ]
    )
    role_failures = _role_failures(
        no_recommendation=no_recommendation,
        existing_reason=existing_reason,
        critic=critic,
        role_evaluation_coverage=role_evaluation_coverage,
        considered_candidate_count_by_role=_safe_mapping(
            source_output.get("considered_candidate_count_by_role")
            or no_recommendation.get("considered_candidate_count_by_role")
            or existing_reason.get("considered_candidate_count_by_role")
        ),
    )
    partial_available_components = _partial_available_components(
        no_recommendation=no_recommendation,
        existing_reason=existing_reason,
        hard_requirements_met=hard_requirements_met,
        general_notes=[
            *_string_list(no_recommendation.get("general_notes")),
            *_string_list(source_output.get("general_notes")),
        ],
    )
    unverified_requirements = _unique_items(
        [
            *_generic_items(existing_reason.get("unverified_requirements")),
            *_generic_items(no_recommendation.get("unverified_requirements")),
            *_mapping_rows(source_output.get("unverified_requirements")),
            *_generic_items(critic.get("unverified_requirements")),
            *validation_unverified,
        ]
    )
    hard_mismatch_risks = _unique_items(
        [
            *_generic_items(existing_reason.get("hard_mismatch_risks")),
            *_mapping_rows(existing_reason.get("hard_mismatches")),
            *_generic_items(no_recommendation.get("hard_mismatch_risks")),
            *_mapping_rows(no_recommendation.get("hard_mismatches")),
            *_mapping_rows(source_output.get("hard_mismatch_risks")),
            *_generic_items(critic.get("hard_mismatch_risks")),
            *validation_hard_mismatches,
        ]
    )
    recommended_next_actions = _unique_text(
        [
            *_string_list(existing_reason.get("recommended_next_actions")),
            *_string_list(no_recommendation.get("recommended_next_actions")),
            *_string_list(no_recommendation.get("recommended_repair_actions")),
            *_string_list(critic.get("recommended_repair_actions")),
        ]
    )
    engineer_checks = _unique_text(
        [
            *_string_list(existing_reason.get("engineer_checks")),
            *_string_list(existing_reason.get("engineering_checks")),
            *_string_list(existing_reason.get("manual_checks")),
            *_string_list(no_recommendation.get("engineer_checks")),
            *_string_list(no_recommendation.get("engineering_checks")),
            *_string_list(source_output.get("engineer_checks")),
            *_string_list(contract.get("engineer_checks")),
        ]
    )
    diagnostic_notes = _unique_text(
        [
            *_string_list(existing_reason.get("diagnostic_notes")),
            *_string_list(no_recommendation.get("general_notes")),
            *_string_list(source_output.get("general_notes")),
            *[str(row.get("warning") or "") for row in coverage_diagnostics],
            *[_safe_text(item, limit=180) for item in warnings],
        ]
    )

    composer_summary_ru = _summary_text(
        no_recommendation,
        existing_reason=existing_reason,
        validation_rejected=validation_rejected,
    )
    customer_safe_summary_ru = _customer_summary_text(
        no_recommendation,
        existing_reason=existing_reason,
        composer_summary_ru=composer_summary_ru,
    )
    if valid_without_no_recommendation:
        return {
            "primary_recommendation_status": input_status,
            "final_status_source": _clean_scalar(final_status_source),
            "llm_fallback_reason": output_fallback,
            "no_recommendation_reason": {},
            "partial_available_components": partial_available_components,
            "failed_requirements": failed_requirements,
            "role_failures": role_failures,
            "unverified_requirements": unverified_requirements,
            "hard_mismatch_risks": hard_mismatch_risks,
            "recommended_next_actions": recommended_next_actions,
            "engineer_checks": engineer_checks,
            "composer_summary_ru": composer_summary_ru,
            "customer_safe_summary_ru": customer_safe_summary_ru,
        }
    role_names = _unique_text(
        [
            *[
                str(row.get("role") or "").strip()
                for row in role_failures
                if str(row.get("role") or "").strip()
            ],
            *_missing_role_names(existing_reason.get("missing_roles")),
            *_missing_role_names(no_recommendation.get("missing_roles")),
        ]
    )
    hard_incompatibility = _unique_text(
        [
            *_string_list(existing_reason.get("hard_incompatibility")),
            *[_item_summary_text(item) for item in hard_mismatch_risks],
        ]
    )
    missing_required_capabilities = _unique_mapping_rows(
        [
            *_mapping_rows(existing_reason.get("missing_required_capabilities")),
            *_mapping_rows(no_recommendation.get("missing_required_capabilities")),
            *[
                item
                for item in failed_requirements
                if isinstance(item, Mapping)
                and (
                    item.get("capability_id")
                    or item.get("requirement_text")
                    or item.get("source_text")
                )
            ],
        ]
    )

    no_recommendation_reason = {
        **existing_reason,
        "summary": customer_safe_summary_ru or SAFE_NO_RECOMMENDATION_SUMMARY_RU,
        "fallback_reason": reason_fallback,
        "product_group": _clean_scalar(product_group)
        or _clean_scalar(existing_reason.get("product_group")),
        "primary_object": _clean_scalar(primary_object)
        or _clean_scalar(existing_reason.get("primary_object")),
        "failed_requirements": failed_requirements,
        "role_failures": role_failures,
        "partial_available_components": partial_available_components,
        "hard_requirements_met": hard_requirements_met,
        "hard_requirements_failed": hard_requirements_failed,
        "unverified_requirements": unverified_requirements,
        "hard_mismatch_risks": hard_mismatch_risks,
        "recommended_next_actions": recommended_next_actions,
        "engineer_checks": engineer_checks,
        "engineering_checks": engineer_checks,
        "diagnostic_notes": diagnostic_notes,
        "composer_summary_ru": composer_summary_ru,
        "customer_safe_summary_ru": customer_safe_summary_ru,
        "structured_no_recommendation": structured_no_recommendation,
        "missing_roles": role_names,
        "missing_required_capabilities": missing_required_capabilities,
        "hard_incompatibility": hard_incompatibility,
    }
    _copy_optional_mapping(
        no_recommendation_reason,
        "requirement_coverage_summary",
        _first_mapping(
            _safe_mapping(no_recommendation.get("requirement_coverage_summary")),
            _safe_mapping(source_output.get("requirement_coverage_summary")),
            _safe_mapping(existing_reason.get("requirement_coverage_summary")),
        ),
    )
    _copy_optional_mapping(
        no_recommendation_reason,
        "considered_candidate_count_by_role",
        _first_mapping(
            _safe_mapping(no_recommendation.get("considered_candidate_count_by_role")),
            _safe_mapping(source_output.get("considered_candidate_count_by_role")),
            _safe_mapping(existing_reason.get("considered_candidate_count_by_role")),
        ),
    )
    _copy_optional_mapping(
        no_recommendation_reason,
        "role_evaluation_coverage_by_role",
        role_evaluation_coverage,
    )
    _copy_optional_mapping(
        no_recommendation_reason,
        "role_evaluation_summaries",
        _safe_mapping(role_evaluation_summaries),
    )
    _copy_optional_mapping(
        no_recommendation_reason,
        "completeness_critic_result",
        critic,
    )
    if coverage_diagnostics:
        no_recommendation_reason["coverage_diagnostics"] = coverage_diagnostics
    if original_request_text:
        no_recommendation_reason["original_request_text"] = str(
            original_request_text
        )

    status = input_status or (
        "no_recommendation" if no_recommendation_reason else "not_available"
    )
    return {
        "primary_recommendation_status": status,
        "final_status_source": _clean_scalar(final_status_source),
        "llm_fallback_reason": output_fallback,
        "no_recommendation_reason": no_recommendation_reason,
        "partial_available_components": partial_available_components,
        "failed_requirements": failed_requirements,
        "role_failures": role_failures,
        "unverified_requirements": unverified_requirements,
        "hard_mismatch_risks": hard_mismatch_risks,
        "recommended_next_actions": recommended_next_actions,
        "engineer_checks": engineer_checks,
        "composer_summary_ru": composer_summary_ru,
        "customer_safe_summary_ru": customer_safe_summary_ru,
    }


def normalize_composer_report_json(report_json: Mapping[str, Any]) -> dict[str, Any]:
    """Apply Composer no_recommendation normalization to a persisted report payload."""

    result = dict(report_json)
    final_bom = _safe_mapping(result.get("final_bom_after_repair"))
    existing_reason = _safe_mapping(result.get("no_recommendation_reason"))
    critic = _safe_mapping(result.get("completeness_critic_result"))
    validation_hard_mismatches = _mapping_rows(result.get("validation_hard_mismatches"))
    validation_unverified = _mapping_rows(
        result.get("validation_unverified_requirements")
    )
    primary_status = _clean_scalar(result.get("primary_recommendation_status"))
    final_source = _clean_scalar(result.get("final_status_source"))
    has_structured_sources = bool(
        _safe_mapping(final_bom.get("no_recommendation"))
        or existing_reason
        or critic
        or validation_hard_mismatches
        or validation_unverified
        or primary_status == "no_recommendation"
        or final_source in {"composer_no_recommendation", "composer_rejected_by_validation"}
    )
    if not has_structured_sources:
        return result

    normalized = normalize_composer_result(
        product_group=_clean_scalar(result.get("product_group")),
        primary_object=_clean_scalar(result.get("primary_object")),
        original_request_text=_clean_scalar(
            result.get("original_request_text") or result.get("source_text")
        ),
        requirement_contract=_safe_mapping(result.get("requirement_contract")),
        role_evaluation_summaries=_safe_mapping(
            result.get("role_evaluation_summaries")
        ),
        role_evaluation_coverage_by_role=_safe_mapping(
            result.get("role_evaluation_coverage_by_role")
        ),
        bom_composer_output=final_bom,
        completeness_critic_result=critic,
        repair_composer_output=final_bom,
        code_validation_result={
            "validation_hard_mismatches": validation_hard_mismatches,
            "validation_unverified_requirements": validation_unverified,
            "validation_summary": _safe_mapping(result.get("ai_validation_summary")),
            "rejected_recommendations": _mapping_rows(
                result.get("rejected_ai_recommendations_debug_safe")
            ),
        },
        final_status_source=final_source,
        primary_recommendation_status=primary_status,
        llm_fallback_reason=_clean_scalar(result.get("llm_fallback_reason")),
        existing_no_recommendation_reason=existing_reason,
        warnings=_string_list(result.get("llm_internal_warnings")),
    )
    for key in (
        "primary_recommendation_status",
        "final_status_source",
        "llm_fallback_reason",
        "no_recommendation_reason",
        "partial_available_components",
        "failed_requirements",
        "role_failures",
        "unverified_requirements",
        "hard_mismatch_risks",
        "recommended_next_actions",
        "engineer_checks",
        "composer_summary_ru",
        "customer_safe_summary_ru",
    ):
        if key in normalized:
            result[key] = normalized[key]
    return result


def _is_structured_no_recommendation(value: Mapping[str, Any]) -> bool:
    if not value:
        return False
    if _text_or_none(value.get("reason") or value.get("summary")):
        return True
    if any(
        _generic_items(value.get(key))
        for key in (
            "failed_requirements",
            "hard_requirements_failed",
            "role_failures",
            "role_level_reasons",
            "failures_by_role",
            "unverified_requirements",
            "hard_mismatch_risks",
            "recommended_next_actions",
            "recommended_repair_actions",
        )
    ):
        return True
    return bool(_string_list(value.get("missing_roles"))) or bool(
        _mapping_rows(value.get("role_analysis"))
    )


def _normalized_reason_fallback(
    *,
    structured_no_recommendation: bool,
    validation_rejected: bool,
    fallback_reason: str | None,
) -> str:
    if fallback_reason == COMPOSER_NO_SAFE_COMPLETE_BOM:
        return COMPOSER_NO_SAFE_COMPLETE_BOM
    if validation_rejected:
        return COMPOSER_NO_SAFE_COMPLETE_BOM
    if structured_no_recommendation:
        return COMPOSER_STRUCTURED_NO_RECOMMENDATION
    return fallback_reason or COMPOSER_NO_SAFE_COMPLETE_BOM


def _normalized_output_fallback(
    *,
    structured_no_recommendation: bool,
    reason_fallback: str,
    fallback_reason: str | None,
) -> str:
    if structured_no_recommendation:
        return reason_fallback
    return fallback_reason or reason_fallback


def _validation_rejected(
    validation: Mapping[str, Any],
    final_status_source: str | None,
) -> bool:
    final_source = str(final_status_source or "").strip()
    if "validation" in final_source:
        return True
    if _mapping_rows(validation.get("validation_hard_mismatches")):
        return True
    summary = _safe_mapping(validation.get("validation_summary"))
    for key in ("validation_rejected_count", "rejected", "rejected_fatal"):
        try:
            if int(summary.get(key) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return bool(_mapping_rows(validation.get("rejected_recommendations")))


def _role_failures(
    *,
    no_recommendation: Mapping[str, Any],
    existing_reason: Mapping[str, Any],
    critic: Mapping[str, Any],
    role_evaluation_coverage: Mapping[str, Any],
    considered_candidate_count_by_role: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in [
        *_mapping_rows(existing_reason.get("role_failures")),
        *_role_failure_alias_rows(no_recommendation),
    ]:
        rows.append(
            _normalize_role_failure(
                value,
                role_evaluation_coverage=role_evaluation_coverage,
                considered_candidate_count_by_role=considered_candidate_count_by_role,
            )
        )
    for value in _mapping_rows(no_recommendation.get("role_analysis")):
        status = str(value.get("status") or "").strip().casefold()
        if status and status in _SUCCESS_STATUSES:
            continue
        if not status and not (
            value.get("reason") or value.get("explanation") or value.get("role")
        ):
            continue
        rows.append(
            _normalize_role_failure(
                value,
                role_evaluation_coverage=role_evaluation_coverage,
                considered_candidate_count_by_role=considered_candidate_count_by_role,
            )
        )
    for role in [
        *_plain_role_names(existing_reason.get("missing_roles")),
        *_plain_role_names(no_recommendation.get("missing_roles")),
        *_string_list(critic.get("missing_roles")),
    ]:
        rows.append(
            _normalize_role_failure(
                {
                    "role": role,
                    "reason": "Required role is not safely covered by available candidates.",
                },
                role_evaluation_coverage=role_evaluation_coverage,
                considered_candidate_count_by_role=considered_candidate_count_by_role,
            )
        )
    return _unique_mapping_rows([row for row in rows if row.get("role")])


def _role_failure_alias_rows(no_recommendation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("role_failures", "role_level_reasons"):
        rows.extend(_mapping_rows(no_recommendation.get(key)))
    rows.extend(_role_failure_rows_from_mapping(no_recommendation.get("failures_by_role")))
    rows.extend(_role_failure_rows_from_mapping(no_recommendation.get("missing_roles")))
    return rows


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


def _missing_role_names(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return _unique_text([str(role or "").strip() for role in value])
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        names: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                names.append(
                    str(
                        item.get("role")
                        or item.get("component_role")
                        or item.get("name")
                        or ""
                    ).strip()
                )
            else:
                names.append(str(item or "").strip())
        return _unique_text(names)
    text = str(value or "").strip()
    return [text] if text else []


def _plain_role_names(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _unique_text(
            [str(item or "").strip() for item in value if not isinstance(item, Mapping)]
        )
    return []


def _normalize_role_failure(
    row: Mapping[str, Any],
    *,
    role_evaluation_coverage: Mapping[str, Any],
    considered_candidate_count_by_role: Mapping[str, Any],
) -> dict[str, Any]:
    role = str(row.get("role") or row.get("component_role") or "").strip()
    reason = _safe_text(
        row.get("reason")
        or row.get("explanation")
        or row.get("user_message")
        or row.get("status")
        or "No safe candidate covers this role.",
        limit=260,
    )
    candidate_coverage = _safe_mapping(row.get("candidate_coverage"))
    coverage = _safe_mapping(role_evaluation_coverage.get(role))
    if coverage:
        candidate_coverage = {**coverage, **candidate_coverage}
    considered_count = _safe_int(considered_candidate_count_by_role.get(role))
    if considered_count is not None and "considered_count" not in candidate_coverage:
        candidate_coverage["considered_count"] = considered_count
    considered_ids = _string_list(row.get("considered_candidate_ids"))
    if considered_ids:
        candidate_coverage["considered_candidate_ids"] = considered_ids
        candidate_coverage.setdefault("considered_count", len(considered_ids))
    result = {
        "role": role,
        "reason": reason,
        "candidate_coverage": candidate_coverage,
        "suggested_action": _safe_text(
            row.get("suggested_action")
            or row.get("recommended_action")
            or row.get("repair_action"),
            limit=240,
        ),
    }
    return {key: value for key, value in result.items() if value not in ("", {}, [])}


def _partial_available_components(
    *,
    no_recommendation: Mapping[str, Any],
    existing_reason: Mapping[str, Any],
    hard_requirements_met: Sequence[Any],
    general_notes: Sequence[str],
) -> list[dict[str, Any]]:
    rows = [
        *_mapping_rows(existing_reason.get("partial_available_components")),
        *_mapping_rows(no_recommendation.get("partial_available_components")),
    ]
    for item in hard_requirements_met:
        if isinstance(item, Mapping):
            row = dict(item)
            row.setdefault("source", "hard_requirements_met")
            rows.append(row)
        else:
            text = _item_summary_text(item)
            if text:
                rows.append(
                    {"source": "hard_requirements_met", "reason": text}
                )
    for note in general_notes:
        text = _safe_text(note, limit=240)
        if text:
            rows.append({"source": "general_notes", "reason": text})
    for value in _mapping_rows(no_recommendation.get("role_analysis")):
        status = str(value.get("status") or "").strip().casefold()
        if status not in _SUCCESS_STATUSES:
            continue
        rows.append(
            {
                "role": value.get("role"),
                "reason": value.get("reason")
                or value.get("explanation")
                or value.get("status"),
                "candidate_coverage": value.get("candidate_coverage") or {},
            }
        )
    return _unique_mapping_rows(rows)


def _coverage_diagnostics(
    role_evaluation_coverage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, raw_coverage in role_evaluation_coverage.items():
        coverage = _safe_mapping(raw_coverage)
        if not coverage:
            continue
        all_considered = coverage.get("all_candidates_considered")
        considered_count = _first_int(
            coverage,
            "considered_count",
            "considered_candidate_count",
            "candidates_considered",
            "evaluated_candidate_count",
        )
        candidate_count = _first_int(
            coverage,
            "candidate_count",
            "total_candidate_count",
            "total_candidates",
            "expected_candidate_count",
        )
        failed_chunk_count = _first_int(
            coverage,
            "failed_chunk_count",
            "failed_chunks",
        )
        incomplete = all_considered is False
        if (
            considered_count is not None
            and candidate_count is not None
            and considered_count < candidate_count
        ):
            incomplete = True
        if not incomplete:
            continue
        role_text = str(role or "").strip()
        warning = _coverage_warning_text(
            role=role_text,
            considered_count=considered_count,
            candidate_count=candidate_count,
            failed_chunk_count=failed_chunk_count,
        )
        rows.append(
            {
                "role": role_text,
                "all_candidates_considered": all_considered,
                "considered_count": considered_count,
                "candidate_count": candidate_count,
                "failed_chunk_count": failed_chunk_count,
                "warning": warning,
            }
        )
    return _unique_mapping_rows(rows)


def _coverage_warning_text(
    *,
    role: str,
    considered_count: int | None,
    candidate_count: int | None,
    failed_chunk_count: int | None,
) -> str:
    if considered_count is not None and candidate_count is not None:
        count_text = f"considered {considered_count}/{candidate_count} candidates"
    else:
        count_text = "did not confirm that every candidate was considered"
    failed_text = (
        f"; failed_chunk_count={failed_chunk_count}"
        if failed_chunk_count is not None
        else ""
    )
    return f"Coverage warning for role {role}: {count_text}{failed_text}."


def _critic_failed_requirements(critic: Mapping[str, Any]) -> list[Any]:
    rows: list[Any] = []
    rows.extend(
        {"role": role, "reason": "Completeness critic reported the role missing."}
        for role in _string_list(critic.get("missing_roles"))
    )
    rows.extend(_generic_items(critic.get("insufficient_quantities")))
    rows.extend(_generic_items(critic.get("hard_mismatch_risks")))
    return rows


def _summary_text(
    no_recommendation: Mapping[str, Any],
    *,
    existing_reason: Mapping[str, Any],
    validation_rejected: bool,
) -> str:
    if (
        existing_reason.get("coverage_rejected")
        or existing_reason.get("no_recommendation_coverage_rejected")
    ):
        existing_summary = _safe_text(existing_reason.get("summary"), limit=400)
        if existing_summary:
            return existing_summary
    direct = _safe_text(
        no_recommendation.get("summary")
        or no_recommendation.get("reason")
        or no_recommendation.get("explanation_ru")
        or existing_reason.get("composer_summary_ru")
        or existing_reason.get("summary"),
        limit=400,
    )
    if direct:
        return direct
    if validation_rejected:
        return (
            "Composer вернул BOM, но проверка кода отклонила его как небезопасный "
            "для подготовки КП."
        )
    return SAFE_NO_RECOMMENDATION_SUMMARY_RU


def _customer_summary_text(
    no_recommendation: Mapping[str, Any],
    *,
    existing_reason: Mapping[str, Any],
    composer_summary_ru: str,
) -> str:
    if (
        existing_reason.get("coverage_rejected")
        or existing_reason.get("no_recommendation_coverage_rejected")
    ):
        existing_summary = _safe_text(existing_reason.get("summary"), limit=500)
        if existing_summary:
            return existing_summary
    return _safe_text(
        no_recommendation.get("customer_safe_summary_ru")
        or no_recommendation.get("explanation_ru")
        or existing_reason.get("customer_safe_summary_ru")
        or composer_summary_ru
        or SAFE_NO_RECOMMENDATION_SUMMARY_RU,
        limit=500,
    )


def _item_summary_text(item: Any) -> str:
    if isinstance(item, Mapping):
        return _safe_text(
            item.get("message")
            or item.get("reason")
            or item.get("requirement_text")
            or item.get("source_text")
            or item.get("status")
            or item.get("type"),
            limit=240,
        )
    return _safe_text(item, limit=240)


def _safe_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                result.append(text)
        return result
    text = str(value or "").strip()
    return [text] if text else []


def _generic_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, str):
        text = _safe_text(value, limit=500)
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result: list[Any] = []
        for item in value:
            if isinstance(item, Mapping):
                result.append(dict(item))
            else:
                text = _safe_text(item, limit=500)
                if text:
                    result.append(text)
        return result
    text = _safe_text(value, limit=500)
    return [text] if text else []


def _safe_text(value: Any, *, limit: int) -> str:
    text = sanitize_user_facing_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _clean_scalar(value: Any) -> str:
    return str(value or "").strip()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _safe_int(mapping.get(key))
        if value is not None:
            return value
    return None


def _unique_text(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _unique_items(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(value) if isinstance(value, Mapping) else value)
    return result


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


def _first_mapping(*values: Mapping[str, Any]) -> dict[str, Any]:
    for value in values:
        if value:
            return dict(value)
    return {}


def _copy_optional_mapping(
    target: dict[str, Any],
    key: str,
    value: Mapping[str, Any],
) -> None:
    if value:
        target[key] = dict(value)
