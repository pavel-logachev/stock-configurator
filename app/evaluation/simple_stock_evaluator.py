from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from app.evaluation.simple_stock_contracts import (
    EVALUATOR_VERSION,
    REPORT_SCHEMA_VERSION,
    ArtifactFile,
    BlindReview,
    EvaluationCaseRun,
    EvaluationRun,
    GoldenCase,
    GoldenDataset,
    HumanAnnotation,
    RunBindings,
)

MAX_CONTROL_FILE_BYTES = 5_000_000
MAX_CASE_ARTIFACT_BYTES = 25_000_000


class EvaluationInputError(ValueError):
    def __init__(self, code: str, details: list[str] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.details = details or []


@dataclass
class CaseAssessment:
    case_id: str
    output_sha256: str
    matrix_sha256: str
    annotation_sha256: str | None
    structural_valid: bool = False
    selected_product_id_count: int = 0
    grounded_product_id_count: int = 0
    semantic_score: float | None = None
    unsupported_material_claim_count: int = 0
    critical_error_count: int = 0
    business_weighted_loss: float | None = None
    human_reviewed: bool = False
    latency_ms: int | None = None
    cost_usd: float | None = None
    retries: int = 0
    fallback_used: bool = False
    failures: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def to_report(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": _status(self.failures, self.blockers),
            "output_sha256": self.output_sha256,
            "matrix_sha256": self.matrix_sha256,
            "annotation_sha256": self.annotation_sha256,
            "structural_valid": self.structural_valid,
            "selected_product_id_count": self.selected_product_id_count,
            "grounded_product_id_count": self.grounded_product_id_count,
            "semantic_score": self.semantic_score,
            "unsupported_material_claim_count": self.unsupported_material_claim_count,
            "critical_error_count": self.critical_error_count,
            "business_weighted_loss": self.business_weighted_loss,
            "human_reviewed": self.human_reviewed,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "retries": self.retries,
            "fallback_used": self.fallback_used,
            "failures": sorted(set(self.failures)),
            "blockers": sorted(set(self.blockers)),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        if path.stat().st_size > MAX_CONTROL_FILE_BYTES:
            raise EvaluationInputError("input.file_too_large")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationInputError("input.file_not_found") from exc
    except (OSError, UnicodeError) as exc:
        raise EvaluationInputError("input.file_unreadable") from exc
    except (json.JSONDecodeError, RecursionError) as exc:
        line_number = getattr(exc, "lineno", 0)
        raise EvaluationInputError("input.invalid_json", [f"line:{line_number}"]) from exc
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        details = [
            f"{'.'.join(str(part) for part in error['loc'])}:{error['type']}"
            for error in exc.errors(include_input=False)
        ]
        raise EvaluationInputError("input.schema_invalid", details) from exc


def evaluate_simple_stock_runs(
    *,
    dataset_path: Path,
    baseline_run_path: Path,
    candidate_run_path: Path,
    production_baseline_path: Path,
    blind_review_path: Path | None = None,
) -> dict[str, Any]:
    dataset = load_model(dataset_path, GoldenDataset)
    baseline_run = load_model(baseline_run_path, EvaluationRun)
    candidate_run = load_model(candidate_run_path, EvaluationRun)
    production_baseline = _load_json_mapping(production_baseline_path)

    dataset_sha256 = sha256_file(dataset_path)
    baseline_run_sha256 = sha256_file(baseline_run_path)
    candidate_run_sha256 = sha256_file(candidate_run_path)
    production_baseline_sha256 = sha256_file(production_baseline_path)

    failures: list[str] = []
    blockers: list[str] = []
    _validate_dataset_gate(dataset, blockers)
    _validate_run_bindings(
        baseline_run.bindings,
        run_root=baseline_run_path.parent,
        dataset_sha256=dataset_sha256,
        production_baseline_sha256=production_baseline_sha256,
        failures=failures,
        prefix="baseline",
    )
    _validate_run_bindings(
        candidate_run.bindings,
        run_root=candidate_run_path.parent,
        dataset_sha256=dataset_sha256,
        production_baseline_sha256=production_baseline_sha256,
        failures=failures,
        prefix="candidate",
    )
    _validate_baseline_identity(
        baseline_run.bindings,
        production_baseline,
        failures=failures,
    )
    changed_axes = _changed_axes(baseline_run.bindings, candidate_run.bindings)
    if len(changed_axes) > 1:
        failures.append("candidate.multiple_change_axes")

    accepted_cases = [case for case in dataset.cases if case.status == "accepted"]
    expected_case_ids = {case.case_id for case in accepted_cases}
    _validate_case_coverage(
        expected_case_ids,
        baseline_run,
        failures=failures,
        prefix="baseline",
    )
    _validate_case_coverage(
        expected_case_ids,
        candidate_run,
        failures=failures,
        prefix="candidate",
    )

    baseline_by_id = {case.case_id: case for case in baseline_run.cases}
    candidate_by_id = {case.case_id: case for case in candidate_run.cases}
    baseline_assessments: list[CaseAssessment] = []
    candidate_assessments: list[CaseAssessment] = []
    for golden_case in accepted_cases:
        baseline_case = baseline_by_id.get(golden_case.case_id)
        candidate_case = candidate_by_id.get(golden_case.case_id)
        if baseline_case is not None:
            baseline_assessments.append(
                _assess_case(
                    golden_case,
                    baseline_case,
                    baseline_run.bindings,
                    run_root=baseline_run_path.parent,
                )
            )
        if candidate_case is not None:
            candidate_assessments.append(
                _assess_case(
                    golden_case,
                    candidate_case,
                    candidate_run.bindings,
                    run_root=candidate_run_path.parent,
                )
            )

    baseline_metrics = _aggregate_metrics(accepted_cases, baseline_assessments)
    candidate_metrics = _aggregate_metrics(accepted_cases, candidate_assessments)
    _apply_candidate_quality_gates(
        dataset,
        baseline_metrics,
        candidate_metrics,
        failures=failures,
        blockers=blockers,
    )
    for assessment in baseline_assessments:
        failures.extend(f"baseline.{code}" for code in assessment.failures)
        blockers.extend(f"baseline.{code}" for code in assessment.blockers)
    for assessment in candidate_assessments:
        failures.extend(f"candidate.{code}" for code in assessment.failures)
        blockers.extend(f"candidate.{code}" for code in assessment.blockers)

    review_summary = _validate_blind_review(
        blind_review_path=blind_review_path,
        dataset_sha256=dataset_sha256,
        baseline_run_sha256=baseline_run_sha256,
        candidate_run_sha256=candidate_run_sha256,
        expected_case_ids=expected_case_ids,
        failures=failures,
        blockers=blockers,
    )

    failures = sorted(set(failures))
    blockers = sorted(set(blockers))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "status": _status(failures, blockers),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "status": dataset.status,
            "sha256": dataset_sha256,
            "accepted_case_count": len(accepted_cases),
            "minimum_accepted_cases": dataset.minimum_accepted_cases,
        },
        "production_baseline": {
            "sha256": production_baseline_sha256,
            "production_commit": production_baseline.get("production_commit"),
        },
        "comparison": {
            "baseline_run_id": baseline_run.run_id,
            "baseline_run_sha256": baseline_run_sha256,
            "candidate_run_id": candidate_run.run_id,
            "candidate_run_sha256": candidate_run_sha256,
            "changed_axes": changed_axes,
            "baseline_bindings": _bindings_report(baseline_run.bindings),
            "candidate_bindings": _bindings_report(candidate_run.bindings),
        },
        "baseline": {
            "metrics": baseline_metrics,
            "cases": [assessment.to_report() for assessment in baseline_assessments],
        },
        "candidate": {
            "metrics": candidate_metrics,
            "cases": [assessment.to_report() for assessment in candidate_assessments],
        },
        "blind_review": review_summary,
        "failures": failures,
        "blockers": blockers,
        "privacy": {
            "raw_prompts_in_report": False,
            "raw_outputs_in_report": False,
            "product_ids_in_report": False,
        },
    }


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_CONTROL_FILE_BYTES:
            raise EvaluationInputError("input.file_too_large")
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationInputError("input.file_not_found") from exc
    except (OSError, UnicodeError) as exc:
        raise EvaluationInputError("input.file_unreadable") from exc
    except (json.JSONDecodeError, RecursionError) as exc:
        line_number = getattr(exc, "lineno", 0)
        raise EvaluationInputError("input.invalid_json", [f"line:{line_number}"]) from exc
    if not isinstance(value, dict):
        raise EvaluationInputError("input.mapping_required")
    return value


def _bindings_report(bindings: RunBindings) -> dict[str, Any]:
    return {
        "code_revision": bindings.code_revision,
        "pipeline_version": bindings.pipeline_version,
        "stages": bindings.stages.model_dump(),
        "model_id": bindings.model_id,
        "model_config_sha256": bindings.model_settings_artifact.sha256,
        "prompt_bundle_sha256": bindings.prompt_bundle.sha256,
        "evaluator_version": bindings.evaluator_version,
    }


def _validate_dataset_gate(dataset: GoldenDataset, blockers: list[str]) -> None:
    accepted_count = sum(case.status == "accepted" for case in dataset.cases)
    if dataset.status != "accepted":
        blockers.append("dataset.not_accepted")
    if accepted_count < dataset.minimum_accepted_cases:
        blockers.append("dataset.minimum_case_count_not_met")


def _validate_run_bindings(
    bindings: RunBindings,
    *,
    run_root: Path,
    dataset_sha256: str,
    production_baseline_sha256: str,
    failures: list[str],
    prefix: str,
) -> None:
    if bindings.dataset_sha256 != dataset_sha256:
        failures.append(f"{prefix}.dataset_hash_mismatch")
    if bindings.production_baseline_sha256 != production_baseline_sha256:
        failures.append(f"{prefix}.production_baseline_hash_mismatch")
    _validate_binding_artifact(
        bindings.model_settings_artifact,
        run_root=run_root,
        failures=failures,
        code=f"{prefix}.model_config",
    )
    _validate_binding_artifact(
        bindings.prompt_bundle,
        run_root=run_root,
        failures=failures,
        code=f"{prefix}.prompt_bundle",
    )


def _validate_binding_artifact(
    artifact: ArtifactFile,
    *,
    run_root: Path,
    failures: list[str],
    code: str,
) -> None:
    root = run_root.resolve()
    path = (root / Path(artifact.path)).resolve()
    if not path.is_relative_to(root):
        failures.append(f"{code}_path_escape")
    elif not path.is_file():
        failures.append(f"{code}_missing")
    elif path.stat().st_size > MAX_CONTROL_FILE_BYTES:
        failures.append(f"{code}_too_large")
    elif sha256_file(path) != artifact.sha256:
        failures.append(f"{code}_hash_mismatch")


def _validate_baseline_identity(
    bindings: RunBindings,
    production_baseline: dict[str, Any],
    *,
    failures: list[str],
) -> None:
    expected_stages = production_baseline.get("stages")
    expected_llm = production_baseline.get("llm")
    if bindings.pipeline_version != production_baseline.get("pipeline_version"):
        failures.append("baseline.pipeline_version_mismatch")
    if bindings.stages.model_dump() != expected_stages:
        failures.append("baseline.stage_versions_mismatch")
    if not isinstance(expected_llm, dict) or bindings.model_id != expected_llm.get("model"):
        failures.append("baseline.model_mismatch")


def _changed_axes(baseline: RunBindings, candidate: RunBindings) -> list[str]:
    axes: list[str] = []
    if baseline.stages.route_prompt_version != candidate.stages.route_prompt_version:
        axes.append("route")
    if baseline.stages.matrix_schema_version != candidate.stages.matrix_schema_version:
        axes.append("matrix")
    if (
        baseline.stages.composer_prompt_version != candidate.stages.composer_prompt_version
        or baseline.prompt_bundle.sha256 != candidate.prompt_bundle.sha256
    ):
        axes.append("composer")
    if baseline.stages.reconciler_version != candidate.stages.reconciler_version:
        axes.append("reconciler")
    if (
        baseline.model_id != candidate.model_id
        or baseline.model_settings_artifact.sha256 != candidate.model_settings_artifact.sha256
    ):
        axes.append("model")
    return axes


def _validate_case_coverage(
    expected_case_ids: set[str],
    run: EvaluationRun,
    *,
    failures: list[str],
    prefix: str,
) -> None:
    actual_case_ids = {case.case_id for case in run.cases}
    if actual_case_ids != expected_case_ids:
        failures.append(f"{prefix}.case_coverage_mismatch")


def _assess_case(
    golden_case: GoldenCase,
    case_run: EvaluationCaseRun,
    bindings: RunBindings,
    *,
    run_root: Path,
) -> CaseAssessment:
    assessment = CaseAssessment(
        case_id=golden_case.case_id,
        output_sha256=case_run.output.sha256,
        matrix_sha256=case_run.matrix.sha256,
        annotation_sha256=(case_run.annotation.sha256 if case_run.annotation else None),
        latency_ms=case_run.latency_ms,
        cost_usd=case_run.cost_usd,
        retries=case_run.retries,
        fallback_used=bool(case_run.fallback_path),
    )
    output_path = _resolve_artifact(run_root, case_run.output, assessment)
    matrix_path = _resolve_artifact(run_root, case_run.matrix, assessment)
    if output_path is None or matrix_path is None:
        return assessment
    if (
        golden_case.matrix_source is None
        or golden_case.matrix_source.sha256 != case_run.matrix.sha256
    ):
        assessment.failures.append("case.matrix_hash_not_golden")

    output = _read_artifact_mapping(output_path, assessment, "output")
    matrix = _read_artifact_mapping(matrix_path, assessment, "matrix")
    if output is None or matrix is None:
        return assessment

    _check_output_contract(output, golden_case, bindings, assessment)
    _check_product_id_grounding(output, matrix, assessment)
    _load_annotation(case_run, run_root, golden_case, assessment)
    assessment.structural_valid = not any(
        code.startswith(("output.", "case.")) for code in assessment.failures
    )
    return assessment


def _resolve_artifact(
    run_root: Path,
    artifact: ArtifactFile,
    assessment: CaseAssessment,
) -> Path | None:
    root = run_root.resolve()
    path = (root / Path(artifact.path)).resolve()
    if not path.is_relative_to(root):
        assessment.failures.append("artifact.path_escape")
        return None
    if not path.is_file():
        assessment.blockers.append("artifact.file_missing")
        return None
    if path.stat().st_size > MAX_CASE_ARTIFACT_BYTES:
        assessment.failures.append("artifact.file_too_large")
        return None
    if sha256_file(path) != artifact.sha256:
        assessment.failures.append("artifact.hash_mismatch")
        return None
    return path


def _read_artifact_mapping(
    path: Path,
    assessment: CaseAssessment,
    artifact_kind: str,
) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        assessment.failures.append(f"{artifact_kind}.invalid_json")
        return None
    if not isinstance(value, dict):
        assessment.failures.append(f"{artifact_kind}.mapping_required")
        return None
    return value


def _check_output_contract(
    output: dict[str, Any],
    golden_case: GoldenCase,
    bindings: RunBindings,
    assessment: CaseAssessment,
) -> None:
    expectations = golden_case.expectations
    result_state = str(output.get("v3_result_state") or output.get("result_state") or "")
    pipeline_version = str(output.get("pipeline_version") or "")
    final_status_source = str(output.get("final_status_source") or "")
    quote = output.get("validated_quote")
    quote = quote if isinstance(quote, dict) else {}
    lines = quote.get("lines")
    lines = lines if isinstance(lines, list) else []
    validation_errors = _validated_string_list(
        output.get("v3_validation_errors", output.get("validation_errors")),
        field_name="validation_errors",
        assessment=assessment,
    )
    validation_warnings = _validated_string_list(
        output.get("v3_validation_warnings", output.get("validation_warnings")),
        field_name="validation_warnings",
        assessment=assessment,
    )

    if result_state not in expectations.allowed_result_states:
        assessment.failures.append("output.result_state_not_allowed")
    if pipeline_version != expectations.required_pipeline_version:
        assessment.failures.append("output.pipeline_version_not_expected")
    if pipeline_version != bindings.pipeline_version:
        assessment.failures.append("output.pipeline_binding_mismatch")
    if (
        expectations.allowed_final_status_sources
        and final_status_source not in expectations.allowed_final_status_sources
    ):
        assessment.failures.append("output.final_status_source_not_allowed")
    if len(lines) < expectations.minimum_quote_lines:
        assessment.failures.append("output.quote_line_count_too_low")
    if len(validation_errors) > expectations.maximum_validation_errors:
        assessment.failures.append("output.validation_error_limit_exceeded")
    if len(validation_warnings) > expectations.maximum_validation_warnings:
        assessment.failures.append("output.validation_warning_limit_exceeded")

    engineering_review_required = output.get("engineering_review_required")
    if engineering_review_required is None:
        engineering_review_required = quote.get("engineering_review_required")
    if not isinstance(engineering_review_required, bool):
        assessment.failures.append("output.engineering_review_boundary_invalid")
    elif engineering_review_required is not expectations.require_engineering_review:
        assessment.failures.append("output.engineering_review_boundary_mismatch")

    actual_versions = _output_versions(output, quote)
    expected_versions = bindings.stages.model_dump()
    for key, expected_value in expected_versions.items():
        if actual_versions.get(key) != expected_value:
            assessment.failures.append(f"output.{key}_mismatch")


def _output_versions(output: dict[str, Any], quote: dict[str, Any]) -> dict[str, str]:
    diagnostics = output.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    route_decision = output.get("simple_route_decision")
    route_decision = route_decision if isinstance(route_decision, dict) else {}
    diagnostic_route_decision = diagnostics.get("simple_route_decision")
    diagnostic_route_decision = (
        diagnostic_route_decision if isinstance(diagnostic_route_decision, dict) else {}
    )
    quote_integrity = quote.get("quote_integrity")
    quote_integrity = quote_integrity if isinstance(quote_integrity, dict) else {}
    return {
        "route_prompt_version": str(
            diagnostics.get("route_prompt_version")
            or route_decision.get("prompt_version")
            or diagnostic_route_decision.get("prompt_version")
            or ""
        ),
        "matrix_schema_version": str(diagnostics.get("matrix_schema_version") or ""),
        "composer_prompt_version": str(diagnostics.get("composer_prompt_version") or ""),
        "reconciler_version": str(
            quote_integrity.get("version")
            or diagnostics.get("quote_integrity_reconciler_version")
            or ""
        ),
    }


def _check_product_id_grounding(
    output: dict[str, Any],
    matrix: dict[str, Any],
    assessment: CaseAssessment,
) -> None:
    quote = output.get("validated_quote")
    quote = quote if isinstance(quote, dict) else {}
    lines = quote.get("lines")
    lines = lines if isinstance(lines, list) else []
    selected_ids: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            assessment.failures.append("output.quote_line_not_mapping")
            continue
        candidate_id = str(line.get("component_candidate_id") or "").strip()
        if not candidate_id:
            assessment.failures.append("output.component_candidate_id_missing")
            continue
        selected_ids.append(candidate_id)
    matrix_ids = _collect_values(matrix, "component_candidate_id")
    assessment.selected_product_id_count = len(selected_ids)
    assessment.grounded_product_id_count = sum(value in matrix_ids for value in selected_ids)
    if selected_ids and assessment.grounded_product_id_count != len(selected_ids):
        assessment.failures.append("output.unknown_component_candidate_id")
    if selected_ids and not matrix_ids:
        assessment.blockers.append("matrix.no_component_candidate_ids")


def _load_annotation(
    case_run: EvaluationCaseRun,
    run_root: Path,
    golden_case: GoldenCase,
    assessment: CaseAssessment,
) -> None:
    if case_run.annotation is None:
        assessment.blockers.append("human.annotation_missing")
        return
    annotation_path = _resolve_artifact(run_root, case_run.annotation, assessment)
    if annotation_path is None:
        return
    try:
        annotation = load_model(annotation_path, HumanAnnotation)
    except EvaluationInputError:
        assessment.failures.append("human.annotation_invalid")
        return
    if annotation.case_id != golden_case.case_id:
        assessment.failures.append("human.annotation_case_mismatch")
    if annotation.output_sha256 != case_run.output.sha256:
        assessment.failures.append("human.annotation_output_hash_mismatch")
    criteria = {item.criterion_id: item.status for item in annotation.atomic_criteria}
    for criterion_id in golden_case.expectations.required_atomic_criteria:
        if criteria.get(criterion_id) != "pass":
            assessment.failures.append("human.required_atomic_criterion_not_passed")
    assessment.semantic_score = annotation.semantic_score
    assessment.unsupported_material_claim_count = annotation.unsupported_material_claim_count
    assessment.critical_error_count = len(annotation.critical_error_codes)
    assessment.business_weighted_loss = annotation.business_weighted_loss
    assessment.human_reviewed = not any(
        code.startswith("human.annotation") for code in assessment.failures
    )


def _collect_values(value: Any, key: str) -> set[str]:
    found: set[str] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for current_key, current_value in current.items():
                if current_key == key and isinstance(current_value, str):
                    cleaned = current_value.strip()
                    if cleaned:
                        found.add(cleaned)
                if isinstance(current_value, (dict, list)):
                    pending.append(current_value)
        elif isinstance(current, list):
            pending.extend(item for item in current if isinstance(item, (dict, list)))
    return found


def _aggregate_metrics(
    golden_cases: list[GoldenCase],
    assessments: list[CaseAssessment],
) -> dict[str, Any]:
    case_by_id = {case.case_id: case for case in golden_cases}
    case_count = len(assessments)
    selected_count = sum(item.selected_product_id_count for item in assessments)
    grounded_count = sum(item.grounded_product_id_count for item in assessments)
    annotated = [item for item in assessments if item.semantic_score is not None]
    weighted_denominator = sum(case_by_id[item.case_id].business_weight for item in annotated)
    semantic_score = _weighted_average(
        annotated,
        case_by_id,
        value=lambda item: item.semantic_score,
        denominator=weighted_denominator,
    )
    business_loss = _weighted_average(
        annotated,
        case_by_id,
        value=lambda item: item.business_weighted_loss,
        denominator=weighted_denominator,
    )
    latencies = [item.latency_ms for item in assessments if item.latency_ms is not None]
    costs = [item.cost_usd for item in assessments if item.cost_usd is not None]
    return {
        "case_count": case_count,
        "structured_valid_rate": _ratio(
            sum(item.structural_valid for item in assessments), case_count
        ),
        "grounded_product_id_rate": (
            _ratio(grounded_count, selected_count) if selected_count else float(bool(case_count))
        ),
        "semantic_score": semantic_score,
        "critical_error_count": sum(item.critical_error_count for item in assessments),
        "unsupported_material_claim_count": sum(
            item.unsupported_material_claim_count for item in assessments
        ),
        "business_weighted_loss": business_loss,
        "human_review_rate": _ratio(sum(item.human_reviewed for item in assessments), case_count),
        "latency_evidence_rate": _ratio(len(latencies), case_count),
        "mean_latency_ms": _mean(latencies),
        "cost_evidence_rate": _ratio(len(costs), case_count),
        "mean_cost_usd": _mean(costs),
        "fallback_rate": _ratio(sum(item.fallback_used for item in assessments), case_count),
        "retry_count": sum(item.retries for item in assessments),
    }


def _weighted_average(
    assessments: list[CaseAssessment],
    case_by_id: dict[str, GoldenCase],
    *,
    value: Any,
    denominator: float,
) -> float | None:
    if not assessments or denominator <= 0:
        return None
    total = 0.0
    for assessment in assessments:
        current = value(assessment)
        if current is None:
            return None
        total += float(current) * case_by_id[assessment.case_id].business_weight
    return round(total / denominator, 6)


def _apply_candidate_quality_gates(
    dataset: GoldenDataset,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    failures: list[str],
    blockers: list[str],
) -> None:
    thresholds = dataset.thresholds
    if candidate["structured_valid_rate"] < thresholds.structured_valid_rate:
        failures.append("candidate.structured_valid_rate_below_threshold")
    if candidate["grounded_product_id_rate"] < thresholds.grounded_product_id_rate:
        failures.append("candidate.grounded_product_id_rate_below_threshold")
    if candidate["critical_error_count"] > thresholds.critical_error_count:
        failures.append("candidate.critical_error_count_exceeded")
    if candidate["unsupported_material_claim_count"] > thresholds.unsupported_material_claim_count:
        failures.append("candidate.unsupported_material_claim_count_exceeded")
    if candidate["semantic_score"] is None:
        blockers.append("candidate.semantic_score_missing")
    elif candidate["semantic_score"] < thresholds.minimum_semantic_score:
        failures.append("candidate.semantic_score_below_threshold")
    if candidate["business_weighted_loss"] is None:
        blockers.append("candidate.business_weighted_loss_missing")
    elif candidate["business_weighted_loss"] > thresholds.maximum_business_weighted_loss:
        failures.append("candidate.business_weighted_loss_exceeded")
    if candidate["human_review_rate"] < 1:
        blockers.append("candidate.human_review_incomplete")
    _compare_regression_metrics(baseline, candidate, failures=failures, blockers=blockers)
    _check_operational_regression(
        baseline,
        candidate,
        metric="mean_latency_ms",
        evidence_metric="latency_evidence_rate",
        maximum_ratio=thresholds.maximum_latency_regression_ratio,
        failures=failures,
        blockers=blockers,
    )
    _check_operational_regression(
        baseline,
        candidate,
        metric="mean_cost_usd",
        evidence_metric="cost_evidence_rate",
        maximum_ratio=thresholds.maximum_cost_regression_ratio,
        failures=failures,
        blockers=blockers,
    )


def _compare_regression_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    failures: list[str],
    blockers: list[str],
) -> None:
    if baseline["human_review_rate"] < 1:
        blockers.append("baseline.human_review_incomplete")
    for metric in ("structured_valid_rate", "grounded_product_id_rate", "semantic_score"):
        baseline_value = baseline[metric]
        candidate_value = candidate[metric]
        if baseline_value is None or candidate_value is None:
            blockers.append(f"comparison.{metric}_missing")
        elif candidate_value < baseline_value:
            failures.append(f"candidate.{metric}_regressed")
    for metric in (
        "critical_error_count",
        "unsupported_material_claim_count",
        "business_weighted_loss",
    ):
        baseline_value = baseline[metric]
        candidate_value = candidate[metric]
        if baseline_value is None or candidate_value is None:
            blockers.append(f"comparison.{metric}_missing")
        elif candidate_value > baseline_value:
            failures.append(f"candidate.{metric}_regressed")


def _check_operational_regression(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    metric: str,
    evidence_metric: str,
    maximum_ratio: float,
    failures: list[str],
    blockers: list[str],
) -> None:
    if baseline[evidence_metric] < 1 or candidate[evidence_metric] < 1:
        blockers.append(f"comparison.{metric}_evidence_incomplete")
        return
    baseline_value = baseline[metric]
    candidate_value = candidate[metric]
    if baseline_value is None or candidate_value is None:
        blockers.append(f"comparison.{metric}_missing")
        return
    if baseline_value == 0:
        if candidate_value > 0:
            failures.append(f"candidate.{metric}_regressed")
        return
    if candidate_value / baseline_value > maximum_ratio:
        failures.append(f"candidate.{metric}_regressed")


def _validate_blind_review(
    *,
    blind_review_path: Path | None,
    dataset_sha256: str,
    baseline_run_sha256: str,
    candidate_run_sha256: str,
    expected_case_ids: set[str],
    failures: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    if blind_review_path is None:
        blockers.append("blind_review.missing")
        return {"status": "blocked", "decision_counts": {}}
    review = load_model(blind_review_path, BlindReview)
    if review.dataset_sha256 != dataset_sha256:
        failures.append("blind_review.dataset_hash_mismatch")
    if review.baseline_run_sha256 != baseline_run_sha256:
        failures.append("blind_review.baseline_run_hash_mismatch")
    if review.candidate_run_sha256 != candidate_run_sha256:
        failures.append("blind_review.candidate_run_hash_mismatch")
    decisions_by_case = {decision.case_id: decision.decision for decision in review.decisions}
    if set(decisions_by_case) != expected_case_ids:
        failures.append("blind_review.case_coverage_mismatch")
    decision_counts: dict[str, int] = {}
    for decision in decisions_by_case.values():
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        if decision == "baseline":
            failures.append("blind_review.candidate_rejected_for_case")
        elif decision == "reject_both":
            failures.append("blind_review.both_rejected_for_case")
    return {
        "status": _status(
            [code for code in failures if code.startswith("blind_review.")],
            [code for code in blockers if code.startswith("blind_review.")],
        ),
        "review_sha256": sha256_file(blind_review_path),
        "decision_counts": dict(sorted(decision_counts.items())),
    }


def _validated_string_list(
    value: Any,
    *,
    field_name: str,
    assessment: CaseAssessment,
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        assessment.failures.append(f"output.{field_name}_invalid")
        return []
    return value


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _mean(values: list[int | float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _status(failures: list[str], blockers: list[str]) -> str:
    if failures:
        return "failed"
    if blockers:
        return "blocked"
    return "passed"
