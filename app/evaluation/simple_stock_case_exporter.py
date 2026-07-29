from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.product_repository import FullCategoryMatrixRow
from app.db.models import DistributorProduct, DistributorStockPrice, MatchRun
from app.evaluation.simple_stock_contracts import (
    ANNOTATION_SCHEMA_VERSION,
    ArtifactFile,
    GoldenCase,
    GoldenDataset,
    GoldenExpectations,
    HumanAnnotation,
    PipelineStages,
)
from app.evaluation.simple_stock_evaluator import sha256_file
from app.matching.simple_stock_matrix import build_simple_stock_matrix_group_package

CASE_EXPORT_SCHEMA_VERSION = "simple-stock-case-export.v1"
CASE_EXPORTER_VERSION = "simple-stock-case-exporter.v1"
CASE_SOURCE_SCHEMA_VERSION = "simple-stock-case-source.v1"
REVIEW_DRAFT_SCHEMA_VERSION = "simple-stock-case-review-draft.v1"
REVIEW_RECEIPT_SCHEMA_VERSION = "simple-stock-case-review-receipt.v1"

MAX_CONTROL_FILE_BYTES = 5_000_000
MAX_CASE_ARTIFACT_BYTES = 25_000_000
DEFAULT_ATOMIC_CRITERIA = (
    "request-role-coverage",
    "compatibility-and-enablement",
    "commercial-price-choice",
    "no-unsupported-claims",
)

_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class CaseExportError(ValueError):
    def __init__(self, code: str, details: Sequence[str] = ()) -> None:
        super().__init__(code)
        self.code = code
        self.details = list(details)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MatrixEvidence(StrictModel):
    mode: Literal["reconstructed_as_of"]
    as_of: datetime
    earliest_snapshot_at: datetime | None = None
    latest_snapshot_at: datetime | None = None
    matched_diagnostics: list[str] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)


class CaseArtifacts(StrictModel):
    source: ArtifactFile
    output: ArtifactFile
    matrix: ArtifactFile
    review_draft: ArtifactFile


class CaseExportManifest(StrictModel):
    schema_version: Literal[CASE_EXPORT_SCHEMA_VERSION]
    exporter_version: Literal[CASE_EXPORTER_VERSION]
    case_id: str
    match_run_id: int = Field(gt=0)
    exported_at: datetime
    source_ref: str
    run_created_at: datetime
    privacy_class: Literal["sensitive_local_only"]
    production_code_revision: str = Field(min_length=7, max_length=64)
    pipeline_version: str
    result_state: str
    final_status_source: str
    distributor_code: str
    category_ids: list[str]
    stages: PipelineStages
    model_id: str
    required_atomic_criteria: list[str]
    tags: list[str]
    business_weight: float = Field(ge=0.1, le=100)
    expectations: GoldenExpectations
    artifacts: CaseArtifacts
    matrix_evidence: MatrixEvidence
    golden_review_eligible: bool
    blockers: list[str] = Field(default_factory=list)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("case_id must use lowercase safe identifiers")
        return value


class DraftCriterion(StrictModel):
    criterion_id: str
    status: Literal["pending", "pass", "fail", "not_applicable"] = "pending"
    evidence_refs: list[str] = Field(default_factory=list)


class ReviewDraft(StrictModel):
    schema_version: Literal[REVIEW_DRAFT_SCHEMA_VERSION]
    case_id: str
    output_sha256: str
    decision: Literal["pending", "accept", "reject"] = "pending"
    reviewer_role: str | None = Field(default=None, max_length=100)
    reviewed_at: datetime | None = None
    semantic_score: float | None = Field(default=None, ge=0, le=1)
    unsupported_material_claim_count: int | None = Field(default=None, ge=0)
    critical_error_codes: list[str] = Field(default_factory=list)
    business_weighted_loss: float | None = Field(default=None, ge=0)
    atomic_criteria: list[DraftCriterion]
    tags: list[str]
    business_weight: float = Field(ge=0.1, le=100)
    expectations: GoldenExpectations


class ReviewReceipt(StrictModel):
    schema_version: Literal[REVIEW_RECEIPT_SCHEMA_VERSION]
    case_id: str
    decision: Literal["accepted", "rejected"]
    finalized_at: datetime
    annotation: ArtifactFile
    golden_case: ArtifactFile
    manifest_sha256: str


async def enforce_postgresql_read_only_transaction(session: AsyncSession) -> None:
    bind = session.get_bind()
    dialect_name = bind.dialect.name if bind is not None else ""
    if dialect_name != "postgresql":
        raise CaseExportError("database.postgresql_required")
    await session.execute(text("SET TRANSACTION READ ONLY"))


async def export_case_from_session(
    session: AsyncSession,
    *,
    match_run_id: int,
    output_root: Path,
    production_baseline_path: Path,
    dataset_path: Path,
    case_id: str | None = None,
) -> CaseExportManifest:
    baseline = _read_json_mapping(production_baseline_path)
    dataset = _read_model(dataset_path, GoldenDataset)
    match_run = await session.scalar(select(MatchRun).where(MatchRun.id == match_run_id))
    if match_run is None:
        raise CaseExportError("match_run.not_found")

    report = _mapping(match_run.report_json)
    distributor_code = _required_text(report.get("distributor_code"), "run.distributor_missing")
    category_ids = _string_list(report.get("category_ids"))
    if not category_ids:
        raise CaseExportError("run.category_ids_missing")
    model_id = _report_model(report)
    max_package_chars = _baseline_max_package_chars(baseline)
    rows = await _list_matrix_rows_as_of(
        session,
        distributor_code=distributor_code,
        category_ids=category_ids,
        as_of=match_run.created_at,
    )
    matrix_package = build_simple_stock_matrix_group_package(
        distributor_code=distributor_code,
        category_ids=category_ids,
        rows=rows,
        max_package_chars=max_package_chars,
        model=model_id,
    )
    resolved_case_id = case_id or f"prod-{match_run.id}-{_safe_slug(distributor_code)}"
    return write_case_bundle(
        match_run=match_run,
        matrix_payload=matrix_package.payload,
        matrix_rows=rows,
        output_root=output_root,
        baseline=baseline,
        dataset=dataset,
        case_id=resolved_case_id,
    )


def write_case_bundle(
    *,
    match_run: MatchRun,
    matrix_payload: Mapping[str, Any],
    matrix_rows: Sequence[FullCategoryMatrixRow],
    output_root: Path,
    baseline: Mapping[str, Any],
    dataset: GoldenDataset,
    case_id: str,
    exported_at: datetime | None = None,
) -> CaseExportManifest:
    _validate_case_id(case_id)
    report = _mapping(match_run.report_json)
    pipeline_version = _required_text(
        report.get("pipeline_version"),
        "run.pipeline_version_missing",
    )
    if pipeline_version != "simple_stock_quote":
        raise CaseExportError("run.pipeline_not_simple_stock")
    if not str(match_run.source_text or "").strip():
        raise CaseExportError("run.source_text_missing")

    distributor_code = _required_text(report.get("distributor_code"), "run.distributor_missing")
    category_ids = _string_list(report.get("category_ids"))
    if not category_ids:
        raise CaseExportError("run.category_ids_missing")
    result_state = _required_text(
        report.get("v3_result_state") or report.get("result_state"),
        "run.result_state_missing",
    )
    final_status_source = _required_text(
        report.get("final_status_source"),
        "run.final_status_source_missing",
    )
    model_id = _report_model(report)
    stages, stage_blockers = _pipeline_stages(report, baseline)
    matrix_evidence = _matrix_evidence(
        report,
        matrix_payload,
        matrix_rows=matrix_rows,
        as_of=match_run.created_at,
    )
    blockers = sorted({*stage_blockers, *matrix_evidence.mismatches})

    existing_case = next((item for item in dataset.cases if item.case_id == case_id), None)
    criteria = list(
        existing_case.expectations.required_atomic_criteria
        if existing_case is not None
        else DEFAULT_ATOMIC_CRITERIA
    )
    expectations = (
        existing_case.expectations
        if existing_case is not None
        else _default_expectations(report, pipeline_version, final_status_source, result_state)
    )
    tags = (
        list(existing_case.tags)
        if existing_case is not None
        else ["normal", _safe_slug(distributor_code)]
    )
    business_weight = existing_case.business_weight if existing_case is not None else 1.0

    source_payload = {
        "schema_version": CASE_SOURCE_SCHEMA_VERSION,
        "case_id": case_id,
        "source_ref": f"production://match-runs/{match_run.id}",
        "match_run_id": match_run.id,
        "run_created_at": _aware(match_run.created_at).isoformat(),
        "source": match_run.source,
        "source_text": match_run.source_text,
        "spec_json": _mapping(match_run.spec_json),
    }
    output_payload = report
    matrix_payload_dict = dict(matrix_payload)

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = (output_root / case_id).resolve()
    if not target.is_relative_to(output_root):
        raise CaseExportError("output.path_escape")
    if target.exists():
        raise CaseExportError("output.bundle_exists")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{case_id}-", dir=output_root)).resolve()
    if not temp_dir.is_relative_to(output_root):
        raise CaseExportError("output.temp_path_escape")
    try:
        source_artifact = _write_json_artifact(temp_dir, "source.json", source_payload)
        output_artifact = _write_json_artifact(temp_dir, "output.json", output_payload)
        matrix_artifact = _write_json_artifact(temp_dir, "matrix.json", matrix_payload_dict)
        review_payload = {
            "schema_version": REVIEW_DRAFT_SCHEMA_VERSION,
            "case_id": case_id,
            "output_sha256": output_artifact.sha256,
            "decision": "pending",
            "reviewer_role": None,
            "reviewed_at": None,
            "semantic_score": None,
            "unsupported_material_claim_count": None,
            "critical_error_codes": [],
            "business_weighted_loss": None,
            "atomic_criteria": [
                {
                    "criterion_id": criterion_id,
                    "status": "pending",
                    "evidence_refs": [],
                }
                for criterion_id in criteria
            ],
            "tags": tags,
            "business_weight": business_weight,
            "expectations": expectations.model_dump(mode="json"),
        }
        review_artifact = _write_json_artifact(temp_dir, "review.json", review_payload)
        manifest = CaseExportManifest(
            schema_version=CASE_EXPORT_SCHEMA_VERSION,
            exporter_version=CASE_EXPORTER_VERSION,
            case_id=case_id,
            match_run_id=match_run.id,
            exported_at=_aware(exported_at or datetime.now(UTC)),
            source_ref=f"production://match-runs/{match_run.id}",
            run_created_at=_aware(match_run.created_at),
            privacy_class="sensitive_local_only",
            production_code_revision=_baseline_code_revision(baseline),
            pipeline_version=pipeline_version,
            result_state=result_state,
            final_status_source=final_status_source,
            distributor_code=distributor_code,
            category_ids=category_ids,
            stages=stages,
            model_id=model_id,
            required_atomic_criteria=criteria,
            tags=tags,
            business_weight=business_weight,
            expectations=expectations,
            artifacts=CaseArtifacts(
                source=source_artifact,
                output=output_artifact,
                matrix=matrix_artifact,
                review_draft=review_artifact,
            ),
            matrix_evidence=matrix_evidence,
            golden_review_eligible=not blockers,
            blockers=blockers,
        )
        _write_json_artifact(
            temp_dir,
            "manifest.json",
            manifest.model_dump(mode="json"),
        )
        os.replace(temp_dir, target)
        return manifest
    except Exception:
        if temp_dir.exists() and temp_dir.is_relative_to(output_root):
            shutil.rmtree(temp_dir)
        raise


def finalize_case_review(
    *,
    bundle_dir: Path,
    dataset_path: Path,
    finalized_at: datetime | None = None,
) -> ReviewReceipt:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    review_path = bundle_dir / "review.json"
    manifest = _read_model(manifest_path, CaseExportManifest)
    draft = _read_model(review_path, ReviewDraft)
    dataset = _read_model(dataset_path, GoldenDataset)
    _validate_review_draft(manifest, draft, dataset)
    _validate_bundle_artifacts(bundle_dir, manifest)

    reviewer_role = str(draft.reviewer_role or "").strip()
    reviewed_at = _aware_or_error(draft.reviewed_at, "review.reviewed_at_required")
    accepted = draft.decision == "accept"
    try:
        annotation = HumanAnnotation(
            schema_version=ANNOTATION_SCHEMA_VERSION,
            case_id=draft.case_id,
            output_sha256=draft.output_sha256,
            reviewer_role=reviewer_role,
            reviewed_at=reviewed_at,
            semantic_score=draft.semantic_score,
            unsupported_material_claim_count=draft.unsupported_material_claim_count,
            critical_error_codes=draft.critical_error_codes,
            business_weighted_loss=draft.business_weighted_loss,
            atomic_criteria=[
                {
                    "criterion_id": item.criterion_id,
                    "status": item.status,
                    "evidence_refs": item.evidence_refs,
                }
                for item in draft.atomic_criteria
            ],
        )
        golden_case = GoldenCase(
            case_id=draft.case_id,
            status="accepted" if accepted else "rejected",
            tags=manifest.tags,
            business_weight=manifest.business_weight,
            source={
                "ref": manifest.source_ref,
                "sha256": manifest.artifacts.source.sha256,
                "privacy_class": "sensitive_local_only",
            },
            matrix_source={
                "ref": f"local://golden/simple-stock/{draft.case_id}/matrix.json",
                "sha256": manifest.artifacts.matrix.sha256,
                "privacy_class": "sensitive_local_only",
            },
            expectations=manifest.expectations,
            reviewed_by_role=reviewer_role,
            reviewed_at=reviewed_at,
        )
    except ValidationError as exc:
        details = [
            f"{'.'.join(str(part) for part in error['loc'])}:{error['type']}"
            for error in exc.errors(include_input=False)
        ]
        raise CaseExportError("review.schema_invalid", details) from exc

    annotation_path = bundle_dir / "annotation.json"
    golden_case_path = bundle_dir / "golden-case.json"
    receipt_path = bundle_dir / "review-receipt.json"
    for path in (annotation_path, golden_case_path, receipt_path):
        if path.exists():
            raise CaseExportError("review.output_exists", [path.name])

    created_paths: list[Path] = []
    try:
        annotation_artifact = _write_json_artifact(
            bundle_dir,
            annotation_path.name,
            annotation.model_dump(mode="json"),
        )
        created_paths.append(annotation_path)
        golden_case_artifact = _write_json_artifact(
            bundle_dir,
            golden_case_path.name,
            golden_case.model_dump(mode="json"),
        )
        created_paths.append(golden_case_path)
        receipt = ReviewReceipt(
            schema_version=REVIEW_RECEIPT_SCHEMA_VERSION,
            case_id=draft.case_id,
            decision="accepted" if accepted else "rejected",
            finalized_at=_aware(finalized_at or datetime.now(UTC)),
            annotation=annotation_artifact,
            golden_case=golden_case_artifact,
            manifest_sha256=sha256_file(manifest_path),
        )
        _write_json_artifact(
            bundle_dir,
            receipt_path.name,
            receipt.model_dump(mode="json"),
        )
        created_paths.append(receipt_path)
        return receipt
    except Exception:
        for path in reversed(created_paths):
            path.unlink(missing_ok=True)
        raise


def read_case_manifest(bundle_dir: Path) -> CaseExportManifest:
    """Load a case manifest through the same bounded schema gate as finalization."""
    return _read_model(bundle_dir.resolve() / "manifest.json", CaseExportManifest)


def read_review_draft(bundle_dir: Path) -> ReviewDraft:
    """Load an editable review draft without exposing the underlying raw JSON parser."""
    return _read_model(bundle_dir.resolve() / "review.json", ReviewDraft)


def validate_case_bundle(bundle_dir: Path) -> CaseExportManifest:
    """Verify immutable artifacts and return the validated manifest."""
    resolved = bundle_dir.resolve()
    manifest = read_case_manifest(resolved)
    _validate_bundle_artifacts(resolved, manifest)
    return manifest


async def _list_matrix_rows_as_of(
    session: AsyncSession,
    *,
    distributor_code: str,
    category_ids: Sequence[str],
    as_of: datetime,
) -> list[FullCategoryMatrixRow]:
    latest_by_category = (
        select(
            DistributorProduct.category_id.label("category_id"),
            func.max(DistributorStockPrice.synced_at).label("latest_synced_at"),
        )
        .join(
            DistributorStockPrice,
            (DistributorProduct.distributor_code == DistributorStockPrice.distributor_code)
            & (DistributorProduct.item_id == DistributorStockPrice.item_id),
        )
        .where(
            DistributorProduct.distributor_code == distributor_code,
            DistributorProduct.category_id.in_(list(category_ids)),
            DistributorProduct.created_at <= as_of,
            DistributorStockPrice.synced_at <= as_of,
        )
        .group_by(DistributorProduct.category_id)
        .subquery()
    )
    result = await session.execute(
        select(DistributorProduct, DistributorStockPrice)
        .join(
            DistributorStockPrice,
            (DistributorProduct.distributor_code == DistributorStockPrice.distributor_code)
            & (DistributorProduct.item_id == DistributorStockPrice.item_id),
        )
        .join(
            latest_by_category,
            (DistributorProduct.category_id == latest_by_category.c.category_id)
            & (DistributorStockPrice.synced_at == latest_by_category.c.latest_synced_at),
        )
        .where(
            DistributorProduct.distributor_code == distributor_code,
            DistributorProduct.category_id.in_(list(category_ids)),
            DistributorProduct.created_at <= as_of,
        )
        .order_by(
            DistributorProduct.category_id,
            DistributorProduct.item_id,
            DistributorStockPrice.location,
            DistributorStockPrice.id,
        )
    )
    return [
        FullCategoryMatrixRow(product=product, stock=stock)
        for product, stock in result.all()
    ]


def _matrix_evidence(
    report: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    matrix_rows: Sequence[FullCategoryMatrixRow],
    as_of: datetime,
) -> MatrixEvidence:
    diagnostics = _mapping(report.get("diagnostics"))
    matrix_diagnostics = _mapping(matrix.get("diagnostics"))
    expected = {
        "matrix.schema_version": diagnostics.get("matrix_schema_version"),
        "matrix.row_count": diagnostics.get("matrix_row_count"),
        "matrix.position_count": diagnostics.get("matrix_position_count"),
        "matrix.char_count": diagnostics.get("matrix_char_count"),
        "matrix.status": diagnostics.get("matrix_status"),
        "matrix.model": diagnostics.get("model"),
        "matrix.distributor_code": report.get("distributor_code"),
        "matrix.category_ids": _string_list(report.get("category_ids")),
    }
    actual = {
        "matrix.schema_version": matrix.get("schema_version"),
        "matrix.row_count": matrix_diagnostics.get("row_count"),
        "matrix.position_count": matrix_diagnostics.get("position_count"),
        "matrix.char_count": matrix_diagnostics.get("char_count"),
        "matrix.status": matrix_diagnostics.get("status"),
        "matrix.model": matrix.get("model"),
        "matrix.distributor_code": matrix.get("distributor_code"),
        "matrix.category_ids": _string_list(matrix.get("category_ids")),
    }
    matched: list[str] = []
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        if expected_value in (None, "", []):
            mismatches.append(f"{key}.evidence_missing")
        elif actual[key] != expected_value:
            mismatches.append(f"{key}.mismatch")
        else:
            matched.append(key)

    snapshot_times = sorted({_aware(row.stock.synced_at) for row in matrix_rows})
    return MatrixEvidence(
        mode="reconstructed_as_of",
        as_of=_aware(as_of),
        earliest_snapshot_at=snapshot_times[0] if snapshot_times else None,
        latest_snapshot_at=snapshot_times[-1] if snapshot_times else None,
        matched_diagnostics=matched,
        mismatches=mismatches,
    )


def _pipeline_stages(
    report: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> tuple[PipelineStages, list[str]]:
    baseline_stages = _mapping(baseline.get("stages"))
    diagnostics = _mapping(report.get("diagnostics"))
    route_decision = _mapping(
        report.get("simple_route_decision") or diagnostics.get("simple_route_decision")
    )
    observed = {
        "route_prompt_version": route_decision.get("prompt_version"),
        "matrix_schema_version": diagnostics.get("matrix_schema_version"),
        "composer_prompt_version": diagnostics.get("composer_prompt_version"),
        "reconciler_version": diagnostics.get("quote_integrity_reconciler"),
    }
    values: dict[str, str] = {}
    blockers: list[str] = []
    for key in (
        "route_prompt_version",
        "matrix_schema_version",
        "composer_prompt_version",
        "reconciler_version",
    ):
        expected = str(baseline_stages.get(key) or "").strip()
        if not expected:
            raise CaseExportError("baseline.stage_missing", [key])
        actual = str(observed.get(key) or "").strip()
        if not actual:
            blockers.append(f"pipeline.{key}.missing")
        elif actual != expected:
            blockers.append(f"pipeline.{key}.mismatch")
        values[key] = actual or expected
    return PipelineStages(**values), blockers


def _validate_review_draft(
    manifest: CaseExportManifest,
    draft: ReviewDraft,
    dataset: GoldenDataset,
) -> None:
    failures: list[str] = []
    if draft.case_id != manifest.case_id:
        failures.append("review.case_id_mismatch")
    if draft.output_sha256 != manifest.artifacts.output.sha256:
        failures.append("review.output_hash_mismatch")
    if (
        draft.tags != manifest.tags
        or draft.business_weight != manifest.business_weight
        or draft.expectations != manifest.expectations
    ):
        failures.append("review.contract_changed")
    if draft.decision == "pending":
        failures.append("review.decision_pending")
    if not str(draft.reviewer_role or "").strip():
        failures.append("review.reviewer_role_required")
    if draft.reviewed_at is None or draft.reviewed_at.utcoffset() is None:
        failures.append("review.reviewed_at_required")
    if draft.semantic_score is None:
        failures.append("review.semantic_score_required")
    if draft.unsupported_material_claim_count is None:
        failures.append("review.unsupported_claim_count_required")
    if draft.business_weighted_loss is None:
        failures.append("review.business_weighted_loss_required")

    criteria = {item.criterion_id: item.status for item in draft.atomic_criteria}
    required = set(manifest.required_atomic_criteria)
    if set(criteria) != required or len(criteria) != len(draft.atomic_criteria):
        failures.append("review.atomic_criteria_mismatch")
    if any(status == "pending" for status in criteria.values()):
        failures.append("review.atomic_criteria_pending")

    if draft.decision == "accept":
        thresholds = dataset.thresholds
        if not manifest.golden_review_eligible:
            failures.append("review.bundle_not_eligible")
        if (
            draft.semantic_score is not None
            and draft.semantic_score < thresholds.minimum_semantic_score
        ):
            failures.append("review.semantic_score_below_threshold")
        if (
            draft.unsupported_material_claim_count is not None
            and draft.unsupported_material_claim_count
            > thresholds.unsupported_material_claim_count
        ):
            failures.append("review.unsupported_claim_count_exceeded")
        if len(draft.critical_error_codes) > thresholds.critical_error_count:
            failures.append("review.critical_error_count_exceeded")
        if (
            draft.business_weighted_loss is not None
            and draft.business_weighted_loss > thresholds.maximum_business_weighted_loss
        ):
            failures.append("review.business_weighted_loss_exceeded")
        if any(criteria.get(item) != "pass" for item in required):
            failures.append("review.required_atomic_criterion_not_passed")
    if failures:
        raise CaseExportError("review.invalid", sorted(set(failures)))


def _validate_bundle_artifacts(bundle_dir: Path, manifest: CaseExportManifest) -> None:
    failures: list[str] = []
    for name, artifact in (
        ("source", manifest.artifacts.source),
        ("output", manifest.artifacts.output),
        ("matrix", manifest.artifacts.matrix),
    ):
        path = (bundle_dir / artifact.path).resolve()
        if not path.is_relative_to(bundle_dir):
            failures.append(f"artifact.{name}.path_escape")
        elif not path.is_file():
            failures.append(f"artifact.{name}.missing")
        elif path.stat().st_size > MAX_CASE_ARTIFACT_BYTES:
            failures.append(f"artifact.{name}.too_large")
        elif sha256_file(path) != artifact.sha256:
            failures.append(f"artifact.{name}.hash_mismatch")
    if failures:
        raise CaseExportError("bundle.invalid", failures)


def _default_expectations(
    report: Mapping[str, Any],
    pipeline_version: str,
    final_status_source: str,
    result_state: str,
) -> GoldenExpectations:
    quote = _mapping(report.get("validated_quote"))
    lines = quote.get("lines")
    minimum_quote_lines = 1 if isinstance(lines, list) and lines else 0
    return GoldenExpectations(
        allowed_result_states=[result_state],
        required_pipeline_version=pipeline_version,
        allowed_final_status_sources=[final_status_source],
        minimum_quote_lines=minimum_quote_lines,
        maximum_validation_errors=0,
        maximum_validation_warnings=0,
        require_engineering_review=True,
        required_atomic_criteria=list(DEFAULT_ATOMIC_CRITERIA),
    )


def _write_json_artifact(root: Path, name: str, payload: Mapping[str, Any]) -> ArtifactFile:
    path = root / name
    data = _json_bytes(payload)
    if len(data) > MAX_CASE_ARTIFACT_BYTES:
        raise CaseExportError("artifact.file_too_large", [name])
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise CaseExportError("artifact.file_exists", [name]) from exc
    return ArtifactFile(path=name, sha256=hashlib.sha256(data).hexdigest())


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )
    return (rendered + "\n").encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _aware(value).isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > MAX_CONTROL_FILE_BYTES:
            raise CaseExportError("input.file_too_large", [path.name])
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise CaseExportError("input.file_not_found", [path.name]) from exc
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CaseExportError("input.invalid_json", [path.name]) from exc
    if not isinstance(payload, dict):
        raise CaseExportError("input.mapping_required", [path.name])
    return payload


def _read_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    payload = _read_json_mapping(path)
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        details = [
            f"{'.'.join(str(part) for part in error['loc'])}:{error['type']}"
            for error in exc.errors(include_input=False)
        ]
        raise CaseExportError("input.schema_invalid", details) from exc


def _report_model(report: Mapping[str, Any]) -> str:
    diagnostics = _mapping(report.get("diagnostics"))
    return _required_text(diagnostics.get("model"), "run.model_missing")


def _baseline_max_package_chars(baseline: Mapping[str, Any]) -> int:
    llm = _mapping(baseline.get("llm"))
    value = llm.get("max_package_chars")
    if not isinstance(value, int) or value <= 0:
        raise CaseExportError("baseline.max_package_chars_invalid")
    return value


def _baseline_code_revision(baseline: Mapping[str, Any]) -> str:
    return _required_text(
        baseline.get("production_commit"),
        "baseline.production_commit_missing",
    )


def _validate_case_id(value: str) -> None:
    if not _SAFE_ID_RE.fullmatch(value):
        raise CaseExportError("case_id.invalid")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return slug or "unknown"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return [item for item in dict.fromkeys(str(item or "").strip() for item in value) if item]


def _required_text(value: Any, code: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise CaseExportError(code)
    return cleaned


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.utcoffset() is None else value


def _aware_or_error(value: datetime | None, code: str) -> datetime:
    if value is None or value.utcoffset() is None:
        raise CaseExportError(code)
    return value


__all__ = [
    "CASE_EXPORTER_VERSION",
    "CASE_EXPORT_SCHEMA_VERSION",
    "CaseExportError",
    "CaseExportManifest",
    "ReviewDraft",
    "ReviewReceipt",
    "enforce_postgresql_read_only_transaction",
    "export_case_from_session",
    "finalize_case_review",
    "read_case_manifest",
    "read_review_draft",
    "validate_case_bundle",
    "write_case_bundle",
]
