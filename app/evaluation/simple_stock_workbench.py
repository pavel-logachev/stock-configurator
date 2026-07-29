from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MatchRun
from app.evaluation.simple_stock_case_exporter import (
    CaseExportError,
    DraftCriterion,
    ReviewDraft,
    ReviewReceipt,
    export_case_from_session,
    finalize_case_review,
    read_case_manifest,
    read_review_draft,
    validate_case_bundle,
)
from app.evaluation.simple_stock_evaluator import sha256_file

CATALOG_SCHEMA_VERSION = "simple-stock-candidate-catalog.v1"
BATCH_EXPORT_SCHEMA_VERSION = "simple-stock-batch-export-receipt.v1"
QUALITY_SUMMARY_SCHEMA_VERSION = "simple-stock-quality-summary.v1"
MAX_CATALOG_LIMIT = 100
MAX_CATALOG_SCAN = 1_000
MAX_BATCH_EXPORT = 20
MAX_LOCAL_CASES = 500
MAX_DISPLAY_TEXT = 100_000

_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CatalogCandidate(StrictModel):
    match_run_id: int
    created_at: datetime
    run_status: str
    pipeline_version: str
    result_state: str | None
    final_status_source: str | None
    distributor_code: str | None
    category_count: int
    total_candidates: int
    matched_items: int
    engineer_review_required: bool
    exportable: bool
    blockers: list[str]


class CandidateCatalog(StrictModel):
    schema_version: Literal[CATALOG_SCHEMA_VERSION]
    generated_at: datetime
    limit: int
    before_id: int | None
    scanned_count: int
    candidates: list[CatalogCandidate]
    privacy: Literal["safe_metadata_only"] = "safe_metadata_only"


class BatchExportItem(StrictModel):
    match_run_id: int
    case_id: str | None = None
    status: Literal["exported", "blocked", "already_exists"]
    golden_review_eligible: bool | None = None
    blockers: list[str] = Field(default_factory=list)
    error: str | None = None


class BatchExportReceipt(StrictModel):
    schema_version: Literal[BATCH_EXPORT_SCHEMA_VERSION]
    batch_id: str
    created_at: datetime
    requested_count: int
    exported_count: int
    blocked_count: int
    already_exists_count: int
    database_mode: Literal["transaction_read_only"] = "transaction_read_only"
    database_writes: Literal[0] = 0
    llm_calls: Literal[0] = 0
    items: list[BatchExportItem]


class CriterionMutation(StrictModel):
    criterion_id: str = Field(min_length=1, max_length=100)
    status: Literal["pending", "pass", "fail", "not_applicable"]
    evidence_refs: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if any(len(item) > 500 for item in cleaned):
            raise ValueError("evidence reference exceeds 500 characters")
        return list(dict.fromkeys(cleaned))


class ReviewMutation(StrictModel):
    decision: Literal["pending", "accept", "reject"]
    reviewer_role: str | None = Field(default=None, max_length=100)
    semantic_score: float | None = Field(default=None, ge=0, le=1)
    unsupported_material_claim_count: int | None = Field(default=None, ge=0, le=10_000)
    critical_error_codes: list[str] = Field(default_factory=list, max_length=50)
    business_weighted_loss: float | None = Field(default=None, ge=0, le=1_000_000)
    atomic_criteria: list[CriterionMutation] = Field(max_length=50)

    @field_validator("critical_error_codes")
    @classmethod
    def validate_critical_codes(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if any(len(item) > 100 for item in cleaned):
            raise ValueError("critical error code exceeds 100 characters")
        return list(dict.fromkeys(cleaned))


class QualitySummary(StrictModel):
    schema_version: Literal[QUALITY_SUMMARY_SCHEMA_VERSION]
    generated_at: datetime
    total_cases: int
    eligible_cases: int
    blocked_cases: int
    invalid_cases: int
    pending_cases: int
    accepted_cases: int
    rejected_cases: int
    finalized_cases: int
    human_review_coverage: float
    average_semantic_score: float | None
    unsupported_material_claim_count: int
    business_weighted_loss: float


async def list_match_run_catalog(
    session: AsyncSession,
    *,
    production_baseline_path: Path,
    limit: int = 25,
    before_id: int | None = None,
) -> CandidateCatalog:
    if not 1 <= limit <= MAX_CATALOG_LIMIT:
        raise CaseExportError("catalog.limit_invalid")
    if before_id is not None and before_id <= 0:
        raise CaseExportError("catalog.before_id_invalid")
    baseline = _read_json_mapping(production_baseline_path)
    scan_limit = min(max(limit * 10, limit), MAX_CATALOG_SCAN)
    statement = select(MatchRun).order_by(MatchRun.id.desc()).limit(scan_limit)
    if before_id is not None:
        statement = statement.where(MatchRun.id < before_id)
    result = await session.execute(statement)
    runs = list(result.scalars())
    candidates: list[CatalogCandidate] = []
    for run in runs:
        candidate = _catalog_candidate(run, baseline)
        if candidate.pipeline_version != "simple_stock_quote":
            continue
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return CandidateCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        generated_at=datetime.now(UTC),
        limit=limit,
        before_id=before_id,
        scanned_count=len(runs),
        candidates=candidates,
    )


async def export_case_batch(
    session: AsyncSession,
    *,
    match_run_ids: Sequence[int],
    output_root: Path,
    production_baseline_path: Path,
    dataset_path: Path,
    receipt_root: Path | None = None,
    batch_id: str | None = None,
) -> BatchExportReceipt:
    ids = list(dict.fromkeys(match_run_ids))
    if not ids or len(ids) > MAX_BATCH_EXPORT or any(item <= 0 for item in ids):
        raise CaseExportError("batch.match_run_ids_invalid")
    resolved_batch_id = batch_id or _new_batch_id()
    _validate_safe_id(resolved_batch_id, "batch.id_invalid")

    items: list[BatchExportItem] = []
    for match_run_id in ids:
        try:
            manifest = await export_case_from_session(
                session,
                match_run_id=match_run_id,
                output_root=output_root,
                production_baseline_path=production_baseline_path,
                dataset_path=dataset_path,
            )
            items.append(
                BatchExportItem(
                    match_run_id=match_run_id,
                    case_id=manifest.case_id,
                    status="exported",
                    golden_review_eligible=manifest.golden_review_eligible,
                    blockers=manifest.blockers,
                )
            )
        except CaseExportError as exc:
            case_id = _expected_case_id(match_run_id, output_root)
            status: Literal["blocked", "already_exists"] = (
                "already_exists" if exc.code == "output.bundle_exists" else "blocked"
            )
            items.append(
                BatchExportItem(
                    match_run_id=match_run_id,
                    case_id=case_id,
                    status=status,
                    blockers=exc.details,
                    error=exc.code,
                )
            )

    receipt = BatchExportReceipt(
        schema_version=BATCH_EXPORT_SCHEMA_VERSION,
        batch_id=resolved_batch_id,
        created_at=datetime.now(UTC),
        requested_count=len(ids),
        exported_count=sum(item.status == "exported" for item in items),
        blocked_count=sum(item.status == "blocked" for item in items),
        already_exists_count=sum(item.status == "already_exists" for item in items),
        items=items,
    )
    destination = (receipt_root or output_root.parent / "batches").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _write_append_only_json(
        destination / f"{resolved_batch_id}.json",
        receipt.model_dump(mode="json"),
    )
    return receipt


class LocalCaseStore:
    def __init__(self, *, case_root: Path, dataset_path: Path) -> None:
        self.case_root = case_root.resolve()
        self.dataset_path = dataset_path.resolve()
        self.case_root.mkdir(parents=True, exist_ok=True)
        if not self.dataset_path.is_file():
            raise CaseExportError("dataset.not_found")

    def list_cases(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for path in sorted(self.case_root.iterdir(), key=lambda item: item.name, reverse=True):
            if len(result) >= MAX_LOCAL_CASES:
                break
            if not path.is_dir() or not _SAFE_ID_RE.fullmatch(path.name):
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(self.case_root):
                continue
            result.append(self._case_list_item(resolved))
        return result

    def quality_summary(self) -> QualitySummary:
        cases = self.list_cases()
        finalized = [item for item in cases if item["review_state"] in {"accepted", "rejected"}]
        scores: list[float] = []
        unsupported_count = 0
        weighted_loss = 0.0
        for item in finalized:
            bundle = self._bundle_dir(item["case_id"])
            receipt = _validated_review_receipt(bundle)
            if receipt is None:
                continue
            annotation_path = (bundle / receipt.annotation.path).resolve()
            annotation = _read_json_mapping(annotation_path)
            score = annotation.get("semantic_score")
            if isinstance(score, int | float):
                scores.append(float(score))
            claims = annotation.get("unsupported_material_claim_count")
            if isinstance(claims, int):
                unsupported_count += claims
            loss = annotation.get("business_weighted_loss")
            if isinstance(loss, int | float):
                weighted_loss += float(loss)
        total = len(cases)
        finalized_count = len(finalized)
        return QualitySummary(
            schema_version=QUALITY_SUMMARY_SCHEMA_VERSION,
            generated_at=datetime.now(UTC),
            total_cases=total,
            eligible_cases=sum(item["golden_review_eligible"] for item in cases),
            blocked_cases=sum(item["integrity_state"] == "blocked" for item in cases),
            invalid_cases=sum(item["integrity_state"] == "invalid" for item in cases),
            pending_cases=sum(item["review_state"] == "pending" for item in cases),
            accepted_cases=sum(item["review_state"] == "accepted" for item in cases),
            rejected_cases=sum(item["review_state"] == "rejected" for item in cases),
            finalized_cases=finalized_count,
            human_review_coverage=(finalized_count / total if total else 0.0),
            average_semantic_score=(sum(scores) / len(scores) if scores else None),
            unsupported_material_claim_count=unsupported_count,
            business_weighted_loss=weighted_loss,
        )

    def case_detail(self, case_id: str) -> dict[str, Any]:
        bundle = self._bundle_dir(case_id)
        manifest = read_case_manifest(bundle)
        integrity_errors: list[str] = []
        try:
            validate_case_bundle(bundle)
        except CaseExportError as exc:
            integrity_errors = [exc.code, *exc.details]

        review: ReviewDraft | None = None
        try:
            review = read_review_draft(bundle)
        except CaseExportError as exc:
            integrity_errors.extend([exc.code, *exc.details])

        try:
            validated_receipt = _validated_review_receipt(bundle)
        except CaseExportError as exc:
            validated_receipt = None
            integrity_errors.extend([exc.code, *exc.details])
        receipt = (
            validated_receipt.model_dump(mode="json")
            if validated_receipt is not None
            else None
        )
        base = {
            "case_id": manifest.case_id,
            "match_run_id": manifest.match_run_id,
            "run_created_at": manifest.run_created_at.isoformat(),
            "exported_at": manifest.exported_at.isoformat(),
            "privacy_class": manifest.privacy_class,
            "production_code_revision": manifest.production_code_revision,
            "pipeline_version": manifest.pipeline_version,
            "result_state": manifest.result_state,
            "final_status_source": manifest.final_status_source,
            "distributor_code": manifest.distributor_code,
            "category_count": len(manifest.category_ids),
            "model_id": manifest.model_id,
            "stages": manifest.stages.model_dump(mode="json"),
            "matrix_evidence": manifest.matrix_evidence.model_dump(mode="json"),
            "artifact_hashes": {
                "source": manifest.artifacts.source.sha256,
                "output": manifest.artifacts.output.sha256,
                "matrix": manifest.artifacts.matrix.sha256,
            },
            "golden_review_eligible": manifest.golden_review_eligible,
            "blockers": manifest.blockers,
            "integrity_state": "verified" if not integrity_errors else "invalid",
            "integrity_errors": sorted(set(integrity_errors)),
            "review_state": _review_state(receipt),
            "review": review.model_dump(mode="json") if review is not None else None,
            "receipt": receipt,
            "source": None,
            "quote": None,
            "validation": None,
        }
        if integrity_errors:
            return base

        source = _read_json_mapping(bundle / manifest.artifacts.source.path)
        output = _read_json_mapping(bundle / manifest.artifacts.output.path)
        base["source"] = {
            "source": _bounded_text(source.get("source")),
            "source_text": _bounded_text(source.get("source_text")),
        }
        base["quote"] = _curated_quote(output.get("validated_quote"))
        base["validation"] = {
            "errors": _display_list(output.get("v3_validation_errors")),
            "warnings": _display_list(output.get("v3_validation_warnings")),
            "error_details": _display_list(output.get("v3_validation_error_details")),
        }
        return base

    def save_review(self, case_id: str, mutation: ReviewMutation) -> ReviewDraft:
        bundle = self._bundle_dir(case_id)
        manifest = validate_case_bundle(bundle)
        if (bundle / "review-receipt.json").exists():
            raise CaseExportError("review.already_finalized")
        criteria = {item.criterion_id: item for item in mutation.atomic_criteria}
        if len(criteria) != len(mutation.atomic_criteria) or set(criteria) != set(
            manifest.required_atomic_criteria
        ):
            raise CaseExportError("review.atomic_criteria_mismatch")
        reviewer_role = str(mutation.reviewer_role or "").strip() or None
        reviewed_at = (
            datetime.now(UTC)
            if mutation.decision != "pending" and reviewer_role is not None
            else None
        )
        draft = ReviewDraft(
            schema_version="simple-stock-case-review-draft.v1",
            case_id=manifest.case_id,
            output_sha256=manifest.artifacts.output.sha256,
            decision=mutation.decision,
            reviewer_role=reviewer_role,
            reviewed_at=reviewed_at,
            semantic_score=mutation.semantic_score,
            unsupported_material_claim_count=mutation.unsupported_material_claim_count,
            critical_error_codes=mutation.critical_error_codes,
            business_weighted_loss=mutation.business_weighted_loss,
            atomic_criteria=[
                DraftCriterion(
                    criterion_id=criterion_id,
                    status=criteria[criterion_id].status,
                    evidence_refs=criteria[criterion_id].evidence_refs,
                )
                for criterion_id in manifest.required_atomic_criteria
            ],
            tags=manifest.tags,
            business_weight=manifest.business_weight,
            expectations=manifest.expectations,
        )
        _atomic_replace_json(bundle / "review.json", draft.model_dump(mode="json"))
        return draft

    def finalize_review(self, case_id: str, *, confirm_case_id: str) -> dict[str, Any]:
        if confirm_case_id != case_id:
            raise CaseExportError("review.confirmation_mismatch")
        bundle = self._bundle_dir(case_id)
        receipt = finalize_case_review(bundle_dir=bundle, dataset_path=self.dataset_path)
        return receipt.model_dump(mode="json")

    def _case_list_item(self, bundle: Path) -> dict[str, Any]:
        try:
            manifest = read_case_manifest(bundle)
        except CaseExportError:
            return {
                "case_id": bundle.name,
                "run_created_at": None,
                "pipeline_version": None,
                "result_state": None,
                "distributor_code": None,
                "golden_review_eligible": False,
                "integrity_state": "invalid",
                "review_state": "pending",
                "blocker_count": 1,
            }
        try:
            validate_case_bundle(bundle)
            integrity_state = "verified" if manifest.golden_review_eligible else "blocked"
        except CaseExportError:
            integrity_state = "invalid"
        try:
            receipt = _validated_review_receipt(bundle)
        except CaseExportError:
            receipt = None
            integrity_state = "invalid"
        return {
            "case_id": manifest.case_id,
            "run_created_at": manifest.run_created_at.isoformat(),
            "pipeline_version": manifest.pipeline_version,
            "result_state": manifest.result_state,
            "distributor_code": manifest.distributor_code,
            "golden_review_eligible": manifest.golden_review_eligible,
            "integrity_state": integrity_state,
            "review_state": _review_state(
                receipt.model_dump(mode="json") if receipt is not None else None
            ),
            "blocker_count": len(manifest.blockers),
        }

    def _bundle_dir(self, case_id: str) -> Path:
        _validate_safe_id(case_id, "case_id.invalid")
        bundle = (self.case_root / case_id).resolve()
        if not bundle.is_relative_to(self.case_root):
            raise CaseExportError("case.path_escape")
        if not bundle.is_dir():
            raise CaseExportError("case.not_found")
        return bundle


def _catalog_candidate(run: MatchRun, baseline: Mapping[str, Any]) -> CatalogCandidate:
    report = _mapping(run.report_json)
    diagnostics = _mapping(report.get("diagnostics"))
    route = _mapping(
        report.get("simple_route_decision") or diagnostics.get("simple_route_decision")
    )
    stages = _mapping(baseline.get("stages"))
    observed = {
        "route_prompt_version": route.get("prompt_version"),
        "matrix_schema_version": diagnostics.get("matrix_schema_version"),
        "composer_prompt_version": diagnostics.get("composer_prompt_version"),
        "reconciler_version": diagnostics.get("quote_integrity_reconciler"),
    }
    blockers: list[str] = []
    pipeline_version = str(report.get("pipeline_version") or "")
    if pipeline_version != "simple_stock_quote":
        blockers.append("pipeline.not_simple_stock")
    for field in ("distributor_code", "category_ids", "final_status_source"):
        if report.get(field) in (None, "", []):
            blockers.append(f"run.{field}.missing")
    if not str(report.get("v3_result_state") or report.get("result_state") or "").strip():
        blockers.append("run.result_state.missing")
    if not str(run.source_text or "").strip():
        blockers.append("run.source_text.missing")
    if not str(diagnostics.get("model") or "").strip():
        blockers.append("run.model.missing")
    for key, expected in stages.items():
        if key not in observed:
            continue
        actual = str(observed[key] or "").strip()
        if not actual:
            blockers.append(f"pipeline.{key}.missing")
        elif actual != str(expected or "").strip():
            blockers.append(f"pipeline.{key}.mismatch")
    return CatalogCandidate(
        match_run_id=run.id,
        created_at=_aware(run.created_at),
        run_status=run.status,
        pipeline_version=pipeline_version,
        result_state=str(report.get("v3_result_state") or report.get("result_state") or "") or None,
        final_status_source=str(report.get("final_status_source") or "") or None,
        distributor_code=str(report.get("distributor_code") or "") or None,
        category_count=len(_string_list(report.get("category_ids"))),
        total_candidates=run.total_candidates,
        matched_items=run.matched_items,
        engineer_review_required=run.engineer_review_required,
        exportable=not blockers,
        blockers=sorted(set(blockers)),
    )


def _curated_quote(value: Any) -> dict[str, Any]:
    quote = _mapping(value)
    lines: list[dict[str, Any]] = []
    raw_lines = quote.get("lines")
    if isinstance(raw_lines, list):
        for raw in raw_lines[:100]:
            line = _mapping(raw)
            lines.append(
                {
                    "line_id": _bounded_text(line.get("line_id"), 200),
                    "requirement_id": _bounded_text(line.get("requirement_id"), 200),
                    "role": _bounded_text(line.get("role"), 200),
                    "part_number": _bounded_text(line.get("part_number"), 500),
                    "item_name": _bounded_text(line.get("item_name"), 5_000),
                    "quantity": line.get("quantity"),
                    "available_quantity": line.get("available_quantity"),
                    "fit_status": _bounded_text(line.get("fit_status"), 200),
                    "reason": _bounded_text(line.get("reason"), 5_000),
                    "unit_price_value": line.get("unit_price_value"),
                    "unit_price_currency": _bounded_text(line.get("unit_price_currency"), 20),
                    "line_total_value": line.get("line_total_value"),
                    "line_total_currency": _bounded_text(line.get("line_total_currency"), 20),
                }
            )
    return {
        "title": _bounded_text(quote.get("title"), 2_000),
        "client_summary": _bounded_text(quote.get("client_summary"), 20_000),
        "coverage_summary": _bounded_text(quote.get("coverage_summary"), 20_000),
        "why_selected": _bounded_text(quote.get("why_selected"), 20_000),
        "completeness_status": _bounded_text(quote.get("completeness_status"), 200),
        "operational_status": _bounded_text(quote.get("operational_status"), 200),
        "total_price_value": quote.get("total_price_value"),
        "total_price_currency": _bounded_text(quote.get("total_price_currency"), 20),
        "lines": lines,
        "assumptions": _display_list(quote.get("assumptions")),
        "key_deviations": _display_list(quote.get("key_deviations")),
        "engineer_checks": _display_list(quote.get("engineer_checks")),
        "procurement_gaps": _display_list(quote.get("procurement_gaps")),
    }


def _display_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value[:100]:
        if isinstance(item, str):
            result.append(_bounded_text(item, 5_000))
        elif isinstance(item, Mapping):
            result.append(
                _bounded_text(
                    json.dumps(item, ensure_ascii=False, sort_keys=True, default=str),
                    5_000,
                )
            )
        else:
            result.append(_bounded_text(item, 5_000))
    return result


def _review_state(receipt: Mapping[str, Any] | None) -> str:
    if receipt is None:
        return "pending"
    decision = str(receipt.get("decision") or "")
    return decision if decision in {"accepted", "rejected"} else "invalid"


def _validated_review_receipt(bundle: Path) -> ReviewReceipt | None:
    path = bundle / "review-receipt.json"
    if not path.exists():
        return None
    payload = _read_json_mapping(path)
    try:
        receipt = ReviewReceipt.model_validate(payload)
    except ValidationError as exc:
        details = [
            f"{'.'.join(str(part) for part in error['loc'])}:{error['type']}"
            for error in exc.errors(include_input=False)
        ]
        raise CaseExportError("review.receipt_invalid", details) from exc
    failures: list[str] = []
    manifest_path = bundle / "manifest.json"
    if sha256_file(manifest_path) != receipt.manifest_sha256:
        failures.append("review.manifest_hash_mismatch")
    for name, artifact in (
        ("annotation", receipt.annotation),
        ("golden_case", receipt.golden_case),
    ):
        artifact_path = (bundle / artifact.path).resolve()
        if not artifact_path.is_relative_to(bundle):
            failures.append(f"review.{name}.path_escape")
        elif not artifact_path.is_file():
            failures.append(f"review.{name}.missing")
        elif sha256_file(artifact_path) != artifact.sha256:
            failures.append(f"review.{name}.hash_mismatch")
    if failures:
        raise CaseExportError("review.receipt_invalid", failures)
    return receipt


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 25_000_000:
            raise CaseExportError("input.file_too_large", [path.name])
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise CaseExportError("input.file_not_found", [path.name]) from exc
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CaseExportError("input.invalid_json", [path.name]) from exc
    if not isinstance(value, dict):
        raise CaseExportError("input.mapping_required", [path.name])
    return value


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _json_bytes(payload)
    if len(data) > 5_000_000:
        raise CaseExportError("review.file_too_large")
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name).resolve()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _write_append_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = _json_bytes(payload)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise CaseExportError("batch.receipt_exists", [path.name]) from exc


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    ).encode("utf-8")


def _expected_case_id(match_run_id: int, output_root: Path) -> str | None:
    prefix = f"prod-{match_run_id}-"
    matches = [item.name for item in output_root.glob(f"{prefix}*") if item.is_dir()]
    return sorted(matches)[0] if matches else None


def _new_batch_id() -> str:
    stamp = datetime.now(UTC).strftime("batch-%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _validate_safe_id(value: str, code: str) -> None:
    if not _SAFE_ID_RE.fullmatch(value):
        raise CaseExportError(code)


def _bounded_text(value: Any, limit: int = MAX_DISPLAY_TEXT) -> str:
    text = str(value or "")
    return text if len(text) <= limit else f"{text[:limit]}\n[truncated]"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return [item for item in dict.fromkeys(str(item or "").strip() for item in value) if item]


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.utcoffset() is None else value


__all__ = [
    "BatchExportReceipt",
    "CandidateCatalog",
    "CatalogCandidate",
    "LocalCaseStore",
    "MAX_BATCH_EXPORT",
    "MAX_CATALOG_LIMIT",
    "QualitySummary",
    "ReviewMutation",
    "export_case_batch",
    "list_match_run_catalog",
]
