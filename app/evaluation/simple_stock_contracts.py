from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DATASET_SCHEMA_VERSION = "simple-stock-golden-dataset.v1"
RUN_SCHEMA_VERSION = "simple-stock-eval-run.v1"
ANNOTATION_SCHEMA_VERSION = "simple-stock-human-annotation.v1"
BLIND_REVIEW_SCHEMA_VERSION = "simple-stock-blind-review.v1"
REPORT_SCHEMA_VERSION = "simple-stock-eval-report.v1"
EVALUATOR_VERSION = "simple-stock-offline-evaluator.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SAFE_SOURCE_REF_RE = re.compile(r"^(?:production|local|synthetic)://[A-Za-z0-9._/-]+$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourcePointer(StrictModel):
    ref: str = Field(min_length=1, max_length=500)
    sha256: str | None = None
    privacy_class: Literal["internal", "sensitive_local_only"]

    @field_validator("ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        if not _SAFE_SOURCE_REF_RE.fullmatch(value):
            raise ValueError("source ref must be a safe opaque production/local/synthetic URI")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class ArtifactFile(StrictModel):
    path: str = Field(min_length=1, max_length=500)
    sha256: str

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        if "\\" in value or ":" in value:
            raise ValueError("artifact path must use forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must stay inside the run bundle directory")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value


class GoldenExpectations(StrictModel):
    allowed_result_states: list[str] = Field(min_length=1)
    required_pipeline_version: str = Field(min_length=1)
    allowed_final_status_sources: list[str] = Field(default_factory=list)
    minimum_quote_lines: int = Field(default=0, ge=0)
    maximum_validation_errors: int = Field(default=0, ge=0)
    maximum_validation_warnings: int = Field(default=0, ge=0)
    require_engineering_review: bool = True
    required_atomic_criteria: list[str] = Field(default_factory=list)

    @field_validator("allowed_result_states", "allowed_final_status_sources")
    @classmethod
    def unique_non_empty_values(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("values must be non-empty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("values must be unique")
        return cleaned

    @field_validator("required_atomic_criteria")
    @classmethod
    def validate_criterion_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("criterion ids must be unique")
        for value in values:
            if not _SAFE_ID_RE.fullmatch(value):
                raise ValueError("criterion ids must use lowercase safe identifiers")
        return values


class GoldenCase(StrictModel):
    case_id: str
    status: Literal["pending_review", "accepted", "rejected"]
    tags: list[str] = Field(min_length=1)
    business_weight: float = Field(ge=0.1, le=100)
    source: SourcePointer
    matrix_source: SourcePointer | None = None
    expectations: GoldenExpectations
    reviewed_by_role: str | None = None
    reviewed_at: datetime | None = None

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("case_id must use lowercase safe identifiers")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("tags must be unique")
        for value in values:
            if not _SAFE_ID_RE.fullmatch(value):
                raise ValueError("tags must use lowercase safe identifiers")
        return values

    @model_validator(mode="after")
    def accepted_case_is_hash_bound(self) -> GoldenCase:
        if self.status != "accepted":
            return self
        if self.source.sha256 is None:
            raise ValueError("accepted case requires source sha256")
        if self.matrix_source is None or self.matrix_source.sha256 is None:
            raise ValueError("accepted case requires hash-bound matrix_source")
        if not self.reviewed_by_role or self.reviewed_at is None:
            raise ValueError("accepted case requires reviewer role and timestamp")
        if self.reviewed_at.utcoffset() is None:
            raise ValueError("accepted case review timestamp must include timezone")
        return self


class EvaluationThresholds(StrictModel):
    minimum_semantic_score: float = Field(default=0.8, ge=0, le=1)
    structured_valid_rate: float = Field(default=1.0, ge=0, le=1)
    grounded_product_id_rate: float = Field(default=1.0, ge=0, le=1)
    critical_error_count: int = Field(default=0, ge=0)
    unsupported_material_claim_count: int = Field(default=0, ge=0)
    maximum_business_weighted_loss: float = Field(default=0.2, ge=0)
    maximum_latency_regression_ratio: float = Field(default=1.25, ge=1)
    maximum_cost_regression_ratio: float = Field(default=1.25, ge=1)


class GoldenDataset(StrictModel):
    schema_version: Literal[DATASET_SCHEMA_VERSION]
    dataset_id: str
    status: Literal["bootstrap", "accepted", "retired"]
    privacy_class: Literal["internal", "sensitive_local_only"]
    production_baseline_ref: str
    minimum_accepted_cases: int = Field(ge=1)
    thresholds: EvaluationThresholds
    cases: list[GoldenCase]

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("dataset_id must use lowercase safe identifiers")
        return value

    @model_validator(mode="after")
    def dataset_has_unique_cases(self) -> GoldenDataset:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("dataset case ids must be unique")
        if self.status == "accepted":
            accepted_count = sum(case.status == "accepted" for case in self.cases)
            if accepted_count < self.minimum_accepted_cases:
                raise ValueError("accepted dataset has too few accepted cases")
        return self


class PipelineStages(StrictModel):
    route_prompt_version: str
    matrix_schema_version: str
    composer_prompt_version: str
    reconciler_version: str


class RunBindings(StrictModel):
    dataset_sha256: str
    production_baseline_sha256: str
    code_revision: str = Field(min_length=7, max_length=64)
    pipeline_version: str
    stages: PipelineStages
    model_id: str
    model_settings_artifact: ArtifactFile
    prompt_bundle: ArtifactFile
    evaluator_version: Literal[EVALUATOR_VERSION]

    @field_validator("dataset_sha256", "production_baseline_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("binding hashes must be 64 lowercase hexadecimal characters")
        return value


class EvaluationCaseRun(StrictModel):
    case_id: str
    output: ArtifactFile
    matrix: ArtifactFile
    annotation: ArtifactFile | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    retries: int = Field(default=0, ge=0)
    fallback_path: str | None = Field(default=None, max_length=200)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("case_id must use lowercase safe identifiers")
        return value


class EvaluationRun(StrictModel):
    schema_version: Literal[RUN_SCHEMA_VERSION]
    run_id: str
    candidate_label: str = Field(min_length=1, max_length=100)
    bindings: RunBindings
    cases: list[EvaluationCaseRun] = Field(min_length=1)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("run_id must use lowercase safe identifiers")
        return value

    @model_validator(mode="after")
    def run_has_unique_cases(self) -> EvaluationRun:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("run case ids must be unique")
        return self


class AtomicCriterionAnnotation(StrictModel):
    criterion_id: str
    status: Literal["pass", "fail", "not_applicable"]
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("criterion_id must use lowercase safe identifiers")
        return value


class HumanAnnotation(StrictModel):
    schema_version: Literal[ANNOTATION_SCHEMA_VERSION]
    case_id: str
    output_sha256: str
    reviewer_role: str = Field(min_length=1, max_length=100)
    reviewed_at: datetime
    semantic_score: float = Field(ge=0, le=1)
    unsupported_material_claim_count: int = Field(ge=0)
    critical_error_codes: list[str] = Field(default_factory=list)
    business_weighted_loss: float = Field(ge=0)
    atomic_criteria: list[AtomicCriterionAnnotation] = Field(default_factory=list)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("case_id must use lowercase safe identifiers")
        return value

    @field_validator("critical_error_codes")
    @classmethod
    def validate_critical_error_codes(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("critical error codes must be unique")
        for value in values:
            if not _SAFE_ID_RE.fullmatch(value):
                raise ValueError("critical error codes must use lowercase safe identifiers")
        return values

    @field_validator("output_sha256")
    @classmethod
    def validate_output_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("output_sha256 must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("review timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def annotation_has_unique_criteria(self) -> HumanAnnotation:
        criterion_ids = [item.criterion_id for item in self.atomic_criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("annotation criterion ids must be unique")
        return self


class BlindCaseDecision(StrictModel):
    case_id: str
    decision: Literal["baseline", "candidate", "tie", "reject_both"]
    reason_codes: list[str] = Field(min_length=1)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str) -> str:
        if not _SAFE_ID_RE.fullmatch(value):
            raise ValueError("case_id must use lowercase safe identifiers")
        return value

    @field_validator("reason_codes")
    @classmethod
    def validate_reason_codes(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("reason codes must be unique")
        for value in values:
            if not _SAFE_ID_RE.fullmatch(value):
                raise ValueError("reason codes must use lowercase safe identifiers")
        return values


class BlindReview(StrictModel):
    schema_version: Literal[BLIND_REVIEW_SCHEMA_VERSION]
    dataset_sha256: str
    baseline_run_sha256: str
    candidate_run_sha256: str
    reviewer_role: str = Field(min_length=1, max_length=100)
    reviewed_at: datetime
    decisions: list[BlindCaseDecision] = Field(min_length=1)

    @field_validator("dataset_sha256", "baseline_run_sha256", "candidate_run_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("review hashes must be 64 lowercase hexadecimal characters")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("review timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def review_has_unique_cases(self) -> BlindReview:
        case_ids = [decision.case_id for decision in self.decisions]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("blind review case ids must be unique")
        return self
