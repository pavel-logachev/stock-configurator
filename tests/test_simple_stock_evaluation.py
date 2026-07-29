from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.cli.evaluate_simple_stock_pipeline import run as run_cli
from app.evaluation.simple_stock_contracts import GoldenDataset
from app.evaluation.simple_stock_evaluator import (
    evaluate_simple_stock_runs,
    load_model,
    sha256_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_BASELINE_PATH = PROJECT_ROOT / "config" / "production_pipeline_baseline.json"
CRITERIA = [
    "request-role-coverage",
    "compatibility-and-enablement",
    "commercial-price-choice",
    "no-unsupported-claims",
]


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _output(*, candidate_id: str, composer_version: str) -> dict[str, Any]:
    return {
        "pipeline_version": "simple_stock_quote",
        "primary_recommendation_status": "llm_final",
        "final_status_source": "simple_stock_quote_llm_accepted",
        "v3_result_state": "quote_draft_review_required",
        "v3_validation_errors": [],
        "v3_validation_warnings": [],
        "validated_quote": {
            "engineering_review_required": True,
            "lines": [{"component_candidate_id": candidate_id}],
            "quote_integrity": {"version": "quote_integrity_reconciler_v9"},
        },
        "diagnostics": {
            "route_prompt_version": "simple_stock_route_v19",
            "matrix_schema_version": "simple_stock_matrix.v8",
            "composer_prompt_version": composer_version,
        },
    }


def _annotation(*, output_sha256: str, semantic_score: float = 0.95) -> dict[str, Any]:
    return {
        "schema_version": "simple-stock-human-annotation.v1",
        "case_id": "synthetic-case",
        "output_sha256": output_sha256,
        "reviewer_role": "test-reviewer",
        "reviewed_at": "2026-07-22T12:00:00Z",
        "semantic_score": semantic_score,
        "unsupported_material_claim_count": 0,
        "critical_error_codes": [],
        "business_weighted_loss": 0.05,
        "atomic_criteria": [
            {
                "criterion_id": criterion_id,
                "status": "pass",
                "evidence_refs": [f"synthetic://evidence/{criterion_id}"],
            }
            for criterion_id in CRITERIA
        ],
    }


def _prepare_comparison(
    tmp_path: Path,
    *,
    candidate_id: str = "synthetic:p1",
    include_blind_review: bool = True,
) -> dict[str, Path | None]:
    matrix_path = tmp_path / "matrix.json"
    matrix_sha256 = _write_json(
        matrix_path,
        {
            "schema_version": "simple_stock_matrix.v8",
            "category_sections": [{"positions": [{"component_candidate_id": "synthetic:p1"}]}],
        },
    )
    dataset_path = tmp_path / "dataset.json"
    _write_json(
        dataset_path,
        {
            "schema_version": "simple-stock-golden-dataset.v1",
            "dataset_id": "synthetic-golden-v1",
            "status": "accepted",
            "privacy_class": "internal",
            "production_baseline_ref": "config/production_pipeline_baseline.json",
            "minimum_accepted_cases": 1,
            "thresholds": {
                "minimum_semantic_score": 0.8,
                "structured_valid_rate": 1.0,
                "grounded_product_id_rate": 1.0,
                "critical_error_count": 0,
                "unsupported_material_claim_count": 0,
                "maximum_business_weighted_loss": 0.2,
                "maximum_latency_regression_ratio": 1.25,
                "maximum_cost_regression_ratio": 1.25,
            },
            "cases": [
                {
                    "case_id": "synthetic-case",
                    "status": "accepted",
                    "tags": ["synthetic", "normal"],
                    "business_weight": 5,
                    "source": {
                        "ref": "synthetic://requests/synthetic-case",
                        "sha256": _hash_text("synthetic request"),
                        "privacy_class": "internal",
                    },
                    "matrix_source": {
                        "ref": "synthetic://matrices/synthetic-case",
                        "sha256": matrix_sha256,
                        "privacy_class": "internal",
                    },
                    "expectations": {
                        "allowed_result_states": ["quote_draft_review_required"],
                        "required_pipeline_version": "simple_stock_quote",
                        "allowed_final_status_sources": ["simple_stock_quote_llm_accepted"],
                        "minimum_quote_lines": 1,
                        "maximum_validation_errors": 0,
                        "maximum_validation_warnings": 0,
                        "require_engineering_review": True,
                        "required_atomic_criteria": CRITERIA,
                    },
                    "reviewed_by_role": "test-reviewer",
                    "reviewed_at": "2026-07-22T12:00:00Z",
                }
            ],
        },
    )
    dataset_sha256 = sha256_file(dataset_path)
    production_baseline_sha256 = sha256_file(PRODUCTION_BASELINE_PATH)
    model_config_path = tmp_path / "model-config.json"
    model_config_sha256 = _write_json(
        model_config_path,
        {"model": "example/model", "thinking_enabled": False},
    )
    baseline_prompt_path = tmp_path / "baseline-prompt.txt"
    baseline_prompt_path.write_text("synthetic baseline prompt", encoding="utf-8")
    baseline_prompt_sha256 = sha256_file(baseline_prompt_path)
    candidate_prompt_path = tmp_path / "candidate-prompt.txt"
    candidate_prompt_path.write_text("synthetic candidate prompt", encoding="utf-8")
    candidate_prompt_sha256 = sha256_file(candidate_prompt_path)

    baseline_output_path = tmp_path / "baseline-output.json"
    baseline_output_sha256 = _write_json(
        baseline_output_path,
        _output(candidate_id="synthetic:p1", composer_version="simple_stock_composer_v70"),
    )
    candidate_output_path = tmp_path / "candidate-output.json"
    candidate_output_sha256 = _write_json(
        candidate_output_path,
        _output(candidate_id=candidate_id, composer_version="simple_stock_composer_v71"),
    )
    baseline_annotation_path = tmp_path / "baseline-annotation.json"
    baseline_annotation_sha256 = _write_json(
        baseline_annotation_path,
        _annotation(output_sha256=baseline_output_sha256),
    )
    candidate_annotation_path = tmp_path / "candidate-annotation.json"
    candidate_annotation_sha256 = _write_json(
        candidate_annotation_path,
        _annotation(output_sha256=candidate_output_sha256),
    )

    def run_payload(*, candidate: bool) -> dict[str, Any]:
        prefix = "candidate" if candidate else "baseline"
        composer_version = "simple_stock_composer_v71" if candidate else "simple_stock_composer_v70"
        return {
            "schema_version": "simple-stock-eval-run.v1",
            "run_id": f"{prefix}-run",
            "candidate_label": prefix,
            "bindings": {
                "dataset_sha256": dataset_sha256,
                "production_baseline_sha256": production_baseline_sha256,
                "code_revision": "abcdef1234567890",
                "pipeline_version": "simple_stock_quote",
                "stages": {
                    "route_prompt_version": "simple_stock_route_v19",
                    "matrix_schema_version": "simple_stock_matrix.v8",
                    "composer_prompt_version": composer_version,
                    "reconciler_version": "quote_integrity_reconciler_v9",
                },
                "model_id": "example/model",
                "model_settings_artifact": {
                    "path": "model-config.json",
                    "sha256": model_config_sha256,
                },
                "prompt_bundle": {
                    "path": f"{prefix}-prompt.txt",
                    "sha256": (candidate_prompt_sha256 if candidate else baseline_prompt_sha256),
                },
                "evaluator_version": "simple-stock-offline-evaluator.v1",
            },
            "cases": [
                {
                    "case_id": "synthetic-case",
                    "output": {
                        "path": f"{prefix}-output.json",
                        "sha256": (
                            candidate_output_sha256 if candidate else baseline_output_sha256
                        ),
                    },
                    "matrix": {"path": "matrix.json", "sha256": matrix_sha256},
                    "annotation": {
                        "path": f"{prefix}-annotation.json",
                        "sha256": (
                            candidate_annotation_sha256 if candidate else baseline_annotation_sha256
                        ),
                    },
                    "latency_ms": 1100 if candidate else 1000,
                    "cost_usd": 0.11 if candidate else 0.1,
                    "retries": 0,
                    "fallback_path": None,
                }
            ],
        }

    baseline_run_path = tmp_path / "baseline-run.json"
    candidate_run_path = tmp_path / "candidate-run.json"
    _write_json(baseline_run_path, run_payload(candidate=False))
    _write_json(candidate_run_path, run_payload(candidate=True))

    blind_review_path: Path | None = None
    if include_blind_review:
        blind_review_path = tmp_path / "blind-review.json"
        _write_json(
            blind_review_path,
            {
                "schema_version": "simple-stock-blind-review.v1",
                "dataset_sha256": dataset_sha256,
                "baseline_run_sha256": sha256_file(baseline_run_path),
                "candidate_run_sha256": sha256_file(candidate_run_path),
                "reviewer_role": "test-reviewer",
                "reviewed_at": "2026-07-22T12:30:00Z",
                "decisions": [
                    {
                        "case_id": "synthetic-case",
                        "decision": "tie",
                        "reason_codes": ["equivalent-quality"],
                    }
                ],
            },
        )
    return {
        "dataset": dataset_path,
        "baseline_run": baseline_run_path,
        "candidate_run": candidate_run_path,
        "blind_review": blind_review_path,
    }


def _evaluate(paths: dict[str, Path | None]) -> dict[str, Any]:
    return evaluate_simple_stock_runs(
        dataset_path=paths["dataset"],  # type: ignore[arg-type]
        baseline_run_path=paths["baseline_run"],  # type: ignore[arg-type]
        candidate_run_path=paths["candidate_run"],  # type: ignore[arg-type]
        production_baseline_path=PRODUCTION_BASELINE_PATH,
        blind_review_path=paths["blind_review"],
    )


def test_offline_evaluator_passes_hash_bound_single_axis_candidate(tmp_path: Path) -> None:
    report = _evaluate(_prepare_comparison(tmp_path))

    assert report["status"] == "passed"
    assert report["comparison"]["changed_axes"] == ["composer"]
    assert report["candidate"]["metrics"]["grounded_product_id_rate"] == 1.0
    assert report["candidate"]["metrics"]["human_review_rate"] == 1.0
    assert len(report["candidate"]["cases"][0]["output_sha256"]) == 64
    assert len(report["candidate"]["cases"][0]["annotation_sha256"]) == 64
    assert report["failures"] == []
    assert report["blockers"] == []
    assert report["privacy"] == {
        "raw_prompts_in_report": False,
        "raw_outputs_in_report": False,
        "product_ids_in_report": False,
    }


def test_offline_evaluator_fails_unknown_product_identifier(tmp_path: Path) -> None:
    report = _evaluate(_prepare_comparison(tmp_path, candidate_id="synthetic:unknown"))

    assert report["status"] == "failed"
    assert "candidate.output.unknown_component_candidate_id" in report["failures"]
    assert "synthetic:unknown" not in json.dumps(report)


def test_offline_evaluator_blocks_without_blind_review(tmp_path: Path) -> None:
    report = _evaluate(_prepare_comparison(tmp_path, include_blind_review=False))

    assert report["status"] == "blocked"
    assert "blind_review.missing" in report["blockers"]


def test_offline_evaluator_rejects_multiple_semantic_change_axes(tmp_path: Path) -> None:
    paths = _prepare_comparison(tmp_path)
    candidate_run_path = paths["candidate_run"]
    assert isinstance(candidate_run_path, Path)
    candidate_run = json.loads(candidate_run_path.read_text(encoding="utf-8"))
    candidate_run["bindings"]["model_id"] = "synthetic/other-model"
    _write_json(candidate_run_path, candidate_run)

    report = _evaluate(paths)

    assert report["status"] == "failed"
    assert "candidate.multiple_change_axes" in report["failures"]


def test_offline_evaluator_rejects_changed_binding_artifact(tmp_path: Path) -> None:
    paths = _prepare_comparison(tmp_path)
    (tmp_path / "candidate-prompt.txt").write_text(
        "changed after run bundle was created",
        encoding="utf-8",
    )

    report = _evaluate(paths)

    assert report["status"] == "failed"
    assert "candidate.prompt_bundle_hash_mismatch" in report["failures"]


def test_offline_evaluator_fails_closed_on_malformed_validation_fields(
    tmp_path: Path,
) -> None:
    paths = _prepare_comparison(tmp_path)
    candidate_output_path = tmp_path / "candidate-output.json"
    candidate_output = json.loads(candidate_output_path.read_text(encoding="utf-8"))
    candidate_output["v3_validation_errors"] = "not-a-list"
    candidate_output_sha256 = _write_json(candidate_output_path, candidate_output)

    annotation_path = tmp_path / "candidate-annotation.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["output_sha256"] = candidate_output_sha256
    annotation_sha256 = _write_json(annotation_path, annotation)

    candidate_run_path = paths["candidate_run"]
    assert isinstance(candidate_run_path, Path)
    candidate_run = json.loads(candidate_run_path.read_text(encoding="utf-8"))
    candidate_case = candidate_run["cases"][0]
    candidate_case["output"]["sha256"] = candidate_output_sha256
    candidate_case["annotation"]["sha256"] = annotation_sha256
    _write_json(candidate_run_path, candidate_run)

    report = _evaluate(paths)

    assert report["status"] == "failed"
    assert "candidate.output.validation_errors_invalid" in report["failures"]


def test_bootstrap_dataset_is_explicitly_not_release_ready() -> None:
    dataset = load_model(
        PROJECT_ROOT / "evaluation" / "simple_stock" / "v1" / "dataset.json",
        GoldenDataset,
    )

    assert dataset.status == "bootstrap"
    assert dataset.minimum_accepted_cases == 20
    assert [case.case_id for case in dataset.cases] == ["synthetic-server-basic"]
    assert dataset.cases[0].status == "pending_review"
    assert dataset.cases[0].source.ref == "synthetic://requests/server-basic"
    assert dataset.cases[0].source.sha256 == (
        "da457a9f0290a3b6c56f66092e14ed275d5857aff6ec2e8679220a506910b83d"
    )


def test_cli_blocks_invalid_schema_without_echoing_raw_input(
    tmp_path: Path,
    capsys: Any,
) -> None:
    invalid_dataset = tmp_path / "invalid-dataset.json"
    _write_json(
        invalid_dataset,
        {
            "schema_version": "simple-stock-golden-dataset.v1",
            "request_text": "DO_NOT_ECHO_PRIVATE_CUSTOMER_TEXT",
        },
    )

    exit_code = run_cli(
        [
            "--dataset",
            str(invalid_dataset),
            "--baseline-run",
            str(tmp_path / "missing-baseline.json"),
            "--candidate-run",
            str(tmp_path / "missing-candidate.json"),
            "--production-baseline",
            str(PRODUCTION_BASELINE_PATH),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert '"status": "blocked"' in captured.out
    assert "DO_NOT_ECHO_PRIVATE_CUSTOMER_TEXT" not in captured.out


def test_cli_writes_sanitized_pass_report(tmp_path: Path, capsys: Any) -> None:
    paths = _prepare_comparison(tmp_path)
    report_path = tmp_path / "report.json"

    exit_code = run_cli(
        [
            "--dataset",
            str(paths["dataset"]),
            "--baseline-run",
            str(paths["baseline_run"]),
            "--candidate-run",
            str(paths["candidate_run"]),
            "--production-baseline",
            str(PRODUCTION_BASELINE_PATH),
            "--blind-review",
            str(paths["blind_review"]),
            "--output",
            str(report_path),
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["status"] == "passed"
    assert report["privacy"]["raw_outputs_in_report"] is False
    assert "synthetic:p1" not in captured.out
    assert "synthetic:p1" not in report_path.read_text(encoding="utf-8")
