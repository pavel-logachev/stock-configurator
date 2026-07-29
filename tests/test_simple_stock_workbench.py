from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.evaluation.simple_stock_workbench as workbench_module
from app.core.database import Base
from app.db.models import MatchRun
from app.evaluation.simple_stock_case_exporter import CaseExportError, write_case_bundle
from app.evaluation.simple_stock_contracts import GoldenDataset
from app.evaluation.simple_stock_workbench import (
    LocalCaseStore,
    ReviewMutation,
    export_case_batch,
    list_match_run_catalog,
)
from app.evaluation.simple_stock_workbench_app import create_workbench_app


class AsyncSessionAdapter:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def execute(self, statement: Any) -> Any:
        return self._session.execute(statement)


@pytest.fixture
def db_session() -> Generator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


def test_catalog_returns_bounded_safe_metadata_without_raw_payload(
    tmp_path: Path,
    db_session: Session,
) -> None:
    db_session.add_all(
        [
            _persisted_run(402, pipeline="other_pipeline", source_text="other raw secret"),
            _persisted_run(401, source_text="private customer secret"),
        ]
    )
    db_session.commit()
    baseline_path = _baseline_path(tmp_path)

    catalog = asyncio.run(
        list_match_run_catalog(
            AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            production_baseline_path=baseline_path,
            limit=1,
        )
    )

    assert len(catalog.candidates) == 1
    assert catalog.candidates[0].match_run_id == 401
    assert catalog.candidates[0].exportable is True
    rendered = catalog.model_dump_json()
    assert "private customer secret" not in rendered
    assert "raw_vendor_secret" not in rendered
    assert "source_text" not in rendered
    assert "report_json" not in rendered


def test_catalog_marks_stage_drift_without_exporting_raw_data(
    tmp_path: Path,
    db_session: Session,
) -> None:
    run = _persisted_run(401)
    run.report_json["diagnostics"]["composer_prompt_version"] = "simple_stock_composer_v69"
    db_session.add(run)
    db_session.commit()

    catalog = asyncio.run(
        list_match_run_catalog(
            AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            production_baseline_path=_baseline_path(tmp_path),
            limit=10,
        )
    )

    assert catalog.candidates[0].exportable is False
    assert catalog.candidates[0].blockers == [
        "pipeline.composer_prompt_version.mismatch"
    ]


def test_batch_export_is_bounded_and_writes_append_only_safe_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_export(_session: Any, *, match_run_id: int, **_kwargs: Any) -> Any:
        if match_run_id == 12:
            raise CaseExportError("run.pipeline_not_simple_stock", ["safe.blocker"])
        return SimpleNamespace(
            case_id=f"prod-{match_run_id}-treolan",
            golden_review_eligible=True,
            blockers=[],
        )

    monkeypatch.setattr(workbench_module, "export_case_from_session", fake_export)
    receipt = asyncio.run(
        export_case_batch(
            SimpleNamespace(),  # type: ignore[arg-type]
            match_run_ids=[11, 12],
            output_root=tmp_path / "local" / "cases",
            receipt_root=tmp_path / "local" / "batches",
            production_baseline_path=tmp_path / "baseline.json",
            dataset_path=tmp_path / "dataset.json",
            batch_id="batch-test-1",
        )
    )

    assert receipt.requested_count == 2
    assert receipt.exported_count == 1
    assert receipt.blocked_count == 1
    assert receipt.database_writes == 0
    assert receipt.llm_calls == 0
    receipt_path = tmp_path / "local" / "batches" / "batch-test-1.json"
    assert receipt_path.is_file()
    assert "safe.blocker" in receipt_path.read_text(encoding="utf-8")

    with pytest.raises(CaseExportError, match="batch.receipt_exists"):
        asyncio.run(
            export_case_batch(
                SimpleNamespace(),  # type: ignore[arg-type]
                match_run_ids=[11],
                output_root=tmp_path / "local" / "cases",
                receipt_root=tmp_path / "local" / "batches",
                production_baseline_path=tmp_path / "baseline.json",
                dataset_path=tmp_path / "dataset.json",
                batch_id="batch-test-1",
            )
        )

    with pytest.raises(CaseExportError, match="batch.match_run_ids_invalid"):
        asyncio.run(
            export_case_batch(
                SimpleNamespace(),  # type: ignore[arg-type]
                match_run_ids=list(range(1, 22)),
                output_root=tmp_path / "local" / "cases",
                production_baseline_path=tmp_path / "baseline.json",
                dataset_path=tmp_path / "dataset.json",
            )
        )


def test_local_case_store_saves_only_mutable_review_fields_and_summarizes(
    tmp_path: Path,
) -> None:
    case_root, dataset_path, manifest = _ready_case(tmp_path)
    store = LocalCaseStore(case_root=case_root, dataset_path=dataset_path)
    mutation = ReviewMutation.model_validate(_complete_review_payload())

    draft = store.save_review(manifest.case_id, mutation)

    assert draft.tags == manifest.tags
    assert draft.business_weight == manifest.business_weight
    assert draft.expectations == manifest.expectations
    assert draft.output_sha256 == manifest.artifacts.output.sha256
    assert not list((case_root / manifest.case_id).glob(".review.json.*"))
    summary = store.quality_summary()
    assert summary.total_cases == 1
    assert summary.pending_cases == 1
    assert summary.finalized_cases == 0
    assert summary.human_review_coverage == 0


def test_workbench_api_enforces_local_host_csrf_and_explicit_finalization(
    tmp_path: Path,
) -> None:
    case_root, dataset_path, manifest = _ready_case(tmp_path)
    app = create_workbench_app(
        case_root=case_root,
        dataset_path=dataset_path,
        csrf_token="test-csrf-token",
    )
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "Evaluation Workbench" in index.text
        assert 'content="test-csrf-token"' in index.text
        assert "default-src 'none'" in index.headers["content-security-policy"]
        assert index.headers["x-frame-options"] == "DENY"

        denied_host = client.get("/api/cases", headers={"host": "evil.example"})
        assert denied_host.status_code == 403
        assert denied_host.json()["error"] == "security.host_denied"

        denied_csrf = client.post(
            f"/api/cases/{manifest.case_id}/review",
            json=_complete_review_payload(),
        )
        assert denied_csrf.status_code == 403
        assert denied_csrf.json()["error"] == "security.csrf_denied"

        denied_origin = client.post(
            f"/api/cases/{manifest.case_id}/review",
            json=_complete_review_payload(),
            headers={
                "x-workbench-csrf": "test-csrf-token",
                "origin": "https://evil.example",
            },
        )
        assert denied_origin.status_code == 403
        assert denied_origin.json()["error"] == "security.origin_denied"

        saved = client.post(
            f"/api/cases/{manifest.case_id}/review",
            json=_complete_review_payload(),
            headers={"x-workbench-csrf": "test-csrf-token"},
        )
        assert saved.status_code == 200
        assert saved.json()["status"] == "saved"

        mismatch = client.post(
            f"/api/cases/{manifest.case_id}/finalize",
            json={"confirm_case_id": "prod-other"},
            headers={"x-workbench-csrf": "test-csrf-token"},
        )
        assert mismatch.status_code == 422
        assert mismatch.json()["error"] == "review.confirmation_mismatch"

        finalized = client.post(
            f"/api/cases/{manifest.case_id}/finalize",
            json={"confirm_case_id": manifest.case_id},
            headers={"x-workbench-csrf": "test-csrf-token"},
        )
        assert finalized.status_code == 200
        assert finalized.json()["receipt"]["decision"] == "accepted"
        summary = client.get("/api/summary").json()
        assert summary["finalized_cases"] == 1
        assert summary["human_review_coverage"] == 1

        annotation_path = case_root / manifest.case_id / "annotation.json"
        annotation_path.write_text("{}\n", encoding="utf-8")
        tampered_summary = client.get("/api/summary").json()
        assert tampered_summary["invalid_cases"] == 1
        assert tampered_summary["finalized_cases"] == 0
        assert tampered_summary["human_review_coverage"] == 0


def test_tampered_bundle_hides_sensitive_content_and_disables_write(
    tmp_path: Path,
) -> None:
    case_root, dataset_path, manifest = _ready_case(
        tmp_path,
        source_text='<img src=x onerror="alert(1)">private',
    )
    bundle = case_root / manifest.case_id
    (bundle / "output.json").write_text("{}\n", encoding="utf-8")
    store = LocalCaseStore(case_root=case_root, dataset_path=dataset_path)

    detail = store.case_detail(manifest.case_id)

    assert detail["integrity_state"] == "invalid"
    assert detail["source"] is None
    assert detail["quote"] is None
    with pytest.raises(CaseExportError, match="bundle.invalid"):
        store.save_review(
            manifest.case_id,
            ReviewMutation.model_validate(_complete_review_payload()),
        )


def test_workbench_frontend_uses_text_content_and_has_no_external_assets() -> None:
    root = Path("app/evaluation/workbench_static")
    html = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "workbench.js").read_text(encoding="utf-8")
    css = (root / "workbench.css").read_text(encoding="utf-8")

    assert 'href="http://' not in html
    assert 'href="https://' not in html
    assert 'src="http://' not in html
    assert 'src="https://' not in html
    assert "innerHTML" not in script
    assert "textContent" in script
    assert "prefers-reduced-motion" in css
    assert "@media (max-width: 430px)" in css


def _ready_case(
    tmp_path: Path,
    *,
    source_text: str = "private request alpha",
) -> tuple[Path, Path, Any]:
    matrix = _matrix_payload()
    match_run = _match_run(matrix, source_text=source_text)
    case_root = tmp_path / "local" / "cases"
    manifest = write_case_bundle(
        match_run=match_run,
        matrix_payload=matrix,
        matrix_rows=_matrix_rows(),
        output_root=case_root,
        baseline=_baseline(),
        dataset=_dataset(),
        case_id="prod-301-treolan",
    )
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(_dataset().model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return case_root, dataset_path, manifest


def _complete_review_payload() -> dict[str, Any]:
    return {
        "decision": "accept",
        "reviewer_role": "solution-engineer",
        "semantic_score": 0.95,
        "unsupported_material_claim_count": 0,
        "critical_error_codes": [],
        "business_weighted_loss": 0.0,
        "atomic_criteria": [
            {"criterion_id": criterion, "status": "pass", "evidence_refs": []}
            for criterion in (
                "request-role-coverage",
                "compatibility-and-enablement",
                "commercial-price-choice",
                "no-unsupported-claims",
            )
        ],
    }


def _persisted_run(
    run_id: int,
    *,
    pipeline: str = "simple_stock_quote",
    source_text: str = "request",
) -> MatchRun:
    matrix = _matrix_payload()
    report = _match_run(matrix, source_text=source_text).report_json
    report["pipeline_version"] = pipeline
    report["raw_vendor_payload"] = "raw_vendor_secret"
    return MatchRun(
        id=run_id,
        source="v3_full_category_text",
        source_text=source_text,
        status="completed",
        engineer_review_required=True,
        total_candidates=17,
        matched_items=8,
        missing_requirements_json=[],
        risk_flags_json=[],
        spec_json={"raw": "do-not-return"},
        report_json=report,
        report_markdown=None,
        created_at=datetime(2026, 7, 22, 12, run_id % 60, tzinfo=UTC),
    )


def _baseline_path(tmp_path: Path) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(_baseline()), encoding="utf-8")
    return path


def _baseline() -> dict[str, Any]:
    return {
        "production_commit": "551ea85b84d6a0926285532100ebb97baf163954",
        "pipeline_version": "simple_stock_quote",
        "stages": {
            "route_prompt_version": "simple_stock_route_v19",
            "matrix_schema_version": "simple_stock_matrix.v8",
            "composer_prompt_version": "simple_stock_composer_v70",
            "reconciler_version": "quote_integrity_reconciler_v9",
        },
        "llm": {"model": "qwen/qwen3.7-max", "max_package_chars": 5_000_000},
    }


def _dataset() -> GoldenDataset:
    payload = json.loads(
        Path("evaluation/simple_stock/v1/dataset.json").read_text(encoding="utf-8-sig")
    )
    return GoldenDataset.model_validate(payload)


def _match_run(matrix: dict[str, Any], *, source_text: str = "private request alpha") -> Any:
    diagnostics = matrix["diagnostics"]
    report = {
        "match_run_id": 301,
        "pipeline_version": "simple_stock_quote",
        "v3_result_state": "quote_draft_review_required",
        "final_status_source": "simple_stock_quote_llm_accepted",
        "distributor_code": "treolan",
        "category_ids": ["cat-a"],
        "simple_route_decision": {"prompt_version": "simple_stock_route_v19"},
        "validated_quote": {
            "title": "Тестовая конфигурация",
            "client_summary": "Результат для ручной проверки.",
            "coverage_summary": "Покрыта одна роль.",
            "total_price_value": "100",
            "total_price_currency": "USD",
            "engineering_review_required": True,
            "assumptions": ["Проверить совместимость"],
            "engineer_checks": ["Сверить память"],
            "procurement_gaps": [],
            "key_deviations": [],
            "lines": [
                {
                    "component_candidate_id": "treolan:item-secret",
                    "part_number": "PN-1",
                    "item_name": "Product",
                    "quantity": 1,
                    "line_total_value": "100",
                    "line_total_currency": "USD",
                }
            ],
        },
        "v3_validation_errors": [],
        "v3_validation_warnings": [],
        "v3_validation_error_details": [],
        "diagnostics": {
            "matrix_schema_version": matrix["schema_version"],
            "matrix_row_count": diagnostics["row_count"],
            "matrix_position_count": diagnostics["position_count"],
            "matrix_char_count": diagnostics["char_count"],
            "matrix_status": diagnostics["status"],
            "model": matrix["model"],
            "composer_prompt_version": "simple_stock_composer_v70",
            "quote_integrity_reconciler": "quote_integrity_reconciler_v9",
        },
    }
    return SimpleNamespace(
        id=301,
        source="v3_full_category_text",
        source_text=source_text,
        spec_json={"source_text": source_text},
        report_json=report,
        created_at=datetime(2026, 7, 2, 20, 54, 11, tzinfo=UTC),
    )


def _matrix_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "simple_stock_matrix.v8",
        "distributor_code": "treolan",
        "category_ids": ["cat-a"],
        "model": "qwen/qwen3.7-max",
        "category_sections": [],
        "diagnostics": {
            "row_count": 1,
            "position_count": 1,
            "char_count": 0,
            "status": "ready_for_llm",
        },
    }
    while True:
        char_count = len(
            json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
        )
        if payload["diagnostics"]["char_count"] == char_count:
            return payload
        payload["diagnostics"]["char_count"] = char_count


def _matrix_rows() -> list[Any]:
    return [
        SimpleNamespace(
            stock=SimpleNamespace(synced_at=datetime(2026, 7, 2, 19, 0, tzinfo=UTC))
        )
    ]
