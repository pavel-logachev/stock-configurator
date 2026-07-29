from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.evaluation.simple_stock_case_exporter as exporter_module
from app.core.database import Base
from app.db.models import DistributorProduct, DistributorStockPrice
from app.evaluation.simple_stock_case_exporter import (
    CASE_EXPORT_SCHEMA_VERSION,
    CaseExportError,
    _list_matrix_rows_as_of,
    enforce_postgresql_read_only_transaction,
    finalize_case_review,
    write_case_bundle,
)
from app.evaluation.simple_stock_contracts import GoldenDataset


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


def test_write_case_bundle_is_hash_bound_and_excludes_raw_data_from_manifest(
    tmp_path: Path,
) -> None:
    matrix = _matrix_payload()
    match_run = _match_run(matrix)
    manifest = write_case_bundle(
        match_run=match_run,
        matrix_payload=matrix,
        matrix_rows=_matrix_rows(),
        output_root=tmp_path / "local",
        baseline=_baseline(),
        dataset=_dataset(),
        case_id="prod-301-treolan",
        exported_at=datetime(2026, 7, 22, 15, 0, tzinfo=UTC),
    )

    bundle = tmp_path / "local" / "prod-301-treolan"
    assert manifest.schema_version == CASE_EXPORT_SCHEMA_VERSION
    assert manifest.golden_review_eligible is True
    assert manifest.blockers == []
    assert set(manifest.matrix_evidence.matched_diagnostics) == {
        "matrix.schema_version",
        "matrix.row_count",
        "matrix.position_count",
        "matrix.char_count",
        "matrix.status",
        "matrix.model",
        "matrix.distributor_code",
        "matrix.category_ids",
    }
    for name, artifact in (
        ("source.json", manifest.artifacts.source),
        ("output.json", manifest.artifacts.output),
        ("matrix.json", manifest.artifacts.matrix),
        ("review.json", manifest.artifacts.review_draft),
    ):
        path = bundle / name
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256

    manifest_text = (bundle / "manifest.json").read_text(encoding="utf-8")
    review_text = (bundle / "review.json").read_text(encoding="utf-8")
    assert "private request alpha" not in manifest_text
    assert "treolan:item-secret" not in manifest_text
    assert "private request alpha" not in review_text
    assert "treolan:item-secret" not in review_text


def test_write_case_bundle_blocks_review_when_matrix_diagnostics_do_not_match(
    tmp_path: Path,
) -> None:
    matrix = _matrix_payload()
    match_run = _match_run(matrix)
    match_run.report_json["diagnostics"]["matrix_char_count"] += 1

    manifest = write_case_bundle(
        match_run=match_run,
        matrix_payload=matrix,
        matrix_rows=_matrix_rows(),
        output_root=tmp_path / "local",
        baseline=_baseline(),
        dataset=_dataset(),
        case_id="prod-301-treolan",
    )

    assert manifest.golden_review_eligible is False
    assert manifest.blockers == ["matrix.char_count.mismatch"]


def test_write_case_bundle_blocks_review_when_stage_evidence_is_missing(
    tmp_path: Path,
) -> None:
    matrix = _matrix_payload()
    match_run = _match_run(matrix)
    del match_run.report_json["diagnostics"]["composer_prompt_version"]

    manifest = write_case_bundle(
        match_run=match_run,
        matrix_payload=matrix,
        matrix_rows=_matrix_rows(),
        output_root=tmp_path / "local",
        baseline=_baseline(),
        dataset=_dataset(),
        case_id="prod-301-treolan",
    )

    assert manifest.golden_review_eligible is False
    assert manifest.blockers == ["pipeline.composer_prompt_version.missing"]


def test_write_case_bundle_rejects_path_escape_and_duplicate_case(tmp_path: Path) -> None:
    matrix = _matrix_payload()
    match_run = _match_run(matrix)
    kwargs = {
        "match_run": match_run,
        "matrix_payload": matrix,
        "matrix_rows": _matrix_rows(),
        "output_root": tmp_path / "local",
        "baseline": _baseline(),
        "dataset": _dataset(),
    }
    with pytest.raises(CaseExportError, match="case_id.invalid"):
        write_case_bundle(**kwargs, case_id="../escape")

    write_case_bundle(**kwargs, case_id="prod-301-treolan")
    manifest_path = tmp_path / "local" / "prod-301-treolan" / "manifest.json"
    original_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(CaseExportError, match="output.bundle_exists"):
        write_case_bundle(**kwargs, case_id="prod-301-treolan")
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == original_hash


def test_finalize_review_materializes_annotation_and_golden_case(tmp_path: Path) -> None:
    bundle, manifest = _export_ready_bundle(tmp_path)
    review_path = bundle / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review.update(
        {
            "decision": "accept",
            "reviewer_role": "solution-engineer",
            "reviewed_at": "2026-07-22T18:00:00+03:00",
            "semantic_score": 0.95,
            "unsupported_material_claim_count": 0,
            "business_weighted_loss": 0.0,
        }
    )
    for criterion in review["atomic_criteria"]:
        criterion["status"] = "pass"
        criterion["evidence_refs"] = [
            f"local://evidence/{criterion['criterion_id']}"
        ]
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    receipt = finalize_case_review(
        bundle_dir=bundle,
        dataset_path=_dataset_path(tmp_path),
        finalized_at=datetime(2026, 7, 22, 15, 1, tzinfo=UTC),
    )

    assert receipt.decision == "accepted"
    annotation = json.loads((bundle / "annotation.json").read_text(encoding="utf-8"))
    golden_case = json.loads((bundle / "golden-case.json").read_text(encoding="utf-8"))
    assert annotation["output_sha256"] == manifest.artifacts.output.sha256
    assert golden_case["status"] == "accepted"
    assert golden_case["source"]["sha256"] == manifest.artifacts.source.sha256
    assert golden_case["matrix_source"]["sha256"] == manifest.artifacts.matrix.sha256


def test_finalize_review_fails_closed_for_ineligible_bundle(tmp_path: Path) -> None:
    matrix = _matrix_payload()
    match_run = _match_run(matrix)
    match_run.report_json["diagnostics"]["matrix_row_count"] += 1
    manifest = write_case_bundle(
        match_run=match_run,
        matrix_payload=matrix,
        matrix_rows=_matrix_rows(),
        output_root=tmp_path / "local",
        baseline=_baseline(),
        dataset=_dataset(),
        case_id="prod-301-treolan",
    )
    bundle = tmp_path / "local" / manifest.case_id
    _complete_review(bundle / "review.json")

    with pytest.raises(CaseExportError) as exc_info:
        finalize_case_review(bundle_dir=bundle, dataset_path=_dataset_path(tmp_path))
    assert "review.bundle_not_eligible" in exc_info.value.details
    assert not (bundle / "annotation.json").exists()


def test_finalize_review_detects_tampered_output(tmp_path: Path) -> None:
    bundle, _manifest = _export_ready_bundle(tmp_path)
    _complete_review(bundle / "review.json")
    (bundle / "output.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(CaseExportError) as exc_info:
        finalize_case_review(bundle_dir=bundle, dataset_path=_dataset_path(tmp_path))
    assert "artifact.output.hash_mismatch" in exc_info.value.details
    assert not (bundle / "annotation.json").exists()


def test_finalize_review_rejects_changed_evaluation_contract(tmp_path: Path) -> None:
    bundle, _manifest = _export_ready_bundle(tmp_path)
    _complete_review(bundle / "review.json")
    review_path = bundle / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["expectations"]["minimum_quote_lines"] = 0
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CaseExportError) as exc_info:
        finalize_case_review(bundle_dir=bundle, dataset_path=_dataset_path(tmp_path))
    assert "review.contract_changed" in exc_info.value.details
    assert not (bundle / "annotation.json").exists()


def test_matrix_query_uses_latest_snapshot_not_newer_than_match_run(
    db_session: Session,
) -> None:
    old = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)
    future = old + timedelta(hours=2)
    db_session.add(
        DistributorProduct(
            distributor_code="treolan",
            item_id="item-1",
            product_key="key-1",
            part_number="PN-1",
            producer="Vendor",
            category_id="cat-a",
            item_name="Product",
            item_name_rus=None,
            product_name=None,
            product_description=None,
            product_notes=None,
            hscode=None,
            ean=None,
            is_in_mpt_registry=None,
            is_project_item=None,
            traceable=None,
            condition=None,
            warranty=None,
            original_country_iso_code=None,
            vat_percent=None,
            serial_number_availability=None,
            catalog_path_json=[],
            package_json={},
            raw_json={},
            synced_at=old,
            created_at=old - timedelta(days=1),
            updated_at=old,
        )
    )
    db_session.flush()
    old_row = _stock_row("treolan", "item-1", old, quantity=1)
    future_row = _stock_row("treolan", "item-1", future, quantity=99)
    db_session.add_all([old_row, future_row])
    db_session.commit()

    rows = asyncio.run(
        _list_matrix_rows_as_of(
            AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            distributor_code="treolan",
            category_ids=["cat-a"],
            as_of=old + timedelta(hours=1),
        )
    )

    assert len(rows) == 1
    assert rows[0].stock.quantity_value == 1
    assert rows[0].stock.synced_at.replace(tzinfo=UTC) == old


def test_read_only_guard_rejects_non_postgresql_before_query() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.executed = False

        def get_bind(self) -> Any:
            return SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        async def execute(self, _statement: Any) -> None:
            self.executed = True

    session = FakeSession()
    with pytest.raises(CaseExportError, match="database.postgresql_required"):
        asyncio.run(
            enforce_postgresql_read_only_transaction(session)  # type: ignore[arg-type]
        )
    assert session.executed is False


def test_read_only_guard_enables_postgresql_transaction_mode() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def get_bind(self) -> Any:
            return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        async def execute(self, statement: Any) -> None:
            self.statements.append(str(statement))

    session = FakeSession()
    asyncio.run(enforce_postgresql_read_only_transaction(session))  # type: ignore[arg-type]
    assert session.statements == ["SET TRANSACTION READ ONLY"]


def test_finalize_review_removes_partial_outputs_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _manifest = _export_ready_bundle(tmp_path)
    _complete_review(bundle / "review.json")
    original_writer = exporter_module._write_json_artifact

    def failing_writer(
        root: Path,
        name: str,
        payload: dict[str, Any],
    ) -> Any:
        if name == "golden-case.json":
            raise OSError("simulated local write failure")
        return original_writer(root, name, payload)

    monkeypatch.setattr(exporter_module, "_write_json_artifact", failing_writer)
    with pytest.raises(OSError, match="simulated local write failure"):
        finalize_case_review(bundle_dir=bundle, dataset_path=_dataset_path(tmp_path))

    assert not (bundle / "annotation.json").exists()
    assert not (bundle / "golden-case.json").exists()
    assert not (bundle / "review-receipt.json").exists()


def _export_ready_bundle(tmp_path: Path) -> tuple[Path, Any]:
    matrix = _matrix_payload()
    manifest = write_case_bundle(
        match_run=_match_run(matrix),
        matrix_payload=matrix,
        matrix_rows=_matrix_rows(),
        output_root=tmp_path / "local",
        baseline=_baseline(),
        dataset=_dataset(),
        case_id="prod-301-treolan",
    )
    _dataset_path(tmp_path)
    return tmp_path / "local" / manifest.case_id, manifest


def _complete_review(path: Path) -> None:
    review = json.loads(path.read_text(encoding="utf-8"))
    review.update(
        {
            "decision": "accept",
            "reviewer_role": "solution-engineer",
            "reviewed_at": "2026-07-22T18:00:00+03:00",
            "semantic_score": 0.95,
            "unsupported_material_claim_count": 0,
            "business_weighted_loss": 0.0,
        }
    )
    for criterion in review["atomic_criteria"]:
        criterion["status"] = "pass"
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.json"
    if not path.exists():
        path.write_text(
            json.dumps(_dataset().model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return path


def _dataset() -> GoldenDataset:
    payload = json.loads(
        Path("evaluation/simple_stock/v1/dataset.json").read_text(encoding="utf-8-sig")
    )
    return GoldenDataset.model_validate(payload)


def _baseline() -> dict[str, Any]:
    return {
        "production_commit": "ba20f7faa5282561baa50beec88b8b8aa5ff2371",
        "pipeline_version": "simple_stock_quote",
        "stages": {
            "route_prompt_version": "simple_stock_route_v19",
            "matrix_schema_version": "simple_stock_matrix.v8",
            "composer_prompt_version": "simple_stock_composer_v70",
            "reconciler_version": "quote_integrity_reconciler_v9",
        },
        "llm": {
            "model": "qwen/qwen3.7-max",
            "max_package_chars": 5_000_000,
        },
    }


def _match_run(matrix: dict[str, Any]) -> Any:
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
            "engineering_review_required": True,
            "lines": [
                {
                    "component_candidate_id": "treolan:item-secret",
                    "quantity": 1,
                }
            ],
        },
        "v3_validation_errors": [],
        "v3_validation_warnings": [],
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
        source_text="private request alpha",
        spec_json={"source_text": "private request alpha"},
        report_json=report,
        created_at=datetime(2026, 7, 2, 20, 54, 11, tzinfo=UTC),
    )


def _matrix_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "simple_stock_matrix.v8",
        "distributor_code": "treolan",
        "category_ids": ["cat-a"],
        "model": "qwen/qwen3.7-max",
        "category_sections": [
            {
                "category_id": "cat-a",
                "positions": [
                    {
                        "component_candidate_id": "treolan:item-secret",
                        "part_number": "PN-1",
                        "offers": [
                            {
                                "price": {"value": "100", "currency": "USD"},
                                "available_quantity": 1,
                            }
                        ],
                    }
                ],
            }
        ],
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
            stock=SimpleNamespace(
                synced_at=datetime(2026, 7, 2, 19, 0, tzinfo=UTC),
            )
        )
    ]


def _stock_row(
    distributor_code: str,
    item_id: str,
    synced_at: datetime,
    *,
    quantity: int,
) -> DistributorStockPrice:
    return DistributorStockPrice(
        distributor_code=distributor_code,
        item_id=item_id,
        product_key="key-1",
        shipment_city="Moscow",
        location="stock",
        location_description=None,
        location_type="stock",
        quantity_value=quantity,
        quantity_is_greater_than=False,
        can_reserve=True,
        departure_date=None,
        arrival_date=None,
        delivery_date=None,
        price_order_value=100,
        price_order_currency="USD",
        price_list_value=None,
        price_list_currency=None,
        end_user_value=None,
        end_user_currency=None,
        raw_json={},
        synced_at=synced_at,
    )
