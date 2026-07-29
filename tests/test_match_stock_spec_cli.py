from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Generator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import app.cli.list_match_runs as list_match_runs_cli
import app.cli.match_stock_spec as match_stock_spec_cli
import app.cli.preview_llm_configurator_package as preview_llm_configurator_package_cli
from app.core.config import get_llm_settings, get_web_evidence_settings
from app.core.database import Base
from app.db.models import DistributorProduct, DistributorStockPrice, MatchRun
from app.matching import match_engine as match_engine_module

SERVER_REQUEST = "Нужно 2 сервера 2U, 2 процессора, 512 ГБ RAM, SSD, 2 БП, склад Москва"
NETWORK_MATCH_71_TEXT = (
    "Нужен 1 коммутатор 48 портов 1G PoE+, 4 uplink 10G SFP+, "
    "L3, stacking, склад Москва, один самый дешевый вариант для КП"
)
NETWORK_MATCH_74_PLAIN_POE_TEXT = (
    "Нужен 1 коммутатор 48 портов 1G PoE, 4 uplink 10G SFP+, "
    "L3, stacking желательно, склад Москва, один самый дешевый вариант для КП"
)
COMPLEX_SERVER_78_TEXT = """
Execution: 1U
Sockets: 2
CPU: Intel 6th generation, 2 pcs, at least 24 cores
RAM: 256 GB DDR5 RDIMM
Disks: 6 x SSD 1920 GB SATA, 2 x SSD 480 GB SATA
Controller: LSI Logic 9400-8i / LSI 9500-8i
Network adapter: Intel X710-DA2 2x10GbE SFP+
Power: 2 x 2000W hot-swap Platinum, C13-C14 cables, C13-Schuko cables
Cooling: 8 fans N+1
Interfaces: USB 3.0, serial RJ-45, VGA, remote management RJ-45
""".strip()


class AsyncSessionAdapter:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, instance: object) -> None:
        self._session.add(instance)

    async def flush(self) -> None:
        self._session.flush()

    async def commit(self) -> None:
        self._session.commit()

    async def rollback(self) -> None:
        self._session.rollback()

    async def scalar(self, statement: Any) -> Any:
        return self._session.scalar(statement)

    async def execute(self, statement: Any) -> Any:
        return self._session.execute(statement)


class SessionContext:
    def __init__(self, adapter: AsyncSessionAdapter) -> None:
        self._adapter = adapter

    async def __aenter__(self) -> AsyncSessionAdapter:
        return self._adapter

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc is not None:
            await self._adapter.rollback()


@pytest.fixture()
def db_session() -> Generator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        yield session

    engine.dispose()


def test_match_stock_spec_cli_uses_fallback_extractor_and_saves_match_run(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_nerpa_product(db_session)
    _patch_session_factory(monkeypatch, db_session)
    report_dir = Path("data/match_reports_test")
    monkeypatch.setattr(match_stock_spec_cli, "DEFAULT_REPORT_DIR", report_dir)
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(match_stock_spec_cli.run(["--text", SERVER_REQUEST]))
    captured = capsys.readouterr()

    match_run_count = db_session.scalar(select(func.count()).select_from(MatchRun))
    match_run = db_session.scalar(select(MatchRun))

    assert exit_code == 0
    assert "Match Engine V0 Report" in captured.out
    assert "Найдены варианты, но полное соответствие не подтверждено" in captured.out
    assert "D5720-181125SA04" in captured.out
    assert match_run_count == 1
    assert match_run is not None
    assert match_run.engineer_review_required is True
    assert (report_dir / f"{match_run.id}.json").exists()
    assert (report_dir / f"{match_run.id}.md").exists()

    get_llm_settings.cache_clear()


def test_list_match_runs_cli_prints_recent_runs(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_match_run(db_session)
    _patch_session_factory(monkeypatch, db_session)

    exit_code = asyncio.run(list_match_runs_cli.run(["--limit", "10"]))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Match runs" in captured.out
    assert "partial_stock_matched" in captured.out
    assert "Нужно 2 сервера" in captured.out


def test_preview_llm_configurator_package_cli_prints_safe_coverage_summary(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-2U-2S",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 2U dual socket DDR5 server platform 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="cpu-16",
        part_number="CPU-16C",
        producer="Intel",
        category_id="V110103",
        item_name="Intel Xeon Silver 4416 16 core tray processor",
        quantity=4,
        price=Decimal("500"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-64",
        part_number="RAM-DDR5-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-3840",
        part_number="SSD-3840",
        producer="KIOXIA",
        category_id="V110106",
        item_name="KIOXIA CD8-R Server SSD NVMe 3.84TB U.2",
        quantity=4,
        price=Decimal("300"),
    )
    _patch_session_factory(monkeypatch, db_session)
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                (
                    "Нужно 2 сервера 2U, 2 CPU Intel Xeon не менее 16 ядер, "
                    "512 ГБ RAM DDR5, 2 SSD NVMe 3.84 ТБ"
                ),
                "--json",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["normalized_requirements"]
    assert preview["count_by_role"]["cpu"] >= 1
    assert preview["matrix_coverage_by_role"]["selection_strategy"] == "bucketed_broad_matrix_v3"
    assert preview["first_candidates_by_role"]["cpu"][0]["part_number"] == "CPU-16C"
    assert preview["package_approximate_size"]["chars"] > 0
    assert preview["package_budget"]["final_chars"] == preview["package_approximate_size"][
        "chars"
    ]
    assert "raw_json" not in captured.out
    assert "test-key" not in captured.out


def test_preview_cli_progress_logs_go_to_stderr_and_stdout_stays_json(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_session_factory(monkeypatch, db_session)

    async def fake_build_package_from_text(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        match_engine_module._log_full_matrix_progress(
            "full_matrix_start",
            {"product_group": "server", "max_seconds": 900},
        )
        match_engine_module._log_full_matrix_progress(
            "role_evaluator_start",
            {"role": "cpu", "chunk": 0},
        )
        match_engine_module._log_full_matrix_progress(
            "role_evaluator_done",
            {"role": "cpu", "chunk": 0},
        )
        match_engine_module._log_full_matrix_progress(
            "reducer_start",
            {"role": "cpu"},
        )
        match_engine_module._log_full_matrix_progress(
            "reducer_done",
            {"role": "cpu"},
        )
        return {
            "product_group": "server",
            "primary_object": "server",
            "component_candidate_matrix": {},
            "full_matrix_evaluation_used": False,
            "full_matrix_evaluation_fallback_reason": (
                "full_matrix_evaluation_timeout_but_package_under_budget"
            ),
            "package_budget": {"over_budget": False, "final_chars": 2000, "max_chars": 200000},
            "package_skipped_reason": None,
            "llm_fallback_reason": None,
        }

    monkeypatch.setattr(
        preview_llm_configurator_package_cli,
        "build_llm_configurator_package_from_text",
        fake_build_package_from_text,
    )

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(["--text", "server #78", "--json"])
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["full_matrix_evaluation_used"] is False
    assert preview["full_matrix_evaluation_fallback_reason"] == (
        "full_matrix_evaluation_timeout_but_package_under_budget"
    )
    assert "full_matrix_start" not in captured.out
    assert "role_evaluator_start" not in captured.out
    assert "full_matrix_start" in captured.err
    assert "role_evaluator_done" in captured.err


def test_preview_show_candidates_only_limits_display() -> None:
    package = {
        "product_group": "server",
        "component_candidate_matrix": {
            "cpu": [
                {
                    "component_candidate_id": f"cpu-{index}",
                    "part_number": f"CPU-{index}",
                }
                for index in range(5)
            ]
        },
        "broad_matrix_count_by_role": {"cpu": 5},
        "composer_package_candidate_count_by_role": {"cpu": 5},
        "composer_package_candidate_total": 5,
        "composer_package_candidate_ids_by_role": {
            "cpu": [f"cpu-{index}" for index in range(5)]
        },
        "dropped_before_composer_count_by_role": {"cpu": 0},
        "package_candidate_exposure_policy": {
            "mode": "full_broad_matrix"
        },
        "package_budget": {"over_budget": False, "final_chars": 1000},
    }

    preview = preview_llm_configurator_package_cli._preview_summary(
        package,
        show_candidates=1,
        show_filter_diagnostics=False,
        show_dropped_categories=False,
    )

    assert len(preview["first_candidates_by_role"]["cpu"]) == 1
    assert preview["composer_package_candidate_count_by_role"]["cpu"] == 5
    assert preview["composer_package_candidate_total"] == 5


def test_preview_cli_dump_composer_package_keeps_stdout_json(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dump_path = Path(".tmp_pytest/composer-package-dump-test.json")
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.unlink(missing_ok=True)
    _patch_session_factory(monkeypatch, db_session)

    async def fake_build_package_from_text(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "product_group": "server",
            "component_candidate_matrix": {
                "cpu": [{"component_candidate_id": "cpu-1", "part_number": "CPU-1"}]
            },
            "broad_matrix_count_by_role": {"cpu": 1},
            "composer_package_candidate_count_by_role": {"cpu": 1},
            "composer_package_candidate_total": 1,
            "composer_package_candidate_ids_by_role": {"cpu": ["cpu-1"]},
            "dropped_before_composer_count_by_role": {"cpu": 0},
            "package_candidate_exposure_policy": {
                "mode": "full_broad_matrix"
            },
            "package_budget": {"over_budget": False, "final_chars": 1000},
        }

    monkeypatch.setattr(
        preview_llm_configurator_package_cli,
        "build_llm_configurator_package_from_text",
        fake_build_package_from_text,
    )

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                "server #86",
                "--json",
                "--dump-composer-package",
                str(dump_path),
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)
    dump = json.loads(dump_path.read_text(encoding="utf-8"))
    dump_path.unlink(missing_ok=True)

    assert exit_code == 0
    assert preview["composer_package_dump_path"] == str(dump_path)
    assert "Composer package dump written" in captured.err
    assert dump["package_candidate_summary"]["composer_package_candidate_total"] == 1
    assert dump["composer_package"]["composer_package_candidate_ids_by_role"] == {
        "cpu": ["cpu-1"]
    }
    assert captured.out.strip().startswith("{")


def test_preview_llm_configurator_package_cli_uses_llm_semantic_planner_by_default(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-1U-2S",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 1U dual socket DDR5 server platform",
        quantity=1,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="nic-1",
        part_number="NIC-X710",
        producer="Intel",
        category_id="V120116",
        item_name="Intel X710-DA2 dual-port 10GbE SFP+ server adapter",
        quantity=2,
        price=Decimal("200"),
    )
    _patch_session_factory(monkeypatch, db_session)
    monkeypatch.setattr(
        match_engine_module,
        "OpenAICompatibleLlmClient",
        _PreviewSemanticPlannerClient,
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "preview-secret")
    monkeypatch.setenv("LLM_MODEL", "qwen-test")
    monkeypatch.setenv("LLM_CONFIGURATOR_ENABLED", "true")
    monkeypatch.setenv("LLM_CONFIGURATOR_MODE", "composer")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                (
                    "1U 2-socket server, Intel CPUs, DDR5 RAM, SATA SSD, "
                    "Intel X710-DA2 2x10GbE SFP+, C13-C14 power cables"
                ),
                "--json",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["semantic_planner_source"] == "llm"
    assert preview["semantic_planner_used"] is True
    assert preview["semantic_planner_provider"] == "openai-compatible"
    assert preview["semantic_planner_model"] == "qwen-test"
    assert preview["product_group"] == "server"
    assert "network_adapter" in preview["matrix_blueprint_roles"]
    assert preview["category_plan"]["server_platform"] == ["V110100"]
    assert preview["category_plan"]["network_adapter"] == ["V120116"]
    assert "preview-secret" not in captured.out
    get_llm_settings.cache_clear()


def test_preview_llm_configurator_package_cli_no_network_still_uses_semantic_planner(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-1U-2S",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 1U dual socket DDR5 server platform",
        quantity=1,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="nic-1",
        part_number="NIC-X710",
        producer="Intel",
        category_id="V120116",
        item_name="Intel X710-DA2 dual-port 10GbE SFP+ server adapter",
        quantity=2,
        price=Decimal("200"),
    )
    _patch_session_factory(monkeypatch, db_session)
    _PreviewSemanticPlannerClient.semantic_calls = 0
    monkeypatch.setattr(
        match_engine_module,
        "OpenAICompatibleLlmClient",
        _PreviewSemanticPlannerClient,
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "preview-secret")
    monkeypatch.setenv("LLM_MODEL", "qwen-test")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                COMPLEX_SERVER_78_TEXT,
                "--json",
                "--no-network",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert _PreviewSemanticPlannerClient.semantic_calls >= 1
    assert preview["semantic_planner_source"] == "llm"
    assert preview["product_group"] == "server"
    assert preview["required_capabilities"]
    assert "network_adapter" in preview["matrix_blueprint_roles"]
    assert "preview-secret" not in captured.out
    get_llm_settings.cache_clear()


def test_preview_llm_configurator_package_cli_semantic_only_smokes_planner(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_session_factory(monkeypatch, db_session)
    _PreviewSemanticPlannerClient.semantic_calls = 0
    monkeypatch.setattr(
        match_engine_module,
        "OpenAICompatibleLlmClient",
        _PreviewSemanticPlannerClient,
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "preview-secret")
    monkeypatch.setenv("LLM_MODEL", "qwen-test")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                COMPLEX_SERVER_78_TEXT,
                "--json",
                "--semantic-only",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["semantic_only"] is True
    assert _PreviewSemanticPlannerClient.semantic_calls >= 2
    assert preview["semantic_planner_source"] == "llm"
    assert preview["product_group"] == "server"
    assert "requirement_source_coverage_percent" in preview
    assert "unclassified_source_fragments" in preview
    assert "component_candidate_matrix" not in preview
    assert "preview-secret" not in captured.out
    get_llm_settings.cache_clear()


def test_preview_cli_semantic_only_intent_timeout_returns_clean_json(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_session_factory(monkeypatch, db_session)
    monkeypatch.setattr(
        match_engine_module,
        "OpenAICompatibleLlmClient",
        _HangingPreviewSemanticPlannerClient,
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "preview-secret")
    monkeypatch.setenv("LLM_MODEL", "qwen-test")
    monkeypatch.setenv("LLM_SEMANTIC_PLANNER_MAX_SECONDS", "0.2")
    monkeypatch.setenv("LLM_SEMANTIC_PLANNER_STAGE_TIMEOUT_SECONDS", "0.05")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                COMPLEX_SERVER_78_TEXT,
                "--json",
                "--semantic-only",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["semantic_only"] is True
    assert preview["product_group"] == "unknown"
    assert preview["semantic_planner_source"] == "fallback_after_llm_timeout"
    assert preview["semantic_planner_fallback_reason"] == "semantic_planner_timeout"
    assert preview["semantic_planner_error_type"] == "SemanticPlannerTimeout"
    assert preview["semantic_planner_stage"] == "intent_router"
    assert preview["semantic_planner_stage_timeouts"][0]["stage"] == "intent_router"
    assert preview["requirement_classifier_status"] == "failed"
    assert "semantic_stage_start stage=intent_router" in captured.err
    assert "semantic_stage_timeout stage=intent_router" in captured.err
    assert "preview-secret" not in captured.out
    get_llm_settings.cache_clear()


def test_preview_llm_configurator_package_cli_fails_closed_for_78_without_planner(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_nerpa_product(db_session)
    _patch_session_factory(monkeypatch, db_session)
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                COMPLEX_SERVER_78_TEXT,
                "--json",
                "--show-candidates",
                "8",
                "--show-dropped-categories",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["semantic_planner_source"] == (
        "complex_request_requires_llm_semantic_planner"
    )
    assert preview["semantic_planner_source"] is not None
    assert preview["semantic_planner_fallback_reason"] == (
        "complex_request_requires_llm_semantic_planner"
    )
    assert preview["product_group"] == "unknown"
    assert preview["required_capabilities"] == []
    assert preview["category_plan"] == {}
    assert preview["count_by_role"] == {}
    assert preview["first_candidates_by_role"] == {}
    assert preview["package_skipped_reason"] == (
        "complex_request_requires_llm_semantic_planner"
    )
    assert preview["package_approximate_size"]["chars"] < 120000
    get_llm_settings.cache_clear()


def test_preview_llm_configurator_package_cli_deterministic_only_is_explicit(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_component_product(
        db_session,
        item_id="dac-1",
        part_number="DAC-10G",
        producer="NetVendor",
        category_id="V120150",
        item_name="10G SFP+ DAC cable 3m",
        quantity=2,
        price=Decimal("30"),
    )
    _patch_session_factory(monkeypatch, db_session)

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                "Need DAC SFP+ 10G 3m",
                "--json",
                "--deterministic-only",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["semantic_planner_source"] == "deterministic_preview_only"
    assert preview["selected_product_group_reason"] == (
        "Preview was run without LLM semantic planner"
    )
    assert preview["product_group"] == "network"


def test_preview_llm_configurator_package_cli_uses_match_71_network_pipeline(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_match_71_network_products(db_session)
    _patch_session_factory(monkeypatch, db_session)
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                NETWORK_MATCH_71_TEXT,
                "--json",
                "--show-candidates",
                "8",
                "--show-dropped-categories",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)
    switch_candidates = preview["first_candidates_by_role"]["switch"]
    part_numbers = [candidate["part_number"] for candidate in switch_candidates]

    assert exit_code == 0
    assert preview["product_group"] == "network"
    assert preview["category_plan"]["switch"] == ["V120100"]
    assert preview["count_by_role"]["switch"] >= 1
    assert switch_candidates[0]["part_number"] == "SW-48P-4SFP"
    assert {"SW-5P", "SW-8P", "SW-16P", "SW-24P"}.isdisjoint(part_numbers)
    assert "ready_server" not in preview["count_by_role"]


def test_preview_llm_configurator_package_cli_preserves_plain_poe_network_pipeline(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_match_71_network_products(db_session)
    _patch_session_factory(monkeypatch, db_session)
    monkeypatch.setenv("LLM_PROVIDER", "disabled")
    get_llm_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                NETWORK_MATCH_74_PLAIN_POE_TEXT,
                "--json",
                "--show-candidates",
                "8",
                "--show-dropped-categories",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)
    switch_capability = next(
        capability
        for capability in preview["required_capabilities"]
        if capability["role"] == "switch"
    )
    switch_requirements = switch_capability["parsed_requirements"]

    assert exit_code == 0
    assert preview["product_group"] == "network"
    assert switch_requirements["poe_required"] is True
    assert switch_requirements["poe_standard"] == "PoE"
    assert switch_requirements["poe_standard"] != "PoE+"
    assert "stacking_required" not in switch_requirements
    assert any(
        capability["role"] == "switch"
        and capability["hard"] is False
        and capability["parsed_requirements"].get("stacking_required") is True
        for capability in preview["optional_capabilities"]
    )
    assert preview["category_plan"]["switch"] == ["V120100"]
    assert preview["count_by_role"]["switch"] >= 1
    assert preview["first_candidates_by_role"]["switch"]


def test_preview_llm_configurator_package_cli_can_show_evidence_tasks_without_network(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-2U-2S",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 2U dual socket DDR5 server platform 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    _patch_session_factory(monkeypatch, db_session)
    monkeypatch.setenv("WEB_EVIDENCE_ENABLED", "false")
    get_web_evidence_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                "Need 2 servers 2U, 2 CPU, 512 GB RAM DDR5",
                "--with-evidence-preview",
                "--no-network",
                "--json",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["web_evidence_preview"]["status"] == "web evidence disabled"
    assert "TAVILY_API_KEY" not in captured.out
    assert "test-key" not in captured.out
    get_web_evidence_settings.cache_clear()


def test_preview_llm_configurator_package_cli_routerai_no_network_hides_secrets(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="RS720-E11-RS24U",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS RS720-E11-RS24U server platform",
        quantity=2,
        price=Decimal("2000"),
    )
    _patch_session_factory(monkeypatch, db_session)
    monkeypatch.setenv("WEB_EVIDENCE_ENABLED", "true")
    monkeypatch.setenv("WEB_EVIDENCE_PROVIDER", "routerai")
    monkeypatch.setenv("WEB_EVIDENCE_MODEL", "deepseek/deepseek-v4-pro:online")
    monkeypatch.setenv("LLM_API_KEY", "router-secret")
    get_web_evidence_settings.cache_clear()

    exit_code = asyncio.run(
        preview_llm_configurator_package_cli.run(
            [
                "--text",
                "Need 2 servers 2U, 2 CPU, 512 GB RAM DDR5",
                "--with-evidence-preview",
                "--no-network",
                "--json",
            ]
        )
    )
    captured = capsys.readouterr()
    preview = json.loads(captured.out)

    assert exit_code == 0
    assert preview["web_evidence_preview"]["provider"] == "routerai"
    assert preview["web_evidence_preview"]["model"] == "deepseek/deepseek-v4-pro:online"
    assert preview["web_evidence_preview"]["status"] == (
        "web evidence planned; no network request sent"
    )
    assert "router-secret" not in captured.out
    get_web_evidence_settings.cache_clear()


def _patch_session_factory(monkeypatch: pytest.MonkeyPatch, db_session: Session) -> None:
    adapter = AsyncSessionAdapter(db_session)

    def session_factory() -> SessionContext:
        return SessionContext(adapter)

    monkeypatch.setattr(match_stock_spec_cli, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(list_match_runs_cli, "get_session_factory", lambda: session_factory)
    monkeypatch.setattr(
        preview_llm_configurator_package_cli,
        "get_session_factory",
        lambda: session_factory,
    )


class _PreviewSemanticPlannerClient:
    semantic_calls = 0

    def __init__(self, settings: object, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def close(self) -> None:
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "AI Semantic Matrix Planner V2" in system_prompt:
            type(self).semantic_calls += 1
            json.loads(user_prompt)
            return {
                "primary_product_group": "server",
                "primary_object": "server",
                "confidence": "high",
                "classification_reason": "The request is a server with embedded NIC.",
                "matrix_blueprint": {
                    "roles": [
                        {
                            "role": "server_platform",
                            "required": True,
                            "source_text": "1U 2-socket server",
                            "characteristics_to_match": {
                                "form_factor": "1U",
                                "socket_count": 2,
                            },
                            "hard_capability_ids": ["server_platform.1u.2s"],
                        },
                        {
                            "role": "network_adapter",
                            "required": True,
                            "source_text": "Intel X710-DA2 2x10GbE SFP+",
                            "characteristics_to_match": {
                                "min_ports_per_server": 2,
                                "speed": "10GbE",
                                "media": "SFP+",
                            },
                            "hard_capability_ids": ["network_adapter.10gbe.sfp+"],
                        },
                        {
                            "role": "cpu",
                            "required": True,
                            "source_text": "Intel CPUs",
                            "characteristics_to_match": {"vendor": "Intel"},
                            "hard_capability_ids": ["cpu.intel"],
                        },
                        {
                            "role": "ram",
                            "required": True,
                            "source_text": "DDR5 RAM",
                            "characteristics_to_match": {"type": "DDR5"},
                            "hard_capability_ids": ["ram.ddr5"],
                        },
                        {
                            "role": "storage",
                            "required": True,
                            "source_text": "SATA SSD",
                            "characteristics_to_match": {"interface": "SATA"},
                            "hard_capability_ids": ["storage.sata"],
                        },
                        {
                            "role": "storage_controller",
                            "required": True,
                            "source_text": "LSI HBA",
                            "characteristics_to_match": {"controller_type": "HBA"},
                            "hard_capability_ids": ["storage_controller.hba"],
                        },
                        {
                            "role": "power_supply",
                            "required": True,
                            "source_text": "2 x 2000W PSU",
                            "characteristics_to_match": {"wattage_w": 2000},
                            "hard_capability_ids": ["power_supply.2000w"],
                        },
                        {
                            "role": "cable",
                            "required": True,
                            "source_text": "C13-C14 power cables",
                            "characteristics_to_match": {"cable_type": "power"},
                            "hard_capability_ids": ["power_cable.c13"],
                        },
                    ]
                },
                "required_capabilities": [],
                "optional_capabilities": [],
                "embedded_requirements": [
                    {
                        "role": "network_adapter",
                        "reason": "SFP+ is inside the requested server.",
                    }
                ],
                "not_primary_product_groups": [
                    {
                        "product_group": "network",
                        "reason": "No standalone network device or DAC requested.",
                    }
                ],
                "logistics_constraints": {},
                "commercial_instructions": [],
                "response_instructions": [],
                "engineer_review_instructions": [],
                "unsupported_or_unmapped_requirements": [],
                "planner_warnings": [],
            }
        if "Distributor Category Planner" in system_prompt:
            return {
                "category_plan": [
                    {
                        "role": "server_platform",
                        "selected_category_ids": ["V110100"],
                        "purpose": "base_device",
                        "capability_ids": ["server_platform.1u.2s"],
                        "hard_optional_relation": "hard",
                        "reason": "server platform category",
                        "confidence": "high",
                    },
                    {
                        "role": "network_adapter",
                        "selected_category_ids": ["V120116"],
                        "purpose": "component",
                        "capability_ids": ["network_adapter.10gbe.sfp+"],
                        "hard_optional_relation": "hard",
                        "reason": "server NIC category",
                        "confidence": "high",
                    },
                ],
                "missing_category_roles": [],
                "category_plan_warnings": [],
            }
        raise AssertionError(f"unexpected preview LLM prompt: {system_prompt[:80]}")


class _HangingPreviewSemanticPlannerClient:
    def __init__(self, settings: object, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def close(self) -> None:
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        json.loads(user_prompt)
        if "Stage A Minimal AI Intent Router" in system_prompt:
            time.sleep(0.3)
            return {}
        raise AssertionError(f"unexpected preview LLM prompt: {system_prompt[:80]}")


def _seed_nerpa_product(db_session: Session) -> None:
    synced_at = datetime(2026, 5, 9, 12, 0, tzinfo=UTC)
    now = datetime(2026, 5, 9, 12, 5, tzinfo=UTC)
    db_session.add(
        DistributorProduct(
            distributor_code="ocs",
            item_id="1000841882",
            product_key="1000841882",
            part_number="D5720-181125SA04",
            producer="NERPA",
            category_id="V1100",
            item_name="Server NERPA D5720 2U 2x CPU SSD 2x PSU 2x DDR4 64GB",
            item_name_rus="Сервер NERPA",
            product_name="NERPA D5720-181125SA04",
            product_description=None,
            product_notes=None,
            hscode="8471490000",
            ean="04600000000012",
            is_in_mpt_registry=False,
            is_project_item=False,
            traceable=False,
            condition="Regular",
            warranty="Distributor warranty 12 months",
            original_country_iso_code="RU",
            vat_percent=Decimal("20"),
            serial_number_availability=None,
            catalog_path_json=[{"category_id": "V1100", "name": "Servers in assembly"}],
            package_json={"weight": 25.0},
            raw_json={"product": {"itemId": "1000841882"}},
            synced_at=synced_at,
            created_at=now,
            updated_at=now,
        )
    )
    db_session.add(
        DistributorStockPrice(
            distributor_code="ocs",
            item_id="1000841882",
            product_key="1000841882",
            shipment_city="Москва",
            location="MSK",
            location_description="Moscow",
            location_type="ShipmentCity",
            quantity_value=3,
            quantity_is_greater_than=False,
            can_reserve=True,
            departure_date=None,
            arrival_date=None,
            delivery_date=None,
            price_order_value=Decimal("6900"),
            price_order_currency="USD",
            price_list_value=Decimal("6900"),
            price_list_currency="USD",
            end_user_value=Decimal("7100"),
            end_user_currency="USD",
            raw_json={"productKey": "1000841882", "location": "MSK"},
            synced_at=synced_at,
            created_at=now,
        )
    )
    db_session.commit()


def _seed_component_product(
    db_session: Session,
    *,
    item_id: str,
    part_number: str,
    producer: str,
    category_id: str,
    item_name: str,
    quantity: int,
    price: Decimal,
) -> None:
    synced_at = datetime(2026, 5, 9, 12, 0, tzinfo=UTC)
    now = datetime(2026, 5, 9, 12, 5, tzinfo=UTC)
    db_session.add(
        DistributorProduct(
            distributor_code="ocs",
            item_id=item_id,
            product_key=item_id,
            part_number=part_number,
            producer=producer,
            category_id=category_id,
            item_name=item_name,
            item_name_rus=item_name,
            product_name=f"{producer} {part_number}",
            product_description=None,
            product_notes=None,
            hscode="8473308000",
            ean=None,
            is_in_mpt_registry=False,
            is_project_item=False,
            traceable=False,
            condition="Regular",
            warranty="Distributor warranty 12 months",
            original_country_iso_code="CN",
            vat_percent=Decimal("20"),
            serial_number_availability=None,
            catalog_path_json=[{"category_id": category_id, "name": "Server component"}],
            package_json={},
            raw_json={"product": {"itemId": item_id, "partNumber": part_number}},
            synced_at=synced_at,
            created_at=now,
            updated_at=now,
        )
    )
    db_session.add(
        DistributorStockPrice(
            distributor_code="ocs",
            item_id=item_id,
            product_key=item_id,
            shipment_city="Moscow",
            location="MSK",
            location_description="Moscow",
            location_type="ShipmentCity",
            quantity_value=quantity,
            quantity_is_greater_than=False,
            can_reserve=True,
            departure_date=None,
            arrival_date=None,
            delivery_date=None,
            price_order_value=price,
            price_order_currency="USD",
            price_list_value=price,
            price_list_currency="USD",
            end_user_value=price,
            end_user_currency="USD",
            raw_json={"productKey": item_id, "location": "MSK"},
            synced_at=synced_at,
            created_at=now,
        )
    )
    db_session.commit()


def _seed_match_71_network_products(db_session: Session) -> None:
    rows = [
        (
            "switch-good",
            "SW-48P-4SFP",
            "48-port 1G RJ45 PoE+ switch 740W 4 uplink 10G SFP+ L3 stacking",
            "1200",
        ),
        ("switch-5p", "SW-5P", "5-port desktop switch", "25"),
        ("switch-8p", "SW-8P", "8-port 1G PoE switch", "100"),
        ("switch-16p", "SW-16P", "16-port 1G PoE switch", "180"),
        ("switch-24p", "SW-24P", "24-port 1G PoE+ switch 2 uplink 10G SFP+", "260"),
    ]
    for item_id, part_number, name, price in rows:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="NetVendor",
            category_id="V120100",
            item_name=name,
            quantity=10,
            price=Decimal(price),
        )


def _seed_match_run(db_session: Session) -> None:
    now = datetime(2026, 5, 9, 12, 30, tzinfo=UTC)
    db_session.add(
        MatchRun(
            source="text",
            source_text=SERVER_REQUEST,
            status="partial_stock_matched",
            engineer_review_required=True,
            total_candidates=1,
            matched_items=0,
            missing_requirements_json=["RAM ниже требования"],
            risk_flags_json=["Гарантия из OCS"],
            spec_json={"items": []},
            report_json={"status": "partial_stock_matched"},
            report_markdown="# Match Engine V0 Report",
            created_at=now,
        )
    )
    db_session.commit()
