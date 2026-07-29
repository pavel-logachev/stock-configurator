from __future__ import annotations

import asyncio
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import app.cli.disable_ocs_category as disable_ocs_category_cli
import app.cli.discover_anchor_ocs_categories as discover_anchor_cli
import app.cli.enable_ocs_anchor_categories as enable_ocs_anchor_categories_cli
import app.cli.enable_ocs_category as enable_ocs_category_cli
import app.cli.enable_server_ocs_categories as enable_server_ocs_categories_cli
from app.catalog.category_repository import CategoryRepository, CategoryUpsert
from app.catalog.ocs_anchor_categories import load_ocs_anchor_categories
from app.catalog.ocs_server_categories import enabled_server_categories, server_category_role_label
from app.core.database import Base
from app.db.models import Distributor, DistributorCategory, SyncRun
from app.distributors.ocs.sync_categories import flatten_ocs_categories, sync_ocs_categories

OCS_CATEGORY_SAMPLE: list[dict[str, Any]] = [
    {
        "category": "V11",
        "name": "Серверы",
        "children": [
            {
                "category": "V1100",
                "name": "Серверы в сборе",
                "children": [],
            },
            {
                "category": "V1101",
                "name": "Комплектующие для серверов",
                "children": [
                    {
                        "category": "V110100",
                        "name": "Серверные платформы",
                        "children": [],
                    }
                ],
            },
        ],
    }
]


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


class AsyncSessionContext:
    def __init__(self, session: Session) -> None:
        self._adapter = AsyncSessionAdapter(session)

    async def __aenter__(self) -> AsyncSessionAdapter:
        return self._adapter

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class FakeOcsClient:
    def __init__(self, payload: Any | None = None, exc: Exception | None = None) -> None:
        self._payload = payload
        self._exc = exc

    async def get_categories(self) -> Any:
        if self._exc is not None:
            raise self._exc
        return self._payload


@pytest.fixture()
def db_session() -> Generator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        yield session

    engine.dispose()


def test_flatten_ocs_categories_recursively_maps_level_parent_and_path() -> None:
    synced_at = datetime(2026, 5, 9, tzinfo=UTC)
    rows = flatten_ocs_categories(OCS_CATEGORY_SAMPLE, synced_at=synced_at)

    assert [row.category_id for row in rows] == ["V11", "V1100", "V1101", "V110100"]
    assert [row.level for row in rows] == [0, 1, 1, 2]
    assert [row.parent_category_id for row in rows] == [None, "V11", "V11", "V1101"]
    assert rows[3].path_json == [
        {"category_id": "V11", "name": "Серверы"},
        {"category_id": "V1101", "name": "Комплектующие для серверов"},
        {"category_id": "V110100", "name": "Серверные платформы"},
    ]
    assert all(row.synced_at == synced_at for row in rows)


def test_flatten_ocs_categories_falls_back_to_id_when_category_is_empty() -> None:
    rows = flatten_ocs_categories([{"category": "", "id": "legacy", "name": "Legacy"}])

    assert [row.category_id for row in rows] == ["legacy"]


def test_category_repository_upsert_updates_existing_row_without_duplicates(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    repository = CategoryRepository(adapter)  # type: ignore[arg-type]
    synced_at = datetime(2026, 5, 9, tzinfo=UTC)

    async def run() -> None:
        await repository.upsert_category(
            CategoryUpsert(
                distributor_code="ocs",
                category_id="servers",
                parent_category_id=None,
                name="Servers",
                level=0,
                path_json=[{"category_id": "servers", "name": "Servers"}],
                raw_json={"id": "servers", "name": "Servers"},
                synced_at=synced_at,
                enabled_for_sync=True,
            )
        )
        await repository.upsert_category(
            CategoryUpsert(
                distributor_code="ocs",
                category_id="servers",
                parent_category_id=None,
                name="Updated servers",
                level=0,
                path_json=[{"category_id": "servers", "name": "Updated servers"}],
                raw_json={"id": "servers", "name": "Updated servers"},
                synced_at=synced_at,
            )
        )

    asyncio.run(run())

    count = db_session.scalar(select(func.count()).select_from(DistributorCategory))
    category = db_session.scalar(select(DistributorCategory))

    assert count == 1
    assert category is not None
    assert category.name == "Updated servers"
    assert category.enabled_for_sync is True


def test_category_repository_set_category_enabled_enables_existing_category(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    repository = CategoryRepository(adapter)  # type: ignore[arg-type]
    synced_at = datetime(2026, 5, 9, tzinfo=UTC)
    rows = flatten_ocs_categories(OCS_CATEGORY_SAMPLE, synced_at=synced_at)

    async def run() -> DistributorCategory | None:
        for row in rows:
            await repository.upsert_category(row)
        return await repository.set_category_enabled("ocs", "V1100", True)

    category = asyncio.run(run())
    saved = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "V1100")
    )

    assert category is not None
    assert category.enabled_for_sync is True
    assert saved is not None
    assert saved.enabled_for_sync is True


def test_category_repository_set_category_enabled_disables_existing_category(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    repository = CategoryRepository(adapter)  # type: ignore[arg-type]
    synced_at = datetime(2026, 5, 9, tzinfo=UTC)

    async def run() -> DistributorCategory | None:
        await repository.upsert_category(
            CategoryUpsert(
                distributor_code="ocs",
                category_id="V1100",
                parent_category_id=None,
                name="Servers in assembly",
                level=0,
                path_json=[{"category_id": "V1100", "name": "Servers in assembly"}],
                raw_json={"category": "V1100", "name": "Servers in assembly"},
                synced_at=synced_at,
                enabled_for_sync=True,
            )
        )
        return await repository.set_category_enabled("ocs", "V1100", False)

    category = asyncio.run(run())
    saved = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "V1100")
    )

    assert category is not None
    assert category.enabled_for_sync is False
    assert saved is not None
    assert saved.enabled_for_sync is False


def test_enable_ocs_category_cli_returns_error_for_missing_category(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_cli_session_factory(monkeypatch, enable_ocs_category_cli, db_session)

    exit_code = asyncio.run(enable_ocs_category_cli.run(["--category-id", "missing"]))
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "OCS category was not found: missing" in captured.err


def test_enable_ocs_category_cli_enables_existing_category(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_category(db_session, category_id="V1100", enabled_for_sync=False)
    _patch_cli_session_factory(monkeypatch, enable_ocs_category_cli, db_session)

    exit_code = asyncio.run(
        enable_ocs_category_cli.run(
            [
                "--category-id",
                "V1100",
                "--comment",
                "Servers in assembly",
            ]
        )
    )
    captured = capsys.readouterr()
    saved = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "V1100")
    )

    assert exit_code == 0
    assert saved is not None
    assert saved.enabled_for_sync is True
    assert "enabled V1100" in captured.out
    assert "enabled_for_sync=true" in captured.out
    assert "comment: Servers in assembly" in captured.out


def test_enable_server_ocs_categories_cli_reports_enabled_already_enabled_and_missing(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_category(db_session, category_id="V1100", enabled_for_sync=False)
    _seed_category(db_session, category_id="V110100", enabled_for_sync=True)
    _seed_category(db_session, category_id="V110103", enabled_for_sync=False)
    _patch_cli_session_factory(monkeypatch, enable_server_ocs_categories_cli, db_session)

    exit_code = asyncio.run(enable_server_ocs_categories_cli.run())
    captured = capsys.readouterr()

    v1100 = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "V1100")
    )

    assert exit_code == 0
    assert v1100 is not None
    assert v1100.enabled_for_sync is True
    assert "V1100" in captured.out
    assert "ready_server" in captured.out
    assert "enabled" in captured.out
    assert "V110100" in captured.out
    assert "already_enabled" in captured.out
    assert "V110103" in captured.out
    assert "CPU / серверные процессоры" in captured.out
    assert "V110104" in captured.out
    assert "not_found" in captured.out
    v110103 = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "V110103")
    )
    assert v110103 is not None
    assert v110103.enabled_for_sync is True


def test_enable_server_ocs_categories_cli_is_idempotent(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_category(db_session, category_id="V1100", enabled_for_sync=False)
    _patch_cli_session_factory(monkeypatch, enable_server_ocs_categories_cli, db_session)

    first_exit_code = asyncio.run(enable_server_ocs_categories_cli.run())
    _ = capsys.readouterr()
    second_exit_code = asyncio.run(enable_server_ocs_categories_cli.run())
    captured = capsys.readouterr()

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert "V1100" in captured.out
    assert "already_enabled" in captured.out


def test_discover_anchor_ocs_categories_maps_local_candidates() -> None:
    categories = [
        {
            "category_id": "S-ARRAY",
            "name": "СХД и системы хранения",
            "path_json": [{"name": "Storage"}, {"name": "СХД"}],
            "parent_category_id": "S",
        },
        {
            "category_id": "N-SWITCH",
            "name": "Коммутаторы Ethernet",
            "path_json": [{"name": "Network"}, {"name": "Switches"}],
            "parent_category_id": "N",
        },
        {
            "category_id": "SUP-3Y",
            "name": "Поддержка и лицензии",
            "path_json": [{"name": "Support"}],
            "parent_category_id": "SUP",
        },
    ]

    candidates = discover_anchor_cli.discover_anchor_candidates(
        categories,
        product_counts={"S-ARRAY": 2, "N-SWITCH": 5, "SUP-3Y": 3},
        group="all",
    )

    candidate_keys = {
        (candidate.group, candidate.suggested_role, candidate.category_id)
        for candidate in candidates
    }
    assert ("storage", "storage_system", "S-ARRAY") in candidate_keys
    assert ("network", "switch", "N-SWITCH") in candidate_keys
    assert ("support_license", "support", "SUP-3Y") in candidate_keys
    assert all("secret" not in ",".join(candidate.matched_terms) for candidate in candidates)


def test_enable_ocs_anchor_categories_cli_is_idempotent_and_skips_missing(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = Path(".tmp_ocs_anchor_categories_test.yaml").resolve()
    try:
        config_path.write_text(
            """
anchors:
  - group: storage
    role: storage_system
    category_id: S-ARRAY
    comment: reviewed storage anchor
    enabled_default: false
    review_status: approved
  - group: storage
    role: drive
    category_id: S-DRIVE-MISSING
    comment: missing local category
    enabled_default: false
    review_status: approved
  - group: storage
    role: support
    category_id: S-SUPPORT-CANDIDATE
    comment: not approved yet
    enabled_default: false
    review_status: candidate
""".strip(),
            encoding="utf-8",
        )
        load_ocs_anchor_categories.cache_clear()
        _seed_category(db_session, category_id="S-ARRAY", enabled_for_sync=False)
        _seed_category(
            db_session,
            category_id="S-SUPPORT-CANDIDATE",
            enabled_for_sync=False,
        )
        _patch_cli_session_factory(
            monkeypatch,
            enable_ocs_anchor_categories_cli,
            db_session,
        )

        first_exit = asyncio.run(
            enable_ocs_anchor_categories_cli.run(
                ["--group", "storage", "--config", str(config_path)]
            )
        )
        _ = capsys.readouterr()
        second_exit = asyncio.run(
            enable_ocs_anchor_categories_cli.run(
                ["--group", "storage", "--config", str(config_path)]
            )
        )
        captured = capsys.readouterr()

        enabled = db_session.scalar(
            select(DistributorCategory).where(
                DistributorCategory.category_id == "S-ARRAY"
            )
        )
        candidate = db_session.scalar(
            select(DistributorCategory).where(
                DistributorCategory.category_id == "S-SUPPORT-CANDIDATE"
            )
        )
        assert first_exit == 0
        assert second_exit == 0
        assert enabled is not None
        assert enabled.enabled_for_sync is True
        assert candidate is not None
        assert candidate.enabled_for_sync is False
        assert "already_enabled" in captured.out
        assert "not_found" in captured.out
        assert "S-SUPPORT-CANDIDATE" not in captured.out
    finally:
        load_ocs_anchor_categories.cache_clear()
        config_path.unlink(missing_ok=True)


def test_server_category_profile_contains_server_cpu_category() -> None:
    categories = {category.category_id: category for category in enabled_server_categories()}

    assert categories["V110103"].name_ru == "Серверные процессоры"
    assert categories["V110103"].role == "cpu"
    assert categories["V110103"].enabled_by_default is True
    assert server_category_role_label("cpu") == "CPU / серверные процессоры"
    assert "V0205" not in categories
    assert "V110111" not in categories


def test_disable_ocs_category_cli_disables_existing_category(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_category(db_session, category_id="V1100", enabled_for_sync=True)
    _patch_cli_session_factory(monkeypatch, disable_ocs_category_cli, db_session)

    exit_code = asyncio.run(disable_ocs_category_cli.run(["--category-id", "V1100"]))
    captured = capsys.readouterr()
    saved = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "V1100")
    )

    assert exit_code == 0
    assert saved is not None
    assert saved.enabled_for_sync is False
    assert "disabled V1100" in captured.out
    assert "enabled_for_sync=false" in captured.out


def test_category_repository_list_enabled_categories_returns_only_enabled_rows(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    repository = CategoryRepository(adapter)  # type: ignore[arg-type]
    synced_at = datetime(2026, 5, 9, tzinfo=UTC)
    rows = flatten_ocs_categories(OCS_CATEGORY_SAMPLE, synced_at=synced_at)

    async def run() -> list[DistributorCategory]:
        for row in rows:
            await repository.upsert_category(row)
        await repository.set_category_enabled("ocs", "V1100", True)
        await repository.set_category_enabled("ocs", "V110100", True)
        await repository.upsert_category(
            CategoryUpsert(
                distributor_code="other",
                category_id="enabled-too",
                parent_category_id=None,
                name="Other enabled category",
                level=0,
                path_json=[{"category_id": "enabled-too", "name": "Other enabled category"}],
                raw_json={"category": "enabled-too", "name": "Other enabled category"},
                synced_at=synced_at,
                enabled_for_sync=True,
            )
        )
        return await repository.list_enabled_categories("ocs")

    categories = asyncio.run(run())

    assert [category.category_id for category in categories] == ["V1100", "V110100"]


def _seed_category(
    db_session: Session,
    *,
    category_id: str,
    enabled_for_sync: bool,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    repository = CategoryRepository(adapter)  # type: ignore[arg-type]
    synced_at = datetime(2026, 5, 9, tzinfo=UTC)

    async def run() -> None:
        await repository.upsert_category(
            CategoryUpsert(
                distributor_code="ocs",
                category_id=category_id,
                parent_category_id=None,
                name="Servers in assembly",
                level=0,
                path_json=[{"category_id": category_id, "name": "Servers in assembly"}],
                raw_json={"category": category_id, "name": "Servers in assembly"},
                synced_at=synced_at,
                enabled_for_sync=enabled_for_sync,
            )
        )

    asyncio.run(run())


def _patch_cli_session_factory(
    monkeypatch: pytest.MonkeyPatch,
    cli_module: Any,
    db_session: Session,
) -> None:
    def fake_session_factory() -> AsyncSessionContext:
        return AsyncSessionContext(db_session)

    monkeypatch.setattr(cli_module, "get_session_factory", lambda: fake_session_factory)


def test_category_repository_search_matches_category_id_name_and_path(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    repository = CategoryRepository(adapter)  # type: ignore[arg-type]
    synced_at = datetime(2026, 5, 9, tzinfo=UTC)
    rows = flatten_ocs_categories(OCS_CATEGORY_SAMPLE, synced_at=synced_at)

    async def run() -> tuple[list[DistributorCategory], list[DistributorCategory]]:
        for row in rows:
            await repository.upsert_category(row)

        by_category_id = await repository.list_categories(
            distributor_code="ocs",
            search="V110100",
        )
        by_path = await repository.list_categories(
            distributor_code="ocs",
            search="Комплектующие для серверов",
        )
        return by_category_id, by_path

    by_category_id, by_path = asyncio.run(run())

    assert [category.category_id for category in by_category_id] == ["V110100"]
    assert "V110100" in {category.category_id for category in by_path}


def test_sync_ocs_categories_accepts_real_category_field_and_is_idempotent(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)

    result = asyncio.run(
        sync_ocs_categories(adapter, client=FakeOcsClient(OCS_CATEGORY_SAMPLE))  # type: ignore[arg-type]
    )
    second_result = asyncio.run(
        sync_ocs_categories(adapter, client=FakeOcsClient(OCS_CATEGORY_SAMPLE))  # type: ignore[arg-type]
    )

    sync_run = db_session.scalar(select(SyncRun))
    distributor = db_session.scalar(select(Distributor).where(Distributor.code == "ocs"))
    category_count = db_session.scalar(select(func.count()).select_from(DistributorCategory))
    v1100 = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "V1100")
    )
    v110100 = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "V110100")
    )

    assert result.status == "success"
    assert result.categories_processed == 4
    assert second_result.status == "success"
    assert second_result.categories_processed == 4
    assert sync_run is not None
    assert sync_run.status == "success"
    assert sync_run.sync_type == "categories"
    assert sync_run.items_processed == 4
    assert sync_run.finished_at is not None
    assert distributor is not None
    assert distributor.name == "OCS"
    assert distributor.enabled is True
    assert category_count == 4
    assert v1100 is not None
    assert v1100.parent_category_id == "V11"
    assert v110100 is not None
    assert v110100.level == 2


def test_flatten_ocs_categories_error_mentions_missing_category_or_id() -> None:
    with pytest.raises(ValueError, match="OCS category node does not contain category/id"):
        flatten_ocs_categories([{"name": "No code"}])


def test_sync_ocs_categories_marks_sync_run_failed_on_client_error(db_session: Session) -> None:
    adapter = AsyncSessionAdapter(db_session)

    result = asyncio.run(
        sync_ocs_categories(
            adapter,
            client=FakeOcsClient(exc=RuntimeError("client exploded")),
        )  # type: ignore[arg-type]
    )

    sync_run = db_session.scalar(select(SyncRun))
    category_count = db_session.scalar(select(func.count()).select_from(DistributorCategory))

    assert result.status == "failed"
    assert result.categories_processed == 0
    assert result.error_message == "client exploded"
    assert sync_run is not None
    assert sync_run.status == "failed"
    assert sync_run.items_processed == 0
    assert sync_run.error_message == "client exploded"
    assert sync_run.finished_at is not None
    assert category_count == 0
