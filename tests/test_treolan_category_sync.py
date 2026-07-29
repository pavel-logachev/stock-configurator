from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.core.database import Base
from app.db.models import Distributor, DistributorCategory, SyncRun
from app.distributors.treolan.parsing import flatten_treolan_categories
from app.distributors.treolan.sync_categories import sync_treolan_categories

TREOLAN_CATEGORY_XML = """
<catalog>
  <category id="A" name="Servers">
    <category id="A-CPU" name="Processors" />
    <category id="A-RAM" name="Memory">
      <position id="P-1" name="DIMM" />
    </category>
  </category>
</catalog>
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


class FakeTreolanCategoriesClient:
    def __init__(self, payload: str | None = None, exc: Exception | None = None) -> None:
        self._payload = payload or TREOLAN_CATEGORY_XML
        self._exc = exc

    async def get_categories(self) -> str:
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


def test_flatten_treolan_categories_maps_nested_tree_without_positions() -> None:
    rows = flatten_treolan_categories(TREOLAN_CATEGORY_XML)

    assert [row.category_id for row in rows] == ["A", "A-CPU", "A-RAM"]
    assert [row.parent_category_id for row in rows] == [None, "A", "A"]
    assert [row.level for row in rows] == [0, 1, 1]
    assert rows[2].path_json == [
        {"category_id": "A", "name": "Servers"},
        {"category_id": "A-RAM", "name": "Memory"},
    ]
    assert rows[0].raw_json["position_count"] == 0
    assert rows[2].raw_json["position_count"] == 1


def test_sync_treolan_categories_saves_distributor_and_categories(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)

    result = asyncio.run(
        sync_treolan_categories(adapter, client=FakeTreolanCategoriesClient())  # type: ignore[arg-type]
    )

    distributor = db_session.scalar(select(Distributor).where(Distributor.code == "treolan"))
    category_count = db_session.scalar(select(func.count()).select_from(DistributorCategory))
    sync_run = db_session.scalar(select(SyncRun))
    ram_category = db_session.scalar(
        select(DistributorCategory).where(DistributorCategory.category_id == "A-RAM")
    )

    assert result.status == "success"
    assert result.categories_processed == 3
    assert distributor is not None
    assert distributor.name == "Treolan"
    assert category_count == 3
    assert ram_category is not None
    assert ram_category.parent_category_id == "A"
    assert sync_run is not None
    assert sync_run.status == "success"
    assert sync_run.items_processed == 3


def test_sync_treolan_categories_marks_sync_run_failed_on_client_error(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)

    result = asyncio.run(
        sync_treolan_categories(
            adapter,
            client=FakeTreolanCategoriesClient(exc=RuntimeError("Treolan exploded")),
        )  # type: ignore[arg-type]
    )

    sync_run = db_session.scalar(select(SyncRun))
    category_count = db_session.scalar(select(func.count()).select_from(DistributorCategory))

    assert result.status == "failed"
    assert result.categories_processed == 0
    assert result.error_message == "Treolan exploded"
    assert sync_run is not None
    assert sync_run.status == "failed"
    assert category_count == 0
