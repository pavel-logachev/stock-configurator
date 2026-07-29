from __future__ import annotations

import asyncio
from collections.abc import Generator
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.catalog.category_repository import CategoryRepository, CategoryUpsert
from app.catalog.product_repository import ProductRepository
from app.core.config import TreolanSettings
from app.core.database import Base
from app.db.models import DistributorProduct, DistributorStockPrice, SyncRun
from app.distributors.treolan.parsing import extract_treolan_positions
from app.distributors.treolan.sync_products import sync_treolan_products

TREOLAN_PRODUCT_XML = """
<catalog>
  <category id="S-CPU" name="Processors">
    <position
      id="033002/187"
      prid="123456"
      articul="BX8071514500"
      name="Intel Core i5 processor"
      rusDescr="Intel Core i5 boxed processor"
      vendor="Intel"
      vendor-id="10"
      gp="36 months"
      price="150.50"
      dprice="140.25"
      currency="USD"
      discount="7"
      outoftrade="false"
      uchmark="false"
      sale="false"
      freenom="3"
      freeptrans="2"
      ntdate="2026-07-01"
      ntstatus="incoming"
      width="10"
      length="12"
      height="5"
      brutto="0.5"
      GTIN="5032037000000"
      isTraceable="true"
      codeTNVED="8542319000"
    />
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


class FakeTreolanProductsClient:
    def __init__(
        self,
        payload_by_category: dict[str, str] | None = None,
        exc_by_category: dict[str, Exception] | None = None,
    ) -> None:
        self.payload_by_category = payload_by_category or {}
        self.exc_by_category = exc_by_category or {}
        self.calls: list[dict[str, Any]] = []

    async def gen_catalog_v2(
        self,
        *,
        category: str = "",
        vendorid: str = "0",
        keywords: str = "",
        criterion: int | None = None,
        in_articul: bool = True,
        in_name: bool = True,
        in_mark: bool = False,
        show_nc: int | None = None,
        free_nom: bool | None = None,
    ) -> str:
        self.calls.append(
            {
                "category": category,
                "vendorid": vendorid,
                "keywords": keywords,
                "criterion": criterion,
                "in_articul": in_articul,
                "in_name": in_name,
                "in_mark": in_mark,
                "show_nc": show_nc,
                "free_nom": free_nom,
            }
        )
        if category in self.exc_by_category:
            raise self.exc_by_category[category]
        return self.payload_by_category.get(category, "<catalog />")


@pytest.fixture()
def db_session() -> Generator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        yield session

    engine.dispose()


def test_extract_treolan_positions_keeps_full_position_attributes() -> None:
    nodes = extract_treolan_positions(TREOLAN_PRODUCT_XML)

    assert len(nodes) == 1
    wrapper = nodes[0].to_wrapper()
    position = wrapper["position"]
    assert position["id"] == "033002/187"
    assert position["articul"] == "BX8071514500"
    assert position["rusDescr"] == "Intel Core i5 boxed processor"
    assert wrapper["category_id"] == "S-CPU"
    assert wrapper["category_path_json"] == [
        {"category_id": "S-CPU", "name": "Processors"}
    ]


def test_sync_treolan_products_saves_product_and_stock_rows(db_session: Session) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="S-CPU", enabled_for_sync=True)
    _seed_category(db_session, category_id="S-RAM", enabled_for_sync=False)
    client = FakeTreolanProductsClient(
        {
            "S-CPU": TREOLAN_PRODUCT_XML,
            "S-RAM": "<catalog><position id='SHOULD-NOT-SYNC' /></catalog>",
        }
    )
    settings = TreolanSettings(
        treolan_login="login",
        treolan_password="password",
        treolan_shipment_city="Treolan",
    )

    result = asyncio.run(
        sync_treolan_products(adapter, client=client, settings=settings)  # type: ignore[arg-type]
    )
    latest_product_count, latest_stock_count = asyncio.run(_latest_counts(adapter))

    product = db_session.scalar(select(DistributorProduct))
    stock_rows = list(
        db_session.scalars(
            select(DistributorStockPrice).order_by(DistributorStockPrice.location)
        )
    )
    sync_run = db_session.scalar(select(SyncRun).where(SyncRun.sync_type == "products"))

    assert result.status == "success"
    assert result.enabled_categories == 1
    assert result.products_processed == 1
    assert result.stock_rows_inserted == 2
    assert latest_product_count == 1
    assert latest_stock_count == 2
    assert [call["category"] for call in client.calls] == ["S-CPU"]
    assert product is not None
    assert product.distributor_code == "treolan"
    assert product.item_id == "033002/187"
    assert product.product_key == "123456"
    assert product.part_number == "BX8071514500"
    assert product.producer == "Intel"
    assert product.category_id == "S-CPU"
    assert product.item_name == "Intel Core i5 processor"
    assert product.item_name_rus == "Intel Core i5 boxed processor"
    assert product.hscode == "8542319000"
    assert product.ean == "5032037000000"
    assert product.traceable is True
    assert product.warranty == "36 months"
    assert product.package_json["brutto"] == "0.5"
    assert product.raw_json["position"]["vendor-id"] == "10"
    assert {"treolan.articul", "treolan.rusDescr"}.issubset(
        {item["name"] for item in product.raw_json["content_properties"]}
    )
    assert {row.location for row in stock_rows} == {"stock", "transit"}
    assert [row.quantity_value for row in stock_rows] == [3, 2]
    assert all(row.price_order_value == Decimal("140.2500") for row in stock_rows)
    assert all(row.price_order_currency == "USD" for row in stock_rows)
    assert all(row.price_list_value == Decimal("150.5000") for row in stock_rows)
    assert stock_rows[0].can_reserve is True
    assert stock_rows[1].can_reserve is False
    assert stock_rows[1].arrival_date == date(2026, 7, 1)
    assert sync_run is not None
    assert sync_run.status == "success"
    assert sync_run.items_processed == 1


def test_sync_treolan_products_can_refresh_selected_categories_without_enabled_flag(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="S-CPU", enabled_for_sync=False)
    _seed_category(db_session, category_id="S-RAM", enabled_for_sync=True)
    client = FakeTreolanProductsClient(
        {
            "S-CPU": TREOLAN_PRODUCT_XML,
            "S-RAM": "<catalog><position id='SHOULD-NOT-SYNC' /></catalog>",
        }
    )
    settings = TreolanSettings(
        treolan_login="login",
        treolan_password="password",
        treolan_shipment_city="Treolan",
    )

    result = asyncio.run(
        sync_treolan_products(
            adapter,
            client=client,
            settings=settings,
            category_ids=["S-CPU"],
        )  # type: ignore[arg-type]
    )

    product = db_session.scalar(select(DistributorProduct))

    assert result.status == "success"
    assert result.enabled_categories == 1
    assert result.products_processed == 1
    assert result.stock_rows_inserted == 2
    assert [call["category"] for call in client.calls] == ["S-CPU"]
    assert product is not None
    assert product.category_id == "S-CPU"


def test_sync_treolan_products_returns_clear_error_when_no_categories_enabled(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="S-CPU", enabled_for_sync=False)
    client = FakeTreolanProductsClient({"S-CPU": TREOLAN_PRODUCT_XML})

    result = asyncio.run(
        sync_treolan_products(adapter, client=client)  # type: ignore[arg-type]
    )

    sync_run = db_session.scalar(select(SyncRun).where(SyncRun.sync_type == "products"))
    product_count = db_session.scalar(select(func.count()).select_from(DistributorProduct))

    assert result.status == "failed"
    assert result.enabled_categories == 0
    assert "No Treolan categories are enabled for product sync" in (
        result.error_message or ""
    )
    assert client.calls == []
    assert sync_run is not None
    assert sync_run.status == "failed"
    assert product_count == 0


def test_sync_treolan_products_marks_sync_run_failed_on_category_error(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="S-CPU", enabled_for_sync=True)
    client = FakeTreolanProductsClient(
        exc_by_category={"S-CPU": RuntimeError("Treolan exploded")}
    )

    result = asyncio.run(
        sync_treolan_products(adapter, client=client)  # type: ignore[arg-type]
    )

    sync_run = db_session.scalar(select(SyncRun).where(SyncRun.sync_type == "products"))
    product_count = db_session.scalar(select(func.count()).select_from(DistributorProduct))
    stock_count = db_session.scalar(select(func.count()).select_from(DistributorStockPrice))

    assert result.status == "failed"
    assert result.enabled_categories == 1
    assert result.products_processed == 0
    assert result.error_message == "Treolan exploded"
    assert sync_run is not None
    assert sync_run.status == "failed"
    assert product_count == 0
    assert stock_count == 0


def _seed_category(
    db_session: Session,
    *,
    category_id: str,
    enabled_for_sync: bool,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    repository = CategoryRepository(adapter)  # type: ignore[arg-type]
    synced_at = datetime(2026, 6, 16, tzinfo=UTC)

    async def run() -> None:
        await repository.upsert_category(
            CategoryUpsert(
                distributor_code="treolan",
                category_id=category_id,
                parent_category_id=None,
                name="Treolan test category",
                level=0,
                path_json=[{"category_id": category_id, "name": "Treolan test category"}],
                raw_json={"category": category_id, "name": "Treolan test category"},
                synced_at=synced_at,
                enabled_for_sync=enabled_for_sync,
            )
        )

    asyncio.run(run())


async def _latest_counts(adapter: AsyncSessionAdapter) -> tuple[int, int]:
    repository = ProductRepository(adapter)  # type: ignore[arg-type]
    return (
        await repository.get_latest_product_count("treolan"),
        await repository.get_latest_stock_count("treolan"),
    )
