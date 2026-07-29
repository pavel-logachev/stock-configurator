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
from app.core.database import Base
from app.db.models import DistributorProduct, DistributorStockPrice, SyncRun
from app.distributors.ocs.sync_products import sync_ocs_products

OCS_PRODUCT_SAMPLE: list[dict[str, Any]] = [
    {
        "product": {
            "itemId": "1000841882",
            "productKey": "1000841882",
            "partNumber": "D5720-181125SA04",
            "producer": "NERPA",
            "category": "V1100",
            "itemName": "Server NERPA D5720",
            "itemNameRus": "Server",
            "productName": "NERPA D5720-181125SA04",
            "productDescription": None,
            "productNotes": None,
            "warranty": "Distributor warranty 12 months",
            "originalCountryISOCode": "RU",
            "hsCode": "8471490000",
            "eaN128": "04600000000012",
            "isInMPTRegistry": False,
            "condition": "Regular",
        },
        "isAvailableForOrder": True,
        "packageInformation": {
            "weight": 25.0,
            "width": 0.6,
            "height": 0.28,
            "depth": 1.0,
            "volume": 0.168,
            "minOrderQuantity": 1,
            "multiplicity": 1,
            "units": "pcs",
        },
        "price": {
            "priceList": {"value": 6900.0, "currency": "USD"},
            "order": {"value": 6900.0, "currency": "USD"},
            "endUser": {"value": 7100.0, "currency": "USD"},
            "discountB2B": 0,
        },
        "locations": [
            {
                "location": "MSK",
                "description": "Moscow",
                "type": "ShipmentCity",
                "quantity": {"value": 3, "isGreatThan": False},
                "canReserve": True,
                "arrivalDate": None,
                "deliveryDate": "2026-05-08T00:00:00",
            },
            {
                "location": "SPB",
                "description": "Saint Petersburg",
                "type": "ShipmentCity",
                "quantity": {"value": 7, "isGreatThan": True},
                "canReserve": False,
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


class FakeOcsProductClient:
    def __init__(
        self,
        payload_by_category: dict[str, Any] | None = None,
        exc_by_category: dict[str, Exception] | None = None,
    ) -> None:
        self.payload_by_category = payload_by_category or {}
        self.exc_by_category = exc_by_category or {}
        self.calls: list[dict[str, Any]] = []

    async def get_products_by_category(
        self,
        category: str,
        shipment_city: str,
        only_available: bool = True,
        include_regular: bool = True,
        include_sale: bool = False,
        include_uncondition: bool = False,
        include_missing: bool = False,
        with_descriptions: bool = False,
    ) -> Any:
        self.calls.append(
            {
                "category": category,
                "shipment_city": shipment_city,
                "only_available": only_available,
                "include_regular": include_regular,
                "include_sale": include_sale,
                "include_uncondition": include_uncondition,
                "include_missing": include_missing,
                "with_descriptions": with_descriptions,
            }
        )
        if category in self.exc_by_category:
            raise self.exc_by_category[category]
        return self.payload_by_category.get(category, [])


@pytest.fixture()
def db_session() -> Generator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        yield session

    engine.dispose()


def test_sync_ocs_products_uses_only_enabled_categories_and_saves_product_stock(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="V1100", enabled_for_sync=True)
    _seed_category(db_session, category_id="V1101", enabled_for_sync=False)
    client = FakeOcsProductClient(
        {
            "V1100": {"products": OCS_PRODUCT_SAMPLE},
            "V1101": [{"itemId": "SHOULD-NOT-SYNC"}],
        }
    )

    result = asyncio.run(sync_ocs_products(adapter, client=client))  # type: ignore[arg-type]
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
    assert [call["category"] for call in client.calls] == ["V1100"]
    assert client.calls[0]["only_available"] is True
    assert client.calls[0]["include_regular"] is True
    assert client.calls[0]["include_sale"] is False
    assert client.calls[0]["include_uncondition"] is False
    assert client.calls[0]["include_missing"] is False
    assert client.calls[0]["with_descriptions"] is False
    assert product is not None
    assert product.item_id == "1000841882"
    assert product.product_key == "1000841882"
    assert product.part_number == "D5720-181125SA04"
    assert product.producer == "NERPA"
    assert product.category_id == "V1100"
    assert product.item_name == "Server NERPA D5720"
    assert product.hscode == "8471490000"
    assert product.ean == "04600000000012"
    assert product.is_in_mpt_registry is False
    assert product.original_country_iso_code == "RU"
    assert product.package_json["weight"] == 25.0
    assert product.raw_json["product"]["itemId"] == "1000841882"
    assert product.raw_json["isAvailableForOrder"] is True
    assert len(stock_rows) == 2
    assert {row.location for row in stock_rows} == {"MSK", "SPB"}
    assert all(row.price_order_value == Decimal("6900.0000") for row in stock_rows)
    assert all(row.price_order_currency == "USD" for row in stock_rows)
    assert all(row.price_list_value == Decimal("6900.0000") for row in stock_rows)
    assert all(row.price_list_currency == "USD" for row in stock_rows)
    assert all(row.end_user_value == Decimal("7100.0000") for row in stock_rows)
    assert stock_rows[0].delivery_date == date(2026, 5, 8)
    assert [row.quantity_value for row in stock_rows] == [3, 7]
    assert [row.quantity_is_greater_than for row in stock_rows] == [False, True]
    assert [row.can_reserve for row in stock_rows] == [True, False]
    assert stock_rows[0].raw_json["productKey"] == "1000841882"
    assert sync_run is not None
    assert sync_run.status == "success"
    assert sync_run.items_processed == 1
    assert sync_run.finished_at is not None


def test_second_sync_upserts_products_and_adds_stock_snapshot_rows(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="V1100", enabled_for_sync=True)
    client = FakeOcsProductClient({"V1100": OCS_PRODUCT_SAMPLE})

    first_result = asyncio.run(sync_ocs_products(adapter, client=client))  # type: ignore[arg-type]
    second_result = asyncio.run(sync_ocs_products(adapter, client=client))  # type: ignore[arg-type]

    product_count = db_session.scalar(select(func.count()).select_from(DistributorProduct))
    stock_count = db_session.scalar(select(func.count()).select_from(DistributorStockPrice))
    sync_run_count = db_session.scalar(
        select(func.count()).select_from(SyncRun).where(SyncRun.sync_type == "products")
    )

    assert first_result.status == "success"
    assert second_result.status == "success"
    assert product_count == 1
    assert stock_count == 4
    assert sync_run_count == 2


def test_sync_ocs_products_can_refresh_selected_categories_without_enabled_flag(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="V1100", enabled_for_sync=False)
    _seed_category(db_session, category_id="V1101", enabled_for_sync=True)
    client = FakeOcsProductClient(
        {
            "V1100": {"products": OCS_PRODUCT_SAMPLE},
            "V1101": [{"itemId": "SHOULD-NOT-SYNC"}],
        }
    )

    result = asyncio.run(
        sync_ocs_products(
            adapter,
            client=client,
            category_ids=["V1100"],
        )  # type: ignore[arg-type]
    )

    product = db_session.scalar(select(DistributorProduct))

    assert result.status == "success"
    assert result.enabled_categories == 1
    assert result.products_processed == 1
    assert result.stock_rows_inserted == 2
    assert [call["category"] for call in client.calls] == ["V1100"]
    assert product is not None
    assert product.category_id == "V1100"


def test_sync_ocs_products_returns_clear_error_when_no_categories_enabled(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="V1100", enabled_for_sync=False)
    client = FakeOcsProductClient({"V1100": OCS_PRODUCT_SAMPLE})

    result = asyncio.run(sync_ocs_products(adapter, client=client))  # type: ignore[arg-type]

    sync_run = db_session.scalar(select(SyncRun).where(SyncRun.sync_type == "products"))
    product_count = db_session.scalar(select(func.count()).select_from(DistributorProduct))

    assert result.status == "failed"
    assert result.enabled_categories == 0
    assert "No OCS categories are enabled for product sync" in (result.error_message or "")
    assert client.calls == []
    assert sync_run is not None
    assert sync_run.status == "failed"
    assert sync_run.items_processed == 0
    assert product_count == 0


def test_sync_ocs_products_fails_with_clear_error_without_product_item_id(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="V1100", enabled_for_sync=True)
    client = FakeOcsProductClient(
        {
            "V1100": [
                {
                    "product": {"productKey": "missing-item-id"},
                    "price": {"order": {"value": 10, "currency": "USD"}},
                    "locations": [],
                }
            ]
        }
    )

    result = asyncio.run(sync_ocs_products(adapter, client=client))  # type: ignore[arg-type]

    sync_run = db_session.scalar(select(SyncRun).where(SyncRun.sync_type == "products"))
    product_count = db_session.scalar(select(func.count()).select_from(DistributorProduct))
    stock_count = db_session.scalar(select(func.count()).select_from(DistributorStockPrice))

    assert result.status == "failed"
    assert result.products_processed == 0
    assert result.error_message == "OCS product wrapper does not contain product.itemId"
    assert sync_run is not None
    assert sync_run.status == "failed"
    assert sync_run.error_message == "OCS product wrapper does not contain product.itemId"
    assert product_count == 0
    assert stock_count == 0


def test_sync_ocs_products_marks_sync_run_failed_on_category_error(
    db_session: Session,
) -> None:
    adapter = AsyncSessionAdapter(db_session)
    _seed_category(db_session, category_id="V1100", enabled_for_sync=True)
    client = FakeOcsProductClient(exc_by_category={"V1100": RuntimeError("OCS exploded")})

    result = asyncio.run(sync_ocs_products(adapter, client=client))  # type: ignore[arg-type]

    sync_run = db_session.scalar(select(SyncRun).where(SyncRun.sync_type == "products"))
    product_count = db_session.scalar(select(func.count()).select_from(DistributorProduct))
    stock_count = db_session.scalar(select(func.count()).select_from(DistributorStockPrice))

    assert result.status == "failed"
    assert result.enabled_categories == 1
    assert result.products_processed == 0
    assert result.error_message == "OCS exploded"
    assert sync_run is not None
    assert sync_run.status == "failed"
    assert sync_run.error_message == "OCS exploded"
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


async def _latest_counts(adapter: AsyncSessionAdapter) -> tuple[int, int]:
    repository = ProductRepository(adapter)  # type: ignore[arg-type]
    return (
        await repository.get_latest_product_count("ocs"),
        await repository.get_latest_stock_count("ocs"),
    )
