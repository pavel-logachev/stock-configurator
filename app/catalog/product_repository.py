from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DistributorProduct, DistributorStockPrice


@dataclass(frozen=True)
class ProductUpsert:
    distributor_code: str
    item_id: str
    product_key: str | None
    part_number: str | None
    producer: str | None
    category_id: str | None
    item_name: str | None
    item_name_rus: str | None
    product_name: str | None
    product_description: str | None
    product_notes: str | None
    hscode: str | None
    ean: str | None
    is_in_mpt_registry: bool | None
    is_project_item: bool | None
    traceable: bool | None
    condition: str | None
    warranty: str | None
    original_country_iso_code: str | None
    vat_percent: Decimal | None
    serial_number_availability: str | None
    catalog_path_json: list[dict[str, Any]]
    package_json: dict[str, Any]
    raw_json: dict[str, Any]
    synced_at: datetime


@dataclass(frozen=True)
class StockPriceInsert:
    distributor_code: str
    item_id: str
    product_key: str | None
    shipment_city: str
    location: str | None
    location_description: str | None
    location_type: str | None
    quantity_value: int | None
    quantity_is_greater_than: bool | None
    can_reserve: bool | None
    departure_date: date | None
    arrival_date: date | None
    delivery_date: date | None
    price_order_value: Decimal | None
    price_order_currency: str | None
    price_list_value: Decimal | None
    price_list_currency: str | None
    end_user_value: Decimal | None
    end_user_currency: str | None
    raw_json: dict[str, Any]
    synced_at: datetime


@dataclass(frozen=True)
class FullCategoryMatrixRow:
    product: DistributorProduct
    stock: DistributorStockPrice


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProductRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_product(self, row: ProductUpsert) -> DistributorProduct:
        product = await self._session.scalar(
            select(DistributorProduct).where(
                DistributorProduct.distributor_code == row.distributor_code,
                DistributorProduct.item_id == row.item_id,
            )
        )
        now = _utc_now()

        if product is None:
            product = DistributorProduct(
                distributor_code=row.distributor_code,
                item_id=row.item_id,
                created_at=now,
                updated_at=now,
            )
            self._session.add(product)
        else:
            product.updated_at = now

        product.product_key = row.product_key
        product.part_number = row.part_number
        product.producer = row.producer
        product.category_id = row.category_id
        product.item_name = row.item_name
        product.item_name_rus = row.item_name_rus
        product.product_name = row.product_name
        product.product_description = row.product_description
        product.product_notes = row.product_notes
        product.hscode = row.hscode
        product.ean = row.ean
        product.is_in_mpt_registry = row.is_in_mpt_registry
        product.is_project_item = row.is_project_item
        product.traceable = row.traceable
        product.condition = row.condition
        product.warranty = row.warranty
        product.original_country_iso_code = row.original_country_iso_code
        product.vat_percent = row.vat_percent
        product.serial_number_availability = row.serial_number_availability
        product.catalog_path_json = row.catalog_path_json
        product.package_json = row.package_json
        product.raw_json = row.raw_json
        product.synced_at = row.synced_at

        await self._session.flush()
        return product

    async def insert_stock_price_rows(
        self,
        rows: list[StockPriceInsert],
    ) -> list[DistributorStockPrice]:
        inserted: list[DistributorStockPrice] = []
        now = _utc_now()

        for row in rows:
            stock_price = DistributorStockPrice(
                distributor_code=row.distributor_code,
                item_id=row.item_id,
                product_key=row.product_key,
                shipment_city=row.shipment_city,
                location=row.location,
                location_description=row.location_description,
                location_type=row.location_type,
                quantity_value=row.quantity_value,
                quantity_is_greater_than=row.quantity_is_greater_than,
                can_reserve=row.can_reserve,
                departure_date=row.departure_date,
                arrival_date=row.arrival_date,
                delivery_date=row.delivery_date,
                price_order_value=row.price_order_value,
                price_order_currency=row.price_order_currency,
                price_list_value=row.price_list_value,
                price_list_currency=row.price_list_currency,
                end_user_value=row.end_user_value,
                end_user_currency=row.end_user_currency,
                raw_json=row.raw_json,
                synced_at=row.synced_at,
                created_at=now,
            )
            self._session.add(stock_price)
            inserted.append(stock_price)

        await self._session.flush()
        return inserted

    async def get_latest_product_count(self, distributor_code: str) -> int:
        count = await self._session.scalar(
            select(func.count())
            .select_from(DistributorProduct)
            .where(DistributorProduct.distributor_code == distributor_code)
        )
        return int(count or 0)

    async def get_latest_stock_count(self, distributor_code: str) -> int:
        latest_synced_at = await self._latest_stock_synced_at(distributor_code)
        if latest_synced_at is None:
            return 0

        count = await self._session.scalar(
            select(func.count())
            .select_from(DistributorStockPrice)
            .where(
                DistributorStockPrice.distributor_code == distributor_code,
                DistributorStockPrice.synced_at == latest_synced_at,
            )
        )
        return int(count or 0)

    async def list_recent_products(
        self,
        distributor_code: str,
        *,
        limit: int = 20,
    ) -> list[DistributorProduct]:
        result = await self._session.execute(
            select(DistributorProduct)
            .where(DistributorProduct.distributor_code == distributor_code)
            .order_by(DistributorProduct.synced_at.desc(), DistributorProduct.id.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_latest_stock_for_category(
        self,
        distributor_code: str,
        category_id: str,
    ) -> list[DistributorStockPrice]:
        cleaned_category_id = str(category_id or "").strip()
        if not cleaned_category_id:
            return []
        latest_by_category = self._latest_stock_synced_at_by_category_subquery(
            distributor_code,
            [cleaned_category_id],
        )

        result = await self._session.execute(
            select(DistributorStockPrice)
            .join(
                DistributorProduct,
                (DistributorProduct.distributor_code == DistributorStockPrice.distributor_code)
                & (DistributorProduct.item_id == DistributorStockPrice.item_id),
            )
            .join(
                latest_by_category,
                (DistributorProduct.category_id == latest_by_category.c.category_id)
                & (
                    DistributorStockPrice.synced_at
                    == latest_by_category.c.latest_synced_at
                ),
            )
            .where(
                DistributorStockPrice.distributor_code == distributor_code,
                DistributorProduct.category_id == cleaned_category_id,
            )
            .order_by(
                DistributorStockPrice.item_id,
                DistributorStockPrice.location,
                DistributorStockPrice.id,
            )
        )
        return list(result.scalars().all())

    async def list_latest_full_category_matrix(
        self,
        distributor_code: str,
        category_id: str,
    ) -> list[FullCategoryMatrixRow]:
        return await self.list_latest_full_category_group_matrix(
            distributor_code,
            [category_id],
        )

    async def list_latest_full_category_group_matrix(
        self,
        distributor_code: str,
        category_ids: Sequence[str],
    ) -> list[FullCategoryMatrixRow]:
        cleaned_category_ids = [
            category_id
            for category_id in dict.fromkeys(str(value or "").strip() for value in category_ids)
            if category_id
        ]
        if not cleaned_category_ids:
            return []

        latest_by_category = self._latest_stock_synced_at_by_category_subquery(
            distributor_code,
            cleaned_category_ids,
        )

        result = await self._session.execute(
            select(DistributorProduct, DistributorStockPrice)
            .join(
                DistributorStockPrice,
                (DistributorProduct.distributor_code == DistributorStockPrice.distributor_code)
                & (DistributorProduct.item_id == DistributorStockPrice.item_id),
            )
            .join(
                latest_by_category,
                (DistributorProduct.category_id == latest_by_category.c.category_id)
                & (
                    DistributorStockPrice.synced_at
                    == latest_by_category.c.latest_synced_at
                ),
            )
            .where(
                DistributorProduct.distributor_code == distributor_code,
                DistributorProduct.category_id.in_(cleaned_category_ids),
            )
            .order_by(
                DistributorProduct.category_id,
                DistributorProduct.item_id,
                DistributorStockPrice.location,
                DistributorStockPrice.id,
            )
        )
        return [
            FullCategoryMatrixRow(product=product, stock=stock)
            for product, stock in result.all()
        ]

    async def list_latest_stock_preview_for_categories(
        self,
        distributor_code: str,
        category_ids: Sequence[str],
        *,
        per_category_limit: int = 2,
    ) -> list[dict[str, Any]]:
        cleaned_category_ids = [
            category_id
            for category_id in dict.fromkeys(str(value or "").strip() for value in category_ids)
            if category_id
        ]
        if not cleaned_category_ids or per_category_limit <= 0:
            return []

        latest_by_category = self._latest_stock_synced_at_by_category_subquery(
            distributor_code,
            cleaned_category_ids,
        )
        preview_rank = func.row_number().over(
            partition_by=DistributorProduct.category_id,
            order_by=(
                DistributorStockPrice.price_order_currency,
                DistributorStockPrice.price_order_value,
                DistributorProduct.item_id,
                DistributorStockPrice.location,
                DistributorStockPrice.id,
            ),
        ).label("preview_rank")
        preview_rows = (
            select(
                DistributorProduct.category_id.label("category_id"),
                DistributorProduct.item_id.label("item_id"),
                DistributorProduct.part_number.label("part_number"),
                DistributorProduct.producer.label("producer"),
                DistributorProduct.item_name.label("item_name"),
                DistributorProduct.item_name_rus.label("item_name_rus"),
                DistributorProduct.product_name.label("product_name"),
                DistributorProduct.product_description.label("product_description"),
                DistributorProduct.product_notes.label("product_notes"),
                DistributorStockPrice.price_order_value.label("price_value"),
                DistributorStockPrice.price_order_currency.label("price_currency"),
                DistributorStockPrice.quantity_value.label("quantity_value"),
                DistributorStockPrice.quantity_is_greater_than.label(
                    "quantity_is_greater_than"
                ),
                preview_rank,
            )
            .join(
                DistributorStockPrice,
                (DistributorProduct.distributor_code == DistributorStockPrice.distributor_code)
                & (DistributorProduct.item_id == DistributorStockPrice.item_id),
            )
            .join(
                latest_by_category,
                (DistributorProduct.category_id == latest_by_category.c.category_id)
                & (
                    DistributorStockPrice.synced_at
                    == latest_by_category.c.latest_synced_at
                ),
            )
            .where(
                DistributorProduct.distributor_code == distributor_code,
                DistributorProduct.category_id.in_(cleaned_category_ids),
            )
            .subquery()
        )

        result = await self._session.execute(
            select(preview_rows)
            .where(preview_rows.c.preview_rank <= per_category_limit)
            .order_by(preview_rows.c.category_id, preview_rows.c.preview_rank)
        )
        return [dict(row._mapping) for row in result.all()]

    async def list_latest_stock_counts_by_category(
        self,
        distributor_code: str,
        category_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        cleaned_category_ids = [
            category_id
            for category_id in dict.fromkeys(str(value or "").strip() for value in category_ids)
            if category_id
        ]
        if not cleaned_category_ids:
            return []

        latest_by_category = self._latest_stock_synced_at_by_category_subquery(
            distributor_code,
            cleaned_category_ids,
        )

        result = await self._session.execute(
            select(
                DistributorProduct.category_id.label("category_id"),
                func.count(func.distinct(DistributorProduct.item_id)).label(
                    "position_count"
                ),
                func.count(DistributorStockPrice.id).label("stock_row_count"),
            )
            .join(
                DistributorStockPrice,
                (DistributorProduct.distributor_code == DistributorStockPrice.distributor_code)
                & (DistributorProduct.item_id == DistributorStockPrice.item_id),
            )
            .join(
                latest_by_category,
                (DistributorProduct.category_id == latest_by_category.c.category_id)
                & (
                    DistributorStockPrice.synced_at
                    == latest_by_category.c.latest_synced_at
                ),
            )
            .where(
                DistributorProduct.distributor_code == distributor_code,
                DistributorProduct.category_id.in_(cleaned_category_ids),
            )
            .group_by(DistributorProduct.category_id)
            .order_by(DistributorProduct.category_id)
        )
        return [dict(row._mapping) for row in result.all()]

    async def list_latest_stock_for_item_ids(
        self,
        distributor_code: str,
        item_ids: list[str],
    ) -> list[DistributorStockPrice]:
        if not item_ids:
            return []

        latest_by_item = (
            select(
                DistributorStockPrice.item_id.label("item_id"),
                func.max(DistributorStockPrice.synced_at).label("latest_synced_at"),
            )
            .where(
                DistributorStockPrice.distributor_code == distributor_code,
                DistributorStockPrice.item_id.in_(item_ids),
            )
            .group_by(DistributorStockPrice.item_id)
            .subquery()
        )

        result = await self._session.execute(
            select(DistributorStockPrice)
            .join(
                latest_by_item,
                (DistributorStockPrice.item_id == latest_by_item.c.item_id)
                & (
                    DistributorStockPrice.synced_at
                    == latest_by_item.c.latest_synced_at
                ),
            )
            .where(
                DistributorStockPrice.distributor_code == distributor_code,
                DistributorStockPrice.item_id.in_(item_ids),
            )
            .order_by(
                DistributorStockPrice.item_id,
                DistributorStockPrice.location,
                DistributorStockPrice.id,
            )
        )
        return list(result.scalars().all())

    async def _latest_stock_synced_at(self, distributor_code: str) -> datetime | None:
        return await self._session.scalar(
            select(func.max(DistributorStockPrice.synced_at)).where(
                DistributorStockPrice.distributor_code == distributor_code
            )
        )

    def _latest_stock_synced_at_by_category_subquery(
        self,
        distributor_code: str,
        category_ids: Sequence[str],
    ) -> Any:
        return (
            select(
                DistributorProduct.category_id.label("category_id"),
                func.max(DistributorStockPrice.synced_at).label("latest_synced_at"),
            )
            .join(
                DistributorStockPrice,
                (DistributorProduct.distributor_code == DistributorStockPrice.distributor_code)
                & (DistributorProduct.item_id == DistributorStockPrice.item_id),
            )
            .where(
                DistributorProduct.distributor_code == distributor_code,
                DistributorProduct.category_id.in_(category_ids),
            )
            .group_by(DistributorProduct.category_id)
            .subquery()
        )
