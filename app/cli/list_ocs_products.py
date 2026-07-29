from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from decimal import Decimal

from app.catalog.product_repository import ProductRepository
from app.core.database import get_session_factory
from app.db.models import DistributorStockPrice

DISTRIBUTOR_CODE = "ocs"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List saved OCS products.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to print.")
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    limit = max(args.limit, 1)
    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = ProductRepository(session)
        products = await repository.list_recent_products(DISTRIBUTOR_CODE, limit=limit)
        stock_rows = await repository.list_latest_stock_for_item_ids(
            DISTRIBUTOR_CODE,
            [product.item_id for product in products],
        )

    print("OCS products")
    if not products:
        print("No products found.")
        return 0

    stock_by_item_id: dict[str, list[DistributorStockPrice]] = {}
    for row in stock_rows:
        stock_by_item_id.setdefault(row.item_id, []).append(row)

    print("item_id\tpart_number\tproducer\tcategory_id\titem_name\tprice/order\tavailable")
    for product in products:
        item_stock_rows = stock_by_item_id.get(product.item_id, [])
        print(
            "\t".join(
                [
                    product.item_id,
                    product.part_number or "",
                    product.producer or "",
                    product.category_id or "",
                    product.item_name or "",
                    _price_order(item_stock_rows),
                    _available_summary(item_stock_rows),
                ]
            )
        )

    return 0


def _price_order(rows: list[DistributorStockPrice]) -> str:
    for row in rows:
        if row.price_order_value is None:
            continue
        value = _format_decimal(row.price_order_value)
        if row.price_order_currency:
            return f"{value} {row.price_order_currency}"
        return value
    return ""


def _available_summary(rows: list[DistributorStockPrice]) -> str:
    if not rows:
        return ""

    quantities = [row.quantity_value for row in rows if row.quantity_value is not None]
    total = sum(quantities) if quantities else None
    suffix = "+" if any(row.quantity_is_greater_than for row in rows) else ""
    reservable = sum(1 for row in rows if row.can_reserve is True)
    locations = len(rows)

    if total is None:
        return f"{locations} locations, {reservable} reservable"
    return f"{total}{suffix} units, {locations} locations, {reservable} reservable"


def _format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f")


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
