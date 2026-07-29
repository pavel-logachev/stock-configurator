from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.category_repository import CategoryRepository
from app.catalog.product_repository import ProductRepository, ProductUpsert, StockPriceInsert
from app.core.config import TreolanSettings, get_treolan_settings
from app.distributors.treolan.client import TreolanClient
from app.distributors.treolan.parsing import extract_treolan_positions

DISTRIBUTOR_CODE = "treolan"
DISTRIBUTOR_NAME = "Treolan"
SYNC_TYPE = "products"


class TreolanProductsClient(Protocol):
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
        pass


class NoEnabledTreolanCategoriesError(ValueError):
    """Raised when product sync has no enabled Treolan categories to process."""


@dataclass(frozen=True)
class ProductSyncResult:
    distributor: str
    status: str
    enabled_categories: int
    products_processed: int
    stock_rows_inserted: int
    sync_run_id: int
    error_message: str | None = None


@dataclass(frozen=True)
class SyncCategory:
    category_id: str


async def sync_treolan_products(
    session: AsyncSession,
    *,
    client: TreolanProductsClient | None = None,
    settings: TreolanSettings | None = None,
    category_repository: CategoryRepository | None = None,
    product_repository: ProductRepository | None = None,
    category_ids: Sequence[str] | None = None,
) -> ProductSyncResult:
    treolan_settings = settings or get_treolan_settings()
    category_repo = category_repository or CategoryRepository(session)
    product_repo = product_repository or ProductRepository(session)

    await category_repo.ensure_distributor(
        code=DISTRIBUTOR_CODE,
        name=DISTRIBUTOR_NAME,
        enabled=True,
    )
    sync_run = await category_repo.start_sync_run(
        distributor_code=DISTRIBUTOR_CODE,
        sync_type=SYNC_TYPE,
    )
    sync_run_id = sync_run.id
    await session.commit()

    enabled_categories_count = 0
    products_processed = 0
    stock_rows_inserted = 0

    try:
        enabled_categories = await _target_categories(
            category_repo,
            category_ids=category_ids,
        )
        enabled_categories_count = len(enabled_categories)
        if not enabled_categories:
            message = (
                "No Treolan categories are enabled for product sync. "
                "Enable at least one category with enable_treolan_category."
                if category_ids is None
                else "No Treolan categories were selected for product sync."
            )
            raise NoEnabledTreolanCategoriesError(message)

        if client is None:
            async with TreolanClient(settings=treolan_settings) as treolan_client:
                products_processed, stock_rows_inserted = await _sync_enabled_categories(
                    enabled_categories,
                    client=treolan_client,
                    product_repository=product_repo,
                    settings=treolan_settings,
                )
        else:
            products_processed, stock_rows_inserted = await _sync_enabled_categories(
                enabled_categories,
                client=client,
                product_repository=product_repo,
                settings=treolan_settings,
            )

        await category_repo.finish_sync_run(
            sync_run_id,
            status="success",
            items_processed=products_processed,
        )
        await session.commit()
        return ProductSyncResult(
            distributor=DISTRIBUTOR_CODE,
            status="success",
            enabled_categories=enabled_categories_count,
            products_processed=products_processed,
            stock_rows_inserted=stock_rows_inserted,
            sync_run_id=sync_run_id,
        )
    except Exception as exc:
        await session.rollback()
        error_message = _error_message(exc)
        await category_repo.finish_sync_run(
            sync_run_id,
            status="failed",
            items_processed=products_processed,
            error_message=error_message,
        )
        await session.commit()
        return ProductSyncResult(
            distributor=DISTRIBUTOR_CODE,
            status="failed",
            enabled_categories=enabled_categories_count,
            products_processed=products_processed,
            stock_rows_inserted=stock_rows_inserted,
            sync_run_id=sync_run_id,
            error_message=error_message,
        )


async def _target_categories(
    category_repository: CategoryRepository,
    *,
    category_ids: Sequence[str] | None,
) -> list[Any]:
    if category_ids is None:
        return await category_repository.list_enabled_categories(DISTRIBUTOR_CODE)

    cleaned_category_ids = [
        category_id
        for category_id in dict.fromkeys(str(value or "").strip() for value in category_ids)
        if category_id
    ]
    return [SyncCategory(category_id=category_id) for category_id in cleaned_category_ids]


async def _sync_enabled_categories(
    enabled_categories: Sequence[Any],
    *,
    client: TreolanProductsClient,
    product_repository: ProductRepository,
    settings: TreolanSettings,
) -> tuple[int, int]:
    synced_at = datetime.now(UTC)
    products_processed = 0
    stock_rows_inserted = 0

    for category in enabled_categories:
        category_id = str(category.category_id)
        payload = await client.gen_catalog_v2(category=category_id)
        for node in extract_treolan_positions(
            payload,
            fallback_category_id=category_id,
        ):
            wrapper = node.to_wrapper(fallback_category_id=category_id)
            product_row = build_product_upsert(
                wrapper,
                fallback_category_id=category_id,
                synced_at=synced_at,
            )
            stock_rows = build_stock_price_rows(
                wrapper,
                item_id=product_row.item_id,
                product_key=product_row.product_key,
                shipment_city=settings.treolan_shipment_city,
                synced_at=synced_at,
            )

            await product_repository.upsert_product(product_row)
            await product_repository.insert_stock_price_rows(stock_rows)
            products_processed += 1
            stock_rows_inserted += len(stock_rows)

    return products_processed, stock_rows_inserted


def build_product_upsert(
    wrapper: Mapping[str, Any],
    *,
    fallback_category_id: str,
    synced_at: datetime,
) -> ProductUpsert:
    position = _position_from_wrapper(wrapper)
    item_id = _first_string(position, ("id",))
    if item_id is None:
        raise ValueError("Treolan position does not contain id")

    category_path = _category_path(wrapper)
    raw_json = _jsonable_dict(
        {
            **dict(wrapper),
            "content_properties": _content_properties(position),
        }
    )

    return ProductUpsert(
        distributor_code=DISTRIBUTOR_CODE,
        item_id=item_id,
        product_key=_first_string(position, ("prid", "productKey", "product_key")),
        part_number=_first_string(position, ("articul", "partNumber", "part_number")),
        producer=_first_string(position, ("vendor", "producer", "brand", "manufacturer")),
        category_id=_first_string(wrapper, ("category_id",)) or fallback_category_id,
        item_name=_first_string(position, ("name",)),
        item_name_rus=_first_string(position, ("rusDescr", "rusdescr", "description")),
        product_name=_first_string(position, ("name",)),
        product_description=_first_string(position, ("rusDescr", "rusdescr", "description")),
        product_notes=_product_notes(position),
        hscode=_first_string(position, ("codeTNVED", "codetnved", "hscode", "hsCode")),
        ean=_first_string(position, ("GTIN", "gtin", "ean")),
        is_in_mpt_registry=None,
        is_project_item=None,
        traceable=_first_bool(position, ("isTraceable", "istraceable", "traceable")),
        condition=_condition(position),
        warranty=_first_string(position, ("gp", "warranty")),
        original_country_iso_code=None,
        vat_percent=None,
        serial_number_availability=None,
        catalog_path_json=category_path,
        package_json=_package_json(position),
        raw_json=raw_json,
        synced_at=synced_at,
    )


def build_stock_price_rows(
    wrapper: Mapping[str, Any],
    *,
    item_id: str,
    product_key: str | None,
    shipment_city: str,
    synced_at: datetime,
) -> list[StockPriceInsert]:
    position = _position_from_wrapper(wrapper)
    price_order = _to_decimal(_first_value(position, ("dprice", "price")))
    price_list = _to_decimal(_first_value(position, ("price",)))
    currency = _first_string(position, ("currency",))
    rows: list[StockPriceInsert] = []

    stock_quantity = _to_int(_first_value(position, ("freenom", "freeNom")))
    if stock_quantity is not None and stock_quantity > 0:
        rows.append(
            _stock_row(
                position,
                item_id=item_id,
                product_key=product_key,
                shipment_city=shipment_city,
                location="stock",
                location_description="Treolan available stock",
                location_type="stock",
                quantity_value=stock_quantity,
                can_reserve=True,
                arrival_date=None,
                price_order=price_order,
                price_list=price_list,
                currency=currency,
                synced_at=synced_at,
            )
        )

    transit_quantity = _to_int(_first_value(position, ("freeptrans", "freePtrans")))
    if transit_quantity is not None and transit_quantity > 0:
        rows.append(
            _stock_row(
                position,
                item_id=item_id,
                product_key=product_key,
                shipment_city=shipment_city,
                location="transit",
                location_description=_first_string(position, ("ntstatus", "ntStatus"))
                or "Treolan transit",
                location_type="transit",
                quantity_value=transit_quantity,
                can_reserve=False,
                arrival_date=_to_date(_first_value(position, ("ntdate", "ntDate"))),
                price_order=price_order,
                price_list=price_list,
                currency=currency,
                synced_at=synced_at,
            )
        )

    return rows


def _stock_row(
    position: Mapping[str, Any],
    *,
    item_id: str,
    product_key: str | None,
    shipment_city: str,
    location: str,
    location_description: str,
    location_type: str,
    quantity_value: int | None,
    can_reserve: bool,
    arrival_date: date | None,
    price_order: Decimal | None,
    price_list: Decimal | None,
    currency: str | None,
    synced_at: datetime,
) -> StockPriceInsert:
    return StockPriceInsert(
        distributor_code=DISTRIBUTOR_CODE,
        item_id=item_id,
        product_key=product_key,
        shipment_city=shipment_city,
        location=location,
        location_description=location_description,
        location_type=location_type,
        quantity_value=quantity_value,
        quantity_is_greater_than=False,
        can_reserve=can_reserve,
        departure_date=None,
        arrival_date=arrival_date,
        delivery_date=None,
        price_order_value=price_order,
        price_order_currency=currency,
        price_list_value=price_list,
        price_list_currency=currency,
        end_user_value=None,
        end_user_currency=None,
        raw_json=_jsonable_dict(
            {
                "position": dict(position),
                "productKey": product_key,
                "itemId": item_id,
                "stock_source": location,
            }
        ),
        synced_at=synced_at,
    )


def _position_from_wrapper(wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    position = wrapper.get("position")
    if isinstance(position, Mapping):
        return position
    return wrapper


def _category_path(wrapper: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = wrapper.get("category_path_json")
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    path: list[dict[str, Any]] = []
    for node in value:
        if isinstance(node, Mapping):
            path.append(_jsonable_dict(node))
        else:
            path.append({"value": node})
    return path


def _package_json(position: Mapping[str, Any]) -> dict[str, Any]:
    package: dict[str, Any] = {}
    for key in (
        "width",
        "length",
        "height",
        "brutto",
        "vendor-id",
        "vendor_id",
        "discount",
        "sale",
        "uchmark",
        "outoftrade",
        "ntdate",
        "ntstatus",
        "currency",
    ):
        if key in position and str(position[key] or "").strip():
            package[key] = position[key]
    return _jsonable_dict(package)


def _content_properties(position: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key, value in position.items():
        text = _string_or_none(value)
        if text is None:
            continue
        result.append({"name": f"treolan.{key}", "value": text})
    return result


def _product_notes(position: Mapping[str, Any]) -> str | None:
    notes: list[str] = []
    for key in ("sale", "uchmark", "outoftrade", "ntstatus"):
        value = _string_or_none(position.get(key))
        if value is not None:
            notes.append(f"{key}: {value}")
    return "; ".join(notes) if notes else None


def _condition(position: Mapping[str, Any]) -> str | None:
    out_of_trade = _first_bool(position, ("outoftrade", "outOfTrade"))
    if out_of_trade is True:
        return "out_of_trade"
    sale = _first_bool(position, ("sale",))
    if sale is True:
        return "sale"
    return None


def _first_string(node: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    return _string_or_none(_first_value(node, keys))


def _first_bool(node: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    return _to_bool(_first_value(node, keys))


def _first_value(node: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in node:
            return node[key]
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None or isinstance(value, Mapping | Sequence) and not isinstance(value, str):
        return None
    text = str(value).strip()
    return text or None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "да"}:
        return True
    if normalized in {"false", "0", "no", "n", "нет"}:
        return False
    return None


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, str) and not value.strip():
        return None

    try:
        return Decimal(str(value).replace(",", "."))
    except (InvalidOperation, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        return int(value)
    match = re.search(r"-?\d+", str(value).replace(" ", ""))
    if match is None:
        return None
    return int(match.group(0))


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            try:
                return datetime.strptime(text[:10], fmt).date()
            except ValueError:
                continue
    return None


def _jsonable_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))


def _error_message(exc: Exception) -> str:
    return str(exc)[:2000]
