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
from app.core.config import OcsSettings, get_ocs_settings
from app.distributors.ocs.client import OcsClient

DISTRIBUTOR_CODE = "ocs"
DISTRIBUTOR_NAME = "OCS"
SYNC_TYPE = "products"

PRODUCT_ROOT_KEYS = ("products", "items", "data", "result")
ITEM_ID_KEYS = ("itemId", "item_id", "itemid", "id")
PRODUCT_KEY_KEYS = ("productKey", "product_key", "productkey")
PART_NUMBER_KEYS = ("partNumber", "part_number", "partnumber")
PRODUCER_KEYS = ("producer", "brand", "vendor", "manufacturer")
CATEGORY_KEYS = ("category", "categoryId", "category_id", "categoryid")
CATEGORY_ID_KEYS = ("category", "categoryId", "category_id", "categoryid", "id", "code")
LOCATION_KEYS = ("locations", "stock", "stocks", "warehouses")


class OcsProductsClient(Protocol):
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
        pass


class NoEnabledOcsCategoriesError(ValueError):
    """Raised when product sync has no enabled OCS categories to process."""


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


async def sync_ocs_products(
    session: AsyncSession,
    *,
    client: OcsProductsClient | None = None,
    settings: OcsSettings | None = None,
    category_repository: CategoryRepository | None = None,
    product_repository: ProductRepository | None = None,
    category_ids: Sequence[str] | None = None,
    only_available: bool = True,
    with_descriptions: bool = False,
) -> ProductSyncResult:
    ocs_settings = settings or get_ocs_settings()
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
                "No OCS categories are enabled for product sync. "
                "Enable at least one category with enable_ocs_category."
                if category_ids is None
                else "No OCS categories were selected for product sync."
            )
            raise NoEnabledOcsCategoriesError(message)

        if client is None:
            async with OcsClient(settings=ocs_settings) as ocs_client:
                products_processed, stock_rows_inserted = await _sync_enabled_categories(
                    enabled_categories,
                    client=ocs_client,
                    product_repository=product_repo,
                    shipment_city=ocs_settings.ocs_shipment_city,
                    only_available=only_available,
                    with_descriptions=with_descriptions,
                )
        else:
            products_processed, stock_rows_inserted = await _sync_enabled_categories(
                enabled_categories,
                client=client,
                product_repository=product_repo,
                shipment_city=ocs_settings.ocs_shipment_city,
                only_available=only_available,
                with_descriptions=with_descriptions,
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
    client: OcsProductsClient,
    product_repository: ProductRepository,
    shipment_city: str,
    only_available: bool,
    with_descriptions: bool,
) -> tuple[int, int]:
    synced_at = datetime.now(UTC)
    products_processed = 0
    stock_rows_inserted = 0

    for category in enabled_categories:
        category_id = str(category.category_id)
        payload = await client.get_products_by_category(
            category_id,
            shipment_city,
            only_available=only_available,
            include_regular=True,
            include_sale=False,
            include_uncondition=False,
            include_missing=False,
            with_descriptions=with_descriptions,
        )

        for wrapper_node in extract_ocs_products(payload):
            wrapper = _as_mapping(wrapper_node, entity="OCS product wrapper")
            product_row = build_product_upsert(
                wrapper,
                fallback_category_id=category_id,
                synced_at=synced_at,
            )
            stock_rows = build_stock_price_rows(
                wrapper,
                item_id=product_row.item_id,
                product_key=product_row.product_key,
                shipment_city=shipment_city,
                synced_at=synced_at,
            )

            await product_repository.upsert_product(product_row)
            await product_repository.insert_stock_price_rows(stock_rows)
            products_processed += 1
            stock_rows_inserted += len(stock_rows)

    return products_processed, stock_rows_inserted


def extract_ocs_products(payload: Any) -> list[Any]:
    if _is_sequence(payload):
        return list(payload)

    if isinstance(payload, Mapping):
        for key in PRODUCT_ROOT_KEYS:
            nested = payload.get(key)
            if _is_sequence(nested):
                return list(nested)
            if isinstance(nested, Mapping):
                nested_products = extract_ocs_products(nested)
                if nested_products:
                    return nested_products

        if "product" in payload or any(key in payload for key in ITEM_ID_KEYS):
            return [payload]

    return []


def build_product_upsert(
    wrapper: Mapping[str, Any],
    *,
    fallback_category_id: str,
    synced_at: datetime,
) -> ProductUpsert:
    product = _product_from_wrapper(wrapper)
    item_id = _first_string(product, ("itemId", "itemid"))
    if item_id is None:
        raise ValueError("OCS product wrapper does not contain product.itemId")

    return ProductUpsert(
        distributor_code=DISTRIBUTOR_CODE,
        item_id=item_id,
        product_key=_first_string(product, PRODUCT_KEY_KEYS),
        part_number=_first_string(product, PART_NUMBER_KEYS),
        producer=_first_string(product, PRODUCER_KEYS),
        category_id=_category_id(product, fallback=fallback_category_id),
        item_name=_first_string(product, ("itemName", "item_name", "itemname", "name")),
        item_name_rus=_first_string(product, ("itemNameRus", "item_name_rus", "itemnamerus")),
        product_name=_first_string(product, ("productName", "product_name", "productname")),
        product_description=_first_string(
            product,
            ("productDescription", "product_description", "productdescription", "description"),
        ),
        product_notes=_first_string(product, ("productNotes", "product_notes", "productnotes")),
        hscode=_first_string(product, ("hsCode", "hscode", "hSCode", "hs_code")),
        ean=_first_string(product, ("eaN128", "ean128", "ean", "EAN")),
        is_in_mpt_registry=_first_bool(
            product,
            ("isInMPTRegistry", "isInMptRegistry", "is_in_mpt_registry", "isinmptregistry"),
        ),
        is_project_item=_first_bool(product, ("isProjectItem", "is_project_item", "isprojectitem")),
        traceable=_first_bool(product, ("traceable",)),
        condition=_first_string(product, ("condition",)),
        warranty=_first_string(product, ("warranty",)),
        original_country_iso_code=_first_string(
            product,
            (
                "originalCountryIsoCode",
                "originalCountryISOCode",
                "original_country_iso_code",
                "originalcountryisocode",
            ),
        ),
        vat_percent=_to_decimal(_first_value(product, ("vatPercent", "vat_percent", "vat"))),
        serial_number_availability=_first_string(
            product,
            (
                "serialNumberAvailability",
                "serial_number_availability",
                "serialnumberavailability",
            ),
        ),
        catalog_path_json=_catalog_path(product),
        package_json=_package_json(wrapper),
        raw_json=_jsonable_dict(wrapper),
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
    rows: list[StockPriceInsert] = []
    wrapper_price = _first_value(wrapper, ("price", "prices")) or {}

    for location in _locations(wrapper):
        location_map = _as_mapping(location, entity="OCS product location")
        row_price = _first_value(location_map, ("price", "prices")) or wrapper_price
        order_value, order_currency = _price_bucket(row_price, "order")
        list_value, list_currency = _price_bucket(row_price, "list")
        end_user_value, end_user_currency = _price_bucket(row_price, "endUser")
        quantity_value, quantity_is_greater_than = _quantity(location_map)

        rows.append(
            StockPriceInsert(
                distributor_code=DISTRIBUTOR_CODE,
                item_id=item_id,
                product_key=product_key,
                shipment_city=shipment_city,
                location=_location_identifier(location_map),
                location_description=_first_string(
                    location_map,
                    (
                        "locationDescription",
                        "location_description",
                        "description",
                        "address",
                        "name",
                    ),
                ),
                location_type=_first_string(
                    location_map,
                    ("locationType", "location_type", "type", "warehouseType"),
                ),
                quantity_value=quantity_value,
                quantity_is_greater_than=quantity_is_greater_than,
                can_reserve=_first_bool(location_map, ("canReserve", "can_reserve", "canreserve")),
                departure_date=_first_date(location_map, ("departureDate", "departure_date")),
                arrival_date=_first_date(location_map, ("arrivalDate", "arrival_date")),
                delivery_date=_first_date(location_map, ("deliveryDate", "delivery_date")),
                price_order_value=order_value,
                price_order_currency=order_currency,
                price_list_value=list_value,
                price_list_currency=list_currency,
                end_user_value=end_user_value,
                end_user_currency=end_user_currency,
                raw_json=_jsonable_dict(
                    {
                        "location": dict(location_map),
                        "price": row_price,
                        "productKey": product_key,
                        "itemId": item_id,
                    }
                ),
                synced_at=synced_at,
            )
        )

    return rows


def _product_from_wrapper(wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    product = wrapper.get("product") or {}
    if isinstance(product, Mapping):
        return product
    return {}


def _locations(product: Mapping[str, Any]) -> list[Any]:
    for key in LOCATION_KEYS:
        value = product.get(key)
        if _is_sequence(value):
            return list(value)
        if isinstance(value, Mapping):
            for nested in ("items", "data", "locations", "stocks"):
                nested_value = value.get(nested)
                if _is_sequence(nested_value):
                    return list(nested_value)
            return list(value.values())
    return []


def _category_id(product: Mapping[str, Any], *, fallback: str) -> str:
    value = _first_value(product, CATEGORY_KEYS)
    if isinstance(value, Mapping):
        category_id = _first_string(value, CATEGORY_ID_KEYS)
        return category_id or fallback
    return _string_or_none(value) or fallback


def _catalog_path(product: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = _first_value(product, ("catalogPath", "catalog_path", "catalogpath"))
    if not _is_sequence(value):
        return []

    path: list[dict[str, Any]] = []
    for node in value:
        if isinstance(node, Mapping):
            path.append(_jsonable_dict(node))
        else:
            path.append({"value": node})
    return path


def _package_json(wrapper: Mapping[str, Any]) -> dict[str, Any]:
    value = _first_value(
        wrapper,
        ("packageInformation", "package_information", "package", "packageJson", "package_json"),
    )
    if isinstance(value, Mapping):
        return _jsonable_dict(value)
    if value is None:
        return {}
    return {"value": _jsonable(value)}


def _quantity(location: Mapping[str, Any]) -> tuple[int | None, bool | None]:
    quantity = _first_value(location, ("quantity", "qty", "available", "stockQuantity"))
    greater_than = _first_bool(
        location,
        (
            "quantityIsGreatThan",
            "quantityIsGreaterThan",
            "quantity_is_greater_than",
            "isGreaterThan",
        ),
    )

    if isinstance(quantity, Mapping):
        greater_than = _to_bool(
            _first_value(
                quantity,
                ("isGreatThan", "isGreaterThan", "is_greater_than", "greaterThan"),
            )
        )
        quantity = _first_value(quantity, ("value", "quantity", "qty", "available"))

    if isinstance(quantity, str) and quantity.strip().startswith(">"):
        greater_than = True

    return _to_int(quantity), greater_than


def _price_bucket(price: Any, bucket: str) -> tuple[Decimal | None, str | None]:
    if price is None:
        return None, None

    if not isinstance(price, Mapping):
        if bucket == "order":
            return _to_decimal(price), None
        return None, None

    bucket_keys = {
        "order": ("order", "orderPrice", "priceOrder", "price_order"),
        "list": ("list", "listPrice", "priceList", "price_list"),
        "endUser": ("endUser", "end_user", "enduser", "endUserPrice"),
    }[bucket]
    value = _first_value(price, bucket_keys)
    currency = _first_string(price, (f"{bucket}Currency", f"{bucket}_currency", "currency"))

    if isinstance(value, Mapping):
        currency = _first_string(value, ("currency", "currencyCode", "currency_code")) or currency
        value = _first_value(value, ("value", "amount", "price"))

    return _to_decimal(value), currency


def _location_identifier(location: Mapping[str, Any]) -> str | None:
    value = _first_value(
        location,
        ("location", "locationCode", "location_code", "warehouse", "code", "id"),
    )
    if isinstance(value, Mapping):
        return _first_string(value, ("code", "id", "name"))
    return _string_or_none(value)


def _first_string(node: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    return _string_or_none(_first_value(node, keys))


def _first_bool(node: Mapping[str, Any], keys: tuple[str, ...]) -> bool | None:
    return _to_bool(_first_value(node, keys))


def _first_date(node: Mapping[str, Any], keys: tuple[str, ...]) -> date | None:
    return _to_date(_first_value(node, keys))


def _first_value(node: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
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
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
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
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


def _as_mapping(node: Any, *, entity: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise ValueError(f"{entity} must be an object, got {type(node).__name__}")
    return node


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _jsonable_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return _jsonable(dict(value))


def _error_message(exc: Exception) -> str:
    return str(exc)[:2000]
