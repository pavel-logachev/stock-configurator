from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.catalog.product_repository import FullCategoryMatrixRow
from app.core.config import DEFAULT_LLM_MODEL

MATRIX_READY_FOR_LLM = "matrix_ready_for_llm"
MATRIX_TOO_LARGE_FOR_MODEL = "matrix_too_large_for_model"
MATRIX_EMPTY_AFTER_CATEGORY_SELECTION = "matrix_empty_after_category_selection"
SCHEMA_VERSION = "v3_full_category_matrix.category_sections.v1"
MATRIX_PAYLOAD_SCHEMA_VERSION_V7 = "matrix_payload_schema_v7"
OCS_CONTENT_CACHE_KEY = "__ocs_content_cache_v1"


@dataclass(frozen=True)
class FullCategoryMatrixPackage:
    payload: dict[str, Any]
    json_payload: str
    char_count: int
    status: str


def build_full_category_matrix_package(
    *,
    distributor_code: str,
    category_id: str,
    rows: list[FullCategoryMatrixRow],
    max_package_chars: int,
    model: str = DEFAULT_LLM_MODEL,
) -> FullCategoryMatrixPackage:
    return build_full_category_matrix_group_package(
        distributor_code=distributor_code,
        category_ids=[category_id],
        rows=rows,
        max_package_chars=max_package_chars,
        model=model,
    )


def build_full_category_matrix_group_package(
    *,
    distributor_code: str,
    category_ids: Sequence[str],
    rows: list[FullCategoryMatrixRow],
    max_package_chars: int,
    model: str = DEFAULT_LLM_MODEL,
) -> FullCategoryMatrixPackage:
    cleaned_category_ids = [
        category_id
        for category_id in dict.fromkeys(str(value or "").strip() for value in category_ids)
        if category_id
    ]
    primary_category_id = cleaned_category_ids[0] if cleaned_category_ids else ""
    category_sections = _serialize_category_sections(
        _price_order_rows(rows, category_ids=cleaned_category_ids)
    )
    matrix_index = _build_matrix_index(category_sections)
    fact_reference_count = _fact_reference_count(category_sections)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "matrix_payload_schema_version": MATRIX_PAYLOAD_SCHEMA_VERSION_V7,
        "distributor_code": distributor_code,
        "category_id": primary_category_id,
        "category_ids": cleaned_category_ids,
        "model": model,
        "matrix_policy": {
            "row_scope": (
                "all_latest_products_with_all_stock_rows_grouped_by_distributor_subcategory"
            ),
            "semantic_trimming": False,
            "semantic_ranking": False,
            "compatibility_prefiltering": False,
            "category_is_atomic": True,
            "oversize_behavior": MATRIX_TOO_LARGE_FOR_MODEL,
            "section_order": "selected_category_tree_order_without_section_loss",
            "row_order": "subcategory_then_usd_first_price_ascending_without_row_loss",
            "field_order": "distributor_product_block_then_all_stock_rows",
            "mechanical_price_ordering": True,
            "primary_matrix_view": "category_sections",
            "matrix_index": (
                "lossless mechanical index over every product block; not a "
                "shortlist, not semantic ranking, not compatibility proof"
            ),
            "fact_references": (
                "stable fact_id values attached to catalog fields/properties; "
                "Composer may cite them, code verifies only existence/ownership"
            ),
            "raw_payload_policy": (
                "raw_json_is_excluded_from_llm_payload; "
                "package_json_is_exposed_only_as_compact_package_facts"
            ),
        },
        "row_legend": {
            "category_sections": (
                "Primary matrix view: distributor category/subcategory sections. "
                "Each section contains all stocked/priced products from that exact "
                "subcategory."
            ),
            "products": (
                "Complete distributor product blocks inside a section. The code does "
                "not assign BOM roles or split product descriptions into fields for "
                "technical reasoning. A product block may describe an individual "
                "component, an accessory, a barebone platform, or a complete/"
                "preconfigured system with included parts."
            ),
            "component_candidate_id": (
                "Stable product candidate ID; quote lines must use this value "
                "from a product block."
            ),
            "stock_row_id": (
                "Specific stock/price bucket ID; quote lines must use this value "
                "from one of that product block's stock_rows."
            ),
            "product": "Distributor product facts exactly as available in the catalog feed.",
            "stock_rows": (
                "Warehouse, quantity, reservation, delivery and price facts for "
                "each exact stock bucket of the product."
            ),
            "price_order_value": "Unit price used for quote arithmetic in price_order_currency.",
            "matrix_index": (
                "Compact row for each product block: category path, producer, "
                "part number, item name, total stock and minimum price by currency."
            ),
            "fact_refs": (
                "Mechanical references to facts inside the same product block. "
                "They are evidence pointers, not normalized technical claims."
            ),
        },
        "matrix_index": matrix_index,
        "category_sections": category_sections,
        "diagnostics": {
            "row_count": len(rows),
            "stock_row_count": len(rows),
            "category_count": len(cleaned_category_ids),
            "section_count": _section_count(rows),
            "component_count": _component_count(rows),
            "matrix_index_count": len(matrix_index),
            "fact_reference_count": fact_reference_count,
            "max_package_chars": max_package_chars,
            "char_count": 0,
            "status": MATRIX_READY_FOR_LLM,
            "raw_json_included": False,
            "package_json_included": False,
        },
    }

    json_payload = _stable_json(payload)
    char_count = len(json_payload)
    while True:
        status = _matrix_status(
            row_count=len(rows),
            char_count=char_count,
            max_package_chars=max_package_chars,
        )
        payload["diagnostics"]["char_count"] = char_count
        payload["diagnostics"]["status"] = status
        json_payload = _stable_json(payload)
        next_char_count = len(json_payload)
        if next_char_count == char_count:
            return FullCategoryMatrixPackage(
                payload=payload,
                json_payload=json_payload,
                char_count=char_count,
                status=status,
            )
        char_count = next_char_count


def build_full_category_matrix_summary(
    package: FullCategoryMatrixPackage,
    *,
    output_path: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "matrix_payload_schema_version": package.payload.get(
            "matrix_payload_schema_version",
        ),
        "status": package.status,
        "row_count": package.payload["diagnostics"]["row_count"],
        "component_count": package.payload["diagnostics"].get("component_count"),
        "stock_row_count": package.payload["diagnostics"].get("stock_row_count"),
        "char_count": package.char_count,
        "max_package_chars": package.payload["diagnostics"]["max_package_chars"],
        "distributor_code": package.payload["distributor_code"],
        "category_id": package.payload["category_id"],
        "category_ids": package.payload.get("category_ids", []),
        "model": package.payload["model"],
        "output_path": output_path,
    }


def _price_order_rows(
    rows: list[FullCategoryMatrixRow],
    *,
    category_ids: Sequence[str],
) -> list[FullCategoryMatrixRow]:
    category_position = {category_id: index for index, category_id in enumerate(category_ids)}
    return sorted(
        rows,
        key=lambda row: (
            category_position.get(row.product.category_id or "", len(category_position)),
            _price_sort_currency_priority(row.stock.price_order_currency),
            _price_sort_currency(row.stock.price_order_currency),
            _price_sort_value(row.stock.price_order_value),
            row.product.item_id or "",
            row.stock.location or "",
            row.stock.id,
        ),
    )


def _price_sort_currency_priority(value: str | None) -> int:
    return 0 if str(value or "").upper() == "USD" else 1


def _price_sort_currency(value: str | None) -> str:
    return str(value or "ZZZ")


def _price_sort_value(value: Decimal | None) -> Decimal:
    return value if value is not None else Decimal("999999999999")


def _component_count(rows: list[FullCategoryMatrixRow]) -> int:
    return len({(row.product.distributor_code, row.product.item_id) for row in rows})


def _section_count(rows: list[FullCategoryMatrixRow]) -> int:
    return len({str(row.product.category_id or "") for row in rows})


def _matrix_status(
    *,
    row_count: int,
    char_count: int,
    max_package_chars: int,
) -> str:
    if row_count == 0:
        return MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
    if char_count > max_package_chars:
        return MATRIX_TOO_LARGE_FOR_MODEL
    return MATRIX_READY_FOR_LLM


def _serialize_category_sections(rows: list[FullCategoryMatrixRow]) -> list[dict[str, Any]]:
    sections_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        category_id = str(row.product.category_id or "").strip()
        section = sections_by_id.get(category_id)
        if section is None:
            section = {
                "category_id": category_id,
                "category_path": _category_path_text(row.product.catalog_path_json),
                "products": [],
            }
            sections_by_id[category_id] = section
        _append_serialized_product(section["products"], row)
    return list(sections_by_id.values())


def _append_serialized_product(
    products: list[dict[str, Any]],
    row: FullCategoryMatrixRow,
) -> None:
    serialized_row = _serialize_row(row)
    component_candidate_id = serialized_row["component_candidate_id"]
    for product in products:
        if product.get("component_candidate_id") == component_candidate_id:
            product["stock_rows"].append(serialized_row["stock"])
            return
    products.append(
        {
            "component_candidate_id": component_candidate_id,
            "product": serialized_row["product"],
            "fact_refs": _fact_refs_for_product(
                component_candidate_id,
                serialized_row["product"],
            ),
            "stock_rows": [serialized_row["stock"]],
        }
    )


def _build_matrix_index(category_sections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index: list[dict[str, Any]] = []
    for section in category_sections:
        category_id = str(section.get("category_id") or "").strip()
        category_path = str(section.get("category_path") or "").strip()
        products = section.get("products") or []
        if not isinstance(products, Sequence) or isinstance(products, (str, bytes, bytearray)):
            continue
        for raw_product in products:
            if not isinstance(raw_product, Mapping):
                continue
            product = _mapping(raw_product.get("product"))
            stock_rows = _stock_rows(raw_product.get("stock_rows"))
            index.append(
                {
                    "component_candidate_id": raw_product.get("component_candidate_id"),
                    "category_id": category_id or product.get("category_id"),
                    "category_path": category_path,
                    "producer": product.get("producer"),
                    "part_number": product.get("part_number"),
                    "item_name": product.get("item_name") or product.get("product_name"),
                    "total_stock_quantity": _total_stock_quantity(stock_rows),
                    "minimum_price_by_currency": _minimum_price_by_currency(stock_rows),
                    "stock_row_count": len(stock_rows),
                }
            )
    return sorted(
        index,
        key=lambda item: (
            str(item.get("category_path") or ""),
            str(item.get("category_id") or ""),
            str(item.get("producer") or ""),
            str(item.get("part_number") or ""),
            str(item.get("item_name") or ""),
            str(item.get("component_candidate_id") or ""),
        ),
    )


def _fact_reference_count(category_sections: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for section in category_sections:
        products = section.get("products") or []
        if not isinstance(products, Sequence) or isinstance(products, (str, bytes, bytearray)):
            continue
        for raw_product in products:
            if isinstance(raw_product, Mapping):
                count += len(_raw_sequence(raw_product.get("fact_refs")))
    return count


def _fact_refs_for_product(
    component_candidate_id: str,
    product: Mapping[str, Any],
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for field_name in (
        "producer",
        "part_number",
        "item_name",
        "item_name_rus",
        "product_name",
        "product_description",
        "product_notes",
        "product_key",
        "condition",
        "warranty",
        "hscode",
        "ean",
        "catalog_path_json",
        "package_facts",
    ):
        value = product.get(field_name)
        if _is_empty_fact_value(value):
            continue
        refs.append(
            {
                "fact_id": f"F:{component_candidate_id}:{field_name}",
                "field": field_name,
                "value": _fact_value_preview(value),
            }
        )

    properties = product.get("content_properties") or []
    if isinstance(properties, Sequence) and not isinstance(properties, (str, bytes, bytearray)):
        for index, item in enumerate(properties):
            if _is_empty_fact_value(item):
                continue
            refs.append(
                {
                    "fact_id": f"F:{component_candidate_id}:property:{index}",
                    "field": "content_properties",
                    "index": index,
                    "value": _fact_value_preview(item),
                }
            )
    return refs


def _stock_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _total_stock_quantity(stock_rows: Sequence[Mapping[str, Any]]) -> int:
    total = 0
    for stock_row in stock_rows:
        quantity = stock_row.get("quantity_value")
        if isinstance(quantity, bool) or quantity in (None, ""):
            continue
        try:
            total += int(quantity)
        except (TypeError, ValueError):
            continue
    return total


def _minimum_price_by_currency(stock_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    minimums: dict[str, Decimal] = {}
    for stock_row in stock_rows:
        currency = str(stock_row.get("price_order_currency") or "").strip()
        if not currency:
            continue
        value = stock_row.get("price_order_value")
        try:
            price = Decimal(str(value))
        except Exception:
            continue
        current = minimums.get(currency)
        if current is None or price < current:
            minimums[currency] = price
    return {currency: str(minimums[currency]) for currency in sorted(minimums)}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _raw_sequence(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(value)


def _is_empty_fact_value(value: Any) -> bool:
    return value in (None, "", [], {})


def _fact_value_preview(value: Any) -> Any:
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, Decimal | datetime | date):
        return _json_value(value)
    if isinstance(value, bool | int | float) or value is None:
        return value
    json_value = json.loads(json.dumps(value, ensure_ascii=False, default=str))
    if isinstance(json_value, list):
        return json_value[:12]
    return json_value


def _category_path_text(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                text = str(
                    item.get("name")
                    or item.get("title")
                    or item.get("category_name")
                    or item.get("category_id")
                    or item.get("id")
                    or ""
                ).strip()
            else:
                text = str(item or "").strip()
            if text:
                parts.append(text)
        return " / ".join(parts)
    return ""


def _serialize_row(row: FullCategoryMatrixRow) -> dict[str, Any]:
    product = row.product
    stock = row.stock
    component_candidate_id = f"{product.distributor_code}:{product.item_id}"
    stock_row_id = f"{stock.distributor_code}:{stock.item_id}:{stock.id}"

    return {
        "component_candidate_id": component_candidate_id,
        "stock_row_id": stock_row_id,
        "product": {
            "distributor_code": product.distributor_code,
            "item_id": product.item_id,
            "category_id": product.category_id,
            "producer": product.producer,
            "part_number": product.part_number,
            "item_name": product.item_name,
            "item_name_rus": product.item_name_rus,
            "product_name": product.product_name,
            "product_description": product.product_description,
            "product_notes": product.product_notes,
            "product_key": product.product_key,
            "condition": product.condition,
            "warranty": product.warranty,
            "hscode": product.hscode,
            "ean": product.ean,
            "is_in_mpt_registry": product.is_in_mpt_registry,
            "is_project_item": product.is_project_item,
            "traceable": product.traceable,
            "original_country_iso_code": product.original_country_iso_code,
            "vat_percent": _json_value(product.vat_percent),
            "serial_number_availability": product.serial_number_availability,
            "catalog_path_json": product.catalog_path_json,
            "package_facts": _compact_package_facts(product.package_json),
            "content_properties": _compact_content_properties(product.raw_json),
            "synced_at": _json_value(product.synced_at),
        },
        "stock": {
            "stock_row_id": stock_row_id,
            "distributor_code": stock.distributor_code,
            "item_id": stock.item_id,
            "product_key": stock.product_key,
            "shipment_city": stock.shipment_city,
            "location": stock.location,
            "location_description": stock.location_description,
            "location_type": stock.location_type,
            "quantity_value": stock.quantity_value,
            "quantity_is_greater_than": stock.quantity_is_greater_than,
            "can_reserve": stock.can_reserve,
            "price_order_value": _json_value(stock.price_order_value),
            "price_order_currency": stock.price_order_currency,
            "departure_date": _json_value(stock.departure_date),
            "arrival_date": _json_value(stock.arrival_date),
            "delivery_date": _json_value(stock.delivery_date),
            "price_list_value": _json_value(stock.price_list_value),
            "price_list_currency": stock.price_list_currency,
            "end_user_value": _json_value(stock.end_user_value),
            "end_user_currency": stock.end_user_currency,
            "synced_at": _json_value(stock.synced_at),
        },
    }


def _compact_package_facts(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        compact_item = _compact_json_item(item)
        if compact_item in ({}, []):
            continue
        result[key_text] = compact_item
        if len(result) >= 12:
            break
    return result


def _compact_json_item(value: Any) -> Any:
    scalar = _compact_scalar(value)
    if scalar is not None or value is None:
        return scalar
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            compact_item = _compact_json_item(item)
            if compact_item in ({}, []):
                continue
            result[key_text] = compact_item
            if len(result) >= 12:
                break
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            compact_item = _compact_json_item(item)
            if compact_item in ({}, []):
                continue
            result.append(compact_item)
            if len(result) >= 12:
                break
        return result
    return {}


def _compact_content_properties(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    candidates: list[Any] = []
    cache = value.get(OCS_CONTENT_CACHE_KEY)
    if isinstance(cache, Mapping):
        candidates.append(cache.get("properties"))
    for key in ("content_properties", "properties", "Properties", "attributes", "specifications"):
        candidates.append(value.get(key))

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        for item in _property_items(candidate):
            compact = _compact_property(item)
            if not compact:
                continue
            marker = (
                str(compact.get("name") or ""),
                str(compact.get("value") or ""),
                str(compact.get("unit") or ""),
            )
            if marker in seen:
                continue
            seen.add(marker)
            result.append(compact)
            if len(result) >= 32:
                return result
    return result


def _property_items(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _compact_property(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        name = _first_compact_property_value(
            value,
            ("name", "Name", "property", "Property", "title", "Title", "label", "Label"),
        )
        property_value = _first_compact_property_value(
            value,
            ("value", "Value", "displayValue", "DisplayValue", "text", "Text"),
        )
        unit = _first_compact_property_value(
            value,
            ("unit", "Unit", "measure", "Measure", "uom", "UOM"),
        )
        result: dict[str, Any] = {}
        if name is not None:
            result["name"] = name
        if property_value is not None:
            result["value"] = property_value
        if unit is not None:
            result["unit"] = unit
        return result
    scalar = _compact_scalar(value)
    if scalar is None:
        return {}
    return {"value": scalar}


def _first_compact_property_value(
    value: Mapping[str, Any],
    keys: Sequence[str],
) -> Any:
    for key in keys:
        compact = _compact_scalar(value.get(key))
        if compact is not None:
            return compact
    return None


def _compact_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text[:500] if text else None
    if isinstance(value, bool | int | float | Decimal | datetime | date):
        return _json_value(value)
    return None


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
    )
