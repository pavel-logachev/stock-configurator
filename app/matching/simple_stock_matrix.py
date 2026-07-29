from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from app.catalog.product_repository import FullCategoryMatrixRow
from app.core.config import DEFAULT_LLM_MODEL
from app.matching.full_category_matrix import (
    MATRIX_EMPTY_AFTER_CATEGORY_SELECTION,
    MATRIX_READY_FOR_LLM,
    MATRIX_TOO_LARGE_FOR_MODEL,
)

SIMPLE_STOCK_MATRIX_SCHEMA_VERSION = "simple_stock_matrix.v8"
_TECHNICAL_PRICE_INDEX_CANDIDATE_LIMIT = 5
_TECHNICAL_PRICE_INDEX_MIN_CANDIDATES = 2
_TECHNICAL_PRICE_INDEX_MAX_CANDIDATES = 24
_TECHNICAL_PRICE_INDEX_PAIR_PER_TOKEN_LIMIT = 1
_TECHNICAL_PRICE_INDEX_DESCRIPTION_LIMIT = 70
_TECHNICAL_WORD_TOKENS = (
    "DDR3",
    "DDR4",
    "DDR5",
    "RDIMM",
    "LRDIMM",
    "UDIMM",
    "ECC",
    "NVME",
    "NVMe",
    "SATA",
    "SAS",
    "SSD",
    "HDD",
    "SFF",
    "LFF",
    "U.2",
    "U2",
    "M.2",
    "M2",
    "SFP28",
    "QSFP28",
    "SFP+",
    "RJ-45",
    "RJ45",
    "HBA",
    "FC",
    "FIBRE",
    "CHANNEL",
    "RAID",
    "OCP",
    "PCIE",
    "PCI-E",
    "GEN3",
    "GEN4",
    "GEN5",
    "RACK",
    "TOWER",
    "UPS",
)


@dataclass(frozen=True)
class SimpleStockMatrixPackage:
    payload: dict[str, Any]
    json_payload: str
    char_count: int
    status: str
    stock_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ProductCard:
    product: Any
    component_candidate_id: str
    rows: list[FullCategoryMatrixRow] = field(default_factory=list)


def build_simple_stock_matrix_group_package(
    *,
    distributor_code: str,
    category_ids: Sequence[str],
    rows: list[FullCategoryMatrixRow],
    max_package_chars: int,
    model: str = DEFAULT_LLM_MODEL,
) -> SimpleStockMatrixPackage:
    cleaned_category_ids = [
        category_id
        for category_id in dict.fromkeys(str(value or "").strip() for value in category_ids)
        if category_id
    ]
    ordered_rows = _price_order_rows(
        _dedupe_stock_fact_rows(rows),
        category_ids=cleaned_category_ids,
    )
    product_cards = _product_cards(ordered_rows)
    stock_rows = [
        _stock_row(row, component_candidate_id=card.component_candidate_id)
        for card in product_cards
        for row in card.rows
    ]
    category_sections = _category_sections(product_cards)
    payload: dict[str, Any] = {
        "schema_version": SIMPLE_STOCK_MATRIX_SCHEMA_VERSION,
        "distributor_code": distributor_code,
        "category_ids": cleaned_category_ids,
        "model": model,
        "matrix_note": (
            "Complete stocked/priced positions for the selected category descendants. "
            "Each position is one product-level card for LLM reasoning: part_number, "
            "description, price offers and total stock. The LLM must select products by "
            "component_candidate_id only; exact stock_row_id values are materialized by "
            "code after the LLM response."
        ),
        "field_legend": {
            "position_id": "Product candidate ID to cite in quote lines.",
            "component_candidate_id": "Product candidate ID to cite in quote lines.",
            "category_path": "Distributor category/subcategory path.",
            "part_number": "Distributor/vendor part number when available.",
            "description": "Minimal product description assembled from distributor text facts.",
            "offers": "Mechanically aggregated price/stock options for this product.",
            "source_item_count": (
                "Number of exact distributor item records merged into this product card."
            ),
            "position_order": (
                "Inside each category section, earlier same-currency candidates are cheaper."
            ),
            "price_rank_in_currency": (
                "Same-category price rank for each offer currency; 1 is cheapest."
            ),
            "price_delta_vs_cheapest": (
                "Same-category price delta from the cheapest candidate in that currency."
            ),
            "technical_price_index": (
                "Mechanical per-category token-pair index: technical token pair -> cheapest "
                "stocked candidates in the same category/currency. It is a recall aid, not a "
                "compatibility decision or SKU recommendation."
            ),
        },
        "category_sections": category_sections,
        "diagnostics": {
            "row_count": len(rows),
            "stock_row_count": len(rows),
            "effective_stock_row_count": len(stock_rows),
            "deduplicated_stock_row_count": max(0, len(rows) - len(stock_rows)),
            "position_count": len(product_cards),
            "product_card_count": len(product_cards),
            "merged_product_card_count": _merged_product_card_count(product_cards),
            "merged_source_item_count": _merged_source_item_count(product_cards),
            "category_count": len(cleaned_category_ids),
            "section_count": len(category_sections),
            "max_package_chars": max_package_chars,
            "char_count": 0,
            "status": MATRIX_READY_FOR_LLM,
            "mechanical_price_ordering": True,
            "llm_visible_stock_row_ids": False,
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
            return SimpleStockMatrixPackage(
                payload=payload,
                json_payload=json_payload,
                char_count=char_count,
                status=status,
                stock_rows=stock_rows,
            )
        char_count = next_char_count


def stock_rows_by_id(package: SimpleStockMatrixPackage) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in package.stock_rows:
        stock_row_id = str(row.get("stock_row_id") or "").strip()
        if stock_row_id:
            result[stock_row_id] = dict(row)
    if result:
        return result
    for section in _sequence(package.payload.get("category_sections")):
        if not isinstance(section, Mapping):
            continue
        for position in _sequence(section.get("positions")):
            if not isinstance(position, Mapping):
                continue
            stock_row_id = str(position.get("stock_row_id") or "").strip()
            if stock_row_id:
                result[stock_row_id] = dict(position)
    return result


def product_cards_by_id(package: SimpleStockMatrixPackage) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for section in _sequence(package.payload.get("category_sections")):
        if not isinstance(section, Mapping):
            continue
        for position in _sequence(section.get("positions")):
            if not isinstance(position, Mapping):
                continue
            component_id = str(position.get("component_candidate_id") or "").strip()
            if component_id:
                result[component_id] = dict(position)
    return result


def _category_sections(product_cards: Sequence[_ProductCard]) -> list[dict[str, Any]]:
    sections_by_id: dict[str, dict[str, Any]] = {}
    for card in product_cards:
        category_id = str(card.product.category_id or "").strip()
        section = sections_by_id.get(category_id)
        if section is None:
            section = {
                "category_id": category_id,
                "category_path": _category_path_text(card.product.catalog_path_json),
                "positions": [],
            }
            sections_by_id[category_id] = section
        section["positions"].append(_position(card))
    sections = list(sections_by_id.values())
    _annotate_price_order_metadata(sections)
    _annotate_technical_price_index(sections)
    return sections


def _position(card: _ProductCard) -> dict[str, Any]:
    product = card.product
    position_id = card.component_candidate_id
    source_item_ids = _source_item_ids(card.rows)
    return _compact_dict({
        "position_id": position_id,
        "component_candidate_id": position_id,
        "category_id": product.category_id,
        "category_path": _category_path_text(product.catalog_path_json),
        "part_number": _text_or_none(product.part_number),
        "description": _description(product),
        "offers": _offers_for_product(card.rows),
        "source_item_count": len(source_item_ids) if len(source_item_ids) > 1 else None,
    })


def _stock_row(row: FullCategoryMatrixRow, *, component_candidate_id: str) -> dict[str, Any]:
    product = row.product
    stock = row.stock
    stock_row_id = f"{stock.distributor_code}:{stock.item_id}:{stock.id}"
    return _compact_dict({
        "position_id": component_candidate_id,
        "component_candidate_id": component_candidate_id,
        "stock_row_id": stock_row_id,
        "source_item_id": _clean_text(product.item_id),
        "category_id": product.category_id,
        "category_path": _category_path_text(product.catalog_path_json),
        "part_number": _text_or_none(product.part_number),
        "description": _description(product),
        "price": {
            "value": _json_value(stock.price_order_value),
            "currency": stock.price_order_currency,
        },
        "stock": _compact_dict({
            "quantity_value": stock.quantity_value,
            "quantity_is_greater_than": stock.quantity_is_greater_than,
        }),
    })


def _dedupe_stock_fact_rows(
    rows: Sequence[FullCategoryMatrixRow],
) -> list[FullCategoryMatrixRow]:
    result: list[FullCategoryMatrixRow] = []
    seen: set[tuple[str, str, Any]] = set()
    for row in rows:
        stock = row.stock
        key = (
            str(stock.distributor_code or ""),
            str(stock.item_id or ""),
            stock.id,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _product_cards(rows: Sequence[FullCategoryMatrixRow]) -> list[_ProductCard]:
    by_position: dict[tuple[str, ...], _ProductCard] = {}
    order: list[tuple[str, ...]] = []
    for row in rows:
        product = row.product
        key = _product_card_key(row)
        card = by_position.get(key)
        if card is None:
            card = _ProductCard(
                product=product,
                component_candidate_id=_component_candidate_id_for_rows([row]),
            )
            by_position[key] = card
            order.append(key)
        card.rows.append(row)
    cards = [by_position[key] for key in order]
    for card in cards:
        card.component_candidate_id = _component_candidate_id_for_rows(card.rows)
    return cards


def _product_card_key(row: FullCategoryMatrixRow) -> tuple[str, ...]:
    product = row.product
    distributor_code = _identity_text(product.distributor_code)
    category_id = _identity_text(product.category_id)
    producer = _identity_text(product.producer)
    part_number = _identity_part_number(product.part_number)
    if distributor_code and category_id and producer and part_number:
        return ("identity", distributor_code, category_id, producer, part_number)
    return ("item", distributor_code, _identity_text(product.item_id))


def _component_candidate_id_for_rows(rows: Sequence[FullCategoryMatrixRow]) -> str:
    if not rows:
        return ""
    product = rows[0].product
    distributor_code = _clean_text(product.distributor_code) or ""
    source_item_ids = _source_item_ids(rows)
    item_id = source_item_ids[0] if source_item_ids else (_clean_text(product.item_id) or "")
    return f"{distributor_code}:{item_id}"


def _source_item_ids(rows: Sequence[FullCategoryMatrixRow]) -> list[str]:
    return sorted(
        {
            item_id
            for row in rows
            if (item_id := (_clean_text(row.product.item_id) or ""))
        }
    )


def _offers_for_product(rows: Sequence[FullCategoryMatrixRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        stock = row.stock
        price_value = _json_value(stock.price_order_value)
        currency = _clean_text(stock.price_order_currency)
        if price_value is None or not currency:
            continue
        key = (str(price_value), currency)
        offer = grouped.setdefault(
            key,
            {
                "price": {"value": price_value, "currency": currency},
                "available_quantity": 0,
                "quantity_is_greater_than": False,
                "stock_bucket_count": 0,
            },
        )
        quantity = stock.quantity_value
        if isinstance(quantity, int):
            offer["available_quantity"] += quantity
        offer["quantity_is_greater_than"] = bool(
            offer["quantity_is_greater_than"] or stock.quantity_is_greater_than
        )
        offer["stock_bucket_count"] += 1

    offers = list(grouped.values())
    offers.sort(
        key=lambda offer: (
            str(offer["price"].get("currency") or "ZZZ"),
            _decimal_sort_value(offer["price"].get("value")),
        )
    )
    for index, offer in enumerate(offers, start=1):
        offer["offer_index"] = index
    return offers


def _annotate_price_order_metadata(sections: Sequence[dict[str, Any]]) -> None:
    for section in sections:
        position_prices: list[tuple[dict[str, Any], dict[str, Decimal]]] = []
        for position in _sequence(section.get("positions")):
            if not isinstance(position, dict):
                continue
            best_prices = _best_offer_prices_by_currency(position)
            if best_prices:
                position_prices.append((position, best_prices))

        currencies = sorted(
            {
                currency
                for _position, best_prices in position_prices
                for currency in best_prices
            }
        )
        for currency in currencies:
            priced_positions = [
                (best_prices[currency], position)
                for position, best_prices in position_prices
                if currency in best_prices
            ]
            priced_positions.sort(
                key=lambda item: (
                    item[0],
                    str(item[1].get("part_number") or ""),
                    str(item[1].get("component_candidate_id") or ""),
                )
            )
            if not priced_positions:
                continue
            cheapest_price = priced_positions[0][0]
            rank = 0
            previous_price: Decimal | None = None
            for price, position in priced_positions:
                if previous_price is None or price != previous_price:
                    rank += 1
                    previous_price = price
                position.setdefault("price_rank_in_currency", {})[currency] = rank
                position.setdefault("price_delta_vs_cheapest", {})[
                    currency
                ] = _format_price_delta(price - cheapest_price)


def _annotate_technical_price_index(sections: Sequence[dict[str, Any]]) -> None:
    for section in sections:
        buckets: dict[tuple[str, str], list[tuple[Decimal, dict[str, Any]]]] = {}
        for position in _sequence(section.get("positions")):
            if not isinstance(position, dict):
                continue
            tokens = _technical_tokens(position)
            if not tokens:
                continue
            best_prices = _best_offer_prices_by_currency(position)
            for currency, price in best_prices.items():
                candidate = _technical_index_candidate(position, currency=currency, price=price)
                if candidate is None:
                    continue
                for token in tokens:
                    buckets.setdefault((token, currency), []).append((price, candidate))

        index_entries: list[dict[str, Any]] = []
        for (token, currency), priced_candidates in sorted(buckets.items()):
            candidates = _dedupe_and_sort_technical_candidates(priced_candidates)
            if len(candidates) < _TECHNICAL_PRICE_INDEX_MIN_CANDIDATES:
                continue
            index_entries.append(
                {
                    "token": token,
                    "currency": currency,
                    "candidates": candidates[:_TECHNICAL_PRICE_INDEX_CANDIDATE_LIMIT],
                }
            )
        if index_entries:
            section["technical_price_index"] = index_entries


def _technical_tokens(position: Mapping[str, Any]) -> list[str]:
    text = " ".join(
        part
        for part in (
            _clean_text(position.get("part_number")),
            _clean_text(position.get("description")),
        )
        if part
    )
    if not text:
        return []
    normalized = text.upper().replace(",", ".")
    tokens: set[str] = set()
    patterns = (
        r"\b\d+(?:\.\d+)?\s*(?:GB|TB)\b",
        r"\b\d+(?:\.\d+)?\s*(?:MT/S|MHZ|GHZ|GT/S|GB/S|GBPS|MB/S|MBPS)\b",
        r"\b\d+(?:\.\d+)?\s*GFC\b",
        r"\b\d+(?:\.\d+)?\s*GBE\b",
        r"\b\d{2,5}\s*W\b",
        r"\b\d+\s*(?:YR|YEAR|YEARS)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            token = _normalize_technical_token(match.group(0))
            tokens.add(token)
            numeric_prefix = re.match(r"^(\d+(?:\.\d+)?)", token)
            if numeric_prefix and any(
                marker in token for marker in ("MT/S", "MHZ", "GT/S", "GB/S", "GBPS")
            ):
                tokens.add(numeric_prefix.group(1))

    if re.search(r"(?<![A-Z0-9])(?:DDR[345]|RDIMM|LRDIMM|UDIMM)(?![A-Z0-9])", normalized):
        for match in re.finditer(r"\b([1-9]\d{3,4})\b", normalized):
            value = int(match.group(1))
            if 1000 <= value <= 8000:
                tokens.add(str(value))

    for match in re.finditer(r"\b(\d{1,2})\s*[- ]?PORTS?\b", normalized, flags=re.IGNORECASE):
        tokens.add(f"{match.group(1)}-port")

    for token in _TECHNICAL_WORD_TOKENS:
        pattern = rf"(?<![A-Z0-9]){re.escape(token.upper())}(?![A-Z0-9])"
        if re.search(pattern, normalized):
            tokens.add(_normalize_technical_token(token))

    return sorted(token for token in tokens if 2 <= len(token) <= 24)


def _normalize_technical_token(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or "").strip().upper()).replace(",", ".")
    text = text.replace("YEARS", "yr").replace("YEAR", "yr").replace("YR", "yr")
    text = text.replace("PORTS", "port").replace("PORT", "port")
    text = text.replace("RJ45", "RJ-45")
    if text.endswith("P") and text[:-1].isdigit():
        return f"{text[:-1]}-port"
    if text.endswith("PORT") and text[:-4].isdigit():
        return f"{text[:-4]}-port"
    return text


def _technical_index_candidate(
    position: Mapping[str, Any],
    *,
    currency: str,
    price: Decimal,
) -> dict[str, Any] | None:
    component_id = _clean_text(position.get("component_candidate_id"))
    if not component_id:
        return None
    offer = _best_offer_for_currency(position, currency=currency, price=price)
    candidate = {
        "component_candidate_id": component_id,
        "part_number": _clean_text(position.get("part_number")),
        "description": _limit_text(
            _clean_text(position.get("description")),
            _TECHNICAL_PRICE_INDEX_DESCRIPTION_LIMIT,
        ),
        "price": {"value": _json_value(price), "currency": currency},
    }
    if offer is not None:
        candidate["available_quantity"] = offer.get("available_quantity")
        candidate["quantity_is_greater_than"] = bool(offer.get("quantity_is_greater_than"))
    return _compact_dict(candidate)


def _best_offer_for_currency(
    position: Mapping[str, Any],
    *,
    currency: str,
    price: Decimal,
) -> Mapping[str, Any] | None:
    for offer in _sequence(position.get("offers")):
        if not isinstance(offer, Mapping):
            continue
        offer_price = offer.get("price")
        if not isinstance(offer_price, Mapping):
            continue
        if _clean_text(offer_price.get("currency")) != currency:
            continue
        if _decimal_price_value(offer_price.get("value")) == price:
            return offer
    return None


def _dedupe_and_sort_technical_candidates(
    priced_candidates: Sequence[tuple[Decimal, dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_id: dict[str, tuple[Decimal, dict[str, Any]]] = {}
    for price, candidate in priced_candidates:
        component_id = str(candidate.get("component_candidate_id") or "")
        current = by_id.get(component_id)
        if current is None or price < current[0]:
            by_id[component_id] = (price, candidate)
    return [
        candidate
        for _price, candidate in sorted(
            by_id.values(),
            key=lambda item: (
                item[0],
                str(item[1].get("part_number") or ""),
                str(item[1].get("component_candidate_id") or ""),
            ),
        )
    ]


def _best_offer_prices_by_currency(position: Mapping[str, Any]) -> dict[str, Decimal]:
    best_prices: dict[str, Decimal] = {}
    for offer in _sequence(position.get("offers")):
        if not isinstance(offer, Mapping):
            continue
        price = offer.get("price")
        if not isinstance(price, Mapping):
            continue
        currency = _clean_text(price.get("currency"))
        value = _decimal_price_value(price.get("value"))
        if not currency or value is None:
            continue
        current = best_prices.get(currency)
        if current is None or value < current:
            best_prices[currency] = value
    return best_prices


def _description(product: Any) -> str | None:
    parts = _unique_clean_texts(
        [
            product.producer,
            product.item_name,
            product.item_name_rus,
            product.product_name,
            product.product_description,
            product.product_notes,
        ]
    )
    property_facts = _properties_text(product.raw_json)
    if property_facts:
        parts.extend(_unique_clean_texts([property_facts], existing=parts))
    return _limit_text(" / ".join(parts), 650) if parts else None


def _properties_text(raw_json: Any) -> str | None:
    if not isinstance(raw_json, Mapping):
        return None
    cache = raw_json.get("__ocs_content_cache_v1")
    candidates = []
    if isinstance(cache, Mapping):
        candidates.append(cache.get("properties"))
    for key in ("content_properties", "properties", "Properties", "attributes", "specifications"):
        candidates.append(raw_json.get(key))

    properties: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for item in _property_items(candidate):
            text = _property_text(item)
            if not text or text in seen:
                continue
            seen.add(text)
            properties.append(text)
            if len(properties) >= 12:
                return "; ".join(properties)
    return "; ".join(properties) if properties else None


def _property_items(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _property_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        name = _first_mapping_text(
            value,
            ("name", "Name", "property", "Property", "title", "Title", "label", "Label"),
        )
        property_value = _first_mapping_text(
            value,
            ("value", "Value", "displayValue", "DisplayValue", "text", "Text"),
        )
        unit = _first_mapping_text(value, ("unit", "Unit", "measure", "Measure", "uom", "UOM"))
        parts = [name, property_value, unit]
        return " ".join(part for part in parts if part) or None
    return _clean_text(value)


def _first_mapping_text(value: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        text = _clean_text(value.get(key))
        if text:
            return text
    return None


def _price_order_rows(
    rows: Sequence[FullCategoryMatrixRow],
    *,
    category_ids: Sequence[str],
) -> list[FullCategoryMatrixRow]:
    category_position = {category_id: index for index, category_id in enumerate(category_ids)}
    return sorted(
        rows,
        key=lambda row: (
            category_position.get(row.product.category_id or "", len(category_position)),
            0 if str(row.stock.price_order_currency or "").upper() == "USD" else 1,
            str(row.stock.price_order_currency or "ZZZ"),
            (
                row.stock.price_order_value
                if row.stock.price_order_value is not None
                else Decimal("999999999999")
            ),
            row.product.item_id or "",
            row.stock.location or "",
            row.stock.id,
        ),
    )


def _matrix_status(*, row_count: int, char_count: int, max_package_chars: int) -> str:
    if row_count == 0:
        return MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
    if char_count > max_package_chars:
        return MATRIX_TOO_LARGE_FOR_MODEL
    return MATRIX_READY_FOR_LLM


def _position_count(rows: Sequence[FullCategoryMatrixRow]) -> int:
    return len({(row.product.distributor_code, row.product.item_id) for row in rows})


def _merged_product_card_count(cards: Sequence[_ProductCard]) -> int:
    return sum(1 for card in cards if len(_source_item_ids(card.rows)) > 1)


def _merged_source_item_count(cards: Sequence[_ProductCard]) -> int:
    return sum(max(0, len(_source_item_ids(card.rows)) - 1) for card in cards)


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


def _unique_clean_texts(values: Sequence[Any], *, existing: Sequence[str] = ()) -> list[str]:
    result: list[str] = []
    seen = {text.casefold() for text in existing}
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _compact_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value is not None and value != "" and value != []
    }


def _text_or_none(value: Any) -> str | None:
    return _clean_text(value)


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return " ".join(text.split())


def _identity_text(value: Any) -> str:
    return (_clean_text(value) or "").upper()


def _identity_part_number(value: Any) -> str:
    text = _identity_text(value)
    return re.sub(r"\s+", "", text)


def _limit_text(value: str | None, limit: int) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _decimal_sort_value(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("999999999999")


def _decimal_price_value(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _format_price_delta(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"+{text}" if value > 0 else text


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
