from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import OcsSettings, get_ocs_settings
from app.db.models import DistributorProduct
from app.distributors.ocs.client import (
    OcsClient,
    OcsClientError,
    OcsConfigurationError,
    OcsForbiddenError,
    OcsNotFoundError,
    OcsRateLimitError,
    OcsServerError,
    OcsUnauthorizedError,
)

CONTENT_CACHE_KEY = "__ocs_content_cache_v1"
CONTENT_FETCH_ROLES = {
    "platform_candidates",
    "cpu_candidates",
    "ram_candidates",
    "drive_candidates",
    "ssd_candidates",
    "hdd_candidates",
    "storage_controller_candidates",
    "network_adapter_candidates",
    "power_supply_candidates",
    "cable_candidates",
    "other_accessory_candidates",
    "gpu_candidates",
    "transceiver_candidates",
    "rail_kit_candidates",
    "license_candidates",
    "support_candidates",
}


async def enrich_matrix_with_ocs_content(
    *,
    session: AsyncSession,
    component_candidate_matrix: dict[str, Any],
    products: list[DistributorProduct],
    settings: OcsSettings | None = None,
) -> dict[str, Any]:
    settings = settings or get_ocs_settings()
    diagnostics: dict[str, Any] = {
        "enabled": bool(settings.ocs_content_enabled),
        "available": False,
        "batch_size": settings.ocs_content_batch_size,
        "max_items_per_run": settings.ocs_content_max_items_per_run,
        "cache_ttl_hours": settings.ocs_content_cache_ttl_hours,
        "cached_items": 0,
        "fetched_items": 0,
        "requested_items": 0,
        "skipped_reason": None,
        "error_type": None,
        "http_status": None,
    }

    if not settings.ocs_content_enabled:
        diagnostics["skipped_reason"] = "ocs_content_disabled"
        return diagnostics
    if not settings.ocs_api_key.strip():
        diagnostics["skipped_reason"] = "ocs_api_key_missing"
        return diagnostics
    if settings.ocs_content_max_items_per_run <= 0:
        diagnostics["skipped_reason"] = "ocs_content_max_items_zero"
        return diagnostics
    diagnostics["available"] = True

    product_by_key = {
        (product.distributor_code, product.item_id): product
        for product in products
        if product.distributor_code and product.item_id
    }
    rows = _candidate_rows(component_candidate_matrix)
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    products_to_fetch: dict[tuple[str, str], DistributorProduct] = {}

    for row in rows:
        distributor_code = str(row.get("distributor_code") or "ocs").strip()
        item_id = str(row.get("item_id") or "").strip()
        if not item_id:
            continue
        key = (distributor_code, item_id)
        product = product_by_key.get(key)
        if product is None:
            continue
        rows_by_key.setdefault(key, []).append(row)
        cached = _cached_properties(product, settings=settings)
        if cached is not None:
            _attach_properties(row, cached)
            diagnostics["cached_items"] += 1
            continue
        if len(products_to_fetch) < settings.ocs_content_max_items_per_run:
            products_to_fetch[key] = product

    if not products_to_fetch:
        diagnostics["skipped_reason"] = (
            None if diagnostics["cached_items"] else "no_candidate_items_for_content"
        )
        return diagnostics

    diagnostics["requested_items"] = len(products_to_fetch)
    try:
        async with OcsClient(settings=settings) as client:
            for batch_keys in _chunks(
                list(products_to_fetch),
                max(1, settings.ocs_content_batch_size),
            ):
                batch_ids = [item_id for _, item_id in batch_keys]
                response = await client.get_content_batch(batch_ids)
                content_by_item_id = _content_by_item_id(response)
                for key in batch_keys:
                    product = products_to_fetch[key]
                    properties = _compact_properties(content_by_item_id.get(key[1]))
                    _cache_properties(product, properties)
                    for row in rows_by_key.get(key, []):
                        _attach_properties(row, properties)
                    diagnostics["fetched_items"] += 1
        await session.flush()
    except OcsConfigurationError as exc:
        diagnostics["available"] = False
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["skipped_reason"] = "ocs_content_configuration_error"
    except (OcsUnauthorizedError, OcsForbiddenError) as exc:
        diagnostics["available"] = False
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["http_status"] = exc.status_code
        diagnostics["skipped_reason"] = "content_forbidden"
    except (OcsNotFoundError, OcsRateLimitError) as exc:
        diagnostics["available"] = False
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["http_status"] = exc.status_code
        diagnostics["skipped_reason"] = "content_unavailable"
    except OcsClientError as exc:
        diagnostics["available"] = False
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["http_status"] = exc.status_code
        diagnostics["skipped_reason"] = "content_unavailable"
    except OcsServerError as exc:
        diagnostics["available"] = False
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["http_status"] = exc.status_code
        diagnostics["skipped_reason"] = "content_unavailable"

    return diagnostics


def _candidate_rows(component_candidate_matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in CONTENT_FETCH_ROLES:
        value = component_candidate_matrix.get(key)
        if not isinstance(value, list):
            continue
        rows.extend(row for row in value if isinstance(row, dict))
    return rows


def _cached_properties(
    product: DistributorProduct,
    *,
    settings: OcsSettings,
) -> list[dict[str, Any]] | None:
    raw = product.raw_json if isinstance(product.raw_json, Mapping) else {}
    cache = raw.get(CONTENT_CACHE_KEY)
    if not isinstance(cache, Mapping):
        return None
    fetched_at = _parse_datetime(cache.get("fetched_at"))
    if fetched_at is None:
        return None
    if settings.ocs_content_cache_ttl_hours > 0:
        expires_at = fetched_at + timedelta(hours=settings.ocs_content_cache_ttl_hours)
        if datetime.now(UTC) > expires_at:
            return None
    properties = cache.get("properties")
    if not isinstance(properties, list):
        return None
    return [dict(item) for item in properties if isinstance(item, Mapping)]


def _cache_properties(product: DistributorProduct, properties: list[dict[str, Any]]) -> None:
    raw = dict(product.raw_json or {})
    raw[CONTENT_CACHE_KEY] = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "properties": properties,
    }
    product.raw_json = raw
    flag_modified(product, "raw_json")


def _attach_properties(row: dict[str, Any], properties: list[dict[str, Any]]) -> None:
    if properties:
        row["ocs_content_properties"] = properties


def _content_by_item_id(response: Any) -> dict[str, Any]:
    if isinstance(response, list):
        return {
            item_id: item
            for item in response
            if isinstance(item, Mapping) and (item_id := _content_item_id(item))
        }
    if not isinstance(response, Mapping):
        return {}
    for key in ("items", "content", "contents", "data", "results"):
        nested = response.get(key)
        if isinstance(nested, list):
            return _content_by_item_id(nested)
    result: dict[str, Any] = {}
    for key, value in response.items():
        if isinstance(value, Mapping):
            item_id = _content_item_id(value) or str(key)
            result[item_id] = value
    if not result and _content_item_id(response):
        result[_content_item_id(response)] = response
    return result


def _content_item_id(content: Mapping[str, Any]) -> str:
    for key in ("itemId", "item_id", "itemID", "id"):
        text = str(content.get(key) or "").strip()
        if text:
            return text
    product = content.get("product")
    if isinstance(product, Mapping):
        return str(product.get("itemId") or product.get("item_id") or "").strip()
    return ""


def _compact_properties(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, Mapping):
        return []
    properties = _raw_properties(content)
    rows: list[dict[str, Any]] = []
    for item in properties:
        if not isinstance(item, Mapping):
            continue
        name = _short_text(
            item.get("name")
            or item.get("propertyName")
            or item.get("description")
            or item.get("key"),
            limit=80,
        )
        value = item.get("value")
        if isinstance(value, list):
            value = ", ".join(str(part) for part in value if str(part).strip())
        value_text = _short_text(value, limit=120)
        if not name or not value_text:
            continue
        row = {
            "name": name,
            "description": _short_text(item.get("description"), limit=120),
            "unit": _short_text(item.get("unit"), limit=24),
            "type": _short_text(item.get("type"), limit=32),
            "value": value_text,
        }
        rows.append({key: val for key, val in row.items() if val})
        if len(rows) >= 30:
            break
    return rows


def _raw_properties(content: Mapping[str, Any]) -> list[Any]:
    for key in ("properties", "Properties", "propertyList", "attributes", "specifications"):
        value = content.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, Mapping):
            return [
                {"name": property_name, "value": property_value}
                for property_name, property_value in value.items()
            ]
    return []


def _chunks(values: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _short_text(value: Any, *, limit: int) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:limit]
