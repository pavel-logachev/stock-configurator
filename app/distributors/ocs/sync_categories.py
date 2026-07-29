from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.category_repository import CategoryRepository, CategoryUpsert
from app.distributors.ocs.client import OcsClient

DISTRIBUTOR_CODE = "ocs"
DISTRIBUTOR_NAME = "OCS"
SYNC_TYPE = "categories"

ROOT_KEYS = ("categories", "items", "data", "result")
ID_KEYS = ("category", "id", "category_id", "categoryId", "categoryid", "code", "slug")
NAME_KEYS = ("name", "title", "label")
CHILDREN_KEYS = (
    "children",
    "childs",
    "items",
    "categories",
    "subcategories",
    "subCategories",
    "sub_categories",
)


class OcsCategoriesClient(Protocol):
    async def get_categories(self) -> Any:
        pass


@dataclass(frozen=True)
class CategorySyncResult:
    distributor: str
    status: str
    categories_processed: int
    sync_run_id: int
    error_message: str | None = None


def flatten_ocs_categories(
    payload: Any,
    *,
    synced_at: datetime | None = None,
    distributor_code: str = DISTRIBUTOR_CODE,
) -> list[CategoryUpsert]:
    rows: list[CategoryUpsert] = []
    current_synced_at = synced_at or datetime.now(UTC)

    for root in _extract_root_nodes(payload):
        _append_category_rows(
            root,
            rows=rows,
            distributor_code=distributor_code,
            parent_category_id=None,
            level=0,
            path_json=[],
            synced_at=current_synced_at,
        )

    return rows


async def sync_ocs_categories(
    session: AsyncSession,
    *,
    client: OcsCategoriesClient | None = None,
    repository: CategoryRepository | None = None,
) -> CategorySyncResult:
    repo = repository or CategoryRepository(session)
    await repo.ensure_distributor(code=DISTRIBUTOR_CODE, name=DISTRIBUTOR_NAME, enabled=True)
    sync_run = await repo.start_sync_run(
        distributor_code=DISTRIBUTOR_CODE,
        sync_type=SYNC_TYPE,
    )
    sync_run_id = sync_run.id
    await session.commit()

    processed = 0
    try:
        if client is None:
            async with OcsClient() as ocs_client:
                payload = await ocs_client.get_categories()
        else:
            payload = await client.get_categories()

        rows = flatten_ocs_categories(payload, synced_at=datetime.now(UTC))
        for row in rows:
            await repo.upsert_category(row)
            processed += 1

        await repo.finish_sync_run(
            sync_run_id,
            status="success",
            items_processed=processed,
        )
        await session.commit()
        return CategorySyncResult(
            distributor=DISTRIBUTOR_CODE,
            status="success",
            categories_processed=processed,
            sync_run_id=sync_run_id,
        )
    except Exception as exc:
        await session.rollback()
        error_message = _error_message(exc)
        await repo.finish_sync_run(
            sync_run_id,
            status="failed",
            items_processed=processed,
            error_message=error_message,
        )
        await session.commit()
        return CategorySyncResult(
            distributor=DISTRIBUTOR_CODE,
            status="failed",
            categories_processed=processed,
            sync_run_id=sync_run_id,
            error_message=error_message,
        )


def _append_category_rows(
    node: Any,
    *,
    rows: list[CategoryUpsert],
    distributor_code: str,
    parent_category_id: str | None,
    level: int,
    path_json: list[dict[str, str]],
    synced_at: datetime,
) -> None:
    category_node = _as_mapping(node)
    category_id = _category_id(category_node)
    name = _category_name(category_node, fallback=category_id)
    current_path = [*path_json, {"category_id": category_id, "name": name}]

    rows.append(
        CategoryUpsert(
            distributor_code=distributor_code,
            category_id=category_id,
            parent_category_id=parent_category_id,
            name=name,
            level=level,
            path_json=current_path,
            raw_json=_jsonable_dict(category_node),
            synced_at=synced_at,
        )
    )

    for child in _children(category_node):
        _append_category_rows(
            child,
            rows=rows,
            distributor_code=distributor_code,
            parent_category_id=category_id,
            level=level + 1,
            path_json=current_path,
            synced_at=synced_at,
        )


def _extract_root_nodes(payload: Any) -> list[Any]:
    if _is_sequence(payload):
        return list(payload)

    if isinstance(payload, Mapping):
        for key in ROOT_KEYS:
            nested = payload.get(key)
            if _is_sequence(nested):
                return list(nested)
            if isinstance(nested, Mapping):
                nested_roots = _extract_root_nodes(nested)
                if nested_roots:
                    return nested_roots

        if any(key in payload for key in ID_KEYS):
            return [payload]

    return []


def _as_mapping(node: Any) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise ValueError(f"OCS category node must be an object, got {type(node).__name__}")
    return node


def _category_id(node: Mapping[str, Any]) -> str:
    has_id_key = False
    for key in ID_KEYS:
        if key not in node:
            continue

        has_id_key = True
        value = node[key]
        if value is None:
            continue

        category_id = str(value).strip()
        if category_id:
            return category_id

    if has_id_key:
        raise ValueError("OCS category node contains an empty category/id")
    raise ValueError("OCS category node does not contain category/id")


def _category_name(node: Mapping[str, Any], *, fallback: str) -> str:
    value = _first_value(node, NAME_KEYS)
    if value is None:
        return fallback

    name = str(value).strip()
    return name or fallback


def _children(node: Mapping[str, Any]) -> list[Any]:
    for key in CHILDREN_KEYS:
        value = node.get(key)
        if _is_sequence(value):
            return list(value)
    return []


def _first_value(node: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in node:
            return node[key]
    return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _jsonable_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))


def _error_message(exc: Exception) -> str:
    return str(exc)[:2000]
