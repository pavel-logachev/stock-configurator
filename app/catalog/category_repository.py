from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import String, cast, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Distributor, DistributorCategory, SyncRun


@dataclass(frozen=True)
class CategoryUpsert:
    distributor_code: str
    category_id: str
    parent_category_id: str | None
    name: str
    level: int
    path_json: list[dict[str, str]]
    raw_json: dict[str, Any]
    synced_at: datetime
    enabled_for_sync: bool | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CategoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def ensure_distributor(
        self,
        *,
        code: str,
        name: str,
        enabled: bool = True,
    ) -> Distributor:
        distributor = await self._session.scalar(
            select(Distributor).where(Distributor.code == code)
        )
        now = _utc_now()

        if distributor is None:
            distributor = Distributor(
                code=code,
                name=name,
                enabled=enabled,
                created_at=now,
                updated_at=now,
            )
            self._session.add(distributor)
        else:
            distributor.name = name
            distributor.enabled = enabled
            distributor.updated_at = now

        await self._session.flush()
        return distributor

    async def start_sync_run(self, *, distributor_code: str, sync_type: str) -> SyncRun:
        now = _utc_now()
        sync_run = SyncRun(
            distributor_code=distributor_code,
            sync_type=sync_type,
            status="running",
            started_at=now,
            items_processed=0,
            created_at=now,
        )
        self._session.add(sync_run)
        await self._session.flush()
        return sync_run

    async def finish_sync_run(
        self,
        sync_run_id: int,
        *,
        status: str,
        items_processed: int,
        error_message: str | None = None,
    ) -> SyncRun:
        sync_run = await self._session.scalar(select(SyncRun).where(SyncRun.id == sync_run_id))
        if sync_run is None:
            raise ValueError(f"sync_run {sync_run_id} was not found")

        sync_run.status = status
        sync_run.finished_at = _utc_now()
        sync_run.items_processed = items_processed
        sync_run.error_message = error_message
        await self._session.flush()
        return sync_run

    async def get_sync_run(self, sync_run_id: int) -> SyncRun | None:
        return await self._session.scalar(select(SyncRun).where(SyncRun.id == sync_run_id))

    async def upsert_category(self, row: CategoryUpsert) -> DistributorCategory:
        category = await self._session.scalar(
            select(DistributorCategory).where(
                DistributorCategory.distributor_code == row.distributor_code,
                DistributorCategory.category_id == row.category_id,
            )
        )
        now = _utc_now()

        if category is None:
            category = DistributorCategory(
                distributor_code=row.distributor_code,
                category_id=row.category_id,
                parent_category_id=row.parent_category_id,
                name=row.name,
                level=row.level,
                path_json=row.path_json,
                enabled_for_sync=bool(row.enabled_for_sync),
                raw_json=row.raw_json,
                synced_at=row.synced_at,
                created_at=now,
                updated_at=now,
            )
            self._session.add(category)
        else:
            category.parent_category_id = row.parent_category_id
            category.name = row.name
            category.level = row.level
            category.path_json = row.path_json
            category.raw_json = row.raw_json
            category.synced_at = row.synced_at
            category.updated_at = now
            if row.enabled_for_sync is not None:
                category.enabled_for_sync = row.enabled_for_sync

        await self._session.flush()
        return category

    async def set_category_enabled(
        self,
        distributor_code: str,
        category_id: str,
        enabled: bool,
    ) -> DistributorCategory | None:
        category = await self._session.scalar(
            select(DistributorCategory).where(
                DistributorCategory.distributor_code == distributor_code,
                DistributorCategory.category_id == category_id,
            )
        )
        if category is None:
            return None

        category.enabled_for_sync = enabled
        category.updated_at = _utc_now()
        await self._session.flush()
        return category

    async def get_category(
        self,
        distributor_code: str,
        category_id: str,
    ) -> DistributorCategory | None:
        return await self._session.scalar(
            select(DistributorCategory).where(
                DistributorCategory.distributor_code == distributor_code,
                DistributorCategory.category_id == category_id,
            )
        )

    async def list_enabled_categories(
        self,
        distributor_code: str,
    ) -> list[DistributorCategory]:
        result = await self._session.execute(
            select(DistributorCategory)
            .where(
                DistributorCategory.distributor_code == distributor_code,
                DistributorCategory.enabled_for_sync.is_(True),
            )
            .order_by(
                DistributorCategory.level,
                DistributorCategory.name,
                DistributorCategory.category_id,
            )
        )
        return list(result.scalars().all())

    async def list_all_categories(
        self,
        distributor_code: str,
    ) -> list[DistributorCategory]:
        result = await self._session.execute(
            select(DistributorCategory)
            .where(DistributorCategory.distributor_code == distributor_code)
            .order_by(
                DistributorCategory.level,
                DistributorCategory.name,
                DistributorCategory.category_id,
            )
        )
        return list(result.scalars().all())

    async def list_category_ids_with_descendants(
        self,
        *,
        distributor_code: str,
        category_ids: Sequence[str],
    ) -> list[str]:
        root_ids = [
            category_id
            for category_id in dict.fromkeys(
                str(value or "").strip() for value in category_ids
            )
            if category_id
        ]
        if not root_ids:
            return []

        result = await self._session.execute(
            select(DistributorCategory)
            .where(DistributorCategory.distributor_code == distributor_code)
            .order_by(
                DistributorCategory.level,
                DistributorCategory.name,
                DistributorCategory.category_id,
            )
        )
        categories = list(result.scalars().all())
        by_parent: dict[str | None, list[DistributorCategory]] = {}
        existing_ids = {category.category_id for category in categories}
        for category in categories:
            by_parent.setdefault(category.parent_category_id, []).append(category)

        expanded_ids: list[str] = []
        seen: set[str] = set()

        def add_category_tree(category_id: str) -> None:
            if category_id in seen:
                return
            seen.add(category_id)
            expanded_ids.append(category_id)
            for child in by_parent.get(category_id, []):
                add_category_tree(child.category_id)

        for root_id in root_ids:
            add_category_tree(root_id)
            if root_id not in existing_ids:
                continue

        return expanded_ids

    async def list_categories(
        self,
        *,
        distributor_code: str,
        search: str | None = None,
        root_only: bool = False,
        limit: int = 50,
    ) -> list[DistributorCategory]:
        statement = select(DistributorCategory).where(
            DistributorCategory.distributor_code == distributor_code
        )

        if root_only:
            statement = statement.where(DistributorCategory.parent_category_id.is_(None))

        if search:
            search_text = search.strip()
            pattern = f"%{search_text}%"
            path_patterns = [pattern]
            escaped_search_text = json.dumps(search_text, ensure_ascii=True)[1:-1]
            if escaped_search_text != search_text:
                path_patterns.append(f"%{escaped_search_text}%")

            statement = statement.where(
                or_(
                    DistributorCategory.category_id.ilike(pattern),
                    DistributorCategory.name.ilike(pattern),
                    *(
                        cast(DistributorCategory.path_json, String).ilike(path_pattern)
                        for path_pattern in path_patterns
                    ),
                )
            )

        statement = statement.order_by(
            DistributorCategory.level,
            DistributorCategory.name,
            DistributorCategory.category_id,
        ).limit(limit)

        result = await self._session.execute(statement)
        return list(result.scalars().all())
