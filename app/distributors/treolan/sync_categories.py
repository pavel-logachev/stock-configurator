from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.catalog.category_repository import CategoryRepository, CategoryUpsert
from app.distributors.treolan.client import TreolanClient
from app.distributors.treolan.parsing import flatten_treolan_categories

DISTRIBUTOR_CODE = "treolan"
DISTRIBUTOR_NAME = "Treolan"
SYNC_TYPE = "categories"


class TreolanCategoriesClient(Protocol):
    async def get_categories(self) -> str:
        pass


@dataclass(frozen=True)
class CategorySyncResult:
    distributor: str
    status: str
    categories_processed: int
    sync_run_id: int
    error_message: str | None = None


async def sync_treolan_categories(
    session: AsyncSession,
    *,
    client: TreolanCategoriesClient | None = None,
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
            async with TreolanClient() as treolan_client:
                payload = await treolan_client.get_categories()
        else:
            payload = await client.get_categories()

        synced_at = datetime.now(UTC)
        for row in flatten_treolan_categories(payload):
            await repo.upsert_category(
                CategoryUpsert(
                    distributor_code=DISTRIBUTOR_CODE,
                    category_id=row.category_id,
                    parent_category_id=row.parent_category_id,
                    name=row.name,
                    level=row.level,
                    path_json=row.path_json,
                    raw_json=row.raw_json,
                    synced_at=synced_at,
                )
            )
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


def _error_message(exc: Exception) -> str:
    return str(exc)[:2000]
