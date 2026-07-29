from __future__ import annotations

import asyncio
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.distributors.registry import CategoryRefreshResult, build_default_distributor_registry

_REFRESH_LOCKS: dict[str, asyncio.Lock] = {}
_REFRESH_LOCKS_GUARD = asyncio.Lock()


async def refresh_distributor_categories(
    session: AsyncSession,
    *,
    distributor_code: str,
    category_ids: Sequence[str],
) -> CategoryRefreshResult:
    normalized_distributor_code = str(distributor_code or "").strip().lower()
    cleaned_category_ids = [
        category_id
        for category_id in dict.fromkeys(str(value or "").strip() for value in category_ids)
        if category_id
    ]
    if not cleaned_category_ids:
        return CategoryRefreshResult(
            distributor_code=normalized_distributor_code,
            status="skipped",
            category_count=0,
            error_message="No categories selected for refresh.",
        )

    lock = await _refresh_lock(normalized_distributor_code)
    async with lock:
        registry = build_default_distributor_registry()
        connector = registry.get(normalized_distributor_code)
        if connector is None or connector.refresh_categories is None:
            return CategoryRefreshResult(
                distributor_code=normalized_distributor_code,
                status="unsupported_distributor",
                category_count=len(cleaned_category_ids),
                error_message=(
                    f"Distributor {normalized_distributor_code!r} does not support "
                    "on-demand category refresh."
                ),
            )
        return await connector.refresh_categories(session, cleaned_category_ids)


async def _refresh_lock(distributor_code: str) -> asyncio.Lock:
    async with _REFRESH_LOCKS_GUARD:
        lock = _REFRESH_LOCKS.get(distributor_code)
        if lock is None:
            lock = asyncio.Lock()
            _REFRESH_LOCKS[distributor_code] = lock
        return lock
