from __future__ import annotations

import asyncio

from app.catalog.category_repository import CategoryRepository
from app.catalog.ocs_server_categories import enabled_server_categories, server_category_role_label
from app.core.database import get_session_factory

DISTRIBUTOR_CODE = "ocs"


async def run() -> int:
    session_factory = get_session_factory()
    profile = enabled_server_categories()
    rows: list[tuple[str, str, str, str, str, bool | None]] = []

    async with session_factory() as session:
        repository = CategoryRepository(session)
        for category_profile in profile:
            category = await repository.get_category(
                DISTRIBUTOR_CODE,
                category_profile.category_id,
            )
            if category is None:
                rows.append(
                    (
                        category_profile.category_id,
                        category_profile.name_ru,
                        category_profile.role,
                        server_category_role_label(category_profile.role),
                        "not_found",
                        None,
                    )
                )
                continue

            if category.enabled_for_sync:
                status = "already_enabled"
            else:
                category = await repository.set_category_enabled(
                    DISTRIBUTOR_CODE,
                    category_profile.category_id,
                    True,
                )
                status = "enabled"

            rows.append(
                (
                    category_profile.category_id,
                    category.name if category is not None else category_profile.name_ru,
                    category_profile.role,
                    server_category_role_label(category_profile.role),
                    status,
                    category.enabled_for_sync if category is not None else None,
                )
            )

        await session.commit()

    print("Server OCS categories")
    print("category_id\tname\trole\trole_label\tstatus\tenabled_for_sync")
    for category_id, name, role, role_label, status, enabled_for_sync in rows:
        enabled_text = "" if enabled_for_sync is None else str(enabled_for_sync).lower()
        print("\t".join([category_id, name, role, role_label, status, enabled_text]))

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
