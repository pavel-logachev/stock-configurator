from __future__ import annotations

import asyncio

from app.catalog.category_repository import CategoryRepository
from app.cli.ocs_category_format import format_path
from app.core.database import get_session_factory

DISTRIBUTOR_CODE = "treolan"


async def run() -> int:
    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = CategoryRepository(session)
        categories = await repository.list_enabled_categories(DISTRIBUTOR_CODE)

    print("Enabled Treolan categories")
    if not categories:
        print("No enabled categories found.")
        return 0

    print("category_id\tname\tenabled_for_sync\tlevel\tpath")
    for category in categories:
        print(
            "\t".join(
                [
                    category.category_id,
                    category.name,
                    str(category.enabled_for_sync).lower(),
                    str(category.level),
                    format_path(category.path_json),
                ]
            )
        )

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
