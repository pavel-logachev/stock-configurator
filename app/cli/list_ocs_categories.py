from __future__ import annotations

import argparse
import asyncio

from app.catalog.category_repository import CategoryRepository
from app.cli.ocs_category_format import format_path
from app.core.database import get_session_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List saved OCS categories.")
    parser.add_argument("--search", help="Search in category id, name, and saved path.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum rows to print.")
    return parser.parse_args()


async def run() -> int:
    args = parse_args()
    limit = max(args.limit, 1)
    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = CategoryRepository(session)
        categories = await repository.list_categories(
            distributor_code="ocs",
            search=args.search,
            root_only=not bool(args.search),
            limit=limit,
        )

    if args.search:
        print(f'OCS categories matching "{args.search}"')
    else:
        print('Root OCS categories. Use --search "сервер" to search the full tree.')

    if not categories:
        print("No categories found.")
        return 0

    print("category_id\tlevel\tname\tpath\tenabled_for_sync")
    for category in categories:
        print(
            "\t".join(
                [
                    category.category_id,
                    str(category.level),
                    category.name,
                    format_path(category.path_json),
                    str(category.enabled_for_sync).lower(),
                ]
            )
        )

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
