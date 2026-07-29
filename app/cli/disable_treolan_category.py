from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from app.catalog.category_repository import CategoryRepository
from app.cli.ocs_category_format import print_category_summary
from app.core.database import get_session_factory

DISTRIBUTOR_CODE = "treolan"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Disable a Treolan category for product sync.")
    parser.add_argument("--category-id", required=True, help="Treolan category id to disable.")
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = CategoryRepository(session)
        category = await repository.set_category_enabled(
            DISTRIBUTOR_CODE,
            args.category_id,
            False,
        )
        if category is None:
            await session.rollback()
            print(f"Treolan category was not found: {args.category_id}", file=sys.stderr)
            return 1

        await session.commit()
        print_category_summary("disabled", category)

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
