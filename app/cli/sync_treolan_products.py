from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.database import get_session_factory
from app.distributors.treolan.sync_products import sync_treolan_products


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Treolan products for enabled categories.")
    return parser.parse_args()


async def run() -> int:
    parse_args()
    session_factory = get_session_factory()

    async with session_factory() as session:
        result = await sync_treolan_products(session)

    print("Treolan products sync summary")
    print(f"distributor: {result.distributor}")
    print(f"status: {result.status}")
    print(f"enabled categories: {result.enabled_categories}")
    print(f"products processed: {result.products_processed}")
    print(f"stock rows inserted: {result.stock_rows_inserted}")
    print(f"sync_run_id: {result.sync_run_id}")

    if result.error_message:
        print(f"error: {result.error_message}", file=sys.stderr)

    return 0 if result.status == "success" else 1


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
