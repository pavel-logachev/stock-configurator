from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from app.core.database import get_session_factory
from app.distributors.ocs.sync_products import sync_ocs_products


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync OCS products for enabled categories.")
    parser.add_argument(
        "--only-available",
        type=_parse_bool,
        default=True,
        help="Sync only available products: true or false. Default: true.",
    )
    parser.add_argument(
        "--with-descriptions",
        type=_parse_bool,
        default=False,
        help="Request product descriptions from OCS: true or false. Default: false.",
    )
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_factory = get_session_factory()

    async with session_factory() as session:
        result = await sync_ocs_products(
            session,
            only_available=args.only_available,
            with_descriptions=args.with_descriptions,
        )

    print("OCS products sync summary")
    print(f"distributor: {result.distributor}")
    print(f"status: {result.status}")
    print(f"enabled categories: {result.enabled_categories}")
    print(f"products processed: {result.products_processed}")
    print(f"stock rows inserted: {result.stock_rows_inserted}")
    print(f"sync_run_id: {result.sync_run_id}")

    if result.error_message:
        print(f"error: {result.error_message}", file=sys.stderr)

    return 0 if result.status == "success" else 1


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
