from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from app.catalog.category_repository import CategoryRepository
from app.catalog.ocs_anchor_categories import (
    ANCHOR_GROUPS,
    CONFIG_PATH,
    anchor_categories_for_group,
)
from app.core.database import get_session_factory

DISTRIBUTOR_CODE = "ocs"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enable reviewed local OCS anchor categories for product sync."
    )
    parser.add_argument(
        "--group",
        required=True,
        choices=(*sorted(ANCHOR_GROUPS), "all-approved"),
        help="Anchor group to enable.",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help="Path to reviewed OCS anchor category config.",
    )
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    anchors = anchor_categories_for_group(args.group, path=Path(args.config))
    rows: list[tuple[str, str, str, str, bool | None]] = []
    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = CategoryRepository(session)
        for anchor in anchors:
            category = await repository.get_category(DISTRIBUTOR_CODE, anchor.category_id)
            if category is None:
                rows.append(
                    (
                        anchor.group,
                        anchor.role,
                        anchor.category_id,
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
                    anchor.category_id,
                    True,
                )
                status = "enabled"
            rows.append(
                (
                    anchor.group,
                    anchor.role,
                    anchor.category_id,
                    status,
                    category.enabled_for_sync if category is not None else None,
                )
            )
        await session.commit()

    print("OCS anchor categories")
    print("group\trole\tcategory_id\tstatus\tenabled_for_sync")
    for group, role, category_id, status, enabled_for_sync in rows:
        enabled_text = "" if enabled_for_sync is None else str(enabled_for_sync).lower()
        print("\t".join([group, role, category_id, status, enabled_text]))
    if not rows:
        print("No approved/enabled_default anchors found for requested group.")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
