from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from app.catalog.product_repository import ProductRepository
from app.core.config import get_llm_settings
from app.core.database import get_session_factory
from app.matching.full_category_matrix import (
    build_full_category_matrix_group_package,
    build_full_category_matrix_package,
    build_full_category_matrix_summary,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_llm_settings()
    parser = argparse.ArgumentParser(
        description="Dump the full stocked/priced matrix for one distributor category.",
    )
    parser.add_argument(
        "--category-id",
        action="append",
        required=True,
        help="Distributor category_id to dump. May be provided multiple times.",
    )
    parser.add_argument("--distributor-code", default="ocs", help="Distributor code.")
    parser.add_argument(
        "--max-package-chars",
        type=int,
        default=settings.llm_configurator_max_package_chars,
        help="LLM package character budget used only for diagnostics.",
    )
    parser.add_argument(
        "--model",
        default=settings.llm_model,
        help="Model name used only for diagnostics.",
    )
    parser.add_argument("--output", help="Path for the full matrix JSON payload.")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full matrix JSON payload instead of only a summary.",
    )
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = ProductRepository(session)
        category_ids = [str(value).strip() for value in args.category_id if str(value).strip()]
        if len(category_ids) == 1:
            rows = await repository.list_latest_full_category_matrix(
                args.distributor_code,
                category_ids[0],
            )
        else:
            rows = await repository.list_latest_full_category_group_matrix(
                args.distributor_code,
                category_ids,
            )

    if len(category_ids) == 1:
        package = build_full_category_matrix_package(
            distributor_code=args.distributor_code,
            category_id=category_ids[0],
            rows=rows,
            max_package_chars=max(args.max_package_chars, 1),
            model=args.model,
        )
    else:
        package = build_full_category_matrix_group_package(
            distributor_code=args.distributor_code,
            category_ids=category_ids,
            rows=rows,
            max_package_chars=max(args.max_package_chars, 1),
            model=args.model,
        )

    output_path = None
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(package.json_payload, encoding="utf-8")
        output_path = str(output)

    if args.print_json:
        print(package.json_payload)
        return 0

    summary = build_full_category_matrix_summary(package, output_path=output_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
