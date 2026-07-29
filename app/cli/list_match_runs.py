from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from app.core.database import get_session_factory
from app.matching.match_repository import MatchRepository


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List recent Match Engine runs.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum rows to print.")
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = MatchRepository(session)
        match_runs = await repository.list_recent_match_runs(limit=args.limit)

    print("Match runs")
    if not match_runs:
        print("No match runs found.")
        return 0

    print("id\tstatus\ttotal_candidates\tmatched_items\tcreated_at\tsource_text")
    for match_run in match_runs:
        print(
            "\t".join(
                [
                    str(match_run.id),
                    match_run.status,
                    str(match_run.total_candidates),
                    str(match_run.matched_items),
                    match_run.created_at.isoformat(),
                    _brief(match_run.source_text),
                ]
            )
        )
    return 0


def _brief(value: str | None, *, limit: int = 80) -> str:
    if not value:
        return ""
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
