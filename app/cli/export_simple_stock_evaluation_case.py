from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from app.core.database import get_session_factory
from app.evaluation.simple_stock_case_exporter import (
    CaseExportError,
    enforce_postgresql_read_only_transaction,
    export_case_from_session,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one persisted simple-stock MatchRun into a private, hash-bound local "
            "evaluation bundle without LLM or database writes."
        ),
    )
    parser.add_argument("--match-run-id", type=int, required=True)
    parser.add_argument("--case-id", help="Optional safe case id; derived from the run by default.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/simple_stock/local/cases"),
        help="Private ignored directory that will contain one append-only case bundle.",
    )
    parser.add_argument(
        "--production-baseline",
        type=Path,
        default=Path("config/production_pipeline_baseline.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/simple_stock/v1/dataset.json"),
    )
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.match_run_id <= 0:
        print(_error_payload(CaseExportError("match_run.id_invalid")))
        return 2
    private_root = Path("evaluation/simple_stock/local").resolve()
    output_root = args.output_root.resolve()
    if not output_root.is_relative_to(private_root):
        print(_error_payload(CaseExportError("output.private_root_required")))
        return 2

    session_factory = get_session_factory()
    try:
        async with session_factory() as session, session.begin():
            await enforce_postgresql_read_only_transaction(session)
            manifest = await export_case_from_session(
                session,
                match_run_id=args.match_run_id,
                output_root=output_root,
                production_baseline_path=args.production_baseline,
                dataset_path=args.dataset,
                case_id=args.case_id,
            )
    except CaseExportError as exc:
        print(_error_payload(exc))
        return 2
    except Exception as exc:
        print(_error_payload(CaseExportError("database.read_failed", [type(exc).__name__])))
        return 2

    payload = {
        "status": "ready_for_review" if manifest.golden_review_eligible else "blocked",
        "case_id": manifest.case_id,
        "match_run_id": manifest.match_run_id,
        "bundle": str((output_root / manifest.case_id).resolve()),
        "golden_review_eligible": manifest.golden_review_eligible,
        "matched_matrix_diagnostics": manifest.matrix_evidence.matched_diagnostics,
        "blockers": manifest.blockers,
        "privacy_class": manifest.privacy_class,
        "database_mode": "transaction_read_only",
        "llm_calls": 0,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if manifest.golden_review_eligible else 2


def _error_payload(exc: CaseExportError) -> str:
    return json.dumps(
        {
            "status": "blocked",
            "error": exc.code,
            "details": exc.details,
            "database_writes": 0,
            "llm_calls": 0,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
