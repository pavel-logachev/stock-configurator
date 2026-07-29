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
)
from app.evaluation.simple_stock_workbench import (
    MAX_BATCH_EXPORT,
    export_case_batch,
    list_match_run_catalog,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List safe MatchRun metadata or export a bounded batch of private, hash-bound "
            "simple-stock evaluation bundles without database writes or LLM calls."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="List safe metadata only.")
    catalog.add_argument("--limit", type=int, default=25)
    catalog.add_argument("--before-id", type=int)

    export = subparsers.add_parser("export", help="Export up to 20 append-only bundles.")
    selection = export.add_mutually_exclusive_group(required=True)
    selection.add_argument("--match-run-id", type=int, action="append", dest="match_run_ids")
    selection.add_argument("--latest", type=int)
    export.add_argument("--batch-id")
    export.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/simple_stock/local/cases"),
    )
    export.add_argument(
        "--receipt-root",
        type=Path,
        default=Path("evaluation/simple_stock/local/batches"),
    )

    for command in (catalog, export):
        command.add_argument(
            "--production-baseline",
            type=Path,
            default=Path("config/production_pipeline_baseline.json"),
        )
    export.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/simple_stock/v1/dataset.json"),
    )
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "catalog":
            payload = await _catalog(args)
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        payload = await _export(args)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["blocked_count"] == 0 else 2
    except CaseExportError as exc:
        print(_error_payload(exc))
        return 2
    except Exception as exc:
        print(_error_payload(CaseExportError("database.read_failed", [type(exc).__name__])))
        return 2


async def _catalog(args: argparse.Namespace) -> dict[str, object]:
    session_factory = get_session_factory()
    async with session_factory() as session, session.begin():
        await enforce_postgresql_read_only_transaction(session)
        catalog = await list_match_run_catalog(
            session,
            production_baseline_path=args.production_baseline,
            limit=args.limit,
            before_id=args.before_id,
        )
    payload = catalog.model_dump(mode="json")
    payload.update({"database_mode": "transaction_read_only", "database_writes": 0, "llm_calls": 0})
    return payload


async def _export(args: argparse.Namespace) -> dict[str, object]:
    private_root = Path("evaluation/simple_stock/local").resolve()
    output_root = args.output_root.resolve()
    receipt_root = args.receipt_root.resolve()
    for path in (output_root, receipt_root):
        if not path.is_relative_to(private_root):
            raise CaseExportError("output.private_root_required")
    ids = list(args.match_run_ids or [])
    if args.latest is not None and not 1 <= args.latest <= MAX_BATCH_EXPORT:
        raise CaseExportError("batch.latest_invalid")
    if len(ids) > MAX_BATCH_EXPORT:
        raise CaseExportError("batch.match_run_ids_invalid")

    session_factory = get_session_factory()
    async with session_factory() as session, session.begin():
        await enforce_postgresql_read_only_transaction(session)
        if args.latest is not None:
            catalog = await list_match_run_catalog(
                session,
                production_baseline_path=args.production_baseline,
                limit=MAX_BATCH_EXPORT,
            )
            ids = [item.match_run_id for item in catalog.candidates if item.exportable][
                : args.latest
            ]
            if not ids:
                raise CaseExportError("batch.no_exportable_candidates")
        receipt = await export_case_batch(
            session,
            match_run_ids=ids,
            output_root=output_root,
            receipt_root=receipt_root,
            production_baseline_path=args.production_baseline,
            dataset_path=args.dataset,
            batch_id=args.batch_id,
        )
    return receipt.model_dump(mode="json")


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
