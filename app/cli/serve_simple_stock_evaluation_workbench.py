from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from app.evaluation.simple_stock_case_exporter import CaseExportError
from app.evaluation.simple_stock_workbench_app import create_workbench_app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the private simple-stock evaluation workbench on localhost only."
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--case-root",
        type=Path,
        default=Path("evaluation/simple_stock/local/cases"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/simple_stock/v1/dataset.json"),
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1_024 <= args.port <= 65_535:
        raise SystemExit("port must be between 1024 and 65535")
    private_root = Path("evaluation/simple_stock/local").resolve()
    case_root = args.case_root.resolve()
    if not case_root.is_relative_to(private_root):
        raise SystemExit(CaseExportError("output.private_root_required").code)
    app = create_workbench_app(case_root=case_root, dataset_path=args.dataset)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        access_log=False,
        log_level="warning",
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
