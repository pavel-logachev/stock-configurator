from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.evaluation.simple_stock_case_exporter import (
    CaseExportError,
    finalize_case_review,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed private review.json and materialize hash-bound annotation "
            "and golden-case fragments."
        ),
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/simple_stock/v1/dataset.json"),
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    private_root = Path("evaluation/simple_stock/local").resolve()
    bundle = args.bundle.resolve()
    if not bundle.is_relative_to(private_root):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error": "output.private_root_required",
                    "details": [],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    try:
        receipt = finalize_case_review(
            bundle_dir=bundle,
            dataset_path=args.dataset,
        )
    except CaseExportError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error": exc.code,
                    "details": exc.details,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error": "review.failed",
                    "details": [type(exc).__name__],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "status": receipt.decision,
                "case_id": receipt.case_id,
                "annotation_sha256": receipt.annotation.sha256,
                "golden_case_sha256": receipt.golden_case.sha256,
                "manifest_sha256": receipt.manifest_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
