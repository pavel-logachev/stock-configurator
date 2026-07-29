from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.evaluation.simple_stock_contracts import EVALUATOR_VERSION, REPORT_SCHEMA_VERSION
from app.evaluation.simple_stock_evaluator import (
    EvaluationInputError,
    evaluate_simple_stock_runs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare offline simple-stock baseline and candidate run bundles.",
    )
    parser.add_argument("--dataset", required=True, help="Golden dataset manifest JSON.")
    parser.add_argument("--baseline-run", required=True, help="Accepted baseline run bundle JSON.")
    parser.add_argument("--candidate-run", required=True, help="Candidate run bundle JSON.")
    parser.add_argument(
        "--production-baseline",
        default="config/production_pipeline_baseline.json",
        help="Captured production pipeline baseline JSON.",
    )
    parser.add_argument(
        "--blind-review",
        help="Optional hash-bound blind comparison review JSON.",
    )
    parser.add_argument("--output", help="Optional path for the sanitized evaluation report.")
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate_simple_stock_runs(
            dataset_path=Path(args.dataset),
            baseline_run_path=Path(args.baseline_run),
            candidate_run_path=Path(args.candidate_run),
            production_baseline_path=Path(args.production_baseline),
            blind_review_path=Path(args.blind_review) if args.blind_review else None,
        )
    except EvaluationInputError as exc:
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "evaluator_version": EVALUATOR_VERSION,
            "status": "blocked",
            "failures": [],
            "blockers": [exc.code],
            "input_error_details": exc.details,
            "privacy": {
                "raw_prompts_in_report": False,
                "raw_outputs_in_report": False,
                "product_ids_in_report": False,
            },
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] == "passed":
        return 0
    if report["status"] == "failed":
        return 1
    return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
