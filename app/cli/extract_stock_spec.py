from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.llm.base import LlmError
from app.llm.stock_spec_extractor import extract_stock_spec_from_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Stock Spec from a free-form request.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Free-form user request text.")
    source.add_argument("--file", type=Path, help="UTF-8 text file with a user request.")
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
        result = extract_stock_spec_from_text(text)
    except OSError as exc:
        print(f"Could not read request file: {exc}", file=sys.stderr)
        return 1
    except LlmError as exc:
        print(f"Stock Spec extraction failed: {exc}", file=sys.stderr)
        return 1

    print("confirmation_text:")
    print(result.confirmation_text)
    print()
    print("spec_json:")
    print(json.dumps(result.spec_json.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
    print()
    print("unclear_points:")
    print(json.dumps(result.unclear_points, ensure_ascii=False, indent=2))
    print()
    print("risk_flags:")
    print(json.dumps(result.risk_flags, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
