from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from app.catalog.product_repository import ProductRepository
from app.core.config import get_llm_settings
from app.core.database import get_session_factory
from app.llm.full_category_composer import compose_full_category_quote
from app.llm.openai_compatible import OpenAICompatibleLlmClient
from app.matching.full_category_matrix import build_full_category_matrix_group_package


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_llm_settings()
    parser = argparse.ArgumentParser(
        description="Run a v3 full-category-matrix quote preview.",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Original customer request text.")
    text_group.add_argument("--text-file", help="Path to a UTF-8 file with request text.")
    parser.add_argument(
        "--category-id",
        action="append",
        required=True,
        help="Distributor category_id to include. May be provided multiple times.",
    )
    parser.add_argument("--distributor-code", default="ocs", help="Distributor code.")
    parser.add_argument(
        "--max-package-chars",
        type=int,
        default=settings.llm_configurator_max_package_chars,
        help="Hard prompt/package character budget.",
    )
    parser.add_argument(
        "--model",
        default=settings.llm_model,
        help="Model name used in package diagnostics.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the preview result JSON.",
    )
    parser.add_argument(
        "--matrix-output",
        help="Optional path for the full matrix JSON payload.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Build the full matrix package only and skip the LLM call.",
    )
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_llm_settings().model_copy(
        update={
            "llm_model": args.model,
            "llm_configurator_max_package_chars": max(args.max_package_chars, 1),
        }
    )
    request_text = _request_text(args)
    category_ids = [str(value).strip() for value in args.category_id if str(value).strip()]
    session_factory = get_session_factory()

    async with session_factory() as session:
        repository = ProductRepository(session)
        rows = await repository.list_latest_full_category_group_matrix(
            args.distributor_code,
            category_ids,
        )

    package = build_full_category_matrix_group_package(
        distributor_code=args.distributor_code,
        category_ids=category_ids,
        rows=rows,
        max_package_chars=settings.llm_configurator_max_package_chars,
        model=settings.llm_model,
    )

    if args.matrix_output:
        matrix_output = Path(args.matrix_output)
        matrix_output.parent.mkdir(parents=True, exist_ok=True)
        matrix_output.write_text(package.json_payload, encoding="utf-8")

    if args.no_llm:
        result = {
            "pipeline_version": "v3_full_category_matrix",
            "llm_configurator_used": False,
            "diagnostics": {
                "matrix_status": package.status,
                "matrix_char_count": package.char_count,
                "matrix_row_count": package.payload["diagnostics"]["row_count"],
                "category_ids": package.payload.get("category_ids", []),
                "max_package_chars": settings.llm_configurator_max_package_chars,
                "model": settings.llm_model,
            },
        }
    else:
        client = OpenAICompatibleLlmClient(
            settings=settings,
            read_timeout_seconds=settings.llm_configurator_read_timeout_seconds,
            max_output_tokens=settings.llm_configurator_max_output_tokens,
            max_retries=0,
        )
        try:
            outcome = compose_full_category_quote(
                user_request=request_text,
                matrix_package=package,
                settings=settings,
                llm_client=client,
            )
        finally:
            client.close()
        result = outcome.to_report_json()

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(_summary(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _request_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8").strip()
    return str(args.text or "").strip()


def _summary(result: dict[str, object]) -> dict[str, object]:
    diagnostics = result.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    quote = result.get("validated_quote")
    quote = quote if isinstance(quote, dict) else {}
    return {
        "pipeline_version": result.get("pipeline_version"),
        "llm_configurator_used": result.get("llm_configurator_used"),
        "primary_recommendation_status": result.get("primary_recommendation_status"),
        "final_status_source": result.get("final_status_source"),
        "matrix_status": diagnostics.get("matrix_status"),
        "matrix_row_count": diagnostics.get("matrix_row_count"),
        "matrix_char_count": diagnostics.get("matrix_char_count"),
        "prompt_char_count": diagnostics.get("prompt_char_count"),
        "category_ids": diagnostics.get("category_ids"),
        "model": diagnostics.get("model"),
        "total_price_value": quote.get("total_price_value"),
        "total_price_currency": quote.get("total_price_currency"),
        "engineering_review_required": quote.get("engineering_review_required"),
        "line_count": len(quote.get("lines", [])) if isinstance(quote.get("lines"), list) else 0,
        "validation_errors": result.get("v3_validation_errors"),
        "no_recommendation_reason": result.get("no_recommendation_reason"),
    }


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
