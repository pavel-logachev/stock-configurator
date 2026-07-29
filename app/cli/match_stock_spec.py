from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.core.database import get_session_factory
from app.llm.base import LlmError
from app.matching.ai_match_orchestrator import (
    AiMatchOrchestratorRequest,
    run_ai_match_orchestrator,
)
from app.matching.match_engine import (
    MatchCandidateResult,
    extract_stock_spec_for_text_match,
)
from app.matching.match_repository import MatchCandidateCreate, MatchRepository, MatchRunCreate
from app.matching.spec_schema import StockSpec
from app.reports.match_report import build_match_markdown_report

DEFAULT_REPORT_DIR = Path("data/match_reports")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match a Stock Spec against local distributor stock."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Free-form user request text.")
    source.add_argument("--file", type=Path, help="UTF-8 .txt request or .json Stock Spec file.")
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        spec, source, source_text = _load_stock_spec(args)
        session_factory = get_session_factory()
        async with session_factory() as session:
            orchestrator_result = await run_ai_match_orchestrator(
                AiMatchOrchestratorRequest(text=source_text, spec=spec),
                session,
            )
            match_result = orchestrator_result.match_result
            report_markdown = build_match_markdown_report(match_result)
            report_json = orchestrator_result.report_json
            repository = MatchRepository(session)
            match_run = await repository.create_match_run(
                MatchRunCreate(
                    source=source,
                    source_text=source_text,
                    status=match_result.status,
                    engineer_review_required=match_result.engineer_review_required,
                    total_candidates=match_result.total_candidates,
                    matched_items=match_result.matched_items,
                    missing_requirements_json=match_result.missing_requirements,
                    risk_flags_json=match_result.risk_flags,
                    spec_json=spec.model_dump(mode="json", exclude_none=True),
                    report_json=report_json,
                    report_markdown=report_markdown,
                    candidates=[
                        _candidate_to_create(candidate) for candidate in match_result.candidates
                    ],
                )
            )
            report_json = {"match_run_id": match_run.id, **report_json}
            match_run.report_json = report_json
            await session.flush()
            await session.commit()
    except OSError as exc:
        print(f"Could not read or write report file: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"Stock Spec JSON validation failed: {exc}", file=sys.stderr)
        return 1
    except LlmError as exc:
        print(f"Stock Spec extraction failed: {exc}", file=sys.stderr)
        return 1

    try:
        _save_report_files(match_run.id, report_json, report_markdown)
    except OSError as exc:
        print(f"Report was saved to DB, but file export failed: {exc}", file=sys.stderr)

    print(report_markdown)
    return 0


def _load_stock_spec(args: argparse.Namespace) -> tuple[StockSpec, str, str | None]:
    if args.text is not None:
        extraction = extract_stock_spec_for_text_match(args.text)
        return extraction.spec_json, "text", args.text

    file_path: Path = args.file
    content = file_path.read_text(encoding="utf-8")
    if file_path.suffix.casefold() == ".json":
        spec = StockSpec.model_validate_json(content)
        return spec, f"file:{file_path}", spec.source_text

    extraction = extract_stock_spec_for_text_match(content)
    return extraction.spec_json, f"file:{file_path}", content


def _candidate_to_create(candidate: MatchCandidateResult) -> MatchCandidateCreate:
    return MatchCandidateCreate(
        distributor_code=candidate.distributor_code,
        item_id=candidate.item_id,
        product_key=candidate.product_key,
        part_number=candidate.part_number,
        producer=candidate.producer,
        category_id=candidate.category_id,
        item_name=candidate.item_name,
        confidence_score=candidate.confidence_score,
        price_value=candidate.price_value,
        price_currency=candidate.price_currency,
        available_quantity=candidate.available_quantity,
        reservable_locations=candidate.reservable_locations,
        matched_requirements_json=candidate.matched_requirements,
        missing_requirements_json=candidate.missing_requirements,
        risk_flags_json=candidate.risk_flags,
        raw_json=candidate.raw,
    )


def _save_report_files(
    match_run_id: int,
    report_json: dict[str, object],
    report_markdown: str,
) -> None:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEFAULT_REPORT_DIR / f"{match_run_id}.json"
    markdown_path = DEFAULT_REPORT_DIR / f"{match_run_id}.md"
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(report_markdown, encoding="utf-8")


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
