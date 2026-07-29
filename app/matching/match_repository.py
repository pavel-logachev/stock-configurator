from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import MatchCandidate, MatchRun


@dataclass(frozen=True)
class MatchCandidateCreate:
    distributor_code: str
    item_id: str
    product_key: str | None
    part_number: str | None
    producer: str | None
    category_id: str | None
    item_name: str | None
    confidence_score: int
    price_value: Decimal | None
    price_currency: str | None
    available_quantity: int | None
    reservable_locations: int
    matched_requirements_json: list[str]
    missing_requirements_json: list[str]
    risk_flags_json: list[str]
    raw_json: dict[str, Any]


@dataclass(frozen=True)
class MatchRunCreate:
    source: str | None
    source_text: str | None
    status: str
    engineer_review_required: bool
    total_candidates: int
    matched_items: int
    missing_requirements_json: list[str]
    risk_flags_json: list[str]
    spec_json: dict[str, Any]
    report_json: dict[str, Any]
    report_markdown: str | None
    candidates: list[MatchCandidateCreate]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MatchRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_match_run(self, row: MatchRunCreate) -> MatchRun:
        now = _utc_now()
        match_run = MatchRun(
            source=row.source,
            source_text=row.source_text,
            status=row.status,
            engineer_review_required=row.engineer_review_required,
            total_candidates=row.total_candidates,
            matched_items=row.matched_items,
            missing_requirements_json=row.missing_requirements_json,
            risk_flags_json=row.risk_flags_json,
            spec_json=row.spec_json,
            report_json=row.report_json,
            report_markdown=row.report_markdown,
            created_at=now,
        )
        self._session.add(match_run)
        await self._session.flush()

        for candidate_row in row.candidates:
            self._session.add(
                MatchCandidate(
                    match_run_id=match_run.id,
                    distributor_code=candidate_row.distributor_code,
                    item_id=candidate_row.item_id,
                    product_key=candidate_row.product_key,
                    part_number=candidate_row.part_number,
                    producer=candidate_row.producer,
                    category_id=candidate_row.category_id,
                    item_name=candidate_row.item_name,
                    confidence_score=candidate_row.confidence_score,
                    price_value=candidate_row.price_value,
                    price_currency=candidate_row.price_currency,
                    available_quantity=candidate_row.available_quantity,
                    reservable_locations=candidate_row.reservable_locations,
                    matched_requirements_json=candidate_row.matched_requirements_json,
                    missing_requirements_json=candidate_row.missing_requirements_json,
                    risk_flags_json=candidate_row.risk_flags_json,
                    raw_json=candidate_row.raw_json,
                    created_at=now,
                )
            )

        await self._session.flush()
        return match_run

    async def get_match_run(self, match_run_id: int) -> MatchRun | None:
        result = await self._session.execute(
            select(MatchRun)
            .options(selectinload(MatchRun.candidates))
            .where(MatchRun.id == match_run_id)
        )
        return result.scalar_one_or_none()

    async def list_match_runs(self, *, limit: int = 10) -> list[MatchRun]:
        result = await self._session.execute(
            select(MatchRun)
            .order_by(MatchRun.created_at.desc(), MatchRun.id.desc())
            .limit(max(limit, 1))
        )
        return list(result.scalars().all())

    async def get_match_report_markdown(self, match_run_id: int) -> str | None:
        result = await self._session.execute(
            select(MatchRun.report_markdown).where(MatchRun.id == match_run_id)
        )
        return result.scalar_one_or_none()

    async def list_recent_match_runs(self, *, limit: int = 10) -> list[MatchRun]:
        return await self.list_match_runs(limit=limit)
