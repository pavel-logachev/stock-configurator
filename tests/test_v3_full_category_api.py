from __future__ import annotations

import asyncio
from typing import Any

from app.api.routes import match as match_routes
from app.llm.simple_stock_composer import (
    SIMPLE_STOCK_QUOTE_ACCEPTED,
    SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
)
from app.matching.v3_full_category_quote_service import V3FullCategoryQuoteResult


def test_create_v3_full_category_quote_returns_bot_contract(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_run_v3_full_category_quote(**kwargs: Any) -> V3FullCategoryQuoteResult:
        calls.append(kwargs)
        return V3FullCategoryQuoteResult(
            profile="server",
            category_ids=["V1100"],
            distributor_code="ocs",
            result_state="quote_draft_review_required",
            report_json={
                "pipeline_version": "v3_full_category_matrix",
                "llm_configurator_used": True,
                "primary_recommendation_status": "valid",
                "final_status_source": "v3_full_category_quote_validated",
                "validated_quote": {
                    "engineering_review_required": True,
                    "lines": [],
                },
                "diagnostics": {
                    "matrix_row_count": 559,
                    "model": "qwen/qwen3.7-plus",
                },
                "v3_validation_errors": [],
                "v3_validation_warnings": [],
            },
        )

    monkeypatch.setattr(
        match_routes,
        "run_v3_full_category_quote",
        fake_run_v3_full_category_quote,
    )

    async def fake_persist_v3_full_category_quote_result(**kwargs: Any) -> None:
        result = kwargs["result"]
        result.report_json.update(
            {
                "match_run_id": 42,
                "report_url": "/api/v1/match/42/report.md",
                "report_xlsx_url": "/api/v1/match/42/report.xlsx",
            }
        )

    monkeypatch.setattr(
        match_routes,
        "_persist_v3_full_category_quote_result",
        fake_persist_v3_full_category_quote_result,
    )

    response = asyncio.run(
        match_routes.create_v3_full_category_quote(
            object(),  # type: ignore[arg-type]
            match_routes.V3FullCategoryQuoteRequest(
                text=" Need server ",
            ),
        )
    )

    assert response.result_state == "quote_draft_review_required"
    assert response.profile == "server"
    assert response.category_ids == ["V1100"]
    assert response.engineering_review_required is True
    assert response.diagnostics["model"] == "qwen/qwen3.7-plus"
    assert response.match_run_id == 42
    assert response.report_url == "/api/v1/match/42/report.md"
    assert response.report_xlsx_url == "/api/v1/match/42/report.xlsx"
    assert calls[0]["text"] == "Need server"
    assert calls[0]["profile"] is None


def test_create_simple_stock_quote_returns_bot_contract(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_run_simple_stock_quote(**kwargs: Any) -> V3FullCategoryQuoteResult:
        calls.append(kwargs)
        return V3FullCategoryQuoteResult(
            profile="server",
            category_ids=["V1100"],
            distributor_code="ocs",
            result_state="quote_draft_review_required",
            report_json={
                "pipeline_version": SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
                "llm_configurator_used": True,
                "primary_recommendation_status": "valid",
                "final_status_source": SIMPLE_STOCK_QUOTE_ACCEPTED,
                "validated_quote": {
                    "engineering_review_required": True,
                    "lines": [],
                },
                "diagnostics": {
                    "simple_clean_route": True,
                    "legacy_v7_1_contract_used": False,
                },
                "v3_validation_errors": [],
                "v3_validation_warnings": ["simple_code_validation_bypassed"],
            },
        )

    monkeypatch.setattr(
        match_routes,
        "run_simple_stock_quote",
        fake_run_simple_stock_quote,
    )

    async def fake_persist_v3_full_category_quote_result(**kwargs: Any) -> None:
        result = kwargs["result"]
        result.report_json.update(
            {
                "match_run_id": 43,
                "report_url": "/api/v1/match/43/report.md",
                "report_xlsx_url": "/api/v1/match/43/report.xlsx",
            }
        )

    monkeypatch.setattr(
        match_routes,
        "_persist_v3_full_category_quote_result",
        fake_persist_v3_full_category_quote_result,
    )

    response = asyncio.run(
        match_routes.create_simple_stock_quote(
            object(),  # type: ignore[arg-type]
            match_routes.V3FullCategoryQuoteRequest(
                text=" Need server ",
            ),
        )
    )

    assert response.result_state == "quote_draft_review_required"
    assert response.pipeline_version == SIMPLE_STOCK_QUOTE_PIPELINE_VERSION
    assert response.diagnostics["simple_clean_route"] is True
    assert response.match_run_id == 43
    assert calls[0]["text"] == "Need server"
