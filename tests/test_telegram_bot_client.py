from __future__ import annotations

import asyncio
import json

import httpx

from app.telegram_bot.stock_api_client import StockApiClient


def test_stock_api_client_posts_match_text() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=201,
            json={
                "match_run_id": 42,
                "status": "partial_stock_matched",
                "engineer_review_required": True,
                "total_candidates": 2,
                "matched_items": 0,
                "risk_flags": ["engineer_review_required"],
                "missing_requirements": ["RAM below requirement"],
                "candidates": [
                    {
                        "part_number": "D5720-181125SA04",
                        "item_id": "1000841882",
                        "confidence_score": 80,
                        "price_value": "6900",
                        "price_currency": "USD",
                        "available_quantity": 3,
                    }
                ],
                "report_markdown": "# Report",
                "report_url": "/api/v1/match/42/report.md",
            },
        )

    async def run_test() -> dict[str, object]:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            http_client=http_client,
        )
        try:
            return await client.create_match(" Need 2 servers ")
        finally:
            await client.aclose()

    result = asyncio.run(run_test())

    assert result["match_run_id"] == 42
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url == "http://stock-api:8000/api/v1/match"
    assert json.loads(request.content) == {"text": "Need 2 servers"}


def test_stock_api_client_posts_v3_full_category_quote() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "profile": "storage",
                "category_ids": ["V2101"],
                "distributor_code": "ocs",
                "result_state": "quote_draft_review_required",
                "engineering_review_required": True,
                "validated_quote": {"lines": []},
                "diagnostics": {"matrix_row_count": 1},
            },
        )

    async def run_test() -> dict[str, object]:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            v3_timeout_seconds=900,
            http_client=http_client,
        )
        try:
            return await client.create_v3_full_category_quote(
                profile="storage",
                distributor_code="treolan",
                user_text=" Need NAS ",
            )
        finally:
            await client.aclose()

    result = asyncio.run(run_test())

    assert result["result_state"] == "quote_draft_review_required"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url == "http://stock-api:8000/api/v1/match/v3/full-category"
    assert json.loads(request.content) == {
        "text": "Need NAS",
        "profile": "storage",
        "distributor_code": "treolan",
    }
    assert request.extensions["timeout"]["read"] == 900


def test_stock_api_client_posts_simple_stock_quote_with_profile() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "profile": "storage",
                "category_ids": ["V2101"],
                "distributor_code": "treolan",
                "result_state": "quote_draft_review_required",
                "engineering_review_required": True,
                "validated_quote": {"lines": []},
                "diagnostics": {
                    "simple_clean_route": True,
                    "legacy_v7_1_contract_used": False,
                },
            },
        )

    async def run_test() -> dict[str, object]:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            v3_timeout_seconds=900,
            http_client=http_client,
        )
        try:
            return await client.create_simple_stock_quote(
                profile="storage",
                distributor_code="treolan",
                user_text=" Need NAS ",
            )
        finally:
            await client.aclose()

    result = asyncio.run(run_test())

    assert result["result_state"] == "quote_draft_review_required"
    request = requests[0]
    assert request.method == "POST"
    assert request.url == "http://stock-api:8000/api/v1/match/v3/simple-stock-quote"
    assert json.loads(request.content) == {
        "text": "Need NAS",
        "profile": "storage",
        "distributor_code": "treolan",
    }
    assert request.extensions["timeout"]["read"] == 900


def test_stock_api_client_posts_v3_auto_quote_without_profile() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "profile": "server",
                "category_ids": ["V1100"],
                "distributor_code": "ocs",
                "result_state": "quote_draft_review_required",
                "engineering_review_required": True,
                "validated_quote": {"lines": []},
                "diagnostics": {
                    "matrix_row_count": 1,
                    "profile_router_used": True,
                },
            },
        )

    async def run_test() -> dict[str, object]:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            http_client=http_client,
        )
        try:
            return await client.create_v3_full_category_quote_auto(" Need server ")
        finally:
            await client.aclose()

    result = asyncio.run(run_test())

    assert result["profile"] == "server"
    request = requests[0]
    assert request.method == "POST"
    assert request.url == "http://stock-api:8000/api/v1/match/v3/simple-stock-quote"
    assert json.loads(request.content) == {"text": "Need server"}


def test_stock_api_client_posts_v3_auto_quote_with_distributor() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "profile": "server",
                "category_ids": ["server-root"],
                "distributor_code": "treolan",
                "result_state": "quote_draft_review_required",
                "engineering_review_required": True,
                "validated_quote": {"lines": []},
                "diagnostics": {
                    "matrix_row_count": 1,
                    "profile_router_used": True,
                },
            },
        )

    async def run_test() -> dict[str, object]:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            http_client=http_client,
        )
        try:
            return await client.create_v3_full_category_quote_auto(
                " Need server ",
                distributor_code="treolan",
            )
        finally:
            await client.aclose()

    result = asyncio.run(run_test())

    assert result["distributor_code"] == "treolan"
    request = requests[0]
    assert request.method == "POST"
    assert request.url == "http://stock-api:8000/api/v1/match/v3/simple-stock-quote"
    assert json.loads(request.content) == {
        "text": "Need server",
        "distributor_code": "treolan",
    }


def test_stock_api_client_gets_markdown_report() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code=200, text="# Match Engine V0 Report")

    async def run_test() -> str:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            http_client=http_client,
        )
        try:
            return await client.get_match_report_markdown(42)
        finally:
            await client.aclose()

    report = asyncio.run(run_test())

    assert report == "# Match Engine V0 Report"
    assert requests[0].method == "GET"
    assert requests[0].url == "http://stock-api:8000/api/v1/match/42/report.md"


def test_stock_api_client_gets_existing_match_summary() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "id": 70,
                "status": "partial_stock_matched",
                "engineer_review_required": True,
                "total_candidates": 1,
                "matched_items": 0,
                "risk_flags": [],
                "missing_requirements": [],
                "candidates": [],
                "ready_stock_candidates": [],
                "build_candidates": [],
                "report_json": {
                    "match_run_id": 70,
                    "llm_configurator_used": True,
                },
            },
        )

    async def run_test() -> dict[str, object]:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            http_client=http_client,
        )
        try:
            return await client.get_match_summary(70)
        finally:
            await client.aclose()

    result = asyncio.run(run_test())

    assert result["match_run_id"] == 70
    assert result["report_url"] == "/api/v1/match/70/report.md"
    assert result["report_xlsx_url"] == "/api/v1/match/70/report.xlsx"
    assert [request.method for request in requests] == ["GET"]
    assert requests[0].url == "http://stock-api:8000/api/v1/match/70"


def test_stock_api_client_normalizes_existing_v3_match_summary() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            json={
                "id": 88,
                "status": "quote_draft_review_required",
                "engineer_review_required": True,
                "total_candidates": 1,
                "matched_items": 1,
                "risk_flags": ["engineering_review_required"],
                "missing_requirements": [],
                "candidates": [],
                "ready_stock_candidates": [],
                "build_candidates": [],
                "report_json": {
                    "match_run_id": 88,
                    "pipeline_version": "v3_full_category_matrix",
                    "v3_result_state": "quote_draft_review_required",
                    "v3_profile": "server",
                    "category_ids": ["V1100"],
                    "distributor_code": "ocs",
                    "validated_quote": {"engineering_review_required": True, "lines": []},
                    "diagnostics": {"matrix_row_count": 1},
                },
            },
        )

    async def run_test() -> dict[str, object]:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            http_client=http_client,
        )
        try:
            return await client.get_match_summary(88)
        finally:
            await client.aclose()

    result = asyncio.run(run_test())

    assert result["match_run_id"] == 88
    assert result["pipeline_version"] == "v3_full_category_matrix"
    assert result["result_state"] == "quote_draft_review_required"
    assert result["profile"] == "server"
    assert result["report_xlsx_url"] == "/api/v1/match/88/report.xlsx"
    assert requests[0].url == "http://stock-api:8000/api/v1/match/88"


def test_stock_api_client_gets_excel_report() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status_code=200,
            content=b"xlsx-bytes",
            headers={
                "content-type": (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            },
        )

    async def run_test() -> bytes:
        http_client = httpx.AsyncClient(
            base_url="http://stock-api:8000",
            transport=httpx.MockTransport(handler),
        )
        client = StockApiClient(
            base_url="http://stock-api:8000",
            timeout_seconds=60,
            http_client=http_client,
        )
        try:
            return await client.get_match_report_xlsx(42)
        finally:
            await client.aclose()

    report = asyncio.run(run_test())

    assert report == b"xlsx-bytes"
    assert requests[0].method == "GET"
    assert requests[0].url == "http://stock-api:8000/api/v1/match/42/report.xlsx"
