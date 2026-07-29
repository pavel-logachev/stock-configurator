import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import app.distributors.ocs.content_enrichment as content_enrichment_module
from app.core.config import OcsSettings
from app.distributors.ocs.client import (
    OcsClient,
    OcsClientError,
    OcsForbiddenError,
    OcsNotFoundError,
    OcsRateLimitError,
    OcsUnauthorizedError,
)

SECRET = "super-secret-token"


def _settings(api_key: str = SECRET) -> OcsSettings:
    return OcsSettings(
        ocs_base_url="https://ocs.example.test",
        ocs_api_key=api_key,
        ocs_shipment_city="Москва",
        ocs_timeout_seconds=30,
        ocs_request_delay_seconds=0,
        ocs_content_batch_size=2,
        ocs_content_max_items_per_run=3,
    )


async def _call_with_transport(
    handler: httpx.MockTransport,
    action: Callable[[OcsClient], Awaitable[Any]],
) -> Any:
    async with httpx.AsyncClient(transport=handler) as http_client:
        client = OcsClient(settings=_settings(), http_client=http_client)
        return await action(client)


def test_get_shipment_cities_successful_response() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=["Москва", "Санкт-Петербург"])

    result = asyncio.run(
        _call_with_transport(
            httpx.MockTransport(handler),
            lambda client: client.get_shipment_cities(),
        )
    )

    assert result == ["Москва", "Санкт-Петербург"]
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/v2/logistic/shipment/cities"
    assert requests[0].headers["accept"] == "application/json"
    assert requests[0].headers["X-API-Key"] == SECRET


def test_get_stock_locations_passes_cyrillic_shipment_city_in_params() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{"name": "Main"}])

    result = asyncio.run(
        _call_with_transport(
            httpx.MockTransport(handler),
            lambda client: client.get_stock_locations("Москва"),
        )
    )

    assert result == [{"name": "Main"}]
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/v2/logistic/stocks/locations"
    assert requests[0].url.params["shipmentcity"] == "Москва"


def test_get_categories_uses_catalog_categories_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{"id": "servers"}])

    result = asyncio.run(
        _call_with_transport(
            httpx.MockTransport(handler),
            lambda client: client.get_categories(),
        )
    )

    assert result == [{"id": "servers"}]
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/v2/catalog/categories"


def test_get_products_by_category_uses_category_path_and_filter_params() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{"itemId": "123"}])

    result = asyncio.run(
        _call_with_transport(
            httpx.MockTransport(handler),
            lambda client: client.get_products_by_category(
                "servers",
                "Москва",
                only_available=False,
                include_regular=True,
                include_sale=True,
                include_uncondition=True,
                include_missing=True,
                with_descriptions=True,
            ),
        )
    )

    assert result == [{"itemId": "123"}]
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/v2/catalog/categories/servers/products"
    assert requests[0].url.params["shipmentcity"] == "Москва"
    assert requests[0].url.params["onlyavailable"] == "false"
    assert requests[0].url.params["includeregular"] == "true"
    assert requests[0].url.params["includesale"] == "true"
    assert requests[0].url.params["includeuncondition"] == "true"
    assert requests[0].url.params["includemissing"] == "true"
    assert requests[0].url.params["withdescriptions"] == "true"
    assert "category" not in requests[0].url.params


def test_get_products_batch_posts_item_ids_and_filter_params() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{"itemId": "123"}])

    result = asyncio.run(
        _call_with_transport(
            httpx.MockTransport(handler),
            lambda client: client.get_products_batch(
                ["123", "ABC"],
                "Москва",
                only_available=True,
                include_regular=False,
                include_sale=True,
                include_uncondition=False,
                include_missing=True,
                with_descriptions=False,
            ),
        )
    )

    assert result == [{"itemId": "123"}]
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v2/catalog/products/batch"
    assert requests[0].url.params["shipmentcity"] == "Москва"
    assert requests[0].url.params["onlyavailable"] == "true"
    assert requests[0].url.params["includeregular"] == "false"
    assert requests[0].url.params["includesale"] == "true"
    assert requests[0].url.params["includeuncondition"] == "false"
    assert requests[0].url.params["includemissing"] == "true"
    assert requests[0].url.params["withdescriptions"] == "false"
    assert "itemid" not in requests[0].url.params
    assert json.loads(requests[0].content) == ["123", "ABC"]


def test_get_content_batch_posts_item_ids() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"123": {"title": "Server"}})

    result = asyncio.run(
        _call_with_transport(
            httpx.MockTransport(handler),
            lambda client: client.get_content_batch(["123", "ABC"]),
        )
    )

    assert result == {"123": {"title": "Server"}}
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v2/content/batch"
    assert not requests[0].url.params
    assert json.loads(requests[0].content) == ["123", "ABC"]


def test_get_content_uses_content_item_ids_path() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"123": {"title": "Server"}})

    result = asyncio.run(
        _call_with_transport(
            httpx.MockTransport(handler),
            lambda client: client.get_content(["123", "ABC"]),
        )
    )

    assert result == {"123": {"title": "Server"}}
    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/v2/content/123,ABC"
    assert requests[0].headers["X-API-Key"] == SECRET


@pytest.mark.parametrize(
    ("exception", "expected_reason", "expected_status"),
    [
        (OcsUnauthorizedError("unauthorized", status_code=401), "content_forbidden", 401),
        (OcsForbiddenError("forbidden", status_code=403), "content_forbidden", 403),
        (OcsNotFoundError("missing", status_code=404), "content_unavailable", 404),
        (OcsRateLimitError("rate limited", status_code=429), "content_unavailable", 429),
        (OcsClientError("network failed"), "content_unavailable", None),
    ],
)
def test_ocs_content_errors_are_optional_enrichment(
    monkeypatch: pytest.MonkeyPatch,
    exception: OcsClientError,
    expected_reason: str,
    expected_status: int | None,
) -> None:
    class FailingContentClient:
        def __init__(self, settings: OcsSettings) -> None:
            self.settings = settings

        async def __aenter__(self) -> "FailingContentClient":
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def get_content_batch(self, item_ids: list[str]) -> Any:
            assert item_ids == ["123"]
            raise exception

    async def flush() -> None:
        raise AssertionError("content failure must not flush cache writes")

    monkeypatch.setattr(content_enrichment_module, "OcsClient", FailingContentClient)
    matrix = {
        "cpu_candidates": [
            {
                "component_candidate_id": "cpu-1",
                "distributor_code": "ocs",
                "item_id": "123",
            }
        ]
    }
    products = [
        SimpleNamespace(distributor_code="ocs", item_id="123", raw_json={}),
    ]

    diagnostics = asyncio.run(
        content_enrichment_module.enrich_matrix_with_ocs_content(
            session=SimpleNamespace(flush=flush),
            component_candidate_matrix=matrix,
            products=products,
            settings=_settings(),
        )
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["available"] is False
    assert diagnostics["requested_items"] == 1
    assert diagnostics["fetched_items"] == 0
    assert diagnostics["skipped_reason"] == expected_reason
    assert diagnostics["error_type"] == type(exception).__name__
    assert diagnostics["http_status"] == expected_status
    assert "ocs_content_properties" not in matrix["cpu_candidates"][0]


def test_ocs_content_settings_are_bounded_and_share_rate_limit() -> None:
    client = OcsClient(settings=_settings())

    assert client.rate_limit_summary()["requests_per_hour_limit"] == 180
    assert client._settings.ocs_content_batch_size == 2
    assert client._settings.ocs_content_max_items_per_run == 3
    asyncio.run(client.aclose())


def test_401_response_becomes_unauthorized_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    with pytest.raises(OcsUnauthorizedError) as exc_info:
        asyncio.run(
            _call_with_transport(
                httpx.MockTransport(handler),
                lambda client: client.get_categories(),
            )
        )

    assert exc_info.value.status_code == 401


def test_403_response_becomes_forbidden_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    with pytest.raises(OcsForbiddenError) as exc_info:
        asyncio.run(
            _call_with_transport(
                httpx.MockTransport(handler),
                lambda client: client.get_content_batch(["123"]),
            )
        )

    assert exc_info.value.status_code == 403


def test_429_response_becomes_rate_limit_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "too many requests"})

    with pytest.raises(OcsRateLimitError) as exc_info:
        asyncio.run(
            _call_with_transport(
                httpx.MockTransport(handler),
                lambda client: client.get_categories(),
            )
        )

    assert exc_info.value.status_code == 429


def test_token_does_not_get_into_exception_text_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad token: {SECRET}")

    caplog.set_level(logging.DEBUG)

    with pytest.raises(OcsUnauthorizedError) as exc_info:
        asyncio.run(
            _call_with_transport(
                httpx.MockTransport(handler),
                lambda client: client.get_categories(),
            )
        )

    assert SECRET not in str(exc_info.value)
    assert SECRET not in caplog.text
