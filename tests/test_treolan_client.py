from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from app.core.config import TreolanSettings
from app.distributors.treolan.client import (
    TreolanClient,
    TreolanConfigurationError,
    TreolanUnauthorizedError,
)

LOGIN = "treolan-user"
PASSWORD = "treolan-secret-password"


def _settings(
    *,
    login: str = LOGIN,
    password: str = PASSWORD,
) -> TreolanSettings:
    return TreolanSettings(
        treolan_base_url="https://treolan.example.test",
        treolan_login=login,
        treolan_password=password,
        treolan_timeout_seconds=30,
        treolan_request_delay_seconds=0,
        treolan_requests_per_minute_limit=0,
    )


async def _call_with_transport(
    handler: httpx.MockTransport,
    action: Callable[[TreolanClient], Awaitable[Any]],
) -> Any:
    async with httpx.AsyncClient(transport=handler) as http_client:
        client = TreolanClient(settings=_settings(), http_client=http_client)
        return await action(client)


def test_gen_catalog_v2_posts_rpc_encoded_soap_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text=(
                '<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">'
                "<soap:Body>"
                "<GenCatalogV2Response>"
                "<Result>"
                "&lt;catalog&gt;&lt;category id='C1' name='Servers'/&gt;&lt;/catalog&gt;"
                "</Result>"
                "</GenCatalogV2Response>"
                "</soap:Body>"
                "</soap:Envelope>"
            ),
        )

    result = asyncio.run(
        _call_with_transport(
            httpx.MockTransport(handler),
            lambda client: client.gen_catalog_v2(category="C1"),
        )
    )

    assert result == "<catalog><category id='C1' name='Servers'/></catalog>"
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/ws/service.asmx"
    assert requests[0].headers["SOAPAction"] == (
        '"http://tempuri.org/treolan/action/WebService.GenCatalogV2"'
    )
    body = requests[0].content.decode()
    assert "<m:GenCatalogV2" in body
    assert f"<login>{LOGIN}</login>" in body
    assert f"<password>{PASSWORD}</password>" in body
    assert "<category>C1</category>" in body
    assert "<vendorid>0</vendorid>" in body
    assert "<freeNom>true</freeNom>" in body


def test_missing_credentials_raise_configuration_error() -> None:
    with pytest.raises(TreolanConfigurationError):
        TreolanClient(settings=_settings(login="", password=""))


def test_http_error_redacts_credentials() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad credentials: {LOGIN} / {PASSWORD}")

    with pytest.raises(TreolanUnauthorizedError) as exc_info:
        asyncio.run(
            _call_with_transport(
                httpx.MockTransport(handler),
                lambda client: client.get_categories(),
            )
        )

    message = str(exc_info.value)
    assert LOGIN not in message
    assert PASSWORD not in message
    assert "[redacted]" in message
