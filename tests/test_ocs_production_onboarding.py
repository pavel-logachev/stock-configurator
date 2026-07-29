from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

import app.cli.smoke_ocs_connector as smoke_cli
import app.distributors.ocs.client as ocs_client_module
from app.core.config import OcsSettings
from app.distributors.ocs.client import OcsClient, OcsRateLimitError, OcsUnauthorizedError

SECRET = "unit-test-placeholder"


def test_ocs_settings_reads_api_base_url_alias_and_hides_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OCS_BASE_URL", raising=False)
    monkeypatch.delenv("OCS_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("OCS_API_BASE_URL", "https://prod-ocs.example.test/api")
    monkeypatch.setenv("OCS_API_KEY", SECRET)
    monkeypatch.setenv("OCS_REQUEST_TIMEOUT_SECONDS", "12")
    monkeypatch.setenv("OCS_REQUESTS_PER_HOUR_LIMIT", "180")

    settings = OcsSettings()

    assert settings.ocs_base_url == "https://prod-ocs.example.test/api"
    assert settings.ocs_api_key == SECRET
    assert settings.ocs_timeout_seconds == 12
    assert settings.ocs_requests_per_hour_limit == 180
    assert SECRET not in repr(settings)
    assert SECRET not in str(settings)


def test_ocs_settings_keeps_existing_base_url_env_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OCS_BASE_URL", "https://testconnector.b2b.ocs.ru")
    monkeypatch.setenv("OCS_API_BASE_URL", "https://prod-ocs.example.test")
    monkeypatch.setenv("OCS_API_KEY", SECRET)

    settings = OcsSettings()

    assert settings.ocs_base_url == "https://testconnector.b2b.ocs.ru"


def test_ocs_client_rate_limits_between_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    current_time = 100.0

    def monotonic() -> float:
        return current_time

    async def fake_sleep(seconds: float) -> None:
        nonlocal current_time
        sleeps.append(round(seconds, 3))
        current_time += seconds

    monkeypatch.setattr(ocs_client_module.time, "monotonic", monotonic)
    monkeypatch.setattr(ocs_client_module.asyncio, "sleep", fake_sleep)

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    settings = OcsSettings(
        ocs_base_url="https://ocs.example.test",
        ocs_api_key=SECRET,
        ocs_shipment_city="Moscow",
        ocs_timeout_seconds=30,
        ocs_request_delay_seconds=0,
        ocs_requests_per_hour_limit=200,
    )

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
            client = OcsClient(settings=settings, http_client=http_client)
            await client.get_categories()
            await client.get_categories()

    asyncio.run(run())

    assert len(requests) == 2
    assert sleeps == [18.0]


class FakeSuccessSmokeClient:
    def __init__(self, settings: OcsSettings) -> None:
        self.settings = settings

    async def __aenter__(self) -> FakeSuccessSmokeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get_categories(self) -> Any:
        return {"categories": [{"category": "V1100"}, {"category": "V120100"}]}

    def rate_limit_summary(self) -> dict[str, float | int]:
        return {"requests_per_hour_limit": 180, "minimum_interval_seconds": 20.0}


class FakeUnauthorizedSmokeClient:
    def __init__(self, settings: OcsSettings) -> None:
        self.settings = settings

    async def __aenter__(self) -> FakeUnauthorizedSmokeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get_categories(self) -> Any:
        raise OcsUnauthorizedError(f"bad token {SECRET}", status_code=401)

    def rate_limit_summary(self) -> dict[str, float | int]:
        return {"requests_per_hour_limit": 180, "minimum_interval_seconds": 20.0}


class FakeRateLimitSmokeClient:
    def __init__(self, settings: OcsSettings) -> None:
        self.settings = settings

    async def __aenter__(self) -> FakeRateLimitSmokeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get_categories(self) -> Any:
        raise OcsRateLimitError("too many requests", status_code=429)

    def rate_limit_summary(self) -> dict[str, float | int]:
        return {"requests_per_hour_limit": 180, "minimum_interval_seconds": 20.0}


def test_smoke_ocs_connector_prints_safe_success_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = OcsSettings(
        ocs_base_url="https://prod-ocs.example.test/api",
        ocs_api_key=SECRET,
        ocs_requests_per_hour_limit=180,
    )

    exit_code = asyncio.run(
        smoke_cli.run_smoke(settings=settings, client_factory=FakeSuccessSmokeClient)
    )
    captured = capsys.readouterr()
    combined_output = captured.out + captured.err

    assert exit_code == 0
    assert "base_url_host=prod-ocs.example.test" in captured.out
    assert "auth_configured=true" in captured.out
    assert "endpoint=catalog.categories" in captured.out
    assert "http_status=200" in captured.out
    assert "items_count=2" in captured.out
    assert "rate_limit.requests_per_hour_limit=180" in captured.out
    assert SECRET not in combined_output
    assert "https://prod-ocs.example.test/api" not in combined_output


def test_smoke_ocs_connector_masks_secret_on_auth_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = OcsSettings(
        ocs_base_url="https://prod-ocs.example.test/api",
        ocs_api_key=SECRET,
        ocs_requests_per_hour_limit=180,
    )

    exit_code = asyncio.run(
        smoke_cli.run_smoke(settings=settings, client_factory=FakeUnauthorizedSmokeClient)
    )
    captured = capsys.readouterr()
    combined_output = captured.out + captured.err

    assert exit_code == 1
    assert "http_status=401" in captured.out
    assert "error_type=OcsUnauthorizedError" in captured.err
    assert "check OCS_API_KEY and confirm the VPS allowed IP" in captured.err
    assert SECRET not in combined_output


def test_smoke_ocs_connector_explains_rate_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = OcsSettings(
        ocs_base_url="https://prod-ocs.example.test/api",
        ocs_api_key=SECRET,
        ocs_requests_per_hour_limit=180,
    )

    exit_code = asyncio.run(
        smoke_cli.run_smoke(settings=settings, client_factory=FakeRateLimitSmokeClient)
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "http_status=429" in captured.out
    assert "error_type=OcsRateLimitError" in captured.err
    assert "OCS rate limit reached" in captured.err


def test_docs_do_not_contain_ocs_api_key_values() -> None:
    paths = [Path("README.md"), Path(".env.example"), *Path("docs").glob("*.md")]
    token_assignment = re.compile(r"OCS_API_KEY[ \t]*=[ \t]*([^\s`]+)")
    allowed_values = {
        "",
        "<set",
        "<set-in-vps-env-only>",
        "<set-in-vps-.env-only>",
        "change-me",
        "placeholder",
    }
    offenders: list[str] = []

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for match in token_assignment.finditer(text):
            value = match.group(1).strip()
            if value not in allowed_values and not value.startswith("<"):
                offenders.append(f"{path}:{value}")

    assert offenders == []
