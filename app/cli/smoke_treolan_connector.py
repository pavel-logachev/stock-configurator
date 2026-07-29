from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import Protocol
from urllib.parse import quote, urlparse

from app.core.config import TreolanSettings, get_treolan_settings
from app.distributors.treolan.client import TreolanClient, TreolanError, TreolanHttpError
from app.distributors.treolan.parsing import flatten_treolan_categories

ENDPOINT_NAME = "GetCategories"


class SmokeTreolanClient(Protocol):
    async def get_categories(self) -> str:
        pass

    def rate_limit_summary(self) -> Mapping[str, float | int]:
        pass


class SmokeTreolanClientContext(AbstractAsyncContextManager[SmokeTreolanClient], Protocol):
    pass


def _base_url_host(base_url: str) -> str:
    parsed = urlparse(base_url)
    return parsed.hostname or "invalid"


def _rate_limit_summary(settings: TreolanSettings) -> dict[str, float | int]:
    requests_per_minute_limit = settings.treolan_requests_per_minute_limit
    minimum_interval_seconds = 0
    if requests_per_minute_limit > 0:
        minimum_interval_seconds = round(60 / requests_per_minute_limit, 3)
    return {
        "requests_per_minute_limit": requests_per_minute_limit,
        "minimum_interval_seconds": minimum_interval_seconds,
        "request_delay_seconds": settings.treolan_request_delay_seconds,
    }


def _print_common_summary(settings: TreolanSettings, *, auth_configured: bool) -> None:
    rate_limit = _rate_limit_summary(settings)
    print("Treolan connector smoke summary")
    print(f"base_url_host={_base_url_host(settings.treolan_base_url)}")
    print(f"auth_configured={str(auth_configured).lower()}")
    print(f"endpoint={ENDPOINT_NAME}")
    print(
        "rate_limit.requests_per_minute_limit="
        f"{rate_limit['requests_per_minute_limit']}"
    )
    print(f"rate_limit.minimum_interval_seconds={rate_limit['minimum_interval_seconds']}")
    print(f"rate_limit.request_delay_seconds={rate_limit['request_delay_seconds']}")


def _redact(message: str, settings: TreolanSettings) -> str:
    redacted = message
    for secret in (settings.treolan_login.strip(), settings.treolan_password.strip()):
        if not secret:
            continue
        redacted = redacted.replace(secret, "[redacted]")
        encoded_secret = quote(secret, safe="")
        if encoded_secret != secret:
            redacted = redacted.replace(encoded_secret, "[redacted]")
    base_url = settings.treolan_base_url.rstrip("/")
    if base_url:
        redacted = redacted.replace(base_url, "[base_url]")
    return redacted


def _advice(exc: TreolanError) -> str:
    status_code = exc.status_code if isinstance(exc, TreolanHttpError) else None
    if status_code in {401, 403}:
        return "check TREOLAN_LOGIN/TREOLAN_PASSWORD and B2B API access; secrets are not printed"
    if status_code == 429:
        return "Treolan rate limit reached; wait before retrying and keep sync conservative"
    return "check network connectivity, DNS, TLS and Treolan connector availability"


async def run_smoke(
    *,
    settings: TreolanSettings | None = None,
    client_factory: Callable[[TreolanSettings], SmokeTreolanClientContext] | None = None,
) -> int:
    treolan_settings = settings or get_treolan_settings()
    auth_configured = bool(
        treolan_settings.treolan_login.strip()
        and treolan_settings.treolan_password.strip()
    )
    _print_common_summary(treolan_settings, auth_configured=auth_configured)

    if not auth_configured:
        print("http_status=not_requested")
        print("items_count=0")
        print("error_type=TreolanConfigurationError", file=sys.stderr)
        print(
            "message=TREOLAN_LOGIN and TREOLAN_PASSWORD are not set. "
            "Put them only in the VPS .env.",
            file=sys.stderr,
        )
        return 1

    factory = client_factory or TreolanClient
    try:
        async with factory(treolan_settings) as client:
            payload = await client.get_categories()
    except TreolanError as exc:
        status_code = "unavailable"
        if isinstance(exc, TreolanHttpError) and exc.status_code:
            status_code = exc.status_code
        print(f"http_status={status_code}")
        print("items_count=0")
        print(f"error_type={type(exc).__name__}", file=sys.stderr)
        print(f"message={_redact(str(exc), treolan_settings)}", file=sys.stderr)
        print(f"advice={_advice(exc)}", file=sys.stderr)
        return 1

    items_count = len(flatten_treolan_categories(payload))
    print("http_status=200")
    print(f"items_count={items_count}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run_smoke()))


if __name__ == "__main__":
    main()
