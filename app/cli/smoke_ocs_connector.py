from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol
from urllib.parse import quote, urlparse

from app.core.config import OcsSettings, get_ocs_settings
from app.distributors.ocs.client import OcsClient, OcsError, OcsHttpError

ENDPOINT_NAME = "catalog.categories"


class SmokeOcsClient(Protocol):
    async def get_categories(self) -> Any:
        pass

    def rate_limit_summary(self) -> Mapping[str, float | int]:
        pass


class SmokeOcsClientContext(AbstractAsyncContextManager[SmokeOcsClient], Protocol):
    pass


def _root_count(value: Any, preferred_keys: tuple[str, ...]) -> int:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return len(value)
    if isinstance(value, Mapping):
        for key in preferred_keys:
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(
                nested, str | bytes | bytearray
            ):
                return len(nested)
        return len(value)
    return 0


def _base_url_host(base_url: str) -> str:
    parsed = urlparse(base_url)
    return parsed.hostname or "invalid"


def _rate_limit_summary(settings: OcsSettings) -> dict[str, float | int]:
    requests_per_hour_limit = settings.ocs_requests_per_hour_limit
    minimum_interval_seconds = 0
    if requests_per_hour_limit > 0:
        minimum_interval_seconds = round(3600 / requests_per_hour_limit, 3)
    return {
        "requests_per_hour_limit": requests_per_hour_limit,
        "minimum_interval_seconds": minimum_interval_seconds,
        "request_delay_seconds": settings.ocs_request_delay_seconds,
    }


def _print_common_summary(settings: OcsSettings, *, auth_configured: bool) -> None:
    rate_limit = _rate_limit_summary(settings)
    print("OCS connector smoke summary")
    print(f"base_url_host={_base_url_host(settings.ocs_base_url)}")
    print(f"auth_configured={str(auth_configured).lower()}")
    print(f"endpoint={ENDPOINT_NAME}")
    print(f"rate_limit.requests_per_hour_limit={rate_limit['requests_per_hour_limit']}")
    print(f"rate_limit.minimum_interval_seconds={rate_limit['minimum_interval_seconds']}")
    print(f"rate_limit.request_delay_seconds={rate_limit['request_delay_seconds']}")


def _redact(message: str, settings: OcsSettings) -> str:
    redacted = message
    api_key = settings.ocs_api_key.strip()
    if api_key:
        redacted = redacted.replace(api_key, "[redacted]")
        encoded_api_key = quote(api_key, safe="")
        if encoded_api_key != api_key:
            redacted = redacted.replace(encoded_api_key, "[redacted]")
    base_url = settings.ocs_base_url.rstrip("/")
    if base_url:
        redacted = redacted.replace(base_url, "[base_url]")
    return redacted


def _advice(exc: OcsError) -> str:
    status_code = exc.status_code if isinstance(exc, OcsHttpError) else None
    if status_code in {401, 403}:
        return "check OCS_API_KEY and confirm the VPS allowed IP; token is not printed"
    if status_code == 429:
        return "OCS rate limit reached; wait before retrying and keep sync conservative"
    return "check network connectivity, DNS, TLS and connector availability"


async def run_smoke(
    *,
    settings: OcsSettings | None = None,
    client_factory: Callable[[OcsSettings], SmokeOcsClientContext] | None = None,
) -> int:
    ocs_settings = settings or get_ocs_settings()
    auth_configured = bool(ocs_settings.ocs_api_key.strip())
    _print_common_summary(ocs_settings, auth_configured=auth_configured)

    if not auth_configured:
        print("http_status=not_requested")
        print("items_count=0")
        print("error_type=OcsConfigurationError", file=sys.stderr)
        print(
            "message=OCS_API_KEY is not set. Put it only in the VPS .env.",
            file=sys.stderr,
        )
        return 1

    factory = client_factory or OcsClient
    try:
        async with factory(ocs_settings) as client:
            payload = await client.get_categories()
    except OcsError as exc:
        status_code = "unavailable"
        if isinstance(exc, OcsHttpError) and exc.status_code:
            status_code = exc.status_code
        print(f"http_status={status_code}")
        print("items_count=0")
        print(f"error_type={type(exc).__name__}", file=sys.stderr)
        print(f"message={_redact(str(exc), ocs_settings)}", file=sys.stderr)
        print(f"advice={_advice(exc)}", file=sys.stderr)
        return 1

    items_count = _root_count(payload, ("categories", "items", "data", "result"))
    print("http_status=200")
    print(f"items_count={items_count}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run_smoke()))


if __name__ == "__main__":
    main()
