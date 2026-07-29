from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import quote

import httpx

from app.core.config import OcsSettings, get_ocs_settings
from app.distributors.base import DistributorClient, DistributorOffer


class OcsError(Exception):
    """Base error for OCS integration failures."""


class OcsConfigurationError(OcsError):
    """Raised when OCS settings are incomplete."""


class OcsHttpError(OcsError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OcsClientError(OcsHttpError):
    """Raised for network errors and non-specific OCS 4xx responses."""


class OcsUnauthorizedError(OcsClientError):
    """Raised when OCS rejects the API key."""


class OcsForbiddenError(OcsClientError):
    """Raised when the API key has no access to the requested resource."""


class OcsNotFoundError(OcsClientError):
    """Raised when OCS does not have the requested resource."""


class OcsRateLimitError(OcsClientError):
    """Raised when OCS asks the client to slow down."""


class OcsServerError(OcsHttpError):
    """Raised for OCS 5xx responses."""


class OcsClient(DistributorClient):
    name = "ocs"

    CHECK_CONNECTION_PATH = "/api/v2/CheckConnection"
    SHIPMENT_CITIES_PATH = "/api/v2/logistic/shipment/cities"
    STOCK_LOCATIONS_PATH = "/api/v2/logistic/stocks/locations"
    CATEGORIES_PATH = "/api/v2/catalog/categories"
    PRODUCTS_BY_CATEGORY_PATH_TEMPLATE = "/api/v2/catalog/categories/{category}/products"
    PRODUCTS_BATCH_PATH = "/api/v2/catalog/products/batch"
    CONTENT_PATH_TEMPLATE = "/api/v2/content/{item_ids}"
    CONTENT_BATCH_PATH = "/api/v2/content/batch"

    def __init__(
        self,
        settings: OcsSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_ocs_settings()
        self._base_url = self._settings.ocs_base_url.rstrip("/")
        self._api_key = self._settings.ocs_api_key.strip()
        self._timeout_seconds = self._settings.ocs_timeout_seconds
        self._request_delay_seconds = self._settings.ocs_request_delay_seconds
        self._requests_per_hour_limit = self._settings.ocs_requests_per_hour_limit
        self._request_interval_seconds = self._minimum_request_interval_seconds(
            self._requests_per_hour_limit
        )
        self._last_request_started_at: float | None = None
        self._rate_limit_lock = asyncio.Lock()
        self._http_client = http_client or httpx.AsyncClient()
        self._owns_http_client = http_client is None

        if not self._api_key:
            raise OcsConfigurationError(
                "OCS_API_KEY is not set. Set it in the environment or .env."
            )

    async def __aenter__(self) -> OcsClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def fetch_offers(self) -> list[DistributorOffer]:
        raise NotImplementedError("OCS offer ingestion will be implemented in a later step.")

    async def check_connection(self) -> Any:
        return await self._get(self.CHECK_CONNECTION_PATH)

    async def get_shipment_cities(self) -> Any:
        return await self._get(self.SHIPMENT_CITIES_PATH)

    async def get_stock_locations(self, shipment_city: str) -> Any:
        return await self._get(
            self.STOCK_LOCATIONS_PATH,
            params={"shipmentcity": shipment_city},
        )

    async def get_categories(self) -> Any:
        return await self._get(self.CATEGORIES_PATH)

    async def get_products_by_category(
        self,
        category: str,
        shipment_city: str,
        only_available: bool = True,
        include_regular: bool = True,
        include_sale: bool = False,
        include_uncondition: bool = False,
        include_missing: bool = False,
        with_descriptions: bool = False,
    ) -> Any:
        params = {
            "shipmentcity": shipment_city,
            **self._product_filter_params(
                only_available=only_available,
                include_regular=include_regular,
                include_sale=include_sale,
                include_uncondition=include_uncondition,
                include_missing=include_missing,
                with_descriptions=with_descriptions,
            ),
        }
        return await self._get(self._products_by_category_path(category), params=params)

    async def get_products_batch(
        self,
        item_ids: list[str],
        shipment_city: str,
        only_available: bool = True,
        include_regular: bool = True,
        include_sale: bool = False,
        include_uncondition: bool = False,
        include_missing: bool = False,
        with_descriptions: bool = True,
    ) -> Any:
        self._validate_item_ids(item_ids)
        params = {
            "shipmentcity": shipment_city,
            **self._product_filter_params(
                only_available=only_available,
                include_regular=include_regular,
                include_sale=include_sale,
                include_uncondition=include_uncondition,
                include_missing=include_missing,
                with_descriptions=with_descriptions,
            ),
        }
        return await self._post(self.PRODUCTS_BATCH_PATH, params=params, json=list(item_ids))

    async def get_content_batch(self, item_ids: list[str]) -> Any:
        self._validate_item_ids(item_ids)
        return await self._post(self.CONTENT_BATCH_PATH, json=list(item_ids))

    async def get_content(self, item_ids: list[str]) -> Any:
        self._validate_item_ids(item_ids)
        encoded_item_ids = quote(",".join(item_ids), safe=",")
        return await self._get(
            self.CONTENT_PATH_TEMPLATE.format(item_ids=encoded_item_ids)
        )

    async def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params)

    async def _post(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
    ) -> Any:
        return await self._request("POST", path, params=params, json=json)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
    ) -> Any:
        await self._throttle()

        try:
            response = await self._http_client.request(
                method,
                self._url(path),
                headers=self._headers(),
                params=self._clean_params(params),
                json=json,
                timeout=self._timeout_seconds,
            )
        except httpx.RequestError as exc:
            message = self._sanitize(f"OCS request failed: {exc}")
            raise OcsClientError(message) from exc

        self._raise_for_status(response)
        return self._decode_response(response)

    def _headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "X-API-Key": self._api_key,
        }

    async def _throttle(self) -> None:
        async with self._rate_limit_lock:
            min_interval_seconds = max(
                self._request_delay_seconds,
                self._request_interval_seconds,
            )
            if self._last_request_started_at is None:
                if self._request_delay_seconds > 0:
                    await asyncio.sleep(self._request_delay_seconds)
                self._last_request_started_at = time.monotonic()
                return

            if min_interval_seconds <= 0:
                self._last_request_started_at = time.monotonic()
                return

            elapsed_seconds = time.monotonic() - self._last_request_started_at
            wait_seconds = min_interval_seconds - elapsed_seconds
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)

            self._last_request_started_at = time.monotonic()

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _clean_params(self, params: dict[str, Any] | None) -> dict[str, Any] | None:
        if params is None:
            return None
        return {key: value for key, value in params.items() if value is not None}

    def _products_by_category_path(self, category: str) -> str:
        encoded_category = quote(category, safe="")
        return self.PRODUCTS_BY_CATEGORY_PATH_TEMPLATE.format(category=encoded_category)

    def _product_filter_params(
        self,
        *,
        only_available: bool,
        include_regular: bool,
        include_sale: bool,
        include_uncondition: bool,
        include_missing: bool,
        with_descriptions: bool,
    ) -> dict[str, bool]:
        return {
            "onlyavailable": only_available,
            "includeregular": include_regular,
            "includesale": include_sale,
            "includeuncondition": include_uncondition,
            "includemissing": include_missing,
            "withdescriptions": with_descriptions,
        }

    def _validate_item_ids(self, item_ids: list[str]) -> None:
        if not item_ids:
            raise ValueError("item_ids must contain at least one item id.")

    def rate_limit_summary(self) -> dict[str, float | int]:
        return {
            "requests_per_hour_limit": self._requests_per_hour_limit,
            "minimum_interval_seconds": round(self._request_interval_seconds, 3),
            "request_delay_seconds": self._request_delay_seconds,
        }

    def _raise_for_status(self, response: httpx.Response) -> None:
        status_code = response.status_code
        if status_code < 400:
            return

        message = self._error_message(response)

        if status_code == 401:
            raise OcsUnauthorizedError(message, status_code=status_code)
        if status_code == 403:
            raise OcsForbiddenError(message, status_code=status_code)
        if status_code == 404:
            raise OcsNotFoundError(message, status_code=status_code)
        if status_code == 429:
            raise OcsRateLimitError(message, status_code=status_code)
        if 400 <= status_code < 500:
            raise OcsClientError(message, status_code=status_code)
        if status_code >= 500:
            raise OcsServerError(message, status_code=status_code)

    def _error_message(self, response: httpx.Response) -> str:
        try:
            request = response.request
        except RuntimeError:
            request = None
        location = request.url.path if request else "unknown endpoint"
        message = f"OCS API returned HTTP {response.status_code} for {location}"

        body = response.text.strip()
        if body:
            message = f"{message}: {body[:500]}"

        return self._sanitize(message)

    def _decode_response(self, response: httpx.Response) -> Any:
        if not response.content:
            return None

        try:
            return response.json()
        except ValueError:
            return response.text

    def _sanitize(self, value: str) -> str:
        sanitized = value.replace(self._api_key, "[redacted]")
        encoded_api_key = quote(self._api_key, safe="")
        if encoded_api_key != self._api_key:
            sanitized = sanitized.replace(encoded_api_key, "[redacted]")
        return sanitized

    def _minimum_request_interval_seconds(self, requests_per_hour_limit: int) -> float:
        if requests_per_hour_limit <= 0:
            return 0
        return 3600 / requests_per_hour_limit
