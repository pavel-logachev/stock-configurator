from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

import httpx

from app.core.config import TreolanSettings, get_treolan_settings
from app.distributors.base import DistributorClient, DistributorOffer


class TreolanError(Exception):
    """Base error for Treolan integration failures."""


class TreolanConfigurationError(TreolanError):
    """Raised when Treolan settings are incomplete."""


class TreolanHttpError(TreolanError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TreolanClientError(TreolanHttpError):
    """Raised for network, SOAP fault and Treolan 4xx errors."""


class TreolanUnauthorizedError(TreolanClientError):
    """Raised when Treolan rejects credentials."""


class TreolanRateLimitError(TreolanClientError):
    """Raised when Treolan asks the client to slow down."""


class TreolanServerError(TreolanHttpError):
    """Raised for Treolan 5xx responses."""


class TreolanClient(DistributorClient):
    name = "treolan"

    SERVICE_PATH = "/ws/service.asmx"
    MESSAGE_NAMESPACE = "http://tempuri.org/treolan/message/"
    SOAP_ACTION_PREFIX = "http://tempuri.org/treolan/action/WebService."
    SOAP_ENCODING = "http://schemas.xmlsoap.org/soap/encoding/"
    SOAP_ENVELOPE_NAMESPACE = "http://schemas.xmlsoap.org/soap/envelope/"

    def __init__(
        self,
        settings: TreolanSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_treolan_settings()
        self._base_url = self._settings.treolan_base_url.rstrip("/")
        self._login = self._settings.treolan_login.strip()
        self._password = self._settings.treolan_password.strip()
        self._timeout_seconds = self._settings.treolan_timeout_seconds
        self._request_delay_seconds = self._settings.treolan_request_delay_seconds
        self._requests_per_minute_limit = self._settings.treolan_requests_per_minute_limit
        self._request_interval_seconds = self._minimum_request_interval_seconds(
            self._requests_per_minute_limit
        )
        self._last_request_started_at: float | None = None
        self._rate_limit_lock = asyncio.Lock()
        self._http_client = http_client or httpx.AsyncClient()
        self._owns_http_client = http_client is None

        if not self._login or not self._password:
            raise TreolanConfigurationError(
                "TREOLAN_LOGIN and TREOLAN_PASSWORD must be set in the environment or .env."
            )

    async def __aenter__(self) -> TreolanClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def fetch_offers(self) -> list[DistributorOffer]:
        raise NotImplementedError(
            "Treolan offer ingestion uses category/product sync into canonical tables."
        )

    async def get_categories(self) -> str:
        return await self._call("GetCategories", {})

    async def gen_catalog_v2(
        self,
        *,
        category: str = "",
        vendorid: str = "0",
        keywords: str = "",
        criterion: int | None = None,
        in_articul: bool = True,
        in_name: bool = True,
        in_mark: bool = False,
        show_nc: int | None = None,
        free_nom: bool | None = None,
    ) -> str:
        return await self._call(
            "GenCatalogV2",
            {
                "category": category,
                "vendorid": vendorid,
                "keywords": keywords,
                "criterion": (
                    self._settings.treolan_catalog_criterion
                    if criterion is None
                    else criterion
                ),
                "inArticul": in_articul,
                "inName": in_name,
                "inMark": in_mark,
                "showNc": (
                    self._settings.treolan_catalog_show_nc
                    if show_nc is None
                    else show_nc
                ),
                "freeNom": (
                    self._settings.treolan_catalog_free_nom_only
                    if free_nom is None
                    else free_nom
                ),
            },
        )

    async def product_info_v2(self, articul: str) -> str:
        return await self._call(
            "ProductInfoV2",
            {
                "Login": self._login,
                "password": self._password,
                "Articul": articul,
            },
            include_credentials=False,
        )

    async def _call(
        self,
        method_name: str,
        params: dict[str, Any],
        *,
        include_credentials: bool = True,
    ) -> str:
        await self._throttle()

        effective_params = dict(params)
        if include_credentials:
            effective_params = {
                "login": self._login,
                "password": self._password,
                **effective_params,
            }

        body = self._soap_body(method_name, effective_params)
        try:
            response = await self._http_client.post(
                self._service_url(),
                content=body,
                headers={
                    "Content-Type": "text/xml; charset=utf-8",
                    "SOAPAction": f'"{self.SOAP_ACTION_PREFIX}{method_name}"',
                },
                timeout=self._timeout_seconds,
            )
        except httpx.RequestError as exc:
            message = self._sanitize(f"Treolan request failed: {exc}")
            raise TreolanClientError(message) from exc

        self._raise_for_status(response)
        return self._decode_soap_response(response)

    def _soap_body(self, method_name: str, params: dict[str, Any]) -> str:
        param_xml = "".join(
            f"<{name}>{self._xml_text(value)}</{name}>" for name, value in params.items()
        )
        return (
            '<?xml version="1.0" encoding="utf-8"?>'
            f'<soap:Envelope xmlns:soap="{self.SOAP_ENVELOPE_NAMESPACE}" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xmlns:xsd="http://www.w3.org/2001/XMLSchema">'
            "<soap:Body>"
            f'<m:{method_name} xmlns:m="{self.MESSAGE_NAMESPACE}" '
            f'soap:encodingStyle="{self.SOAP_ENCODING}">'
            f"{param_xml}"
            f"</m:{method_name}>"
            "</soap:Body>"
            "</soap:Envelope>"
        )

    def _decode_soap_response(self, response: httpx.Response) -> str:
        if not response.content:
            return ""

        text = response.text
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            return text

        fault = self._first_text(root, "faultstring")
        if fault:
            raise TreolanClientError(self._sanitize(f"Treolan SOAP fault: {fault}"))

        result = self._first_text(root, "Result")
        return result if result is not None else text

    def _raise_for_status(self, response: httpx.Response) -> None:
        status_code = response.status_code
        if status_code < 400:
            return

        message = self._error_message(response)
        if status_code in {401, 403}:
            raise TreolanUnauthorizedError(message, status_code=status_code)
        if status_code == 429:
            raise TreolanRateLimitError(message, status_code=status_code)
        if 400 <= status_code < 500:
            raise TreolanClientError(message, status_code=status_code)
        raise TreolanServerError(message, status_code=status_code)

    def _error_message(self, response: httpx.Response) -> str:
        try:
            request = response.request
        except RuntimeError:
            request = None
        location = request.url.path if request else "unknown endpoint"
        message = f"Treolan API returned HTTP {response.status_code} for {location}"
        body = response.text.strip()
        if body:
            message = f"{message}: {body[:500]}"
        return self._sanitize(message)

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

    def rate_limit_summary(self) -> dict[str, float | int]:
        return {
            "requests_per_minute_limit": self._requests_per_minute_limit,
            "minimum_interval_seconds": round(self._request_interval_seconds, 3),
            "request_delay_seconds": self._request_delay_seconds,
        }

    def _service_url(self) -> str:
        return f"{self._base_url}{self.SERVICE_PATH}"

    def _sanitize(self, value: str) -> str:
        sanitized = value
        for secret in (self._login, self._password):
            if not secret:
                continue
            sanitized = sanitized.replace(secret, "[redacted]")
            encoded_secret = quote(secret, safe="")
            if encoded_secret != secret:
                sanitized = sanitized.replace(encoded_secret, "[redacted]")
        if self._base_url:
            sanitized = sanitized.replace(self._base_url, "[base_url]")
        return sanitized

    def _minimum_request_interval_seconds(self, requests_per_minute_limit: int) -> float:
        if requests_per_minute_limit <= 0:
            return 0
        return 60 / requests_per_minute_limit

    def _first_text(self, root: ElementTree.Element, local_name: str) -> str | None:
        for element in root.iter():
            if self._local_name(element.tag) == local_name:
                return element.text or ""
        return None

    def _local_name(self, tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def _xml_text(self, value: Any) -> str:
        import html

        text = ("true" if value else "false") if isinstance(value, bool) else str(value)
        return html.escape(text, quote=False)
