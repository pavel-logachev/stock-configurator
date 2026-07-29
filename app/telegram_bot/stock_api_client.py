from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class StockApiClientError(RuntimeError):
    """Raised when the bot cannot complete a stock-api request."""


class StockApiClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        v3_timeout_seconds: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._v3_timeout = httpx.Timeout(
            timeout_seconds if v3_timeout_seconds is None else v3_timeout_seconds
        )
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Accept": "application/json"},
        )

    async def create_match(self, user_text: str) -> dict[str, Any]:
        text = user_text.strip()
        if not text:
            raise StockApiClientError("Match request text is empty.")

        response = await self._request(
            "POST",
            "/api/v1/match",
            operation="create match",
            json={"text": text},
        )
        payload = self._json_response(response, operation="create match")
        self._validate_match_summary(payload)
        return payload

    async def create_v3_full_category_quote(
        self,
        *,
        profile: str | None = None,
        distributor_code: str | None = None,
        user_text: str,
    ) -> dict[str, Any]:
        text = user_text.strip()
        if not text:
            raise StockApiClientError("V3 full-category request text is empty.")

        request_json: dict[str, Any] = {"text": text}
        if profile:
            request_json["profile"] = profile
        if distributor_code:
            request_json["distributor_code"] = distributor_code.strip()

        response = await self._request(
            "POST",
            "/api/v1/match/v3/full-category",
            operation="create v3 full-category quote",
            json=request_json,
            timeout=self._v3_timeout,
        )
        payload = self._json_response(
            response,
            operation="create v3 full-category quote",
        )
        self._validate_v3_full_category_quote(payload)
        return payload

    async def create_simple_stock_quote(
        self,
        *,
        profile: str | None = None,
        distributor_code: str | None = None,
        user_text: str,
    ) -> dict[str, Any]:
        text = user_text.strip()
        if not text:
            raise StockApiClientError("Simple stock quote request text is empty.")

        request_json: dict[str, Any] = {"text": text}
        if profile:
            request_json["profile"] = profile
        if distributor_code:
            request_json["distributor_code"] = distributor_code.strip()

        response = await self._request(
            "POST",
            "/api/v1/match/v3/simple-stock-quote",
            operation="create simple stock quote",
            json=request_json,
            timeout=self._v3_timeout,
        )
        payload = self._json_response(
            response,
            operation="create simple stock quote",
        )
        self._validate_v3_full_category_quote(payload)
        return payload

    async def create_v3_full_category_quote_auto(
        self,
        user_text: str,
        *,
        distributor_code: str | None = None,
    ) -> dict[str, Any]:
        return await self.create_simple_stock_quote(
            user_text=user_text,
            distributor_code=distributor_code,
        )

    async def get_match_summary(self, match_run_id: int) -> dict[str, Any]:
        response = await self._request(
            "GET",
            f"/api/v1/match/{match_run_id}",
            operation="load match",
        )
        payload = self._json_response(response, operation="load match")
        summary = self._normalize_match_summary(payload, requested_match_run_id=match_run_id)
        self._validate_match_summary(summary)
        return summary

    async def get_match_report_markdown(self, match_run_id: int) -> str:
        response = await self._request(
            "GET",
            f"/api/v1/match/{match_run_id}/report.md",
            operation="load match report",
        )
        return response.text

    async def get_match_report_xlsx(self, match_run_id: int) -> bytes:
        response = await self._request(
            "GET",
            f"/api/v1/match/{match_run_id}/report.xlsx",
            operation="load match Excel report",
            headers={"Accept": XLSX_MEDIA_TYPE},
        )
        if not response.content:
            raise StockApiClientError("Stock API returned an empty Excel report.")
        return response.content

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        *,
        operation: str,
        **kwargs: Any,
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._response_error_detail(exc.response)
            raise StockApiClientError(
                f"Stock API could not {operation}: HTTP {exc.response.status_code}. {detail}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise StockApiClientError(f"Stock API timed out while trying to {operation}.") from exc
        except httpx.RequestError as exc:
            raise StockApiClientError(
                f"Stock API is unavailable while trying to {operation}."
            ) from exc
        return response

    @staticmethod
    def _json_response(response: httpx.Response, *, operation: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise StockApiClientError(f"Stock API returned invalid JSON for {operation}.") from exc
        if not isinstance(payload, dict):
            raise StockApiClientError(f"Stock API returned an invalid response for {operation}.")
        return payload

    @staticmethod
    def _response_error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return "Response detail was not JSON."

        if isinstance(payload, Mapping):
            detail = payload.get("detail")
            if detail:
                return "Response detail was provided by stock-api."
        return "No response detail."

    @staticmethod
    def _validate_match_summary(payload: Mapping[str, Any]) -> None:
        required_keys = {
            "match_run_id",
            "status",
            "engineer_review_required",
            "total_candidates",
            "matched_items",
            "risk_flags",
            "missing_requirements",
            "candidates",
            "report_url",
        }
        missing_keys = sorted(required_keys.difference(payload))
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise StockApiClientError(f"Stock API response is missing required fields: {missing}.")

    @staticmethod
    def _validate_v3_full_category_quote(payload: Mapping[str, Any]) -> None:
        required_keys = {
            "category_ids",
            "diagnostics",
            "distributor_code",
            "engineering_review_required",
            "result_state",
            "validated_quote",
        }
        missing_keys = sorted(required_keys.difference(payload))
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise StockApiClientError(
                f"Stock API v3 response is missing required fields: {missing}."
            )

    @staticmethod
    def _normalize_match_summary(
        payload: Mapping[str, Any],
        *,
        requested_match_run_id: int,
    ) -> dict[str, Any]:
        report_json = payload.get("report_json")
        summary: dict[str, Any] = dict(report_json) if isinstance(report_json, Mapping) else {}
        summary.update(payload)

        match_run_id = (
            payload.get("match_run_id")
            or payload.get("id")
            or summary.get("match_run_id")
            or requested_match_run_id
        )
        summary["match_run_id"] = match_run_id
        summary.setdefault("confirmation_text", None)
        summary.setdefault("risk_flags", [])
        summary.setdefault("missing_requirements", [])
        summary.setdefault("candidates", [])
        summary.setdefault("ready_stock_candidates", [])
        summary.setdefault("build_candidates", [])
        summary.setdefault("report_url", f"/api/v1/match/{match_run_id}/report.md")
        summary.setdefault("report_xlsx_url", f"/api/v1/match/{match_run_id}/report.xlsx")
        if summary.get("pipeline_version") in {
            "v3_full_category_matrix",
            "simple_stock_quote",
        }:
            summary.setdefault("result_state", summary.get("v3_result_state"))
            summary.setdefault("profile", summary.get("v3_profile"))
            summary.setdefault("category_ids", summary.get("category_ids") or [])
            summary.setdefault("distributor_code", summary.get("distributor_code") or "ocs")
            summary.setdefault("validated_quote", summary.get("validated_quote") or {})
            summary.setdefault("diagnostics", summary.get("diagnostics") or {})
            summary.setdefault(
                "engineering_review_required",
                bool(
                    (summary.get("validated_quote") or {}).get(
                        "engineering_review_required",
                        summary.get("result_state") == "quote_draft_review_required",
                    )
                ),
            )
        return summary
