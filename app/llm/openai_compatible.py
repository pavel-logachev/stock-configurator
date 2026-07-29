from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from app.core.config import LlmSettings, get_llm_settings
from app.llm.base import (
    LlmClientError,
    LlmConfigurationError,
    LlmInvalidJsonError,
    LlmReadTimeoutError,
    LlmServerError,
)


class OpenAICompatibleLlmClient:
    CHAT_COMPLETIONS_PATH = "/chat/completions"
    RETRYABLE_HTTP_STATUSES = {502, 503, 504, 520, 522, 524}

    def __init__(
        self,
        settings: LlmSettings | None = None,
        http_client: httpx.Client | None = None,
        *,
        timeout_seconds: float | None = None,
        read_timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
        use_response_format: bool = True,
        max_retries: int = 1,
        thinking_enabled: bool | None = None,
        thinking_budget_tokens: int | None = None,
    ) -> None:
        self._settings = settings or get_llm_settings()
        self._base_url = self._settings.llm_base_url.strip().rstrip("/")
        self._api_key = self._settings.llm_api_key.strip()
        self._model = self._settings.llm_model.strip()
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else self._settings.llm_timeout_seconds
        )
        self._read_timeout_seconds = read_timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._use_response_format = use_response_format
        self._max_retries = max(0, max_retries)
        self._thinking_enabled = (
            bool(thinking_enabled)
            if thinking_enabled is not None
            else bool(self._settings.llm_configurator_thinking_enabled)
        )
        self._thinking_budget_tokens = (
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else self._settings.llm_configurator_thinking_budget_tokens
        )
        self._thinking_fallback_reason: str | None = None
        self._http_client = http_client or httpx.Client()
        self._owns_http_client = http_client is None

        self._validate_settings()

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def safe_diagnostics(self) -> dict[str, Any]:
        return {
            "llm_thinking_enabled": self._thinking_enabled,
            "llm_thinking_budget_tokens": self._thinking_budget_tokens,
            "llm_thinking_fallback_reason": self._thinking_fallback_reason,
        }

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = self._build_payload(
            system_prompt,
            user_prompt,
            use_response_format=self._use_response_format,
            include_thinking=self._thinking_enabled,
        )

        response = self._post_chat_completion(payload)
        if (
            response.status_code in {400, 422}
            and self._thinking_enabled
            and self._looks_like_thinking_rejection(response)
        ):
            self._thinking_fallback_reason = "thinking_params_rejected"
            payload = self._build_payload(
                system_prompt,
                user_prompt,
                use_response_format=self._use_response_format,
                include_thinking=False,
            )
            response = self._post_chat_completion(payload)
        if (
            response.status_code == 400
            and self._use_response_format
            and self._looks_like_response_format_rejection(response)
        ):
            fallback_payload = self._build_payload(
                system_prompt,
                user_prompt,
                use_response_format=False,
                include_thinking=(
                    self._thinking_enabled
                    and self._thinking_fallback_reason is None
                ),
            )
            response = self._post_chat_completion(fallback_payload)
            if (
                response.status_code in {400, 422}
                and self._thinking_enabled
                and self._thinking_fallback_reason is None
                and self._looks_like_thinking_rejection(response)
            ):
                self._thinking_fallback_reason = "thinking_params_rejected"
                fallback_payload = self._build_payload(
                    system_prompt,
                    user_prompt,
                    use_response_format=False,
                    include_thinking=False,
                )
                response = self._post_chat_completion(fallback_payload)

        self._raise_for_status(response)
        return self._decode_chat_completion(response)

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        use_response_format: bool,
        include_thinking: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        if self._max_output_tokens is not None:
            payload["max_tokens"] = self._max_output_tokens
        if use_response_format:
            payload["response_format"] = {"type": "json_object"}
        if include_thinking:
            payload["enable_thinking"] = True
            if self._thinking_budget_tokens is not None:
                payload["thinking_budget"] = self._thinking_budget_tokens
        return payload

    def _post_chat_completion(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http_client.post(
                    self._completion_url(),
                    headers=self._headers(),
                    json=payload,
                    timeout=self._timeout(),
                )
            except httpx.ReadTimeout as exc:
                message = self._sanitize(f"LLM request read timed out: {exc}")
                raise LlmReadTimeoutError(message) from exc
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt < self._max_retries:
                    continue
                message = self._sanitize(f"LLM request failed: {exc}")
                raise LlmClientError(message) from exc
            except httpx.RequestError as exc:
                message = self._sanitize(f"LLM request failed: {exc}")
                raise LlmClientError(message) from exc

            if (
                response.status_code in self.RETRYABLE_HTTP_STATUSES
                and attempt < self._max_retries
            ):
                continue
            return response

        raise LlmClientError("LLM request failed before receiving a response.")

    def _validate_settings(self) -> None:
        if not self._base_url:
            raise LlmConfigurationError("LLM_BASE_URL is not set.")
        if not self._base_url.startswith(("http://", "https://")):
            raise LlmConfigurationError("LLM_BASE_URL must start with http:// or https://.")
        if not self._api_key:
            raise LlmConfigurationError("LLM_API_KEY is not set.")
        if not self._model:
            raise LlmConfigurationError("LLM_MODEL is not set.")

    def _headers(self) -> dict[str, str]:
        return {
            "accept": "application/json",
            "authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }

    def _completion_url(self) -> str:
        parsed = urlparse(self._base_url)
        path = parsed.path.rstrip("/")
        if path.endswith(self.CHAT_COMPLETIONS_PATH):
            return self._base_url
        return f"{self._base_url}{self.CHAT_COMPLETIONS_PATH}"

    def _timeout(self) -> httpx.Timeout:
        if self._read_timeout_seconds is None:
            return httpx.Timeout(self._timeout_seconds)
        return httpx.Timeout(self._timeout_seconds, read=self._read_timeout_seconds)

    def _raise_for_status(self, response: httpx.Response) -> None:
        status_code = response.status_code
        if status_code < 400:
            return

        message = self._error_message(response)
        if status_code >= 500:
            raise LlmServerError(message, status_code=status_code)
        raise LlmClientError(message, status_code=status_code)

    def _looks_like_response_format_rejection(self, response: httpx.Response) -> bool:
        text = response.text.casefold()
        return "response_format" in text or "json_schema" in text

    def _looks_like_thinking_rejection(self, response: httpx.Response) -> bool:
        text = response.text.casefold()
        return (
            "enable_thinking" in text
            or "thinking_budget" in text
            or "thinking" in text
            or "unknown field" in text
            or "unsupported field" in text
            or "extra fields" in text
        )

    def _error_message(self, response: httpx.Response) -> str:
        try:
            request = response.request
        except RuntimeError:
            request = None

        location = request.url.path if request else "unknown endpoint"
        message = f"LLM API returned HTTP {response.status_code} for {location}"
        body = response.text.strip()
        if body:
            message = f"{message}: {body[:500]}"
        return self._sanitize(message)

    def _decode_chat_completion(self, response: httpx.Response) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise LlmInvalidJsonError(
                "LLM response body was not valid JSON.",
                parse_stage="response_body",
                json_extract_status="parse_error",
                invalid_json_reason="response_body_not_json",
                preview_sanitized=self._preview(response.text),
            ) from exc

        if not isinstance(body, dict):
            raise LlmInvalidJsonError(
                "LLM response body must be a JSON object.",
                parse_stage="response_body",
                json_extract_status="invalid_type",
                invalid_json_reason="response_body_not_object",
                preview_sanitized=self._preview(body),
            )

        content = self._extract_message_content(body)
        if isinstance(content, dict):
            return content
        if isinstance(content, list):
            return {"recommendations": content}
        if not isinstance(content, str):
            raise LlmInvalidJsonError(
                "LLM response message content must be a JSON string.",
                parse_stage="message_content",
                json_extract_status="invalid_type",
                invalid_json_reason="message_content_not_string",
                preview_sanitized=self._preview(content),
            )

        try:
            parsed = self._parse_message_content_json(content)
        except LlmInvalidJsonError:
            raise

        if not isinstance(parsed, dict):
            if isinstance(parsed, list):
                return {"recommendations": parsed}
            raise LlmInvalidJsonError(
                "LLM response content must be a JSON object.",
                parse_stage="message_content",
                json_extract_status="invalid_type",
                invalid_json_reason="message_content_root_not_object",
                preview_sanitized=self._preview(content),
            )
        return parsed

    def _parse_message_content_json(self, content: str) -> Any:
        normalized = self._normalize_json_content(content)
        try:
            return json.loads(normalized)
        except json.JSONDecodeError as first_exc:
            candidate = self._extract_first_balanced_json(normalized)
            if candidate is not None and candidate != normalized:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as second_exc:
                    raise LlmInvalidJsonError(
                        "LLM response content was not valid JSON.",
                        parse_stage="message_content",
                        json_extract_status="repair_parse_error",
                        invalid_json_reason=str(second_exc),
                        preview_sanitized=self._preview(content),
                    ) from second_exc
            raise LlmInvalidJsonError(
                "LLM response content was not valid JSON.",
                parse_stage="message_content",
                json_extract_status="parse_error",
                invalid_json_reason=str(first_exc),
                preview_sanitized=self._preview(content),
            ) from first_exc

    def _normalize_json_content(self, content: str) -> str:
        text = content.lstrip("\ufeff").strip()
        text = re.sub(r"(?is)<think\b[^>]*>.*?</think>", "", text).strip()
        fence_match = re.fullmatch(r"(?is)```(?:json)?\s*(.*?)\s*```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        return text

    def _extract_first_balanced_json(self, text: str) -> str | None:
        for index, character in enumerate(text):
            if character not in "{[":
                continue
            candidate = self._balanced_json_from(text, index)
            if candidate is not None:
                return candidate
        return None

    def _balanced_json_from(self, text: str, start: int) -> str | None:
        opening = text[start]
        closing_by_opening = {"{": "}", "[": "]"}
        stack = [closing_by_opening[opening]]
        in_string = False
        escape = False
        for index in range(start + 1, len(text)):
            character = text[index]
            if in_string:
                if escape:
                    escape = False
                elif character == "\\":
                    escape = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
                continue
            if character in closing_by_opening:
                stack.append(closing_by_opening[character])
                continue
            if stack and character == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start : index + 1].strip()
        return None

    def _extract_message_content(self, body: dict[str, Any]) -> Any:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmInvalidJsonError(
                "LLM response does not contain choices.",
                parse_stage="response_body",
                json_extract_status="missing_choices",
                invalid_json_reason="response_body_missing_choices",
                preview_sanitized=self._preview(body),
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LlmInvalidJsonError(
                "LLM response choice must be a JSON object.",
                parse_stage="response_body",
                json_extract_status="invalid_type",
                invalid_json_reason="response_choice_not_object",
                preview_sanitized=self._preview(first_choice),
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LlmInvalidJsonError(
                "LLM response choice does not contain message.",
                parse_stage="response_body",
                json_extract_status="missing_message",
                invalid_json_reason="response_choice_missing_message",
                preview_sanitized=self._preview(first_choice),
            )

        return message.get("content")

    def _sanitize(self, value: str) -> str:
        sanitized = value.replace(self._api_key, "[redacted]")
        encoded_api_key = quote(self._api_key, safe="")
        if encoded_api_key != self._api_key:
            sanitized = sanitized.replace(encoded_api_key, "[redacted]")
        sanitized = re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+",
            "[redacted]",
            sanitized,
        )
        sanitized = re.sub(
            r"(?i)\b(api[_-]?key|authorization|token)\b\s*[:=]\s*['\"]?[^'\"\s,}]+",
            "[redacted]",
            sanitized,
        )
        return sanitized

    def _preview(self, value: Any, *, limit: int = 500) -> str:
        text = value if isinstance(value, str) else str(value)
        text = " ".join(text.replace("\x00", "").split())
        return self._sanitize(text)[:limit]
