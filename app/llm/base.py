from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class LlmError(Exception):
    """Base error for LLM integration failures."""


class LlmConfigurationError(LlmError):
    """Raised when LLM settings are incomplete or unsupported."""


class LlmHttpError(LlmError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LlmClientError(LlmHttpError):
    """Raised for network errors and non-specific LLM 4xx responses."""


class LlmReadTimeoutError(LlmClientError):
    """Raised when the LLM request read phase times out and must not be retried."""


class LlmServerError(LlmHttpError):
    """Raised for LLM 5xx responses."""


class LlmInvalidJsonError(LlmError):
    """Raised when an LLM response is not valid JSON for the requested schema."""

    def __init__(
        self,
        message: str,
        *,
        parse_stage: str | None = None,
        json_extract_status: str | None = None,
        invalid_json_reason: str | None = None,
        preview_sanitized: str | None = None,
    ) -> None:
        super().__init__(message)
        self.parse_stage = parse_stage
        self.json_extract_status = json_extract_status
        self.invalid_json_reason = invalid_json_reason
        self.preview_sanitized = preview_sanitized


@runtime_checkable
class LlmClient(Protocol):
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Generate a JSON object from prompts."""
