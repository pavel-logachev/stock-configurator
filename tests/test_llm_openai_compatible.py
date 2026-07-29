from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.core.config import LlmSettings
from app.llm.base import LlmClientError, LlmInvalidJsonError, LlmReadTimeoutError
from app.llm.openai_compatible import OpenAICompatibleLlmClient

SECRET = "super-secret-llm-token"


def _settings(base_url: str = "https://llm.example.test/v1") -> LlmSettings:
    return LlmSettings(
        llm_provider="openai-compatible",
        llm_base_url=base_url,
        llm_api_key=SECRET,
        llm_model="test-model",
        llm_timeout_seconds=10,
    )


def test_openai_compatible_client_posts_chat_completions_and_parses_json(
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"answer": 42})}}]},
        )

    caplog.set_level(logging.DEBUG)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        result = client.generate_json("system", "user")

    assert result == {"answer": 42}
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/v1/chat/completions"
    assert requests[0].headers["authorization"] == f"Bearer {SECRET}"

    payload = json.loads(requests[0].content)
    assert payload["model"] == "test-model"
    assert payload["temperature"] == 0
    assert "top_p" not in payload
    assert "seed" not in payload
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert SECRET not in caplog.text


def test_openai_compatible_client_keeps_api_key_out_of_error_text_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad token: {SECRET}")

    caplog.set_level(logging.DEBUG)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        with pytest.raises(LlmClientError) as exc_info:
            client.generate_json("system", "user")

    assert SECRET not in str(exc_info.value)
    assert SECRET not in caplog.text


def test_openai_compatible_client_does_not_double_append_full_chat_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(
            settings=_settings("https://llm.example.test/v1/chat/completions"),
            http_client=http_client,
        )
        result = client.generate_json("system", "user")

    assert result == {"ok": True}
    assert requests[0].url.path == "/v1/chat/completions"


def test_openai_compatible_client_raises_clear_error_for_invalid_json_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        with pytest.raises(LlmInvalidJsonError, match="content was not valid JSON") as exc_info:
            client.generate_json("system", "user")

    assert exc_info.value.parse_stage == "message_content"
    assert exc_info.value.json_extract_status == "parse_error"
    assert "not-json" in (exc_info.value.preview_sanitized or "")


def test_openai_compatible_client_parses_fenced_json_content() -> None:
    result = _generate_from_content('```json\n{"answer": 42}\n```')

    assert result == {"answer": 42}


def test_openai_compatible_client_extracts_json_after_leading_prose() -> None:
    result = _generate_from_content('Here is the JSON:\n{"answer": 42}\nThanks.')

    assert result == {"answer": 42}


def test_openai_compatible_client_strips_think_blocks_before_json() -> None:
    result = _generate_from_content('<think>hidden reasoning</think>\n{"answer": 42}')

    assert result == {"answer": 42}


def test_openai_compatible_client_wraps_root_array_as_recommendations() -> None:
    result = _generate_from_content('[{"recommendation_id": "llm_rec_1"}]')

    assert result == {"recommendations": [{"recommendation_id": "llm_rec_1"}]}


def test_openai_compatible_client_invalid_json_preview_is_sanitized() -> None:
    content = f"Authorization: Bearer {SECRET}\nnot-json api_key={SECRET}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        with pytest.raises(LlmInvalidJsonError) as exc_info:
            client.generate_json("system", "user")

    preview = exc_info.value.preview_sanitized or ""
    assert SECRET not in preview
    assert "Authorization" not in preview
    assert "api_key" not in preview


def test_openai_compatible_client_uses_composer_timeout_and_max_tokens() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(
            settings=_settings(),
            http_client=http_client,
            timeout_seconds=30,
            read_timeout_seconds=900,
            max_output_tokens=16384,
            use_response_format=False,
        )
        result = client.generate_json("system", "user")

    payload = json.loads(requests[0].content)
    assert result == {"ok": True}
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 16384
    assert "top_p" not in payload
    assert "seed" not in payload
    assert "response_format" not in payload
    assert requests[0].extensions["timeout"]["connect"] == 30
    assert requests[0].extensions["timeout"]["read"] == 900


def test_openai_compatible_client_read_timeout_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("read timed out", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        with pytest.raises(LlmReadTimeoutError):
            client.generate_json("system", "user")

    assert calls == 1


def test_openai_compatible_client_connect_timeout_can_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("connect timed out", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        result = client.generate_json("system", "user")

    assert calls == 2
    assert result == {"ok": True}


def test_openai_compatible_client_retries_transient_http_status() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503, text="temporarily unavailable")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        result = client.generate_json("system", "user")

    assert len(requests) == 2
    assert result == {"ok": True}


def test_openai_compatible_client_falls_back_when_response_format_is_rejected() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if "response_format" in payload:
            return httpx.Response(400, text="response_format is not supported")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        result = client.generate_json("system", "user")

    first_payload = json.loads(requests[0].content)
    second_payload = json.loads(requests[1].content)
    assert result == {"ok": True}
    assert first_payload["response_format"] == {"type": "json_object"}
    assert "response_format" not in second_payload


def test_openai_compatible_client_does_not_send_thinking_by_default() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        result = client.generate_json("system", "user")

    payload = json.loads(requests[0].content)
    assert result == {"ok": True}
    assert "enable_thinking" not in payload
    assert "thinking_budget" not in payload
    assert client.safe_diagnostics()["llm_thinking_enabled"] is False


def test_openai_compatible_client_falls_back_when_thinking_is_rejected() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        if payload.get("enable_thinking"):
            return httpx.Response(400, text="unknown field enable_thinking")
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    settings = _settings().model_copy(
        update={
            "llm_configurator_thinking_enabled": True,
            "llm_configurator_thinking_budget_tokens": 1024,
        }
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=settings, http_client=http_client)
        result = client.generate_json("system", "user")

    first_payload = json.loads(requests[0].content)
    second_payload = json.loads(requests[1].content)
    diagnostics = client.safe_diagnostics()

    assert result == {"ok": True}
    assert first_payload["enable_thinking"] is True
    assert first_payload["thinking_budget"] == 1024
    assert "enable_thinking" not in second_payload
    assert "thinking_budget" not in second_payload
    assert diagnostics["llm_thinking_enabled"] is True
    assert diagnostics["llm_thinking_fallback_reason"] == "thinking_params_rejected"
    assert SECRET not in json.dumps(diagnostics)


def _generate_from_content(content: str) -> dict[str, object]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAICompatibleLlmClient(settings=_settings(), http_client=http_client)
        return client.generate_json("system", "user")
