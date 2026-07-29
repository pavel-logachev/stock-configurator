from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx
import pytest

import app.cli.smoke_routerai_web_evidence as smoke_cli
from app.core.config import get_llm_settings, get_web_evidence_settings
from app.evidence.web_evidence import RouterAIWebEvidenceProvider


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None]:
    get_llm_settings.cache_clear()
    get_web_evidence_settings.cache_clear()
    yield
    get_llm_settings.cache_clear()
    get_web_evidence_settings.cache_clear()


def test_smoke_routerai_cli_fake_provider_success_with_sources(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_routerai_response(_routerai_component_json()))

    _set_routerai_env(monkeypatch)
    provider = _provider(handler)

    exit_code = smoke_cli.run(
        [
            "--query",
            "ASUS RS720-E11-RS24U datasheet LGA4677 DDR5 NVMe",
            "--component-id",
            "test-asus",
            "--role",
            "platform",
        ],
        provider=provider,
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert calls == 1
    assert summary["provider"] == "routerai"
    assert summary["model"] == "deepseek/deepseek-v4-pro:online"
    assert summary["http_status"] == 200
    assert summary["parse_status"] == "parsed"
    assert summary["evidence_status"] == "found"
    assert summary["sources_count"] > 0
    assert summary["source_domains"] == ["servers.asus.com"]
    assert summary["extracted_facts"]["socket_family"] == "LGA4677"


def test_smoke_routerai_cli_fake_provider_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_routerai_response("not json router-secret"),
        )

    _set_routerai_env(monkeypatch)
    provider = _provider(handler)

    exit_code = smoke_cli.run(
        [
            "--query",
            "ASUS RS720-E11-RS24U datasheet LGA4677 DDR5 NVMe",
            "--component-id",
            "test-asus",
            "--role",
            "platform",
        ],
        provider=provider,
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 1
    assert summary["http_status"] == 200
    assert summary["parse_status"] == "parse_error"
    assert summary["evidence_status"] == "error"
    assert summary["error_type"] == "RuntimeError"
    assert "not json" in summary["error_preview"]
    assert "router-secret" not in captured.out


def test_smoke_routerai_cli_no_network_does_not_call_provider(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_routerai_env(monkeypatch)

    exit_code = smoke_cli.run(
        [
            "--query",
            "ASUS RS720-E11-RS24U datasheet LGA4677 DDR5 NVMe",
            "--component-id",
            "test-asus",
            "--role",
            "platform",
            "--no-network",
        ],
        provider=_ExplodingProvider(),
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["no_network"] is True
    assert summary["http_status"] is None
    assert summary["parse_status"] == "not_requested"
    assert summary["sources_count"] == 0


def test_smoke_routerai_cli_relation_no_network(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_routerai_env(monkeypatch)

    exit_code = smoke_cli.run(
        [
            "--relation",
            "platform_cpu",
            "--platform-name",
            "ASUS RS720-E11-RS24U",
            "--cpu-name",
            "Intel Xeon Gold 6530",
            "--no-network",
        ],
        provider=_ExplodingProvider(),
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["relation_type"] == "platform_cpu"
    assert summary["status"] == "not_requested"
    assert "ASUS RS720-E11-RS24U" in summary["question"]
    assert "Intel Xeon Gold 6530" in summary["question"]
    assert "router-secret" not in captured.out


def test_smoke_routerai_cli_redacts_secrets_from_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "super-secret-key"

    def handler(request: httpx.Request) -> httpx.Response:
        body = _routerai_component_json(notes=f"contains {secret}")
        return httpx.Response(200, json=_routerai_response(body))

    _set_routerai_env(monkeypatch, api_key=secret)
    provider = _provider(handler, api_key=secret)

    exit_code = smoke_cli.run(
        [
            "--query",
            "ASUS RS720-E11-RS24U datasheet LGA4677 DDR5 NVMe",
            "--component-id",
            "test-asus",
            "--role",
            "platform",
        ],
        provider=provider,
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert secret not in captured.out
    assert summary["extracted_facts"]["notes"] == "contains [redacted]"


class _ExplodingProvider:
    provider_name = "routerai"

    def collect_evidence(self, **kwargs: Any) -> None:
        raise AssertionError("network should not be called in --no-network mode")

    def search(self, **kwargs: Any) -> None:
        raise AssertionError("network should not be called in --no-network mode")


def _set_routerai_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str = "router-secret",
) -> None:
    monkeypatch.setenv("WEB_EVIDENCE_PROVIDER", "routerai")
    monkeypatch.setenv("WEB_EVIDENCE_MODEL", "deepseek/deepseek-v4-pro:online")
    monkeypatch.setenv("LLM_BASE_URL", "https://routerai.example.test/api/v1")
    monkeypatch.setenv("LLM_API_KEY", api_key)
    monkeypatch.delenv("WEB_EVIDENCE_API_KEY", raising=False)
    get_llm_settings.cache_clear()
    get_web_evidence_settings.cache_clear()


def _provider(
    handler: Any,
    *,
    api_key: str = "router-secret",
) -> RouterAIWebEvidenceProvider:
    return RouterAIWebEvidenceProvider(
        base_url="https://routerai.example.test/api/v1",
        api_key=api_key,
        model="deepseek/deepseek-v4-pro:online",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _routerai_response(content: object) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _routerai_component_json(*, notes: str = "official datasheet") -> dict[str, object]:
    return {
        "components": [
            {
                "component_candidate_id": "test-asus",
                "role": "platform",
                "evidence_status": "found",
                "confidence": "high",
                "facts": {
                    "vendor": "ASUS",
                    "platform_family": "ASUS RS720-E11-RS24U",
                    "supported_cpu_generation": "4th/5th Gen Intel Xeon Scalable",
                    "socket_family": "LGA4677",
                    "memory_type": "DDR5",
                    "nvme_support": True,
                    "notes": notes,
                },
                "sources": [
                    {
                        "url": "https://servers.asus.com/products/servers/rs720-e11-rs24u",
                        "title": "ASUS RS720-E11-RS24U",
                        "source_type": "official_vendor",
                        "trust_score": 0.95,
                    }
                ],
                "warnings": [],
            }
        ],
        "general_notes": [],
    }
