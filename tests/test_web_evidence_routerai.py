from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import httpx

from app.core.config import LlmSettings, WebEvidenceSettings
from app.evidence.web_evidence import (
    EvidenceSearchCache,
    EvidenceSearchTask,
    RouterAIWebEvidenceProvider,
    build_evidence_tasks_for_proposals,
    build_relation_evidence_task,
    build_web_search_provider,
    collect_web_evidence,
)


def test_routerai_provider_uses_llm_settings_fallback() -> None:
    provider = build_web_search_provider(
        WebEvidenceSettings(
            web_evidence_enabled=True,
            web_evidence_provider="routerai",
            web_evidence_model="deepseek/deepseek-v4-pro:online",
        ),
        llm_settings=LlmSettings(
            llm_base_url="https://routerai.example.test/api/v1",
            llm_api_key="llm-secret",
            llm_model="deepseek/deepseek-v4-pro",
        ),
    )

    assert isinstance(provider, RouterAIWebEvidenceProvider)
    assert provider.base_url == "https://routerai.example.test/api/v1"
    assert provider.model == "deepseek/deepseek-v4-pro:online"


def test_routerai_provider_builds_chat_completion_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        captured["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_routerai_response(_routerai_component_json()))

    provider = RouterAIWebEvidenceProvider(
        base_url="https://routerai.example.test/api/v1",
        api_key="router-secret",
        model="deepseek/deepseek-v4-pro:online",
        max_output_tokens=2048,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    pack = collect_web_evidence(
        tasks=[_task()],
        settings=_routerai_settings(max_output_tokens=2048),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert captured["url"] == "https://routerai.example.test/api/v1/chat/completions"
    assert captured["authorization"] == "Bearer router-secret"
    assert payload["model"] == "deepseek/deepseek-v4-pro:online"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 2048
    assert pack.completed_tasks == 1


def test_routerai_provider_parses_strict_json_content() -> None:
    provider = _routerai_provider_with_response(_routerai_component_json())

    pack = collect_web_evidence(
        tasks=[_task()],
        settings=_routerai_settings(),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    assert pack.components[0].evidence_status == "found"
    assert pack.components[0].confidence == "high"
    assert pack.components[0].facts["socket_family"] == "LGA4189"
    assert pack.components[0].sources[0].domain == "i.dell.com"


def test_routerai_provider_strips_markdown_code_fences() -> None:
    content = "```json\n" + json.dumps(_routerai_component_json()) + "\n```"
    provider = _routerai_provider_with_response(content)

    pack = collect_web_evidence(
        tasks=[_task()],
        settings=_routerai_settings(),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    assert pack.completed_tasks == 1
    assert pack.components[0].facts["memory_type"] == "DDR4"


def test_routerai_provider_invalid_json_returns_error_pack() -> None:
    provider = _routerai_provider_with_response("not json")

    pack = collect_web_evidence(
        tasks=[_task()],
        settings=_routerai_settings(),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    assert pack.error_count == 1
    assert pack.components[0].evidence_status == "error"
    assert pack.diagnostics["evidence_error_count"] == 1
    assert pack.diagnostics["evidence_error_type"] == "RuntimeError"
    assert pack.diagnostics["evidence_raw_response_parse_status"] == "parse_error"
    assert "router-secret" not in json.dumps(pack.model_dump(), ensure_ascii=False)


def test_routerai_provider_no_sources_is_not_found_without_facts() -> None:
    content = _routerai_component_json()
    content["components"][0]["sources"] = []
    content["components"][0]["facts"] = {"memory_type": "DDR4"}
    provider = _routerai_provider_with_response(content)

    pack = collect_web_evidence(
        tasks=[_task()],
        settings=_routerai_settings(),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    assert pack.components[0].evidence_status == "not_found"
    assert pack.components[0].confidence == "unknown"
    assert pack.components[0].facts == {}
    assert pack.completed_tasks == 0


def test_evidence_task_generation_adds_relation_tasks() -> None:
    rows = {
        "platform-1": _component_row("platform-1", "platform", "ASUS RS720-E11-RS24U"),
        "cpu-1": _component_row("cpu-1", "cpu", "Intel Xeon Gold 6530"),
        "ram-1": _component_row("ram-1", "ram", "Micron 32GB DDR5 RDIMM"),
        "ssd-1": _component_row("ssd-1", "ssd", "KIOXIA CD8-R U.3 NVMe"),
    }

    tasks = build_evidence_tasks_for_proposals(
        [
            {
                "recommendation_id": "rec-1",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "platform": "platform-1",
                    "cpu": "cpu-1",
                    "ram": "ram-1",
                    "storage": "ssd-1",
                },
            }
        ],
        component_rows_by_id=rows,
        max_queries=16,
        normalized_requirements={"ram_min_gb": 512},
    )

    target_types = {task.target_type for task in tasks}
    assert "component_platform" in target_types
    assert "component_cpu" in target_types
    assert "component_ram" in target_types
    assert "component_storage" in target_types
    assert "relation_platform_cpu" in target_types
    assert "relation_platform_ram" in target_types
    assert "relation_platform_storage" in target_types
    assert "relation_build_sanity" in target_types


def test_routerai_relation_task_payload_contains_component_names() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_routerai_response(_routerai_relation_json()))

    task = build_relation_evidence_task(
        relation_type="platform_cpu",
        recommendation_id="rec-1",
        platform={"component_candidate_id": "platform-1", "name": "ASUS RS720-E11-RS24U"},
        cpu={"component_candidate_id": "cpu-1", "name": "Intel Xeon Gold 6530"},
    )
    provider = RouterAIWebEvidenceProvider(
        base_url="https://routerai.example.test/api/v1",
        api_key="router-secret",
        model="deepseek/deepseek-v4-pro:online",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    pack = collect_web_evidence(
        tasks=[task],
        settings=_routerai_settings(),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    payload = captured["json"]
    assert isinstance(payload, dict)
    messages = payload["messages"]
    assert "compatibility question" in messages[0]["content"]
    user_payload = json.loads(messages[1]["content"])
    relation_task = user_payload["relation_tasks"][0]
    assert relation_task["components"]["platform"]["name"] == "ASUS RS720-E11-RS24U"
    assert relation_task["components"]["cpu"]["name"] == "Intel Xeon Gold 6530"
    assert pack.relation_evidence[0].status == "partially_confirmed"


def test_routerai_relation_no_sources_support_missing_maps_to_not_confirmed() -> None:
    task = build_relation_evidence_task(
        relation_type="platform_cpu",
        recommendation_id="rec-1",
        platform={"component_candidate_id": "platform-1", "name": "ASUS RS720-E11-RS24U"},
        cpu={"component_candidate_id": "cpu-1", "name": "Intel Xeon Gold 6530"},
    )
    provider = _routerai_provider_with_response(
        {
            "components": [],
            "relation_evidence": [
                {
                    "relation_type": "platform_cpu",
                    "recommendation_id": "rec-1",
                    "status": "error",
                    "confidence": "unknown",
                    "missing_evidence": [
                        "CPU support list for this platform/CPU pair was not found."
                    ],
                    "engineering_checks": [
                        "Check the platform CPU support list with an engineer."
                    ],
                    "sources": [],
                }
            ],
        }
    )

    pack = collect_web_evidence(
        tasks=[task],
        settings=_routerai_settings(),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    relation = pack.relation_evidence[0]
    assert relation.status == "not_confirmed"
    assert relation.confidence == "unknown"
    assert relation.sources == []
    assert relation.platform_name == "ASUS RS720-E11-RS24U"
    assert relation.cpu_name == "Intel Xeon Gold 6530"
    assert relation.missing_evidence == [
        "CPU support list for this platform/CPU pair was not found."
    ]
    assert relation.engineering_checks == [
        "Check the platform CPU support list with an engineer."
    ]
    assert pack.error_count == 0
    assert pack.diagnostics["evidence_error_count"] == 0
    assert pack.diagnostics["evidence_status_summary"] == {"not_confirmed": 1}
    assert pack.diagnostics["relation_not_confirmed_count"] == 1
    assert pack.diagnostics["relation_partially_confirmed_count"] == 0
    assert pack.diagnostics["relation_mismatch_count"] == 0


def test_routerai_relation_generic_facts_missing_qvl_maps_to_partially_confirmed() -> None:
    task = build_relation_evidence_task(
        relation_type="platform_ram",
        recommendation_id="rec-1",
        platform={"component_candidate_id": "platform-1", "name": "ASUS RS720-E11-RS24U"},
        ram={"component_candidate_id": "ram-1", "name": "Micron 64GB DDR5 RDIMM"},
    )
    provider = _routerai_provider_with_response(
        {
            "components": [],
            "relation_evidence": [
                {
                    "relation_type": "platform_ram",
                    "recommendation_id": "rec-1",
                    "status": "error",
                    "confidence": "unknown",
                    "confirmed_facts": ["Platform uses DDR5 RDIMM memory"],
                    "missing_evidence": ["Exact RAM QVL row was not found."],
                    "engineering_checks": ["Check platform memory QVL with an engineer."],
                    "sources": [
                        {
                            "url": "https://servers.asus.com/products/servers/rs720-e11-rs24u",
                            "title": "ASUS RS720-E11-RS24U specifications",
                            "domain": "servers.asus.com",
                            "source_type": "official_vendor",
                            "trust_score": 0.95,
                        }
                    ],
                }
            ],
        }
    )

    pack = collect_web_evidence(
        tasks=[task],
        settings=_routerai_settings(),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    relation = pack.relation_evidence[0]
    assert relation.status == "partially_confirmed"
    assert relation.confidence == "medium"
    assert len(relation.sources) == 1
    assert relation.platform_name == "ASUS RS720-E11-RS24U"
    assert relation.ram_name == "Micron 64GB DDR5 RDIMM"
    assert relation.confirmed_facts == ["Platform uses DDR5 RDIMM memory"]
    assert relation.missing_evidence == ["Exact RAM QVL row was not found."]
    assert pack.error_count == 0
    assert pack.diagnostics["evidence_error_count"] == 0
    assert pack.diagnostics["evidence_status_summary"] == {"partially_confirmed": 1}
    assert pack.diagnostics["relation_not_confirmed_count"] == 0
    assert pack.diagnostics["relation_partially_confirmed_count"] == 1


def test_routerai_relation_provider_exception_maps_to_error() -> None:
    task = build_relation_evidence_task(
        relation_type="platform_cpu",
        recommendation_id="rec-1",
        platform={"component_candidate_id": "platform-1", "name": "ASUS RS720-E11-RS24U"},
        cpu={"component_candidate_id": "cpu-1", "name": "Intel Xeon Gold 6530"},
    )
    provider = _routerai_provider_with_response("not json")

    pack = collect_web_evidence(
        tasks=[task],
        settings=_routerai_settings(),
        provider=provider,
        cache=EvidenceSearchCache(cache_dir=_cache_dir(), ttl_hours=0),
    )

    relation = pack.relation_evidence[0]
    assert relation.status == "error"
    assert relation.confidence == "unknown"
    assert relation.platform_name == "ASUS RS720-E11-RS24U"
    assert relation.cpu_name == "Intel Xeon Gold 6530"
    assert pack.error_count == 1
    assert pack.diagnostics["evidence_error_count"] == 1
    assert pack.diagnostics["evidence_status_summary"] == {"error": 1}
    assert pack.diagnostics["relation_not_confirmed_count"] == 0
    assert pack.diagnostics["relation_partially_confirmed_count"] == 0
    assert pack.diagnostics["evidence_error_type"] == "RuntimeError"
    assert pack.diagnostics["evidence_raw_response_parse_status"] == "parse_error"


def test_routerai_cache_is_sanitized() -> None:
    cache_dir = _cache_dir()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_routerai_response(_routerai_component_json()))

    provider = RouterAIWebEvidenceProvider(
        base_url="https://routerai.example.test/api/v1",
        api_key="router-secret",
        model="deepseek/deepseek-v4-pro:online",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    cache = EvidenceSearchCache(cache_dir=cache_dir, ttl_hours=168)
    settings = _routerai_settings()

    first = collect_web_evidence(
        tasks=[_task()],
        settings=settings,
        provider=provider,
        cache=cache,
    )
    second = collect_web_evidence(
        tasks=[_task()],
        settings=settings,
        provider=provider,
        cache=cache,
    )

    assert calls == 1
    assert first.completed_tasks == second.completed_tasks == 1
    cache_text = "\n".join(path.read_text(encoding="utf-8") for path in cache_dir.glob("*.json"))
    assert "router-secret" not in cache_text
    assert "authorization" not in cache_text.casefold()
    assert "messages" not in cache_text.casefold()


def test_docker_compose_keeps_web_evidence_keys_out_of_stock_bot() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    stock_bot_section = compose.split("  stock-bot:", maxsplit=1)[1].split(
        "  stock-postgres:",
        maxsplit=1,
    )[0]

    assert "WEB_EVIDENCE_" not in stock_bot_section
    assert "TAVILY_API_KEY" not in stock_bot_section
    assert "LLM_API_KEY" not in stock_bot_section


def _routerai_provider_with_response(content: object) -> RouterAIWebEvidenceProvider:
    return RouterAIWebEvidenceProvider(
        base_url="https://routerai.example.test/api/v1",
        api_key="router-secret",
        model="deepseek/deepseek-v4-pro:online",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=_routerai_response(content))
            )
        ),
    )


def _routerai_response(content: object) -> dict[str, object]:
    return {"choices": [{"message": {"content": content}}]}


def _routerai_component_json() -> dict[str, object]:
    return {
        "components": [
            {
                "component_candidate_id": "platform-1",
                "role": "platform",
                "evidence_status": "found",
                "confidence": "high",
                "facts": {
                    "vendor": "Dell",
                    "platform_family": "Dell PowerEdge R750xs",
                    "supported_cpu_generation": "3rd Gen Intel Xeon Scalable",
                    "socket_family": "LGA4189",
                    "memory_type": "DDR4",
                },
                "sources": [
                    {
                        "url": "https://i.dell.com/r750xs.pdf",
                        "title": "Dell PowerEdge R750xs Spec Sheet",
                        "source_type": "official_vendor",
                        "trust_score": 0.95,
                    }
                ],
                "warnings": [],
            }
        ],
        "general_notes": [],
    }


def _routerai_relation_json() -> dict[str, object]:
    return {
        "components": [],
        "relation_evidence": [
            {
                "relation_type": "platform_cpu",
                "recommendation_id": "rec-1",
                "components": {
                    "platform": {
                        "component_id": "platform-1",
                        "name": "ASUS RS720-E11-RS24U",
                    },
                    "cpu": {"component_id": "cpu-1", "name": "Intel Xeon Gold 6530"},
                },
                "status": "partially_confirmed",
                "confidence": "medium",
                "confirmed_facts": ["LGA4677", "4th Gen Xeon"],
                "missing_evidence": ["CPU support list not found"],
                "engineering_checks": ["Check CPU support list"],
                "sources": [
                    {
                        "url": "https://servers.asus.com/products/servers/rs720-e11-rs24u",
                        "title": "ASUS RS720-E11-RS24U",
                        "domain": "servers.asus.com",
                        "source_type": "official_vendor",
                        "trust_score": 0.95,
                    }
                ],
            }
        ],
    }


def _component_row(component_id: str, role: str, name: str) -> dict[str, object]:
    return {
        "component_candidate_id": component_id,
        "role": role,
        "producer": name.split(" ", maxsplit=1)[0],
        "part_number": name.split(" ", maxsplit=1)[-1],
        "name": name,
    }


def _task() -> EvidenceSearchTask:
    return EvidenceSearchTask(
        task_id="task-1",
        target_type="platform",
        component_candidate_id="platform-1",
        role="server_platform",
        producer="Dell",
        part_number="R750XS",
        name="Dell PowerEdge R750xs",
        queries=["Dell R750XS datasheet"],
        reason="test",
    )


def _routerai_settings(max_output_tokens: int = 4096) -> WebEvidenceSettings:
    return WebEvidenceSettings(
        web_evidence_enabled=True,
        web_evidence_provider="routerai",
        web_evidence_model="deepseek/deepseek-v4-pro:online",
        web_evidence_max_output_tokens=max_output_tokens,
        web_evidence_cache_ttl_hours=0,
    )


def _cache_dir() -> Path:
    return Path(".tmp_pytest") / "routerai_evidence_cache" / uuid4().hex
