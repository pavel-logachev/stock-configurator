from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

UNKNOWN_FACT = "unknown"
DEFAULT_CACHE_DIR = Path("data/evidence_cache")
LEGACY_COMPONENT_TARGET_TYPES = {"platform", "cpu", "ram", "storage", "ready_server"}
COMPONENT_TARGET_TYPES = {
    "component_platform",
    "component_cpu",
    "component_ram",
    "component_storage",
}
RELATION_TARGET_TYPES = {
    "relation_platform_cpu",
    "relation_platform_ram",
    "relation_platform_storage",
    "relation_build_sanity",
}
RELATION_TYPES = {
    "platform_cpu",
    "platform_ram",
    "platform_storage",
    "build_sanity",
}
RELATION_TYPE_TO_TARGET_TYPE = {
    "platform_cpu": "relation_platform_cpu",
    "platform_ram": "relation_platform_ram",
    "platform_storage": "relation_platform_storage",
    "build_sanity": "relation_build_sanity",
}
TARGET_TYPE_TO_RELATION_TYPE = {
    target_type: relation_type
    for relation_type, target_type in RELATION_TYPE_TO_TARGET_TYPE.items()
}
TASK_TARGET_TYPES = (
    COMPONENT_TARGET_TYPES | RELATION_TARGET_TYPES | LEGACY_COMPONENT_TARGET_TYPES
)
ROLE_TO_TARGET_TYPE = {
    "server_platform": "component_platform",
    "platform": "component_platform",
    "cpu": "component_cpu",
    "ram": "component_ram",
    "ssd": "component_storage",
    "hdd": "component_storage",
    "storage": "component_storage",
    "ready_server": "ready_server",
}
COMPONENT_MATRIX_KEYS = [
    ("ready_server_candidates", "ready_server"),
    ("platform_candidates", "component_platform"),
    ("cpu_candidates", "component_cpu"),
    ("ram_candidates", "component_ram"),
    ("ssd_candidates", "component_storage"),
    ("hdd_candidates", "component_storage"),
]

EvidenceTargetType = Literal[
    "platform",
    "cpu",
    "ram",
    "storage",
    "ready_server",
    "component_platform",
    "component_cpu",
    "component_ram",
    "component_storage",
    "relation_platform_cpu",
    "relation_platform_ram",
    "relation_platform_storage",
    "relation_build_sanity",
]


class EvidenceSearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    url: str = ""
    snippet: str = ""
    domain: str = ""


class EvidenceSearchTask(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    target_type: EvidenceTargetType
    component_candidate_id: str = ""
    role: str = ""
    producer: str = ""
    part_number: str = ""
    name: str = ""
    queries: list[str] = Field(default_factory=list)
    reason: str = ""
    recommendation_id: str = ""
    platform_component_id: str = ""
    platform_name: str = ""
    platform_part_number: str = ""
    cpu_component_id: str = ""
    cpu_name: str = ""
    cpu_part_number: str = ""
    ram_component_id: str = ""
    ram_name: str = ""
    ram_part_number: str = ""
    storage_component_id: str = ""
    storage_name: str = ""
    storage_part_number: str = ""
    normalized_requirements: dict[str, Any] = Field(default_factory=dict)
    question: str = ""


class EvidenceSource(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = ""
    title: str = ""
    snippet: str = ""
    domain: str = ""
    source_type: Literal[
        "official_vendor",
        "cpu_vendor",
        "memory_vendor",
        "storage_vendor",
        "distributor",
        "unknown",
    ] = "unknown"
    trust_score: float = 0.0
    retrieved_at: str


class EvidenceFact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    value: str | int | float | bool
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    source_domains: list[str] = Field(default_factory=list)


class ComponentEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    component_candidate_id: str
    role: str = ""
    part_number: str = ""
    name: str = ""
    evidence_status: Literal["found", "not_found", "disabled", "error"] = "not_found"
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    facts: dict[str, Any] = Field(default_factory=dict)
    fact_list: list[EvidenceFact] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RelationEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relation_type: Literal[
        "platform_cpu",
        "platform_ram",
        "platform_storage",
        "build_sanity",
    ]
    recommendation_id: str = ""
    components: dict[str, Any] = Field(default_factory=dict)
    platform_name: str = ""
    cpu_name: str = ""
    ram_name: str = ""
    storage_name: str = ""
    status: Literal[
        "confirmed",
        "partially_confirmed",
        "not_confirmed",
        "mismatch",
        "error",
    ] = "not_confirmed"
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    confirmed_facts: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    mismatch_facts: list[str] = Field(default_factory=list)
    engineering_checks: list[str] = Field(default_factory=list)
    sources: list[EvidenceSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool
    provider: str
    total_tasks: int = 0
    completed_tasks: int = 0
    error_count: int = 0
    components: list[ComponentEvidence] = Field(default_factory=list)
    relation_evidence: list[RelationEvidence] = Field(default_factory=list)
    evidence_summary: str = ""
    search_tasks: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


ROUTERAI_EVIDENCE_SYSTEM_PROMPT = """
You are an evidence research assistant for checking server component compatibility.

Use web search / online sources. Do not merely find a datasheet: answer the concrete
compatibility question for the given component pair or build. Prefer official
vendor, datasheet, support-list and QVL sources from Dell, HPE, Lenovo, ASUS,
Supermicro, Intel, AMD, KIOXIA, Samsung, Micron, and Gooxi. Return strict JSON only.
Do not invent facts, part numbers, sockets, CPU generations, support lists, QVL rows,
or URLs. If sources are not found, return not_found/not_confirmed with confidence
"unknown" and empty sources. If only general specs are found but no support list or
QVL is found, use status="partially_confirmed" for relation evidence and include a
missing_evidence item. Do not claim full compatibility without an official support
list, QVL, datasheet, or support page for the exact relationship. If a source is not
official, lower confidence.

Expected JSON:
{
  "components": [
    {
      "component_candidate_id": "...",
      "role": "platform|cpu|ram|storage|ready_server",
      "evidence_status": "found|not_found|error",
      "confidence": "high|medium|low|unknown",
      "facts": {
        "vendor": "...",
        "platform_family": "...",
        "cpu_generation": "...",
        "socket_family": "...",
        "supported_cpu_generation": "...",
        "memory_type": "...",
        "dimm_slots": "...",
        "drive_bays": "...",
        "nvme_support": "...",
        "form_factor": "...",
        "psu_info": "...",
        "storage_interface": "...",
        "capacity": "...",
        "notes": "..."
      },
      "sources": [
        {
          "url": "...",
          "title": "...",
          "domain": "...",
          "source_type": "official_vendor|cpu_vendor|memory_vendor|storage_vendor|"
                         "distributor|unknown",
          "trust_score": 0.0
        }
      ],
      "warnings": []
    }
  ],
  "relation_evidence": [
    {
      "relation_type": "platform_cpu|platform_ram|platform_storage|build_sanity",
      "recommendation_id": "...",
      "components": {
        "platform": {"component_id": "...", "name": "...", "part_number": "..."},
        "cpu": {"component_id": "...", "name": "...", "part_number": "..."},
        "ram": {"component_id": "...", "name": "...", "part_number": "..."},
        "storage": {"component_id": "...", "name": "...", "part_number": "..."}
      },
      "status": "confirmed|partially_confirmed|not_confirmed|mismatch|error",
      "confidence": "high|medium|low|unknown",
      "confirmed_facts": [],
      "missing_evidence": [],
      "mismatch_facts": [],
      "engineering_checks": [],
      "sources": []
    }
  ],
  "general_notes": []
}
""".strip()


class WebSearchProvider(Protocol):
    provider_name: str

    def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout: float,
    ) -> list[EvidenceSearchResult]:
        """Return external search results for one sanitized query."""


class DisabledWebSearchProvider:
    provider_name = "disabled"

    def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout: float,
    ) -> list[EvidenceSearchResult]:
        return []


class FakeWebSearchProvider:
    provider_name = "fake"

    def __init__(
        self,
        results_by_query: Mapping[str, Sequence[Mapping[str, Any] | EvidenceSearchResult]]
        | None = None,
    ) -> None:
        self._results_by_query = {
            str(query).casefold(): list(results)
            for query, results in (results_by_query or {}).items()
        }
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout: float,
    ) -> list[EvidenceSearchResult]:
        self.queries.append(query)
        query_key = query.casefold()
        matched: list[Mapping[str, Any] | EvidenceSearchResult] = []
        for key, results in self._results_by_query.items():
            if key in query_key or query_key in key:
                matched.extend(results)
        return [_coerce_search_result(result) for result in matched[:max_results]]


class TavilySearchProvider:
    provider_name = "tavily"
    SEARCH_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._http_client = http_client or httpx.Client()
        self._owns_http_client = http_client is None

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout: float,
    ) -> list[EvidenceSearchResult]:
        if not self._api_key:
            raise RuntimeError("Tavily API key is not configured.")
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            response = self._http_client.post(
                self.SEARCH_URL,
                json=payload,
                timeout=httpx.Timeout(timeout),
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(self._sanitize(f"Tavily search failed: {exc}")) from exc
        results = body.get("results") if isinstance(body, Mapping) else None
        if not isinstance(results, list):
            return []
        rows: list[EvidenceSearchResult] = []
        for result in results[:max_results]:
            if not isinstance(result, Mapping):
                continue
            url = str(result.get("url") or "")
            rows.append(
                EvidenceSearchResult(
                    title=str(result.get("title") or ""),
                    url=url,
                    snippet=str(result.get("content") or result.get("snippet") or ""),
                    domain=_domain_from_url(url),
                )
            )
        return rows

    def _sanitize(self, value: str) -> str:
        sanitized = value.replace(self._api_key, "[redacted]")
        encoded_api_key = quote(self._api_key, safe="")
        if encoded_api_key != self._api_key:
            sanitized = sanitized.replace(encoded_api_key, "[redacted]")
        return sanitized


class RouterAIWebEvidenceProvider:
    provider_name = "routerai"
    CHAT_COMPLETIONS_PATH = "/chat/completions"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_output_tokens: int = 4096,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._api_key = api_key.strip()
        self._model = model.strip()
        self._max_output_tokens = max(1, int(max_output_tokens or 4096))
        self._http_client = http_client or httpx.Client()
        self._owns_http_client = http_client is None

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout: float,
    ) -> list[EvidenceSearchResult]:
        return []

    def collect_evidence(
        self,
        *,
        tasks: Sequence[EvidenceSearchTask],
        settings: Any,
        cache: EvidenceSearchCache | None = None,
        normalized_requirements: Any = None,
    ) -> EvidencePack:
        search_tasks = [_task_summary(task) for task in tasks]
        if not tasks:
            return EvidencePack(
                enabled=True,
                provider=self.provider_name,
                evidence_summary="no evidence tasks generated",
                search_tasks=search_tasks,
                diagnostics=safe_evidence_diagnostics(
                    {
                        "enabled": True,
                        "provider": self.provider_name,
                        "total_tasks": 0,
                        "completed_tasks": 0,
                        "error_count": 0,
                        "components": [],
                    },
                    model=self._model,
                    raw_response_parse_status="not_requested",
                ),
            )
        if not self._base_url or not self._api_key or not self._model:
            return EvidencePack(
                enabled=True,
                provider=self.provider_name,
                total_tasks=len(tasks),
                error_count=1,
                components=[
                    _error_component_evidence(
                        task,
                        message="RouterAI evidence provider is not configured.",
                    )
                    for task in tasks
                    if not _is_relation_target(task.target_type)
                ],
                relation_evidence=[
                    _error_relation_evidence(
                        task,
                        message="RouterAI evidence provider is not configured.",
                    )
                    for task in tasks
                    if _is_relation_target(task.target_type)
                ],
                evidence_summary="web evidence unavailable; RouterAI provider is not configured",
                search_tasks=search_tasks,
                diagnostics=safe_evidence_diagnostics(
                    {
                        "enabled": True,
                        "provider": self.provider_name,
                        "total_tasks": len(tasks),
                        "completed_tasks": 0,
                        "error_count": 1,
                        "components": [],
                    },
                    model=self._model,
                    error_type="configuration_error",
                    raw_response_parse_status="not_requested",
                ),
            )

        task_payload = _routerai_task_payload(
            tasks=tasks,
            normalized_requirements=normalized_requirements,
        )
        cache_key = _routerai_cache_key(model=self._model, task_payload=task_payload)
        evidence_cache = cache or EvidenceSearchCache(
            ttl_hours=int(getattr(settings, "web_evidence_cache_ttl_hours", 168))
        )
        cached = evidence_cache.get_pack(provider=self.provider_name, cache_key=cache_key)
        if cached is not None:
            cached.diagnostics = safe_evidence_diagnostics(
                cached,
                model=self._model,
                raw_response_parse_status=str(
                    cached.diagnostics.get("evidence_raw_response_parse_status")
                    or "cache_hit"
                ),
            )
            return cached

        timeout = float(getattr(settings, "web_evidence_timeout_seconds", 120) or 120)
        parse_status = "not_started"
        error_type = ""
        http_status: int | None = None
        error_preview = ""
        try:
            response = self._http_client.post(
                self._completion_url(),
                headers=self._headers(),
                json=self._chat_payload(task_payload),
                timeout=httpx.Timeout(timeout),
            )
            http_status = response.status_code
            if response.status_code >= 400:
                error_preview = self._preview(response.text)
                raise RuntimeError(self._error_message(response))
            try:
                body = response.json()
            except ValueError:
                error_preview = self._preview(response.text)
                raise
            parse_status = "body_json_parsed"
            error_preview = self._response_content_preview(body)
            evidence_json = _decode_routerai_evidence_json(body)
            parse_status = "parsed"
            pack = _routerai_evidence_pack_from_json(
                evidence_json,
                response_body=body,
                tasks=tasks,
                settings=settings,
                provider_name=self.provider_name,
                search_tasks=search_tasks,
            )
        except (httpx.HTTPError, ValueError, TypeError, RuntimeError) as exc:
            error_type = type(exc).__name__
            if not error_preview:
                error_preview = self._preview(str(exc))
            if parse_status in {"not_started", "body_json_parsed"}:
                parse_status = (
                    "parse_error"
                    if isinstance(exc, (ValueError, RuntimeError))
                    else "provider_error"
                )
            message = self._sanitize(f"RouterAI evidence request failed: {exc}")
            pack = EvidencePack(
                enabled=True,
                provider=self.provider_name,
                total_tasks=len(tasks),
                error_count=1,
                components=[
                    _error_component_evidence(task, message=message)
                    for task in tasks
                    if not _is_relation_target(task.target_type)
                ],
                relation_evidence=[
                    _error_relation_evidence(task, message=message)
                    for task in tasks
                    if _is_relation_target(task.target_type)
                ],
                evidence_summary="web evidence unavailable; Composer fallback used",
                search_tasks=search_tasks,
            )
        diagnostics = safe_evidence_diagnostics(
            pack,
            model=self._model,
            error_type=error_type,
            raw_response_parse_status=parse_status,
        )
        if http_status is not None:
            diagnostics["evidence_http_status"] = http_status
        if error_type and error_preview:
            diagnostics["evidence_error_preview"] = error_preview
        pack.diagnostics = diagnostics
        evidence_cache.set_pack(provider=self.provider_name, cache_key=cache_key, pack=pack)
        return pack

    def _chat_payload(self, task_payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": ROUTERAI_EVIDENCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        task_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self._max_output_tokens,
            "response_format": {"type": "json_object"},
        }

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

    def _error_message(self, response: httpx.Response) -> str:
        message = f"RouterAI evidence API returned HTTP {response.status_code}"
        body = response.text.strip()
        if body:
            message = f"{message}: {body[:500]}"
        return self._sanitize(message)

    def _response_content_preview(self, body: Any) -> str:
        try:
            content = _routerai_message_content(body) if isinstance(body, Mapping) else body
        except RuntimeError:
            content = body
        return self._preview(content)

    def _preview(self, value: Any, *, limit: int = 300) -> str:
        if isinstance(value, (Mapping, list)):
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            text = str(value or "")
        text = re.sub(r"\s+", " ", text).strip()
        return self._sanitize(text)[:limit]

    def _sanitize(self, value: str) -> str:
        if not self._api_key:
            return value
        sanitized = value.replace(self._api_key, "[redacted]")
        encoded_api_key = quote(self._api_key, safe="")
        if encoded_api_key != self._api_key:
            sanitized = sanitized.replace(encoded_api_key, "[redacted]")
        return sanitized


class EvidenceSearchCache:
    def __init__(self, *, cache_dir: Path = DEFAULT_CACHE_DIR, ttl_hours: int = 168) -> None:
        self._cache_dir = cache_dir
        self._ttl = timedelta(hours=max(0, ttl_hours))

    def get(self, *, provider: str, query: str) -> list[EvidenceSearchResult] | None:
        if self._ttl.total_seconds() <= 0:
            return None
        path = self._path(provider=provider, query=query)
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            timestamp = datetime.fromisoformat(str(body.get("timestamp") or ""))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            if datetime.now(UTC) - timestamp > self._ttl:
                return None
            results = body.get("results")
            if not isinstance(results, list):
                return None
            return [_coerce_search_result(result) for result in results]
        except (OSError, ValueError, TypeError):
            return None

    def set(
        self,
        *,
        provider: str,
        query: str,
        results: Sequence[EvidenceSearchResult],
    ) -> None:
        path = self._path(provider=provider, query=query)
        body = {
            "provider": _safe_cache_name(provider),
            "query": query,
            "timestamp": datetime.now(UTC).isoformat(),
            "results": [result.model_dump() for result in results],
        }
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("Could not write evidence search cache.", exc_info=True)

    def get_pack(self, *, provider: str, cache_key: str) -> EvidencePack | None:
        if self._ttl.total_seconds() <= 0:
            return None
        path = self._path(provider=provider, query=f"pack:{cache_key}")
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
            timestamp = datetime.fromisoformat(str(body.get("timestamp") or ""))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            if datetime.now(UTC) - timestamp > self._ttl:
                return None
            pack = body.get("evidence_pack")
            if not isinstance(pack, Mapping):
                return None
            return EvidencePack.model_validate(pack)
        except (OSError, ValueError, TypeError):
            return None

    def set_pack(self, *, provider: str, cache_key: str, pack: EvidencePack) -> None:
        path = self._path(provider=provider, query=f"pack:{cache_key}")
        body = {
            "provider": _safe_cache_name(provider),
            "cache_key": cache_key,
            "timestamp": datetime.now(UTC).isoformat(),
            "evidence_pack": pack.model_dump(),
        }
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.debug("Could not write evidence pack cache.", exc_info=True)

    def _path(self, *, provider: str, query: str) -> Path:
        digest = hashlib.sha256(f"{provider}\n{query}".encode()).hexdigest()
        return self._cache_dir / f"{_safe_cache_name(provider)}-{digest}.json"


def build_web_search_provider(
    settings: Any,
    *,
    llm_settings: Any | None = None,
) -> WebSearchProvider:
    provider = str(getattr(settings, "web_evidence_provider", "disabled") or "").strip().lower()
    if provider in {"", "disabled"}:
        return DisabledWebSearchProvider()
    if provider == "tavily":
        api_key = str(getattr(settings, "tavily_api_key", "") or "").strip()
        if not api_key:
            return DisabledWebSearchProvider()
        return TavilySearchProvider(api_key=api_key)
    if provider == "routerai":
        base_url = (
            str(getattr(settings, "web_evidence_base_url", "") or "").strip()
            or str(getattr(settings, "llm_base_url", "") or "").strip()
            or str(getattr(llm_settings, "llm_base_url", "") or "").strip()
        )
        api_key = (
            str(getattr(settings, "web_evidence_api_key", "") or "").strip()
            or str(getattr(settings, "llm_api_key", "") or "").strip()
            or str(getattr(llm_settings, "llm_api_key", "") or "").strip()
        )
        model = str(
            getattr(settings, "web_evidence_model", "deepseek/deepseek-v4-pro:online")
            or ""
        ).strip()
        max_output_tokens = int(getattr(settings, "web_evidence_max_output_tokens", 4096))
        if not base_url or not api_key or not model:
            return DisabledWebSearchProvider()
        return RouterAIWebEvidenceProvider(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_output_tokens=max_output_tokens,
        )
    return DisabledWebSearchProvider()


def build_evidence_tasks_for_proposals(
    recommendations: Sequence[Any],
    *,
    component_rows_by_id: Mapping[str, Mapping[str, Any]],
    stock_rows_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    max_queries: int,
    normalized_requirements: Mapping[str, Any] | None = None,
) -> list[EvidenceSearchTask]:
    rows: list[tuple[str, str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    stock_rows = stock_rows_by_id or {}
    for recommendation in recommendations:
        rec = _as_mapping(recommendation)
        source_type = str(rec.get("source_type") or rec.get("candidate_type") or "").strip()
        source_id = str(rec.get("source_candidate_id") or "").strip()
        if source_type == "ready_server" and source_id and source_id not in seen:
            row = stock_rows.get(source_id)
            if row is not None:
                rows.append(("ready_server", source_id, row))
                seen.add(source_id)
        elif source_id:
            source_row = stock_rows.get(source_id)
            if source_row is not None:
                for component in _mapping_rows(source_row.get("components")):
                    component_id = str(component.get("component_candidate_id") or "").strip()
                    if not component_id or component_id in seen:
                        continue
                    row = component_rows_by_id.get(component_id, component)
                    rows.append((_target_type_for_row(row), component_id, row))
                    seen.add(component_id)
        component_ids = rec.get("component_candidate_ids")
        if isinstance(component_ids, Mapping):
            for prompt_role in ("platform", "cpu", "ram", "storage", "ssd", "hdd"):
                component_id = str(component_ids.get(prompt_role) or "").strip()
                if not component_id or component_id in seen:
                    continue
                row = component_rows_by_id.get(component_id)
                if row is None:
                    continue
                target_type = ROLE_TO_TARGET_TYPE.get(prompt_role, _target_type_for_row(row))
                rows.append((target_type, component_id, row))
                seen.add(component_id)
        for component in _mapping_rows(rec.get("components")):
            component_id = str(component.get("component_candidate_id") or "").strip()
            if not component_id or component_id in seen:
                continue
            target_type = _target_type_for_row(component)
            rows.append((target_type, component_id, component))
            seen.add(component_id)
    component_tasks = _tasks_from_rows(rows, max_queries=max_queries, apply_budget=False)
    relation_tasks = _relation_tasks_for_recommendations(
        recommendations,
        component_rows_by_id=component_rows_by_id,
        normalized_requirements=normalized_requirements or {},
    )
    return _apply_query_budget([*component_tasks, *relation_tasks], max_queries=max_queries)


def build_relation_evidence_tasks_for_recommendations(
    recommendations: Sequence[Any],
    *,
    component_rows_by_id: Mapping[str, Mapping[str, Any]],
    normalized_requirements: Mapping[str, Any] | None = None,
) -> list[EvidenceSearchTask]:
    return _relation_tasks_for_recommendations(
        recommendations,
        component_rows_by_id=component_rows_by_id,
        normalized_requirements=normalized_requirements or {},
    )


def build_evidence_tasks_from_component_matrix(
    component_candidate_matrix: Mapping[str, Any],
    *,
    max_queries: int,
    per_role_limit: int = 1,
) -> list[EvidenceSearchTask]:
    rows: list[tuple[str, str, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for matrix_key, target_type in COMPONENT_MATRIX_KEYS:
        count = 0
        for row in _mapping_rows(component_candidate_matrix.get(matrix_key)):
            component_id = str(
                row.get("component_candidate_id") or row.get("candidate_id") or ""
            ).strip()
            if not component_id or component_id in seen:
                continue
            rows.append((target_type, component_id, row))
            seen.add(component_id)
            count += 1
            if count >= per_role_limit:
                break
    return _tasks_from_rows(rows, max_queries=max_queries)


def collect_web_evidence(
    *,
    tasks: Sequence[EvidenceSearchTask],
    settings: Any,
    provider: WebSearchProvider | None = None,
    cache: EvidenceSearchCache | None = None,
    normalized_requirements: Any = None,
    llm_settings: Any | None = None,
) -> EvidencePack:
    enabled = bool(getattr(settings, "web_evidence_enabled", False))
    provider_name = str(getattr(settings, "web_evidence_provider", "disabled") or "disabled")
    search_tasks = [_task_summary(task) for task in tasks]
    if not enabled:
        pack = EvidencePack(
            enabled=False,
            provider="disabled",
            evidence_summary="web evidence disabled",
            search_tasks=search_tasks,
        )
        pack.diagnostics = safe_evidence_diagnostics(
            pack,
            model=str(getattr(settings, "web_evidence_model", "") or ""),
            raw_response_parse_status="not_requested",
        )
        return pack

    effective_provider = provider or build_web_search_provider(
        settings,
        llm_settings=llm_settings,
    )
    provider_name = getattr(effective_provider, "provider_name", provider_name)
    if isinstance(effective_provider, DisabledWebSearchProvider):
        pack = EvidencePack(
            enabled=True,
            provider=provider_name,
            total_tasks=len(tasks),
            components=[
                _disabled_component_evidence(task, provider_name=provider_name)
                for task in tasks
                if not _is_relation_target(task.target_type)
            ],
            relation_evidence=[
                _disabled_relation_evidence(task, provider_name=provider_name)
                for task in tasks
                if _is_relation_target(task.target_type)
            ],
            evidence_summary="web evidence provider disabled or not configured",
            search_tasks=search_tasks,
        )
        pack.diagnostics = safe_evidence_diagnostics(
            pack,
            model=str(getattr(settings, "web_evidence_model", "") or ""),
            raw_response_parse_status="not_requested",
        )
        return pack
    if hasattr(effective_provider, "collect_evidence"):
        pack = effective_provider.collect_evidence(  # type: ignore[attr-defined]
            tasks=tasks,
            settings=settings,
            cache=cache,
            normalized_requirements=normalized_requirements,
        )
        pack.diagnostics = safe_evidence_diagnostics(
            pack,
            model=str(getattr(settings, "web_evidence_model", "") or ""),
            raw_response_parse_status=str(
                getattr(pack, "diagnostics", {}).get("evidence_raw_response_parse_status")
                or "provider_collect"
            ),
        )
        return pack

    trusted_domains = _trusted_domains(settings)
    max_results = max(1, int(getattr(settings, "web_evidence_max_results_per_query", 5)))
    timeout = float(getattr(settings, "web_evidence_timeout_seconds", 120) or 120)
    max_snippet_chars = max(0, int(getattr(settings, "web_evidence_max_snippet_chars", 1200)))
    evidence_cache = cache or EvidenceSearchCache(
        ttl_hours=int(getattr(settings, "web_evidence_cache_ttl_hours", 168))
    )
    components: list[ComponentEvidence] = []
    relation_evidence: list[RelationEvidence] = []
    error_count = 0
    completed_tasks = 0
    for task in tasks:
        sources: list[EvidenceSource] = []
        warnings: list[str] = []
        task_error = False
        for query in task.queries:
            try:
                results = evidence_cache.get(provider=provider_name, query=query)
                if results is None:
                    results = effective_provider.search(
                        query,
                        max_results=max_results,
                        timeout=timeout,
                    )
                    evidence_cache.set(provider=provider_name, query=query, results=results)
            except Exception as exc:
                task_error = True
                error_count += 1
                warnings.append(_safe_error_message(exc))
                continue
            for result in results[:max_results]:
                source = _source_from_result(
                    result,
                    trusted_domains=trusted_domains,
                    max_snippet_chars=max_snippet_chars,
                )
                if source.url and source.url not in {existing.url for existing in sources}:
                    sources.append(source)
        if sources:
            completed_tasks += 1
        if _is_relation_target(task.target_type):
            relation_evidence.append(
                _relation_evidence_from_sources(
                    task,
                    sources=sources,
                    task_error=task_error,
                    warnings=warnings,
                )
            )
        else:
            components.append(
                _component_evidence_from_sources(
                    task,
                    sources=sources,
                    task_error=task_error,
                    warnings=warnings,
                )
            )

    pack = EvidencePack(
        enabled=True,
        provider=provider_name,
        total_tasks=len(tasks),
        completed_tasks=completed_tasks,
        error_count=error_count,
        components=components,
        relation_evidence=relation_evidence,
        evidence_summary=_evidence_summary(
            components,
            relation_evidence=relation_evidence,
            error_count=error_count,
        ),
        search_tasks=search_tasks,
    )
    pack.diagnostics = safe_evidence_diagnostics(
        pack,
        model=str(getattr(settings, "web_evidence_model", "") or ""),
        raw_response_parse_status="not_applicable",
    )
    return pack


def evidence_pack_has_found_sources(evidence_pack: Mapping[str, Any] | EvidencePack) -> bool:
    pack = _pack_mapping(evidence_pack)
    for component in _mapping_rows(pack.get("components")):
        if component.get("evidence_status") == "found" and _mapping_rows(component.get("sources")):
            return True
    for relation in _mapping_rows(pack.get("relation_evidence")):
        if _mapping_rows(relation.get("sources")):
            return True
    return False


def evidence_components_by_id(
    evidence_pack: Mapping[str, Any] | EvidencePack | None,
) -> dict[str, Mapping[str, Any]]:
    if evidence_pack is None:
        return {}
    pack = _pack_mapping(evidence_pack)
    result: dict[str, Mapping[str, Any]] = {}
    for component in _mapping_rows(pack.get("components")):
        component_id = str(component.get("component_candidate_id") or "").strip()
        if component_id:
            result[component_id] = component
    return result


def evidence_relations_by_recommendation_id(
    evidence_pack: Mapping[str, Any] | EvidencePack | None,
) -> dict[str, list[Mapping[str, Any]]]:
    if evidence_pack is None:
        return {}
    pack = _pack_mapping(evidence_pack)
    result: dict[str, list[Mapping[str, Any]]] = {}
    for relation in _mapping_rows(pack.get("relation_evidence")):
        recommendation_id = str(relation.get("recommendation_id") or "").strip()
        if not recommendation_id:
            continue
        result.setdefault(recommendation_id, []).append(relation)
    return result


def safe_evidence_diagnostics(
    evidence_pack: Mapping[str, Any] | EvidencePack | None,
    *,
    model: str = "",
    error_type: str = "",
    raw_response_parse_status: str = "",
) -> dict[str, Any]:
    pack = _pack_mapping(evidence_pack) if evidence_pack is not None else {}
    existing = pack.get("diagnostics")
    existing_diagnostics = existing if isinstance(existing, Mapping) else {}
    components = _mapping_rows(pack.get("components"))
    relations = _mapping_rows(pack.get("relation_evidence"))
    status_summary: dict[str, int] = {}
    for component in components:
        status = str(component.get("evidence_status") or "unknown").strip() or "unknown"
        status_summary[status] = status_summary.get(status, 0) + 1
    for relation in relations:
        status = str(relation.get("status") or "unknown").strip() or "unknown"
        status_summary[status] = status_summary.get(status, 0) + 1
    if not status_summary and pack.get("total_tasks"):
        status_summary["unknown"] = int(pack.get("total_tasks") or 0)

    provider = str(pack.get("provider") or "").strip()
    model_value = str(
        model
        or existing_diagnostics.get("evidence_model")
        or pack.get("model")
        or ""
    ).strip()
    error_value = str(
        error_type or existing_diagnostics.get("evidence_error_type") or ""
    ).strip()
    parse_status = str(
        raw_response_parse_status
        or existing_diagnostics.get("evidence_raw_response_parse_status")
        or "not_applicable"
    ).strip()
    http_status: int | None = None
    raw_http_status = existing_diagnostics.get("evidence_http_status")
    if raw_http_status not in (None, ""):
        try:
            http_status = int(raw_http_status)
        except (TypeError, ValueError):
            http_status = None
    error_preview = str(existing_diagnostics.get("evidence_error_preview") or "").strip()[:300]

    source_count = 0
    for component in components:
        source_count += len(_mapping_rows(component.get("sources")))
    for relation in relations:
        source_count += len(_mapping_rows(relation.get("sources")))
    if not source_count:
        source_count = int(existing_diagnostics.get("evidence_sources_count") or 0)
    tasks_by_type: dict[str, int] = {}
    for task in _mapping_rows(pack.get("search_tasks")):
        target_type = str(task.get("target_type") or "unknown").strip() or "unknown"
        tasks_by_type[target_type] = tasks_by_type.get(target_type, 0) + 1
    if not tasks_by_type and existing_diagnostics.get("evidence_tasks_count_by_type"):
        raw_tasks_by_type = existing_diagnostics.get("evidence_tasks_count_by_type")
        if isinstance(raw_tasks_by_type, Mapping):
            tasks_by_type = {
                str(key): int(value)
                for key, value in raw_tasks_by_type.items()
                if str(key).strip()
            }
    relation_recommendations = {
        str(relation.get("recommendation_id") or "").strip()
        for relation in relations
        if str(relation.get("recommendation_id") or "").strip()
    }

    return {
        "evidence_tasks_count": int(pack.get("total_tasks") or 0),
        "evidence_tasks_count_by_type": tasks_by_type,
        "evidence_completed_count": int(pack.get("completed_tasks") or 0),
        "evidence_error_count": int(pack.get("error_count") or 0),
        "evidence_provider": provider,
        "evidence_model": model_value,
        "evidence_status_summary": status_summary,
        "relation_evidence_count": len(relations),
        "recommendations_with_relation_evidence": len(relation_recommendations),
        "relation_mismatch_count": sum(
            1 for relation in relations if relation.get("status") == "mismatch"
        ),
        "relation_partially_confirmed_count": sum(
            1 for relation in relations if relation.get("status") == "partially_confirmed"
        ),
        "relation_not_confirmed_count": sum(
            1 for relation in relations if relation.get("status") == "not_confirmed"
        ),
        "evidence_error_type": error_value,
        "evidence_raw_response_parse_status": parse_status,
        "evidence_http_status": http_status,
        "evidence_error_preview": error_preview,
        "evidence_mode": str(existing_diagnostics.get("evidence_mode") or "separate"),
        "online_composer_used": bool(existing_diagnostics.get("online_composer_used")),
        "evidence_used": bool(
            existing_diagnostics.get("evidence_used")
            if "evidence_used" in existing_diagnostics
            else source_count > 0
        ),
        "evidence_sources_count": source_count,
        "online_composer_error_type": str(
            existing_diagnostics.get("online_composer_error_type") or ""
        ),
        "online_composer_parse_status": str(
            existing_diagnostics.get("online_composer_parse_status") or ""
        ),
        "online_composer_empty_response_repair_attempted": bool(
            existing_diagnostics.get(
                "online_composer_empty_response_repair_attempted"
            )
        ),
        "online_composer_empty_response_repair_success": bool(
            existing_diagnostics.get("online_composer_empty_response_repair_success")
        ),
        "structured_no_recommendation_used": bool(
            existing_diagnostics.get("structured_no_recommendation_used")
        ),
        "evidence_requests_count": int(
            existing_diagnostics.get("evidence_requests_count")
            or (1 if pack.get("enabled") else 0)
        ),
    }


def _tasks_from_rows(
    rows: Sequence[tuple[str, str, Mapping[str, Any]]],
    *,
    max_queries: int,
    apply_budget: bool = True,
) -> list[EvidenceSearchTask]:
    tasks: list[EvidenceSearchTask] = []
    for target_type, component_id, row in rows:
        normalized_target = (
            target_type if target_type in TASK_TARGET_TYPES else _target_type_for_row(row)
        )
        queries = _queries_for_row(row, target_type=normalized_target)
        if not queries:
            continue
        producer = str(row.get("producer") or "").strip()
        part_number = str(row.get("part_number") or "").strip()
        name = str(row.get("name") or row.get("item_name") or "").strip()
        tasks.append(
            EvidenceSearchTask(
                task_id=_task_id(component_id, normalized_target),
                target_type=normalized_target,  # type: ignore[arg-type]
                component_candidate_id=component_id,
                role=str(row.get("role") or "").strip(),
                producer=producer,
                part_number=part_number,
                name=name,
                queries=queries,
                reason=f"proposal_pool_{normalized_target}_compatibility_check",
            )
        )
    return _apply_query_budget(tasks, max_queries=max_queries) if apply_budget else tasks


def _apply_query_budget(
    tasks: Sequence[EvidenceSearchTask],
    *,
    max_queries: int,
) -> list[EvidenceSearchTask]:
    remaining = max(0, int(max_queries or 0))
    if remaining <= 0:
        return []
    budgeted: list[EvidenceSearchTask] = []
    original_queries: list[list[str]] = []
    selected_queries: list[list[str]] = []
    for task in tasks:
        queries = _unique_texts(_sanitize_query(query) for query in task.queries if query)
        if not queries or remaining <= 0:
            continue
        original_queries.append(queries)
        selected_queries.append([queries[0]])
        budgeted.append(task.model_copy(update={"queries": [queries[0]]}))
        remaining -= 1
    if remaining <= 0:
        return budgeted
    for index, task in enumerate(budgeted):
        if remaining <= 0:
            break
        for query in original_queries[index][1:3]:
            if remaining <= 0:
                break
            if query in selected_queries[index]:
                continue
            selected_queries[index].append(query)
            remaining -= 1
        budgeted[index] = task.model_copy(update={"queries": selected_queries[index]})
    return budgeted


def _queries_for_row(row: Mapping[str, Any], *, target_type: str) -> list[str]:
    producer = str(row.get("producer") or "").strip()
    part_number = str(row.get("part_number") or "").strip()
    name = str(row.get("name") or row.get("item_name") or "").strip()
    base = " ".join(part for part in [producer, part_number] if part).strip()
    display = name or base
    queries: list[str] = []
    target_kind = _component_target_kind(target_type)
    if target_kind in {"platform", "ready_server"}:
        if base:
            queries.append(f"{base} datasheet")
        if display:
            queries.append(f"{display} specifications CPU support")
            queries.append(f"{display} memory DDR5 NVMe bays")
    elif target_kind == "cpu":
        if base:
            queries.append(f"{base} processor specifications")
        if display:
            queries.append(f"{display} socket cores generation")
        if "intel" in f"{producer} {display}".casefold() or "xeon" in display.casefold():
            queries.append(f"Intel {part_number or display} ARK")
    elif target_kind == "ram":
        if base:
            queries.append(f"{base} DDR5 RDIMM capacity")
        if display:
            queries.append(f"{display} memory module specs")
    elif target_kind == "storage":
        if base:
            queries.append(f"{base} NVMe 3.84TB specs")
        if display:
            queries.append(f"{display} interface form factor")
    return _unique_texts(_sanitize_query(query) for query in queries if query)


def _relation_tasks_for_recommendations(
    recommendations: Sequence[Any],
    *,
    component_rows_by_id: Mapping[str, Mapping[str, Any]],
    normalized_requirements: Mapping[str, Any],
) -> list[EvidenceSearchTask]:
    tasks: list[EvidenceSearchTask] = []
    seen: set[str] = set()
    for recommendation in recommendations:
        rec = _as_mapping(recommendation)
        recommendation_id = str(rec.get("recommendation_id") or "").strip()
        if not recommendation_id:
            continue
        component_ids = rec.get("component_candidate_ids")
        if not isinstance(component_ids, Mapping):
            continue
        platform = _component_ref(component_ids.get("platform"), component_rows_by_id)
        if not platform:
            continue
        cpu = _component_ref(component_ids.get("cpu"), component_rows_by_id)
        ram = _component_ref(component_ids.get("ram"), component_rows_by_id)
        storage = _component_ref(
            component_ids.get("storage") or component_ids.get("ssd") or component_ids.get("hdd"),
            component_rows_by_id,
        )
        relation_inputs = [
            ("platform_cpu", cpu, None, None),
            ("platform_ram", None, ram, None),
            ("platform_storage", None, None, storage),
            ("build_sanity", cpu, ram, storage),
        ]
        for relation_type, cpu_ref, ram_ref, storage_ref in relation_inputs:
            if relation_type != "build_sanity" and not (cpu_ref or ram_ref or storage_ref):
                continue
            if relation_type == "build_sanity" and not (cpu_ref or ram_ref or storage_ref):
                continue
            task = build_relation_evidence_task(
                relation_type=relation_type,
                recommendation_id=recommendation_id,
                platform=platform,
                cpu=cpu_ref,
                ram=ram_ref,
                storage=storage_ref,
                normalized_requirements=normalized_requirements,
            )
            dedupe_key = ":".join(
                [
                    task.recommendation_id,
                    task.target_type,
                    task.platform_component_id,
                    task.cpu_component_id,
                    task.ram_component_id,
                    task.storage_component_id,
                ]
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            tasks.append(task)
    return tasks


def build_relation_evidence_task(
    *,
    relation_type: str,
    recommendation_id: str,
    platform: Mapping[str, Any] | None = None,
    cpu: Mapping[str, Any] | None = None,
    ram: Mapping[str, Any] | None = None,
    storage: Mapping[str, Any] | None = None,
    normalized_requirements: Mapping[str, Any] | None = None,
) -> EvidenceSearchTask:
    normalized_relation_type = (
        relation_type if relation_type in RELATION_TYPES else "build_sanity"
    )
    target_type = RELATION_TYPE_TO_TARGET_TYPE[normalized_relation_type]
    platform_ref = _relation_component_payload(platform)
    cpu_ref = _relation_component_payload(cpu)
    ram_ref = _relation_component_payload(ram)
    storage_ref = _relation_component_payload(storage)
    requirements = dict(normalized_requirements or {})
    question = _relation_question(
        normalized_relation_type,
        platform=platform_ref,
        cpu=cpu_ref,
        ram=ram_ref,
        storage=storage_ref,
        requirements=requirements,
    )
    queries = _queries_for_relation(
        normalized_relation_type,
        platform=platform_ref,
        cpu=cpu_ref,
        ram=ram_ref,
        storage=storage_ref,
        requirements=requirements,
    )
    return EvidenceSearchTask(
        task_id=_relation_task_id(
            recommendation_id=recommendation_id,
            relation_type=normalized_relation_type,
            platform=platform_ref,
            cpu=cpu_ref,
            ram=ram_ref,
            storage=storage_ref,
        ),
        target_type=target_type,  # type: ignore[arg-type]
        component_candidate_id="",
        role=normalized_relation_type,
        queries=queries,
        reason=f"proposal_relation_{normalized_relation_type}_compatibility_check",
        recommendation_id=recommendation_id,
        platform_component_id=platform_ref.get("component_id", ""),
        platform_name=platform_ref.get("name", ""),
        platform_part_number=platform_ref.get("part_number", ""),
        cpu_component_id=cpu_ref.get("component_id", ""),
        cpu_name=cpu_ref.get("name", ""),
        cpu_part_number=cpu_ref.get("part_number", ""),
        ram_component_id=ram_ref.get("component_id", ""),
        ram_name=ram_ref.get("name", ""),
        ram_part_number=ram_ref.get("part_number", ""),
        storage_component_id=storage_ref.get("component_id", ""),
        storage_name=storage_ref.get("name", ""),
        storage_part_number=storage_ref.get("part_number", ""),
        normalized_requirements=requirements,
        question=question,
    )


def _component_ref(
    component_id: Any,
    component_rows_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    clean_id = str(component_id or "").strip()
    if not clean_id:
        return {}
    row = component_rows_by_id.get(clean_id)
    if row is None:
        return {}
    return {
        **dict(row),
        "component_candidate_id": clean_id,
    }


def _relation_component_payload(component: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(component, Mapping):
        return {}
    component_id = str(
        component.get("component_candidate_id") or component.get("candidate_id") or ""
    ).strip()
    return {
        "component_id": component_id,
        "name": str(component.get("name") or component.get("item_name") or "").strip(),
        "part_number": str(component.get("part_number") or "").strip(),
        "producer": str(component.get("producer") or "").strip(),
    }


def _queries_for_relation(
    relation_type: str,
    *,
    platform: Mapping[str, str],
    cpu: Mapping[str, str],
    ram: Mapping[str, str],
    storage: Mapping[str, str],
    requirements: Mapping[str, Any],
) -> list[str]:
    platform_name = _display_component(platform)
    cpu_name = _display_component(cpu)
    ram_part = str(ram.get("part_number") or "").strip()
    storage_name = _display_component(storage)
    storage_part = str(storage.get("part_number") or "").strip()
    queries: list[str] = []
    if relation_type == "platform_cpu":
        cpu_model = _cpu_model_text(cpu_name)
        if platform_name and cpu_name:
            queries.append(f"{platform_name} supported CPU list {cpu_name}")
        if platform.get("part_number") and cpu.get("part_number"):
            queries.append(
                f"{platform.get('part_number')} {cpu.get('part_number')} compatibility"
            )
        if platform_name and cpu_model:
            queries.append(f"{platform_name} {cpu_model} support")
            queries.append(f"{platform_name} CPU support LGA4677 {cpu_model}")
        if _same_vendor(platform, cpu) and platform_name and cpu.get("part_number"):
            queries.append(
                f"{platform.get('producer')} {platform_name} {cpu.get('part_number')} support"
            )
    elif relation_type == "platform_ram":
        if platform_name:
            queries.append(f"{platform_name} DDR5 RDIMM memory support")
        if platform_name and ram_part:
            producer = str(ram.get("producer") or "").strip()
            queries.append(f"{platform_name} {producer} {ram_part} QVL")
        total_ram = _requirement_ram_text(requirements)
        module_ram = _ram_module_text(ram)
        if platform_name:
            queries.append(
                _sanitize_query(
                    f"{platform_name} {module_ram} DDR5 RDIMM support {total_ram}"
                )
            )
    elif relation_type == "platform_storage":
        if platform_name:
            queries.append(f"{platform_name} NVMe U.2 U.3 backplane support")
            queries.append(f"{platform_name} 2.5 U.3 NVMe drive bays")
        if platform_name and (storage_part or storage_name):
            producer = str(storage.get("producer") or "").strip()
            queries.append(
                f"{platform_name} {producer} {storage_part or storage_name} NVMe compatibility"
            )
    else:
        cpu_model = _cpu_model_text(cpu_name)
        total_ram = _requirement_ram_text(requirements)
        storage_label = "NVMe" if _looks_like_nvme(storage_name) else storage_name
        if platform_name:
            queries.append(
                _sanitize_query(
                    f"{platform_name} dual Xeon {cpu_model} {total_ram} "
                    f"DDR5 {storage_label} 2U configuration"
                )
            )
    return _unique_texts(_sanitize_query(query) for query in queries if query)


def _relation_question(
    relation_type: str,
    *,
    platform: Mapping[str, str],
    cpu: Mapping[str, str],
    ram: Mapping[str, str],
    storage: Mapping[str, str],
    requirements: Mapping[str, Any],
) -> str:
    platform_name = _display_component(platform)
    if relation_type == "platform_cpu":
        return f"Is CPU {_display_component(cpu)} officially supported by platform {platform_name}?"
    if relation_type == "platform_ram":
        return (
            f"Does platform {platform_name} support RAM {_display_component(ram)} "
            f"for the requested {_requirement_ram_text(requirements)} configuration?"
        )
    if relation_type == "platform_storage":
        return (
            f"Does platform {platform_name} backplane support storage "
            f"{_display_component(storage)}?"
        )
    return (
        f"Is the build based on {platform_name}, {_display_component(cpu)}, "
        f"{_display_component(ram)}, and {_display_component(storage)} sane enough "
        "for engineer review?"
    )


def _display_component(component: Mapping[str, str]) -> str:
    return " ".join(
        part
        for part in [
            str(component.get("producer") or "").strip(),
            str(component.get("part_number") or "").strip(),
            str(component.get("name") or "").strip(),
        ]
        if part
    ).strip()


def _cpu_model_text(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(
        r"\b(?:Intel\s+)?Xeon\s+(?:Gold|Silver|Platinum|Bronze)?\s*\w+\b",
        text,
        re.IGNORECASE,
    )
    if match is not None:
        return _sanitize_query(match.group(0))
    match = re.search(r"\bEPYC\s+\w+\b", text, re.IGNORECASE)
    if match is not None:
        return _sanitize_query(match.group(0))
    return text


def _requirement_ram_text(requirements: Mapping[str, Any]) -> str:
    for key in ("ram_min_gb", "ram_gb", "memory_gb"):
        value = _safe_int(requirements.get(key))
        if value:
            return f"{value}GB"
    return ""


def _ram_module_text(ram: Mapping[str, str]) -> str:
    text = _display_component(ram)
    match = re.search(r"\b(\d{1,4})\s*GB\b", text, re.IGNORECASE)
    return f"{match.group(1)}GB" if match is not None else ""


def _looks_like_nvme(value: str) -> bool:
    return bool(re.search(r"\bNVMe\b|\bU\.?2\b|\bU\.?3\b", value, re.IGNORECASE))


def _same_vendor(left: Mapping[str, str], right: Mapping[str, str]) -> bool:
    left_vendor = str(left.get("producer") or "").strip().casefold()
    right_vendor = str(right.get("producer") or "").strip().casefold()
    return bool(left_vendor and left_vendor == right_vendor)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _component_evidence_from_sources(
    task: EvidenceSearchTask,
    *,
    sources: list[EvidenceSource],
    task_error: bool,
    warnings: list[str],
) -> ComponentEvidence:
    if not sources:
        status = "error" if task_error else "not_found"
        clean_warnings = warnings or ["No external evidence found."]
        return ComponentEvidence(
            component_candidate_id=task.component_candidate_id,
            role=task.role or task.target_type,
            part_number=task.part_number,
            name=task.name,
            evidence_status=status,
            confidence="unknown",
            warnings=clean_warnings,
        )

    facts = _extract_facts(task, sources)
    confidence = _component_confidence(facts, sources)
    fact_list = [
        EvidenceFact(
            name=key,
            value=value,
            confidence=confidence,
            source_domains=sorted({source.domain for source in sources if source.domain}),
        )
        for key, value in facts.items()
        if value not in (None, "", [], UNKNOWN_FACT)
    ]
    clean_warnings = list(warnings)
    if not facts:
        clean_warnings.append("External sources found, but no compatibility facts were extracted.")
    return ComponentEvidence(
        component_candidate_id=task.component_candidate_id,
        role=task.role or task.target_type,
        part_number=task.part_number,
        name=task.name,
        evidence_status="found",
        confidence=confidence,
        facts=facts,
        fact_list=fact_list,
        sources=sources,
        warnings=_unique_texts(clean_warnings),
    )


def _error_component_evidence(task: EvidenceSearchTask, *, message: str) -> ComponentEvidence:
    return ComponentEvidence(
        component_candidate_id=task.component_candidate_id,
        role=task.role or task.target_type,
        part_number=task.part_number,
        name=task.name,
        evidence_status="error",
        confidence="unknown",
        warnings=[_safe_error_message(RuntimeError(message))],
    )


def _routerai_task_payload(
    *,
    tasks: Sequence[EvidenceSearchTask],
    normalized_requirements: Any,
) -> dict[str, Any]:
    component_tasks = [task for task in tasks if not _is_relation_target(task.target_type)]
    relation_tasks = [task for task in tasks if _is_relation_target(task.target_type)]
    return {
        "normalized_requirements": normalized_requirements if normalized_requirements else {},
        "components": [
            {
                "task_id": task.task_id,
                "component_candidate_id": task.component_candidate_id,
                "role": task.role or task.target_type,
                "target_type": task.target_type,
                "producer": task.producer,
                "part_number": task.part_number,
                "name": task.name,
                "queries": task.queries,
                "reason": task.reason,
            }
            for task in component_tasks
        ],
        "relation_tasks": [
            {
                "task_id": task.task_id,
                "target_type": task.target_type,
                "relation_type": _relation_type_for_task(task),
                "recommendation_id": task.recommendation_id,
                "question": task.question,
                "components": _relation_components_for_task(task),
                "normalized_requirements": task.normalized_requirements,
                "queries": task.queries,
                "reason": task.reason,
            }
            for task in relation_tasks
        ],
        "instructions": [
            "Use online search and source URLs.",
            "Prefer official vendor, datasheet, support list, and QVL sources.",
            "Answer relation_tasks as explicit compatibility questions.",
            (
                "Return not_found with confidence unknown, empty facts, and empty "
                "sources when sources are missing."
            ),
            (
                "For relation_tasks, return partially_confirmed when only general "
                "specs match but the exact support list or QVL is missing."
            ),
            "For found evidence, include source URL, title, and domain.",
            "Do not infer support without a source.",
            "Do not invent sources.",
        ],
    }


def _routerai_cache_key(*, model: str, task_payload: Mapping[str, Any]) -> str:
    body = json.dumps(task_payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(body.encode()).hexdigest()
    return f"{_safe_cache_name(model)}-{digest}"


def _decode_routerai_evidence_json(body: Any) -> Mapping[str, Any]:
    if not isinstance(body, Mapping):
        raise RuntimeError("RouterAI response body must be a JSON object.")
    content = _routerai_message_content(body)
    if isinstance(content, Mapping):
        return content
    if not isinstance(content, str):
        raise RuntimeError("RouterAI response content must be a JSON object or string.")
    text = _strip_json_code_fence(content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("RouterAI evidence content was not valid JSON.") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("RouterAI evidence content was not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise RuntimeError("RouterAI evidence content must be a JSON object.")
    return parsed


def _routerai_message_content(body: Mapping[str, Any]) -> Any:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("RouterAI response does not contain choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise RuntimeError("RouterAI response choice must be a JSON object.")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise RuntimeError("RouterAI response choice does not contain message.")
    return message.get("content")


def _strip_json_code_fence(content: str) -> str:
    text = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    return text


def _routerai_evidence_pack_from_json(
    payload: Mapping[str, Any],
    *,
    response_body: Mapping[str, Any],
    tasks: Sequence[EvidenceSearchTask],
    settings: Any,
    provider_name: str,
    search_tasks: list[dict[str, Any]],
) -> EvidencePack:
    trusted_domains = _trusted_domains(settings)
    max_snippet_chars = max(0, int(getattr(settings, "web_evidence_max_snippet_chars", 1200)))
    metadata_sources = _routerai_metadata_sources(
        response_body,
        trusted_domains=trusted_domains,
        max_snippet_chars=max_snippet_chars,
    )
    components_by_id = {
        str(component.get("component_candidate_id") or "").strip(): component
        for component in _mapping_rows(payload.get("components"))
    }
    relation_by_key = {
        _relation_key_from_mapping(relation): relation
        for relation in _mapping_rows(payload.get("relation_evidence"))
        if _relation_key_from_mapping(relation)
    }
    components: list[ComponentEvidence] = []
    relations: list[RelationEvidence] = []
    for task in tasks:
        if _is_relation_target(task.target_type):
            relation = relation_by_key.get(_relation_key_for_task(task))
            if relation is None:
                relations.append(_missing_relation_evidence(task))
                continue
            relations.append(
                _routerai_relation_evidence(
                    relation,
                    task=task,
                    trusted_domains=trusted_domains,
                    max_snippet_chars=max_snippet_chars,
                    metadata_sources=metadata_sources if len(tasks) == 1 else [],
                )
            )
            continue
        component = components_by_id.get(task.component_candidate_id)
        if component is None:
            components.append(
                ComponentEvidence(
                    component_candidate_id=task.component_candidate_id,
                    role=task.role or task.target_type,
                    part_number=task.part_number,
                    name=task.name,
                    evidence_status="not_found",
                    confidence="unknown",
                    warnings=["RouterAI evidence did not return this component."],
                )
            )
            continue
        components.append(
            _routerai_component_evidence(
                component,
                task=task,
                trusted_domains=trusted_domains,
                max_snippet_chars=max_snippet_chars,
                metadata_sources=metadata_sources if len(tasks) == 1 else [],
            )
        )
    error_count = sum(1 for component in components if component.evidence_status == "error")
    error_count += sum(1 for relation in relations if relation.status == "error")
    completed_tasks = sum(
        1 for component in components if component.evidence_status == "found" and component.sources
    )
    completed_tasks += sum(1 for relation in relations if relation.sources)
    return EvidencePack(
        enabled=True,
        provider=provider_name,
        total_tasks=len(tasks),
        completed_tasks=completed_tasks,
        error_count=error_count,
        components=components,
        relation_evidence=relations,
        evidence_summary=_evidence_summary(
            components,
            relation_evidence=relations,
            error_count=error_count,
        ),
        search_tasks=search_tasks,
    )


def _routerai_component_evidence(
    component: Mapping[str, Any],
    *,
    task: EvidenceSearchTask,
    trusted_domains: set[str],
    max_snippet_chars: int,
    metadata_sources: Sequence[EvidenceSource],
) -> ComponentEvidence:
    raw_sources = _mapping_rows(component.get("sources"))
    sources = [
        source
        for source in (
            _source_from_routerai_mapping(
                raw_source,
                trusted_domains=trusted_domains,
                max_snippet_chars=max_snippet_chars,
            )
            for raw_source in raw_sources
        )
        if source is not None
    ]
    if not sources and metadata_sources:
        sources = list(metadata_sources)

    facts = _clean_routerai_facts(component.get("facts"))
    status = _routerai_status(component.get("evidence_status"), facts=facts, sources=sources)
    confidence = _routerai_confidence(component.get("confidence"))
    if not sources:
        facts = {}
        confidence = "unknown"
    if status == "found" and confidence == "unknown":
        confidence = _component_confidence(facts, sources)
    warnings = _unique_texts(str(value) for value in _string_sequence(component.get("warnings")))
    if status == "not_found" and not warnings:
        warnings.append("No external evidence found.")

    fact_list = [
        EvidenceFact(
            name=key,
            value=value,
            confidence=confidence,
            source_domains=sorted({source.domain for source in sources if source.domain}),
        )
        for key, value in facts.items()
        if value not in (None, "", [], UNKNOWN_FACT)
    ]
    return ComponentEvidence(
        component_candidate_id=task.component_candidate_id,
        role=str(component.get("role") or task.role or task.target_type).strip(),
        part_number=task.part_number,
        name=task.name,
        evidence_status=status,
        confidence=confidence,
        facts=facts,
        fact_list=fact_list,
        sources=sources,
        warnings=warnings,
    )


def _routerai_relation_evidence(
    relation: Mapping[str, Any],
    *,
    task: EvidenceSearchTask,
    trusted_domains: set[str],
    max_snippet_chars: int,
    metadata_sources: Sequence[EvidenceSource],
) -> RelationEvidence:
    raw_sources = _mapping_rows(relation.get("sources"))
    sources = [
        source
        for source in (
            _source_from_routerai_mapping(
                raw_source,
                trusted_domains=trusted_domains,
                max_snippet_chars=max_snippet_chars,
            )
            for raw_source in raw_sources
        )
        if source is not None
    ]
    if not sources and metadata_sources:
        sources = list(metadata_sources)

    relation_type = _routerai_relation_type(relation.get("relation_type"), task=task)
    missing = _unique_texts(
        str(value) for value in _string_sequence(relation.get("missing_evidence"))
    )
    mismatch = _unique_texts(
        str(value) for value in _string_sequence(relation.get("mismatch_facts"))
    )
    checks = _unique_texts(
        str(value) for value in _string_sequence(relation.get("engineering_checks"))
    )
    confirmed = _unique_texts(
        str(value) for value in _string_sequence(relation.get("confirmed_facts"))
    )
    status = _routerai_relation_status(
        relation.get("status"),
        sources=sources,
        confirmed=confirmed,
        missing=missing,
        mismatch=mismatch,
    )
    confidence = _routerai_confidence(relation.get("confidence"))
    if status == "not_confirmed":
        confidence = (
            "unknown"
            if not sources
            else (confidence if confidence != "unknown" else "low")
        )
    if status == "partially_confirmed" and confidence == "unknown":
        confidence = "medium" if sources else "unknown"
    if status == "confirmed" and confidence == "unknown":
        confidence = "high" if sources else "unknown"
    if status == "mismatch" and confidence == "unknown":
        confidence = "high" if sources else "medium"
    if status == "partially_confirmed" and not missing:
        missing.append(_default_relation_missing_text(relation_type))
    if status == "mismatch" and not mismatch:
        mismatch.append("External evidence reports a relation mismatch.")
    if status in {"partially_confirmed", "not_confirmed"} and not checks:
        checks.append(_default_relation_engineering_check(relation_type))
    warnings = _unique_texts(str(value) for value in _string_sequence(relation.get("warnings")))
    if status == "not_confirmed" and not missing:
        missing.append(_default_relation_missing_text(relation_type))
    components = _clean_relation_components(
        relation.get("components") if isinstance(relation.get("components"), Mapping) else None,
        task=task,
    )
    return RelationEvidence(
        relation_type=relation_type,  # type: ignore[arg-type]
        recommendation_id=str(
            relation.get("recommendation_id") or task.recommendation_id
        ).strip(),
        components=components,
        **_relation_label_fields(task, components=components),
        status=status,
        confidence=confidence,
        confirmed_facts=confirmed,
        missing_evidence=missing,
        mismatch_facts=mismatch,
        engineering_checks=checks,
        sources=sources,
        warnings=warnings,
    )


def _relation_evidence_from_sources(
    task: EvidenceSearchTask,
    *,
    sources: list[EvidenceSource],
    task_error: bool,
    warnings: list[str],
) -> RelationEvidence:
    relation_type = _relation_type_for_task(task)
    if not sources:
        status = "error" if task_error else "not_confirmed"
        return RelationEvidence(
            relation_type=relation_type,  # type: ignore[arg-type]
            recommendation_id=task.recommendation_id,
            components=_relation_components_for_task(task),
            **_relation_label_fields(task),
            status=status,  # type: ignore[arg-type]
            confidence="unknown",
            missing_evidence=[_default_relation_missing_text(relation_type)],
            engineering_checks=[_default_relation_engineering_check(relation_type)],
            warnings=warnings or ["No external relation evidence found."],
        )
    text = _combined_text(
        [
            task.platform_name,
            task.platform_part_number,
            task.cpu_name,
            task.cpu_part_number,
            task.ram_name,
            task.ram_part_number,
            task.storage_name,
            task.storage_part_number,
            *[source.title for source in sources],
            *[source.snippet for source in sources],
        ]
    )
    mismatch = _relation_mismatch_facts(text, relation_type=relation_type)
    status = "mismatch" if mismatch else _relation_status_from_text(text, relation_type)
    confirmed = _relation_confirmed_facts(text, relation_type=relation_type)
    missing = []
    if status in {"partially_confirmed", "not_confirmed"}:
        missing.append(_default_relation_missing_text(relation_type))
    checks = []
    if status != "confirmed":
        checks.append(_default_relation_engineering_check(relation_type))
    confidence: Literal["high", "medium", "low", "unknown"]
    if status == "mismatch":
        confidence = "high"
    elif status == "confirmed":
        confidence = "high" if any(source.trust_score >= 0.85 for source in sources) else "medium"
    elif status == "partially_confirmed":
        confidence = "medium"
    else:
        confidence = "low"
    return RelationEvidence(
        relation_type=relation_type,  # type: ignore[arg-type]
        recommendation_id=task.recommendation_id,
        components=_relation_components_for_task(task),
        **_relation_label_fields(task),
        status=status,  # type: ignore[arg-type]
        confidence=confidence,
        confirmed_facts=confirmed,
        missing_evidence=missing,
        mismatch_facts=mismatch,
        engineering_checks=checks,
        sources=sources,
        warnings=_unique_texts(warnings),
    )


def _missing_relation_evidence(task: EvidenceSearchTask) -> RelationEvidence:
    relation_type = _relation_type_for_task(task)
    return RelationEvidence(
        relation_type=relation_type,  # type: ignore[arg-type]
        recommendation_id=task.recommendation_id,
        components=_relation_components_for_task(task),
        **_relation_label_fields(task),
        status="error",
        confidence="unknown",
        missing_evidence=["RouterAI evidence did not return this relation."],
        engineering_checks=[_default_relation_engineering_check(relation_type)],
    )


def _disabled_relation_evidence(
    task: EvidenceSearchTask,
    *,
    provider_name: str,
) -> RelationEvidence:
    relation_type = _relation_type_for_task(task)
    return RelationEvidence(
        relation_type=relation_type,  # type: ignore[arg-type]
        recommendation_id=task.recommendation_id,
        components=_relation_components_for_task(task),
        **_relation_label_fields(task),
        status="not_confirmed",
        confidence="unknown",
        missing_evidence=[_default_relation_missing_text(relation_type)],
        engineering_checks=[_default_relation_engineering_check(relation_type)],
        warnings=[f"Web evidence provider {provider_name} is disabled or not configured."],
    )


def _error_relation_evidence(task: EvidenceSearchTask, *, message: str) -> RelationEvidence:
    relation_type = _relation_type_for_task(task)
    return RelationEvidence(
        relation_type=relation_type,  # type: ignore[arg-type]
        recommendation_id=task.recommendation_id,
        components=_relation_components_for_task(task),
        **_relation_label_fields(task),
        status="error",
        confidence="unknown",
        missing_evidence=[_default_relation_missing_text(relation_type)],
        engineering_checks=[_default_relation_engineering_check(relation_type)],
        warnings=[_safe_error_message(RuntimeError(message))],
    )


def _routerai_relation_type(value: Any, *, task: EvidenceSearchTask) -> str:
    relation_type = str(value or "").strip()
    if relation_type in RELATION_TYPES:
        return relation_type
    return _relation_type_for_task(task)


def _routerai_relation_status(
    value: Any,
    *,
    sources: Sequence[EvidenceSource],
    confirmed: Sequence[str],
    missing: Sequence[str],
    mismatch: Sequence[str],
) -> Literal["confirmed", "partially_confirmed", "not_confirmed", "mismatch"]:
    status = str(value or "").strip().lower()
    if status in {"mismatch", "incompatible", "not_compatible", "unsupported", "conflict"}:
        return "mismatch"
    if mismatch:
        return "mismatch"
    if not sources:
        return "not_confirmed"
    if status == "confirmed":
        return "confirmed"
    if status == "partially_confirmed":
        return "partially_confirmed"
    if confirmed:
        if missing or status in {"error", "not_confirmed"}:
            return "partially_confirmed"
        return "confirmed"
    if status in {"not_confirmed", "not_found", "error"}:
        return "not_confirmed"
    return "partially_confirmed"


def _relation_status_from_text(text: str, relation_type: str) -> str:
    lowered = text.casefold()
    support_markers = (
        "supported cpu list",
        "cpu support list",
        "support list",
        "qvl",
        "qualified vendor list",
        "validated",
        "compatible",
    )
    if relation_type == "platform_cpu" and any(marker in lowered for marker in support_markers):
        return "confirmed"
    if relation_type == "platform_ram" and any(marker in lowered for marker in support_markers):
        return "confirmed"
    if relation_type == "platform_storage" and any(marker in lowered for marker in support_markers):
        return "confirmed"
    if relation_type == "build_sanity" and (
        "configuration" in lowered or "datasheet" in lowered or "specification" in lowered
    ):
        return "partially_confirmed"
    return "partially_confirmed"


def _relation_mismatch_facts(text: str, *, relation_type: str) -> list[str]:
    lowered = text.casefold()
    mismatch: list[str] = []
    if re.search(r"\bnot\s+(?:supported|compatible)\b|\bunsupported\b", lowered):
        mismatch.append("Source says the selected relation is not supported.")
    if relation_type == "platform_cpu":
        sockets = sorted(set(re.findall(r"\b(?:FC)?LGA\s*(\d{4})\b", text, re.IGNORECASE)))
        if len(sockets) >= 2:
            mismatch.append("Source text contains conflicting CPU/platform sockets.")
    if relation_type == "platform_ram" and "ddr4" in lowered and "ddr5" in lowered:
        mismatch.append("Source text contains conflicting DDR4/DDR5 memory evidence.")
    if relation_type == "platform_storage" and "no nvme" in lowered:
        mismatch.append("Source says NVMe is not supported.")
    return _unique_texts(mismatch)


def _relation_confirmed_facts(text: str, *, relation_type: str) -> list[str]:
    facts: list[str] = []
    for pattern, label in (
        (r"\bLGA\s*4677\b|\bFCLGA\s*4677\b", "LGA4677"),
        (r"\bLGA\s*4189\b|\bFCLGA\s*4189\b", "LGA4189"),
        (r"\bDDR\s*5\b", "DDR5"),
        (r"\bDDR\s*4\b", "DDR4"),
        (r"\bNVMe\b", "NVMe"),
        (r"\bU\.?2\b", "U.2"),
        (r"\bU\.?3\b", "U.3"),
        (r"\b2\s*U\b", "2U"),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            facts.append(label)
    if relation_type == "platform_cpu" and re.search(
        r"\bCPU support list\b|\bsupported CPU list\b", text, re.IGNORECASE
    ):
        facts.append("CPU support list")
    if relation_type == "platform_ram" and re.search(r"\bQVL\b", text, re.IGNORECASE):
        facts.append("memory QVL")
    if relation_type == "platform_storage" and re.search(
        r"\bbackplane\b", text, re.IGNORECASE
    ):
        facts.append("backplane")
    return _unique_texts(facts)


def _default_relation_missing_text(relation_type: str) -> str:
    labels = {
        "platform_cpu": "CPU support list for this platform/CPU pair was not found.",
        "platform_ram": "RAM QVL/support evidence for this platform/RAM pair was not found.",
        "platform_storage": (
            "Backplane/support evidence for this platform/storage pair was not found."
        ),
        "build_sanity": "Whole-build vendor configuration evidence was not found.",
    }
    return labels.get(relation_type, "Relation support evidence was not found.")


def _default_relation_engineering_check(relation_type: str) -> str:
    labels = {
        "platform_cpu": "Check the platform CPU support list with an engineer.",
        "platform_ram": (
            "Check platform memory QVL, DIMM rules, and total capacity with an engineer."
        ),
        "platform_storage": "Check NVMe/U.2/U.3 backplane and controller support with an engineer.",
        "build_sanity": "Run a final whole-build engineering compatibility review.",
    }
    return labels.get(relation_type, "Check relation compatibility with an engineer.")


def _clean_relation_components(
    value: Mapping[str, Any] | None,
    *,
    task: EvidenceSearchTask,
) -> dict[str, Any]:
    fallback = _relation_components_for_task(task)
    if not isinstance(value, Mapping):
        return fallback
    result: dict[str, Any] = {}
    for key in ("platform", "cpu", "ram", "storage"):
        row = value.get(key)
        if isinstance(row, Mapping):
            result[key] = {
                "component_id": str(
                    row.get("component_id") or row.get("component_candidate_id") or ""
                ).strip(),
                "name": str(row.get("name") or "").strip(),
                "part_number": str(row.get("part_number") or "").strip(),
            }
        elif key in fallback:
            result[key] = fallback[key]
    return result or fallback


def _source_from_routerai_mapping(
    value: Mapping[str, Any],
    *,
    trusted_domains: set[str],
    max_snippet_chars: int,
) -> EvidenceSource | None:
    url = str(value.get("url") or value.get("link") or value.get("href") or "").strip()
    title = str(value.get("title") or value.get("name") or "").strip()
    snippet = str(
        value.get("snippet") or value.get("content") or value.get("text") or ""
    ).strip()
    if not url and not title:
        return None
    domain = str(value.get("domain") or _domain_from_url(url)).strip().casefold()
    source_type = _safe_source_type(value.get("source_type"), domain, trusted_domains)
    trust_score = _safe_trust_score(
        value.get("trust_score"),
        source_type=source_type,
        domain=domain,
        trusted_domains=trusted_domains,
    )
    return EvidenceSource(
        url=url,
        title=title[:300],
        snippet=snippet[:max(0, max_snippet_chars)],
        domain=domain,
        source_type=source_type,
        trust_score=trust_score,
        retrieved_at=datetime.now(UTC).isoformat(),
    )


def _routerai_metadata_sources(
    body: Mapping[str, Any],
    *,
    trusted_domains: set[str],
    max_snippet_chars: int,
) -> list[EvidenceSource]:
    raw_sources: list[Any] = []
    for key in ("citations", "sources", "references"):
        value = body.get(key)
        if isinstance(value, list):
            raw_sources.extend(value)
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, Mapping):
            message = first_choice.get("message")
            if isinstance(message, Mapping):
                for key in ("citations", "sources", "references", "annotations"):
                    value = message.get(key)
                    if isinstance(value, list):
                        raw_sources.extend(value)
    sources: list[EvidenceSource] = []
    for raw_source in raw_sources:
        source = _coerce_routerai_metadata_source(
            raw_source,
            trusted_domains=trusted_domains,
            max_snippet_chars=max_snippet_chars,
        )
        if source is not None and source.url not in {existing.url for existing in sources}:
            sources.append(source)
    return sources


def _coerce_routerai_metadata_source(
    value: Any,
    *,
    trusted_domains: set[str],
    max_snippet_chars: int,
) -> EvidenceSource | None:
    if isinstance(value, str):
        return _source_from_routerai_mapping(
            {"url": value},
            trusted_domains=trusted_domains,
            max_snippet_chars=max_snippet_chars,
        )
    if not isinstance(value, Mapping):
        return None
    if isinstance(value.get("url_citation"), Mapping):
        value = value["url_citation"]
    return _source_from_routerai_mapping(
        value,
        trusted_domains=trusted_domains,
        max_snippet_chars=max_snippet_chars,
    )


def _clean_routerai_facts(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    allowed = {
        "vendor",
        "platform_family",
        "cpu_generation",
        "socket_family",
        "supported_cpu_generation",
        "memory_type",
        "dimm_slots",
        "drive_bays",
        "nvme_support",
        "form_factor",
        "psu_info",
        "storage_interface",
        "capacity",
        "notes",
        "cores",
    }
    facts: dict[str, Any] = {}
    for key, raw_value in value.items():
        clean_key = str(key or "").strip()
        if clean_key not in allowed or raw_value in (None, "", [], UNKNOWN_FACT):
            continue
        facts[clean_key] = raw_value
    return facts


def _routerai_status(
    value: Any,
    *,
    facts: Mapping[str, Any],
    sources: Sequence[EvidenceSource],
) -> Literal["found", "not_found", "error"]:
    status = str(value or "").strip()
    if status in {"found", "not_found", "error"}:
        if status == "found" and not facts and not sources:
            return "not_found"
        if status == "found" and not sources:
            return "not_found"
        return status  # type: ignore[return-value]
    return "found" if sources else "not_found"


def _routerai_confidence(value: Any) -> Literal["high", "medium", "low", "unknown"]:
    confidence = str(value or "").strip()
    if confidence in {"high", "medium", "low", "unknown"}:
        return confidence  # type: ignore[return-value]
    return "unknown"


def _safe_source_type(value: Any, domain: str, trusted_domains: set[str]) -> str:
    source_type = str(value or "").strip()
    if source_type in {
        "official_vendor",
        "cpu_vendor",
        "memory_vendor",
        "storage_vendor",
        "distributor",
        "unknown",
    }:
        return source_type
    return _source_type(domain, trusted_domains)


def _safe_trust_score(
    value: Any,
    *,
    source_type: str,
    domain: str,
    trusted_domains: set[str],
) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = _trust_score(source_type, domain, trusted_domains)
    return min(1.0, max(0.0, score))


def _string_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _extract_facts(task: EvidenceSearchTask, sources: Sequence[EvidenceSource]) -> dict[str, Any]:
    text = _combined_text(
        [
            task.producer,
            task.part_number,
            task.name,
            *[source.title for source in sources],
            *[source.snippet for source in sources],
        ]
    )
    facts: dict[str, Any] = {}
    vendor = _detect_vendor(text, fallback=task.producer)
    if vendor != UNKNOWN_FACT:
        facts["vendor"] = vendor

    target_kind = _component_target_kind(task.target_type)
    if target_kind in {"platform", "ready_server"}:
        facts.update(_extract_platform_facts(text))
    elif target_kind == "cpu":
        facts.update(_extract_cpu_facts(text))
    elif target_kind == "ram":
        facts.update(_extract_ram_facts(text))
    elif target_kind == "storage":
        facts.update(_extract_storage_facts(text))
    return {key: value for key, value in facts.items() if value not in (None, "", [], UNKNOWN_FACT)}


def _extract_platform_facts(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    facts: dict[str, Any] = {}
    if re.search(r"\bpoweredge\s+r750xs\b|\br750xs\b|\bpoweredge\s+r750\b|\br750\b", lowered):
        facts.update(
            {
                "platform_family": "Dell PowerEdge R750/R750xs",
                "supported_cpu_generation": "3rd Gen Intel Xeon Scalable",
                "socket_family": "LGA4189",
                "memory_type": "DDR4",
            }
        )
    if re.search(r"\bpoweredge\s+r760\b|\br760\b|\bpoweredge\s+r660\b|\br660\b", lowered):
        facts.update(
            {
                "platform_family": "Dell PowerEdge R760/R660",
                "supported_cpu_generation": "4th/5th Gen Intel Xeon Scalable",
                "socket_family": "LGA4677",
                "memory_type": "DDR5",
            }
        )
    if re.search(r"\brs720-e11-rs24u\b|\brs720\s*e11\b", lowered):
        facts.update(
            {
                "platform_family": "ASUS RS720-E11-RS24U",
                "supported_cpu_generation": "4th/5th Gen Intel Xeon Scalable",
                "socket_family": "LGA4677",
                "memory_type": "DDR5",
                "nvme_support": True,
            }
        )
    if re.search(r"\bsys-621c-tn12r\b|\b621c-tn12r\b", lowered):
        facts.update(
            {
                "platform_family": "Supermicro SYS-621C-TN12R",
                "supported_cpu_generation": "4th/5th Gen Intel Xeon Scalable",
                "socket_family": "LGA4677",
                "memory_type": "DDR5",
                "nvme_support": True,
                "storage_interface": "NVMe/SAS/SATA",
            }
        )
    if re.search(r"\b(?:epyc\s*)?(?:7001|7002|7003)\b|\bsp3\b", lowered):
        facts.setdefault("supported_cpu_generation", "AMD EPYC 7001/7002/7003")
        facts.setdefault("socket_family", "SP3")
        facts.setdefault("memory_type", "DDR4")
    if re.search(r"\b(?:epyc\s*)?9004\b|\bsp5\b|\blga\s*6096\b|\bgenoa\b", lowered):
        facts.setdefault("supported_cpu_generation", "AMD EPYC 9004")
        facts.setdefault("socket_family", "SP5" if "sp5" in lowered else "LGA6096")
        facts.setdefault("memory_type", "DDR5")
    if re.search(r"\bice\s*lake\b|\b3rd\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b", lowered):
        facts.setdefault("supported_cpu_generation", "3rd Gen Intel Xeon Scalable")
        facts.setdefault("socket_family", "LGA4189")
        facts.setdefault("memory_type", "DDR4")
    if re.search(
        r"\bsapphire\s+rapids\b|\brapid\s+sapphire\b|\b4th\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b",
        lowered,
    ):
        facts.setdefault("supported_cpu_generation", "4th Gen Intel Xeon Scalable")
        facts.setdefault("socket_family", "LGA4677")
        facts.setdefault("memory_type", "DDR5")
    facts.update(_extract_common_platform_facts(text))
    return facts


def _extract_common_platform_facts(text: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    socket = _detect_socket(text)
    if socket != UNKNOWN_FACT:
        facts.setdefault("socket_family", socket)
    memory_type = _detect_memory_type(text)
    if memory_type != UNKNOWN_FACT:
        facts.setdefault("memory_type", memory_type)
    dimm_slots = _first_int_match(
        text,
        [
            r"\b(\d{1,3})\s*(?:dimm|ddr[45]\s+dimm)\s+slots?\b",
            r"\b(\d{1,3})\s*slots?\s+(?:dimm|ddr[45])\b",
        ],
    )
    if dimm_slots is not None:
        facts["dimm_slots"] = dimm_slots
    drive_bays = _drive_bay_text(text)
    if drive_bays:
        facts["drive_bays"] = drive_bays
    if re.search(r"\bnvme\b|\bu\.?2\b|\bu\.?3\b", text, re.IGNORECASE):
        facts["nvme_support"] = True
    if re.search(r"\bno\s+nvme\b|\bwithout\s+nvme\b", text, re.IGNORECASE):
        facts["nvme_support"] = False
    form_factor = _form_factor(text)
    if form_factor:
        facts["form_factor"] = form_factor
    psu_info = _psu_info(text)
    if psu_info:
        facts["psu_info"] = psu_info
    return facts


def _extract_cpu_facts(text: str) -> dict[str, Any]:
    lowered = text.casefold()
    facts: dict[str, Any] = {}
    if re.search(r"\bxeon\s+6\b|\b65\d{2}p\b|\b6[57]\d{2}p\b", lowered):
        facts.update({"vendor": "Intel", "cpu_generation": "Xeon 6", "socket_family": "LGA4710"})
    if re.search(r"\bxeon\s+gold\s+6326\b|\b6326\b", lowered):
        facts.update(
            {
                "vendor": "Intel",
                "cpu_generation": "3rd Gen Intel Xeon Scalable",
                "socket_family": "LGA4189",
            }
        )
    if re.search(r"\bxeon\s+gold\s+5220r\b|\b5220r\b", lowered):
        facts.update(
            {
                "vendor": "Intel",
                "cpu_generation": "2nd Gen Intel Xeon Scalable",
                "socket_family": "LGA3647",
            }
        )
    if re.search(r"\b4th\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b|\bsapphire\s+rapids\b", lowered):
        facts.update({"vendor": "Intel", "cpu_generation": "4th Gen Intel Xeon Scalable"})
        facts.setdefault("socket_family", "LGA4677")
    if re.search(r"\b5th\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b|\bemerald\s+rapids\b", lowered):
        facts.update({"vendor": "Intel", "cpu_generation": "5th Gen Intel Xeon Scalable"})
        facts.setdefault("socket_family", "LGA4677")
    if re.search(r"\bepyc\s+9\d{3}\b|\bepyc\s+9004\b|\bgenoa\b", lowered):
        facts.update({"vendor": "AMD", "cpu_generation": "AMD EPYC 9004", "socket_family": "SP5"})
    socket = _detect_socket(text)
    if socket != UNKNOWN_FACT:
        facts["socket_family"] = socket
    cores = _first_int_match(text, [r"\b(\d{1,3})\s*(?:cores?|c)\b"])
    if cores is not None:
        facts["cores"] = cores
    return facts


def _extract_ram_facts(text: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    memory_type = _detect_memory_type(text)
    if memory_type != UNKNOWN_FACT:
        facts["memory_type"] = memory_type
    if re.search(r"\brdimm\b", text, re.IGNORECASE):
        facts["notes"] = "RDIMM"
    if re.search(r"\blrdimm\b", text, re.IGNORECASE):
        facts["notes"] = "LRDIMM"
    capacity_gb = _capacity_gb(text)
    if capacity_gb is not None:
        facts["capacity"] = f"{capacity_gb}GB"
    speed_match = re.search(
        r"\b(2[0-9]{3}|3[0-9]{3}|4[0-9]{3}|5[0-9]{3}|6[0-9]{3})"
        r"\s*(?:mt/s|mhz)\b",
        text,
        re.IGNORECASE,
    )
    if speed_match is not None:
        facts["notes"] = " ".join(
            part for part in [str(facts.get("notes") or ""), f"{speed_match.group(1)} MT/s"] if part
        )
    return facts


def _extract_storage_facts(text: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    interface = _storage_interface(text)
    if interface != UNKNOWN_FACT:
        facts["storage_interface"] = interface
    capacity = _storage_capacity(text)
    if capacity:
        facts["capacity"] = capacity
    form_factor = _storage_form_factor(text)
    if form_factor:
        facts["form_factor"] = form_factor
    return facts


def _source_from_result(
    result: EvidenceSearchResult,
    *,
    trusted_domains: set[str],
    max_snippet_chars: int,
) -> EvidenceSource:
    url = str(result.url or "").strip()
    domain = str(result.domain or _domain_from_url(url)).strip().casefold()
    source_type = _source_type(domain, trusted_domains)
    return EvidenceSource(
        url=url,
        title=str(result.title or "").strip()[:300],
        snippet=str(result.snippet or "").strip()[:max(0, max_snippet_chars)],
        domain=domain,
        source_type=source_type,
        trust_score=_trust_score(source_type, domain, trusted_domains),
        retrieved_at=datetime.now(UTC).isoformat(),
    )


def _source_type(domain: str, trusted_domains: set[str]) -> str:
    if not domain:
        return "unknown"
    if _domain_is_trusted(domain, trusted_domains):
        if any(marker in domain for marker in ("intel.com", "amd.com", "ark.intel.com")):
            return "cpu_vendor"
        if any(marker in domain for marker in ("samsung.com", "kioxia.com", "micron.com")):
            return "storage_vendor"
        return "official_vendor"
    if any(marker in domain for marker in ("intel.com", "amd.com")):
        return "cpu_vendor"
    if any(marker in domain for marker in ("samsung.com", "kioxia.com", "micron.com")):
        return "storage_vendor"
    if any(marker in domain for marker in ("ocs.ru", "shop", "store", "distributor")):
        return "distributor"
    return "unknown"


def _trust_score(source_type: str, domain: str, trusted_domains: set[str]) -> float:
    if source_type == "official_vendor":
        return 0.95
    if source_type == "cpu_vendor":
        return 0.92
    if source_type in {"memory_vendor", "storage_vendor"}:
        return 0.88
    if _domain_is_trusted(domain, trusted_domains):
        return 0.8
    if source_type == "distributor":
        return 0.55
    return 0.3


def _component_confidence(
    facts: Mapping[str, Any],
    sources: Sequence[EvidenceSource],
) -> Literal["high", "medium", "low", "unknown"]:
    if not sources:
        return "unknown"
    trusted_sources = [source for source in sources if source.trust_score >= 0.85]
    if trusted_sources and len(facts) >= 2:
        return "high"
    if facts:
        return "medium"
    return "low"


def _disabled_component_evidence(
    task: EvidenceSearchTask,
    *,
    provider_name: str,
) -> ComponentEvidence:
    return ComponentEvidence(
        component_candidate_id=task.component_candidate_id,
        role=task.role or task.target_type,
        part_number=task.part_number,
        name=task.name,
        evidence_status="disabled",
        confidence="unknown",
        warnings=[f"Web evidence provider {provider_name} is disabled or not configured."],
    )


def _evidence_summary(
    components: Sequence[ComponentEvidence],
    *,
    relation_evidence: Sequence[RelationEvidence] | None = None,
    error_count: int,
) -> str:
    found = sum(1 for component in components if component.evidence_status == "found")
    relations = list(relation_evidence or [])
    relation_found = sum(1 for relation in relations if relation.sources)
    total = len(components) + len(relations)
    if total == 0:
        return "no evidence tasks generated"
    if found == 0 and relation_found == 0 and error_count:
        return f"external evidence unavailable; {error_count} search errors"
    if found == 0 and relation_found == 0:
        return f"no external evidence found for {total} evidence tasks"
    return (
        f"external evidence found for {found} component tasks and "
        f"{relation_found} relation tasks of {total} evidence tasks"
    )


def _detect_vendor(text: str, *, fallback: str = "") -> str:
    source = f"{fallback} {text}".casefold()
    vendors = (
        ("Dell", r"\bdell\b|\bpoweredge\b"),
        ("HPE", r"\bhpe\b|\bhewlett\s+packard\b"),
        ("Lenovo", r"\blenovo\b|\bthinksystem\b"),
        ("ASUS", r"\basus\b"),
        ("Supermicro", r"\bsuper\s*micro\b|\bsupermicro\b"),
        ("Intel", r"\bintel\b|\bxeon\b"),
        ("AMD", r"\bamd\b|\bepyc\b"),
        ("Samsung", r"\bsamsung\b"),
        ("KIOXIA", r"\bkioxia\b"),
        ("Micron", r"\bmicron\b"),
        ("Gooxi", r"\bgooxi\b"),
    )
    for vendor, pattern in vendors:
        if re.search(pattern, source, re.IGNORECASE):
            return vendor
    return fallback.strip() or UNKNOWN_FACT


def _detect_socket(text: str) -> str:
    lga_match = re.search(r"\b(?:FC)?LGA\s*(3647|4189|4677|4710|4094|6096)\b", text, re.IGNORECASE)
    if lga_match is not None:
        return f"LGA{lga_match.group(1)}"
    sp_match = re.search(r"\bSP\s*([35])\b", text, re.IGNORECASE)
    if sp_match is not None:
        return f"SP{sp_match.group(1)}"
    lowered = text.casefold()
    if re.search(r"\bice\s*lake\b|\b3rd\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b", lowered):
        return "LGA4189"
    if re.search(
        r"\bsapphire\s+rapids\b|\brapid\s+sapphire\b|\b4th\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b",
        lowered,
    ):
        return "LGA4677"
    if re.search(r"\b(?:epyc\s*)?(?:7001|7002|7003)\b|\bsp3\b", lowered):
        return "SP3"
    if re.search(r"\b(?:epyc\s*)?9004\b|\bsp5\b|\bgenoa\b", lowered):
        return "SP5"
    return UNKNOWN_FACT


def _detect_memory_type(text: str) -> str:
    if re.search(r"\bDDR\s*5\b", text, re.IGNORECASE):
        return "DDR5"
    if re.search(r"\bDDR\s*4\b", text, re.IGNORECASE):
        return "DDR4"
    lowered = text.casefold()
    if re.search(r"\bice\s*lake\b|\b3rd\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b", lowered):
        return "DDR4"
    if re.search(
        r"\bsapphire\s+rapids\b|\brapid\s+sapphire\b|\b4th\s+gen(?:eration)?\s+(?:intel\s+)?xeon\b|\blga\s*4677\b",
        lowered,
    ):
        return "DDR5"
    if re.search(r"\b(?:epyc\s*)?(?:7001|7002|7003)\b|\bsp3\b", lowered):
        return "DDR4"
    if re.search(r"\b(?:epyc\s*)?9004\b|\bsp5\b|\bgenoa\b", lowered):
        return "DDR5"
    return UNKNOWN_FACT


def _storage_interface(text: str) -> str:
    if re.search(r"\bNVMe\b|\bU\.?2\b|\bU\.?3\b", text, re.IGNORECASE):
        return "NVMe"
    if re.search(r"\bSAS\b", text, re.IGNORECASE):
        return "SAS"
    if re.search(r"\bSATA\b", text, re.IGNORECASE):
        return "SATA"
    return UNKNOWN_FACT


def _storage_capacity(text: str) -> str:
    tb_match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*TB\b", text, re.IGNORECASE)
    if tb_match is not None:
        return f"{tb_match.group(1).replace(',', '.')}TB"
    gb_match = re.search(r"\b(\d{2,5})\s*GB\b", text, re.IGNORECASE)
    if gb_match is not None:
        return f"{gb_match.group(1)}GB"
    return ""


def _storage_form_factor(text: str) -> str:
    values: list[str] = []
    for pattern, label in (
        (r"\bU\.?2\b", "U.2"),
        (r"\bU\.?3\b", "U.3"),
        (r"\bM\.?2\b", "M.2"),
        (r"\b2\.5(?:\s*inch|[\"”])?\b", "2.5"),
        (r"\b3\.5(?:\s*inch|[\"”])?\b", "3.5"),
    ):
        if re.search(pattern, text, re.IGNORECASE):
            values.append(label)
    return ", ".join(_unique_texts(values))


def _drive_bay_text(text: str) -> str:
    values: list[str] = []
    count_match = re.search(
        r"\b(\d{1,2})\s*x?\s*(?:2\.5|3\.5)[\"”]?\s*(?:drive\s+)?bays?\b",
        text,
        re.IGNORECASE,
    )
    if count_match is not None:
        values.append(count_match.group(0))
    for marker in ("NVMe", "SAS", "SATA", "U.2", "U.3"):
        if re.search(rf"\b{re.escape(marker).replace('\\.', '\\.?')}\b", text, re.IGNORECASE):
            values.append(marker)
    return ", ".join(_unique_texts(values))


def _form_factor(text: str) -> str:
    match = re.search(r"\b([124])\s*U\b", text, re.IGNORECASE)
    return f"{match.group(1)}U" if match is not None else ""


def _psu_info(text: str) -> str:
    if re.search(r"\b(?:dual|redundant|2\s*x|2x)\s*(?:psu|power\s+suppl)", text, re.IGNORECASE):
        return "dual/redundant PSU"
    return ""


def _capacity_gb(text: str) -> int | None:
    gb_match = re.search(r"\b(\d{1,4})\s*GB\b", text, re.IGNORECASE)
    if gb_match is not None:
        return int(gb_match.group(1))
    tb_match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*TB\b", text, re.IGNORECASE)
    if tb_match is not None:
        return int(float(tb_match.group(1).replace(",", ".")) * 1024)
    return None


def _first_int_match(text: str, patterns: Sequence[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    return None


def _target_type_for_row(row: Mapping[str, Any]) -> str:
    return ROLE_TO_TARGET_TYPE.get(str(row.get("role") or "").strip(), "component_platform")


def _task_id(component_id: str, target_type: str) -> str:
    digest = hashlib.sha1(f"{target_type}:{component_id}".encode()).hexdigest()
    return f"ev_{digest[:12]}"


def _relation_task_id(
    *,
    recommendation_id: str,
    relation_type: str,
    platform: Mapping[str, str],
    cpu: Mapping[str, str],
    ram: Mapping[str, str],
    storage: Mapping[str, str],
) -> str:
    raw = json.dumps(
        {
            "recommendation_id": recommendation_id,
            "relation_type": relation_type,
            "platform": platform.get("component_id") or platform.get("part_number"),
            "cpu": cpu.get("component_id") or cpu.get("part_number"),
            "ram": ram.get("component_id") or ram.get("part_number"),
            "storage": storage.get("component_id") or storage.get("part_number"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(raw.encode()).hexdigest()
    return f"ev_rel_{digest[:12]}"


def _task_summary(task: EvidenceSearchTask) -> dict[str, Any]:
    summary = {
        "task_id": task.task_id,
        "target_type": task.target_type,
        "component_candidate_id": task.component_candidate_id,
        "role": task.role,
        "producer": task.producer,
        "part_number": task.part_number,
        "name": task.name,
        "queries": task.queries,
        "reason": task.reason,
    }
    if _is_relation_target(task.target_type):
        summary.update(
            {
                "relation_type": _relation_type_for_task(task),
                "recommendation_id": task.recommendation_id,
                "question": task.question,
                "components": _relation_components_for_task(task),
                "normalized_requirements": task.normalized_requirements,
            }
        )
    return summary


def _is_relation_target(target_type: str) -> bool:
    return str(target_type or "") in RELATION_TARGET_TYPES


def _component_target_kind(target_type: str) -> str:
    target = str(target_type or "").strip()
    if target.startswith("component_"):
        return target.removeprefix("component_")
    if target == "ready_server":
        return "ready_server"
    if target in {"platform", "cpu", "ram", "storage"}:
        return target
    return "platform"


def _relation_type_for_task(task: EvidenceSearchTask) -> str:
    relation_type = TARGET_TYPE_TO_RELATION_TYPE.get(str(task.target_type or ""))
    if relation_type:
        return relation_type
    role = str(task.role or "").strip()
    return role if role in RELATION_TYPES else "build_sanity"


def _relation_components_for_task(task: EvidenceSearchTask) -> dict[str, Any]:
    components = {
        "platform": {
            "component_id": task.platform_component_id,
            "name": task.platform_name,
            "part_number": task.platform_part_number,
        },
        "cpu": {
            "component_id": task.cpu_component_id,
            "name": task.cpu_name,
            "part_number": task.cpu_part_number,
        },
        "ram": {
            "component_id": task.ram_component_id,
            "name": task.ram_name,
            "part_number": task.ram_part_number,
        },
        "storage": {
            "component_id": task.storage_component_id,
            "name": task.storage_name,
            "part_number": task.storage_part_number,
        },
    }
    return {
        key: value
        for key, value in components.items()
        if any(str(item or "").strip() for item in value.values())
    }


def _relation_label_fields(
    task: EvidenceSearchTask,
    *,
    components: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    def component_name(role: str, fallback: str) -> str:
        row = components.get(role) if isinstance(components, Mapping) else None
        if isinstance(row, Mapping):
            value = str(row.get("name") or "").strip()
            if value:
                return value
        return fallback

    return {
        "platform_name": component_name("platform", task.platform_name),
        "cpu_name": component_name("cpu", task.cpu_name),
        "ram_name": component_name("ram", task.ram_name),
        "storage_name": component_name("storage", task.storage_name),
    }


def _relation_key_for_task(task: EvidenceSearchTask) -> str:
    return f"{task.recommendation_id}:{_relation_type_for_task(task)}"


def _relation_key_from_mapping(value: Mapping[str, Any]) -> str:
    recommendation_id = str(value.get("recommendation_id") or "").strip()
    relation_type = str(value.get("relation_type") or "").strip()
    if not recommendation_id or relation_type not in RELATION_TYPES:
        return ""
    return f"{recommendation_id}:{relation_type}"


def _safe_cache_name(value: str) -> str:
    return (
        re.sub(r"[^a-z0-9_.-]+", "_", str(value or "provider").casefold()).strip("_")
        or "provider"
    )


def _sanitize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _safe_error_message(exc: Exception) -> str:
    text = str(exc).strip()
    if not text:
        return type(exc).__name__
    return text[:300]


def _domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.casefold()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _trusted_domains(settings: Any) -> set[str]:
    raw = str(getattr(settings, "web_evidence_trusted_domains", "") or "")
    return {
        domain.strip().casefold().removeprefix("www.")
        for domain in raw.split(",")
        if domain.strip()
    }


def _domain_is_trusted(domain: str, trusted_domains: set[str]) -> bool:
    clean = domain.casefold().removeprefix("www.")
    return any(clean == trusted or clean.endswith(f".{trusted}") for trusted in trusted_domains)


def _coerce_search_result(value: Mapping[str, Any] | EvidenceSearchResult) -> EvidenceSearchResult:
    if isinstance(value, EvidenceSearchResult):
        return value
    url = str(value.get("url") or "")
    return EvidenceSearchResult(
        title=str(value.get("title") or ""),
        url=url,
        snippet=str(value.get("snippet") or value.get("content") or ""),
        domain=str(value.get("domain") or _domain_from_url(url)),
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _pack_mapping(value: Mapping[str, Any] | EvidencePack) -> Mapping[str, Any]:
    if isinstance(value, EvidencePack):
        return value.model_dump()
    return value if isinstance(value, Mapping) else {}


def _combined_text(values: Sequence[Any]) -> str:
    return " ".join(str(value or "") for value in values if value not in (None, ""))


def _unique_texts(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result
