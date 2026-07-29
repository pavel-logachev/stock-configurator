from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from app.core.config import (
    LlmSettings,
    WebEvidenceSettings,
    get_llm_settings,
    get_web_evidence_settings,
)
from app.evidence.web_evidence import (
    RELATION_TYPES,
    ROLE_TO_TARGET_TYPE,
    TASK_TARGET_TYPES,
    EvidencePack,
    EvidenceSearchTask,
    RouterAIWebEvidenceProvider,
    WebSearchProvider,
    build_relation_evidence_task,
    collect_web_evidence,
)

SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PROXY")
ERROR_PREVIEW_CHARS = 240


class _NoopEvidenceCache:
    def get(self, *, provider: str, query: str) -> None:
        return None

    def set(self, *, provider: str, query: str, results: Sequence[Any]) -> None:
        return None

    def get_pack(self, *, provider: str, cache_key: str) -> None:
        return None

    def set_pack(self, *, provider: str, cache_key: str, pack: EvidencePack) -> None:
        return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one sanitized RouterAI online web evidence smoke request."
    )
    parser.add_argument("--query", help="Evidence search query to send.")
    parser.add_argument("--component-id", help="Synthetic component id.")
    parser.add_argument(
        "--role",
        help="Component role, for example platform, cpu, ram, storage, or ready_server.",
    )
    parser.add_argument(
        "--relation",
        choices=sorted(RELATION_TYPES),
        help="Compatibility relation to check.",
    )
    parser.add_argument("--platform-name", default="", help="Platform display name.")
    parser.add_argument("--platform-part-number", default="", help="Platform part number.")
    parser.add_argument("--cpu-name", default="", help="CPU display name.")
    parser.add_argument("--cpu-part-number", default="", help="CPU part number.")
    parser.add_argument("--ram-name", default="", help="RAM display name.")
    parser.add_argument("--ram-part-number", default="", help="RAM part number.")
    parser.add_argument("--storage-name", default="", help="Storage display name.")
    parser.add_argument("--storage-part-number", default="", help="Storage part number.")
    parser.add_argument("--producer", default="", help="Optional producer/vendor hint.")
    parser.add_argument("--part-number", default="", help="Optional part-number hint.")
    parser.add_argument("--name", default="", help="Optional display name hint.")
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Print the planned sanitized summary without calling RouterAI.",
    )
    return parser.parse_args(argv)


def run(
    argv: Sequence[str] | None = None,
    *,
    provider: WebSearchProvider | None = None,
    stdout: Any = None,
) -> int:
    args = parse_args(argv)
    settings = get_web_evidence_settings()
    llm_settings = get_llm_settings()
    task = _task_from_args(args)

    if args.no_network:
        summary = _no_network_summary(settings=settings, task=task)
        _print_summary(summary, stdout=stdout)
        return 0

    configured_provider = str(settings.web_evidence_provider or "").strip().lower()
    if provider is None and configured_provider != "routerai":
        summary = _base_summary(settings=settings, task=task) | {
            "parse_status": "not_requested",
            "evidence_status": "configuration_error",
            "error_type": "unsupported_provider",
            "error_preview": (
                f"WEB_EVIDENCE_PROVIDER is {settings.web_evidence_provider!r}; expected 'routerai'."
            ),
        }
        _print_summary(summary, stdout=stdout)
        return 1

    effective_provider = provider or _build_routerai_provider(settings, llm_settings)
    smoke_settings = settings.model_copy(
        update={
            "web_evidence_enabled": True,
            "web_evidence_provider": "routerai",
        }
    )
    try:
        pack = collect_web_evidence(
            tasks=[task],
            settings=smoke_settings,
            provider=effective_provider,
            cache=_NoopEvidenceCache(),
            llm_settings=llm_settings,
        )
    finally:
        if provider is None and hasattr(effective_provider, "close"):
            effective_provider.close()  # type: ignore[attr-defined]

    summary = _summary_from_pack(pack, settings=smoke_settings, task=task)
    _print_summary(summary, stdout=stdout)
    return _exit_code(summary)


def _build_routerai_provider(
    settings: WebEvidenceSettings,
    llm_settings: LlmSettings,
) -> RouterAIWebEvidenceProvider:
    base_url = settings.web_evidence_base_url.strip() or llm_settings.llm_base_url.strip()
    api_key = settings.web_evidence_api_key.strip() or llm_settings.llm_api_key.strip()
    return RouterAIWebEvidenceProvider(
        base_url=base_url,
        api_key=api_key,
        model=settings.web_evidence_model.strip(),
        max_output_tokens=settings.web_evidence_max_output_tokens,
    )


def _task_from_args(args: argparse.Namespace) -> EvidenceSearchTask:
    if args.relation:
        return build_relation_evidence_task(
            relation_type=str(args.relation),
            recommendation_id="smoke_relation",
            platform={
                "component_candidate_id": "smoke-platform",
                "name": str(args.platform_name or "").strip(),
                "part_number": str(args.platform_part_number or "").strip(),
            },
            cpu={
                "component_candidate_id": "smoke-cpu",
                "name": str(args.cpu_name or "").strip(),
                "part_number": str(args.cpu_part_number or "").strip(),
            },
            ram={
                "component_candidate_id": "smoke-ram",
                "name": str(args.ram_name or "").strip(),
                "part_number": str(args.ram_part_number or "").strip(),
            },
            storage={
                "component_candidate_id": "smoke-storage",
                "name": str(args.storage_name or "").strip(),
                "part_number": str(args.storage_part_number or "").strip(),
            },
            normalized_requirements={},
        )
    if not args.query or not args.component_id or not args.role:
        raise SystemExit("--query, --component-id, and --role are required without --relation.")
    role = str(args.role or "").strip()
    target_type = ROLE_TO_TARGET_TYPE.get(role, role)
    if target_type not in TASK_TARGET_TYPES:
        target_type = "component_platform"
    query = _single_line(args.query)
    return EvidenceSearchTask(
        task_id=f"smoke_{_safe_task_fragment(args.component_id)}",
        target_type=target_type,  # type: ignore[arg-type]
        component_candidate_id=str(args.component_id).strip(),
        role=role,
        producer=str(args.producer or "").strip(),
        part_number=str(args.part_number or "").strip(),
        name=str(args.name or query).strip(),
        queries=[query],
        reason="routerai_online_web_evidence_smoke",
    )


def _no_network_summary(
    *,
    settings: WebEvidenceSettings,
    task: EvidenceSearchTask,
) -> dict[str, Any]:
    summary = _base_summary(settings=settings, task=task) | {
        "no_network": True,
        "parse_status": "not_requested",
        "evidence_status": "not_requested",
        "error_type": "",
    }
    if str(task.target_type).startswith("relation_"):
        summary["status"] = "not_requested"
    return summary


def _base_summary(
    *,
    settings: WebEvidenceSettings,
    task: EvidenceSearchTask,
) -> dict[str, Any]:
    summary = {
        "provider": str(settings.web_evidence_provider or "").strip() or "disabled",
        "model": str(settings.web_evidence_model or "").strip(),
        "http_status": None,
        "parse_status": "",
        "evidence_status": "",
        "sources_count": 0,
        "source_domains": [],
        "extracted_facts": {},
        "error_type": "",
        "component_id": task.component_candidate_id,
        "role": task.role or task.target_type,
        "query": task.queries[0] if task.queries else "",
    }
    if str(task.target_type).startswith("relation_"):
        summary.update(
            {
                "relation_type": task.role,
                "status": "",
                "confidence": "",
                "confirmed_facts": [],
                "missing_evidence": [],
                "question": task.question,
            }
        )
    return summary


def _summary_from_pack(
    pack: EvidencePack,
    *,
    settings: WebEvidenceSettings,
    task: EvidenceSearchTask,
) -> dict[str, Any]:
    diagnostics = pack.diagnostics if isinstance(pack.diagnostics, Mapping) else {}
    component = _first_component(pack, task.component_candidate_id)
    relation = _first_relation(pack, task)
    sources = [source for row in pack.components for source in row.sources]
    sources.extend(source for row in pack.relation_evidence for source in row.sources)
    domains = sorted({source.domain for source in sources if source.domain})
    facts = dict(component.facts) if component is not None else {}
    source_count = _int_value(diagnostics.get("evidence_sources_count"))
    if source_count is None:
        source_count = len(sources)

    summary = _base_summary(settings=settings, task=task) | {
        "provider": str(diagnostics.get("evidence_provider") or pack.provider or ""),
        "model": str(diagnostics.get("evidence_model") or settings.web_evidence_model or ""),
        "http_status": _int_value(diagnostics.get("evidence_http_status")),
        "parse_status": str(
            diagnostics.get("evidence_raw_response_parse_status") or "not_applicable"
        ),
        "evidence_status": component.evidence_status if component is not None else "",
        "sources_count": source_count,
        "source_domains": domains,
        "extracted_facts": facts,
        "error_type": str(diagnostics.get("evidence_error_type") or ""),
        "evidence_summary": pack.evidence_summary,
    }
    if relation is not None:
        summary.update(
            {
                "relation_type": relation.relation_type,
                "status": relation.status,
                "confidence": relation.confidence,
                "confirmed_facts": relation.confirmed_facts,
                "missing_evidence": relation.missing_evidence,
                "mismatch_facts": relation.mismatch_facts,
                "engineering_checks": relation.engineering_checks,
                "evidence_status": relation.status,
            }
        )

    preview = str(diagnostics.get("evidence_error_preview") or "").strip()
    if not preview and component is not None and component.warnings:
        preview = str(component.warnings[0])
    if summary["error_type"] and preview:
        summary["error_preview"] = _short_preview(preview)
    return summary


def _first_component(pack: EvidencePack, component_id: str) -> Any:
    for component in pack.components:
        if component.component_candidate_id == component_id:
            return component
    return pack.components[0] if pack.components else None


def _first_relation(pack: EvidencePack, task: EvidenceSearchTask) -> Any:
    if not str(task.target_type).startswith("relation_"):
        return None
    relation_type = task.role
    for relation in pack.relation_evidence:
        if (
            relation.recommendation_id == task.recommendation_id
            and relation.relation_type == relation_type
        ):
            return relation
    return pack.relation_evidence[0] if pack.relation_evidence else None


def _exit_code(summary: Mapping[str, Any]) -> int:
    if summary.get("no_network"):
        return 0
    if str(summary.get("error_type") or "").strip():
        return 1
    if str(summary.get("parse_status") or "") != "parsed":
        return 1
    if summary.get("relation_type"):
        status = str(summary.get("status") or "").strip()
        if status in {"confirmed", "partially_confirmed"}:
            return 0 if (_int_value(summary.get("sources_count")) or 0) > 0 else 1
        return 1
    if str(summary.get("evidence_status") or "") != "found":
        return 1
    return 0 if (_int_value(summary.get("sources_count")) or 0) > 0 else 1


def _print_summary(summary: Mapping[str, Any], *, stdout: Any = None) -> None:
    safe_summary = _redact_obj(summary, secrets=_env_secret_values())
    stream = stdout or sys.stdout
    print(json.dumps(safe_summary, ensure_ascii=False, indent=2, sort_keys=True), file=stream)


def _env_secret_values() -> set[str]:
    secrets: set[str] = set()
    for key, value in os.environ.items():
        if not value or len(value) < 4:
            continue
        upper_key = key.upper()
        if any(marker in upper_key for marker in SECRET_ENV_MARKERS):
            secrets.add(value)
            encoded = quote(value, safe="")
            if encoded != value:
                secrets.add(encoded)
    return secrets


def _redact_obj(value: Any, *, secrets: set[str]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secrets=secrets)
    if isinstance(value, Mapping):
        return {key: _redact_obj(item, secrets=secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_obj(item, secrets=secrets) for item in value]
    return value


def _redact_text(value: str, *, secrets: set[str]) -> str:
    redacted = value
    for secret in sorted(secrets, key=len, reverse=True):
        redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _short_preview(value: str) -> str:
    return _single_line(value)[:ERROR_PREVIEW_CHARS]


def _single_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_task_fragment(value: Any) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "component")).strip("_")
    return safe[:64] or "component"


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
