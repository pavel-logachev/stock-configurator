from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from app.reports.match_text import pluralize_ru

GENERIC_TITLE_MARKERS = (
    "вариант",
    "альтернатива",
    "рекомендац",
    "технически",
    "резерв",
)


def humanized_recommendation_title(
    recommendation: Mapping[str, Any],
    recommendations: Iterable[Mapping[str, Any]],
    *,
    index: int,
    default_title: str,
) -> str:
    title = str(recommendation.get("title") or "").strip() or default_title
    peers = [item for item in recommendations if isinstance(item, Mapping)]
    if not _is_generic_title(title) or not _has_same_platform_peer(recommendation, peers):
        return title

    difference_title = _difference_title(recommendation, peers)
    if difference_title:
        return difference_title
    if index > 1:
        return "Альтернатива на той же платформе"
    return title


def _is_generic_title(title: str) -> bool:
    lowered = title.casefold()
    return any(marker in lowered for marker in GENERIC_TITLE_MARKERS)


def _has_same_platform_peer(
    recommendation: Mapping[str, Any],
    peers: list[Mapping[str, Any]],
) -> bool:
    platform = _platform_key(recommendation)
    if not platform:
        return False
    return sum(1 for peer in peers if _platform_key(peer) == platform) > 1


def _platform_key(recommendation: Mapping[str, Any]) -> str:
    component = _component_by_role(recommendation, "server_platform")
    if component is not None:
        return _component_article(component).casefold()
    summary = recommendation.get("component_summary")
    if isinstance(summary, Mapping):
        return str(summary.get("platform") or "").strip().casefold()
    return str(
        recommendation.get("part_number")
        or recommendation.get("display_name")
        or recommendation.get("item_name")
        or ""
    ).strip().casefold()


def _difference_title(
    recommendation: Mapping[str, Any],
    peers: list[Mapping[str, Any]],
) -> str:
    for role, builder in (
        ("cpu", _cpu_title),
        ("ram", _ram_title),
        ("ssd", _storage_title),
        ("hdd", _storage_title),
    ):
        signature = _component_signature(recommendation, role)
        if not signature:
            continue
        if any(
            peer is not recommendation and _component_signature(peer, role) not in {"", signature}
            for peer in peers
            if _platform_key(peer) == _platform_key(recommendation)
        ):
            component = _component_by_role(recommendation, role)
            title = builder(component) if component is not None else ""
            if title:
                return title
    return ""


def _cpu_title(component: Mapping[str, Any]) -> str:
    cores = _component_cpu_cores(component)
    if cores is not None:
        return f"Вариант с CPU {cores} {pluralize_ru(cores, 'ядро', 'ядра', 'ядер')}"
    article = _component_article(component)
    return f"Вариант с CPU {article}" if article else ""


def _ram_title(component: Mapping[str, Any]) -> str:
    quantity = _int_value(component.get("quantity_required"))
    module_gb = _ram_module_gb(component)
    if quantity is not None and module_gb is not None:
        return f"Вариант с RAM {quantity * module_gb} ГБ к подбору"
    if module_gb is not None:
        return f"Вариант с RAM-модулями {module_gb} ГБ"
    article = _component_article(component)
    return f"Вариант с другой RAM ({article})" if article else "Вариант с другой RAM"


def _storage_title(component: Mapping[str, Any]) -> str:
    role = str(component.get("role") or "ssd").upper()
    capacity = _storage_capacity_tb(component)
    if capacity is not None:
        return f"Вариант с {role} {_format_number(capacity)} ТБ"
    article = _component_article(component)
    return f"Вариант с {role} {article}" if article else f"Вариант с другим {role}"


def _component_signature(recommendation: Mapping[str, Any], role: str) -> str:
    component = _component_by_role(recommendation, role)
    if component is None:
        return ""
    facts = component.get("facts")
    fact_values: list[str] = []
    if isinstance(facts, Mapping):
        fact_values = [
            str(facts.get(key) or "")
            for key in (
                "cpu_cores",
                "cpu_family",
                "ram_capacity_gb",
                "ram_type",
                "storage_capacity_tb",
                "storage_interface",
            )
            if facts.get(key) not in (None, "", "unknown")
        ]
    return "|".join(
        value
        for value in [
            str(component.get("producer") or "").strip(),
            str(component.get("part_number") or "").strip(),
            str(component.get("quantity_required") or "").strip(),
            *fact_values,
        ]
        if value
    ).casefold()


def _component_by_role(
    recommendation: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any] | None:
    components = recommendation.get("components")
    if not isinstance(components, Iterable) or isinstance(components, str):
        return None
    for component in components:
        if isinstance(component, Mapping) and component.get("role") == role:
            return component
    return None


def _component_article(component: Mapping[str, Any]) -> str:
    parts = [
        str(component.get("producer") or "").strip(),
        str(component.get("part_number") or "").strip(),
    ]
    display = " ".join(part for part in parts if part)
    if display:
        return display
    return str(
        component.get("item_name")
        or component.get("name")
        or component.get("item_id")
        or ""
    ).strip()


def _component_cpu_cores(component: Mapping[str, Any]) -> int | None:
    value = component.get("cpu_cores")
    facts = component.get("facts")
    if value in (None, "") and isinstance(facts, Mapping):
        value = facts.get("cpu_cores")
    parsed = _int_value(value)
    if parsed is not None:
        return parsed
    text = " ".join(
        str(component.get(key) or "")
        for key in ("part_number", "item_name", "name")
    )
    match = re.search(r"\b(\d{1,3})\s*(?:core|cores|ядер|ядра)\b", text, re.IGNORECASE)
    return _int_value(match.group(1)) if match else None


def _ram_module_gb(component: Mapping[str, Any]) -> int | None:
    value = component.get("ram_module_capacity_gb")
    facts = component.get("facts")
    if value in (None, "") and isinstance(facts, Mapping):
        value = facts.get("ram_capacity_gb")
    parsed = _int_value(value)
    if parsed is not None:
        return parsed
    text = " ".join(
        str(component.get(key) or "")
        for key in ("part_number", "item_name", "name")
    )
    match = re.search(r"\b(\d{2,4})\s*(?:gb|гб)\b", text, re.IGNORECASE)
    return _int_value(match.group(1)) if match else None


def _storage_capacity_tb(component: Mapping[str, Any]) -> Decimal | None:
    value = component.get("storage_capacity_tb")
    facts = component.get("facts")
    if value in (None, "") and isinstance(facts, Mapping):
        value = facts.get("storage_capacity_tb")
    if value in (None, ""):
        text = " ".join(
            str(component.get(key) or "")
            for key in ("part_number", "item_name", "name")
        )
        match = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(tb|тб|gb|гб)\b", text, re.IGNORECASE)
        if not match:
            return None
        amount = Decimal(match.group(1).replace(",", "."))
        return amount if match.group(2).casefold() in {"tb", "тб"} else amount / Decimal(1024)
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _format_number(value: Decimal) -> str:
    if value == value.to_integral():
        return str(int(value))
    return format(value.normalize(), "f")


def _int_value(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
