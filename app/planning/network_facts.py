from __future__ import annotations

import re
from collections.abc import Mapping
from math import ceil
from typing import Any

UNKNOWN_FACT = "unknown"

_MEDIA_PREFIX = r"(?:\b|(?<=[xXхХ*])|(?<=\u00d7))"
_PORT_MULTIPLIER = r"(?:x|х|\*|\u00d7)"

_MEDIA_PATTERNS: tuple[tuple[str, str], ...] = (
    ("QSFP56", rf"{_MEDIA_PREFIX}QSFP\s*56\b|{_MEDIA_PREFIX}QSFP56\b"),
    ("QSFP28", rf"{_MEDIA_PREFIX}QSFP\s*28\b|{_MEDIA_PREFIX}QSFP28\b"),
    ("QSFP+", rf"{_MEDIA_PREFIX}QSFP\+(?!\w)"),
    ("QSFP", rf"{_MEDIA_PREFIX}QSFP\b"),
    ("SFP28", rf"{_MEDIA_PREFIX}SFP\s*28\b|{_MEDIA_PREFIX}SFP28\b"),
    ("SFP+", rf"{_MEDIA_PREFIX}SFP\+(?!\w)"),
    ("SFP", rf"{_MEDIA_PREFIX}SFP\b"),
    ("RJ45", r"\bRJ\s*-?\s*45\b|BASE\s*-?\s*T(?:X)?\b|10GBASE\s*-?\s*T\b"),
)

_INTERFACE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("OCP", r"\bOCP\b|\bOCP\s*3\.?0\b"),
    ("PCIe", r"\bPCI\s*-?\s*E\b|\bPCIe\b"),
)


def extract_network_facts(text: str | None, raw_json: Any = None) -> dict[str, Any]:
    source = _joined_text(text, raw_json)
    ports_count = _extract_ports_count(source)
    speed_gbps = _extract_speed_gbps(source)
    media = _extract_media(source)
    interface = _extract_interface(source)
    return {
        "ports_count": ports_count,
        "speed": f"{speed_gbps}GbE" if speed_gbps is not None else UNKNOWN_FACT,
        "speed_gbps": speed_gbps,
        "media": media or UNKNOWN_FACT,
        "interface": interface or UNKNOWN_FACT,
    }


def network_requirement_from_sources(
    *,
    text: str | None = None,
    explicit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    explicit = explicit if isinstance(explicit, Mapping) else {}
    full_text = _joined_text(text, explicit)
    facts = extract_network_facts(full_text)
    min_ports = _int_value(
        explicit.get("min_ports_per_server")
        or explicit.get("ports_per_server")
        or explicit.get("ports")
        or explicit.get("port_count")
    )
    if min_ports is None:
        min_ports = facts["ports_count"]
    speed = _normalize_speed(
        explicit.get("speed")
        or explicit.get("min_speed")
        or explicit.get("required_speed")
        or facts["speed"]
    )
    media = _normalize_media(explicit.get("media") or explicit.get("connector") or facts["media"])
    interface = _normalize_interface(explicit.get("interface") or facts["interface"])

    required = _truthy(explicit.get("required")) or _advanced_network_requested(full_text)
    if required and min_ports is None and (speed != UNKNOWN_FACT or media != UNKNOWN_FACT):
        min_ports = 1
    return {
        "required": bool(required),
        "min_ports_per_server": min_ports,
        "speed": speed,
        "media": media,
        "interface": interface,
    }


def network_facts_satisfy_requirement(
    facts: Mapping[str, Any] | None,
    requirement: Mapping[str, Any] | None,
) -> bool:
    return _network_facts_satisfy_requirement(
        facts,
        requirement,
        require_full_port_count=True,
    )


def network_adapter_facts_satisfy_requirement(
    facts: Mapping[str, Any] | None,
    requirement: Mapping[str, Any] | None,
) -> bool:
    return _network_facts_satisfy_requirement(
        facts,
        requirement,
        require_full_port_count=False,
    )


def _network_facts_satisfy_requirement(
    facts: Mapping[str, Any] | None,
    requirement: Mapping[str, Any] | None,
    *,
    require_full_port_count: bool,
) -> bool:
    requirement = requirement if isinstance(requirement, Mapping) else {}
    if not requirement.get("required"):
        return True
    facts = facts if isinstance(facts, Mapping) else {}
    required_ports = _int_value(requirement.get("min_ports_per_server")) or 1
    ports = _int_value(_fact_value(facts, "ports_count", "network_ports_count"))
    if ports is None or ports <= 0:
        return False
    if require_full_port_count and ports < required_ports:
        return False
    if not _speed_satisfies(
        _fact_value(facts, "speed_gbps", "network_speed_gbps", "speed", "network_speed"),
        requirement.get("speed"),
    ):
        return False
    if not _media_satisfies(
        _fact_value(facts, "media", "network_media"),
        requirement.get("media"),
    ):
        return False
    required_interface = _normalize_interface(requirement.get("interface"))
    if required_interface != UNKNOWN_FACT:
        return (
            _normalize_interface(_fact_value(facts, "interface", "network_interface"))
            == required_interface
        )
    return True


def required_network_adapter_quantity(
    facts: Mapping[str, Any] | None,
    requirement: Mapping[str, Any] | None,
    *,
    server_quantity: int,
) -> int | None:
    requirement = requirement if isinstance(requirement, Mapping) else {}
    facts = facts if isinstance(facts, Mapping) else {}
    required_ports = _int_value(requirement.get("min_ports_per_server")) or 1
    ports_per_adapter = _int_value(_fact_value(facts, "ports_count", "network_ports_count"))
    if ports_per_adapter is None or ports_per_adapter <= 0:
        return None
    return max(1, server_quantity) * ceil(required_ports / ports_per_adapter)


def _fact_value(facts: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = facts.get(key)
        if value is not None and value != "":
            return value
    return None


def _advanced_network_requested(text: str) -> bool:
    speed_gbps = _extract_speed_gbps(text)
    if speed_gbps is not None and speed_gbps >= 10:
        return True
    media = _extract_media(text)
    if (media is not None and media != "RJ45") or re.search(r"\bOCP\s*NIC\b", text, re.I):
        return True
    if re.search(r"\b(?:NIC|network adapter|ethernet adapter)\b", text, re.I):
        return True
    return bool(
        re.search(
            r"(?:сетев\w+|порт\w+)[^\n,;]{0,40}"
            r"\b(?:10|25|40|56|100|200|400)\s*g",
            text,
            re.I,
        )
    )


def _extract_ports_count(text: str) -> int | None:
    label_values = {
        "single": 1,
        "dual": 2,
        "double": 2,
        "quad": 4,
        "one": 1,
        "two": 2,
        "four": 4,
    }
    speed_or_media = (
        r"(?:"
        r"(?:1000|100|(?:1|10|25|40|56|100|200|400)\s*g?)\s*base\s*-?\s*t[x]?"
        r"|gigabit"
        r"|"
        r"(?:1|10|25|40|56|100|200|400)\s*"
        r"(?:gb\s*/\s*s|gbit\s*/\s*s|g\s*bps|g\s*bit\s*/?\s*s?|g\s*b\s*e?|g\s*e?)"
        r"|SFP\s*28|SFP28|SFP\+|QSFP\s*28|QSFP28|QSFP\s*56|QSFP56|QSFP|RJ\s*-?\s*45|BASE\s*-?\s*T"
        r")"
    )
    multiplier = _PORT_MULTIPLIER
    for match in re.finditer(
        rf"\b(\d{{1,2}})\s*{multiplier}\s*(?={speed_or_media})",
        text,
        re.I,
    ):
        value = int(match.group(1))
        if 1 <= value <= 16:
            return value
    for label, value in label_values.items():
        if re.search(rf"\b{label}\s*[- ]?\s*ports?\b", text, re.I):
            return value
    contextual_label_values = {
        "single": 1,
        "dual": 2,
        "quad": 4,
        "one": 1,
        "two": 2,
        "four": 4,
    }
    excluded_next_words = r"(?:socket|cpu|processor|processors|node|nodes)"
    for label, value in contextual_label_values.items():
        if re.search(
            rf"\b{label}\b(?!\s+{excluded_next_words})[^\n,;]{{0,40}}{speed_or_media}",
            text,
            re.I,
        ) or re.search(
            rf"{speed_or_media}[^\n,;]{{0,40}}\b{label}\b(?!\s+{excluded_next_words})",
            text,
            re.I,
        ):
            return value
    if re.search(r"\bдвух\s*порт|\bdual\s*port", text, re.I):
        return 2
    if re.search(r"\bчетыр[её]х\s*порт|\bquad\s*port", text, re.I):
        return 4
    patterns = (
        r"\b(\d{1,2})\s*[- ]?\s*ports?\b",
        rf"\b(\d{{1,2}})\s*{multiplier}\s*(?:ports?|порт\w*)\b",
        r"\b(\d{1,2})\s*(?:сетев\w+\s+)?порт\w*\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match is None:
            continue
        value = int(match.group(1))
        if 1 <= value <= 16:
            return value
    return None


def _extract_speed_gbps(text: str) -> int | None:
    if re.search(
        rf"{_MEDIA_PREFIX}1000\s*base\s*-?\s*t[x]?\b"
        rf"|{_MEDIA_PREFIX}gigabit\b",
        text,
        re.I,
    ):
        return 1
    matches: list[int] = []
    for match in re.finditer(
        r"(?<!\d)(1|10|25|40|56|100|200|400)\s*"
        r"(?:"
        r"gb\s*/\s*s|gbit\s*/\s*s|"
        r"g\s*bps|g\s*bit\s*/?\s*s?|"
        r"g\s*b\s*e?|g\s*e?|"
        r"гбит\s*/\s*с"
        r")\b",
        text,
        re.I,
    ):
        matches.append(int(match.group(1)))
    return max(matches) if matches else None


def _extract_media(text: str) -> str | None:
    for media, pattern in _MEDIA_PATTERNS:
        if re.search(pattern, text, re.I):
            return media
    return None


def _extract_interface(text: str) -> str | None:
    for interface, pattern in _INTERFACE_PATTERNS:
        if re.search(pattern, text, re.I):
            return interface
    return None


def _speed_satisfies(actual: Any, required: Any) -> bool:
    required_gbps = _speed_to_gbps(required)
    if required_gbps is None:
        return True
    actual_gbps = _speed_to_gbps(actual)
    return actual_gbps is not None and actual_gbps >= required_gbps


def _media_satisfies(actual: Any, required: Any) -> bool:
    required_media = _normalize_media(required)
    if required_media == UNKNOWN_FACT:
        return True
    actual_media = _normalize_media(actual)
    if required_media == "QSFP":
        return actual_media.startswith("QSFP")
    if required_media == "SFP":
        return actual_media.startswith("SFP")
    return actual_media == required_media


def _normalize_speed(value: Any) -> str:
    gbps = _speed_to_gbps(value)
    return f"{gbps}GbE" if gbps is not None else UNKNOWN_FACT


def _speed_to_gbps(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        result = int(value)
        return result if result > 0 else None
    text = str(value).strip()
    match = re.search(r"\b(1|10|25|40|56|100|200|400)(?=\D|$)", text, re.I)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_media(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if not text or text == UNKNOWN_FACT:
        return UNKNOWN_FACT
    aliases = {
        "RJ-45": "RJ45",
        "BASET": "RJ45",
        "BASE-T": "RJ45",
        "10GBASE-T": "RJ45",
    }
    return aliases.get(text, text)


def _normalize_interface(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "")
    if not text or text == UNKNOWN_FACT.upper():
        return UNKNOWN_FACT
    if text.startswith("OCP"):
        return "OCP"
    if text in {"PCIE", "PCI-E"}:
        return "PCIe"
    return text


def _joined_text(text: str | None, raw_json: Any = None) -> str:
    parts = [text or ""]
    parts.extend(_raw_texts(raw_json))
    return " ".join(part for part in parts if part)


def _raw_texts(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        rows: list[str] = []
        for key, item in value.items():
            if isinstance(item, str | int | float | bool):
                rows.append(f"{key}: {item}")
            else:
                rows.extend(_raw_texts(item))
        return rows
    if isinstance(value, list):
        rows: list[str] = []
        for item in value:
            rows.extend(_raw_texts(item))
        return rows
    if isinstance(value, str):
        return [value]
    return []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
