from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def platform_power_bundle_satisfies(
    text: str | None,
    *,
    required_psu_count: int | None = None,
    raw_json: Any = None,
) -> bool:
    required = required_psu_count or 2
    if required <= 1:
        return True
    if required > 2:
        return False
    source = _joined_text(text, raw_json)
    if not source.strip():
        return False
    return bool(
        re.search(r"\b1\s*\+\s*1\b", source, re.IGNORECASE)
        or re.search(r"\b(?:redundant|redundancy|hot\s*swap)\b", source, re.IGNORECASE)
        or re.search(
            r"\b(?:crps|pws)\b.{0,40}\b(?:2\s*x|1\s*\+\s*1)\b",
            source,
            re.IGNORECASE,
        )
        or re.search(
            r"\b2\s*x\s*\d{3,4}\s*w\b.{0,40}\b(?:psu|power|pws|crps)?\b",
            source,
            re.IGNORECASE,
        )
        or re.search(r"\b2\s*x\s*(?:psu|power\s*supply|pws|crps)\b", source, re.IGNORECASE)
        or re.search(
            r"\b2\s+(?:"
            r"\u0431\u043b\u043e\u043a\w*\s+\u043f\u0438\u0442\u0430\u043d\w*|"
            r"\u0431\u043f"
            r")\b",
            source,
            re.IGNORECASE,
        )
    )


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
