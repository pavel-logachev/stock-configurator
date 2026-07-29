from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ocs_anchor_categories.yaml"

ANCHOR_GROUPS = {"server", "network", "storage", "support_license", "accessory"}
REVIEW_STATUSES = {"candidate", "approved", "rejected"}


@dataclass(frozen=True)
class OcsAnchorCategory:
    group: str
    role: str
    category_id: str
    comment: str
    enabled_default: bool
    review_status: str
    category_name: str = ""
    category_path: str = ""
    allowed_roles: tuple[str, ...] = ()
    category_kind: str = "mixed"
    base_device_allowed: bool = False
    notes: str = ""

    @property
    def enable_allowed(self) -> bool:
        return self.enabled_default or self.review_status == "approved"


@lru_cache
def load_ocs_anchor_categories(
    path: str | Path = CONFIG_PATH,
) -> tuple[OcsAnchorCategory, ...]:
    config_path = Path(path)
    if not config_path.exists():
        return ()
    rows = _parse_minimal_yaml(config_path.read_text(encoding="utf-8"))
    return tuple(_anchor_from_row(row) for row in rows)


def anchor_categories_for_group(
    group: str,
    *,
    path: str | Path = CONFIG_PATH,
) -> tuple[OcsAnchorCategory, ...]:
    group = str(group or "").strip()
    anchors = load_ocs_anchor_categories(path)
    if group == "all-approved":
        return tuple(anchor for anchor in anchors if anchor.review_status == "approved")
    if group not in ANCHOR_GROUPS:
        raise ValueError(f"Unsupported OCS anchor group: {group}")
    return tuple(
        anchor
        for anchor in anchors
        if anchor.group == group and anchor.enable_allowed
    )


def _anchor_from_row(row: dict[str, Any]) -> OcsAnchorCategory:
    if "group" not in row and "product_group" in row:
        row["group"] = row["product_group"]
    if "role" not in row:
        row["role"] = row.get("default_role") or row.get("suggested_role") or ""
    required = {"group", "category_id", "review_status"}
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"OCS anchor category row is missing: {', '.join(missing)}")
    group = str(row["group"]).strip()
    review_status = str(row["review_status"]).strip()
    if group not in ANCHOR_GROUPS:
        raise ValueError(f"Unsupported OCS anchor group: {group}")
    if review_status not in REVIEW_STATUSES:
        raise ValueError(f"Unsupported OCS anchor review_status: {review_status}")
    allowed_roles = _yaml_list(row.get("allowed_roles"))
    category_kind = str(row.get("category_kind") or "mixed").strip()
    base_device_allowed = (
        _as_bool(row["base_device_allowed"])
        if "base_device_allowed" in row
        else category_kind == "base_device"
    )
    return OcsAnchorCategory(
        group=group,
        role=str(row.get("role") or "").strip(),
        category_id=str(row["category_id"]).strip(),
        comment=str(row.get("comment") or row.get("notes") or "").strip(),
        enabled_default=_as_bool(row.get("enabled_default", review_status == "approved")),
        review_status=review_status,
        category_name=str(row.get("category_name") or row.get("name") or "").strip(),
        category_path=str(row.get("category_path") or row.get("path") or "").strip(),
        allowed_roles=allowed_roles,
        category_kind=category_kind,
        base_device_allowed=base_device_allowed,
        notes=str(row.get("notes") or "").strip(),
    )


def _parse_minimal_yaml(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_rows = False

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped in {"categories:", "anchors:"}:
            in_rows = True
            continue
        if not in_rows:
            continue
        if stripped.startswith("- "):
            if current:
                rows.append(current)
            current = {}
            remainder = stripped[2:].strip()
            if remainder:
                _set_yaml_key_value(current, remainder)
            continue
        if current is None:
            raise ValueError("OCS anchor profile has a property before a list item.")
        _set_yaml_key_value(current, stripped)

    if current:
        rows.append(current)
    return rows


def _set_yaml_key_value(target: dict[str, Any], line: str) -> None:
    if ":" not in line:
        raise ValueError(f"Unsupported OCS anchor profile line: {line}")
    key, value = line.split(":", 1)
    target[key.strip()] = _yaml_scalar(value.strip())


def _yaml_scalar(value: str) -> str | bool:
    if value.casefold() == "true":
        return True
    if value.casefold() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return value
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _yaml_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list | tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value or "").strip()
    if not text:
        return ()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return tuple(
        item.strip().strip("'\"")
        for item in text.split(",")
        if item.strip().strip("'\"")
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y", "да"}:
        return True
    if normalized in {"false", "0", "no", "n", "нет"}:
        return False
    raise ValueError(f"Unsupported boolean value in OCS anchor profile: {value!r}")
