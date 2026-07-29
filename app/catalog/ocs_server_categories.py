from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "ocs_server_categories.yaml"


@dataclass(frozen=True)
class OcsServerCategory:
    category_id: str
    name_ru: str
    role: str
    enabled_by_default: bool


@lru_cache
def load_server_category_profile(path: str | Path = CONFIG_PATH) -> tuple[OcsServerCategory, ...]:
    rows = _parse_minimal_yaml(Path(path).read_text(encoding="utf-8"))
    return tuple(_category_from_row(row) for row in rows)


def enabled_server_categories() -> tuple[OcsServerCategory, ...]:
    return tuple(
        category
        for category in load_server_category_profile()
        if category.enabled_by_default
    )


def server_category_by_id() -> dict[str, OcsServerCategory]:
    return {category.category_id: category for category in load_server_category_profile()}


def server_category_role(category_id: str | None) -> str | None:
    if category_id is None:
        return None
    category = server_category_by_id().get(category_id)
    return category.role if category is not None else None


def server_category_role_label(role: str | None) -> str:
    labels = {
        "ready_server": "Готовые серверы",
        "server_platform": "Серверные платформы",
        "cpu": "CPU / серверные процессоры",
        "ram": "Серверная оперативная память",
        "ssd": "Серверные SSD",
        "hdd": "Серверные HDD",
        "storage_controller": "Серверные контроллеры",
        "network_adapter": "Серверные сетевые адаптеры",
    }
    if role is None:
        return ""
    return labels.get(role, role)


def server_category_ids_for_role(role: str) -> tuple[str, ...]:
    return tuple(
        category.category_id
        for category in load_server_category_profile()
        if category.role == role
    )


def _category_from_row(row: dict[str, Any]) -> OcsServerCategory:
    if "name" in row and "name_ru" not in row:
        row = {**row, "name_ru": row["name"]}

    required = {"category_id", "name_ru", "role", "enabled_by_default"}
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"Server OCS category profile row is missing: {', '.join(missing)}")

    return OcsServerCategory(
        category_id=str(row["category_id"]).strip(),
        name_ru=str(row["name_ru"]).strip(),
        role=str(row["role"]).strip(),
        enabled_by_default=_as_bool(row["enabled_by_default"]),
    )


def _parse_minimal_yaml(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_categories = False

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped == "categories:":
            in_categories = True
            continue
        if not in_categories:
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
            raise ValueError("Server OCS category profile has a property before a list item.")
        _set_yaml_key_value(current, stripped)

    if current:
        rows.append(current)
    if not rows:
        raise ValueError("Server OCS category profile does not contain categories.")
    return rows


def _set_yaml_key_value(target: dict[str, Any], line: str) -> None:
    if ":" not in line:
        raise ValueError(f"Unsupported server category profile line: {line}")
    key, value = line.split(":", 1)
    target[key.strip()] = _yaml_scalar(value.strip())


def _yaml_scalar(value: str) -> str | bool:
    if value.casefold() == "true":
        return True
    if value.casefold() == "false":
        return False
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y", "да"}:
        return True
    if normalized in {"false", "0", "no", "n", "нет"}:
        return False
    raise ValueError(f"Unsupported boolean value in server category profile: {value!r}")
