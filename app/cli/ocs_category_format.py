from __future__ import annotations

from typing import Any

from app.db.models import DistributorCategory


def format_path(path_json: Any) -> str:
    if not isinstance(path_json, list):
        return str(path_json)

    parts: list[str] = []
    for item in path_json:
        if isinstance(item, dict):
            parts.append(str(item.get("name") or item.get("category_id") or ""))
        else:
            parts.append(str(item))
    return " > ".join(part for part in parts if part)


def print_category_summary(
    action: str,
    category: DistributorCategory,
    *,
    comment: str | None = None,
) -> None:
    print(f"{action} {category.category_id}")
    print(f"name: {category.name}")
    print(f"path: {format_path(category.path_json)}")
    print(f"enabled_for_sync={str(category.enabled_for_sync).lower()}")
    if comment:
        print(f"comment: {comment}")
