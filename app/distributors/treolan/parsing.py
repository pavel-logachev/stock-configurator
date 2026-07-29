from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree


@dataclass(frozen=True)
class TreolanCategoryNode:
    category_id: str
    name: str
    parent_category_id: str | None
    level: int
    path_json: list[dict[str, str]]
    raw_json: dict[str, Any]


@dataclass(frozen=True)
class TreolanPositionNode:
    position: dict[str, Any]
    category_id: str | None
    category_path_json: list[dict[str, str]]
    category_name: str | None = None

    def to_wrapper(self, *, fallback_category_id: str | None = None) -> dict[str, Any]:
        return {
            "position": self.position,
            "category_id": self.category_id or fallback_category_id,
            "category_name": self.category_name,
            "category_path_json": self.category_path_json,
        }


def parse_treolan_xml(payload: str) -> ElementTree.Element:
    text = str(payload or "").strip()
    if not text:
        raise ValueError("Treolan XML payload is empty")

    root = _from_string(text)
    result_text = _soap_result_text(root)
    if result_text is None:
        return root

    result_text = result_text.strip()
    if not result_text:
        raise ValueError("Treolan SOAP Result is empty")
    return _from_string(result_text)


def flatten_treolan_categories(payload: str) -> list[TreolanCategoryNode]:
    root = parse_treolan_xml(payload)
    rows: list[TreolanCategoryNode] = []
    seen: set[str] = set()

    def walk(
        element: ElementTree.Element,
        *,
        parent_category_id: str | None,
        level: int,
        path_json: list[dict[str, str]],
    ) -> None:
        tag = _local_name(element.tag).casefold()
        if tag != "category":
            for child in list(element):
                walk(
                    child,
                    parent_category_id=parent_category_id,
                    level=level,
                    path_json=path_json,
                )
            return

        attrs = _attrs(element)
        category_id = _category_id(attrs)
        name = _category_name(attrs, fallback=category_id)
        explicit_parent = _first_string(
            attrs,
            (
                "parentid",
                "parent-id",
                "parent_id",
                "parent",
                "pid",
                "parentCategoryId",
            ),
        )
        effective_parent_id = explicit_parent or parent_category_id
        current_path = [*path_json, {"category_id": category_id, "name": name}]
        if category_id not in seen:
            rows.append(
                TreolanCategoryNode(
                    category_id=category_id,
                    name=name,
                    parent_category_id=effective_parent_id,
                    level=_to_int(attrs.get("level")) or level,
                    path_json=current_path,
                    raw_json={
                        "attributes": _jsonable_dict(attrs),
                        "child_category_count": _child_count(element, "category"),
                        "position_count": _child_count(element, "position"),
                    },
                )
            )
            seen.add(category_id)

        for child in list(element):
            walk(
                child,
                parent_category_id=category_id,
                level=level + 1,
                path_json=current_path,
            )

    walk(root, parent_category_id=None, level=0, path_json=[])
    return rows


def extract_treolan_positions(
    payload: str,
    *,
    fallback_category_id: str | None = None,
) -> list[TreolanPositionNode]:
    root = parse_treolan_xml(payload)
    positions: list[TreolanPositionNode] = []

    def walk(
        element: ElementTree.Element,
        *,
        current_category_id: str | None,
        current_category_name: str | None,
        path_json: list[dict[str, str]],
    ) -> None:
        tag = _local_name(element.tag).casefold()
        category_id = current_category_id
        category_name = current_category_name
        category_path = path_json

        if tag == "category":
            attrs = _attrs(element)
            category_id = _category_id(attrs)
            category_name = _category_name(attrs, fallback=category_id)
            category_path = [
                *path_json,
                {"category_id": category_id, "name": category_name},
            ]
        elif tag == "position":
            attrs = _attrs(element)
            positions.append(
                TreolanPositionNode(
                    position=_jsonable_dict(attrs),
                    category_id=category_id or fallback_category_id,
                    category_name=category_name,
                    category_path_json=category_path,
                )
            )
            return

        for child in list(element):
            walk(
                child,
                current_category_id=category_id,
                current_category_name=category_name,
                path_json=category_path,
            )

    walk(
        root,
        current_category_id=fallback_category_id,
        current_category_name=None,
        path_json=[],
    )
    return positions


def _from_string(value: str) -> ElementTree.Element:
    text = _strip_invalid_xml_chars(value.strip())
    attempts = [text]
    unescaped = html.unescape(text)
    if unescaped != text:
        attempts.append(unescaped)

    last_error: ElementTree.ParseError | None = None
    for attempt in attempts:
        try:
            return ElementTree.fromstring(attempt)
        except ElementTree.ParseError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("Treolan XML payload is empty")


def _soap_result_text(root: ElementTree.Element) -> str | None:
    for element in root.iter():
        if _local_name(element.tag) == "Result":
            return element.text or ""
    return None


def _attrs(element: ElementTree.Element) -> dict[str, str]:
    attrs = {_local_name(key): value for key, value in element.attrib.items()}
    if element.text and element.text.strip() and "text" not in attrs:
        attrs["text"] = element.text.strip()
    return attrs


def _category_id(attrs: Mapping[str, Any]) -> str:
    category_id = _first_string(attrs, ("id", "category", "category_id", "categoryId", "code"))
    if category_id:
        return category_id
    raise ValueError("Treolan category node does not contain id/category")


def _category_name(attrs: Mapping[str, Any], *, fallback: str) -> str:
    name = _first_string(attrs, ("name", "title", "label", "descr", "description"))
    return name or fallback


def _child_count(element: ElementTree.Element, tag_name: str) -> int:
    return sum(1 for child in list(element) if _local_name(child.tag).casefold() == tag_name)


def _first_string(node: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key not in node:
            continue
        text = str(node[key] or "").strip()
        if text:
            return text
    return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"-?\d+", str(value).replace(" ", ""))
    if match is None:
        return None
    return int(match.group(0))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _strip_invalid_xml_chars(value: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)


def _jsonable_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))
