from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

GENERIC_CONSTRAINT_OPERATORS = {
    "=",
    "~=",
    "contains",
    "one_of",
    ">=",
    "<=",
    "range",
    "compatible_with",
    "quantity_at_least",
}


def constraints_by_role_from_role_plan(
    role_plan: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Adapt existing capability rows to generic distiller constraints."""

    constraints: dict[str, list[dict[str, Any]]] = {}
    for capability in _capabilities(role_plan.get("required_capabilities")):
        _append_capability_constraints(
            constraints,
            capability,
            default_hardness="hard",
        )
    for capability in _capabilities(role_plan.get("optional_capabilities")):
        _append_capability_constraints(
            constraints,
            capability,
            default_hardness="optional",
        )
    return constraints


def flat_constraints_from_role_plan(role_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for rows in constraints_by_role_from_role_plan(role_plan).values():
        result.extend(rows)
    return result


def _append_capability_constraints(
    constraints: dict[str, list[dict[str, Any]]],
    capability: Mapping[str, Any],
    *,
    default_hardness: str,
) -> None:
    role = str(capability.get("role") or "").strip()
    if not role:
        return

    hardness = _hardness(capability, default_hardness=default_hardness)
    source_text = str(capability.get("source_text") or "").strip()
    capability_id = str(capability.get("capability_id") or "").strip()
    parsed = capability.get("parsed_requirements")
    parsed_requirements = dict(parsed) if isinstance(parsed, Mapping) else {}

    if capability_id:
        constraints.setdefault(role, []).append(
            {
                "role": role,
                "key": "capability_id",
                "operator": "contains",
                "value": capability_id,
                "unit": None,
                "hardness": hardness,
                "machine_verifiable": False,
                "source_text": source_text,
            }
        )

    for key, value in parsed_requirements.items():
        normalized_key = str(key).strip()
        if not normalized_key or value in (None, ""):
            continue
        constraints.setdefault(role, []).append(
            {
                "role": role,
                "key": normalized_key,
                "operator": _operator_for(normalized_key, value),
                "value": _jsonable_constraint_value(value),
                "unit": _unit_for_key(normalized_key),
                "hardness": hardness,
                "machine_verifiable": _machine_verifiable(normalized_key, value),
                "source_text": source_text,
            }
        )


def _capabilities(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _hardness(capability: Mapping[str, Any], *, default_hardness: str) -> str:
    if default_hardness == "optional":
        return "optional"
    hard = capability.get("hard")
    if hard is False:
        return "optional"
    text = str(capability.get("hardness") or "").strip().lower()
    if text in {"hard", "optional"}:
        return text
    return "hard"


def _operator_for(key: str, value: Any) -> str:
    lowered = key.casefold()
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "contains"
    if _looks_like_range(value):
        return "range"
    if any(token in lowered for token in ("compatible", "support", "supported")):
        return "compatible_with"
    if any(token in lowered for token in ("quantity", "qty")):
        return "quantity_at_least"
    if _is_number_like(value) and any(
        token in lowered
        for token in ("min", "minimum", "at_least", "count", "ports", "slots", "sockets")
    ):
        return ">="
    if _is_number_like(value) and any(token in lowered for token in ("max", "maximum")):
        return "<="
    if isinstance(value, str) and any(token in lowered for token in ("hint", "model", "family")):
        return "~="
    return "="


def _unit_for_key(key: str) -> str | None:
    lowered = key.casefold()
    suffix_units = {
        "_gb": "GB",
        "_tb": "TB",
        "_w": "W",
        "_mhz": "MHz",
        "_ghz": "GHz",
        "_gbps": "Gbps",
        "_ports": "ports",
        "_slots": "slots",
        "_sockets": "sockets",
    }
    for suffix, unit in suffix_units.items():
        if lowered.endswith(suffix):
            return unit
    return None


def _machine_verifiable(key: str, value: Any) -> bool:
    lowered = key.casefold()
    if any(token in lowered for token in ("hint", "note", "text", "description")):
        return False
    if isinstance(value, bool | int | float):
        return True
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _looks_like_range(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = {str(key).casefold() for key in value}
    return bool(keys & {"min", "minimum", "from"} and keys & {"max", "maximum", "to"})


def _is_number_like(value: Any) -> bool:
    if isinstance(value, int | float):
        return True
    try:
        float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return False
    return True


def _jsonable_constraint_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_constraint_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_jsonable_constraint_value(item) for item in value]
    return value
