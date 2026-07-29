from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

VERBOSE_FULL_MATRIX_MODE = "verbose_full_matrix"
COMPACT_FULL_MATRIX_MODE = "compact_full_matrix"
CHUNKED_FULL_MATRIX_MODE = "chunked_full_matrix"

COMPACT_PACKAGE_FORMAT = "compact_candidate_rows_v1"

_TOP_LEVEL_DROP_KEYS = {
    "role_candidate_pools",
}

_CANDIDATE_DROP_KEYS = {
    "raw",
    "raw_json",
    "package_json",
    "catalog_path_json",
    "product_description",
    "item_name",
    "item_name_rus",
    "product_name",
    "name",
    "content_properties",
    "ocs_content_properties",
    "matrix_distiller_evidence",
    "matrix_distiller_matched_constraints",
    "matrix_distiller_missing_facts",
    "matrix_distiller_mismatch_reasons",
    "matrix_distiller_price_stock_notes",
    "matrix_distiller_compatibility_assumptions",
    "matrix_distiller_engineer_checks",
    "eligibility_warnings",
    "fit_reasons",
    "match_warnings",
    "uncertainty_reasons",
    "evidence_summary",
    "stock_locations",
    "catalog_path",
}

_DIRECT_CANDIDATE_FIELDS = (
    "component_candidate_id",
    "role",
    "category_id",
    "item_id",
    "product_key",
    "producer",
    "part_number",
    "available_quantity",
    "price_value",
    "price_currency",
)

_NAME_FIELDS = (
    "compact_name",
    "short_name",
    "item_name",
    "name",
    "product_name",
    "item_name_rus",
)

_FACT_FIELDS = (
    "cpu_socket",
    "socket",
    "cpu_cores",
    "core_count",
    "cpu_generation",
    "cpu_frequency",
    "frequency_ghz",
    "tdp",
    "tdp_w",
    "ram_capacity_gb",
    "ram_module_capacity_gb",
    "ram_type",
    "ram_speed",
    "module_count",
    "storage_capacity_tb",
    "storage_capacity_gb",
    "raw_capacity_tb",
    "usable_capacity_tb",
    "drive_capacity_tb",
    "drive_capacity_gb",
    "drive_type",
    "drive_interface",
    "form_factor",
    "dwpd",
    "endurance",
    "tbw",
    "controller_type",
    "host_protocol",
    "host_port_count",
    "host_port_speed",
    "host_port_speed_gbps",
    "host_port_media",
    "jbod_support",
    "hot_swap_support",
    "hot_swap",
    "port_count",
    "port_speed",
    "port_speed_gbps",
    "port_media",
    "uplink_count",
    "uplink_speed",
    "uplink_speed_gbps",
    "uplink_media",
    "network_speed",
    "network_speed_gbps",
    "network_ports_count",
    "network_media",
    "network_interface",
    "transceiver_form_factor",
    "poe_supported",
    "poe_budget_w",
    "poe_standard",
    "l2_supported",
    "l3_supported",
    "stacking_supported",
    "managed_status",
    "psu_wattage",
    "wattage",
    "efficiency",
    "redundant_psu",
    "socket_count",
    "dimm_slots",
    "drive_bay_count",
    "drive_bay_form_factor",
    "cooling",
    "fan_count",
    "fans",
    "airflow",
    "onboard_ports",
    "management_port",
    "connector_type",
    "cable_type",
    "cable_length",
    "purpose",
    "warranty",
    "warranty_months",
    "condition",
    "fit_tier",
    "fit_label",
)

_FACT_ALIASES = {
    "cpu_socket": ("cpu_socket", "socket"),
    "cpu_frequency": ("cpu_frequency", "frequency_ghz"),
    "tdp": ("tdp", "tdp_w"),
    "ram_capacity_gb": ("ram_capacity_gb", "ram_module_capacity_gb"),
    "drive_capacity_tb": ("drive_capacity_tb", "storage_capacity_tb"),
    "network_speed_gbps": ("network_speed_gbps", "port_speed_gbps"),
    "network_ports_count": ("network_ports_count", "port_count"),
    "psu_wattage": ("psu_wattage", "wattage"),
    "hot_swap": ("hot_swap", "hot_swap_support"),
}

_UNKNOWN_VALUES = {
    "unknown",
    "none",
    "n/a",
    "na",
    "not specified",
    "unspecified",
    "-",
}


def compact_composer_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact full-matrix Composer package without dropping candidates."""

    compact = dict(package)
    matrix = package.get("component_candidate_matrix")
    compact_matrix: dict[str, list[dict[str, Any]]] = {}
    removed = Counter()

    if isinstance(matrix, Mapping):
        for prompt_role, rows in matrix.items():
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                continue
            compact_rows: list[dict[str, Any]] = []
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                compact_row, row_removed = compact_candidate_row(row, str(prompt_role))
                removed.update(row_removed)
                compact_rows.append(compact_row)
            if compact_rows:
                compact_matrix[str(prompt_role)] = compact_rows
    compact["component_candidate_matrix"] = compact_matrix

    for key in _TOP_LEVEL_DROP_KEYS:
        if key in compact:
            removed[key] += 1
            compact.pop(key, None)

    compact["composer_package_format"] = COMPACT_PACKAGE_FORMAT
    compact["compact_candidate_representation"] = {
        "candidate_loss_allowed": False,
        "semantic_filtering_allowed": False,
        "top_n_allowed": False,
        "unknown_fact_policy": "omitted means unknown, not satisfied",
    }
    compact["removed_verbose_fields"] = sorted(removed)
    compact["removed_verbose_field_counts"] = dict(sorted(removed.items()))
    return _strip_empty_values(compact)


def compact_candidate_row(
    row: Mapping[str, Any],
    prompt_role: str,
) -> tuple[dict[str, Any], Counter[str]]:
    removed = Counter()
    facts = _safe_mapping(row.get("extracted_facts"))
    output: dict[str, Any] = {}

    for key in _DIRECT_CANDIDATE_FIELDS:
        value = row.get(key)
        if key == "role" and _empty_or_unknown(value):
            value = prompt_role
        _put(output, key, value)

    if not output.get("component_candidate_id"):
        _put(output, "component_candidate_id", row.get("candidate_id"))
    if not output.get("role"):
        _put(output, "role", prompt_role)

    compact_name = _first_non_empty(*(row.get(key) for key in _NAME_FIELDS))
    _put(output, "compact_name", _short_text(compact_name, limit=120))

    shipment_city, location = _stock_location(row)
    _put(output, "shipment_city", shipment_city)
    _put(output, "location", location)

    for key in _FACT_FIELDS:
        aliases = _FACT_ALIASES.get(key, (key,))
        value = _first_non_empty(
            *(row.get(alias) for alias in aliases),
            *(facts.get(alias) for alias in aliases),
        )
        _put(output, key, _short_text(value, limit=160) if isinstance(value, str) else value)

    for key, value in row.items():
        if key in output:
            continue
        if key in _CANDIDATE_DROP_KEYS:
            removed[key] += 1
            continue
        if key == "extracted_facts":
            kept_fact_keys = {
                alias
                for fact_key in _FACT_FIELDS
                for alias in _FACT_ALIASES.get(fact_key, (fact_key,))
            }
            extra_facts = {
                str(fact_key): _jsonable(fact_value)
                for fact_key, fact_value in facts.items()
                if str(fact_key) not in kept_fact_keys
                and _scalar_fact(fact_value)
                and not _empty_or_unknown(fact_value)
            }
            if extra_facts:
                output["key_facts"] = dict(sorted(extra_facts.items()))
            else:
                removed[key] += 1
            continue
        if _empty_or_unknown(value):
            removed[key] += 1

    return _strip_empty_values(output), removed


def composer_package_compaction_diagnostics(
    verbose_package: Mapping[str, Any],
    compact_package: Mapping[str, Any],
    *,
    selected_package: Mapping[str, Any] | None = None,
    selected_package_mode: str,
) -> dict[str, Any]:
    selected = selected_package or compact_package
    verbose_ids_by_role = candidate_ids_by_role(verbose_package)
    compact_ids_by_role = candidate_ids_by_role(compact_package)
    verbose_ids = _id_set(verbose_ids_by_role)
    compact_ids = _id_set(compact_ids_by_role)
    verbose_counts = _candidate_count_by_role(verbose_package, verbose_ids_by_role)
    compact_counts = _candidate_count_by_role(compact_package, compact_ids_by_role)
    verbose_total = _candidate_total(verbose_package, verbose_counts)
    compact_total = _candidate_total(compact_package, compact_counts)
    package_candidate_loss = (
        verbose_ids != compact_ids
        or verbose_total != compact_total
        or verbose_counts != compact_counts
    )

    verbose_chars = json_size_chars(verbose_package)
    compact_chars = json_size_chars(compact_package)
    selected_chars = json_size_chars(selected)
    removed_fields = _removed_verbose_fields(compact_package)
    return {
        "v2_package_mode": selected_package_mode,
        "selected_package_mode": selected_package_mode,
        "verbose_context_chars": verbose_chars,
        "compact_context_chars": compact_chars,
        "selected_context_chars": selected_chars,
        "verbose_context_size": _context_size(verbose_chars),
        "compact_context_size": _context_size(compact_chars),
        "selected_context_size": _context_size(selected_chars),
        "chars_by_section": chars_by_section(selected),
        "avg_chars_per_candidate_by_role": avg_chars_per_candidate_by_role(selected),
        "removed_verbose_fields": removed_fields,
        "removed_verbose_field_counts": _safe_mapping(
            compact_package.get("removed_verbose_field_counts")
        ),
        "verbose_candidate_count_by_role": verbose_counts,
        "verbose_candidate_count_total": verbose_total,
        "compact_candidate_count_by_role": compact_counts,
        "compact_candidate_total": compact_total,
        "compact_candidate_ids_by_role": compact_ids_by_role,
        "compact_candidate_ids_hash": candidate_ids_hash(compact_ids_by_role),
        "compact_package_full_matrix_used": not package_candidate_loss,
        "package_candidate_loss": package_candidate_loss,
        "package_candidate_loss_details": {
            "missing_in_compact": sorted(verbose_ids.difference(compact_ids))[:50],
            "extra_in_compact": sorted(compact_ids.difference(verbose_ids))[:50],
            "verbose_total": verbose_total,
            "compact_total": compact_total,
            "verbose_count_by_role": verbose_counts,
            "compact_count_by_role": compact_counts,
        }
        if package_candidate_loss
        else {},
    }


def candidate_ids_by_role(package: Mapping[str, Any]) -> dict[str, list[str]]:
    matrix = package.get("component_candidate_matrix")
    result: dict[str, list[str]] = {}
    if not isinstance(matrix, Mapping):
        return result
    for role, rows in matrix.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        ids: list[str] = []
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            component_id = str(
                row.get("component_candidate_id") or row.get("candidate_id") or ""
            ).strip()
            if not component_id or component_id in seen:
                continue
            seen.add(component_id)
            ids.append(component_id)
        if ids:
            result[str(role)] = ids
    return result


def candidate_ids_hash(ids_by_role: Mapping[str, Sequence[str]]) -> str:
    pairs = [
        f"{role}:{component_id}"
        for role, ids in ids_by_role.items()
        for component_id in ids
    ]
    payload = "\n".join(sorted(pairs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chars_by_section(package: Mapping[str, Any]) -> dict[str, int]:
    diagnostics_sections = {
        "matrix_distiller_diagnostics": package.get("matrix_distiller_diagnostics"),
        "role_coverage_summary": package.get("role_coverage_summary"),
        "component_matrix_coverage_summary": package.get(
            "component_matrix_coverage_summary"
        ),
        "category_catalog_summary": package.get("category_catalog_summary"),
        "category_plan_warnings": package.get("category_plan_warnings"),
        "package_budget_warnings": package.get("package_budget_warnings"),
    }
    return {
        "source_request_chars": json_size_chars(package.get("user_request")),
        "normalized_requirements_chars": json_size_chars(
            package.get("normalized_requirements")
        ),
        "role_plan_chars": json_size_chars(package.get("role_plan")),
        "category_plan_chars": json_size_chars(package.get("category_plan")),
        "matrix_chars": json_size_chars(package.get("component_candidate_matrix")),
        "ready_candidates_chars": json_size_chars(package.get("ready_stock_candidates")),
        "rule_based_build_candidates_chars": json_size_chars(
            package.get("rule_based_build_candidates")
        ),
        "diagnostics_chars": json_size_chars(diagnostics_sections),
    }


def avg_chars_per_candidate_by_role(package: Mapping[str, Any]) -> dict[str, int]:
    matrix = package.get("component_candidate_matrix")
    result: dict[str, int] = {}
    if not isinstance(matrix, Mapping):
        return result
    for role, rows in matrix.items():
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            continue
        row_sizes = [json_size_chars(row) for row in rows if isinstance(row, Mapping)]
        if row_sizes:
            result[str(role)] = round(sum(row_sizes) / len(row_sizes))
    return result


def json_size_chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _candidate_count_by_role(
    package: Mapping[str, Any],
    ids_by_role: Mapping[str, Sequence[str]],
) -> dict[str, int]:
    counts = _safe_mapping(package.get("composer_package_candidate_count_by_role"))
    if counts:
        return {
            str(role): int(count)
            for role, count in counts.items()
            if _int_or_none(count) is not None and int(count) > 0
        }
    return {role: len(ids) for role, ids in ids_by_role.items() if ids}


def _candidate_total(package: Mapping[str, Any], counts: Mapping[str, int]) -> int:
    total = _int_or_none(package.get("composer_package_candidate_total"))
    if total is not None:
        return total
    return sum(int(count or 0) for count in counts.values())


def _removed_verbose_fields(compact_package: Mapping[str, Any]) -> list[str]:
    fields = compact_package.get("removed_verbose_fields")
    if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)):
        return [str(field) for field in fields if str(field or "").strip()]
    return []


def _id_set(ids_by_role: Mapping[str, Sequence[str]]) -> set[str]:
    return {
        str(component_id)
        for ids in ids_by_role.values()
        for component_id in ids
        if str(component_id or "").strip()
    }


def _context_size(chars: int) -> dict[str, int]:
    return {"chars": chars, "tokens_estimate": max(1, chars // 4)}


def _stock_location(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    shipment_city = row.get("shipment_city")
    location = row.get("location")
    locations = row.get("stock_locations")
    if isinstance(locations, Sequence) and not isinstance(locations, (str, bytes)):
        for item in locations:
            if not isinstance(item, Mapping):
                continue
            if _empty_or_unknown(shipment_city):
                shipment_city = item.get("shipment_city")
            if _empty_or_unknown(location):
                location = item.get("location")
            if not _empty_or_unknown(shipment_city) or not _empty_or_unknown(location):
                break
    return (
        str(shipment_city).strip() if not _empty_or_unknown(shipment_city) else None,
        str(location).strip() if not _empty_or_unknown(location) else None,
    )


def _put(output: dict[str, Any], key: str, value: Any) -> None:
    if _empty_or_unknown(value):
        return
    output[key] = _jsonable(value)


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if not _empty_or_unknown(value):
            return value
    return None


def _short_text(value: Any, *, limit: int) -> Any:
    if not isinstance(value, str):
        return value
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _strip_empty_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            stripped = _strip_empty_values(item)
            if _empty_or_unknown(stripped):
                continue
            result[str(key)] = stripped
        return result
    if isinstance(value, list):
        return [
            stripped
            for item in value
            if not _empty_or_unknown(stripped := _strip_empty_values(item))
        ]
    return _jsonable(value)


def _empty_or_unknown(value: Any) -> bool:
    if value is None:
        return True
    if value is False:
        return False
    if isinstance(value, (int, float, Decimal)):
        return False
    if isinstance(value, str):
        text = value.strip()
        return not text or text.casefold() in _UNKNOWN_VALUES
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return not value
    return False


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _scalar_fact(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, Decimal)) or value is None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
