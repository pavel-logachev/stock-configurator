from __future__ import annotations

import json

from app.llm.composer_package_compactor import (
    COMPACT_FULL_MATRIX_MODE,
    compact_composer_package,
    composer_package_compaction_diagnostics,
)


def test_compact_package_preserves_all_candidate_ids_counts_and_roles() -> None:
    verbose = _verbose_package(candidate_count=30)

    compact = compact_composer_package(verbose)
    diagnostics = composer_package_compaction_diagnostics(
        verbose,
        compact,
        selected_package=compact,
        selected_package_mode=COMPACT_FULL_MATRIX_MODE,
    )

    assert diagnostics["package_candidate_loss"] is False
    assert diagnostics["verbose_candidate_count_total"] == 30
    assert diagnostics["compact_candidate_total"] == 30
    assert diagnostics["verbose_candidate_count_by_role"] == {
        "cpu": 10,
        "ram": 10,
        "server_platform": 10,
    }
    assert diagnostics["compact_candidate_count_by_role"] == {
        "cpu": 10,
        "ram": 10,
        "server_platform": 10,
    }
    assert {
        row["component_candidate_id"]
        for rows in compact["component_candidate_matrix"].values()
        for row in rows
    } == {
        row["component_candidate_id"]
        for rows in verbose["component_candidate_matrix"].values()
        for row in rows
    }


def test_compact_package_is_smaller_and_removes_verbose_noise() -> None:
    verbose = _verbose_package(candidate_count=60)

    compact = compact_composer_package(verbose)
    verbose_chars = len(json.dumps(verbose, ensure_ascii=False, sort_keys=True))
    compact_chars = len(json.dumps(compact, ensure_ascii=False, sort_keys=True))

    assert compact_chars < verbose_chars * 0.55
    assert "raw_json" in compact["removed_verbose_fields"]
    assert "package_json" in compact["removed_verbose_fields"]
    assert "role_candidate_pools" in compact["removed_verbose_fields"]
    matrix_payload = json.dumps(
        compact["component_candidate_matrix"],
        ensure_ascii=False,
    )
    assert "raw_json" not in matrix_payload
    assert "package_json" not in matrix_payload
    assert "empty_evidence" not in matrix_payload
    assert "unknown" not in matrix_payload.casefold()


def test_compact_package_keeps_required_candidate_fields() -> None:
    compact = compact_composer_package(_verbose_package(candidate_count=3))

    row = compact["component_candidate_matrix"]["cpu"][0]

    assert row["component_candidate_id"] == "cpu-0"
    assert row["role"] == "cpu"
    assert row["category_id"] == "cat-cpu"
    assert row["item_id"] == "cpu-item-0"
    assert row["producer"] == "TestVendor"
    assert row["part_number"] == "CPU-0"
    assert row["compact_name"] == "CPU verbose product name 0"
    assert row["available_quantity"] == 8
    assert row["price_value"] == "100"
    assert row["price_currency"] == "USD"
    assert row["cpu_cores"] == 24
    assert row["cpu_socket"] == "LGA4677"


def test_compact_package_does_not_top_n_trim() -> None:
    verbose = _verbose_package(candidate_count=420)

    compact = compact_composer_package(verbose)

    assert sum(
        len(rows) for rows in compact["component_candidate_matrix"].values()
    ) == 420


def _verbose_package(*, candidate_count: int) -> dict[str, object]:
    roles = {
        "cpu": "cpu",
        "ram": "ram",
        "platform": "server_platform",
    }
    matrix: dict[str, list[dict[str, object]]] = {role: [] for role in roles}
    for index in range(candidate_count):
        role = ("cpu", "ram", "platform")[index % 3]
        role_index = len(matrix[role])
        matrix[role].append(_verbose_row(role, role_index))
    counts = {internal: len(matrix[role]) for role, internal in roles.items()}
    return {
        "original_request_text": "Need a server.",
        "user_request": "Need a server.",
        "product_group": "server",
        "normalized_requirements": {
            "pipeline_version": "v2_composer_first",
            "composer_first": True,
        },
        "component_candidate_matrix": matrix,
        "composer_package_candidate_count_by_role": counts,
        "composer_package_candidate_total": candidate_count,
        "role_candidate_pools": {
            role: {
                "candidate_ids": [
                    row["component_candidate_id"] for row in rows
                ]
            }
            for role, rows in matrix.items()
        },
    }


def _verbose_row(role: str, index: int) -> dict[str, object]:
    prefix = "platform" if role == "platform" else role
    facts = {
        "cpu_socket": "LGA4677" if role in {"cpu", "platform"} else "unknown",
        "cpu_cores": 24 if role == "cpu" else None,
        "ram_capacity_gb": 64 if role == "ram" else None,
        "ram_type": "DDR5" if role in {"ram", "platform"} else None,
        "raw": {"large": "x" * 100},
        "debug": {"why": "not for prompt"},
    }
    return {
        "candidate_id": f"{prefix}-{index}",
        "component_candidate_id": f"{prefix}-{index}",
        "role": role,
        "category_id": f"cat-{prefix}",
        "item_id": f"{prefix}-item-{index}",
        "product_key": f"ocs:{prefix}-item-{index}",
        "producer": "TestVendor",
        "part_number": f"{prefix.upper()}-{index}",
        "item_name": f"{prefix.upper()} verbose product name {index}",
        "product_name": f"{prefix.upper()} verbose product name {index}",
        "product_description": "long description " * 50,
        "catalog_path_json": [{"category_id": f"cat-{prefix}", "name": prefix}],
        "catalog_path": [{"category_id": f"cat-{prefix}", "name": prefix}],
        "package_json": {"large": "payload " * 100},
        "raw_json": {"source": "payload " * 100},
        "price_value": "100",
        "price_currency": "USD",
        "available_quantity": 8,
        "stock_locations": [
            {"shipment_city": "Moscow", "location": "main", "quantity_value": 8}
        ],
        "extracted_facts": facts,
        "fit_tier": "possible_fit",
        "eligibility_warnings": [],
        "empty_evidence": [],
        "unknown_spam": "unknown",
    }
