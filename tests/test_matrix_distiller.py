from __future__ import annotations

import json
import time
from decimal import Decimal
from typing import Any

from app.llm.base import LlmClientError
from app.llm.configuration_composer import build_llm_configurator_package
from app.llm.matrix_distiller import (
    compact_candidate_for_distiller,
    distill_component_candidate_matrix,
    distill_role_candidates,
    split_candidate_chunks,
)
from app.planning.generic_constraints import constraints_by_role_from_role_plan


class _FakeDistillerClient:
    def __init__(self, response_factory: Any) -> None:
        self.response_factory = response_factory
        self.packages: list[dict[str, Any]] = []

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert "AI Matrix Distiller" in system_prompt
        package = json.loads(user_prompt)
        self.packages.append(package)
        return self.response_factory(package)


def test_generic_constraints_adapter_keeps_role_key_operator_and_source() -> None:
    constraints = constraints_by_role_from_role_plan(
        {
            "required_capabilities": [
                {
                    "role": "network_adapter",
                    "capability_id": "network_adapter.10gbe.sfpplus",
                    "hard": True,
                    "source_text": "Intel X710-DA2 2x10GbE SFP+",
                    "parsed_requirements": {
                        "min_ports_per_server": 2,
                        "speed": "10GbE",
                        "media": "SFP+",
                    },
                }
            ],
            "optional_capabilities": [
                {
                    "role": "power_supply",
                    "capability_id": "power_supply.redundant",
                    "parsed_requirements": {"redundant_supported": True},
                }
            ],
        }
    )

    network = constraints["network_adapter"]
    power = constraints["power_supply"]

    assert any(row["key"] == "min_ports_per_server" and row["operator"] == ">=" for row in network)
    assert any(row["key"] == "capability_id" and row["operator"] == "contains" for row in network)
    assert all(row["hardness"] == "hard" for row in network)
    assert power[0]["hardness"] == "optional"
    assert network[0]["source_text"] == "Intel X710-DA2 2x10GbE SFP+"


def test_distiller_rejects_unknown_ids_and_preserves_db_price_stock() -> None:
    rows = [
        _candidate("cpu-good", price=Decimal("100"), stock=4),
        _candidate("cpu-bad", price=Decimal("50"), stock=9),
    ]

    def response(package: dict[str, Any]) -> dict[str, Any]:
        ids = [candidate["component_candidate_id"] for candidate in package["candidates"]]
        return {
            "role": package["role"],
            "evaluated_candidates": [
                {
                    "component_candidate_id": ids[0],
                    "fit_tier": "strong_fit",
                    "facts": {"llm_price_value": "1", "cores": 32},
                    "matched_constraints": ["cores"],
                    "missing_facts": [],
                    "mismatch_reasons": [],
                    "evidence": "name says 32 cores",
                    "confidence": "high",
                },
                {
                    "component_candidate_id": "invented-id",
                    "fit_tier": "strong_fit",
                    "facts": {},
                    "matched_constraints": [],
                    "missing_facts": [],
                    "mismatch_reasons": [],
                    "evidence": "",
                    "confidence": "high",
                },
                {
                    "component_candidate_id": ids[1],
                    "fit_tier": "wrong_role",
                    "facts": {},
                    "matched_constraints": [],
                    "missing_facts": [],
                    "mismatch_reasons": ["not a CPU"],
                    "evidence": "diagnostic only",
                    "confidence": "medium",
                },
            ],
            "shortlist_candidate_ids": {
                "strong_fit": [ids[0], "invented-id"],
                "possible_fit": [],
                "fallback_unknown": [],
            },
        }

    result = distill_role_candidates(
        product_group="server",
        role="cpu",
        constraints=[],
        candidate_rows=rows,
        llm_client=_FakeDistillerClient(response),
        role_limit=4,
        chunk_max_chars=10000,
    )

    assert [row["component_candidate_id"] for row in result.rows] == ["cpu-good"]
    assert result.rows[0]["price_value"] == "100"
    assert result.rows[0]["available_quantity"] == 4
    assert result.rows[0]["matrix_distiller_facts"]["cores"] == 32
    assert result.diagnostics["unknown_component_candidate_ids"] == ["invented-id"]


def test_distiller_candidate_payload_does_not_require_ocs_content_properties() -> None:
    compact = compact_candidate_for_distiller(
        _candidate("cpu-product-only", price=Decimal("100"), stock=4)
        | {
            "item_name_rus": "Intel Xeon product name",
            "product_name": "Intel CPU Product",
            "product_description": "CPU catalog description from DB",
            "catalog_path": [{"name": "Processors"}, {"name": "Server CPUs"}],
        }
    )

    assert compact["component_candidate_id"] == "cpu-product-only"
    assert compact["role"] == "cpu"
    assert compact["item_name"] == "Server component cpu-product-only"
    assert compact["item_name_rus"] == "Intel Xeon product name"
    assert compact["product_name"] == "Intel CPU Product"
    assert compact["product_description"] == "CPU catalog description from DB"
    assert compact["catalog_path"] == ["Processors", "Server CPUs"]
    assert "content_properties" not in compact


def test_distiller_chunks_and_applies_role_limits_deterministically() -> None:
    rows = [
        _candidate(f"ssd-{index:02d}", price=Decimal(index + 1), stock=10)
        for index in range(20)
    ]
    compact = [
        {
            "component_candidate_id": row["component_candidate_id"],
            "item_name": row["name"] * 4,
        }
        for row in rows
    ]
    chunks = split_candidate_chunks(
        product_group="server",
        role="ssd",
        constraints=[],
        candidates=compact,
        max_chars=700,
    )

    assert len(chunks) > 1

    def response(package: dict[str, Any]) -> dict[str, Any]:
        evaluated = []
        for candidate in package["candidates"]:
            index = int(candidate["component_candidate_id"].split("-")[1])
            evaluated.append(
                {
                    "component_candidate_id": candidate["component_candidate_id"],
                    "fit_tier": "strong_fit" if index % 2 == 0 else "fallback_unknown",
                    "facts": {},
                    "matched_constraints": [],
                    "missing_facts": [],
                    "mismatch_reasons": [],
                    "evidence": "",
                    "confidence": "medium",
                }
            )
        return {
            "role": package["role"],
            "evaluated_candidates": evaluated,
            "shortlist_candidate_ids": {
                "strong_fit": [],
                "possible_fit": [],
                "fallback_unknown": [],
            },
        }

    first_client = _FakeDistillerClient(response)
    first = distill_role_candidates(
        product_group="server",
        role="ssd",
        constraints=[],
        candidate_rows=rows,
        llm_client=first_client,
        role_limit=5,
        chunk_max_chars=700,
    )
    second = distill_role_candidates(
        product_group="server",
        role="ssd",
        constraints=[],
        candidate_rows=list(reversed(rows)),
        llm_client=_FakeDistillerClient(response),
        role_limit=5,
        chunk_max_chars=700,
    )

    assert len(first.rows) == 5
    evaluator_packages = [
        package for package in first_client.packages if "evaluated_candidates" not in package
    ]
    reducer_packages = [
        package for package in first_client.packages if "evaluated_candidates" in package
    ]
    assert sum(len(package["candidates"]) for package in evaluator_packages) == 20
    assert reducer_packages
    assert reducer_packages[-1]["evaluated_count"] == 20
    assert first.diagnostics["evaluated_count"] == 20
    assert first.diagnostics["role_reducer_summary"]["selected_count"] == 5
    assert all(row["fit_tier"] == "strong_fit" for row in first.rows)
    assert [row["component_candidate_id"] for row in first.rows] == [
        "ssd-00",
        "ssd-02",
        "ssd-04",
        "ssd-06",
        "ssd-08",
    ]
    assert [row["component_candidate_id"] for row in second.rows] == [
        "ssd-18",
        "ssd-16",
        "ssd-14",
        "ssd-12",
        "ssd-10",
    ]


def test_distiller_retries_failed_chunk_and_continues_other_chunks() -> None:
    rows = [
        _candidate(f"ssd-{index:02d}", price=Decimal(index + 1), stock=10)
        | {"product_description": "x" * 1200}
        for index in range(6)
    ]

    class PartiallyFailingClient:
        def __init__(self) -> None:
            self.evaluator_calls = 0
            self.reducer_calls = 0

        def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
            assert "AI Matrix Distiller" in system_prompt
            package = json.loads(user_prompt)
            if "evaluated_candidates" in package:
                self.reducer_calls += 1
                return {
                    "role": package["role"],
                    "selected_candidate_ids": [
                        row["component_candidate_id"]
                        for row in package["evaluated_candidates"]
                    ],
                    "role_summary": "Reduced after one failed chunk.",
                    "no_viable_reason": None,
                    "rejected_summary": [],
                }

            self.evaluator_calls += 1
            candidate_ids = [
                candidate["component_candidate_id"]
                for candidate in package["candidates"]
            ]
            if "ssd-02" in candidate_ids:
                raise LlmClientError("temporary role evaluator failure", status_code=502)
            return {
                "role": package["role"],
                "evaluated_candidates": [
                    {
                        "component_candidate_id": component_id,
                        "fit_tier": "strong_fit",
                        "facts": {},
                        "matched_constraints": [],
                        "missing_facts": [],
                        "mismatch_reasons": [],
                        "evidence": "",
                        "confidence": "medium",
                    }
                    for component_id in candidate_ids
                ],
                "shortlist_candidate_ids": {
                    "strong_fit": [],
                    "possible_fit": [],
                    "fallback_unknown": [],
                },
            }

    client = PartiallyFailingClient()
    result = distill_component_candidate_matrix(
        product_group="server",
        component_candidate_matrix={
            "product_group": "server",
            "ssd_candidates": rows,
        },
        constraints_by_role={},
        llm_client=client,
        chunk_max_chars=1000,
    )

    diagnostics = result.diagnostics
    role_diagnostics = diagnostics["role_diagnostics"]["ssd"]

    assert diagnostics["full_matrix_evaluation_used"] is True
    assert diagnostics["evaluated_candidate_count_by_role"]["ssd"] < 6
    assert diagnostics["full_matrix_failed_chunks"]
    assert diagnostics["full_matrix_failed_chunks"][0]["role"] == "ssd"
    assert diagnostics["full_matrix_failed_chunks"][0]["http_status"] == 502
    assert role_diagnostics["retried_chunk_count"] == 1
    assert role_diagnostics["failed_chunk_count"] == 1
    assert role_diagnostics["failed_chunks"][0]["attempt_count"] == 2
    assert client.reducer_calls == 1
    assert result.component_candidate_matrix["ssd_candidates"]


def test_distiller_times_out_chunk_retries_once_and_continues_role() -> None:
    progress_events: list[tuple[str, dict[str, Any]]] = []

    class HangingPlatformClient:
        def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
            assert "AI Matrix Distiller" in system_prompt
            package = json.loads(user_prompt)
            if "evaluated_candidates" in package:
                return {
                    "role": package["role"],
                    "selected_candidate_ids": [
                        row["component_candidate_id"]
                        for row in package["evaluated_candidates"]
                    ],
                    "role_summary": "Reduced.",
                    "no_viable_reason": None,
                    "rejected_summary": [],
                }
            if package["role"] == "server_platform":
                time.sleep(5)
            return {
                "role": package["role"],
                "evaluated_candidates": [
                    {
                        "component_candidate_id": candidate["component_candidate_id"],
                        "fit_tier": "strong_fit",
                        "facts": {},
                        "matched_constraints": [],
                        "missing_facts": [],
                        "mismatch_reasons": [],
                        "evidence": "",
                        "confidence": "medium",
                    }
                    for candidate in package["candidates"]
                ],
            }

    result = distill_component_candidate_matrix(
        product_group="server",
        component_candidate_matrix={
            "product_group": "server",
            "platform_candidates": [
                _candidate("server_platform-timeout", price=Decimal("100"), stock=2)
            ],
            "cpu_candidates": [_candidate("cpu-ok", price=Decimal("50"), stock=4)],
        },
        constraints_by_role={},
        llm_client=HangingPlatformClient(),
        max_seconds=1,
        chunk_timeout_seconds=0.02,
        progress_callback=lambda event, fields: progress_events.append((event, dict(fields))),
    )

    diagnostics = result.diagnostics
    failed_chunk = diagnostics["full_matrix_failed_chunks"][0]

    assert diagnostics["full_matrix_evaluation_used"] is True
    assert diagnostics["evaluated_candidate_count_by_role"]["cpu"] == 1
    assert failed_chunk["role"] == "server_platform"
    assert failed_chunk["attempt_count"] == 2
    assert failed_chunk["timeout_kind"] == "chunk_timeout"
    assert diagnostics["role_diagnostics"]["server_platform"]["retried_chunk_count"] == 1
    assert any(event == "role_evaluator_timeout" for event, _fields in progress_events)
    assert any(
        event == "reducer_done"
        for event, fields in progress_events
        if fields["role"] == "cpu"
    )


def test_server_78_like_distilled_package_stays_under_budget_without_raw_json() -> None:
    matrix = {
        "product_group": "server",
        "semantic_planner_source": "llm",
        "category_planner_source": "ai_category_planner",
        "category_plan": {
            "server_platform": ["V110100"],
            "cpu": ["V110103"],
            "ram": ["V110104"],
            "storage": ["V110106"],
            "storage_controller": ["V110107"],
            "network_adapter": ["V120116"],
            "power_supply": ["V110108"],
            "cable": ["V110109"],
        },
        "required_capabilities": [
            {
                "role": "cpu",
                "capability_id": "cpu.intel.min_24c",
                "hard": True,
                "parsed_requirements": {"min_cores_per_cpu": 24, "vendor": "Intel"},
            }
        ],
        "platform_candidates": [
            _candidate(f"platform-{index:02d}", price=Decimal(1000 + index), stock=2)
            | {"raw_json": {"huge": "x" * 5000}}
            for index in range(40)
        ],
        "cpu_candidates": [
            _candidate(f"cpu-{index:02d}", price=Decimal(100 + index), stock=4)
            for index in range(60)
        ],
        "ram_candidates": [
            _candidate(f"ram-{index:02d}", price=Decimal(50 + index), stock=16)
            for index in range(60)
        ],
        "ssd_candidates": [
            _candidate(f"ssd-{index:02d}", price=Decimal(80 + index), stock=16)
            for index in range(80)
        ],
        "storage_controller_candidates": [
            _candidate(f"storage_controller-{index:02d}", price=Decimal(150 + index), stock=4)
            for index in range(30)
        ],
        "network_adapter_candidates": [
            _candidate(f"network_adapter-{index:02d}", price=Decimal(120 + index), stock=4)
            for index in range(30)
        ],
        "power_supply_candidates": [
            _candidate(f"power_supply-{index:02d}", price=Decimal(90 + index), stock=4)
            for index in range(20)
        ],
        "cable_candidates": [
            _candidate(f"cable-{index:02d}", price=Decimal(10 + index), stock=4)
            for index in range(20)
        ],
    }

    def response(package: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": package["role"],
            "evaluated_candidates": [
                {
                    "component_candidate_id": candidate["component_candidate_id"],
                    "fit_tier": "strong_fit",
                    "facts": {},
                    "matched_constraints": [],
                    "missing_facts": [],
                    "mismatch_reasons": [],
                    "evidence": "",
                    "confidence": "medium",
                }
                for candidate in package["candidates"]
            ],
            "shortlist_candidate_ids": {
                "strong_fit": [],
                "possible_fit": [],
                "fallback_unknown": [],
            },
        }

    distilled = distill_component_candidate_matrix(
        product_group="server",
        component_candidate_matrix=matrix,
        constraints_by_role=constraints_by_role_from_role_plan(matrix),
        llm_client=_FakeDistillerClient(response),
        chunk_max_chars=5000,
    ).component_candidate_matrix
    package = build_llm_configurator_package(
        user_request="server #78 like request",
        normalized_requirements=[],
        ready_stock_candidates=[],
        component_candidate_matrix=distilled,
        rule_based_build_candidates=[],
        max_package_chars=1_000_000,
    )
    package_text = json.dumps(package, ensure_ascii=False)

    assert package["matrix_distiller_used"] is True
    assert package["package_budget"]["final_chars"] <= 1_000_000
    assert package["full_matrix_evaluation_used"] is True
    assert package["broad_count_by_role"]["cpu"] == 60
    assert package["evaluated_candidate_count_by_role"]["cpu"] == 60
    assert package["distilled_count_by_role"]["cpu"] == 48
    assert package["selected_candidate_count_by_role"]["network_adapter"] == 30
    assert package["role_chunk_count_by_role"]["ssd"] > 1
    assert (
        package["no_recommendation_coverage"]["coverage_percent_by_role"]["network_adapter"]
        == 100.0
    )
    assert package["llm_cost_diagnostics"]["llm_calls_count"] > 0
    assert "raw_json" not in package_text
    assert "x" * 100 not in package_text


def _candidate(component_id: str, *, price: Decimal, stock: int) -> dict[str, Any]:
    return {
        "component_candidate_id": component_id,
        "role": component_id.split("-")[0],
        "producer": "Vendor",
        "part_number": component_id.upper(),
        "name": f"Server component {component_id}",
        "category_id": "CAT",
        "price_value": str(price),
        "price_currency": "USD",
        "available_quantity": stock,
        "quantity_required": 1,
        "fit_tier": "possible_fit",
        "extracted_facts": {"payload": "must not be sent", "safe_fact": "yes"},
    }
