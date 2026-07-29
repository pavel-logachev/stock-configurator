from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.catalog.category_repository import CategoryRepository
from app.catalog.product_repository import ProductRepository
from app.cli.compose_full_category_quote import _summary as v3_quote_summary
from app.core.config import LlmSettings
from app.core.database import Base
from app.db.models import DistributorCategory, DistributorProduct, DistributorStockPrice
from app.distributors.category_refresh import CategoryRefreshResult
from app.llm.full_category_composer import (
    V3_CODE_VALIDATION_BYPASSED,
    V3_NO_RECOMMENDATION,
    V3_VALIDATED,
    FullCategoryComposerOutcome,
    build_full_category_quote_prompts,
    compose_full_category_quote,
    parse_full_category_composer_payload,
)
from app.llm.simple_stock_composer import (
    SIMPLE_STOCK_QUOTE_ACCEPTED,
    build_simple_stock_quote_prompts,
    compose_simple_stock_quote,
)
from app.matching import simple_stock_quote_service as simple_stock_service_module
from app.matching import v3_full_category_quote_service as v3_service_module
from app.matching.full_category_matrix import (
    MATRIX_EMPTY_AFTER_CATEGORY_SELECTION,
    MATRIX_TOO_LARGE_FOR_MODEL,
    build_full_category_matrix_group_package,
    build_full_category_matrix_package,
)
from app.matching.simple_stock_matrix import build_simple_stock_matrix_group_package
from app.matching.simple_stock_quote_service import route_simple_stock_target
from app.matching.v3_full_category_profiles import (
    V3_FULL_CATEGORY_PROFILES,
    resolve_v3_full_category_profile,
)
from app.matching.v3_full_category_quote_service import (
    V3_STOCK_REFRESH_FAILED,
    route_v3_full_category_target,
    run_v3_full_category_quote,
    v3_result_state,
)


class AsyncSessionAdapter:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def scalar(self, statement: Any) -> Any:
        return self._session.scalar(statement)

    async def execute(self, statement: Any) -> Any:
        return self._session.execute(statement)


class FakeLlmClient:
    def __init__(self, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload_index = min(len(self.calls), len(self.payloads) - 1)
        self.calls.append((system_prompt, user_prompt))
        return self.payloads[payload_index]


def test_full_category_payload_normalizes_string_coverage_contributions() -> None:
    payload = parse_full_category_composer_payload(
        {
            "status": "quote",
            "quote": {
                "title": "Quote",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": "ocs:p1",
                        "stock_row_id": "ocs:p1:s1",
                        "quantity": 1,
                        "coverage_contributions": [
                            "Базовое устройство выбрано как ближайший якорь",
                        ],
                    }
                ],
            },
        }
    )

    assert payload.quote is not None
    assert payload.quote.lines[0].coverage_contributions == [
        {"description": "Базовое устройство выбрано как ближайший якорь"}
    ]


@pytest.fixture()
def db_session() -> Generator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        yield session

    engine.dispose()


def test_full_category_matrix_uses_latest_snapshot_and_selected_category(
    db_session: Session,
) -> None:
    old_sync = datetime(2026, 6, 9, tzinfo=UTC)
    latest_sync = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=latest_sync)
    _seed_product(db_session, item_id="p2", category_id="cat-b", synced_at=latest_sync)
    _seed_stock(db_session, item_id="p1", location="OLD", synced_at=old_sync)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=latest_sync)
    _seed_stock(db_session, item_id="p1", location="SPB", synced_at=latest_sync)
    _seed_stock(db_session, item_id="p2", location="MSK", synced_at=latest_sync)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_matrix("ocs", "cat-a"))

    assert [(row.product.item_id, row.stock.location) for row in rows] == [
        ("p1", "MSK"),
        ("p1", "SPB"),
    ]


def test_full_category_group_matrix_uses_latest_snapshot_per_selected_category(
    db_session: Session,
) -> None:
    cat_a_latest = datetime(2026, 6, 11, tzinfo=UTC)
    cat_b_latest = datetime(2026, 6, 10, tzinfo=UTC)
    old_sync = datetime(2026, 6, 9, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=cat_a_latest)
    _seed_product(db_session, item_id="p2", category_id="cat-b", synced_at=cat_b_latest)
    _seed_stock(db_session, item_id="p1", location="A-OLD", synced_at=old_sync)
    _seed_stock(db_session, item_id="p1", location="A-NEW", synced_at=cat_a_latest)
    _seed_stock(db_session, item_id="p2", location="B-OLD", synced_at=old_sync)
    _seed_stock(db_session, item_id="p2", location="B-NEW", synced_at=cat_b_latest)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(
        repository.list_latest_full_category_group_matrix("ocs", ["cat-a", "cat-b"])
    )

    assert [(row.product.item_id, row.stock.location) for row in rows] == [
        ("p1", "A-NEW"),
        ("p2", "B-NEW"),
    ]


def test_full_category_matrix_package_marks_oversize_without_dropping_rows(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="SPB", synced_at=synced_at)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_matrix("ocs", "cat-a"))

    package = build_full_category_matrix_package(
        distributor_code="ocs",
        category_id="cat-a",
        rows=rows,
        max_package_chars=1,
        model="qwen/qwen3.7-plus",
    )

    assert package.status == MATRIX_TOO_LARGE_FOR_MODEL
    assert package.payload["diagnostics"]["row_count"] == 2
    assert len(package.payload["category_sections"]) == 1
    assert package.payload["matrix_policy"]["semantic_trimming"] is False
    assert package.payload["matrix_policy"]["compatibility_prefiltering"] is False
    assert package.payload["matrix_policy"]["mechanical_price_ordering"] is True
    assert package.payload["matrix_policy"]["primary_matrix_view"] == "category_sections"
    assert package.payload["row_legend"]["stock_row_id"].startswith("Specific stock")
    section = package.payload["category_sections"][0]
    assert section["category_id"] == "cat-a"
    assert section["category_path"] == "Category"
    product = section["products"][0]
    assert product["component_candidate_id"] == "ocs:p1"
    assert product["product"]["package_facts"] == {"weight": 1.5}
    assert "package_json" not in product["product"]
    assert "raw_json" not in product["product"]
    assert all("raw_json" not in row for row in product["stock_rows"])
    assert [row["stock_row_id"] for row in product["stock_rows"]] == [
        "ocs:p1:1",
        "ocs:p1:2",
    ]
    assert package.payload["diagnostics"]["raw_json_included"] is False
    assert package.payload["diagnostics"]["package_json_included"] is False


def test_full_category_matrix_payload_excludes_raw_payload_without_dropping_rows(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="SPB", synced_at=synced_at)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_matrix("ocs", "cat-a"))
    package = build_full_category_matrix_package(
        distributor_code="ocs",
        category_id="cat-a",
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )

    payload_text = package.json_payload
    assert package.payload["diagnostics"]["row_count"] == 2
    assert package.payload["diagnostics"]["stock_row_count"] == 2
    assert len(package.payload["category_sections"]) == 1
    assert len(package.payload["category_sections"][0]["products"]) == 1
    assert '"raw_json"' not in payload_text
    assert '"package_json"' not in payload_text
    assert '"raw_product"' not in payload_text
    assert '"raw_stock"' not in payload_text
    assert '"package_facts"' in payload_text
    assert package.payload["matrix_payload_schema_version"] == "matrix_payload_schema_v7"
    assert package.payload["matrix_index"] == [
        {
            "component_candidate_id": "ocs:p1",
            "category_id": "cat-a",
            "category_path": "Category",
            "producer": "Vendor",
            "part_number": "PN-p1",
            "item_name": "Product p1",
            "total_stock_quantity": 6,
            "minimum_price_by_currency": {"USD": "100.0000"},
            "stock_row_count": 2,
        }
    ]
    product = package.payload["category_sections"][0]["products"][0]
    fact_ids = {item["fact_id"] for item in product["fact_refs"]}
    assert "F:ocs:p1:producer" in fact_ids
    assert "F:ocs:p1:part_number" in fact_ids
    assert "F:ocs:p1:item_name" in fact_ids
    assert "F:ocs:p1:product_description" in fact_ids
    assert package.payload["diagnostics"]["fact_reference_count"] >= 4


def test_full_category_matrix_marks_empty_selection_without_llm_ready_state() -> None:
    package = build_full_category_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-empty"],
        rows=[],
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )

    assert package.status == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
    assert package.payload["diagnostics"]["status"] == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
    assert package.payload["diagnostics"]["row_count"] == 0
    assert package.payload["category_sections"] == []


def test_full_category_group_matrix_preserves_all_selected_category_rows(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_product(db_session, item_id="p2", category_id="cat-b", synced_at=synced_at)
    _seed_product(db_session, item_id="p3", category_id="cat-c", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="p2", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="p3", location="MSK", synced_at=synced_at)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(
        repository.list_latest_full_category_group_matrix("ocs", ["cat-b", "cat-a"])
    )
    package = build_full_category_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-b", "cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )

    assert [(row.product.category_id, row.product.item_id) for row in rows] == [
        ("cat-a", "p1"),
        ("cat-b", "p2"),
    ]
    assert package.payload["category_ids"] == ["cat-b", "cat-a"]
    assert package.payload["diagnostics"]["row_count"] == 2
    assert package.payload["diagnostics"]["component_count"] == 2
    assert len(package.payload["category_sections"]) == 2
    assert [section["category_id"] for section in package.payload["category_sections"]] == [
        "cat-b",
        "cat-a",
    ]
    section_product_ids = [
        section["products"][0]["product"]["item_id"]
        for section in package.payload["category_sections"]
    ]
    assert section_product_ids == [
        "p2",
        "p1",
    ]
    assert package.payload["matrix_policy"]["semantic_trimming"] is False
    assert package.payload["matrix_policy"]["semantic_ranking"] is False


def test_full_category_matrix_package_orders_rows_by_input_category_then_price(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="a-expensive", category_id="cat-a", synced_at=synced_at)
    _seed_product(db_session, item_id="z-cheap", category_id="cat-a", synced_at=synced_at)
    _seed_product(db_session, item_id="r-rur", category_id="cat-a", synced_at=synced_at)
    _seed_product(db_session, item_id="b-category", category_id="cat-b", synced_at=synced_at)
    _seed_stock(
        db_session,
        item_id="a-expensive",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("200.0000"),
    )
    _seed_stock(
        db_session,
        item_id="z-cheap",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("50.0000"),
    )
    _seed_stock(
        db_session,
        item_id="r-rur",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("10.0000"),
        price_order_currency="RUR",
    )
    _seed_stock(
        db_session,
        item_id="b-category",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("300.0000"),
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(
        repository.list_latest_full_category_group_matrix("ocs", ["cat-b", "cat-a"])
    )
    package = build_full_category_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-b", "cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )

    assert package.payload["diagnostics"]["row_count"] == 4
    sections = package.payload["category_sections"]
    assert [section["category_id"] for section in sections] == ["cat-b", "cat-a"]
    assert [row["product"]["item_id"] for row in sections[0]["products"]] == ["b-category"]
    assert [row["product"]["item_id"] for row in sections[1]["products"]] == [
        "z-cheap",
        "a-expensive",
        "r-rur",
    ]
    assert package.payload["matrix_policy"]["semantic_trimming"] is False
    assert package.payload["matrix_policy"]["semantic_ranking"] is False
    assert package.payload["matrix_policy"]["mechanical_price_ordering"] is True


def test_v3_storage_profile_resolves_full_product_group() -> None:
    profile, category_ids = resolve_v3_full_category_profile(profile="storage")

    assert profile == "storage"
    assert category_ids == [
        "V2101",
        "V2103",
        "V2105",
        "V2104",
        "V3100",
        "V3104",
        "V110106",
        "V110112",
    ]


def test_category_repository_expands_target_category_to_descendants(
    db_session: Session,
) -> None:
    repository = CategoryRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    db_session.add_all(
        [
            _category("root", None, "Root", 0, synced_at),
            _category("child-a", "root", "Child A", 1, synced_at),
            _category("child-b", "root", "Child B", 1, synced_at),
            _category("grandchild", "child-a", "Grandchild", 2, synced_at),
        ]
    )
    db_session.commit()

    async def run() -> list[str]:
        return await repository.list_category_ids_with_descendants(
            distributor_code="ocs",
            category_ids=["root"],
        )

    assert asyncio.run(run()) == ["root", "child-a", "grandchild", "child-b"]


def test_v3_request_intake_selects_category_ids_with_llm() -> None:
    fake_client = FakeLlmClient(
        {
            "status": "selected_category",
            "category_ids": ["cat-root"],
            "reason": "The request matches the root category.",
            "resolved_request": {
                "objective": "cheapest_minimum_viable",
                "target_category_ids": ["cat-root"],
                "customer_task_summary": "Product group request",
                "hard_requirements": [
                    {
                        "key": "workload",
                        "value": "requested workload",
                        "source_phrase": "Need a product group quote",
                        "explicit": True,
                    }
                ],
                "soft_preferences": [],
                "unknowns_or_missing_facts": ["Exact model is not specified"],
                "assumptions_allowed_for_draft": [],
                "compatibility_attention_points": [],
                "non_goals": [],
            },
        }
    )

    decision = route_v3_full_category_target(
        text="Need a product group quote",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "cat-root",
                "parent_category_id": None,
                "name": "Root category",
                "level": 0,
                "path": "Root category",
                "enabled_for_sync": True,
            }
        ],
        llm_client=fake_client,
    )

    assert decision.status == "selected_category"
    assert decision.category_ids == ["cat-root"]
    assert decision.profile is None
    assert decision.resolved_request["objective"] == "cheapest_minimum_viable"
    assert decision.resolved_request["target_category_ids"] == ["cat-root"]
    assert decision.resolved_request["hard_requirements"][0]["key"] == "workload"
    assert decision.resolved_request["unknowns_or_missing_facts"] == [
        "Exact model is not specified"
    ]
    assert len(fake_client.calls) == 1
    assert "Request Intake v7" in fake_client.calls[0][0]
    assert "Build one structured requirement ledger" in fake_client.calls[0][0]
    assert "Do not choose products, SKUs, stock rows" in fake_client.calls[0][0]
    assert "Do not duplicate one requirement" in fake_client.calls[0][0]
    assert "objects[].requested_items[].constraints[]" in fake_client.calls[0][0]
    assert "Detailed specification does not mean exact-only" in fake_client.calls[0][0]
    assert "Named vendor, model, generation" in fake_client.calls[0][0]
    assert "Operational quantity and spare quantity are separate" in fake_client.calls[0][0]
    assert "Profile rule" in fake_client.calls[0][0]
    user_payload = json.loads(fake_client.calls[0][1])
    assert user_payload["prompt_version"] == "request_intake_v7_1"
    assert user_payload["resolved_request_schema_version"] == (
        "resolved_request_schema_v7_1"
    )
    assert user_payload["profile_catalog"] == []
    assert user_payload["category_catalog"][0]["category_id"] == "cat-root"
    assert user_payload["intake_policy"]["single_ledger"] is True
    assert user_payload["intake_policy"]["legacy_hard_soft_arrays"] == "do_not_return"
    assert decision.prompt_version == "request_intake_v7_1"
    assert decision.schema_version == "resolved_request_schema_v7_1"
    assert decision.canonical_input_hash


def test_v3_request_intake_selects_known_profile_with_llm() -> None:
    fake_client = FakeLlmClient(
        {
            "status": "selected_profile",
            "profile": "server",
            "category_ids": [],
            "reason": "The request matches the server product profile.",
            "resolved_request": {
                "objective": "cheapest_minimum_viable",
                "profile": "server",
                "customer_task_summary": "Server product group request",
                "hard_requirements": [],
                "soft_preferences": [],
                "unknowns_or_missing_facts": [],
                "assumptions_allowed_for_draft": [],
                "compatibility_attention_points": [],
                "non_goals": [],
            },
        }
    )

    decision = route_v3_full_category_target(
        text="Server with CPU, RAM, disks and SFP+",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": category_id,
                "parent_category_id": None,
                "name": f"Category {category_id}",
                "level": 0,
                "path": f"Category {category_id}",
                "enabled_for_sync": True,
            }
            for category_id in V3_FULL_CATEGORY_PROFILES["server"].category_ids
        ],
        llm_client=fake_client,
    )

    server_profile_category_ids = list(V3_FULL_CATEGORY_PROFILES["server"].category_ids)
    assert decision.status == "selected_profile"
    assert decision.profile == "server"
    assert decision.category_ids == server_profile_category_ids
    assert decision.resolved_request["profile"] == "server"
    assert decision.resolved_request["target_category_ids"] == server_profile_category_ids


def test_v3_request_intake_accepts_v6_retrieval_plan_category_groups() -> None:
    decision = route_v3_full_category_target(
        text="Need a custom stocked system",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "cat-anchor",
                "parent_category_id": None,
                "name": "Anchor",
                "level": 0,
                "path": "Anchor",
                "enabled_for_sync": True,
            },
            {
                "category_id": "cat-component",
                "parent_category_id": None,
                "name": "Component",
                "level": 0,
                "path": "Component",
                "enabled_for_sync": True,
            },
            {
                "category_id": "cat-fallback",
                "parent_category_id": None,
                "name": "Fallback",
                "level": 0,
                "path": "Fallback",
                "enabled_for_sync": True,
            },
        ],
        llm_client=FakeLlmClient(
            {
                "status": "selected_category",
                "profile": None,
                "category_ids": [],
                "reason": "Use v6 retrieval groups.",
                "resolved_request": {
                    "procurement_mode": "best_available",
                    "allow_partial_quote": True,
                    "requirements": [
                        {
                            "id": "R1",
                            "object_id": "O1",
                            "dimension": "product_type",
                            "requested": "custom stocked system",
                            "comparison": "semantic",
                            "value": None,
                            "unit": None,
                            "priority": "core",
                            "substitution": "degrade_allowed",
                            "source_phrase": "custom stocked system",
                        }
                    ],
                    "retrieval_plan": {
                        "anchor_category_ids": ["cat-anchor"],
                        "component_category_ids": ["cat-component"],
                        "fallback_category_ids": ["cat-fallback"],
                    },
                },
            }
        ),
    )

    assert decision.status == "selected_category"
    assert decision.category_ids == ["cat-anchor", "cat-component", "cat-fallback"]
    assert decision.resolved_request["target_category_ids"] == [
        "cat-anchor",
        "cat-component",
        "cat-fallback",
    ]


def test_v3_request_intake_rejects_profile_unavailable_for_catalog() -> None:
    decision = route_v3_full_category_target(
        text="Server with CPU, RAM, disks and SFP+",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "S-CPU",
                "parent_category_id": None,
                "name": "Treolan processors",
                "level": 0,
                "path": "Treolan processors",
                "enabled_for_sync": True,
            }
        ],
        llm_client=FakeLlmClient(
            {
                "status": "selected_profile",
                "profile": "server",
                "category_ids": [],
                "reason": "The request matches servers.",
            }
        ),
    )

    assert decision.status == "schema_error"
    assert decision.profile == "server"
    assert decision.error_type == "V3CategoryIntakeUnavailableProfile"


def test_v3_request_intake_falls_back_to_valid_categories_when_profile_unavailable() -> None:
    decision = route_v3_full_category_target(
        text="Server with CPU, RAM, disks and SFP+",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "treolan-server-root",
                "parent_category_id": None,
                "name": "Treolan servers",
                "level": 0,
                "path": "Treolan servers",
                "enabled_for_sync": True,
            },
            {
                "category_id": "treolan-server-options",
                "parent_category_id": None,
                "name": "Treolan server options",
                "level": 0,
                "path": "Treolan server options",
                "enabled_for_sync": True,
            },
        ],
        llm_client=FakeLlmClient(
            {
                "status": "selected_profile",
                "profile": "server",
                "category_ids": ["treolan-server-root", "treolan-server-options"],
                "reason": "The request matches servers.",
                "resolved_request": {
                    "profile": "server",
                    "customer_task_summary": "Нужны серверы с опциями",
                },
            }
        ),
    )

    assert decision.status == "selected_category"
    assert decision.profile is None
    assert decision.category_ids == ["treolan-server-root", "treolan-server-options"]
    assert decision.resolved_request["profile"] is None
    assert decision.resolved_request["target_category_ids"] == [
        "treolan-server-root",
        "treolan-server-options",
    ]


def test_v3_auto_intake_uses_selected_profile_for_matrix(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_at = datetime(2026, 6, 17, tzinfo=UTC)
    db_session.add_all(
        [
            _category("V11", None, "Servers", 0, synced_at),
            _category("V1100", "V11", "Ready servers", 1, synced_at),
            _category("V12", None, "Network", 0, synced_at),
        ]
    )
    _seed_product(
        db_session,
        item_id="cpu-1",
        category_id="V110103",
        synced_at=synced_at,
        item_name="CPU product",
    )
    _seed_stock(db_session, item_id="cpu-1", location="MSK", synced_at=synced_at)
    db_session.commit()
    captured: dict[str, Any] = {}

    def fake_route(**_kwargs: Any) -> v3_service_module.V3RequestIntakeDecision:
        server_profile_category_ids = list(V3_FULL_CATEGORY_PROFILES["server"].category_ids)
        return v3_service_module.V3RequestIntakeDecision(
            status="selected_profile",
            profile="server",
            category_ids=server_profile_category_ids,
            reason="Server request matched the server product profile.",
            resolved_request={
                "objective": "cheapest_minimum_viable",
                "profile": "server",
                "target_category_ids": server_profile_category_ids,
                "customer_task_summary": "Need a server with components",
                "hard_requirements": [],
            },
        )

    def fake_compose(**kwargs: Any) -> FullCategoryComposerOutcome:
        captured["resolved_request"] = kwargs["resolved_request"]
        captured["category_ids"] = list(kwargs["matrix_package"].payload["category_ids"])
        return FullCategoryComposerOutcome(
            pipeline_version="v3_full_category_matrix",
            used=True,
            status="quote",
            final_status_source=V3_VALIDATED,
            primary_recommendation_status="valid",
            validated_quote={
                "engineering_review_required": True,
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "lines": [],
            },
            diagnostics={},
        )

    monkeypatch.setattr(v3_service_module, "route_v3_full_category_target", fake_route)
    monkeypatch.setattr(
        v3_service_module,
        "_compose_v3_full_category_quote_sync",
        fake_compose,
    )

    result = asyncio.run(
        run_v3_full_category_quote(
            text="Need server with CPU, RAM, disks and SFP+",
            session=AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            distributor_code="ocs",
            settings=LlmSettings(
                _env_file=None,
                v3_refresh_categories_before_llm=False,
            ),
        )
    )

    _, server_profile_category_ids = resolve_v3_full_category_profile(profile="server")
    assert result.profile == "server"
    assert result.category_ids == server_profile_category_ids
    assert captured["category_ids"] == server_profile_category_ids
    assert captured["resolved_request"]["target_category_ids"] == server_profile_category_ids
    assert result.report_json["v3_request_intake"]["status"] == "selected_profile"
    assert result.report_json["diagnostics"]["matrix_row_count"] == 1


def test_v3_request_intake_rejects_unknown_category_ids() -> None:
    decision = route_v3_full_category_target(
        text="Need a product group quote",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "cat-root",
                "parent_category_id": None,
                "name": "Root category",
                "level": 0,
                "path": "Root category",
                "enabled_for_sync": True,
            }
        ],
        llm_client=FakeLlmClient(
            {
                "status": "selected_category",
                "category_ids": ["made-up-category"],
                "reason": "Invented ID.",
            }
        ),
    )

    assert decision.status == "schema_error"
    assert decision.error_type == "V3CategoryIntakeUnknownCategoryIds"


def test_simple_stock_route_uses_clean_router_prompt() -> None:
    server_category_ids = list(V3_FULL_CATEGORY_PROFILES["server"].category_ids[:2])
    fake_client = FakeLlmClient(
        {
            "status": "selected_category",
            "profile": None,
            "category_ids": server_category_ids,
            "reason": "Server request.",
            "resolved_request": {
                "customer_task_summary": "Нужен сервер",
                "target_objects": [
                    {
                        "target_id": "T1",
                        "label": "Requested system",
                        "quantity": 1,
                        "expects_anchor_line": True,
                    }
                ],
                "allow_analogs": True,
                "allow_partial_offer": True,
            },
        }
    )

    decision = route_simple_stock_target(
        text="Нужен сервер",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": category_id,
                "parent_category_id": None,
                "name": category_id,
                "level": 0,
                "path": category_id,
            }
            for category_id in V3_FULL_CATEGORY_PROFILES["server"].category_ids
        ],
        llm_client=fake_client,
    )

    system_prompt, user_prompt = fake_client.calls[0]
    assert decision.status == "selected_category"
    assert decision.profile is None
    assert decision.category_ids == server_category_ids
    assert decision.resolved_request["target_objects"][0]["expects_anchor_line"] is True
    assert "Simple Route" in system_prompt
    assert "minimal set of distributor category subtrees" in system_prompt
    assert "Cover every explicit\n  product role first" in system_prompt
    assert "Whole-object priority" in system_prompt
    assert "ready, assembled, configured or finished devices" in system_prompt
    assert "Component/platform categories\n  are not a substitute" in system_prompt
    assert "Anchor/base completeness" in system_prompt
    assert "expects_anchor_line=true" in system_prompt
    assert "that target/base object is its own explicit product role" in system_prompt
    assert "components-only route is incomplete" in system_prompt
    assert "Supporting component, option, license or accessory branches" in system_prompt
    assert "Do not replace an assembled/ready system" in system_prompt
    assert "Commercial materiality boundary" in system_prompt
    assert "fitment_notes" in system_prompt
    assert "Do not move explicit high-performance cooling requests into fitment_notes" in (
        system_prompt
    )
    assert "Keep each requested cooling role in must_have" in system_prompt
    assert "do not select dedicated accessory subtrees" in system_prompt
    assert "Atomic request roles" in system_prompt
    assert "must_have must be a flat list" in system_prompt
    assert "independently fulfillable commercial roles" in system_prompt
    assert "Split comma, semicolon, plus" in system_prompt
    assert "Do not atomize" in system_prompt
    assert '"requirement_id": "R1"' in system_prompt
    assert "most\n  specific visible stocked base/platform subtree" in system_prompt
    assert "Do not choose a catalog-wide root" in system_prompt
    assert "additional component, option" in system_prompt
    assert "license and material option subtrees" in system_prompt
    assert "smallest subtree or set of descendant subtrees" in system_prompt
    assert "never choose a broad root merely as insurance" in system_prompt
    assert "If a complete selection exceeds max_total_subtree_positions" in system_prompt
    assert "same explicit role coverage is preserved" in system_prompt
    assert "Break ties by lower total subtree_in_stock_positions" in system_prompt
    assert "nearest parent" not in system_prompt
    assert "may exclude relevant sibling roles" not in system_prompt
    assert "narrowest sufficient" not in system_prompt
    assert "target object detection" in system_prompt
    assert "target_objects" in system_prompt
    assert "commercially material quote roles" in system_prompt
    assert "low-value installation" in system_prompt
    assert "category_routing_index" in system_prompt
    assert "stock previews" not in system_prompt
    assert "Never select both an ancestor and its descendant" in system_prompt
    assert "Request Intake v7" not in system_prompt
    assert "Category-role discipline" in system_prompt
    assert "not by specifications or compatibility terms" in system_prompt
    assert "If no stocked category for the\n  requested product role is visible" in system_prompt
    assert '"prompt_version":"simple_stock_route_v19"' in user_prompt
    assert '"category_routing_index"' in user_prompt
    assert '"max_total_subtree_positions":500' in user_prompt
    assert '"category_root_summaries"' not in user_prompt
    assert '"selectable_categories"' not in user_prompt
    assert '"category_catalog"' not in user_prompt
    assert '"max_category_ids":12' in user_prompt
    assert "object_anchor_policy" not in user_prompt


def test_simple_stock_route_user_prompt_builds_category_routing_index() -> None:
    fake_client = FakeLlmClient(
        {
            "status": "selected_category",
            "profile": None,
            "category_ids": ["cpu", "hba"],
            "reason": "Server options are in separate branches.",
            "resolved_request": {
                "customer_task_summary": "Нужен сервер с опциями",
                "allow_analogs": True,
                "allow_partial_offer": True,
            },
        }
    )

    decision = route_simple_stock_target(
        text="Нужен сервер с CPU, RAM и HBA",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "root-server",
                "parent_category_id": None,
                "name": "Servers and options",
                "level": 0,
                "path": "Servers and options",
            },
            {
                "category_id": "cpu",
                "parent_category_id": "root-server",
                "name": "Processors",
                "level": 1,
                "path": "Servers and options / Processors",
            },
            {
                "category_id": "hba",
                "parent_category_id": "root-server",
                "name": "FC HBA adapters",
                "level": 1,
                "path": "Servers and options / FC HBA adapters",
            },
        ],
        category_stock_counts=[
            {
                "category_id": "cpu",
                "position_count": 5,
                "stock_row_count": 7,
            },
            {
                "category_id": "hba",
                "position_count": 3,
                "stock_row_count": 3,
            },
        ],
        llm_client=fake_client,
    )

    _system_prompt, user_prompt = fake_client.calls[0]
    payload = json.loads(user_prompt)
    routing_index = payload["category_routing_index"]
    assert decision.category_ids == ["cpu", "hba"]
    assert decision.selected_subtree_position_count == 8
    assert decision.max_total_subtree_positions == 500
    assert decision.routing_over_budget is False
    assert routing_index == [
        {
            "category_id": "root-server",
            "category_path": "Servers and options",
            "level": 0,
            "children_count": 2,
            "descendant_count": 2,
            "subtree_in_stock_positions": 8,
        },
        {
            "category_id": "cpu",
            "parent_category_id": "root-server",
            "category_path": "Servers and options / Processors",
            "level": 1,
            "children_count": 0,
            "descendant_count": 0,
            "subtree_in_stock_positions": 5,
        },
        {
            "category_id": "hba",
            "parent_category_id": "root-server",
            "category_path": "Servers and options / FC HBA adapters",
            "level": 1,
            "children_count": 0,
            "descendant_count": 0,
            "subtree_in_stock_positions": 3,
        }
    ]


def test_simple_stock_route_does_not_mechanically_add_ready_system_category() -> None:
    fake_client = FakeLlmClient(
        {
            "status": "selected_category",
            "profile": None,
            "category_ids": ["platform", "cpu"],
            "reason": "Components cover the requested server.",
            "resolved_request": {
                "customer_task_summary": "Need configured rack servers.",
                "target_objects": [
                    {
                        "target_id": "T1",
                        "label": "3 assembled rack servers",
                        "quantity": 3,
                        "expects_anchor_line": True,
                    }
                ],
                "must_have": ["3 assembled servers", "CPU options"],
                "allow_analogs": True,
                "allow_partial_offer": True,
            },
        }
    )

    decision = route_simple_stock_target(
        text="Need 3 assembled rack servers with CPU options route anchor 274",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "servers",
                "parent_category_id": None,
                "name": "Servers",
                "level": 0,
                "path": "Servers",
            },
            {
                "category_id": "ready",
                "parent_category_id": "servers",
                "name": "Assembled servers",
                "level": 1,
                "path": "Servers / Assembled servers",
            },
            {
                "category_id": "components",
                "parent_category_id": "servers",
                "name": "Server components",
                "level": 1,
                "path": "Servers / Server components",
            },
            {
                "category_id": "platform",
                "parent_category_id": "components",
                "name": "Server platforms",
                "level": 2,
                "path": "Servers / Server components / Server platforms",
            },
            {
                "category_id": "cpu",
                "parent_category_id": "components",
                "name": "Server processors",
                "level": 2,
                "path": "Servers / Server components / Server processors",
            },
        ],
        category_stock_counts=[
            {"category_id": "ready", "position_count": 2},
            {"category_id": "platform", "position_count": 4},
            {"category_id": "cpu", "position_count": 5},
        ],
        llm_client=fake_client,
    )

    assert decision.status == "selected_category"
    assert decision.category_ids == ["platform", "cpu"]
    assert decision.resolved_request["target_category_ids"] == [
        "platform",
        "cpu",
    ]
    assert "route_category_adjustments" not in decision.resolved_request


def test_simple_stock_route_does_not_add_ready_system_for_component_only_request() -> None:
    fake_client = FakeLlmClient(
        {
            "status": "selected_category",
            "profile": None,
            "category_ids": ["cpu"],
            "reason": "CPU-only request.",
            "resolved_request": {
                "customer_task_summary": "Need CPUs only.",
                "must_have": ["2 Intel server processors"],
                "target_objects": [],
                "allow_analogs": True,
                "allow_partial_offer": True,
            },
        }
    )

    decision = route_simple_stock_target(
        text="Need two server processors only route component 274",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "servers",
                "parent_category_id": None,
                "name": "Servers",
                "level": 0,
                "path": "Servers",
            },
            {
                "category_id": "ready",
                "parent_category_id": "servers",
                "name": "Assembled servers",
                "level": 1,
                "path": "Servers / Assembled servers",
            },
            {
                "category_id": "components",
                "parent_category_id": "servers",
                "name": "Server components",
                "level": 1,
                "path": "Servers / Server components",
            },
            {
                "category_id": "cpu",
                "parent_category_id": "components",
                "name": "Server processors",
                "level": 2,
                "path": "Servers / Server components / Server processors",
            },
        ],
        category_stock_counts=[
            {"category_id": "ready", "position_count": 2},
            {"category_id": "cpu", "position_count": 5},
        ],
        llm_client=fake_client,
    )

    assert decision.status == "selected_category"
    assert decision.category_ids == ["cpu"]
    assert "route_category_adjustments" not in decision.resolved_request


def test_simple_stock_route_does_not_mechanically_add_accessory_sibling_categories() -> None:
    fake_client = FakeLlmClient(
        {
            "status": "selected_category",
            "profile": None,
            "category_ids": ["misc"],
            "reason": "Server accessories are in misc.",
            "resolved_request": {
                "customer_task_summary": "Need servers with fans, heatsinks and riser cables.",
                "target_objects": [
                    {
                        "target_id": "T1",
                        "label": "Configured server",
                        "quantity": 1,
                        "expects_anchor_line": True,
                    }
                ],
                "must_have": ["high performance fans", "heatsinks", "riser", "cables"],
                "allow_analogs": True,
                "allow_partial_offer": True,
            },
        }
    )

    decision = route_simple_stock_target(
        text="Need server fans heatsinks riser cables route accessory 274",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "servers",
                "parent_category_id": None,
                "name": "Servers",
                "level": 0,
                "path": "Servers",
            },
            {
                "category_id": "ready",
                "parent_category_id": "servers",
                "name": "Assembled servers",
                "level": 1,
                "path": "Servers / Assembled servers",
            },
            {
                "category_id": "components",
                "parent_category_id": "servers",
                "name": "Server components",
                "level": 1,
                "path": "Servers / Server components",
            },
            {
                "category_id": "misc",
                "parent_category_id": "components",
                "name": "Misc server accessories",
                "level": 2,
                "path": "Servers / Server components / Misc server accessories",
            },
            {
                "category_id": "cooling",
                "parent_category_id": "components",
                "name": "Server cooling systems",
                "level": 2,
                "path": "Servers / Server components / Server cooling systems",
            },
            {
                "category_id": "cables",
                "parent_category_id": "components",
                "name": "Server cables and risers",
                "level": 2,
                "path": "Servers / Server components / Server cables and risers",
            },
        ],
        category_stock_counts=[
            {"category_id": "ready", "position_count": 2},
            {"category_id": "misc", "position_count": 4},
            {"category_id": "cooling", "position_count": 3},
            {"category_id": "cables", "position_count": 3},
        ],
        llm_client=fake_client,
    )

    assert decision.status == "selected_category"
    assert decision.category_ids == ["misc"]
    assert decision.resolved_request["target_category_ids"] == ["misc"]
    assert "route_category_adjustments" not in decision.resolved_request


def test_simple_stock_route_caches_identical_route_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with simple_stock_service_module._SIMPLE_STOCK_ROUTE_CACHE_LOCK:
        simple_stock_service_module._SIMPLE_STOCK_ROUTE_CACHE.clear()

    fake_client = FakeLlmClient(
        {
            "status": "selected_category",
            "profile": None,
            "category_ids": ["cpu"],
            "reason": "CPU category covers the role.",
            "resolved_request": {
                "customer_task_summary": "Need CPU",
                "allow_analogs": True,
                "allow_partial_offer": True,
            },
        }
    )
    monkeypatch.setattr(
        simple_stock_service_module,
        "_create_llm_client",
        lambda settings: fake_client,
    )

    kwargs = {
        "text": "Need Intel CPU",
        "distributor_code": "treolan",
        "settings": LlmSettings(_env_file=None),
        "category_catalog": [
            {
                "category_id": "cpu",
                "parent_category_id": None,
                "name": "Processors",
                "level": 0,
                "path": "Processors",
            }
        ],
        "category_stock_counts": [{"category_id": "cpu", "position_count": 5}],
    }

    first = route_simple_stock_target(**kwargs)
    second = route_simple_stock_target(**kwargs)

    assert len(fake_client.calls) == 1
    assert first.status == "selected_category"
    assert first.route_cache_status == "miss"
    assert first.selected_subtree_position_count == 5
    assert second.status == "selected_category"
    assert second.route_cache_status == "hit"
    assert second.category_ids == ["cpu"]
    assert second.resolved_request == first.resolved_request

    with simple_stock_service_module._SIMPLE_STOCK_ROUTE_CACHE_LOCK:
        simple_stock_service_module._SIMPLE_STOCK_ROUTE_CACHE.clear()


def test_simple_stock_route_removes_parent_child_overlap() -> None:
    fake_client = FakeLlmClient(
        {
            "status": "selected_category",
            "profile": None,
            "category_ids": ["cpu", "root-server", "hba"],
            "reason": "Mixed overlap.",
            "resolved_request": {
                "customer_task_summary": "Нужен сервер с опциями",
                "allow_analogs": True,
                "allow_partial_offer": True,
            },
        }
    )

    decision = route_simple_stock_target(
        text="Нужен сервер с CPU и HBA",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "root-server",
                "parent_category_id": None,
                "name": "Servers and options",
                "level": 0,
                "path": "Servers and options",
            },
            {
                "category_id": "cpu",
                "parent_category_id": "root-server",
                "name": "Processors",
                "level": 1,
                "path": "Servers and options / Processors",
            },
            {
                "category_id": "hba",
                "parent_category_id": "root-server",
                "name": "FC HBA adapters",
                "level": 1,
                "path": "Servers and options / FC HBA adapters",
            },
        ],
        llm_client=fake_client,
    )

    assert decision.status == "selected_category"
    assert decision.category_ids == ["root-server"]
    assert decision.resolved_request["target_category_ids"] == ["root-server"]


def test_product_repository_lists_limited_stock_preview_by_category(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 21, tzinfo=UTC)
    _seed_product(db_session, item_id="cpu-expensive", category_id="cpu", synced_at=synced_at)
    _seed_product(db_session, item_id="cpu-cheap", category_id="cpu", synced_at=synced_at)
    _seed_product(db_session, item_id="hba", category_id="hba", synced_at=synced_at)
    _seed_stock(
        db_session,
        item_id="cpu-expensive",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("200.0000"),
    )
    _seed_stock(
        db_session,
        item_id="cpu-cheap",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("100.0000"),
    )
    _seed_stock(
        db_session,
        item_id="hba",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("300.0000"),
        quantity_value=1,
        quantity_is_greater_than=True,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    preview = asyncio.run(
        repository.list_latest_stock_preview_for_categories(
            "ocs",
            ["cpu", "hba"],
            per_category_limit=1,
        )
    )

    assert [(row["category_id"], row["item_id"]) for row in preview] == [
        ("cpu", "cpu-cheap"),
        ("hba", "hba"),
    ]
    assert preview[1]["quantity_is_greater_than"] is True


def test_product_repository_lists_latest_stock_counts_by_category(
    db_session: Session,
) -> None:
    old_synced_at = datetime(2026, 6, 20, tzinfo=UTC)
    synced_at = datetime(2026, 6, 21, tzinfo=UTC)
    _seed_product(db_session, item_id="cpu-a", category_id="cpu", synced_at=synced_at)
    _seed_product(db_session, item_id="cpu-b", category_id="cpu", synced_at=synced_at)
    _seed_product(db_session, item_id="hba-a", category_id="hba", synced_at=synced_at)
    _seed_stock(db_session, item_id="cpu-a", location="MSK", synced_at=old_synced_at)
    _seed_stock(db_session, item_id="cpu-a", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="cpu-a", location="SPB", synced_at=synced_at)
    _seed_stock(db_session, item_id="cpu-b", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="hba-a", location="MSK", synced_at=synced_at)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    counts = asyncio.run(
        repository.list_latest_stock_counts_by_category("ocs", ["cpu", "hba"])
    )

    assert counts == [
        {"category_id": "cpu", "position_count": 2, "stock_row_count": 3},
        {"category_id": "hba", "position_count": 1, "stock_row_count": 1},
    ]


def test_simple_stock_route_falls_back_to_valid_categories_when_profile_unavailable() -> None:
    fake_client = FakeLlmClient(
        {
            "status": "selected_profile",
            "profile": "server",
            "category_ids": ["treolan-server-root", "treolan-server-options"],
            "reason": "Server request.",
            "resolved_request": {
                "profile": "server",
                "customer_task_summary": "Нужны серверы с опциями",
                "allow_analogs": True,
                "allow_partial_offer": True,
            },
        }
    )

    decision = route_simple_stock_target(
        text="Нужны серверы с опциями",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "treolan-server-root",
                "parent_category_id": None,
                "name": "Treolan servers",
                "level": 0,
                "path": "Treolan servers",
            },
            {
                "category_id": "treolan-server-options",
                "parent_category_id": None,
                "name": "Treolan server options",
                "level": 0,
                "path": "Treolan server options",
            },
        ],
        llm_client=fake_client,
    )

    system_prompt, _user_prompt = fake_client.calls[0]
    assert 'Return status="selected_category"' in system_prompt
    assert decision.status == "selected_category"
    assert decision.profile is None
    assert decision.category_ids == ["treolan-server-root", "treolan-server-options"]
    assert decision.resolved_request["profile"] is None
    assert decision.resolved_request["target_category_ids"] == [
        "treolan-server-root",
        "treolan-server-options",
    ]


def test_simple_stock_matrix_is_compact_with_price_visibility(db_session: Session) -> None:
    synced_at = datetime(2026, 6, 19, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="p1",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="HPE DL380 Gen11 ready server",
    )
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )

    position = package.payload["category_sections"][0]["positions"][0]
    assert package.payload["schema_version"] == "simple_stock_matrix.v8"
    assert "matrix_index" not in package.payload
    assert "matrix_policy" not in package.payload
    assert package.payload["matrix_note"].startswith("Complete stocked/priced")
    assert "product-level card" in package.payload["matrix_note"]
    assert "component_candidate_id only" in package.payload["matrix_note"]
    assert package.payload["diagnostics"]["mechanical_price_ordering"] is True
    assert package.payload["diagnostics"]["llm_visible_stock_row_ids"] is False
    assert "position_order" in package.payload["field_legend"]
    assert "price_rank_in_currency" in package.payload["field_legend"]
    assert "price_delta_vs_cheapest" in package.payload["field_legend"]
    assert "technical_price_index" in package.payload["field_legend"]
    assert "quote_quantity_limit" not in package.payload["field_legend"]
    assert "fact_refs" not in position
    assert position["component_candidate_id"] == "ocs:p1"
    assert "stock_row_id" not in position
    assert position["part_number"] == "PN-p1"
    assert "Vendor" in position["description"]
    assert "HPE DL380 Gen11" in position["description"]
    assert position["description"].count("HPE DL380 Gen11") == 1
    assert "title" not in position
    assert "name" not in position
    assert "key_facts" not in position
    assert "property_facts" not in position
    assert "unit_price" not in position
    assert "availability" not in position
    assert position["offers"][0]["price"] == {"value": "100.0000", "currency": "USD"}
    assert position["offers"][0]["available_quantity"] == 3
    assert position["price_rank_in_currency"] == {"USD": 1}
    assert position["price_delta_vs_cheapest"] == {"USD": "0"}
    stock_row = _simple_matrix_stock_row_by_item(package, "p1")
    assert stock_row["stock_row_id"].startswith("ocs:p1:")
    assert stock_row["price"] == {"value": "100.0000", "currency": "USD"}
    assert stock_row["stock"]["quantity_value"] == 3


def test_simple_stock_matrix_merges_exact_product_identity_offers(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 24, tzinfo=UTC)
    part_number = "M321R8GA0PB0-CWM"
    _seed_product(
        db_session,
        item_id="1000813252",
        category_id="cat-a",
        synced_at=synced_at,
        producer="Samsung",
        part_number=part_number,
        item_name="Samsung 64GB DDR5-5600 RDIMM USD card",
    )
    _seed_product(
        db_session,
        item_id="44000006787",
        category_id="cat-a",
        synced_at=synced_at,
        producer="samsung ",
        part_number=" M321R8GA0PB0-CWM ",
        item_name="Samsung 64GB DDR5-5600 RDIMM RUR card",
    )
    _seed_stock(
        db_session,
        item_id="1000813252",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("4200.0000"),
        price_order_currency="USD",
        quantity_value=100,
    )
    _seed_stock(
        db_session,
        item_id="44000006787",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("244331.9600"),
        price_order_currency="RUR",
        quantity_value=10,
    )
    db_session.commit()

    package = _simple_group_category_package(db_session, ["cat-a"])

    positions = package.payload["category_sections"][0]["positions"]
    assert len(positions) == 1
    position = positions[0]
    assert position["component_candidate_id"] == "ocs:1000813252"
    assert position["part_number"] == part_number
    assert position["source_item_count"] == 2
    assert package.payload["diagnostics"]["product_card_count"] == 1
    assert package.payload["diagnostics"]["merged_product_card_count"] == 1
    assert package.payload["diagnostics"]["merged_source_item_count"] == 1
    assert [offer["price"] for offer in position["offers"]] == [
        {"value": "244331.9600", "currency": "RUR"},
        {"value": "4200.0000", "currency": "USD"},
    ]

    stock_rows = sorted(package.stock_rows, key=lambda row: row["source_item_id"])
    assert [row["source_item_id"] for row in stock_rows] == [
        "1000813252",
        "44000006787",
    ]
    assert {row["component_candidate_id"] for row in stock_rows} == {"ocs:1000813252"}
    assert {row["stock_row_id"].split(":")[1] for row in stock_rows} == {
        "1000813252",
        "44000006787",
    }


def test_simple_stock_matrix_keeps_same_identity_separate_across_categories(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 24, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="ram-a",
        category_id="cat-a",
        synced_at=synced_at,
        producer="Samsung",
        part_number="M321R8GA0PB0-CWM",
    )
    _seed_product(
        db_session,
        item_id="ram-b",
        category_id="cat-b",
        synced_at=synced_at,
        producer="Samsung",
        part_number="M321R8GA0PB0-CWM",
    )
    _seed_stock(db_session, item_id="ram-a", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="ram-b", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _simple_group_category_package(db_session, ["cat-a", "cat-b"])

    positions = [
        position
        for section in package.payload["category_sections"]
        for position in section["positions"]
    ]
    assert len(positions) == 2
    assert {position["component_candidate_id"] for position in positions} == {
        "ocs:ram-a",
        "ocs:ram-b",
    }


def test_simple_stock_matrix_orders_positions_price_first(db_session: Session) -> None:
    synced_at = datetime(2026, 6, 19, tzinfo=UTC)
    _seed_product(db_session, item_id="expensive", category_id="cat-a", synced_at=synced_at)
    _seed_product(db_session, item_id="cheap", category_id="cat-a", synced_at=synced_at)
    _seed_product(db_session, item_id="rur", category_id="cat-a", synced_at=synced_at)
    _seed_stock(
        db_session,
        item_id="expensive",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("200.0000"),
    )
    _seed_stock(
        db_session,
        item_id="cheap",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("50.0000"),
    )
    _seed_stock(
        db_session,
        item_id="rur",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("10.0000"),
        price_order_currency="RUR",
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )

    positions = package.payload["category_sections"][0]["positions"]
    assert [position["component_candidate_id"] for position in positions] == [
        "ocs:cheap",
        "ocs:expensive",
        "ocs:rur",
    ]
    assert positions[0]["price_rank_in_currency"] == {"USD": 1}
    assert positions[0]["price_delta_vs_cheapest"] == {"USD": "0"}
    assert positions[1]["price_rank_in_currency"] == {"USD": 2}
    assert positions[1]["price_delta_vs_cheapest"] == {"USD": "+150"}
    assert positions[2]["price_rank_in_currency"] == {"RUR": 1}
    assert positions[2]["price_delta_vs_cheapest"] == {"RUR": "0"}
    assert package.payload["field_legend"]["position_order"].startswith(
        "Inside each category section"
    )


def test_simple_stock_matrix_adds_mechanical_technical_price_index(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 7, 1, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="dell-ram",
        category_id="cat-ram",
        synced_at=synced_at,
        producer="Dell",
        part_number="370-BBRN",
        item_name="64GB RDIMM, 5600MT/s, Dual Rank",
    )
    _seed_product(
        db_session,
        item_id="samsung-ram",
        category_id="cat-ram",
        synced_at=synced_at,
        producer="Samsung",
        part_number="M321R8GA0PB0-CWM",
        item_name="Samsung DDR5 64GB RDIMM 5600",
    )
    _seed_product(
        db_session,
        item_id="fc-hba",
        category_id="cat-net",
        synced_at=synced_at,
        producer="Emulex",
        part_number="LPE32002-M2",
        item_name="Emulex LPe32002-M2 Gen 6 32GFC 2-port PCIe HBA",
    )
    _seed_product(
        db_session,
        item_id="fc-hba-expensive",
        category_id="cat-net",
        synced_at=synced_at,
        producer="QLogic",
        part_number="QLE2742-SR-CK",
        item_name="QLogic QLE2742-SR-CK PCIe 2-port 32GFC adapter",
    )
    _seed_stock(
        db_session,
        item_id="dell-ram",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("2590.0000"),
        quantity_value=100,
        quantity_is_greater_than=True,
    )
    _seed_stock(
        db_session,
        item_id="samsung-ram",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("4635.0000"),
        quantity_value=100,
        quantity_is_greater_than=True,
    )
    _seed_stock(
        db_session,
        item_id="fc-hba",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("415.0000"),
        quantity_value=99,
    )
    _seed_stock(
        db_session,
        item_id="fc-hba-expensive",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("1095.0000"),
        quantity_value=60,
    )
    db_session.commit()

    package = _simple_group_category_package(db_session, ["cat-ram", "cat-net"])

    sections = {
        section["category_id"]: section for section in package.payload["category_sections"]
    }
    ram_index = {
        entry["token"]: entry for entry in sections["cat-ram"]["technical_price_index"]
    }
    assert ram_index["64GB"]["candidates"][0]["component_candidate_id"] == "ocs:dell-ram"
    assert ram_index["RDIMM"]["candidates"][0]["component_candidate_id"] == "ocs:dell-ram"
    assert ram_index["5600"]["candidates"][0]["component_candidate_id"] == "ocs:dell-ram"

    net_index = {
        entry["token"]: entry for entry in sections["cat-net"]["technical_price_index"]
    }
    assert net_index["32GFC"]["candidates"][0]["component_candidate_id"] == "ocs:fc-hba"
    assert net_index["2-port"]["candidates"][0]["component_candidate_id"] == "ocs:fc-hba"


def test_simple_stock_composer_sends_only_request_resolution_and_matrix_to_llm(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 25, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="server-base",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Universal rack system base platform",
    )
    _seed_product(
        db_session,
        item_id="fc-hba",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Emulex 32Gb 2-port FC HBA adapter",
    )
    _seed_product(
        db_session,
        item_id="unrelated",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Unrelated stocked product in the selected category",
    )
    _seed_stock(db_session, item_id="server-base", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="fc-hba", location="MSK", synced_at=synced_at)
    _seed_stock(db_session, item_id="unrelated", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _simple_group_category_package(db_session, ["cat-a"])
    prompts = build_simple_stock_quote_prompts(
        user_request="Need one rack system with 1 x 32Gb 2-port FC HBA",
        resolved_request={
            "customer_task_summary": "Need a rack system with an HBA",
            "category_ids": ["cat-a"],
        },
        matrix_package=package,
        model="qwen/qwen3.7-max",
    )

    payload = json.loads(prompts.user_prompt)
    matrix_ids = {
        position["component_candidate_id"]
        for section in payload["stock_matrix"]["category_sections"]
        for position in section["positions"]
    }

    assert set(payload) == {
        "original_request_text",
        "resolved_request",
        "composer_profile",
        "stock_matrix",
    }
    assert prompts.composer_profile == "clean_role_first"
    assert "requirement_ledger" not in payload
    assert "candidate_highlights" not in payload
    assert "target_anchor_highlights" not in payload
    assert "commercial_candidates" not in payload
    assert "evidence_candidates" not in payload
    assert "evidence_review" not in prompts.system_prompt
    assert "requirement_ledger" not in prompts.system_prompt
    assert "evidence candidates" not in prompts.system_prompt
    assert "ocs:server-base" in matrix_ids
    assert "ocs:fc-hba" in matrix_ids
    assert "ocs:unrelated" in matrix_ids


def test_simple_stock_composer_keeps_low_price_and_noise_in_full_matrix(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 24, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="expensive-memory",
        category_id="cat-ram",
        synced_at=synced_at,
        item_name="Vendor 64GB DDR5-5600 RDIMM ECC memory module",
    )
    _seed_product(
        db_session,
        item_id="cheap-memory",
        category_id="cat-ram",
        synced_at=synced_at,
        item_name="Value 64GB DDR5-5600 RDIMM ECC memory module",
    )
    _seed_product(
        db_session,
        item_id="unrelated-cheap",
        category_id="cat-ram",
        synced_at=synced_at,
        item_name="Cheap office keyboard",
    )
    _seed_stock(
        db_session,
        item_id="expensive-memory",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("3800.0000"),
        quantity_value=4,
    )
    _seed_stock(
        db_session,
        item_id="cheap-memory",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("2520.0000"),
        quantity_value=100,
    )
    _seed_stock(
        db_session,
        item_id="unrelated-cheap",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("5.0000"),
        quantity_value=100,
    )
    db_session.commit()

    package = _simple_group_category_package(db_session, ["cat-ram"])
    prompts = build_simple_stock_quote_prompts(
        user_request="Need 48 x 64GB DDR5-5600 RDIMM ECC memory modules",
        resolved_request={"category_ids": ["cat-ram"]},
        matrix_package=package,
    )

    payload = json.loads(prompts.user_prompt)
    matrix_ids = {
        position["component_candidate_id"]
        for section in payload["stock_matrix"]["category_sections"]
        for position in section["positions"]
    }

    assert "ocs:expensive-memory" in matrix_ids
    assert "ocs:cheap-memory" in matrix_ids
    assert "ocs:unrelated-cheap" in matrix_ids
    assert "requirement_ledger" not in payload
    assert "commercial_candidates" not in payload


def test_simple_stock_composer_keeps_target_anchor_candidates_in_matrix(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 23, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="server-base",
        category_id="cat-server",
        synced_at=synced_at,
        item_name=(
            "Сервер HP ProLiant DL380 Gen11/ DL380Gen11 4510 "
            "HP-SATA (8/24 SFF max) 4x1Gb 2x1000W"
        ),
    )
    _seed_product(
        db_session,
        item_id="drive-cage",
        category_id="cat-cage",
        synced_at=synced_at,
        item_name=(
            'Дисковая корзина 8*2.5"/ HPE ProLiant DL380 Gen11 '
            "2U 8SFF x1 Tri-Mode U.3 Drive Cage Kit"
        ),
    )
    _seed_product(
        db_session,
        item_id="heatsink",
        category_id="cat-heatsink",
        synced_at=synced_at,
        item_name="Радиатор/ HPE ProLiant DL380/DL560 Gen11 High Performance 2U Heat Sink Kit",
    )
    _seed_product(
        db_session,
        item_id="fan-kit",
        category_id="cat-fan",
        synced_at=synced_at,
        item_name=(
            "Корпусной вентилятор/ HPE ProLiant DL380/DL560 Gen11 "
            "2U High Performance Fan Kit"
        ),
    )
    _seed_product(
        db_session,
        item_id="riser",
        category_id="cat-riser",
        synced_at=synced_at,
        item_name="Райзер-карта HPE ProLiant DL380 Gen11 x16/x16/x16 Primary Cable Kit",
    )
    _seed_stock(
        db_session,
        item_id="server-base",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("9700.0000"),
        quantity_value=2,
    )
    _seed_stock(
        db_session,
        item_id="drive-cage",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("390.0000"),
    )
    _seed_stock(
        db_session,
        item_id="heatsink",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("132.0000"),
    )
    _seed_stock(
        db_session,
        item_id="fan-kit",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("490.0000"),
    )
    _seed_stock(
        db_session,
        item_id="riser",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("7184.6000"),
        price_order_currency="RUR",
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(
        repository.list_latest_full_category_group_matrix(
            "ocs",
            ["cat-server", "cat-cage", "cat-heatsink", "cat-fan", "cat-riser"],
        )
    )
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-server", "cat-cage", "cat-heatsink", "cat-fan", "cat-riser"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )

    prompts = build_simple_stock_quote_prompts(
        user_request="Подобрать 3 сервера HPE ProLiant DL380 Gen11 8SFF.",
        resolved_request={
            "target_objects": [
                {
                    "target_id": "T1",
                    "label": "3 сервера HPE ProLiant DL380 Gen11 8SFF с полной конфигурацией",
                    "quantity": 3,
                    "expects_anchor_line": True,
                    "requirement_ids": ["R1"],
                }
            ],
            "must_have": [
                {
                    "requirement_id": "R1",
                    "requirement": "HPE ProLiant DL380 Gen11 8SFF",
                }
            ],
        },
        matrix_package=package,
    )

    payload = json.loads(prompts.user_prompt)
    matrix_ids = {
        position["component_candidate_id"]
        for section in payload["stock_matrix"]["category_sections"]
        for position in section["positions"]
    }

    assert "target_anchor_highlights" not in payload
    assert payload["resolved_request"]["target_objects"][0]["expects_anchor_line"] is True
    assert "ocs:server-base" in matrix_ids
    assert "ocs:drive-cage" in matrix_ids
    assert "ocs:heatsink" in matrix_ids
    assert "ocs:fan-kit" in matrix_ids


def test_simple_stock_composer_accepts_llm_quote_with_integrity_reconciliation(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 19, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    position = package.payload["category_sections"][0]["positions"][0]
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "client_status_label": "Черновик КП",
                "selection_mode": "partial_stock_offer",
                "completeness_status": "partial",
                "operational_status": "incomplete_needs_sourcing",
                "client_summary": "Максимум со склада.",
                "coverage_summary": "Частично покрыто.",
                "why_selected": "Выбрана доступная складская строка.",
                "target_decisions": [
                    {
                        "target_id": "T1",
                        "target_label": "Requested system",
                        "anchor_status": "selected",
                        "anchor_line_id": "L1",
                        "reason": "Selected stocked anchor row.",
                    }
                ],
                "lines": [
                    {
                        "component_candidate_id": position["component_candidate_id"],
                        "role": "Складская позиция",
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "procurement_gaps": [
                    {
                        "requested": "Недостающий компонент",
                        "needed_action": "source_from_other_stock",
                        "included_in_total": False,
                    }
                ],
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Нужен сервер",
        resolved_request={
            "customer_task_summary": "Нужен сервер",
            "target_objects": [
                {
                    "target_id": "T1",
                    "label": "Requested system",
                    "quantity": 1,
                    "expects_anchor_line": True,
                }
            ],
        },
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.pipeline_version == "simple_stock_quote"
    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.primary_recommendation_status == "llm_final"
    assert outcome.validation_errors == []
    assert outcome.validation_warnings == []
    assert outcome.validated_quote["lines"][0]["component_candidate_id"] == "ocs:p1"
    assert outcome.validated_quote["lines"][0]["stock_row_id"].startswith("ocs:p1:")
    assert outcome.validated_quote["target_decisions"][0]["anchor_status"] == "selected"
    system_prompt, user_prompt = fake_client.calls[0]
    assert outcome.diagnostics["composer_prompt_version"] == "simple_stock_composer_v70"
    assert "requirement_ledger_version" not in outcome.diagnostics
    assert "requirement_ledger_product_candidates" not in outcome.diagnostics
    assert "requirement_ledger_evidence_candidate_count" not in outcome.diagnostics
    assert "candidate_highlights_version" not in outcome.diagnostics
    assert "target_anchor_highlights_version" not in outcome.diagnostics
    assert len(system_prompt) < 6_900
    assert "Stock Configurator Composer" in system_prompt
    assert "original_request_text: customer request and source of truth" in system_prompt
    assert "stock_matrix: complete product cards" in system_prompt
    assert "requirement_ledger" not in system_prompt
    assert "build the most useful Russian draft quote" in system_prompt
    assert "One lines[] item is one selected product candidate" in system_prompt
    assert "do not return stock_row_id" in system_prompt
    assert "compare all relevant stocked products" in system_prompt
    assert "price_rank_in_currency" in system_prompt
    assert "price_delta_vs_cheapest" in system_prompt
    assert "technical_price_index" in system_prompt
    assert "mechanical recall aid" in system_prompt
    assert "multiple specific token buckets" in system_prompt
    assert "numeric and structural matrix facts" in system_prompt
    assert "stronger evidence than exact wording" in system_prompt
    assert "lacks one expected wording token" in system_prompt
    assert "Missing wording is an engineer_check" in system_prompt
    assert "Concrete conflicts override price" in system_prompt
    assert "Exact part number, exact OEM or no-analog" in system_prompt
    assert "Brand reputation" in system_prompt
    assert "vendor is not a premium reason" in system_prompt
    assert "Цена: выбран самый дешевый сопоставимый вариант." in system_prompt
    assert "If no matrix-grounded disqualifier exists" in system_prompt
    assert "first evaluate stocked\n  ready, preconfigured or assembled products" in system_prompt
    assert "Use component-only coverage only when" in system_prompt
    assert "Low-value mounting/install details" in system_prompt
    assert "not separate\n  quote/gap roles" in system_prompt
    assert "no stocked base reasonably matches the target object" in system_prompt
    assert "lower-bound stock such as \"1+\"" in system_prompt
    assert "Uncertainty is not\n  a procurement gap by itself" in system_prompt
    assert "Platform-proprietary fit overrides generic similarity" in system_prompt
    assert "do not count a generic analog as covered" in system_prompt
    assert "Do not say a" in system_prompt
    assert "role is covered when the line itself depends" in system_prompt
    assert "The platform-fit rule is two-sided" in system_prompt
    assert "treat it as a strong positive candidate" in system_prompt
    assert "Do not create an \"absent from stock\" procurement_gap" in system_prompt
    assert "target_decisions" not in system_prompt
    assert "available_alternatives" not in system_prompt
    assert "considered_candidates" not in system_prompt
    assert "candidate_highlights" not in system_prompt
    assert "target_anchor_highlights" not in system_prompt
    assert "commercial_candidates" not in system_prompt
    assert "candidate_highlights" not in user_prompt
    assert "target_anchor_highlights" not in user_prompt
    assert "commercial_candidates" not in user_prompt
    assert "requirement_ledger" not in user_prompt
    assert "stock_matrix" in user_prompt


def test_simple_stock_composer_normalizes_selected_anchor_target_status(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 25, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="base-system",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Universal rack system base platform",
    )
    _seed_stock(db_session, item_id="base-system", location="MSK", synced_at=synced_at)
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-max",
    )
    position = _simple_matrix_position_by_item(package, "base-system")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "KP draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": position["component_candidate_id"],
                        "role": "Base platform",
                        "fit_status": "analog",
                        "quantity": 1,
                    }
                ],
                "target_decisions": [
                    {
                        "target_id": "T1",
                        "target_label": "Requested base platform",
                        "anchor_candidate_id": position["component_candidate_id"],
                        "anchor_status": "alternative",
                        "anchor_line_id": "L1",
                        "reason": "Selected as an analog base.",
                    }
                ],
                "procurement_gaps": [],
                "available_alternatives": [],
                "engineer_checks": [],
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need one rack system",
        resolved_request={
            "target_objects": [
                {
                    "target_id": "T1",
                    "label": "Requested base platform",
                    "quantity": 1,
                    "expects_anchor_line": True,
                }
            ],
        },
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_model="qwen/qwen3.7-max"),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validated_quote["target_decisions"][0]["anchor_status"] == "selected"
    assert outcome.validated_quote["target_decisions"][0]["anchor_line_id"] == "L1"
    assert outcome.validated_quote["lines"][0]["fit_status"] == "analog"
    assert outcome.validated_quote["quote_integrity"]["version"] == "quote_integrity_reconciler_v9"
    assert outcome.validated_quote["quote_integrity"]["status"] == "mechanically_adjusted"
    assert {
        adjustment["type"]
        for adjustment in outcome.validated_quote["quote_integrity"]["adjustments"]
    } == {"target_decision_anchor_status_normalized"}


def test_simple_stock_composer_accepts_llm_semantics_without_coverage_audit(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 25, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="base-system",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Universal rack system base platform",
    )
    _seed_stock(db_session, item_id="base-system", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _simple_group_category_package(db_session, ["cat-a"])
    position = _simple_matrix_position_by_item(package, "base-system")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "Quote draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": position["component_candidate_id"],
                        "role": "Main system",
                        "quantity": 1,
                    }
                ],
                "procurement_gaps": [],
                "available_alternatives": [],
                "engineer_checks": [],
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need one rack system with 1 x 32Gb 2-port FC HBA",
        resolved_request={
            "must_have": [
                {"requirement_id": "R_BASE", "requirement": "rack system base platform"},
                {"requirement_id": "R_HBA", "requirement": "32Gb 2-port FC HBA"},
            ],
        },
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_model="qwen/qwen3.7-max"),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == []
    assert "requirement_coverage_audit" not in outcome.validated_quote
    assert "requirement_coverage_audit_status" not in outcome.diagnostics
    assert "quote_integrity.requirement_ledger_unaddressed" not in outcome.validation_warnings


def test_simple_stock_composer_reconciles_stock_quantity_price_and_total(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 22, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="ram-micron",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Micron 64GB DDR5-5600 RDIMM",
    )
    _seed_product(
        db_session,
        item_id="ram-samsung",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Samsung 64GB DDR5-5600 RDIMM",
    )
    _seed_stock(
        db_session,
        item_id="ram-micron",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("3299.9500"),
        quantity_value=10,
    )
    _seed_stock(
        db_session,
        item_id="ram-samsung",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("3599.9500"),
        quantity_value=10,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    micron_position = _simple_matrix_position_by_item(package, "ram-micron")
    samsung_position = _simple_matrix_position_by_item(package, "ram-samsung")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": micron_position["component_candidate_id"],
                        "stock_row_id": micron_position["stock_row_id"],
                        "role": "RAM 64GB DDR5-5600 RDIMM",
                        "quantity": 48,
                        "unit_price_value": "1.00",
                        "unit_price_currency": "EUR",
                        "line_total_value": "48.00",
                        "line_total_currency": "EUR",
                    }
                ],
                "available_alternatives": [
                    {
                        "requirement_id": "R_RAM",
                        "item": "Samsung RAM",
                        "component_candidate_id": samsung_position["component_candidate_id"],
                        "stock_row_id": samsung_position["stock_row_id"],
                        "available_quantity": 99,
                        "unit_price_value": "1.00",
                        "unit_price_currency": "EUR",
                        "reason": "Alternative RAM row.",
                    }
                ],
                "total_price_value": "999999.00",
                "total_price_currency": "EUR",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need 48 x 64GB DDR5 RAM",
        resolved_request={"customer_task_summary": "Need 48 x 64GB DDR5 RAM"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == ["quote_integrity.stock_overallocation_adjusted"]
    assert outcome.diagnostics["quote_integrity_status"] == "mechanically_adjusted"

    line = outcome.validated_quote["lines"][0]
    assert line["component_candidate_id"] == micron_position["component_candidate_id"]
    assert line["stock_row_id"] == micron_position["stock_row_id"]
    assert line["quantity"] == 10
    assert line["available_quantity"] == 10
    assert line["part_number"] == "PN-ram-micron"
    assert line["unit_price_value"] == "3299.9500"
    assert line["unit_price_currency"] == "USD"
    assert line["line_total_value"] == "32999.5000"
    assert line["line_total_currency"] == "USD"
    assert outcome.validated_quote["total_price_value"] == "32999.5000"
    assert outcome.validated_quote["total_price_currency"] == "USD"

    gap = outcome.validated_quote["procurement_gaps"][0]
    assert gap["item"] == "RAM 64GB DDR5-5600 RDIMM"
    assert gap["quantity"] == 38
    assert "закрыто 10" in gap["reason"]

    alternative = outcome.validated_quote["available_alternatives"][0]
    assert alternative["requirement_id"] == "R_RAM"
    assert alternative["component_candidate_id"] == samsung_position["component_candidate_id"]
    assert alternative["stock_row_id"] == samsung_position["stock_row_id"]
    assert alternative["part_number"] == "PN-ram-samsung"
    assert alternative["available_quantity"] == 10
    assert alternative["unit_price_value"] == "3599.9500"
    assert alternative["unit_price_currency"] == "USD"
    assert outcome.validated_quote["quote_integrity"]["status"] == "mechanically_adjusted"


def test_simple_stock_composer_keeps_lower_bound_stock_quantity_with_confirmation(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 25, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="dl380-base",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="HPE ProLiant DL380 Gen11 8SFF configured server",
    )
    _seed_stock(
        db_session,
        item_id="dl380-base",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("9700.0000"),
        quantity_value=1,
        quantity_is_greater_than=True,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    position = _simple_matrix_position_by_item(package, "dl380-base")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "coverage_summary": "Базовая платформа подобрана.",
                "lines": [
                    {
                        "line_id": "L1",
                        "requirement_id": "R_BASE",
                        "component_candidate_id": position["component_candidate_id"],
                        "role": "Базовый сервер / платформа",
                        "quantity": 3,
                        "unit_price_currency": "USD",
                    }
                ],
                "procurement_gaps": [],
                "available_alternatives": [],
                "engineer_checks": [],
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need 3 HPE ProLiant DL380 Gen11 8SFF servers",
        resolved_request={"customer_task_summary": "Need 3 HPE ProLiant DL380 Gen11 8SFF"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_model="qwen/qwen3.7-plus"),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == [
        "quote_integrity.stock_lower_bound_requires_confirmation"
    ]
    assert outcome.diagnostics["quote_integrity_status"] == "mechanically_adjusted"

    line = outcome.validated_quote["lines"][0]
    assert line["quantity"] == 3
    assert line["available_quantity"] == 3
    assert line["quantity_value"] == 1
    assert line["quantity_is_greater_than"] is True
    assert line["stock_confirmation_required"] is True
    assert "подтвердить доступность" in line["stock_confirmation_note"]
    assert "quantity_adjusted" not in line
    assert outcome.validated_quote["procurement_gaps"] == []
    assert outcome.validated_quote["total_price_value"] == "29100.0000"
    assert outcome.validated_quote["total_price_currency"] == "USD"
    assert "stock_overallocation_adjusted" not in outcome.validation_warnings


def test_simple_stock_composer_allocates_merged_identity_by_selected_currency(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 24, tzinfo=UTC)
    part_number = "M321R8GA0PB0-CWM"
    _seed_product(
        db_session,
        item_id="1000813252",
        category_id="cat-a",
        synced_at=synced_at,
        producer="Samsung",
        part_number=part_number,
        item_name="Samsung 64GB DDR5-5600 RDIMM USD card",
    )
    _seed_product(
        db_session,
        item_id="44000006787",
        category_id="cat-a",
        synced_at=synced_at,
        producer="Samsung",
        part_number=part_number,
        item_name="Samsung 64GB DDR5-5600 RDIMM RUR card",
    )
    _seed_stock(
        db_session,
        item_id="1000813252",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("4200.0000"),
        price_order_currency="USD",
        quantity_value=100,
    )
    _seed_stock(
        db_session,
        item_id="44000006787",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("244331.9600"),
        price_order_currency="RUR",
        quantity_value=10,
    )
    db_session.commit()

    package = _simple_group_category_package(db_session, ["cat-a"])
    position = package.payload["category_sections"][0]["positions"][0]
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": position["component_candidate_id"],
                        "role": "RAM 64GB DDR5-5600 RDIMM",
                        "quantity": 48,
                        "selected_currency": "RUR",
                    }
                ],
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need 48 x 64GB DDR5 RAM",
        resolved_request={"customer_task_summary": "Need 48 x 64GB DDR5 RAM"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == []
    assert outcome.validated_quote["quote_integrity"]["status"] == "mechanically_adjusted"
    assert outcome.validated_quote["quote_integrity"]["adjustments"] == [
        {
            "type": "component_quantity_split",
            "section": "line",
            "index": 1,
            "component_candidate_id": "ocs:1000813252",
            "requested_quantity": 48,
            "stock_row_count": 2,
        }
    ]

    first_line, second_line = outcome.validated_quote["lines"]
    assert first_line["stock_row_id"].startswith("ocs:44000006787:")
    assert first_line["quantity"] == 10
    assert first_line["unit_price_value"] == "244331.9600"
    assert first_line["unit_price_currency"] == "RUR"
    assert first_line["line_total_value"] == "2443319.6000"
    assert second_line["stock_row_id"].startswith("ocs:1000813252:")
    assert second_line["quantity"] == 38
    assert second_line["unit_price_value"] == "4200.0000"
    assert second_line["unit_price_currency"] == "USD"
    assert second_line["line_total_value"] == "159600.0000"
    assert outcome.validated_quote["total_price_value"] is None
    assert outcome.validated_quote["total_price_currency"] is None
    assert outcome.validated_quote["totals_by_currency"] == [
        {"currency": "RUR", "value": "2443319.6000"},
        {"currency": "USD", "value": "159600.0000"},
    ]

    stale_price_currency_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": position["component_candidate_id"],
                        "role": "RAM 64GB DDR5-5600 RDIMM",
                        "quantity": 48,
                        "unit_price_currency": "USD",
                        "line_total_currency": "USD",
                    }
                ],
            },
        }
    )
    no_selected_currency_outcome = compose_simple_stock_quote(
        user_request="Need 48 x 64GB DDR5 RAM",
        resolved_request={"customer_task_summary": "Need 48 x 64GB DDR5 RAM"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=stale_price_currency_client,
    )

    first_line_without_selection = no_selected_currency_outcome.validated_quote["lines"][0]
    assert first_line_without_selection["stock_row_id"].startswith("ocs:44000006787:")
    assert first_line_without_selection["unit_price_currency"] == "RUR"


def test_simple_stock_composer_reconciles_procurement_gap_considered_candidates(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 22, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="server-base",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="HPE DL380 base server",
    )
    _seed_product(
        db_session,
        item_id="ssd-alt",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Dell 1.92TB SAS RI SFF SSD",
    )
    _seed_stock(
        db_session,
        item_id="server-base",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("4767.00"),
        quantity_value=3,
    )
    _seed_stock(
        db_session,
        item_id="ssd-alt",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("1395.00"),
        quantity_value=10,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    base_position = _simple_matrix_position_by_item(package, "server-base")
    ssd_position = _simple_matrix_position_by_item(package, "ssd-alt")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": base_position["component_candidate_id"],
                        "stock_row_id": base_position["stock_row_id"],
                        "role": "Server base",
                        "quantity": 1,
                        "unit_price_value": "1.00",
                        "unit_price_currency": "EUR",
                        "line_total_value": "1.00",
                        "line_total_currency": "EUR",
                    }
                ],
                "procurement_gaps": [
                    {
                        "requirement_id": "R_SSD",
                        "item": "SSD SATA RI 1.92TB SFF",
                        "quantity": 6,
                        "reason": "Exact SATA SSD is not present in the matrix.",
                        "considered_candidates": [
                            {
                                "component_candidate_id": ssd_position["component_candidate_id"],
                                "stock_row_id": "ocs:ssd-alt:stale",
                                "item": "Dell SSD stale name",
                                "available_quantity": 99,
                                "unit_price_value": "9999.00",
                                "unit_price_currency": "EUR",
                                "reason": "SAS alternative, not exact SATA.",
                            }
                        ],
                    }
                ],
                "total_price_value": "1.00",
                "total_price_currency": "EUR",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need base plus SSD",
        resolved_request={"customer_task_summary": "Need base plus SSD"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == []
    assert outcome.diagnostics["quote_integrity_status"] == "ok"

    gap = outcome.validated_quote["procurement_gaps"][0]
    candidate = gap["considered_candidates"][0]
    assert candidate["component_candidate_id"] == ssd_position["component_candidate_id"]
    assert candidate["stock_row_id"] == ssd_position["stock_row_id"]
    assert candidate["part_number"] == "PN-ssd-alt"
    assert "Dell 1.92TB SAS RI SFF SSD" in candidate["item"]
    assert candidate["available_quantity"] == 10
    assert candidate["unit_price_value"] == "1395.0000"
    assert candidate["unit_price_currency"] == "USD"
    assert candidate["reason"] == "SAS alternative, not exact SATA."

    assert outcome.validated_quote["quote_integrity"]["adjustments"] == []


def test_simple_stock_composer_merges_adjusted_quantity_into_requirement_gap(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 22, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="ram-micron",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Micron 64GB DDR5-5600 RDIMM",
    )
    _seed_stock(
        db_session,
        item_id="ram-micron",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("3299.9500"),
        quantity_value=10,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    micron_position = _simple_matrix_position_by_item(package, "ram-micron")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "requirement_id": "R_RAM",
                        "component_candidate_id": micron_position["component_candidate_id"],
                        "stock_row_id": micron_position["stock_row_id"],
                        "role": "RAM 64GB DDR5-5600 RDIMM",
                        "quantity": 20,
                        "unit_price_value": "1.00",
                        "unit_price_currency": "EUR",
                        "line_total_value": "20.00",
                        "line_total_currency": "EUR",
                        "reason": "Available quantity is 20 of 48.",
                    }
                ],
                "procurement_gaps": [
                    {
                        "requirement_id": "R_RAM",
                        "item": "RAM 64GB DDR5-5600 RDIMM",
                        "quantity": 28,
                        "reason": "Initial LLM shortage.",
                    }
                ],
                "total_price_value": "20.00",
                "total_price_currency": "EUR",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need 48 x 64GB DDR5 RAM",
        resolved_request={"customer_task_summary": "Need 48 x 64GB DDR5 RAM"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.diagnostics["quote_integrity_status"] == "mechanically_adjusted"

    line = outcome.validated_quote["lines"][0]
    assert line["quantity"] == 10
    assert line["reason"] == "Количество скорректировано по доступному складскому остатку."
    assert line["reconciliation_note"] == line["reason"]
    assert line["quantity_adjusted"] is True
    assert line["original_requested_quantity"] == 20
    assert line["shortage_quantity"] == 10

    gaps = outcome.validated_quote["procurement_gaps"]
    assert len(gaps) == 1
    assert gaps[0]["requirement_id"] == "R_RAM"
    assert gaps[0]["quantity"] == 38
    assert gaps[0]["internal_key"] == (
        f"quantity_shortage:R_RAM:{micron_position['component_candidate_id']}"
    )
    assert gaps[0]["component_candidate_id"] == micron_position["component_candidate_id"]
    assert "RAM 64GB DDR5-5600 RDIMM - " in outcome.validated_quote["coverage_summary"]
    assert "не закрыто 10 шт." in gaps[0]["reason"]

    adjustment = outcome.validated_quote["quote_integrity"]["adjustments"][0]
    assert adjustment["requirement_id"] == "R_RAM"


def test_simple_stock_composer_deduplicates_existing_llm_stock_shortage_gap(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 23, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="dl380-base",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="HPE ProLiant DL380 Gen11 8SFF configured server",
    )
    for location in ("MSK", "SPB"):
        _seed_stock(
            db_session,
            item_id="dl380-base",
            location=location,
            synced_at=synced_at,
            price_order_value=Decimal("9700.00"),
            quantity_value=1,
        )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    base_position = _simple_matrix_position_by_item(package, "dl380-base")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "KP draft",
                "selection_mode": "partial_stock_offer",
                "coverage_summary": "Base servers partially covered.",
                "lines": [
                    {
                        "line_id": "L1",
                        "requirement_id": "R_BASE",
                        "component_candidate_id": base_position["component_candidate_id"],
                        "role": "server base chassis",
                        "quantity": 3,
                        "unit_price_currency": "USD",
                    }
                ],
                "procurement_gaps": [
                    {
                        "requirement_id": "R_BASE",
                        "item": "HPE ProLiant DL380 Gen11 8SFF base chassis",
                        "quantity": 1,
                        "reason": "Only 2 of 3 requested bases are available.",
                    }
                ],
                "total_price_value": "29100.00",
                "total_price_currency": "USD",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need 3 HPE ProLiant DL380 Gen11 8SFF servers",
        resolved_request={"customer_task_summary": "Need 3 HPE ProLiant DL380 Gen11 8SFF"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == ["quote_integrity.stock_overallocation_adjusted"]
    assert outcome.diagnostics["quote_integrity_status"] == "mechanically_adjusted"

    gaps = outcome.validated_quote["procurement_gaps"]
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["requirement_id"] == "R_BASE"
    assert gap["quantity"] == 1
    assert gap["component_candidate_id"] == base_position["component_candidate_id"]
    assert gap["duplicate_shortage_deduped"] is True
    assert gap["internal_key"] == (
        f"quantity_shortage:R_BASE:{base_position['component_candidate_id']}"
    )


def test_simple_stock_composer_keeps_unlike_shortage_gaps_separate(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 23, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="fan-kit",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="HPE Gen11 High Performance Fan Kit",
    )
    _seed_product(
        db_session,
        item_id="heatsink",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="HPE Gen11 High Performance Heat Sink Kit",
    )
    for location in ("MSK", "SPB"):
        _seed_stock(
            db_session,
            item_id="fan-kit",
            location=location,
            synced_at=synced_at,
            price_order_value=Decimal("490.00"),
            quantity_value=1,
        )
        _seed_stock(
            db_session,
            item_id="heatsink",
            location=location,
            synced_at=synced_at,
            price_order_value=Decimal("132.00"),
            quantity_value=1,
        )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    fan_position = _simple_matrix_position_by_item(package, "fan-kit")
    heatsink_position = _simple_matrix_position_by_item(package, "heatsink")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "KP draft",
                "selection_mode": "partial_stock_offer",
                "coverage_summary": "Accessories are covered.",
                "lines": [
                    {
                        "line_id": "L1",
                        "requirement_id": "R_ACCESSORIES",
                        "component_candidate_id": fan_position["component_candidate_id"],
                        "role": "High performance fans",
                        "quantity": 3,
                        "unit_price_currency": "USD",
                    },
                    {
                        "line_id": "L2",
                        "requirement_id": "R_ACCESSORIES",
                        "component_candidate_id": heatsink_position["component_candidate_id"],
                        "role": "Heatsinks",
                        "quantity": 3,
                        "unit_price_currency": "USD",
                    },
                ],
                "procurement_gaps": [
                    {
                        "requirement_id": "R_ACCESSORIES",
                        "item": "iLO Advanced license 3yr, Riser, Rails, CMA",
                        "quantity": 3,
                        "reason": "Initial accessory gap.",
                    }
                ],
                "total_price_value": "1866.00",
                "total_price_currency": "USD",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need server accessories",
        resolved_request={"customer_task_summary": "Need server accessories"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == ["quote_integrity.stock_overallocation_adjusted"]
    assert outcome.diagnostics["quote_integrity_status"] == "mechanically_adjusted"

    gaps = outcome.validated_quote["procurement_gaps"]
    gaps_by_item = {gap["item"]: gap for gap in gaps}
    assert set(gaps_by_item) == {
        "iLO Advanced license 3yr, Riser, Rails, CMA",
        "High performance fans",
        "Heatsinks",
    }
    assert gaps_by_item["iLO Advanced license 3yr, Riser, Rails, CMA"]["quantity"] == 3
    assert gaps_by_item["High performance fans"]["quantity"] == 1
    assert gaps_by_item["Heatsinks"]["quantity"] == 1
    assert gaps_by_item["High performance fans"]["component_candidate_id"] == (
        fan_position["component_candidate_id"]
    )
    assert gaps_by_item["Heatsinks"]["component_candidate_id"] == (
        heatsink_position["component_candidate_id"]
    )
    assert gaps_by_item["High performance fans"]["internal_key"] == (
        f"quantity_shortage:R_ACCESSORIES:{fan_position['component_candidate_id']}"
    )
    assert gaps_by_item["Heatsinks"]["internal_key"] == (
        f"quantity_shortage:R_ACCESSORIES:{heatsink_position['component_candidate_id']}"
    )
    assert "Accessories are covered." in outcome.validated_quote["coverage_summary"]
    assert "High performance fans - " in outcome.validated_quote["coverage_summary"]
    assert "Heatsinks - " in outcome.validated_quote["coverage_summary"]


def test_simple_stock_composer_rejects_mismatched_selected_id_pair(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 20, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="hba",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="HPE SN1100Q 16Gb Dual Port FC HBA",
    )
    _seed_product(
        db_session,
        item_id="hdd",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Toshiba HDD 10TB",
    )
    _seed_stock(
        db_session,
        item_id="hba",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("1176.00"),
        quantity_value=10,
    )
    _seed_stock(
        db_session,
        item_id="hdd",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("612.00"),
        quantity_value=3,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    hba_position = _simple_matrix_position_by_item(package, "hba")
    hdd_position = _simple_matrix_position_by_item(package, "hdd")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": hba_position["component_candidate_id"],
                        "stock_row_id": hdd_position["stock_row_id"],
                        "role": "FC HBA",
                        "quantity": 3,
                        "unit_price_value": "1176.00",
                        "unit_price_currency": "USD",
                        "line_total_value": "3528.00",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "999999.00",
                "total_price_currency": "USD",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need 3 x FC HBA",
        resolved_request={"customer_task_summary": "Need 3 x FC HBA"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.status == "quote"
    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_errors == []
    assert outcome.validation_warnings == ["quote_integrity.llm_stock_row_id_ignored"]
    line = outcome.validated_quote["lines"][0]
    assert line["component_candidate_id"] == hba_position["component_candidate_id"]
    assert line["stock_row_id"] == hba_position["stock_row_id"]
    assert line["unit_price_value"] == "1176.0000"
    adjustment = outcome.validated_quote["quote_integrity"]["adjustments"][0]
    assert adjustment["type"] == "llm_stock_row_id_ignored"
    assert adjustment["supplied_stock_row_id"] == hdd_position["stock_row_id"]
    assert adjustment["component_candidate_id"] == hba_position["component_candidate_id"]
    assert outcome.diagnostics["quote_integrity_status"] == "mechanically_adjusted"


def test_simple_stock_composer_repairs_line_stock_row_id_from_unique_component_id(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 22, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="hba",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="HPE SN1100Q 16Gb Dual Port FC HBA",
    )
    _seed_stock(
        db_session,
        item_id="hba",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("1176.00"),
        quantity_value=10,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    hba_position = _simple_matrix_position_by_item(package, "hba")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": hba_position["component_candidate_id"],
                        "stock_row_id": "ocs:hba:stale",
                        "part_number": "PN-hba",
                        "role": "FC HBA",
                        "quantity": 3,
                        "unit_price_value": "1.00",
                        "unit_price_currency": "EUR",
                        "line_total_value": "3.00",
                        "line_total_currency": "EUR",
                    }
                ],
                "total_price_value": "3.00",
                "total_price_currency": "EUR",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need 3 x FC HBA",
        resolved_request={"customer_task_summary": "Need 3 x FC HBA"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_errors == []
    assert outcome.validation_warnings == ["quote_integrity.llm_stock_row_id_ignored"]
    assert outcome.diagnostics["quote_integrity_status"] == "mechanically_adjusted"

    line = outcome.validated_quote["lines"][0]
    assert line["component_candidate_id"] == hba_position["component_candidate_id"]
    assert line["stock_row_id"] == hba_position["stock_row_id"]
    assert line["unit_price_value"] == "1176.0000"
    assert line["line_total_value"] == "3528.0000"

    adjustment = outcome.validated_quote["quote_integrity"]["adjustments"][0]
    assert adjustment == {
        "type": "llm_stock_row_id_ignored",
        "section": "line",
        "index": 1,
        "component_candidate_id": hba_position["component_candidate_id"],
        "supplied_stock_row_id": "ocs:hba:stale",
        "resolution": "allocated_by_component_candidate_id",
    }


def test_simple_stock_composer_repairs_available_alternative_stock_row_id(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 22, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="hba-main",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Primary FC HBA",
    )
    _seed_product(
        db_session,
        item_id="hba-alt",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Alternative FC HBA",
    )
    _seed_stock(
        db_session,
        item_id="hba-main",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("1500.00"),
        quantity_value=5,
    )
    _seed_stock(
        db_session,
        item_id="hba-alt",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("1176.00"),
        quantity_value=1,
        quantity_is_greater_than=True,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    main_position = _simple_matrix_position_by_item(package, "hba-main")
    alt_position = _simple_matrix_position_by_item(package, "hba-alt")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": main_position["component_candidate_id"],
                        "stock_row_id": main_position["stock_row_id"],
                        "role": "Primary FC HBA",
                        "quantity": 1,
                        "unit_price_value": "1500.00",
                        "unit_price_currency": "USD",
                        "line_total_value": "1500.00",
                        "line_total_currency": "USD",
                    }
                ],
                "available_alternatives": [
                    {
                        "requirement_id": "R_HBA",
                        "component_candidate_id": alt_position["component_candidate_id"],
                        "stock_row_id": "ocs:hba-alt:stale",
                        "part_number": "PN-hba-alt",
                        "item": "Alternative FC HBA",
                        "available_quantity": 1,
                        "unit_price_value": "1.00",
                        "unit_price_currency": "EUR",
                        "reason": "Relevant alternative.",
                    }
                ],
                "total_price_value": "1500.00",
                "total_price_currency": "USD",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need FC HBA",
        resolved_request={"customer_task_summary": "Need FC HBA"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == []

    alternative = outcome.validated_quote["available_alternatives"][0]
    assert alternative["component_candidate_id"] == alt_position["component_candidate_id"]
    assert alternative["stock_row_id"] == alt_position["stock_row_id"]
    assert alternative["part_number"] == "PN-hba-alt"
    assert alternative["available_quantity"] == 1
    assert alternative["quantity_value"] == 1
    assert alternative["quantity_is_greater_than"] is True
    assert alternative["stock_confirmation_required"] is True
    assert alternative["unit_price_value"] == "1176.0000"
    assert alternative["unit_price_currency"] == "USD"

    assert outcome.validated_quote["quote_integrity"]["adjustments"] == []


def test_simple_stock_composer_does_not_repair_part_number_conflict(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 22, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="hba-main",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Primary FC HBA",
    )
    _seed_product(
        db_session,
        item_id="hba-alt",
        category_id="cat-a",
        synced_at=synced_at,
        item_name="Alternative FC HBA",
    )
    _seed_stock(
        db_session,
        item_id="hba-main",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("1500.00"),
        quantity_value=5,
    )
    _seed_stock(
        db_session,
        item_id="hba-alt",
        location="MSK",
        synced_at=synced_at,
        price_order_value=Decimal("1176.00"),
        quantity_value=10,
    )
    db_session.commit()

    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_group_matrix("ocs", ["cat-a"]))
    package = build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-a"],
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    main_position = _simple_matrix_position_by_item(package, "hba-main")
    alt_position = _simple_matrix_position_by_item(package, "hba-alt")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "КП draft",
                "selection_mode": "partial_stock_offer",
                "lines": [
                    {
                        "line_id": "L1",
                        "component_candidate_id": main_position["component_candidate_id"],
                        "stock_row_id": main_position["stock_row_id"],
                        "role": "Primary FC HBA",
                        "quantity": 1,
                        "unit_price_value": "1500.00",
                        "unit_price_currency": "USD",
                        "line_total_value": "1500.00",
                        "line_total_currency": "USD",
                    }
                ],
                "available_alternatives": [
                    {
                        "requirement_id": "R_HBA",
                        "component_candidate_id": alt_position["component_candidate_id"],
                        "stock_row_id": "ocs:hba-alt:stale",
                        "part_number": "WRONG-PN",
                        "item": "Alternative FC HBA",
                        "available_quantity": 1,
                        "unit_price_value": "1.00",
                        "unit_price_currency": "EUR",
                        "reason": "Relevant alternative.",
                    }
                ],
                "total_price_value": "1500.00",
                "total_price_currency": "USD",
            },
        }
    )

    outcome = compose_simple_stock_quote(
        user_request="Need FC HBA",
        resolved_request={"customer_task_summary": "Need FC HBA"},
        matrix_package=package,
        settings=LlmSettings(_env_file=None),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == SIMPLE_STOCK_QUOTE_ACCEPTED
    assert outcome.validation_warnings == []
    alternative = outcome.validated_quote["available_alternatives"][0]
    assert alternative["component_candidate_id"] == alt_position["component_candidate_id"]
    assert alternative["part_number"] == "PN-hba-alt"
    assert alternative["stock_row_id"] == alt_position["stock_row_id"]
    assert outcome.validated_quote["quote_integrity"]["adjustments"] == []


def test_v3_request_intake_returns_no_matching_category() -> None:
    decision = route_v3_full_category_target(
        text="Need office chairs",
        settings=LlmSettings(_env_file=None),
        category_catalog=[
            {
                "category_id": "cat-root",
                "parent_category_id": None,
                "name": "Root category",
                "level": 0,
                "path": "Root category",
                "enabled_for_sync": True,
            }
        ],
        llm_client=FakeLlmClient(
            {
                "status": "no_matching_category",
                "category_ids": [],
                "reason": "No catalog category matches.",
            }
        ),
    )

    assert decision.status == "no_matching_category"
    assert decision.profile is None


def test_v3_full_category_quote_refreshes_selected_categories_before_matrix(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    db_session.add(_category("cat-a", None, "Category A", 0, synced_at))
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()
    refresh_calls: list[dict[str, Any]] = []

    async def fake_refresh(*args: Any, **kwargs: Any) -> CategoryRefreshResult:
        refresh_calls.append(
            {
                "distributor_code": kwargs["distributor_code"],
                "category_ids": list(kwargs["category_ids"]),
            }
        )
        return CategoryRefreshResult(
            distributor_code=kwargs["distributor_code"],
            status="success",
            category_count=len(kwargs["category_ids"]),
            products_processed=1,
            stock_rows_inserted=1,
            sync_run_id=42,
        )

    monkeypatch.setattr(
        v3_service_module,
        "refresh_distributor_categories",
        fake_refresh,
    )

    result = asyncio.run(
        run_v3_full_category_quote(
            text="Need one product from cat-a",
            session=AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            category_ids=["cat-a"],
            distributor_code="ocs",
            settings=LlmSettings(_env_file=None),
        )
    )

    diagnostics = result.report_json["diagnostics"]
    assert refresh_calls == [{"distributor_code": "ocs", "category_ids": ["cat-a"]}]
    assert diagnostics["matrix_row_count"] == 1
    assert diagnostics["stock_refresh"] == {
        "enabled": True,
        "distributor_code": "ocs",
        "status": "success",
        "category_count": 1,
        "products_processed": 1,
        "stock_rows_inserted": 1,
        "sync_run_id": 42,
        "error_message": None,
    }


def test_v3_full_category_quote_stops_before_llm_when_stock_refresh_fails(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    db_session.add(_category("cat-a", None, "Category A", 0, synced_at))
    db_session.commit()

    async def fake_refresh(*args: Any, **kwargs: Any) -> CategoryRefreshResult:
        return CategoryRefreshResult(
            distributor_code=kwargs["distributor_code"],
            status="failed",
            category_count=len(kwargs["category_ids"]),
            error_message="Distributor API unavailable",
        )

    monkeypatch.setattr(
        v3_service_module,
        "refresh_distributor_categories",
        fake_refresh,
    )

    result = asyncio.run(
        run_v3_full_category_quote(
            text="Need one product from cat-a",
            session=AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            category_ids=["cat-a"],
            distributor_code="ocs",
            settings=LlmSettings(_env_file=None),
        )
    )

    assert result.result_state == "stock_refresh_failed"
    assert result.report_json["final_status_source"] == V3_STOCK_REFRESH_FAILED
    assert result.report_json["llm_configurator_used"] is False
    assert (
        result.report_json["no_recommendation_reason"]["details"]
        == "Distributor API unavailable"
    )
    assert result.report_json["diagnostics"]["stock_refresh"]["status"] == "failed"


def test_v3_full_category_quote_uses_cached_matrix_when_stock_refresh_fails(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    db_session.add(_category("cat-a", None, "Category A", 0, synced_at))
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()
    captured: dict[str, Any] = {}

    async def fake_refresh(*args: Any, **kwargs: Any) -> CategoryRefreshResult:
        return CategoryRefreshResult(
            distributor_code=kwargs["distributor_code"],
            status="failed",
            category_count=len(kwargs["category_ids"]),
            error_message="Distributor API unavailable",
        )

    def fake_compose(**kwargs: Any) -> FullCategoryComposerOutcome:
        matrix_package = kwargs["matrix_package"]
        captured["matrix_row_count"] = matrix_package.payload["diagnostics"][
            "stock_row_count"
        ]
        return FullCategoryComposerOutcome(
            pipeline_version="v3_full_category_matrix",
            used=True,
            status="quote",
            final_status_source=V3_CODE_VALIDATION_BYPASSED,
            primary_recommendation_status="valid",
            validated_quote={"engineering_review_required": True, "engineer_checks": []},
            diagnostics={},
        )

    monkeypatch.setattr(
        v3_service_module,
        "refresh_distributor_categories",
        fake_refresh,
    )
    monkeypatch.setattr(
        v3_service_module,
        "_compose_v3_full_category_quote_sync",
        fake_compose,
    )

    result = asyncio.run(
        run_v3_full_category_quote(
            text="Need one product from cat-a",
            session=AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            category_ids=["cat-a"],
            distributor_code="ocs",
            settings=LlmSettings(_env_file=None),
        )
    )

    diagnostics = result.report_json["diagnostics"]
    warning = diagnostics["stock_refresh"]["freshness_warning"]
    assert result.result_state == "quote_draft_review_required"
    assert captured["matrix_row_count"] == 1
    assert diagnostics["matrix_row_count"] == 1
    assert diagnostics["stock_refresh"]["status"] == "failed_using_cached_stock"
    assert diagnostics["stock_refresh"]["refresh_status"] == "failed"
    assert diagnostics["stock_refresh"]["fallback_used"] is True
    assert diagnostics["stock_refresh"]["cached_matrix_row_count"] == 1
    assert warning in result.report_json["v3_validation_warnings"]
    assert warning in result.report_json["validated_quote"]["engineer_checks"]


def test_simple_stock_quote_uses_cached_matrix_when_stock_refresh_fails(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    db_session.add(_category("cat-a", None, "Category A", 0, synced_at))
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()
    captured: dict[str, Any] = {}

    async def fake_refresh(*args: Any, **kwargs: Any) -> CategoryRefreshResult:
        return CategoryRefreshResult(
            distributor_code=kwargs["distributor_code"],
            status="failed",
            category_count=len(kwargs["category_ids"]),
            error_message="Distributor API unavailable",
        )

    def fake_compose(**kwargs: Any) -> FullCategoryComposerOutcome:
        matrix_package = kwargs["matrix_package"]
        captured["matrix_row_count"] = matrix_package.payload["diagnostics"]["row_count"]
        return FullCategoryComposerOutcome(
            pipeline_version="simple_stock_quote",
            used=True,
            status="quote",
            final_status_source=SIMPLE_STOCK_QUOTE_ACCEPTED,
            primary_recommendation_status="llm_final",
            validated_quote={"title": "Quote draft", "lines": [], "engineer_checks": []},
            diagnostics={},
        )

    monkeypatch.setattr(
        simple_stock_service_module,
        "refresh_distributor_categories",
        fake_refresh,
    )
    monkeypatch.setattr(
        simple_stock_service_module,
        "_compose_simple_stock_quote_sync",
        fake_compose,
    )

    result = asyncio.run(
        simple_stock_service_module.run_simple_stock_quote(
            text="Need one product from cat-a",
            session=AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            category_ids=["cat-a"],
            distributor_code="ocs",
            settings=LlmSettings(_env_file=None),
        )
    )

    diagnostics = result.report_json["diagnostics"]
    warning = diagnostics["stock_refresh"]["freshness_warning"]
    assert result.result_state == "quote_draft_review_required"
    assert captured["matrix_row_count"] == 1
    assert diagnostics["matrix_row_count"] == 1
    assert diagnostics["stock_refresh"]["status"] == "failed_using_cached_stock"
    assert diagnostics["stock_refresh"]["refresh_status"] == "failed"
    assert diagnostics["stock_refresh"]["fallback_used"] is True
    assert diagnostics["stock_refresh"]["cached_matrix_row_count"] == 1
    assert warning in result.report_json["v3_validation_warnings"]
    assert warning in result.report_json["validated_quote"]["engineer_checks"]


def test_v3_full_category_composer_skips_llm_for_empty_matrix() -> None:
    package = build_full_category_matrix_group_package(
        distributor_code="ocs",
        category_ids=["cat-empty"],
        rows=[],
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {"lines": []},
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from empty category",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.used is False
    assert outcome.final_status_source == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.no_recommendation_reason["fallback_reason"] == (
        MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
    )
    assert outcome.diagnostics["matrix_status"] == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
    assert len(fake_client.calls) == 0


def test_v3_result_state_keeps_empty_matrix_reason_explicit() -> None:
    assert (
        v3_result_state({"final_status_source": MATRIX_EMPTY_AFTER_CATEGORY_SELECTION})
        == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
    )


def test_v3_result_state_treats_validation_bypass_as_review_draft() -> None:
    assert (
        v3_result_state(
            {
                "final_status_source": V3_CODE_VALIDATION_BYPASSED,
                "primary_recommendation_status": "valid",
                "validated_quote": {"engineering_review_required": True},
            }
        )
        == "quote_draft_review_required"
    )


def test_v3_full_category_composer_validates_selected_stock_row(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "Cheapest valid quote",
                "lines": [
                    {
                        "role": "primary",
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 2,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "200.0000",
                        "line_total_currency": "USD",
                        "reason": "Fits request using provided matrix facts.",
                    }
                ],
                "total_price_value": "200.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [
                        f"{row['stock_row_id']} has enough stock and price facts."
                    ],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
                "price_audit": [
                    "No cheaper technically workable row was found in the supplied matrix."
                ],
                "deviation_notes": [
                    (
                        "Исходное требование: точная модель A; выбрано: "
                        "складской аналог B; класс: равноценно; влияние: "
                        "закрывает задачу; согласование не требуется."
                    )
                ],
                "why_selected": "Lowest valid option in the supplied full matrix.",
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.primary_recommendation_status == "valid"
    assert outcome.validated_quote["total_price_value"] == "200.0000"
    assert outcome.validated_quote["lines"][0]["stock_row_id"] == row["stock_row_id"]
    assert outcome.validated_quote["compatibility_check"]["status"] == "compatible"
    assert outcome.validated_quote["price_audit"] == [
        "No cheaper technically workable row was found in the supplied matrix."
    ]
    assert outcome.validated_quote["deviation_notes"] == [
        (
            "Исходное требование: точная модель A; выбрано: складской аналог B; "
            "класс: равноценно; влияние: закрывает задачу; согласование не требуется."
        )
    ]
    assert len(fake_client.calls) == 1
    assert "full_category_matrix" in fake_client.calls[0][1]


def test_v3_full_category_composer_passes_through_no_recommendation(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    fake_client = FakeLlmClient(
        {
            "status": "no_recommendation",
            "no_recommendation": {
                "reason_code": "exact_target_missing",
                "summary": "Точной модели нет.",
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Нужен лучший доступный товар",
        resolved_request={
            "schema_version": "resolved_request_schema_v7",
            "request_mode": "best_available",
            "allow_partial_offer": True,
            "objects": [],
        },
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_NO_RECOMMENDATION
    assert outcome.validation_errors == []
    assert outcome.no_recommendation_reason["reason_code"] == "exact_target_missing"


def test_v3_full_category_composer_preserves_v4_selection_contract(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "Лучший складской аналог",
                "solution_scope": "complete_system",
                "substitution_policy": "allowed_with_disclosed_downgrade",
                "selection_mode": "analog_with_downgrade",
                "lines": [
                    {
                        "role": "primary",
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                        "satisfies_requirement_ids": ["R1"],
                        "reason": "Закрывает основную задачу по строке склада.",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "requirement_coverage": [
                    {
                        "requirement_id": "R1",
                        "requirement": "Запрошен товар",
                        "priority": "target",
                        "outcome": "substituted",
                        "requested": "точная модель",
                        "offered": "складской аналог",
                        "evidence": [row["stock_row_id"]],
                        "impact": "нужно согласовать замену модели",
                    }
                ],
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [
                        f"{row['stock_row_id']} выбран как совместимая складская строка."
                    ],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
                "why_selected": "Это лучший вариант в допустимом классе соответствия.",
                "deviation_notes": [
                    {
                        "requirement_id": "R1",
                        "requested": "точная модель",
                        "offered": "складской аналог",
                        "direction": "different",
                        "severity": "minor",
                        "impact": "требуется согласовать замену модели",
                        "reason": "точной модели нет в складской матрице",
                    }
                ],
                "price_audit": [
                    {
                        "scope": "configuration",
                        "result": "Выбран самый дешевый недоминированный вариант.",
                        "evidence": [row["stock_row_id"]],
                    }
                ],
                "engineer_checks": [
                    "Сверить финальную применимость аналога перед отправкой КП."
                ],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_warnings == ["code_validation_bypassed"]
    assert outcome.validated_quote["solution_scope"] == "complete_system"
    assert outcome.validated_quote["substitution_policy"] == (
        "allowed_with_disclosed_downgrade"
    )
    assert outcome.validated_quote["selection_mode"] == "analog_with_downgrade"
    assert outcome.validated_quote["lines"][0]["satisfies_requirement_ids"] == ["R1"]
    assert outcome.validated_quote["requirement_coverage"][0]["outcome"] == "substituted"
    assert outcome.validated_quote["deviation_notes"][0]["direction"] == "different"
    assert outcome.validated_quote["price_audit"][0]["scope"] == "configuration"


def test_v3_full_category_composer_preserves_v6_partial_quote_contract(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "title": "Ближайшая складская база",
                "client_status_label": "Ближайший складской вариант требует добора",
                "solution_scope": "configured_system",
                "substitution_policy": "allowed_with_disclosed_downgrade",
                "selection_mode": "partial_build",
                "completeness_status": "partial",
                "operational_status": "incomplete_needs_procurement",
                "anchor_component_candidate_id": row["component_candidate_id"],
                "lines": [
                    {
                        "role": "platform",
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                        "satisfies_requirement_ids": ["R1"],
                        "reason": "Складская база выбрана как ближайший якорь.",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "client_summary": "Можно поставить складскую базу, но часть ТЗ нужно добрать.",
                "coverage_summary": "База закрыта, процессоры отсутствуют в матрице.",
                "requirement_coverage": [
                    {
                        "requirement_id": "R1",
                        "requirement": "Серверная база",
                        "priority": "core",
                        "outcome": "met",
                        "requested": "серверная база",
                        "offered": "складская база",
                    },
                    {
                        "requirement_id": "R2",
                        "requirement": "Процессор",
                        "priority": "important",
                        "outcome": "missing",
                        "requested": "2 CPU",
                        "offered": "не выбрано",
                    },
                ],
                "key_deviations": [
                    {
                        "requirement_id": "R2",
                        "requested": "2 CPU",
                        "offered": "не выбрано",
                        "direction": "different",
                        "severity": "material",
                        "impact": "Без добора CPU система не является полной.",
                        "reason": "Совместимый CPU не подтвержден по матрице.",
                    }
                ],
                "procurement_gaps": [
                    {
                        "requirement_id": "R2",
                        "role": "cpu",
                        "requested": "2 CPU",
                        "status": "no_compatible_item_proven",
                        "required_for": "operational_readiness",
                        "impact": "Нужно добрать CPU для работоспособности.",
                        "next_action": "Запросить CPU у дистрибьютора или согласовать другую базу.",
                    }
                ],
                "compatibility_check": {
                    "status": "compatible_selected_lines",
                    "checked_facts": [
                        f"{row['stock_row_id']} quoted as the selected anchor line."
                    ],
                    "blocking_mismatches": [],
                    "selected_line_conflicts": [],
                    "unresolved_risks": [],
                },
                "why_selected": "Это ближайшая складская база с корректной ценой и остатком.",
                "engineer_checks": ["Проверить добираемые CPU перед финальным КП."],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need a configured system from stock",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert outcome.validation_warnings == ["code_validation_bypassed"]
    assert outcome.validated_quote["selection_mode"] == "partial_build"
    assert outcome.validated_quote["completeness_status"] == "partial"
    assert outcome.validated_quote["operational_status"] == "incomplete_needs_procurement"
    assert outcome.validated_quote["client_summary"].startswith("Можно поставить")
    assert outcome.validated_quote["key_deviations"][0]["severity"] == "material"
    assert outcome.validated_quote["procurement_gaps"][0]["status"] == (
        "no_compatible_item_proven"
    )
    assert outcome.validated_quote["compatibility_check"]["status"] == (
        "compatible_selected_lines"
    )


def test_v3_full_category_composer_accepts_v7_partial_with_fact_refs(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fact_id = package.payload["category_sections"][0]["products"][0]["fact_refs"][0][
        "fact_id"
    ]
    resolved_request = {
        "schema_version": "resolved_request_schema_v7_1",
        "request_mode": "best_available",
        "allow_partial_offer": True,
        "objects": [
            {
                "object_id": "O1",
                "functional_class": "товар",
                "deliverable_scope": "standalone_product",
                "object_quantity": 1,
                "primary_item_id": "I1",
                "anchor_policy": "self",
                "requested_items": [
                    {
                        "item_id": "I1",
                        "item_kind": "primary_product",
                        "role": "товар",
                        "quantity": 1,
                        "quantity_basis": "total",
                        "quantity_requirement_id": "R1",
                        "partial_quantity_allowed": True,
                        "source_phrase": "1 товар",
                        "constraints": [
                            {
                                "requirement_id": "R2",
                                "dimension": "product_class",
                                "operator": "semantic_match",
                                "value_text": "товар",
                                "unit": None,
                                "strictness": "core",
                                "substitution": "downgrade_allowed",
                                "source_phrase": "товар",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "selection_mode": "partial_without_anchor",
                "completeness_status": "partial",
                "operational_status": "requires_completion",
                "solution_scope": "standalone_product",
                "substitution_policy": "allowed_with_disclosed_downgrade",
                "client_status_label": "Частичный складской комплект",
                "object_results": [
                    {
                        "object_id": "O1",
                        "anchor_policy": "self",
                        "selection_mode": "partial_without_anchor",
                        "selected_anchor_line_id": None,
                        "anchor_component_candidate_id": None,
                        "summary": (
                            "Anchor-кандидатов в manifest нет, выбрана полезная "
                            "складская позиция."
                        ),
                    }
                ],
                "anchor_search_audit": [
                    {
                        "object_id": "O1",
                        "anchor_policy": "self",
                        "anchor_candidate_count": 0,
                        "outcome": "none_available_in_manifest",
                        "selected_anchor_line_id": None,
                        "selected_anchor_component_candidate_id": None,
                        "reason": "В manifest нет anchor-кандидатов для standalone_product.",
                    }
                ],
                "lines": [
                    {
                        "line_id": "L1",
                        "object_id": "O1",
                        "role": "primary",
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                        "covered_item_ids": ["I1"],
                        "covered_requirement_ids": ["R1", "R2"],
                        "satisfies_requirement_ids": ["R1", "R2"],
                        "technical_status": "review_required",
                        "coverage_contributions": [
                            {
                                "item_id": "I1",
                                "selected_covered_quantity": 1,
                                "remaining_quantity": 0,
                            }
                        ],
                        "fact_ids": [fact_id],
                        "compatibility_statement": (
                            "Строка выбрана из матрицы и проверяется как самостоятельная поставка."
                        ),
                        "reason": "Позиция выбрана из переданной матрицы.",
                    }
                ],
                "total_price": {"value": "100.0000", "currency": "USD"},
                "client_summary": "Можно поставить складскую позицию.",
                "coverage": [
                    {
                        "object_id": "O1",
                        "item_id": "I1",
                        "requirement_ids": ["R1", "R2"],
                        "requested_quantity": 1,
                        "selected_covered_quantity": 1,
                        "remaining_quantity": 0,
                        "status": "covered",
                    }
                ],
                "requirement_coverage": [
                    {"requirement_id": "R1", "outcome": "met"},
                    {"requirement_id": "R2", "outcome": "met"},
                ],
                "key_deviations": [],
                "procurement_gaps": [],
                "compatibility_check": {
                    "status": "review_required_selected_set",
                    "checked_facts": [
                        {
                            "line_id": "L1",
                            "component_candidate_id": row["component_candidate_id"],
                            "stock_row_id": row["stock_row_id"],
                            "fact_ids": [fact_id],
                            "relationship": "independent_supply",
                            "conclusion": (
                                "Строка существует в матрице и может быть "
                                "самостоятельной поставкой."
                            ),
                        }
                    ],
                    "blocking_mismatches": [],
                    "selected_line_conflicts": [],
                    "unresolved_risks": [],
                },
                "dominance_audit": [
                        {
                            "line_id": "L1",
                            "item_id": "I1",
                            "audit_scope": "same_role",
                            "selected_component_candidate_id": row["component_candidate_id"],
                            "selected_stock_row_id": row["stock_row_id"],
                            "price_position": "lowest_eligible",
                            "cheaper_candidate_count": 0,
                            "cheaper_candidates_reviewed": [],
                            "result": "cheapest eligible selected row",
                        }
                ],
                "why_selected": "Это единственная складская позиция в матрице.",
                "assumptions": [],
                "engineer_checks": [
                    "Проверить финальную применимость выбранной позиции к заявке."
                ],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Нужен 1 товар",
        resolved_request=resolved_request,
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validated_quote["selection_mode"] == "partial_without_anchor"
    assert outcome.validated_quote["total_price"] == {
        "value": "100.0000",
        "currency": "USD",
    }
    assert outcome.validated_quote["lines"][0]["fact_ids"] == [fact_id]
    assert outcome.validated_quote["dominance_audit"][0]["result"] == (
        "cheapest eligible selected row"
    )
    assert outcome.diagnostics["composer_prompt_version"] == "composer_v7_1"
    assert outcome.diagnostics["composer_output_schema_version"] == (
        "composer_output_schema_v7_1"
    )


def test_v3_full_category_prompt_requires_immediate_compatibility_gate(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    resolved_request = {
        "objective": "cheapest_minimum_viable",
        "profile": "server",
        "customer_task_summary": "Server for 20-25 VMs",
        "hard_requirements": [
            {
                "key": "ram.capacity_min",
                "value": 256,
                "unit": "GB",
                "source_phrase": "256GB RAM",
                "explicit": True,
            }
        ],
        "unknowns_or_missing_facts": ["hypervisor is not specified"],
    }
    system_prompt, user_prompt = build_full_category_quote_prompts(
        user_request="Need a server for virtualization",
        resolved_request=resolved_request,
        matrix_package=package,
    )

    assert "Stock Configurator Composer v7" in system_prompt
    assert "single semantic selection pass" in system_prompt
    assert "the default output is status=\"quote\"" in system_prompt
    assert "partial_with_anchor" in system_prompt
    assert "partial_without_anchor" in system_prompt
    assert "Included coverage subtraction" in system_prompt
    assert "Residual fulfilment" in system_prompt
    assert "Same-role dominance" in system_prompt
    assert "Every requested_item and every requirement_id" in system_prompt
    assert "fact_ids" in system_prompt
    assert "review_required_selected_set" in system_prompt
    assert "total_price is the exact sum of included lines only" in system_prompt
    user_payload = json.loads(user_prompt)
    assert list(user_payload) == [
        "TASK_CAPSULE",
        "resolved_request",
        "selection_contract",
        "anchor_candidate_manifest",
        "matrix_index",
        "row_legend",
        "BEGIN_FULL_CATEGORY_MATRIX",
        "full_category_matrix",
        "END_FULL_CATEGORY_MATRIX",
        "FINAL_RESPONSE_GATE",
    ]
    assert user_payload["resolved_request"] == resolved_request
    assert user_payload["selection_contract"]["contract_version"] == (
        "selection_contract_v7_1"
    )
    assert user_payload["selection_contract"]["quote_first"] is True
    assert user_payload["selection_contract"]["allow_partial_offer"] is True
    assert user_payload["selection_contract"]["anchor_detection"] == (
        "backend_mechanical_manifest_from_category_roles"
    )
    assert user_payload["selection_contract"]["no_recommendation_allowed"] is False
    assert "anchor_candidate_manifest" in user_payload
    assert user_payload["FINAL_RESPONSE_GATE"]
    assert user_payload["matrix_index"]
    assert user_payload["BEGIN_FULL_CATEGORY_MATRIX"] == "BEGIN FULL CATEGORY MATRIX"
    assert "full_category_matrix" in user_payload


def test_v3_full_category_prompt_requires_lowest_price_minimum_viable_quote(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    system_prompt, _ = build_full_category_quote_prompts(
        user_request="Need cheapest server for 20-25 light VMs",
        matrix_package=package,
    )

    assert "Same-role dominance" in system_prompt
    assert "whole configurations" in system_prompt
    assert "complete/preconfigured products" in system_prompt
    assert "A more expensive line is allowed only with a matrix-supported reason" in system_prompt
    assert "dominates" in system_prompt
    assert "why_selected" in system_prompt
    assert "dominance_audit" in system_prompt
    assert "single semantic selection pass" in system_prompt
    assert "Every line must use an existing component_candidate_id" in system_prompt


def test_v3_full_category_composer_rejects_llm_incompatible_quote(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "incompatible",
                    "checked_facts": [
                        (
                            f"{row['stock_row_id']} selected platform and CPU rows "
                            "describe different sockets."
                        )
                    ],
                    "blocking_mismatches": ["platform_cpu_socket_mismatch"],
                    "unresolved_risks": [],
                },
                "price_audit": [
                    "Cheapest path review failed because the selected rows are incompatible."
                ],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert outcome.validation_error_details == []
    assert outcome.validated_quote["compatibility_check"]["blocking_mismatches"] == [
        "platform_cpu_socket_mismatch"
    ]


def test_v3_full_category_composer_allows_unresolved_risks_as_review_draft(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [
                        (
                            f"{row['stock_row_id']} provides stock and price but "
                            "not BIOS support."
                        )
                    ],
                    "blocking_mismatches": [],
                    "unresolved_risks": ["BIOS CPU support still needs confirmation"],
                },
                "price_audit": [
                    "No cheaper technically workable row was found in the supplied matrix."
                ],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.primary_recommendation_status == "valid"
    assert outcome.validation_warnings == ["code_validation_bypassed"]
    assert outcome.validated_quote["compatibility_check"]["unresolved_risks"] == [
        "BIOS CPU support still needs confirmation"
    ]


def test_v3_full_category_composer_allows_missing_price_audit(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [f"{row['stock_row_id']} has enough stock."],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert outcome.validated_quote["price_audit"] == []
    assert len(fake_client.calls) == 1


def test_v3_full_category_composer_warns_on_missing_compatibility_coverage_per_line(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": ["The selected row is compatible."],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert outcome.validation_warnings == ["code_validation_bypassed"]
    assert outcome.validated_quote["lines"][0]["stock_row_id"] == row["stock_row_id"]
    assert len(fake_client.calls) == 1


def test_v3_full_category_composer_still_requires_compatibility_checked_facts(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert len(fake_client.calls) == 1
    assert outcome.diagnostics["code_validation_bypassed"] is True


def test_v3_full_category_composer_does_not_auto_repair_stock_errors_by_default(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 4,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "400.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "400.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [f"{row['stock_row_id']} is selected."],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need four products from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert len(fake_client.calls) == 1
    assert outcome.diagnostics["code_validation_bypassed"] is True


def test_v3_full_category_composer_treats_greater_than_stock_as_lower_bound(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(
        db_session,
        item_id="p1",
        location="MSK",
        synced_at=synced_at,
        quantity_value=1,
        quantity_is_greater_than=True,
    )
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 2,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "200.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "200.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [f"{row['stock_row_id']} is selected."],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need two products from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validated_quote["lines"][0]["available_quantity"] == 2
    assert outcome.validated_quote["lines"][0]["quantity_value"] == 1
    assert outcome.validated_quote["lines"][0]["quantity_is_greater_than"] is True


def test_v3_full_category_composer_passes_through_greater_than_stock_overrun(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(
        db_session,
        item_id="p1",
        location="MSK",
        synced_at=synced_at,
        quantity_value=1,
        quantity_is_greater_than=True,
    )
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": row["component_candidate_id"],
                        "stock_row_id": row["stock_row_id"],
                        "quantity": 3,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "300.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "300.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [f"{row['stock_row_id']} is selected."],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need three products from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert outcome.validated_quote["lines"][0]["available_quantity"] == 2


def test_v3_full_category_composer_passes_through_unknown_component_id(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p1", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": "ocs:invented",
                        "stock_row_id": "ocs:invented:999",
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [
                        f"{row['stock_row_id']} is claimed compatible."
                    ],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
                "price_audit": [
                    "No cheaper technically workable row was found in the supplied matrix."
                ],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert outcome.validated_quote["lines"][0]["component_candidate_id"] == "ocs:invented"


def test_v3_full_category_composer_resolves_ids_from_unique_stock_row_suffix(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 10, tzinfo=UTC)
    _seed_product(db_session, item_id="p12345", category_id="cat-a", synced_at=synced_at)
    _seed_stock(db_session, item_id="p12345", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _single_category_package(db_session)
    row = _first_matrix_row(package)
    bad_component_candidate_id = row["component_candidate_id"].replace("12345", "145")
    bad_stock_row_id = row["stock_row_id"].replace("12345", "145")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "lines": [
                    {
                        "component_candidate_id": bad_component_candidate_id,
                        "stock_row_id": bad_stock_row_id,
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [
                        f"{bad_stock_row_id} is claimed compatible."
                    ],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
                "price_audit": [
                    "No cheaper technically workable row was found in the supplied matrix."
                ],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Need one product from cat-a",
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert outcome.validation_warnings == ["code_validation_bypassed"]
    validated_line = outcome.validated_quote["lines"][0]
    assert validated_line["component_candidate_id"] == bad_component_candidate_id
    assert validated_line["stock_row_id"] == bad_stock_row_id


def test_v3_quote_summary_includes_engineering_review_flag() -> None:
    summary = v3_quote_summary(
        {
            "pipeline_version": "v3_full_category_matrix",
            "llm_configurator_used": True,
            "primary_recommendation_status": "valid",
            "final_status_source": V3_VALIDATED,
            "diagnostics": {
                "matrix_status": "matrix_ready_for_llm",
                "matrix_row_count": 1,
                "matrix_char_count": 100,
                "prompt_char_count": 200,
                "category_ids": ["cat-a"],
                "model": "qwen/qwen3.7-plus",
            },
            "validated_quote": {
                "engineering_review_required": True,
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "lines": [{"component_candidate_id": "ocs:p1"}],
            },
            "v3_validation_errors": [],
        }
    )

    assert summary["engineering_review_required"] is True


def test_v7_1_rejects_partial_without_anchor_when_manifest_has_anchor(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 19, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="anchor1",
        category_id="V1100",
        synced_at=synced_at,
        item_name="HPE DL380 Gen11 ready server",
    )
    _seed_stock(db_session, item_id="anchor1", location="MSK", synced_at=synced_at)
    _seed_product(
        db_session,
        item_id="ram1",
        category_id="V110104",
        synced_at=synced_at,
        item_name="DDR5 RDIMM",
    )
    _seed_stock(db_session, item_id="ram1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _group_category_package(db_session, ["V1100", "V110104"])
    ram_row = _matrix_row_by_item(package, "ram1")
    ram_fact_id = _first_fact_id_for_item(package, "ram1")
    resolved_request = _v7_1_configured_request()
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "selection_mode": "partial_without_anchor",
                "solution_scope": "configured_system",
                "object_results": [
                    {
                        "object_id": "O1",
                        "anchor_policy": "required",
                        "selection_mode": "partial_without_anchor",
                        "summary": "Выбраны только компоненты.",
                    }
                ],
                "anchor_search_audit": [
                    {
                        "object_id": "O1",
                        "anchor_policy": "required",
                        "anchor_candidate_count": 1,
                        "outcome": "none_available_in_manifest",
                        "reason": "Модель пропустила anchor.",
                    }
                ],
                "lines": [
                    {
                        "line_id": "L1",
                        "object_id": "O1",
                        "role": "ram",
                        "component_candidate_id": ram_row["component_candidate_id"],
                        "stock_row_id": ram_row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                        "covered_item_ids": ["I2"],
                        "covered_requirement_ids": ["R2"],
                        "satisfies_requirement_ids": ["R2"],
                        "technical_status": "independent_item",
                        "coverage_contributions": [],
                        "fact_ids": [ram_fact_id],
                        "compatibility_statement": "Память есть на складе.",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "compatibility_check": {
                    "status": "independent_partial_set",
                    "checked_facts": [
                        {
                            "line_id": "L1",
                            "component_candidate_id": ram_row["component_candidate_id"],
                            "stock_row_id": ram_row["stock_row_id"],
                            "fact_ids": [ram_fact_id],
                            "relationship": "independent_supply",
                            "conclusion": "Строка существует в матрице.",
                        }
                    ],
                    "blocking_mismatches": [],
                    "selected_line_conflicts": [],
                    "unresolved_risks": [],
                },
                "dominance_audit": [
                    {
                        "line_id": "L1",
                        "item_id": "I2",
                        "audit_scope": "same_role",
                        "selected_stock_row_id": ram_row["stock_row_id"],
                        "price_position": "lowest_eligible",
                        "cheaper_candidate_count": 0,
                        "cheaper_candidates_reviewed": [],
                    }
                ],
                "engineer_checks": ["Проверить применимость частичной поставки."],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Сервер в составе с памятью",
        resolved_request=resolved_request,
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validation_errors == []
    assert outcome.validated_quote["selection_mode"] == "partial_without_anchor"


def test_v7_1_accepts_partial_with_selected_anchor_and_gaps(
    db_session: Session,
) -> None:
    synced_at = datetime(2026, 6, 19, tzinfo=UTC)
    _seed_product(
        db_session,
        item_id="anchor1",
        category_id="V1100",
        synced_at=synced_at,
        item_name="HPE DL380 Gen11 ready server",
    )
    _seed_stock(db_session, item_id="anchor1", location="MSK", synced_at=synced_at)
    db_session.commit()

    package = _group_category_package(db_session, ["V1100"])
    anchor_row = _matrix_row_by_item(package, "anchor1")
    anchor_fact_id = _first_fact_id_for_item(package, "anchor1")
    fake_client = FakeLlmClient(
        {
            "status": "quote",
            "quote": {
                "selection_mode": "partial_with_anchor",
                "completeness_status": "partial",
                "operational_status": "incomplete_needs_procurement",
                "solution_scope": "configured_system",
                "object_results": [
                    {
                        "object_id": "O1",
                        "anchor_policy": "required",
                        "selection_mode": "partial_with_anchor",
                        "selected_anchor_line_id": "L1",
                        "anchor_component_candidate_id": anchor_row["component_candidate_id"],
                        "summary": "Выбран ближайший складской anchor.",
                    }
                ],
                "anchor_search_audit": [
                    {
                        "object_id": "O1",
                        "anchor_policy": "required",
                        "anchor_candidate_count": 1,
                        "outcome": "selected",
                        "selected_anchor_line_id": "L1",
                        "selected_anchor_component_candidate_id": anchor_row[
                            "component_candidate_id"
                        ],
                        "reason": "Это единственный складской anchor.",
                    }
                ],
                "lines": [
                    {
                        "line_id": "L1",
                        "object_id": "O1",
                        "role": "server_anchor",
                        "component_candidate_id": anchor_row["component_candidate_id"],
                        "stock_row_id": anchor_row["stock_row_id"],
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                        "covered_item_ids": ["I1"],
                        "covered_requirement_ids": ["R1"],
                        "satisfies_requirement_ids": ["R1"],
                        "technical_status": "anchor",
                        "coverage_contributions": [],
                        "fact_ids": [anchor_fact_id],
                        "compatibility_statement": "Anchor выбран из V1100.",
                    }
                ],
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "procurement_gaps": [
                    {
                        "requirement_id": "R2",
                        "role": "ram",
                        "requested": "DDR5 RAM",
                        "status": "not_in_matrix",
                        "required_for": "requested_spec",
                        "impact": "Память нужно добрать отдельно.",
                        "next_action": "Запросить память у поставщика.",
                    }
                ],
                "compatibility_check": {
                    "status": "review_required_selected_set",
                    "checked_facts": [
                        {
                            "line_id": "L1",
                            "component_candidate_id": anchor_row["component_candidate_id"],
                            "stock_row_id": anchor_row["stock_row_id"],
                            "fact_ids": [anchor_fact_id],
                            "relationship": "anchor_identity",
                            "conclusion": "Anchor существует в матрице.",
                        }
                    ],
                    "blocking_mismatches": [],
                    "selected_line_conflicts": [],
                    "unresolved_risks": [],
                },
                "dominance_audit": [
                    {
                        "line_id": "L1",
                        "item_id": "I1",
                        "audit_scope": "anchor",
                        "selected_stock_row_id": anchor_row["stock_row_id"],
                        "price_position": "lowest_eligible",
                        "cheaper_candidate_count": 0,
                        "cheaper_candidates_reviewed": [],
                    }
                ],
                "engineer_checks": ["Проверить состав anchor перед КП."],
            },
        }
    )

    outcome = compose_full_category_quote(
        user_request="Сервер в составе с памятью",
        resolved_request=_v7_1_configured_request(),
        matrix_package=package,
        settings=LlmSettings(_env_file=None, llm_configurator_max_package_chars=1_000_000),
        llm_client=fake_client,
    )

    assert outcome.final_status_source == V3_CODE_VALIDATION_BYPASSED
    assert outcome.validated_quote["selection_mode"] == "partial_with_anchor"
    assert outcome.validated_quote["object_results"][0]["selected_anchor_line_id"] == "L1"


def _single_category_package(db_session: Session):
    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(repository.list_latest_full_category_matrix("ocs", "cat-a"))
    return build_full_category_matrix_package(
        distributor_code="ocs",
        category_id="cat-a",
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )


def _group_category_package(db_session: Session, category_ids: list[str]):
    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(
        repository.list_latest_full_category_group_matrix("ocs", category_ids)
    )
    return build_full_category_matrix_group_package(
        distributor_code="ocs",
        category_ids=category_ids,
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )


def _simple_group_category_package(db_session: Session, category_ids: list[str]):
    repository = ProductRepository(AsyncSessionAdapter(db_session))  # type: ignore[arg-type]
    rows = asyncio.run(
        repository.list_latest_full_category_group_matrix("ocs", category_ids)
    )
    return build_simple_stock_matrix_group_package(
        distributor_code="ocs",
        category_ids=category_ids,
        rows=rows,
        max_package_chars=1_000_000,
        model="qwen/qwen3.7-plus",
    )


def _first_matrix_row(package: Any) -> dict[str, Any]:
    component = package.payload["category_sections"][0]["products"][0]
    stock_row = component["stock_rows"][0]
    return {
        "component_candidate_id": component["component_candidate_id"],
        "stock_row_id": stock_row["stock_row_id"],
        "product": component["product"],
        "stock": stock_row,
    }


def _matrix_row_by_item(package: Any, item_id: str) -> dict[str, Any]:
    for section in package.payload["category_sections"]:
        for component in section["products"]:
            product = component["product"]
            if product.get("item_id") != item_id:
                continue
            stock_row = component["stock_rows"][0]
            return {
                "component_candidate_id": component["component_candidate_id"],
                "stock_row_id": stock_row["stock_row_id"],
                "product": product,
                "stock": stock_row,
            }
    raise AssertionError(f"matrix item not found: {item_id}")


def _simple_matrix_position_by_item(package: Any, item_id: str) -> dict[str, Any]:
    component_candidate_id = f"ocs:{item_id}"
    for section in package.payload["category_sections"]:
        for position in section["positions"]:
            if position.get("component_candidate_id") == component_candidate_id:
                result = dict(position)
                result["stock_row_id"] = _simple_matrix_stock_row_by_item(
                    package,
                    item_id,
                )["stock_row_id"]
                return result
    raise AssertionError(f"simple matrix item not found: {item_id}")


def _simple_matrix_stock_row_by_item(package: Any, item_id: str) -> dict[str, Any]:
    component_candidate_id = f"ocs:{item_id}"
    for row in package.stock_rows:
        if row.get("component_candidate_id") == component_candidate_id:
            return row
    raise AssertionError(f"simple matrix stock row not found: {item_id}")


def _first_fact_id_for_item(package: Any, item_id: str) -> str:
    for section in package.payload["category_sections"]:
        for component in section["products"]:
            if component["product"].get("item_id") == item_id:
                return component["fact_refs"][0]["fact_id"]
    raise AssertionError(f"matrix fact not found: {item_id}")


def _v7_1_configured_request() -> dict[str, Any]:
    return {
        "schema_version": "resolved_request_schema_v7_1",
        "request_mode": "best_available",
        "allow_partial_offer": True,
        "deliverable_scope": "configured_system",
        "objects": [
            {
                "object_id": "O1",
                "functional_class": "сервер",
                "deliverable_scope": "configured_system",
                "object_quantity": 1,
                "primary_item_id": "I1",
                "anchor_policy": "required",
                "requested_items": [
                    {
                        "item_id": "I1",
                        "item_kind": "primary_product",
                        "role": "server",
                        "quantity": 1,
                        "quantity_basis": "total",
                        "quantity_requirement_id": "R1",
                        "partial_quantity_allowed": True,
                        "source_phrase": "Сервер",
                        "constraints": [],
                    },
                    {
                        "item_id": "I2",
                        "item_kind": "component",
                        "role": "ram",
                        "quantity": 1,
                        "quantity_basis": "total",
                        "quantity_requirement_id": "R2",
                        "partial_quantity_allowed": True,
                        "source_phrase": "Оперативная память",
                        "constraints": [],
                    },
                ],
            }
        ],
        "retrieval_plan": {
            "objects": [
                {
                    "object_id": "O1",
                    "anchor_category_ids": [],
                    "component_category_ids": [],
                    "fallback_category_ids": [],
                }
            ],
            "anchor_category_ids": [],
            "component_category_ids": [],
            "fallback_category_ids": [],
        },
    }


def _category(
    category_id: str,
    parent_category_id: str | None,
    name: str,
    level: int,
    synced_at: datetime,
) -> DistributorCategory:
    path = [{"category_id": category_id, "name": name}]
    return DistributorCategory(
        distributor_code="ocs",
        category_id=category_id,
        parent_category_id=parent_category_id,
        name=name,
        level=level,
        path_json=path,
        enabled_for_sync=True,
        raw_json={},
        synced_at=synced_at,
    )


def _seed_product(
    session: Session,
    *,
    item_id: str,
    category_id: str,
    synced_at: datetime,
    item_name: str | None = None,
    product_description: str | None = None,
    producer: str = "Vendor",
    part_number: str | None = None,
) -> None:
    effective_item_name = item_name or f"Product {item_id}"
    session.add(
        DistributorProduct(
            distributor_code="ocs",
            item_id=item_id,
            product_key=item_id,
            part_number=part_number or f"PN-{item_id}",
            producer=producer,
            category_id=category_id,
            item_name=effective_item_name,
            item_name_rus=effective_item_name,
            product_name=effective_item_name,
            product_description=product_description or "Full product description",
            product_notes="Full product notes",
            hscode="8471",
            ean=f"ean-{item_id}",
            is_in_mpt_registry=False,
            is_project_item=False,
            traceable=False,
            condition="Regular",
            warranty="12 months",
            original_country_iso_code="RU",
            vat_percent=Decimal("20.00"),
            serial_number_availability="available",
            catalog_path_json=[{"category_id": category_id, "name": "Category"}],
            package_json={"weight": 1.5},
            raw_json={"raw_product": item_id},
            synced_at=synced_at,
        )
    )


def _seed_stock(
    session: Session,
    *,
    item_id: str,
    location: str,
    synced_at: datetime,
    price_order_value: Decimal = Decimal("100.0000"),
    price_order_currency: str = "USD",
    quantity_value: int = 3,
    quantity_is_greater_than: bool = False,
) -> None:
    session.add(
        DistributorStockPrice(
            distributor_code="ocs",
            item_id=item_id,
            product_key=item_id,
            shipment_city="Moscow",
            location=location,
            location_description=f"{location} warehouse",
            location_type="ShipmentCity",
            quantity_value=quantity_value,
            quantity_is_greater_than=quantity_is_greater_than,
            can_reserve=True,
            price_order_value=price_order_value,
            price_order_currency=price_order_currency,
            price_list_value=Decimal("120.0000"),
            price_list_currency="USD",
            end_user_value=Decimal("140.0000"),
            end_user_currency="USD",
            raw_json={"raw_stock": item_id, "location": location},
            synced_at=synced_at,
        )
    )
