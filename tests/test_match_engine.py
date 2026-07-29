from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Generator, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import app.llm.configuration_composer as configuration_composer_module
from app.core.config import LlmSettings, WebEvidenceSettings
from app.core.database import Base
from app.db.models import DistributorProduct, DistributorStockPrice
from app.evidence.web_evidence import (
    ComponentEvidence,
    EvidencePack,
    EvidenceSearchCache,
    EvidenceSearchResult,
    EvidenceSearchTask,
    EvidenceSource,
    FakeWebSearchProvider,
    RelationEvidence,
)
from app.llm.base import (
    LlmClientError,
    LlmInvalidJsonError,
    LlmReadTimeoutError,
    LlmServerError,
)
from app.llm.configuration_composer import (
    build_llm_configurator_package,
    compose_llm_configurations,
)
from app.matching import match_engine as match_engine_module
from app.matching.match_engine import (
    STATUS_NO_STOCK_MATCH,
    STATUS_PARTIAL_STOCK_MATCHED,
    extract_stock_spec_for_text_match,
    match_stock_spec,
)
from app.matching.spec_schema import StockSpec, StockSpecItem
from app.reports.match_report import build_match_markdown_report
from app.telegram_bot.formatting import format_match_summary
from app.user_facing_text import contains_cjk_text

NETWORK_MATCH_71_TEXT = (
    "Нужен 1 коммутатор 48 портов 1G PoE+, 4 uplink 10G SFP+, "
    "L3, stacking, склад Москва, один самый дешевый вариант для КП"
)


class AsyncSessionAdapter:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, instance: object) -> None:
        self._session.add(instance)

    async def flush(self) -> None:
        self._session.flush()

    async def commit(self) -> None:
        self._session.commit()

    async def rollback(self) -> None:
        self._session.rollback()

    async def scalar(self, statement: Any) -> Any:
        return self._session.scalar(statement)

    async def execute(self, statement: Any) -> Any:
        return self._session.execute(statement)


@pytest.fixture()
def db_session() -> Generator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine, expire_on_commit=False) as session:
        yield session

    engine.dispose()


def test_match_realistic_nerpa_products_returns_partial_when_ram_is_missing(
    db_session: Session,
) -> None:
    _seed_nerpa_product(
        db_session,
        item_id="1000841882",
        part_number="D5720-181125SA04",
        item_name="Server NERPA D5720 2U 2x CPU SSD 2x Power 2x DDR4 64GB",
        quantity=3,
        can_reserve=True,
    )
    _seed_nerpa_product(
        db_session,
        item_id="1000841883",
        part_number="D5720-181125SA05",
        item_name="Server NERPA D5720 2U 2x CPU SSD 2x PSU 2x DDR4 32GB",
        quantity=1,
        can_reserve=True,
    )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=512), _adapter(db_session))
    )

    assert result.status == STATUS_PARTIAL_STOCK_MATCHED
    assert result.engineer_review_required is True
    assert result.total_candidates == 2
    assert result.candidates[0].available_quantity == 3
    assert any(
        "Оперативная память ниже требования" in requirement
        for candidate in result.candidates
        for requirement in candidate.missing_requirements
    )
    assert any(
        "Количество закрыто" in requirement
        for candidate in result.candidates
        for requirement in candidate.matched_requirements
    )


def test_no_stock_match_when_products_are_absent(db_session: Session) -> None:
    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=1, ram_min_gb=64), _adapter(db_session))
    )

    assert result.status == STATUS_NO_STOCK_MATCH
    assert result.total_candidates == 0
    assert result.engineer_review_required is True
    assert "Складские варианты не найдены." in result.missing_requirements
    assert result.to_report_json()["build_candidates"] == []
    assert any("Сборка из комплектующих" in item for item in result.missing_requirements)


def test_configuration_builder_v0_builds_candidate_from_synthetic_parts(
    db_session: Session,
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-2U-2S",
        producer="TestVendor",
        category_id="V110100",
        item_name="2U dual socket server platform 2x CPU 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-64",
        part_number="RAM-64G",
        producer="TestVendor",
        category_id="V110104",
        item_name="DDR4 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-1",
        part_number="SSD-960G",
        producer="TestVendor",
        category_id="V110106",
        item_name="Server SSD 960GB SATA",
        quantity=2,
        price=Decimal("200"),
    )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=512), _adapter(db_session))
    )
    report_json = result.to_report_json()

    assert result.status == STATUS_PARTIAL_STOCK_MATCHED
    assert result.matched_items == 0
    assert len(report_json["ready_stock_candidates"]) == 0
    assert len(report_json["build_candidates"]) == 1

    build_candidate = result.candidates[0]
    assert build_candidate.candidate_type == "build_from_parts"
    assert build_candidate.total_price_value == Decimal("6800")
    assert build_candidate.total_price_currency == "USD"
    assert build_candidate.completeness_status == "incomplete"
    assert build_candidate.missing_component_roles == ["cpu"]
    assert build_candidate.excluded_from_total_roles == ["cpu"]
    assert build_candidate.cpu_per_server == 2
    assert build_candidate.total_cpu_required == 4
    assert build_candidate.total_price_note == "без CPU"
    assert {component["role"] for component in build_candidate.components} == {
        "server_platform",
        "ram",
        "ssd",
    }
    assert any("подбор CPU" in item for item in build_candidate.missing_components)
    assert build_candidate.engineer_review_required is True
    assert "cpu" in report_json["build_candidates"][0]["excluded_from_total_roles"]
    assert report_json["build_candidates"][0]["total_price_note"] == "без CPU"
    assert any("Совместимость RAM" in warning for warning in build_candidate.risk_flags)


def test_configuration_builder_v0_includes_server_cpu_quantity_when_available(
    db_session: Session,
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-2U-2S",
        producer="TestVendor",
        category_id="V110100",
        item_name="2U dual socket server platform 2x CPU 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="cpu-1",
        part_number="CPU-SERVER",
        producer="TestVendor",
        category_id="V110103",
        item_name="Server CPU for dual socket platform",
        quantity=4,
        price=Decimal("500"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-64",
        part_number="RAM-64G",
        producer="TestVendor",
        category_id="V110104",
        item_name="DDR4 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-1",
        part_number="SSD-960G",
        producer="TestVendor",
        category_id="V110106",
        item_name="Server SSD 960GB SATA",
        quantity=2,
        price=Decimal("200"),
    )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=512), _adapter(db_session))
    )
    build_candidate = result.to_report_json()["build_candidates"][0]

    assert build_candidate["cpu_per_server"] == 2
    assert build_candidate["total_cpu_required"] == 4
    assert build_candidate["total_price_value"] == "8800"
    assert build_candidate["total_price_note"] is None
    assert "cpu" in build_candidate["included_component_roles"]
    assert "cpu" not in build_candidate["missing_component_roles"]
    cpu_components = [
        component
        for component in build_candidate["components"]
        if component["role"] == "cpu"
    ]
    assert len(cpu_components) == 1
    assert cpu_components[0]["category_id"] == "V110103"
    assert cpu_components[0]["quantity_required"] == 4


def test_configuration_builder_v02_scans_cpu_pool_and_excludes_foreign_vendor_kit(
    db_session: Session,
) -> None:
    _seed_component_product(
        db_session,
        item_id="asus-platform",
        part_number="90SF03A1-M00070",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 2U dual socket DDR5 server platform 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="hpe-cpu-kit",
        part_number="P49616-B21",
        producer="HPE",
        category_id="V110103",
        item_name="HPE Intel Xeon Gold 5416S processor kit",
        quantity=100,
        price=Decimal("900"),
    )
    _seed_component_product(
        db_session,
        item_id="intel-cpu",
        part_number="BX807135416S",
        producer="Intel",
        category_id="V110103",
        item_name="Intel Xeon Gold 5416S tray processor",
        quantity=4,
        price=Decimal("700"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-ddr5",
        part_number="RAM-DDR5-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-1",
        part_number="SSD-960G",
        producer="Samsung",
        category_id="V110106",
        item_name="Server SSD 960GB SATA",
        quantity=2,
        price=Decimal("200"),
    )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=512), _adapter(db_session))
    )
    build_candidate = result.to_report_json()["build_candidates"][0]
    cpu_components = [
        component
        for component in build_candidate["components"]
        if component["role"] == "cpu"
    ]

    assert len(cpu_components) == 1
    assert cpu_components[0]["part_number"] == "BX807135416S"
    assert "P49616-B21" not in str(build_candidate)
    assert build_candidate["completeness_status"] == "complete"
    assert "cpu" not in build_candidate["missing_component_roles"]
    assert any(
        "поддерживаемых CPU" in warning
        for warning in build_candidate["compatibility_warnings"]
    )


def test_configuration_builder_v02_all_foreign_cpu_kits_leave_build_incomplete(
    db_session: Session,
) -> None:
    _seed_component_product(
        db_session,
        item_id="asus-platform",
        part_number="90SF03A1-M00070",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 2U dual socket DDR5 server platform 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="hpe-cpu-kit",
        part_number="P49616-B21",
        producer="HPE",
        category_id="V110103",
        item_name="HPE Intel Xeon Gold 5416S processor kit",
        quantity=4,
        price=Decimal("900"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-ddr5",
        part_number="RAM-DDR5-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-1",
        part_number="SSD-960G",
        producer="Samsung",
        category_id="V110106",
        item_name="Server SSD 960GB SATA",
        quantity=2,
        price=Decimal("200"),
    )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=512), _adapter(db_session))
    )
    build_candidate = result.to_report_json()["build_candidates"][0]
    summary = format_match_summary({"match_run_id": 1, **result.to_report_json()})

    assert build_candidate["completeness_status"] == "incomplete"
    assert build_candidate["missing_component_roles"] == ["cpu"]
    assert build_candidate["excluded_from_total_roles"] == ["cpu"]
    assert build_candidate["total_price_value"] == "6800"
    assert build_candidate["total_price_note"] == "без CPU"
    assert "P49616-B21" not in str(build_candidate)
    assert "P49616-B21" not in summary
    assert "CPU не подобраны" in summary


def test_configuration_builder_v02_allows_same_vendor_cpu_kit_with_support_warning(
    db_session: Session,
) -> None:
    _seed_component_product(
        db_session,
        item_id="hpe-platform",
        part_number="P55245-B21",
        producer="HPE",
        category_id="V110100",
        item_name="HPE ProLiant 2U dual socket DDR5 server platform 2x PSU",
        quantity=2,
        price=Decimal("3000"),
    )
    _seed_component_product(
        db_session,
        item_id="hpe-cpu-kit",
        part_number="P49616-B21",
        producer="HPE",
        category_id="V110103",
        item_name="HPE Intel Xeon Gold 5416S processor kit",
        quantity=4,
        price=Decimal("900"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-ddr5",
        part_number="RAM-DDR5-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-1",
        part_number="SSD-960G",
        producer="Samsung",
        category_id="V110106",
        item_name="Server SSD 960GB SATA",
        quantity=2,
        price=Decimal("200"),
    )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=512), _adapter(db_session))
    )
    build_candidate = result.to_report_json()["build_candidates"][0]
    cpu_components = [
        component
        for component in build_candidate["components"]
        if component["role"] == "cpu"
    ]

    assert len(cpu_components) == 1
    assert cpu_components[0]["producer"] == "HPE"
    assert cpu_components[0]["part_number"] == "P49616-B21"
    assert build_candidate["completeness_status"] == "complete"
    assert any(
        "поддерживаемых CPU" in warning
        for warning in build_candidate["compatibility_warnings"]
    )


def test_configuration_builder_v03_builds_candidate_matrix_and_diverse_components(
    db_session: Session,
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-2U-2S",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 2U dual socket DDR5 server platform 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    for item_id, part_number, item_name, price in [
        ("cpu-5416", "BX807135416S", "Intel Xeon Gold 5416S 16 core tray processor", "700"),
        ("cpu-4410", "BX807134410Y", "Intel Xeon Silver 4410Y 12 core tray processor", "500"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="Intel",
            category_id="V110103",
            item_name=item_name,
            quantity=4,
            price=Decimal(price),
        )
    for item_id, part_number, item_name, quantity, price in [
        ("ram-64", "RAM-DDR5-64G", "DDR5 RDIMM 64GB server memory module", 16, "150"),
        ("ram-128", "RAM-DDR5-128G", "DDR5 RDIMM 128GB server memory module", 8, "280"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="Samsung",
            category_id="V110104",
            item_name=item_name,
            quantity=quantity,
            price=Decimal(price),
        )
    for item_id, part_number, item_name, price in [
        ("ssd-960", "SSD-960G", "Server SSD 960GB SATA", "200"),
        ("ssd-1920", "SSD-1920G", "Server SSD 1.92TB SATA", "350"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="Samsung",
            category_id="V110106",
            item_name=item_name,
            quantity=2,
            price=Decimal(price),
        )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=512), _adapter(db_session))
    )
    report_json = result.to_report_json()
    build_candidates = report_json["build_candidates"]
    matrix = report_json["component_candidate_matrix"]

    assert len(build_candidates) >= 2
    assert {component["role"] for component in matrix["platform_candidates"]} == {
        "server_platform"
    }
    assert matrix["cpu_candidates"]
    assert matrix["ram_candidates"]
    assert matrix["ssd_candidates"]
    assert "candidate_id" in build_candidates[0]
    assert report_json["shortlist_for_llm"]

    cpu_parts = _component_part_numbers(build_candidates, "cpu")
    ram_parts = _component_part_numbers(build_candidates, "ram")
    ssd_parts = _component_part_numbers(build_candidates, "ssd")

    assert len(cpu_parts) > 1
    assert len(ram_parts) > 1
    assert len(ssd_parts) > 1


def test_right_size_storage_prefers_384_or_768_over_1536_when_cheaper(
    db_session: Session,
) -> None:
    _seed_right_size_base_components(db_session)
    for item_id, part_number, item_name, quantity, price in [
        ("ssd-3840", "SSD-3840", "Server SSD NVMe 3.84TB U.2", 4, "300"),
        ("ssd-7680", "SSD-7680", "Server SSD NVMe 7.68TB U.2", 4, "520"),
        ("ssd-15360", "SSD-15360", "Server SSD NVMe 15.36TB U.2", 4, "900"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="Samsung",
            category_id="V110106",
            item_name=item_name,
            quantity=quantity,
            price=Decimal(price),
        )

    result = asyncio.run(
        match_stock_spec(_right_size_server_spec(), _adapter(db_session))
    )
    report_json = result.to_report_json()
    top_build = report_json["build_candidates"][0]
    selected_ssd = _first_component(top_build, "ssd")
    matrix_ssd = report_json["component_candidate_matrix"]["ssd_candidates"]

    assert selected_ssd["part_number"] == "SSD-3840"
    assert selected_ssd["quantity_required"] == 4
    assert selected_ssd["fit_label"] == "exact_or_close_fit"
    assert matrix_ssd[0]["part_number"] == "SSD-3840"
    assert matrix_ssd[0]["storage_capacity_tb"] == 3.84
    assert matrix_ssd[0]["storage_over_requirement"] == 1.0
    assert top_build["optimization_mode"] == "cost_minimal_fit"
    assert top_build["right_size_note"] == "Подбор: минимально подходящий по требованиям"


def test_broad_matrix_keeps_close_cpu_buckets_under_small_limit(
    db_session: Session,
) -> None:
    _seed_right_size_base_components(db_session, include_cpu=False)
    for cores in [32, 24, 20, 16]:
        _seed_component_product(
            db_session,
            item_id=f"cpu-{cores}",
            part_number=f"CPU-{cores}C",
            producer="Intel",
            category_id="V110103",
            item_name=f"Intel Xeon Gold {cores} core tray processor",
            quantity=4,
            price=Decimal(str(400 + cores)),
        )
    _seed_component_product(
        db_session,
        item_id="ssd-3840",
        part_number="SSD-3840",
        producer="KIOXIA",
        category_id="V110106",
        item_name="KIOXIA CD8-R Server SSD NVMe 3.84TB U.2",
        quantity=4,
        price=Decimal("300"),
    )

    result = asyncio.run(
        match_stock_spec(
            _right_size_server_spec(),
            _adapter(db_session),
            llm_settings=LlmSettings(
                llm_provider="disabled",
                llm_component_candidates_per_role=3,
            ),
        )
    )
    report_json = result.to_report_json()
    cpu_parts = [
        row["part_number"]
        for row in report_json["component_candidate_matrix"]["cpu_candidates"]
    ]
    coverage = report_json["component_matrix_coverage_summary"]

    assert cpu_parts == ["CPU-16C", "CPU-20C", "CPU-24C"]
    assert coverage["eligible_products_by_role"]["cpu"] == 4
    assert coverage["sent_to_llm_by_role"]["cpu"] == 3
    assert coverage["omitted_by_role"]["cpu"] == 1
    assert coverage["selection_strategy"] == "bucketed_broad_matrix_v3"
    assert "cpu_16_cores" in coverage["bucket_summary_by_role"]["cpu"]


def test_broad_matrix_includes_exact_384_ssd_under_small_limit(
    db_session: Session,
) -> None:
    _seed_right_size_base_components(db_session)
    for item_id, part_number, item_name, price in [
        ("ssd-15360", "SSD-15360", "Server SSD NVMe 15.36TB U.2", "250"),
        ("ssd-7680", "SSD-7680", "Server SSD NVMe 7.68TB U.2", "260"),
        ("ssd-3840", "SSD-3840", "Server SSD NVMe 3.84TB U.2", "300"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="Samsung",
            category_id="V110106",
            item_name=item_name,
            quantity=4,
            price=Decimal(price),
        )

    result = asyncio.run(
        match_stock_spec(
            _right_size_server_spec(),
            _adapter(db_session),
            llm_settings=LlmSettings(
                llm_provider="disabled",
                llm_component_candidates_per_role=2,
            ),
        )
    )
    ssd_parts = [
        row["part_number"]
        for row in result.to_report_json()["component_candidate_matrix"]["ssd_candidates"]
    ]

    assert "SSD-3840" in ssd_parts


def test_right_size_storage_allows_1536_when_smaller_stock_is_insufficient(
    db_session: Session,
) -> None:
    _seed_right_size_base_components(db_session)
    for item_id, part_number, item_name, quantity, price in [
        ("ssd-3840", "SSD-3840", "Server SSD NVMe 3.84TB U.2", 2, "300"),
        ("ssd-7680", "SSD-7680", "Server SSD NVMe 7.68TB U.2", 1, "520"),
        ("ssd-15360", "SSD-15360", "Server SSD NVMe 15.36TB U.2", 4, "900"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="Samsung",
            category_id="V110106",
            item_name=item_name,
            quantity=quantity,
            price=Decimal(price),
        )

    result = asyncio.run(
        match_stock_spec(_right_size_server_spec(), _adapter(db_session))
    )
    top_build = result.to_report_json()["build_candidates"][0]
    selected_ssd = _first_component(top_build, "ssd")

    assert selected_ssd["part_number"] == "SSD-15360"
    assert selected_ssd["fit_label"] == "excessive_overfit"
    assert top_build["storage_over_requirement"] == 4.0
    assert top_build["right_size_note"].startswith("Подбор:")
    assert "Накопитель существенно выше требования" in top_build["right_size_note"]


def test_right_size_cpu_prefers_16_or_24_cores_before_32_when_price_is_not_better(
    db_session: Session,
) -> None:
    _seed_right_size_base_components(db_session, include_cpu=False)
    for item_id, part_number, item_name, price in [
        ("cpu-16", "CPU-16C", "Intel Xeon Silver 4416 16 core tray processor", "500"),
        ("cpu-24", "CPU-24C", "Intel Xeon Gold 5424 24 core tray processor", "650"),
        ("cpu-32", "CPU-32C", "Intel Xeon Gold 6430 32 core tray processor", "900"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="Intel",
            category_id="V110103",
            item_name=item_name,
            quantity=4,
            price=Decimal(price),
        )
    _seed_component_product(
        db_session,
        item_id="ssd-3840",
        part_number="SSD-3840",
        producer="Samsung",
        category_id="V110106",
        item_name="Server SSD NVMe 3.84TB U.2",
        quantity=4,
        price=Decimal("300"),
    )

    result = asyncio.run(
        match_stock_spec(_right_size_server_spec(), _adapter(db_session))
    )
    top_build = result.to_report_json()["build_candidates"][0]
    selected_cpu = _first_component(top_build, "cpu")

    assert selected_cpu["part_number"] == "CPU-16C"
    assert selected_cpu["cpu_cores"] == 16
    assert selected_cpu["cpu_over_requirement"] == 0
    assert selected_cpu["fit_label"] == "exact_or_close_fit"


def test_llm_configurator_disabled_keeps_rule_based_behavior(db_session: Session) -> None:
    _seed_complete_component_set(db_session)

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_settings=LlmSettings(llm_configurator_enabled=False),
        )
    )
    report_json = result.to_report_json()

    assert report_json["llm_configurator_enabled"] is False
    assert report_json["llm_configurator_used"] is False
    assert report_json["llm_recommended_build_candidates"] == []
    assert report_json["llm_fallback_reason"] == "llm_configurator_disabled"
    assert len(report_json["build_candidates"]) == 1


def test_matrix_includes_network_candidates_when_role_plan_requires_it(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    _seed_component_product(
        db_session,
        item_id="nic-25g",
        part_number="NIC-25G-DUAL",
        producer="Mellanox",
        category_id="V120116",
        item_name="Dual-port 25GbE SFP28 PCIe network adapter",
        quantity=2,
        price=Decimal("250"),
    )

    spec = _server_spec(quantity=2, ram_min_gb=512)
    spec.source_text = (
        "Need 2 servers with 2 CPU, 512GB RAM, SSD, "
        "2 ports 25GbE SFP28 per server."
    )
    spec.items[0].requirements["network"] = {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "25GbE",
        "media": "SFP28",
    }
    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_settings=LlmSettings(llm_configurator_enabled=False),
        )
    )
    report_json = result.to_report_json()

    assert "network_adapter" in report_json["required_roles"]
    assert report_json["component_candidate_matrix"]["network_adapter_candidates"]
    assert report_json["missing_required_roles"] == []


def test_matrix_accepts_network_media_attached_to_port_multiplier(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    _seed_component_product(
        db_session,
        item_id="nic-25g-quad",
        part_number="E810-XXVDA4",
        producer="Intel",
        category_id="V120116",
        item_name="Ethernet Network Adapter, 4xSFP28 ports, 25GbE PCIe adapter",
        quantity=2,
        price=Decimal("350"),
    )

    spec = _server_spec(quantity=2, ram_min_gb=512)
    spec.source_text = "Need 2 servers with 2 ports 25GbE SFP28 per server."
    spec.items[0].requirements["network"] = {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "25GbE",
        "media": "SFP28",
    }
    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_settings=LlmSettings(llm_configurator_enabled=False),
        )
    )
    report_json = result.to_report_json()
    network = report_json["component_candidate_matrix"]["network_adapter_candidates"][0]
    diagnostics = report_json["role_coverage_summary"]["network_adapter"][
        "filter_diagnostics"
    ]

    assert network["network_ports_count"] == 4
    assert network["network_speed"] == "25GbE"
    assert network["network_media"] == "SFP28"
    assert diagnostics["after_eligibility_count"] == 1
    assert not any(
        row["reason"] == "network_media_mismatch"
        for row in diagnostics["filtered_reasons_top"]
    )


def test_matrix_builder_keeps_network_technical_mismatch_as_non_selectable_candidate(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    _seed_component_product(
        db_session,
        item_id="nic-1g",
        part_number="NIC-1G-RJ45",
        producer="Intel",
        category_id="V120116",
        item_name="Quad-port 1GbE RJ45 PCIe network adapter",
        quantity=2,
        price=Decimal("50"),
    )

    spec = _server_spec(quantity=2, ram_min_gb=512)
    spec.source_text = "Need 2 servers with 2 ports 25GbE SFP28 per server."
    spec.items[0].requirements["network"] = {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "25GbE",
        "media": "SFP28",
    }
    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_settings=LlmSettings(llm_configurator_enabled=False),
        )
    )
    report_json = result.to_report_json()
    coverage = report_json["role_coverage_summary"]["network_adapter"]
    diagnostics = coverage["filter_diagnostics"]

    assert report_json["category_plan"]["network_adapter"] == ["V120116"]
    assert coverage["category_ids"] == ["V120116"]
    assert coverage["raw_products_count"] == 1
    assert diagnostics["after_category_count"] == 1
    assert diagnostics["after_eligibility_count"] == 1
    assert coverage["after_eligibility_count"] == 1
    assert coverage["fit_tier_counts"] == {}
    assert coverage["missing_candidates"] is True
    assert coverage["missing"] is True


def test_network_adapter_sfp28_uncertainty_reaches_llm_instead_of_missing(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    _seed_component_product(
        db_session,
        item_id="nic-25g-sfp28",
        part_number="NIC-25G-SFP28",
        producer="Intel",
        category_id="V120116",
        item_name="Dual-port 25GbE SFP28 PCIe Ethernet network adapter",
        quantity=2,
        price=Decimal("250"),
    )

    spec = _server_spec(quantity=2, ram_min_gb=512)
    spec.source_text = "Need 2 servers with Intel X710-DA2 2x10GbE SFP+ adapter."
    spec.items[0].requirements["network"] = {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "10GbE",
        "media": "SFP+",
    }
    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_settings=LlmSettings(llm_configurator_enabled=False),
        )
    )
    report_json = result.to_report_json()
    coverage = report_json["role_coverage_summary"]["network_adapter"]
    matrix_rows = report_json["component_candidate_matrix"]["network_adapter_candidates"]

    assert matrix_rows
    assert coverage["sent_to_llm_count"] > 0
    assert coverage["missing"] is False
    assert "network_adapter" not in report_json["missing_required_roles"]
    assert matrix_rows[0]["fit_tier"] in {"possible_fit", "strong_fit"}
    assert "network_media_family_compatibility_check" in matrix_rows[0]["match_warnings"]


def test_network_adapter_objective_rejects_no_stock_no_price_and_wrong_role(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    _seed_component_product(
        db_session,
        item_id="nic-no-stock",
        part_number="NIC-NOSTOCK",
        producer="Intel",
        category_id="V120116",
        item_name="Dual-port 10GbE SFP+ PCIe Ethernet network adapter",
        quantity=0,
        price=Decimal("120"),
    )
    _seed_component_product(
        db_session,
        item_id="nic-no-price",
        part_number="NIC-NOPRICE",
        producer="Intel",
        category_id="V120116",
        item_name="Dual-port 10GbE SFP+ PCIe Ethernet network adapter",
        quantity=2,
        price=Decimal("120"),
    )
    _seed_component_product(
        db_session,
        item_id="fc-hba",
        part_number="FC-HBA",
        producer="StorageVendor",
        category_id="V120116",
        item_name="Dual-port Fibre Channel FC HBA 16G SFP+ adapter",
        quantity=2,
        price=Decimal("90"),
    )
    for row in db_session.query(DistributorStockPrice).filter_by(item_id="nic-no-price"):
        row.price_order_value = None
        row.price_list_value = None
        row.end_user_value = None
    db_session.commit()

    spec = _server_spec(quantity=2, ram_min_gb=512)
    spec.source_text = "Need 2 servers with 2x10GbE SFP+ network adapter."
    spec.items[0].requirements["network"] = {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "10GbE",
        "media": "SFP+",
    }
    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_settings=LlmSettings(llm_configurator_enabled=False),
        )
    )
    coverage = result.to_report_json()["role_coverage_summary"]["network_adapter"]
    reasons = {
        row["reason"]
        for row in coverage["filter_diagnostics"]["filtered_reasons_top"]
    }

    assert coverage["after_eligibility_count"] == 0
    assert {"no_stock", "no_price", "wrong_role_objective"}.issubset(reasons)


def test_server_83_like_network_adapter_candidates_reach_composer(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    for item_id, part_number, name, price in [
        (
            "nic-x710-like",
            "X710-DA2-LIKE",
            "Intel Ethernet Network Adapter X710-DA2 2x10GbE SFP+ PCIe adapter",
            "300",
        ),
        (
            "nic-4x10g",
            "NIC-4X10G-SFP",
            "Quad-port 4x10GbE SFP+ PCIe Ethernet network adapter",
            "350",
        ),
        (
            "nic-25g-sfp28",
            "NIC-25G-SFP28",
            "Dual-port 25GbE SFP28 PCIe Ethernet network adapter",
            "250",
        ),
        (
            "nic-rj45",
            "NIC-10G-RJ45",
            "Dual-port 10GbE RJ45 PCIe Ethernet network adapter",
            "100",
        ),
        (
            "fc-hba",
            "FC-HBA",
            "Dual-port Fibre Channel FC HBA 16G SFP+ adapter",
            "90",
        ),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="TestVendor",
            category_id="V120116",
            item_name=name,
            quantity=2,
            price=Decimal(price),
        )

    spec = _server_spec(quantity=2, ram_min_gb=512)
    spec.source_text = "Need 2 servers with Intel X710-DA2 2x10GbE SFP+ network adapter."
    spec.items[0].requirements["network"] = {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "10GbE",
        "media": "SFP+",
    }
    client = _FakeComposerClient(_server_78_full_llm_response)
    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_configurator_client=client,
            llm_settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        )
    )
    report_json = result.to_report_json()
    coverage = report_json["role_coverage_summary"]["network_adapter"]
    package_rows = client.package["component_candidate_matrix"]["network_adapter"]
    package_names = {row["name"] for row in package_rows}

    assert client.calls == 1
    assert report_json["composer_attempt_decision"]["should_attempt"] is True
    assert report_json["composer_attempt_decision"]["blocked_by"] == []
    assert coverage["sent_to_llm_count"] > 0
    assert "network_adapter" not in report_json["missing_required_roles"]
    assert any("X710" in name for name in package_names)
    assert any("SFP28" in name for name in package_names)
    assert not any("Fibre Channel" in name for name in package_names)


def test_role_coverage_platform_capability_requires_actual_platform_candidate() -> None:
    required_capabilities = [
        {
            "capability_id": "network.25gbe.sfp28",
            "role": "network_adapter",
            "hard": True,
            "source_text": "2 ports 25GbE SFP28 per server",
            "parsed_requirements": {
                "min_ports_per_server": 2,
                "speed": "25GbE",
                "media": "SFP28",
            },
            "can_be_satisfied_by_platform": True,
        }
    ]
    coverage = match_engine_module._role_coverage_summary(
        required_roles=["network_adapter"],
        required_capabilities=required_capabilities,
        category_plan={"network_adapter": ["V120116"]},
        selected_by_role={},
        planner_role_coverage={
            "network_adapter": {
                "can_be_satisfied_by_platform": True,
                "missing_category": False,
            }
        },
        role_filter_diagnostics={
            "network_adapter": {
                "raw_products_count": 1,
                "after_category_count": 1,
                "after_fact_extraction_count": 1,
                "after_eligibility_count": 0,
                "filtered_reasons_top": [
                    {"reason": "network_media_mismatch", "count": 1}
                ],
            }
        },
        platform_satisfaction_counts={"network_adapter": 0},
    )

    assert coverage["network_adapter"]["can_be_satisfied_by_platform"] is True
    assert coverage["network_adapter"]["missing_candidates"] is True
    assert coverage["network_adapter"]["missing"] is True


def test_role_coverage_platform_satisfied_candidate_suppresses_precomposer_missing() -> None:
    coverage = match_engine_module._role_coverage_summary(
        required_roles=["power_supply"],
        required_capabilities=[
            {
                "capability_id": "power_supply.min_2",
                "role": "power_supply",
                "hard": True,
                "source_text": "2 PSU per server",
                "can_be_satisfied_by_platform": True,
            }
        ],
        category_plan={},
        selected_by_role={},
        planner_role_coverage={
            "power_supply": {
                "can_be_satisfied_by_platform": True,
                "missing_category": False,
            }
        },
        role_filter_diagnostics={},
        platform_satisfaction_counts={"power_supply": 1},
    )

    assert coverage["power_supply"]["platform_satisfied_candidates_count"] == 1
    assert coverage["power_supply"]["missing_candidates"] is False
    assert coverage["power_supply"]["missing"] is False


def test_matrix_builder_uses_category_plan_for_gpu_role(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    _seed_component_product(
        db_session,
        item_id="gpu-l40s",
        part_number="GPU-L40S",
        producer="NVIDIA",
        category_id="GPU-CAT",
        item_name="NVIDIA L40S GPU accelerator",
        quantity=2,
        price=Decimal("1500"),
    )
    spec = _server_spec(quantity=2, ram_min_gb=512)
    spec.source_text += " Need NVIDIA GPU accelerator."

    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_settings=LlmSettings(llm_configurator_enabled=False),
        )
    )
    report_json = result.to_report_json()
    matrix = report_json["component_candidate_matrix"]

    assert "gpu" in report_json["required_roles"]
    assert report_json["category_plan"]["gpu"] == ["GPU-CAT"]
    assert matrix["gpu_candidates"][0]["role"] == "gpu"
    assert matrix["gpu_candidates"][0]["capability_id"] == "gpu.requested"
    assert report_json["missing_required_capabilities"] == []


def test_matrix_builder_materializes_validated_ai_category_plan_with_synthetic_server_item(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    _seed_component_product(
        db_session,
        item_id="controller-1",
        part_number="HBA-9500-8I",
        producer="Broadcom",
        category_id="V110107",
        item_name="Broadcom LSI 9500-8i Tri-Mode HBA storage controller",
        quantity=2,
        price=Decimal("450"),
    )
    _seed_component_product(
        db_session,
        item_id="nic-1",
        part_number="X710-DA2",
        producer="Intel",
        category_id="V120116",
        item_name="Intel X710-DA2 dual-port 10GbE SFP+ PCIe network adapter",
        quantity=2,
        price=Decimal("220"),
    )
    products = list(db_session.query(DistributorProduct).all())
    stock_rows_by_key = asyncio.run(
        match_engine_module._load_latest_stock_rows(_adapter(db_session), products)
    )
    role_plan = {
        "product_group": "server",
        "semantic_planner_source": "llm",
        "matrix_blueprint_roles": [
            "server_platform",
            "cpu",
            "ram",
            "storage",
            "storage_controller",
            "network_adapter",
        ],
        "required_roles": [
            "server_platform",
            "cpu",
            "ram",
            "storage",
            "storage_controller",
            "network_adapter",
        ],
        "required_capabilities": [
            {"role": "server_platform", "capability_id": "server_platform.1u", "hard": True},
            {"role": "cpu", "capability_id": "cpu.intel", "hard": True},
            {"role": "ram", "capability_id": "ram.ddr5", "hard": True},
            {"role": "storage", "capability_id": "storage.sata", "hard": True},
            {
                "role": "storage_controller",
                "capability_id": "storage_controller.hba",
                "hard": True,
            },
            {
                "role": "network_adapter",
                "capability_id": "network_adapter.10gbe.sfpplus",
                "hard": True,
                "parsed_requirements": {
                    "min_ports_per_server": 2,
                    "speed": "10GbE",
                    "media": "SFP+",
                },
            },
        ],
        "requirements_by_role": {
            "cpu": {"vendor": "Intel"},
            "ram": {"type": "DDR5"},
            "storage": {"interface": "SATA"},
            "network_adapter": {
                "required": True,
                "min_ports_per_server": 2,
                "speed": "10GbE",
                "media": "SFP+",
            },
        },
    }
    category_plan_result = match_engine_module.CategoryPlanResult(
        category_plan={
            "server_platform": ["V110100"],
            "cpu": ["V110103"],
            "ram": ["V110104"],
            "storage": ["V110106"],
            "storage_controller": ["V110107"],
            "network_adapter": ["V120116"],
        },
        category_planner_source="ai_category_planner",
        category_plan_source="llm",
    )
    spec = StockSpec(items=[], source_text="1U server with Intel CPU DDR5 SATA SSD HBA NIC")

    _, _, _, matrix, normalized = match_engine_module._build_configuration_candidates(
        spec=spec,
        products=products,
        stock_rows_by_key=stock_rows_by_key,
        role_plan=role_plan,
        category_plan_result=category_plan_result,
    )

    assert normalized
    assert matrix["category_planner_source"] == "ai_category_planner"
    assert matrix["platform_candidates"]
    assert matrix["cpu_candidates"]
    assert matrix["ram_candidates"]
    assert matrix["ssd_candidates"]
    assert matrix["storage_controller_candidates"]
    assert matrix["network_adapter_candidates"]
    assert matrix["role_coverage_summary"]["storage"]["sent_to_llm_count"] >= 1


def test_matrix_builder_does_not_add_unplanned_cpu_category(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    products = list(db_session.query(DistributorProduct).all())

    products_by_role = match_engine_module._products_by_server_role(
        products,
        product_group="server",
        category_plan={"server_platform": ["V110100"], "ram": ["V110104"]},
    )

    assert "cpu" not in products_by_role
    assert {product.category_id for product in products_by_role["server_platform"]} == {
        "V110100"
    }


def test_matrix_builder_marks_empty_hard_role_as_missing_required_capability(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    spec = _server_spec(quantity=2, ram_min_gb=512)
    spec.source_text += " Need NVIDIA GPU accelerator."

    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_settings=LlmSettings(llm_configurator_enabled=False),
        )
    )
    report_json = result.to_report_json()

    assert "gpu" in report_json["missing_category_roles"]
    assert any(
        row["role"] == "gpu" and row["status"] == "missing_category"
        for row in report_json["missing_required_capabilities"]
    )


def test_llm_configurator_with_fake_client_adds_valid_recommendation(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)
    fake_client = _FakeComposerClient(_valid_llm_response)

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=fake_client,
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()
    llm_build = report_json["llm_recommended_build_candidates"][0]

    assert report_json["llm_configurator_enabled"] is True
    assert report_json["llm_configurator_used"] is True
    assert llm_build["candidate_id"] == "llm_build_1"
    assert llm_build["total_price_value"] == "8800"
    assert llm_build["total_price_currency"] == "USD"
    assert llm_build["why_selected"] == "Balanced stocked configuration."
    assert "component_candidate_id" in fake_client.package["component_candidate_matrix"]["cpu"][0]
    assert "ready_stock_candidates" in fake_client.package
    assert "rule_based_build_candidates" in fake_client.package
    assert "raw_json" not in str(fake_client.package)


def test_web_evidence_pack_is_included_in_report_json(db_session: Session) -> None:
    _seed_complete_component_set(db_session)
    provider = FakeWebSearchProvider(
        {
            "PLATFORM-2U-2S": [
                {
                    "title": "ASUS platform specifications",
                    "url": "https://servers.asus.com/platform",
                    "snippet": "ASUS platform supports LGA4677, DDR5 and NVMe bays.",
                }
            ],
            "CPU-SERVER": [
                {
                    "title": "Intel CPU specifications",
                    "url": "https://ark.intel.com/cpu",
                    "snippet": "4th Gen Intel Xeon Scalable processor, FCLGA4677 socket.",
                }
            ],
        }
    )

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerAndEvidenceReviewClient(
                _valid_llm_response
            ),
            llm_settings=_llm_composer_settings(),
            web_evidence_settings=_web_evidence_settings(),
            web_search_provider=provider,
        )
    )
    report_json = result.to_report_json()

    assert report_json["web_evidence_pack"]["enabled"] is True
    assert report_json["web_evidence_pack"]["completed_tasks"] >= 1
    assert "web_evidence_pack" in report_json
    assert "llm_evidence_review" in report_json
    diagnostics = report_json["web_evidence_diagnostics"]
    assert diagnostics["evidence_tasks_count"] >= 1
    assert diagnostics["evidence_tasks_count_by_type"]
    assert "relation_evidence_count" in diagnostics
    assert "relation_mismatch_count" in diagnostics
    assert "relation_partially_confirmed_count" in diagnostics
    assert "relation_not_confirmed_count" in diagnostics
    assert diagnostics["evidence_completed_count"] >= 1
    assert diagnostics["evidence_provider"] == "fake"
    assert "test-key" not in json.dumps(diagnostics, ensure_ascii=False)


def test_llm_configurator_package_uses_deterministic_candidate_order() -> None:
    cpu_rows = [
        _package_component_candidate(
            "cpu-a",
            score=90,
            price="200",
            over_requirement=10,
            producer="Beta",
            part_number="CPU-A",
        ),
        _package_component_candidate(
            "cpu-b",
            score=90,
            price="100",
            over_requirement=20,
            producer="Alpha",
            part_number="CPU-B",
        ),
        _package_component_candidate(
            "cpu-c",
            score=95,
            price="500",
            over_requirement=0,
            producer="Gamma",
            part_number="CPU-C",
        ),
        _package_component_candidate(
            "cpu-d",
            score=90,
            price="100",
            over_requirement=5,
            producer="Delta",
            part_number="CPU-D",
        ),
    ]
    ready_candidates = [
        _package_ready_candidate("ready-b", score=80, matched=1, price="900", producer="B"),
        _package_ready_candidate("ready-a", score=90, matched=1, price="1200", producer="A"),
        _package_ready_candidate("ready-c", score=90, matched=2, price="1500", producer="C"),
    ]
    build_candidates = [
        _package_build_candidate("build-b", complete=True, price="9000", score=70),
        _package_build_candidate("build-a", complete=True, price="8000", score=80),
        _package_build_candidate(
            "build-c",
            complete=False,
            price="7000",
            score=95,
            missing_roles=["ram"],
        ),
    ]

    package = build_llm_configurator_package(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=list(reversed(ready_candidates)),
        component_candidate_matrix={"cpu_candidates": list(reversed(cpu_rows))},
        rule_based_build_candidates=list(reversed(build_candidates)),
    )

    assert [
        row["component_candidate_id"]
        for row in package["component_candidate_matrix"]["cpu"]
    ] == ["cpu-c", "cpu-d", "cpu-b", "cpu-a"]
    assert [row["candidate_id"] for row in package["ready_stock_candidates"]] == [
        "ready-c",
        "ready-a",
        "ready-b",
    ]
    assert [row["candidate_id"] for row in package["rule_based_build_candidates"]] == [
        "build-a",
        "build-b",
        "build-c",
    ]


def test_llm_configurator_package_excludes_ready_server_for_non_server_groups() -> None:
    ready_row = _composer_component_candidate(
        "ready-server-row",
        "ServerVendor",
        "READY-ROW",
        "Ready server must not leak into network or storage package",
        1,
        Decimal("5000"),
        {"cpu_cores": 32},
    )
    ready_stock = [
        _package_ready_candidate(
            "ready-stock-row",
            score=90,
            matched=2,
            price="5000",
            producer="ServerVendor",
        )
    ]
    switch = _composer_component_candidate(
        "switch-row",
        "NetVendor",
        "SW-48P",
        "48-port 1G PoE+ switch 4 uplink 10G SFP+ L3 stacking",
        1,
        Decimal("1200"),
        {"port_count": 48, "poe_supported": True, "l3_supported": True},
    )
    storage = _composer_component_candidate(
        "storage-row",
        "StorageVendor",
        "ARR-100",
        "Storage array 120TB usable dual controller FC 32G",
        1,
        Decimal("10000"),
        {"usable_capacity_tb": 120, "host_protocol": "FC"},
    )

    network_package = build_llm_configurator_package(
        user_request="Need a switch.",
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=ready_stock,
        component_candidate_matrix={
            "product_group": "network",
            "ready_server_candidates": [ready_row],
            "switch_candidates": [switch],
        },
        rule_based_build_candidates=[],
    )
    storage_package = build_llm_configurator_package(
        user_request="Need a storage array.",
        normalized_requirements=[_storage_composer_requirements()],
        ready_stock_candidates=ready_stock,
        component_candidate_matrix={
            "product_group": "storage",
            "ready_server_candidates": [ready_row],
            "storage_system_candidates": [storage],
        },
        rule_based_build_candidates=[],
    )
    server_package = build_llm_configurator_package(
        user_request="Need a server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=ready_stock,
        component_candidate_matrix={
            "product_group": "server",
            "ready_server_candidates": [ready_row],
        },
        rule_based_build_candidates=[],
    )

    assert "ready_server" not in network_package["component_candidate_matrix"]
    assert network_package["ready_stock_candidates"] == []
    assert "ready_server" not in storage_package["component_candidate_matrix"]
    assert storage_package["ready_stock_candidates"] == []
    assert server_package["component_candidate_matrix"]["ready_server"]
    assert server_package["ready_stock_candidates"]


def test_llm_configurator_package_excludes_ready_candidates_on_ai_matrix_path() -> None:
    ready_stock = [
        _package_ready_candidate(
            f"ready-{index:03d}",
            score=90,
            matched=4,
            price="9000",
            producer="ServerVendor",
        )
        | {
            "item_name": f"Legacy ready server {index} " + ("full raw stock text " * 200),
            "missing_requirements": ["diagnostic " + ("x" * 200)] * 8,
            "risk_flags": ["risk " + ("y" * 200)] * 8,
        }
        for index in range(80)
    ]
    matrix = _server_78_like_broad_matrix(rows_per_role=1)

    package = build_llm_configurator_package(
        user_request="Need server #78 from component matrix.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=ready_stock,
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        max_package_chars=120000,
    )

    section_sizes = package["package_budget"]["section_size_diagnostics"]
    assert package["ready_stock_candidates"] == []
    assert "ready_server" not in package["component_candidate_matrix"]
    assert (
        package["ready_candidates_excluded_reason"]
        == configuration_composer_module.READY_CANDIDATES_EXCLUDED_AI_CATEGORY_PLAN_REASON
    )
    assert section_sizes["ready_candidates_chars"] == len("[]")
    assert package["package_budget"]["final_chars"] == len(
        json.dumps(package, ensure_ascii=False, sort_keys=True, default=str)
    )
    assert package["package_budget"]["final_chars"] < 120000
    assert package["package_budget"]["over_budget"] is False
    assert package.get("package_skipped_reason") is None


def test_huge_legacy_ready_candidates_cannot_dominate_ai_category_plan_package() -> None:
    ready_stock = [
        _package_ready_candidate(
            f"ready-huge-{index:03d}",
            score=90,
            matched=3,
            price="10000",
            producer="ServerVendor",
        )
        | {
            "item_name": f"Legacy ready candidate {index} " + ("raw catalog " * 500),
            "matched_requirements": ["matched " + ("z" * 500)] * 20,
            "missing_requirements": ["missing " + ("x" * 500)] * 20,
            "risk_flags": ["risk " + ("y" * 500)] * 20,
        }
        for index in range(120)
    ]

    package = build_llm_configurator_package(
        user_request="Need a configured server with CPU RAM SSD NIC PSU cable.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=ready_stock,
        component_candidate_matrix=_server_78_like_broad_matrix(rows_per_role=2),
        rule_based_build_candidates=[],
        max_package_chars=120000,
    )

    section_sizes = package["package_budget"]["section_size_diagnostics"]
    assert package["ready_stock_candidates"] == []
    assert section_sizes["ready_candidates_chars"] == len("[]")
    assert section_sizes["matrix_chars"] > section_sizes["ready_candidates_chars"]
    assert package["package_budget"]["final_chars"] <= 120000
    assert package["package_budget"]["over_budget"] is False


def test_ready_server_package_still_includes_compact_candidates_when_requested() -> None:
    ready_stock = [
        _package_ready_candidate(
            f"ready-{index}",
            score=100 - index,
            matched=4,
            price=str(8000 + index),
            producer="ServerVendor",
        )
        | {"item_name": f"Ready server {index} " + ("with catalog details " * 100)}
        for index in range(8)
    ]

    package = build_llm_configurator_package(
        user_request="Need a ready server from stock.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=ready_stock,
        component_candidate_matrix={"product_group": "server"},
        rule_based_build_candidates=[],
        max_package_chars=120000,
    )

    assert len(package["ready_stock_candidates"]) == (
        configuration_composer_module.READY_SERVER_CANDIDATES_LIMIT
    )
    assert package["ready_candidates_limit"] == (
        configuration_composer_module.READY_SERVER_CANDIDATES_LIMIT
    )
    assert "ready_candidates_excluded_reason" not in package
    assert package["package_budget"]["section_size_diagnostics"][
        "ready_candidates_limit"
    ] == configuration_composer_module.READY_SERVER_CANDIDATES_LIMIT
    assert package["package_budget"]["over_budget"] is False


def test_llm_configurator_package_skips_technical_empty_planning_state() -> None:
    ready_stock = [
        _package_ready_candidate(
            f"ready-{index}",
            score=90,
            matched=1,
            price="5000",
            producer="ServerVendor",
        )
        for index in range(25)
    ]

    package = build_llm_configurator_package(
        user_request=(
            "Need 1U server with CPU, DDR5 RAM, SATA SSD, HBA, "
            "Intel X710 SFP+, C13 power cables"
        ),
        normalized_requirements=[],
        ready_stock_candidates=ready_stock,
        component_candidate_matrix={},
        rule_based_build_candidates=[],
    )

    assert package["semantic_planner_source"] == "planner_unavailable"
    assert package["product_group"] == "unknown"
    assert package["required_capabilities"] == []
    assert package["category_plan"] == {}
    assert package["ready_stock_candidates"] == []
    assert package["component_candidate_matrix"] == {}
    assert package["package_skipped_reason"] == "planner_unavailable"
    assert package["package_budget"]["final_chars"] < 120000


def test_llm_configurator_package_skips_empty_matrix_after_category_plan() -> None:
    package = build_llm_configurator_package(
        user_request="Need a server with CPU and RAM.",
        normalized_requirements=[],
        ready_stock_candidates=[
            _package_ready_candidate(
                "ready-huge",
                score=90,
                matched=1,
                price="9000",
                producer="ServerVendor",
            )
        ],
        component_candidate_matrix={
            "product_group": "server",
            "semantic_planner_source": "llm",
            "category_planner_source": "ai_category_planner",
            "category_plan": {"cpu": ["CPU-CAT"]},
            "role_coverage_summary": {
                "cpu": {
                    "required": True,
                    "category_ids": ["CPU-CAT"],
                    "category_count": 1,
                    "sent_to_llm_count": 0,
                    "raw_products_count": 0,
                    "after_category_count": 0,
                    "after_eligibility_count": 0,
                    "missing_candidates": True,
                    "missing": True,
                }
            },
        },
        rule_based_build_candidates=[
            _package_build_candidate("build-huge", complete=True, price="9000", score=90)
        ],
    )

    assert package["package_skipped_reason"] == "matrix_empty_after_category_plan"
    assert package["ready_stock_candidates"] == []
    assert package["rule_based_build_candidates"] == []
    assert package["role_coverage_summary"]["cpu"]["category_ids"] == ["CPU-CAT"]
    assert package["package_budget"]["final_chars"] < 120000


def test_llm_configurator_package_excludes_unselectable_fit_tiers() -> None:
    good_switch = _composer_component_candidate(
        "switch-good",
        "NetVendor",
        "SW-GOOD",
        "48-port 1G PoE+ switch 4x10G SFP+ L3 stacking",
        1,
        Decimal("1200"),
        {"port_count": 48, "uplink_count": 4, "poe_supported": True},
    )
    good_switch["fit_tier"] = "strong_fit"
    tiny_switch = _composer_component_candidate(
        "switch-tiny",
        "NetVendor",
        "SW-TINY",
        "5-Port 100Base-TX unmanaged switch",
        1,
        Decimal("25"),
        {"port_count": 5, "port_speed_gbps": 0.1},
    )
    tiny_switch["fit_tier"] = "explicit_mismatch"
    cable = _composer_component_candidate(
        "switch-cable",
        "NetVendor",
        "CAB",
        "KVM cable",
        1,
        Decimal("8"),
        {},
    )
    cable["fit_tier"] = "wrong_role"

    package = build_llm_configurator_package(
        user_request="Need a 48-port PoE L3 stacking switch.",
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix={
            "product_group": "network",
            "switch_candidates": [tiny_switch, cable, good_switch],
        },
        rule_based_build_candidates=[],
    )

    switch_ids = [
        row["component_candidate_id"]
        for row in package["component_candidate_matrix"]["switch"]
    ]
    assert switch_ids == ["switch-good"]
    assert package["component_candidate_matrix"]["switch"][0]["fit_tier"] == "strong_fit"


def test_llm_configurator_package_includes_network_role_contract() -> None:
    package = build_llm_configurator_package(
        user_request="Need a 48-port PoE L3 switch.",
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_network_switch_component_matrix(close_switch=True),
        rule_based_build_candidates=[],
    )

    contract = package["component_role_contract"]
    assert contract["product_group"] == "network"
    assert "switch" in contract["allowed_component_role_keys"]
    assert "switch" in contract["base_device_roles"]
    assert "component_candidate_ids.switch, not platform" in (
        contract["primary_role_key_guidance"]
    )
    assert "platform/cpu/ram/storage" in contract["forbidden_role_guidance"]


def test_llm_configurator_package_includes_storage_role_contract() -> None:
    package = build_llm_configurator_package(
        user_request="Need a storage system.",
        normalized_requirements=[_storage_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_storage_component_matrix(),
        rule_based_build_candidates=[],
    )

    contract = package["component_role_contract"]
    assert contract["product_group"] == "storage"
    assert "storage_system" in contract["allowed_component_role_keys"]
    assert contract["base_device_roles"] == ["storage_system"]
    assert "storage_system" in contract["primary_role_key_guidance"]
    assert "Do not use server CPU/RAM/platform roles" in contract["forbidden_role_guidance"]


def test_llm_configurator_package_keeps_candidates_and_reports_over_budget() -> None:
    cpu_rows = [
        _composer_component_candidate(
            f"cpu-{index:03d}",
            "CPUVendor",
            f"CPU-{index:03d}",
            f"Server CPU candidate {index:03d} " + ("with long fit text " * 10),
            4,
            Decimal("500"),
            {
                "cpu_cores": 16 + index,
                "raw": {"payload": "must not enter prompt"},
                "raw_json": {"secret": "must not enter prompt"},
                "debug": True,
            },
        )
        for index in range(90)
    ]
    kwargs = {
        "user_request": "Need 2 servers.",
        "normalized_requirements": [_composer_requirements()],
        "ready_stock_candidates": [],
        "component_candidate_matrix": {"product_group": "server", "cpu_candidates": cpu_rows},
        "rule_based_build_candidates": [],
        "candidates_per_role": 90,
        "max_package_chars": 18000,
    }

    package_a = build_llm_configurator_package(**kwargs)
    package_b = build_llm_configurator_package(**kwargs)
    ids_a = [
        row["component_candidate_id"]
        for row in package_a["component_candidate_matrix"]["cpu"]
    ]
    ids_b = [
        row["component_candidate_id"]
        for row in package_b["component_candidate_matrix"]["cpu"]
    ]
    package_text = json.dumps(package_a, ensure_ascii=False)

    assert package_a["package_budget"]["over_budget"] is True
    assert package_a["package_budget"]["trimmed"] is False
    assert package_a["package_skipped_reason"] == "package_over_budget_before_composer"
    assert ids_a == ids_b
    assert len(ids_a) == 90
    assert '"raw"' not in package_text
    assert '"raw_json"' not in package_text
    assert '"debug"' not in package_text
    assert package_a["composer_package_candidate_count_by_role"]["cpu"] == 90
    assert package_a["dropped_before_composer_count_by_role"]["cpu"] == 0


def test_86_like_broad_package_exposes_all_candidates_under_budget() -> None:
    matrix = _server_85_like_broad_matrix()
    package = build_llm_configurator_package(
        user_request="Need server #86-like broad matrix.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        candidates_per_role=3,
        max_package_chars=1_000_000,
    )
    expected_counts = matrix["broad_count_by_role"]

    assert package["package_budget"]["over_budget"] is False
    assert package.get("package_skipped_reason") is None
    assert package["broad_matrix_count_by_role"] == expected_counts
    assert package["composer_package_candidate_count_by_role"] == expected_counts
    assert package["composer_package_candidate_total"] == sum(expected_counts.values())
    assert all(
        count == 0
        for count in package["dropped_before_composer_count_by_role"].values()
    )
    assert package["package_candidate_exposure_incomplete"] is False
    assert package["package_candidate_exposure_policy"]["mode"] == (
        "full_broad_matrix"
    )


def test_incomplete_package_exposure_blocks_composer() -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=1)
    matrix["broad_count_by_role"] = {
        role: 2 for role in matrix["role_plan"]["required_roles"]
    }
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server with incomplete package exposure.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={"llm_configurator_max_package_chars": 1_000_000}
        ),
    )

    diagnostics = outcome.package_diagnostics
    assert client.package == {}
    assert outcome.fallback_reason == "incomplete_matrix_exposure"
    assert outcome.final_status_source == "matrix_exposure_incomplete"
    assert diagnostics["package_candidate_exposure_incomplete"] is True
    assert diagnostics["package_candidate_exposure_incomplete_roles"] == (
        matrix["role_plan"]["required_roles"]
    )
    assert diagnostics["dropped_before_composer_count_by_role"]["cpu"] == 1
    assert diagnostics["dropped_before_composer_reason_by_role"]["cpu"] == (
        "pre_composer_candidate_drop"
    )


def test_materialization_role_drops_include_candidate_counts() -> None:
    role_coverage_summary = {
        "server_platform": {
            "required": True,
            "category_count": 1,
            "sent_to_llm_count": 0,
            "after_category_count": 1,
            "after_eligibility_count": 0,
            "missing_category": False,
            "missing_candidates": True,
        },
        "cpu": {
            "required": True,
            "category_count": 1,
            "sent_to_llm_count": 1,
            "after_category_count": 1,
            "after_eligibility_count": 1,
            "missing_category": False,
            "missing_candidates": False,
        },
    }

    dropped = match_engine_module._roles_dropped_during_materialization(
        validated_category_plan_roles=["server_platform", "cpu"],
        materialized_matrix_roles=["cpu"],
        role_coverage_summary=role_coverage_summary,
    )
    reasons = match_engine_module._materialization_drop_reasons(
        dropped,
        role_coverage_summary,
    )

    assert dropped == ["server_platform"]
    assert reasons["server_platform"] == (
        "no_products_after_eligibility_filter:"
        "after_category_count=1:after_eligibility_count=0"
    )


def test_package_exposure_incomplete_when_lifecycle_role_has_no_candidates_or_reason() -> None:
    cpu_row = _composer_component_candidate(
        "cpu-only",
        "Intel",
        "CPU-ONLY",
        "Intel CPU candidate",
        2,
        Decimal("100"),
        {"cpu_cores": 24},
    )
    matrix = {
        "product_group": "server",
        "role_plan": {
            "product_group": "server",
            "required_roles": ["server_platform", "cpu"],
            "required_capabilities": [],
        },
        "stage_a_broad_roles": ["server_platform", "cpu"],
        "effective_matrix_roles_before_category_planner": ["server_platform", "cpu"],
        "category_planner_input_roles": ["server_platform", "cpu"],
        "validated_category_plan_roles": ["server_platform", "cpu"],
        "materialized_matrix_roles": ["cpu"],
        "category_plan": {
            "server_platform": ["platform"],
            "cpu": ["cpu"],
        },
        "cpu_candidates": [cpu_row],
        "broad_count_by_role": {"cpu": 1},
    }

    package = build_llm_configurator_package(
        user_request="Need server lifecycle invariant.",
        normalized_requirements=[
            {
                "product_group": "server",
                "required_roles": ["server_platform", "cpu"],
                "required_capabilities": [],
            }
        ],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        max_package_chars=1_000_000,
    )

    assert package["package_candidate_exposure_incomplete"] is True
    assert "server_platform" in package["package_candidate_exposure_incomplete_roles"]
    assert package["package_skipped_reason"] == "incomplete_matrix_exposure"
    trace_by_role = {
        row["role"]: row for row in package["role_lifecycle_trace"]
    }
    assert trace_by_role["server_platform"]["category_planner_input"] is True
    assert trace_by_role["server_platform"]["materialized_matrix"] is False
    assert trace_by_role["server_platform"]["composer_package"] is False


def test_package_exposure_blocks_required_lifecycle_missing_category() -> None:
    cpu_row = _composer_component_candidate(
        "cpu-only",
        "Intel",
        "CPU-ONLY",
        "Intel CPU candidate",
        2,
        Decimal("100"),
        {"cpu_cores": 24},
    )
    matrix = {
        "product_group": "server",
        "primary_object": "server",
        "role_plan": {
            "product_group": "server",
            "primary_object": "server",
            "required_roles": ["server_platform", "cpu"],
            "required_capabilities": [],
        },
        "stage_a_broad_roles": ["server_platform", "cpu"],
        "effective_matrix_roles_before_category_planner": ["server_platform", "cpu"],
        "category_planner_input_roles": ["server_platform", "cpu"],
        "category_planner_output_roles": ["cpu"],
        "validated_category_plan_roles": ["cpu"],
        "materialized_matrix_roles": ["cpu"],
        "category_planner_missing_required_roles": ["server_platform"],
        "category_planner_repair_attempted": True,
        "category_planner_repair_success": False,
        "category_planner_repair_reason": "no_category_found:server_platform",
        "category_planner_unresolved_required_roles": ["server_platform"],
        "category_plan": {"cpu": ["cpu"]},
        "role_coverage_summary": {
            "server_platform": {
                "required": True,
                "category_ids": [],
                "category_count": 0,
                "sent_to_llm_count": 0,
                "missing_category": True,
                "missing_candidates": True,
                "missing": True,
            },
            "cpu": {
                "required": True,
                "category_ids": ["cpu"],
                "category_count": 1,
                "sent_to_llm_count": 1,
                "missing_category": False,
                "missing_candidates": False,
                "missing": False,
            },
        },
        "roles_dropped_reason_by_role": {"server_platform": "missing_category"},
        "cpu_candidates": [cpu_row],
        "broad_count_by_role": {"cpu": 1},
    }

    package = build_llm_configurator_package(
        user_request="Need server lifecycle category invariant.",
        normalized_requirements=[
            {
                "product_group": "server",
                "required_roles": ["server_platform", "cpu"],
                "required_capabilities": [],
            }
        ],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        max_package_chars=1_000_000,
    )

    assert package["package_candidate_exposure_incomplete"] is True
    assert package["package_exposure_blocking_lifecycle_roles"] == [
        "server_platform"
    ]
    assert "server_platform" in package["package_candidate_exposure_incomplete_roles"]
    assert package["package_skipped_reason"] == "category_plan_missing_required_roles"
    assert package["category_planner_unresolved_required_roles"] == [
        "server_platform"
    ]
    assert package["roles_dropped_reason_by_role"]["server_platform"] == (
        "missing_category"
    )


def test_repaired_required_category_materializes_platform_candidates_in_package() -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=1)
    matrix["stage_a_broad_roles"] = ["server_platform", "cpu", "ram", "storage"]
    matrix["effective_matrix_roles_before_category_planner"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
    ]
    matrix["category_planner_input_roles"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
    ]
    matrix["category_planner_output_roles"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
    ]
    matrix["validated_category_plan_roles"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
    ]
    matrix["materialized_matrix_roles"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
    ]
    matrix["category_planner_missing_required_roles"] = ["server_platform"]
    matrix["category_planner_repair_attempted"] = True
    matrix["category_planner_repair_success"] = True
    matrix["category_planner_repaired_roles"] = ["server_platform"]
    matrix["category_planner_unresolved_required_roles"] = []

    package = build_llm_configurator_package(
        user_request="Need repaired server platform category.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        max_package_chars=1_000_000,
    )

    assert package["composer_package_candidate_count_by_role"]["server_platform"] == 1
    assert package["package_candidate_exposure_incomplete"] is False
    assert "server_platform" in package["composer_package_roles"]
    assert package.get("package_skipped_reason") is None


def test_composer_not_attempted_for_missing_required_category_plan_role() -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=1)
    matrix["category_plan"].pop("server_platform")
    matrix["stage_a_broad_roles"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
    ]
    matrix["effective_matrix_roles_before_category_planner"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
    ]
    matrix["category_planner_input_roles"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
    ]
    matrix["category_planner_output_roles"] = ["cpu", "ram", "storage"]
    matrix["validated_category_plan_roles"] = ["cpu", "ram", "storage"]
    matrix["materialized_matrix_roles"] = ["cpu", "ram", "storage"]
    matrix["platform_candidates"] = []
    matrix.setdefault("broad_count_by_role", {}).pop("server_platform", None)
    matrix["category_planner_missing_required_roles"] = ["server_platform"]
    matrix["category_planner_repair_attempted"] = True
    matrix["category_planner_repair_success"] = False
    matrix["category_planner_repair_reason"] = "no_category_found:server_platform"
    matrix["category_planner_unresolved_required_roles"] = ["server_platform"]
    matrix["roles_dropped_reason_by_role"] = {"server_platform": "missing_category"}
    matrix["role_coverage_summary"] = {
        "cpu": {
            "required": True,
            "category_ids": ["V110103"],
            "category_count": 1,
            "sent_to_llm_count": 1,
            "missing_category": False,
            "missing_candidates": False,
            "missing": False,
        },
        "ram": {
            "required": True,
            "category_ids": ["V110104"],
            "category_count": 1,
            "sent_to_llm_count": 1,
            "missing_category": False,
            "missing_candidates": False,
            "missing": False,
        },
        "storage": {
            "required": True,
            "category_ids": ["V110106"],
            "category_count": 1,
            "sent_to_llm_count": 1,
            "missing_category": False,
            "missing_candidates": False,
            "missing": False,
        },
    }
    matrix["role_coverage_summary"]["server_platform"] = {
        "required": True,
        "category_ids": [],
        "category_count": 0,
        "sent_to_llm_count": 0,
        "missing_category": True,
        "missing_candidates": True,
        "missing": True,
    }
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server with missing platform category.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={"llm_configurator_max_package_chars": 1_000_000}
        ),
    )

    assert client.package == {}
    assert outcome.final_status_source == "category_plan_incomplete"
    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.fallback_reason != "composer_no_recommendation"
    assert "category_plan_missing_required_roles" in outcome.composer_attempt_decision[
        "blocked_by"
    ]
    assert outcome.package_diagnostics["package_candidate_exposure_incomplete"] is True
    assert outcome.package_diagnostics["package_exposure_blocking_lifecycle_roles"] == [
        "server_platform"
    ]


def test_high_quality_full_broad_matrix_semantic_diagnostics_do_not_block_composer() -> None:
    request_text = (
        "Need server #78 with CPU, RAM, storage, LSI HBA, 10GbE SFP+, "
        "C13-C14 cables and 2x2000W PSU."
    )
    matrix = _server_78_like_broad_matrix(rows_per_role=1)
    semantic_diagnostics = {
        "requirement_classifier_status": "partial",
        "requirement_source_coverage_percent": 38.71,
        "unclassified_source_fragments": ["C13-C14 cables", "8 fans N+1"],
        "requirement_classifier_repair_quality": "incomplete_source_coverage",
        "requirement_classifier_repair_accepted": False,
    }
    matrix.update(semantic_diagnostics)
    matrix["role_plan"].update(semantic_diagnostics)
    for rows in matrix.values():
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    row["available_quantity"] = 100
    matrix["missing_required_roles_before_llm"] = ["cable"]
    matrix["missing_required_capabilities_before_llm"] = [
        {
            "capability_id": "cable.classifier_gap",
            "role": "cable",
            "hard": True,
            "source_text": "C13-C14 cables",
            "classification": "accessory_or_consumable",
            "fulfillment_mode": "separate_component_required",
        }
    ]
    client = _FakeComposerClient(_server_78_requirement_analysis_response)

    outcome = compose_llm_configurations(
        user_request=request_text,
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={
                "llm_component_candidates_per_role": 42,
                "llm_configurator_max_package_chars": 1_000_000,
            }
        ),
    )

    package = client.package
    decision = outcome.composer_attempt_decision
    blocked_by = " ".join(decision["blocked_by"])

    assert client.calls == 1
    assert outcome.primary_recommendation_status == "valid"
    assert outcome.final_status_source == "composer_validated"
    assert decision["should_attempt"] is True
    assert decision["blocked_by"] == []
    assert "requirement_classifier_status" not in blocked_by
    assert "source_coverage" not in blocked_by
    assert "unclassified_source_fragments" not in blocked_by
    assert package["original_request_text"] == request_text
    assert package["pre_composer_requirement_classifier_status"] == "partial"
    assert package["pre_composer_requirement_source_coverage_percent"] == 38.71
    assert package["pre_composer_unclassified_source_fragments"] == [
        "C13-C14 cables",
        "8 fans N+1",
    ]
    assert package["pre_composer_semantic_diagnostics_are_blocking"] is False
    assert package["composer_package_candidate_count_by_role"] == (
        package["broad_matrix_count_by_role"]
    )
    assert all(
        count == 0 for count in package["dropped_before_composer_count_by_role"].values()
    )
    assert package["package_candidate_exposure_incomplete"] is False
    assert package["component_candidate_matrix"]["cable"][0]["component_candidate_id"] == (
        "cable-0"
    )
    assert outcome.package_diagnostics["composer_requirement_analysis"][
        "fulfillment_decisions"
    ][0]["fulfillment_mode"] == "separate_component_required"
    assert outcome.package_diagnostics["composer_source_coverage_summary"][
        "coverage_note"
    ] == "composer reconstructed requirements from original request"
    assert "main semantic reasoning stage" in client.system_prompts[0]
    assert "requirement_analysis" in client.system_prompts[0]
    assert "fulfillment_decisions" in client.system_prompts[0]


def test_llm_configurator_returns_fallback_when_package_stays_over_budget() -> None:
    client = _FakeComposerClient(_component_matrix_llm_response)
    settings = _llm_composer_settings(
        output_mode="single_best_cost_valid"
    ).model_copy(update={"llm_configurator_max_package_chars": 10000})

    outcome = compose_llm_configurations(
        user_request="x" * 20000,
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix={
            "product_group": "server",
            "cpu_candidates": [
                _composer_component_candidate(
                    "cpu-single",
                    "CPUVendor",
                    "CPU-SINGLE",
                    "Server CPU",
                    4,
                    Decimal("500"),
                    {"cpu_cores": 32},
                )
            ],
        },
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    assert outcome.fallback_reason == "llm_configurator_not_attempted:package_over_budget"
    assert client.package == {}


def test_product_group_unknown_fails_closed_before_composer_even_with_candidates() -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=1)
    matrix["product_group"] = "unknown"
    matrix["role_plan"]["product_group"] = "unknown"
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need technical stock configuration but product group is unknown.",
        normalized_requirements=[{"product_group": "unknown"}],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={"llm_configurator_max_package_chars": 1_000_000}
        ),
    )

    assert client.calls == 0
    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.final_status_source == "planner_unavailable"
    assert outcome.composer_attempt_decision["should_attempt"] is False
    assert "product_group_unknown" in outcome.composer_attempt_decision["blocked_by"]


def test_llm_configurator_does_not_call_composer_when_distiller_failed() -> None:
    client = _FakeComposerClient(_component_matrix_llm_response)
    settings = _llm_composer_settings(output_mode="single_best_cost_valid")

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix={
            "product_group": "server",
            "semantic_planner_source": "llm",
            "category_plan": {"cpu": ["V110103"]},
            "package_skipped_reason": "matrix_distiller_failed",
            "matrix_distiller_used": False,
            "matrix_distiller_source": "error",
            "matrix_distiller_diagnostics": {
                "broad_count_by_role": {"cpu": 120},
                "reason": "llm_unavailable",
            },
        },
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.fallback_reason == (
        "llm_configurator_not_attempted:package_skipped:matrix_distiller_failed"
    )
    assert "сузить серверную матрицу" in outcome.no_recommendation_reason["summary"]
    assert client.package == {}


def test_matrix_distiller_failure_blocks_compact_fallback_without_top_n() -> None:
    settings = _llm_composer_settings().model_copy(
        update={"llm_configurator_max_package_chars": 20000}
    )
    matrix = _distiller_fallback_source_matrix()
    original_package = build_llm_configurator_package(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        max_package_chars=settings.llm_configurator_max_package_chars,
    )
    fallback_matrix = match_engine_module._matrix_distiller_compact_fallback_package(
        matrix=matrix,
        spec=_server_spec(quantity=2, ram_min_gb=512),
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        rule_based_build_candidates=[],
        settings=settings,
        diagnostics={
            "matrix_distiller_source": "error",
            "error_type": LlmServerError.__name__,
            "ocs_content": {
                "enabled": True,
                "available": False,
                "skipped_reason": "content_forbidden",
                "error_type": "OcsForbiddenError",
                "http_status": 403,
            },
        },
    )
    blocked_package = build_llm_configurator_package(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=fallback_matrix,
        rule_based_build_candidates=[],
        max_package_chars=settings.llm_configurator_max_package_chars,
    )
    client = _FakeComposerClient(_component_matrix_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=fallback_matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    assert original_package["package_skipped_reason"] == "package_over_budget_before_composer"
    assert fallback_matrix["matrix_distiller_source"] == "error"
    assert fallback_matrix["matrix_distiller_used"] is False
    assert fallback_matrix["package_skipped_reason"] == "incomplete_matrix_exposure"
    assert fallback_matrix["llm_fallback_reason"] == "incomplete_matrix_exposure"
    assert fallback_matrix["matrix_distiller_diagnostics"][
        "original_distiller_error_type"
    ] == LlmServerError.__name__
    assert fallback_matrix["matrix_distiller_diagnostics"]["ocs_content"][
        "skipped_reason"
    ] == "content_forbidden"
    assert fallback_matrix["matrix_distiller_diagnostics"][
        "fallback_compaction_attempted"
    ] is False
    assert (
        fallback_matrix["matrix_distiller_diagnostics"][
            "fallback_compaction_disabled_reason"
        ]
        == "no_silent_top_n"
    )
    assert (
        fallback_matrix["matrix_distiller_diagnostics"]["fallback_decision"]
        == "block_incomplete_matrix_exposure"
    )
    assert "cpu_candidates" not in fallback_matrix
    assert blocked_package["package_budget"]["over_budget"] is False
    assert blocked_package["package_budget"]["final_chars"] == len(
        json.dumps(blocked_package, ensure_ascii=False, sort_keys=True, default=str)
    )
    assert blocked_package["package_candidate_exposure_incomplete"] is True
    assert blocked_package["package_skipped_reason"] == "incomplete_matrix_exposure"
    assert blocked_package["llm_fallback_reason"] == "incomplete_matrix_exposure"
    assert blocked_package["broad_matrix_count_by_role"] == {
        "server_platform": 1,
        "cpu": 1,
        "ram": 1,
        "ssd": 1,
    }
    assert blocked_package["composer_package_candidate_count_by_role"] == {
        "server_platform": 0,
        "cpu": 0,
        "ram": 0,
        "ssd": 0,
    }
    assert blocked_package["dropped_before_composer_count_by_role"] == {
        "server_platform": 1,
        "cpu": 1,
        "ram": 1,
        "ssd": 1,
    }
    assert all(
        reason == "package_skipped:incomplete_matrix_exposure"
        for reason in blocked_package["dropped_before_composer_reason_by_role"].values()
    )
    assert client.package == {}
    assert outcome.fallback_reason == "incomplete_matrix_exposure"


def test_server_78_like_fallback_limit_one_under_budget_has_no_stale_block() -> None:
    settings = _llm_composer_settings().model_copy(
        update={"llm_configurator_max_package_chars": 120000}
    )
    normalized_requirements = [
        {
            **_composer_requirements(),
            "source_context": "x" * 91800,
        }
    ]
    matrix = _server_78_like_broad_matrix(rows_per_role=8)

    fallback_matrix = match_engine_module._matrix_distiller_compact_fallback_package(
        matrix=matrix,
        spec=_server_spec(quantity=2, ram_min_gb=256),
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        rule_based_build_candidates=[],
        settings=settings,
        diagnostics={
            "matrix_distiller_source": "error",
            "error_type": "MatrixDistillerError",
            "ocs_content": {
                "enabled": True,
                "available": False,
                "skipped_reason": "content_forbidden",
                "error_type": "OcsForbiddenError",
                "http_status": 403,
            },
        },
    )
    package = build_llm_configurator_package(
        user_request="Need server #78.",
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        component_candidate_matrix=fallback_matrix,
        rule_based_build_candidates=[],
        max_package_chars=settings.llm_configurator_max_package_chars,
    )
    client = _FakeComposerClient(_component_matrix_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        component_candidate_matrix=fallback_matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    assert fallback_matrix["broad_count_by_role"]["cpu"] == 8
    assert fallback_matrix["matrix_distiller_source"] == "error"
    assert fallback_matrix["matrix_distiller_diagnostics"][
        "fallback_compaction_attempted"
    ] is False
    assert (
        fallback_matrix["matrix_distiller_diagnostics"][
            "fallback_compaction_disabled_reason"
        ]
        == "no_silent_top_n"
    )
    assert package["package_budget"]["final_chars"] == len(
        json.dumps(package, ensure_ascii=False, sort_keys=True, default=str)
    )
    assert package["package_candidate_exposure_incomplete"] is True
    assert package["package_skipped_reason"] == "incomplete_matrix_exposure"
    assert package["dropped_before_composer_count_by_role"]["cpu"] == 8
    assert client.package == {}
    assert outcome.fallback_reason == "incomplete_matrix_exposure"


def test_distiller_skipped_under_budget_still_allows_online_composer() -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=1)
    for rows in matrix.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                row["available_quantity"] = 100
    matrix["matrix_distiller_used"] = False
    matrix["matrix_distiller_source"] = "skipped"
    matrix["matrix_distiller_diagnostics"] = {
        "reason": "package_within_budget_or_not_distillable",
        "broad_count_by_role": {
            "server_platform": 1,
            "cpu": 1,
            "ram": 1,
            "ssd": 1,
            "storage_controller": 1,
            "network_adapter": 1,
            "power_supply": 1,
            "cable": 1,
        },
        "package_budget": {"max_chars": 200000, "over_budget": False},
        "package_skipped_reason": None,
    }
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={
                "llm_component_candidates_per_role": 42,
                "llm_configurator_max_package_chars": 1000000,
            }
        ),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    assert client.package["matrix_distiller_used"] is False
    assert client.package["matrix_distiller_source"] == "skipped"
    assert client.package["package_budget"]["over_budget"] is False
    assert client.package.get("package_skipped_reason") is None
    assert client.calls == 1
    assert {
        "server_platform",
        "cpu",
        "ram",
        "ssd",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }.issubset(set(client.package["composer_package_roles"]))
    assert client.package["component_candidate_matrix"]["cpu"]
    assert "distilled_count_by_role" not in client.package or not client.package[
        "distilled_count_by_role"
    ]
    assert outcome.used is True
    assert outcome.evidence_pack["diagnostics"]["online_composer_used"] is True


def test_broad_package_under_budget_skips_full_matrix_by_default() -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=1)
    result = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=[_composer_requirements()],
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=_llm_composer_settings().model_copy(
                update={"llm_configurator_max_package_chars": 1000000}
            ),
        )
    )

    diagnostics = result["matrix_distiller_diagnostics"]

    assert result["full_matrix_evaluation_used"] is False
    assert (
        result["full_matrix_evaluation_fallback_reason"]
        == "skipped_full_broad_package_under_high_quality_limit"
    )
    assert diagnostics["reason"] == "broad_package_under_high_quality_limit"
    assert diagnostics["package_budget"]["over_budget"] is False
    assert result.get("package_skipped_reason") is None


def test_full_broad_package_under_budget_skips_full_matrix_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=3)

    async def fake_enrich_matrix_with_ocs_content(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("full-matrix enrichment must not run under budget")

    def fail_build_matrix_distiller_client(_settings: LlmSettings) -> Any:
        raise AssertionError("full-matrix evaluator must not be constructed")

    monkeypatch.setattr(
        match_engine_module,
        "enrich_matrix_with_ocs_content",
        fake_enrich_matrix_with_ocs_content,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_build_matrix_distiller_llm_client",
        fail_build_matrix_distiller_client,
    )

    result = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=[_composer_requirements()],
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=_llm_composer_settings().model_copy(
                update={"llm_configurator_max_package_chars": 100000}
            ),
        )
    )
    captured = capsys.readouterr()
    diagnostics = result["matrix_distiller_diagnostics"]

    assert result["full_matrix_evaluation_used"] is False
    assert (
        result["full_matrix_evaluation_fallback_reason"]
        == "skipped_full_broad_package_under_high_quality_limit"
    )
    assert diagnostics["reason"] == "broad_package_under_high_quality_limit"
    assert diagnostics["package_budget"]["over_budget"] is False
    assert diagnostics["package_budget"]["trimmed"] is False
    assert result.get("package_skipped_reason") is None
    assert "full_matrix_start" not in captured.err


def test_78_like_package_over_old_threshold_uses_full_broad_package(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=16)
    normalized_requirements = [
        {
            **_composer_requirements(),
            "source_context": "x" * 850_000,
        }
    ]

    async def fake_enrich_matrix_with_ocs_content(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("full-matrix enrichment must not run below 1.5M")

    def fail_build_matrix_distiller_client(_settings: LlmSettings) -> Any:
        raise AssertionError("full-matrix evaluator must not be constructed")

    monkeypatch.setattr(
        match_engine_module,
        "enrich_matrix_with_ocs_content",
        fake_enrich_matrix_with_ocs_content,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_build_matrix_distiller_llm_client",
        fail_build_matrix_distiller_client,
    )

    settings = _llm_composer_settings().model_copy(
        update={"llm_configurator_max_package_chars": 1_500_000}
    )
    result = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=normalized_requirements,
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=settings,
        )
    )
    package = build_llm_configurator_package(
        user_request="Need server #78-like broad matrix.",
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        component_candidate_matrix=result,
        rule_based_build_candidates=[],
        max_package_chars=settings.llm_configurator_max_package_chars,
    )
    captured = capsys.readouterr()
    expected_total = sum(package["broad_matrix_count_by_role"].values())

    assert package["package_budget"]["over_budget"] is False
    assert package["package_budget"]["final_chars"] > 200_000
    assert package["package_budget"]["final_chars"] <= 1_500_000
    assert result["full_matrix_evaluation_used"] is False
    assert result["full_matrix_evaluation_fallback_reason"] == (
        "skipped_full_broad_package_under_high_quality_limit"
    )
    assert package["composer_package_candidate_total"] == expected_total
    assert package["composer_package_candidate_count_by_role"] == (
        package["broad_matrix_count_by_role"]
    )
    assert package["package_candidate_exposure_incomplete"] is False
    assert package["package_candidate_exposure_policy"]["mode"] == "full_broad_matrix"
    assert all(
        ratio == 1.0
        for ratio in package["package_candidate_exposure_ratio_by_role"].values()
    )
    assert "full_matrix_start" not in captured.err


def test_provider_context_limit_falls_back_to_full_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=2)
    normalized_requirements = [_composer_requirements()]
    spec = _server_spec(quantity=2, ram_min_gb=256)

    class ContextLimitThenSuccessClient:
        def __init__(self) -> None:
            self.calls = 0
            self.packages: list[dict[str, Any]] = []

        def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
            assert "Do not invent products" in system_prompt
            self.calls += 1
            package = json.loads(user_prompt)
            self.packages.append(package)
            if self.calls == 1:
                raise LlmClientError(
                    "Provider rejected request: context length exceeds maximum context 131072",
                    status_code=400,
                )
            return _server_78_full_llm_response(package)

    async def fake_prepare_match_planning_context(**_kwargs: Any) -> Any:
        return match_engine_module.PlannedMatchContext(
            spec=spec,
            product_group="server",
            products=[],
            stock_rows_by_key={},
            role_plan=matrix["role_plan"],
            category_plan_result=match_engine_module.CategoryPlanResult(
                category_plan=matrix["category_plan"],
                category_plan_entries=[],
                category_catalog_summary={},
                category_planner_source="ai_category_planner",
                category_plan_source="llm",
            ),
            build_candidates=[],
            build_missing=[],
            build_risks=[],
            component_candidate_matrix=matrix,
            normalized_requirements=normalized_requirements,
        )

    async def fake_distill_component_matrix_if_needed(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["full_matrix_trigger_reason"] == (
            "provider_context_limit_fallback_to_full_matrix"
        )
        assert kwargs["allow_broad_under_budget_fallback"] is False
        result = dict(kwargs["component_candidate_matrix"])
        counts = match_engine_module._server_broad_count_by_role(result)
        result["full_matrix_evaluation_used"] = True
        result["matrix_distiller_used"] = True
        result["matrix_distiller_source"] = "llm"
        result["broad_count_by_role"] = counts
        result["evaluated_candidate_count_by_role"] = counts
        result["selected_candidate_count_by_role"] = counts
        return result

    client = ContextLimitThenSuccessClient()
    monkeypatch.setattr(
        match_engine_module,
        "prepare_match_planning_context",
        fake_prepare_match_planning_context,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_distill_component_matrix_if_needed",
        fake_distill_component_matrix_if_needed,
    )

    from app.matching.ai_match_orchestrator import (
        AiMatchOrchestratorRequest,
        run_ai_match_orchestrator,
    )

    orchestrated = asyncio.run(
        run_ai_match_orchestrator(
            AiMatchOrchestratorRequest(spec=spec),
            object(),  # type: ignore[arg-type]
            llm_configurator_client=client,
            llm_settings=_llm_composer_settings(
                output_mode="single_best_cost_valid"
            ).model_copy(
                update={"llm_configurator_max_package_chars": 1_500_000}
            ),
            web_evidence_settings=WebEvidenceSettings(),
        )
    )
    result = orchestrated.match_result
    report = orchestrated.report_json

    assert client.calls == 2
    assert result.llm_fallback_reason != "provider_context_limit_fallback_to_full_matrix"
    assert report["provider_error_type"] == "context_limit"
    assert report["provider_context_limit"]["context_limit"] == 131072
    assert report["full_matrix_evaluation_used"] is True
    assert report["package_strategy_decision"]["reason"] == (
        "provider_context_limit_fallback_to_full_matrix"
    )
    assert client.packages[1]["provider_error_type"] == "context_limit"


def test_package_strategy_planner_unavailable_is_not_over_budget_failure() -> None:
    from app.matching.ai_match_orchestrator import (
        AiMatchOrchestratorRequest,
        _package_strategy_decision,
    )

    decision = _package_strategy_decision(
        {
            "package_budget": {"over_budget": False},
            "package_skipped_reason": "complex_request_requires_llm_semantic_planner",
            "package_candidate_exposure_incomplete": True,
        },
        AiMatchOrchestratorRequest(),
        llm_settings=LlmSettings(),
    )

    assert decision["decision"] == "planner_unavailable"
    assert decision["reason"] == "complex_request_requires_llm_semantic_planner"
    assert decision["package_over_budget"] is False

    timeout_decision = _package_strategy_decision(
        {
            "product_group": "unknown",
            "semantic_planner_source": "fallback_after_llm_timeout",
            "semantic_planner_fallback_reason": "semantic_planner_timeout",
            "package_budget": {"over_budget": True},
            "package_skipped_reason": "semantic_planner_timeout",
        },
        AiMatchOrchestratorRequest(),
        llm_settings=LlmSettings(),
    )

    assert timeout_decision["decision"] == "planner_unavailable"
    assert timeout_decision["reason"] == "semantic_planner_timeout"
    assert timeout_decision["package_over_budget"] is False


def test_force_full_matrix_attempts_distiller_even_under_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=1)

    class DistillerClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, _system_prompt: str, user_prompt: str) -> dict[str, Any]:
            self.calls += 1
            package = json.loads(user_prompt)
            if "evaluated_candidates" in package:
                return {
                    "role": package["role"],
                    "selected_candidate_ids": [
                        row["component_candidate_id"]
                        for row in package["evaluated_candidates"]
                    ],
                    "role_summary": "forced full-matrix reduction",
                    "no_viable_reason": None,
                    "rejected_summary": [],
                }
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
                        "confidence": "medium",
                    }
                    for candidate in package["candidates"]
                ],
            }

    distiller_client = DistillerClient()

    async def fake_enrich_matrix_with_ocs_content(**_kwargs: Any) -> dict[str, Any]:
        return {"enabled": False, "skipped_reason": "test"}

    monkeypatch.setattr(
        match_engine_module,
        "enrich_matrix_with_ocs_content",
        fake_enrich_matrix_with_ocs_content,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_build_matrix_distiller_llm_client",
        lambda _settings: match_engine_module._PlannerClientState(client=distiller_client),
    )

    result = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=[_composer_requirements()],
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=_llm_composer_settings().model_copy(
                update={
                    "llm_configurator_max_package_chars": 1000000,
                    "llm_full_matrix_force": True,
                }
            ),
        )
    )

    assert result["full_matrix_evaluation_used"] is True
    assert result["matrix_distiller_used"] is True
    assert distiller_client.calls > 0


def test_over_budget_package_runs_full_matrix_chunking_instead_of_compact_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=12)
    broad_package = build_llm_configurator_package(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        max_package_chars=5000,
    )

    assert broad_package["package_budget"]["over_budget"] is True
    assert broad_package["composer_package_candidate_total"] == 96
    assert broad_package["package_candidate_exposure_policy"][
        "candidate_matrix_trimming_allowed"
    ] is False
    assert all(
        count == 0
        for count in broad_package["dropped_before_composer_count_by_role"].values()
    )

    class DistillerClient:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, _system_prompt: str, user_prompt: str) -> dict[str, Any]:
            self.calls += 1
            package = json.loads(user_prompt)
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
                        "confidence": "medium",
                    }
                    for candidate in package["candidates"]
                ],
            }

    distiller_client = DistillerClient()

    async def fake_enrich_matrix_with_ocs_content(**_kwargs: Any) -> dict[str, Any]:
        return {"enabled": False, "skipped_reason": "test"}

    monkeypatch.setattr(
        match_engine_module,
        "enrich_matrix_with_ocs_content",
        fake_enrich_matrix_with_ocs_content,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_build_matrix_distiller_llm_client",
        lambda _settings: match_engine_module._PlannerClientState(client=distiller_client),
    )

    distilled = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=[_composer_requirements()],
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=_llm_composer_settings().model_copy(
                update={"llm_configurator_max_package_chars": 5000}
            ),
        )
    )

    assert distilled["full_matrix_evaluation_used"] is True
    assert distilled["evaluated_candidate_count_by_role"]["network_adapter"] == 12
    assert distilled["broad_count_by_role"]["network_adapter"] == 12
    assert distilled["selected_candidate_count_by_role"]["network_adapter"] == 12
    assert distilled["llm_cost_diagnostics"]["llm_calls_count"] > 0
    assert not distilled["matrix_distiller_diagnostics"].get("fallback_compaction_attempted")


def test_full_matrix_failure_under_budget_attempts_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=8)
    for rows in matrix.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                row["available_quantity"] = 100
    normalized_requirements = [
        {
            **_composer_requirements(),
            "source_context": "x" * 120000,
        }
    ]
    settings = _llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
        update={
            "llm_configurator_max_package_chars": 200000,
            "llm_full_matrix_force": True,
        }
    )

    class FailingDistillerClient:
        def generate_json(self, _system_prompt: str, user_prompt: str) -> dict[str, Any]:
            package = json.loads(user_prompt)
            return {"role": package["role"], "evaluated_candidates_missing": []}

    async def fake_enrich_matrix_with_ocs_content(**_kwargs: Any) -> dict[str, Any]:
        return {
            "enabled": True,
            "available": False,
            "skipped_reason": "content_forbidden",
            "error_type": "OcsForbiddenError",
            "http_status": 403,
        }

    monkeypatch.setattr(
        match_engine_module,
        "enrich_matrix_with_ocs_content",
        fake_enrich_matrix_with_ocs_content,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_build_matrix_distiller_llm_client",
        lambda _settings: match_engine_module._PlannerClientState(
            client=FailingDistillerClient()
        ),
    )

    fallback_matrix = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=normalized_requirements,
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=settings,
        )
    )
    package = build_llm_configurator_package(
        user_request="Need server #78.",
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        component_candidate_matrix=fallback_matrix,
        rule_based_build_candidates=[],
        max_package_chars=settings.llm_configurator_max_package_chars,
    )
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        component_candidate_matrix=fallback_matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    diagnostics = fallback_matrix["matrix_distiller_diagnostics"]
    assert fallback_matrix["matrix_distiller_source"] == "error"
    assert fallback_matrix["full_matrix_evaluation_used"] is False
    assert fallback_matrix["full_matrix_evaluation_fallback_reason"] == (
        "incomplete_matrix_exposure"
    )
    assert fallback_matrix["package_skipped_reason"] == "incomplete_matrix_exposure"
    assert fallback_matrix["llm_fallback_reason"] == "incomplete_matrix_exposure"
    assert diagnostics["error_type"] == "MatrixDistillerError"
    assert diagnostics["stage"] == "role_evaluator"
    assert diagnostics["role"] == "server_platform"
    assert diagnostics["chunk_index"] == 0
    assert diagnostics["fallback_decision"] == "block_composer_package_over_budget"
    assert diagnostics["package_budget_at_failure"]["over_budget"] is True
    assert diagnostics["incomplete_matrix_exposure_reason"] == (
        "package_over_budget_after_full_matrix_failure"
    )
    assert diagnostics["ocs_content"]["skipped_reason"] == "content_forbidden"
    assert package["package_candidate_exposure_incomplete"] is True
    assert package["package_skipped_reason"] == "incomplete_matrix_exposure"
    assert client.package == {}
    assert outcome.fallback_reason == "incomplete_matrix_exposure"


def test_full_matrix_failure_over_budget_blocks_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=20)
    normalized_requirements = [
        {
            **_composer_requirements(),
            "source_context": "x" * 91800,
        }
    ]
    settings = _llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
        update={"llm_configurator_max_package_chars": 120000}
    )

    class FailingDistillerClient:
        def generate_json(self, _system_prompt: str, user_prompt: str) -> dict[str, Any]:
            package = json.loads(user_prompt)
            return {"role": package["role"], "evaluated_candidates_missing": []}

    async def fake_enrich_matrix_with_ocs_content(**_kwargs: Any) -> dict[str, Any]:
        return {"enabled": False, "skipped_reason": "test"}

    monkeypatch.setattr(
        match_engine_module,
        "enrich_matrix_with_ocs_content",
        fake_enrich_matrix_with_ocs_content,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_build_matrix_distiller_llm_client",
        lambda _settings: match_engine_module._PlannerClientState(
            client=FailingDistillerClient()
        ),
    )

    blocked_matrix = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=normalized_requirements,
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=settings,
        )
    )
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        component_candidate_matrix=blocked_matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    diagnostics = blocked_matrix["matrix_distiller_diagnostics"]
    assert blocked_matrix["full_matrix_evaluation_used"] is False
    assert blocked_matrix["package_skipped_reason"] == "incomplete_matrix_exposure"
    assert blocked_matrix["llm_fallback_reason"] == "incomplete_matrix_exposure"
    assert diagnostics["error_type"] == "MatrixDistillerError"
    assert diagnostics["stage"] == "role_evaluator"
    assert diagnostics["role"] == "server_platform"
    assert diagnostics["chunk_index"] == 0
    assert (
        diagnostics["incomplete_matrix_exposure_reason"]
        == "package_over_budget_after_full_matrix_failure"
    )
    assert diagnostics["fallback_decision"] == "block_composer_package_over_budget"
    assert diagnostics["package_budget_at_failure"]["over_budget"] is True
    assert outcome.fallback_reason == "incomplete_matrix_exposure"
    assert client.package == {}


def test_full_matrix_timeout_under_budget_attempts_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=8)
    normalized_requirements = [
        {
            **_composer_requirements(),
            "source_context": "x" * 120000,
        }
    ]
    settings = _llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
        update={
            "llm_configurator_max_package_chars": 200000,
            "llm_full_matrix_max_seconds": 0.07,
            "llm_full_matrix_chunk_timeout_seconds": 0.02,
            "llm_full_matrix_force": True,
        }
    )

    class HangingDistillerClient:
        def generate_json(self, _system_prompt: str, _user_prompt: str) -> dict[str, Any]:
            time.sleep(1)
            return {}

    async def fake_enrich_matrix_with_ocs_content(**_kwargs: Any) -> dict[str, Any]:
        return {"enabled": False, "skipped_reason": "test"}

    monkeypatch.setattr(
        match_engine_module,
        "enrich_matrix_with_ocs_content",
        fake_enrich_matrix_with_ocs_content,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_build_matrix_distiller_llm_client",
        lambda _settings: match_engine_module._PlannerClientState(
            client=HangingDistillerClient()
        ),
    )

    fallback_matrix = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=normalized_requirements,
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=settings,
        )
    )
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        component_candidate_matrix=fallback_matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    diagnostics = fallback_matrix["matrix_distiller_diagnostics"]
    assert fallback_matrix["full_matrix_evaluation_used"] is False
    assert fallback_matrix["full_matrix_evaluation_fallback_reason"] == (
        "incomplete_matrix_exposure"
    )
    assert fallback_matrix["package_skipped_reason"] == "incomplete_matrix_exposure"
    assert fallback_matrix["llm_fallback_reason"] == "incomplete_matrix_exposure"
    assert diagnostics["error_type"] == "MatrixDistillerTimeoutError"
    assert diagnostics["fallback_decision"] == "block_composer_package_over_budget"
    assert diagnostics["package_budget_at_failure"]["over_budget"] is True
    assert diagnostics["incomplete_matrix_exposure_reason"] == (
        "full_matrix_evaluation_timeout_package_over_budget"
    )
    assert client.package == {}
    assert outcome.fallback_reason == "incomplete_matrix_exposure"


def test_full_matrix_timeout_over_budget_blocks_composer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=20)
    normalized_requirements = [
        {
            **_composer_requirements(),
            "source_context": "x" * 91800,
        }
    ]
    settings = _llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
        update={
            "llm_configurator_max_package_chars": 120000,
            "llm_full_matrix_max_seconds": 0.07,
            "llm_full_matrix_chunk_timeout_seconds": 0.02,
        }
    )

    class HangingDistillerClient:
        def generate_json(self, _system_prompt: str, _user_prompt: str) -> dict[str, Any]:
            time.sleep(1)
            return {}

    async def fake_enrich_matrix_with_ocs_content(**_kwargs: Any) -> dict[str, Any]:
        return {"enabled": False, "skipped_reason": "test"}

    monkeypatch.setattr(
        match_engine_module,
        "enrich_matrix_with_ocs_content",
        fake_enrich_matrix_with_ocs_content,
    )
    monkeypatch.setattr(
        match_engine_module,
        "_build_matrix_distiller_llm_client",
        lambda _settings: match_engine_module._PlannerClientState(
            client=HangingDistillerClient()
        ),
    )

    blocked_matrix = asyncio.run(
        match_engine_module._distill_component_matrix_if_needed(
            session=None,  # type: ignore[arg-type]
            spec=_server_spec(quantity=2, ram_min_gb=256),
            products=[],
            component_candidate_matrix=matrix,
            normalized_requirements=normalized_requirements,
            ready_stock_candidates=[],
            rule_based_build_candidates=[],
            role_plan=matrix["role_plan"],
            llm_settings=settings,
        )
    )
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=normalized_requirements,
        ready_stock_candidates=[],
        component_candidate_matrix=blocked_matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    diagnostics = blocked_matrix["matrix_distiller_diagnostics"]
    assert blocked_matrix["full_matrix_evaluation_used"] is False
    assert blocked_matrix["package_skipped_reason"] == "incomplete_matrix_exposure"
    assert blocked_matrix["llm_fallback_reason"] == "incomplete_matrix_exposure"
    assert diagnostics["error_type"] == "MatrixDistillerTimeoutError"
    assert (
        diagnostics["incomplete_matrix_exposure_reason"]
        == "full_matrix_evaluation_timeout_package_over_budget"
    )
    assert diagnostics["fallback_decision"] == "block_composer_package_over_budget"
    assert diagnostics["package_budget_at_failure"]["over_budget"] is True
    assert outcome.fallback_reason == "incomplete_matrix_exposure"
    assert client.package == {}


def test_server_81_ready_package_records_attempt_decision_and_calls_composer() -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=1)
    client = _FakeComposerClient(_server_78_full_llm_response)

    outcome = compose_llm_configurations(
        user_request="Need server #81 with CPU RAM storage NIC PSU and cables.",
        normalized_requirements=[
            _composer_requirements()
            | {
                "required_roles": [
                    "server_platform",
                    "cpu",
                    "ram",
                    "ssd",
                    "storage_controller",
                    "network_adapter",
                    "power_supply",
                    "cable",
                ]
            }
        ],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={
                "llm_component_candidates_per_role": 42,
                "llm_configurator_max_package_chars": 1000000,
            }
        ),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    assert client.calls == 1
    assert outcome.proposal_count == 1
    assert outcome.evidence_pack["diagnostics"]["online_composer_used"] is True
    decision = outcome.composer_attempt_decision
    assert decision["enabled"] is True
    assert decision["package_present"] is True
    assert decision["package_over_budget"] is False
    assert decision["package_skipped_reason"] is None
    assert decision["candidate_count_total"] >= 8
    assert decision["required_roles"]
    assert decision["output_mode"] == "single_best_cost_valid"
    assert decision["provider_configured"] is True
    assert decision["should_attempt"] is True
    assert decision["blocked_by"] == []


def test_online_empty_composer_response_repairs_to_valid_recommendation() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=lambda _package: {"recommendations": [], "general_notes": []},
        repair_responder=_component_matrix_llm_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={
                "llm_component_candidates_per_role": 42,
                "llm_configurator_max_package_chars": 1000000,
            }
        ),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    diagnostics = outcome.evidence_pack["diagnostics"]

    assert client.primary_packages[0]["package_budget"]["over_budget"] is False
    assert len(client.primary_packages) == 1
    assert len(client.repair_packages) == 1
    assert client.repair_packages[0]["empty_response_repair_attempt"] == 1
    assert "same candidate matrix" in " ".join(client.repair_packages[0]["repair_instructions"])
    assert "no proposal and no structured no_recommendation" in (
        client.repair_system_prompts[0].replace("\n", " ")
    )
    assert outcome.used is True
    assert outcome.primary_recommendation_status == "valid"
    assert outcome.fallback_reason is None
    assert outcome.proposal_count == 1
    assert outcome.valid_proposals_count == 1
    assert diagnostics["online_composer_used"] is True
    assert diagnostics["online_composer_empty_response_repair_attempted"] is True
    assert diagnostics["online_composer_empty_response_repair_success"] is True
    assert diagnostics["structured_no_recommendation_used"] is False


def test_online_empty_composer_response_repairs_to_structured_no_recommendation() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=lambda _package: {"recommendations": [], "general_notes": []},
        repair_responder=_structured_no_recommendation_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_server_78_like_broad_matrix(rows_per_role=1),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={
                "llm_component_candidates_per_role": 42,
                "llm_configurator_max_package_chars": 1000000,
            }
        ),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    diagnostics = outcome.evidence_pack["diagnostics"]
    reason = outcome.no_recommendation_reason

    assert len(client.primary_packages) == 1
    assert len(client.repair_packages) == 1
    assert outcome.used is False
    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.fallback_reason == "composer_structured_no_recommendation"
    assert reason["structured_no_recommendation"] is True
    assert reason["missing_roles"] == ["network_adapter"]
    assert reason["missing_required_capabilities"][0]["role"] == "network_adapter"
    assert reason["hard_mismatches"][0]["component_candidate_id"].startswith(
        "network_adapter-"
    )
    assert reason["role_analysis"][0]["considered_candidate_ids"]
    assert reason["considered_candidate_ids"]["network_adapter"]
    assert "network_adapter" in reason["explanation_ru"]
    assert diagnostics["online_composer_used"] is True
    assert diagnostics["online_composer_empty_response_repair_attempted"] is True
    assert diagnostics["online_composer_empty_response_repair_success"] is True
    assert diagnostics["structured_no_recommendation_used"] is True


def test_structured_no_recommendation_with_one_of_many_candidates_repairs() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_one_candidate_per_role_no_recommendation_response,
        repair_responder=_server_78_full_llm_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need server #85.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_server_85_like_broad_matrix(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={
                "llm_component_candidates_per_role": 42,
                "llm_configurator_max_package_chars": 1000000,
            }
        ),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    diagnostics = outcome.evidence_pack["diagnostics"]

    assert len(client.primary_packages) == 1
    assert len(client.repair_packages) == 1
    assert client.repair_packages[0]["no_recommendation_coverage_repair_attempt"] == 1
    assert (
        "Your no_recommendation considered too few candidates"
        in client.repair_system_prompts[0]
    )
    assert outcome.primary_recommendation_status == "valid"
    assert outcome.fallback_reason is None
    assert diagnostics["no_recommendation_coverage_gate_passed"] is True
    assert diagnostics["no_recommendation_coverage_repair_attempted"] is True
    assert diagnostics["no_recommendation_coverage_repair_success"] is True
    assert diagnostics["no_recommendation_coverage_repair_reason"] == (
        "repair_returned_valid_bom"
    )


def test_no_recommendation_coverage_repair_accepts_sufficient_no_recommendation() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_one_candidate_per_role_no_recommendation_response,
        repair_responder=_covered_no_recommendation_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need server #85.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_server_85_like_broad_matrix(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={
                "llm_component_candidates_per_role": 42,
                "llm_configurator_max_package_chars": 1000000,
            }
        ),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    diagnostics = outcome.evidence_pack["diagnostics"]
    coverage = outcome.no_recommendation_reason["no_recommendation_coverage"]

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.fallback_reason == "composer_structured_no_recommendation"
    assert coverage["coverage_incomplete"] is False
    assert coverage["considered_count_by_role"]["network_adapter"] == 42
    assert diagnostics["no_recommendation_coverage_gate_passed"] is True
    assert diagnostics["no_recommendation_coverage_repair_attempted"] is True
    assert diagnostics["no_recommendation_coverage_repair_success"] is True
    assert diagnostics["structured_no_recommendation_used"] is True


def test_no_recommendation_coverage_repair_still_incomplete_becomes_technical() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_one_candidate_per_role_no_recommendation_response,
        repair_responder=_one_candidate_per_role_no_recommendation_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need server #85.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_server_85_like_broad_matrix(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
            update={
                "llm_component_candidates_per_role": 42,
                "llm_configurator_max_package_chars": 1000000,
            }
        ),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    diagnostics = outcome.evidence_pack["diagnostics"]
    reason = outcome.no_recommendation_reason
    coverage = reason["no_recommendation_coverage"]

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.fallback_reason == "composer_no_safe_complete_bom"
    assert reason["summary"] == "AI не смог надежно оценить полную матрицу кандидатов."
    assert reason["coverage_rejected"] is True
    assert coverage["coverage_incomplete"] is True
    assert coverage["incomplete_roles"] == [
        "server_platform",
        "cpu",
        "ram",
        "ssd",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    ]
    assert coverage["considered_count_by_role"]["network_adapter"] == 1
    assert coverage["matrix_count_by_role"]["network_adapter"] == 42
    assert coverage["next_action"] == "rerun with full-matrix evaluation or larger budget"
    assert diagnostics["no_recommendation_coverage_gate_passed"] is False
    assert diagnostics["no_recommendation_coverage_repair_attempted"] is True
    assert diagnostics["no_recommendation_coverage_repair_success"] is False
    assert diagnostics["no_recommendation_coverage_rejected"] is True
    assert diagnostics["structured_no_recommendation_used"] is False
    assert outcome.package_diagnostics["bom_critic_used"] is True


def test_no_recommendation_can_rely_on_complete_full_matrix_summaries() -> None:
    matrix = _server_78_like_broad_matrix(rows_per_role=5)
    matrix.update(
        {
            "full_matrix_evaluation_used": True,
            "evaluated_candidate_count_by_role": {
                "server_platform": 5,
                "cpu": 5,
                "ram": 5,
                "ssd": 5,
                "storage_controller": 5,
                "network_adapter": 5,
                "power_supply": 5,
                "cable": 5,
            },
            "broad_count_by_role": {
                "server_platform": 5,
                "cpu": 5,
                "ram": 5,
                "ssd": 5,
                "storage_controller": 5,
                "network_adapter": 5,
                "power_supply": 5,
                "cable": 5,
            },
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need server #84-like full matrix coverage.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_one_candidate_per_role_no_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    coverage = outcome.no_recommendation_reason["no_recommendation_coverage"]

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.fallback_reason == "composer_structured_no_recommendation"
    assert coverage["coverage_incomplete"] is False
    assert coverage["full_matrix_evaluation_complete"] is True
    assert coverage["considered_count_by_role"]["network_adapter"] == 5


def test_online_empty_composer_response_empty_twice_reports_after_repair() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=lambda _package: {"recommendations": [], "general_notes": []},
        repair_responder=lambda _package: {"recommendations": [], "general_notes": []},
    )

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    diagnostics = outcome.evidence_pack["diagnostics"]
    reason = outcome.no_recommendation_reason

    assert len(client.repair_packages) == 1
    assert outcome.fallback_reason == "llm_configurator_no_proposals_after_repair"
    assert outcome.proposal_count == 0
    assert outcome.rejected_recommendations_count == 0
    assert "Composer returned no proposal twice" in reason["summary"]
    assert reason["composer_no_proposal_attempts"] == 2
    assert "Composer returned no proposal twice" in " ".join(
        reason["diagnostic_notes"]
    )
    assert diagnostics["online_composer_used"] is True
    assert diagnostics["online_composer_empty_response_repair_attempted"] is True
    assert diagnostics["online_composer_empty_response_repair_success"] is False
    assert diagnostics["structured_no_recommendation_used"] is False
    assert outcome.parse_diagnostics["llm_json_extract_status"] == "parsed"


def test_empty_normal_composer_response_still_marks_attempt() -> None:
    client = _FakeComposerClient(lambda _package: {"recommendations": [], "general_notes": []})

    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert client.calls == 1
    assert outcome.fallback_reason == "llm_configurator_no_proposals"
    assert outcome.proposal_count == 0
    assert outcome.evidence_pack["diagnostics"]["online_composer_used"] is True
    assert outcome.error_type is None


def test_online_composer_provider_failure_marks_attempt_and_error() -> None:
    outcome = compose_llm_configurations(
        user_request="Need server #78.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        llm_client=_RaisingComposerClient("secret-token", status_code=503),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
    )

    diagnostics = outcome.evidence_pack["diagnostics"]

    assert outcome.used is False
    assert outcome.fallback_reason == "llm_configurator_request_failed"
    assert outcome.error_type == "LlmClientError"
    assert outcome.http_status == 503
    assert diagnostics["online_composer_used"] is True
    assert diagnostics["online_composer_error_type"] == "LlmClientError"
    assert diagnostics["online_composer_parse_status"] == "request_failed"
    assert diagnostics["online_composer_empty_response_repair_attempted"] is False
    assert diagnostics["structured_no_recommendation_used"] is False
    assert outcome.fallback_reason != "llm_configurator_no_valid_recommendations"


def test_over_budget_package_is_not_attempted_with_explicit_decision() -> None:
    client = _FakeComposerClient(_component_matrix_llm_response)
    settings = _llm_composer_settings(
        output_mode="single_best_cost_valid"
    ).model_copy(update={"llm_configurator_max_package_chars": 10000})

    outcome = compose_llm_configurations(
        user_request="x" * 20000,
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix={
            "product_group": "server",
            "cpu_candidates": [
                _composer_component_candidate(
                    "cpu-single",
                    "CPUVendor",
                    "CPU-SINGLE",
                    "Server CPU",
                    4,
                    Decimal("500"),
                    {"cpu_cores": 32},
                )
            ],
        },
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    assert client.calls == 0
    assert outcome.fallback_reason == "llm_configurator_not_attempted:package_over_budget"
    assert outcome.composer_attempt_decision["should_attempt"] is False
    assert "package_over_budget" in outcome.composer_attempt_decision["blocked_by"]


def test_llm_normalizes_qwen_role_and_slot_aliases_before_validation() -> None:
    normalized = configuration_composer_module._normalize_proposal_payload(
        {
            "proposal_role": "lower_price_with_tradeoff",
            "recommendation_slot": "lower_price_with_tradeoff",
            "selected_component_candidate_ids": {
                "platform": "platform-amd",
                "storage": "ssd-3840",
            },
        }
    )

    assert normalized["proposal_role"] == "explicit_tradeoff"
    assert normalized["recommendation_slot"] == "alternative"
    assert normalized["component_candidate_ids"]["ssd"] == "ssd-3840"
    assert "storage" not in normalized["component_candidate_ids"]


@pytest.mark.parametrize(
    ("raw_slot", "expected_slot"),
    [
        ("lower_price_with_tradeoff", "alternative"),
        ("alternative_vendor_or_platform", "alternative"),
        ("budget_option", "price_optimal"),
        ("cheapest", "price_optimal"),
        ("technical", "technical_clean"),
        ("partial_fallback", "partial_fallback"),
    ],
)
def test_llm_normalizes_recommendation_slot_aliases(
    raw_slot: str,
    expected_slot: str,
) -> None:
    normalized = configuration_composer_module._normalize_proposal_payload(
        {"recommendation_slot": raw_slot}
    )

    assert normalized["recommendation_slot"] == expected_slot


def test_default_output_mode_is_single_best_cost_valid() -> None:
    assert LlmSettings().llm_configurator_output_mode == "single_best_cost_valid"


def test_single_best_prompt_requests_exactly_one_primary_recommendation() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_primary_recommendation_response,
        repair_responder=_repair_invalid_json_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_cheaper_equivalent_ram(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )
    system_prompt = client.primary_system_prompts[0]
    package = client.primary_packages[0]

    assert "Return exactly ONE primary recommendation" in system_prompt
    assert "Return one primary_recommendation JSON object" in system_prompt
    assert "required_roles contains network_adapter" in system_prompt
    assert package["output_mode"] == "single_best_cost_valid"
    assert package["proposal_pool_limit"] == 1
    assert outcome.used is True
    assert outcome.grouped_presales_mode_used is False
    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["candidate_type"] == "build_from_parts"
    assert outcome.commercial_summary


def test_single_best_accepts_primary_recommendation_object() -> None:
    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_cheaper_equivalent_ram(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.used is True
    assert len(outcome.recommended_builds) == 1
    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["component_candidate_ids"]["ram"] == "ram-micron-32"
    assert outcome.primary_recommendation["engineering_confidence_code"] == (
        "preliminary_requires_engineer_review"
    )
    assert outcome.primary_recommendation["engineering_confidence"] == (
        "предварительно, нужна инженерная проверка"
    )
    assert outcome.primary_recommendation["components"]
    checks_text = json.dumps(
        outcome.primary_recommendation["engineer_checks"],
        ensure_ascii=False,
    )
    assert "CPU support list" in checks_text
    assert "QVL RAM" in checks_text
    assert "NVMe/U.2/U.3" in checks_text
    commercial_summary = outcome.commercial_summary
    assert commercial_summary["mode"] == "single_best_cost_valid"
    assert "Сервер в сборе" in commercial_summary["copy_paste_text"]
    assert "Платформа" in commercial_summary["copy_paste_text"]
    assert "CPU" in commercial_summary["copy_paste_text"]
    assert "RAM" in commercial_summary["copy_paste_text"]
    assert "SSD" in commercial_summary["copy_paste_text"]
    assert "components" not in commercial_summary
    assert "primary_recommendation" not in commercial_summary
    commercial_summary_text = json.dumps(commercial_summary, ensure_ascii=False)
    for forbidden in (
        "component_candidate_id",
        '"facts"',
        '"evidence"',
        "raw JSON",
        "llm_rec",
        "why_this_platform:",
        "quote_recommendation:",
        "Engineering confidence",
        "Tradeoff",
        "Minimal cost",
        "Proven",
        "Premium platform",
        "高密度",
        "preliminary_requires_engineer_review",
    ):
        assert forbidden not in commercial_summary_text
    assert outcome.configuration_groups == []


def test_single_best_no_recommendation_when_no_complete_build_exists() -> None:
    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_cheaper_equivalent_ram(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_missing_cpu_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.used is False
    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.primary_recommendation == {}
    assert outcome.commercial_summary["status"] == "no_recommendation"
    assert outcome.no_recommendation_reason["summary"] == (
        "Безопасную складскую рекомендацию дать нельзя."
    )
    assert "cpu" in outcome.no_recommendation_reason["missing_roles"]
    reason_text = json.dumps(outcome.no_recommendation_reason, ensure_ascii=False)
    assert "CPU support list" in reason_text
    assert "QVL RAM" in reason_text
    assert "NVMe/U.2/U.3" in reason_text


def test_single_best_rejects_missing_required_network_adapter() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers with 2 ports 25GbE SFP28 per server.",
        normalized_requirements=[_network_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_network(
            include_network=True,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert "network_adapter" in outcome.no_recommendation_reason["missing_roles"]
    reason_text = json.dumps(outcome.no_recommendation_reason, ensure_ascii=False)
    assert "network.25gbe.sfp28" in reason_text
    assert "базы данных" not in reason_text
    assert "склад Москва" not in reason_text


def test_single_best_accepts_dual_port_25gbe_sfp28_adapter_quantity() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers with 2 ports 25GbE SFP28 per server.",
        normalized_requirements=[_network_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_network(
            include_network=True,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_with_network_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["product_group"] == "server"
    assert outcome.primary_recommendation["component_candidate_ids"]["platform"]
    assert outcome.primary_recommendation["component_candidate_ids"]["network_adapter"]
    network = _component_by_role(outcome.primary_recommendation["components"], "network_adapter")
    assert network["quantity_required"] == 2
    assert network["per_server_quantity"] == 1
    assert network["network_ports_count"] == 2
    assert "Сеть" in outcome.commercial_summary["copy_paste_text"]
    assert "Сеть:" in outcome.commercial_summary["copy_paste_text"]
    assert "2 x 25GbE SFP28" in outcome.commercial_summary["copy_paste_text"]
    assert "на сервер / 2 шт. всего" in outcome.commercial_summary["copy_paste_text"]


def test_single_best_accepts_quad_port_25gbe_sfp28_adapter_quantity() -> None:
    matrix = _composer_component_matrix_with_network(include_network=True)
    network = matrix["network_adapter_candidates"][0]
    network["name"] = "Quad-port 25GbE SFP28 PCIe network adapter"
    network["network_ports_count"] = 4
    network["extracted_facts"]["network_ports_count"] = 4
    network["extracted_facts"]["ports_count"] = 4

    outcome = compose_llm_configurations(
        user_request="Need 2 servers with 2 ports 25GbE SFP28 per server.",
        normalized_requirements=[_network_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_with_network_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    network_component = _component_by_role(
        outcome.primary_recommendation["components"],
        "network_adapter",
    )
    assert network_component["quantity_required"] == 2
    assert network_component["per_server_quantity"] == 1
    assert network_component["network_ports_count"] == 4
    assert "4 x 25GbE SFP28" in outcome.commercial_summary["copy_paste_text"]


def test_single_best_network_no_recommendation_when_switch_hard_capability_missing() -> None:
    outcome = compose_llm_configurations(
        user_request=(
            "Нужен коммутатор 48 портов 1G PoE+, 4 uplink 10G SFP+, "
            "L3, stacking, один самый дешевый вариант"
        ),
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_network_switch_component_matrix(close_switch=False),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_network_switch_only_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert "switch" in outcome.no_recommendation_reason["missing_roles"]
    assert not any(
        row.get("role") == "switch" and row.get("status") == "missing_component"
        for row in outcome.no_recommendation_reason["missing_required_capabilities"]
    )
    reason_text = json.dumps(outcome.no_recommendation_reason, ensure_ascii=False)
    assert "uplink" in reason_text or "PoE" in reason_text


def test_single_best_network_rejects_text_only_48_port_request_for_24_port_switch() -> None:
    requirements = {
        "product_group": "network",
        "required_roles": ["switch"],
        "required_capabilities": [],
        "role_plan": {
            "product_group": "network",
            "required_roles": ["switch"],
            "required_capabilities": [],
        },
    }
    matrix = _network_switch_component_matrix(close_switch=True)
    switch = matrix["switch_candidates"][0]
    switch["part_number"] = "SW-24P-4SFP"
    switch["name"] = "24x1G RJ45 PoE switch 4x10G SFP+ L3"
    switch["extracted_facts"]["port_count"] = 24

    outcome = compose_llm_configurations(
        user_request=(
            "Нужен access switch 48 x 1G RJ45 PoE, минимум 4 x 10G SFP+ "
            "uplink, L3, один самый дешевый вариант"
        ),
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_network_switch_only_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.valid_proposals_count == 0
    rejected_text = json.dumps(
        outcome.rejected_recommendations_debug_safe,
        ensure_ascii=False,
    )
    assert "request-text fact: port_count" in rejected_text


def test_single_best_network_normalizes_platform_alias_to_switch() -> None:
    outcome = compose_llm_configurations(
        user_request=NETWORK_MATCH_71_TEXT,
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_network_switch_component_matrix(close_switch=True),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_network_switch_platform_alias_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["component_candidate_ids"] == {
        "switch": "switch-48p"
    }
    assert outcome.validation_summary["rejected_role_mismatch"] == 0
    assert outcome.no_recommendation_reason == {}


def test_single_best_network_primary_from_platform_alias_materializes_one_switch() -> None:
    matrix = _network_switch_component_matrix(close_switch=True)
    switch = matrix["switch_candidates"][0]
    switch["candidate_id"] = "switch-good"
    switch["price_value"] = "483.84"
    switch["quantity_required"] = 1
    switch["available_quantity"] = 5

    outcome = compose_llm_configurations(
        user_request=NETWORK_MATCH_71_TEXT,
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_network_switch_platform_alias_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["product_group"] == "network"
    assert outcome.primary_recommendation["total_price_value"] == "483.84"
    assert outcome.primary_recommendation["component_candidate_ids"]["switch"] == "switch-good"
    switch_component = _component_by_role(outcome.primary_recommendation["components"], "switch")
    assert switch_component["quantity_required"] == 1
    assert switch_component["component_candidate_id"] == "switch-good"
    commercial_summary = outcome.commercial_summary
    assert commercial_summary["product_group"] == "network"
    assert commercial_summary["server_line"]
    assert "РЎРµСЂРІРµСЂ РІ СЃР±РѕСЂРµ" not in commercial_summary["copy_paste_text"]


def test_single_best_network_primary_sanitizes_llm_server_engineer_checks() -> None:
    outcome = compose_llm_configurations(
        user_request=NETWORK_MATCH_71_TEXT,
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_network_switch_component_matrix(close_switch=True),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_network_switch_with_server_checks_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    primary_text = json.dumps(outcome.primary_recommendation, ensure_ascii=False)
    commercial_text = json.dumps(outcome.commercial_summary, ensure_ascii=False)
    recommended_text = json.dumps(outcome.recommended_builds, ensure_ascii=False)
    combined = "\n".join([primary_text, commercial_text, recommended_text])

    assert "порт" in combined or "access/uplink" in combined
    assert "PoE" in combined
    for forbidden in (
        "CPU support",
        "QVL RAM",
        "DIMM",
        "NVMe/U.2/U.3",
        "backplane",
        "кулеры",
        "рейки",
    ):
        assert forbidden not in combined


def test_single_best_network_rejects_platform_alias_to_transceiver() -> None:
    outcome = compose_llm_configurations(
        user_request=NETWORK_MATCH_71_TEXT,
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_network_switch_component_matrix(
            close_switch=True,
            include_transceiver=True,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_network_transceiver_platform_alias_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.validation_summary["rejected_role_mismatch"] == 1
    rejected = outcome.rejected_recommendations_debug_safe[0]
    assert rejected["role_mismatches"][0]["prompt_role"] == "platform"
    assert rejected["role_mismatches"][0]["actual_role"] == "transceiver"
    assert "switch" not in rejected["normalized_core_component_candidate_ids"]


def test_single_best_network_valid_quote_with_transceivers_license_support() -> None:
    outcome = compose_llm_configurations(
        user_request=(
            "Нужен коммутатор 48 портов 1G PoE+, 4 uplink 10G SFP+, "
            "L3, stacking, трансиверы в комплект, лицензия/support 1 год"
        ),
        normalized_requirements=[
            _network_switch_composer_requirements(
                include_transceiver=True,
                include_license=True,
                include_support=True,
            )
        ],
        ready_stock_candidates=[],
        component_candidate_matrix=_network_switch_component_matrix(
            close_switch=True,
            include_transceiver=True,
            include_license=True,
            include_support=True,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_network_full_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["product_group"] == "network"
    assert {component["role"] for component in outcome.primary_recommendation["components"]} >= {
        "switch",
        "transceiver",
        "license",
        "support",
    }
    commercial_text = outcome.commercial_summary["copy_paste_text"]
    assert "Предварительная спецификация для КП" in commercial_text
    assert "Сетевое оборудование - 1 шт." in commercial_text
    assert "Состав:" in commercial_text
    assert "Всего к заказу:" in commercial_text
    assert "Проверить перед КП:" in commercial_text
    for forbidden in ("component_candidate_id", "raw JSON", "llm_rec", "{"):
        assert forbidden not in commercial_text


def test_component_matrix_includes_network_roles_from_validated_category_plan(
    db_session: Session,
) -> None:
    _seed_network_product(
        db_session,
        item_id="net-switch-1",
        part_number="SW-48P-4SFP",
        producer="NetVendor",
        category_id="net-switch",
        item_name="48x1G RJ45 PoE+ switch 740W 4 uplink 10G SFP+ L3 stacking",
        quantity=1,
        price=Decimal("1200"),
    )
    _seed_network_product(
        db_session,
        item_id="net-transceiver-1",
        part_number="SFP-10G-SR",
        producer="NetVendor",
        category_id="net-transceiver",
        item_name="10G SFP+ transceiver module",
        quantity=4,
        price=Decimal("80"),
    )
    _seed_network_product(
        db_session,
        item_id="net-dac-1",
        part_number="DAC-10G-1M",
        producer="NetVendor",
        category_id="net-dac",
        item_name="10G SFP+ DAC cable 1m",
        quantity=4,
        price=Decimal("30"),
    )
    _seed_network_product(
        db_session,
        item_id="net-license-1",
        part_number="LIC-1Y",
        producer="NetVendor",
        category_id="net-license",
        item_name="Switch license subscription 1 year",
        quantity=1,
        price=Decimal("100"),
    )
    _seed_network_product(
        db_session,
        item_id="net-support-1",
        part_number="SUP-1Y",
        producer="NetVendor",
        category_id="net-support",
        item_name="Switch support 1 year",
        quantity=1,
        price=Decimal("120"),
    )

    result = asyncio.run(
        match_stock_spec(_network_switch_spec(), _adapter(db_session))
    )
    matrix = result.to_report_json()["component_candidate_matrix"]

    assert matrix["product_group"] == "network"
    assert "net-switch" in matrix["category_plan"]["switch"]
    assert matrix["switch_candidates"]
    assert matrix["transceiver_candidates"]
    assert matrix["dac_cable_candidates"]
    assert matrix["license_candidates"]
    assert matrix["support_candidates"]


def test_network_matrix_filters_ready_servers_and_obvious_switch_non_devices(
    db_session: Session,
) -> None:
    _seed_nerpa_product(
        db_session,
        item_id="ready-server-in-network-db",
        part_number="READY-SERVER",
        item_name="Server 2U 2x CPU 512GB RAM SSD",
        quantity=2,
        can_reserve=True,
    )
    for item_id, name, price in [
        (
            "switch-good",
            "48-port 1G RJ45 PoE+ switch 740W 4 uplink 10G SFP+ L3 stacking",
            "1200",
        ),
        ("switch-tiny", "5-port desktop switch", "25"),
        ("poe-injector", "PoE injector accessory", "18"),
        ("kvm-cable", "KVM cable for rack console", "12"),
        ("mount-kit", "switch wall mount bracket", "8"),
    ]:
        _seed_network_product(
            db_session,
            item_id=item_id,
            part_number=item_id.upper(),
            producer="NetVendor",
            category_id="net-switch" if item_id.startswith("switch") else f"net-{item_id}",
            item_name=name,
            quantity=10,
            price=Decimal(price),
        )

    result = asyncio.run(match_stock_spec(_network_switch_spec(), _adapter(db_session)))
    matrix = result.to_report_json()["component_candidate_matrix"]
    switch_names = [
        str(candidate["name"]).casefold()
        for candidate in matrix["switch_candidates"]
    ]
    active_roles = {
        role
        for role, count in matrix["component_matrix_coverage_summary"][
            "sent_to_llm_by_role"
        ].items()
        if count
    }

    assert matrix["product_group"] == "network"
    assert matrix["ready_server_candidates"] == []
    assert matrix["switch_candidates"][0]["part_number"] == "SWITCH-GOOD"
    assert matrix["switch_candidates"][0]["fit_tier"] == "strong_fit"
    assert not any("5-port" in name for name in switch_names)
    assert not any("injector" in name for name in switch_names)
    assert not any("kvm" in name for name in switch_names)
    assert not any("mount" in name for name in switch_names)
    assert active_roles.isdisjoint({"ready_server", "platform", "cpu", "ram", "ssd", "hdd"})


def test_network_switch_hard_fit_excludes_small_unicode_port_switches(
    db_session: Session,
) -> None:
    for item_id, name, price in [
        (
            "switch-good",
            "Managed L3 Switch 48х1000Base-T PoE, 6x10GBase-X SFP+, PoE Budget 370W",
            "1200",
        ),
        (
            "switch-fallback",
            "Managed Ethernet switch PoE L3 with SFP+ uplinks",
            "900",
        ),
        ("switch-8p", "8х1000Base-T PoE, 2x1000Base-X SFP", "100"),
        ("switch-16p", "16х1000Base-T PoE switch", "180"),
        ("switch-24p", "24х1000Base-T PoE switch", "260"),
        ("switch-100m", "5x100Base-TX unmanaged desktop switch", "40"),
    ]:
        _seed_network_product(
            db_session,
            item_id=item_id,
            part_number=item_id.upper(),
            producer="NetVendor",
            category_id="net-switch",
            item_name=name,
            quantity=10,
            price=Decimal(price),
        )

    result = asyncio.run(match_stock_spec(_network_switch_spec(), _adapter(db_session)))
    switch_candidates = result.to_report_json()["component_candidate_matrix"][
        "switch_candidates"
    ]
    part_numbers = [candidate["part_number"] for candidate in switch_candidates]
    fit_tiers = {
        candidate["part_number"]: candidate["fit_tier"]
        for candidate in switch_candidates
    }

    assert part_numbers[0] == "SWITCH-GOOD"
    assert fit_tiers["SWITCH-GOOD"] == "possible_fit"
    assert fit_tiers.get("SWITCH-FALLBACK") == "fallback_unknown"
    if "SWITCH-FALLBACK" in part_numbers:
        assert part_numbers.index("SWITCH-GOOD") < part_numbers.index("SWITCH-FALLBACK")
    assert "SWITCH-8P" not in part_numbers
    assert "SWITCH-16P" not in part_numbers
    assert "SWITCH-24P" not in part_numbers
    assert "SWITCH-100M" not in part_numbers


def test_match_71_text_path_builds_network_planning_matrix_and_valid_primary(
    db_session: Session,
) -> None:
    _seed_match_71_network_products(db_session)
    spec = extract_stock_spec_for_text_match(NETWORK_MATCH_71_TEXT).spec_json
    client = _FakeComposerClient(_network_switch_only_response)

    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_configurator_client=client,
            llm_settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        )
    )
    report_json = result.to_report_json()
    matrix = report_json["component_candidate_matrix"]
    switch_candidates = matrix["switch_candidates"]
    part_numbers = [candidate["part_number"] for candidate in switch_candidates]

    assert report_json["product_group"] == "network"
    assert any(
        capability["role"] == "switch"
        and "48" in str(capability["capability_id"])
        for capability in report_json["required_capabilities"]
    )
    assert report_json["category_plan"]["switch"] == ["V120100"]
    assert switch_candidates[0]["part_number"] == "SW-48P-4SFP"
    assert {"SW-5P", "SW-8P", "SW-16P", "SW-24P"}.isdisjoint(part_numbers)
    assert matrix["ready_server_candidates"] == []
    assert report_json["shortlist_for_llm"]
    assert client.package["product_group"] == "network"
    assert client.package["component_candidate_matrix"]["switch"]
    assert report_json["primary_recommendation_status"] == "valid"
    assert report_json["primary_recommendation"]["product_group"] == "network"
    assert report_json["llm_configurator_used"] is True
    assert report_json["llm_proposals_count"] == 1
    assert report_json["valid_proposals_count"] == 1
    assert report_json["commercial_summary"]["product_group"] == "network"


def test_match_71_text_path_rejects_unknown_network_selection(
    db_session: Session,
) -> None:
    _seed_match_71_network_products(db_session)
    spec = extract_stock_spec_for_text_match(NETWORK_MATCH_71_TEXT).spec_json

    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(_network_unknown_switch_response),
            llm_settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        )
    )
    report_json = result.to_report_json()
    reason = report_json["no_recommendation_reason"]
    reason_text = json.dumps(reason, ensure_ascii=False)

    assert report_json["primary_recommendation_status"] == "no_recommendation"
    assert report_json["llm_fallback_reason"] == (
        "llm_configurator_all_recommendations_rejected"
    )
    assert reason["product_group"] == "network"
    assert "switch" in reason_text
    assert "CPU support" not in reason_text
    assert "QVL RAM" not in reason_text
    assert "NVMe/U.2/U.3" not in reason_text
    assert "DIMM" not in reason_text


def test_match_71_text_path_pre_llm_missing_switch_skips_composer(
    db_session: Session,
) -> None:
    _seed_match_71_network_products(db_session, include_good_switch=False)
    spec = extract_stock_spec_for_text_match(NETWORK_MATCH_71_TEXT).spec_json
    client = _FakeComposerClient(_network_switch_only_response)

    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_configurator_client=client,
            llm_settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        )
    )
    report_json = result.to_report_json()
    reason_text = json.dumps(report_json["no_recommendation_reason"], ensure_ascii=False)

    assert client.package == {}
    assert report_json["primary_recommendation_status"] == "no_recommendation"
    assert report_json["llm_configurator_used"] is False
    assert report_json["llm_fallback_reason"] in {
        "llm_configurator_not_attempted:missing_required_roles_before_llm",
        "llm_configurator_not_attempted:hard_capability_coverage_missing_before_llm",
        "llm_configurator_not_attempted:no_eligible_candidates_for_required_role",
        "llm_configurator_not_attempted:package_skipped:matrix_empty_after_category_plan",
    }
    assert "switch" in report_json["missing_required_roles"]
    assert report_json["missing_required_capabilities"]
    assert "PoE" in reason_text or "port" in reason_text
    assert "CPU support" not in reason_text


def test_network_no_recommendation_uses_network_manual_checks() -> None:
    outcome = compose_llm_configurations(
        user_request=NETWORK_MATCH_71_TEXT,
        normalized_requirements=[_network_switch_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_network_switch_component_matrix(close_switch=False),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_network_switch_only_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )
    reason_text = json.dumps(outcome.no_recommendation_reason, ensure_ascii=False)

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.no_recommendation_reason["product_group"] == "network"
    assert "порт" in reason_text or "PoE" in reason_text
    assert "CPU support" not in reason_text
    assert "QVL RAM" not in reason_text
    assert "NVMe/U.2/U.3" not in reason_text


def test_storage_no_recommendation_uses_storage_manual_checks() -> None:
    requirements = _storage_composer_requirements()
    matrix = {
        "product_group": "storage",
        "required_roles": requirements["required_roles"],
        "required_capabilities": requirements["required_capabilities"],
        "missing_required_roles": ["storage_system"],
        "missing_required_capabilities": [
            {
                "capability_id": "storage.usable_100tb.fc_32g",
                "role": "storage_system",
                "status": "missing_candidates",
                "source_text": "СХД 100 ТБ usable, FC 32G",
                "reason": "no eligible storage system",
            }
        ],
        "component_matrix_coverage_summary": {"sent_to_llm_by_role": {}},
    }

    outcome = compose_llm_configurations(
        user_request="Нужна СХД 100 ТБ usable FC 32G",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_storage_full_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )
    reason_text = json.dumps(outcome.no_recommendation_reason, ensure_ascii=False)

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.no_recommendation_reason["product_group"] == "storage"
    assert "capacity" in reason_text or "RAID" in reason_text
    assert "CPU support" not in reason_text
    assert "QVL RAM" not in reason_text
    assert "DIMM" not in reason_text
    assert outcome.used is False


@pytest.mark.parametrize(
    ("text", "port_count", "port_speed_gbps", "port_speed"),
    [
        ("8х1000Base-T", 8, 1, "1GbE"),
        ("16х1000Base-T", 16, 1, "1GbE"),
        ("24х1000Base-T", 24, 1, "1GbE"),
        ("48х1000Base-T", 48, 1, "1GbE"),
        ("48×1000Base-T", 48, 1, "1GbE"),
        ("48X1000Base-T", 48, 1, "1GbE"),
        ("48*1000Base-T", 48, 1, "1GbE"),
        ("48x1G", 48, 1, "1GbE"),
        ("48xGigabit", 48, 1, "1GbE"),
        ("48 портов", 48, None, "unknown"),
        ("48 порт", 48, None, "unknown"),
        ("5-Port", 5, None, "unknown"),
    ],
)
def test_network_switch_fact_extraction_parses_generic_port_patterns(
    text: str,
    port_count: int,
    port_speed_gbps: float | None,
    port_speed: str,
) -> None:
    facts = match_engine_module._extract_network_device_facts(text)

    assert facts["port_count"] == port_count
    assert facts["port_speed_gbps"] == port_speed_gbps
    assert facts["port_speed"] == port_speed


def test_network_switch_fact_extraction_parses_fast_ethernet_below_1g() -> None:
    facts = match_engine_module._extract_network_device_facts("5x100Base-TX switch")

    assert facts["port_count"] == 5
    assert facts["port_speed"] == "100MbE"
    assert facts["port_speed_gbps"] == 0.1
    assert facts["port_speed_gbps"] < 1


def test_network_switch_fact_extraction_parses_ports_speed_poe_l3_and_stacking() -> None:
    tiny = match_engine_module._extract_network_device_facts(
        "5-Port 100Base-TX unmanaged desktop switch"
    )
    enterprise = match_engine_module._extract_network_device_facts(
        "48-Port Gigabit PoE+ switch 4x10G SFP+ L3 Stackable managed"
    )

    assert tiny["port_count"] == 5
    assert tiny["port_speed"] == "100MbE"
    assert tiny["port_speed_gbps"] == 0.1
    assert tiny["managed_status"] == "unmanaged"
    assert enterprise["port_count"] == 48
    assert enterprise["port_speed"] == "1GbE"
    assert enterprise["uplink_count"] == 4
    assert enterprise["uplink_speed"] == "10GbE"
    assert enterprise["uplink_media"] == "SFP+"
    assert enterprise["poe_supported"] is True
    assert enterprise["poe_standard"] == "PoE+"
    assert enterprise["l3_supported"] is True
    assert enterprise["stacking_supported"] is True


def test_storage_matrix_filters_ready_servers_accessories_and_wifi_host_ports(
    db_session: Session,
) -> None:
    _seed_nerpa_product(
        db_session,
        item_id="ready-server-in-storage-db",
        part_number="READY-STORAGE-LEAK",
        item_name="Server 2U 2x CPU 512GB RAM SSD",
        quantity=2,
        can_reserve=True,
    )
    for item_id, category_id, name, price in [
        (
            "storage-array",
            "V2103",
            "Storage array 500TB usable capacity dual controller SSD FC",
            "10000",
        ),
        ("storage-dac", "V2104", "Dell DAC cable for storage shelf", "40"),
        ("storage-ram", "V2104", "QNAP RAM memory module", "80"),
        ("storage-adapter", "V2104", "M.2 adapter card", "35"),
        ("storage-host-port", "V2104", "Storage FC 32G host port interface module", "300"),
        ("wifi-controller", "V120109", "Wi-Fi access point controller", "200"),
        ("server-psu", "V110108", "Server PSU power supply 800W", "100"),
        ("switch-dac", "V120150", "Switch DAC twinax cable 10G", "30"),
        ("storage-support", "V3100", "Storage support service warranty 3 years", "1000"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=item_id.upper(),
            producer="StorageVendor",
            category_id=category_id,
            item_name=name,
            quantity=10,
            price=Decimal(price),
        )

    result = asyncio.run(match_stock_spec(_storage_fc_spec(), _adapter(db_session)))
    matrix = result.to_report_json()["component_candidate_matrix"]
    storage_names = [
        str(candidate["name"]).casefold()
        for candidate in matrix["storage_system_candidates"]
    ]
    support_names = [
        str(candidate["name"]).casefold()
        for candidate in matrix["support_candidates"]
    ]
    host_port_names = [
        str(candidate["name"]).casefold()
        for candidate in matrix["host_port_candidates"]
    ]
    active_roles = {
        role
        for role, count in matrix["component_matrix_coverage_summary"][
            "sent_to_llm_by_role"
        ].items()
        if count
    }

    assert matrix["product_group"] == "storage"
    assert matrix["ready_server_candidates"] == []
    assert matrix["storage_system_candidates"]
    assert "fit_tier" in matrix["storage_system_candidates"][0]
    assert matrix["storage_system_candidates"][0]["part_number"] == "STORAGE-ARRAY"
    assert not any("dac" in name for name in storage_names)
    assert not any("ram" in name for name in storage_names)
    assert not any("adapter" in name for name in storage_names)
    assert support_names == ["storage support service warranty 3 years"]
    assert not any("psu" in name or "dac" in name or "cable" in name for name in support_names)
    assert not any("wi-fi" in name or "access point controller" in name for name in host_port_names)
    assert active_roles.isdisjoint({"ready_server", "platform", "cpu", "ram"})


def test_pipeline_semantic_amd_25gbe_context_does_not_block_composer(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_amd_25gbe_component_set(db_session)

    def fake_plan_semantic_matrix_roles(
        spec: StockSpec,
        *,
        distributor_code: str | None = None,
        planner_client: Any = None,
        deterministic_product_group_hint: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return _semantic_amd_25gbe_role_plan()

    monkeypatch.setattr(
        match_engine_module,
        "plan_semantic_matrix_roles",
        fake_plan_semantic_matrix_roles,
    )
    fake_client = _FakeComposerClient(_primary_recommendation_with_network_response)
    spec = StockSpec(
        items=[StockSpecItem(item_type="server", quantity=2, name="server")],
        source_text=(
            "Нужно подобрать 2 сервера под виртуализацию, базы данных и локальное "
            "NVMe-хранилище, склад Москва. Требования на каждый сервер: "
            "2 процессора AMD EPYC, не менее 32 ядер на процессор; "
            "не менее 768 ГБ RAM DDR5 RDIMM; "
            "4 SSD NVMe не менее 7.68 ТБ на сервер; "
            "минимум 2 сетевых порта 25GbE SFP28; 2 блока питания. "
            "Нужен один самый дешевый складской вариант для КП. "
            "Инженерная проверка перед КП обязательна."
        ),
    )

    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_configurator_client=fake_client,
            llm_settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        )
    )
    report_json = result.to_report_json()

    assert report_json["unsupported_or_unmapped_requirements"] == []
    assert report_json["llm_configurator_used"] is True
    assert report_json["primary_recommendation_status"] == "valid"
    assert "network_adapter" in report_json["required_roles"]
    commercial_text = report_json["commercial_summary"]["copy_paste_text"]
    assert "Сеть:" in commercial_text
    assert "LRES1026PF-2SFP28, 2 x 25GbE SFP28" in commercial_text
    assert "на сервер / 2 шт. всего" in commercial_text
    assert "базы данных" not in report_json["unsupported_or_unmapped_requirements"]
    assert "склад Москва" not in report_json["unsupported_or_unmapped_requirements"]
    assert "КП" not in report_json["unsupported_or_unmapped_requirements"]


def test_semantic_planner_uses_existing_llm_provider_when_composer_disabled(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_complete_component_set(db_session)
    _seed_component_product(
        db_session,
        item_id="nic-10g-sfp",
        part_number="NIC-10G-SFP",
        producer="Intel",
        category_id="V120116",
        item_name="Intel X710-DA2 dual-port 10GbE SFP+ server adapter",
        quantity=2,
        price=Decimal("200"),
    )
    monkeypatch.setattr(
        match_engine_module,
        "OpenAICompatibleLlmClient",
        _FakeSemanticPlannerOpenAIClient,
    )
    settings = _llm_composer_settings(output_mode="single_best_cost_valid").model_copy(
        update={"llm_configurator_enabled": False, "llm_configurator_mode": "disabled"}
    )
    spec = extract_stock_spec_for_text_match(
        "1U 2-socket server, Intel CPUs, DDR5 RAM, SATA SSD, "
        "LSI HBA, Intel X710-DA2 2x10GbE SFP+, C13-C14 power cables"
    ).spec_json

    result = asyncio.run(match_stock_spec(spec, _adapter(db_session), llm_settings=settings))
    report_json = result.to_report_json()

    assert report_json["semantic_planner_source"] == "llm"
    assert report_json["semantic_planner_used"] is True
    assert report_json["semantic_planner_provider"] == "openai-compatible"
    assert report_json["semantic_planner_model"] == "test-model"
    assert report_json["product_group"] == "server"
    assert report_json["primary_object"] == "server"
    assert "network_adapter" in report_json["required_roles"]
    assert report_json["category_plan"]["server_platform"] == ["V110100"]
    assert report_json["category_plan"]["network_adapter"] == ["V120116"]
    assert report_json["llm_configurator_used"] is False


def test_pipeline_complex_78_fails_closed_when_semantic_llm_unavailable(
    db_session: Session,
) -> None:
    spec = extract_stock_spec_for_text_match(
        """
Исполнение: 1U
Сокеты: 2
ПРОЦЕССОР: Intel 6-го поколения 2шт, не менее 24 ядер
ОПЕРАТИВНАЯ ПАМЯТЬ: 256 ГБ DDR5 RDIMM
ДИСКИ: 6 x SSD 1920 GB SATA, 2 x SSD 480 GB SATA
КОНТРОЛЛЕР: LSI Logic 9400-8i / LSI 9500-8i
СЕТЕВОЙ АДАПТЕР: Intel X710-DA2 2x10GbE SFP+
БП: 2 x 2000W hot-swap Platinum, C13-C14 cables, C13-Schuko cables
ОХЛАЖДЕНИЕ: 8 fans N+1
ИНТЕРФЕЙСЫ: USB 3.0, serial RJ-45, VGA, remote management RJ-45
""".strip()
    ).spec_json

    result = asyncio.run(
        match_stock_spec(
            spec,
            _adapter(db_session),
            llm_settings=LlmSettings(llm_provider="disabled"),
        )
    )
    report_json = result.to_report_json()

    assert report_json["product_group"] == "unknown"
    assert report_json["semantic_planner_source"] == (
        "complex_request_requires_llm_semantic_planner"
    )
    assert report_json["semantic_planner_fallback_reason"] == (
        "complex_request_requires_llm_semantic_planner"
    )
    assert report_json["category_plan"] == {}
    assert report_json["primary_recommendation_status"] == "no_recommendation"
    assert "Не удалось безопасно разобрать сложный запрос" in report_json[
        "no_recommendation_reason"
    ]["summary"]
    assert not any(
        capability["role"] in {"dac_cable", "cable"}
        for capability in report_json["required_capabilities"]
    )


def test_pipeline_no_recommendation_propagates_missing_network_capability(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_amd_25gbe_component_set(db_session)

    def fake_plan_semantic_matrix_roles(
        spec: StockSpec,
        *,
        distributor_code: str | None = None,
        planner_client: Any = None,
        deterministic_product_group_hint: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return _semantic_amd_25gbe_role_plan()

    monkeypatch.setattr(
        match_engine_module,
        "plan_semantic_matrix_roles",
        fake_plan_semantic_matrix_roles,
    )

    result = asyncio.run(
        match_stock_spec(
            StockSpec(items=[StockSpecItem(item_type="server", quantity=2, name="server")]),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(_primary_recommendation_response),
            llm_settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        )
    )
    report_json = result.to_report_json()

    assert report_json["primary_recommendation_status"] == "no_recommendation"
    assert any(
        row["role"] == "network_adapter"
        for row in report_json["missing_required_capabilities"]
    )
    assert any(
        row["role"] == "network_adapter"
        for row in report_json["no_recommendation_reason"][
            "missing_required_capabilities"
        ]
    )
    assert any(
        row["role"] == "network_adapter"
        for row in report_json["commercial_summary"]["missing_required_capabilities"]
    )


def test_pipeline_no_eligible_network_candidate_preblocks_with_precise_missing(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_amd_25gbe_component_set(db_session, include_network=False)
    _seed_component_product(
        db_session,
        item_id="nic-10g-sfp",
        part_number="NIC-10G-SFP",
        producer="Intel",
        category_id="V120116",
        item_name="Dual-port 10GbE SFP+ PCIe network adapter",
        quantity=2,
        price=Decimal("120"),
    )

    def fake_plan_semantic_matrix_roles(
        spec: StockSpec,
        *,
        distributor_code: str | None = None,
        planner_client: Any = None,
        deterministic_product_group_hint: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return _semantic_amd_25gbe_role_plan()

    monkeypatch.setattr(
        match_engine_module,
        "plan_semantic_matrix_roles",
        fake_plan_semantic_matrix_roles,
    )

    result = asyncio.run(
        match_stock_spec(
            StockSpec(items=[StockSpecItem(item_type="server", quantity=2, name="server")]),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(
                _primary_recommendation_with_network_response
            ),
            llm_settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
        )
    )
    report_json = result.to_report_json()
    coverage = report_json["role_coverage_summary"]["network_adapter"]
    commercial_text = report_json["commercial_summary"]["copy_paste_text"]

    assert report_json["primary_recommendation_status"] == "no_recommendation"
    assert coverage["can_be_satisfied_by_platform"] is True
    assert coverage["platform_satisfied_candidates_count"] == 0
    assert coverage["after_eligibility_count"] == 1
    assert coverage["missing_candidates"] is True
    assert coverage["missing"] is True
    assert any(
        row["role"] == "network_adapter"
        and row["status"] == "missing_candidates"
        for row in report_json["missing_required_capabilities"]
    )
    assert any(
        row["role"] == "network_adapter"
        for row in report_json["no_recommendation_reason"][
            "missing_required_capabilities"
        ]
    )
    assert any(
        row["role"] == "network_adapter"
        for row in report_json["commercial_summary"]["missing_required_capabilities"]
    )
    assert "Не закрыто требование" in commercial_text
    assert "25GbE" in commercial_text
    assert "SFP28" in commercial_text
    assert "onboard" in commercial_text


def test_single_best_rejects_platform_onboard_1gbe_for_25gbe_sfp28() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers with 2 ports 25GbE SFP28 per server.",
        normalized_requirements=[_network_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_network(
            include_network=False,
            platform_network_facts={
                "network_ports_count": 4,
                "network_speed": "1GbE",
                "network_speed_gbps": 1,
                "network_media": "RJ45",
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert "network_adapter" in outcome.no_recommendation_reason["missing_roles"]
    missing = outcome.no_recommendation_reason["missing_required_capabilities"]
    assert any(row["role"] == "network_adapter" for row in missing)
    assert any(
        row["role"] == "network_adapter"
        for row in outcome.commercial_summary["missing_required_capabilities"]
    )


def test_single_best_accepts_storage_primary_from_component_matrix() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужна СХД 100 ТБ usable, SSD, FC 32G, поддержка 3 года.",
        normalized_requirements=[_storage_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_storage_component_matrix(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_storage_full_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    selected_ids = {
        row["component_candidate_id"]
        for row in outcome.primary_recommendation["components"]
    }
    assert selected_ids == {"storage-system-100u", "drive-ssd-768", "support-3y"}
    assert outcome.primary_recommendation["total_price_value"] == "13400"
    quantities = {
        row["role"]: row["quantity_required"]
        for row in outcome.primary_recommendation["components"]
    }
    assert quantities["storage_system"] == 1
    assert quantities["drive"] == 24
    assert quantities["support"] == 1
    commercial_text = json.dumps(outcome.commercial_summary, ensure_ascii=False)
    assert "СХД - 1 шт." in commercial_text
    for forbidden in ("component_candidate_id", "storage-system-100u", '"raw"', "debug"):
        assert forbidden not in commercial_text


@pytest.mark.parametrize("alias", ["platform", "base_device"])
def test_single_best_storage_normalizes_base_device_alias_to_storage_system(
    alias: str,
) -> None:
    outcome = compose_llm_configurations(
        user_request=(
            "РќСѓР¶РЅР° РЎРҐР” 100 РўР‘ usable, SSD, FC 32G, "
            "РїРѕРґРґРµСЂР¶РєР° 3 РіРѕРґР°."
        ),
        normalized_requirements=[_storage_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_storage_component_matrix(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_storage_alias_response(alias)),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["component_candidate_ids"]["storage_system"] == (
        "storage-system-100u"
    )
    assert alias not in outcome.primary_recommendation["component_candidate_ids"]
    assert outcome.validation_summary["rejected_role_mismatch"] == 0


def test_single_best_storage_no_recommendation_when_fc_speed_not_closed() -> None:
    matrix = _storage_component_matrix()
    matrix["storage_system_candidates"][0]["host_port_speed"] = "16G"
    matrix["storage_system_candidates"][0]["host_port_speed_gbps"] = 16
    matrix["storage_system_candidates"][0]["extracted_facts"]["host_port_speed"] = "16G"
    matrix["storage_system_candidates"][0]["extracted_facts"]["host_port_speed_gbps"] = 16

    outcome = compose_llm_configurations(
        user_request="Нужна СХД 100 ТБ usable, SSD, FC 32G, поддержка 3 года.",
        normalized_requirements=[_storage_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_storage_full_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert any(
        row["role"] == "storage_system"
        for row in outcome.no_recommendation_reason["missing_required_capabilities"]
    )
    assert "Безопасную складскую рекомендацию дать нельзя." in json.dumps(
        outcome.commercial_summary,
        ensure_ascii=False,
    )


def test_single_best_satisfies_power_supply_min_2_from_platform_bundle_1_plus_1() -> None:
    requirements = _power_supply_composer_requirements()
    matrix = _composer_component_matrix_with_network(include_network=False)
    matrix["platform_candidates"][0]["name"] = (
        "ASUS RS521A-E12-RS24U server platform 1+1 1600W PSU"
    )

    outcome = compose_llm_configurations(
        user_request="Need 2 servers with 2 PSU per server.",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    power = next(
        row
        for row in outcome.primary_recommendation["hard_capability_validation"]
        if row["role"] == "power_supply"
    )
    assert power["status"] == "satisfied"
    assert power["satisfied_by"] == "platform_bundle"
    assert power["component_role"] == "server_platform"


def test_single_best_satisfies_power_supply_min_2_from_platform_bundle_2x1300w() -> None:
    requirements = _power_supply_composer_requirements()
    matrix = _composer_component_matrix_with_network(include_network=False)
    matrix["platform_candidates"][0]["name"] = (
        "ASUS RS521A-E12-RS24U server platform CRPS 2x1300W"
    )

    outcome = compose_llm_configurations(
        user_request="Need 2 servers with redundant PSU.",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert not outcome.primary_recommendation["missing_required_capabilities"]


def test_single_best_rejects_power_supply_min_2_without_platform_bundle() -> None:
    requirements = _power_supply_composer_requirements()

    outcome = compose_llm_configurations(
        user_request="Need 2 servers with 2 PSU per server.",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_network(
            include_network=False
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert any(
        row["role"] == "power_supply"
        and row["status"] == "missing_component"
        for row in outcome.no_recommendation_reason["missing_required_capabilities"]
    )
    commercial_text = outcome.commercial_summary["copy_paste_text"]
    assert "Не закрыто требование" in commercial_text
    assert "2 PSU" in commercial_text


def test_single_best_rejects_missing_required_gpu() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers with NVIDIA GPU.",
        normalized_requirements=[_gpu_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_gpu(include_gpu=True),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert "gpu" in outcome.no_recommendation_reason["missing_roles"]


def test_single_best_accepts_required_gpu_and_commercial_summary_shows_gpu() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers with NVIDIA GPU.",
        normalized_requirements=[_gpu_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_gpu(include_gpu=True),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_with_gpu_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert _component_by_role(outcome.primary_recommendation["components"], "gpu")
    assert "GPU:" in outcome.commercial_summary["copy_paste_text"]


def test_single_best_rejects_missing_required_storage_controller() -> None:
    requirements = _right_size_composer_requirements()
    requirements["required_roles"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "storage_controller",
    ]
    requirements["required_capabilities"] = [
        {
            "capability_id": "storage_controller.requested",
            "role": "storage_controller",
            "hard": True,
            "requirement_text": "Need RAID HBA",
            "parsed_requirements": {"required": True},
        }
    ]

    outcome = compose_llm_configurations(
        user_request="Need RAID HBA.",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_network(
            include_network=False,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert "storage_controller" in outcome.no_recommendation_reason["missing_roles"]


def test_single_best_accepts_required_storage_controller_and_summary_shows_controller() -> None:
    requirements = _storage_controller_composer_requirements()

    outcome = compose_llm_configurations(
        user_request="Need RAID HBA.",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_storage_controller(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_with_controller_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert _component_by_role(
        outcome.primary_recommendation["components"],
        "storage_controller",
    )
    assert "Контроллер:" in outcome.commercial_summary["copy_paste_text"]


def test_single_best_no_recommendation_for_unsupported_hard_requirement() -> None:
    requirements = _right_size_composer_requirements()
    requirements["unsupported_or_unmapped_requirements"] = ["quantum flux capacitor"]

    outcome = compose_llm_configurations(
        user_request="Need quantum flux capacitor.",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_network(
            include_network=False,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert "unsupported" in outcome.no_recommendation_reason["missing_roles"]


def test_single_best_network_adapter_stock_shortage_blocks_recommendation() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers with 2 ports 25GbE SFP28 per server.",
        normalized_requirements=[_network_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_network(
            include_network=True,
            network_available_quantity=1,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_primary_recommendation_with_network_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "no_recommendation"
    assert any(
        shortage["role"] == "network_adapter"
        for shortage in outcome.no_recommendation_reason["stock_shortages"]
    )


def test_llm_response_with_qwen_enum_alias_does_not_reject_entire_pool() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers 2U, 2 CPU, 512GB RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_eight_proposals_one_qwen_alias_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.proposal_count == 8
    assert outcome.valid_proposals_count == 8
    assert outcome.validation_summary["rejected_invalid_schema"] == 0
    assert not [
        row
        for row in outcome.rejected_recommendations_debug_safe
        if row["rejection_code"] == "invalid_schema"
    ]


def test_llm_per_item_schema_validation_keeps_valid_proposals() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers 2U, 2 CPU, 512GB RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_one_bad_schema_one_valid_qwen_response),
        settings=_llm_composer_settings(),
    )
    invalid_schema_debug = [
        row
        for row in outcome.rejected_recommendations_debug_safe
        if row["rejection_code"] == "invalid_schema"
    ]

    assert outcome.used is True
    assert outcome.proposal_count == 2
    assert outcome.valid_proposals_count == 1
    assert outcome.validation_summary["rejected_invalid_schema"] == 1
    assert len(invalid_schema_debug) == 1
    assert invalid_schema_debug[0]["proposal_index"] == 0
    assert invalid_schema_debug[0]["recommendation_id"] == "llm_bad_schema"
    assert invalid_schema_debug[0]["validation_errors"][0]["loc"] == ["confidence"]


def test_llm_selected_component_ids_storage_alias_materializes_to_ssd() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers 2U, 2 CPU, 512GB RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_selected_storage_alias_llm_response),
        settings=_llm_composer_settings(),
    )
    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert recommendation["component_candidate_ids"]["ssd"] == "ssd-3840"
    assert "storage" not in recommendation["component_candidate_ids"]
    assert recommendation["quantities"]["ssd"] == 4
    assert recommendation["normalized_bom_quantities"]["ssd"]["quantity_source"] == (
        "code_materialized"
    )


def test_llm_selected_ssd_materializes_generic_drive_hard_quantity() -> None:
    requirements = _composer_requirements()
    requirements["required_roles"] = ["server_platform", "cpu", "ram", "drive"]
    requirements["storage_type_preference"] = None
    requirements["role_plan"] = {
        "product_group": "server",
        "required_roles": requirements["required_roles"],
        "requirements_by_role": {
            "drive": {"required": True, "count_per_server": 2},
        },
    }

    def response(package: dict[str, Any]) -> dict[str, Any]:
        recommendation = _component_matrix_llm_recommendation(
            package,
            recommendation_id="llm_underselected_generic_drive",
            why_selected="LLM selected an SSD but undercounted generic drive quantity.",
            why_selected_short="Selected SSD with too-low quantity.",
        )
        recommendation["quantities"]["storage"] = 1
        return {"recommendations": [recommendation], "general_notes": []}

    outcome = compose_llm_configurations(
        user_request="Need 2 servers with 2 drives per server.",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]
    ssd = _component_by_role(recommendation["components"], "ssd")

    assert outcome.primary_recommendation_status == "valid"
    assert recommendation["quantities"]["ssd"] == 4
    assert ssd["quantity_required"] == 4
    assert recommendation["normalized_bom_quantities"]["ssd"]["llm_quantity"] == 1


@pytest.mark.parametrize(
    "phrase",
    [
        "512 ГБ RAM DDR5",
        "512 GB RAM DDR5",
        "512 ГБ оперативной памяти DDR5",
        "по 512 ГБ RAM на сервер",
        "на каждый сервер 512 ГБ RAM DDR5",
        "512 гб озу ddr5",
        "по 512 гб оперативки DDR5",
    ],
)
def test_match_engine_extracts_ram_amount_and_ddr5_preference(phrase: str) -> None:
    spec = StockSpec(
        items=[StockSpecItem(item_type="server", quantity=2, name="server")],
        source_text=f"Нужно 2 сервера, {phrase}, склад Москва",
    )

    requirements = match_engine_module._normalize_server_requirements(spec, spec.items[0])

    assert requirements.ram_gb_per_server == 512
    assert requirements.ram_type_preference == "DDR5"


def test_llm_configurator_can_create_new_build_from_component_candidate_ids() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert recommendation["source_candidate_id"] is None
    assert recommendation["source_type"] == "build_from_parts"
    assert recommendation["component_candidate_ids"]["platform"] == "platform-amd"
    assert recommendation["component_candidate_ids"]["cpu"] == "cpu-selected"
    assert recommendation["total_price_value"] == "9600"
    assert recommendation["why_selected"] == "LLM composed a new BOM from component matrix."
    assert recommendation["display_name"] == "ASUS RS521A-E12-RS24U"
    assert "Fabricated" not in str(recommendation)


def test_composer_prompt_keeps_optional_addons_out_of_core_bom() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SYSTEM_PROMPT

    assert "optional_component_candidate_ids" in prompt
    assert "not only 3 final cards" in prompt
    assert "Do not put storage_controller, network_adapter" in prompt
    assert "It must not increase the mandatory\n  minimum BOM price" in prompt
    assert "Without external evidence" in prompt


def test_composer_prompt_contains_qwen_strict_protocol() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SYSTEM_PROMPT

    assert "QWEN STRICT COMPOSER PROTOCOL" in prompt
    assert "You are not a free-form chatbot." in prompt
    assert "You are a procurement/presales reasoning engine." in prompt
    assert "Application code is the source of truth for price, stock, quantities" in prompt


def test_composer_prompt_defines_strict_core_bom_rules() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SYSTEM_PROMPT

    assert "CORE BOM RULES" in prompt
    assert "selected_component_candidate_ids/component_candidate_ids are core BOM only" in prompt
    assert "Do not include storage_controller, network_adapter, cable, rail" in prompt
    assert "Optional components must not be used to make the proposal look different" in prompt


def test_composer_prompt_defines_product_group_role_contract() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SYSTEM_PROMPT

    assert "PRODUCT-GROUP ROLE CONTRACT" in prompt
    assert 'component_candidate_ids with "switch": "..."' in prompt
    assert "Do not use server roles such as platform/cpu/ram/storage for network" in prompt
    assert 'component_candidate_ids with "storage_system": "..."' in prompt
    assert "Do not use platform for storage_system" in prompt


def test_single_best_prompt_is_product_group_first_and_universal() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SINGLE_BEST_SYSTEM_PROMPT

    assert "one cheapest valid universal stock quote" in prompt
    assert "Read package.product_group before selecting roles" in prompt
    assert "server, network, storage" in prompt
    assert "For product_group=network" in prompt
    assert "For product_group=storage" in prompt


def test_composer_prompt_contains_quantity_reasoning_examples() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SYSTEM_PROMPT

    assert "QUANTITY REASONING RULES" in prompt
    assert "total_cpu = cpu_per_server * requested_server_count" in prompt
    assert "total_drives = required_drives_per_server * requested_server_count" in prompt
    assert "2 servers, 512 GB RAM per server, 32 GB modules: expected total is 32 modules" in prompt
    assert "2 servers, 512 GB RAM per server, 64 GB modules: expected total is 16 modules" in prompt


def test_composer_prompt_requests_full_proposal_pool_and_self_check() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SYSTEM_PROMPT

    assert "Return a proposal pool up to proposal_pool_limit" in prompt
    assert "Before returning JSON, silently verify:" in prompt
    assert "Does every selected component_candidate_id exist in the input matrix?" in prompt
    assert "Are there duplicate proposals that differ only by optional peripherals?" in prompt


def test_composer_prompt_separates_commercial_and_engineering_confidence() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SYSTEM_PROMPT

    assert "confidence means commercial fit only, not engineering validation" in prompt
    assert "Do not use high confidence to mean engineering compatibility" in prompt
    assert "engineering_confidence must be preliminary_requires_engineer_review" in prompt
    assert "Every recommendation remains preliminary" in prompt
    assert "Engineer review is mandatory before quotation" in prompt


def test_composer_prompt_has_qwen_final_json_contract() -> None:
    prompt = configuration_composer_module.LLM_CONFIGURATOR_SYSTEM_PROMPT

    assert "FINAL OUTPUT CONTRACT" in prompt
    assert "The final answer must be exactly one JSON object." in prompt
    assert "No markdown." in prompt
    assert "No code fences." in prompt
    assert "No prose before or after JSON." in prompt
    assert "No <think> tags." in prompt
    assert "No chain-of-thought." in prompt
    assert "No trailing commas." in prompt
    assert "Use only double quotes." in prompt
    assert "The root object must contain a recommendations array." in prompt
    assert "must not return an empty recommendations list without structured" in prompt
    assert "missing_required_capabilities" in prompt
    assert "hard_mismatches" in prompt
    assert "considered_candidate_ids" in prompt
    assert "explanation_ru" in prompt
    assert "Reasoning must be compressed into short fields" in prompt
    assert "Do not output hidden reasoning." in prompt


def test_llm_accepts_selected_component_ids_alias_and_proposal_role() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_selected_component_ids_alias_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert recommendation["component_candidate_ids"]["platform"] == "platform-amd"
    assert recommendation["proposal_role"] == "cheapest_fit"
    assert recommendation["recommendation_slot"] == "price_optimal"
    assert recommendation["commercial_tradeoff"] == "Cheapest valid core BOM."
    assert "Check vendor QVL before quotation." in recommendation["critical_checks"]
    assert recommendation["component_candidate_ids"]["cpu"] == "cpu-selected"
    assert recommendation["normalized_bom_quantities"]["ram"]["quantity_source"] == (
        "code_materialized"
    )


@pytest.mark.parametrize(
    "wrapper_key",
    [
        "proposals",
        "proposal_pool",
        "recommendations",
        "ai_recommendations",
        "llm_recommendations",
    ],
)
def test_llm_composer_normalizes_proposal_wrapper_keys(wrapper_key: str) -> None:
    def payload_factory(package: dict[str, Any]) -> dict[str, Any]:
        return {
            wrapper_key: [
                _component_matrix_llm_recommendation(
                    package,
                    recommendation_id=f"llm_{wrapper_key}",
                    why_selected="Wrapped proposal should be normalized.",
                    why_selected_short="Wrapped proposal.",
                )
            ],
            "general_notes": [],
        }

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_RawPayloadComposerClient(payload_factory),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.proposal_count == 1
    assert outcome.recommended_builds[0]["component_candidate_ids"]["platform"] == (
        "platform-amd"
    )


def test_llm_composer_normalizes_root_array_as_proposal_list() -> None:
    def payload_factory(package: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            _component_matrix_llm_recommendation(
                package,
                recommendation_id="llm_root_array",
                why_selected="Root array should be normalized.",
                why_selected_short="Root array proposal.",
            )
        ]

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_RawPayloadComposerClient(payload_factory),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.proposal_count == 1
    assert outcome.recommended_builds[0]["recommendation_id"] == "llm_root_array"


def test_llm_optional_peripherals_do_not_inflate_mandatory_core_price() -> None:
    matrix = _composer_component_matrix_with_optional_peripherals()

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_optional_peripherals_in_core_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert recommendation["total_price_value"] == "9600"
    assert recommendation["optional_component_roles"] == [
        "storage_controller",
        "network_adapter",
    ]
    assert recommendation["excluded_from_total_roles"] == [
        "storage_controller",
        "network_adapter",
    ]
    assert recommendation["optional_total_price_value"] == "34000"
    assert recommendation["optional_total_price_currency"] == "USD"
    assert len(recommendation["optional_components"]) == 2
    assert "опциональные Контроллеры, Сетевые адаптеры" in recommendation[
        "total_price_note"
    ]
    assert recommendation["engineering_confidence"] == "preliminary_requires_engineer_review"
    assert "инженерная подтвержденность: предварительно" in recommendation[
        "displayed_confidence"
    ]
    assert recommendation["confidence"] != "high"


def test_selection_skips_duplicate_builds_when_only_optional_peripherals_differ() -> None:
    matrix = _composer_component_matrix_with_optional_peripherals()

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_optional_only_duplicate_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert [row["recommendation_id"] for row in outcome.recommended_builds] == [
        "llm_core_without_optional"
    ]
    assert outcome.valid_proposals_count == 2
    assert outcome.selection_skipped_count == 1
    assert outcome.validation_summary["selection_skipped_duplicate"] == 1
    assert any(
        "duplicate_same_core_bom_optional_peripherals" in warning
        for warning in outcome.internal_warnings
    )


def test_grouped_presales_groups_same_component_base_as_platform_options() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, 2 SSD NVMe.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_same_component_base_platforms_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.grouped_presales_mode_used is True
    assert len(outcome.configuration_groups) == 1

    group = outcome.configuration_groups[0]
    options = group["platform_options"]

    assert group["group_title"].startswith("Intel LGA4677 / DDR5 / NVMe")
    assert "минимальная базовая конфигурация" in group["group_title"]
    assert set(group["component_base"]) == {"cpu", "ram", "storage"}
    assert len(options) == 3
    assert {option["platform"]["part_number"] for option in options} == {
        "PLATFORM-CHEAP",
        "SYS-621C-TN12R",
        "R283-ZK0",
    }
    assert options[0]["role"] == "cheapest_quote"
    assert outcome.quote_recommendation["recommended_group_id"] == group["group_id"]
    assert outcome.selected_configuration_group_id == group["group_id"]
    assert outcome.selected_platform_option_id == options[0]["option_id"]


def test_grouped_presales_keeps_different_component_bases_as_separate_groups() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, 2 SSD NVMe.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_different_component_bases_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert len(outcome.configuration_groups) == 2
    cpu_parts = {
        group["component_base"]["cpu"]["part_number"]
        for group in outcome.configuration_groups
    }
    assert cpu_parts == {"CPU-SELECTED", "CPU-16C"}


def test_grouped_presales_separates_intel_lga4677_and_amd_sp5() -> None:
    matrix = _composer_component_matrix_with_amd_sp5_family()

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, 2 SSD NVMe.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_intel_and_amd_family_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert len(outcome.configuration_groups) == 2
    titles = {group["group_title"] for group in outcome.configuration_groups}
    assert any(title.startswith("Intel LGA4677 / DDR5 / NVMe") for title in titles)
    assert any(title.startswith("AMD SP5 / DDR5 / NVMe") for title in titles)


def test_grouped_presales_hides_same_platform_storage_change_without_tradeoff() -> None:
    matrix = _composer_component_matrix_with_alternative_ssd()

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, 2 SSD NVMe.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_same_platform_storage_no_tradeoff_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.valid_proposals_count == 2
    assert len(outcome.configuration_groups) == 1
    assert sum(len(group["platform_options"]) for group in outcome.configuration_groups) == 1


def test_grouped_presales_component_base_notes_show_materialized_ram_quantity() -> None:
    outcome = compose_llm_configurations(
        user_request=(
            "Нужно подобрать 2 сервера 2U под базу данных. На каждый сервер: "
            "2 процессора Intel Xeon, 512 ГБ RAM DDR5, 2 SSD NVMe 3.84 ТБ."
        ),
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_ram_module(
            32,
            available_quantity=100,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_double_ram_quantity_llm_response),
        settings=_llm_composer_settings(),
    )

    notes = outcome.configuration_groups[0]["component_base_notes"]

    assert "RAM: 16 x 32 ГБ на сервер = 512 ГБ" in notes
    assert "RAM: 32 шт. всего" in notes


def test_repair_critic_detects_cheaper_equivalent_ram_and_saving() -> None:
    matrix = _composer_component_matrix_with_cheaper_equivalent_ram()
    client = _RepairAwareComposerClient(
        primary_responder=_samsung_ram_primary_response,
        repair_responder=_micron_ram_repair_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(),
    )
    repair_package = client.repair_packages[0]
    critique = repair_package["critique_facts"][0]

    assert outcome.repair_critique_count == 1
    assert critique["classification"] == "cheaper_equivalent"
    assert critique["role"] == "ram"
    assert critique["selected"]["producer"] == "Samsung"
    assert critique["selected"]["price_value"] == "1493"
    assert critique["selected"]["quantity_required"] == 32
    assert critique["selected"]["line_total_value"] == "47776"
    assert critique["alternative"]["producer"] == "Micron"
    assert critique["alternative"]["price_value"] == "1300"
    assert critique["alternative"]["available_quantity"] == 100
    assert critique["alternative"]["quantity_required"] == 32
    assert critique["alternative"]["line_total_value"] == "41600"
    assert critique["saving_value"] == "6176"
    assert "512" in " ".join(critique["facts"])
    assert "DDR5" in " ".join(critique["facts"])


def test_repair_ram_ddr4_candidate_for_ddr5_is_blocked_as_type_mismatch() -> None:
    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_ram_alternative(
            _repair_ram_candidate(
                candidate_id="ram-ddr4-32",
                producer="Budget",
                part_number="RAM-DDR4-32G",
                name="Budget 32GB DDR4 RDIMM server memory module",
                available_quantity=100,
                price=Decimal("900"),
                facts={"ram_capacity_gb": 32, "ram_type": "DDR4"},
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_samsung_ram_primary_response),
        settings=_llm_composer_settings(),
    )

    blocked_text = " ".join(outcome.repair_blocked_critique_summary)
    assert outcome.repair_critique_count == 0
    assert outcome.repair_blocked_critique_count == 1
    assert "ram_type_mismatch" in blocked_text


def test_repair_ram_unknown_generation_is_blocked_as_unknown() -> None:
    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_ram_alternative(
            _repair_ram_candidate(
                candidate_id="ram-unknown-32",
                producer="Budget",
                part_number="RAM-UNKNOWN-32G",
                name="Budget 32GB server memory module",
                available_quantity=100,
                price=Decimal("900"),
                facts={"ram_capacity_gb": 32},
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_samsung_ram_primary_response),
        settings=_llm_composer_settings(),
    )

    blocked_text = " ".join(outcome.repair_blocked_critique_summary)
    assert outcome.repair_critique_count == 0
    assert outcome.repair_blocked_critique_count == 1
    assert "ram_type_unknown" in blocked_text


def test_repair_ram_stock_shortage_still_blocks_equivalent_candidate() -> None:
    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_ram_alternative(
            _repair_ram_candidate(
                candidate_id="ram-short-stock-32",
                producer="Budget",
                part_number="RAM-SHORT-32G",
                name="Budget 32GB DDR5 RDIMM server memory module",
                available_quantity=8,
                price=Decimal("900"),
                facts={"ram_capacity_gb": 32, "ram_type": "DDR5"},
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_samsung_ram_primary_response),
        settings=_llm_composer_settings(),
    )

    blocked_text = " ".join(outcome.repair_blocked_critique_summary)
    assert outcome.repair_critique_count == 0
    assert outcome.repair_blocked_critique_count == 1
    assert "stock_shortage" in blocked_text


def test_repair_ram_sodimm_is_not_equivalent_to_server_rdimm() -> None:
    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_ram_alternative(
            _repair_ram_candidate(
                candidate_id="ram-sodimm-32",
                producer="Budget",
                part_number="RAM-SODIMM-32G",
                name="Budget 32GB DDR5 SODIMM memory module",
                available_quantity=100,
                price=Decimal("900"),
                facts={"ram_capacity_gb": 32, "ram_type": "DDR5"},
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_samsung_ram_primary_response),
        settings=_llm_composer_settings(),
    )

    blocked_text = " ".join(outcome.repair_blocked_critique_summary)
    assert outcome.repair_critique_count == 0
    assert outcome.repair_blocked_critique_count == 1
    assert "ram_type_mismatch" in blocked_text


def test_repair_pass_sends_safe_critique_without_raw_matrix_or_secrets() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_samsung_ram_primary_response,
        repair_responder=_micron_ram_repair_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_cheaper_equivalent_ram(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(api_key="super-secret-test-key"),
    )
    repair_package_text = json.dumps(client.repair_packages[0], ensure_ascii=False)

    assert outcome.repair_attempted is True
    assert "critique_facts" in client.repair_packages[0]
    assert "allowed_candidate_alternatives" in client.repair_packages[0]
    assert "component_candidate_matrix" not in client.repair_packages[0]
    assert "super-secret-test-key" not in repair_package_text
    assert "headers" not in repair_package_text
    assert "authorization" not in repair_package_text.casefold()


def test_repair_revised_cheaper_ram_is_materialized_and_repriced() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_samsung_ram_primary_response,
        repair_responder=_micron_ram_repair_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_cheaper_equivalent_ram(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(),
    )
    recommendation = outcome.recommended_builds[0]
    ram_component = next(
        component for component in recommendation["components"] if component["role"] == "ram"
    )

    assert outcome.repair_used is True
    assert outcome.repair_success is True
    assert recommendation["component_candidate_ids"]["ram"] == "ram-micron-32"
    assert recommendation["quantities"]["ram"] == 32
    assert ram_component["line_total_value"] == "41600"
    assert recommendation["total_price_value"] == "48800"
    assert outcome.repair_savings_estimate == "6176"


def test_single_best_repair_revises_primary_to_cheaper_micron_ram() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_samsung_ram_primary_response,
        repair_responder=_micron_ram_repair_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_cheaper_equivalent_ram(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.repair_used is True
    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["component_candidate_ids"]["ram"] == "ram-micron-32"
    assert outcome.recommended_builds[0]["total_price_value"] == "48800"
    assert client.repair_packages[0]["output_mode"] == "single_best_cost_valid"
    assert "Do not return alternatives." in client.repair_packages[0]["repair_instructions"]


def test_repair_failure_keeps_primary_proposals_and_sets_fallback_reason() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_samsung_ram_primary_response,
        repair_responder=_repair_invalid_json_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest quote for 2 servers, 512GB DDR5 RAM per server.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_cheaper_equivalent_ram(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(),
    )
    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert outcome.repair_attempted is True
    assert outcome.repair_success is False
    assert outcome.repair_fallback_reason == "llm_repair_invalid_json"
    assert recommendation["component_candidate_ids"]["ram"] == "ram-samsung-32"
    assert recommendation["total_price_value"] == "54976"


def test_repair_platform_with_unknown_form_factor_is_blocked() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers 2U, Intel Xeon, DDR5 and NVMe.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_blocked_platform(
            _blocked_platform_candidate(
                candidate_id="platform-unknown-form",
                part_number="UNKNOWN-FORM",
                name="Budget dual socket LGA4677 DDR5 NVMe 2x PSU platform",
                facts={
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_socket": "LGA4677",
                    "cpu_sockets": 2,
                    "ram_type": "DDR5",
                    "nvme_support": True,
                },
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_base_platform_primary_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.repair_critique_count == 0
    assert outcome.repair_blocked_critique_count == 1
    assert "platform_form_factor_unknown" in " ".join(
        outcome.repair_blocked_critique_summary
    )


def test_repair_platform_with_incomplete_chassis_text_is_blocked() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 complete servers 2U, Intel Xeon, DDR5 and NVMe.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_blocked_platform(
            _blocked_platform_candidate(
                candidate_id="platform-incomplete",
                part_number="INCOMPLETE",
                name="Budget 2U dual socket LGA4677 DDR5 NVMe chassis no CPU, Memory, HDD, PSU",
                facts={
                    "form_factor": "2U",
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_socket": "LGA4677",
                    "cpu_sockets": 2,
                    "ram_type": "DDR5",
                    "nvme_support": True,
                },
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_base_platform_primary_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.repair_critique_count == 0
    assert "platform_incomplete_chassis" in " ".join(
        outcome.repair_blocked_critique_summary
    )


def test_repair_platform_with_unknown_ram_type_is_blocked_for_ddr5() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers 2U, Intel Xeon, DDR5 and NVMe.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_blocked_platform(
            _blocked_platform_candidate(
                candidate_id="platform-ram-unknown",
                part_number="RAM-UNKNOWN",
                name="Budget Intel Xeon 2U dual socket NVMe 2x PSU platform",
                facts={
                    "form_factor": "2U",
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_sockets": 2,
                    "nvme_support": True,
                },
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_base_platform_primary_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.repair_critique_count == 0
    assert "platform_ram_type_unknown" in " ".join(
        outcome.repair_blocked_critique_summary
    )


def test_repair_platform_with_unknown_nvme_backplane_is_blocked() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers 2U, Intel Xeon, DDR5 and 2 NVMe SSD each.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_blocked_platform(
            _blocked_platform_candidate(
                candidate_id="platform-nvme-unknown",
                part_number="STORAGE-UNKNOWN",
                name="Budget 2U dual socket LGA4677 DDR5 2x PSU platform",
                facts={
                    "form_factor": "2U",
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_socket": "LGA4677",
                    "cpu_sockets": 2,
                    "ram_type": "DDR5",
                },
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_base_platform_primary_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.repair_critique_count == 0
    assert "platform_storage_unknown" in " ".join(
        outcome.repair_blocked_critique_summary
    )


def test_repair_gooxi_like_platform_is_allowed_when_hard_eligibility_passes() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_base_platform_primary_response,
        repair_responder=_gooxi_platform_repair_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest 2U Intel Xeon DDR5 NVMe quote.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_gooxi_repair_platform(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(),
    )
    critique = client.repair_packages[0]["critique_facts"][0]

    assert critique["classification"] == "cheaper_equivalent"
    assert critique["alternative"]["producer"] == "Gooxi"
    assert outcome.repair_success is True
    assert outcome.recommended_builds[0]["component_candidate_ids"]["platform"] == (
        "platform-gooxi-repair"
    )


def test_single_best_gooxi_like_eligible_cheapest_platform_can_become_primary() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_base_platform_primary_response,
        repair_responder=_gooxi_platform_repair_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest 2U Intel Xeon DDR5 NVMe quote.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_gooxi_repair_platform(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["component_candidate_ids"]["platform"] == (
        "platform-gooxi-repair"
    )
    assert outcome.grouped_presales_mode_used is False


def test_single_best_hard_ineligible_cheaper_platform_cannot_become_primary() -> None:
    outcome = compose_llm_configurations(
        user_request="Need 2 servers 2U Intel Xeon DDR5 NVMe.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_blocked_platform(
            _blocked_platform_candidate(
                candidate_id="platform-blocked-incomplete",
                part_number="BLOCKED",
                name="Budget Intel Xeon dual socket NVMe platform",
                facts={
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_sockets": 2,
                    "nvme_support": True,
                },
            )
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_base_platform_primary_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.primary_recommendation_status == "valid"
    assert outcome.primary_recommendation["component_candidate_ids"]["platform"] == (
        "platform-base-eligible"
    )
    assert outcome.repair_critique_count == 0
    assert outcome.repair_blocked_critique_count == 1


def test_blocked_cheaper_platform_is_diagnostic_not_repair_fact() -> None:
    client = _FakeComposerClient(_base_platform_primary_response)

    outcome = compose_llm_configurations(
        user_request="Need 2 servers 2U Intel Xeon DDR5 NVMe.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_blocked_platform(
            _blocked_platform_candidate(
                candidate_id="platform-blocked",
                part_number="BLOCKED",
                name="Budget Intel Xeon dual socket NVMe platform",
                facts={
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_sockets": 2,
                    "nvme_support": True,
                },
            )
        ),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(),
    )

    assert outcome.repair_critique_count == 0
    assert outcome.repair_blocked_critique_count == 1
    assert client.package.get("critique_facts") is None


def test_repair_prompt_limits_qwen_to_cheaper_equivalent_facts() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_base_platform_primary_response,
        repair_responder=_gooxi_platform_repair_response,
    )

    compose_llm_configurations(
        user_request="Need cheapest 2U Intel Xeon DDR5 NVMe quote.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_gooxi_repair_platform(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(),
    )

    prompt = client.repair_system_prompts[0]
    repair_package = client.repair_packages[0]
    assert "Use only critique_facts marked as cheaper_equivalent" in prompt
    assert all(
        fact["classification"] == "cheaper_equivalent"
        for fact in repair_package["critique_facts"]
    )
    assert "not_equivalent_requires_engineering_review" not in json.dumps(
        repair_package["allowed_candidate_alternatives"],
        ensure_ascii=False,
    )


def test_hard_ineligible_platform_cannot_become_repair_cheapest_quote() -> None:
    client = _RepairAwareComposerClient(
        primary_responder=_base_platform_with_samsung_ram_primary_response,
        repair_responder=_blocked_platform_repair_response,
    )

    outcome = compose_llm_configurations(
        user_request="Need cheapest 2 servers 2U Intel Xeon DDR5 NVMe.",
        normalized_requirements=[_repair_requirements_with_hard_platform_dimensions()],
        ready_stock_candidates=[],
        component_candidate_matrix=_matrix_with_hard_ineligible_platform_and_ram_repair(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]
    repair_package_text = json.dumps(client.repair_packages[0], ensure_ascii=False)
    blocked_text = " ".join(outcome.repair_blocked_critique_summary)

    assert outcome.repair_attempted is True
    assert outcome.repair_success is False
    assert outcome.repair_fallback_reason == "llm_repair_revised_no_valid_recommendations"
    assert recommendation["component_candidate_ids"]["platform"] == "platform-base-eligible"
    assert "platform-blocked-incomplete" not in repair_package_text
    assert "INCOMPLETE-CHASSIS" not in json.dumps(outcome.recommended_builds)
    assert "platform_incomplete_chassis" in blocked_text


def test_gooxi_cheapest_platform_is_not_penalized_and_brand_stays_alternative() -> None:
    outcome = compose_llm_configurations(
        user_request="Need cheapest quote with branded alternative.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_gooxi_and_supermicro(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_gooxi_and_supermicro_response),
        settings=_llm_composer_settings(),
    )

    group = outcome.configuration_groups[0]
    options = group["platform_options"]

    assert outcome.used is True
    assert options[0]["role"] == "cheapest_quote"
    assert options[0]["platform"]["producer"] == "Gooxi"
    assert any(
        option["role"] in {"branded_safe", "preferred_for_database", "engineering_clear"}
        and option["platform"]["producer"] == "Supermicro"
        for option in options
    )
    assert "Gooxi" in outcome.quote_recommendation["for_cheapest_quote"]
    assert "Supermicro" not in outcome.quote_recommendation["for_cheapest_quote"]


def test_grouped_presales_sanitizes_cjk_and_deduplicates_engineer_checks() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 512 ГБ RAM DDR5, 2 SSD NVMe.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_cjk_and_duplicate_checks_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    grouped_text = json.dumps(outcome.configuration_groups, ensure_ascii=False)
    assert not contains_cjk_text(grouped_text)

    option = outcome.configuration_groups[0]["platform_options"][0]
    assert "высокоплот" in option["why_this_platform"]
    assert option["engineering_confidence_code"] == "preliminary_requires_engineer_review"
    assert option["engineering_confidence"] == "предварительно, нужна инженерная проверка"
    assert "component_candidate_id" not in grouped_text
    assert "raw JSON" not in grouped_text
    assert "preliminary_requires_engineer_review" in grouped_text

    checks = option["engineer_checks"]
    assert checks.count("Проверить CPU support list платформы и версию BIOS.") == 1
    assert checks.count("Проверить QVL RAM и правила заполнения DIMM.") == 1
    assert len(checks) == len(set(checks))


def test_grouped_presales_titles_distinguish_ram_and_storage_tradeoffs() -> None:
    outcome = compose_llm_configurations(
        user_request=(
            "Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, "
            "2 SSD NVMe не менее 3.84 ТБ."
        ),
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_ram_and_storage_tradeoffs(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_ram_and_storage_tradeoff_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert len(outcome.configuration_groups) == 3

    titles = [group["group_title"] for group in outcome.configuration_groups]
    assert len(titles) == len(set(titles))
    assert "минимальная базовая конфигурация" in titles[0]
    assert "база на 32 ГБ DIMM" in titles[0]

    title_by_ram: dict[int, str] = {}
    title_by_storage: dict[float, str] = {}
    for group in outcome.configuration_groups:
        base = group["component_base"]
        title_by_ram[base["ram"]["ram_module_capacity_gb"]] = group["group_title"]
        title_by_storage[base["storage"]["storage_capacity_tb"]] = group["group_title"]

    assert "вариант с 64 ГБ DIMM" in title_by_ram[64]
    assert "компромисс по RAM/DIMM-слотам" in title_by_ram[64]
    assert "вариант с SSD 7.68 ТБ" in title_by_storage[7.68]
    assert "компромисс по емкости накопителей" in title_by_storage[7.68]
    assert "не являются отдельной обязательной базой" in outcome.quote_recommendation[
        "summary"
    ]


@pytest.mark.parametrize(
    ("server_quantity", "ram_module_gb", "expected_ram_quantity"),
    [
        (2, 32, 32),
        (2, 64, 16),
        (1, 32, 16),
    ],
)
def test_llm_bom_materializer_calculates_ram_quantity_from_requirements(
    server_quantity: int,
    ram_module_gb: int,
    expected_ram_quantity: int,
) -> None:
    requirements = _composer_requirements()
    requirements["server_qty"] = server_quantity
    requirements["total_cpu_required"] = server_quantity * requirements["cpu_per_server"]
    requirements["storage_qty_per_server"] = 2
    matrix = _composer_component_matrix_with_ram_module(
        ram_module_gb,
        available_quantity=max(expected_ram_quantity, 100),
    )

    outcome = compose_llm_configurations(
        user_request="Нужно подобрать серверы: 2 CPU, 512 ГБ RAM DDR5, 2 SSD NVMe.",
        normalized_requirements=[requirements],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_double_ram_quantity_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]
    ram_component = _first_component(recommendation, "ram")

    assert outcome.used is True
    assert recommendation["quantities"]["ram"] == expected_ram_quantity
    assert ram_component["quantity_required"] == expected_ram_quantity
    assert ram_component["per_server_quantity"] == expected_ram_quantity // server_quantity
    assert ram_component["ram_module_capacity_gb"] == ram_module_gb
    assert ram_component["ram_total_gb_per_server"] == 512


def test_llm_bom_materializer_normalizes_ssd_quantity_per_server() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера, на каждый 2 SSD NVMe не менее 3.84 ТБ.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_underselected_ssd_quantity_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]
    ssd_component = _first_component(recommendation, "ssd")

    assert outcome.used is True
    assert recommendation["quantities"]["ssd"] == 4
    assert ssd_component["quantity_required"] == 4
    assert ssd_component["per_server_quantity"] == 2


def test_llm_double_ram_quantity_is_not_shown_in_user_facing_bom() -> None:
    outcome = compose_llm_configurations(
        user_request=(
            "Нужно подобрать 2 сервера 2U под базу данных. На каждый сервер: "
            "2 процессора Intel Xeon, 512 ГБ RAM DDR5, 2 SSD NVMe 3.84 ТБ."
        ),
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_ram_module(
            32,
            available_quantity=100,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_double_ram_quantity_llm_response),
        settings=_llm_composer_settings(),
    )

    text = format_match_summary(
        {
            "match_run_id": 101,
            "llm_configurator_enabled": True,
            "llm_configurator_used": outcome.used,
            "ai_recommendations": outcome.recommended_builds,
        }
    )

    assert "16 x 32 ГБ на сервер" in text
    assert "32 шт. всего" in text
    assert "64 шт. всего" not in text


def test_llm_build_with_unknown_ram_capacity_is_rejected_as_unsafe() -> None:
    matrix = _composer_component_matrix_with_ram_module(
        32,
        available_quantity=100,
    )
    matrix["ram_candidates"][0]["ram_module_capacity_gb"] = None
    matrix["ram_candidates"][0]["extracted_facts"] = {"ram_type": "DDR5"}

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера, 512 ГБ RAM DDR5 на сервер.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is False
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert outcome.validation_rejected_count == 1
    assert "RAM module capacity is unknown" in " ".join(outcome.internal_warnings)


def test_online_composer_uses_web_evidence_model_and_materializes_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingOpenAIComposerClient.reset(_online_composer_llm_response)
    monkeypatch.setattr(
        configuration_composer_module,
        "OpenAICompatibleLlmClient",
        _RecordingOpenAIComposerClient,
    )
    web_settings = _routerai_web_evidence_settings().model_copy(
        update={
            "web_evidence_mode": "online_composer",
            "web_evidence_base_url": "https://evidence.example.test/v1",
            "web_evidence_api_key": "evidence-key",
            "web_evidence_model": "online-model",
            "web_evidence_max_output_tokens": 20000,
        }
    )
    provider = _FakeRouterAIEvidenceProvider(
        {},
        relation_by_type={
            "platform_cpu": {
                "status": "confirmed",
                "confirmed_facts": ["CPU support list includes selected CPU"],
                "domain": "asus.com",
            },
            "platform_ram": {
                "status": "partially_confirmed",
                "confirmed_facts": ["DDR5 RDIMM"],
                "missing_evidence": ["Memory QVL requires engineer review"],
                "domain": "asus.com",
            },
            "platform_storage": {
                "status": "partially_confirmed",
                "confirmed_facts": ["NVMe backplane"],
                "missing_evidence": ["Drive support list requires engineer review"],
                "domain": "asus.com",
            },
            "build_sanity": {
                "status": "partially_confirmed",
                "confirmed_facts": ["2U DDR5 NVMe platform"],
                "missing_evidence": ["Whole build requires engineer review"],
                "domain": "asus.com",
            },
        },
    )

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        settings=_llm_composer_settings(),
        web_evidence_settings=web_settings,
        web_search_provider=provider,
    )

    recommendation = outcome.recommended_builds[0]
    client = _RecordingOpenAIComposerClient.instances[0]

    assert outcome.used is True
    assert client.settings.llm_model == "online-model"
    assert client.settings.llm_base_url == "https://evidence.example.test/v1"
    assert client.settings.llm_api_key == "evidence-key"
    assert client.kwargs["use_response_format"] is False
    assert client.kwargs["max_output_tokens"] == 65536
    assert "Online Composer V1" in client.system_prompt
    assert "proposal_pool_limit" in client.system_prompt
    assert "2-4" not in client.system_prompt
    assert recommendation["component_candidate_ids"]["platform"] == "platform-amd"
    assert recommendation["evidence_summary"]["status"] in {
        "confirmed",
        "partially_confirmed",
    }
    assert recommendation["evidence_summary"]["sources_count"] >= 4
    assert recommendation["evidence_summary"]["relation_evidence_count"] >= 4
    diagnostics = outcome.evidence_pack["diagnostics"]
    assert diagnostics["evidence_mode"] == "online_composer"
    assert diagnostics["online_composer_used"] is True
    assert diagnostics["evidence_sources_count"] >= 4
    assert diagnostics["evidence_requests_count"] == 2
    assert provider.requests_count == 1
    assert {task.target_type for task in provider.tasks} == {
        "relation_platform_cpu",
        "relation_platform_ram",
        "relation_platform_storage",
        "relation_build_sanity",
    }


def test_online_composer_zero_sources_triggers_posthoc_relation_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingOpenAIComposerClient.reset(_online_composer_no_sources_llm_response)
    monkeypatch.setattr(
        configuration_composer_module,
        "OpenAICompatibleLlmClient",
        _RecordingOpenAIComposerClient,
    )
    provider = _FakeRouterAIEvidenceProvider(
        {},
        relation_by_type={
            "platform_cpu": {
                "status": "partially_confirmed",
                "confirmed_facts": ["LGA4677 CPU family"],
                "missing_evidence": ["CPU support list requires engineer review"],
                "domain": "asus.com",
            },
            "platform_ram": {
                "status": "partially_confirmed",
                "confirmed_facts": ["DDR5 RDIMM"],
                "missing_evidence": ["Memory QVL requires engineer review"],
                "domain": "asus.com",
            },
            "platform_storage": {
                "status": "partially_confirmed",
                "confirmed_facts": ["NVMe backplane"],
                "missing_evidence": ["Drive support list requires engineer review"],
                "domain": "asus.com",
            },
            "build_sanity": {
                "status": "partially_confirmed",
                "confirmed_facts": ["2U DDR5 NVMe platform"],
                "missing_evidence": ["Whole build requires engineer review"],
                "domain": "asus.com",
            },
        },
    )
    web_settings = _routerai_web_evidence_settings().model_copy(
        update={"web_evidence_mode": "online_composer"}
    )

    outcome = compose_llm_configurations(
        user_request="Need selected relation evidence after online composer.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        settings=_llm_composer_settings(),
        web_evidence_settings=web_settings,
        web_search_provider=provider,
    )

    assert outcome.used is True
    assert provider.requests_count == 1
    assert all(str(task.target_type).startswith("relation_") for task in provider.tasks)
    assert {task.role for task in provider.tasks} == {
        "platform_cpu",
        "platform_ram",
        "platform_storage",
        "build_sanity",
    }
    diagnostics = outcome.evidence_pack["diagnostics"]
    assert diagnostics["relation_evidence_count"] == 4
    assert diagnostics["evidence_tasks_count_by_type"] == {
        "relation_platform_cpu": 1,
        "relation_platform_ram": 1,
        "relation_platform_storage": 1,
        "relation_build_sanity": 1,
    }
    recommendation = outcome.recommended_builds[0]
    summary = recommendation["evidence_summary"]
    assert summary["status"] == "partially_confirmed"
    assert summary["relation_evidence_count"] == 4
    assert summary["sources_count"] == 4
    assert summary["source_domains"] == ["asus.com"]

    telegram_text = format_match_summary(
        {
            "match_run_id": 40,
            "llm_configurator_enabled": True,
            "llm_configurator_used": True,
            "ai_recommendations_count": 1,
            "ai_recommendations": outcome.recommended_builds,
            "web_evidence_pack": outcome.evidence_pack,
            "web_evidence_diagnostics": diagnostics,
            "llm_proposals_count": outcome.proposal_count,
            "rejected_ai_recommendations_count": outcome.rejected_recommendations_count,
        }
    )
    assert "component_candidate_id" not in telegram_text
    assert "raw JSON" not in telegram_text
    assert "Llm_rec_" not in telegram_text
    assert "llm_rec_" not in telegram_text
    assert "web evidence not found" not in telegram_text
    assert "keep engineer" not in telegram_text


def test_online_composer_posthoc_relation_mismatch_rejects_selected_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingOpenAIComposerClient.reset(_online_composer_no_sources_llm_response)
    monkeypatch.setattr(
        configuration_composer_module,
        "OpenAICompatibleLlmClient",
        _RecordingOpenAIComposerClient,
    )
    provider = _FakeRouterAIEvidenceProvider(
        {},
        relation_by_type={
            "platform_cpu": {
                "status": "mismatch",
                "mismatch_facts": ["CPU is not on the platform support list"],
                "domain": "asus.com",
            }
        },
    )

    outcome = compose_llm_configurations(
        user_request="Need selected relation evidence after online composer.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        settings=_llm_composer_settings(),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_mode": "online_composer"}
        ),
        web_search_provider=provider,
    )

    assert outcome.used is False
    assert outcome.validation_summary["rejected_fatal"] == 1
    assert outcome.evidence_pack["diagnostics"]["relation_mismatch_count"] == 1
    assert "relation_platform_cpu mismatch" in " ".join(outcome.internal_warnings)


def test_online_composer_fallback_runs_posthoc_relation_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RecordingOpenAIComposerClient.reset(
        _component_matrix_llm_response,
        failing_models={"online-model"},
    )
    monkeypatch.setattr(
        configuration_composer_module,
        "OpenAICompatibleLlmClient",
        _RecordingOpenAIComposerClient,
    )
    provider = _FailingEvidenceProvider()
    web_settings = _routerai_web_evidence_settings().model_copy(
        update={
            "web_evidence_mode": "online_composer",
            "web_evidence_model": "online-model",
        }
    )

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        settings=_llm_composer_settings(),
        web_evidence_settings=web_settings,
        web_search_provider=provider,
    )

    models = [client.settings.llm_model for client in _RecordingOpenAIComposerClient.instances]
    diagnostics = outcome.evidence_pack["diagnostics"]

    assert outcome.used is True
    assert models == ["online-model", "test-model"]
    assert provider.requests_count == 1
    assert diagnostics["evidence_mode"] == "online_composer"
    assert diagnostics["online_composer_used"] is True
    assert diagnostics["online_composer_error_type"] == "LlmClientError"
    assert diagnostics["evidence_requests_count"] == 2
    assert diagnostics["evidence_sources_count"] == 0
    assert diagnostics["relation_evidence_count"] == 4


def test_llm_configurator_rejects_unknown_component_candidate_id() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_unknown_component_id_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is False
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert any("unknown component_candidate_id" in warning for warning in outcome.internal_warnings)
    assert outcome.validation_summary["rejected_unknown_component"] == 1
    assert outcome.rejected_reasons_top[0]["reason"] == "unknown_component"
    debug = outcome.rejected_recommendations_debug_safe[0]
    assert debug["rejection_code"] == "unknown_component"
    assert debug["unknown_component_ids"] == ["missing-component-id"]
    assert debug["proposal_index"] == 0
    assert "component_candidate_matrix" not in json.dumps(debug, ensure_ascii=False)


def test_llm_configurator_rejects_component_candidate_id_with_wrong_role() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_wrong_component_role_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is False
    assert any("component role mismatch" in warning for warning in outcome.internal_warnings)


def test_llm_configurator_missing_ram_downgrades_to_partial_build() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера, 512 ГБ RAM DDR5.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_missing_ram_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert recommendation["source_type"] == "partial_build"
    assert recommendation["candidate_type"] == "partial_build"
    assert recommendation["completeness_status"] == "incomplete"
    assert "ram" in recommendation["missing_component_roles"]
    assert recommendation["total_price_note"] == "без RAM"


def test_llm_rejected_missing_platform_has_missing_required_role_debug() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_missing_platform_llm_response),
        settings=_llm_composer_settings(),
    )

    debug = outcome.rejected_recommendations_debug_safe[0]

    assert outcome.used is False
    assert debug["rejection_code"] == "missing_required_role"
    assert "server_platform" in debug["missing_roles"]
    assert outcome.rejected_reasons_top[0]["reason"] == "missing_required_role"


def test_llm_rejected_stock_shortage_has_safe_debug() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_ram_module(
            32,
            available_quantity=10,
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    debug = outcome.rejected_recommendations_debug_safe[0]

    assert outcome.used is False
    assert debug["rejection_code"] == "stock_shortage"
    assert debug["stock_shortages"]
    assert debug["stock_shortages"][0]["role"] == "ram"
    assert outcome.rejected_reasons_top[0]["reason"] == "stock_shortage"


def test_llm_materializer_exception_is_safe_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_materializer(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom api_key=super-secret-llm-token")

    monkeypatch.setattr(
        configuration_composer_module,
        "_materialize_build_quantities",
        raise_materializer,
    )

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    debug = outcome.rejected_recommendations_debug_safe[0]
    debug_text = json.dumps(debug, ensure_ascii=False)

    assert outcome.used is False
    assert debug["exception_type"] == "RuntimeError"
    assert debug["stage"] == "validate_exception"
    assert "boom" in debug["exception_message_sanitized"]
    assert "super-secret-llm-token" not in debug_text
    assert "api_key" not in debug_text


def test_llm_configurator_keeps_valid_recommendation_when_another_is_invalid() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_one_invalid_one_valid_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert len(outcome.recommended_builds) == 1
    assert outcome.recommended_builds[0]["recommendation_id"] == "llm_matrix_build"
    assert outcome.rejected_recommendations_count == 1


def test_llm_configurator_keeps_stable_valid_recommendation_order() -> None:
    kwargs = {
        "user_request": "Нужно 2 сервера.",
        "normalized_requirements": [_composer_requirements()],
        "ready_stock_candidates": [],
        "component_candidate_matrix": _composer_component_matrix_with_alternatives(),
        "rule_based_build_candidates": [],
        "llm_client": _FakeComposerClient(_two_slot_llm_response),
        "settings": _llm_composer_settings(),
    }

    first = compose_llm_configurations(**kwargs)
    second = compose_llm_configurations(
        **{**kwargs, "llm_client": _FakeComposerClient(_two_slot_llm_response)}
    )

    assert [row["recommendation_id"] for row in first.recommended_builds] == [
        "llm_price",
        "llm_technical",
    ]
    assert [row["recommendation_id"] for row in second.recommended_builds] == [
        "llm_price",
        "llm_technical",
    ]
    assert first.recommended_builds[0]["recommendation_slot"] == "price_optimal"
    assert first.recommended_builds[1]["recommendation_slot"] == "technical_clean"


def test_llm_proposal_pool_validates_ten_and_displays_deterministic_top_three() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера 2U, 2 CPU, 512 ГБ RAM DDR5, SSD.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_proposal_pool_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.proposal_count == 10
    assert len(outcome.recommended_builds) == 3
    assert outcome.rejected_recommendations_count == 7
    assert outcome.validation_summary["accepted"] == 3
    assert outcome.validation_summary["rejected_unknown_component"] == 7
    assert [row["recommendation_id"] for row in outcome.recommended_builds] == [
        "llm_pool_price",
        "llm_pool_technical",
        "llm_pool_alternative",
    ]
    assert [row["recommendation_slot"] for row in outcome.recommended_builds] == [
        "price_optimal",
        "technical_clean",
        "alternative",
    ]
    assert outcome.recommended_builds[0]["title"] == "Оптимальный по цене вариант"
    assert outcome.recommended_builds[0]["total_price_value"] == "8400"
    assert outcome.recommended_builds[1]["title"] == "Технически более чистый вариант"
    assert outcome.recommended_builds[2]["platform"]["part_number"] == "R283-ZK0"
    assert "user_request" not in json.dumps(outcome.rejected_reasons_top, ensure_ascii=False)
    assert "test-key" not in json.dumps(outcome.validation_summary, ensure_ascii=False)


def test_llm_proposal_pool_one_safe_keeps_ai_and_telegram_explains_rejections() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_one_safe_proposal_pool_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert len(outcome.recommended_builds) == 1
    assert outcome.proposal_count == 10
    assert outcome.rejected_recommendations_count == 9

    text = format_match_summary(
        {
            "match_run_id": 77,
            "llm_configurator_enabled": True,
            "llm_configurator_used": outcome.used,
            "ai_recommendations": outcome.recommended_builds,
            "llm_proposals_count": outcome.proposal_count,
            "rejected_ai_recommendations_count": outcome.rejected_recommendations_count,
            "ai_validation_summary": outcome.validation_summary,
        }
    )

    assert "Показан 1 безопасный вариант." in text
    assert "Остальные AI-варианты были отклонены валидатором" in text
    assert "Готовые варианты" not in text
    assert "\nСборка из комплектующих\n" not in text


def test_llm_proposal_pool_all_invalid_returns_safe_no_recommendation_mode() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_invalid_proposal_pool_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is False
    assert outcome.recommended_builds == []
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert outcome.proposal_count == 10
    assert outcome.valid_proposals_count == 0
    assert outcome.validation_rejected_count == 10
    assert outcome.selection_skipped_count == 0
    assert outcome.rejected_recommendations_count == 10
    assert outcome.validation_summary["rejected_unknown_component"] == 10

    text = format_match_summary(
        {
            "match_run_id": 78,
            "llm_configurator_enabled": True,
            "llm_configurator_used": False,
            "llm_fallback_reason": outcome.fallback_reason,
            "ai_recommendation_mode": "ai_no_safe_recommendations",
        }
    )
    assert "Безопасную складскую рекомендацию дать нельзя" in text
    assert "Готовые варианты" not in text


def test_llm_proposal_pool_limit_is_sent_greater_than_display_limit() -> None:
    client = _FakeComposerClient(_component_matrix_llm_response)
    settings = _llm_composer_settings().model_copy(
        update={
            "llm_proposal_pool_limit": 12,
            "llm_build_recommendations_limit": 2,
        }
    )

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=client,
        settings=settings,
    )

    assert outcome.used is True
    assert client.package["proposal_pool_limit"] == 12
    assert client.package["final_display_limit"] == 2
    assert client.package["proposal_pool_limit"] > client.package["final_display_limit"]


def test_llm_proposal_pool_display_limit_caps_valid_safe_recommendations() -> None:
    settings = _llm_composer_settings().model_copy(
        update={"llm_build_recommendations_limit": 2}
    )

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_proposal_pool_llm_response),
        settings=settings,
    )

    assert outcome.used is True
    assert [row["recommendation_id"] for row in outcome.recommended_builds] == [
        "llm_pool_price",
        "llm_pool_technical",
    ]
    assert outcome.rejected_recommendations_count == 8
    assert outcome.selection_skipped_count == 1


def test_llm_configurator_selection_keeps_one_when_all_proposals_duplicate() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_duplicate_five_bom_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert len(outcome.recommended_builds) == 1
    assert outcome.fallback_reason is None
    assert outcome.proposal_count == 5
    assert outcome.valid_proposals_count == 5
    assert outcome.validation_rejected_count == 0
    assert outcome.selection_skipped_count == 4
    assert outcome.validation_summary["accepted_after_validation"] == 5
    assert outcome.validation_summary["selection_skipped_duplicate"] == 4
    assert any("duplicate" in warning.casefold() for warning in outcome.internal_warnings)


def test_llm_configurator_selection_keeps_cheapest_when_others_worse_by_price() -> None:
    matrix = _composer_component_matrix_with_alternatives()
    matrix["cpu_candidates"].append(
        _composer_component_candidate(
            "cpu-32",
            "Intel",
            "CPU-32C",
            "Intel Xeon 32 core tray processor",
            4,
            Decimal("900"),
            {
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 32,
            },
        )
    )
    matrix["cpu_candidates"][-1]["fit_label"] = "acceptable_overfit"
    matrix["cpu_candidates"][-1]["cpu_over_requirement"] = 16

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера с Intel Xeon не менее 16 ядер.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_worse_by_price_pool_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.fallback_reason is None
    assert [row["recommendation_id"] for row in outcome.recommended_builds] == [
        "llm_worse_cheapest"
    ]
    assert outcome.valid_proposals_count == 4
    assert outcome.validation_rejected_count == 0
    assert outcome.selection_skipped_count == 3
    assert (
        outcome.validation_summary["selection_skipped_dominated_by_cheaper_equivalent"]
        == 3
    )


def test_llm_configurator_diagnostics_separate_fatal_and_duplicate_selection() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_mixed_invalid_duplicate_pool_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.fallback_reason is None
    assert len(outcome.recommended_builds) == 1
    assert outcome.proposal_count == 5
    assert outcome.valid_proposals_count == 3
    assert outcome.validation_rejected_count == 2
    assert outcome.selection_skipped_count == 2
    assert outcome.validation_summary["rejected_unknown_component"] == 2
    assert outcome.validation_summary["selection_skipped_duplicate"] == 2


def test_llm_configurator_rejects_intel_cpu_for_amd_sp5_platform() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужны 2 сервера с Intel Xeon и DDR5 RAM.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "AMD",
                "cpu_family": "EPYC",
                "cpu_socket": "SP5",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA3647",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[_composer_build_candidate()],
        llm_client=_FakeComposerClient(_fatal_pair_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is False
    assert outcome.recommended_builds == []
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert outcome.rejected_recommendations_count == 1
    assert any("fatal compatibility mismatch" in warning for warning in outcome.internal_warnings)


def test_86_like_lga1700_platform_lga3647_cpu_bom_is_rejected() -> None:
    outcome = compose_llm_configurations(
        user_request="Need #86-like server BOM; reject obvious socket mismatch.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Core",
                "cpu_socket": "LGA1700",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA3647",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[_composer_build_candidate()],
        llm_client=_FakeComposerClient(_fatal_pair_llm_response),
        settings=_llm_composer_settings(output_mode="single_best_cost_valid"),
    )

    assert outcome.used is False
    assert outcome.primary_recommendation_status == "no_recommendation"
    assert outcome.final_status_source == "composer_rejected"
    assert outcome.validation_summary["rejected_platform_cpu_mismatch"] == 1
    assert any("socket mismatch" in warning for warning in outcome.internal_warnings)


def test_llm_configurator_rejects_amd_epyc_cpu_for_intel_lga4677_platform() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужна серверная сборка.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "AMD",
                "cpu_family": "EPYC",
                "cpu_socket": "SP5",
                "cpu_cores": 32,
            },
        ),
        rule_based_build_candidates=[_composer_build_candidate()],
        llm_client=_FakeComposerClient(_fatal_pair_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is False
    assert outcome.recommended_builds == []
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert any("fatal compatibility mismatch" in warning for warning in outcome.internal_warnings)


def test_web_evidence_rejects_dell_r750xs_with_xeon_6_cpu() -> None:
    matrix = _composer_component_matrix(
        platform_facts={"normalized_vendor": "Dell"},
        cpu_facts={"cpu_brand": "Intel", "cpu_family": "Xeon"},
    )
    matrix["platform_candidates"][0].update(
        {
            "producer": "Dell",
            "part_number": "R750XS",
            "name": "Dell PowerEdge R750xs server platform",
        }
    )
    matrix["cpu_candidates"][0].update(
        {
            "producer": "Intel",
            "part_number": "6527P",
            "name": "Intel Xeon 6 6527P processor",
        }
    )
    provider = FakeWebSearchProvider(
        {
            "R750XS": [
                {
                    "title": "Dell PowerEdge R750xs Spec Sheet",
                    "url": "https://i.dell.com/sites/doccontent/shared-content/data-sheets/en/Documents/poweredge-r750xs-spec-sheet.pdf",
                    "snippet": (
                        "PowerEdge R750xs supports 3rd Gen Intel Xeon Scalable "
                        "processors, DDR4 memory and LGA4189 platform."
                    ),
                }
            ],
            "6527P": [
                {
                    "title": "Intel Xeon 6 6527P processor specifications",
                    "url": "https://ark.intel.com/content/www/us/en/ark/products/6527p.html",
                    "snippet": "Intel Xeon 6 processor 6527P uses FCLGA4710 socket.",
                }
            ],
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need Dell R750xs with Xeon 6.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_web_evidence_settings(),
        web_search_provider=provider,
    )

    assert outcome.used is False
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert outcome.validation_summary["rejected_fatal"] == 1
    assert outcome.evidence_pack["completed_tasks"] >= 2
    assert "test-key" not in json.dumps(outcome.evidence_pack, ensure_ascii=False)


def test_web_evidence_keeps_asus_lga4677_with_engineering_check() -> None:
    matrix = _composer_component_matrix(
        platform_facts={"normalized_vendor": "ASUS"},
        cpu_facts={"cpu_brand": "Intel", "cpu_family": "Xeon"},
    )
    matrix["platform_candidates"][0].update(
        {
            "producer": "ASUS",
            "part_number": "RS720-E11-RS24U",
            "name": "ASUS RS720-E11-RS24U server platform",
        }
    )
    matrix["cpu_candidates"][0].update(
        {
            "producer": "Intel",
            "part_number": "5416S",
            "name": "Intel Xeon Gold 5416S processor",
        }
    )
    provider = FakeWebSearchProvider(
        {
            "RS720-E11-RS24U": [
                {
                    "title": "ASUS RS720-E11-RS24U specifications",
                    "url": "https://servers.asus.com/products/Servers/Rack-Servers/RS720-E11-RS24U",
                    "snippet": (
                        "ASUS RS720-E11-RS24U supports LGA4677 4th/5th Gen "
                        "Intel Xeon Scalable processors, DDR5 and NVMe bays."
                    ),
                }
            ],
            "5416S": [
                {
                    "title": "Intel Xeon Gold 5416S specifications",
                    "url": "https://ark.intel.com/content/www/us/en/ark/products/5416s.html",
                    "snippet": "4th Gen Intel Xeon Scalable processor, FCLGA4677 socket, 16 cores.",
                }
            ],
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need ASUS LGA4677 server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_web_evidence_settings(),
        web_search_provider=provider,
    )

    assert outcome.used is True
    recommendation = outcome.recommended_builds[0]
    assert recommendation["evidence_summary"]["sources_count"] >= 2
    assert recommendation["evidence_summary"]["confidence"] in {"high", "medium"}
    assert recommendation["engineer_review_required"] is True


def test_web_evidence_not_found_warns_without_rejecting() -> None:
    outcome = compose_llm_configurations(
        user_request="Need server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_web_evidence_settings(),
        web_search_provider=FakeWebSearchProvider({}),
    )

    assert outcome.used is True
    assert outcome.evidence_pack["completed_tasks"] == 0
    assert outcome.recommended_builds[0]["confidence"] == "low"
    assert outcome.recommended_builds[0]["confidence_score"] < 75
    assert any("Доказательная проверка" in warning for warning in outcome.internal_warnings)
    assert "web evidence not found" not in json.dumps(
        outcome.recommended_builds,
        ensure_ascii=False,
    )
    assert "keep engineer" not in json.dumps(outcome.recommended_builds, ensure_ascii=False)


def test_relation_evidence_confirmed_keeps_recommendation() -> None:
    provider = _FakeRouterAIEvidenceProvider(
        _full_component_evidence_rows(),
        relation_by_type={
            "platform_cpu": {
                "status": "confirmed",
                "confirmed_facts": ["CPU support list includes Xeon Gold 6530"],
                "domain": "servers.asus.com",
            },
            "platform_ram": {
                "status": "partially_confirmed",
                "confirmed_facts": ["DDR5 RDIMM"],
                "missing_evidence": ["QVL row still requires engineer review"],
                "domain": "servers.asus.com",
            },
            "platform_storage": {
                "status": "partially_confirmed",
                "confirmed_facts": ["NVMe U.3 backplane"],
                "missing_evidence": ["Drive support list still requires engineer review"],
                "domain": "servers.asus.com",
            },
            "build_sanity": {
                "status": "partially_confirmed",
                "confirmed_facts": ["2U DDR5 NVMe platform"],
                "missing_evidence": ["Whole build requires engineer review"],
                "domain": "servers.asus.com",
            },
        },
    )

    outcome = compose_llm_configurations(
        user_request="Need ASUS relation evidence.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_max_queries": 20}
        ),
        web_search_provider=provider,
    )

    assert outcome.used is True
    recommendation = outcome.recommended_builds[0]
    assert recommendation["evidence_summary"]["status"] in {
        "confirmed",
        "partially_confirmed",
    }
    assert recommendation["evidence_summary"]["relation_evidence_count"] >= 3
    assert recommendation["evidence_summary"]["sources_count"] >= 4


def test_relation_evidence_mismatch_rejects_recommendation() -> None:
    provider = _FakeRouterAIEvidenceProvider(
        _full_component_evidence_rows(),
        relation_by_type={
            "platform_cpu": {
                "status": "mismatch",
                "mismatch_facts": ["CPU is not on the platform support list"],
                "domain": "servers.asus.com",
            }
        },
    )

    outcome = compose_llm_configurations(
        user_request="Need ASUS relation evidence.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_max_queries": 20}
        ),
        web_search_provider=provider,
    )

    assert outcome.used is False
    assert outcome.validation_summary["rejected_fatal"] == 1
    assert "relation_platform_cpu mismatch" in " ".join(outcome.internal_warnings)


def test_relation_evidence_not_confirmed_warns_without_rejecting() -> None:
    provider = _FakeRouterAIEvidenceProvider(
        _full_component_evidence_rows(),
        relation_by_type={
            "platform_cpu": {
                "status": "not_confirmed",
                "missing_evidence": ["CPU support list not found"],
                "sources": [],
            }
        },
    )

    outcome = compose_llm_configurations(
        user_request="Need ASUS relation evidence.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={"cpu_socket": "LGA4677", "ram_type": "DDR5"},
            cpu_facts={"cpu_socket": "LGA4677", "cpu_cores": 32},
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_max_queries": 20}
        ),
        web_search_provider=provider,
    )

    assert outcome.used is True
    recommendation = outcome.recommended_builds[0]
    assert recommendation["evidence_summary"]["status"] == "partially_confirmed"
    assert any("support list" in warning for warning in outcome.internal_warnings)


def test_routerai_evidence_rejects_dell_r750xs_with_xeon_6_cpu() -> None:
    matrix = _composer_component_matrix(
        platform_facts={"normalized_vendor": "Dell"},
        cpu_facts={"cpu_brand": "Intel", "cpu_family": "Xeon"},
    )
    matrix["platform_candidates"][0].update(
        {
            "producer": "Dell",
            "part_number": "R750XS",
            "name": "Dell PowerEdge R750xs server platform",
        }
    )
    matrix["cpu_candidates"][0].update(
        {
            "producer": "Intel",
            "part_number": "6527P",
            "name": "Intel Xeon 6 6527P processor",
        }
    )
    provider = _FakeRouterAIEvidenceProvider(
        {
            "platform-amd": {
                "facts": {
                    "supported_cpu_generation": "3rd Gen Intel Xeon Scalable",
                    "socket_family": "LGA4189",
                    "memory_type": "DDR4",
                },
                "domain": "i.dell.com",
            },
            "cpu-selected": {
                "facts": {
                    "cpu_generation": "Xeon 6",
                    "socket_family": "LGA4710",
                },
                "domain": "ark.intel.com",
            },
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need Dell R750xs with Xeon 6.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_routerai_web_evidence_settings(),
        web_search_provider=provider,
    )

    assert outcome.used is False
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert outcome.validation_summary["rejected_fatal"] == 1
    assert outcome.evidence_pack["provider"] == "routerai"


def test_routerai_evidence_low_confidence_does_not_reject() -> None:
    matrix = _composer_component_matrix(
        platform_facts={"normalized_vendor": "Dell"},
        cpu_facts={"cpu_brand": "Intel", "cpu_family": "Xeon"},
    )
    provider = _FakeRouterAIEvidenceProvider(
        {
            "platform-amd": {
                "facts": {"socket_family": "LGA4189"},
                "confidence": "low",
                "sources": [],
            },
            "cpu-selected": {
                "facts": {"socket_family": "LGA4710"},
                "confidence": "low",
                "sources": [],
            },
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_routerai_web_evidence_settings(),
        web_search_provider=provider,
    )

    assert outcome.used is True
    assert outcome.recommended_builds
    assert outcome.evidence_pack["provider"] == "routerai"


def test_name_facts_reject_ice_lake_platform_with_ddr5_ram() -> None:
    matrix = _composer_component_matrix(platform_facts={}, cpu_facts={})
    matrix["platform_candidates"][0].update(
        {
            "producer": "Gooxi",
            "part_number": "0.95.002.0103",
            "name": "Gooxi 0.95.002.0103 server platform based on Ice lake",
        }
    )
    matrix["cpu_candidates"][0].update(
        {
            "producer": "Intel",
            "part_number": "CPU-ICX",
            "name": "Intel Xeon Ice Lake processor",
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need DDR5 RAM server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is False
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert any(
        "RAM type does not match platform RAM type" in warning
        for warning in outcome.internal_warnings
    )


def test_name_facts_allow_sapphire_rapids_lga4677_with_ddr5_ram() -> None:
    matrix = _composer_component_matrix(platform_facts={}, cpu_facts={})
    matrix["platform_candidates"][0].update(
        {
            "producer": "Gooxi",
            "part_number": "SPR-LGA4677",
            "name": "Gooxi Intel Gen 4th Sapphire Rapids LGA4677 DDR5 platform",
        }
    )
    matrix["cpu_candidates"][0].update(
        {
            "producer": "Intel",
            "part_number": "CPU-SPR",
            "name": "Intel Xeon Gold 5416S Sapphire Rapids processor",
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need DDR5 RAM server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.recommended_builds[0]["engineer_review_required"] is True


def test_name_facts_reject_amd_sp3_platform_with_ddr5_ram() -> None:
    matrix = _composer_component_matrix(platform_facts={}, cpu_facts={})
    matrix["platform_candidates"][0].update(
        {
            "producer": "AMD",
            "part_number": "EPYC-SP3",
            "name": "AMD EPYC 7003 SP3 server platform",
        }
    )
    matrix["cpu_candidates"][0].update(
        {
            "producer": "AMD",
            "part_number": "EPYC-7003",
            "name": "AMD EPYC 7003 processor SP3",
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need DDR5 RAM server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is False
    assert outcome.fallback_reason == "llm_configurator_all_recommendations_rejected"
    assert any(
        "RAM type does not match platform RAM type" in warning
        for warning in outcome.internal_warnings
    )


def test_confirmed_evidence_ranks_above_not_found_evidence() -> None:
    provider = _FakeRouterAIEvidenceProvider(
        {
            "platform-clean": {
                "facts": {
                    "supported_cpu_generation": "4th Gen Intel Xeon Scalable",
                    "socket_family": "LGA4677",
                    "memory_type": "DDR5",
                    "nvme_support": True,
                },
                "domain": "supermicro.com",
            }
        }
    )

    outcome = compose_llm_configurations(
        user_request="Need DDR5 RAM server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(
            _confirmed_and_not_found_evidence_response
        ),
        settings=_llm_composer_settings(),
        web_evidence_settings=_routerai_web_evidence_settings().model_copy(
            update={"web_evidence_max_queries": 20}
        ),
        web_search_provider=provider,
    )

    assert outcome.used is True
    first = outcome.recommended_builds[0]
    assert first["component_candidate_ids"]["platform"] == "platform-clean"
    assert first["evidence_summary"]["sources_count"] >= 1


def test_web_evidence_tasks_use_proposal_components_only() -> None:
    provider = FakeWebSearchProvider({})

    compose_llm_configurations(
        user_request="Need server.",
        normalized_requirements=[_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix_with_alternatives(),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerAndEvidenceReviewClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
        web_evidence_settings=_web_evidence_settings(),
        web_search_provider=provider,
    )

    queries = "\n".join(provider.queries)
    assert "CPU-" in queries
    assert "SYS-621C-TN12R" not in queries
    assert "R283-ZK0" not in queries


def test_evidence_cache_reads_and_writes_sanitized_json() -> None:
    cache_dir = Path(".tmp_pytest") / "evidence_cache_unit" / uuid4().hex
    cache = EvidenceSearchCache(cache_dir=cache_dir, ttl_hours=168)
    result = FakeWebSearchProvider(
        {
            "x": [
                    {
                        "title": "Dell R750xs",
                        "url": "https://i.dell.com/r750xs.pdf",
                        "snippet": "Public datasheet snippet.",
                    }
            ]
        }
    ).search("x", max_results=1, timeout=1)[0]

    cache.set(provider="tavily", query="Dell R750xs datasheet", results=[result])
    cached = cache.get(provider="tavily", query="Dell R750xs datasheet")

    assert cached is not None
    assert cached[0].title == "Dell R750xs"
    cache_text = "\n".join(path.read_text(encoding="utf-8") for path in cache_dir.iterdir())
    assert "secret" not in cache_text
    assert "api_key" not in cache_text


def test_rule_based_build_does_not_pair_amd_sp5_platform_with_intel_xeon_cpu(
    db_session: Session,
) -> None:
    _seed_component_product(
        db_session,
        item_id="amd-sp5-platform",
        part_number="RS521A-E12-RS24U",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS RS521A-E12-RS24U 2U AMD EPYC SP5 LGA6096 DDR5 platform 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="intel-lga3647-cpu",
        part_number="Xeon-5220R",
        producer="Intel",
        category_id="V110103",
        item_name="Intel Xeon Gold 5220R 24 core LGA3647 processor",
        quantity=4,
        price=Decimal("500"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-ddr5",
        part_number="RAM-DDR5-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-3840",
        part_number="SSD-3840",
        producer="KIOXIA",
        category_id="V110106",
        item_name="KIOXIA CD8-R Server SSD NVMe 3.84TB U.2",
        quantity=4,
        price=Decimal("300"),
    )

    result = asyncio.run(match_stock_spec(_right_size_server_spec(), _adapter(db_session)))
    build = result.to_report_json()["build_candidates"][0]

    assert build["candidate_type"] == "build_from_parts"
    assert build["completeness_status"] == "incomplete"
    assert "cpu" in build["missing_component_roles"]
    assert not any(component["role"] == "cpu" for component in build["components"])


def test_llm_configurator_can_recommend_ready_server_by_source_candidate_id(
    db_session: Session,
) -> None:
    _seed_nerpa_product(
        db_session,
        item_id="ready-1",
        part_number="READY-2U",
        item_name="Server 2U 2x CPU SSD 2x PSU 2x DDR4 64GB",
        quantity=3,
        can_reserve=True,
    )

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=64),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(_ready_server_llm_response),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()
    recommendation = report_json["llm_recommendations"][0]

    assert report_json["llm_configurator_used"] is True
    assert report_json["ready_stock_candidates"]
    assert report_json["component_candidate_matrix"]["ready_server_candidates"] == []
    assert recommendation["source_type"] == "ready_server"
    assert recommendation["source_candidate_id"] in {
        candidate["candidate_id"] for candidate in report_json["ready_stock_candidates"]
    }
    assert recommendation["total_price_value"] == "13800"
    assert recommendation["total_price_currency"] == "USD"


def test_llm_configurator_rejects_ready_server_with_serious_gaps(
    db_session: Session,
) -> None:
    _seed_nerpa_product(
        db_session,
        item_id="ready-gap",
        part_number="READY-GAP",
        item_name="Server 2U 2x CPU SSD 2x PSU 2x DDR4 64GB",
        quantity=3,
        can_reserve=True,
    )

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(_ready_server_llm_response),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()

    assert report_json["llm_configurator_used"] is False
    assert report_json["llm_recommendations"] == []
    assert (
        report_json["llm_fallback_reason"]
        == "llm_configurator_all_recommendations_rejected"
    )
    assert any("serious gaps" in warning for warning in report_json["llm_internal_warnings"])


def test_llm_configurator_keeps_excessive_storage_overfit_when_it_is_only_valid_proposal(
    db_session: Session,
) -> None:
    _seed_right_size_base_components(db_session)
    for item_id, part_number, item_name, price in [
        ("ssd-3840", "SSD-3840", "Server SSD NVMe 3.84TB U.2", "300"),
        ("ssd-15360", "SSD-15360", "Server SSD NVMe 15.36TB U.2", "900"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="Samsung",
            category_id="V110106",
            item_name=item_name,
            quantity=4,
            price=Decimal(price),
        )

    result = asyncio.run(
        match_stock_spec(
            _right_size_server_spec(),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(_excessive_storage_llm_response),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()

    assert report_json["llm_configurator_used"] is True
    assert len(report_json["llm_recommended_build_candidates"]) == 1
    assert report_json["valid_proposals_count"] == 1
    assert report_json["validation_rejected_count"] == 0
    assert report_json["selection_skipped_count"] == 0
    assert any(
        "excessive overfit rejected" in warning
        for warning in report_json["llm_internal_warnings"]
    )
    assert _first_component(report_json["build_candidates"][0], "ssd")["part_number"] == "SSD-3840"


def test_llm_configurator_accepts_overfit_only_with_validator_reason(
    db_session: Session,
) -> None:
    _seed_right_size_base_components(db_session)
    _seed_component_product(
        db_session,
        item_id="ssd-15360",
        part_number="SSD-15360",
        producer="Samsung",
        category_id="V110106",
        item_name="Server SSD NVMe 15.36TB U.2",
        quantity=4,
        price=Decimal("900"),
    )

    result = asyncio.run(
        match_stock_spec(
            _right_size_server_spec(),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(_right_size_overfit_llm_response),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()
    llm_build = report_json["llm_recommended_build_candidates"][0]

    assert report_json["llm_configurator_used"] is True
    assert llm_build["storage_over_requirement"] == 4.0
    assert llm_build["overfit_reason"]
    assert llm_build["right_size_note"].startswith("Подбор:")
    assert "SSD выше минимального требования" in llm_build["right_size_note"]
    assert any(
        "right-size reason was added" in warning
        for warning in report_json["llm_internal_warnings"]
    )


def test_llm_configurator_cpu_over_requirement_uses_evidence_when_no_closer_cpu() -> None:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 32,
        },
    )
    matrix["cpu_candidates"][0]["fit_label"] = "acceptable_overfit"
    matrix["cpu_candidates"][0]["fit_reason"] = (
        "CPU выше минимального требования: 32 ядер вместо 16."
    )
    matrix["cpu_candidates"][0]["cpu_over_requirement"] = 16

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера с Intel Xeon не менее 16 ядер.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert "32 ядра вместо 16 ядер" in recommendation["right_size_note"]
    assert "32 ядер" not in recommendation["right_size_note"]
    assert (
        "Среди складских Intel Xeon для этой платформы не найден более близкий к 16 ядрам "
        "вариант с остатком на 4 CPU и без явных конфликтов совместимости"
    ) in recommendation["right_size_note"]
    assert "CPU-кандидат" not in recommendation["right_size_note"]
    assert "самый дешев" not in recommendation["right_size_note"].casefold()


def test_llm_configurator_cpu_over_requirement_asks_to_check_when_price_not_proven() -> None:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 32,
        },
    )
    matrix["cpu_candidates"][0]["fit_label"] = "acceptable_overfit"
    matrix["cpu_candidates"][0]["cpu_over_requirement"] = 16
    matrix["cpu_candidates"].append(
        _composer_component_candidate(
            "cpu-24",
            "Intel",
            "CPU-24C",
            "Intel Xeon Gold 5424 24 core processor",
            4,
            Decimal("0"),
            {
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        )
    )
    matrix["cpu_candidates"][1]["price_value"] = None
    matrix["cpu_candidates"][1]["fit_label"] = "exact_or_close_fit"

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера с Intel Xeon не менее 16 ядер.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert "32 ядра вместо 16 ядер" in recommendation["right_size_note"]
    assert "Перед КП проверьте альтернативы CPU в матрице компонентов" in recommendation[
        "right_size_note"
    ]


def test_llm_configurator_marks_cpu_over_requirement_as_selection_skip_not_fatal() -> None:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 32,
        },
    )
    matrix["cpu_candidates"][0]["part_number"] = "CPU-32C"
    matrix["cpu_candidates"][0]["price_value"] = "800"
    matrix["cpu_candidates"][0]["fit_label"] = "acceptable_overfit"
    matrix["cpu_candidates"][0]["cpu_over_requirement"] = 16
    matrix["cpu_candidates"].append(
        _composer_component_candidate(
            "cpu-24",
            "Intel",
            "CPU-24C",
            "Intel Xeon Gold 5424 24 core processor",
            4,
            Decimal("400"),
            {
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        )
    )
    matrix["cpu_candidates"][1]["fit_label"] = "exact_or_close_fit"

    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера с Intel Xeon не менее 16 ядер.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=matrix,
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_cpu_32_overfit_response),
        settings=_llm_composer_settings(),
    )

    assert outcome.used is True
    assert outcome.fallback_reason is None
    assert len(outcome.recommended_builds) == 1
    assert outcome.valid_proposals_count == 1
    assert outcome.validation_rejected_count == 0
    assert outcome.selection_skipped_count == 0
    assert any(
        "closer cheaper stocked alternative exists for cpu" in warning
        for warning in outcome.internal_warnings
    )


def test_llm_configurator_storage_exact_requirement_is_stated() -> None:
    outcome = compose_llm_configurations(
        user_request="Нужно 2 сервера с SSD NVMe 3.84 ТБ.",
        normalized_requirements=[_right_size_composer_requirements()],
        ready_stock_candidates=[],
        component_candidate_matrix=_composer_component_matrix(
            platform_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
            },
            cpu_facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_cores": 24,
            },
        ),
        rule_based_build_candidates=[],
        llm_client=_FakeComposerClient(_component_matrix_llm_response),
        settings=_llm_composer_settings(),
    )

    recommendation = outcome.recommended_builds[0]

    assert outcome.used is True
    assert "CPU соответствует минимальному требованию 16 ядер." in recommendation[
        "right_size_note"
    ]
    assert "SSD соответствует минимальному требованию 3.84 ТБ." in recommendation[
        "right_size_note"
    ]


def test_llm_configurator_rejects_unknown_source_candidate_id(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(_unknown_component_response),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()

    assert report_json["llm_configurator_used"] is False
    assert report_json["llm_recommended_build_candidates"] == []
    assert (
        report_json["llm_fallback_reason"]
        == "llm_configurator_all_recommendations_rejected"
    )
    assert report_json["build_candidates"]


def test_llm_configurator_rejects_source_type_mismatch(db_session: Session) -> None:
    _seed_complete_component_set(db_session)

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(_wrong_role_response),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()

    assert report_json["llm_configurator_used"] is False
    assert (
        report_json["llm_fallback_reason"]
        == "llm_configurator_all_recommendations_rejected"
    )
    assert any(
        "source type mismatch" in warning
        for warning in report_json["llm_internal_warnings"]
    )


def test_llm_configurator_falls_back_on_validation_failed(db_session: Session) -> None:
    _seed_complete_component_set(db_session)

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_FakeComposerClient(lambda _package: {"oops": []}),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()

    assert report_json["llm_configurator_used"] is False
    assert report_json["llm_fallback_reason"] == "llm_configurator_validation_failed"
    assert report_json["llm_error_type"] == "ValidationError"
    assert report_json["llm_http_status"] is None
    assert report_json["ai_validation_summary"]["rejected_invalid_schema"] == 0
    assert report_json["rejected_ai_recommendations_debug_safe"] == []
    assert report_json["build_candidates"]


def test_llm_validation_failed_proposals_keep_safe_rejection_debug(
    db_session: Session,
) -> None:
    _seed_complete_component_set(db_session)

    def payload_factory(package: dict[str, Any]) -> dict[str, Any]:
        recommendation = _component_matrix_llm_recommendation(
            package,
            recommendation_id="bad_schema",
            why_selected="Invalid schema should be diagnosed.",
        )
        recommendation["confidence"] = "certain"
        return {"proposals": [recommendation], "general_notes": []}

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_RawPayloadComposerClient(payload_factory),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()
    debug = report_json["rejected_ai_recommendations_debug_safe"][0]
    debug_text = json.dumps(debug, ensure_ascii=False)

    assert report_json["llm_configurator_used"] is False
    assert report_json["llm_fallback_reason"] == "llm_configurator_validation_failed"
    assert report_json["ai_validation_summary"]["rejected_invalid_schema"] == 1
    assert report_json["rejected_reasons_top"][0]["reason"] == "invalid_schema"
    assert debug["rejection_code"] == "invalid_schema"
    assert debug["recommendation_id"] == "bad_schema"
    assert debug["validation_errors"]
    assert "raw_prompt" not in debug_text
    assert "raw_response" not in debug_text
    assert "headers" not in debug_text
    assert "token" not in debug_text.casefold()
    assert "test-key" not in debug_text


def test_llm_configurator_falls_back_on_invalid_json_error(db_session: Session) -> None:
    _seed_complete_component_set(db_session)

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_InvalidJsonComposerClient(),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()

    assert report_json["llm_configurator_used"] is False
    assert report_json["llm_fallback_reason"] == "llm_configurator_invalid_json"
    assert report_json["llm_error_type"] == "LlmInvalidJsonError"
    assert report_json["build_candidates"]


def test_llm_configurator_invalid_json_diagnostics_are_safe(db_session: Session) -> None:
    _seed_complete_component_set(db_session)

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_DiagnosticInvalidJsonComposerClient(),
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()
    diagnostics_text = json.dumps(
        report_json["llm_parse_diagnostics"],
        ensure_ascii=False,
    )

    assert report_json["llm_configurator_used"] is False
    assert report_json["llm_fallback_reason"] == "llm_configurator_invalid_json"
    assert report_json["llm_parse_stage"] == "message_content"
    assert report_json["llm_json_extract_status"] == "parse_error"
    assert report_json["llm_invalid_json_reason"] == "not json after local repair"
    assert "not-json" in report_json["llm_invalid_json_preview_sanitized"]
    assert "super-secret-llm-token" not in diagnostics_text
    assert "Authorization" not in diagnostics_text
    assert "api_key" not in diagnostics_text
    assert "messages" not in diagnostics_text
    assert "component_candidate_matrix" not in diagnostics_text


def test_llm_configurator_read_timeout_is_not_retried(db_session: Session) -> None:
    _seed_complete_component_set(db_session)
    client = _ReadTimeoutComposerClient()

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=client,
            llm_settings=_llm_composer_settings(),
        )
    )
    report_json = result.to_report_json()

    assert client.calls == 1
    assert report_json["llm_configurator_used"] is False
    assert (
        report_json["llm_fallback_reason"]
        == "llm_configurator_read_timeout_not_retried"
    )
    assert report_json["llm_error_type"] == "LlmReadTimeoutError"
    assert report_json["build_candidates"]


def test_llm_configurator_does_not_log_api_key_on_request_failure(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _seed_complete_component_set(db_session)
    secret = "llm-secret-value"
    caplog.set_level(logging.INFO, logger="app.llm.configuration_composer")

    result = asyncio.run(
        match_stock_spec(
            _server_spec(quantity=2, ram_min_gb=512),
            _adapter(db_session),
            llm_configurator_client=_RaisingComposerClient(secret, status_code=524),
            llm_settings=_llm_composer_settings(api_key=secret),
        )
    )

    report_json = result.to_report_json()
    assert report_json["llm_fallback_reason"] == "llm_configurator_request_failed"
    assert report_json["llm_error_type"] == "LlmClientError"
    assert report_json["llm_http_status"] == 524
    assert secret not in caplog.text


def test_configuration_builder_v03_warns_when_component_alternatives_are_limited(
    db_session: Session,
) -> None:
    for item_id, part_number, producer in [
        ("platform-asus", "ASUS-2U", "ASUS"),
        ("platform-gooxi", "GOOXI-2U", "Gooxi"),
    ]:
        _seed_component_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer=producer,
            category_id="V110100",
            item_name=f"{producer} 2U dual socket DDR5 server platform 2x PSU",
            quantity=2,
            price=Decimal("2000"),
        )
    _seed_component_product(
        db_session,
        item_id="intel-cpu",
        part_number="BX807135416S",
        producer="Intel",
        category_id="V110103",
        item_name="Intel Xeon Gold 5416S tray processor",
        quantity=8,
        price=Decimal("700"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-64",
        part_number="RAM-DDR5-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=32,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-960",
        part_number="SSD-960G",
        producer="Samsung",
        category_id="V110106",
        item_name="Server SSD 960GB SATA",
        quantity=4,
        price=Decimal("200"),
    )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=512), _adapter(db_session))
    )

    assert len(result.to_report_json()["build_candidates"]) >= 2
    assert any("Альтернатив по CPU/RAM/SSD" in warning for warning in result.risk_flags)


def test_quantity_risk_when_required_quantity_exceeds_available_stock(
    db_session: Session,
) -> None:
    _seed_nerpa_product(
        db_session,
        item_id="1000841882",
        part_number="D5720-181125SA04",
        item_name="Server NERPA D5720 2U 2x CPU SSD 2x PSU 2x DDR4 32GB",
        quantity=1,
        can_reserve=True,
    )

    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=64), _adapter(db_session))
    )

    assert result.status == STATUS_PARTIAL_STOCK_MATCHED
    assert result.candidates[0].available_quantity == 1
    assert any("Остаток ниже требования" in item for item in result.missing_requirements)
    assert any("По одному варианту не хватает остатка" in item for item in result.risk_flags)


def test_markdown_report_contains_part_number_price_and_risks(db_session: Session) -> None:
    _seed_nerpa_product(
        db_session,
        item_id="1000841882",
        part_number="D5720-181125SA04",
        item_name="Server NERPA D5720 2U 2x CPU SSD 2x PSU 2x DDR4 32GB",
        quantity=1,
        can_reserve=True,
    )
    result = asyncio.run(
        match_stock_spec(_server_spec(quantity=2, ram_min_gb=64), _adapter(db_session))
    )

    markdown = build_match_markdown_report(result)

    assert "D5720-181125SA04" in markdown
    assert "6900 USD" in markdown
    assert "Гарантия у OCS указана 12 месяцев" in markdown
    assert "По одному варианту не хватает остатка" in markdown


def _first_component(candidate: dict[str, Any], role: str) -> dict[str, Any]:
    for component in candidate["components"]:
        if component["role"] == role:
            return component
    raise AssertionError(f"Missing component role {role}")


def _component_by_role(components: Sequence[Mapping[str, Any]], role: str) -> Mapping[str, Any]:
    for component in components:
        if component.get("role") == role:
            return component
    raise AssertionError(f"Missing component role {role}")


def _adapter(db_session: Session) -> AsyncSessionAdapter:
    return AsyncSessionAdapter(db_session)


def _server_spec(*, quantity: int, ram_min_gb: int) -> StockSpec:
    return StockSpec(
        items=[
            StockSpecItem(
                item_type="server",
                quantity=quantity,
                name="server",
                requirements={
                    "form_factor": "2U",
                    "cpu": {"sockets": 2},
                    "ram": {"min_gb": ram_min_gb},
                    "storage": {"type": "SSD"},
                    "power": {"psu_count": 2, "redundant_psu": True},
                },
            )
        ],
        shipment_city="Москва",
        source_text="Нужно 2 сервера 2U, 2 процессора, 512 ГБ RAM, SSD, 2 БП, склад Москва",
    )


def _right_size_server_spec() -> StockSpec:
    return StockSpec(
        items=[
            StockSpecItem(
                item_type="server",
                quantity=2,
                name="server",
                requirements={
                    "form_factor": "2U",
                    "cpu": {
                        "sockets": 2,
                        "vendor": "Intel",
                        "family": "Xeon",
                        "min_cores_per_cpu": 16,
                    },
                    "ram": {"min_gb": 512, "type": "DDR5"},
                    "storage": {
                        "type": "SSD",
                        "interface": "NVMe",
                        "min_capacity": "3.84 TB",
                        "qty_per_server": 2,
                    },
                    "power": {"psu_count": 2, "redundant_psu": True},
                },
            )
        ],
        shipment_city="Москва",
        source_text=(
            "Нужно 2 сервера 2U, 2 CPU Intel Xeon не менее 16 ядер, "
            "512 ГБ RAM DDR5, 2 SSD NVMe не менее 3.84 ТБ на сервер, 2 БП"
        ),
    )


def _network_switch_spec() -> StockSpec:
    return StockSpec(
        items=[
            StockSpecItem(
                item_type="network",
                quantity=1,
                name="коммутатор",
                requirements={},
            )
        ],
        shipment_city="Москва",
        source_text=(
            "Нужен коммутатор 48 портов 1G PoE+, 4 uplink 10G SFP+, "
            "L3, stacking, трансиверы в комплект, 4 DAC, лицензия/support 1 год, "
            "склад Москва, один самый дешевый вариант"
        ),
    )


def _storage_fc_spec() -> StockSpec:
    return StockSpec(
        items=[
            StockSpecItem(
                item_type="storage",
                quantity=1,
                name="storage array",
                requirements={},
            )
        ],
        shipment_city="РњРѕСЃРєРІР°",
        source_text=(
            "Need storage array 100 TB usable capacity, 2 controllers, SSD, "
            "FC 32G, support 3 years, Moscow stock, one cheapest quote."
        ),
    )


def _seed_nerpa_product(
    db_session: Session,
    *,
    item_id: str,
    part_number: str,
    item_name: str,
    quantity: int,
    can_reserve: bool,
) -> None:
    synced_at = datetime(2026, 5, 9, 12, 0, tzinfo=UTC)
    now = datetime(2026, 5, 9, 12, 5, tzinfo=UTC)
    db_session.add(
        DistributorProduct(
            distributor_code="ocs",
            item_id=item_id,
            product_key=item_id,
            part_number=part_number,
            producer="NERPA",
            category_id="V1100",
            item_name=item_name,
            item_name_rus="Сервер NERPA",
            product_name=f"NERPA {part_number}",
            product_description=None,
            product_notes=None,
            hscode="8471490000",
            ean="04600000000012",
            is_in_mpt_registry=False,
            is_project_item=False,
            traceable=False,
            condition="Regular",
            warranty="Distributor warranty 12 months",
            original_country_iso_code="RU",
            vat_percent=Decimal("20"),
            serial_number_availability=None,
            catalog_path_json=[{"category_id": "V1100", "name": "Servers in assembly"}],
            package_json={"weight": 25.0},
            raw_json={"product": {"itemId": item_id, "partNumber": part_number}},
            synced_at=synced_at,
            created_at=now,
            updated_at=now,
        )
    )
    db_session.add(
        DistributorStockPrice(
            distributor_code="ocs",
            item_id=item_id,
            product_key=item_id,
            shipment_city="Москва",
            location="MSK",
            location_description="Moscow",
            location_type="ShipmentCity",
            quantity_value=quantity,
            quantity_is_greater_than=False,
            can_reserve=can_reserve,
            departure_date=None,
            arrival_date=None,
            delivery_date=None,
            price_order_value=Decimal("6900"),
            price_order_currency="USD",
            price_list_value=Decimal("6900"),
            price_list_currency="USD",
            end_user_value=Decimal("7100"),
            end_user_currency="USD",
            raw_json={"productKey": item_id, "location": "MSK"},
            synced_at=synced_at,
            created_at=now,
        )
    )
    db_session.commit()


def _seed_component_product(
    db_session: Session,
    *,
    item_id: str,
    part_number: str,
    producer: str,
    category_id: str,
    item_name: str,
    quantity: int,
    price: Decimal,
) -> None:
    synced_at = datetime(2026, 5, 9, 12, 0, tzinfo=UTC)
    now = datetime(2026, 5, 9, 12, 5, tzinfo=UTC)
    db_session.add(
        DistributorProduct(
            distributor_code="ocs",
            item_id=item_id,
            product_key=item_id,
            part_number=part_number,
            producer=producer,
            category_id=category_id,
            item_name=item_name,
            item_name_rus=item_name,
            product_name=f"{producer} {part_number}",
            product_description=None,
            product_notes=None,
            hscode="8473308000",
            ean=None,
            is_in_mpt_registry=False,
            is_project_item=False,
            traceable=False,
            condition="Regular",
            warranty="Distributor warranty 12 months",
            original_country_iso_code="CN",
            vat_percent=Decimal("20"),
            serial_number_availability=None,
            catalog_path_json=[{"category_id": category_id, "name": "Server component"}],
            package_json={},
            raw_json={"product": {"itemId": item_id, "partNumber": part_number}},
            synced_at=synced_at,
            created_at=now,
            updated_at=now,
        )
    )
    db_session.add(
        DistributorStockPrice(
            distributor_code="ocs",
            item_id=item_id,
            product_key=item_id,
            shipment_city="Москва",
            location="MSK",
            location_description="Moscow",
            location_type="ShipmentCity",
            quantity_value=quantity,
            quantity_is_greater_than=False,
            can_reserve=True,
            departure_date=None,
            arrival_date=None,
            delivery_date=None,
            price_order_value=price,
            price_order_currency="USD",
            price_list_value=price,
            price_list_currency="USD",
            end_user_value=price,
            end_user_currency="USD",
            raw_json={"productKey": item_id, "location": "MSK"},
            synced_at=synced_at,
            created_at=now,
        )
    )
    db_session.commit()


def _seed_network_product(
    db_session: Session,
    *,
    item_id: str,
    part_number: str,
    producer: str,
    category_id: str,
    item_name: str,
    quantity: int,
    price: Decimal,
) -> None:
    _seed_component_product(
        db_session,
        item_id=item_id,
        part_number=part_number,
        producer=producer,
        category_id=category_id,
        item_name=item_name,
        quantity=quantity,
        price=price,
    )


def _seed_match_71_network_products(
    db_session: Session,
    *,
    include_good_switch: bool = True,
) -> None:
    rows = [
        (
            "switch-good",
            "SW-48P-4SFP",
            "48-port 1G RJ45 PoE+ switch 740W 4 uplink 10G SFP+ L3 stacking",
            "1200",
        ),
        ("switch-5p", "SW-5P", "5-port desktop switch", "25"),
        ("switch-8p", "SW-8P", "8-port 1G PoE switch", "100"),
        ("switch-16p", "SW-16P", "16-port 1G PoE switch", "180"),
        ("switch-24p", "SW-24P", "24-port 1G PoE+ switch 2 uplink 10G SFP+", "260"),
    ]
    for item_id, part_number, name, price in rows:
        if item_id == "switch-good" and not include_good_switch:
            continue
        _seed_network_product(
            db_session,
            item_id=item_id,
            part_number=part_number,
            producer="NetVendor",
            category_id="V120100",
            item_name=name,
            quantity=10,
            price=Decimal(price),
        )


def _seed_complete_component_set(db_session: Session) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-2U-2S",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 2U dual socket DDR5 server platform 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    _seed_component_product(
        db_session,
        item_id="cpu-1",
        part_number="CPU-SERVER",
        producer="Intel",
        category_id="V110103",
        item_name="Intel Xeon server CPU",
        quantity=4,
        price=Decimal("500"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-64",
        part_number="RAM-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-1",
        part_number="SSD-960G",
        producer="Samsung",
        category_id="V110106",
        item_name="Server SSD 960GB SATA",
        quantity=2,
        price=Decimal("200"),
    )


def _seed_amd_25gbe_component_set(
    db_session: Session,
    *,
    include_network: bool = True,
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-amd-2u",
        part_number="AMD-2U-2S-PSU",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 2U dual socket AMD EPYC DDR5 NVMe server platform 2x PSU",
        quantity=2,
        price=Decimal("2400"),
    )
    _seed_component_product(
        db_session,
        item_id="cpu-amd-32",
        part_number="EPYC-32C",
        producer="AMD",
        category_id="V110103",
        item_name="AMD EPYC 9354 32 core server CPU",
        quantity=4,
        price=Decimal("900"),
    )
    _seed_component_product(
        db_session,
        item_id="ram-ddr5-64",
        part_number="RAM-DDR5-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=24,
        price=Decimal("160"),
    )
    _seed_component_product(
        db_session,
        item_id="ssd-nvme-7680",
        part_number="SSD-NVME-7680",
        producer="KIOXIA",
        category_id="V110106",
        item_name="KIOXIA Server SSD NVMe 7.68TB",
        quantity=8,
        price=Decimal("420"),
    )
    if include_network:
        _seed_component_product(
            db_session,
            item_id="nic-25g-dual",
            part_number="LRES1026PF-2SFP28",
            producer="ShenzhenLianrui Electronic Co., LTD",
            category_id="V120116",
            item_name=(
                "LR-Link LRES1026PF-2SFP28, 2xSFP28 ports, "
                "25GbE PCIe network adapter"
            ),
            quantity=30,
            price=Decimal("250"),
        )


def _seed_right_size_base_components(
    db_session: Session,
    *,
    include_cpu: bool = True,
) -> None:
    _seed_component_product(
        db_session,
        item_id="platform-1",
        part_number="PLATFORM-2U-2S",
        producer="ASUS",
        category_id="V110100",
        item_name="ASUS 2U dual socket DDR5 server platform 2x PSU",
        quantity=2,
        price=Decimal("2000"),
    )
    if include_cpu:
        _seed_component_product(
            db_session,
            item_id="cpu-16",
            part_number="CPU-16C",
            producer="Intel",
            category_id="V110103",
            item_name="Intel Xeon Silver 4416 16 core tray processor",
            quantity=4,
            price=Decimal("500"),
        )
    _seed_component_product(
        db_session,
        item_id="ram-64",
        part_number="RAM-DDR5-64G",
        producer="Samsung",
        category_id="V110104",
        item_name="DDR5 RDIMM 64GB server memory module",
        quantity=16,
        price=Decimal("150"),
    )


def _llm_composer_settings(
    api_key: str = "test-key",
    *,
    output_mode: str = "grouped_presales",
) -> LlmSettings:
    return LlmSettings(
        llm_provider="openai-compatible",
        llm_base_url="https://llm.example.test/v1",
        llm_api_key=api_key,
        llm_model="test-model",
        llm_configurator_enabled=True,
        llm_configurator_mode="composer",
        llm_configurator_output_mode=output_mode,
    )


def _web_evidence_settings() -> WebEvidenceSettings:
    return WebEvidenceSettings(
        web_evidence_enabled=True,
        web_evidence_provider="fake",
        web_evidence_max_queries=8,
        web_evidence_cache_ttl_hours=0,
    )


def _routerai_web_evidence_settings() -> WebEvidenceSettings:
    return WebEvidenceSettings(
        web_evidence_enabled=True,
        web_evidence_provider="routerai",
        web_evidence_model="deepseek/deepseek-v4-pro:online",
        web_evidence_max_queries=8,
        web_evidence_cache_ttl_hours=0,
    )


def _full_component_evidence_rows() -> dict[str, dict[str, Any]]:
    return {
        "platform-amd": {
            "facts": {
                "supported_cpu_generation": "4th Gen Intel Xeon Scalable",
                "socket_family": "LGA4677",
                "memory_type": "DDR5",
                "nvme_support": True,
            },
            "domain": "servers.asus.com",
        },
        "cpu-selected": {
            "facts": {
                "cpu_generation": "4th Gen Intel Xeon Scalable",
                "socket_family": "LGA4677",
                "cores": 32,
            },
            "domain": "ark.intel.com",
        },
        "ram-64": {
            "facts": {"memory_type": "DDR5", "capacity": "64GB"},
            "domain": "samsung.com",
        },
        "ssd-3840": {
            "facts": {"storage_interface": "NVMe", "capacity": "3.84TB"},
            "domain": "kioxia.com",
        },
    }


class _FakeRouterAIEvidenceProvider:
    provider_name = "routerai"

    def __init__(
        self,
        components_by_id: Mapping[str, Mapping[str, Any]],
        relation_by_type: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._components_by_id = components_by_id
        self._relation_by_type = relation_by_type or {}
        self.requests_count = 0
        self.tasks: list[EvidenceSearchTask] = []

    def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout: float,
    ) -> list[EvidenceSearchResult]:
        return []

    def collect_evidence(
        self,
        *,
        tasks: Sequence[EvidenceSearchTask],
        settings: WebEvidenceSettings,
        cache: EvidenceSearchCache | None = None,
        normalized_requirements: Any = None,
    ) -> EvidencePack:
        self.requests_count += 1
        self.tasks.extend(tasks)
        components: list[ComponentEvidence] = []
        relations: list[RelationEvidence] = []
        for task in tasks:
            if str(task.target_type).startswith("relation_"):
                relation_type = str(task.role or "").strip()
                row = self._relation_by_type.get(relation_type)
                if row is None:
                    relations.append(
                        RelationEvidence(
                            relation_type=relation_type,  # type: ignore[arg-type]
                            recommendation_id=task.recommendation_id,
                            components={},
                            status="not_confirmed",
                            confidence="unknown",
                            missing_evidence=["support list not found"],
                            engineering_checks=["engineer review required"],
                        )
                    )
                    continue
                sources = [
                    EvidenceSource(
                        url=str(source.get("url") or f"https://{source.get('domain')}/source"),
                        title=str(source.get("title") or "RouterAI relation source"),
                        snippet=str(source.get("snippet") or ""),
                        domain=str(source.get("domain") or "example.test"),
                        source_type=str(source.get("source_type") or "official_vendor"),
                        trust_score=float(source.get("trust_score") or 0.95),
                        retrieved_at=datetime.now(UTC).isoformat(),
                    )
                    for source in row.get(
                        "sources",
                        [{"domain": row.get("domain", "example.test")}],
                    )
                ]
                relations.append(
                    RelationEvidence(
                        relation_type=relation_type,  # type: ignore[arg-type]
                        recommendation_id=task.recommendation_id,
                        components={},
                        status=str(row.get("status") or "confirmed"),
                        confidence=str(row.get("confidence") or "high"),
                        confirmed_facts=list(row.get("confirmed_facts") or []),
                        missing_evidence=list(row.get("missing_evidence") or []),
                        mismatch_facts=list(row.get("mismatch_facts") or []),
                        engineering_checks=list(row.get("engineering_checks") or []),
                        sources=sources,
                    )
                )
                continue
            row = self._components_by_id.get(task.component_candidate_id)
            if row is None:
                components.append(
                    ComponentEvidence(
                        component_candidate_id=task.component_candidate_id,
                        role=task.role or task.target_type,
                        part_number=task.part_number,
                        name=task.name,
                        evidence_status="not_found",
                        confidence="unknown",
                        warnings=["No external evidence found."],
                    )
                )
                continue
            sources = [
                EvidenceSource(
                    url=str(source.get("url") or f"https://{source.get('domain')}/source"),
                    title=str(source.get("title") or "RouterAI evidence source"),
                    snippet=str(source.get("snippet") or ""),
                    domain=str(source.get("domain") or "example.test"),
                    source_type=str(source.get("source_type") or "official_vendor"),
                    trust_score=float(source.get("trust_score") or 0.95),
                    retrieved_at=datetime.now(UTC).isoformat(),
                )
                for source in row.get(
                    "sources",
                    [{"domain": row.get("domain", "example.test")}],
                )
            ]
            components.append(
                ComponentEvidence(
                    component_candidate_id=task.component_candidate_id,
                    role=task.role or task.target_type,
                    part_number=task.part_number,
                    name=task.name,
                    evidence_status="found",
                    confidence=str(row.get("confidence") or "high"),
                    facts=dict(row.get("facts") or {}),
                    sources=sources,
                    warnings=list(row.get("warnings") or []),
                )
            )
        completed = sum(1 for component in components if component.sources) + sum(
            1 for relation in relations if relation.sources
        )
        return EvidencePack(
            enabled=True,
            provider=self.provider_name,
            total_tasks=len(tasks),
            completed_tasks=completed,
            components=components,
            relation_evidence=relations,
            evidence_summary=f"external evidence found for {completed} of {len(tasks)} components",
            search_tasks=[task.model_dump() for task in tasks],
        )


class _FakeComposerClient:
    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.package: dict[str, Any] = {}
        self.system_prompts: list[str] = []
        self._last_response: dict[str, Any] | None = None
        self.calls = 0

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert "Do not invent products" in system_prompt
        self.system_prompts.append(system_prompt)
        self.calls += 1
        self.package = json.loads(user_prompt)
        if "critique_facts" in self.package and self._last_response is not None:
            return self._last_response
        self._last_response = self._responder(self.package)
        return self._last_response


class _RepairAwareComposerClient:
    def __init__(self, *, primary_responder: Any, repair_responder: Any) -> None:
        self._primary_responder = primary_responder
        self._repair_responder = repair_responder
        self.primary_packages: list[dict[str, Any]] = []
        self.repair_packages: list[dict[str, Any]] = []
        self.primary_system_prompts: list[str] = []
        self.repair_system_prompts: list[str] = []

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert "Do not invent products" in system_prompt
        package = json.loads(user_prompt)
        if (
            "critique_facts" in package
            or "empty_response_repair_attempt" in package
            or "no_recommendation_coverage_repair_attempt" in package
        ):
            self.repair_system_prompts.append(system_prompt)
            self.repair_packages.append(package)
            return self._repair_responder(package)
        self.primary_system_prompts.append(system_prompt)
        self.primary_packages.append(package)
        return self._primary_responder(package)


class _RawPayloadComposerClient:
    def __init__(self, payload_factory: Any) -> None:
        self._payload_factory = payload_factory
        self.package: dict[str, Any] = {}

    def generate_json(self, system_prompt: str, user_prompt: str) -> Any:
        assert "Do not invent products" in system_prompt
        self.package = json.loads(user_prompt)
        return self._payload_factory(self.package)


class _FakeComposerAndEvidenceReviewClient:
    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.packages: list[dict[str, Any]] = []

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert "Do not invent products" in system_prompt
        package = json.loads(user_prompt)
        self.packages.append(package)
        if "evidence_pack" in package:
            return {
                "evidence_review": [
                    {
                        "recommendation_id": recommendation.get("recommendation_id"),
                        "decision": "keep",
                        "evidence_confidence": "medium",
                        "confirmed_facts": [],
                        "missing_evidence": ["support list still requires engineer review"],
                        "fatal_concerns": [],
                        "engineering_checks": ["Check vendor CPU support list."],
                        "user_note": "Evidence reviewed.",
                    }
                    for recommendation in package.get("recommendations", [])
                ],
                "general_notes": ["Evidence review completed."],
            }
        return self._responder(package)


class _FakeSemanticPlannerOpenAIClient:
    def __init__(self, settings: LlmSettings, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs

    def close(self) -> None:
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "AI Semantic Matrix Planner V2" in system_prompt:
            package = json.loads(user_prompt)
            assert package["deterministic_product_group_hint"] in {
                "network",
                "server",
                "unknown",
            }
            return _semantic_server_78_payload_for_match_engine()
        if "Distributor Category Planner" in system_prompt:
            return {
                "category_plan": [
                    {
                        "role": "server_platform",
                        "selected_category_ids": ["V110100"],
                        "purpose": "base_device",
                        "capability_ids": ["server_platform.1u.2s"],
                        "hard_optional_relation": "hard",
                        "reason": "server platform category",
                        "confidence": "high",
                    },
                    {
                        "role": "cpu",
                        "selected_category_ids": ["V110103"],
                        "purpose": "component",
                        "capability_ids": ["cpu.intel"],
                        "hard_optional_relation": "hard",
                        "reason": "server CPU category",
                        "confidence": "high",
                    },
                    {
                        "role": "ram",
                        "selected_category_ids": ["V110104"],
                        "purpose": "component",
                        "capability_ids": ["ram.ddr5"],
                        "hard_optional_relation": "hard",
                        "reason": "server RAM category",
                        "confidence": "high",
                    },
                    {
                        "role": "storage",
                        "selected_category_ids": ["V110106"],
                        "purpose": "drive",
                        "capability_ids": ["storage.sata"],
                        "hard_optional_relation": "hard",
                        "reason": "server drive category",
                        "confidence": "high",
                    },
                    {
                        "role": "network_adapter",
                        "selected_category_ids": ["V120116"],
                        "purpose": "component",
                        "capability_ids": ["network_adapter.10gbe.sfp+"],
                        "hard_optional_relation": "hard",
                        "reason": "server NIC category",
                        "confidence": "high",
                    },
                ],
                "missing_category_roles": [],
                "category_plan_warnings": [],
            }
        raise AssertionError(f"unexpected LLM prompt: {system_prompt[:80]}")


def _semantic_server_78_payload_for_match_engine() -> dict[str, Any]:
    return {
        "primary_product_group": "server",
        "primary_object": "server",
        "confidence": "high",
        "classification_reason": "The request is a server BOM with embedded NIC/cables.",
        "matrix_blueprint": {
            "roles": [
                {
                    "role": "server_platform",
                    "required": True,
                    "source_text": "1U 2-socket server",
                    "characteristics_to_match": {
                        "form_factor": "1U",
                        "socket_count": 2,
                    },
                    "hard_capability_ids": ["server_platform.1u.2s"],
                },
                {
                    "role": "cpu",
                    "required": True,
                    "source_text": "Intel CPUs",
                    "characteristics_to_match": {"vendor": "Intel"},
                    "hard_capability_ids": ["cpu.intel"],
                },
                {
                    "role": "ram",
                    "required": True,
                    "source_text": "DDR5 RAM",
                    "characteristics_to_match": {"type": "DDR5"},
                    "hard_capability_ids": ["ram.ddr5"],
                },
                {
                    "role": "storage",
                    "required": True,
                    "source_text": "SATA SSD",
                    "characteristics_to_match": {"interface": "SATA"},
                    "hard_capability_ids": ["storage.sata"],
                },
                {
                    "role": "network_adapter",
                    "required": True,
                    "source_text": "Intel X710-DA2 2x10GbE SFP+",
                    "characteristics_to_match": {
                        "min_ports_per_server": 2,
                        "speed": "10GbE",
                        "media": "SFP+",
                    },
                    "hard_capability_ids": ["network_adapter.10gbe.sfp+"],
                },
                {
                    "role": "cable",
                    "required": True,
                    "source_text": "C13-C14 power cables",
                    "characteristics_to_match": {
                        "cable_type": "power",
                        "connector_types": ["C13-C14"],
                    },
                    "hard_capability_ids": ["cable.power"],
                },
            ]
        },
        "required_capabilities": [],
        "optional_capabilities": [],
        "embedded_requirements": [
            {
                "role": "network_adapter",
                "reason": "SFP+ belongs to the NIC inside the server.",
            }
        ],
        "not_primary_product_groups": [
            {
                "product_group": "network",
                "reason": "No standalone switch/DAC requested.",
            }
        ],
        "logistics_constraints": {},
        "commercial_instructions": [],
        "response_instructions": [],
        "engineer_review_instructions": [],
        "unsupported_or_unmapped_requirements": [],
        "planner_warnings": [],
    }


class _RecordingOpenAIComposerClient:
    instances: list[_RecordingOpenAIComposerClient] = []
    response_factory: Any = None
    failing_models: set[str] = set()

    def __init__(self, settings: LlmSettings, **kwargs: Any) -> None:
        self.settings = settings
        self.kwargs = kwargs
        self.system_prompt = ""
        self.package: dict[str, Any] = {}
        type(self).instances.append(self)

    @classmethod
    def reset(cls, response_factory: Any, *, failing_models: set[str] | None = None) -> None:
        cls.instances = []
        cls.response_factory = response_factory
        cls.failing_models = failing_models or set()

    def close(self) -> None:
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.settings.llm_model in type(self).failing_models:
            raise LlmClientError("online composer failed")
        self.system_prompt = system_prompt
        self.package = json.loads(user_prompt)
        if type(self).response_factory is None:
            raise AssertionError("response_factory is not configured")
        return type(self).response_factory(self.package)


class _FailingEvidenceProvider:
    provider_name = "routerai"

    def __init__(self) -> None:
        self.requests_count = 0

    def search(
        self,
        query: str,
        *,
        max_results: int,
        timeout: float,
    ) -> list[EvidenceSearchResult]:
        self.requests_count += 1
        raise AssertionError("online_composer must not run separate search")

    def collect_evidence(
        self,
        *,
        tasks: Sequence[EvidenceSearchTask],
        settings: WebEvidenceSettings,
        cache: EvidenceSearchCache | None = None,
        normalized_requirements: Any = None,
    ) -> EvidencePack:
        self.requests_count += 1
        raise AssertionError("online_composer must not run separate evidence")


class _RaisingComposerClient:
    def __init__(self, secret: str, *, status_code: int | None = None) -> None:
        self._secret = secret
        self._status_code = status_code

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise LlmClientError(
            f"request failed with {self._secret}",
            status_code=self._status_code,
        )


class _InvalidJsonComposerClient:
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise LlmInvalidJsonError("LLM response content was not valid JSON.")


class _DiagnosticInvalidJsonComposerClient:
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise LlmInvalidJsonError(
            "LLM response content was not valid JSON.",
            parse_stage="message_content",
            json_extract_status="parse_error",
            invalid_json_reason="not json after local repair",
            preview_sanitized=(
                "Authorization: Bearer super-secret-llm-token "
                "api_key=super-secret-llm-token not-json"
            ),
        )


class _ReadTimeoutComposerClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls += 1
        raise LlmReadTimeoutError("LLM request read timed out.")


def _valid_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendations": [
            _llm_build_recommendation(
                package,
                recommendation_id="llm_build_1",
                title="Recommended build 1",
                why_selected="Balanced stocked configuration.",
                critical_checks=["Check platform support list."],
            )
        ],
        "general_notes": ["CPU choice is preliminary."],
    }


def _fatal_pair_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendations": [
            _llm_build_recommendation(
                package,
                recommendation_id="llm_build_fatal_pair",
                title="Bad incompatible build",
                why_selected="Uses an incompatible source candidate.",
                quantities={"platform": 2, "cpu": 4, "ram": 16, "ssd": 4},
            )
        ],
        "general_notes": [],
    }


def _component_matrix_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendations": [
            _component_matrix_llm_recommendation(
                package,
                recommendation_id="llm_matrix_build",
                why_selected="LLM composed a new BOM from component matrix.",
                why_selected_short="Собрано из доступной матрицы компонентов.",
            )
        ],
        "general_notes": [],
    }


def _structured_no_recommendation_response(package: dict[str, Any]) -> dict[str, Any]:
    matrix = package.get("component_candidate_matrix", {})
    considered_by_role = {
        role: [
            row.get("component_candidate_id")
            for row in rows
            if row.get("component_candidate_id")
        ]
        for role, rows in matrix.items()
        if isinstance(rows, list)
    }
    network_ids = [
        row.get("component_candidate_id")
        for row in matrix.get("network_adapter", [])
        if row.get("component_candidate_id")
    ]
    considered_network_id = network_ids[0] if network_ids else "network_adapter-missing"
    return {
        "recommendations": [],
        "no_recommendation": {
            "summary": "No safe complete BOM can be produced from the provided matrix.",
            "missing_roles": ["network_adapter"],
            "missing_required_capabilities": [
                {
                    "role": "network_adapter",
                    "capability_id": "network_adapter.10gbe.sfp+",
                    "requirement_text": "10GbE SFP+ network adapter is required.",
                    "reason": (
                        "The matrix has no network_adapter candidate that proves the "
                        "requested media and port requirement."
                    ),
                }
            ],
            "hard_mismatches": [
                {
                    "role": "network_adapter",
                    "component_candidate_id": considered_network_id,
                    "requirement": "10GbE SFP+ network connectivity",
                    "candidate_fact": "candidate facts do not prove the required media",
                    "reason": "Hard network requirement is not safely satisfied.",
                }
            ],
            "stock_shortages": [],
            "role_analysis": [
                *[
                    {
                        "role": role,
                        "status": "satisfied",
                        "considered_candidate_ids": ids,
                        "considered_count": len(ids),
                        "explanation": "Role candidates were reviewed.",
                    }
                    for role, ids in considered_by_role.items()
                    if role != "network_adapter"
                ],
                {
                    "role": "network_adapter",
                    "status": "mismatch",
                    "considered_candidate_ids": network_ids,
                    "considered_count": len(network_ids),
                    "explanation": (
                        "Network adapter role remains unsafe for the hard request."
                    ),
                }
            ],
            "considered_candidate_ids": {**considered_by_role, "network_adapter": network_ids},
            "explanation_ru": (
                "network_adapter: матрица не подтверждает выполнение жесткого "
                "требования 10GbE SFP+."
            ),
        },
        "general_notes": [],
    }


def _one_candidate_per_role_no_recommendation_response(
    package: dict[str, Any],
) -> dict[str, Any]:
    considered: dict[str, list[str]] = {}
    role_analysis: list[dict[str, Any]] = []
    for role, rows in package.get("component_candidate_matrix", {}).items():
        if not isinstance(rows, list) or not rows:
            continue
        component_id = rows[0].get("component_candidate_id")
        if not component_id:
            continue
        considered[role] = [component_id]
        role_analysis.append(
            {
                "role": role,
                "status": "uncertain",
                "considered_candidate_ids": [component_id],
                "explanation": "Only one candidate was checked.",
            }
        )
    return {
        "recommendations": [],
        "no_recommendation": {
            "summary": "No safe complete BOM can be produced from the provided matrix.",
            "missing_roles": [],
            "missing_required_capabilities": [],
            "hard_mismatches": [],
            "stock_shortages": [],
            "role_analysis": role_analysis,
            "considered_candidate_ids": considered,
            "explanation_ru": "Недостаточно надежных совпадений.",
        },
        "general_notes": [],
    }


def _covered_no_recommendation_response(package: dict[str, Any]) -> dict[str, Any]:
    considered: dict[str, list[str]] = {}
    role_analysis: list[dict[str, Any]] = []
    for role, rows in package.get("component_candidate_matrix", {}).items():
        if not isinstance(rows, list) or not rows:
            continue
        ids = [
            row.get("component_candidate_id")
            for row in rows
            if row.get("component_candidate_id")
        ]
        considered[role] = ids
        role_analysis.append(
            {
                "role": role,
                "status": "mismatch",
                "considered_candidate_ids": ids,
                "considered_count": len(ids),
                "explanation": "All provided candidates for this role were checked.",
            }
        )
    return {
        "recommendations": [],
        "no_recommendation": {
            "summary": "No safe complete BOM can be produced after full role review.",
            "missing_roles": [],
            "missing_required_capabilities": [],
            "hard_mismatches": [],
            "stock_shortages": [],
            "role_analysis": role_analysis,
            "considered_candidate_ids": considered,
            "explanation_ru": "Матрица проверена достаточно широко.",
        },
        "general_notes": [],
    }


def _server_78_full_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_server_78_full",
        why_selected="Composer selected a complete server #78 BOM from the matrix.",
        why_selected_short="Complete stocked server #78 BOM.",
    )
    matrix = package["component_candidate_matrix"]
    role_aliases = {
        "storage_controller": "storage_controller",
        "network_adapter": "network_adapter",
        "power_supply": "power_supply",
        "cable": "cable",
    }
    for matrix_role, output_role in role_aliases.items():
        if matrix.get(matrix_role):
            recommendation["component_candidate_ids"][output_role] = matrix[matrix_role][
                0
            ]["component_candidate_id"]
            recommendation["quantities"][output_role] = 2
    recommendation["evidence_summary"] = {
        "status": "confirmed",
        "sources_count": 1,
        "confirmed_facts": ["Selected from package matrix."],
        "not_confirmed": [],
        "source_domains": ["example.test"],
        "notes": "Test evidence summary.",
    }
    return {"recommendations": [recommendation], "general_notes": []}


def _server_78_requirement_analysis_response(package: dict[str, Any]) -> dict[str, Any]:
    response = _server_78_full_llm_response(package)
    response["requirement_analysis"] = {
        "classified_requirements": [
            {
                "requirement_id": "req_cable",
                "source_text": "C13-C14 cables",
                "classification": "accessory_or_consumable",
                "target_role": "cable",
                "hard_or_optional": "hard",
            }
        ],
        "primary_object_feature_requirements": [
            {"requirement_id": "req_1u", "source_text": "server #78"}
        ],
        "purchasable_role_requirements": [
            {"requirement_id": "req_cpu", "target_role": "cpu"}
        ],
        "accessory_or_consumable_requirements": [
            {"requirement_id": "req_cable", "target_role": "cable"}
        ],
        "service_or_support_requirements": [],
        "logistics_or_commercial_constraints": [],
        "engineering_check_requirements": [
            {"requirement_id": "req_fans", "source_text": "8 fans N+1"}
        ],
        "fulfillment_decisions": [
            {
                "requirement_id": "req_cable",
                "fulfillment_mode": "separate_component_required",
                "closed_by": "separate_component",
                "component_candidate_id": "cable-0",
            }
        ],
        "unverified_requirements": [
            {"requirement_id": "req_fans", "source_text": "8 fans N+1"}
        ],
    }
    response["requirement_coverage_summary"] = {
        "coverage_note": "composer reconstructed requirements from original request"
    }
    return response


def _samsung_ram_primary_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_samsung_ram",
        why_selected="Cheapest quote candidate before matrix repair.",
        why_selected_short="Cheapest quote before repair.",
    )
    recommendation["proposal_role"] = "cheapest_fit"
    recommendation["recommendation_slot"] = "price_optimal"
    recommendation["component_candidate_ids"]["ram"] = (
        _required_component_candidate_id_by_part(package, "ram", "RAM-SAMSUNG-32G")
    )
    recommendation["quantities"] = {"platform": 2, "cpu": 4, "ram": 32, "storage": 4}
    return {"recommendations": [recommendation], "general_notes": []}


def _primary_recommendation_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="primary_qwen_build",
        why_selected="Cheapest complete stocked build that satisfies hard requirements.",
        why_selected_short="Cheapest complete stocked build.",
    )
    return {
        "primary_recommendation": {
            "candidate_type": recommendation["source_type"],
            "title": "Cheapest valid complete stock build",
            "component_candidate_ids": recommendation["component_candidate_ids"],
            "why_selected": recommendation["why_selected"],
            "assumptions": ["Code materializes quantities and prices."],
            "engineer_checks": ["Check platform support list."],
        },
        "general_notes": [],
    }


def _primary_recommendation_missing_cpu_response(package: dict[str, Any]) -> dict[str, Any]:
    response = _primary_recommendation_response(package)
    component_ids = dict(response["primary_recommendation"]["component_candidate_ids"])
    component_ids.pop("cpu", None)
    response["primary_recommendation"]["component_candidate_ids"] = component_ids
    return response


def _primary_recommendation_with_network_response(package: dict[str, Any]) -> dict[str, Any]:
    response = _primary_recommendation_response(package)
    component_ids = dict(response["primary_recommendation"]["component_candidate_ids"])
    component_ids["network_adapter"] = package["component_candidate_matrix"][
        "network_adapter"
    ][0]["component_candidate_id"]
    response["primary_recommendation"]["component_candidate_ids"] = component_ids
    return response


def _network_switch_only_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_recommendation": {
            "candidate_type": "build_from_parts",
            "title": "Network switch quote",
            "component_candidate_ids": {
                "switch": package["component_candidate_matrix"]["switch"][0][
                    "component_candidate_id"
                ],
            },
            "why_selected": "Cheapest switch candidate from the component matrix.",
            "engineer_checks": ["Проверить сетевые характеристики перед КП."],
        },
        "general_notes": [],
    }


def _network_switch_platform_alias_response(package: dict[str, Any]) -> dict[str, Any]:
    response = _network_switch_only_response(package)
    switch_id = response["primary_recommendation"]["component_candidate_ids"].pop("switch")
    response["primary_recommendation"]["component_candidate_ids"]["platform"] = switch_id
    return response


def _network_switch_with_server_checks_response(package: dict[str, Any]) -> dict[str, Any]:
    response = _network_switch_only_response(package)
    response["primary_recommendation"]["engineer_checks"] = [
        "Проверить CPU support list платформы и версию BIOS.",
        "Проверить QVL RAM и правила заполнения DIMM.",
        "Проверить NVMe/U.2/U.3 backplane.",
        "Проверить комплектацию БП, кулеры, рейки и кабели.",
        "Проверить портовую схему access/uplink и PoE budget.",
    ]
    return response


def _network_unknown_switch_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_recommendation": {
            "candidate_type": "build_from_parts",
            "title": "Network switch quote",
            "component_candidate_ids": {"switch": "switch-not-in-matrix"},
            "why_selected": "Invalid selection should be rejected.",
            "engineer_checks": ["Check network requirements before quote."],
        },
        "general_notes": [],
    }


def _network_transceiver_platform_alias_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_recommendation": {
            "candidate_type": "build_from_parts",
            "title": "Wrong network alias quote",
            "component_candidate_ids": {
                "platform": package["component_candidate_matrix"]["transceiver"][0][
                    "component_candidate_id"
                ],
            },
            "why_selected": "Wrong alias target should be rejected.",
            "engineer_checks": ["Check network requirements before quote."],
        },
        "general_notes": [],
    }


def _network_full_response(package: dict[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "primary_recommendation": {
            "candidate_type": "build_from_parts",
            "title": "Network quote",
            "component_candidate_ids": {
                "switch": matrix["switch"][0]["component_candidate_id"],
                "transceiver": matrix["transceiver"][0]["component_candidate_id"],
                "license": matrix["license"][0]["component_candidate_id"],
                "support": matrix["support"][0]["component_candidate_id"],
            },
            "why_selected": "Cheapest complete stocked network quote.",
            "engineer_checks": ["Проверить сетевые характеристики перед КП."],
        },
        "general_notes": [],
    }


def _storage_alias_response(alias: str) -> Any:
    def response(package: dict[str, Any]) -> dict[str, Any]:
        result = _storage_full_response(package)
        component_ids = dict(result["primary_recommendation"]["component_candidate_ids"])
        storage_id = component_ids.pop("storage_system")
        component_ids[alias] = storage_id
        result["primary_recommendation"]["component_candidate_ids"] = component_ids
        return result

    return response


def _primary_recommendation_with_gpu_response(package: dict[str, Any]) -> dict[str, Any]:
    response = _primary_recommendation_response(package)
    component_ids = dict(response["primary_recommendation"]["component_candidate_ids"])
    component_ids["gpu"] = package["component_candidate_matrix"]["gpu"][0][
        "component_candidate_id"
    ]
    response["primary_recommendation"]["component_candidate_ids"] = component_ids
    return response


def _primary_recommendation_with_controller_response(
    package: dict[str, Any],
) -> dict[str, Any]:
    response = _primary_recommendation_response(package)
    component_ids = dict(response["primary_recommendation"]["component_candidate_ids"])
    component_ids["storage_controller"] = package["component_candidate_matrix"][
        "storage_controller"
    ][0]["component_candidate_id"]
    response["primary_recommendation"]["component_candidate_ids"] = component_ids
    return response


def _base_platform_primary_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_base_platform",
        why_selected="Primary cheapest quote before repair.",
        why_selected_short="Primary quote before repair.",
    )
    recommendation["proposal_role"] = "cheapest_fit"
    recommendation["recommendation_slot"] = "price_optimal"
    recommendation["component_candidate_ids"]["platform"] = (
        _required_component_candidate_id_by_part(package, "platform", "BASE-ELIGIBLE")
    )
    recommendation["quantities"] = {"platform": 2, "cpu": 4, "ram": 16, "storage": 4}
    return {"recommendations": [recommendation], "general_notes": []}


def _base_platform_with_samsung_ram_primary_response(
    package: dict[str, Any],
) -> dict[str, Any]:
    response = _base_platform_primary_response(package)
    recommendation = response["recommendations"][0]
    recommendation["component_candidate_ids"]["ram"] = (
        _required_component_candidate_id_by_part(package, "ram", "RAM-SAMSUNG-32G")
    )
    recommendation["quantities"] = {"platform": 2, "cpu": 4, "ram": 32, "storage": 4}
    return response


def _micron_ram_repair_response(package: dict[str, Any]) -> dict[str, Any]:
    original = dict(package["original_accepted_proposals"][0])
    component_ids = dict(original["component_candidate_ids"])
    component_ids["ram"] = package["allowed_candidate_alternatives"]["ram"][0][
        "component_candidate_id"
    ]
    return {
        "recommendations": [
            {
                "recommendation_id": "llm_repaired_micron_ram",
                "proposal_role": "cheapest_fit",
                "recommendation_slot": "price_optimal",
                "source_type": "build_from_parts",
                "source_candidate_id": None,
                "component_candidate_ids": component_ids,
                "quantities": {"platform": 2, "cpu": 4, "ram": 32, "storage": 4},
                "decision": "recommend",
                "title": "Repaired cheapest quote",
                "why_selected": "Uses cheaper equivalent Micron 32GB DDR5 RAM.",
                "why_selected_short": "Cheaper equivalent RAM.",
                "right_size_note": "Minimal sufficient fit; code materializes quantities.",
                "commercial_tradeoff": "Micron is cheaper with sufficient stock.",
                "critical_checks": ["Check platform support list."],
                "engineering_review_required": True,
                "confidence": "medium",
            }
        ],
        "general_notes": [],
    }


def _gooxi_platform_repair_response(package: dict[str, Any]) -> dict[str, Any]:
    original = dict(package["original_accepted_proposals"][0])
    component_ids = dict(original["component_candidate_ids"])
    component_ids["platform"] = package["allowed_candidate_alternatives"]["platform"][0][
        "component_candidate_id"
    ]
    return {
        "recommendations": [
            {
                "recommendation_id": "llm_repaired_gooxi_platform",
                "proposal_role": "cheapest_fit",
                "recommendation_slot": "price_optimal",
                "source_type": "build_from_parts",
                "source_candidate_id": None,
                "component_candidate_ids": component_ids,
                "quantities": {"platform": 2, "cpu": 4, "ram": 16, "storage": 4},
                "decision": "recommend",
                "title": "Repaired cheapest Gooxi quote",
                "why_selected": "Uses cheaper eligible Gooxi platform.",
                "why_selected_short": "Cheaper eligible platform.",
                "right_size_note": "Minimal sufficient fit; code materializes quantities.",
                "commercial_tradeoff": "Gooxi passes hard eligibility and is cheaper.",
                "critical_checks": ["Check platform support list."],
                "engineering_review_required": True,
                "confidence": "medium",
            }
        ],
        "general_notes": [],
    }


def _blocked_platform_repair_response(package: dict[str, Any]) -> dict[str, Any]:
    original = dict(package["original_accepted_proposals"][0])
    component_ids = dict(original["component_candidate_ids"])
    component_ids["platform"] = "platform-blocked-incomplete"
    if package["allowed_candidate_alternatives"].get("ram"):
        component_ids["ram"] = package["allowed_candidate_alternatives"]["ram"][0][
            "component_candidate_id"
        ]
    return {
        "recommendations": [
            {
                "recommendation_id": "llm_bad_dell_repair",
                "proposal_role": "cheapest_fit",
                "recommendation_slot": "price_optimal",
                "source_type": "build_from_parts",
                "source_candidate_id": None,
                "component_candidate_ids": component_ids,
                "quantities": {"platform": 2, "cpu": 4, "ram": 32, "storage": 4},
                "decision": "recommend",
                "title": "Bad repaired quote",
                "why_selected": "Attempts to use a blocked cheap incomplete chassis.",
                "why_selected_short": "Blocked chassis.",
                "right_size_note": "Invalid repair.",
                "commercial_tradeoff": "Should be rejected by repair guard.",
                "critical_checks": ["Check platform support list."],
                "engineering_review_required": True,
                "confidence": "medium",
            }
        ],
        "general_notes": [],
    }


def _repair_invalid_json_response(package: dict[str, Any]) -> dict[str, Any]:
    raise LlmInvalidJsonError(
        "LLM response content was not valid JSON.",
        parse_stage="message_content",
        json_extract_status="parse_error",
        invalid_json_reason="not-json",
    )


def _gooxi_and_supermicro_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendations: list[dict[str, Any]] = []
    for recommendation_id, part_number, role, slot, why in [
        (
            "llm_gooxi_cheapest",
            "GOOXI-CHEAP",
            "cheapest_fit",
            "price_optimal",
            "Gooxi is the cheapest valid platform in the same family.",
        ),
        (
            "llm_supermicro_brand",
            "SYS-621C-TN12R",
            "alternative_platform",
            "alternative",
            "Supermicro is kept as a branded safe alternative.",
        ),
    ]:
        recommendation = _component_matrix_llm_recommendation(
            package,
            recommendation_id=recommendation_id,
            why_selected=why,
            why_selected_short=why,
        )
        recommendation["proposal_role"] = role
        recommendation["recommendation_slot"] = slot
        recommendation["component_candidate_ids"]["platform"] = (
            _required_component_candidate_id_by_part(package, "platform", part_number)
        )
        recommendations.append(recommendation)
    return {"recommendations": recommendations, "general_notes": []}


def _same_component_base_platforms_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendations: list[dict[str, Any]] = []
    for recommendation_id, platform_part, why in [
        ("llm_group_budget", "PLATFORM-CHEAP", "Самая низкая цена платформы."),
        ("llm_group_clean", "SYS-621C-TN12R", "Брендовая платформа с понятной проверкой."),
        ("llm_group_alt", "R283-ZK0", "Альтернативная платформа в той же семье."),
    ]:
        recommendation = _component_matrix_llm_recommendation(
            package,
            recommendation_id=recommendation_id,
            why_selected=why,
            why_selected_short=why,
        )
        recommendation["component_candidate_ids"]["platform"] = (
            _required_component_candidate_id_by_part(package, "platform", platform_part)
        )
        recommendations.append(recommendation)
    return {"recommendations": recommendations, "general_notes": []}


def _different_component_bases_response(package: dict[str, Any]) -> dict[str, Any]:
    base = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_base_cpu_selected",
        why_selected="Базовый CPU из матрицы.",
        why_selected_short="Базовый CPU из матрицы.",
    )
    base["component_candidate_ids"]["cpu"] = _required_component_candidate_id_by_part(
        package,
        "cpu",
        "CPU-SELECTED",
    )
    alternative = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_base_cpu_16",
        why_selected="Компромисс по CPU для отдельной компонентной базы.",
        why_selected_short="Отдельная база с другим CPU.",
    )
    alternative["proposal_role"] = "explicit_tradeoff"
    alternative["commercial_tradeoff"] = "Другой CPU меняет компонентную базу и цену."
    alternative["component_candidate_ids"]["cpu"] = _required_component_candidate_id_by_part(
        package,
        "cpu",
        "CPU-16C",
    )
    return {"recommendations": [base, alternative], "general_notes": []}


def _intel_and_amd_family_response(package: dict[str, Any]) -> dict[str, Any]:
    intel = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_family_intel",
        why_selected="Intel LGA4677 family.",
        why_selected_short="Intel LGA4677 family.",
    )
    amd = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_family_amd",
        why_selected="AMD SP5 family.",
        why_selected_short="AMD SP5 family.",
    )
    amd["component_candidate_ids"]["platform"] = _required_component_candidate_id_by_part(
        package,
        "platform",
        "AMD-SP5-PLATFORM",
    )
    amd["component_candidate_ids"]["cpu"] = _required_component_candidate_id_by_part(
        package,
        "cpu",
        "EPYC-CPU",
    )
    return {"recommendations": [intel, amd], "general_notes": []}


def _same_platform_storage_no_tradeoff_response(package: dict[str, Any]) -> dict[str, Any]:
    primary = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_storage_primary",
        why_selected="Базовый NVMe SSD закрывает требование.",
        why_selected_short="Базовый NVMe SSD закрывает требование.",
    )
    storage_alt = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_storage_alt_without_tradeoff",
        why_selected="Другой SSD без существенной причины.",
        why_selected_short="Другой SSD без существенной причины.",
    )
    storage_alt["component_candidate_ids"]["storage"] = (
        _required_component_candidate_id_by_part(package, "ssd", "SSD-ALT")
    )
    return {"recommendations": [primary, storage_alt], "general_notes": []}


def _cjk_and_duplicate_checks_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_cjk_cleaning",
        why_selected="Платформа с高密度 NVMe закрывает запрос.",
        why_selected_short="Платформа с高密度 NVMe закрывает запрос.",
    )
    recommendation["commercial_tradeoff"] = (
        "tradeoffs: с高密度 NVMe без смены CPU component_candidate_id: secret-id"
    )
    recommendation["critical_checks"] = [
        "Проверить совместимость CPU с платформой.",
        "Проверить список поддерживаемых CPU платформы.",
        "Проверить QVL памяти.",
        "Проверить совместимость RAM.",
        "raw JSON: component_candidate_id: secret-id",
    ]
    recommendation["engineer_checks"] = [
        "Проверить CPU support list / BIOS.",
        "Проверить QVL RAM.",
    ]
    return {"recommendations": [recommendation], "general_notes": []}


def _ram_and_storage_tradeoff_response(package: dict[str, Any]) -> dict[str, Any]:
    base = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_base_32g_3840",
        why_selected="Минимальная базовая конфигурация на 32 ГБ DIMM.",
        why_selected_short="Минимальная базовая конфигурация на 32 ГБ DIMM.",
    )
    base["component_candidate_ids"]["ram"] = _required_component_candidate_id_by_part(
        package,
        "ram",
        "RAM-32G",
    )

    ram_tradeoff = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_tradeoff_64g",
        why_selected="Вариант с 64 ГБ DIMM уменьшает число занятых слотов.",
        why_selected_short="Вариант с 64 ГБ DIMM уменьшает число занятых слотов.",
    )
    ram_tradeoff["proposal_role"] = "explicit_tradeoff"
    ram_tradeoff["commercial_tradeoff"] = (
        "64 ГБ DIMM уменьшает число занятых RAM/DIMM-слотов, но меняет компонентную базу."
    )
    ram_tradeoff["component_candidate_ids"]["ram"] = _required_component_candidate_id_by_part(
        package,
        "ram",
        "RAM-64G",
    )

    storage_tradeoff = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_tradeoff_7680",
        why_selected="Вариант с SSD 7.68 ТБ дает запас по емкости.",
        why_selected_short="Вариант с SSD 7.68 ТБ дает запас по емкости.",
    )
    storage_tradeoff["proposal_role"] = "explicit_tradeoff"
    storage_tradeoff["commercial_tradeoff"] = (
        "SSD 7.68 ТБ дороже, зато дает запас по емкости накопителей."
    )
    storage_tradeoff["component_candidate_ids"]["ram"] = (
        _required_component_candidate_id_by_part(package, "ram", "RAM-32G")
    )
    storage_tradeoff["component_candidate_ids"]["storage"] = (
        _required_component_candidate_id_by_part(package, "ssd", "SSD-7680")
    )

    return {"recommendations": [base, ram_tradeoff, storage_tradeoff], "general_notes": []}


def _missing_platform_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_missing_platform",
        why_selected="LLM omitted the mandatory platform.",
        why_selected_short="Missing platform.",
    )
    recommendation["component_candidate_ids"].pop("platform")
    recommendation["quantities"].pop("platform")
    return {"recommendations": [recommendation], "general_notes": []}


def _selected_component_ids_alias_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_selected_ids_alias",
        why_selected="LLM used the Qwen strict schema alias for the core BOM.",
        why_selected_short="Core BOM selected from matrix IDs.",
    )
    recommendation["selected_component_candidate_ids"] = recommendation.pop(
        "component_candidate_ids"
    )
    recommendation["proposal_role"] = "cheapest_fit"
    recommendation["commercial_tradeoff"] = "Cheapest valid core BOM."
    recommendation["engineer_checks"] = ["Check vendor QVL before quotation."]
    recommendation["engineering_confidence"] = "preliminary_requires_engineer_review"
    return {"recommendations": [recommendation], "general_notes": []}


def _selected_storage_alias_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_selected_storage_alias",
        why_selected="LLM used selected_component_candidate_ids with storage alias.",
        why_selected_short="Core BOM selected from matrix IDs.",
    )
    recommendation["selected_component_candidate_ids"] = recommendation.pop(
        "component_candidate_ids"
    )
    recommendation["selected_component_candidate_ids"]["storage"] = (
        recommendation["selected_component_candidate_ids"].pop("storage")
    )
    return {"recommendations": [recommendation], "general_notes": []}


def _eight_proposals_one_qwen_alias_response(package: dict[str, Any]) -> dict[str, Any]:
    proposals = []
    for index in range(8):
        recommendation = _component_matrix_llm_recommendation(
            package,
            recommendation_id=f"llm_qwen_alias_{index}",
            why_selected=f"Valid proposal {index}.",
            why_selected_short=f"Valid proposal {index}.",
        )
        if index == 3:
            recommendation["proposal_role"] = "lower_price_with_tradeoff"
            recommendation["recommendation_slot"] = "lower_price_with_tradeoff"
        proposals.append(recommendation)
    return {"recommendations": proposals, "general_notes": []}


def _one_bad_schema_one_valid_qwen_response(package: dict[str, Any]) -> dict[str, Any]:
    invalid = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_bad_schema",
        why_selected="Invalid schema should not drop the valid proposal.",
        why_selected_short="Invalid schema.",
    )
    invalid["confidence"] = "certain"
    invalid["proposal_role"] = "lower_price_with_tradeoff"
    valid = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_valid_after_bad_schema",
        why_selected="Valid proposal survives per-item schema validation.",
        why_selected_short="Valid proposal.",
    )
    return {"recommendations": [invalid, valid], "general_notes": []}


def _double_ram_quantity_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_double_ram_quantity",
        why_selected="LLM selected valid component IDs but overestimated RAM quantity.",
        why_selected_short="Selected valid component IDs.",
    )
    recommendation["quantities"]["ram"] = 64
    return {"recommendations": [recommendation], "general_notes": []}


def _underselected_ssd_quantity_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_underselected_ssd_quantity",
        why_selected="LLM selected valid component IDs but undercounted SSD quantity.",
        why_selected_short="Selected valid component IDs.",
    )
    recommendation["quantities"]["storage"] = 2
    recommendation["quantities"]["ssd"] = 2
    return {"recommendations": [recommendation], "general_notes": []}


def _optional_peripherals_in_core_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_optional_wrong_core",
        why_selected="Core server is valid; controller and NIC are useful checks only.",
        why_selected_short="Core server is valid; optional add-ons need engineer review.",
    )
    recommendation["confidence"] = "high"
    recommendation["component_candidate_ids"]["storage_controller"] = (
        _required_component_candidate_id_by_part(
            package,
            "storage_controller",
            "RAID-OPTION",
        )
    )
    recommendation["component_candidate_ids"]["network_adapter"] = (
        _required_component_candidate_id_by_part(
            package,
            "network_adapter",
            "NIC-25G",
        )
    )
    recommendation["quantities"]["storage_controller"] = 2
    recommendation["quantities"]["network_adapter"] = 2
    return {"recommendations": [recommendation], "general_notes": []}


def _optional_only_duplicate_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    core = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_core_without_optional",
        why_selected="Core BOM only.",
        why_selected_short="Core BOM only.",
    )
    with_optional = _optional_peripherals_in_core_llm_response(package)["recommendations"][0]
    with_optional["recommendation_id"] = "llm_core_with_optional_peripherals"
    with_optional["why_selected"] = "Same core BOM with optional peripherals."
    with_optional["why_selected_short"] = "Same core BOM with optional peripherals."
    return {"recommendations": [core, with_optional], "general_notes": []}


def _online_composer_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_online_matrix_build",
        why_selected="Online composer selected IDs from the component matrix.",
        why_selected_short="Выбрано из складской матрицы после проверки источников.",
    )
    recommendation["evidence_summary"] = {
        "status": "confirmed",
        "sources_count": 3,
        "confirmed_facts": ["DDR5", "LGA4677", "NVMe"],
        "not_confirmed": [],
        "source_domains": ["asus.com", "intel.com", "kioxia.com"],
        "notes": "Official source domains found.",
    }
    return {"recommendations": [recommendation], "general_notes": []}


def _online_composer_no_sources_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_online_no_sources_build",
        why_selected="Online composer selected IDs but did not find relation sources.",
        why_selected_short="Selected from stock matrix; external relation sources were not found.",
    )
    recommendation["evidence_summary"] = {
        "status": "not_confirmed",
        "sources_count": 0,
        "confirmed_facts": [],
        "not_confirmed": ["External relation sources were not found."],
        "source_domains": [],
        "notes": "No online sources found.",
    }
    return {"recommendations": [recommendation], "general_notes": []}


def _cpu_32_overfit_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_cpu_32_overfit",
        why_selected="LLM selected the 32-core CPU.",
        why_selected_short="Выбран 32-ядерный CPU.",
    )
    for candidate in package["component_candidate_matrix"]["cpu"]:
        if candidate["part_number"] == "CPU-32C":
            recommendation["component_candidate_ids"]["cpu"] = candidate[
                "component_candidate_id"
            ]
            break
    return {"recommendations": [recommendation], "general_notes": []}


def _two_slot_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    price = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_price",
        why_selected="Самый дешевый валидированный вариант.",
        why_selected_short="Самый дешевый валидированный вариант.",
    )
    price["recommendation_slot"] = "price_optimal"
    price["title"] = "Оптимальный по цене вариант"
    technical = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_technical",
        why_selected="Технически более чистый валидированный вариант.",
        why_selected_short="Технически более чистый валидированный вариант.",
    )
    cpu_20_id = _component_candidate_id_by_part(package, "cpu", "CPU-20C")
    if cpu_20_id:
        technical["component_candidate_ids"]["cpu"] = cpu_20_id
    technical["recommendation_slot"] = "technical_clean"
    technical["title"] = "Технически более чистый вариант"
    return {"recommendations": [price, technical], "general_notes": []}


def _confirmed_and_not_found_evidence_response(package: dict[str, Any]) -> dict[str, Any]:
    no_evidence = _pool_recommendation(
        package,
        recommendation_id="llm_no_evidence",
        platform_part="PLATFORM-CHEAP",
        cpu_part="CPU-16C",
        recommendation_slot="price_optimal",
        why_selected="Cheaper platform without external evidence.",
    )
    confirmed = _pool_recommendation(
        package,
        recommendation_id="llm_confirmed_evidence",
        platform_part="SYS-621C-TN12R",
        cpu_part="CPU-16C",
        recommendation_slot="technical_clean",
        why_selected="Platform has external DDR5 evidence.",
    )
    return {"recommendations": [no_evidence, confirmed], "general_notes": []}


def _unknown_component_id_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_matrix_unknown_component",
        why_selected="Uses an unknown component.",
    )
    recommendation["component_candidate_ids"]["cpu"] = "missing-component-id"
    return {"recommendations": [recommendation], "general_notes": []}


def _wrong_component_role_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_matrix_wrong_role",
        why_selected="Uses a platform id as CPU.",
    )
    recommendation["component_candidate_ids"]["cpu"] = recommendation["component_candidate_ids"][
        "platform"
    ]
    return {"recommendations": [recommendation], "general_notes": []}


def _missing_ram_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id="llm_matrix_partial_ram",
        source_type="build_from_parts",
        why_selected="Platform, CPU and SSD are stocked; RAM must be sourced separately.",
    )
    recommendation["component_candidate_ids"].pop("ram")
    recommendation["quantities"].pop("ram")
    recommendation["what_is_missing"] = ["ram"]
    return {"recommendations": [recommendation], "general_notes": []}


def _one_invalid_one_valid_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    invalid = _unknown_component_id_llm_response(package)["recommendations"][0]
    valid = _component_matrix_llm_response(package)["recommendations"][0]
    return {"recommendations": [invalid, valid], "general_notes": []}


def _proposal_pool_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    proposals = [
        _pool_recommendation(
            package,
            recommendation_id="llm_pool_technical",
            platform_part="SYS-621C-TN12R",
            cpu_part="CPU-16C",
            recommendation_slot="technical_clean",
            why_selected="Clean platform evidence with DDR5/NVMe/2U/PSU hints.",
        ),
        _invalid_pool_recommendation(package, "llm_pool_invalid_1"),
        _pool_recommendation(
            package,
            recommendation_id="llm_pool_price",
            platform_part="PLATFORM-CHEAP",
            cpu_part="CPU-16C",
            recommendation_slot="price_optimal",
            why_selected="Cheapest full stocked BOM in the proposal pool.",
            critical_checks=["Cheaper platform needs PSU bundle review."],
        ),
        _pool_recommendation(
            package,
            recommendation_id="llm_pool_alternative",
            platform_part="R283-ZK0",
            cpu_part="CPU-20C",
            recommendation_slot="alternative_vendor_or_platform",
            why_selected="Different stocked platform and nearby CPU option.",
            critical_checks=["Alternative platform needs vendor support review."],
        ),
    ]
    proposals.extend(
        _invalid_pool_recommendation(package, f"llm_pool_invalid_{index}")
        for index in range(2, 8)
    )
    return {"recommendations": proposals, "general_notes": []}


def _one_safe_proposal_pool_response(package: dict[str, Any]) -> dict[str, Any]:
    proposals = [
        _pool_recommendation(
            package,
            recommendation_id="llm_pool_price",
            platform_part="PLATFORM-CHEAP",
            cpu_part="CPU-16C",
            recommendation_slot="price_optimal",
            why_selected="Only safe proposal in this pool.",
        )
    ]
    proposals.extend(
        _invalid_pool_recommendation(package, f"llm_pool_invalid_{index}")
        for index in range(1, 10)
    )
    return {"recommendations": proposals, "general_notes": []}


def _invalid_proposal_pool_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendations": [
            _invalid_pool_recommendation(package, f"llm_pool_invalid_{index}")
            for index in range(1, 11)
        ],
        "general_notes": [],
    }


def _duplicate_bom_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    first = _pool_recommendation(
        package,
        recommendation_id="llm_duplicate_1",
        platform_part="PLATFORM-CHEAP",
        cpu_part="CPU-16C",
        recommendation_slot="price_optimal",
        why_selected="First copy of the same BOM.",
    )
    second = dict(first)
    second["recommendation_id"] = "llm_duplicate_2"
    second["why_selected"] = "Second copy of the same BOM."
    return {"recommendations": [first, second], "general_notes": []}


def _duplicate_five_bom_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    first = _pool_recommendation(
        package,
        recommendation_id="llm_duplicate_1",
        platform_part="PLATFORM-CHEAP",
        cpu_part="CPU-16C",
        recommendation_slot="price_optimal",
        why_selected="Best representative of the same BOM.",
    )
    recommendations = [first]
    for index in range(2, 6):
        duplicate = dict(first)
        duplicate["recommendation_id"] = f"llm_duplicate_{index}"
        duplicate["why_selected"] = f"Duplicate copy {index} of the same BOM."
        recommendations.append(duplicate)
    return {"recommendations": recommendations, "general_notes": []}


def _worse_by_price_pool_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    cheapest = _pool_recommendation(
        package,
        recommendation_id="llm_worse_cheapest",
        platform_part="PLATFORM-CHEAP",
        cpu_part="CPU-16C",
        recommendation_slot="price_optimal",
        why_selected="Cheapest right-sized stocked BOM.",
    )
    worse = []
    for index, platform_part in enumerate(
        ["SYS-621C-TN12R", "R283-ZK0", "RS521A-E12-RS24U"],
        start=1,
    ):
        worse.append(
            _pool_recommendation(
                package,
                recommendation_id=f"llm_worse_{index}",
                platform_part=platform_part,
                cpu_part="CPU-32C",
                recommendation_slot="lower_price_with_tradeoff",
                why_selected="More expensive overfit CPU option.",
            )
        )
    return {"recommendations": [cheapest, *worse], "general_notes": []}


def _mixed_invalid_duplicate_pool_response(package: dict[str, Any]) -> dict[str, Any]:
    duplicates = _duplicate_five_bom_llm_response(package)["recommendations"][:3]
    invalid = [
        _invalid_pool_recommendation(package, "llm_mixed_invalid_1"),
        _invalid_pool_recommendation(package, "llm_mixed_invalid_2"),
    ]
    return {"recommendations": [*invalid, *duplicates], "general_notes": []}


def _pool_recommendation(
    package: dict[str, Any],
    *,
    recommendation_id: str,
    platform_part: str,
    cpu_part: str,
    recommendation_slot: str,
    why_selected: str,
    critical_checks: list[str] | None = None,
) -> dict[str, Any]:
    recommendation = _component_matrix_llm_recommendation(
        package,
        recommendation_id=recommendation_id,
        why_selected=why_selected,
        why_selected_short=why_selected,
    )
    recommendation["recommendation_slot"] = recommendation_slot
    recommendation["title"] = f"LLM proposal {recommendation_id}"
    platform_component_id = _required_component_candidate_id_by_part(
        package, "platform", platform_part
    )
    recommendation["component_candidate_ids"]["platform"] = platform_component_id
    recommendation["component_candidate_ids"]["cpu"] = _required_component_candidate_id_by_part(
        package,
        "cpu",
        cpu_part,
    )
    recommendation["critical_checks"] = critical_checks or []
    return recommendation


def _invalid_pool_recommendation(
    package: dict[str, Any],
    recommendation_id: str,
) -> dict[str, Any]:
    recommendation = _pool_recommendation(
        package,
        recommendation_id=recommendation_id,
        platform_part="PLATFORM-CHEAP",
        cpu_part="CPU-16C",
        recommendation_slot="lower_price_with_tradeoff",
        why_selected="Invalid proposal uses an unknown component.",
    )
    recommendation["component_candidate_ids"]["cpu"] = f"missing-{recommendation_id}"
    return recommendation


def _component_candidate_id_by_part(
    package: dict[str, Any],
    role: str,
    part_number: str,
) -> str | None:
    for candidate in package["component_candidate_matrix"][role]:
        if candidate["part_number"] == part_number:
            return candidate["component_candidate_id"]
    return None


def _required_component_candidate_id_by_part(
    package: dict[str, Any],
    role: str,
    part_number: str,
) -> str:
    component_id = _component_candidate_id_by_part(package, role, part_number)
    if component_id is None:
        raise AssertionError(f"Missing {role} component with part {part_number}")
    return component_id


def _composer_requirements() -> dict[str, Any]:
    return {
        "server_qty": 2,
        "cpu_per_server": 2,
        "total_cpu_required": 4,
        "ram_gb_per_server": 512,
        "ram_type_preference": "DDR5",
        "storage_required": True,
        "storage_type_preference": "SSD",
        "storage_qty_per_server": 2,
        "optimization_mode": "cost_minimal_fit",
    }


def _right_size_composer_requirements() -> dict[str, Any]:
    requirements = _composer_requirements()
    requirements.update(
        {
            "cpu_min_cores_per_cpu": 16,
            "cpu_vendor_preference": "Intel",
            "cpu_family_preference": "Xeon",
            "storage_min_capacity_tb": 3.84,
            "storage_interface_preference": "NVMe",
        }
    )
    return requirements


def _semantic_amd_25gbe_role_plan() -> dict[str, Any]:
    required_capabilities = [
        {
            "capability_id": "power_supply.min_2",
            "role": "power_supply",
            "hard": True,
            "source_text": "2 блока питания",
            "parsed_requirements": {"psu_count_per_server": 2},
            "can_be_satisfied_by_platform": True,
        },
        {
            "capability_id": "cpu.amd_epyc.min_32_cores",
            "role": "cpu",
            "hard": True,
            "source_text": "2 процессора AMD EPYC, не менее 32 ядер на процессор",
            "parsed_requirements": {
                "vendor": "AMD",
                "family": "EPYC",
                "cpu_per_server": 2,
                "min_cores_per_cpu": 32,
            },
        },
        {
            "capability_id": "ram.ddr5_rdimm.min_768gb",
            "role": "ram",
            "hard": True,
            "source_text": "не менее 768 ГБ RAM DDR5 RDIMM",
            "parsed_requirements": {
                "min_gb_per_server": 768,
                "type": "DDR5",
                "form_factor": "RDIMM",
            },
        },
        {
            "capability_id": "storage.nvme.min_4x_7_68tb",
            "role": "storage",
            "hard": True,
            "source_text": "4 SSD NVMe не менее 7.68 ТБ на сервер",
            "parsed_requirements": {
                "drives_per_server": 4,
                "interface": "NVMe",
                "min_capacity_tb": 7.68,
            },
        },
        {
            "capability_id": "network.25gbe.sfp28",
            "role": "network_adapter",
            "hard": True,
            "source_text": "минимум 2 сетевых порта 25GbE SFP28",
            "can_be_satisfied_by_platform": True,
            "parsed_requirements": {
                "required": True,
                "min_ports_per_server": 2,
                "speed": "25GbE",
                "media": "SFP28",
            },
        },
    ]
    return {
        "product_group": "server",
        "requirements": [
            {
                "requirement_id": "ctx_1",
                "source_text": "под виртуализацию, базы данных и локальное NVMe-хранилище",
                "classification": "workload_context",
                "hard": False,
                "parsed_requirements": {},
            },
            {
                "requirement_id": "log_1",
                "source_text": "склад Москва",
                "classification": "logistics_constraint",
                "hard": False,
                "parsed_requirements": {"shipment_city": "Москва"},
            },
        ],
        "required_capabilities": required_capabilities,
        "optional_capabilities": [],
        "workload_context": [
            "под виртуализацию, базы данных и локальное NVMe-хранилище"
        ],
        "logistics_constraints": {"shipment_city": "Москва"},
        "commercial_instructions": [
            {
                "source_text": "один самый дешевый складской вариант для КП",
                "parsed_requirements": {
                    "optimization_goal": "cheapest_valid_stock_quote"
                },
            }
        ],
        "response_instructions": [],
        "engineer_review_required": True,
        "unsupported_or_unmapped_requirements": [],
        "planner_warnings": [],
        "required_roles": [
            "server_platform",
            "cpu",
            "ram",
            "storage",
            "network_adapter",
            "power_supply",
        ],
        "optional_roles": [],
        "requirements_by_role": {
            row["role"]: row["parsed_requirements"] | {"required": True}
            for row in required_capabilities
        },
        "role_catalog": [
            "ready_server",
            "server_platform",
            "cpu",
            "ram",
            "storage",
            "network_adapter",
            "storage_controller",
            "gpu",
            "transceiver",
            "cable",
            "power_supply",
            "rail_kit",
            "license",
            "support",
            "other_accessory",
        ],
    }


def _network_composer_requirements() -> dict[str, Any]:
    requirements = _right_size_composer_requirements()
    requirements.update(
        {
            "required_roles": [
                "server_platform",
                "cpu",
                "ram",
                "storage",
                "network_adapter",
            ],
            "network_required": True,
            "network_min_ports_per_server": 2,
            "network_speed": "25GbE",
            "network_media": "SFP28",
            "network_requirement": {
                "required": True,
                "min_ports_per_server": 2,
                "speed": "25GbE",
                "media": "SFP28",
                "interface": "unknown",
            },
            "required_capabilities": [
                {
                    "capability_id": "network.25gbe.sfp28",
                    "role": "network_adapter",
                    "hard": True,
                    "source_text": "2 ports 25GbE SFP28 per server",
                    "parsed_requirements": {
                        "required": True,
                        "min_ports_per_server": 2,
                        "speed": "25GbE",
                        "media": "SFP28",
                    },
                }
            ],
        }
    )
    return requirements


def _network_switch_composer_requirements(
    *,
    include_transceiver: bool = False,
    include_license: bool = False,
    include_support: bool = False,
) -> dict[str, Any]:
    required_roles = ["switch"]
    capabilities: list[dict[str, Any]] = [
        {
            "capability_id": "switch.48.1gbe.rj45.4.10gbe.sfp+.poe.l3.stacking",
            "role": "switch",
            "hard": True,
            "source_text": "48 портов 1G PoE+, 4 uplink 10G SFP+, L3, stacking",
            "parsed_requirements": {
                "required": True,
                "count": 1,
                "device_count": 1,
                "port_count": 48,
                "port_speed": "1GbE",
                "port_media": "RJ45",
                "uplink_count": 4,
                "uplink_speed": "10GbE",
                "uplink_media": "SFP+",
                "poe_required": True,
                "poe_budget_w": 740,
                "poe_standard": "PoE+",
                "l3_required": True,
                "stacking_required": True,
            },
        }
    ]
    requirements_by_role: dict[str, dict[str, Any]] = {
        "switch": dict(capabilities[0]["parsed_requirements"])
    }
    if include_transceiver:
        required_roles.append("transceiver")
        transceiver = {
            "required": True,
            "count": 4,
            "port_speed": "10GbE",
            "transceiver_form_factor": "SFP+",
        }
        requirements_by_role["transceiver"] = transceiver
        capabilities.append(
            {
                "capability_id": "transceiver.10gbe.sfp+",
                "role": "transceiver",
                "hard": True,
                "source_text": "трансиверы в комплект",
                "parsed_requirements": transceiver,
            }
        )
    if include_license:
        required_roles.append("license")
        license_req = {"required": True, "count": 1, "term_years": 1}
        requirements_by_role["license"] = license_req
        capabilities.append(
            {
                "capability_id": "license.1y",
                "role": "license",
                "hard": True,
                "source_text": "лицензия 1 год",
                "parsed_requirements": license_req,
            }
        )
    if include_support:
        required_roles.append("support")
        support_req = {"required": True, "count": 1, "term_years": 1}
        requirements_by_role["support"] = support_req
        capabilities.append(
            {
                "capability_id": "support.1y",
                "role": "support",
                "hard": True,
                "source_text": "support 1 год",
                "parsed_requirements": support_req,
            }
        )
    return {
        "product_group": "network",
        "server_qty": 1,
        "device_qty": 1,
        "required_roles": required_roles,
        "required_capabilities": capabilities,
        "unsupported_or_unmapped_requirements": [],
        "role_plan": {
            "product_group": "network",
            "required_roles": required_roles,
            "requirements_by_role": requirements_by_role,
            "required_capabilities": capabilities,
        },
    }


def _power_supply_composer_requirements() -> dict[str, Any]:
    requirements = _right_size_composer_requirements()
    requirements["required_roles"] = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "power_supply",
    ]
    requirements["psu_count_per_server"] = 2
    requirements["required_capabilities"] = [
        {
            "capability_id": "power_supply.min_2",
            "role": "power_supply",
            "hard": True,
            "source_text": "2 PSU per server",
            "parsed_requirements": {"psu_count_per_server": 2},
            "can_be_satisfied_by_platform": True,
        }
    ]
    requirements["role_plan"] = {
        "required_capabilities": requirements["required_capabilities"],
        "required_roles": requirements["required_roles"],
        "requirements_by_role": {
            "power_supply": {
                "required": True,
                "psu_count_per_server": 2,
            }
        },
    }
    return requirements


def _gpu_composer_requirements() -> dict[str, Any]:
    requirements = _right_size_composer_requirements()
    requirements.update(
        {
            "required_roles": [
                "server_platform",
                "cpu",
                "ram",
                "storage",
                "gpu",
            ],
            "required_capabilities": [
                {
                    "capability_id": "gpu.requested",
                    "role": "gpu",
                    "hard": True,
                    "requirement_text": "Need NVIDIA GPU",
                    "parsed_requirements": {"required": True},
                }
            ],
        }
    )
    return requirements


def _storage_controller_composer_requirements() -> dict[str, Any]:
    requirements = _right_size_composer_requirements()
    requirements.update(
        {
            "required_roles": [
                "server_platform",
                "cpu",
                "ram",
                "storage",
                "storage_controller",
            ],
            "required_capabilities": [
                {
                    "capability_id": "storage_controller.requested",
                    "role": "storage_controller",
                    "hard": True,
                    "requirement_text": "Need RAID HBA",
                    "parsed_requirements": {"required": True},
                }
            ],
        }
    )
    return requirements


def _component_matrix_llm_recommendation(
    package: dict[str, Any],
    *,
    recommendation_id: str,
    why_selected: str,
    why_selected_short: str | None = None,
    source_type: str = "build_from_parts",
) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "recommendation_id": recommendation_id,
        "source_type": source_type,
        "source_candidate_id": None,
        "component_candidate_ids": {
            "platform": matrix["platform"][0]["component_candidate_id"],
            "cpu": matrix["cpu"][0]["component_candidate_id"],
            "ram": matrix["ram"][0]["component_candidate_id"],
            "storage": matrix["ssd"][0]["component_candidate_id"],
        },
        "quantities": {"platform": 2, "cpu": 4, "ram": 16, "storage": 4},
        "decision": "recommend",
        "title": "Matrix-composed build",
        "display_name": "Fabricated text must not be displayed as BOM",
        "why_selected": why_selected,
        "why_selected_short": why_selected_short or why_selected,
        "right_size_note": "Минимально закрывает требования по CPU, RAM и SSD.",
        "critical_checks": ["Проверить CPU support list платформы."],
        "engineering_review_required": True,
        "confidence": "medium",
    }


def _composer_component_matrix(
    *,
    platform_facts: dict[str, Any],
    cpu_facts: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "platform_candidates": [
            _composer_component_candidate(
                "platform-amd",
                "ASUS",
                "RS521A-E12-RS24U",
                "ASUS RS521A-E12-RS24U server platform",
                2,
                Decimal("2000"),
                platform_facts,
            )
        ],
        "cpu_candidates": [
            _composer_component_candidate(
                "cpu-selected",
                "CPUVendor",
                "CPU-SELECTED",
                "Selected server CPU",
                4,
                Decimal("500"),
                cpu_facts,
            )
        ],
        "ram_candidates": [
            _composer_component_candidate(
                "ram-64",
                "Samsung",
                "RAM-64G",
                "DDR5 RDIMM 64GB server memory module",
                16,
                Decimal("150"),
                {"ram_capacity_gb": 64, "ram_type": "DDR5"},
            )
        ],
        "ssd_candidates": [
            _composer_component_candidate(
                "ssd-3840",
                "KIOXIA",
                "SSD-3840",
                "KIOXIA CD8-R Server SSD NVMe 3.84TB",
                4,
                Decimal("300"),
                {"storage_capacity_tb": 3.84, "storage_interface": "NVMe"},
            )
        ],
    }


def _distiller_fallback_source_matrix() -> dict[str, Any]:
    matrix: dict[str, Any] = {
        "product_group": "server",
        "semantic_planner_source": "llm",
        "category_planner_source": "ai_category_planner",
        "category_plan": {
            "server_platform": ["test-platform"],
            "cpu": ["test-cpu"],
            "ram": ["test-ram"],
            "storage": ["test-ssd"],
        },
        "role_plan": {
            "product_group": "server",
            "required_roles": ["server_platform", "cpu", "ram", "ssd"],
            "required_capabilities": [],
        },
        **_composer_component_matrix(
            platform_facts={"socket_count": 2, "memory_type": "DDR5"},
            cpu_facts={"cpu_cores": 32, "normalized_vendor": "Intel"},
        ),
    }
    bulky_text = "x" * 15000
    for rows in matrix.values():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["fit_reason"] = bulky_text
            row["product_description"] = "Catalog product text"
            row["item_name_rus"] = row.get("name")
            row["product_name"] = row.get("name")
            row["catalog_path"] = [{"name": "Server components"}]
    return matrix


def _server_78_like_broad_matrix(*, rows_per_role: int) -> dict[str, Any]:
    matrix: dict[str, Any] = {
        "product_group": "server",
        "primary_object": "server",
        "semantic_planner_source": "llm",
        "semantic_planner_confidence": "high",
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
        "role_plan": {
            "product_group": "server",
            "primary_object": "server",
            "required_roles": [
                "server_platform",
                "cpu",
                "ram",
                "ssd",
                "storage_controller",
                "network_adapter",
                "power_supply",
                "cable",
            ],
            "required_capabilities": [],
        },
    }
    role_specs = [
        ("platform_candidates", "server_platform", "V110100", "ASUS", "PLATFORM"),
        ("cpu_candidates", "cpu", "V110103", "Intel", "CPU"),
        ("ram_candidates", "ram", "V110104", "Samsung", "RAM"),
        ("ssd_candidates", "ssd", "V110106", "KIOXIA", "SSD"),
        (
            "storage_controller_candidates",
            "storage_controller",
            "V110107",
            "LSI",
            "HBA",
        ),
        (
            "network_adapter_candidates",
            "network_adapter",
            "V120116",
            "Intel",
            "NIC",
        ),
        ("power_supply_candidates", "power_supply", "V110108", "Delta", "PSU"),
        ("cable_candidates", "cable", "V110109", "CableCo", "CABLE"),
    ]
    facts_by_role = {
        "server_platform": {"socket_count": 2, "memory_type": "DDR5"},
        "cpu": {"cpu_cores": 24, "normalized_vendor": "Intel"},
        "ram": {"ram_capacity_gb": 64, "ram_type": "DDR5"},
        "ssd": {"storage_capacity_tb": 1.92, "storage_interface": "SATA"},
        "storage_controller": {"controller_count": 1, "storage_interface": "SAS"},
        "network_adapter": {
            "network_ports_count": 2,
            "network_speed": "10GbE",
            "network_media": "SFP+",
        },
        "power_supply": {"power_w": 2000, "redundant_psu": True},
        "cable": {"cable_type": "C13-C14"},
    }
    for matrix_key, role, category_id, producer, part_prefix in role_specs:
        rows = []
        for index in range(rows_per_role):
            row = _composer_component_candidate(
                f"{role}-{index}",
                producer,
                f"{part_prefix}-{index:03d}",
                f"{producer} {role} candidate {index}",
                2,
                Decimal("100"),
                facts_by_role[role],
            )
            row.update(
                {
                    "component_candidate_id": f"{role}-{index}",
                    "role": role,
                    "category_id": category_id,
                    "category_name": f"{role} category",
                    "item_id": f"{role}-item-{index}",
                    "product_key": f"ocs:{role}-item-{index}",
                    "item_name": f"{producer} {role} candidate {index}",
                    "item_name_rus": f"{producer} {role} candidate {index}",
                    "product_name": f"{producer} {role} candidate {index}",
                    "product_description": " ".join([role] * 90),
                    "catalog_path": [{"name": "Server"}, {"name": role}],
                }
            )
            rows.append(row)
        matrix[matrix_key] = rows
    return matrix


def _server_85_like_broad_matrix() -> dict[str, Any]:
    matrix = _server_78_like_broad_matrix(rows_per_role=42)
    role_limits = {
        "platform_candidates": 26,
        "cpu_candidates": 13,
        "ram_candidates": 42,
        "ssd_candidates": 13,
        "storage_controller_candidates": 42,
        "network_adapter_candidates": 42,
        "power_supply_candidates": 10,
        "cable_candidates": 3,
    }
    for key, limit in role_limits.items():
        matrix[key] = matrix[key][:limit]
        for row in matrix[key]:
            row["available_quantity"] = 100
    matrix["full_matrix_evaluation_used"] = False
    matrix["full_matrix_evaluation_fallback_reason"] = (
        "full_matrix_evaluation_failed_but_package_under_budget"
    )
    matrix["broad_count_by_role"] = {
        "server_platform": 26,
        "cpu": 13,
        "ram": 42,
        "ssd": 13,
        "storage_controller": 42,
        "network_adapter": 42,
        "power_supply": 10,
        "cable": 3,
    }
    return matrix


def _storage_composer_requirements() -> dict[str, Any]:
    required_capabilities = [
        {
            "capability_id": "storage.usable_100tb.fc_32g",
            "role": "storage_system",
            "hard": True,
            "source_text": "СХД 100 ТБ usable, FC 32G",
            "parsed_requirements": {
                "usable_capacity_tb": 100,
                "host_protocol": "FC",
                "host_port_speed": "32G",
            },
        },
        {
            "capability_id": "drive.24x_7_68tb_ssd",
            "role": "drive",
            "hard": True,
            "source_text": "24 диска SSD 7.68 TB",
            "parsed_requirements": {
                "drive_count": 24,
                "drive_type": "SSD",
                "drive_capacity_tb": 7.68,
            },
        },
        {
            "capability_id": "support.36m",
            "role": "support",
            "hard": True,
            "source_text": "поддержка 3 года",
            "parsed_requirements": {
                "support_required": True,
                "warranty_months": 36,
            },
        },
    ]
    return {
        "product_group": "storage",
        "server_qty": 1,
        "system_qty": 1,
        "storage_required": True,
        "usable_capacity_tb": 100,
        "host_protocol": "FC",
        "host_port_speed": "32G",
        "drive_count": 24,
        "drive_type": "SSD",
        "drive_capacity_tb": 7.68,
        "support_required": True,
        "warranty_months": 36,
        "required_roles": ["storage_system", "drive", "support"],
        "required_capabilities": required_capabilities,
        "role_plan": {
            "product_group": "storage",
            "required_roles": ["storage_system", "drive", "support"],
            "required_capabilities": required_capabilities,
            "requirements_by_role": {
                "storage_system": {
                    "usable_capacity_tb": 100,
                    "host_protocol": "FC",
                    "host_port_speed": "32G",
                },
                "drive": {
                    "drive_count": 24,
                    "drive_type": "SSD",
                    "drive_capacity_tb": 7.68,
                },
                "support": {"support_required": True, "warranty_months": 36},
            },
        },
    }


def _storage_component_matrix() -> dict[str, list[dict[str, Any]]]:
    storage_system = _composer_component_candidate(
        "storage-system-100u",
        "StorageVendor",
        "ARR-100U",
        "Storage array 120TB usable dual controller FC 32G",
        1,
        Decimal("10000"),
        {
            "usable_capacity_tb": 120,
            "raw_capacity_tb": 200,
            "controller_count": 2,
            "host_protocol": "FC",
            "host_port_speed": "32G",
            "host_port_speed_gbps": 32,
            "raw": {"debug": True},
        },
    )
    storage_system.update(
        {
            "usable_capacity_tb": 120,
            "raw_capacity_tb": 200,
            "controller_count": 2,
            "host_protocol": "FC",
            "host_port_speed": "32G",
            "host_port_speed_gbps": 32,
        }
    )
    drive = _composer_component_candidate(
        "drive-ssd-768",
        "DiskVendor",
        "SSD-7680",
        "SSD drive 7.68TB SAS",
        24,
        Decimal("100"),
        {
            "drive_capacity_tb": 7.68,
            "drive_type": "SSD",
            "drive_interface": "SAS",
        },
    )
    drive.update(
        {
            "drive_capacity_tb": 7.68,
            "drive_type": "SSD",
            "drive_interface": "SAS",
        }
    )
    support = _composer_component_candidate(
        "support-3y",
        "StorageVendor",
        "SUP-3Y",
        "Storage support 3 years",
        1,
        Decimal("1000"),
        {"warranty_months": 36},
    )
    support["warranty_months"] = 36
    return {
        "product_group": "storage",
        "storage_system_candidates": [storage_system],
        "drive_candidates": [drive],
        "support_candidates": [support],
        "required_capabilities": _storage_composer_requirements()["required_capabilities"],
    }


def _storage_full_response(package: dict[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "primary_recommendation": {
            "candidate_type": "build_from_parts",
            "title": "Storage quote",
            "component_candidate_ids": {
                "storage_system": matrix["storage_system"][0]["component_candidate_id"],
                "drive": matrix["drive"][0]["component_candidate_id"],
                "support": matrix["support"][0]["component_candidate_id"],
            },
            "why_selected": "Cheapest complete stocked storage quote.",
            "engineer_checks": ["Проверить СХД инженером перед КП."],
        },
        "general_notes": [],
    }


def _composer_component_matrix_with_network(
    *,
    include_network: bool,
    platform_network_facts: dict[str, Any] | None = None,
    network_available_quantity: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    platform_facts = {
        "cpu_brand": "Intel",
        "cpu_family": "Xeon",
        "cpu_socket": "LGA4677",
        "ram_type": "DDR5",
        **(platform_network_facts or {}),
    }
    matrix = _composer_component_matrix(
        platform_facts=platform_facts,
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 24,
        },
    )
    if include_network:
        network = _composer_component_candidate(
            "nic-25g-dual",
            "Mellanox",
            "NIC-25G-DUAL",
            "Dual-port 25GbE SFP28 PCIe network adapter",
            network_available_quantity,
            Decimal("250"),
            {
                "network_ports_count": 2,
                "network_speed": "25GbE",
                "network_speed_gbps": 25,
                "network_media": "SFP28",
                "network_interface": "PCIe",
            },
        )
        network["quantity_required"] = 1
        network["available_quantity"] = network_available_quantity
        matrix["network_adapter_candidates"] = [network]
    return matrix


def _network_switch_component_matrix(
    *,
    close_switch: bool,
    include_transceiver: bool = False,
    include_license: bool = False,
    include_support: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    switch_facts = {
        "port_count": 48,
        "port_speed": "1GbE",
        "port_speed_gbps": 1,
        "port_media": "RJ45",
        "uplink_count": 4 if close_switch else 2,
        "uplink_speed": "10GbE" if close_switch else "1GbE",
        "uplink_speed_gbps": 10 if close_switch else 1,
        "uplink_media": "SFP+" if close_switch else "RJ45",
        "poe_supported": close_switch,
        "poe_budget_w": 740 if close_switch else 0,
        "poe_standard": "PoE+" if close_switch else "unknown",
        "l3_supported": close_switch,
        "stacking_supported": close_switch,
    }
    matrix: dict[str, Any] = {
        "product_group": "network",
        "switch_candidates": [
            _composer_component_candidate(
                "switch-48p",
                "NetVendor",
                "SW-48P-4SFP",
                "48x1G RJ45 PoE+ switch 740W 4x10G SFP+ L3 stacking",
                1,
                Decimal("1200"),
                switch_facts,
            )
        ],
    }
    if include_transceiver:
        matrix["transceiver_candidates"] = [
            _composer_component_candidate(
                "sfp-10g",
                "NetVendor",
                "SFP-10G-SR",
                "10G SFP+ transceiver module",
                4,
                Decimal("80"),
                {
                    "port_speed": "10GbE",
                    "port_speed_gbps": 10,
                    "transceiver_form_factor": "SFP+",
                },
            )
        ]
    if include_license:
        matrix["license_candidates"] = [
            _composer_component_candidate(
                "license-1y",
                "NetVendor",
                "LIC-1Y",
                "Switch license subscription 1 year",
                1,
                Decimal("100"),
                {},
            )
        ]
    if include_support:
        matrix["support_candidates"] = [
            _composer_component_candidate(
                "support-1y",
                "NetVendor",
                "SUP-1Y",
                "Switch support 1 year",
                1,
                Decimal("120"),
                {},
            )
        ]
    return matrix


def _composer_component_matrix_with_gpu(
    *,
    include_gpu: bool,
) -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix_with_network(include_network=False)
    if include_gpu:
        matrix["gpu_candidates"] = [
            _composer_component_candidate(
                "gpu-l40s",
                "NVIDIA",
                "GPU-L40S",
                "NVIDIA L40S GPU accelerator",
                2,
                Decimal("1500"),
                {},
            )
        ]
    return matrix


def _composer_component_matrix_with_storage_controller() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix_with_network(include_network=False)
    matrix["storage_controller_candidates"] = [
        _composer_component_candidate(
            "raid-hba",
            "Broadcom",
            "RAID-HBA",
            "Broadcom tri-mode RAID HBA controller",
            2,
            Decimal("350"),
            {},
        )
    ]
    return matrix


def _composer_component_matrix_with_ram_module(
    ram_module_gb: int,
    *,
    available_quantity: int,
) -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 24,
        },
    )
    matrix["ram_candidates"] = [
        _composer_component_candidate(
            f"ram-{ram_module_gb}",
            "Micron",
            f"RAM-{ram_module_gb}G",
            f"Micron DDR5 RDIMM {ram_module_gb}GB server memory module",
            available_quantity,
            Decimal("100"),
            {"ram_capacity_gb": ram_module_gb, "ram_type": "DDR5"},
        )
    ]
    matrix["ram_candidates"][0]["quantity_required"] = available_quantity
    matrix["ram_candidates"][0]["available_quantity"] = available_quantity
    matrix["ram_candidates"][0]["ram_module_capacity_gb"] = ram_module_gb
    return matrix


def _composer_component_matrix_with_optional_peripherals() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 24,
        },
    )
    matrix["storage_controller_candidates"] = [
        _composer_component_candidate(
            "controller-raid",
            "Broadcom",
            "RAID-OPTION",
            "Broadcom RAID controller optional add-on",
            2,
            Decimal("9000"),
            {},
        )
    ]
    matrix["network_adapter_candidates"] = [
        _composer_component_candidate(
            "nic-25g",
            "Mellanox",
            "NIC-25G",
            "25G Ethernet network adapter optional add-on",
            2,
            Decimal("8000"),
            {},
        )
    ]
    return matrix


def _composer_component_matrix_with_alternatives() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 24,
        },
    )
    matrix["platform_candidates"].extend(
        [
            _composer_component_candidate(
                "platform-cheap",
                "Budget",
                "PLATFORM-CHEAP",
                "Budget server platform LGA4677 DDR5",
                2,
                Decimal("1500"),
                {
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_socket": "LGA4677",
                    "ram_type": "DDR5",
                },
            ),
            _composer_component_candidate(
                "platform-clean",
                "Supermicro",
                "SYS-621C-TN12R",
                "Supermicro 2U dual LGA4677 DDR5 NVMe redundant PSU platform",
                2,
                Decimal("2300"),
                {
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_socket": "LGA4677",
                    "ram_type": "DDR5",
                },
            ),
            _composer_component_candidate(
                "platform-alt",
                "Gigabyte",
                "R283-ZK0",
                "Gigabyte 2U dual LGA4677 DDR5 server platform",
                2,
                Decimal("2100"),
                {
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_socket": "LGA4677",
                    "ram_type": "DDR5",
                },
            ),
        ]
    )
    matrix["cpu_candidates"].extend(
        [
            _composer_component_candidate(
                "cpu-16",
                "Intel",
                "CPU-16C",
                "Intel Xeon 16 core tray processor",
                4,
                Decimal("450"),
                {
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_socket": "LGA4677",
                    "cpu_cores": 16,
                },
            ),
            _composer_component_candidate(
                "cpu-20",
                "Intel",
                "CPU-20C",
                "Intel Xeon 20 core tray processor",
                4,
                Decimal("550"),
                {
                    "cpu_brand": "Intel",
                    "cpu_family": "Xeon",
                    "cpu_socket": "LGA4677",
                    "cpu_cores": 20,
                },
            ),
        ]
    )
    return matrix


def _composer_component_matrix_with_cheaper_equivalent_ram() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
            "nvme_support": True,
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 24,
        },
    )
    matrix["ram_candidates"] = [
        _composer_component_candidate(
            "ram-samsung-32",
            "Samsung",
            "RAM-SAMSUNG-32G",
            "Samsung M321R4GA3EB2-CCPKF 32GB DDR5-5600 RDIMM server memory module",
            100,
            Decimal("1493"),
            {"ram_capacity_gb": 32, "ram_type": "DDR5-5600 RDIMM"},
        ),
        _composer_component_candidate(
            "ram-micron-32",
            "Micron",
            "RAM-MICRON-32G",
            "Micron MTC20F1045S1RC48BA2 32GB DDR5-4800 ECC Registered server RAM",
            100,
            Decimal("1300"),
            {"ram_capacity_gb": 32, "ram_type": "unknown"},
        ),
    ]
    return matrix


def _composer_component_matrix_with_ram_alternative(
    alternative: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix_with_cheaper_equivalent_ram()
    matrix["ram_candidates"] = [
        matrix["ram_candidates"][0],
        alternative,
    ]
    return matrix


def _repair_ram_candidate(
    *,
    candidate_id: str,
    producer: str,
    part_number: str,
    name: str,
    available_quantity: int,
    price: Decimal,
    facts: dict[str, Any],
) -> dict[str, Any]:
    return _composer_component_candidate(
        candidate_id,
        producer,
        part_number,
        name,
        available_quantity,
        price,
        facts,
    )


def _composer_component_matrix_with_gooxi_and_supermicro() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix_with_alternatives()
    matrix["platform_candidates"] = [
        _composer_component_candidate(
            "platform-gooxi",
            "Gooxi",
            "GOOXI-CHEAP",
            "Gooxi 2U dual LGA4677 DDR5 NVMe platform",
            2,
            Decimal("1200"),
            {
                "normalized_vendor": "gooxi",
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
                "nvme_support": True,
            },
        ),
        _composer_component_candidate(
            "platform-supermicro",
            "Supermicro",
            "SYS-621C-TN12R",
            "Supermicro 2U dual LGA4677 DDR5 NVMe redundant PSU platform",
            2,
            Decimal("2300"),
            {
                "normalized_vendor": "supermicro",
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "ram_type": "DDR5",
                "nvme_support": True,
            },
        ),
    ]
    return matrix


def _repair_requirements_with_hard_platform_dimensions() -> dict[str, Any]:
    requirements = _right_size_composer_requirements()
    requirements["form_factor"] = "2U"
    requirements["psu_count_per_server"] = 2
    return requirements


def _eligible_base_platform_candidate(
    *,
    candidate_id: str = "platform-base-eligible",
    producer: str = "Supermicro",
    part_number: str = "BASE-ELIGIBLE",
    price: Decimal = Decimal("2500"),
) -> dict[str, Any]:
    return _composer_component_candidate(
        candidate_id,
        producer,
        part_number,
        f"{producer} 2U dual socket LGA4677 DDR5 NVMe 2x PSU server platform",
        2,
        price,
        {
            "normalized_vendor": producer.casefold(),
            "form_factor": "2U",
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_sockets": 2,
            "ram_type": "DDR5",
            "nvme_support": True,
        },
    )


def _blocked_platform_candidate(
    *,
    candidate_id: str,
    part_number: str,
    name: str,
    facts: dict[str, Any],
    warnings: list[str] | None = None,
    price: Decimal = Decimal("1200"),
) -> dict[str, Any]:
    candidate = _composer_component_candidate(
        candidate_id,
        "Budget",
        part_number,
        name,
        2,
        price,
        facts,
    )
    if warnings is not None:
        candidate["eligibility_warnings"] = warnings
    return candidate


def _composer_component_matrix_with_blocked_platform(
    blocked_platform: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix_with_cheaper_equivalent_ram()
    matrix["platform_candidates"] = [
        _eligible_base_platform_candidate(),
        blocked_platform,
    ]
    return matrix


def _composer_component_matrix_with_gooxi_repair_platform() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix(
        platform_facts={
            "normalized_vendor": "supermicro",
            "form_factor": "2U",
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_sockets": 2,
            "ram_type": "DDR5",
            "nvme_support": True,
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 24,
        },
    )
    matrix["platform_candidates"] = [
        _eligible_base_platform_candidate(price=Decimal("2500")),
        _composer_component_candidate(
            "platform-gooxi-repair",
            "Gooxi",
            "GOOXI-ELIGIBLE",
            "Gooxi 2U dual socket LGA4677 DDR5 NVMe 2x PSU platform",
            2,
            Decimal("1200"),
            {
                "normalized_vendor": "gooxi",
                "form_factor": "2U",
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
                "cpu_socket": "LGA4677",
                "cpu_sockets": 2,
                "ram_type": "DDR5",
                "nvme_support": True,
            },
        ),
    ]
    return matrix


def _matrix_with_hard_ineligible_platform_and_ram_repair() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix_with_blocked_platform(
        _blocked_platform_candidate(
            candidate_id="platform-blocked-incomplete",
            part_number="INCOMPLETE-CHASSIS",
            name="Budget 2U dual socket LGA4677 DDR5 NVMe chassis no CPU, Memory, HDD, PSU",
            facts={
                "cpu_brand": "Intel",
                "cpu_family": "Xeon",
            },
        )
    )
    return matrix


def _composer_component_matrix_with_amd_sp5_family() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix_with_alternatives()
    matrix["platform_candidates"].append(
        _composer_component_candidate(
            "platform-amd-sp5",
            "ASUS",
            "AMD-SP5-PLATFORM",
            "ASUS AMD EPYC SP5 DDR5 NVMe server platform",
            2,
            Decimal("2200"),
            {
                "cpu_brand": "AMD",
                "cpu_family": "EPYC",
                "cpu_socket": "SP5",
                "ram_type": "DDR5",
            },
        )
    )
    matrix["cpu_candidates"].append(
        _composer_component_candidate(
            "cpu-epyc-sp5",
            "AMD",
            "EPYC-CPU",
            "AMD EPYC 9004 SP5 processor",
            4,
            Decimal("650"),
            {
                "cpu_brand": "AMD",
                "cpu_family": "EPYC",
                "cpu_socket": "SP5",
                "cpu_cores": 24,
            },
        )
    )
    return matrix


def _composer_component_matrix_with_alternative_ssd() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 24,
        },
    )
    matrix["ssd_candidates"].append(
        _composer_component_candidate(
            "ssd-alt",
            "KIOXIA",
            "SSD-ALT",
            "KIOXIA alternative NVMe 3.84TB SSD",
            4,
            Decimal("305"),
            {"storage_capacity_tb": 3.84, "storage_interface": "NVMe"},
        )
    )
    return matrix


def _composer_component_matrix_with_ram_and_storage_tradeoffs() -> dict[str, list[dict[str, Any]]]:
    matrix = _composer_component_matrix(
        platform_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "ram_type": "DDR5",
        },
        cpu_facts={
            "cpu_brand": "Intel",
            "cpu_family": "Xeon",
            "cpu_socket": "LGA4677",
            "cpu_cores": 24,
        },
    )
    matrix["ram_candidates"] = [
        _composer_component_candidate(
            "ram-32",
            "Micron",
            "RAM-32G",
            "Micron DDR5 RDIMM 32GB server memory module",
            100,
            Decimal("60"),
            {"ram_capacity_gb": 32, "ram_type": "DDR5"},
        ),
        _composer_component_candidate(
            "ram-64",
            "Micron",
            "RAM-64G",
            "Micron DDR5 RDIMM 64GB server memory module",
            100,
            Decimal("150"),
            {"ram_capacity_gb": 64, "ram_type": "DDR5"},
        ),
    ]
    matrix["ssd_candidates"].append(
        _composer_component_candidate(
            "ssd-7680",
            "KIOXIA",
            "SSD-7680",
            "KIOXIA CD8-R Server SSD NVMe 7.68TB",
            20,
            Decimal("700"),
            {"storage_capacity_tb": 7.68, "storage_interface": "NVMe"},
        )
    )
    return matrix


def _composer_component_candidate(
    candidate_id: str,
    producer: str,
    part_number: str,
    name: str,
    quantity_required: int,
    price: Decimal,
    facts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "producer": producer,
        "part_number": part_number,
        "name": name,
        "price_value": str(price),
        "price_currency": "USD",
        "available_quantity": quantity_required,
        "quantity_required": quantity_required,
        "extracted_facts": facts,
        "fit_label": "exact_or_close_fit",
        "fit_reason": "Тестовый компонент закрывает требование.",
    }


def _package_component_candidate(
    candidate_id: str,
    *,
    score: int,
    price: str,
    over_requirement: int,
    producer: str,
    part_number: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "producer": producer,
        "normalized_vendor": producer,
        "part_number": part_number,
        "name": part_number,
        "price_value": price,
        "price_currency": "USD",
        "available_quantity": 10,
        "quantity_required": 1,
        "over_requirement": over_requirement,
        "score": score,
        "fit_label": "exact_or_close_fit",
        "fit_reason": "Тестовый компонент закрывает требование.",
    }


def _package_ready_candidate(
    candidate_id: str,
    *,
    score: int,
    matched: int,
    price: str,
    producer: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "producer": producer,
        "part_number": candidate_id.upper(),
        "item_name": candidate_id,
        "score": score,
        "price_value": price,
        "price_currency": "USD",
        "available_quantity": 2,
        "matched_requirements": [f"matched-{index}" for index in range(matched)],
        "missing_requirements": [],
        "risk_flags": [],
    }


def _package_build_candidate(
    candidate_id: str,
    *,
    complete: bool,
    price: str,
    score: int,
    missing_roles: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "candidate_type": "build_from_parts",
        "platform": {"producer": "ASUS", "part_number": candidate_id.upper()},
        "components": [],
        "completeness_status": "complete" if complete else "incomplete",
        "missing_component_roles": missing_roles or [],
        "compatibility_warnings": [],
        "total_price_value": price,
        "total_price_currency": "USD",
        "score": score,
    }


def _composer_build_candidate() -> dict[str, Any]:
    return {
        "candidate_id": "build-fatal",
        "candidate_type": "build_from_parts",
        "platform": {
            "producer": "ASUS",
            "part_number": "RS521A-E12-RS24U",
            "name": "ASUS RS521A-E12-RS24U server platform",
        },
        "components": [
            {
                "role": "server_platform",
                "component_candidate_id": "platform-amd",
                "producer": "ASUS",
                "part_number": "RS521A-E12-RS24U",
                "quantity_required": 2,
            },
            {
                "role": "cpu",
                "component_candidate_id": "cpu-selected",
                "producer": "CPUVendor",
                "part_number": "CPU-SELECTED",
                "quantity_required": 4,
            },
            {
                "role": "ram",
                "component_candidate_id": "ram-64",
                "producer": "Samsung",
                "part_number": "RAM-64G",
                "quantity_required": 16,
            },
            {
                "role": "ssd",
                "component_candidate_id": "ssd-3840",
                "producer": "KIOXIA",
                "part_number": "SSD-3840",
                "quantity_required": 4,
            },
        ],
        "completeness_status": "complete",
        "missing_component_roles": [],
        "compatibility_warnings": [],
        "total_price_value": "8800",
        "total_price_currency": "USD",
    }


def _ready_server_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    source_candidate = package["ready_stock_candidates"][0]
    return {
        "recommendations": [
            {
                "recommendation_id": "llm_ready_1",
                "source_type": "ready_server",
                "source_candidate_id": source_candidate["candidate_id"],
                "decision": "recommend_with_checks",
                "title": "Готовый складской вариант с проверками",
                "display_name": source_candidate["name"],
                "components": {
                    "platform": source_candidate["name"],
                    "cpu": "по данным готового сервера",
                    "ram": "по данным готового сервера",
                    "storage": "SSD",
                },
                "quantities": {"server": 2},
                "total_price_value": None,
                "total_price_currency": None,
                "price_note": "за весь запрос",
                "why_selected": "Готовый сервер закрывает ключевые требования по складу.",
                "why_selected_short": "Готовый сервер закрывает ключевые требования по складу.",
                "right_size_note": "Подбор: готовый складской вариант с проверками",
                "what_is_missing": [],
                "critical_checks": ["Проверить фактическую комплектацию готового сервера."],
                "engineering_review_required": True,
                "confidence": "medium",
            }
        ],
        "general_notes": [],
    }


def _excessive_storage_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendations": [
            _llm_build_recommendation(
                package,
                recommendation_id="llm_build_excessive_storage",
                source_candidate_id=_build_candidate_id_by_component_part(
                    package,
                    "ssd",
                    "SSD-15360",
                ),
                title="Overfit storage build",
                quantities={"platform": 2, "cpu": 4, "ram": 16, "ssd": 4},
                why_selected="Uses available components.",
            )
        ],
        "general_notes": [],
    }


def _right_size_overfit_llm_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendations": [
            _llm_build_recommendation(
                package,
                recommendation_id="llm_build_overfit_with_reason",
                source_candidate_id=_build_candidate_id_by_component_part(
                    package,
                    "ssd",
                    "SSD-15360",
                ),
                title="Overfit storage build",
                quantities={"platform": 2, "cpu": 4, "ram": 16, "ssd": 4},
                why_selected="Uses available components.",
            )
        ],
        "general_notes": [],
    }


def _unknown_component_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendations": [
            _llm_build_recommendation(
                package,
                recommendation_id="llm_build_bad",
                source_candidate_id="missing-source-id",
                title="Bad build",
                why_selected="Uses an unknown source candidate.",
                confidence="low",
            )
        ],
        "general_notes": [],
    }


def _wrong_role_response(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommendations": [
            _llm_build_recommendation(
                package,
                recommendation_id="llm_build_wrong_role",
                source_type="ready_server",
                title="Wrong source type",
                why_selected="Uses a build candidate as ready server.",
                confidence="low",
            )
        ],
        "general_notes": [],
    }


def _llm_build_recommendation(
    package: dict[str, Any],
    *,
    recommendation_id: str,
    title: str,
    why_selected: str,
    source_type: str = "build_from_parts",
    source_candidate_id: str | None = None,
    quantities: dict[str, int] | None = None,
    critical_checks: list[str] | None = None,
    confidence: str = "medium",
) -> dict[str, Any]:
    source_candidate = package["rule_based_build_candidates"][0]
    return {
        "recommendation_id": recommendation_id,
        "source_type": source_type,
        "source_candidate_id": source_candidate_id or source_candidate["candidate_id"],
        "decision": "recommend",
        "title": title,
        "display_name": title,
        "components": {
            "platform": source_candidate["platform"]["part_number"],
            "cpu": "CPU",
            "ram": "RAM",
            "storage": "SSD",
        },
        "quantities": quantities or {"platform": 2, "cpu": 4, "ram": 16, "ssd": 2},
        "total_price_value": None,
        "total_price_currency": None,
        "price_note": "за весь запрос",
        "why_selected": why_selected,
        "why_selected_short": why_selected,
        "right_size_note": "Подбор: минимально подходящий по требованиям",
        "what_is_missing": [],
        "critical_checks": critical_checks or [],
        "engineering_review_required": True,
        "confidence": confidence,
    }


def _build_candidate_id_by_component_part(
    package: dict[str, Any],
    role: str,
    part_number: str,
) -> str:
    for candidate in package["rule_based_build_candidates"]:
        for component in candidate["components"]:
            if component["role"] == role and component["part_number"] == part_number:
                return candidate["candidate_id"]
    raise AssertionError(f"Missing build candidate with {role} part {part_number}")


def _component_part_numbers(build_candidates: list[dict[str, Any]], role: str) -> set[str]:
    values: set[str] = set()
    for candidate in build_candidates:
        for component in candidate["components"]:
            if component["role"] == role:
                values.add(component["part_number"])
    return values
