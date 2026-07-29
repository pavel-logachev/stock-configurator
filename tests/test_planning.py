from __future__ import annotations

import json
import time
from typing import Any

import pytest

from app.planning.category_planner import (
    CompactCategory,
    build_compact_category_catalog,
    plan_distributor_categories,
    validate_category_plan,
)
from app.planning.network_facts import (
    extract_network_facts,
    network_adapter_facts_satisfy_requirement,
    required_network_adapter_quantity,
)
from app.planning.requirement_planner import plan_universal_requirements
from app.planning.role_planner import (
    plan_network_roles,
    plan_semantic_matrix_roles,
    plan_server_roles,
    plan_storage_roles,
)

AMD_25GBE_REQUEST = (
    "Нужно подобрать 2 сервера под виртуализацию, базы данных и локальное "
    "NVMe-хранилище, склад Москва. Требования на каждый сервер: "
    "2 процессора AMD EPYC, не менее 32 ядер на процессор; "
    "не менее 768 ГБ RAM DDR5 RDIMM; "
    "4 SSD NVMe не менее 7.68 ТБ на сервер; "
    "минимум 2 сетевых порта 25GbE SFP28; "
    "2 блока питания. Нужен один самый дешевый складской вариант для КП. "
    "Альтернативы не нужны. Если безопасную комплектную рекомендацию дать нельзя, "
    "честно напиши, что рекомендации нет и что нужно проверить вручную. "
    "Инженерная проверка перед КП обязательна."
)

COMPLEX_SERVER_78_TEXT = """
Исполнение: 1U
Сокеты: 2

ПРОЦЕССОР:
процессора Intel 6-го поколения 2шт.
не менее 24-ядерный процессор с частотой не менее от 2.2 ГГц;

ОПЕРАТИВНАЯ ПАМЯТЬ:
не менее 256 ГБ DDR5 RDIMM, не менее 8 модулей по 32 ГБ
не менее 6400 МГц

ДИСКИ:
6 x SSD 1920 GB SATA 6 Gb/s DWPD 3 Intel D3-S4620/D3-S4610 2.5 SFF
2 x SSD 480 GB SATA 6 Gb/s DWPD 3 Intel D3-S4620/D3-S4610 2.5 SFF
не менее 8 SFF slots на передней панели

КОНТРОЛЛЕР:
LSI Logic 9400-8i / LSI 9500-8i
hot-swap, JBOD

СЕТЕВОЙ АДАПТЕР:
Intel X710-DA2 2x10GbE SFP+

БП:
2 x 2000W hot-swap Platinum
C13-C14 cables
C13-Schuko cables
1+1 redundancy

ОХЛАЖДЕНИЕ:
8 fans N+1

ИНТЕРФЕЙСЫ:
USB 3.0, serial RJ-45, VGA, remote management RJ-45
""".strip()


class _FakeCategoryPlannerClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.package: dict[str, Any] = {}

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert (
            "Select category_id values only from category_catalog" in system_prompt
            or "Distributor Category Planner Repair" in system_prompt
        )
        self.package = json.loads(user_prompt)
        return self.payload


class _SequencedCategoryPlannerClient:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.packages: list[dict[str, Any]] = []
        self.system_prompts: list[str] = []

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert (
            "Select category_id values only from category_catalog" in system_prompt
            or "Distributor Category Planner Repair" in system_prompt
        )
        self.system_prompts.append(system_prompt)
        self.packages.append(json.loads(user_prompt))
        return self.payloads.pop(0)


class _FakeRequirementPlannerClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.package: dict[str, Any] = {}

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert "Universal Requirement Planner" in system_prompt
        assert 'role="unmapped"' in system_prompt
        self.package = json.loads(user_prompt)
        return self.payload


class _FakeSemanticMatrixPlannerClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.package: dict[str, Any] = {}
        self.system_prompt = ""

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        assert "AI Semantic Matrix Planner V2" in system_prompt
        assert "category_id" in system_prompt
        assert "component_candidate_id" in system_prompt
        self.system_prompt = system_prompt
        self.package = json.loads(user_prompt)
        return self.payload


def _semantic_amd_25gbe_plan_payload() -> dict[str, Any]:
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
            {
                "requirement_id": "req_1",
                "source_text": "2 процессора AMD EPYC, не менее 32 ядер на процессор",
                "classification": "hard_technical_requirement",
                "role": "cpu",
                "capability_id": "cpu.amd_epyc.min_32_cores",
                "hard": True,
                "parsed_requirements": {
                    "vendor": "AMD",
                    "family": "EPYC",
                    "cpu_per_server": 2,
                    "min_cores_per_cpu": 32,
                },
            },
            {
                "requirement_id": "req_2",
                "source_text": "не менее 768 ГБ RAM DDR5 RDIMM",
                "classification": "hard_technical_requirement",
                "role": "ram",
                "capability_id": "ram.ddr5_rdimm.min_768gb",
                "hard": True,
                "parsed_requirements": {
                    "min_gb_per_server": 768,
                    "type": "DDR5",
                    "form_factor": "RDIMM",
                },
            },
            {
                "requirement_id": "req_3",
                "source_text": "4 SSD NVMe не менее 7.68 ТБ на сервер",
                "classification": "hard_technical_requirement",
                "role": "storage",
                "capability_id": "storage.nvme.min_4x_7_68tb",
                "hard": True,
                "parsed_requirements": {
                    "drives_per_server": 4,
                    "interface": "NVMe",
                    "min_capacity_tb": 7.68,
                },
            },
            {
                "requirement_id": "req_4",
                "source_text": "минимум 2 сетевых порта 25GbE SFP28",
                "classification": "hard_technical_requirement",
                "role": "network_adapter",
                "capability_id": "network.25gbe.sfp28",
                "hard": True,
                "parsed_requirements": {
                    "min_ports_per_server": 2,
                    "speed": "25GbE",
                    "media": "SFP28",
                },
            },
            {
                "requirement_id": "req_5",
                "source_text": "2 блока питания",
                "classification": "hard_technical_requirement",
                "role": "power_supply",
                "capability_id": "power_supply.min_2",
                "hard": True,
                "parsed_requirements": {"psu_count_per_server": 2},
            },
            {
                "requirement_id": "opt_1",
                "source_text": "один самый дешевый складской вариант для КП",
                "classification": "commercial_instruction",
                "hard": False,
                "parsed_requirements": {
                    "optimization_goal": "cheapest_valid_stock_quote",
                    "alternatives_required": False,
                },
            },
            {
                "requirement_id": "out_1",
                "source_text": (
                    "если рекомендацию дать нельзя, написать, что нужно проверить вручную"
                ),
                "classification": "output_instruction",
                "hard": False,
                "parsed_requirements": {},
            },
            {
                "requirement_id": "eng_1",
                "source_text": "Инженерная проверка перед КП обязательна",
                "classification": "engineer_review_instruction",
                "hard": False,
                "parsed_requirements": {},
            },
        ],
        "required_capabilities": [],
        "optional_capabilities": [],
        "workload_context": [],
        "logistics_constraints": {},
        "commercial_instructions": [],
        "response_instructions": [],
        "engineer_review_required": True,
        "unsupported_or_unmapped_requirements": [],
        "planner_warnings": [],
    }


def _semantic_complex_server_78_payload() -> dict[str, Any]:
    return {
        "primary_product_group": "server",
        "primary_object": "server",
        "confidence": "high",
        "classification_reason": (
            "The whole request specifies a 1U two-socket server; NIC and power "
            "cables are embedded server requirements."
        ),
        "matrix_blueprint": {
            "roles": [
                {
                    "role": "server_platform",
                    "required": True,
                    "source_text": "Исполнение: 1U; Сокеты: 2; не менее 8 SFF slots",
                    "characteristics_to_match": {
                        "form_factor": "1U",
                        "socket_count": 2,
                        "front_sff_slots_min": 8,
                    },
                    "hard_capability_ids": ["server_platform.1u.2s.8sff"],
                },
                {
                    "role": "cpu",
                    "required": True,
                    "source_text": (
                        "процессора Intel 6-го поколения 2шт., "
                        "не менее 24 ядер, от 2.2 ГГц"
                    ),
                    "characteristics_to_match": {
                        "vendor": "Intel",
                        "generation": "6th",
                        "cpu_per_server": 2,
                        "min_cores_per_cpu": 24,
                        "min_base_frequency_ghz": 2.2,
                    },
                    "hard_capability_ids": ["cpu.intel.6th.min_24c.2_2ghz"],
                },
                {
                    "role": "ram",
                    "required": True,
                    "source_text": "не менее 256 ГБ DDR5 RDIMM, 8 модулей по 32 ГБ, 6400 МГц",
                    "characteristics_to_match": {
                        "min_gb_per_server": 256,
                        "type": "DDR5",
                        "form_factor": "RDIMM",
                        "module_count_min": 8,
                        "module_capacity_gb": 32,
                        "speed_mhz_min": 6400,
                    },
                    "hard_capability_ids": ["ram.ddr5_rdimm.256gb.8x32.6400"],
                },
                {
                    "role": "storage",
                    "required": True,
                    "source_text": "6 x SSD 1920 GB SATA; 2 x SSD 480 GB SATA; DWPD 3; 2.5 SFF",
                    "characteristics_to_match": {
                        "drive_type": "SSD",
                        "interface": "SATA",
                        "form_factor": "2.5 SFF",
                        "dwpd_min": 3,
                        "drives": [
                            {"count": 6, "capacity_gb": 1920},
                            {"count": 2, "capacity_gb": 480},
                        ],
                    },
                    "hard_capability_ids": ["storage.sata_ssd.6x1920.2x480.dwpd3"],
                },
                {
                    "role": "storage_controller",
                    "required": True,
                    "source_text": "LSI Logic 9400-8i / LSI 9500-8i, hot-swap, JBOD",
                    "characteristics_to_match": {
                        "model_hints": ["LSI Logic 9400-8i", "LSI 9500-8i"],
                        "jbod_required": True,
                        "hot_swap_required": True,
                    },
                    "hard_capability_ids": ["storage_controller.lsi.9400_9500.jbod"],
                },
                {
                    "role": "network_adapter",
                    "required": True,
                    "source_text": "Intel X710-DA2 2x10GbE SFP+",
                    "characteristics_to_match": {
                        "model_hint": "Intel X710-DA2",
                        "min_ports_per_server": 2,
                        "speed": "10GbE",
                        "media": "SFP+",
                    },
                    "hard_capability_ids": ["network_adapter.10gbe.sfpplus.x710_da2"],
                },
                {
                    "role": "power_supply",
                    "required": True,
                    "source_text": "2 x 2000W hot-swap Platinum, 1+1 redundancy",
                    "characteristics_to_match": {
                        "psu_count_per_server": 2,
                        "wattage_w_min": 2000,
                        "hot_swap_required": True,
                        "efficiency": "Platinum",
                        "redundancy": "1+1",
                    },
                    "hard_capability_ids": ["power_supply.2x2000w.platinum.1plus1"],
                },
                {
                    "role": "power_cable",
                    "required": True,
                    "source_text": "C13-C14 cables; C13-Schuko cables",
                    "characteristics_to_match": {
                        "cable_type": "power",
                        "connector_types": ["C13-C14", "C13-Schuko"],
                    },
                    "hard_capability_ids": ["power_cable.c13_c14.c13_schuko"],
                },
                {
                    "role": "cooling",
                    "required": True,
                    "source_text": "8 fans N+1",
                    "characteristics_to_match": {
                        "fan_count": 8,
                        "redundancy": "N+1",
                    },
                    "hard_capability_ids": ["cooling.8fans.nplus1"],
                },
                {
                    "role": "server_platform",
                    "required": True,
                    "source_text": "USB 3.0, serial RJ-45, VGA, remote management RJ-45",
                    "characteristics_to_match": {
                        "usb_3_required": True,
                        "serial_rj45_required": True,
                        "vga_required": True,
                        "remote_management_rj45_required": True,
                    },
                    "hard_capability_ids": ["server_platform.management_ports"],
                },
            ]
        },
        "required_capabilities": [],
        "optional_capabilities": [],
        "classified_requirements": [
            {
                "requirement_id": "req_platform",
                "source_text": (
                    "РСЃРїРѕР»РЅРµРЅРёРµ: 1U; РЎРѕРєРµС‚С‹: 2; "
                    "РЅРµ РјРµРЅРµРµ 8 SFF slots"
                ),
                "classification": "primary_object_feature",
                "product_group": "server",
                "target_role": "server_platform",
                "target_primary_object": "server",
                "hard_or_optional": "hard",
                "reason": "Form factor, sockets and bays are platform features.",
                "confidence": "high",
                "should_block_before_composer": False,
                "should_appear_in_composer_brief": True,
                "should_be_validated_after_composer": True,
                "category_needed": False,
                "parsed_requirements": {
                    "form_factor": "1U",
                    "socket_count": 2,
                    "front_sff_slots_min": 8,
                },
            },
            {
                "requirement_id": "req_cpu",
                "source_text": (
                    "РїСЂРѕС†РµСЃСЃРѕСЂР° Intel 6-РіРѕ РїРѕРєРѕР»РµРЅРёСЏ 2С€С‚., "
                    "РЅРµ РјРµРЅРµРµ 24 СЏРґРµСЂ, РѕС‚ 2.2 Р“Р“С†"
                ),
                "classification": "purchasable_component_role",
                "product_group": "server",
                "target_role": "cpu",
                "target_primary_object": "server",
                "hard_or_optional": "hard",
                "reason": "CPU is a separate BOM component.",
                "confidence": "high",
                "should_block_before_composer": True,
                "should_appear_in_composer_brief": True,
                "should_be_validated_after_composer": True,
                "category_needed": True,
            },
            {
                "requirement_id": "req_cable",
                "source_text": "C13-C14 cables; C13-Schuko cables",
                "classification": "accessory_or_consumable",
                "product_group": "server",
                "target_role": "cable",
                "target_primary_object": "server",
                "hard_or_optional": "hard",
                "reason": "Power cables are separate accessories.",
                "confidence": "high",
                "should_block_before_composer": False,
                "should_appear_in_composer_brief": True,
                "should_be_validated_after_composer": True,
                "category_needed": True,
            },
            {
                "requirement_id": "req_cooling",
                "source_text": "8 fans N+1",
                "classification": "primary_object_feature",
                "product_group": "server",
                "target_role": "server_platform",
                "target_primary_object": "server",
                "hard_or_optional": "hard",
                "reason": "Cooling redundancy is a platform feature, not a SKU role.",
                "confidence": "high",
                "should_block_before_composer": False,
                "should_appear_in_composer_brief": True,
                "should_be_validated_after_composer": True,
                "suggested_engineer_check_ru": "Check chassis fan count and N+1 cooling.",
                "category_needed": False,
                "parsed_requirements": {"fan_count": 8, "redundancy": "N+1"},
            },
            {
                "requirement_id": "req_ports",
                "source_text": "USB 3.0, serial RJ-45, VGA, remote management RJ-45",
                "classification": "primary_object_feature",
                "product_group": "server",
                "target_role": "server_platform",
                "target_primary_object": "server",
                "hard_or_optional": "hard",
                "reason": "Management and local ports are platform features.",
                "confidence": "high",
                "should_block_before_composer": False,
                "should_appear_in_composer_brief": True,
                "should_be_validated_after_composer": True,
                "category_needed": False,
                "parsed_requirements": {
                    "usb_3_required": True,
                    "serial_rj45_required": True,
                    "vga_required": True,
                    "remote_management_rj45_required": True,
                },
            },
        ],
        "embedded_requirements": [
            {
                "product_group": "network",
                "role": "network_adapter",
                "source_text": "Intel X710-DA2 2x10GbE SFP+",
                "reason": "NIC inside the server BOM.",
            },
            {
                "product_group": "accessory",
                "role": "cable",
                "source_text": "C13-C14 cables; C13-Schuko cables",
                "reason": "Power cables inside server supply scope.",
            },
        ],
        "not_primary_product_groups": [
            {
                "product_group": "network",
                "reason": "SFP+ is on the requested server NIC, not a switch.",
            },
            {
                "product_group": "network_cable",
                "reason": "C13-C14/C13-Schuko are power cables, not DAC.",
            },
        ],
        "logistics_constraints": {},
        "commercial_instructions": [],
        "response_instructions": [],
        "engineer_review_instructions": ["Engineering review mandatory."],
        "unsupported_or_unmapped_requirements": [],
        "planner_warnings": [],
    }


def test_role_planner_extracts_25gbe_sfp28_network_hard_role() -> None:
    plan = plan_server_roles("2 сервера, 2 сетевых порта 25GbE SFP28 на сервер")

    assert "network_adapter" in plan["required_roles"]
    assert "power_supply" not in plan["required_roles"]
    assert any(
        capability["role"] == "network_adapter"
        and capability["capability_id"] == "network_adapter.25gbe.sfp28"
        for capability in plan["required_capabilities"]
    )
    network = plan["requirements_by_role"]["network_adapter"]
    assert network == {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "25GbE",
        "media": "SFP28",
        "interface": "unknown",
    }


def test_network_requirement_planner_preserves_switch_hard_capabilities() -> None:
    plan = plan_network_roles(
        "Нужен коммутатор 48 портов 1G PoE+, 4 uplink 10G SFP+, "
        "L3, stacking, склад Москва, один самый дешевый вариант"
    )

    assert plan["product_group"] == "network"
    assert "switch" in plan["required_roles"]
    switch = plan["requirements_by_role"]["switch"]
    assert switch["port_count"] == 48
    assert switch["port_speed"] == "1GbE"
    assert switch["port_media"] == "RJ45"
    assert switch["uplink_count"] == 4
    assert switch["uplink_speed"] == "10GbE"
    assert switch["uplink_media"] == "SFP+"
    assert switch["poe_required"] is True
    assert switch["poe_standard"] == "PoE+"
    assert switch["l3_required"] is True
    assert switch["stacking_required"] is True
    assert plan["unsupported_or_unmapped_requirements"] == []
    assert plan["logistics_constraints"]["shipment_city"] == "Москва"
    assert any(
        instruction["parsed_requirements"].get("optimization_goal")
        == "cheapest_valid_stock_quote"
        for instruction in plan["commercial_instructions"]
    )


def test_network_requirement_planner_preserves_plain_poe_as_hard_capability() -> None:
    plan = plan_network_roles(
        "Нужен 1 коммутатор 48 портов 1G PoE, 4 uplink 10G SFP+, "
        "L3, stacking желательно, склад Москва, один самый дешевый вариант для КП"
    )

    assert plan["product_group"] == "network"
    assert plan["required_roles"] == ["switch"]
    assert plan["required_capabilities"]
    switch = plan["requirements_by_role"]["switch"]
    assert switch["port_count"] == 48
    assert switch["port_speed"] == "1GbE"
    assert switch["port_media"] == "RJ45"
    assert switch["uplink_count"] == 4
    assert switch["uplink_speed"] == "10GbE"
    assert switch["uplink_media"] == "SFP+"
    assert switch["poe_required"] is True
    assert switch["poe_standard"] == "PoE"
    assert switch["poe_standard"] != "PoE+"
    assert switch["l3_required"] is True
    assert "stacking_required" not in switch
    assert any(
        capability["role"] == "switch"
        and capability["hard"] is False
        and capability["parsed_requirements"].get("stacking_required") is True
        for capability in plan["optional_capabilities"]
    )
    source_text = plan["required_capabilities"][0]["source_text"].casefold()
    assert "склад" not in source_text
    assert "кп" not in source_text
    assert "дешев" not in source_text
    assert plan["logistics_constraints"]["shipment_city"] == "Москва"
    assert any(
        instruction["parsed_requirements"].get("optimization_goal")
        == "cheapest_valid_stock_quote"
        for instruction in plan["commercial_instructions"]
    )


def test_network_requirement_planner_preserves_poe_plus_as_stricter() -> None:
    plan = plan_network_roles(
        "Нужен коммутатор 48 портов 1G PoE+, 4 uplink 10G SFP+, L3"
    )

    switch = plan["requirements_by_role"]["switch"]
    assert switch["poe_required"] is True
    assert switch["poe_standard"] == "PoE+"


@pytest.mark.parametrize(
    ("poe_phrase", "expected_standard"),
    [
        ("Power over Ethernet", "PoE"),
        ("802.3af", "PoE"),
        ("802.3at", "PoE+"),
        ("PoE++", "PoE++"),
        ("802.3bt", "PoE++"),
    ],
)
def test_network_requirement_planner_maps_poe_standard_variants(
    poe_phrase: str,
    expected_standard: str,
) -> None:
    plan = plan_network_roles(
        f"Нужен коммутатор 48 портов 1G {poe_phrase}, 4 uplink 10G SFP+, L3"
    )

    switch = plan["requirements_by_role"]["switch"]
    assert switch["poe_required"] is True
    assert switch["poe_standard"] == expected_standard


def test_network_requirement_planner_treats_optional_poe_as_soft_preference() -> None:
    plan = plan_network_roles("Нужен коммутатор 48 портов 1G, PoE желательно")

    switch = plan["requirements_by_role"]["switch"]
    assert "poe_required" not in switch
    assert any(
        capability["role"] == "switch"
        and capability["hard"] is False
        and capability["parsed_requirements"].get("poe_required") is True
        and capability["parsed_requirements"].get("poe_standard") == "PoE"
        for capability in plan["optional_capabilities"]
    )


def test_network_requirement_planner_keeps_deterministic_poe_when_llm_plan_is_empty() -> None:
    client = _FakeRequirementPlannerClient(
        {
            "product_group": "network",
            "required_capabilities": [],
            "optional_capabilities": [],
            "unsupported_or_unmapped_requirements": [],
            "planner_warnings": [],
        }
    )

    plan = plan_network_roles(
        "Нужен 1 коммутатор 48 портов 1G PoE, 4 uplink 10G SFP+, "
        "L3, stacking желательно, склад Москва, один самый дешевый вариант для КП",
        planner_client=client,
    )

    switch = plan["requirements_by_role"]["switch"]
    assert plan["required_roles"] == ["switch"]
    assert switch["port_count"] == 48
    assert switch["poe_required"] is True
    assert switch["poe_standard"] == "PoE"
    assert any(
        capability["parsed_requirements"].get("stacking_required") is True
        for capability in plan["optional_capabilities"]
    )
    assert "requirement_planner_deterministic_semantic_fallback_preserved" in plan[
        "planner_warnings"
    ]


@pytest.mark.parametrize(
    "port_text",
    [
        "48х1000Base-T",
        "48×1000Base-T",
        "48xGigabit",
    ],
)
def test_network_requirement_planner_parses_switch_unicode_port_patterns(
    port_text: str,
) -> None:
    plan = plan_network_roles(
        f"Нужен коммутатор {port_text} PoE+, 4 uplink 10G SFP+, L3"
    )

    switch = plan["requirements_by_role"]["switch"]
    assert switch["port_count"] == 48
    assert switch["port_speed"] == "1GbE"
    assert switch["uplink_count"] == 4
    assert switch["uplink_speed"] == "10GbE"
    assert switch["uplink_media"] == "SFP+"


def test_storage_requirement_planner_preserves_hard_capabilities() -> None:
    plan = plan_storage_roles(
        "Нужна СХД 100 ТБ usable, 2 контроллера, SSD, FC 32G, "
        "поддержка 3 года, склад Москва, один самый дешевый вариант для КП"
    )

    assert plan["product_group"] == "storage"
    assert "storage_system" in plan["required_roles"]
    assert "controller" in plan["required_roles"]
    assert "drive" in plan["required_roles"] or "ssd" in plan["required_roles"]
    assert "host_port" in plan["required_roles"]
    assert "support" in plan["required_roles"]
    system = plan["requirements_by_role"]["storage_system"]
    controller = plan["requirements_by_role"]["controller"]
    drive = plan["requirements_by_role"].get("drive") or plan["requirements_by_role"]["ssd"]
    host_port = plan["requirements_by_role"]["host_port"]
    support = plan["requirements_by_role"]["support"]

    assert system["usable_capacity_tb"] == 100
    assert controller["controller_count"] == 2
    assert drive["drive_type"] == "SSD"
    assert "drive_interface" in drive
    assert host_port["host_protocol"] == "FC"
    assert host_port["host_port_speed"] == "32G"
    assert support["support_required"] is True
    assert support["warranty_months"] == 36
    assert plan["logistics_constraints"]["shipment_city"] == "Москва"
    assert any(
        instruction["parsed_requirements"].get("optimization_goal")
        == "cheapest_valid_stock_quote"
        for instruction in plan["commercial_instructions"]
    )
    unsupported_text = " ".join(plan["unsupported_or_unmapped_requirements"]).casefold()
    for commercial_or_logistics in ("склад", "москва", "кп", "дешев"):
        assert commercial_or_logistics not in unsupported_text


def test_storage_requirement_planner_keeps_drive_type_alternatives_and_optional_10gbe() -> None:
    plan = plan_storage_roles(
        "Нужно NAS-хранилище на 40 TB raw, SSD или HDD можно, 10GbE желательно, склад Москва"
    )

    assert plan["product_group"] == "storage"
    assert "storage_system" in plan["required_roles"]
    assert "drive" in plan["required_roles"]
    assert not {"ssd", "hdd"}.issubset(set(plan["required_roles"]))
    drive = plan["requirements_by_role"]["drive"]
    assert drive["raw_capacity_tb"] == 40
    assert drive["drive_type"] == "any"
    assert drive["acceptable_drive_types"] == ["SSD", "HDD"]
    assert any(
        capability["role"] == "host_port"
        and capability["hard"] is False
        and capability["parsed_requirements"]["host_port_speed"] == "10G"
        for capability in plan["optional_capabilities"]
    )


def test_storage_requirement_planner_treats_only_ssd_as_hard() -> None:
    plan = plan_storage_roles("Нужна СХД 40 TB raw, только SSD, поддержка 3 года")

    assert "ssd" in plan["required_roles"] or "drive" in plan["required_roles"]
    drive = plan["requirements_by_role"].get("ssd") or plan["requirements_by_role"]["drive"]
    assert drive["drive_type"] == "SSD"
    assert "support" in plan["required_roles"]
    assert plan["requirements_by_role"]["support"]["warranty_months"] == 36


def test_requirement_planner_keeps_storage_source_text_concise() -> None:
    plan = plan_storage_roles(
        "Need storage array 100 TB usable capacity, 2 controllers, SSD, "
        "FC 32G, support 3 years, Moscow stock, one cheapest quote."
    )
    source_texts = [
        str(capability.get("source_text") or "")
        for capability in plan["required_capabilities"]
    ]

    assert source_texts
    assert all(text.count("Need storage array") <= 1 for text in source_texts)
    assert all(len(text) <= 180 for text in source_texts)


def test_category_plan_validation_rejects_incompatible_network_base_categories() -> None:
    role_plan = {"product_group": "network", "required_roles": ["switch"], "optional_roles": []}
    catalog = [
        CompactCategory(
            "ocs",
            "V120100",
            "Switches",
            allowed_candidate_roles=("switch",),
            inferred_category_kind="base_device",
            product_group_context="network",
            base_device_allowed=True,
        ),
        CompactCategory(
            "ocs",
            "V120107",
            "SFP transceivers",
            allowed_candidate_roles=("transceiver",),
            inferred_category_kind="transceiver",
            product_group_context="network",
            base_device_allowed=False,
        ),
        CompactCategory(
            "ocs",
            "V2103",
            "NAS storage",
            allowed_candidate_roles=("storage_system",),
            inferred_category_kind="base_device",
            product_group_context="storage",
            base_device_allowed=True,
        ),
    ]

    result = validate_category_plan(
        category_plan=[
            {
                "role": "switch",
                "selected_category_ids": ["V120100", "V120107", "V2103"],
                "purpose": "base_device",
            }
        ],
        distributor_code="ocs",
        product_group="network",
        compact_catalog=catalog,
        role_plan=role_plan,
    )

    assert result["category_plan"] == {"switch": ["V120100"]}
    assert result["rejected"] is True
    assert any("V120107:switch" in warning for warning in result["warnings"])
    assert any("V2103:switch" in warning for warning in result["warnings"])


def test_category_plan_validation_rejects_storage_support_from_psu_or_cable() -> None:
    role_plan = {"product_group": "storage", "required_roles": ["support"], "optional_roles": []}
    catalog = [
        CompactCategory(
            "ocs",
            "SUP",
            "Storage support",
            allowed_candidate_roles=("support",),
            inferred_category_kind="support",
            product_group_context="storage",
            base_device_allowed=False,
        ),
        CompactCategory(
            "ocs",
            "PSU",
            "Server power supplies",
            allowed_candidate_roles=("power_supply",),
            inferred_category_kind="component",
            product_group_context="storage",
            base_device_allowed=False,
        ),
        CompactCategory(
            "ocs",
            "DAC",
            "Switch DAC cables",
            allowed_candidate_roles=("cable",),
            inferred_category_kind="cable",
            product_group_context="network",
            base_device_allowed=False,
        ),
    ]

    result = validate_category_plan(
        category_plan={"support": ["SUP", "PSU", "DAC"]},
        distributor_code="ocs",
        product_group="storage",
        compact_catalog=catalog,
        role_plan=role_plan,
    )

    assert result["category_plan"] == {"support": ["SUP"]}
    assert any("PSU:support" in warning for warning in result["warnings"])
    assert any("DAC:support" in warning for warning in result["warnings"])


def test_server_power_and_cable_metadata_accepts_server_scope_and_rejects_network_dac() -> None:
    role_plan = {
        "product_group": "server",
        "required_roles": ["power_supply", "cable"],
        "optional_roles": [],
    }
    catalog = [
        CompactCategory(
            "ocs",
            "PSU",
            "Server power supplies",
            allowed_candidate_roles=("power_supply",),
            inferred_category_kind="component",
            product_group_context="server",
            base_device_allowed=False,
        ),
        CompactCategory(
            "ocs",
            "POWER-CABLE",
            "C13 C14 server power cable",
            allowed_candidate_roles=("cable",),
            inferred_category_kind="cable",
            product_group_context="server",
            base_device_allowed=False,
        ),
        CompactCategory(
            "ocs",
            "DAC",
            "10G SFP+ DAC cable",
            allowed_candidate_roles=("cable",),
            inferred_category_kind="cable",
            product_group_context="network",
            base_device_allowed=False,
        ),
    ]

    result = validate_category_plan(
        category_plan={
            "power_supply": ["PSU"],
            "power_cable": ["POWER-CABLE", "DAC"],
        },
        distributor_code="ocs",
        product_group="server",
        compact_catalog=catalog,
        role_plan=role_plan,
    )

    assert result["category_plan"] == {
        "power_supply": ["PSU"],
        "cable": ["POWER-CABLE"],
    }
    assert any(
        "DAC:cable:product_group_context_network" in warning
        for warning in result["warnings"]
    )


def test_compact_category_catalog_slices_by_product_group_from_anchor_metadata() -> None:
    product_rows = [
        {"category_id": "V120100", "item_name": "48-port Ethernet switch"},
        {"category_id": "V120109", "item_name": "Wi-Fi access point controller"},
        {"category_id": "V2103", "item_name": "NAS storage system"},
        {"category_id": "V2104", "item_name": "Storage FC 32G host port module"},
    ]

    network_catalog = build_compact_category_catalog(
        distributor_code="ocs",
        product_group="network",
        product_rows=product_rows,
    )
    storage_catalog = build_compact_category_catalog(
        distributor_code="ocs",
        product_group="storage",
        product_rows=product_rows,
    )
    network_ids = {category.category_id for category in network_catalog}
    storage_ids = {category.category_id for category in storage_catalog}
    storage_by_id = {category.category_id: category for category in storage_catalog}

    assert "V120100" in network_ids
    assert "V2103" not in network_ids
    assert "V2103" in storage_ids
    assert "V2104" in storage_ids
    assert "V120109" not in storage_ids
    assert "host_port" in storage_by_id["V2104"].allowed_candidate_roles


def test_compact_server_catalog_uses_server_power_cable_metadata_not_network_dac() -> None:
    product_rows = [
        {"category_id": "V110108", "item_name": "2000W hot-swap server PSU"},
        {"category_id": "V110109", "item_name": "C13 C14 server power cable"},
        {"category_id": "V120150", "item_name": "10G SFP+ DAC cable"},
    ]

    server_catalog = build_compact_category_catalog(
        distributor_code="ocs",
        product_group="server",
        product_rows=product_rows,
        matrix_roles=["power_supply", "cable"],
    )
    server_ids = {category.category_id for category in server_catalog}
    by_id = {category.category_id: category for category in server_catalog}

    assert "V110108" in server_ids
    assert "V110109" in server_ids
    assert "V120150" not in server_ids
    assert by_id["V110108"].product_group_context == "server"
    assert by_id["V110109"].product_group_context == "server"


def test_category_planner_storage_uses_only_catalog_categories() -> None:
    role_plan = plan_storage_roles(
        "СХД 100 ТБ usable, SSD, FC 32G, поддержка 3 года"
    )
    catalog = [
        CompactCategory("ocs", "S-ARRAY", "СХД", "Storage / СХД", product_count=2),
        CompactCategory("ocs", "S-DRIVE", "SSD накопители", "Storage / SSD", product_count=5),
        CompactCategory("ocs", "S-SUPPORT", "Поддержка", "Services / Support", product_count=3),
    ]
    client = _FakeCategoryPlannerClient(
        {
            "category_plan": [
                {
                    "role": "storage_system",
                    "selected_category_ids": ["invented"],
                    "reason": "bad id",
                }
            ]
        }
    )

    result = plan_distributor_categories(
        distributor_code="ocs",
        product_group="storage",
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=client,
    )

    flattened_ids = {
        category_id
        for ids in result.category_plan.values()
        for category_id in ids
    }
    assert "invented" not in flattened_ids
    assert flattened_ids <= {"S-ARRAY", "S-DRIVE", "S-SUPPORT"}
    assert any(
        warning == "category_plan_id_not_in_catalog:invented"
        for warning in result.category_plan_warnings
    )


def test_role_planner_extracts_power_supply_redundancy_hard_role() -> None:
    plan = plan_server_roles("2 servers with redundant 1+1 PSU")

    assert "power_supply" in plan["required_roles"]
    capability = next(
        row
        for row in plan["required_capabilities"]
        if row["role"] == "power_supply"
    )
    assert capability["capability_id"] == "power_supply.min_2"
    assert capability["can_be_satisfied_by_platform"] is True
    assert capability["parsed_requirements"]["psu_count_per_server"] == 2


def test_role_planner_extracts_100gbe_qsfp_and_ignores_plain_1gbe() -> None:
    advanced = plan_server_roles("Нужен сервер с 100GbE QSFP28 портом")
    plain = plan_server_roles("Нужен сервер с 2 портами 1GbE RJ45")

    assert "network_adapter" in advanced["required_roles"]
    assert advanced["requirements_by_role"]["network_adapter"]["speed"] == "100GbE"
    assert advanced["requirements_by_role"]["network_adapter"]["media"] == "QSFP28"
    assert "network_adapter" not in plain["required_roles"]


def test_role_planner_extracts_storage_controller_and_unknown_hard_ask() -> None:
    plan = plan_server_roles("Обязательно RAID HBA и quantum flux capacitor")

    assert "storage_controller" in plan["required_roles"]
    assert plan["unsupported_or_unmapped_requirements"] == ["quantum flux capacitor"]


@pytest.mark.parametrize(
    ("request_text", "role"),
    [
        ("Need NVIDIA L40S GPU accelerator", "gpu"),
        ("Need RAID tri-mode HBA controller", "storage_controller"),
        ("Need FC HBA for storage fabric", "storage_controller"),
        ("Need SFP28 transceiver modules", "transceiver"),
        ("Need 25G DAC cable", "cable"),
        ("Need iDRAC license", "license"),
        ("Need 5 year support", "support"),
    ],
)
def test_universal_requirement_planner_extracts_non_network_hard_roles(
    request_text: str,
    role: str,
) -> None:
    plan = plan_universal_requirements(request_text)

    assert role in plan["required_roles"]
    assert any(capability["role"] == role for capability in plan["required_capabilities"])


def test_universal_requirement_planner_accepts_fake_client_without_live_llm() -> None:
    client = _FakeRequirementPlannerClient(
        {
            "product_group": "server",
            "required_capabilities": [
                {
                    "capability_id": "gpu.requested",
                    "role": "gpu",
                    "requirement_text": "Need GPU",
                    "hard": True,
                    "parsed_requirements": {"count_per_server": 1},
                }
            ],
            "optional_capabilities": [],
            "unsupported_or_unmapped_requirements": [],
            "planner_warnings": [],
        }
    )

    plan = plan_universal_requirements("Need GPU", planner_client=client)

    assert "gpu" in plan["required_roles"]
    assert client.package["source_text"] == "Need GPU"


def test_requirement_planner_fake_llm_semantically_classifies_amd_25gbe_request() -> None:
    client = _FakeRequirementPlannerClient(_semantic_amd_25gbe_plan_payload())

    plan = plan_universal_requirements(AMD_25GBE_REQUEST, planner_client=client)

    assert plan["unsupported_or_unmapped_requirements"] == []
    assert "network_adapter" in plan["required_roles"]
    assert any(
        capability["role"] == "network_adapter"
        and capability["capability_id"] == "network.25gbe.sfp28"
        for capability in plan["required_capabilities"]
    )
    assert any(
        capability["role"] == "power_supply"
        and capability["capability_id"] == "power_supply.min_2"
        and capability["can_be_satisfied_by_platform"] is True
        for capability in plan["required_capabilities"]
    )
    assert any(
        requirement["classification"] == "workload_context"
        and "базы данных" in requirement["source_text"]
        for requirement in plan["requirements"]
    )
    assert any(
        requirement["classification"] == "workload_context"
        and "виртуализацию" in requirement["source_text"]
        for requirement in plan["requirements"]
    )
    assert any(
        requirement["classification"] == "workload_context"
        and "локальное NVMe-хранилище" in requirement["source_text"]
        for requirement in plan["requirements"]
    )
    assert plan["logistics_constraints"]["shipment_city"] == "Москва"
    assert any(
        instruction["parsed_requirements"].get("optimization_goal")
        == "cheapest_valid_stock_quote"
        for instruction in plan["commercial_instructions"]
    )
    assert any("проверить вручную" in text for text in plan["response_instructions"])
    assert plan["engineer_review_required"] is True
    assert any(
        "Инженерная проверка" in text
        for text in plan["engineer_review_instructions"]
    )
    assert {row["role_id"] for row in client.package["product_group_profile"]["role_catalog"]} >= {
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "network_adapter",
    }


def test_ai_semantic_matrix_planner_v2_classifies_complex_server_78() -> None:
    client = _FakeSemanticMatrixPlannerClient(_semantic_complex_server_78_payload())

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
    )

    assert plan["product_group"] == "server"
    assert plan["primary_product_group"] == "server"
    assert plan["primary_object"] == "server"
    assert plan["semantic_planner_source"] == "llm"
    assert plan["semantic_planner_confidence"] == "high"
    assert plan["deterministic_product_group_hint"] == "network"
    assert plan["semantic_planner_disagreement"] is True
    assert set(plan["matrix_blueprint_roles"]) >= {
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }
    assert {
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }.issubset(set(plan["required_roles"]))
    assert "unmapped" not in plan["required_roles"]
    assert any(
        capability["role"] == "network_adapter"
        and capability["parsed_requirements"]["speed"] == "10GbE"
        and capability["parsed_requirements"]["media"] == "SFP+"
        for capability in plan["required_capabilities"]
    )
    assert any(
        capability["role"] == "cable"
        and capability.get("original_role") == "power_cable"
        and "C13-Schuko" in capability["parsed_requirements"]["connector_types"]
        for capability in plan["required_capabilities"]
    )
    cooling = next(
        capability
        for capability in plan["required_capabilities"]
        if capability["source_text"] == "8 fans N+1"
    )
    assert cooling["role"] == "server_platform"
    assert cooling["requirement_classification"] == "primary_object_feature"
    assert cooling["category_needed"] is False
    assert any(
        row["source_text"] == "8 fans N+1"
        and row["classification"] == "primary_object_feature"
        and row["target_role"] == "server_platform"
        for row in plan["primary_object_feature_requirements"]
    )
    assert plan["unmapped_requirements_blocking"] == []
    assert any(
        row.get("product_group") == "network"
        for row in plan["not_primary_product_groups"]
        if isinstance(row, dict)
    )
    assert client.package["deterministic_product_group_hint"] == "network"
    profile_ids = {
        profile["product_group_id"]
        for profile in client.package["product_group_profiles"]
    }
    assert profile_ids == {
        "server",
        "network",
        "storage",
    }


def test_requirement_classifier_network_features_and_sfp_accessory() -> None:
    client = _FakeSemanticMatrixPlannerClient(
        {
            "primary_product_group": "network",
            "primary_object": "switch",
            "confidence": "high",
            "classification_reason": "Switch request with separate optics.",
            "matrix_blueprint": {
                "roles": [
                    {
                        "role": "switch",
                        "required": True,
                        "source_text": "48 ports 1G PoE, L3, stacking",
                        "characteristics_to_match": {
                            "port_count": 48,
                            "poe_required": True,
                            "l3_required": True,
                            "stacking_required": True,
                        },
                    },
                    {
                        "role": "transceiver",
                        "required": True,
                        "source_text": "SFP+ modules 4 pcs",
                        "characteristics_to_match": {
                            "transceiver_count": 4,
                            "form_factor": "SFP+",
                        },
                    },
                ]
            },
            "classified_requirements": [
                {
                    "requirement_id": "net_feature",
                    "source_text": "48 ports 1G PoE, L3, stacking",
                    "classification": "primary_object_feature",
                    "product_group": "network",
                    "target_role": "switch",
                    "target_primary_object": "switch",
                    "hard_or_optional": "hard",
                    "reason": "Switch capabilities are features of the switch.",
                    "confidence": "high",
                    "category_needed": False,
                    "parsed_requirements": {
                        "port_count": 48,
                        "poe_required": True,
                        "l3_required": True,
                        "stacking_required": True,
                    },
                },
                {
                    "requirement_id": "net_sfp",
                    "source_text": "SFP+ modules 4 pcs",
                    "classification": "accessory_or_consumable",
                    "product_group": "network",
                    "target_role": "transceiver",
                    "target_primary_object": "switch",
                    "hard_or_optional": "hard",
                    "reason": "Optics are separate accessories.",
                    "confidence": "high",
                    "category_needed": True,
                },
            ],
        }
    )

    plan = plan_semantic_matrix_roles(
        "Need 48p PoE L3 stacking switch and 4 SFP+ modules",
        planner_client=client,
    )

    assert plan["required_roles"] == ["switch", "transceiver"]
    switch_feature = plan["primary_object_feature_requirements"][0]
    assert switch_feature["target_role"] == "switch"
    assert switch_feature["category_needed"] is False
    assert plan["accessory_or_consumable_requirements"][0]["target_role"] == "transceiver"


def test_requirement_fulfillment_separate_cable_creates_bom_role() -> None:
    client = _FakeSemanticMatrixPlannerClient(
        {
            "primary_product_group": "server",
            "primary_object": "server",
            "confidence": "high",
            "classification_reason": "Cable is requested as a separate accessory.",
            "matrix_blueprint": {"roles": []},
            "classified_requirements": [
                {
                    "requirement_id": "power_cable",
                    "source_text": "power cable set for each server",
                    "classification": "accessory_or_consumable",
                    "product_group": "server",
                    "target_role": "cable",
                    "target_primary_object": "server",
                    "hard_or_optional": "hard",
                    "fulfillment_mode": "separate_component_required",
                    "evidence_source": "request_text",
                    "evidence_text": "User asks for a power cable set.",
                    "should_create_bom_role": True,
                    "parsed_requirements": {"connector_types": ["power"]},
                }
            ],
        }
    )

    plan = plan_semantic_matrix_roles("Need server and power cable set", planner_client=client)

    cable = plan["classified_requirements"][0]
    assert cable["fulfillment_mode"] == "separate_component_required"
    assert cable["should_create_bom_role"] is True
    assert "cable" in plan["required_roles"]
    assert any(capability["role"] == "cable" for capability in plan["required_capabilities"])


def test_requirement_fulfillment_bundle_with_evidence_does_not_create_bom_role() -> None:
    client = _FakeSemanticMatrixPlannerClient(
        {
            "primary_product_group": "server",
            "primary_object": "server",
            "confidence": "high",
            "classification_reason": "Accessory is documented as part of the kit.",
            "matrix_blueprint": {"roles": []},
            "classified_requirements": [
                {
                    "requirement_id": "kit_accessory",
                    "source_text": "rack mounting kit",
                    "classification": "accessory_or_consumable",
                    "product_group": "server",
                    "target_role": "rail_kit",
                    "fulfillment_target_role": "server_platform",
                    "target_primary_object": "server",
                    "hard_or_optional": "hard",
                    "fulfillment_mode": "included_in_bundle_or_kit",
                    "evidence_source": "package_json",
                    "evidence_text": "Package contents list includes rack mounting kit.",
                    "should_create_bom_role": False,
                    "parsed_requirements": {"kit_required": True},
                }
            ],
        }
    )

    plan = plan_semantic_matrix_roles("Need server with rack mounting kit", planner_client=client)

    row = plan["classified_requirements"][0]
    assert row["fulfillment_mode"] == "included_in_bundle_or_kit"
    assert row["evidence_text"]
    assert row["should_create_bom_role"] is False
    assert "rail_kit" not in plan["required_roles"]
    assert not any(capability["role"] == "rail_kit" for capability in plan["required_capabilities"])


def test_requirement_fulfillment_bundle_without_evidence_becomes_unverified() -> None:
    client = _FakeSemanticMatrixPlannerClient(
        {
            "primary_product_group": "server",
            "primary_object": "server",
            "confidence": "medium",
            "classification_reason": "Bundle claim lacks proof.",
            "matrix_blueprint": {"roles": []},
            "classified_requirements": [
                {
                    "requirement_id": "kit_accessory",
                    "source_text": "mounting accessory included",
                    "classification": "accessory_or_consumable",
                    "product_group": "server",
                    "target_role": "rail_kit",
                    "fulfillment_target_role": "server_platform",
                    "target_primary_object": "server",
                    "hard_or_optional": "hard",
                    "fulfillment_mode": "included_in_bundle_or_kit",
                    "should_create_bom_role": False,
                    "engineer_check_ru": "Подтвердить комплектность у поставщика.",
                }
            ],
        }
    )

    plan = plan_semantic_matrix_roles(
        "Need server, mounting accessory included",
        planner_client=client,
    )

    row = plan["classified_requirements"][0]
    assert row["fulfillment_mode"] == "unverified_requires_confirmation"
    assert row["should_create_bom_role"] is False
    assert "rail_kit" not in plan["required_roles"]
    assert plan["engineer_review_required"] is True


def test_requirement_fulfillment_support_is_service_mode() -> None:
    client = _FakeSemanticMatrixPlannerClient(
        {
            "primary_product_group": "network",
            "primary_object": "switch",
            "confidence": "high",
            "classification_reason": "Support is a service line.",
            "matrix_blueprint": {"roles": []},
            "classified_requirements": [
                {
                    "requirement_id": "support_3y",
                    "source_text": "3 year support",
                    "classification": "service_or_support",
                    "product_group": "network",
                    "target_role": "support",
                    "target_primary_object": "switch",
                    "hard_or_optional": "hard",
                    "fulfillment_mode": "service_or_support",
                    "should_create_bom_role": True,
                    "parsed_requirements": {"term_years": 3},
                }
            ],
        }
    )

    plan = plan_semantic_matrix_roles("Need switch with 3 year support", planner_client=client)

    assert plan["classified_requirements"][0]["fulfillment_mode"] == "service_or_support"
    assert "support" in plan["required_roles"]


def test_requirement_classifier_storage_usable_capacity_raid_is_system_feature() -> None:
    client = _FakeSemanticMatrixPlannerClient(
        {
            "primary_product_group": "storage",
            "primary_object": "storage_system",
            "confidence": "high",
            "classification_reason": "Storage system capacity and RAID request.",
            "matrix_blueprint": {
                "roles": [
                    {
                        "role": "storage_system",
                        "required": True,
                        "source_text": "usable 100TB RAID6",
                        "characteristics_to_match": {
                            "usable_capacity_tb": 100,
                            "redundancy_level": "RAID6",
                        },
                    }
                ]
            },
            "classified_requirements": [
                {
                    "requirement_id": "storage_feature",
                    "source_text": "usable 100TB RAID6",
                    "classification": "primary_object_feature",
                    "product_group": "storage",
                    "target_role": "storage_system",
                    "target_primary_object": "storage_system",
                    "hard_or_optional": "hard",
                    "reason": "Usable capacity and RAID are system-level features.",
                    "confidence": "high",
                    "category_needed": False,
                    "parsed_requirements": {
                        "usable_capacity_tb": 100,
                        "redundancy_level": "RAID6",
                    },
                }
            ],
        }
    )

    plan = plan_semantic_matrix_roles("Need storage usable 100TB RAID6", planner_client=client)

    assert plan["required_roles"] == ["storage_system"]
    assert plan["primary_object_feature_requirements"][0]["target_role"] == "storage_system"
    assert plan["primary_object_feature_requirements"][0]["category_needed"] is False


def test_ai_semantic_matrix_planner_empty_and_invalid_fallbacks_keep_plan() -> None:
    empty_client = _FakeSemanticMatrixPlannerClient({})
    empty_plan = plan_semantic_matrix_roles(
        "Need DAC SFP+ 10G 3m",
        planner_client=empty_client,
        deterministic_product_group_hint="network",
    )

    assert empty_plan["product_group"] == "network"
    assert empty_plan["semantic_planner_source"] == "fallback_after_llm_empty"
    assert "dac_cable" in empty_plan["required_roles"]

    invalid_client = _FakeSemanticMatrixPlannerClient(
        {
            "primary_product_group": "server",
            "primary_object": "server",
            "confidence": "high",
            "classification_reason": "Invalid because category_id is forbidden.",
            "matrix_blueprint": {
                "roles": [
                    {
                        "role": "server_platform",
                        "required": True,
                        "source_text": "server",
                        "category_id": "invented",
                    }
                ]
            },
        }
    )
    invalid_plan = plan_semantic_matrix_roles(
        "Need DAC SFP+ 10G 3m",
        planner_client=invalid_client,
        deterministic_product_group_hint="network",
    )

    assert invalid_plan["product_group"] == "network"
    assert invalid_plan["semantic_planner_source"] == "fallback_after_llm_invalid"
    assert "dac_cable" in invalid_plan["required_roles"]
    assert any(
        warning.startswith("semantic_matrix_planner_invalid")
        for warning in invalid_plan["planner_warnings"]
    )


def test_ai_semantic_matrix_planner_fail_closed_for_complex_78_without_llm() -> None:
    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=None,
        deterministic_product_group_hint="network",
    )

    assert plan["product_group"] == "unknown"
    assert (
        plan["semantic_planner_source"]
        == "complex_request_requires_llm_semantic_planner"
    )
    assert plan["semantic_planner_used"] is False
    assert (
        plan["semantic_planner_fallback_reason"]
        == "complex_request_requires_llm_semantic_planner"
    )
    assert plan["required_capabilities"] == []
    assert "network" not in plan["required_roles"]
    assert "cable" not in plan["required_roles"]
    assert "Не удалось безопасно разобрать сложный запрос" in plan[
        "selected_product_group_reason"
    ]


def test_ai_semantic_matrix_planner_invalid_complex_78_fails_closed() -> None:
    invalid_client = _FakeSemanticMatrixPlannerClient(
        {
            "primary_product_group": "server",
            "primary_object": "server",
            "confidence": "high",
            "classification_reason": "Invalid because category_id is forbidden.",
            "matrix_blueprint": {
                "roles": [
                    {
                        "role": "server_platform",
                        "required": True,
                        "source_text": "server",
                        "category_id": "invented",
                    }
                ]
            },
        }
    )

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=invalid_client,
        deterministic_product_group_hint="network",
    )

    assert plan["product_group"] == "unknown"
    assert plan["semantic_planner_source"] == "fallback_after_llm_invalid"
    assert plan["semantic_planner_error_type"] == "ValueError"
    assert plan["semantic_planner_parse_status"] == "invalid_contract"
    assert (
        plan["semantic_planner_fallback_reason"]
        == "complex_request_requires_llm_semantic_planner"
    )


def test_ai_semantic_matrix_planner_simple_fallback_allowed_without_llm() -> None:
    dac_plan = plan_semantic_matrix_roles(
        "Need DAC SFP+ 10G 3m",
        planner_client=None,
        deterministic_product_group_hint="network",
    )
    switch_plan = plan_semantic_matrix_roles(
        "Нужен коммутатор 48 портов 1G PoE, 4 uplink 10G SFP+",
        planner_client=None,
        deterministic_product_group_hint="network",
    )
    server_plan = plan_semantic_matrix_roles(
        "Нужен сервер 2U, 2 CPU, 512 ГБ RAM DDR5, SSD",
        planner_client=None,
        deterministic_product_group_hint="server",
    )

    assert dac_plan["product_group"] == "network"
    assert dac_plan["semantic_planner_source"] == "deterministic_fallback"
    assert "dac_cable" in dac_plan["required_roles"]
    assert switch_plan["product_group"] == "network"
    assert "switch" in switch_plan["required_roles"]
    assert server_plan["product_group"] == "server"
    assert {"server_platform", "cpu", "ram", "storage"}.issubset(
        set(server_plan["required_roles"])
    )


def _semantic_v3_server_intent() -> dict[str, Any]:
    return {
        "product_group": "server",
        "primary_object": "server",
        "language": "ru",
        "complexity": "complex",
        "required_bom_roles_guess": [
            "server_platform",
            "cpu",
            "ram",
            "storage/ssd",
            "storage_controller",
            "network_adapter",
            "power_supply",
            "cable",
        ],
        "primary_object_feature_hints": [
            "Исполнение: 1U",
            "Сокеты: 2",
            "не менее 8 SFF slots на передней панели",
            "8 fans N+1",
            "USB 3.0, serial RJ-45, VGA, remote management RJ-45",
        ],
        "accessory_hints": [
            {"source_text": "C13-C14 cables; C13-Schuko cables", "role": "cable"}
        ],
        "service_support_hints": [],
        "logistics_hints": [],
        "confidence": "high",
        "reason": "Server BOM with platform features and separate components.",
    }


def _semantic_v3_server_classifier() -> dict[str, Any]:
    roles = [
        ("cpu", "процессора Intel 6-го поколения 2шт."),
        ("ram", "не менее 256 ГБ DDR5 RDIMM"),
        ("storage", "6 x SSD 1920 GB SATA; 2 x SSD 480 GB SATA"),
        ("storage_controller", "LSI Logic 9400-8i / LSI 9500-8i"),
        ("network_adapter", "Intel X710-DA2 2x10GbE SFP+"),
        ("power_supply", "2 x 2000W hot-swap Platinum"),
        ("cable", "C13-C14 cables; C13-Schuko cables"),
    ]
    classified = [
        {
            "requirement_id": f"role_{index}",
            "source_text": source_text,
            "classification": "purchasable_component_role",
            "product_group": "server",
            "target_role": role,
            "target_primary_object": "server",
            "hard_or_optional": "hard",
            "reason": "Separate BOM role.",
            "confidence": "high",
            "category_needed": True,
            "parsed_requirements": {},
        }
        for index, (role, source_text) in enumerate(roles, start=1)
    ]
    for row in classified:
        if row["target_role"] == "cable":
            row["classification"] = "accessory_or_consumable"
            row["reason"] = "Power cables are accessories."
    request_lines = [
        line.strip() for line in COMPLEX_SERVER_78_TEXT.splitlines() if line.strip()
    ]
    role_source_overrides = {
        "cpu": "; ".join(
            line
            for line in request_lines
            if "Intel" in line or "24-" in line or "2.2" in line
        ),
        "ram": "; ".join(
            line
            for line in request_lines
            if "DDR5 RDIMM" in line or "6400" in line
        ),
        "storage": "; ".join(
            line
            for line in request_lines
            if "SSD 1920" in line or "SSD 480" in line
        ),
        "storage_controller": "; ".join(
            line
            for line in request_lines
            if "LSI" in line or "hot-swap" in line or "JBOD" in line
        ),
        "power_supply": "; ".join(
            line
            for line in request_lines
            if "2000W" in line or "1+1" in line
        ),
    }
    for row in classified:
        override = role_source_overrides.get(row["target_role"])
        if override:
            row["source_text"] = override
    classified.append(
        {
            "requirement_id": "feature_cooling",
            "source_text": "8 fans N+1",
            "classification": "primary_object_feature",
            "product_group": "server",
            "target_role": "server_platform",
            "target_primary_object": "server",
            "hard_or_optional": "hard",
            "reason": "Cooling redundancy is a platform feature.",
            "confidence": "high",
            "should_block_before_composer": False,
            "category_needed": False,
            "parsed_requirements": {"redundancy": "N+1"},
        }
    )
    feature_sources = [
        source
        for source in _semantic_v3_server_intent()["primary_object_feature_hints"]
        if source != "8 fans N+1"
    ]
    for index, source_text in enumerate(feature_sources, start=1):
        classified.append(
            {
                "requirement_id": f"feature_source_{index}",
                "source_text": source_text,
                "classification": "primary_object_feature",
                "product_group": "server",
                "target_role": "server_platform",
                "target_primary_object": "server",
                "hard_or_optional": "hard",
                "reason": "Server platform feature.",
                "confidence": "high",
                "should_block_before_composer": False,
                "category_needed": False,
                "parsed_requirements": {"unverified_constraint": True},
            }
        )
    return {"classified_requirements": classified}


def _semantic_v3_server_partial_classifier() -> dict[str, Any]:
    roles = [
        ("cpu", "2 x Intel Xeon CPU"),
        ("ram", "256 GB DDR5 RDIMM"),
        ("storage", "SSD 1920 GB SATA"),
    ]
    return {
        "matrix_blueprint": {
            "roles": [
                {
                    "role": role,
                    "required": True,
                    "source_text": source_text,
                    "characteristics_to_match": {},
                    "hard_capability_ids": [f"{role}.requested"],
                }
                for role, source_text in roles
            ]
        },
        "classified_requirements": [
            {
                "requirement_id": f"{role}.requested",
                "source_text": source_text,
                "classification": "purchasable_component_role",
                "product_group": "server",
                "target_role": role,
                "target_primary_object": "server",
                "hard_or_optional": "hard",
                "reason": "Partial classifier emitted only core component roles.",
                "confidence": "medium",
                "category_needed": True,
                "fulfillment_mode": "separate_component_required",
                "parsed_requirements": {},
            }
            for role, source_text in roles
        ],
    }


def _semantic_v3_synthetic_role_only_classifier() -> dict[str, Any]:
    roles = [
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    ]
    return {
        "matrix_blueprint": {
            "roles": [
                {
                    "role": role,
                    "required": True,
                    "source_text": role,
                    "characteristics_to_match": {},
                    "hard_capability_ids": [],
                }
                for role in roles
            ]
        },
        "classified_requirements": [
            {
                "requirement_id": f"{role}.requested",
                "source_text": role,
                "classification": "purchasable_component_role",
                "product_group": "server",
                "target_role": role,
                "target_primary_object": "server",
                "hard_or_optional": "hard",
                "reason": "Synthetic role-only repair.",
                "confidence": "medium",
                "category_needed": True,
                "fulfillment_mode": "separate_component_required",
                "parsed_requirements": {},
            }
            for role in roles
        ],
    }


class _SequencedSemanticPlannerClient:
    def __init__(
        self,
        *,
        intent: dict[str, Any] | None,
        classifier: dict[str, Any],
        repair_classifier: dict[str, Any],
    ) -> None:
        self.intent = intent
        self.classifier = classifier
        self.repair_classifier = repair_classifier
        self.stages: list[str] = []
        self.payloads: dict[str, dict[str, Any]] = {}

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = json.loads(user_prompt)
        if "Stage A Minimal AI Intent Router" in system_prompt:
            self.stages.append("intent")
            self.payloads["intent"] = payload
            return self.intent or {}
        if (
            "Stage B AI Requirement Classifier" in system_prompt
            and "repair attempt" in system_prompt
        ):
            self.stages.append("classifier_repair")
            self.payloads["classifier_repair"] = payload
            return self.repair_classifier
        if "Stage B AI Requirement Classifier" in system_prompt:
            self.stages.append("classifier")
            self.payloads["classifier"] = payload
            return self.classifier
        raise AssertionError(f"unexpected prompt: {system_prompt[:120]}")


class _HangingSemanticPlannerClient:
    def __init__(
        self,
        *,
        hang_stages: set[str],
        intent: dict[str, Any] | None = None,
        classifier: dict[str, Any] | None = None,
        repair_classifier: dict[str, Any] | None = None,
    ) -> None:
        self.hang_stages = hang_stages
        self.intent = intent if intent is not None else _semantic_v3_server_intent()
        self.classifier = classifier if classifier is not None else _semantic_v3_server_classifier()
        self.repair_classifier = (
            repair_classifier
            if repair_classifier is not None
            else _semantic_v3_server_classifier()
        )
        self.stages: list[str] = []

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        json.loads(user_prompt)
        stage = self._stage_from_prompt(system_prompt)
        self.stages.append(stage)
        if stage in self.hang_stages:
            time.sleep(0.3)
            return {}
        if stage == "intent_router":
            return self.intent or {}
        if stage == "intent_router_repair":
            return self.intent or {}
        if stage == "requirement_classifier":
            return self.classifier or {}
        if stage == "requirement_classifier_repair":
            return self.repair_classifier or {}
        raise AssertionError(f"unexpected prompt: {system_prompt[:120]}")

    @staticmethod
    def _stage_from_prompt(system_prompt: str) -> str:
        if "Stage A Minimal AI Intent Router" in system_prompt:
            if "repair attempt" in system_prompt:
                return "intent_router_repair"
            return "intent_router"
        if "Stage B AI Requirement Classifier" in system_prompt:
            if "repair attempt" in system_prompt:
                return "requirement_classifier_repair"
            return "requirement_classifier"
        raise AssertionError(f"unexpected prompt: {system_prompt[:120]}")


def test_semantic_planner_v3_intent_router_timeout_fails_closed(capsys) -> None:
    client = _HangingSemanticPlannerClient(hang_stages={"intent_router"})

    started = time.monotonic()
    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
        semantic_planner_max_seconds=0.2,
        semantic_planner_stage_timeout_seconds=0.05,
    )
    elapsed = time.monotonic() - started
    captured = capsys.readouterr()

    assert elapsed < 1
    assert plan["product_group"] == "unknown"
    assert plan["semantic_planner_source"] == "fallback_after_llm_timeout"
    assert plan["semantic_planner_fallback_reason"] == "semantic_planner_timeout"
    assert plan["semantic_planner_error_type"] == "SemanticPlannerTimeout"
    assert plan["semantic_planner_stage"] == "intent_router"
    assert plan["semantic_planner_timeout_reason"] == "stage_timeout"
    assert plan["semantic_planner_stage_timeouts"][0]["stage"] == "intent_router"
    assert plan["requirement_classifier_status"] == "failed"
    assert "semantic_stage_start stage=intent_router" in captured.err
    assert "semantic_stage_timeout stage=intent_router" in captured.err
    assert captured.out == ""


def test_semantic_planner_v3_classifier_timeout_uses_minimal_fallback(capsys) -> None:
    client = _HangingSemanticPlannerClient(hang_stages={"requirement_classifier"})

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
        semantic_planner_max_seconds=0.2,
        semantic_planner_stage_timeout_seconds=0.05,
    )
    captured = capsys.readouterr()

    assert plan["product_group"] == "server"
    assert plan["primary_object"] == "server"
    assert plan["semantic_planner_source"] == "llm_minimal_fallback"
    assert plan["semantic_planner_fallback_reason"] == "semantic_planner_timeout"
    assert plan["semantic_planner_error_type"] == "SemanticPlannerTimeout"
    assert plan["requirement_classifier_status"] == "partial"
    assert plan["semantic_planner_minimal_router_used"] is True
    assert plan["semantic_planner_minimal_fallback_used"] is True
    assert plan["semantic_planner_repair_attempted"] is False
    assert plan["semantic_planner_stage_timeouts"][0]["stage"] == (
        "requirement_classifier"
    )
    assert "server_platform" in plan["matrix_blueprint_roles"]
    assert "network_adapter" in plan["required_roles"]
    assert "semantic_stage_timeout stage=requirement_classifier" in captured.err
    assert "semantic_minimal_fallback_used" in captured.err


def test_semantic_planner_v3_repair_timeout_uses_minimal_fallback(capsys) -> None:
    client = _HangingSemanticPlannerClient(
        hang_stages={"requirement_classifier_repair"},
        classifier={},
    )

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
        semantic_planner_max_seconds=0.25,
        semantic_planner_stage_timeout_seconds=0.05,
    )
    captured = capsys.readouterr()

    assert plan["product_group"] == "server"
    assert plan["semantic_planner_source"] == "llm_minimal_fallback"
    assert plan["semantic_planner_fallback_reason"] == "semantic_planner_timeout"
    assert plan["semantic_planner_error_type"] == "SemanticPlannerTimeout"
    assert plan["requirement_classifier_status"] == "partial"
    assert plan["semantic_planner_repair_attempted"] is True
    assert plan["semantic_planner_repair_success"] is False
    assert plan["semantic_planner_stage_timeouts"][0]["stage"] == (
        "requirement_classifier_repair"
    )
    assert "semantic_repair_timeout stage=requirement_classifier_repair" in captured.err
    assert "semantic_minimal_fallback_used" in captured.err


def test_semantic_planner_v3_empty_classifier_repair_succeeds() -> None:
    client = _SequencedSemanticPlannerClient(
        intent=_semantic_v3_server_intent(),
        classifier={},
        repair_classifier=_semantic_v3_server_classifier(),
    )

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
    )

    assert plan["product_group"] == "server"
    assert plan["semantic_planner_source"] == "llm_repaired"
    assert plan["semantic_planner_repair_attempted"] is True
    assert plan["semantic_planner_repair_success"] is True
    assert plan["requirement_classifier_status"] == "repaired"
    assert plan["requirement_classifier_repair_accepted"] is True
    assert plan["requirement_classifier_repair_quality"] == "accepted"
    assert plan["requirement_source_coverage_percent"] >= 80
    assert plan["unclassified_source_fragments"] == []
    assert plan["classified_requirements"]
    feature_sources = {
        row["source_text"] for row in plan["primary_object_feature_requirements"]
    }
    assert "8 fans N+1" in feature_sources
    assert any("USB 3.0" in source for source in feature_sources)
    cable_sources = {
        row["source_text"] for row in plan["accessory_or_consumable_requirements"]
    }
    assert any("C13-C14" in source and "C13-Schuko" in source for source in cable_sources)
    assert "network_adapter" in plan["required_roles"]
    assert client.stages == ["intent", "classifier", "classifier_repair"]
    repair_payload = client.payloads["classifier_repair"]
    assert repair_payload["source_text"] == COMPLEX_SERVER_78_TEXT
    assert repair_payload["stage_a_output"]["product_group"] == "server"
    assert repair_payload["source_fragments"]
    assert "Classify every source fragment" in repair_payload["repair_instruction"]


def test_semantic_planner_v3_repair_synthetic_roles_is_incomplete() -> None:
    client = _SequencedSemanticPlannerClient(
        intent=_semantic_v3_server_intent(),
        classifier={},
        repair_classifier=_semantic_v3_synthetic_role_only_classifier(),
    )

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
    )

    assert plan["product_group"] == "server"
    assert plan["semantic_planner_source"] == "llm_repaired"
    assert plan["semantic_planner_repair_attempted"] is True
    assert plan["semantic_planner_repair_success"] is False
    assert plan["requirement_classifier_status"] == "incomplete_repair"
    assert plan["requirement_classifier_repair_accepted"] is False
    assert plan["requirement_classifier_repair_quality"] == "synthetic_only"
    assert (
        plan["requirement_classifier_incomplete_reason"]
        == "repair_returned_only_synthetic_role_requirements"
    )
    assert plan["synthetic_requirement_count"] >= 8
    assert plan["source_backed_requirement_count"] == 0
    assert plan["requirement_source_coverage_percent"] < 80
    assert any(
        "8 fans N+1" in fragment for fragment in plan["unclassified_source_fragments"]
    )
    assert any(
        "C13-C14" in fragment for fragment in plan["unclassified_source_fragments"]
    )
    assert plan["primary_object_feature_requirements"] == []
    assert client.stages == ["intent", "classifier", "classifier_repair"]


def test_semantic_planner_v3_empty_classifier_twice_uses_minimal_ai_fallback() -> None:
    client = _SequencedSemanticPlannerClient(
        intent=_semantic_v3_server_intent(),
        classifier={},
        repair_classifier={},
    )

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
    )

    assert plan["product_group"] == "server"
    assert plan["semantic_planner_source"] == "llm_minimal_fallback"
    assert plan["semantic_planner_minimal_fallback_used"] is True
    assert plan["requirement_classifier_status"] == "partial"
    assert set(plan["matrix_blueprint_roles"]) >= {
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }
    assert any(
        row["source_text"] == "8 fans N+1"
        and row["classification"] == "primary_object_feature"
        for row in plan["primary_object_feature_requirements"]
    )
    assert plan["unmapped_requirements_blocking"] == []


def test_semantic_planner_v3_partial_classifier_preserves_stage_a_broad_roles() -> None:
    client = _SequencedSemanticPlannerClient(
        intent=_semantic_v3_server_intent(),
        classifier=_semantic_v3_server_partial_classifier(),
        repair_classifier={},
    )

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
    )

    expected_roles = {
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }
    classifier_roles = {"cpu", "ram", "storage"}
    dropped_by_classifier = expected_roles - classifier_roles
    trace_by_role = {
        row["role"]: row for row in plan["role_lifecycle_trace"]
    }

    assert plan["product_group"] == "server"
    assert set(plan["semantic_matrix_blueprint_roles"]) == classifier_roles
    assert set(plan["requirement_classifier_roles"]) == classifier_roles
    assert expected_roles.issubset(set(plan["stage_a_broad_roles"]))
    assert expected_roles.issubset(
        set(plan["effective_matrix_roles_before_category_planner"])
    )
    assert expected_roles.issubset(set(plan["category_planner_input_roles"]))
    assert expected_roles.issubset(set(plan["required_roles"]))
    assert dropped_by_classifier.issubset(set(plan["roles_dropped_after_stage_a"]))
    assert plan["roles_dropped_before_category_planner"] == []
    assert plan["roles_dropped_reason_by_role"]["network_adapter"] == (
        "not_emitted_by_requirement_classifier_preserved_by_union"
    )
    assert "stage_a" in plan["role_source_by_role"]["network_adapter"]
    assert "requirement_classifier" not in plan["role_source_by_role"]["network_adapter"]
    assert trace_by_role["network_adapter"]["stage_a"] is True
    assert trace_by_role["network_adapter"]["requirement_classifier"] is False
    assert trace_by_role["network_adapter"]["category_planner_input"] is True


def test_semantic_planner_v3_empty_intent_fails_closed_for_complex_request() -> None:
    client = _SequencedSemanticPlannerClient(
        intent=None,
        classifier={},
        repair_classifier={},
    )

    plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=client,
        deterministic_product_group_hint="network",
    )

    assert plan["product_group"] == "unknown"
    assert plan["semantic_planner_source"] == "fallback_after_llm_empty"
    assert plan["semantic_planner_fallback_reason"] == (
        "complex_request_requires_llm_semantic_planner"
    )
    assert plan["semantic_planner_empty_response_reason"] == (
        "semantic_planner_empty_response_after_repair"
    )
    assert plan["requirement_classifier_status"] == "failed"
    assert plan["matrix_blueprint_roles"] == []


def test_semantic_planner_v3_network_79_classifier_path_remains_valid() -> None:
    client = _SequencedSemanticPlannerClient(
        intent={
            "product_group": "network",
            "primary_object": "switch",
            "language": "ru",
            "complexity": "medium",
            "required_bom_roles_guess": ["switch", "transceiver"],
            "primary_object_feature_hints": [
                "48 портов 1G PoE, 4 uplink 10G SFP+, L3"
            ],
            "accessory_hints": ["SFP+ modules 4 pcs"],
            "service_support_hints": [],
            "logistics_hints": [],
            "confidence": "high",
            "reason": "Standalone network switch request.",
        },
        classifier={
            "classified_requirements": [
                {
                    "requirement_id": "switch_features",
                    "source_text": "48 портов 1G PoE, 4 uplink 10G SFP+, L3",
                    "classification": "primary_object_feature",
                    "product_group": "network",
                    "target_role": "switch",
                    "target_primary_object": "switch",
                    "hard_or_optional": "hard",
                    "reason": "Switch port and L3 properties are switch features.",
                    "confidence": "high",
                    "category_needed": False,
                    "parsed_requirements": {"port_count": 48},
                },
                {
                    "requirement_id": "sfp_modules",
                    "source_text": "SFP+ modules 4 pcs",
                    "classification": "accessory_or_consumable",
                    "product_group": "network",
                    "target_role": "transceiver",
                    "target_primary_object": "switch",
                    "hard_or_optional": "hard",
                    "reason": "Optics are separate consumables.",
                    "confidence": "high",
                    "category_needed": True,
                },
            ]
        },
        repair_classifier={},
    )

    plan = plan_semantic_matrix_roles(
        "Нужен коммутатор 48 портов 1G PoE, 4 uplink 10G SFP+, L3 и 4 SFP+ модуля",
        planner_client=client,
        deterministic_product_group_hint="network",
    )

    assert plan["product_group"] == "network"
    assert plan["semantic_planner_source"] == "llm"
    assert plan["requirement_classifier_status"] == "complete"
    assert "switch" in plan["matrix_blueprint_roles"]
    assert "transceiver" in plan["required_roles"]
    assert {"switch", "transceiver"}.issubset(
        set(plan["category_planner_input_roles"])
    )
    assert "server_platform" not in plan["category_planner_input_roles"]
    assert plan["primary_object_feature_requirements"][0]["target_role"] == "switch"


def test_category_planner_uses_semantic_matrix_blueprint_for_complex_server_78() -> None:
    role_plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=_FakeSemanticMatrixPlannerClient(
            _semantic_complex_server_78_payload()
        ),
        deterministic_product_group_hint="network",
    )
    catalog = [
        CompactCategory(
            distributor_code="test",
            category_id="platform",
            category_name="Server platforms",
            allowed_candidate_roles=("server_platform",),
            product_group_context="server",
            inferred_category_kind="base_device",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="cpu",
            category_name="Server CPUs",
            allowed_candidate_roles=("cpu",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="ram",
            category_name="Server RAM",
            allowed_candidate_roles=("ram",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="ssd",
            category_name="SATA SSD drives",
            allowed_candidate_roles=("storage",),
            product_group_context="server",
            inferred_category_kind="drive",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="hba",
            category_name="HBA storage controllers",
            allowed_candidate_roles=("storage_controller",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="nic",
            category_name="PCIe 10GbE SFP+ network adapters",
            allowed_candidate_roles=("network_adapter",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="psu",
            category_name="Server power supplies",
            allowed_candidate_roles=("power_supply",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="power-cable",
            category_name="C13 C14 Schuko power cables",
            allowed_candidate_roles=("cable",),
            product_group_context="accessory",
            inferred_category_kind="cable",
        ),
    ]

    result = plan_distributor_categories(
        distributor_code="test",
        product_group=role_plan["product_group"],
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=None,
    )

    assert role_plan["product_group"] == "server"
    assert set(result.category_plan) >= {
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }
    assert result.category_plan["network_adapter"] == ["nic"]
    assert result.category_plan["cable"] == ["power-cable"]
    assert "dac_cable" not in result.category_plan


def test_category_planner_reports_missing_categories_for_preserved_broad_roles() -> None:
    role_plan = plan_semantic_matrix_roles(
        COMPLEX_SERVER_78_TEXT,
        planner_client=_SequencedSemanticPlannerClient(
            intent=_semantic_v3_server_intent(),
            classifier=_semantic_v3_server_partial_classifier(),
            repair_classifier={},
        ),
        deterministic_product_group_hint="network",
    )
    catalog = [
        CompactCategory(
            distributor_code="test",
            category_id="platform",
            category_name="Server platforms",
            allowed_candidate_roles=("server_platform",),
            product_group_context="server",
            inferred_category_kind="base_device",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="cpu",
            category_name="Server CPUs",
            allowed_candidate_roles=("cpu",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="ram",
            category_name="Server RAM",
            allowed_candidate_roles=("ram",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
        CompactCategory(
            distributor_code="test",
            category_id="ssd",
            category_name="SATA SSD drives",
            allowed_candidate_roles=("storage",),
            product_group_context="server",
            inferred_category_kind="drive",
        ),
    ]

    result = plan_distributor_categories(
        distributor_code="test",
        product_group=role_plan["product_group"],
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=None,
    )

    expected_roles = {
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }
    missing_roles = {"storage_controller", "cable"}

    assert expected_roles.issubset(set(result.category_planner_input_roles))
    assert missing_roles.issubset(set(result.missing_category_roles))
    assert result.roles_dropped_after_category_planner == []
    assert result.roles_dropped_reason_by_role["storage_controller"] == "missing_category"
    assert result.role_coverage_summary["storage_controller"]["missing_category"] is True
    assert result.role_coverage_summary["network_adapter"][
        "can_be_satisfied_by_platform"
    ] is True
    assert result.role_coverage_summary["power_supply"][
        "can_be_satisfied_by_platform"
    ] is True


def test_requirement_planner_unmapped_hard_technical_requirement_is_preserved() -> None:
    client = _FakeRequirementPlannerClient(
        {
            "product_group": "server",
            "requirements": [
                {
                    "requirement_id": "req_1",
                    "source_text": "quantum flux capacitor",
                    "classification": "hard_technical_requirement",
                    "role": "quantum_accelerator",
                    "capability_id": "quantum.flux_capacitor",
                    "hard": True,
                    "parsed_requirements": {},
                }
            ],
            "required_capabilities": [],
            "optional_capabilities": [],
            "unsupported_or_unmapped_requirements": [],
            "planner_warnings": [],
        }
    )

    plan = plan_universal_requirements("Need quantum flux capacitor", planner_client=client)

    assert plan["unsupported_or_unmapped_requirements"] == []
    assert "unmapped" in plan["required_roles"]
    capability = next(
        row
        for row in plan["required_capabilities"]
        if row["capability_id"] == "quantum.flux_capacitor"
    )
    assert capability["role"] == "unmapped"
    assert capability["original_role"] == "quantum_accelerator"
    assert capability["source_text"] == "quantum flux capacitor"


@pytest.mark.parametrize(
    ("text", "ports_count"),
    [
        ("Dual-port 25GbE SFP28 PCIe adapter", 2),
        ("Dual-port 25 GbE SFP28 PCIe adapter", 2),
        ("Dual-port 25Gbe SFP28 PCIe adapter", 2),
        ("Dual-port 25Gb/s SFP28 PCIe adapter", 2),
        ("Dual-port 25 Gbps SFP28 PCIe adapter", 2),
        ("Dual-port 25Gb Ethernet SFP28 PCIe adapter", 2),
        ("Dual-Port 25 Gb/s SFP28 Ethernet PCI Express x8", 2),
        ("Quad-Port 25 Gb/s SFP28 Ethernet PCI Express 4.0 x16", 4),
        ("2x25GbE SFP28", 2),
        ("4x25G SFP28", 4),
        ("25GbE Dual SFP28", 2),
        ("4xSFP28 ports, 25GbE", 4),
        ("2xSFP28 ports, 25GbE", 2),
        ("4*SFP28 25G", 4),
        ("2 x SFP28, 25GbE", 2),
    ],
)
def test_network_facts_extract_25gbe_sfp28_variants(
    text: str,
    ports_count: int,
) -> None:
    facts = extract_network_facts(text)

    assert facts["ports_count"] == ports_count
    assert facts["speed"] == "25GbE"
    assert facts["speed_gbps"] == 25
    assert facts["media"] == "SFP28"


def test_network_facts_extract_interface_when_present() -> None:
    dual = extract_network_facts("Dual-port 25GbE SFP28 OCP adapter")
    quad = extract_network_facts("Quad-Port 25 Gb/s SFP28 PCIe")

    assert dual["ports_count"] == 2
    assert dual["speed"] == "25GbE"
    assert dual["media"] == "SFP28"
    assert dual["interface"] == "OCP"
    assert quad["ports_count"] == 4
    assert quad["speed"] == "25GbE"
    assert quad["media"] == "SFP28"
    assert quad["interface"] == "PCIe"


def test_network_adapter_eligibility_and_quantity_for_25gbe_sfp28() -> None:
    requirement = {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "25GbE",
        "media": "SFP28",
    }
    dual = extract_network_facts("Dual-port 25GbE SFP28 PCIe adapter")
    quad = extract_network_facts("Quad-port 25GbE SFP28 PCIe adapter")

    assert network_adapter_facts_satisfy_requirement(dual, requirement)
    assert required_network_adapter_quantity(
        dual,
        requirement,
        server_quantity=2,
    ) == 2
    assert network_adapter_facts_satisfy_requirement(quad, requirement)
    assert required_network_adapter_quantity(
        quad,
        requirement,
        server_quantity=2,
    ) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Dual-port 10GbE SFP+ PCIe adapter",
        "Dual-port 25GbE RJ45 PCIe adapter",
        "25GbE SFP28 PCIe adapter",
        "Generic PCIe Ethernet adapter",
    ],
)
def test_network_adapter_eligibility_rejects_mismatches_and_unknowns(text: str) -> None:
    requirement = {
        "required": True,
        "min_ports_per_server": 2,
        "speed": "25GbE",
        "media": "SFP28",
    }
    facts = extract_network_facts(text)

    assert not network_adapter_facts_satisfy_requirement(facts, requirement)


def test_network_facts_unknown_speed_media_remain_unknown() -> None:
    facts = extract_network_facts("Generic PCIe Ethernet adapter")

    assert facts["speed"] == "unknown"
    assert facts["speed_gbps"] is None
    assert facts["media"] == "unknown"


def test_category_planner_maps_network_adapter_from_supplied_catalog_only() -> None:
    role_plan = {
        "product_group": "server",
        "required_roles": ["server_platform", "cpu", "ram", "storage", "network_adapter"],
        "optional_roles": [],
        "requirements_by_role": {},
    }
    catalog = [
        CompactCategory("dist-a", "srv", "Server platforms"),
        CompactCategory("dist-a", "nic", "Server network adapters 25GbE SFP28"),
        CompactCategory("dist-a", "cpu", "Server processors"),
        CompactCategory("dist-a", "ram", "Server memory RDIMM"),
        CompactCategory("dist-a", "ssd", "Server SSD NVMe"),
    ]

    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=catalog,
    )

    assert result.category_plan["network_adapter"] == ["nic"]


def test_category_planner_uses_semantic_required_capabilities_package() -> None:
    role_plan = {
        "product_group": "server",
        "required_capabilities": [
            {
                "capability_id": "network.25gbe.sfp28",
                "role": "network_adapter",
                "hard": True,
                "source_text": "минимум 2 сетевых порта 25GbE SFP28",
                "parsed_requirements": {
                    "min_ports_per_server": 2,
                    "speed": "25GbE",
                    "media": "SFP28",
                },
            }
        ],
        "optional_capabilities": [],
        "required_roles": [],
        "optional_roles": [],
        "role_catalog": ["network_adapter"],
    }
    catalog = [
        CompactCategory("dist-a", "nic", "Server network adapters 25GbE SFP28"),
    ]
    client = _FakeCategoryPlannerClient(
        {
            "category_plan": [
                {
                    "capability_id": "network.25gbe.sfp28",
                    "role": "network_adapter",
                    "selected_category_ids": ["nic"],
                    "reason": "Catalog category matches 25GbE SFP28 NICs.",
                }
            ],
            "missing_category_roles": [],
            "category_plan_warnings": [],
        }
    )

    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=client,
    )

    assert client.package["required_capabilities"] == role_plan["required_capabilities"]
    assert "source_text" not in client.package
    assert result.category_plan["network_adapter"] == ["nic"]
    assert result.category_plan_entries[0]["capability_id"] == "network.25gbe.sfp28"


def test_category_planner_rejects_ids_not_supplied_to_llm() -> None:
    role_plan = {
        "product_group": "server",
        "required_roles": ["network_adapter"],
        "optional_roles": [],
    }
    catalog = [
        CompactCategory("dist-a", "nic", "Server network adapters"),
    ]
    client = _FakeCategoryPlannerClient(
        {"category_plan": {"network_adapter": ["invented-id"]}}
    )

    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=client,
    )

    assert result.category_plan["network_adapter"] == ["nic"]
    assert any("not_in_catalog" in warning for warning in result.category_plan_warnings)
    assert client.package["category_catalog"] == [catalog[0].to_prompt_json()]


def test_ai_category_planner_keeps_valid_catalog_ids_and_drops_invented_ids() -> None:
    role_plan = {
        "product_group": "server",
        "required_roles": ["server_platform", "cpu"],
        "optional_roles": [],
    }
    catalog = [
        CompactCategory(
            "dist-a",
            "platform",
            "Server platforms",
            allowed_candidate_roles=("server_platform",),
            product_group_context="server",
            inferred_category_kind="base_device",
        ),
        CompactCategory(
            "dist-a",
            "cpu",
            "Server CPUs",
            allowed_candidate_roles=("cpu",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
    ]
    client = _FakeCategoryPlannerClient(
        {
            "category_plan": [
                {
                    "role": "server_platform",
                    "selected_category_ids": ["platform", "invented-platform"],
                    "purpose": "base_device",
                },
                {
                    "role": "cpu",
                    "selected_category_ids": ["cpu"],
                    "purpose": "component",
                },
            ]
        }
    )

    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=client,
    )

    assert result.category_planner_source == "ai_category_planner"
    assert result.category_plan == {
        "server_platform": ["platform"],
        "cpu": ["cpu"],
    }
    assert any(
        warning == "category_plan_id_not_in_catalog:invented-platform"
        for warning in result.category_plan_warnings
    )


def test_category_planner_repairs_missing_required_lifecycle_role() -> None:
    role_plan = {
        "product_group": "server",
        "primary_object": "server",
        "required_roles": ["server_platform", "cpu"],
        "category_planner_input_roles": ["server_platform", "cpu"],
        "effective_matrix_roles_before_category_planner": ["server_platform", "cpu"],
        "required_capabilities": [
            {
                "capability_id": "server_platform.required",
                "role": "server_platform",
                "hard": True,
            }
        ],
    }
    catalog = [
        CompactCategory(
            "dist-a",
            "platform",
            "Server platforms",
            allowed_candidate_roles=("server_platform",),
            product_group_context="server",
            inferred_category_kind="base_device",
        ),
        CompactCategory(
            "dist-a",
            "cpu",
            "Server CPUs",
            allowed_candidate_roles=("cpu",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
    ]
    client = _SequencedCategoryPlannerClient(
        [
            {
                "category_plan": [
                    {
                        "role": "cpu",
                        "selected_category_ids": ["cpu"],
                        "purpose": "component",
                    }
                ]
            },
            {
                "category_plan": [
                    {
                        "role": "server_platform",
                        "selected_category_ids": ["platform"],
                        "purpose": "base_device",
                        "reason": "Required platform category from catalog.",
                    }
                ],
                "no_category_found": [],
            },
        ]
    )

    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=client,
    )

    assert result.category_plan["server_platform"] == ["platform"]
    assert result.category_plan["cpu"] == ["cpu"]
    assert result.category_planner_missing_required_roles == ["server_platform"]
    assert result.category_planner_repair_attempted is True
    assert result.category_planner_repair_success is True
    assert result.category_planner_repaired_roles == ["server_platform"]
    assert result.category_planner_unresolved_required_roles == []
    assert "server_platform" in result.category_planner_output_roles
    assert client.packages[1]["missing_required_roles"] == ["server_platform"]
    assert client.packages[1]["original_category_planner_output"]["category_plan"][0][
        "role"
    ] == "cpu"


def test_category_planner_blocks_when_repair_finds_no_required_category() -> None:
    role_plan = {
        "product_group": "server",
        "primary_object": "server",
        "required_roles": ["server_platform"],
        "category_planner_input_roles": ["server_platform"],
        "effective_matrix_roles_before_category_planner": ["server_platform"],
    }
    catalog = [
        CompactCategory(
            "dist-a",
            "cpu",
            "Server CPUs",
            allowed_candidate_roles=("cpu",),
            product_group_context="server",
            inferred_category_kind="component",
        ),
    ]
    client = _SequencedCategoryPlannerClient(
        [
            {"category_plan": []},
            {
                "category_plan": [],
                "no_category_found": [
                    {
                        "role": "server_platform",
                        "reason": "No server platform category in catalog slice.",
                    }
                ],
            },
        ]
    )

    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=client,
    )

    assert result.category_plan == {}
    assert result.missing_category_roles == ["server_platform"]
    assert result.roles_dropped_reason_by_role["server_platform"] == "missing_category"
    assert result.category_planner_missing_required_roles == ["server_platform"]
    assert result.category_planner_repair_attempted is True
    assert result.category_planner_repair_success is False
    assert result.category_planner_unresolved_required_roles == ["server_platform"]
    assert result.category_planner_repair_reason.startswith("no_category_found:")


def test_network_category_planner_rejects_invented_category_id() -> None:
    role_plan = {
        "product_group": "network",
        "required_roles": ["switch"],
        "optional_roles": [],
        "required_capabilities": [
            {
                "capability_id": "switch.48.1gbe.4.10gbe.poe.l3.stacking",
                "role": "switch",
                "hard": True,
                "source_text": "48 портов 1G PoE+, 4 uplink 10G SFP+, L3, stacking",
                "parsed_requirements": {
                    "port_count": 48,
                    "port_speed": "1GbE",
                    "uplink_count": 4,
                    "uplink_speed": "10GbE",
                    "uplink_media": "SFP+",
                    "poe_required": True,
                    "l3_required": True,
                    "stacking_required": True,
                },
            }
        ],
    }
    catalog = [
        CompactCategory(
            "ocs",
            "net-switch",
            "Коммутаторы Ethernet PoE",
            sample_product_names=("48x1G PoE+ switch 4x10G SFP+ L3 stacking",),
        ),
    ]
    client = _FakeCategoryPlannerClient(
        {"category_plan": {"switch": ["invented-switch-category"]}}
    )

    result = plan_distributor_categories(
        distributor_code="ocs",
        product_group="network",
        role_plan=role_plan,
        compact_catalog=catalog,
        llm_client=client,
    )

    assert result.category_plan["switch"] == ["net-switch"]
    assert any("not_in_catalog" in warning for warning in result.category_plan_warnings)
    assert client.package["category_catalog"] == [catalog[0].to_prompt_json()]


def test_validate_category_plan_rejects_wrong_distributor_and_allows_other_catalog_ids() -> None:
    role_plan = {"required_roles": ["network_adapter"], "optional_roles": []}
    dist_a_catalog = [CompactCategory("dist-a", "nic-a", "Network adapters")]
    dist_b_catalog = [CompactCategory("dist-b", "nic-b", "Network adapters")]

    invalid = validate_category_plan(
        category_plan={"network_adapter": ["nic-b"]},
        distributor_code="dist-a",
        product_group="server",
        compact_catalog=dist_a_catalog,
        role_plan=role_plan,
    )
    valid_b = plan_distributor_categories(
        distributor_code="dist-b",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=dist_b_catalog,
    )

    assert invalid["rejected"] is True
    assert invalid["category_plan"] == {}
    assert valid_b.category_plan["network_adapter"] == ["nic-b"]


def test_category_planner_maps_same_role_to_distributor_specific_catalog_ids() -> None:
    role_plan = {"required_roles": ["gpu"], "optional_roles": []}

    dist_a = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=[CompactCategory("dist-a", "gpu-a", "GPU accelerators")],
    )
    dist_b = plan_distributor_categories(
        distributor_code="dist-b",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=[CompactCategory("dist-b", "gpu-b", "NVIDIA GPU boards")],
    )

    assert dist_a.category_plan["gpu"] == ["gpu-a"]
    assert dist_b.category_plan["gpu"] == ["gpu-b"]


def test_category_planner_maps_unmapped_capability_when_catalog_matches() -> None:
    role_plan = {
        "product_group": "server",
        "required_capabilities": [
            {
                "capability_id": "quantum.flux_capacitor",
                "role": "unmapped",
                "hard": True,
                "source_text": "quantum flux capacitor",
                "category_search_intent": "quantum flux capacitor module",
                "parsed_requirements": {},
            }
        ],
        "optional_capabilities": [],
        "required_roles": ["unmapped"],
        "optional_roles": [],
    }

    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan=role_plan,
        compact_catalog=[
            CompactCategory("dist-a", "quantum-cat", "Quantum flux capacitor modules"),
            CompactCategory("dist-a", "cpu-cat", "Server processors"),
        ],
    )

    assert result.category_plan["unmapped"] == ["quantum-cat"]
    assert result.role_coverage_summary["unmapped"]["missing_category"] is False


@pytest.mark.parametrize(
    ("role", "category_id", "category_name"),
    [
        ("gpu", "gpu-cat", "NVIDIA GPU accelerators"),
        ("storage_controller", "hba-cat", "RAID HBA controllers"),
        ("transceiver", "optic-cat", "SFP QSFP transceiver optics"),
    ],
)
def test_category_planner_maps_hard_roles_from_catalog(
    role: str,
    category_id: str,
    category_name: str,
) -> None:
    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan={"required_roles": [role], "optional_roles": []},
        compact_catalog=[CompactCategory("dist-a", category_id, category_name)],
    )

    assert result.category_plan[role] == [category_id]


def test_category_planner_marks_missing_category_for_hard_capability() -> None:
    result = plan_distributor_categories(
        distributor_code="dist-a",
        product_group="server",
        role_plan={"required_roles": ["gpu"], "optional_roles": []},
        compact_catalog=[CompactCategory("dist-a", "cpu", "Server processors")],
    )

    assert "gpu" in result.missing_category_roles
    assert result.role_coverage_summary["gpu"]["missing_category"] is True
