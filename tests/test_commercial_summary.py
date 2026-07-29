from __future__ import annotations

import json

from app.reports.commercial_summary import (
    build_primary_commercial_summary,
    primary_commercial_excel_rows,
    primary_commercial_telegram_lines,
)


def test_primary_commercial_summary_network_line_shows_normalized_facts() -> None:
    summary = build_primary_commercial_summary(
        {},
        {
            "components": [
                {
                    "role": "server_platform",
                    "producer": "ASUS",
                    "part_number": "AMD-2U-2S-PSU",
                    "item_name": "ASUS 2U dual socket AMD EPYC server platform",
                    "quantity_required": 2,
                    "available_quantity": 2,
                },
                {
                    "role": "network_adapter",
                    "producer": "ShenzhenLianrui Electronic Co., LTD",
                    "part_number": "LRES1026PF-2SFP28",
                    "item_name": "LR-Link LRES1026PF-2SFP28 network adapter",
                    "quantity_required": 2,
                    "per_server_quantity": 1,
                    "server_quantity": 2,
                    "available_quantity": 30,
                    "facts": {
                        "ports_count": 2,
                        "speed": "25GbE",
                        "media": "SFP28",
                    },
                },
            ],
            "total_price_value": "1000",
            "total_price_currency": "USD",
        },
        match_run_id=68,
    )

    assert summary is not None
    text = summary["copy_paste_text"]
    assert "\u0421\u0435\u0442\u044c:" in text
    assert "LRES1026PF-2SFP28, 2 x 25GbE SFP28" in text
    assert (
        "1 \u0448\u0442. \u043d\u0430 \u0441\u0435\u0440\u0432\u0435\u0440 / "
        "2 \u0448\u0442. \u0432\u0441\u0435\u0433\u043e"
    ) in text
    assert "\u0441\u043a\u043b\u0430\u0434 30" in text

    summary_text = json.dumps(summary, ensure_ascii=False)
    for forbidden in ("component_candidate_id", '"facts"', "raw JSON", "llm_rec"):
        assert forbidden not in summary_text


def test_primary_commercial_summary_network_line_can_use_required_capability() -> None:
    summary = build_primary_commercial_summary(
        {},
        {
            "components": [
                {
                    "role": "server_platform",
                    "producer": "ASUS",
                    "part_number": "AMD-2U-2S-PSU",
                    "quantity_required": 2,
                    "available_quantity": 2,
                },
                {
                    "role": "network_adapter",
                    "producer": "AdapterVendor",
                    "part_number": "NIC-25G-DUAL",
                    "quantity_required": 2,
                    "per_server_quantity": 1,
                    "server_quantity": 2,
                    "available_quantity": 10,
                },
            ],
            "required_capabilities": [
                {
                    "capability_id": "network_adapter.25gbe.sfp28",
                    "role": "network_adapter",
                    "hard": True,
                    "parsed_requirements": {
                        "min_ports_per_server": 2,
                        "speed": "25GbE",
                        "media": "SFP28",
                    },
                }
            ],
        },
        match_run_id=69,
    )

    assert summary is not None
    text = summary["copy_paste_text"]
    assert "\u0421\u0435\u0442\u044c:" in text
    assert "NIC-25G-DUAL, 2 x 25GbE SFP28" in text


def test_network_product_group_commercial_surfaces_hide_internals() -> None:
    summary = build_primary_commercial_summary(
        {"product_group": "network"},
        {
            "product_group": "network",
            "components": [
                {
                    "role": "switch",
                    "component_candidate_id": "switch-48p",
                    "producer": "NetVendor",
                    "part_number": "SW-48P-4SFP",
                    "item_name": "48x1G RJ45 PoE+ switch 740W 4x10G SFP+ L3 stacking",
                    "quantity_required": 1,
                    "server_quantity": 1,
                    "per_server_quantity": 1,
                    "available_quantity": 3,
                    "facts": {"raw": {"debug": True}},
                    "port_count": 48,
                    "port_speed": "1GbE",
                    "port_media": "RJ45",
                    "uplink_count": 4,
                    "uplink_speed": "10GbE",
                    "uplink_media": "SFP+",
                    "poe_supported": True,
                    "poe_budget_w": 740,
                    "poe_standard": "PoE+",
                    "l3_supported": True,
                    "stacking_supported": True,
                },
                {
                    "role": "license",
                    "component_candidate_id": "license-1y",
                    "producer": "NetVendor",
                    "part_number": "LIC-1Y",
                    "quantity_required": 1,
                    "server_quantity": 1,
                    "per_server_quantity": 1,
                    "available_quantity": 3,
                },
            ],
            "total_price_value": "1300",
            "total_price_currency": "USD",
        },
        match_run_id=70,
    )

    assert summary is not None
    telegram_text = "\n".join(primary_commercial_telegram_lines(summary))
    excel_text = json.dumps(primary_commercial_excel_rows(summary), ensure_ascii=False)
    combined = f"{telegram_text}\n{excel_text}"

    for expected in (
        "Предварительная спецификация для КП",
        "Сетевое оборудование - 1 шт.",
        "Состав",
        "Всего к заказу",
        "Комментарий",
        "Проверить перед КП",
    ):
        assert expected in combined
    for forbidden in ("component_candidate_id", "switch-48p", "raw", "debug", "llm_rec", "{"):
        assert forbidden not in combined


def test_network_commercial_surfaces_sanitize_server_engineer_checks() -> None:
    commercial = {
        "product_group": "network",
        "server_quantity": 1,
        "title": "Предварительная спецификация для КП",
        "server_line": "Сетевое оборудование - 1 шт.",
        "price_line": "Ориентировочно за 1 шт. сетевого оборудования: 483.84 USD",
        "per_server_lines": ["Коммутатор: Origo OS3254P/370W/A1A"],
        "total_order_lines": ["Коммутатор: 1 шт., склад 1"],
        "comment_lines": ["Перед КП нужна инженерная проверка сетевой схемы."],
        "engineer_checks": [
            "Проверить CPU support list платформы и версию BIOS.",
            "Проверить QVL RAM и правила заполнения DIMM.",
            "Проверить NVMe/U.2/U.3 backplane.",
            "Проверить access/uplink схему и PoE budget.",
        ],
    }

    telegram_text = "\n".join(primary_commercial_telegram_lines(commercial))
    excel_text = json.dumps(primary_commercial_excel_rows(commercial), ensure_ascii=False)
    combined = f"{telegram_text}\n{excel_text}"

    assert "access/uplink" in combined
    assert "PoE" in combined
    for forbidden in ("CPU support", "QVL RAM", "DIMM", "NVMe/U.2/U.3", "backplane"):
        assert forbidden not in combined


def test_storage_product_group_commercial_surfaces_hide_internals() -> None:
    summary = build_primary_commercial_summary(
        {"product_group": "storage"},
        {
            "product_group": "storage",
            "components": [
                {
                    "role": "storage_system",
                    "component_candidate_id": "storage-system-1",
                    "producer": "StorageVendor",
                    "part_number": "ARR-100U",
                    "item_name": "Storage array 100TB usable dual controller FC 32G",
                    "quantity_required": 1,
                    "server_quantity": 1,
                    "per_server_quantity": 1,
                    "available_quantity": 2,
                    "usable_capacity_tb": 100,
                    "host_protocol": "FC",
                    "host_port_speed": "32G",
                    "facts": {"raw": {"debug": True}},
                },
                {
                    "role": "support",
                    "component_candidate_id": "support-3y",
                    "producer": "StorageVendor",
                    "part_number": "SUP-3Y",
                    "quantity_required": 1,
                    "server_quantity": 1,
                    "per_server_quantity": 1,
                    "available_quantity": 2,
                    "warranty_months": 36,
                },
            ],
            "total_price_value": "5000",
            "total_price_currency": "USD",
        },
        match_run_id=71,
    )

    assert summary is not None
    telegram_text = "\n".join(primary_commercial_telegram_lines(summary))
    excel_text = json.dumps(primary_commercial_excel_rows(summary), ensure_ascii=False)
    combined = f"{telegram_text}\n{excel_text}"

    for expected in (
        "Предварительная спецификация для КП",
        "СХД - 1 шт.",
        "Состав",
        "Всего к заказу",
        "Комментарий",
        "Проверить перед КП",
    ):
        assert expected in combined
    for forbidden in ("component_candidate_id", "storage-system-1", '"raw"', "debug", "{"):
        assert forbidden not in combined


def test_storage_commercial_surfaces_sanitize_server_engineer_checks() -> None:
    commercial = {
        "product_group": "storage",
        "server_quantity": 1,
        "title": "Предварительная спецификация для КП",
        "server_line": "СХД - 1 шт.",
        "price_line": "Ориентировочно за 1 шт. СХД: 5000 USD",
        "per_server_lines": ["СХД: StorageVendor ARR-100U"],
        "total_order_lines": ["СХД: 1 шт., склад 2"],
        "comment_lines": ["Перед КП нужна инженерная проверка СХД."],
        "engineer_checks": [
            "Проверить CPU support list платформы и версию BIOS.",
            "Проверить QVL RAM и правила заполнения DIMM.",
            "Проверить NVMe/U.2/U.3 backplane.",
            "Проверить raw/usable capacity и RAID.",
        ],
    }

    telegram_text = "\n".join(primary_commercial_telegram_lines(commercial))
    excel_text = json.dumps(primary_commercial_excel_rows(commercial), ensure_ascii=False)
    combined = f"{telegram_text}\n{excel_text}"

    assert "raw/usable" in combined
    assert "RAID" in combined
    for forbidden in ("CPU support", "QVL RAM", "DIMM", "NVMe/U.2/U.3", "backplane"):
        assert forbidden not in combined
