from __future__ import annotations

from typing import Any

import pytest

from app.core.config import LlmSettings
from app.llm.stock_spec_extractor import extract_stock_spec_from_text

SERVER_REQUEST = "Нужно 2 сервера 2U, 2 процессора, 512 ГБ RAM, SSD, 2 БП, склад Москва"


class FakeLlmClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, str]] = []

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.payload


def test_rule_based_fallback_extracts_basic_server_request() -> None:
    result = extract_stock_spec_from_text(
        SERVER_REQUEST,
        settings=LlmSettings(llm_provider="disabled"),
    )

    assert result.spec_json.shipment_city == "Москва"
    assert result.unclear_points == []
    assert "llm_disabled_rule_based_fallback" in result.risk_flags

    item = result.spec_json.items[0]
    assert item.item_type == "server"
    assert item.quantity == 2
    assert item.requirements["form_factor"] == "2U"
    assert item.requirements["cpu"]["sockets"] == 2
    assert item.requirements["ram"]["min_gb"] == 512
    assert item.requirements["storage"]["type"] == "SSD"
    assert item.requirements["power"]["redundant_psu"] is True


@pytest.mark.parametrize(
    ("phrase", "expected_ram_type"),
    [
        ("512 ГБ RAM DDR5", "DDR5"),
        ("512 GB RAM DDR5", "DDR5"),
        ("512 ГБ оперативной памяти DDR5", "DDR5"),
        ("по 512 ГБ RAM на сервер", "DDR5"),
        ("на каждый сервер 512 ГБ RAM DDR5", "DDR5"),
        ("512 гб озу ddr5", "DDR5"),
        ("по 512 гб оперативки DDR5", "DDR5"),
    ],
)
def test_rule_based_fallback_extracts_ram_amount_and_type(
    phrase: str,
    expected_ram_type: str | None,
) -> None:
    result = extract_stock_spec_from_text(
        f"Нужно 2 сервера, {phrase}, склад Москва",
        settings=LlmSettings(llm_provider="disabled"),
    )

    item = result.spec_json.items[0]
    assert item.ram_gb_per_server == 512
    assert item.requirements["ram"]["min_gb"] == 512
    if expected_ram_type is None:
        assert item.ram_type_preference is None
        assert "type" not in item.requirements["ram"]
    else:
        assert item.ram_type_preference == expected_ram_type
        assert item.requirements["ram"]["type"] == expected_ram_type


def test_llm_extractor_accepts_fake_client_and_adds_source_text() -> None:
    payload = {
        "spec_json": {
            "items": [
                {
                    "item_type": "server",
                    "quantity": 3,
                    "requirements": {"ram": {"min_gb": 1024}},
                }
            ],
            "shipment_city": "Москва",
        },
        "confirmation_text": "Извлечены требования к 3 серверам.",
        "unclear_points": ["Не указан форм-фактор."],
        "risk_flags": [],
    }
    client = FakeLlmClient(payload)

    result = extract_stock_spec_from_text("Нужно три сервера с 1 ТБ RAM", llm_client=client)

    assert result.spec_json.items[0].quantity == 3
    assert result.spec_json.items[0].requirements["ram"]["min_gb"] == 1024
    assert result.spec_json.source_text == "Нужно три сервера с 1 ТБ RAM"
    assert result.unclear_points == ["Не указан форм-фактор."]
    assert client.calls
    assert "Нужно три сервера" in client.calls[0]["user_prompt"]
