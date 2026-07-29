from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from app.core.config import LlmSettings, get_llm_settings
from app.llm.base import LlmClient, LlmConfigurationError, LlmInvalidJsonError
from app.llm.openai_compatible import OpenAICompatibleLlmClient
from app.llm.prompts import STOCK_SPEC_SYSTEM_PROMPT, build_stock_spec_user_prompt
from app.matching.spec_schema import StockSpec, StockSpecExtractionResult, StockSpecItem

DISABLED_PROVIDERS = {"", "disabled", "none", "off"}
OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openai-compatible", "openai_compatible"}


def extract_stock_spec_from_text(
    text: str,
    *,
    llm_client: LlmClient | None = None,
    settings: LlmSettings | None = None,
) -> StockSpecExtractionResult:
    clean_text = text.strip()
    if not clean_text:
        return StockSpecExtractionResult(
            spec_json=StockSpec(source_text=text),
            confirmation_text="Не удалось извлечь Stock Spec: текст запроса пустой.",
            unclear_points=["Текст запроса пустой."],
            risk_flags=["empty_request"],
        )

    if llm_client is not None:
        return _extract_with_llm(clean_text, llm_client)

    effective_settings = settings or get_llm_settings()
    provider = effective_settings.llm_provider.strip().lower()
    if provider in DISABLED_PROVIDERS:
        return _extract_with_rules(clean_text)

    client = _build_llm_client(effective_settings)
    try:
        return _extract_with_llm(clean_text, client)
    finally:
        client.close()


def _build_llm_client(settings: LlmSettings) -> OpenAICompatibleLlmClient:
    provider = settings.llm_provider.strip().lower()
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleLlmClient(settings=settings)
    raise LlmConfigurationError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. "
        "Use disabled or openai-compatible."
    )


def _extract_with_llm(text: str, llm_client: LlmClient) -> StockSpecExtractionResult:
    payload = llm_client.generate_json(
        STOCK_SPEC_SYSTEM_PROMPT,
        build_stock_spec_user_prompt(text),
    )
    return _validate_llm_payload(payload, text)


def _validate_llm_payload(payload: dict[str, Any], source_text: str) -> StockSpecExtractionResult:
    try:
        result = StockSpecExtractionResult.model_validate(payload)
    except ValidationError as exc:
        raise LlmInvalidJsonError(
            "LLM JSON did not match the Stock Spec extraction schema."
        ) from exc

    if result.spec_json.source_text:
        return result

    spec_json = result.spec_json.model_copy(update={"source_text": source_text})
    return result.model_copy(update={"spec_json": spec_json})


def _extract_with_rules(text: str) -> StockSpecExtractionResult:
    requirements: dict[str, Any] = {}
    lower_text = text.casefold()

    quantity = _find_quantity(lower_text)
    item_type = "server" if _mentions_server(lower_text) else "unknown"

    form_factor = _find_form_factor(lower_text)
    if form_factor:
        requirements["form_factor"] = form_factor

    cpu_sockets = _find_cpu_sockets(lower_text)
    if cpu_sockets is not None:
        requirements["cpu"] = {"sockets": cpu_sockets}

    ram_min_gb = _find_ram_min_gb(lower_text)
    ram_type = _find_ram_type(lower_text)
    if ram_min_gb is not None and ram_type is None and _ram_per_server_phrase(lower_text):
        ram_type = "DDR5"
    if ram_min_gb is not None:
        requirements["ram"] = {"min_gb": ram_min_gb}
        if ram_type is not None:
            requirements["ram"]["type"] = ram_type

    storage_type = _find_storage_type(lower_text)
    if storage_type:
        requirements["storage"] = {"type": storage_type}

    if _mentions_network_adapter(lower_text):
        requirements["network"] = {"adapter_required": True}

    psu_count = _find_psu_count(lower_text)
    if psu_count is not None:
        requirements["power"] = {
            "psu_count": psu_count,
            "redundant_psu": psu_count >= 2,
        }

    shipment_city = _find_shipment_city(lower_text)
    item = StockSpecItem(
        item_type=item_type,
        quantity=quantity,
        name="server" if item_type == "server" else None,
        server_qty=quantity if item_type == "server" else None,
        form_factor=form_factor,
        cpu_per_server=cpu_sockets,
        total_cpu_required=cpu_sockets * quantity if cpu_sockets is not None else None,
        ram_gb_per_server=ram_min_gb,
        ram_type_preference=ram_type,
        storage_required=storage_type is not None,
        storage_type_preference=storage_type,
        psu_count_per_server=psu_count,
        location=shipment_city,
        requirements=requirements,
    )
    spec = StockSpec(
        items=[item],
        shipment_city=shipment_city,
        server_qty=quantity if item_type == "server" else None,
        form_factor=form_factor,
        cpu_per_server=cpu_sockets,
        total_cpu_required=cpu_sockets * quantity if cpu_sockets is not None else None,
        ram_gb_per_server=ram_min_gb,
        ram_type_preference=ram_type,
        storage_required=storage_type is not None,
        storage_type_preference=storage_type,
        psu_count_per_server=psu_count,
        location=shipment_city,
        source_text=text,
    )

    unclear_points: list[str] = []
    if item_type == "unknown":
        unclear_points.append("Тип оборудования не распознан.")
    if not requirements:
        unclear_points.append("Технические требования не распознаны.")

    return StockSpecExtractionResult(
        spec_json=spec,
        confirmation_text=_build_fallback_confirmation(item, shipment_city),
        unclear_points=unclear_points,
        risk_flags=["llm_disabled_rule_based_fallback"],
    )


def _mentions_server(text: str) -> bool:
    return bool(re.search(r"\bservers?\b|\bсервер(?:а|ов|ы)?\b", text, re.IGNORECASE))


def _find_quantity(text: str) -> int:
    match = re.search(r"\b(\d+)\s*(?:servers?|сервер(?:а|ов|ы)?)\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 1


def _find_form_factor(text: str) -> str | None:
    match = re.search(r"\b([1-9]\d?)\s*u\b", text, re.IGNORECASE)
    if not match:
        return None
    return f"{match.group(1)}U"


def _find_cpu_sockets(text: str) -> int | None:
    match = re.search(
        r"\b(\d+)\s*(?:cpu|процессор(?:а|ов)?|проца|сокет(?:а|ов)?)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1))


def _find_ram_min_gb(text: str) -> int | None:
    ram_marker = r"(?:ram|озу|оператив(?:ной)?\s+памяти|оперативк[аи])"
    patterns = (
        rf"\b(?:по\s+)?(\d+)\s*(?:гб|gb)\s*{ram_marker}\b",
        rf"\b{ram_marker}\s*(?:на\s+сервер)?\s*(\d+)\s*(?:гб|gb)\b",
        r"\b(?:по\s+)?(\d+)\s*(?:гб|gb)\s*(?=ddr\s*[345]\b)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    match = re.search(r"\b(?:по\s+)?(\d+)\s*(?:гб|gb)\s*\b", text, re.IGNORECASE)
    if not match:
        return None
    segment = text[max(0, match.start() - 40) : match.end() + 40]
    if re.search(
        rf"{ram_marker}|ddr\s*[345]|\bmemory\b|\bпамят",
        segment,
        re.IGNORECASE,
    ) and not re.search(
        r"\b(?:ssd|hdd|nvme|sas|sata|накопител|диск)\b",
        segment,
        re.IGNORECASE,
    ):
        return int(match.group(1))
    return None


def _ram_per_server_phrase(text: str) -> bool:
    ram_marker = r"(?:ram|озу|оператив(?:ной)?\s+памяти|оперативк[аи])"
    return bool(
        re.search(
            rf"\bпо\s+\d+\s*(?:гб|gb)\s*{ram_marker}\s+на\s+сервер\b",
            text,
            re.IGNORECASE,
        )
    )


def _find_ram_type(text: str) -> str | None:
    match = re.search(r"\bddr\s*([345])\b", text, re.IGNORECASE)
    if match is None:
        return None
    return f"DDR{match.group(1)}"


def _find_storage_type(text: str) -> str | None:
    if re.search(r"\bssd\b|ссд", text, re.IGNORECASE):
        return "SSD"
    if re.search(r"\bhdd\b|жестк(?:ий|ие|их)\s+диск", text, re.IGNORECASE):
        return "HDD"
    return None


def _mentions_network_adapter(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:nic|ethernet|lan|10g|25g|40g|100g)\b|сетев|порт",
            text,
            re.IGNORECASE,
        )
    )


def _find_psu_count(text: str) -> int | None:
    match = re.search(
        r"\b(\d+)\s*(?:бп|psu|блока?\s+питания|блоков\s+питания)\b",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return int(match.group(1))


def _find_shipment_city(text: str) -> str | None:
    if re.search(r"\bмоскв(?:а|е|ы|у|ой)?\b|\bmoscow\b", text, re.IGNORECASE):
        return "Москва"
    return None


def _build_fallback_confirmation(item: StockSpecItem, shipment_city: str | None) -> str:
    parts = [f"quantity={item.quantity}", f"item_type={item.item_type}"]
    if item.requirements:
        parts.append(f"requirements={item.requirements}")
    if shipment_city:
        parts.append(f"shipment_city={shipment_city}")
    return "Извлечены требования rule-based parser: " + ", ".join(parts) + "."
