from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

CJK_PATTERN = re.compile(
    "["
    "\u3040-\u30ff"
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\uf900-\ufaff"
    "\uac00-\ud7af"
    "\U00020000-\U0002ebef"
    "]"
)

DETAILED_ENGINEER_CHECK_DEFAULTS = [
    "Проверить CPU support list платформы и версию BIOS.",
    "Проверить QVL RAM и правила заполнения DIMM.",
    "Проверить NVMe/U.2/U.3 backplane.",
    "Проверить комплектацию БП, кулеры, рейки и кабели.",
    "Проверить гарантию и срок поставки перед КП.",
]

GROUPED_ENGINEER_CHECK_DEFAULTS = [
    "CPU support list / BIOS.",
    "QVL RAM и правила заполнения DIMM.",
    "NVMe/U.2/U.3 backplane.",
    "Комплектация БП, кулеры, рейки и кабели.",
    "Гарантия и срок поставки.",
]

NETWORK_ENGINEER_CHECK_DEFAULTS = [
    "Проверить количество и тип портов.",
    "Проверить access/uplink схему.",
    "Проверить скорости, media и совместимость трансиверов/DAC.",
    "Проверить PoE standard и PoE budget.",
    "Проверить L2/L3 feature set.",
    "Проверить stacking compatibility, лицензии и кабели, если stacking требуется.",
    "Проверить airflow, support/warranty и срок поставки.",
    "Проверить финальную спецификацию перед КП.",
]

GROUPED_NETWORK_ENGINEER_CHECK_DEFAULTS = [
    "Количество и тип портов.",
    "Access/uplink схема.",
    "Скорости, media и трансиверы/DAC.",
    "PoE standard/budget.",
    "L2/L3, stacking, лицензии и support.",
]

STORAGE_ENGINEER_CHECK_DEFAULTS = [
    "Проверить raw/usable capacity и RAID/модель избыточности.",
    "Проверить количество и резервирование контроллеров.",
    "Проверить тип, интерфейс и количество дисков.",
    "Проверить FC/iSCSI/NVMe-oF/SAS порты.",
    "Проверить совместимость полок и кабелей.",
    "Проверить лицензии, support/warranty и срок поставки.",
    "Проверить финальную спецификацию СХД перед КП.",
]

GROUPED_STORAGE_ENGINEER_CHECK_DEFAULTS = [
    "Raw/usable capacity и RAID/redundancy.",
    "Контроллеры и резервирование.",
    "Тип, интерфейс и количество дисков.",
    "FC/iSCSI/NVMe-oF/SAS порты.",
    "Полки/кабели, лицензии и support.",
]

_INTERNAL_LABEL_RE = re.compile(
    r"\b(?:"
    r"group_title|architecture_summary|why_group_matters|why_this_platform|"
    r"tradeoffs?|engineer_checks?|quote_recommendation|commercial_tradeoff|"
    r"right_size_note|raw_json|raw json"
    r")\s*[:=]\s*",
    flags=re.IGNORECASE,
)

_COMPONENT_ID_RE = re.compile(
    r"\bcomponent_candidate_id\s*[:=]\s*[\w./:-]+",
    flags=re.IGNORECASE,
)

_LLM_REC_RE = re.compile(r"\bllm_rec_[\w-]+\b\s*:?", flags=re.IGNORECASE)

_USER_TEXT_REPLACEMENTS = {
    "с高密度": "с высокоплотными",
    "高密度": "высокоплотный",
    "overfit": "компонент выше требования",
    "Overfit": "Компонент выше требования",
    "cores": "ядра",
    "Cores": "Ядра",
    "fit label": "оценка соответствия",
    "fit_label": "оценка соответствия",
    "Fit label": "Оценка соответствия",
    "raw JSON": "служебные данные",
    "raw json": "служебные данные",
    "raw_json": "служебные данные",
    "CPU-кандидат": "CPU",
    "cpu-кандидат": "CPU",
    "валидный": "прошедший проверки",
    "валидная": "прошедшая проверки",
    "валидные": "прошедшие проверки",
    "CPU support list": "список поддерживаемых CPU",
    "support list": "список поддерживаемых компонентов",
    "storage": "накопители",
    "Check platform": "Проверить платформу:",
    "check platform": "проверить платформу:",
    "Check": "Проверить",
    "check": "проверить",
    "web evidence not found": "внешние источники не подтвердили совместимость",
    "keep engineer": "требуется инженерная проверка",
}

_GRAMMAR_FIXES = {
    "32 ядер": "32 ядра",
    "24 ядер": "24 ядра",
    "16 ядра": "16 ядер",
    "с высокоплотный": "с высокоплотными",
    "оперативная память требует": "Оперативная память требует",
    "Совместимость оперативная память": "Совместимость оперативной памяти",
    "совместимость оперативная память": "совместимость оперативной памяти",
    "Тип оперативная память": "Тип оперативной памяти",
    "тип оперативная память": "тип оперативной памяти",
}

_ENGINEER_CHECK_CANONICAL_DETAILED = {
    "cpu": "Проверить CPU support list платформы и версию BIOS.",
    "ram": "Проверить QVL RAM и правила заполнения DIMM.",
    "storage": "Проверить NVMe/U.2/U.3 backplane.",
    "kit": "Проверить комплектацию БП, кулеры, рейки и кабели.",
    "warranty": "Проверить гарантию и срок поставки перед КП.",
}

_ENGINEER_CHECK_CANONICAL_GROUPED = {
    "cpu": "CPU support list / BIOS.",
    "ram": "QVL RAM и правила заполнения DIMM.",
    "storage": "NVMe/U.2/U.3 backplane.",
    "kit": "Комплектация БП, кулеры, рейки и кабели.",
    "warranty": "Гарантия и срок поставки.",
}


def sanitize_user_facing_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    text = _COMPONENT_ID_RE.sub("", text)
    text = _LLM_REC_RE.sub("", text)
    text = _INTERNAL_LABEL_RE.sub("", text)
    text = re.sub(r"\bcomponent_candidate_id\b", "", text, flags=re.IGNORECASE)
    for source, replacement in _USER_TEXT_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    text = CJK_PATTERN.sub("", text)
    for source, replacement in _GRAMMAR_FIXES.items():
        text = text.replace(source, replacement)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ;:-")


def contains_cjk_text(value: Any) -> bool:
    return CJK_PATTERN.search(str(value or "")) is not None


def human_engineering_confidence_label(value: Any) -> str:
    raw = str(value or "").strip()
    labels = {
        "preliminary_requires_engineer_review": "предварительно, нужна инженерная проверка",
        "confirmed": "подтверждено, финально проверить комплектность",
        "partially_confirmed": "частично подтверждено, нужна инженерная проверка",
        "not_confirmed_requires_engineer_review": (
            "не подтверждено, нужна инженерная проверка"
        ),
        "not_confirmed": "не подтверждено, нужна инженерная проверка",
    }
    if raw in labels:
        return labels[raw]
    text = sanitize_user_facing_text(raw)
    return text if text else "предварительно, нужна инженерная проверка"


def deduplicate_engineer_checks(
    values: Sequence[Any] | Iterable[Any],
    *,
    defaults: Sequence[str] = DETAILED_ENGINEER_CHECK_DEFAULTS,
    grouped_summary: bool = False,
    max_items: int | None = None,
) -> list[str]:
    canonical = (
        _ENGINEER_CHECK_CANONICAL_GROUPED
        if grouped_summary
        else _ENGINEER_CHECK_CANONICAL_DETAILED
    )
    ordered_values = [*defaults, *list(values)]
    result: list[str] = []
    seen_categories: set[str] = set()
    seen_texts: set[str] = set()
    for value in ordered_values:
        text = sanitize_user_facing_text(value)
        if not text:
            continue
        categories = _engineer_check_categories(text)
        if categories:
            for category in categories:
                if category in seen_categories:
                    continue
                seen_categories.add(category)
                candidate = canonical[category]
                key = _engineer_check_text_key(candidate)
                if key not in seen_texts:
                    seen_texts.add(key)
                    result.append(candidate)
            continue
        key = _engineer_check_text_key(text)
        if key in seen_texts:
            continue
        seen_texts.add(key)
        result.append(text)
    if max_items is not None:
        return result[:max_items]
    return result


def sanitize_engineer_checks_for_product_group(
    values: Sequence[Any] | Iterable[Any],
    *,
    product_group: str | None,
    defaults: Sequence[str] | None = None,
    grouped_summary: bool = False,
    max_items: int | None = None,
) -> list[str]:
    normalized = _normalized_product_group(product_group)
    if normalized == "server":
        server_defaults = defaults
        if server_defaults is None:
            server_defaults = (
                GROUPED_ENGINEER_CHECK_DEFAULTS
                if grouped_summary
                else DETAILED_ENGINEER_CHECK_DEFAULTS
            )
        return deduplicate_engineer_checks(
            values,
            defaults=server_defaults,
            grouped_summary=grouped_summary,
            max_items=max_items,
        )

    default_values = (
        list(defaults)
        if defaults is not None
        else product_group_engineer_check_defaults(
            normalized,
            grouped_summary=grouped_summary,
        )
    )
    ordered_values = [*default_values, *list(values)]
    result: list[str] = []
    seen_texts: set[str] = set()
    for value in ordered_values:
        text = sanitize_user_facing_text(value)
        if not text or _is_forbidden_engineer_check(text, normalized):
            continue
        key = _engineer_check_text_key(text)
        if key in seen_texts:
            continue
        seen_texts.add(key)
        result.append(text)

    if not result:
        for value in product_group_engineer_check_defaults(
            normalized,
            grouped_summary=grouped_summary,
        ):
            text = sanitize_user_facing_text(value)
            key = _engineer_check_text_key(text)
            if text and key not in seen_texts:
                seen_texts.add(key)
                result.append(text)

    if max_items is not None:
        return result[:max_items]
    return result


def product_group_engineer_check_defaults(
    product_group: str | None,
    *,
    grouped_summary: bool = False,
) -> list[str]:
    normalized = _normalized_product_group(product_group)
    if normalized == "network":
        return list(
            GROUPED_NETWORK_ENGINEER_CHECK_DEFAULTS
            if grouped_summary
            else NETWORK_ENGINEER_CHECK_DEFAULTS
        )
    if normalized == "storage":
        return list(
            GROUPED_STORAGE_ENGINEER_CHECK_DEFAULTS
            if grouped_summary
            else STORAGE_ENGINEER_CHECK_DEFAULTS
        )
    return list(
        GROUPED_ENGINEER_CHECK_DEFAULTS
        if grouped_summary
        else DETAILED_ENGINEER_CHECK_DEFAULTS
    )


def grouped_engineer_check_summary(
    values: Sequence[Any] | Iterable[Any],
    *,
    product_group: str | None = None,
) -> list[str]:
    return sanitize_engineer_checks_for_product_group(
        values,
        product_group=product_group,
        grouped_summary=True,
        max_items=len(product_group_engineer_check_defaults(product_group, grouped_summary=True)),
    )


def _normalized_product_group(product_group: str | None) -> str:
    normalized = str(product_group or "server").strip().casefold()
    if normalized in {"network", "storage"}:
        return normalized
    return "server"


def _is_forbidden_engineer_check(text: str, product_group: str) -> bool:
    lowered = text.casefold()
    if _is_server_cpu_check(lowered) or _is_server_ram_check(lowered):
        return True
    if product_group == "network":
        return _is_server_storage_check(lowered) or _is_server_kit_check(lowered)
    if product_group == "storage":
        return _is_server_backplane_check(lowered) or _is_server_kit_check(lowered)
    return False


def _is_server_cpu_check(lowered: str) -> bool:
    return (
        "cpu" in lowered
        or "bios" in lowered
        or "процессор" in lowered
        or "процессорн" in lowered
        or "платформ" in lowered and "поддерж" in lowered
    )


def _is_server_ram_check(lowered: str) -> bool:
    return (
        "qvl" in lowered
        or "ram" in lowered
        or "dimm" in lowered
        or "rdimm" in lowered
        or "lrdimm" in lowered
        or "udimm" in lowered
        or "оперативн" in lowered
        or "памят" in lowered
    )


def _is_server_storage_check(lowered: str) -> bool:
    return (
        _is_server_backplane_check(lowered)
        or "ssd" in lowered
        or "hdd" in lowered
        or "накопител" in lowered
        or "диск" in lowered
    )


def _is_server_backplane_check(lowered: str) -> bool:
    return (
        "backplane" in lowered
        or "u.2" in lowered
        or "u.3" in lowered
        or "nvme/u" in lowered
    )


def _is_server_kit_check(lowered: str) -> bool:
    return (
        "cooler" in lowered
        or "кулер" in lowered
        or "охлаж" in lowered
        or "rail" in lowered
        or "рейк" in lowered
        or "сервер" in lowered and ("платформ" in lowered or "шасси" in lowered)
    )


def _engineer_check_categories(text: str) -> list[str]:
    lowered = text.casefold()
    categories: list[str] = []
    if (
        "cpu" in lowered
        or "bios" in lowered
        or "биос" in lowered
        or "процессор" in lowered
        or "поддерживаемых cpu" in lowered
    ):
        categories.append("cpu")
    if (
        "qvl" in lowered
        or "ram" in lowered
        or "dimm" in lowered
        or "памят" in lowered
    ):
        categories.append("ram")
    if (
        "nvme" in lowered
        or "u.2" in lowered
        or "u.3" in lowered
        or "backplane" in lowered
        or "ssd" in lowered
        or "hdd" in lowered
        or "накоп" in lowered
    ):
        categories.append("storage")
    if (
        "psu" in lowered
        or "tdp" in lowered
        or "cool" in lowered
        or "бп" in lowered
        or "питани" in lowered
        or "охлаж" in lowered
        or "кулер" in lowered
        or "рейк" in lowered
        or "rail" in lowered
        or "кабел" in lowered
        or "cable" in lowered
        or "салаз" in lowered
        or "sled" in lowered
    ):
        categories.append("kit")
    if (
        "гарант" in lowered
        or "warranty" in lowered
        or "срок" in lowered
        or "постав" in lowered
        or "delivery" in lowered
        or "lead time" in lowered
    ):
        categories.append("warranty")
    return _unique_preserving_order(categories)


def _engineer_check_text_key(text: str) -> str:
    key = sanitize_user_facing_text(text).casefold()
    return re.sub(r"[\W_]+", " ", key).strip()


def _unique_preserving_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
