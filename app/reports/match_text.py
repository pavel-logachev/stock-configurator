from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

STATUS_MESSAGES = {
    "partial_stock_matched": "Найдены варианты, но полное соответствие не подтверждено",
    "matched": "Есть варианты, которые предварительно подходят",
    "stock_matched": "Есть варианты, которые предварительно подходят",
    "no_stock_match": "Подходящих складских вариантов не найдено",
    "failed": "Не удалось выполнить подбор",
    "error": "Не удалось выполнить подбор",
}


def pluralize_ru(count: Any, one: str, few: str, many: str) -> str:
    """Return the Russian word form for a numeric count."""
    try:
        number = abs(int(count))
    except (TypeError, ValueError):
        number = 0
    if 11 <= number % 100 <= 14:
        return many
    last_digit = number % 10
    if last_digit == 1:
        return one
    if last_digit in {2, 3, 4}:
        return few
    return many


def human_match_status(status: Any) -> str:
    status_text = str(status or "unknown")
    return STATUS_MESSAGES.get(status_text, f"Статус подбора: {status_text}")


def human_match_status_for_summary(status: Any) -> str:
    status_text = str(status or "unknown")
    message = human_match_status(status_text)
    if status_text in STATUS_MESSAGES:
        return message[:1].lower() + message[1:]
    return message


def yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if value is None:
        return "нет"
    return "да" if str(value).strip().lower() in {"1", "true", "yes", "да"} else "нет"


def humanize_check_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""

    if text == "engineer_review_required":
        return "Нужна инженерная проверка результата."

    ram_match = re.search(
        r"(?:RAM|Оперативная память).*?(?:ниже|below).*?"
        r"(?:найдено\s*)?(\d+)\s*(?:GB|ГБ).*?"
        r"(?:требуется|required)\s*(\d+)\s*(?:GB|ГБ)",
        text,
        re.IGNORECASE,
    )
    if ram_match:
        found, required = ram_match.groups()
        return (
            "Оперативная память ниже требования: "
            f"найдено {found} ГБ, требуется {required} ГБ."
        )

    if "RAM below requirement" in text:
        return "Оперативная память ниже требования."

    stock_match = re.search(
        r"(?:Недостаточный остаток|Остаток ниже требования|Insufficient stock).*?"
        r"(?:доступно|available)\s*(\d+).*?"
        r"(?:требуется|required)\s*(\d+)",
        text,
        re.IGNORECASE,
    )
    if stock_match:
        available, required = stock_match.groups()
        return (
            "По одному варианту не хватает остатка: "
            f"доступно {available} шт., требуется {required} шт."
        )

    if re.search(r"Гарантия.*OCS.*12|OCS.*12.*warranty", text, re.IGNORECASE):
        return "Гарантия у OCS указана 12 месяцев, нужно сверить с требованиями."

    if "Нет кандидата" in text and ("Stock Spec" in text or "V0" in text):
        return "Нет варианта, который полностью закрывает запрос по правилам текущей версии."

    replacements_before_ram = {
        "Совместимость RAM": "Совместимость оперативной памяти",
        "совместимость RAM": "совместимость оперативной памяти",
        "совместимости RAM": "совместимости оперативной памяти",
        "Тип RAM": "Тип оперативной памяти",
        "тип RAM": "тип оперативной памяти",
        "CPU support list": "список поддерживаемых CPU",
        "support list CPU kit": "список поддерживаемых CPU для комплекта CPU",
        "support list": "список поддерживаемых компонентов",
        "Storage": "Накопители",
        "storage": "накопители",
        "Vendor CPU": "Производитель CPU",
        "vendor CPU": "производитель CPU",
        "Check platform": "Проверить платформу:",
        "check platform": "проверить платформу:",
        "Check": "Проверить",
        "check": "проверить",
        "cores": "ядра",
        "Cores": "Ядра",
        "overfit": "выше требования",
        "Overfit": "Выше требования",
    }
    for source, replacement in replacements_before_ram.items():
        text = text.replace(source, replacement)

    replacements = {
        "Stock Spec": "запрос",
        "Match Engine V0": "текущая версия подбора",
        "V0": "текущей версии",
        "RAM": "оперативная память",
        "кандидата": "варианта",
        "Кандидаты": "Варианты",
        "кандидаты": "варианты",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)

    grammar_fixes = {
        "Совместимость оперативная память": "Совместимость оперативной памяти",
        "совместимость оперативная память": "совместимость оперативной памяти",
        "Совместимости оперативная память": "Совместимости оперативной памяти",
        "совместимости оперативная память": "совместимости оперативной памяти",
        "Тип оперативная память": "Тип оперативной памяти",
        "тип оперативная память": "тип оперативной памяти",
        "оперативная память требует": "Оперативная память требует",
    }
    for source, replacement in grammar_fixes.items():
        text = text.replace(source, replacement)

    return _ensure_sentence(_capitalize_first_letter(text))


def candidate_display_name(candidate: Mapping[str, Any]) -> str:
    producer = _clean_text(candidate.get("producer"))
    part_number = _clean_text(candidate.get("part_number"))
    item_name = _clean_text(candidate.get("item_name"))
    item_id = _clean_text(candidate.get("item_id"))
    product_key = _clean_text(candidate.get("product_key"))

    if producer and part_number:
        if producer.casefold() in part_number.casefold():
            return part_number
        return f"{producer} {part_number}"
    if part_number:
        return part_number
    if item_name:
        return item_name
    if item_id:
        return item_id
    if product_key:
        return product_key
    return "артикул не указан"


def candidate_article(candidate: Mapping[str, Any]) -> str:
    return _clean_text(candidate.get("part_number")) or _clean_text(candidate.get("item_id")) or ""


def candidate_outcome(candidate: Mapping[str, Any]) -> str:
    if candidate.get("candidate_type") == "build_from_parts":
        completeness_label = _clean_text(candidate.get("completeness_label"))
        if completeness_label:
            return completeness_label
        completeness_status = _clean_text(candidate.get("completeness_status"))
        if completeness_status == "incomplete":
            return "Неполная сборка"
        if completeness_status == "complete":
            return "Предварительная сборка"

    missing = as_string_list(candidate.get("missing_requirements"))
    if missing:
        return "Требует проверки требований"

    score = _as_int(candidate.get("confidence_score"))
    if score is not None and score >= 70:
        return "Предварительно подходит"
    return "Нужна инженерная проверка"


def candidate_comment(candidate: Mapping[str, Any]) -> str:
    checks = humanized_checks(
        risk_flags=as_string_list(candidate.get("risk_flags")),
        missing_requirements=as_string_list(candidate.get("missing_requirements")),
    )
    return " ".join(checks)


def humanized_checks(
    *,
    risk_flags: Iterable[Any],
    missing_requirements: Iterable[Any],
) -> list[str]:
    checks: list[str] = []
    for value in [*risk_flags, *missing_requirements]:
        for text in humanize_check_items(value):
            if text and _check_key(text) not in {_check_key(check) for check in checks}:
                checks.append(text)
    return checks


def humanize_check_items(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []

    lowered = text.casefold()
    if _is_dropped_generic_check(lowered):
        return []

    canonical_checks: list[str] = []

    if "optional_or_engineer_check" in lowered or "опциональ" in lowered:
        canonical_checks.append(
            "Проверить, нужны ли опциональные контроллеры или сетевые адаптеры; "
            "они не входят в минимальный обязательный расчет."
        )
        return _dedupe_preserve_order(canonical_checks)

    if _is_generic_platform_compatibility_check(lowered):
        canonical_checks.extend(
            [
                "Проверить CPU по списку поддерживаемых процессоров платформы.",
                (
                    "Подтвердить правила заполнения DIMM-слотов для выбранного количества "
                    "модулей RAM на сервер."
                ),
                "Проверить совместимость NVMe/SATA SSD с backplane/контроллером платформы.",
            ]
        )
    else:
        if _mentions_cpu_support(lowered):
            canonical_checks.append(
                "Проверить CPU по списку поддерживаемых процессоров платформы."
            )
        if _mentions_ram_layout(lowered):
            canonical_checks.append(
                "Подтвердить правила заполнения DIMM-слотов для выбранного количества "
                "модулей RAM на сервер."
            )
        if _mentions_storage_backplane(lowered):
            canonical_checks.append(
                "Проверить совместимость NVMe/SATA SSD с backplane/контроллером платформы."
            )
        if _mentions_platform_bom(lowered):
            canonical_checks.append(
                "Подтвердить комплектацию 2 БП, корзины и кабели по спецификации платформы."
            )
        if _mentions_warranty(lowered):
            canonical_checks.append("Сверить гарантию OCS и требования заказчика.")

    if canonical_checks:
        return _dedupe_preserve_order(canonical_checks)

    fallback = humanize_check_text(text)
    if not fallback or _is_dropped_generic_check(fallback.casefold()):
        return []
    return [fallback]


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    keys: set[str] = set()
    for value in values:
        key = _check_key(value)
        if key and key not in keys:
            keys.add(key)
            result.append(value)
    return result


def _check_key(value: str) -> str:
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def _is_dropped_generic_check(lowered: str) -> bool:
    banned_fragments = (
        "llm composer не является финальной инженерной проверкой совместимости",
        "не является финальной инженерной проверкой совместимости",
    )
    return any(fragment in lowered for fragment in banned_fragments)


def _is_generic_platform_compatibility_check(lowered: str) -> bool:
    return (
        "инженер" in lowered
        and "совместим" in lowered
        and "платформ" in lowered
        and (
            "оператив" in lowered
            or "ram" in lowered
            or "накопител" in lowered
            or "storage" in lowered
            or "адаптер" in lowered
        )
    )


def _mentions_cpu_support(lowered: str) -> bool:
    return (
        (
            "cpu" in lowered
            or "процессор" in lowered
            or ("support list" in lowered and "platform" in lowered)
        )
        and (
            "support list" in lowered
            or "спис" in lowered
            or "совместим" in lowered
            or "платформ" in lowered
            or "preliminary" in lowered
        )
    )


def _mentions_ram_layout(lowered: str) -> bool:
    return (
        ("ram" in lowered or "оператив" in lowered or "dimm" in lowered)
        and (
            "совместим" in lowered
            or "слот" in lowered
            or "slot" in lowered
            or "правил" in lowered
            or "платформ" in lowered
            or "провер" in lowered
        )
    )


def _mentions_storage_backplane(lowered: str) -> bool:
    return (
        (
            "ssd" in lowered
            or "hdd" in lowered
            or "nvme" in lowered
            or "накопител" in lowered
            or "storage" in lowered
        )
        and (
            "backplane" in lowered
            or "контроллер" in lowered
            or "controller" in lowered
            or "совместим" in lowered
            or "платформ" in lowered
            or "провер" in lowered
        )
    )


def _mentions_platform_bom(lowered: str) -> bool:
    return (
        "бп" in lowered
        or "psu" in lowered
        or "корзин" in lowered
        or "кабел" in lowered
        or ("комплектац" in lowered and "платформ" in lowered)
    )


def _mentions_warranty(lowered: str) -> bool:
    return "гарант" in lowered or "warranty" in lowered


def short_conclusion(
    *,
    status: Any,
    engineer_review_required: Any,
) -> str:
    conclusion = human_match_status(status)
    if yes_no(engineer_review_required) == "да":
        return f"{conclusion}. Перед финальным предложением нужна проверка инженера."
    return f"{conclusion}. Инженерная проверка не отмечена как обязательная."


def format_price(value: Any, currency: Any) -> str:
    if value is None or value == "":
        return "не указана"

    if isinstance(value, Decimal):
        normalized = value.normalize()
        if normalized == normalized.to_integral():
            formatted = str(normalized.quantize(Decimal("1")))
        else:
            formatted = format(normalized, "f")
    else:
        formatted = str(value)

    currency_text = str(currency or "").strip()
    return f"{formatted} {currency_text}".strip()


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def recommended_action(text: str) -> str:
    lowered = text.casefold()
    if "cpu" in lowered or "процессор" in lowered:
        return "Проверить и подтвердить серверные процессоры и совместимость с платформой."
    if "памят" in lowered or "ram" in lowered:
        return "Сверить конфигурацию памяти с требованием."
    if "остат" in lowered:
        return "Проверить доступный остаток и возможность резерва."
    if "гарант" in lowered:
        return "Сверить гарантию с требованиями."
    if "полностью закрывает" in lowered:
        return "Проверить состав варианта и закрытие всех требований."
    return "Проверить вручную перед финальным предложением."


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _ensure_sentence(text: str) -> str:
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."


def _capitalize_first_letter(text: str) -> str:
    if not text:
        return ""
    return text[:1].upper() + text[1:]


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
