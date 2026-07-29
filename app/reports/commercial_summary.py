from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from app.reports.match_text import format_price, pluralize_ru
from app.user_facing_text import (
    contains_cjk_text,
    human_engineering_confidence_label,
    sanitize_engineer_checks_for_product_group,
    sanitize_user_facing_text,
)

COMMERCIAL_ENGINEER_CHECKS = [
    "CPU support list / BIOS",
    "QVL RAM и правила заполнения DIMM",
    "NVMe/U.2/U.3 backplane",
    "БП, кулеры, рейки и кабели",
    "гарантию и срок поставки",
]

PRIMARY_COMMERCIAL_TITLE = "Предварительная спецификация для КП"
PRIMARY_COMMERCIAL_COMMENT_LINES = [
    (
        "Подобрана минимальная по цене складская конфигурация, закрывающая "
        "требования запроса. Перед выпуском КП нужна инженерная проверка "
        "совместимости и комплектации."
    ),
]
NETWORK_COMMERCIAL_COMMENT_LINES = [
    (
        "Подобрана минимальная по цене складская спецификация сетевого оборудования, "
        "закрывающая hard-требования запроса. Перед выпуском КП обязательна "
        "инженерная проверка совместимости и комплектации."
    ),
]
NETWORK_COMMERCIAL_ENGINEER_CHECKS = [
    "количество и тип портов",
    "портовую схему access/uplink",
    "скорости, media и совместимость трансиверов/DAC",
    "PoE standard и PoE budget",
    "L2/L3 feature set",
    "stacking compatibility, лицензии и кабели, если stacking требуется",
    "airflow, support/warranty и срок поставки",
    "финальную спецификацию перед КП",
]
STORAGE_COMMERCIAL_COMMENT_LINES = [
    (
        "Подобрана предварительная складская спецификация СХД, закрывающая hard-требования "
        "по локальным данным. Перед выпуском КП обязательна инженерная проверка емкости, "
        "протоколов, лицензий/support и комплектности."
    ),
]
STORAGE_COMMERCIAL_ENGINEER_CHECKS = [
    "raw/usable емкость и модель RAID/erasure",
    "количество и резервирование контроллеров",
    "тип, интерфейс и количество дисков",
    "FC/iSCSI/NVMe-oF/SAS, скорость, media и количество host-портов",
    "совместимость полок и кабелей",
    "лицензии/support/warranty и срок поставки",
    "финальную спецификацию СХД перед КП",
]
LEGACY_PRIMARY_COMMERCIAL_TITLE = "Рекомендуемый вариант для самого дешевого КП"

REASON_TEMPLATES = {
    "cheapest_quote": "минимальная цена среди прошедших проверку складских вариантов",
    "preferred_for_database": (
        "более спокойный вариант под БД: бренд/запас по платформе/"
        "инженерная проверяемость"
    ),
    "engineering_clear": "альтернатива с более понятной инженерной проверкой",
    "branded_safe": "брендовая альтернатива",
}

_CPU_MODEL_RE = re.compile(
    r"\b(?:Intel\s+)?Xeon\s+(?:Gold|Silver|Platinum|Bronze|Max)\s+[A-Z0-9-]+\b",
    flags=re.IGNORECASE,
)
_CAPACITY_GB_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:GB|ГБ)\b", flags=re.IGNORECASE)
_CAPACITY_TB_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:TB|ТБ)\b", flags=re.IGNORECASE)
_DDR_RE = re.compile(r"\bDDR[345]\b", flags=re.IGNORECASE)
_SSD_SERIES_RE = re.compile(r"\bCD\d+-R\b", flags=re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_ALLOWED_TECH_WORDS = {
    "BIOS",
    "CPU",
    "DDR",
    "DIMM",
    "GB",
    "HDD",
    "NVME",
    "OCS",
    "PCI",
    "QVL",
    "RAM",
    "RDIMM",
    "SSD",
    "TB",
}


@dataclass(frozen=True)
class CommercialSummary:
    match_run_id: Any
    goal: str
    group_count: int
    platform_options_count: int
    shown_options_count: int
    server_quantity: int | None
    main_group: Mapping[str, Any]
    main_option: Mapping[str, Any]
    alternative_option: Mapping[str, Any] | None


def build_grouped_commercial_summary(
    summary: Mapping[str, Any],
    configuration_groups: list[Mapping[str, Any]],
    *,
    match_run_id: Any,
) -> CommercialSummary | None:
    main_group, main_option = _main_commercial_option(configuration_groups)
    if main_group is None or main_option is None:
        return None
    alternative = _alternative_commercial_option(configuration_groups, main_option)
    shown_options_count = 1 + (1 if alternative else 0)
    return CommercialSummary(
        match_run_id=match_run_id,
        goal=_goal_text(summary, main_group, main_option),
        group_count=len(configuration_groups),
        platform_options_count=sum(
            len(_mapping_rows(group.get("platform_options")))
            for group in configuration_groups
        ),
        shown_options_count=shown_options_count,
        server_quantity=_group_server_quantity(main_group),
        main_group=main_group,
        main_option=main_option,
        alternative_option=alternative,
    )


def build_primary_commercial_summary(
    summary: Mapping[str, Any],
    primary_recommendation: Mapping[str, Any],
    *,
    match_run_id: Any,
) -> dict[str, Any] | None:
    if not isinstance(primary_recommendation, Mapping) or not primary_recommendation:
        return None
    if (
        _safe_string_list(primary_recommendation.get("missing_components"))
        or _safe_string_list(primary_recommendation.get("missing_requirements"))
        or _mapping_rows(primary_recommendation.get("missing_required_capabilities"))
    ):
        return _no_recommendation_commercial_summary(
            primary_recommendation,
            match_run_id=match_run_id,
    )
    existing = summary.get("commercial_summary")
    existing_summary = existing if isinstance(existing, Mapping) else {}
    product_group = _primary_product_group(summary, primary_recommendation, existing_summary)
    is_network = product_group == "network"
    is_storage = product_group == "storage"
    components = _mapping_rows(primary_recommendation.get("components"))
    server_quantity = _primary_server_quantity(primary_recommendation)
    if server_quantity is None:
        server_quantity = _int_value(existing_summary.get("server_quantity"))
    total_price_value = (
        primary_recommendation.get("total_price_value")
        or existing_summary.get("total_price_value")
    )
    total_price_currency = (
        primary_recommendation.get("total_price_currency")
        or existing_summary.get("total_price_currency")
    )
    bom_rows = _primary_bom_rows(
        components=components,
        server_quantity=server_quantity,
        primary_recommendation=primary_recommendation,
    )
    per_server_lines = _per_server_lines_from_bom_rows(
        bom_rows,
        product_group=product_group,
    )
    total_order_lines = _total_order_lines_from_bom_rows(bom_rows)
    price = _option_price_text(
        {
            "total_price_value": total_price_value,
            "total_price_currency": total_price_currency,
        }
    )
    price_line = (
        f"Ориентировочно за {_order_quantity_text(server_quantity, product_group)}: {price}"
        if price
        else ""
    )
    server_line = (
        f"Сетевое оборудование - {_piece_quantity(server_quantity)}"
        if is_network
        else (
            f"СХД - {_piece_quantity(server_quantity)}"
            if is_storage
            else f"Сервер в сборе - {_piece_quantity(server_quantity)}"
        )
    )
    comment_lines = _commercial_comment_lines(product_group)
    engineer_checks = _commercial_engineer_checks(product_group)
    lines = _primary_copy_lines(
        server_quantity=server_quantity,
        per_server_lines=per_server_lines,
        total_order_lines=total_order_lines,
        price_line=price_line,
        comment_lines=comment_lines,
        engineer_checks=engineer_checks,
        product_group=product_group,
    )
    if not bom_rows:
        existing_lines = _safe_summary_lines(existing_summary.get("lines"))
        if existing_lines:
            lines = _normalize_legacy_primary_lines(existing_lines)
    return {
        "mode": "single_best_cost_valid",
        "match_run_id": match_run_id,
        "title": PRIMARY_COMMERCIAL_TITLE,
        "product_group": product_group,
        "server_quantity": server_quantity,
        "device_quantity": server_quantity if is_network or is_storage else None,
        "total_price_value": total_price_value,
        "total_price_currency": total_price_currency,
        "server_line": server_line,
        "price_line": price_line or _safe_summary_text(existing_summary.get("price_line")),
        "per_server_lines": per_server_lines,
        "total_order_lines": total_order_lines,
        "comment_lines": comment_lines,
        "engineer_checks": engineer_checks,
        "bom_rows": bom_rows,
        "copy_paste_text": "\n".join(lines),
        "lines": lines,
    }


def _no_recommendation_commercial_summary(
    primary_recommendation: Mapping[str, Any],
    *,
    match_run_id: Any,
) -> dict[str, Any]:
    missing_required_capabilities = _mapping_rows(
        primary_recommendation.get("missing_required_capabilities")
    )
    reasons = [
        *_safe_string_list(primary_recommendation.get("missing_components")),
        *_safe_string_list(primary_recommendation.get("missing_requirements")),
    ]
    capability_lines: list[str] = []
    for capability in missing_required_capabilities:
        source_text = _safe_summary_text(
            capability.get("source_text")
            or capability.get("requirement_text")
            or capability.get("capability_id")
            or capability.get("role")
        )
        user_message = _safe_summary_text(
            capability.get("user_message") or capability.get("reason") or ""
        )
        if source_text:
            capability_lines.append(f"Не закрыто требование: {source_text}.")
        if user_message:
            capability_lines.append(f"Причина: {user_message}")
        reason = str(capability.get("reason") or capability.get("role") or "").strip()
        if reason:
            reasons.append(reason)
    lines = [
        "Безопасную складскую рекомендацию дать нельзя.",
        *capability_lines,
        *[f"- {_safe_summary_text(reason)}" for reason in reasons if _safe_summary_text(reason)],
    ]
    return {
        "mode": "single_best_cost_valid",
        "status": "no_recommendation",
        "match_run_id": match_run_id,
        "title": "Безопасную складскую рекомендацию дать нельзя.",
        "reasons": reasons,
        "missing_required_capabilities": list(missing_required_capabilities),
        "copy_paste_text": "\n".join(lines),
        "lines": lines,
    }


def _commercial_comment_lines(product_group: str) -> list[str]:
    if product_group == "network":
        return list(NETWORK_COMMERCIAL_COMMENT_LINES)
    if product_group == "storage":
        return list(STORAGE_COMMERCIAL_COMMENT_LINES)
    return list(PRIMARY_COMMERCIAL_COMMENT_LINES)


def _commercial_engineer_checks(product_group: str) -> list[str]:
    if product_group == "network":
        return list(NETWORK_COMMERCIAL_ENGINEER_CHECKS)
    if product_group == "storage":
        return list(STORAGE_COMMERCIAL_ENGINEER_CHECKS)
    return list(COMMERCIAL_ENGINEER_CHECKS)


def _sanitized_commercial_engineer_checks(
    product_group: str,
    values: Iterable[Any],
) -> list[str]:
    if product_group not in {"network", "storage"}:
        return _safe_string_list(values) or _commercial_engineer_checks(product_group)
    return sanitize_engineer_checks_for_product_group(
        _safe_string_list(values),
        product_group=product_group,
        defaults=_commercial_engineer_checks(product_group),
    )


def primary_commercial_telegram_lines(commercial: Mapping[str, Any]) -> list[str]:
    lines = [
        f"AI-подбор по складу №{commercial.get('match_run_id', '?')}",
        "",
    ]
    copy_lines = _safe_summary_lines(commercial.get("lines"))
    if not copy_lines:
        product_group = str(commercial.get("product_group") or "server").strip()
        copy_lines = _primary_copy_lines(
            server_quantity=_int_value(commercial.get("server_quantity")),
            per_server_lines=_safe_summary_lines(commercial.get("per_server_lines")),
            total_order_lines=_safe_summary_lines(commercial.get("total_order_lines")),
            price_line=_price_line(commercial),
            comment_lines=_safe_summary_lines(commercial.get("comment_lines"))
            or _commercial_comment_lines(product_group),
            engineer_checks=_sanitized_commercial_engineer_checks(
                product_group,
                commercial.get("engineer_checks", []),
            ),
            product_group=product_group,
        )
    lines.extend(copy_lines)
    lines.extend(["", "Подробный отчет отправлен Excel-файлом."])
    return lines


def primary_commercial_excel_rows(commercial: Mapping[str, Any]) -> list[tuple[str, str]]:
    server_quantity = _int_value(commercial.get("server_quantity"))
    product_group = str(commercial.get("product_group") or "server").strip()
    is_network = product_group == "network"
    is_storage = product_group == "storage"
    equipment_label = (
        "Сетевое оборудование"
        if is_network
        else ("СХД" if is_storage else "Сервер в сборе")
    )
    composition_label = "Состав" if is_network or is_storage else "Состав 1 сервера"
    rows = [
        ("Заголовок", str(commercial.get("title") or PRIMARY_COMMERCIAL_TITLE)),
        (equipment_label, _piece_quantity(server_quantity)),
    ]
    price_line = str(commercial.get("price_line") or "").strip()
    if price_line:
        rows.append(("Итоговая цена", price_line))
    else:
        price = _option_price_text(commercial)
        if price:
            rows.append(("Итоговая цена", price))
    per_server_lines = _safe_string_list(commercial.get("per_server_lines"))
    total_order_lines = _safe_string_list(commercial.get("total_order_lines"))
    if per_server_lines:
        rows.append((composition_label, "\n".join(per_server_lines)))
    if total_order_lines:
        rows.append(("Всего к заказу", "\n".join(total_order_lines)))
    comment_lines = _safe_string_list(commercial.get("comment_lines")) or list(
        _commercial_comment_lines(product_group)
    )
    engineer_checks = _sanitized_commercial_engineer_checks(
        product_group,
        commercial.get("engineer_checks", []),
    )
    rows.extend(
        [
            (
                "Комментарий",
                "\n".join(comment_lines),
            ),
            (
                "Проверить перед КП",
                "\n".join(f"- {check}" for check in engineer_checks),
            ),
        ]
    )
    return rows


def grouped_commercial_telegram_lines(commercial: CommercialSummary) -> list[str]:
    lines = [
        f"AI-подбор по складу №{commercial.match_run_id}",
        f"Цель: {commercial.goal}.",
        "Основа: склад OCS. Перед КП нужна инженерная проверка.",
        "",
        PRIMARY_COMMERCIAL_TITLE,
        "",
        f"Сервер в сборе - {_piece_quantity(commercial.server_quantity)}",
    ]
    lines.extend(_component_offer_lines(commercial.main_group, commercial.main_option))
    price = _option_price_text(commercial.main_option)
    if price:
        lines.extend(
            [
                "",
                (
                    f"Ориентировочно за {_server_quantity_text(commercial.server_quantity)}: "
                    f"{price}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "Комментарий:",
            f"- {commercial_reason_for_option(commercial.main_option)};",
            "- компоненты закрывают требования по CPU, RAM и NVMe;",
            "- инженерная проверка обязательна.",
        ]
    )

    if commercial.alternative_option is not None:
        lines.extend(
            [
                "",
                "Альтернатива спокойнее для инженеров:",
                _alternative_line(commercial.alternative_option, commercial.server_quantity),
            ]
        )

    more_line = _more_variants_line(commercial)
    if more_line:
        lines.extend(["", more_line])

    lines.extend(
        [
            "",
            "Проверить перед КП:",
            *[f"- {check}" for check in COMMERCIAL_ENGINEER_CHECKS],
            "",
            "Подробный отчет отправлен Excel-файлом.",
        ]
    )
    return lines


def grouped_commercial_excel_rows(
    commercial: CommercialSummary,
) -> list[tuple[str, str]]:
    rows = [
        ("Цель", commercial.goal),
        ("Основа", "склад OCS; перед КП нужна инженерная проверка"),
        ("Сервер в сборе", _piece_quantity(commercial.server_quantity)),
    ]
    for line in _component_offer_lines(commercial.main_group, commercial.main_option):
        label, value = line.split(":", 1)
        rows.append((label, value.strip()))
    price = _option_price_text(commercial.main_option)
    if price:
        rows.append(("Цена за весь запрос", price))
    if commercial.alternative_option is not None:
        rows.append(
            (
                "Альтернатива спокойнее для инженеров",
                _alternative_line(
                    commercial.alternative_option,
                    commercial.server_quantity,
                    trailing_period=False,
                ),
            )
        )
    rows.extend(
        [
            (
                "Комментарий",
                "\n".join(
                    [
                        f"- {commercial_reason_for_option(commercial.main_option)};",
                        "- компоненты закрывают требования по CPU, RAM и NVMe;",
                        "- инженерная проверка обязательна.",
                    ]
                ),
            ),
            (
                "Проверить перед КП",
                "\n".join(f"- {check}" for check in COMMERCIAL_ENGINEER_CHECKS),
            ),
        ]
    )
    return rows


def _primary_copy_lines(
    *,
    server_quantity: int | None,
    per_server_lines: list[str],
    total_order_lines: list[str],
    price_line: str,
    comment_lines: list[str],
    engineer_checks: list[str] | None = None,
    product_group: str = "server",
) -> list[str]:
    is_network = product_group == "network"
    is_storage = product_group == "storage"
    equipment_line = (
        f"Сетевое оборудование - {_piece_quantity(server_quantity)}"
        if is_network
        else (
            f"СХД - {_piece_quantity(server_quantity)}"
            if is_storage
            else f"Сервер в сборе - {_piece_quantity(server_quantity)}"
        )
    )
    composition_title = "Состав:" if is_network or is_storage else "Состав 1 сервера:"
    lines = [
        PRIMARY_COMMERCIAL_TITLE,
        "",
        equipment_line,
    ]
    if price_line:
        lines.append(price_line)
    lines.extend(
        [
            "",
            composition_title,
            *per_server_lines,
            "",
            "Всего к заказу:",
            *total_order_lines,
            "",
            "Комментарий:",
            *comment_lines,
            "",
            "Проверить перед КП:",
            *[f"- {check}" for check in (engineer_checks or COMMERCIAL_ENGINEER_CHECKS)],
        ]
    )
    return [_safe_summary_text(line) for line in lines]


def _normalize_legacy_primary_lines(lines: list[str]) -> list[str]:
    normalized: list[str] = []
    skip_why = False
    for line in lines:
        text = _safe_summary_text(line)
        if not text:
            normalized.append("")
            continue
        if text == LEGACY_PRIMARY_COMMERCIAL_TITLE:
            normalized.append(PRIMARY_COMMERCIAL_TITLE)
            continue
        if text.startswith("Почему этот вариант"):
            skip_why = True
            if normalized and normalized[-1] != "":
                normalized.append("")
            normalized.extend(["Комментарий:", *PRIMARY_COMMERCIAL_COMMENT_LINES])
            continue
        if text.startswith("Что проверить перед КП"):
            normalized.append("Проверить перед КП:")
            continue
        if skip_why and text.startswith("- "):
            continue
        skip_why = False
        normalized.append(text)
    return normalized


def _line_by_prefix(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line
    return ""


def _price_line(commercial: Mapping[str, Any]) -> str:
    server_quantity = _int_value(commercial.get("server_quantity"))
    price = _option_price_text(commercial)
    if not price:
        return ""
    return f"Ориентировочно за {_server_quantity_text(server_quantity)}: {price}"


def _safe_summary_text(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if contains_cjk_text(text):
        text = sanitize_user_facing_text(text)
    text = text.replace(LEGACY_PRIMARY_COMMERCIAL_TITLE, PRIMARY_COMMERCIAL_TITLE)
    text = re.sub(
        r"\bcomponent_candidate_id\s*[:=]\s*[\w./:-]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\bllm_rec_[\w-]+\b\s*:?.*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:why_this_platform|quote_recommendation)\s*[:=]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    forbidden = (
        "component_candidate_id",
        "facts",
        "evidence",
        "raw JSON",
        "llm_rec",
        "why_this_platform:",
        "quote_recommendation:",
        "Engineering confidence",
        "Tradeoff",
        "Minimal cost",
        "Proven",
        "Premium platform",
        "preliminary_requires_engineer_review",
    )
    for marker in forbidden:
        text = re.sub(re.escape(marker), "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _safe_string_list(value: Any) -> list[str]:
    return [line for line in _safe_summary_lines(value) if line]


def _safe_summary_lines(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, str):
        return []
    return [
        _safe_summary_text(item) if str(item or "").strip() else ""
        for item in value
    ]


def _primary_component_offer_lines(commercial: Mapping[str, Any]) -> list[str]:
    primary = commercial.get("primary_recommendation")
    primary_mapping = primary if isinstance(primary, Mapping) else {}
    components = _mapping_rows(commercial.get("components"))
    if not components:
        components = _mapping_rows(primary_mapping.get("components"))
    server_quantity = _int_value(commercial.get("server_quantity")) or _primary_server_quantity(
        primary_mapping
    )
    lines: list[str] = []
    platform = _component_by_role(components, "server_platform")
    if platform is None and isinstance(primary_mapping.get("platform"), Mapping):
        platform_candidate = primary_mapping["platform"]
        if str(platform_candidate.get("role") or "").strip() == "server_platform":
            platform = platform_candidate  # type: ignore[assignment]
    if isinstance(platform, Mapping):
        platform_quantity = _int_value(platform.get("quantity_required")) or server_quantity
        lines.append(
            "Платформа: "
            f"{commercial_component_name(platform)} - {_piece_quantity(platform_quantity)}, "
            f"склад {_stock_text(platform.get('available_quantity'))}"
        )
    storage_system = _component_by_role(components, "storage_system")
    if isinstance(storage_system, Mapping):
        lines.append(_component_quantity_line("СХД", storage_system, server_quantity))
    cpu = _component_by_role(components, "cpu")
    if isinstance(cpu, Mapping):
        lines.append(_component_quantity_line("CPU", cpu, server_quantity))
    ram = _component_by_role(components, "ram")
    if isinstance(ram, Mapping):
        lines.append(_ram_quantity_line(ram, server_quantity))
    storage = (
        _component_by_role(components, "drive")
        or _component_by_role(components, "ssd")
        or _component_by_role(components, "hdd")
    )
    if isinstance(storage, Mapping):
        label = "SSD" if storage.get("role") == "ssd" else "HDD"
        lines.append(_component_quantity_line(label, storage, server_quantity))
    return lines


def _primary_bom_rows(
    *,
    components: list[Mapping[str, Any]],
    server_quantity: int | None,
    primary_recommendation: Mapping[str, Any],
) -> list[dict[str, str]]:
    primary_mapping = primary_recommendation if isinstance(primary_recommendation, Mapping) else {}
    source_components = components
    if not source_components:
        source_components = _mapping_rows(primary_mapping.get("components"))
    rows: list[dict[str, str]] = []
    platform = _component_by_role(source_components, "server_platform")
    if platform is None and isinstance(primary_mapping.get("platform"), Mapping):
        platform_candidate = primary_mapping["platform"]
        if str(platform_candidate.get("role") or "").strip() == "server_platform":
            platform = platform_candidate  # type: ignore[assignment]
    if isinstance(platform, Mapping):
        rows.append(
            _bom_row(
                "Платформа",
                platform,
                server_quantity,
                primary_recommendation=primary_mapping,
            )
        )
    storage_system = _component_by_role(source_components, "storage_system")
    if isinstance(storage_system, Mapping):
        rows.append(
            _bom_row(
                "СХД",
                storage_system,
                server_quantity,
                primary_recommendation=primary_mapping,
            )
        )
    cpu = _component_by_role(source_components, "cpu")
    if isinstance(cpu, Mapping):
        rows.append(
            _bom_row("CPU", cpu, server_quantity, primary_recommendation=primary_mapping)
        )
    ram = _component_by_role(source_components, "ram")
    if isinstance(ram, Mapping):
        rows.append(
            _bom_row("RAM", ram, server_quantity, primary_recommendation=primary_mapping)
        )
    storage = (
        _component_by_role(source_components, "drive")
        or _component_by_role(source_components, "ssd")
        or _component_by_role(
            source_components,
            "hdd",
        )
    )
    if isinstance(storage, Mapping):
        label = "Диски" if storage.get("role") == "drive" else (
            "SSD" if storage.get("role") == "ssd" else "HDD"
        )
        rows.append(
            _bom_row(label, storage, server_quantity, primary_recommendation=primary_mapping)
        )
    network = _component_by_role(source_components, "network_adapter")
    if isinstance(network, Mapping):
        rows.append(
            _bom_row("Сеть", network, server_quantity, primary_recommendation=primary_mapping)
        )
    already_added = {
        "server_platform",
        "storage_system",
        "cpu",
        "ram",
        "drive",
        "ssd",
        "hdd",
        "network_adapter",
    }
    for component in source_components:
        role = str(component.get("role") or "").strip()
        if not role or role in already_added:
            continue
        rows.append(
            _bom_row(
                _commercial_role_label(role),
                component,
                server_quantity,
                primary_recommendation=primary_mapping,
            )
        )
        already_added.add(role)
    return rows


def _bom_row(
    label: str,
    component: Mapping[str, Any],
    server_quantity: int | None,
    *,
    primary_recommendation: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    per_server = _per_server_quantity(component, server_quantity)
    total = _int_value(component.get("quantity_required"))
    per_server_text = _piece_quantity(per_server)
    note = ""
    if label == "RAM":
        total_gb = _ram_total_gb_per_server(component, per_server)
        if total_gb is not None:
            per_server_text = f"{_piece_quantity(per_server)} = {_format_number(total_gb)} ГБ"
            note = f"{_format_number(total_gb)} ГБ на сервер"
    name = _name_with_capability_display(
        commercial_component_name(component),
        _component_capability_display(component, primary_recommendation),
    )
    return {
        "label": label,
        "name": _safe_summary_text(name),
        "per_server": per_server_text,
        "total": _piece_quantity(total),
        "stock": _stock_text(component.get("available_quantity")),
        "note": note,
    }


def _component_capability_display(
    component: Mapping[str, Any],
    primary_recommendation: Mapping[str, Any] | None,
) -> str:
    role = str(component.get("role") or "").strip()
    if role == "network_adapter":
        return _network_adapter_capability_display(component, primary_recommendation)
    if role in {"switch", "router", "firewall", "access_point"}:
        return _network_device_capability_display(component)
    if role in {"transceiver", "dac_cable", "cable"}:
        return _network_link_capability_display(component)
    return ""


def _network_adapter_capability_display(
    component: Mapping[str, Any],
    primary_recommendation: Mapping[str, Any] | None,
) -> str:
    ports = _int_value(
        _component_fact_value(component, "network_ports_count", "ports_count")
    )
    speed = _network_speed_display(
        _component_fact_value(
            component,
            "network_speed",
            "speed",
            "network_speed_gbps",
            "speed_gbps",
        )
    )
    media = _network_media_display(
        _component_fact_value(component, "network_media", "media")
    )
    if ports is None or not speed or not media:
        requirement = _primary_role_requirement_for_display(
            primary_recommendation,
            "network_adapter",
        )
        ports = ports or _int_value(
            _component_fact_value(
                requirement,
                "min_ports_per_server",
                "ports_per_server",
                "ports_count",
            )
        )
        speed = speed or _network_speed_display(
            _component_fact_value(requirement, "speed", "network_speed")
        )
        media = media or _network_media_display(
            _component_fact_value(requirement, "media", "network_media")
        )
    if ports is None or ports <= 0 or not speed or not media:
        return ""
    return f"{ports} x {speed} {media}"


def _network_device_capability_display(component: Mapping[str, Any]) -> str:
    parts: list[str] = []
    ports = _int_value(_component_fact_value(component, "port_count"))
    port_speed = _network_speed_display(
        _component_fact_value(component, "port_speed", "port_speed_gbps")
    )
    port_media = _network_media_display(_component_fact_value(component, "port_media"))
    if ports is not None and port_speed:
        port_text = f"{ports} x {port_speed}"
        if port_media:
            port_text = f"{port_text} {port_media}"
        parts.append(port_text)
    uplinks = _int_value(_component_fact_value(component, "uplink_count"))
    uplink_speed = _network_speed_display(
        _component_fact_value(component, "uplink_speed", "uplink_speed_gbps")
    )
    uplink_media = _network_media_display(_component_fact_value(component, "uplink_media"))
    if uplinks is not None and uplink_speed:
        uplink_text = f"uplink {uplinks} x {uplink_speed}"
        if uplink_media:
            uplink_text = f"{uplink_text} {uplink_media}"
        parts.append(uplink_text)
    poe_standard = _component_fact_value(component, "poe_standard")
    poe_budget = _int_value(_component_fact_value(component, "poe_budget_w"))
    if _truthy(_component_fact_value(component, "poe_supported")):
        poe_text = str(poe_standard or "PoE").strip()
        if poe_budget is not None:
            poe_text = f"{poe_text} {poe_budget}W"
        parts.append(poe_text)
    if _truthy(_component_fact_value(component, "l3_supported")):
        parts.append("L3")
    if _truthy(_component_fact_value(component, "stacking_supported")):
        parts.append("stacking")
    return ", ".join(parts)


def _network_link_capability_display(component: Mapping[str, Any]) -> str:
    speed = _network_speed_display(
        _component_fact_value(
            component,
            "port_speed",
            "uplink_speed",
            "port_speed_gbps",
            "uplink_speed_gbps",
        )
    )
    media = _network_media_display(
        _component_fact_value(
            component,
            "transceiver_form_factor",
            "port_media",
            "uplink_media",
        )
    )
    return " ".join(part for part in (speed, media) if part)


def _primary_role_requirement_for_display(
    primary_recommendation: Mapping[str, Any] | None,
    role: str,
) -> Mapping[str, Any]:
    primary = primary_recommendation if isinstance(primary_recommendation, Mapping) else {}
    for key in ("network_requirement", "network_adapter_requirement"):
        requirement = primary.get(key)
        if isinstance(requirement, Mapping):
            return requirement
    role_plan = primary.get("role_plan")
    if isinstance(role_plan, Mapping):
        requirements_by_role = role_plan.get("requirements_by_role")
        if isinstance(requirements_by_role, Mapping):
            requirement = requirements_by_role.get(role)
            if isinstance(requirement, Mapping):
                return requirement
    for key in ("required_capabilities", "hard_capability_validation"):
        for capability in _mapping_rows(primary.get(key)):
            if str(capability.get("role") or "").strip() != role:
                continue
            parsed = capability.get("parsed_requirements")
            if isinstance(parsed, Mapping):
                return parsed
    return {}


def _name_with_capability_display(name: str, display: str) -> str:
    name = " ".join(str(name or "").split())
    display = " ".join(str(display or "").split())
    if not display:
        return name
    if display.casefold() in name.casefold():
        return name
    if not name:
        return display
    return f"{name}, {display}"


def _component_fact_value(component: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = component.get(key)
        if _known_fact_value(value):
            return value
    for container_key in ("facts", "extracted_facts", "parsed_requirements"):
        container = component.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            value = container.get(key)
            if _known_fact_value(value):
                return value
    return None


def _known_fact_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    return str(value).strip().casefold() != "unknown"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "y",
        "да",
        "poe",
        "poe+",
        "poe++",
    }


def _network_speed_display(value: Any) -> str:
    if not _known_fact_value(value):
        return ""
    if isinstance(value, int | float):
        gbps = int(value)
        return f"{gbps}GbE" if gbps > 0 else ""
    text = str(value).strip()
    match = re.search(r"\b(1|10|16|25|32|40|56|64|100|200|400)(?=\D|$)", text, re.I)
    if match:
        return f"{int(match.group(1))}GbE"
    return text


def _network_media_display(value: Any) -> str:
    if not _known_fact_value(value):
        return ""
    return str(value).strip().upper().replace(" ", "")


def _per_server_lines_from_bom_rows(
    rows: list[Mapping[str, str]],
    *,
    product_group: str = "server",
) -> list[str]:
    lines: list[str] = []
    is_network = product_group == "network"
    is_storage = product_group == "storage"
    for row in rows:
        label = str(row.get("label") or "").strip()
        name = str(row.get("name") or "").strip()
        per_server = str(row.get("per_server") or "").strip()
        total = str(row.get("total") or "").strip()
        stock = str(row.get("stock") or "").strip()
        if label and name and per_server:
            if is_network or is_storage:
                quantity_text = total or per_server
                lines.append(f"- {label}: {name} - {quantity_text}")
                continue
            if label == "Сеть" and total:
                lines.append(
                    f"- {label}: {name} - {per_server} на сервер / "
                    f"{total} всего, склад {stock or 'неизвестно'}"
                )
                continue
            lines.append(f"- {label}: {name} - {per_server}")
    return lines


def _total_order_lines_from_bom_rows(rows: list[Mapping[str, str]]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        label = str(row.get("label") or "").strip()
        total = str(row.get("total") or "").strip()
        stock = str(row.get("stock") or "").strip()
        if label and total:
            lines.append(f"- {label}: {total}, склад {stock or 'неизвестно'}")
    return lines


def _primary_product_group(
    summary: Mapping[str, Any],
    primary_recommendation: Mapping[str, Any],
    existing_summary: Mapping[str, Any],
) -> str:
    candidates: list[Any] = [
        primary_recommendation.get("product_group"),
        existing_summary.get("product_group"),
        summary.get("product_group"),
    ]
    requirements = _first_requirements(summary)
    candidates.append(requirements.get("product_group"))
    role_plan = requirements.get("role_plan")
    if isinstance(role_plan, Mapping):
        candidates.append(role_plan.get("product_group"))
    matrix = summary.get("component_candidate_matrix")
    if isinstance(matrix, Mapping):
        candidates.append(matrix.get("product_group"))
    for candidate in candidates:
        product_group = str(candidate or "").strip()
        if product_group:
            return product_group
    return "server"


def _primary_server_quantity(primary_recommendation: Mapping[str, Any]) -> int | None:
    components = _mapping_rows(primary_recommendation.get("components"))
    platform = _component_by_role(components, "server_platform")
    if platform is not None:
        quantity = _int_value(platform.get("quantity_required"))
        if quantity is not None:
            return quantity
    storage_system = _component_by_role(components, "storage_system")
    if storage_system is not None:
        quantity = _int_value(storage_system.get("quantity_required"))
        if quantity is not None:
            return quantity
    for component in components:
        quantity = _int_value(component.get("server_quantity"))
        if quantity is not None:
            return quantity
    summary = primary_recommendation.get("commercial_summary")
    if isinstance(summary, Mapping):
        return _int_value(summary.get("server_quantity"))
    return None


def _component_by_role(
    components: list[Mapping[str, Any]],
    role: str,
) -> Mapping[str, Any] | None:
    for component in components:
        if str(component.get("role") or "").strip() == role:
            return component
    return None


def _commercial_role_label(role: str) -> str:
    labels = {
        "switch": "Коммутатор",
        "router": "Маршрутизатор",
        "firewall": "Межсетевой экран",
        "access_point": "Точка доступа",
        "storage_system": "СХД",
        "controller": "Контроллер СХД",
        "controller_module": "Модуль контроллера",
        "disk_shelf": "Дисковая полка",
        "drive": "Диски",
        "cache": "Кэш",
        "host_port": "Host-порт",
        "protocol_module": "Протокольный модуль",
        "storage_controller": "Контроллер",
        "gpu": "GPU",
        "transceiver": "Трансивер",
        "dac_cable": "DAC-кабель",
        "cable": "Кабель",
        "power_supply": "БП",
        "rail_kit": "Рельсы",
        "license": "Лицензия",
        "support": "Поддержка",
        "stacking_module": "Модуль стекирования",
        "other_accessory": "Аксессуар",
    }
    return labels.get(role, role)


def commercial_component_name(component: Mapping[str, Any]) -> str:
    role = str(component.get("role") or "").strip()
    if role in {"server_platform", "platform"}:
        return _platform_name(component)
    if role == "cpu":
        return _cpu_name(component)
    if role == "ram":
        return _ram_name(component)
    if role in {"drive", "ssd", "hdd", "storage"}:
        return _storage_name(component)
    if role == "storage_system":
        return _producer_part_name(component)
    if role == "network_adapter":
        return _network_adapter_name(component)
    return _producer_part_name(component)


def commercial_reason_for_option(option: Mapping[str, Any]) -> str:
    role = str(option.get("role") or option.get("option_role") or "").strip()
    if role in REASON_TEMPLATES:
        return REASON_TEMPLATES[role]
    reason = _safe_russian_sentence(option.get("why_this_platform"))
    return reason or "складской вариант для сравнения"


def commercial_safe_russian_text(value: Any) -> str:
    return _safe_russian_sentence(value)


def commercial_engineering_confidence(value: Any) -> str:
    return human_engineering_confidence_label(value)


def _main_commercial_option(
    configuration_groups: list[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    fallback: tuple[Mapping[str, Any] | None, Mapping[str, Any] | None] = (None, None)
    fallback_price: Decimal | None = None
    for group in configuration_groups:
        for option in _mapping_rows(group.get("platform_options")):
            role = str(option.get("role") or option.get("option_role") or "").strip()
            if role == "cheapest_quote":
                return group, option
            price = _money_decimal(option.get("total_price_value"))
            if fallback[1] is None or (
                price is not None and (fallback_price is None or price < fallback_price)
            ):
                fallback = (group, option)
                fallback_price = price
    return fallback


def _alternative_commercial_option(
    configuration_groups: list[Mapping[str, Any]],
    main_option: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    options = [
        option
        for group in configuration_groups
        for option in _mapping_rows(group.get("platform_options"))
        if option is not main_option
    ]
    for role in ("preferred_for_database", "engineering_clear", "branded_safe"):
        for option in options:
            option_role = str(option.get("role") or option.get("option_role") or "").strip()
            if option_role == role:
                return option
    return options[0] if options else None


def _component_offer_lines(
    group: Mapping[str, Any],
    option: Mapping[str, Any],
) -> list[str]:
    base = group.get("component_base")
    component_base = base if isinstance(base, Mapping) else {}
    server_quantity = _group_server_quantity(group)
    lines: list[str] = []
    platform = option.get("platform")
    if isinstance(platform, Mapping):
        platform_quantity = _int_value(platform.get("quantity_required")) or server_quantity
        lines.append(
            "Платформа: "
            f"{commercial_component_name(platform)} - {_piece_quantity(platform_quantity)}, "
            f"склад {_stock_text(platform.get('available_quantity'))}"
        )
    cpu = component_base.get("cpu")
    if isinstance(cpu, Mapping):
        lines.append(_component_quantity_line("CPU", cpu, server_quantity))
    ram = component_base.get("ram")
    if isinstance(ram, Mapping):
        lines.append(_ram_quantity_line(ram, server_quantity))
    storage = component_base.get("storage")
    if isinstance(storage, Mapping):
        label = "SSD" if storage.get("role") == "ssd" else "HDD"
        lines.append(_component_quantity_line(label, storage, server_quantity))
    return lines


def _component_quantity_line(
    label: str,
    component: Mapping[str, Any],
    server_quantity: int | None,
) -> str:
    per_server = _per_server_quantity(component, server_quantity)
    total = _int_value(component.get("quantity_required"))
    return (
        f"{label}: {commercial_component_name(component)} - "
        f"{_piece_quantity(per_server)} на сервер / {_piece_quantity(total)} всего, "
        f"склад {_stock_text(component.get('available_quantity'))}"
    )


def _ram_quantity_line(
    component: Mapping[str, Any],
    server_quantity: int | None,
) -> str:
    per_server = _per_server_quantity(component, server_quantity)
    total = _int_value(component.get("quantity_required"))
    total_gb = _ram_total_gb_per_server(component, per_server)
    total_gb_text = f" = {_format_number(total_gb)} ГБ" if total_gb is not None else ""
    return (
        f"RAM: {commercial_component_name(component)} - "
        f"{_piece_quantity(per_server)} на сервер{total_gb_text} / "
        f"{_piece_quantity(total)} всего, склад {_stock_text(component.get('available_quantity'))}"
    )


def _goal_text(
    summary: Mapping[str, Any],
    group: Mapping[str, Any],
    option: Mapping[str, Any],
) -> str:
    quantity = _group_server_quantity(group)
    quantity_text = _server_quantity_text(quantity)
    form_factor = _form_factor_text(summary, option)
    workload = _workload_text(summary)
    parts = [quantity_text]
    if form_factor:
        parts.append(form_factor)
    parts.append(workload)
    return " ".join(part for part in parts if part)


def _form_factor_text(summary: Mapping[str, Any], option: Mapping[str, Any]) -> str:
    requirements = _first_requirements(summary)
    for key in ("form_factor", "server_form_factor", "chassis_form_factor"):
        value = str(requirements.get(key) or "").strip()
        if value and value != "unknown":
            return value.upper()
    platform = option.get("platform")
    haystack = ""
    if isinstance(platform, Mapping):
        haystack = " ".join(
            str(platform.get(key) or "")
            for key in ("item_name", "name", "display_name", "part_number")
        )
        facts = platform.get("facts")
        if isinstance(facts, Mapping):
            for key in ("form_factor", "server_form_factor", "chassis_form_factor"):
                value = str(facts.get(key) or "").strip()
                if value and value != "unknown":
                    return value.upper()
    match = re.search(r"\b[1248]U\b", haystack, flags=re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return "2U"


def _workload_text(summary: Mapping[str, Any]) -> str:
    source = str(summary.get("source_text") or summary.get("user_request") or "")
    requirements = _first_requirements(summary)
    source += " " + " ".join(str(value) for value in requirements.values())
    lowered = source.casefold()
    if any(marker in lowered for marker in ("бд", "база данных", "database", "db")):
        return "под базу данных"
    return "под базу данных"


def _first_requirements(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = summary.get("normalized_requirements")
    if isinstance(normalized, Mapping):
        return normalized
    if isinstance(normalized, Iterable) and not isinstance(normalized, str):
        for row in normalized:
            if isinstance(row, Mapping):
                return row
    matrix = summary.get("component_candidate_matrix")
    if isinstance(matrix, Mapping):
        matrix_requirements = matrix.get("normalized_requirements")
        if isinstance(matrix_requirements, Mapping):
            return matrix_requirements
    return {}


def _alternative_line(
    option: Mapping[str, Any],
    server_quantity: int | None,
    *,
    trailing_period: bool = True,
) -> str:
    platform = option.get("platform")
    platform_text = (
        commercial_component_name(platform)
        if isinstance(platform, Mapping)
        else "платформа требует уточнения"
    )
    price = _option_price_text(option)
    scope = f"за {_server_quantity_text(server_quantity)}" if server_quantity else "за весь запрос"
    text = " - ".join(part for part in (platform_text, price) if part)
    if scope:
        text = f"{text} {scope}"
    return text + "." if trailing_period else text


def _more_variants_line(commercial: CommercialSummary) -> str:
    extra_options = max(0, commercial.platform_options_count - commercial.shown_options_count)
    extra_groups = max(0, commercial.group_count - 1)
    extra = extra_options + extra_groups
    if extra <= 0:
        return ""
    noun = pluralize_ru(extra, "вариант/семейство", "варианта/семейства", "вариантов/семейств")
    return f"Еще {extra} {noun} в Excel."


def _option_price_text(option: Mapping[str, Any]) -> str:
    amount = _money_decimal(option.get("total_price_value"))
    currency = str(option.get("total_price_currency") or "").strip()
    if amount is None:
        return format_price(option.get("total_price_value"), currency)
    if amount == amount.to_integral():
        formatted = f"{int(amount):,}".replace(",", " ")
    else:
        formatted = f"{amount:,.2f}".replace(",", " ")
    return f"{formatted} {currency}".strip()


def _group_server_quantity(group: Mapping[str, Any]) -> int | None:
    for option in _mapping_rows(group.get("platform_options")):
        platform = option.get("platform")
        if isinstance(platform, Mapping):
            quantity = _int_value(platform.get("quantity_required"))
            if quantity is not None:
                return quantity
    base = group.get("component_base")
    if isinstance(base, Mapping):
        for component in base.values():
            if isinstance(component, Mapping):
                quantity = _int_value(component.get("server_quantity"))
                if quantity is not None:
                    return quantity
    return None


def _per_server_quantity(
    component: Mapping[str, Any],
    server_quantity: int | None,
) -> int | None:
    per_server = _int_value(component.get("per_server_quantity"))
    if per_server is not None:
        return per_server
    quantity = _int_value(component.get("quantity_required"))
    if quantity is None:
        return None
    if server_quantity and server_quantity > 0:
        return max(1, quantity // server_quantity)
    return quantity


def _cpu_name(component: Mapping[str, Any]) -> str:
    part_number = _part_number(component)
    candidates = _text_candidates(component)
    for candidate in candidates:
        match = _CPU_MODEL_RE.search(candidate)
        if match:
            model = " ".join(match.group(0).split())
            if not model.casefold().startswith("intel "):
                model = f"Intel {model}"
            return _with_part_number(model, part_number)
    return _producer_part_name(component).replace("Intel Corporation", "Intel")


def _ram_name(component: Mapping[str, Any]) -> str:
    part_number = _part_number(component)
    producer = _producer(component)
    capacity = _ram_capacity_gb(component)
    memory_type = _first_match(_DDR_RE, _text_candidates(component))
    form_factor = _ram_form_factor(component)
    if capacity is None and not memory_type and not form_factor:
        return _producer_part_name(component)
    parts = [producer]
    if capacity:
        parts.append(f"{_format_number(capacity)}GB")
    if memory_type:
        parts.append(memory_type.upper())
    if form_factor:
        parts.append(form_factor)
    display = " ".join(part for part in parts if part)
    if not display:
        display = _producer_part_name(component)
    return _with_part_number(display, part_number)


def _storage_name(component: Mapping[str, Any]) -> str:
    part_number = _part_number(component)
    producer = _producer(component)
    candidates = _text_candidates(component)
    series = _first_match(_SSD_SERIES_RE, candidates)
    capacity = _storage_capacity_tb(component)
    interface = _storage_interface(component)
    if not series and capacity is None and not interface:
        return _producer_part_name(component)
    parts = [producer, series]
    if capacity is not None:
        parts.append(f"{_format_number(capacity)}TB")
    if interface:
        parts.append(interface)
    display = " ".join(part for part in parts if part)
    if not display:
        display = _producer_part_name(component)
    return _with_part_number(display, part_number)


def _network_adapter_name(component: Mapping[str, Any]) -> str:
    part_number = _part_number(component)
    if part_number:
        for candidate in _text_candidates(component):
            match = re.search(re.escape(part_number), candidate, re.IGNORECASE)
            if match is None:
                continue
            display = candidate[: match.end()].strip(" -_/():,;")
            if display and display.casefold() != part_number.casefold():
                return display
    return _producer_part_name(component)


def _platform_name(component: Mapping[str, Any]) -> str:
    part_number = _part_number(component)
    for candidate in _text_candidates(component):
        model = _gooxi_model(candidate)
        if model:
            return _with_part_number(f"Gooxi {model}", part_number)
    return _producer_part_name(component)


def _gooxi_model(value: str) -> str:
    match = re.search(
        r"\bGooxi\b[\s:/()_-]+([A-Z0-9][A-Z0-9]+(?:-[A-Z0-9]+)+)\b",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    model = match.group(1).strip(" -_/():")
    if re.fullmatch(r"\d+(?:\.\d+)+", model):
        return ""
    return model.upper()


def _producer_part_name(component: Mapping[str, Any]) -> str:
    producer = _producer(component)
    part_number = _part_number(component)
    display = " ".join(part for part in (producer, part_number) if part)
    if display:
        return display
    return str(
        component.get("item_name")
        or component.get("name")
        or component.get("display_name")
        or component.get("item_id")
        or ""
    ).strip()


def _producer(component: Mapping[str, Any]) -> str:
    producer = str(component.get("producer") or component.get("vendor") or "").strip()
    return producer.replace("Intel Corporation", "Intel")


def _part_number(component: Mapping[str, Any]) -> str:
    return str(component.get("part_number") or "").strip()


def _with_part_number(display: str, part_number: str) -> str:
    display = " ".join(str(display or "").split())
    part_number = part_number.strip()
    if not part_number or not display:
        return display or part_number
    if part_number.casefold() in display.casefold():
        return display
    return f"{display} ({part_number})"


def _text_candidates(component: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in (
        "item_name",
        "name",
        "display_name",
        "model",
        "product_name",
        "marketing_name",
        "normalized_name",
        "part_number",
    ):
        value = str(component.get(key) or "").strip()
        if value:
            candidates.append(value)
    facts = component.get("facts")
    if isinstance(facts, Mapping):
        for value in facts.values():
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    return candidates


def _ram_capacity_gb(component: Mapping[str, Any]) -> Decimal | None:
    value = (
        component.get("ram_module_capacity_gb")
        or _fact(component, "ram_capacity_gb")
        or _fact(component, "capacity_gb")
    )
    if value not in (None, ""):
        return _money_decimal(value)
    for candidate in _text_candidates(component):
        match = _CAPACITY_GB_RE.search(candidate)
        if match:
            return _money_decimal(match.group(1).replace(",", "."))
    return None


def _ram_form_factor(component: Mapping[str, Any]) -> str:
    for key in ("ram_form_factor", "memory_form_factor", "form_factor"):
        value = str(_fact(component, key) or component.get(key) or "").strip()
        if value and value != "unknown":
            return _normalize_ram_form_factor(value)
    haystack = " ".join(_text_candidates(component)).casefold()
    if "rdimm" in haystack or "registered" in haystack:
        return "RDIMM"
    if "lrdimm" in haystack:
        return "LRDIMM"
    if "udimm" in haystack:
        return "UDIMM"
    return ""


def _normalize_ram_form_factor(value: str) -> str:
    lowered = value.casefold()
    if "lrdimm" in lowered:
        return "LRDIMM"
    if "rdimm" in lowered or "registered" in lowered:
        return "RDIMM"
    if "udimm" in lowered:
        return "UDIMM"
    return value.upper()


def _storage_capacity_tb(component: Mapping[str, Any]) -> Decimal | None:
    value = (
        component.get("drive_capacity_tb")
        or _fact(component, "drive_capacity_tb")
        or component.get("storage_capacity_tb")
        or _fact(component, "storage_capacity_tb")
    )
    if value not in (None, ""):
        return _money_decimal(value)
    for candidate in _text_candidates(component):
        match = _CAPACITY_TB_RE.search(candidate)
        if match:
            return _money_decimal(match.group(1).replace(",", "."))
    return None


def _storage_interface(component: Mapping[str, Any]) -> str:
    candidates = [
        str(_fact(component, "drive_interface") or component.get("drive_interface") or ""),
        str(_fact(component, "storage_interface") or component.get("storage_interface") or ""),
        *list(_text_candidates(component)),
    ]
    haystack = " ".join(candidates).casefold()
    parts: list[str] = []
    if "u.3" in haystack:
        parts.append("U.3")
    elif "u.2" in haystack:
        parts.append("U.2")
    if "nvme" in haystack:
        parts.append("NVMe")
    elif "sata" in haystack:
        parts.append("SATA")
    return " ".join(parts)


def _ram_total_gb_per_server(
    component: Mapping[str, Any],
    per_server_quantity: int | None,
) -> Decimal | None:
    explicit = component.get("ram_total_gb_per_server")
    if explicit not in (None, ""):
        return _money_decimal(explicit)
    capacity = _ram_capacity_gb(component)
    if capacity is None or per_server_quantity is None:
        return None
    return capacity * per_server_quantity


def _first_match(pattern: re.Pattern[str], values: list[str]) -> str:
    for value in values:
        match = pattern.search(value)
        if match:
            return match.group(0)
    return ""


def _fact(component: Mapping[str, Any], key: str) -> Any:
    facts = component.get("facts")
    if isinstance(facts, Mapping):
        return facts.get(key)
    return None


def _safe_russian_sentence(value: Any) -> str:
    text = sanitize_user_facing_text(value)
    if not text or contains_cjk_text(text):
        return ""
    if not re.search(r"[А-Яа-яЁё]", text):
        return ""
    if any(marker in text for marker in ("component_candidate_id", "raw JSON", "llm_rec")):
        return ""
    latin_words = {
        word.upper()
        for word in _WORD_RE.findall(text)
        if word.upper() not in _ALLOWED_TECH_WORDS
    }
    if latin_words:
        return ""
    return text


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, str):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _piece_quantity(value: int | None) -> str:
    if value is None:
        return "количество уточняется"
    return f"{value} шт."


def _server_quantity_text(value: int | None) -> str:
    if value is None:
        return "весь запрос"
    return f"{value} {pluralize_ru(value, 'сервер', 'сервера', 'серверов')}"


def _order_quantity_text(value: int | None, product_group: str) -> str:
    if product_group == "network":
        if value is None:
            return "весь запрос"
        return f"{value} шт. сетевого оборудования"
    if product_group == "storage":
        if value is None:
            return "весь запрос"
        return f"{value} шт. СХД"
    return _server_quantity_text(value)


def _stock_text(value: Any) -> str:
    if value in (None, ""):
        return "неизвестно"
    return str(value)


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _money_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value).replace(" ", ""))
    except (InvalidOperation, ValueError):
        return None


def _format_number(value: int | Decimal) -> str:
    amount = value if isinstance(value, Decimal) else Decimal(value)
    if amount == amount.to_integral():
        return str(int(amount))
    return format(amount.normalize(), "f")
