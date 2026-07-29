from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from app.reports.commercial_summary import (
    build_grouped_commercial_summary,
    build_primary_commercial_summary,
    grouped_commercial_telegram_lines,
    primary_commercial_telegram_lines,
)
from app.reports.match_text import (
    as_string_list,
    candidate_display_name,
    format_price,
    human_match_status_for_summary,
    humanized_checks,
    pluralize_ru,
    yes_no,
)
from app.reports.recommendation_titles import humanized_recommendation_title
from app.user_facing_text import (
    grouped_engineer_check_summary,
    sanitize_engineer_checks_for_product_group,
    sanitize_user_facing_text,
)

TELEGRAM_MESSAGE_LIMIT = 4096
TELEGRAM_SAFE_MESSAGE_LIMIT = 3900
TELEGRAM_SAFE_CAPTION_LIMIT = 1000
V3_TELEGRAM_FIELD_LIMIT = 320
V3_TELEGRAM_LIST_ITEM_LIMIT = 220
MAX_SUMMARY_CANDIDATES = 3
MAX_SUMMARY_CHECKS = 5
TELEGRAM_TEXT_FALLBACK = "Выбрано по сочетанию цены, наличия и соответствия требованиям."
LLM_TELEGRAM_NOTE = (
    "Важно: LLM может ошибаться, перед КП нужна проверка инженера."
)


@dataclass(frozen=True)
class ReportDelivery:
    mode: Literal["message", "file"]
    text: str | None = None
    filename: str | None = None
    content: bytes | None = None


def format_v3_full_category_quote(payload: Mapping[str, Any]) -> str:
    state = str(payload.get("result_state") or "unknown")
    profile = str(payload.get("profile") or "custom")
    category_ids = as_string_list(payload.get("category_ids"))
    distributor_code = _v3_user_text(payload.get("distributor_code"))
    diagnostics = _v3_mapping(payload.get("diagnostics"))
    row_count = diagnostics.get("matrix_row_count")
    quote = _v3_mapping(payload.get("validated_quote"))
    no_recommendation = _v3_mapping(payload.get("no_recommendation_reason"))
    validation_failure = _v3_mapping(payload.get("validation_failure_reason"))

    lines = [
        "КП draft",
        f"Статус: {_v3_state_label(state)}",
    ]
    if distributor_code:
        lines.append(f"Склад: {_v3_distributor_label(distributor_code)}")

    if quote:
        total = _v3_total_price_text(quote)
        if total:
            lines.extend(["", f"Итого: {total}"])

        selection = _v3_selection_summary(quote)
        if selection:
            lines.extend(["", f"Класс подбора: {selection}"])

        client_status = _v3_short_text(_v3_user_text(quote.get("client_status_label")))
        if client_status:
            lines.extend(["", f"Тип предложения: {client_status}"])

        client_summary = _v3_short_text(_v3_user_text(quote.get("client_summary")))
        if client_summary:
            lines.extend(["", client_summary])

        coverage_summary = _v3_short_text(_v3_user_text(quote.get("coverage_summary")))
        if coverage_summary:
            lines.extend(["", f"Покрытие ТЗ: {coverage_summary}"])

        target_decisions = _v3_target_decision_lines(quote, limit=3)
        if target_decisions:
            lines.extend(["", "Целевые объекты:", *[f"- {item}" for item in target_decisions]])

        priority_checks = _v3_priority_review_lines(quote)
        if priority_checks:
            lines.extend(
                ["", "Перед отправкой проверить:", *[f"- {item}" for item in priority_checks]]
            )

        deviations = _v3_quote_deviation_items(quote, limit=4)
        if deviations:
            lines.extend(["", "Отклонения от ТЗ:", *[f"- {item}" for item in deviations]])

        gaps = _v3_user_text_list(quote.get("procurement_gaps"), limit=5)
        if gaps:
            lines.extend(
                [
                    "",
                    "Что нужно добрать или согласовать:",
                    *[f"- {item}" for item in gaps],
                ]
            )

        alternatives = _v3_available_alternative_list(
            quote.get("available_alternatives"),
            limit=4,
        )
        if alternatives:
            lines.extend(
                [
                    "",
                    "Доступные варианты для согласования (не включены в итог):",
                    *[f"- {item}" for item in alternatives],
                ]
            )

        quote_lines = _v3_quote_lines(quote)
        if quote_lines:
            lines.extend(["", "Спецификация для КП:", *quote_lines])

        why_selected = _v3_short_text(_v3_user_text(quote.get("why_selected")))
        if why_selected:
            lines.extend(["", f"Комментарий: {why_selected}"])

        price_audit = _v3_user_text_list(quote.get("price_audit"), limit=3)
        if price_audit:
            lines.extend(["", "Проверка цены:", *[f"- {item}" for item in price_audit]])

        assumptions = _v3_user_text_list(quote.get("assumptions"), limit=3)
        if assumptions:
            lines.extend(["", "Допущения:", *[f"- {item}" for item in assumptions]])

        compatibility = _v3_compatibility_check_lines(quote.get("compatibility_check"))
        if compatibility:
            lines.extend(["", "Проверка совместимости:", *compatibility])

        checks = _v3_user_text_list(quote.get("engineer_checks"), limit=5)
        if checks and not priority_checks:
            lines.extend(["", "Перед отправкой проверить:", *[f"- {item}" for item in checks]])
        elif bool(payload.get("engineering_review_required")):
            lines.extend(
                [
                    "",
                    "Перед отправкой проверить:",
                    "- Финально сверить совместимость перед КП клиенту.",
                ]
            )

    elif validation_failure:
        summary = _v3_short_text(
            _v3_user_text(validation_failure.get("summary"))
            or _v3_validation_failure_summary(payload)
        )
        if summary:
            lines.extend(["", summary])

    elif no_recommendation:
        if state == "mechanical_validation_failed":
            lines.extend(
                [
                    "",
                    (
                        "LLM собрала вариант, но он не прошел проверку склада, "
                        "ID или цен. Я не показываю его как КП."
                    ),
                ]
            )
        else:
            summary = _v3_no_recommendation_summary(no_recommendation)
            if summary:
                lines.extend(["", summary])
            details = _v3_no_recommendation_details(no_recommendation)
            if details:
                lines.extend(["", f"Детали: {details}"])
            failed = _v3_no_recommendation_failed_items(no_recommendation, limit=4)
            if failed:
                lines.extend(["", "Что не закрыто:", *[f"- {item}" for item in failed]])
            actions = _v3_no_recommendation_next_actions(no_recommendation, limit=3)
            if actions:
                lines.extend(["", "Что сделать дальше:", *[f"- {item}" for item in actions]])

    validation_errors = as_string_list(
        payload.get("validation_errors") or payload.get("v3_validation_errors")
    )
    if validation_errors:
        validation_details = _v3_validation_error_details(payload)
        validation_error_lines = (
            _v3_human_mechanical_errors(validation_errors, validation_details)
            if state == "mechanical_validation_failed"
            else validation_errors[:4]
        )
        validation_header = (
            "Что не сошлось:"
            if state == "mechanical_validation_failed"
            else "Механическая проверка:"
        )
        lines.extend(
            [
                "",
                validation_header,
                *[f"- {item}" for item in validation_error_lines],
            ]
        )

    if state in {"quote_draft_review_required", "quote_candidate_customer_ready"}:
        lines.extend(["", _v3_final_note(quote)])

    service_parts = []
    if diagnostics.get("composer_prompt_version") != "composer_v7":
        service_parts = _v3_service_parts(payload, profile, category_ids, row_count)
    if service_parts:
        lines.extend(["", "Служебно: " + "; ".join(service_parts)])

    return "\n".join(lines)


def choose_v3_full_category_quote_delivery(
    payload: Mapping[str, Any],
    *,
    safe_limit: int = TELEGRAM_SAFE_MESSAGE_LIMIT,
) -> ReportDelivery:
    text = format_v3_full_category_quote(payload)
    return ReportDelivery(mode="message", text=_v3_fit_telegram_message(text, safe_limit))


def format_v3_excel_caption(
    payload: Mapping[str, Any],
    *,
    match_run_id: int,
    safe_limit: int = TELEGRAM_SAFE_CAPTION_LIMIT,
) -> str:
    state = str(payload.get("result_state") or payload.get("v3_result_state") or "")
    no_recommendation = _v3_mapping(payload.get("no_recommendation_reason"))
    quote = _v3_mapping(payload.get("validated_quote"))
    lines = [f"Подбор по складу №{match_run_id}"]

    if state == "no_recommendation" and no_recommendation:
        summary = _v3_no_recommendation_summary(no_recommendation)
        if summary:
            lines.append(summary)
        failed = _v3_no_recommendation_failed_items(no_recommendation, limit=1)
        if failed:
            lines.append(f"Не закрыто: {failed[0]}")
    elif quote:
        total = _v3_total_price_text(quote)
        if total:
            lines.append(f"Итого: {total}")
        lines.append(_v3_state_label(state))
        status_label = _v3_short_text(_v3_user_text(quote.get("client_status_label")), 120)
        if status_label:
            lines.append(status_label)
        summary = _v3_short_text(_v3_user_text(quote.get("client_summary")), 220)
        if summary:
            lines.append(summary)
        coverage = _v3_short_text(_v3_user_text(quote.get("coverage_summary")), 180)
        if coverage:
            lines.append(f"Покрытие: {coverage}")
        target_decisions = _v3_target_decision_lines(quote, limit=1)
        for item in target_decisions:
            lines.append(f"Объект: {_v3_short_text(item, 170)}")
        deviations = _v3_quote_deviation_items(quote, limit=2)
        for item in deviations:
            lines.append(f"Отличие: {_v3_short_text(item, 160)}")
        gaps = _v3_user_text_list(quote.get("procurement_gaps"), limit=1)
        for item in gaps:
            lines.append(f"Добрать: {_v3_short_text(item, 160)}")
        alternatives = _v3_available_alternative_list(
            quote.get("available_alternatives"),
            limit=1,
        )
        for item in alternatives:
            lines.append(f"Вариант для согласования: {_v3_short_text(item, 160)}")
    else:
        lines.append(_v3_state_label(state))

    lines.append("Полная детализация в Excel.")
    return _v3_fit_caption("\n".join(line for line in lines if line), safe_limit)


def _v3_state_label(state: str) -> str:
    labels = {
        "quote_draft_review_required": "черновик КП, нужна инженерная проверка",
        "quote_candidate_customer_ready": "кандидат для КП",
        "no_recommendation": "нет валидной рекомендации",
        "matrix_too_large_for_model": "матрица не поместилась в модель",
        "matrix_empty_after_category_selection": (
            "в выбранной категории нет складских строк с ценой"
        ),
        "provider_error": "ошибка LLM-провайдера",
        "provider_not_configured": "LLM не настроена",
        "mechanical_validation_failed": "ответ LLM не прошел проверку контракта или склада",
        "schema_validation_failed": "ответ LLM в неверном формате",
        "stock_refresh_failed": "не удалось обновить склад перед КП",
    }
    return labels.get(state, state or "неизвестно")


def _v3_quote_lines(quote: Mapping[str, Any]) -> list[str]:
    raw_lines = quote.get("lines")
    if not isinstance(raw_lines, list):
        return []

    result: list[str] = []
    display_lines = _v3_group_quote_lines(raw_lines)
    for index, raw_line in enumerate(display_lines[:8], start=1):
        if not isinstance(raw_line, Mapping):
            continue
        role = _v3_user_text(raw_line.get("role"))
        title = _v3_line_title(raw_line)
        quantity = _int_value(raw_line.get("quantity")) or 1
        unit_price = _v3_price_text(
            raw_line.get("unit_price_value"),
            raw_line.get("unit_price_currency"),
        )
        line_total = _v3_price_text(
            raw_line.get("line_total_value"),
            raw_line.get("line_total_currency"),
        )
        price_parts = []
        if unit_price:
            price_parts.append(f"{unit_price}/шт.")
        if line_total:
            price_parts.append(f"сумма {line_total}")
        role_prefix = f"{_v3_role_label(role)}: " if role else ""
        result.append(f"{index}. {role_prefix}{title}")
        commercial_parts = [f"кол-во {quantity}", *price_parts]
        result.append(f"   {'; '.join(commercial_parts)}")
        note = _v3_short_text(_v3_line_note(raw_line))
        if note:
            result.append(f"   Примечание: {note}")

    extra_count = len(display_lines) - 8
    if extra_count > 0:
        result.append(f"... и еще {extra_count} поз.")
    return result


def _v3_line_note(raw_line: Mapping[str, Any]) -> str:
    parts: list[str] = []
    reconciliation_note = _v3_user_text(raw_line.get("reconciliation_note"))
    if reconciliation_note:
        return reconciliation_note
    reason = _v3_user_text(raw_line.get("reason"))
    if reason:
        parts.append(reason)
    included = _v3_included_components_text(raw_line.get("included_components_summary"))
    note_text = " ".join(parts).casefold()
    if (
        included
        and "в составе по матрице" not in note_text
        and included.casefold() not in note_text
    ):
        parts.append(f"В составе по матрице: {included}")
    return _v3_user_text(". ".join(parts))


def _v3_included_components_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return _v3_user_text(value)
    if isinstance(value, Mapping):
        labels = [
            ("cpu", "CPU"),
            ("ram", "RAM"),
            ("raid_or_controller", "RAID/controller"),
            ("storage", "storage"),
            ("network", "network"),
            ("power", "power"),
            ("rails", "rails"),
            ("hba_or_options", "HBA/options"),
        ]
        parts = []
        for key, label in labels:
            text = _v3_user_text(value.get(key))
            if text and not _v3_is_missing_component_summary(text):
                parts.append(f"{label}: {text}")
        return "; ".join(parts)
    if isinstance(value, Iterable):
        return "; ".join(
            item
            for item in (_v3_user_text(item) for item in value)
            if item and not _v3_is_missing_component_summary(item)
        )
    return _v3_user_text(value)


def _v3_is_missing_component_summary(value: str) -> bool:
    normalized = " ".join(str(value or "").casefold().split())
    return normalized in {"не указано", "not specified", "unknown", "n/a", "none"}


def _v3_validation_error_details(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_details = (
        payload.get("validation_error_details")
        or payload.get("v3_validation_error_details")
    )
    if not isinstance(raw_details, list):
        return []
    return [item for item in raw_details if isinstance(item, Mapping)]


def _v3_validation_failure_summary(payload: Mapping[str, Any]) -> str:
    details = _v3_validation_error_details(payload)
    stages = {
        str(item.get("stage") or "").strip()
        for item in details
        if str(item.get("stage") or "").strip()
    }
    if stages.intersection({"schema", "contract"}):
        return (
            "Черновик КП получен, но ответ модели не прошёл проверку структуры "
            "и полноты контракта."
        )
    if stages.intersection({"reference", "stock", "price", "arithmetic"}):
        return (
            "Черновик КП получен, но содержит недействительные складские ссылки, "
            "остатки или цены."
        )
    return "Черновик КП получен, но не прошёл механическую проверку."


def _v3_human_mechanical_errors(
    errors: Iterable[str],
    details: Iterable[Mapping[str, Any]] | None = None,
) -> list[str]:
    result: list[str] = []
    for detail in details or []:
        message = _v3_user_text(detail.get("message_ru"))
        if message and message not in result:
            result.append(message)
        if len(result) >= 4:
            return result
    for error in errors:
        label = _v3_human_mechanical_error(error)
        if label and label not in result:
            result.append(label)
        if len(result) >= 4:
            break
    return result or ["служебная проверка склада и цен не сошлась с ответом LLM"]


def _v3_human_mechanical_error(error: str) -> str:
    exact_labels = {
        "schema.quote_missing": "Ответ модели не содержит объект quote.",
        "schema.quote_lines_missing": "Ответ модели не содержит ни одной строки КП.",
        "contract.anchor_required_not_selected": (
            "Для объекта есть складские anchor-кандидаты, но модель не выбрала "
            "базовое устройство."
        ),
        "contract.partial_without_anchor_forbidden": (
            "Модель вернула частичный вариант без anchor, хотя anchor-кандидаты "
            "были переданы."
        ),
        "contract.object_results_missing": "Модель не вернула object_results по объектам запроса.",
        "contract.object_results_coverage_mismatch": (
            "object_results не покрывает все объекты запроса."
        ),
        "contract.anchor_search_audit_missing": (
            "Модель не вернула anchor_search_audit по объектам запроса."
        ),
        "contract.anchor_manifest_coverage_mismatch": (
            "anchor_search_audit не соответствует объектам из selection_contract."
        ),
        "contract.compatibility_line_checks_missing": (
            "Модель не вернула построчные проверки совместимости."
        ),
        "contract.compatibility_line_check_coverage_mismatch": (
            "Проверки совместимости не покрывают все строки КП."
        ),
        "contract.dominance_audit_missing": "Модель не вернула dominance_audit по строкам КП.",
        "contract.dominance_audit_line_coverage_mismatch": (
            "dominance_audit не покрывает все строки КП."
        ),
        "reference.unknown_component_candidate_id": (
            "LLM выбрала component ID, которого нет в переданной матрице."
        ),
        "reference.unknown_stock_row_id": (
            "LLM выбрала строку склада, которой нет в переданной матрице."
        ),
        "reference.stock_row_product_mismatch": (
            "Выбранная строка склада относится к другому товару."
        ),
        "stock.insufficient_quantity": "Выбрано больше единиц, чем есть в выбранной строке склада.",
        "price.unit_price_mismatch": "Цена в строке не совпала с ценой склада.",
        "arithmetic.quote_total_mismatch": "Итоговая сумма не совпала с суммой строк.",
    }
    if error in exact_labels:
        return exact_labels[error]
    if "quantity_exceeds_stock" in error:
        return "выбрано больше единиц, чем есть в выбранной строке склада"
    if "stock_row_overallocated" in error:
        return "одна строка склада использована сверх доступного остатка"
    if "total_price_mismatch" in error:
        return "итоговая сумма не совпала с суммой строк"
    if "line_total_mismatch" in error:
        return "сумма одной из строк посчитана неверно"
    if "unit_price_mismatch" in error:
        return "цена в строке не совпала с ценой склада"
    if "unknown_component_candidate_id" in error:
        return "LLM выбрала component ID, которого нет в переданной матрице"
    if "unknown_stock_row_id" in error:
        return "LLM выбрала строку склада, которой нет в переданной матрице"
    if "stock_row_component_mismatch" in error:
        return "выбранная строка склада относится к другому компоненту"
    if "compatibility_check" in error:
        return "LLM не подтвердила техническую совместимость выбранных строк"
    if "price_order" in error:
        return "по выбранной строке не хватило складской цены"
    if "currency" in error:
        return "валюта в ответе не совпала с валютой склада"
    return "ответ LLM не совпал с фактической матрицей склада и цен"


def _v3_compatibility_check_lines(value: Any) -> list[str]:
    check = _v3_mapping(value)
    if not check:
        return []

    result: list[str] = []
    status = _v3_user_text(check.get("status"))
    if status:
        result.append(f"- Статус: {_v3_compatibility_status_label(status)}")

    for fact in _v3_user_text_list(check.get("checked_facts"), limit=4):
        result.append(f"- Подтверждено: {fact}")
    for mismatch in _v3_user_text_list(check.get("blocking_mismatches"), limit=3):
        result.append(f"- Блокер: {mismatch}")
    for conflict in _v3_user_text_list(check.get("selected_line_conflicts"), limit=3):
        result.append(f"- Конфликт строк: {conflict}")
    for risk in _v3_user_text_list(check.get("unresolved_risks"), limit=3):
        result.append(f"- Риск: {risk}")
    return result


def _v3_final_note(quote: Mapping[str, Any]) -> str:
    selection_mode = str(quote.get("selection_mode") or "").strip().lower()
    completeness_status = str(quote.get("completeness_status") or "").strip().lower()
    if (
        selection_mode in {
            "partial_build",
            "partial_with_anchor",
            "partial_without_anchor",
            "anchor_only",
        }
        or completeness_status in {"partial", "anchor_only"}
        or _v3_user_text_list(quote.get("procurement_gaps"), limit=1)
    ):
        return (
            "Это ближайший доступный складской вариант по переданной матрице. "
            "Он не закрывает ТЗ полностью; недостающие позиции и согласования указаны выше."
        )
    deviations = _v3_user_text_list(quote.get("deviation_notes"), limit=1)
    if deviations:
        return (
            "Это лучший доступный складской аналог по переданной матрице. "
            "Отклонения от ТЗ указаны выше; перед отправкой клиенту нужна "
            "инженерная и коммерческая проверка."
        )
    compatibility = _v3_mapping(quote.get("compatibility_check"))
    unresolved_risks = _v3_user_text_list(
        compatibility.get("unresolved_risks"),
        limit=1,
    )
    if unresolved_risks:
        return (
            "Это предварительный вариант КП по складской матрице. Перед отправкой "
            "клиенту инженер должен закрыть указанные риски совместимости."
        )
    return (
        "Это самый дешевый минимально достаточный технически рабочий вариант из "
        "переданной складской матрицы. Перед отправкой клиенту нужна инженерная проверка."
    )


def _v3_line_title(line: Mapping[str, Any]) -> str:
    title = _v3_user_text(line.get("title"))
    if title:
        return title

    producer = _v3_user_text(line.get("producer"))
    part_number = _v3_user_text(line.get("part_number"))
    item_name = _v3_user_text(line.get("item_name") or line.get("product_name"))
    if producer or part_number:
        return " ".join(part for part in (producer, part_number, item_name) if part)

    return "позиция из матрицы"


def _v3_price_text(value: Any, currency: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        amount = (
            value
            if isinstance(value, Decimal)
            else Decimal(str(value).replace(" ", "").replace(",", "."))
        )
    except (InvalidOperation, ValueError):
        return format_price(value, currency)

    quantized = amount.quantize(Decimal("0.01"))
    if quantized == quantized.to_integral_value():
        formatted = f"{int(quantized):,}".replace(",", " ")
    else:
        formatted = f"{quantized:,.2f}".replace(",", " ").replace(".", ",")

    currency_text = str(currency or "").strip()
    return f"{formatted} {currency_text}".strip()


def _v3_total_price_text(quote: Mapping[str, Any]) -> str:
    total = _v3_price_text(
        quote.get("total_price_value"),
        quote.get("total_price_currency"),
    )
    if total:
        return total
    return _v3_totals_by_currency_text(quote.get("totals_by_currency"))


def _v3_totals_by_currency_text(value: Any) -> str:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        total = _v3_price_text(
            item.get("value") or item.get("total_price_value") or item.get("amount"),
            item.get("currency") or item.get("total_price_currency"),
        )
        if total:
            parts.append(total)
    return " + ".join(parts)


def _v3_group_quote_lines(raw_lines: Iterable[Any]) -> list[Mapping[str, Any]]:
    grouped: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str, str], int] = {}
    quantity_sums: dict[tuple[str, str, str, str], Decimal] = {}
    total_sums: dict[tuple[str, str, str, str], Decimal] = {}

    for raw_line in raw_lines:
        if not isinstance(raw_line, Mapping):
            continue
        key = _v3_quote_line_group_key(raw_line)
        quantity = _v3_decimal_value(raw_line.get("quantity")) or Decimal("1")
        line_total = _v3_decimal_value(raw_line.get("line_total_value"))
        if key is None:
            grouped.append(dict(raw_line))
            continue
        if key not in by_key:
            by_key[key] = len(grouped)
            grouped.append(dict(raw_line))
            quantity_sums[key] = quantity
            if line_total is not None:
                total_sums[key] = line_total
            continue

        index = by_key[key]
        quantity_sums[key] += quantity
        if line_total is not None and key in total_sums:
            total_sums[key] += line_total
        merged = grouped[index]
        merged["quantity"] = _v3_decimal_json(quantity_sums[key])
        if key in total_sums:
            merged["line_total_value"] = _v3_decimal_json(total_sums[key])
        else:
            unit_price = _v3_decimal_value(merged.get("unit_price_value"))
            if unit_price is not None:
                merged["line_total_value"] = _v3_decimal_json(
                    unit_price * quantity_sums[key]
                )
        merged["line_total_currency"] = merged.get("unit_price_currency")
    return grouped


def _v3_quote_line_group_key(line: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    product_key = _v3_user_text(
        line.get("component_candidate_id") or line.get("part_number")
    )
    part_number = _v3_user_text(line.get("part_number"))
    unit_price = _v3_decimal_value(line.get("unit_price_value"))
    currency = _v3_user_text(line.get("unit_price_currency"))
    if not product_key or unit_price is None or not currency:
        return None
    return (product_key, part_number, str(unit_price.normalize()), currency)


def _v3_decimal_value(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return (
            value
            if isinstance(value, Decimal)
            else Decimal(str(value).replace(" ", "").replace(",", "."))
        )
    except (InvalidOperation, ValueError):
        return None


def _v3_decimal_json(value: Decimal) -> int | str:
    if value == value.to_integral_value():
        return int(value)
    return format(value.normalize(), "f")


def _v3_distributor_label(distributor_code: str) -> str:
    labels = {
        "ocs": "OCS",
        "treolan": "Treolan",
    }
    return labels.get(distributor_code.lower(), distributor_code)


def _v3_role_label(role: str) -> str:
    labels = {
        "platform": "Платформа",
        "server": "Сервер",
        "configured_system": "Готовая система",
        "cpu": "Процессор",
        "processor": "Процессор",
        "ram": "Оперативная память",
        "memory": "Оперативная память",
        "storage": "Накопитель",
        "ssd": "Накопитель",
        "hdd": "Накопитель",
        "drive": "Диск",
        "nas": "СХД/NAS",
        "storage_system": "СХД",
        "network": "Сеть",
        "switch": "Коммутатор",
        "network_adapter": "Сетевой адаптер",
        "power_supply": "Блок питания",
        "cable": "Кабель",
        "license": "Лицензия",
        "support": "Поддержка",
    }
    return labels.get(role.lower(), role)


def _v3_service_parts(
    payload: Mapping[str, Any],
    profile: str,
    category_ids: list[str],
    row_count: Any,
) -> list[str]:
    result: list[str] = []
    match_run_id = payload.get("match_run_id")
    if match_run_id:
        result.append(f"ID {match_run_id}")
    if profile:
        result.append(f"профиль {_v3_profile_label(profile)}")
    if category_ids:
        result.append(f"категорий {len(category_ids)}")
    if row_count is not None:
        result.append(f"матрица {row_count} строк")
    return result


def _v3_profile_label(profile: str) -> str:
    labels = {
        "server": "серверы",
        "storage": "СХД/хранилища",
        "network": "сеть",
        "custom": "авто",
    }
    return labels.get(profile.lower(), profile)


def _v3_selection_summary(quote: Mapping[str, Any]) -> str:
    parts = [
        _v3_solution_scope_label(quote.get("solution_scope")),
        _v3_substitution_policy_label(quote.get("substitution_policy")),
        _v3_selection_mode_label(quote.get("selection_mode")),
        _v3_completeness_status_label(quote.get("completeness_status")),
        _v3_operational_status_label(quote.get("operational_status")),
    ]
    return "; ".join(part for part in parts if part)


def _v3_solution_scope_label(value: Any) -> str:
    labels = {
        "complete_system": "готовая система",
        "configured_system": "конфигурируемая система",
        "standalone_product": "отдельный товар",
        "replacement_component": "замена компонента",
        "expansion_or_upgrade": "расширение или апгрейд",
        "accessory": "аксессуар",
        "multi_product_solution": "комплект",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_substitution_policy_label(value: Any) -> str:
    labels = {
        "forbidden": "без аналогов",
        "allowed_no_downgrade": "аналоги без ухудшения",
        "allowed_with_disclosed_downgrade": "аналоги с раскрытием отклонений",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_selection_mode_label(value: Any) -> str:
    labels = {
        "exact": "точное соответствие",
        "exact_complete": "полное точное соответствие",
        "equivalent_or_better": "равноценно или лучше",
        "equivalent_complete": "полный функциональный аналог",
        "analog_with_downgrade": "аналог с отклонением",
        "downgraded_complete": "полный аналог с отклонениями",
        "partial_build": "частичная складская сборка",
        "partial_with_anchor": "частичная сборка со складской базой",
        "partial_without_anchor": "частичный складской комплект",
        "anchor_only": "только складская база",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_completeness_status_label(value: Any) -> str:
    labels = {
        "complete": "комплектность: закрыто полностью",
        "partial": "комплектность: частично",
        "anchor_only": "комплектность: только база",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_operational_status_label(value: Any) -> str:
    labels = {
        "ready": "работоспособность: готово после проверки",
        "ready_after_customer_environment_check": (
            "работоспособность: нужна проверка среды заказчика"
        ),
        "incomplete_needs_procurement": "работоспособность: нужно добрать позиции",
        "operational": "работоспособность: готово после проверки",
        "operational_with_deviations": "работоспособность: есть отклонения",
        "requires_completion": "работоспособность: нужно добрать позиции",
        "not_claimed": "работоспособность: не заявляется",
        "not_applicable": "работоспособность: не применимо",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _v3_user_text(value: Any) -> str:
    return sanitize_user_facing_text(value)


def _v3_user_text_list(value: Any, *, limit: int) -> list[str]:
    result: list[str] = []
    for item in _v3_iter_list_items(value):
        text = _v3_short_text(_v3_display_item(item), V3_TELEGRAM_LIST_ITEM_LIMIT)
        if text:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _v3_available_alternative_list(value: Any, *, limit: int) -> list[str]:
    result: list[str] = []
    for item in _v3_iter_list_items(value):
        text = _v3_short_text(_v3_available_alternative_text(item), V3_TELEGRAM_LIST_ITEM_LIMIT)
        if text:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _v3_available_alternative_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _v3_user_text(value)

    item = _v3_user_text(
        value.get("item")
        or value.get("item_name")
        or value.get("description")
        or value.get("title")
        or value.get("part_number")
        or value.get("role")
    )
    quantity = _v3_quantity_text(
        value.get("available_quantity")
        or value.get("stock_quantity")
        or value.get("quantity")
        or value.get("qty"),
        lower_bound=bool(value.get("quantity_is_greater_than")),
    )
    price = _v3_price_text(
        value.get("unit_price_value") or value.get("price_value") or value.get("price"),
        value.get("unit_price_currency") or value.get("price_currency") or value.get("currency"),
    )
    reason = _v3_user_text(
        value.get("reason")
        or value.get("comment")
        or value.get("summary")
        or value.get("difference")
    )
    requirement = _v3_user_text(
        value.get("requirement_id")
        or value.get("requirement")
        or value.get("requested")
        or value.get("role")
    )
    parts = [
        f"требование: {requirement}" if requirement else "",
        item,
        f"доступно: {quantity}" if quantity else "",
        f"цена: {price}" if price else "",
        reason,
    ]
    return _v3_user_text(". ".join(part for part in parts if part))


def _v3_target_decision_lines(quote: Mapping[str, Any], *, limit: int) -> list[str]:
    result: list[str] = []
    for item in _v3_iter_list_items(quote.get("target_decisions")):
        if not isinstance(item, Mapping):
            text = _v3_short_text(_v3_user_text(item), 180)
            if text:
                result.append(text)
        else:
            label = _v3_user_text(
                item.get("target_label") or item.get("label") or item.get("target_id")
            )
            status = _v3_target_anchor_status_label(item.get("anchor_status"))
            line_id = _v3_user_text(item.get("anchor_line_id"))
            reason = _v3_short_text(_v3_user_text(item.get("reason")), 120)
            parts = [
                label,
                status,
                f"строка {line_id}" if line_id else "",
                reason,
            ]
            text = _v3_short_text(" - ".join(part for part in parts if part), 180)
            if text:
                result.append(text)
        if len(result) >= limit:
            break
    return result


def _v3_target_anchor_status_label(value: Any) -> str:
    labels = {
        "selected": "выбран",
        "covered": "закрыт выбранной строкой",
        "covered_by_selected_line": "закрыт выбранной строкой",
        "not_found": "не найден в матрице",
        "not_required": "отдельная якорная строка не нужна",
        "rejected": "не выбран",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_quote_deviation_items(quote: Mapping[str, Any], *, limit: int) -> list[str]:
    values = [
        *_v3_iter_list_items(quote.get("key_deviations")),
        *_v3_iter_list_items(quote.get("deviation_notes")),
    ]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _v3_short_text(_v3_display_item(item), V3_TELEGRAM_LIST_ITEM_LIMIT)
        key = " ".join(text.casefold().split())
        if text and key not in seen:
            result.append(text)
            seen.add(key)
        if len(result) >= limit:
            break
    return result


def _v3_iter_list_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable):
        return [item for item in value if item not in (None, "")]
    return [value]


def _v3_display_item(value: Any) -> str:
    if not isinstance(value, Mapping):
        return _v3_user_text(value)

    if "item" in value or "quantity" in value:
        return _v3_procurement_gap_text(value)

    if "status" in value and ("required_for" in value or "next_action" in value):
        role = _v3_user_text(value.get("role"))
        requested = _v3_user_text(value.get("requested"))
        status = _v3_gap_status_label(value.get("status"))
        required_for = _v3_gap_required_for_label(value.get("required_for"))
        impact = _v3_user_text(value.get("impact"))
        next_action = _v3_user_text(value.get("next_action"))
        parts = [
            f"роль: {role}" if role else "",
            f"просили: {requested}" if requested else "",
            status,
            required_for,
            f"влияние: {impact}" if impact else "",
            f"дальше: {next_action}" if next_action else "",
        ]
        return _v3_user_text(". ".join(part for part in parts if part))

    if "requested" in value or "offered" in value:
        parts: list[str] = []
        requirement_id = _v3_user_text(value.get("requirement_id"))
        requirement = _v3_user_text(value.get("requirement"))
        if requirement_id or requirement:
            label = " ".join(part for part in (requirement_id, requirement) if part)
            parts.append(f"Требование: {label}")
        requested = _v3_user_text(value.get("requested"))
        offered = _v3_user_text(value.get("offered"))
        if requested or offered:
            parts.append(
                f"просили: {requested or 'не указано'}; "
                f"предложено: {offered or 'не указано'}"
            )
        direction = _v3_deviation_direction_label(value.get("direction"))
        severity = _v3_deviation_severity_label(value.get("severity"))
        if direction or severity:
            parts.append("; ".join(part for part in (direction, severity) if part))
        impact = _v3_user_text(value.get("impact"))
        if impact:
            parts.append(f"влияние: {impact}")
        reason = _v3_user_text(value.get("reason"))
        if reason:
            parts.append(f"почему: {reason}")
        return _v3_user_text(". ".join(parts))

    if "outcome" in value or "priority" in value:
        requirement_id = _v3_user_text(value.get("requirement_id"))
        requirement = _v3_user_text(value.get("requirement"))
        priority = _v3_requirement_priority_label(value.get("priority"))
        outcome = _v3_requirement_outcome_label(value.get("outcome"))
        offered = _v3_user_text(value.get("offered"))
        impact = _v3_user_text(value.get("impact"))
        parts = [
            part
            for part in (
                " ".join(part for part in (requirement_id, requirement) if part),
                priority,
                outcome,
                f"предложено: {offered}" if offered else "",
                impact,
            )
            if part
        ]
        return _v3_user_text(". ".join(parts))

    if "result" in value:
        result = _v3_user_text(value.get("result"))
        evidence = ", ".join(_v3_user_text_list(value.get("evidence"), limit=2))
        return _v3_user_text("; ".join(part for part in (result, evidence) if part))

    if any(key in value for key in ("description", "comment", "summary", "needed_action")):
        requirement_id = _v3_user_text(value.get("requirement_id"))
        requested = _v3_user_text(value.get("requested"))
        action = _v3_user_text(value.get("needed_action"))
        description = _v3_user_text(
            value.get("description") or value.get("comment") or value.get("summary")
        )
        reason = _v3_user_text(value.get("reason"))
        parts = [
            requirement_id,
            f"просили: {requested}" if requested else "",
            description,
            f"действие: {action}" if action else "",
            f"причина: {reason}" if reason else "",
        ]
        return _v3_user_text(". ".join(part for part in parts if part))

    if "requirement" in value or "reason" in value:
        requirement_id = _v3_user_text(value.get("requirement_id"))
        requirement = _v3_user_text(value.get("requirement"))
        reason = _v3_user_text(value.get("reason"))
        parts = []
        label = " ".join(part for part in (requirement_id, requirement) if part)
        if label:
            parts.append(f"Требование: {label}")
        if reason:
            parts.append(f"причина: {reason}")
        return _v3_user_text(". ".join(parts))

    return _v3_user_text(value)


def _v3_procurement_gap_text(value: Mapping[str, Any]) -> str:
    item = _v3_user_text(
        value.get("item")
        or value.get("requested")
        or value.get("requirement")
        or value.get("role")
    )
    quantity = _v3_quantity_text(value.get("quantity") or value.get("qty"))
    reason = _v3_user_text(
        value.get("reason")
        or value.get("comment")
        or value.get("summary")
        or value.get("impact")
        or value.get("next_action")
    )
    headline = " - ".join(part for part in (item, quantity) if part)
    if headline and reason:
        return _v3_user_text(f"{headline}\n  {reason}")
    return _v3_user_text(headline or reason)


def _v3_quantity_text(value: Any, *, lower_bound: bool = False) -> str:
    text = _v3_user_text(value)
    if not text:
        return ""
    if lower_bound and not text.endswith("+"):
        text = f"{text}+"
    if re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        return f"{text} шт."
    if re.fullmatch(r"\d+(?:[.,]\d+)?\+", text):
        return f"{text} шт."
    return text


def _v3_deviation_direction_label(value: Any) -> str:
    labels = {
        "upgrade": "лучше исходного требования",
        "downgrade": "хуже исходного требования",
        "different": "другое отличие",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_deviation_severity_label(value: Any) -> str:
    labels = {
        "minor": "незначимое отклонение",
        "material": "существенное отклонение",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_gap_status_label(value: Any) -> str:
    labels = {
        "not_in_matrix": "нет в переданной матрице",
        "out_of_stock": "нет доступного остатка",
        "quantity_shortage": "не хватает количества",
        "no_compatible_item_proven": "совместимый вариант не подтвержден по матрице",
        "not_included": "не входит в выбранную позицию",
        "unknown": "нужно уточнение",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_gap_required_for_label(value: Any) -> str:
    labels = {
        "operational_readiness": "нужно для работоспособности",
        "requested_spec": "нужно для полного закрытия ТЗ",
        "optional_preference": "опциональное пожелание",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_requirement_priority_label(value: Any) -> str:
    labels = {
        "non_negotiable": "обязательное требование",
        "locked": "зафиксированное требование",
        "core": "основная задача",
        "important": "важное требование",
        "target": "целевое требование",
        "preference": "предпочтение",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_requirement_outcome_label(value: Any) -> str:
    labels = {
        "met": "закрыто",
        "exceeded": "закрыто с запасом",
        "equivalent": "закрыто аналогом",
        "degraded": "закрыто с ухудшением",
        "partially_met": "закрыто частично",
        "missing": "не закрыто",
        "unknown": "нужно уточнение",
        "substituted": "заменено аналогом",
        "not_met": "не закрыто",
        "not_applicable": "не применимо",
    }
    return labels.get(str(value or "").strip().lower(), _v3_user_text(value))


def _v3_short_text(text: str, max_chars: int = V3_TELEGRAM_FIELD_LIMIT) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip(" ,.;:") + "..."


def _v3_priority_review_lines(quote: Mapping[str, Any]) -> list[str]:
    compatibility = _v3_mapping(quote.get("compatibility_check"))
    items: list[str] = []
    seen_keys: set[str] = set()
    for value in [
        *_v3_stock_confirmation_lines(quote, limit=3),
        *_v3_user_text_list(compatibility.get("unresolved_risks"), limit=2),
        *_v3_user_text_list(quote.get("engineer_checks"), limit=3),
    ]:
        key = _v3_review_item_key(value)
        if value and key not in seen_keys:
            items.append(value)
            seen_keys.add(key)
        if len(items) >= 5:
            break
    return items


def _v3_stock_confirmation_lines(
    quote: Mapping[str, Any],
    *,
    limit: int,
) -> list[str]:
    quote_lines = [
        line for line in _v3_iter_list_items(quote.get("lines")) if isinstance(line, Mapping)
    ]
    items: list[str] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    for line in quote_lines:
        if not line.get("stock_confirmation_required"):
            continue
        text = _v3_stock_confirmation_text(
            title=_v3_line_title(line),
            included_quantity=_v3_quantity_text(line.get("quantity")),
            stock_quantity=_v3_quantity_text(
                line.get("quantity_value"),
                lower_bound=True,
            ),
        )
        key = (
            _v3_user_text(line.get("component_candidate_id")),
            _v3_user_text(line.get("stock_row_id")),
            _v3_user_text(line.get("quantity")),
            _v3_user_text(line.get("quantity_value")),
        )
        if text and key not in seen_keys:
            items.append(text)
            seen_keys.add(key)
        if len(items) >= limit:
            return items

    integrity = _v3_mapping(quote.get("quote_integrity"))
    for adjustment in _v3_iter_list_items(integrity.get("adjustments")):
        if not isinstance(adjustment, Mapping):
            continue
        if _v3_user_text(adjustment.get("type")) != "stock_lower_bound_quantity_confirm":
            continue
        component_id = _v3_user_text(adjustment.get("component_candidate_id"))
        stock_row_id = _v3_user_text(adjustment.get("stock_row_id"))
        included_quantity = _v3_quantity_text(adjustment.get("included_quantity"))
        stock_quantity = _v3_quantity_text(
            adjustment.get("displayed_available_quantity"),
            lower_bound=True,
        )
        key = (component_id, stock_row_id, included_quantity, stock_quantity)
        if key in seen_keys:
            continue
        line = _v3_quote_line_for_stock_confirmation(
            quote_lines,
            component_id=component_id,
            stock_row_id=stock_row_id,
        )
        title = _v3_line_title(line) if line else component_id or stock_row_id
        text = _v3_stock_confirmation_text(
            title=title,
            included_quantity=included_quantity,
            stock_quantity=stock_quantity,
        )
        if text:
            items.append(text)
            seen_keys.add(key)
        if len(items) >= limit:
            break
    return items


def _v3_quote_line_for_stock_confirmation(
    quote_lines: list[Mapping[str, Any]],
    *,
    component_id: str,
    stock_row_id: str,
) -> Mapping[str, Any] | None:
    if stock_row_id:
        for line in quote_lines:
            if _v3_user_text(line.get("stock_row_id")) == stock_row_id:
                return line
    if component_id:
        for line in quote_lines:
            if _v3_user_text(line.get("component_candidate_id")) == component_id:
                return line
    return None


def _v3_stock_confirmation_text(
    *,
    title: str,
    included_quantity: str,
    stock_quantity: str,
) -> str:
    if not title and not included_quantity and not stock_quantity:
        return ""
    return _v3_short_text(
        (
            f"{title or 'Позиция из матрицы'}: в КП {included_quantity or 'не указано'}; "
            f"склад показывает {stock_quantity or 'нижнюю границу остатка'}. "
            "Подтвердить доступность выбранного количества."
        ),
        V3_TELEGRAM_LIST_ITEM_LIMIT,
    )


def _v3_review_item_key(value: str) -> str:
    normalized = " ".join(str(value or "").casefold().split())
    for marker in (
        "радиатор",
        "heatsink",
        "carrier",
        "caddy",
        "контроллер",
        "raid",
        "sfp",
        "трансивер",
        "кабель",
        "cable",
    ):
        if marker in normalized:
            return marker
    return re.sub(r"\W+", " ", normalized)[:96]


def _v3_fit_telegram_message(text: str, safe_limit: int) -> str:
    if len(text) <= safe_limit:
        return text
    note = "\n\nПолная детализация отправлена в Excel-файле."
    limit = max(0, safe_limit - len(note) - 3)
    kept_lines: list[str] = []
    current_len = 0
    for line in text.splitlines():
        next_len = current_len + len(line) + (1 if kept_lines else 0)
        if next_len > limit:
            break
        kept_lines.append(line)
        current_len = next_len
    compact = "\n".join(kept_lines).rstrip()
    if not compact:
        compact = text[:limit].rstrip(" ,.;:")
    return compact + "..." + note


def _v3_fit_caption(text: str, safe_limit: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= safe_limit:
        return clean
    return clean[: max(0, safe_limit - 3)].rstrip(" ,.;:") + "..."


def _v3_no_recommendation_summary(no_recommendation: Mapping[str, Any]) -> str:
    if str(no_recommendation.get("fallback_reason") or "") == "v3_stock_refresh_failed":
        return (
            "КП не сформировано: не удалось обновить склад перед подбором. "
            "LLM Composer не запускался."
        )
    summary = _v3_short_text(_v3_user_text(no_recommendation.get("summary")))
    if not summary:
        return "КП не сформировано: не удалось закрыть требования по складской матрице."
    if _v3_looks_english(summary):
        failed = _v3_no_recommendation_failed_items(no_recommendation, limit=1)
        if failed and any(marker in failed[0].lower() for marker in ("cpu", "процессор")):
            return "КП не сформировано: указанная модель процессора не найдена на выбранном складе."
        return "КП не сформировано: не удалось закрыть требования по складской матрице."
    return summary


def _v3_no_recommendation_details(no_recommendation: Mapping[str, Any]) -> str:
    details = _v3_short_text(_v3_user_text(no_recommendation.get("details")))
    return "" if _v3_looks_english(details) else details


def _v3_no_recommendation_failed_items(
    no_recommendation: Mapping[str, Any],
    *,
    limit: int,
) -> list[str]:
    result: list[str] = []
    for item in _v3_iter_list_items(no_recommendation.get("failed_requirements")):
        text = _v3_known_missing_requirement(_v3_display_item(item))
        text = _v3_short_text(text, V3_TELEGRAM_LIST_ITEM_LIMIT)
        if text:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _v3_no_recommendation_next_actions(
    no_recommendation: Mapping[str, Any],
    *,
    limit: int,
) -> list[str]:
    raw_actions = [_v3_user_text(item) for item in as_string_list(
        no_recommendation.get("recommended_next_actions")
    )]
    if any(_v3_looks_english(item) for item in raw_actions):
        return [
            "Разрешить ближайший технически подходящий аналог по отсутствующей позиции.",
            "Если нужна строго указанная модель, проверить другой склад или поставку под заказ.",
        ][:limit]
    return [_v3_short_text(item, V3_TELEGRAM_LIST_ITEM_LIMIT) for item in raw_actions if item][
        :limit
    ]


def _v3_known_missing_requirement(text: str) -> str:
    clean = _v3_short_text(text, V3_TELEGRAM_LIST_ITEM_LIMIT)
    if not clean:
        return ""
    cpu_match = re.match(
        r"(?i)^CPU model:\s*(.+?)\s+is missing from (?:the )?matrix\.?$",
        clean,
    )
    if cpu_match:
        model = cpu_match.group(1).strip()
        model = _v3_russianize_core_count(model)
        return f"Не найдено в матрице склада: процессор {model}."
    missing_match = re.match(
        r"(?i)^(.+?)\s+is missing from (?:the )?matrix\.?$",
        clean,
    )
    if missing_match:
        return f"Не найдено в матрице склада: {missing_match.group(1).strip()}."
    return clean


def _v3_compatibility_status_label(status: str) -> str:
    labels = {
        "compatible": "совместимо",
        "compatible_selected_lines": "выбранные строки совместимы",
        "anchor_only": "выбрана только складская база",
        "confirmed_selected_set": "выбранные строки подтверждены по матрице",
        "review_required_selected_set": "выбранные строки требуют инженерной проверки",
        "independent_partial_set": "частичный независимый складской комплект",
        "incompatible": "несовместимо",
        "insufficient_facts": "недостаточно данных",
    }
    return labels.get(status.lower(), status)


def _v3_russianize_core_count(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        count = int(match.group(1))
        last_two = count % 100
        last = count % 10
        if 11 <= last_two <= 14:
            word = "ядер"
        elif last == 1:
            word = "ядро"
        elif 2 <= last <= 4:
            word = "ядра"
        else:
            word = "ядер"
        return f"{count} {word}"

    return re.sub(
        r"\b(\d+)\s+(?:cores?|ядро|ядра|ядер)\b",
        replace,
        text,
        flags=re.IGNORECASE,
    )


def _v3_looks_english(text: str) -> bool:
    clean = str(text or "")
    if not clean:
        return False
    latin = len(re.findall(r"[A-Za-z]", clean))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", clean))
    if latin >= 12 and cyrillic == 0:
        return True
    return latin >= 40 and latin > cyrillic * 2


def _safe_filename_part(value: Any) -> str:
    text = _v3_user_text(value).strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = text.strip("_")
    return text or "custom"


def format_match_summary(summary: Mapping[str, Any]) -> str:
    match_run_id = summary.get("match_run_id", "?")
    configuration_groups = _configuration_group_mappings(summary)
    ai_recommendations = _ai_recommendation_mappings(summary)
    if _llm_enabled(summary):
        primary_recommendation = _primary_recommendation_mapping(summary)
        if (
            summary.get("primary_recommendation_status") == "valid"
            and primary_recommendation
        ):
            return _output_gate(
                _format_primary_commercial_summary(
                    summary,
                    primary_recommendation=primary_recommendation,
                    match_run_id=match_run_id,
                ),
                match_run_id=match_run_id,
            )
        if summary.get("grouped_presales_mode_used") is True and configuration_groups:
            return _output_gate(
                _format_grouped_presales_summary(
                    summary,
                    configuration_groups=configuration_groups,
                    match_run_id=match_run_id,
                ),
                match_run_id=match_run_id,
            )
        if ai_recommendations:
            return _output_gate(
                _format_llm_match_summary(
                    summary,
                    recommendations=ai_recommendations,
                    match_run_id=match_run_id,
                ),
                match_run_id=match_run_id,
            )
        if _is_no_safe_ai_reason(summary):
            return _safe_no_recommendation_message(match_run_id, summary=summary)
        return _safe_ai_unavailable_message(match_run_id)

    ready_candidates = _summary_candidates(
        summary,
        field_name="ready_stock_candidates",
        fallback_candidate_type="ready_server",
    )
    build_candidates = _summary_candidates(
        summary,
        field_name="build_candidates",
        fallback_candidate_type="build_from_parts",
    )
    checks = _summary_checks(summary)
    lines = [
        f"Подбор по складу №{match_run_id}",
        f"Итог: {human_match_status_for_summary(summary.get('status'))}",
        _found_variants_text(_int_value(summary.get("total_candidates")) or 0),
        f"Полностью подходят: {summary.get('matched_items', 0)}",
        f"Нужна проверка инженера: {yes_no(summary.get('engineer_review_required'))}",
        "",
        "Готовые варианты",
        *_candidate_lines(ready_candidates),
        "",
        "Сборка из комплектующих",
        *_build_candidate_lines(build_candidates),
    ]
    fallback_notice = _llm_fallback_notice(summary)
    if fallback_notice:
        lines.extend(["", fallback_notice])

    if checks:
        check_lines = ["- " + check for check in checks[:MAX_SUMMARY_CHECKS]]
        if len(checks) > MAX_SUMMARY_CHECKS:
            check_lines.append(f"- ... и еще {len(checks) - MAX_SUMMARY_CHECKS}")
        lines.extend(["", "Что нужно проверить", *check_lines])

    lines.extend(
        [
            "",
            "Это предварительный подбор по складу, не финальная инженерная спецификация.",
        ]
    )
    return "\n".join(lines)


def _format_llm_match_summary(
    summary: Mapping[str, Any],
    *,
    recommendations: list[Mapping[str, Any]],
    match_run_id: Any,
) -> str:
    shown_candidates = recommendations[:MAX_SUMMARY_CANDIDATES]
    checks = _llm_summary_checks(summary, shown_candidates)
    lines = [
        f"AI-подбор по складу №{match_run_id}",
        _found_ai_recommendations_text(len(recommendations)),
        "Основа: текущие остатки и цены OCS",
    ]
    evidence_notice = _telegram_evidence_notice(summary, shown_candidates)
    if evidence_notice:
        lines.append(evidence_notice)
    repair_notice = _telegram_repair_notice(summary)
    if repair_notice:
        lines.append(repair_notice)
    lines.extend(
        [
            LLM_TELEGRAM_NOTE,
            "",
            "Рекомендации",
            "",
        ]
    )
    if len(recommendations) < MAX_SUMMARY_CANDIDATES:
        lines.extend(
            [
                _shown_safe_variants_message(len(recommendations), summary),
                "",
            ]
        )
    lines.extend(_llm_recommendation_lines(shown_candidates))

    if checks:
        lines.extend(["", "Что проверить инженеру:"])
        lines.extend("- " + check for check in checks[:MAX_SUMMARY_CHECKS])

    lines.extend(["", "Подробный отчет отправлен Excel-файлом."])
    return "\n".join(lines)


def _format_grouped_presales_summary(
    summary: Mapping[str, Any],
    *,
    configuration_groups: list[Mapping[str, Any]],
    match_run_id: Any,
) -> str:
    commercial = build_grouped_commercial_summary(
        summary,
        configuration_groups,
        match_run_id=match_run_id,
    )
    if commercial is None:
        return _safe_ai_unavailable_message(match_run_id)
    return "\n".join(grouped_commercial_telegram_lines(commercial))


def _format_primary_commercial_summary(
    summary: Mapping[str, Any],
    *,
    primary_recommendation: Mapping[str, Any],
    match_run_id: Any,
) -> str:
    commercial = build_primary_commercial_summary(
        summary,
        primary_recommendation,
        match_run_id=match_run_id,
    )
    if commercial is None:
        return _safe_ai_unavailable_message(match_run_id)
    return "\n".join(primary_commercial_telegram_lines(commercial))


def _group_component_per_server_lines(group: Mapping[str, Any]) -> list[str]:
    base = group.get("component_base")
    if not isinstance(base, Mapping):
        return []
    lines: list[str] = []
    cpu = base.get("cpu")
    if isinstance(cpu, Mapping):
        article = _component_article_from_component(cpu)
        quantity = _int_value(cpu.get("per_server_quantity"))
        if article and quantity is not None:
            lines.append(f"CPU: {quantity} x {article}")
    ram = base.get("ram")
    if isinstance(ram, Mapping):
        article = _component_article_from_component(ram)
        per_server = _int_value(ram.get("per_server_quantity"))
        module_gb = _ram_module_capacity_gb(ram)
        total_gb = _ram_total_gb_per_server(ram, per_server)
        if article and per_server is not None and module_gb is not None:
            total_text = f" = {total_gb} ГБ" if total_gb is not None else ""
            lines.append(f"RAM: {article} - {per_server} x {module_gb} ГБ{total_text}")
    storage = base.get("storage")
    if isinstance(storage, Mapping):
        label = "SSD" if storage.get("role") == "ssd" else "HDD"
        article = _component_article_from_component(storage)
        per_server = _int_value(storage.get("per_server_quantity"))
        capacity = _storage_capacity_tb(storage)
        interface = _component_fact_text(storage, "storage_interface")
        if article and per_server is not None:
            capacity_text = (
                f" x {_format_number(capacity)} ТБ" if capacity is not None else ""
            )
            interface_text = (
                f" {interface}" if interface and interface != "unknown" else ""
            )
            lines.append(f"{label}: {article} - {per_server}{capacity_text}{interface_text}")
    return lines


def _group_component_total_lines(group: Mapping[str, Any]) -> list[str]:
    base = group.get("component_base")
    if not isinstance(base, Mapping):
        return []
    storage = base.get("storage")
    storage_label = (
        "SSD"
        if isinstance(storage, Mapping) and storage.get("role") == "ssd"
        else "HDD"
    )
    rows = [
        ("CPU", base.get("cpu")),
        ("RAM", base.get("ram")),
        (storage_label, storage),
    ]
    lines: list[str] = []
    for label, component in rows:
        if not isinstance(component, Mapping):
            continue
        quantity = _int_value(component.get("quantity_required"))
        stock = _stock_count_text(component.get("available_quantity"))
        if label == "RAM":
            module_gb = _ram_module_capacity_gb(component)
            module_text = f", модули по {module_gb} ГБ" if module_gb is not None else ""
            lines.append(f"{label}: {_quantity_units(quantity)} всего{module_text}, склад: {stock}")
        else:
            lines.append(f"{label}: {_quantity_units(quantity)} всего, склад: {stock}")
    return lines


def _group_server_quantity(group: Mapping[str, Any]) -> int | None:
    options = _candidate_mappings(group.get("platform_options"))
    for option in options:
        platform = option.get("platform")
        if isinstance(platform, Mapping):
            quantity = _int_value(platform.get("quantity_required"))
            if quantity is not None:
                return quantity
    base = group.get("component_base")
    if isinstance(base, Mapping):
        for component in base.values():
            if isinstance(component, Mapping):
                server_qty = _int_value(component.get("server_quantity"))
                if server_qty is not None:
                    return server_qty
    return None


def _platform_option_role_label(option: Mapping[str, Any]) -> str:
    labels = {
        "cheapest_quote": "Самый дешевый для КП",
        "preferred_for_database": "Более спокойный под БД",
        "branded_safe": "Брендовый / инженерно понятный",
        "engineering_clear": "Инженерно понятный",
        "high_headroom": "С запасом",
        "alternative": "Альтернатива",
    }
    return labels.get(str(option.get("role") or option.get("option_role") or ""), "Альтернатива")


def _platform_option_platform_text(option: Mapping[str, Any]) -> str:
    platform = option.get("platform")
    if isinstance(platform, Mapping):
        article = _component_article_from_component(platform)
        if article:
            return article
    return "платформа требует уточнения"


def _platform_option_amount_text(
    option: Mapping[str, Any],
    group: Mapping[str, Any],
) -> str:
    amount = format_price(option.get("total_price_value"), option.get("total_price_currency"))
    server_qty = _group_server_quantity(group)
    if server_qty is None:
        return amount
    return (
        f"{amount} за {server_qty} "
        f"{pluralize_ru(server_qty, 'сервер', 'сервера', 'серверов')}"
    )


def _quote_recommendation_lines(quote: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    cheapest = make_telegram_sentence(
        quote.get("for_cheapest_quote"),
        max_chars=180,
        fallback="",
    )
    database = make_telegram_sentence(
        quote.get("for_database_preferred"),
        max_chars=180,
        fallback="",
    )
    if cheapest:
        lines.append(f"- Для КП: {cheapest}")
    if database and database != cheapest:
        lines.append(f"- Более спокойный технический вариант: {database}")
    summary = make_telegram_sentence(quote.get("summary"), max_chars=220, fallback="")
    if summary:
        lines.append(f"- Комментарий: {summary}")
    return lines


def _grouped_engineer_checks(groups: list[Mapping[str, Any]]) -> list[str]:
    checks: list[str] = []
    product_group = "server"
    for group in groups:
        group_product = str(group.get("product_group") or "").strip()
        if group_product:
            product_group = group_product
        for check in as_string_list(group.get("engineer_checks")):
            short = make_telegram_sentence(check, max_chars=130, fallback="")
            if short:
                checks.append(short)
        for option in _candidate_mappings(group.get("platform_options")):
            for check in as_string_list(option.get("engineer_checks")):
                short = make_telegram_sentence(check, max_chars=130, fallback="")
                if short:
                    checks.append(short)
    return grouped_engineer_check_summary(checks, product_group=product_group)


def _llm_enabled(summary: Mapping[str, Any]) -> bool:
    return summary.get("llm_configurator_enabled") is True or summary.get(
        "llm_configurator_used"
    ) is True


def _telegram_repair_notice(summary: Mapping[str, Any]) -> str:
    if summary.get("llm_repair_success") is not True and summary.get("llm_repair_used") is not True:
        return ""
    return (
        "AI перепроверил цены по матрице и выбрал более дешевую эквивалентную "
        "RAM/SSD, если она была доступна."
    )


def _telegram_evidence_notice(
    summary: Mapping[str, Any],
    recommendations: list[Mapping[str, Any]],
) -> str:
    pack = summary.get("web_evidence_pack")
    if isinstance(pack, Mapping) and pack.get("enabled") is not True:
        return ""

    summaries = [
        evidence_summary
        for candidate in recommendations
        if isinstance((evidence_summary := candidate.get("evidence_summary")), Mapping)
    ]
    if not summaries and not isinstance(pack, Mapping):
        return ""
    total_sources = sum(_int_value(item.get("sources_count")) or 0 for item in summaries)
    if total_sources <= 0:
        diagnostics = pack.get("diagnostics") if isinstance(pack, Mapping) else {}
        if isinstance(diagnostics, Mapping):
            total_sources = _int_value(diagnostics.get("evidence_sources_count")) or 0
    if total_sources > 0 and any(
        str(item.get("status") or "").strip() == "partially_confirmed"
        for item in summaries
    ):
        return (
            "Проверка: часть совместимости подтверждена внешними источниками, "
            "финальная проверка инженером обязательна."
        )
    if total_sources <= 0:
        has_evidence_attempt = bool(summaries)
        if isinstance(pack, Mapping):
            has_evidence_attempt = has_evidence_attempt or bool(
                pack.get("enabled")
                and (
                    (_int_value(pack.get("total_tasks")) or 0) > 0
                    or (_int_value(pack.get("completed_tasks")) or 0) > 0
                    or pack.get("diagnostics")
                )
            )
        if not has_evidence_attempt:
            return ""
        return (
            "Внешние источники не подтвердили совместимость выбранных связок; "
            "инженерная проверка обязательна."
        )
    return (
        "Проверка: найдены внешние источники по выбранным конфигурациям; "
        "финальная проверка инженером обязательна."
    )


def _telegram_recommendation_evidence_line(candidate: Mapping[str, Any]) -> str:
    summary = candidate.get("evidence_summary")
    if not isinstance(summary, Mapping):
        return ""
    if (_int_value(summary.get("sources_count")) or 0) <= 0:
        return ""
    missing = [
        *as_string_list(summary.get("missing")),
        *as_string_list(summary.get("not_confirmed")),
    ]
    fatal_concerns = as_string_list(summary.get("fatal_concerns"))
    status = str(summary.get("status") or "").strip()
    if status == "mismatch":
        if "platform_cpu" in _summary_relation_types(summary):
            return (
                "Доказательная проверка: источники не подтвердили совместимость "
                "CPU с платформой."
            )
        return "Доказательная проверка: источники нашли конфликт совместимости."
    if missing or fatal_concerns:
        if status == "partially_confirmed":
            confirmed = as_string_list(summary.get("confirmed")) or as_string_list(
                summary.get("confirmed_facts")
            )
            if confirmed:
                facts = ", ".join(_short_evidence_fact(fact) for fact in confirmed[:3])
                return (
                    f"Доказательная проверка: подтверждены {facts}; "
                    "support list CPU/RAM нужно сверить инженеру."
                )
        return "Доказательная проверка: часть совместимости не подтверждена источниками."
    confirmed = as_string_list(summary.get("confirmed")) or as_string_list(
        summary.get("confirmed_facts")
    )
    if confirmed:
        facts = ", ".join(_short_evidence_fact(fact) for fact in confirmed[:3])
        return f"Доказательная проверка: подтверждено {facts} по найденным источникам."
    if status == "confirmed":
        return "Доказательная проверка: совместимость подтверждена найденными источниками."
    return "Доказательная проверка: явных конфликтов по найденным источникам не выявлено."


def _short_evidence_fact(value: Any) -> str:
    text = _clean_user_text(value)
    text = re.sub(r"^(тип памяти|socket|интерфейс накопителя|NVMe|емкость):\s*", "", text)
    return make_telegram_sentence(text, max_chars=40, fallback=text[:40]).rstrip(".")


def _summary_relation_types(summary: Mapping[str, Any]) -> set[str]:
    relations = summary.get("relation_evidence")
    if not isinstance(relations, list):
        return set()
    return {
        relation_type
        for relation in relations
        if isinstance(relation, Mapping)
        if (relation_type := str(relation.get("relation_type") or "").strip())
    }


def _is_no_safe_ai_reason(summary: Mapping[str, Any]) -> bool:
    if summary.get("primary_recommendation_status") == "no_recommendation":
        return True
    valid_proposals = _int_value(summary.get("valid_proposals_count")) or 0
    validation_summary = summary.get("ai_validation_summary")
    if valid_proposals == 0 and isinstance(validation_summary, Mapping):
        valid_proposals = _int_value(validation_summary.get("accepted_after_validation")) or 0
    recommendations = _int_value(summary.get("ai_recommendations_count")) or 0
    if valid_proposals > 0 or recommendations > 0:
        return False
    reason = str(summary.get("llm_fallback_reason") or "").strip()
    mode = str(summary.get("ai_recommendation_mode") or "").strip()
    return mode == "ai_no_safe_recommendations" or reason in {
        "llm_configurator_all_recommendations_rejected",
        "llm_configurator_no_valid_recommendations",
        "llm_configurator_no_complete_recommendation",
        "composer_structured_no_recommendation",
        "composer_no_safe_complete_bom",
    }


def _safe_no_recommendation_message(
    match_run_id: Any,
    *,
    summary: Mapping[str, Any] | None = None,
) -> str:
    reason = summary.get("no_recommendation_reason") if isinstance(summary, Mapping) else {}
    reason_mapping = reason if isinstance(reason, Mapping) else {}
    lines = [
        f"AI-подбор по складу №{match_run_id}",
        "Безопасную складскую рекомендацию дать нельзя.",
    ]
    summary_text = sanitize_user_facing_text(reason_mapping.get("summary"))
    if (
        reason_mapping.get("reason_code") == "complex_request_requires_llm_semantic_planner"
        and summary_text
    ):
        lines.extend(["", summary_text])
    missing_roles = as_string_list(reason_mapping.get("missing_roles"))
    missing_required_capabilities = _candidate_mappings(
        reason_mapping.get("missing_required_capabilities")
    )
    partial_available_components = _candidate_mappings(
        reason_mapping.get("partial_available_components")
    )
    failed_requirements = _reason_items(reason_mapping.get("failed_requirements"))
    role_failures = _candidate_mappings(reason_mapping.get("role_failures"))
    unverified_requirements = _reason_items(
        reason_mapping.get("unverified_requirements")
    )
    hard_mismatch_risks = _reason_items(reason_mapping.get("hard_mismatch_risks"))
    recommended_next_actions = as_string_list(
        reason_mapping.get("recommended_next_actions")
    )
    normalized_engineer_checks = as_string_list(
        reason_mapping.get("engineer_checks")
        or reason_mapping.get("engineering_checks")
    )
    stock_shortages = _candidate_mappings(reason_mapping.get("stock_shortages"))
    hard_incompatibility = as_string_list(reason_mapping.get("hard_incompatibility"))
    product_group = str(
        reason_mapping.get("product_group")
        or (summary or {}).get("product_group")
        or "server"
    ).strip()
    manual_checks = sanitize_engineer_checks_for_product_group(
        normalized_engineer_checks or as_string_list(reason_mapping.get("manual_checks")),
        product_group=product_group,
    )
    if partial_available_components:
        lines.extend(["", "Что можно закрыть со склада:"])
        lines.extend(
            _reason_bullet_line(component)
            for component in partial_available_components[:5]
        )
    if failed_requirements:
        lines.extend(["", "Что не закрывается:"])
        lines.extend(_reason_bullet_line(item) for item in failed_requirements[:6])
    elif missing_required_capabilities:
        lines.extend(["", "Не закрыты требования:"])
        for capability in missing_required_capabilities[:5]:
            source_text = sanitize_user_facing_text(
                capability.get("source_text")
                or capability.get("requirement_text")
                or capability.get("capability_id")
                or capability.get("role")
            )
            reason_text = sanitize_user_facing_text(
                capability.get("user_message")
                or capability.get("reason")
                or capability.get("status")
            )
            if source_text:
                lines.append(f"- {source_text}")
            if reason_text:
                lines.append(f"  Причина: {reason_text}")
    if role_failures:
        lines.extend(["", "Проблемы по ролям:"])
        lines.extend(_role_failure_line(item) for item in role_failures[:6])
    elif missing_roles:
        lines.extend(
            ["", "Не хватает ролей:", *[f"- {_bom_role_label(role)}" for role in missing_roles]]
        )
    if stock_shortages:
        lines.append("")
        lines.append("Недостаточно склада:")
        for shortage in stock_shortages[:5]:
            role = _bom_role_label(str(shortage.get("role") or ""))
            required = shortage.get("required_quantity")
            available = shortage.get("available_quantity")
            lines.append(f"- {role}: нужно {required}, склад {available}")
    if hard_mismatch_risks:
        lines.extend(["", "Почему нельзя показать BOM как КП-ready:"])
        lines.extend(_reason_bullet_line(item) for item in hard_mismatch_risks[:6])
    elif hard_incompatibility:
        lines.extend(
            [
                "",
                "Жесткая несовместимость:",
                *[f"- {item}" for item in hard_incompatibility[:5]],
            ]
        )
    if unverified_requirements:
        lines.extend(["", "Что осталось неподтвержденным:"])
        lines.extend(
            _reason_bullet_line(item) for item in unverified_requirements[:5]
        )
    if recommended_next_actions:
        lines.extend(["", "Что нужно дозакупить / проверить:"])
        lines.extend(
            f"- {sanitize_user_facing_text(item)}"
            for item in recommended_next_actions[:5]
        )
    if manual_checks:
        lines.extend(
            [
                "",
                "Что проверить вручную:",
                *[f"- {check}" for check in manual_checks[:5]],
            ]
        )
    coverage_line = _no_recommendation_coverage_line(reason_mapping)
    if coverage_line:
        lines.extend(["", coverage_line])
    lines.extend(["", "Подробная матрица компонентов отправлена Excel-файлом."])
    return "\n".join(lines)


def _reason_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, Iterable) or isinstance(value, str):
        text = sanitize_user_facing_text(value)
        return [text] if text else []
    result: list[Any] = []
    for item in value:
        if isinstance(item, Mapping):
            result.append(item)
            continue
        text = sanitize_user_facing_text(item)
        if text:
            result.append(text)
    return result


def _reason_bullet_line(item: Any) -> str:
    if not isinstance(item, Mapping):
        return f"- {sanitize_user_facing_text(item)}"
    role = sanitize_user_facing_text(item.get("role") or item.get("component_role"))
    text = sanitize_user_facing_text(
        item.get("source_text")
        or item.get("requirement_text")
        or item.get("requirement")
        or item.get("message")
        or item.get("reason")
        or item.get("user_message")
        or item.get("status")
        or item.get("type")
    )
    action = sanitize_user_facing_text(
        item.get("suggested_action") or item.get("recommended_action")
    )
    parts = [part for part in (role, text, action) if part]
    return "- " + ": ".join(parts) if parts else "- требуется проверка"


def _role_failure_line(item: Mapping[str, Any]) -> str:
    base = _reason_bullet_line(item)
    coverage = item.get("candidate_coverage")
    if not isinstance(coverage, Mapping):
        return base
    considered = coverage.get("considered_count")
    total = coverage.get("candidate_count")
    if considered is None and isinstance(coverage.get("considered_candidate_ids"), list):
        considered = len(coverage["considered_candidate_ids"])
    if considered is None and total is None:
        return base
    if total is None:
        return f"{base} (рассмотрено кандидатов: {considered})"
    return f"{base} (coverage: {considered}/{total})"


def _no_recommendation_coverage_line(reason: Mapping[str, Any]) -> str:
    coverage = reason.get("no_recommendation_coverage")
    if isinstance(coverage, Mapping):
        percent_by_role = coverage.get("coverage_percent_by_role")
        if isinstance(percent_by_role, Mapping) and percent_by_role:
            parts = [
                f"{role}: {round(float(value), 1)}%"
                for role, value in list(percent_by_role.items())[:4]
                if isinstance(value, int | float)
            ]
            if parts:
                return "Coverage по матрице: " + ", ".join(parts) + "."
    role_coverage = reason.get("role_evaluation_coverage_by_role")
    if isinstance(role_coverage, Mapping) and role_coverage:
        covered = 0
        total = 0
        for value in role_coverage.values():
            if not isinstance(value, Mapping):
                continue
            total += 1
            if value.get("all_candidates_considered") is True:
                covered += 1
        if total:
            return f"Coverage role evaluation: {covered}/{total} ролей полностью рассмотрены."
    return ""


def _safe_ai_unavailable_message(match_run_id: Any) -> str:
    return "\n".join(
        [
            f"AI-подбор по складу №{match_run_id}",
            "Расширенный AI-анализ сейчас недоступен: LLM не вернула безопасный ответ.",
            "Подробная матрица компонентов отправлена Excel-файлом для инженерной проверки.",
            "Перед КП нужна инженерная проверка.",
        ]
    )


def _output_gate(text: str, *, match_run_id: Any) -> str:
    forbidden_exact_lines = {"Готовые варианты", "Сборка из комплектующих"}
    forbidden_fragments = (
        "component_candidate_id",
        "fit_label",
        "raw_json",
        "raw JSON",
        '"recommendations"',
        "preliminary_requires_engineer_review",
        "Minimal cost",
        "Proven Supermicro",
        "Premium platform",
        "на заказ",
        "overfit",
        "cores",
        "за 2 сервера за весь запрос",
        "fatal ",
        "fatal:",
        "fatal compatibility",
        "fatal socket",
        "Llm_rec_",
        "llm_rec_",
        "web evidence not found",
        "keep engineer",
    )
    lines = [line.strip() for line in text.splitlines()]
    lowered = text.casefold()
    if any(line in forbidden_exact_lines for line in lines):
        return _safe_no_recommendation_message(match_run_id)
    if any(fragment.casefold() in lowered for fragment in forbidden_fragments):
        return _safe_no_recommendation_message(match_run_id)
    return text


def choose_excel_report_delivery(
    report_xlsx: bytes,
    *,
    match_run_id: int,
) -> ReportDelivery:
    return ReportDelivery(
        mode="file",
        filename=f"stock_match_{match_run_id}.xlsx",
        content=report_xlsx,
    )


def choose_report_delivery(
    report_markdown: str,
    *,
    match_run_id: int,
    safe_limit: int = TELEGRAM_SAFE_MESSAGE_LIMIT,
) -> ReportDelivery:
    filename = f"report_{match_run_id}.md"
    return ReportDelivery(
        mode="file",
        filename=filename,
        content=report_markdown.encode("utf-8"),
    )


def _summary_checks(summary: Mapping[str, Any]) -> list[str]:
    checks = humanized_checks(
        risk_flags=as_string_list(summary.get("risk_flags")),
        missing_requirements=as_string_list(summary.get("missing_requirements")),
    )
    return checks


def _found_ai_recommendations_text(count: int) -> str:
    action = pluralize_ru(count, "Найдена", "Найдены", "Найдено")
    noun = pluralize_ru(count, "AI-рекомендация", "AI-рекомендации", "AI-рекомендаций")
    return f"{action} {count} {noun}"


def _found_variants_text(count: int) -> str:
    action = pluralize_ru(count, "Найден", "Найдены", "Найдено")
    noun = pluralize_ru(count, "вариант", "варианта", "вариантов")
    return f"{action} {count} {noun}"


def _shown_safe_variants_message(count: int, summary: Mapping[str, Any] | None = None) -> str:
    action = pluralize_ru(count, "Показан", "Показаны", "Показано")
    noun = pluralize_ru(
        count,
        "безопасный вариант",
        "безопасных варианта",
        "безопасных вариантов",
    )
    base = f"{action} {count} {noun}."
    if not summary:
        return (
            f"{base} Остальные AI-варианты были отклонены валидатором "
            "или уступили выбранному по цене/рискам."
        )
    rejected = _int_value(summary.get("rejected_ai_recommendations_count")) or 0
    proposals = _int_value(summary.get("llm_proposals_count")) or count + rejected
    if rejected <= 0 or proposals <= count:
        return base
    return (
        f"{base} Остальные AI-варианты были отклонены валидатором "
        "или уступили по цене/рискам."
    )


def _short_rejection_mix(summary: Mapping[str, Any]) -> str:
    validation_summary = summary.get("ai_validation_summary")
    if not isinstance(validation_summary, Mapping):
        return "часть из-за рисков совместимости, часть из-за неполной комплектации или худшей цены"
    compatibility = (
        (_int_value(validation_summary.get("rejected_fatal")) or 0)
        + (_int_value(validation_summary.get("rejected_role_mismatch")) or 0)
    )
    incomplete = (
        (_int_value(validation_summary.get("rejected_missing_required")) or 0)
        + (_int_value(validation_summary.get("rejected_stock")) or 0)
    )
    duplicate_or_price = _selection_skipped_count(validation_summary)
    parts: list[str] = []
    if compatibility:
        parts.append("часть из-за рисков совместимости")
    if incomplete:
        parts.append("часть из-за неполной комплектации")
    if duplicate_or_price:
        parts.append("часть как дубликаты или хуже по цене")
    if not parts:
        return "часть из-за рисков совместимости, часть из-за неполной комплектации или худшей цены"
    return ", ".join(parts)


def _ai_variant_plural(count: int) -> str:
    return pluralize_ru(count, "AI-вариант", "AI-варианта", "AI-вариантов")


def _selection_skipped_count(validation_summary: Mapping[str, Any]) -> int:
    explicit = _int_value(validation_summary.get("selection_skipped_count"))
    if explicit is not None:
        return explicit
    return (
        (_int_value(validation_summary.get("selection_skipped_duplicate")) or 0)
        + (
            _int_value(
                validation_summary.get("selection_skipped_dominated_by_cheaper_equivalent")
            )
            or 0
        )
        + (_int_value(validation_summary.get("selection_skipped_worse_by_price")) or 0)
        + (
            _int_value(
                validation_summary.get(
                    "selection_skipped_same_platform_without_meaningful_difference"
                )
            )
            or 0
        )
        + (
            _int_value(validation_summary.get("selection_skipped_lower_ranked_alternative"))
            or 0
        )
        + (_int_value(validation_summary.get("rejected_duplicate")) or 0)
        + (_int_value(validation_summary.get("rejected_right_size")) or 0)
        + (_int_value(validation_summary.get("rejected_other")) or 0)
    )


def _llm_fallback_notice(summary: Mapping[str, Any]) -> str:
    if summary.get("llm_configurator_enabled") is False:
        return "Расширенный AI-анализ выключен."
    if summary.get("llm_configurator_used") is not False:
        return ""
    reason = str(summary.get("llm_fallback_reason") or "").strip()
    if reason in {
        "llm_configurator_all_recommendations_rejected",
        "llm_configurator_no_valid_recommendations",
    }:
        return (
            "Безопасную складскую рекомендацию дать нельзя. "
            "Проверьте матрицу компонентов вручную."
        )
    return "Расширенный AI-анализ недоступен, показан предварительный rule-based подбор."


def _llm_summary_checks(
    summary: Mapping[str, Any],
    llm_build_candidates: list[Mapping[str, Any]],
) -> list[str]:
    requirements = _first_normalized_requirements(summary)
    checks: list[str] = []
    for candidate in llm_build_candidates:
        evidence_summary = candidate.get("evidence_summary")
        evidence_checks = (
            as_string_list(evidence_summary.get("engineering_checks"))
            if isinstance(evidence_summary, Mapping)
            else []
        )
        raw_risk_flags = [
            *as_string_list(candidate.get("critical_checks")),
            *as_string_list(candidate.get("critical_risks")),
            *as_string_list(candidate.get("compatibility_warnings")),
            *evidence_checks,
        ]
        candidate_checks = humanized_checks(
            risk_flags=[
                value for value in raw_risk_flags if not _is_internal_evidence_debug_check(value)
            ],
            missing_requirements=[
                *as_string_list(candidate.get("missing_components")),
                *as_string_list(candidate.get("missing_requirements")),
            ],
        )
        for check in candidate_checks:
            if check and check not in checks:
                checks.append(check)
    clean_checks: list[str] = []
    for check in _filter_contradictory_checks(checks, requirements):
        short_check = make_telegram_sentence(check, max_chars=180, fallback="")
        if short_check and short_check not in clean_checks:
            clean_checks.append(short_check)
    return clean_checks[:MAX_SUMMARY_CHECKS]


def _is_internal_evidence_debug_check(value: str) -> bool:
    lowered = value.casefold()
    return (
        "web evidence not found" in lowered
        or "llm_rec_" in lowered
        or "keep engineer" in lowered
    )


def _ai_recommendation_mappings(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for field_name in (
        "ai_recommendations",
        "llm_recommendations",
        "llm_recommended_build_candidates",
    ):
        recommendations = _candidate_mappings(summary.get(field_name))
        if recommendations:
            return recommendations
    return []


def _configuration_group_mappings(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _candidate_mappings(summary.get("configuration_groups"))


def _primary_recommendation_mapping(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    primary = summary.get("primary_recommendation")
    return primary if isinstance(primary, Mapping) else {}


def _summary_candidates(
    summary: Mapping[str, Any],
    *,
    field_name: str,
    fallback_candidate_type: str,
) -> list[Mapping[str, Any]]:
    value = summary.get(field_name)
    if value is None:
        value = [
            candidate
            for candidate in _iter_candidate_mappings(summary.get("candidates"))
            if _candidate_type(candidate) == fallback_candidate_type
        ]
    return _first_candidate_mappings(value)


def _first_candidate_mappings(value: Any) -> list[Mapping[str, Any]]:
    return _candidate_mappings(value)[:MAX_SUMMARY_CANDIDATES]


def _candidate_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, str):
        return []

    candidates: list[Mapping[str, Any]] = []
    for candidate in value:
        if isinstance(candidate, Mapping):
            candidates.append(candidate)
    return candidates


def _iter_candidate_mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, str):
        return []
    return [candidate for candidate in value if isinstance(candidate, Mapping)]


def _candidate_type(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("candidate_type")
    if isinstance(value, str) and value:
        return value
    return "ready_server"


def _candidate_lines(candidates: list[Mapping[str, Any]]) -> list[str]:
    if not candidates:
        return ["- не найдено"]

    return [
        (
            f"{index}. {candidate_display_name(candidate)} - "
            f"остаток: {_stock_text(candidate.get('available_quantity'))}; "
            f"цена: {format_price(candidate.get('price_value'), candidate.get('price_currency'))}"
        )
        for index, candidate in enumerate(candidates, start=1)
    ]


def _build_candidate_lines(
    candidates: list[Mapping[str, Any]],
) -> list[str]:
    if not candidates:
        return [
            "- пока не предложена - нет достаточных складских данных по платформам/комплектующим."
        ]

    lines: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        status = _build_status_text(candidate)
        platform = _component_article(candidate, "server_platform") or candidate_display_name(
            candidate
        )
        composition = _build_composition_text(candidate)
        total_price = _build_total_price_text(candidate)
        detail_lines = _build_component_detail_lines(candidate)
        lines.extend(
            [
                f"{index}. {status} {platform}",
                f"   Состав: {composition}",
                *detail_lines,
                f"   {_price_line_label(candidate)}: {total_price}",
                "   Нужна инженерная проверка",
            ]
        )
    return lines


def _llm_recommendation_lines(candidates: list[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        if lines:
            lines.append("")
        title = humanized_recommendation_title(
            candidate,
            candidates,
            index=index,
            default_title=_default_recommendation_title(candidate),
        )
        lines.append(
            f"{index}. "
            f"{make_telegram_sentence(title, max_chars=120, fallback=_clean_user_text(title))}"
        )
        if _recommendation_source_type(candidate) == READY_SERVER_CANDIDATE_TYPE:
            lines.extend(_ready_server_recommendation_lines(candidate))
        else:
            lines.extend(_build_recommendation_lines(candidate))
    return lines


def _ready_server_recommendation_lines(candidate: Mapping[str, Any]) -> list[str]:
    total_price = _build_total_price_text(candidate)
    why_selected = _why_selected_short_text(candidate)
    right_size_note = _right_size_note_text(candidate)
    article = _ready_server_article(candidate)
    quantity = _server_quantity(candidate)
    available = candidate.get("available_quantity")
    checks = _ready_server_checks(candidate)

    lines = [
        "   Тип: готовый сервер",
        f"   Готовая позиция - {_quantity_units(quantity)}",
    ]
    if article:
        lines.append(f"   Артикул: {article}")
    lines.append(f"   Остаток: {_stock_count_text(available)}")
    lines.append("")
    lines.append(f"   Цена за весь запрос: {total_price}")
    if right_size_note:
        lines.append(f"   {right_size_note}")
    confidence_line = _displayed_confidence_line(candidate)
    if confidence_line:
        lines.append(f"   {confidence_line}")
    if why_selected:
        lines.append(f"   Почему выбрана: {why_selected}")
    evidence_line = _telegram_recommendation_evidence_line(candidate)
    if evidence_line:
        lines.append(f"   {evidence_line}")
    if checks:
        lines.append(f"   Что проверить: {checks[0]}")
    lines.append("   Нужна инженерная проверка")
    return lines


def _build_recommendation_lines(candidate: Mapping[str, Any]) -> list[str]:
    source_type = _recommendation_source_type(candidate)
    total_price = _build_total_price_text(candidate)
    why_selected = _why_selected_short_text(candidate)
    right_size_note = _right_size_note_text(candidate)
    server_quantity = _server_quantity(candidate)
    lines = [
        f"   Тип: {_recommendation_type_label(candidate)}",
    ]
    if source_type == PARTIAL_BUILD_CANDIDATE_TYPE:
        lines.append(f"   {_partial_build_label(candidate)}")
    else:
        lines.append(f"   Сборный сервер - {_quantity_units(server_quantity)}")

    per_server = _bom_per_server_lines(candidate)
    if per_server:
        lines.extend(["", "   В составе 1 сервера:"])
        lines.extend(f"   - {line}" for line in per_server)

    totals = _bom_total_lines(candidate)
    if totals:
        lines.extend(["", "   Всего к подбору:"])
        lines.extend(f"   - {line}" for line in totals)

    optional_lines = _optional_component_lines(candidate)
    if optional_lines:
        lines.extend(["", "   Опционально / проверить инженеру:"])
        lines.extend(f"   - {line}" for line in optional_lines)

    lines.append("")
    lines.append(f"   {_price_line_label(candidate)}: {total_price}")
    if right_size_note:
        lines.append(f"   {right_size_note}")
    confidence_line = _displayed_confidence_line(candidate)
    if confidence_line:
        lines.append(f"   {confidence_line}")
    if why_selected:
        lines.append(f"   Почему выбрана: {why_selected}")
    evidence_line = _telegram_recommendation_evidence_line(candidate)
    if evidence_line:
        lines.append(f"   {evidence_line}")
    lines.append("   Нужна инженерная проверка")
    return lines


def _why_selected_text(candidate: Mapping[str, Any]) -> str:
    text = str(candidate.get("why_selected") or "").strip()
    if text:
        return _clean_user_text(text)
    reasons = as_string_list(candidate.get("rank_reason"))
    return _clean_user_text(reasons[0]) if reasons else ""


def _why_selected_short_text(candidate: Mapping[str, Any]) -> str:
    short = _clean_user_text(candidate.get("why_selected_short"))
    if short:
        return make_telegram_sentence(short, max_chars=180)
    return make_telegram_sentence(_why_selected_text(candidate), max_chars=180)


def make_telegram_sentence(
    text: Any,
    max_chars: int = 180,
    fallback: str = TELEGRAM_TEXT_FALLBACK,
) -> str:
    text = _clean_user_text(text)
    if not text:
        return fallback
    if len(text) <= max_chars and _is_meaningful_phrase(text):
        return text
    first_sentence_end = _first_sentence_end(text)
    source = text[:first_sentence_end].strip() if first_sentence_end is not None else text
    if len(source) <= max_chars and _is_meaningful_phrase(source):
        return source
    if len(source) > max_chars:
        shortened = _truncate_meaningful_sentence(source, limit=max_chars)
        if shortened and _is_meaningful_phrase(shortened):
            return shortened
    sentence_end = max(
        text.rfind(".", 0, max_chars),
        text.rfind("!", 0, max_chars),
        text.rfind("?", 0, max_chars),
    )
    if sentence_end >= 20:
        candidate = text[: sentence_end + 1].strip()
        if _is_meaningful_phrase(candidate):
            return candidate
    shortened = _truncate_meaningful_sentence(text, limit=max_chars)
    if shortened and _is_meaningful_phrase(shortened):
        return shortened
    return fallback


def _truncate_meaningful_sentence(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text if _is_meaningful_phrase(text) else ""
    if limit <= 4:
        return ""
    cutoff = max(1, limit - 3)
    candidate = text[:cutoff].rstrip(" .,;:-")
    word_boundary = max(
        candidate.rfind(" "),
        candidate.rfind(","),
        candidate.rfind(";"),
        candidate.rfind(":"),
    )
    if word_boundary < 20:
        return ""
    candidate = candidate[:word_boundary].rstrip(" .,;:-")
    candidate = _strip_weak_trailing_words(candidate)
    if not _is_meaningful_phrase(candidate):
        return ""
    return f"{candidate}..."


def _strip_weak_trailing_words(text: str) -> str:
    weak_words = {
        "в",
        "во",
        "на",
        "с",
        "со",
        "к",
        "ко",
        "по",
        "при",
        "для",
        "без",
        "из",
        "у",
        "о",
        "об",
        "от",
        "до",
        "и",
        "или",
        "а",
        "но",
        "что",
        "если",
        "как",
        "чем",
        "более",
        "менее",
        "достаточном",
        "достаточной",
        "достаточных",
        "достаточным",
    }
    candidate = text.strip()
    while candidate:
        last_word = re.search(r"([\wА-Яа-яЁё-]+)$", candidate)
        if last_word is None or last_word.group(1).casefold() not in weak_words:
            break
        candidate = candidate[: last_word.start()].rstrip(" .,;:-")
    return candidate


def _is_meaningful_phrase(text: str) -> bool:
    stripped = text.strip(" .,!?:;-'\"…")
    if len(stripped) < 3:
        return False
    return bool(re.search(r"[A-Za-zА-Яа-яЁё0-9]{3,}", stripped))


def _first_sentence_end(text: str) -> int | None:
    match = next(re.finditer(r"[.!?](?:\s|$)", text), None)
    if match is None:
        return None
    return match.start() + 1


def _llm_build_status_text(candidate: Mapping[str, Any]) -> str:
    status = str(candidate.get("completeness_status") or "").strip()
    missing_roles = as_string_list(candidate.get("missing_component_roles"))
    if status == "incomplete" or any(role in missing_roles for role in ("cpu", "ram", "ssd")):
        return "Частичная сборка "
    return ""


def _default_recommendation_title(candidate: Mapping[str, Any]) -> str:
    source_type = _recommendation_source_type(candidate)
    if source_type == READY_SERVER_CANDIDATE_TYPE:
        return "Готовый складской вариант"
    if source_type == PARTIAL_BUILD_CANDIDATE_TYPE:
        return "Частичная сборка"
    return "Сборка из комплектующих"


READY_SERVER_CANDIDATE_TYPE = "ready_server"
BUILD_CANDIDATE_TYPE = "build_from_parts"
PARTIAL_BUILD_CANDIDATE_TYPE = "partial_build"


def _recommendation_source_type(candidate: Mapping[str, Any]) -> str:
    source_type = str(candidate.get("source_type") or candidate.get("candidate_type") or "")
    if source_type == BUILD_CANDIDATE_TYPE:
        missing = as_string_list(candidate.get("missing_component_roles"))
        if candidate.get("completeness_status") == "incomplete" or missing:
            return PARTIAL_BUILD_CANDIDATE_TYPE
    if source_type in {
        READY_SERVER_CANDIDATE_TYPE,
        BUILD_CANDIDATE_TYPE,
        PARTIAL_BUILD_CANDIDATE_TYPE,
    }:
        return source_type
    return BUILD_CANDIDATE_TYPE


def _recommendation_type_label(candidate: Mapping[str, Any]) -> str:
    labels = {
        READY_SERVER_CANDIDATE_TYPE: "готовый сервер",
        BUILD_CANDIDATE_TYPE: "сборка из комплектующих",
        PARTIAL_BUILD_CANDIDATE_TYPE: "частичная сборка",
    }
    return labels[_recommendation_source_type(candidate)]


def _recommendation_display_name(candidate: Mapping[str, Any]) -> str:
    source_type = _recommendation_source_type(candidate)
    if source_type == READY_SERVER_CANDIDATE_TYPE:
        summary = candidate.get("component_summary")
        if isinstance(summary, Mapping):
            platform = str(summary.get("platform") or "").strip()
            if platform:
                return platform
        display = candidate_display_name(candidate)
        if display != "артикул не указан":
            return display
        return str(candidate.get("display_name") or "").strip()

    platform = _component_article(candidate, "server_platform")
    if platform:
        return platform
    display = candidate_display_name(candidate)
    return "" if display == "артикул не указан" else display


def _recommendation_component_detail_lines(candidate: Mapping[str, Any]) -> list[str]:
    component_lines = _build_component_detail_lines(candidate, include_storage=True)
    if component_lines:
        return component_lines

    summary = candidate.get("component_summary")
    if isinstance(summary, Mapping):
        rows: list[str] = []
        for label, key in (("CPU", "cpu"), ("RAM", "ram"), ("SSD/HDD", "storage")):
            value = str(summary.get(key) or "").strip()
            if value:
                rows.append(f"   {label}: {_brief_component_text(value, limit=130)}")
        return rows
    return []


def _right_size_note_text(candidate: Mapping[str, Any]) -> str:
    note = _clean_user_text(candidate.get("right_size_note"))
    fallback = "Подбор: выбрано по сочетанию цены, наличия и соответствия требованиям."
    if note:
        if note.startswith("Компромисс:"):
            note = "Подбор:" + note.removeprefix("Компромисс:")
        if not note.startswith("Подбор:"):
            note = f"Подбор: {note}"
        return make_telegram_sentence(note, max_chars=220, fallback=fallback)

    overfit_reason = _clean_user_text(candidate.get("overfit_reason"))
    if overfit_reason:
        return make_telegram_sentence(
            f"Подбор: компонент выше требования. {overfit_reason}",
            max_chars=220,
            fallback=fallback,
        )
    return "Подбор: минимально подходящий по требованиям"


def _component_article(candidate: Mapping[str, Any], role: str) -> str:
    component = _component_by_role(candidate, role)
    if component is None:
        return ""

    display_parts = [
        str(component.get("producer") or "").strip(),
        str(component.get("part_number") or "").strip(),
    ]
    display = " ".join(part for part in display_parts if part)
    if not display:
        display = str(component.get("item_name") or component.get("item_id") or "").strip()
    return display


def _component_by_role(candidate: Mapping[str, Any], role: str) -> Mapping[str, Any] | None:
    components = candidate.get("components")
    if not isinstance(components, Iterable) or isinstance(components, str):
        return None

    for component in components:
        if not isinstance(component, Mapping) or component.get("role") != role:
            continue
        return component
    return None


def _build_component_detail_lines(
    candidate: Mapping[str, Any],
    *,
    include_storage: bool = False,
) -> list[str]:
    lines: list[str] = []
    cpu_line = _cpu_order_line(candidate)
    if cpu_line:
        lines.append(f"   CPU: {cpu_line}")
    ram_line = _ram_order_line(candidate)
    if ram_line:
        lines.append(f"   RAM: {ram_line}")
    if include_storage:
        storage_line = _storage_order_line(candidate)
        if storage_line:
            lines.append(f"   SSD/HDD: {storage_line}")
    return lines


def _bom_per_server_lines(candidate: Mapping[str, Any]) -> list[str]:
    server_quantity = _server_quantity(candidate)
    lines: list[str] = []
    for role in ("server_platform", "cpu", "ram", "ssd", "hdd"):
        component = _component_by_role(candidate, role)
        if component is None:
            continue
        line = _bom_per_server_line(component, server_quantity)
        if line:
            lines.append(line)
    return lines


def _bom_total_lines(candidate: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for role in ("server_platform", "cpu", "ram", "ssd", "hdd"):
        component = _component_by_role(candidate, role)
        if component is None:
            continue
        label = _bom_role_label(role)
        quantity = _int_value(component.get("quantity_required"))
        quantity_text = _quantity_units(quantity)
        stock_text = _stock_count_text(component.get("available_quantity"))
        if role == "ram":
            module_gb = _ram_module_capacity_gb(component)
            if module_gb is not None and quantity is not None:
                lines.append(
                    f"{label}: {quantity_text} всего, модули по {module_gb} ГБ, "
                    f"склад: {stock_text}"
                )
                continue
        lines.append(f"{label}: {quantity_text} всего, склад: {stock_text}")
    return lines


def _optional_component_lines(candidate: Mapping[str, Any]) -> list[str]:
    components = candidate.get("optional_components")
    if not isinstance(components, Iterable) or isinstance(components, str):
        return []
    lines: list[str] = []
    for component in components:
        if not isinstance(component, Mapping):
            continue
        label = _bom_role_label(str(component.get("role") or ""))
        article = _component_article_from_component(component)
        quantity = _int_value(component.get("quantity_required"))
        stock_text = _stock_count_text(component.get("available_quantity"))
        if article:
            lines.append(
                f"{label}: {article} - {_quantity_units(quantity)} всего, "
                f"склад: {stock_text}"
            )
    optional_total = _format_amount(candidate.get("optional_total_price_value"))
    optional_currency = str(candidate.get("optional_total_price_currency") or "").strip()
    if optional_total:
        lines.append(
            "Опциональная сумма отдельно: "
            + " ".join(part for part in [optional_total, optional_currency] if part)
        )
    return lines


def _bom_per_server_line(
    component: Mapping[str, Any],
    server_quantity: int | None,
) -> str:
    role = str(component.get("role") or "").strip()
    label = _bom_role_label(role)
    article = _component_article_from_component(component)
    if not article:
        return ""
    total_quantity = _int_value(component.get("quantity_required"))
    per_server_quantity = _int_value(component.get("per_server_quantity"))
    if per_server_quantity is None:
        per_server_quantity = _per_server_quantity(total_quantity, server_quantity, role)
    quantity_text = _quantity_units(per_server_quantity)
    suffix = ""
    if role == "ram":
        module_gb = _ram_module_capacity_gb(component)
        ram_total_gb = _ram_total_gb_per_server(component, per_server_quantity)
        if module_gb is not None and per_server_quantity is not None:
            total_text = (
                f" = {_format_number(ram_total_gb)} ГБ"
                if ram_total_gb is not None
                else ""
            )
            return (
                f"{label}: {article} - {per_server_quantity} x {module_gb} ГБ "
                f"на сервер{total_text}"
            )
    elif role in {"ssd", "hdd"}:
        capacity = _storage_capacity_tb(component)
        if capacity is not None:
            suffix = f" x {_format_number(capacity)} ТБ на сервер"
    return f"{label}: {article} - {quantity_text}{suffix}"


def _partial_build_label(candidate: Mapping[str, Any]) -> str:
    missing_roles = as_string_list(candidate.get("missing_component_roles"))
    if not missing_roles:
        missing_roles = as_string_list(candidate.get("missing_components"))
    labels = [
        _bom_role_label(role)
        for role in ("cpu", "ram", "ssd", "hdd")
        if role in missing_roles
    ]
    if not labels:
        for value in missing_roles:
            label = _missing_role_label_from_text(value)
            if label and label not in labels:
                labels.append(label)
    suffix = "/".join(labels) if labels else "обязательные компоненты"
    return f"Частичная сборка - не хватает {suffix}"


def _ready_server_article(candidate: Mapping[str, Any]) -> str:
    article = str(candidate.get("part_number") or "").strip()
    if article:
        return article
    display = candidate_display_name(candidate)
    if display != "артикул не указан":
        return display
    return str(candidate.get("display_name") or "").strip()


def _ready_server_checks(candidate: Mapping[str, Any]) -> list[str]:
    checks = humanized_checks(
        risk_flags=[
            *as_string_list(candidate.get("critical_checks")),
            *as_string_list(candidate.get("critical_risks")),
            *as_string_list(candidate.get("compatibility_warnings")),
            *as_string_list(candidate.get("risk_flags")),
        ],
        missing_requirements=[
            *as_string_list(candidate.get("what_is_missing")),
            *as_string_list(candidate.get("missing_components")),
            *as_string_list(candidate.get("missing_requirements")),
        ],
    )
    return [
        make_telegram_sentence(check, max_chars=160, fallback="")
        for check in checks
        if make_telegram_sentence(check, max_chars=160, fallback="")
    ]


def _component_article_from_component(component: Mapping[str, Any]) -> str:
    display_parts = [
        str(component.get("producer") or "").strip(),
        str(component.get("part_number") or "").strip(),
    ]
    display = " ".join(part for part in display_parts if part)
    if display:
        return display
    return str(component.get("item_name") or component.get("item_id") or "").strip()


def _component_fact_text(component: Mapping[str, Any], key: str) -> str:
    facts = component.get("facts")
    if isinstance(facts, Mapping):
        value = facts.get(key)
        if value not in (None, ""):
            return str(value)
    value = component.get(key)
    if value not in (None, ""):
        return str(value)
    return ""


def _unique_texts(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _bom_role_label(role: str) -> str:
    labels = {
        "server_platform": "Платформа",
        "switch": "Коммутатор",
        "router": "Маршрутизатор",
        "firewall": "Межсетевой экран",
        "access_point": "Точка доступа",
        "cpu": "CPU",
        "ram": "RAM",
        "ssd": "SSD",
        "hdd": "HDD",
        "storage_controller": "Контроллер",
        "network_adapter": "Сетевой адаптер",
        "transceiver": "Трансивер",
        "dac_cable": "DAC-кабель",
        "cable": "Кабель",
        "license": "Лицензия",
        "support": "Поддержка",
        "power_supply": "Блок питания",
        "stacking_module": "Модуль стекирования",
        "other_accessory": "Аксессуар",
    }
    return labels.get(role, role.upper() if role else "Компонент")


def _missing_role_label_from_text(value: str) -> str:
    lowered = value.casefold()
    if "cpu" in lowered or "процесс" in lowered:
        return "CPU"
    if "ram" in lowered or "памят" in lowered:
        return "RAM"
    if "ssd" in lowered:
        return "SSD"
    if "hdd" in lowered:
        return "HDD"
    return ""


def _per_server_quantity(
    total_quantity: int | None,
    server_quantity: int | None,
    role: str,
) -> int | None:
    if role == "server_platform":
        return 1
    if total_quantity is None:
        return None
    if server_quantity is not None and server_quantity > 0:
        return max(1, total_quantity // server_quantity)
    return total_quantity


def _quantity_units(value: int | None) -> str:
    if value is None:
        return "количество уточняется"
    return f"{value} шт."


def _stock_count_text(value: Any) -> str:
    if value in (None, ""):
        return "неизвестно"
    return str(value)


def _ram_total_gb_per_server(
    component: Mapping[str, Any],
    per_server_quantity: int | None,
) -> int | None:
    if per_server_quantity is None:
        return None
    explicit_total = _int_value(component.get("ram_total_gb_per_server"))
    if explicit_total is not None:
        return explicit_total
    module_gb = _ram_module_capacity_gb(component)
    if module_gb is None:
        return None
    return module_gb * per_server_quantity


def _ram_module_capacity_gb(component: Mapping[str, Any]) -> int | None:
    module_gb = _int_value(component.get("ram_module_capacity_gb"))
    facts = component.get("facts")
    if module_gb is None and isinstance(facts, Mapping):
        module_gb = _int_value(facts.get("ram_capacity_gb"))
    return module_gb


def _storage_capacity_tb(component: Mapping[str, Any]) -> Decimal | None:
    value = component.get("storage_capacity_tb")
    facts = component.get("facts")
    if value in (None, "") and isinstance(facts, Mapping):
        value = facts.get("storage_capacity_tb")
    if value in (None, ""):
        return None
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _format_number(value: int | Decimal) -> str:
    amount = value if isinstance(value, Decimal) else Decimal(value)
    if amount == amount.to_integral():
        return str(int(amount))
    return format(amount.normalize(), "f")


def _cpu_order_line(candidate: Mapping[str, Any]) -> str:
    component = _component_by_role(candidate, "cpu")
    if component is None:
        return ""
    display = _component_article(candidate, "cpu")
    quantity = component.get("quantity_required")
    quantity_text = f", всего к подбору: {quantity} шт." if quantity is not None else ""
    return f"{_brief_component_text(display)}{quantity_text}"


def _ram_order_line(candidate: Mapping[str, Any]) -> str:
    component = _component_by_role(candidate, "ram")
    if component is None:
        return ""
    quantity = _int_value(component.get("quantity_required"))
    facts = component.get("facts")
    capacity = None
    if isinstance(facts, Mapping):
        capacity = _int_value(facts.get("ram_capacity_gb"))
    if quantity is not None and capacity is not None:
        return f"{quantity} модулей, всего к подбору: {quantity * capacity} ГБ"
    if quantity is not None:
        return f"{quantity} модулей всего к подбору"
    return ""


def _storage_order_line(candidate: Mapping[str, Any]) -> str:
    component = _component_by_role(candidate, "ssd") or _component_by_role(candidate, "hdd")
    if component is None:
        return ""
    display = _component_article(candidate, str(component.get("role") or "ssd"))
    quantity = component.get("quantity_required")
    quantity_text = f", всего к подбору: {quantity} шт." if quantity is not None else ""
    return f"{_brief_component_text(display)}{quantity_text}"


def _brief_component_text(value: str, *, limit: int = 72) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return _truncate_at_word_boundary(text, limit=limit)


def _truncate_at_word_boundary(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "..."[:limit]
    cutoff = max(1, limit - 3)
    candidate = text[:cutoff].rstrip()
    word_boundary = max(candidate.rfind(" "), candidate.rfind(","), candidate.rfind(";"))
    if word_boundary >= 20:
        candidate = candidate[:word_boundary].rstrip()
    elif len(candidate) < len(text) and not text[len(candidate) : len(candidate) + 1].isspace():
        fallback_boundary = max(text.rfind(" ", 0, limit), text.rfind(",", 0, limit))
        if fallback_boundary >= 10:
            candidate = text[:fallback_boundary].rstrip()
        else:
            return text[:limit].rstrip(".,;: ")
    return candidate.rstrip(".,;:") + "..."


def _clean_user_text(value: Any) -> str:
    return sanitize_user_facing_text(value)


def _first_normalized_requirements(summary: Mapping[str, Any]) -> Mapping[str, Any]:
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


def _filter_contradictory_checks(
    checks: list[str],
    requirements: Mapping[str, Any],
) -> list[str]:
    result: list[str] = []
    for check in checks:
        if _is_contradictory_check(check, requirements):
            continue
        if check not in result:
            result.append(check)
    return result


def _is_contradictory_check(
    check: str,
    requirements: Mapping[str, Any],
) -> bool:
    lowered = check.casefold()
    ram_type = str(requirements.get("ram_type_preference") or "").strip()
    if ram_type and ram_type != "unknown" and "тип оперативной памяти не указан" in lowered:
        return True
    if requirements.get("ram_gb_per_server") not in (None, "", "unknown") and (
        "объем оперативной памяти не указан" in lowered
        or "требуемый объем оперативной памяти" in lowered
    ):
        return True
    storage_interface = str(requirements.get("storage_interface_preference") or "").strip()
    if (
        storage_interface
        and storage_interface != "unknown"
        and "интерфейс накопителя не указан" in lowered
    ):
        return True
    cpu_vendor = str(requirements.get("cpu_vendor_preference") or "").strip()
    return bool(cpu_vendor and cpu_vendor != "unknown" and (
        "vendor cpu не указан" in lowered
        or "производитель cpu не указан" in lowered
    ))


def _build_status_text(candidate: Mapping[str, Any]) -> str:
    status = str(candidate.get("completeness_status") or "").strip()
    if status == "incomplete":
        return "Неполная сборка"
    return "Предварительная сборка"


def _build_composition_text(candidate: Mapping[str, Any]) -> str:
    roles = as_string_list(candidate.get("included_component_roles"))
    if not roles:
        roles = _component_roles(candidate)

    ordered_roles = [
        "server_platform",
        "cpu",
        "ram",
        "ssd",
        "hdd",
        "storage_controller",
        "network_adapter",
    ]
    labels = [
        _composition_role_label(role)
        for role in ordered_roles
        if role in roles and _composition_role_label(role)
    ]
    if not labels:
        labels = ["компоненты не подобраны"]

    missing_roles = as_string_list(candidate.get("missing_component_roles"))
    suffixes: list[str] = []
    for role in ("cpu", "ram", "ssd", "hdd"):
        if role in missing_roles:
            suffixes.append(f"{_composition_role_label(role)} не подобраны")

    result = ", ".join(labels)
    if suffixes:
        result += "; " + "; ".join(suffixes)
    return result


def _component_roles(candidate: Mapping[str, Any]) -> list[str]:
    components = candidate.get("components")
    if not isinstance(components, Iterable) or isinstance(components, str):
        return []

    roles: list[str] = []
    for component in components:
        if not isinstance(component, Mapping):
            continue
        role = str(component.get("role") or "").strip()
        if role and role not in roles:
            roles.append(role)
    return roles


def _composition_role_label(role: str) -> str:
    labels = {
        "server_platform": "платформа",
        "switch": "коммутаторы",
        "router": "маршрутизаторы",
        "firewall": "межсетевые экраны",
        "access_point": "точки доступа",
        "cpu": "CPU",
        "ram": "RAM",
        "ssd": "SSD",
        "hdd": "HDD",
        "storage_controller": "контроллеры",
        "network_adapter": "сетевые адаптеры",
        "transceiver": "трансиверы",
        "dac_cable": "DAC-кабели",
        "cable": "кабели",
        "license": "лицензии",
        "support": "поддержка",
        "power_supply": "блоки питания",
        "stacking_module": "модули стекирования",
        "other_accessory": "аксессуары",
    }
    return labels.get(role, "")


def _build_total_price_text(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("total_price_value") or candidate.get("price_value")
    currency = candidate.get("total_price_currency") or candidate.get("price_currency")
    formatted = _format_amount(value)
    if formatted is None:
        return "сумма не рассчитана"
    currency_text = str(currency or "").strip()
    return f"{formatted} {currency_text}".strip()


def _price_line_label(candidate: Mapping[str, Any]) -> str:
    quantity = _server_quantity(candidate)
    scope = "за весь запрос" if quantity is None else f"за {_server_quantity_text(quantity)}"

    note = _price_note_text(candidate)
    if note:
        scope = f"{scope} {note}"
    return f"Ориентировочно {scope}"


def _price_note_text(candidate: Mapping[str, Any]) -> str:
    note = str(candidate.get("price_note") or candidate.get("total_price_note") or "").strip()
    if not note:
        return ""
    lowered = note.casefold()
    if lowered.startswith("за "):
        return ""
    return note


def _displayed_confidence_line(candidate: Mapping[str, Any]) -> str:
    explicit = _clean_user_text(candidate.get("displayed_confidence"))
    if explicit:
        return explicit
    commercial = str(
        candidate.get("commercial_fit_confidence") or candidate.get("confidence") or ""
    ).strip()
    commercial_label = {
        "high": "высокое",
        "medium": "среднее",
        "low": "низкое",
    }.get(commercial, "")
    if not commercial_label:
        return ""
    evidence = candidate.get("evidence_summary")
    engineering_label = "предварительно, требуется проверка"
    if isinstance(evidence, Mapping):
        source_count = _int_value(evidence.get("sources_count")) or 0
        status = str(evidence.get("status") or "").strip()
        missing = [
            *as_string_list(evidence.get("missing")),
            *as_string_list(evidence.get("not_confirmed")),
        ]
        fatal = as_string_list(evidence.get("fatal_concerns"))
        if status == "mismatch" or fatal:
            engineering_label = "не подтверждено, требуется проверка"
        elif source_count > 0 and (status == "partially_confirmed" or missing):
            engineering_label = "частично проверено источниками, требуется проверка"
        elif source_count > 0 and status == "confirmed":
            engineering_label = "проверено по источникам, требуется финальная проверка"
    return (
        f"Коммерческое соответствие: {commercial_label}; "
        f"инженерная подтвержденность: {engineering_label}."
    )


def _format_amount(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value).replace(" ", ""))
    except (InvalidOperation, ValueError):
        return None

    if amount == amount.to_integral():
        return f"{int(amount):,}".replace(",", " ")
    return f"{amount:,.2f}".replace(",", " ")


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _server_quantity(candidate: Mapping[str, Any]) -> int | None:
    quantity = _int_value(candidate.get("quantity_required"))
    if quantity is not None and quantity > 0:
        return quantity

    components = candidate.get("components")
    if not isinstance(components, Iterable) or isinstance(components, str):
        return None

    for component in components:
        if not isinstance(component, Mapping) or component.get("role") != "server_platform":
            continue
        try:
            return int(component.get("quantity_required"))
        except (TypeError, ValueError):
            return None
    return None


def _server_quantity_text(value: int | None) -> str:
    if value is None:
        return "серверы"
    return f"{value} {pluralize_ru(value, 'сервер', 'сервера', 'серверов')}"


def _stock_text(value: Any) -> str:
    if value is None:
        return "неизвестно"
    return f"{value} шт."
