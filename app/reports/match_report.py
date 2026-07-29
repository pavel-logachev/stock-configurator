from __future__ import annotations

from app.matching.match_engine import (
    BUILD_CANDIDATE_TYPE,
    READY_SERVER_CANDIDATE_TYPE,
    MatchCandidateResult,
    MatchResult,
)
from app.reports.match_text import (
    candidate_outcome,
    format_price,
    human_match_status,
    humanize_check_text,
    yes_no,
)


def build_match_markdown_report(result: MatchResult) -> str:
    ready_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.candidate_type == READY_SERVER_CANDIDATE_TYPE
    ]
    build_candidates = [
        candidate
        for candidate in result.candidates
        if candidate.candidate_type == BUILD_CANDIDATE_TYPE
    ]
    lines = [
        "# Match Engine V0 Report",
        "",
        f"- Статус: {human_match_status(result.status)}",
        f"- Нужна проверка инженера: {yes_no(result.engineer_review_required)}",
        f"- Кандидатов: {result.total_candidates}",
        f"- Полных совпадений: {result.matched_items}",
        "",
        "## Краткий вывод",
        "",
        _summary(result),
        "",
        "## Готовые варианты со склада",
        "",
    ]

    if not ready_candidates:
        lines.append("Готовые серверы со склада не найдены.")
    else:
        for index, candidate in enumerate(ready_candidates, start=1):
            lines.extend(_candidate_lines(index, candidate))

    lines.extend(["", "## Сборки из комплектующих", ""])
    if not build_candidates:
        lines.append(
            "Сборка из комплектующих пока не предложена - нет достаточных складских данных "
            "по платформам/комплектующим."
        )
    else:
        for index, candidate in enumerate(build_candidates, start=1):
            lines.extend(_build_candidate_lines(index, candidate))

    lines.extend(
        [
            "",
            "## Что проверить",
            "",
            *(_list_lines(result.risk_flags) if result.risk_flags else ["- Нет"]),
            "",
            "## Engineer Review",
            "",
            "- Требуется инженерная проверка: да",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _summary(result: MatchResult) -> str:
    if result.status in {"matched", "stock_matched"}:
        return "Есть варианты, которые предварительно закрывают требования по правилам подбора."
    if result.status == "partial_stock_matched":
        return "Найдены варианты, но часть требований или рисков требует проверки инженером."
    return "Подходящих складских вариантов не найдено."


def _candidate_lines(index: int, candidate: MatchCandidateResult) -> list[str]:
    price = format_price(candidate.price_value, candidate.price_currency)
    available = (
        str(candidate.available_quantity)
        if candidate.available_quantity is not None
        else "не найдено"
    )
    candidate_map = {
        "confidence_score": candidate.confidence_score,
        "missing_requirements": candidate.missing_requirements,
    }
    return [
        f"### {index}. {candidate.part_number or candidate.item_id}",
        "",
        f"- Дистрибьютор: {candidate.distributor_code}",
        f"- Item ID: {candidate.item_id}",
        f"- Part number: {candidate.part_number or ''}",
        f"- Производитель: {candidate.producer or ''}",
        f"- Категория: {candidate.category_id or ''}",
        f"- Наименование: {candidate.item_name or ''}",
        f"- Оценка соответствия: {candidate.confidence_score} из 100",
        f"- Итог по варианту: {candidate_outcome(candidate_map)}",
        f"- Цена: {price}",
        f"- Остаток: {available}",
        f"- Резервируемые локации: {candidate.reservable_locations}",
        "",
        "Что закрыто:",
        *(
            _list_lines(candidate.matched_requirements)
            if candidate.matched_requirements
            else ["- Нет"]
        ),
        "",
        "Что не закрыто:",
        *(
            _list_lines(candidate.missing_requirements)
            if candidate.missing_requirements
            else ["- Нет"]
        ),
        "",
        "Риски:",
        *(_list_lines(candidate.risk_flags) if candidate.risk_flags else ["- Нет"]),
        "",
    ]


def _build_candidate_lines(index: int, candidate: MatchCandidateResult) -> list[str]:
    price = format_price(candidate.total_price_value, candidate.total_price_currency)
    components = candidate.components
    platform = _components_text(components, "server_platform")
    cpu = _components_text(components, "cpu")
    ram = _components_text(components, "ram")
    storage = ", ".join(
        value
        for value in [
            _components_text(components, "ssd"),
            _components_text(components, "hdd"),
        ]
        if value
    )
    controllers = _components_text(components, "storage_controller")
    network = _components_text(components, "network_adapter")
    checks = _unique_texts([*candidate.compatibility_warnings, *candidate.missing_components])
    cpu_text = cpu or (
        "не подобраны" if "cpu" in candidate.missing_component_roles else "не указаны"
    )
    price_note = f" ({candidate.total_price_note})" if candidate.total_price_note else ""

    return [
        f"### {index}. {candidate.item_name or 'Предварительная сборка'}",
        "",
        f"- Итог по сборке: {candidate_outcome(candidate.to_report_json())}",
        f"- Платформа: {platform or 'не выбрана'}",
        f"- CPU: {cpu_text}",
        f"- RAM: {ram or 'не выбрана'}",
        f"- Накопители: {storage or 'не выбраны'}",
        f"- Контроллеры: {controllers or 'не выбраны'}",
        f"- Сетевые адаптеры: {network or 'не выбраны'}",
        f"- Ориентировочная сумма: {price}{price_note}",
        "- Требуется инженерная проверка совместимости: да",
        "",
        "Что проверить инженеру:",
        *(_list_lines(checks) if checks else ["- Требуется инженерная проверка совместимости."]),
        "",
    ]


def _components_text(components: list[dict[str, object]], role: str) -> str:
    values: list[str] = []
    for component in components:
        if component.get("role") != role:
            continue
        display_parts = [
            str(component.get("producer") or "").strip(),
            str(component.get("part_number") or "").strip(),
        ]
        display = " ".join(part for part in display_parts if part)
        if not display:
            display = str(component.get("item_name") or component.get("item_id") or "").strip()
        quantity = component.get("quantity_required")
        stock = component.get("available_quantity")
        stock_text = "не найден" if stock is None else str(stock)
        values.append(f"{display} - требуется {quantity} шт., остаток {stock_text}")
    return "; ".join(values)


def _unique_texts(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _list_lines(values: list[str]) -> list[str]:
    return [f"- {humanize_check_text(value)}" for value in values]
