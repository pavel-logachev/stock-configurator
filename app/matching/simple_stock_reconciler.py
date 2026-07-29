from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from app.matching.simple_stock_matrix import (
    SimpleStockMatrixPackage,
    product_cards_by_id,
    stock_rows_by_id,
)

QUOTE_INTEGRITY_RECONCILER_VERSION = "quote_integrity_reconciler_v9"
RECONCILED_QUANTITY_NOTE = "Количество скорректировано по доступному складскому остатку."
SPLIT_QUANTITY_NOTE = "Количество распределено по нескольким складским остаткам выбранной позиции."
LOWER_BOUND_STOCK_NOTE = (
    "Точный остаток указан дистрибьютором как больше отображаемого значения; "
    "перед отправкой КП нужно подтвердить доступность выбранного количества."
)
OPEN_STOCK_CAPACITY = 1_000_000_000


@dataclass(frozen=True)
class QuoteIntegrityResult:
    status: str
    quote: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    error_details: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StockAllocation:
    stock_row_id: str
    row: Mapping[str, Any]
    quantity: int
    available_quantity: int


def reconcile_simple_stock_quote(
    quote: Mapping[str, Any],
    matrix_package: SimpleStockMatrixPackage,
) -> QuoteIntegrityResult:
    """Materialize LLM-selected product candidates into exact stock rows.

    LLM2 owns semantic product choice and returns product-level
    component_candidate_id values. This reconciler does only deterministic
    mechanics: pick real stock rows for selected products, copy canonical price
    facts, split quantities across stock buckets and recalculate totals.
    """

    matrix_rows = stock_rows_by_id(matrix_package)
    product_cards = product_cards_by_id(matrix_package)
    component_index = _component_index(matrix_rows)
    reconciled = deepcopy(dict(quote))
    raw_lines = _sequence(reconciled.get("lines"))
    if not raw_lines:
        return _mechanical_error(
            errors=["quote_integrity.lines_missing"],
            details=[{"type": "lines_missing"}],
            quote=reconciled,
        )

    errors: list[str] = []
    error_details: list[dict[str, Any]] = []
    warnings: list[str] = []
    adjustments: list[dict[str, Any]] = []
    remaining_by_stock_row = {
        stock_row_id: _allocation_capacity(row, quantity)
        for stock_row_id, row in matrix_rows.items()
        if (quantity := _available_quantity(row)) is not None
    }
    reconciled_lines: list[dict[str, Any]] = []
    shortage_gaps: list[dict[str, Any]] = []

    for index, raw_line in enumerate(raw_lines, start=1):
        if not isinstance(raw_line, Mapping):
            errors.append(f"quote_integrity.line_{index}.invalid")
            error_details.append({"type": "invalid_line", "line_index": index})
            continue

        component_id = _component_id(raw_line)
        if not component_id or component_id not in component_index:
            errors.append(f"quote_integrity.line_{index}.unknown_component_candidate_id")
            error_details.append(
                {
                    "type": "unknown_component_candidate_id",
                    "line_index": index,
                    "component_candidate_id": component_id,
                }
            )
            continue

        quantity = _positive_int(raw_line.get("quantity"))
        if quantity is None:
            errors.append(f"quote_integrity.line_{index}.invalid_quantity")
            error_details.append(
                {
                    "type": "invalid_quantity",
                    "line_index": index,
                    "component_candidate_id": component_id,
                    "quantity": raw_line.get("quantity"),
                }
            )
            continue

        allocations, shortage_quantity = _allocate_component_rows(
            component_id=component_id,
            requested_quantity=quantity,
            rows=component_index[component_id],
            preferred_currency=_preferred_currency(raw_line),
            remaining_by_stock_row=remaining_by_stock_row,
        )
        supplied_stock_row_id = _clean_text(raw_line.get("stock_row_id"))
        if supplied_stock_row_id and supplied_stock_row_id not in {
            allocation.stock_row_id for allocation in allocations
        }:
            _append_unique(warnings, "quote_integrity.llm_stock_row_id_ignored")
            adjustments.append(
                {
                    "type": "llm_stock_row_id_ignored",
                    "section": "line",
                    "index": index,
                    "component_candidate_id": component_id,
                    "supplied_stock_row_id": supplied_stock_row_id,
                    "resolution": "allocated_by_component_candidate_id",
                }
            )

        if shortage_quantity > 0:
            _append_unique(warnings, "quote_integrity.stock_overallocation_adjusted")
            requirement_id = _line_requirement_id(raw_line)
            adjustments.append(
                {
                    "type": "stock_overallocation",
                    "component_candidate_id": component_id,
                    "requirement_id": requirement_id or None,
                    "requested_quantity": quantity,
                    "included_quantity": quantity - shortage_quantity,
                    "shortage_quantity": shortage_quantity,
                }
            )
            shortage_gaps.append(
                _shortage_gap(
                    raw_line,
                    component_id=component_id,
                    product_card=product_cards.get(component_id, {}),
                    requirement_id=requirement_id,
                    requested_quantity=quantity,
                    included_quantity=quantity - shortage_quantity,
                    shortage_quantity=shortage_quantity,
                )
            )

        for allocation in allocations:
            if not _allocation_uses_lower_bound_stock(allocation):
                continue
            _append_unique(warnings, "quote_integrity.stock_lower_bound_requires_confirmation")
            adjustments.append(
                {
                    "type": "stock_lower_bound_quantity_confirm",
                    "section": "line",
                    "index": index,
                    "component_candidate_id": component_id,
                    "stock_row_id": allocation.stock_row_id,
                    "displayed_available_quantity": allocation.available_quantity,
                    "included_quantity": allocation.quantity,
                    "resolution": "kept_llm_quantity_with_stock_confirmation",
                }
            )

        if len(allocations) > 1:
            adjustments.append(
                {
                    "type": "component_quantity_split",
                    "section": "line",
                    "index": index,
                    "component_candidate_id": component_id,
                    "requested_quantity": quantity,
                    "stock_row_count": len(allocations),
                }
            )

        for allocation_index, allocation in enumerate(allocations, start=1):
            line = _materialized_line(
                raw_line,
                component_id=component_id,
                allocation=allocation,
                raw_line_index=index,
                allocation_index=allocation_index,
                allocation_count=len(allocations),
                requested_quantity=quantity,
                shortage_quantity=shortage_quantity,
            )
            reconciled_lines.append(line)

    if errors:
        return _mechanical_error(errors=errors, details=error_details, quote=reconciled)
    if not reconciled_lines:
        return _mechanical_error(
            errors=["quote_integrity.lines_empty_after_reconciliation"],
            details=[{"type": "lines_empty_after_reconciliation"}],
            quote=reconciled,
        )

    reconciled["lines"] = reconciled_lines
    _normalize_target_decisions_against_lines(
        reconciled,
        lines=reconciled_lines,
        adjustments=adjustments,
    )
    merged_procurement_gaps = _merge_procurement_gaps(
        reconciled.get("procurement_gaps"),
        shortage_gaps,
    )
    _append_stock_shortage_coverage_note(reconciled, shortage_gaps)
    reconciled["procurement_gaps"] = _reconcile_procurement_gap_candidates(
        merged_procurement_gaps,
        component_index=component_index,
        warnings=warnings,
        adjustments=adjustments,
    )
    reconciled["available_alternatives"] = _reconcile_available_alternatives(
        reconciled.get("available_alternatives"),
        component_index=component_index,
        warnings=warnings,
        adjustments=adjustments,
    )

    totals_by_currency = _totals_by_currency(reconciled_lines)
    if len(totals_by_currency) == 1:
        currency, value = next(iter(totals_by_currency.items()))
        reconciled["total_price_value"] = _decimal_text(value)
        reconciled["total_price_currency"] = currency
        reconciled.pop("totals_by_currency", None)
    else:
        reconciled["total_price_value"] = None
        reconciled["total_price_currency"] = None
        reconciled["totals_by_currency"] = [
            {"currency": currency, "value": _decimal_text(value)}
            for currency, value in totals_by_currency.items()
        ]

    status = "mechanically_adjusted" if adjustments or warnings else "ok"
    reconciled["quote_integrity"] = {
        "version": QUOTE_INTEGRITY_RECONCILER_VERSION,
        "status": status,
        "adjustments": adjustments,
        "warnings": warnings,
    }
    return QuoteIntegrityResult(
        status=status,
        quote=reconciled,
        warnings=warnings,
        diagnostics=_diagnostics(status, adjustments=adjustments, errors=[]),
    )


def _materialized_line(
    raw_line: Mapping[str, Any],
    *,
    component_id: str,
    allocation: StockAllocation,
    raw_line_index: int,
    allocation_index: int,
    allocation_count: int,
    requested_quantity: int,
    shortage_quantity: int,
) -> dict[str, Any]:
    row = allocation.row
    unit_price, currency = _matrix_price(row)
    assert unit_price is not None and currency
    line = dict(raw_line)
    base_line_id = _clean_text(line.get("line_id")) or f"L{raw_line_index}"
    if allocation_count > 1:
        line["line_id"] = f"{base_line_id}.{allocation_index}"
        line["source_line_id"] = base_line_id
        line["reconciliation_note"] = SPLIT_QUANTITY_NOTE
    else:
        line["line_id"] = base_line_id
    if shortage_quantity > 0:
        line["reason"] = RECONCILED_QUANTITY_NOTE
        line["reconciliation_note"] = RECONCILED_QUANTITY_NOTE
        line["quantity_adjusted"] = True
        line["original_requested_quantity"] = requested_quantity
        line["shortage_quantity"] = shortage_quantity

    line["component_candidate_id"] = component_id
    line["stock_row_id"] = allocation.stock_row_id
    line["part_number"] = _clean_text(row.get("part_number"))
    line["item_name"] = _matrix_item_name(row)
    line["quantity"] = allocation.quantity
    quantity_is_greater_than = _stock_quantity_is_greater_than(row)
    line["available_quantity"] = (
        allocation.quantity
        if quantity_is_greater_than and allocation.quantity > allocation.available_quantity
        else allocation.available_quantity
    )
    line["quantity_value"] = allocation.available_quantity
    line["quantity_is_greater_than"] = quantity_is_greater_than
    if quantity_is_greater_than and allocation.quantity > allocation.available_quantity:
        line["stock_confirmation_required"] = True
        line["stock_confirmation_note"] = LOWER_BOUND_STOCK_NOTE
    line["unit_price_value"] = _decimal_text(unit_price)
    line["unit_price_currency"] = currency
    line["line_total_value"] = _decimal_text(unit_price * allocation.quantity)
    line["line_total_currency"] = currency
    return line


def _allocate_component_rows(
    *,
    component_id: str,
    requested_quantity: int,
    rows: Sequence[Mapping[str, Any]],
    preferred_currency: str,
    remaining_by_stock_row: dict[str, int],
) -> tuple[list[StockAllocation], int]:
    candidates = [
        row
        for row in rows
        if _allocatable_quantity(row, remaining_by_stock_row) > 0
        and _matrix_price(row)[0] is not None
        and _matrix_price(row)[1]
    ]
    if not candidates:
        return [], requested_quantity

    ordered_rows = _ordered_allocation_rows(candidates, preferred_currency)
    plan_rows = _single_currency_plan(ordered_rows, requested_quantity, preferred_currency)
    if not plan_rows:
        plan_rows = ordered_rows

    need = requested_quantity
    allocations: list[StockAllocation] = []
    for row in plan_rows:
        stock_row_id = _clean_text(row.get("stock_row_id"))
        if not stock_row_id:
            continue
        available_quantity = _available_quantity(row)
        if available_quantity is None:
            continue
        remaining = remaining_by_stock_row.get(stock_row_id, 0)
        if remaining <= 0:
            continue
        quantity = min(need, remaining)
        if quantity <= 0:
            continue
        remaining_by_stock_row[stock_row_id] = remaining - quantity
        allocations.append(
            StockAllocation(
                stock_row_id=stock_row_id,
                row=row,
                quantity=quantity,
                available_quantity=available_quantity,
            )
        )
        need -= quantity
        if need <= 0:
            break

    return allocations, max(0, need)


def _ordered_allocation_rows(
    rows: Sequence[Mapping[str, Any]],
    preferred_currency: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _currency_rank(_matrix_price(row)[1], preferred_currency),
            _matrix_price(row)[0] or Decimal("999999999999"),
            _clean_text(row.get("stock_row_id")),
        ),
    )


def _single_currency_plan(
    rows: Sequence[Mapping[str, Any]],
    requested_quantity: int,
    preferred_currency: str,
) -> list[Mapping[str, Any]]:
    preferred = _clean_text(preferred_currency).upper()
    if preferred:
        preferred_rows = [
            row
            for row in rows
            if _clean_text(_matrix_price(row)[1]).upper() == preferred
        ]
        return (
            preferred_rows
            if _rows_can_cover_requested(preferred_rows, requested_quantity)
            else []
        )
    return []


def _currency_rank(currency: str | None, preferred_currency: str) -> tuple[int, str]:
    normalized = _clean_text(currency).upper()
    preferred = _clean_text(preferred_currency).upper()
    if preferred and normalized == preferred:
        return (0, normalized)
    if preferred:
        return (1, normalized or "ZZZ")
    return (0, normalized or "ZZZ")


def _reconcile_procurement_gap_candidates(
    value: Any,
    *,
    component_index: Mapping[str, Sequence[Mapping[str, Any]]],
    warnings: list[str],
    adjustments: list[dict[str, Any]],
) -> list[Any]:
    gaps: list[Any] = []
    for gap_index, raw_gap in enumerate(_sequence(value), start=1):
        if not isinstance(raw_gap, Mapping):
            gaps.append(raw_gap)
            continue
        gap = dict(raw_gap)
        if "considered_candidates" in gap:
            gap["considered_candidates"] = _reconcile_product_candidates(
                gap.get("considered_candidates"),
                section=f"procurement_gap_{gap_index}.considered_candidate",
                component_index=component_index,
                warnings=warnings,
                adjustments=adjustments,
            )
        gaps.append(gap)
    return gaps


def _reconcile_available_alternatives(
    value: Any,
    *,
    component_index: Mapping[str, Sequence[Mapping[str, Any]]],
    warnings: list[str],
    adjustments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _reconcile_product_candidates(
        value,
        section="available_alternative",
        component_index=component_index,
        warnings=warnings,
        adjustments=adjustments,
    )


def _reconcile_product_candidates(
    value: Any,
    *,
    section: str,
    component_index: Mapping[str, Sequence[Mapping[str, Any]]],
    warnings: list[str],
    adjustments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, raw_item in enumerate(_sequence(value), start=1):
        if not isinstance(raw_item, Mapping):
            continue
        component_id = _component_id(raw_item)
        if not component_id or component_id not in component_index:
            warning = _candidate_warning(section, "unknown_component_candidate_id")
            _append_unique(warnings, warning)
            adjustments.append(
                {
                    "type": _candidate_adjustment_type(section, "removed"),
                    "section": section,
                    "index": index,
                    "component_candidate_id": component_id,
                    "reason": "unknown_component_candidate_id",
                }
            )
            continue

        representative = _representative_row(
            component_index[component_id],
            preferred_currency=_preferred_currency(raw_item),
        )
        if representative is None:
            warning = _candidate_warning(section, "incomplete_stock_fact")
            _append_unique(warnings, warning)
            adjustments.append(
                {
                    "type": _candidate_adjustment_type(section, "removed"),
                    "section": section,
                    "index": index,
                    "component_candidate_id": component_id,
                    "reason": "incomplete_stock_fact",
                }
            )
            continue

        unit_price, currency = _matrix_price(representative)
        assert unit_price is not None and currency
        item = dict(raw_item)
        item["item"] = _matrix_item_name(representative)
        item["component_candidate_id"] = component_id
        item["stock_row_id"] = _clean_text(representative.get("stock_row_id"))
        part_number = _clean_text(representative.get("part_number"))
        if part_number:
            item["part_number"] = part_number
        available_quantity = _total_available_quantity(
            component_index[component_id],
            currency=currency,
        )
        quantity_is_greater_than = _total_available_quantity_is_lower_bound(
            component_index[component_id],
            currency=currency,
        )
        item["available_quantity"] = available_quantity
        item["quantity_value"] = available_quantity
        item["quantity_is_greater_than"] = quantity_is_greater_than
        if quantity_is_greater_than:
            item["stock_confirmation_required"] = True
            item["stock_confirmation_note"] = LOWER_BOUND_STOCK_NOTE
        item["unit_price_value"] = _decimal_text(unit_price)
        item["unit_price_currency"] = currency
        candidates.append(item)
    return candidates


def _representative_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    preferred_currency: str,
) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if (_available_quantity(row) or 0) > 0
        and _matrix_price(row)[0] is not None
        and _matrix_price(row)[1]
    ]
    if not candidates:
        return None
    return _ordered_allocation_rows(candidates, preferred_currency)[0]


def _component_index(
    matrix_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    for row in matrix_rows.values():
        component_id = _clean_text(row.get("component_candidate_id"))
        if not component_id:
            continue
        index.setdefault(component_id, []).append(row)
    for rows in index.values():
        rows.sort(
            key=lambda row: (
                _currency_rank(_matrix_price(row)[1], ""),
                _matrix_price(row)[0] or Decimal("999999999999"),
                _clean_text(row.get("stock_row_id")),
            )
        )
    return index


def _merge_procurement_gaps(value: Any, shortage_gaps: Sequence[Mapping[str, Any]]) -> list[Any]:
    gaps = list(_sequence(value))
    for shortage_gap in shortage_gaps:
        replaced = False
        for index, existing in enumerate(gaps):
            if not isinstance(existing, Mapping):
                continue
            if _gaps_should_merge(existing, shortage_gap):
                merged = dict(existing)
                if _gaps_are_duplicate_shortage(existing, shortage_gap):
                    merged["quantity"] = max(
                        _positive_int(existing.get("quantity")) or 0,
                        _positive_int(shortage_gap.get("quantity")) or 0,
                    )
                    merged["duplicate_shortage_deduped"] = True
                else:
                    merged["quantity"] = _sum_quantities(
                        existing.get("quantity"),
                        shortage_gap.get("quantity"),
                    )
                merged["reason"] = _merge_gap_reason(
                    existing.get("reason"),
                    shortage_gap.get("reason"),
                )
                for key, value_item in shortage_gap.items():
                    if key in {"quantity", "reason"}:
                        continue
                    if key not in merged or not _clean_text(merged.get(key)):
                        merged[key] = value_item
                gaps[index] = merged
                replaced = True
                break
        if not replaced:
            gaps.append(dict(shortage_gap))
    return gaps


def _gaps_should_merge(existing: Mapping[str, Any], shortage_gap: Mapping[str, Any]) -> bool:
    existing_key = _clean_text(existing.get("internal_key"))
    shortage_key = _clean_text(shortage_gap.get("internal_key"))
    if existing_key and shortage_key and existing_key == shortage_key:
        return True

    existing_component_id = _component_id(existing)
    shortage_component_id = _component_id(shortage_gap)
    if existing_component_id and shortage_component_id:
        existing_requirement_id = _clean_text(existing.get("requirement_id")).casefold()
        shortage_requirement_id = _clean_text(shortage_gap.get("requirement_id")).casefold()
        return (
            existing_component_id == shortage_component_id
            and existing_requirement_id == shortage_requirement_id
        )

    existing_requirement_id = _clean_text(existing.get("requirement_id")).casefold()
    shortage_requirement_id = _clean_text(shortage_gap.get("requirement_id")).casefold()
    existing_item = _clean_text(existing.get("item")).casefold()
    shortage_item = _clean_text(shortage_gap.get("item")).casefold()
    return bool(
        existing_requirement_id
        and existing_requirement_id == shortage_requirement_id
        and existing_item
        and (
            existing_item == shortage_item
            or _texts_materially_overlap(existing_item, shortage_item)
        )
    )


def _gaps_are_duplicate_shortage(
    existing: Mapping[str, Any],
    shortage_gap: Mapping[str, Any],
) -> bool:
    if _clean_text(shortage_gap.get("gap_type")) != "quantity_shortage":
        return False

    existing_quantity = _positive_int(existing.get("quantity"))
    shortage_quantity = _positive_int(shortage_gap.get("quantity"))
    if existing_quantity is None or shortage_quantity is None:
        return False
    if existing_quantity != shortage_quantity:
        return False

    existing_requirement_id = _clean_text(existing.get("requirement_id")).casefold()
    shortage_requirement_id = _clean_text(shortage_gap.get("requirement_id")).casefold()
    if not existing_requirement_id or existing_requirement_id != shortage_requirement_id:
        return False

    existing_key = _clean_text(existing.get("internal_key"))
    shortage_key = _clean_text(shortage_gap.get("internal_key"))
    if existing_key and shortage_key and existing_key == shortage_key:
        return True

    existing_component_id = _component_id(existing)
    shortage_component_id = _component_id(shortage_gap)
    if existing_component_id and shortage_component_id:
        return existing_component_id == shortage_component_id

    return _texts_materially_overlap(
        _clean_text(existing.get("item")),
        _clean_text(shortage_gap.get("item")),
    )


def _texts_materially_overlap(left: str, right: str) -> bool:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens & right_tokens
    return len(shared) >= 2 and len(shared) >= min(len(left_tokens), len(right_tokens)) / 2


def _meaningful_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[A-Za-zА-Яа-я0-9]+", value.casefold()):
        if len(token) < 4 or any(char.isdigit() for char in token):
            continue
        tokens.add(token)
    return tokens


def _merge_gap_reason(left: Any, right: Any) -> str:
    left_text = _clean_text(left)
    right_text = _clean_text(right)
    if left_text and right_text and left_text != right_text:
        return f"{left_text} {right_text}"
    return right_text or left_text


def _shortage_gap(
    line: Mapping[str, Any],
    *,
    component_id: str,
    product_card: Mapping[str, Any],
    requirement_id: str,
    requested_quantity: int,
    included_quantity: int,
    shortage_quantity: int,
) -> dict[str, Any]:
    item = (
        _clean_text(line.get("role"))
        or _clean_text(product_card.get("description"))
        or component_id
    )
    internal_key = (
        f"quantity_shortage:{requirement_id}:{component_id}"
        if requirement_id
        else f"stock_shortage:{component_id}"
    )
    reason = (
        f"После сверки со складским остатком закрыто {included_quantity} шт. "
        f"из запрошенных {requested_quantity} шт.; не закрыто {shortage_quantity} шт."
    )
    gap = {
        "item": item,
        "quantity": shortage_quantity,
        "reason": reason,
        "component_candidate_id": component_id,
        "gap_type": "quantity_shortage",
        "requested_quantity": requested_quantity,
        "included_quantity": included_quantity,
        "shortage_quantity": shortage_quantity,
        "internal_key": internal_key,
    }
    if requirement_id:
        gap["requirement_id"] = requirement_id
    return gap


def _append_stock_shortage_coverage_note(
    quote: dict[str, Any],
    shortage_gaps: Sequence[Mapping[str, Any]],
) -> None:
    if not shortage_gaps:
        return

    notes: list[str] = []
    for gap in shortage_gaps[:5]:
        item = _clean_text(gap.get("item")) or _clean_text(gap.get("component_candidate_id"))
        requested = _positive_int(gap.get("requested_quantity"))
        included = _positive_int(gap.get("included_quantity"))
        shortage = _positive_int(gap.get("shortage_quantity")) or _positive_int(
            gap.get("quantity")
        )
        if not item or shortage is None:
            continue
        if requested is not None and included is not None:
            notes.append(f"{item} - закрыто {included} из {requested}, не закрыто {shortage}")
        else:
            notes.append(f"{item} - не закрыто {shortage}")

    if not notes:
        return
    if len(shortage_gaps) > len(notes):
        notes.append(f"и еще {len(shortage_gaps) - len(notes)} поз.")

    note_text = "После сверки складских остатков частично закрыто: " + "; ".join(notes) + "."
    quote["stock_shortage_summary"] = notes
    current = _clean_text(quote.get("coverage_summary"))
    if not current:
        quote["coverage_summary"] = note_text
        return
    if note_text not in current:
        quote["coverage_summary"] = f"{current} {note_text}"


def _line_requirement_id(line: Mapping[str, Any]) -> str:
    for key in ("requirement_id", "quantity_requirement_id"):
        value = _clean_text(line.get(key))
        if value:
            return value
    for key in (
        "satisfies_requirement_ids",
        "covered_requirement_ids",
        "covers_requirement_ids",
        "requirement_ids",
    ):
        for value in _sequence(line.get(key)):
            requirement_id = _clean_text(value)
            if requirement_id:
                return requirement_id
    return ""


def _normalize_target_decisions_against_lines(
    quote: dict[str, Any],
    *,
    lines: Sequence[Mapping[str, Any]],
    adjustments: list[dict[str, Any]],
) -> None:
    target_decisions = _sequence(quote.get("target_decisions"))
    if not target_decisions:
        return

    line_ids: set[str] = set()
    component_line_ids: dict[str, list[str]] = {}
    for line in lines:
        line_id = _clean_text(line.get("line_id"))
        component_id = _component_id(line)
        if line_id:
            line_ids.add(line_id)
        if component_id:
            component_line_ids.setdefault(component_id, [])
            if line_id:
                component_line_ids[component_id].append(line_id)

    normalized: list[Any] = []
    changed = False
    for index, raw_item in enumerate(target_decisions, start=1):
        if not isinstance(raw_item, Mapping):
            normalized.append(raw_item)
            continue

        item = dict(raw_item)
        anchor_line_id = _clean_text(item.get("anchor_line_id"))
        anchor_candidate_id = _clean_text(
            item.get("anchor_candidate_id")
            or item.get("component_candidate_id")
            or item.get("position_id")
        )
        matched_line_id = anchor_line_id if anchor_line_id in line_ids else ""
        matched_by_candidate = bool(
            anchor_candidate_id and anchor_candidate_id in component_line_ids
        )
        if not matched_line_id and matched_by_candidate:
            candidate_line_ids = component_line_ids.get(anchor_candidate_id, [])
            if len(candidate_line_ids) == 1:
                matched_line_id = candidate_line_ids[0]

        if matched_line_id or matched_by_candidate:
            original_status = _clean_text(item.get("anchor_status")).lower()
            item_changed = False
            if matched_line_id and anchor_line_id != matched_line_id:
                item["anchor_line_id"] = matched_line_id
                item_changed = True
            if original_status != "selected":
                item["anchor_status"] = "selected"
                item_changed = True
            if item_changed:
                adjustments.append(
                    {
                        "type": "target_decision_anchor_status_normalized",
                        "section": "target_decision",
                        "index": index,
                        "anchor_candidate_id": anchor_candidate_id or None,
                        "anchor_line_id": anchor_line_id or None,
                        "resolved_anchor_line_id": matched_line_id or None,
                        "original_anchor_status": original_status or None,
                        "resolved_anchor_status": "selected",
                        "resolution": "selected_line_presence",
                    }
                )
                changed = True

        normalized.append(item)

    if changed:
        quote["target_decisions"] = normalized


def _candidate_warning(section: str, reason: str) -> str:
    prefix = (
        "considered_candidate"
        if section.startswith("procurement_gap_")
        else "available_alternative"
    )
    return f"quote_integrity.{prefix}_{reason}"


def _candidate_adjustment_type(section: str, suffix: str) -> str:
    prefix = (
        "considered_candidate"
        if section.startswith("procurement_gap_")
        else "available_alternative"
    )
    return f"{prefix}_{suffix}"


def _sum_quantities(left: Any, right: Any) -> int:
    left_int = _positive_int(left) or 0
    right_int = _positive_int(right) or 0
    return left_int + right_int


def _totals_by_currency(lines: Sequence[Mapping[str, Any]]) -> dict[str, Decimal]:
    totals: dict[str, Decimal] = {}
    for line in lines:
        currency = _clean_text(line.get("line_total_currency"))
        value = _decimal_value(line.get("line_total_value"))
        if not currency or value is None:
            continue
        totals[currency] = totals.get(currency, Decimal("0")) + value
    return totals


def _matrix_price(row: Mapping[str, Any]) -> tuple[Decimal | None, str | None]:
    price = _mapping(row.get("price"))
    return _decimal_value(price.get("value")), _clean_text(price.get("currency")) or None


def _available_quantity(row: Mapping[str, Any]) -> int | None:
    stock = _mapping(row.get("stock"))
    value = stock.get("quantity_value")
    quantity = _positive_int(value)
    if quantity is not None:
        return quantity
    if value in (0, "0"):
        return 0
    return None


def _stock_quantity_is_greater_than(row: Mapping[str, Any]) -> bool:
    return bool(_mapping(row.get("stock")).get("quantity_is_greater_than"))


def _allocation_capacity(row: Mapping[str, Any], quantity: int) -> int:
    if quantity > 0 and _stock_quantity_is_greater_than(row):
        return OPEN_STOCK_CAPACITY
    return quantity


def _rows_can_cover_requested(
    rows: Sequence[Mapping[str, Any]],
    requested_quantity: int,
) -> bool:
    total = 0
    for row in rows:
        quantity = _available_quantity(row) or 0
        if quantity > 0 and _stock_quantity_is_greater_than(row):
            return True
        total += quantity
    return total >= requested_quantity


def _allocation_uses_lower_bound_stock(allocation: StockAllocation) -> bool:
    return (
        allocation.available_quantity > 0
        and _stock_quantity_is_greater_than(allocation.row)
        and allocation.quantity > allocation.available_quantity
    )


def _allocatable_quantity(
    row: Mapping[str, Any],
    remaining_by_stock_row: Mapping[str, int],
) -> int:
    stock_row_id = _clean_text(row.get("stock_row_id"))
    if not stock_row_id:
        return 0
    return max(0, remaining_by_stock_row.get(stock_row_id, _available_quantity(row) or 0))


def _total_available_quantity(
    rows: Sequence[Mapping[str, Any]],
    *,
    currency: str,
) -> int:
    total = 0
    for row in rows:
        if _matrix_price(row)[1] != currency:
            continue
        total += _available_quantity(row) or 0
    return total


def _total_available_quantity_is_lower_bound(
    rows: Sequence[Mapping[str, Any]],
    *,
    currency: str,
) -> bool:
    for row in rows:
        if _matrix_price(row)[1] != currency:
            continue
        if (_available_quantity(row) or 0) > 0 and _stock_quantity_is_greater_than(row):
            return True
    return False


def _matrix_item_name(row: Mapping[str, Any]) -> str:
    return (
        _clean_text(row.get("description"))
        or _clean_text(row.get("part_number"))
        or _clean_text(row.get("stock_row_id"))
    )


def _component_id(value: Mapping[str, Any]) -> str:
    return _clean_text(value.get("component_candidate_id") or value.get("position_id"))


def _preferred_currency(value: Mapping[str, Any]) -> str:
    for key in ("selected_currency", "preferred_currency"):
        text = _clean_text(value.get(key))
        if text:
            return text.upper()
    return ""


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if number <= 0 or number != number.to_integral_value():
        return None
    return int(number)


def _decimal_value(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def _decimal_text(value: Decimal) -> str:
    return str(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str | bytes | bytearray):
        return []
    if isinstance(value, Sequence):
        return list(value)
    return []


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _diagnostics(
    status: str,
    *,
    adjustments: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
) -> dict[str, Any]:
    return {
        "quote_integrity_reconciler": QUOTE_INTEGRITY_RECONCILER_VERSION,
        "quote_integrity_status": status,
        "quote_integrity_adjustment_count": len(adjustments),
        "quote_integrity_error_count": len(errors),
    }


def _mechanical_error(
    *,
    errors: list[str],
    details: list[dict[str, Any]],
    quote: Mapping[str, Any],
) -> QuoteIntegrityResult:
    status = "mechanical_error"
    return QuoteIntegrityResult(
        status=status,
        quote=dict(quote),
        errors=errors,
        error_details=details,
        diagnostics=_diagnostics(status, adjustments=[], errors=errors),
    )
