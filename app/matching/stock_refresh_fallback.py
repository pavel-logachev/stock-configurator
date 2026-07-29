from __future__ import annotations

from collections.abc import Mapping
from typing import Any

STOCK_REFRESH_FAILED_USING_CACHED_STOCK = "failed_using_cached_stock"
STOCK_REFRESH_CACHE_WARNING_RU = (
    "Обновление склада перед подбором не прошло; расчет выполнен по последнему "
    "сохраненному снимку склада. Перед отправкой подтвердить актуальные остатки "
    "и цены у дистрибьютора."
)


def stock_refresh_cached_fallback_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    cached_matrix_row_count: int,
) -> dict[str, Any]:
    result = dict(diagnostics)
    result["refresh_status"] = result.get("status")
    result["status"] = STOCK_REFRESH_FAILED_USING_CACHED_STOCK
    result["fallback_used"] = True
    result["fallback_source"] = "latest_cached_stock"
    result["cached_matrix_row_count"] = cached_matrix_row_count
    result["freshness_warning"] = STOCK_REFRESH_CACHE_WARNING_RU
    return result


def stock_refresh_used_cached_fallback(diagnostics: Mapping[str, Any]) -> bool:
    return (
        diagnostics.get("status") == STOCK_REFRESH_FAILED_USING_CACHED_STOCK
        and diagnostics.get("fallback_used") is True
    )


def add_stock_refresh_cache_warning(report_json: dict[str, Any]) -> None:
    _append_unique_list_value(
        report_json.setdefault("v3_validation_warnings", []),
        STOCK_REFRESH_CACHE_WARNING_RU,
    )

    quote = report_json.get("validated_quote")
    if not isinstance(quote, dict):
        return

    checks = quote.get("engineer_checks")
    if checks is None:
        quote["engineer_checks"] = [STOCK_REFRESH_CACHE_WARNING_RU]
    elif isinstance(checks, list):
        _append_unique_list_value(checks, STOCK_REFRESH_CACHE_WARNING_RU)
    else:
        quote["engineer_checks"] = [str(checks), STOCK_REFRESH_CACHE_WARNING_RU]


def _append_unique_list_value(target: Any, value: str) -> None:
    if not isinstance(target, list):
        return
    if value not in target:
        target.append(value)
