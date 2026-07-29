from __future__ import annotations

from app.reports.match_text import pluralize_ru
from app.telegram_bot.formatting import (
    _shown_safe_variants_message,
    choose_excel_report_delivery,
    choose_report_delivery,
    choose_v3_full_category_quote_delivery,
    format_match_summary,
    format_v3_excel_caption,
    format_v3_full_category_quote,
    make_telegram_sentence,
)
from app.user_facing_text import contains_cjk_text


def test_format_v3_full_category_quote_shows_draft_quote() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "storage",
            "match_run_id": 42,
            "category_ids": ["V2101"],
            "result_state": "quote_draft_review_required",
            "engineering_review_required": True,
            "diagnostics": {
                "matrix_row_count": 218,
                "model": "qwen/qwen3.7-plus",
            },
            "validated_quote": {
                "total_price_value": "1730.0000",
                "total_price_currency": "USD",
                "why_selected": "Lowest technically acceptable option.",
                "target_decisions": [
                    {
                        "target_label": "Requested NAS",
                        "anchor_status": "selected",
                        "anchor_line_id": "L1",
                        "reason": "Anchor selected from stock.",
                    }
                ],
                "price_audit": [
                    (
                        "Reviewed cheaper NAS rows and found no technically "
                        "workable lower-price bundle."
                    )
                ],
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": [
                        "NAS row supports selected SATA HDD capacity and quantity."
                    ],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
                "lines": [
                    {
                        "role": "nas",
                        "producer": "ASUS",
                        "part_number": "90IX01V0-BW3S00",
                        "item_name": "NAS enclosure",
                        "quantity": 1,
                        "unit_price_value": "292.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "292.0000",
                        "line_total_currency": "USD",
                        "reason": (
                            "База. В составе по матрице: CPU: Xeon 4510; "
                            "RAM: 64GB DDR5."
                        ),
                        "included_components_summary": {
                            "cpu": "Xeon 4510",
                            "ram": "64GB DDR5",
                            "raid_or_controller": "MR408i-o",
                            "storage": "2x960GB SATA",
                            "network": "4x1GbE",
                            "power": "2x1000W",
                        },
                    },
                ],
                "assumptions": ["RAID usable capacity must be checked."],
                "engineer_checks": ["Check HDD compatibility list."],
            },
        }
    )

    assert "КП draft" in text
    assert "Служебно: ID 42" in text
    assert "ASUS 90IX01V0-BW3S00 NAS enclosure" in text
    assert "В составе по матрице: CPU: Xeon 4510" in text
    assert text.count("В составе по матрице") == 1
    assert "RAM: 64GB DDR5" in text
    assert "1 730 USD" in text
    assert "Целевые объекты:" in text
    assert "Requested NAS - выбран - строка L1" in text
    assert "Спецификация для КП:" in text
    assert "Проверка цены:" in text
    assert "Reviewed cheaper NAS rows" in text
    assert "Проверка совместимости:" in text
    assert "NAS row supports selected SATA HDD capacity and quantity" in text
    assert "самый дешевый минимально достаточный технически рабочий вариант" in text


def test_format_v3_full_category_quote_shows_multi_currency_total_and_groups_lines() -> None:
    payload = {
        "result_state": "quote_draft_review_required",
        "validated_quote": {
            "total_price_value": None,
            "total_price_currency": None,
            "totals_by_currency": [
                {"currency": "USD", "value": "218147.00"},
                {"currency": "RUR", "value": "2867815.05"},
            ],
            "lines": [
                {
                    "component_candidate_id": "ocs:3000048380",
                    "role": "heatsink",
                    "producer": "HPE",
                    "part_number": "P48818-B21",
                    "item_name": "HPE High Performance Heat Sink Kit",
                    "quantity": 1,
                    "unit_price_value": "132.00",
                    "unit_price_currency": "USD",
                    "line_total_value": "132.00",
                    "line_total_currency": "USD",
                    "reason": "Первый складской остаток.",
                },
                {
                    "component_candidate_id": "ocs:3000048380",
                    "role": "heatsink",
                    "producer": "HPE",
                    "part_number": "P48818-B21",
                    "item_name": "HPE High Performance Heat Sink Kit",
                    "quantity": 1,
                    "unit_price_value": "132.00",
                    "unit_price_currency": "USD",
                    "line_total_value": "132.00",
                    "line_total_currency": "USD",
                    "reason": "Второй складской остаток.",
                },
            ],
        },
    }

    text = format_v3_full_category_quote(payload)
    caption = format_v3_excel_caption(payload, match_run_id=253)

    assert "Итого: 218 147 USD + 2 867 815,05 RUR" in text
    assert "Итого: 218 147 USD + 2 867 815,05 RUR" in caption
    assert text.count("HPE P48818-B21 HPE High Performance Heat Sink Kit") == 1
    assert "кол-во 2" in text
    assert "сумма 264 USD" in text


def test_format_v3_full_category_quote_shows_gap_item_quantity_reason() -> None:
    text = format_v3_full_category_quote(
        {
            "result_state": "quote_draft_review_required",
            "validated_quote": {
                "client_status_label": "Частичное покрытие из наличия",
                "selection_mode": "partial_stock_offer",
                "completeness_status": "partial",
                "operational_status": "incomplete_needs_sourcing",
                "procurement_gaps": [
                    {
                        "item": "Intel Xeon Gold 6544Y",
                        "quantity": 6,
                        "reason": "Точная позиция не найдена в переданной матрице.",
                    }
                ],
                "lines": [],
            },
        }
    )

    assert "Что нужно добрать или согласовать:" in text
    assert "Intel Xeon Gold 6544Y - 6 шт." in text
    assert "Точная позиция не найдена в переданной матрице." in text
    assert "причина: Точная позиция" not in text


def test_format_v3_full_category_quote_shows_available_alternatives() -> None:
    text = format_v3_full_category_quote(
        {
            "result_state": "quote_draft_review_required",
            "validated_quote": {
                "client_status_label": "Частичное покрытие из наличия",
                "selection_mode": "partial_stock_offer",
                "completeness_status": "partial",
                "operational_status": "incomplete_needs_sourcing",
                "available_alternatives": [
                    {
                        "requirement_id": "R_HBA",
                        "item": "HPE SN1100Q 16Gb Dual Port FC HBA",
                        "stock_row_id": "treolan:014002/1:123",
                        "available_quantity": 3,
                        "quantity_is_greater_than": True,
                        "unit_price_value": "1176.00",
                        "unit_price_currency": "USD",
                        "reason": "Ниже скорость: 16Gb FC вместо запрошенных 32Gb FC.",
                    }
                ],
                "lines": [],
            },
        }
    )

    assert "Доступные варианты для согласования (не включены в итог):" in text
    assert "требование: R_HBA" in text
    assert "HPE SN1100Q 16Gb Dual Port FC HBA" in text
    assert "доступно: 3+ шт." in text
    assert "цена: 1 176 USD" in text
    assert "16Gb FC вместо запрошенных 32Gb FC" in text
    assert "Что нужно добрать или согласовать:" not in text


def test_format_v3_full_category_quote_prefers_reconciliation_note() -> None:
    text = format_v3_full_category_quote(
        {
            "result_state": "quote_draft_review_required",
            "validated_quote": {
                "lines": [
                    {
                        "role": "Server RAM",
                        "part_number": "MTC40F2046S1RC56BD1",
                        "item_name": "Micron 64GB DDR5-5600 RDIMM",
                        "quantity": 10,
                        "unit_price_value": "3299.9500",
                        "unit_price_currency": "USD",
                        "line_total_value": "32999.5000",
                        "line_total_currency": "USD",
                        "reason": "Предложено максимальное доступное количество 20 из 48.",
                        "reconciliation_note": (
                            "Количество скорректировано по доступному складскому остатку."
                        ),
                    }
                ],
            },
        }
    )

    assert "Количество скорректировано по доступному складскому остатку." in text
    assert "20 из 48" not in text


def test_format_v3_full_category_quote_surfaces_lower_bound_stock_confirmation() -> None:
    text = format_v3_full_category_quote(
        {
            "result_state": "quote_draft_review_required",
            "engineering_review_required": True,
            "validated_quote": {
                "lines": [
                    {
                        "component_candidate_id": "ocs:3000048406",
                        "stock_row_id": "ocs:3000048406:106561",
                        "role": "target/base system",
                        "producer": "HPE",
                        "part_number": "P71674-425",
                        "item_name": "HPE ProLiant DL380 Gen11",
                        "quantity": 3,
                        "quantity_value": 1,
                        "quantity_is_greater_than": True,
                        "stock_confirmation_required": True,
                        "unit_price_value": "9700.00",
                        "unit_price_currency": "USD",
                        "line_total_value": "29100.00",
                        "line_total_currency": "USD",
                    }
                ],
                "quote_integrity": {
                    "adjustments": [
                        {
                            "type": "stock_lower_bound_quantity_confirm",
                            "component_candidate_id": "ocs:3000048406",
                            "stock_row_id": "ocs:3000048406:106561",
                            "displayed_available_quantity": 1,
                            "included_quantity": 3,
                        }
                    ]
                },
            },
        }
    )

    assert "Перед отправкой проверить:" in text
    assert "P71674-425" in text
    assert "в КП 3 шт." in text
    assert "склад показывает 1+ шт." in text
    assert "Подтвердить доступность выбранного количества." in text


def test_choose_v3_full_category_quote_delivery_keeps_long_result_as_compact_message() -> None:
    payload = {
        "profile": "server",
        "category_ids": ["V1100"],
        "result_state": "quote_draft_review_required",
        "engineering_review_required": True,
        "diagnostics": {
            "matrix_row_count": 561,
            "model": "qwen/qwen3.7-plus",
        },
        "validated_quote": {
            "total_price_value": "8727.5000",
            "total_price_currency": "USD",
            "why_selected": "Long explanation. " * 120,
            "compatibility_check": {
                "status": "compatible",
                "checked_facts": ["Selected rows are accounted for. " * 80],
                "blocking_mismatches": [],
                "unresolved_risks": ["Needs engineering review. " * 20],
            },
            "engineer_checks": ["Confirm heatsink compatibility before quote. " * 8],
            "lines": [
                {
                    "role": "platform",
                    "producer": "Gooxi",
                    "part_number": "0.96.001.0003",
                    "item_name": "Server platform",
                    "quantity": 1,
                    "unit_price_value": "2300.0000",
                    "unit_price_currency": "USD",
                    "line_total_value": "2300.0000",
                    "line_total_currency": "USD",
                    "reason": "Line reason. " * 80,
                }
            ],
        },
    }

    delivery = choose_v3_full_category_quote_delivery(payload, safe_limit=500)

    assert delivery.mode == "message"
    assert delivery.text is not None
    assert len(delivery.text) <= 500
    assert "КП draft" in delivery.text
    assert "Перед отправкой проверить:" in delivery.text
    assert "Needs engineering review" in delivery.text
    assert "Полная детализация отправлена в Excel-файле." in delivery.text
    assert delivery.filename is None
    assert delivery.content is None


def test_format_v3_full_category_quote_shows_deviation_notes() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "server",
            "match_run_id": 188,
            "category_ids": ["V1100"],
            "result_state": "quote_draft_review_required",
            "engineering_review_required": True,
            "diagnostics": {"matrix_row_count": 538},
            "validated_quote": {
                "total_price_value": "8121.3000",
                "total_price_currency": "USD",
                "why_selected": "Выбран ближайший складской аналог.",
                "deviation_notes": [
                    (
                        "Исходное требование: HPE DL380 Gen12; выбрано: "
                        "складская платформа Gen11; класс: слабее/другое; "
                        "влияние: требуется согласование замены поколения."
                    )
                ],
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": ["selected rows are coherent"],
                    "blocking_mismatches": [],
                    "unresolved_risks": [],
                },
                "lines": [
                    {
                        "role": "platform",
                        "producer": "HPE",
                        "part_number": "DL380-G11",
                        "item_name": "Server platform",
                        "quantity": 1,
                        "unit_price_value": "8121.3000",
                        "unit_price_currency": "USD",
                        "line_total_value": "8121.3000",
                        "line_total_currency": "USD",
                    }
                ],
            },
        }
    )

    assert "Отклонения от ТЗ:" in text
    assert "HPE DL380 Gen12" in text
    assert "требуется согласование замены поколения" in text
    assert "лучший доступный складской аналог" in text


def test_format_v3_full_category_quote_shows_v6_partial_gaps() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "server",
            "match_run_id": 193,
            "category_ids": ["V1100"],
            "result_state": "quote_draft_review_required",
            "engineering_review_required": True,
            "diagnostics": {"matrix_row_count": 538},
            "validated_quote": {
                "client_status_label": "Ближайший складской вариант требует добора",
                "selection_mode": "partial_build",
                "completeness_status": "partial",
                "operational_status": "incomplete_needs_procurement",
                "total_price_value": "100.0000",
                "total_price_currency": "USD",
                "client_summary": "Можно поставить базовую платформу со склада.",
                "coverage_summary": "База закрыта, CPU нужно добрать.",
                "key_deviations": [
                    {
                        "requirement_id": "R2",
                        "requested": "2 CPU",
                        "offered": "не выбрано",
                        "direction": "different",
                        "severity": "material",
                        "impact": "Без CPU система неполная.",
                        "reason": "Совместимый CPU не подтвержден по матрице.",
                    }
                ],
                "procurement_gaps": [
                    {
                        "requirement_id": "R2",
                        "role": "cpu",
                        "requested": "2 CPU",
                        "status": "no_compatible_item_proven",
                        "required_for": "operational_readiness",
                        "impact": "Нужно добрать CPU.",
                        "next_action": "Запросить CPU под выбранную базу.",
                    }
                ],
                "compatibility_check": {
                    "status": "compatible_selected_lines",
                    "checked_facts": ["stock-row-1 quoted as anchor"],
                    "blocking_mismatches": [],
                    "selected_line_conflicts": [],
                    "unresolved_risks": [],
                },
                "lines": [
                    {
                        "role": "platform",
                        "producer": "Vendor",
                        "part_number": "BASE-1",
                        "item_name": "Server platform",
                        "quantity": 1,
                        "unit_price_value": "100.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "100.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "engineer_checks": ["Проверить CPU перед финальным КП."],
            },
        }
    )

    assert "Тип предложения: Ближайший складской вариант требует добора" in text
    assert "Покрытие ТЗ: База закрыта, CPU нужно добрать." in text
    assert "Что нужно добрать или согласовать:" in text
    assert "совместимый вариант не подтвержден по матрице" in text
    assert "Он не закрывает ТЗ полностью" in text
    assert "частичная складская сборка" in text


def test_format_v3_excel_caption_keeps_quote_context_when_card_fails() -> None:
    caption = format_v3_excel_caption(
        {
            "profile": "server",
            "match_run_id": 195,
            "category_ids": ["V1100"],
            "result_state": "quote_draft_review_required",
            "pipeline_version": "v3_full_category_matrix",
            "validated_quote": {
                "client_status_label": "Ближайший складской вариант требует добора",
                "total_price_value": "34282.0000",
                "total_price_currency": "USD",
                "client_summary": (
                    "Предложен сервер HPE ProLiant DL380 Gen11 как ближайший "
                    "доступный аналог DL380 Gen12."
                ),
                "coverage_summary": "Диски SAS и HBA закрыты, iLO и рельсы нужно добрать.",
                "target_decisions": [
                    {
                        "target_label": "HPE DL380 Gen11",
                        "anchor_status": "selected",
                        "anchor_line_id": "L1",
                        "reason": "Ближайшая складская база.",
                    }
                ],
                "key_deviations": [
                    {
                        "requirement_id": "R2",
                        "requested": "HPE DL380 Gen12",
                        "offered": "HPE DL380 Gen11",
                        "direction": "downgrade",
                        "severity": "material",
                        "impact": "Предыдущее поколение платформы.",
                        "reason": "Gen12 нет в матрице.",
                    }
                ],
                "procurement_gaps": [
                    {
                        "role": "iLO",
                        "requested": "iLO 6 Advanced",
                        "status": "not_in_matrix",
                        "required_for": "requested_spec",
                        "impact": "Нужно заказать отдельно.",
                    }
                ],
            },
        },
        match_run_id=195,
    )

    assert "Подбор по складу №195" in caption
    assert "Итого: 34 282 USD" in caption
    assert "Ближайший складской вариант требует добора" in caption
    assert "HPE ProLiant DL380 Gen11" in caption
    assert "Покрытие:" in caption
    assert "Объект: HPE DL380 Gen11 - выбран - строка L1" in caption
    assert "Отличие:" in caption
    assert "Добрать:" in caption
    assert "Полная детализация в Excel." in caption
    assert len(caption) <= 1000


def test_format_v3_full_category_quote_russianizes_english_no_recommendation() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "custom",
            "match_run_id": 170,
            "category_ids": ["cat-" + str(index) for index in range(107)],
            "result_state": "no_recommendation",
            "distributor_code": "treolan",
            "diagnostics": {"matrix_row_count": 696},
            "no_recommendation_reason": {
                "summary": (
                    "The exact CPU model requested (Intel Xeon Silver 4310) is not "
                    "available in the provided distributor matrix."
                ),
                "failed_requirements": [
                    "CPU model: Intel Xeon Silver 4310 (12 cores) is missing from the matrix."
                ],
                "recommended_next_actions": [
                    "Confirm if an alternative CPU model available in the matrix is acceptable."
                ],
            },
        }
    )

    assert "КП не сформировано: указанная модель процессора" in text
    assert "Не найдено в матрице склада: процессор Intel Xeon Silver 4310" in text
    assert "Разрешить ближайший технически подходящий аналог" in text
    assert "категорий 107" in text
    assert "cat-1, cat-2" not in text
    assert "The exact CPU model requested" not in text


def test_format_v3_full_category_quote_formats_structured_failed_requirements() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "server",
            "match_run_id": 192,
            "category_ids": ["V1100"],
            "result_state": "no_recommendation",
            "distributor_code": "ocs",
            "diagnostics": {"matrix_row_count": 538},
            "no_recommendation_reason": {
                "summary": "КП не сформировано: нет валидного аналога.",
                "failed_requirements": [
                    {
                        "requirement_id": "R1",
                        "requirement": "Сервер HPE DL380 Gen12",
                        "reason": "Поколение Gen12 отсутствует на складе.",
                    }
                ],
                "recommended_next_actions": ["Согласовать ближайший складской аналог."],
            },
        }
    )

    assert "Требование: R1 Сервер HPE DL380 Gen12" in text
    assert "причина: Поколение Gen12 отсутствует на складе." in text
    assert "{'requirement_id'" not in text
    assert '"requirement_id"' not in text


def test_format_v3_full_category_quote_marks_unresolved_risks_as_preliminary() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "server",
            "category_ids": ["V1100"],
            "result_state": "quote_draft_review_required",
            "engineering_review_required": True,
            "diagnostics": {
                "matrix_row_count": 559,
                "model": "qwen/qwen3.7-plus",
            },
            "validated_quote": {
                "total_price_value": "19397.0000",
                "total_price_currency": "USD",
                "why_selected": "Commercially reasonable server draft.",
                "price_audit": [
                    (
                        "Rejected cheaper platform row ocs:cheap because matrix "
                        "did not prove CPU generation support."
                    )
                ],
                "compatibility_check": {
                    "status": "compatible",
                    "checked_facts": ["Platform and CPU rows both mention LGA4677."],
                    "blocking_mismatches": [],
                    "unresolved_risks": ["Confirm U.2 backplane cabling."],
                },
                "lines": [
                    {
                        "role": "platform",
                        "producer": "Vandor",
                        "part_number": "P4-P222412N2S-16",
                        "item_name": "Server platform",
                        "quantity": 1,
                        "unit_price_value": "3897.0000",
                        "unit_price_currency": "USD",
                        "line_total_value": "3897.0000",
                        "line_total_currency": "USD",
                    }
                ],
                "engineer_checks": ["Confirm U.2 backplane cabling."],
            },
        }
    )

    assert "Риск: Confirm U.2 backplane cabling" in text
    assert "предварительный вариант КП" in text
    assert "технически приемлемый вариант" not in text


def test_format_v3_full_category_quote_humanizes_mechanical_failure() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "server",
            "category_ids": ["V1100"],
            "result_state": "mechanical_validation_failed",
            "diagnostics": {
                "matrix_row_count": 559,
                "model": "qwen/qwen3.7-plus",
            },
            "no_recommendation_reason": {
                "summary": "LLM quote failed mechanical checks.",
                "failed_requirements": [
                    "line_2:quantity_exceeds_stock",
                    "stock_row_overallocated:ocs:3000033081:4303",
                    "total_price_mismatch",
                ],
            },
            "validation_errors": [
                "line_2:quantity_exceeds_stock",
                "stock_row_overallocated:ocs:3000033081:4303",
                "total_price_mismatch",
            ],
        }
    )

    assert "LLM собрала вариант, но он не прошел проверку склада" in text
    assert "Что не сошлось:" in text
    assert "выбрано больше единиц" in text
    assert "итоговая сумма не совпала" in text
    assert "Что не закрыто:" not in text
    assert "mechanical checks" not in text
    assert "quantity_exceeds_stock" not in text


def test_format_v3_full_category_quote_explains_empty_matrix_state() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "custom",
            "category_ids": ["cat-empty"],
            "result_state": "matrix_empty_after_category_selection",
            "diagnostics": {
                "matrix_row_count": 0,
                "model": "qwen/qwen3.7-plus",
            },
            "no_recommendation_reason": {
                "summary": "Selected v3 category matrix has no stocked/priced rows.",
                "failed_requirements": ["matrix_empty_after_category_selection"],
            },
        }
    )

    assert "в выбранной категории нет складских строк с ценой" in text
    assert "матрица 0 строк" in text
    assert "Selected v3 category matrix has no stocked/priced rows" not in text


def test_format_v3_full_category_quote_explains_stock_refresh_failure() -> None:
    text = format_v3_full_category_quote(
        {
            "profile": "server",
            "category_ids": ["V1100", "V1101"],
            "result_state": "stock_refresh_failed",
            "diagnostics": {
                "model": "qwen/qwen3.7-plus",
                "stock_refresh": {
                    "enabled": True,
                    "status": "failed",
                    "error_message": "Treolan API returned empty response.",
                },
            },
            "no_recommendation_reason": {
                "summary": (
                    "Selected distributor stock could not be refreshed before the "
                    "paid v3 Composer call."
                ),
                "fallback_reason": "v3_stock_refresh_failed",
                "details": "Treolan API returned empty response.",
            },
        }
    )

    assert "не удалось обновить склад перед КП" in text
    assert "LLM Composer не запускался" in text
    assert "категорий 2" in text
    assert "Служебно:" in text
    assert "не удалось закрыть требования" not in text
    assert "Selected distributor stock could not be refreshed" not in text
    assert "Treolan API returned empty response" not in text


def test_pluralize_ru_recommendation_counts() -> None:
    assert _shown_safe_variants_message(1).startswith("Показан 1 безопасный вариант.")
    assert _shown_safe_variants_message(2).startswith("Показаны 2 безопасных варианта.")
    assert _shown_safe_variants_message(3).startswith("Показаны 3 безопасных варианта.")
    assert _shown_safe_variants_message(5).startswith("Показано 5 безопасных вариантов.")
    duplicate_message = _shown_safe_variants_message(
        1,
        {
            "llm_proposals_count": 10,
            "rejected_ai_recommendations_count": 9,
            "selection_skipped_count": 4,
            "ai_validation_summary": {
                "rejected_fatal": 3,
                "rejected_missing_required": 2,
                "selection_skipped_duplicate": 4,
            },
        },
    )
    assert duplicate_message.startswith("Показан 1 безопасный вариант.")
    assert "уступили по цене/рискам" in duplicate_message
    assert _shown_safe_variants_message(
        2,
        {"llm_proposals_count": 10, "rejected_ai_recommendations_count": 8},
    ).startswith("Показаны 2 безопасных варианта.")
    assert pluralize_ru(21, "вариант", "варианта", "вариантов") == "вариант"
    assert pluralize_ru(22, "вариант", "варианта", "вариантов") == "варианта"


def test_format_match_summary_uses_russian_labels_without_api_labels() -> None:
    summary = {
        "match_run_id": 42,
        "status": "partial_stock_matched",
        "engineer_review_required": True,
        "total_candidates": 2,
        "matched_items": 0,
        "risk_flags": ["engineer_review_required"],
        "missing_requirements": ["RAM ниже требования: найдено 64 GB, требуется 512 GB"],
        "candidates": [
            {
                "producer": "NERPA",
                "part_number": "D5720-181125SA04",
                "item_id": "1000841882",
                "available_quantity": 3,
                "price_value": "6900",
                "price_currency": "USD",
            },
            {
                "producer": "NERPA",
                "part_number": "D5720-181125SA05",
                "item_id": "1000841883",
                "available_quantity": 1,
                "price_value": "6800",
                "price_currency": "USD",
            },
        ],
    }

    text = format_match_summary(summary)

    assert "Подбор по складу №42" in text
    assert "Итог: найдены варианты, но полное соответствие не подтверждено" in text
    assert "Найдены 2 варианта" in text
    assert "Полностью подходят: 0" in text
    assert "Нужна проверка инженера: да" in text
    assert "Готовые варианты" in text
    assert "Сборка из комплектующих" in text
    assert (
        "пока не предложена - нет достаточных складских данных по платформам/комплектующим"
        in text
    )
    assert "Что нужно проверить" in text
    assert "NERPA D5720-181125SA04" in text
    assert "NERPA D5720-181125SA05" in text
    assert (
        "Оперативная память ниже требования: найдено 64 ГБ, требуется 512 ГБ."
        in text
    )
    assert "Это предварительный подбор по складу" in text

    forbidden_labels = [
        "status",
        "candidates",
        "matched_items",
        "engineer_review_required",
        "partial_stock_matched",
        "Stock Spec",
        "Match Engine V0",
        "gaps",
        "build_from_parts",
        "ready_server",
    ]
    for label in forbidden_labels:
        assert label not in text


def test_format_match_summary_shows_build_candidates_without_technical_labels() -> None:
    summary = {
        "match_run_id": 43,
        "status": "partial_stock_matched",
        "engineer_review_required": True,
        "total_candidates": 1,
        "matched_items": 0,
        "risk_flags": [],
        "missing_requirements": [],
        "ready_stock_candidates": [],
        "build_candidates": [
            {
                "candidate_type": "build_from_parts",
                "item_name": "Предварительная сборка на платформе TestVendor PLATFORM-2U",
                "price_value": "6800",
                "price_currency": "USD",
                "total_price_value": "6800",
                "total_price_currency": "USD",
                "completeness_status": "incomplete",
                "completeness_label": "Неполная сборка - требуется подбор CPU.",
                "included_component_roles": ["server_platform", "ram", "ssd"],
                "missing_component_roles": ["cpu"],
                "excluded_from_total_roles": ["cpu"],
                "cpu_per_server": 2,
                "total_cpu_required": 4,
                "total_price_note": "без CPU",
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "TestVendor",
                        "part_number": "PLATFORM-2U",
                        "quantity_required": 2,
                        "available_quantity": 2,
                    },
                    {
                        "role": "ram",
                        "producer": "TestVendor",
                        "part_number": "RAM-64G",
                        "quantity_required": 16,
                        "available_quantity": 16,
                    },
                    {
                        "role": "ssd",
                        "producer": "TestVendor",
                        "part_number": "SSD-960G",
                        "quantity_required": 2,
                        "available_quantity": 2,
                    },
                ],
            }
        ],
        "candidates": [],
    }

    text = format_match_summary(summary)

    assert "Готовые варианты" in text
    assert "- не найдено" in text
    assert "Сборка из комплектующих" in text
    assert "Неполная сборка TestVendor PLATFORM-2U" in text
    assert "Состав: платформа, RAM, SSD; CPU не подобраны" in text
    assert "Ориентировочно за 2 сервера без CPU: 6 800 USD" in text
    assert "TestVendor PLATFORM-2U" in text
    assert "TestVendor RAM-64G" not in text
    assert "TestVendor SSD-960G" not in text
    assert "Нужна инженерная проверка" in text
    assert "build_from_parts" not in text
    assert "server_platform" not in text
    assert "candidate_type" not in text


def test_format_match_summary_shows_compact_build_with_cpu_component() -> None:
    summary = {
        "match_run_id": 44,
        "status": "partial_stock_matched",
        "engineer_review_required": True,
        "total_candidates": 1,
        "matched_items": 0,
        "risk_flags": [],
        "missing_requirements": [],
        "ready_stock_candidates": [],
        "build_candidates": [
            {
                "candidate_type": "build_from_parts",
                "producer": "ASUS",
                "part_number": "90SF03A1-M00070",
                "price_value": "74500",
                "price_currency": "USD",
                "total_price_value": "74500",
                "total_price_currency": "USD",
                "completeness_status": "complete",
                "included_component_roles": ["server_platform", "cpu", "ram", "ssd"],
                "missing_component_roles": [],
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "90SF03A1-M00070",
                        "quantity_required": 2,
                        "available_quantity": 2,
                    },
                    {
                        "role": "cpu",
                        "producer": "Intel",
                        "part_number": "CPU",
                        "quantity_required": 4,
                    },
                    {
                        "role": "ram",
                        "producer": "Samsung",
                        "part_number": "RAM",
                        "quantity_required": 16,
                    },
                    {
                        "role": "ssd",
                        "producer": "Samsung",
                        "part_number": "SSD",
                        "quantity_required": 2,
                    },
                ],
            }
        ],
        "candidates": [],
    }

    text = format_match_summary(summary)

    assert "Предварительная сборка ASUS 90SF03A1-M00070" in text
    assert "Состав: платформа, CPU, RAM, SSD" in text
    assert "CPU: Intel CPU, всего к подбору: 4 шт." in text
    assert "RAM: 16 модулей всего к подбору" in text
    assert "Ориентировочно за 2 сервера: 74 500 USD" in text
    assert "Samsung RAM" not in text


def test_format_match_summary_uses_valid_llm_recommendations_without_ids() -> None:
    summary = {
        "match_run_id": 45,
        "status": "partial_stock_matched",
        "engineer_review_required": True,
        "total_candidates": 2,
        "matched_items": 0,
        "risk_flags": [
            "Совместимость RAM с платформой требуется проверить инженеру.",
            "Тип RAM не указан; требуется проверка совместимости RAM с платформой.",
            "Требуется инженерная проверка совместимости платформы, RAM, накопителей и адаптеров.",
        ],
        "missing_requirements": [],
        "normalized_requirements": [
            {
                "server_qty": 2,
                "ram_gb_per_server": 512,
                "ram_type_preference": "DDR5",
            }
        ],
        "ready_stock_candidates": [
            {
                "producer": "NERPA",
                "part_number": "READY-SERVER",
                "available_quantity": 1,
                "price_value": "10000",
                "price_currency": "USD",
            }
        ],
        "build_candidates": [
            {
                "candidate_type": "build_from_parts",
                "total_price_value": "9999",
                "total_price_currency": "USD",
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "Rule",
                        "part_number": "RULE-PLATFORM",
                        "quantity_required": 2,
                    }
                ],
            }
        ],
        "llm_configurator_used": True,
        "llm_general_notes": ["CPU choice is preliminary."],
        "llm_recommended_build_candidates": [
            {
                "candidate_type": "build_from_parts",
                "total_price_value": "8800",
                "total_price_currency": "USD",
                "rank_reason": ["Balanced stocked configuration."],
                "right_size_note": "Подбор: минимально подходящий по требованиям",
                "compatibility_warnings": ["Check platform support list."],
                "included_component_roles": ["server_platform", "cpu", "ram", "ssd"],
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "LLM-PLATFORM",
                        "quantity_required": 2,
                        "component_candidate_id": "platform-secret-id",
                    },
                    {
                        "role": "cpu",
                        "producer": "Intel",
                        "part_number": "CPU",
                        "quantity_required": 4,
                    },
                    {
                        "role": "ram",
                        "producer": "Samsung",
                        "part_number": "RAM",
                        "quantity_required": 16,
                        "facts": {"ram_capacity_gb": 64},
                    },
                    {
                        "role": "ssd",
                        "producer": "Samsung",
                        "part_number": "SSD",
                        "quantity_required": 2,
                    },
                ],
            }
        ],
        "candidates": [],
    }

    text = format_match_summary(summary)

    assert "AI-подбор по складу №45" in text
    assert "Найдена 1 AI-рекомендация" in text
    assert "Основа: текущие остатки и цены OCS" in text
    assert "LLM может ошибаться" in text
    assert "Рекомендации" in text
    assert "Показан 1 безопасный вариант." in text
    assert "Тип: сборка из комплектующих" in text
    assert "Сборный сервер - 2 шт." in text
    assert "В составе 1 сервера" in text
    assert "Платформа: ASUS LLM-PLATFORM - 1 шт." in text
    assert "CPU: Intel CPU - 2 шт." in text
    assert "RAM: Samsung RAM - 8 x 64 ГБ на сервер = 512 ГБ" in text
    assert "SSD: Samsung SSD - 1 шт." in text
    assert "Всего к подбору" in text
    assert "Платформа: 2 шт. всего, склад: неизвестно" in text
    assert "CPU: 4 шт. всего, склад: неизвестно" in text
    assert "RAM: 16 шт. всего, модули по 64 ГБ, склад: неизвестно" in text
    assert "SSD: 2 шт. всего, склад: неизвестно" in text
    assert "Ориентировочно за 2 сервера: 8 800 USD" in text
    assert "за 2 сервера за весь запрос" not in text
    assert "на заказ" not in text
    assert "Подбор: минимально подходящий по требованиям" in text
    assert "Почему выбрана: Balanced stocked configuration." in text
    assert "Что проверить инженеру:" in text
    assert "Тип оперативной памяти не указан" not in text
    assert "Проверить CPU по списку поддерживаемых процессоров платформы." in text
    assert "Проверить совместимость NVMe/SATA SSD" not in text
    assert "Требуется инженерная проверка совместимости платформы" not in text
    assert "Совместимость оперативной памяти требуется проверить инженеру" not in text
    checks_block = text.split("Что проверить инженеру:", 1)[1].split(
        "Подробный отчет", 1
    )[0]
    check_lines = [line for line in checks_block.splitlines() if line.startswith("- ")]
    assert len(check_lines) <= 5
    assert len(check_lines) == len(set(check_lines))
    assert "Подробный отчет отправлен Excel-файлом." in text
    assert "Готовые варианты" not in text
    assert "READY-SERVER" not in text
    assert "RULE-PLATFORM" not in text
    assert "CPU choice is preliminary." not in text
    assert "overfit" not in text
    assert "cores" not in text
    assert "fit_label" not in text
    assert "component_candidate_id" not in text
    assert "platform-secret-id" not in text
    assert "\nСборка из комплектующих\n" not in text
    assert "Совместимость оперативная память" not in text
    assert "Тип оперативная память" not in text


def test_format_match_summary_prefers_grouped_presales_output() -> None:
    summary = {
        "match_run_id": 146,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "grouped_presales_mode_used": True,
        "configuration_groups": [_telegram_grouped_presales_group()],
        "quote_recommendation": {
            "for_cheapest_quote": "ASUS PLATFORM-CHEAP - 8 600 USD",
            "for_database_preferred": "Supermicro SYS-621C-TN12R - 10 200 USD",
            "summary": "Minimal cost build meeting all core specs.",
        },
        "ai_recommendations": [
            {
                "title": "SHOULD-NOT-SHOW",
                "components": [],
            }
        ],
    }

    text = format_match_summary(summary)

    assert "AI-подбор по складу №146" in text
    assert "Предварительная спецификация для КП" in text
    assert "Рекомендуемый вариант для самого дешевого КП" not in text
    assert "Сервер в сборе - 2 шт." in text
    assert "Платформа: ASUS PLATFORM-CHEAP - 2 шт., склад 2" in text
    assert "CPU: Intel Xeon Gold 6530 - 2 шт. на сервер / 4 шт. всего, склад 20" in text
    assert (
        "RAM: Micron 32GB DDR5 RDIMM (MTC20F1045S1RC48BA2) - "
        "16 шт. на сервер = 512 ГБ / 32 шт. всего, склад 100"
    ) in text
    assert (
        "SSD: KIOXIA CD8-R 3.84TB U.3 NVMe (KCD8XRUG3T84) - "
        "2 шт. на сервер / 4 шт. всего, склад 20"
    ) in text
    assert "Ориентировочно за 2 сервера: 8 600 USD" in text
    assert "Комментарий:" in text
    assert "Альтернатива спокойнее для инженеров:" in text
    assert "Supermicro SYS-621C-TN12R - 10 200 USD за 2 сервера." in text
    assert "SHOULD-NOT-SHOW" not in text
    assert "Minimal cost" not in text
    assert "Proven" not in text
    assert "Premium platform" not in text
    assert "preliminary_requires_engineer_review" not in text
    assert "why_this_platform:" not in text
    assert "quote_recommendation:" not in text
    assert "component_candidate_id" not in text
    assert "raw JSON" not in text
    assert "llm_rec" not in text


def test_format_match_summary_uses_single_primary_recommendation_by_default() -> None:
    summary = {
        "match_run_id": 149,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "output_mode": "single_best_cost_valid",
        "primary_recommendation_status": "valid",
        "primary_recommendation": _telegram_primary_recommendation(),
        "grouped_presales_mode_used": False,
        "configuration_groups": [_telegram_grouped_presales_group()],
    }

    text = format_match_summary(summary)

    assert "AI-подбор по складу №149" in text
    assert "Предварительная спецификация для КП" in text
    assert "Рекомендуемый вариант для самого дешевого КП" not in text
    assert "Сервер в сборе - 2 шт." in text
    assert "Состав 1 сервера:" in text
    assert "- Платформа: ASUS PLATFORM-CHEAP - 1 шт." in text
    assert "- CPU: Intel Xeon Gold 6530 - 2 шт." in text
    assert "- RAM: Micron 32GB DDR5 RDIMM (MTC20F1045S1RC48BA2) - 16 шт. = 512 ГБ" in text
    assert "- SSD: KIOXIA CD8-R 3.84TB U.3 NVMe (KCD8XRUG3T84) - 2 шт." in text
    assert "Всего к заказу:" in text
    assert "- Платформа: 2 шт., склад 2" in text
    assert "- CPU: 4 шт., склад 20" in text
    assert "- RAM: 32 шт., склад 100" in text
    assert "- SSD: 4 шт., склад 20" in text
    assert "Ориентировочно за 2 сервера: 8 600 USD" in text
    assert "Комментарий:" in text
    assert "Подобрана минимальная по цене складская конфигурация" in text
    assert "Проверить перед КП:" in text
    assert "Почему этот вариант:" not in text
    assert "Конфигурационная база" not in text
    assert "Варианты платформ" not in text
    assert "Еще " not in text
    assert "Альтернатива спокойнее" not in text
    assert "Minimal cost" not in text
    assert "component_candidate_id" not in text
    assert "llm_rec" not in text


def test_format_match_summary_uses_readable_gooxi_platform_model() -> None:
    primary = _telegram_primary_recommendation()
    platform = primary["components"][0]  # type: ignore[index]
    assert isinstance(platform, dict)
    platform.update(
        {
            "producer": "Shenzhen GuoxinHengyu Technology (Gooxi)",
            "part_number": "0.95.002.1070",
            "item_name": (
                "Shenzhen GuoxinHengyu Technology (Gooxi) "
                "Gooxi SL201-D12R-G4 2U server platform"
            ),
        }
    )
    summary = {
        "match_run_id": 151,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "output_mode": "single_best_cost_valid",
        "primary_recommendation_status": "valid",
        "primary_recommendation": primary,
        "grouped_presales_mode_used": False,
        "configuration_groups": [],
    }

    text = format_match_summary(summary)

    assert "Gooxi SL201-D12R-G4 (0.95.002.1070)" in text
    assert "Shenzhen GuoxinHengyu Technology" not in text


def test_format_match_summary_uses_safe_single_best_commercial_summary_lines() -> None:
    summary = {
        "match_run_id": 150,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "output_mode": "single_best_cost_valid",
        "primary_recommendation_status": "valid",
        "primary_recommendation": {
            "candidate_type": "build_from_parts",
            "total_price_value": "62200",
            "total_price_currency": "USD",
        },
        "commercial_summary": {
            "mode": "single_best_cost_valid",
            "lines": [
                "Рекомендуемый вариант для самого дешевого КП",
                "",
                "Сервер в сборе - 2 шт.",
                "Платформа: SAFE SUMMARY PLATFORM - 2 шт., склад 4",
                "CPU: SAFE SUMMARY CPU - 2 шт. на сервер / 4 шт. всего, склад 10",
                "RAM: SAFE SUMMARY RAM - 16 шт. на сервер = 512 ГБ / 32 шт. всего, склад 100",
                "SSD: SAFE SUMMARY SSD - 2 шт. на сервер / 4 шт. всего, склад 16",
                "component_candidate_id: secret-summary-id",
                "llm_rec_1: hidden diagnostic",
                "",
                "Ориентировочно за 2 сервера: 62 200 USD",
                "",
                "Почему этот вариант:",
                "- минимальная цена среди прошедших проверку складских вариантов",
                "- компоненты закрывают требования запроса",
                "- инженерная проверка обязательна",
                "",
                "Что проверить перед КП:",
                "- CPU support list / BIOS",
            ],
        },
        "grouped_presales_mode_used": False,
        "configuration_groups": [_telegram_grouped_presales_group()],
    }

    text = format_match_summary(summary)

    assert "SAFE SUMMARY PLATFORM" in text
    assert "SAFE SUMMARY CPU" in text
    assert "Предварительная спецификация для КП" in text
    assert "Рекомендуемый вариант для самого дешевого КП" not in text
    assert "Комментарий:" in text
    assert "Проверить перед КП:" in text
    assert "Почему этот вариант:" not in text
    assert "Конфигурационная база" not in text
    assert "component_candidate_id" not in text
    assert "secret-summary-id" not in text
    assert "hidden diagnostic" not in text
    assert "preliminary_requires_engineer_review" not in text
    assert "llm_rec" not in text


def test_format_match_summary_grouped_sanitizes_cjk_and_debug_fields() -> None:
    group = _telegram_grouped_presales_group()
    group["group_title"] = "Intel LGA4677 / DDR5 / NVMe с高密度"
    group["why_group_matters"] = "why_group_matters: с高密度 NVMe"
    group["engineer_checks"] = [
        "Проверить совместимость CPU с платформой.",
        "Проверить список поддерживаемых CPU платформы.",
    ]
    platform_options = group["platform_options"]
    platform_options[0]["why_this_platform"] = (
        "why_this_platform: llm_rec_1 с高密度 NVMe; raw JSON; "
        "component_candidate_id: platform-secret-id"
    )
    platform_options[0]["engineer_checks"] = [
        "Проверить совместимость CPU с платформой.",
        "Проверить список поддерживаемых CPU платформы.",
        "Проверить QVL памяти.",
        "Проверить совместимость RAM.",
    ]
    platform_options[1]["engineer_checks"] = [
        "Проверить QVL RAM.",
        "Проверить QVL памяти.",
    ]
    summary = {
        "match_run_id": 147,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "grouped_presales_mode_used": True,
        "configuration_groups": [group],
        "quote_recommendation": {
            "for_cheapest_quote": "ASUS PLATFORM-CHEAP - 8 600 USD",
            "summary": (
                "quote_recommendation: с高密度 NVMe; llm_rec_2; "
                "component_candidate_id: quote-secret"
            ),
        },
    }

    text = format_match_summary(summary)

    assert not contains_cjk_text(text)
    assert "component_candidate_id" not in text
    assert "platform-secret-id" not in text
    assert "quote-secret" not in text
    assert "raw JSON" not in text
    assert "llm_rec" not in text
    checks_block = text.split("Проверить перед КП:", 1)[1].split(
        "Подробный отчет",
        1,
    )[0]
    check_lines = [line for line in checks_block.splitlines() if line.startswith("- ")]
    assert len(check_lines) <= 5
    assert len(check_lines) == len(set(check_lines))
    assert "- CPU support list / BIOS" in check_lines
    assert "- QVL RAM и правила заполнения DIMM" in check_lines


def test_format_match_summary_repair_notice_hides_raw_diagnostics() -> None:
    summary = {
        "match_run_id": 148,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "grouped_presales_mode_used": True,
        "configuration_groups": [_telegram_grouped_presales_group()],
        "quote_recommendation": {
            "for_cheapest_quote": "Gooxi GOOXI-CHEAP - 48 800 USD",
            "summary": "safe summary",
        },
        "llm_repair_used": True,
        "llm_repair_success": True,
        "llm_repair_critique_summary": [
            "component_candidate_id ram-secret raw JSON headers Authorization"
        ],
        "llm_repair_blocked_critique_summary": [
            "blocked_candidate_id platform-secret raw JSON headers Authorization"
        ],
        "llm_repair_fallback_reason": "should-not-show",
    }

    text = format_match_summary(summary)

    assert "component_candidate_id" not in text
    assert "ram-secret" not in text
    assert "raw JSON" not in text
    assert "Authorization" not in text
    assert "blocked_candidate_id" not in text
    assert "platform-secret" not in text
    assert "should-not-show" not in text


def test_format_match_summary_mentions_evidence_only_when_used() -> None:
    recommendation = {
        "source_type": "build_from_parts",
        "title": "Evidence checked build",
        "total_price_value": "1000",
        "total_price_currency": "USD",
        "components": [
            {
                "role": "server_platform",
                "producer": "ASUS",
                "part_number": "RS720-E11-RS24U",
                "quantity_required": 1,
                "available_quantity": 1,
            }
        ],
        "evidence_summary": {
            "sources_count": 2,
            "missing": [],
            "fatal_concerns": [],
        },
        "why_selected_short": "Подходит по цене и наличию.",
    }
    summary = {
        "match_run_id": 90,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "web_evidence_pack": {
            "enabled": True,
            "completed_tasks": 2,
        },
        "ai_recommendations": [recommendation],
    }

    text = format_match_summary(summary)
    without_evidence_text = format_match_summary(
        {
            **summary,
            "match_run_id": 91,
            "web_evidence_pack": {"enabled": False, "completed_tasks": 0},
            "ai_recommendations": [{**recommendation, "evidence_summary": {}}],
        }
    )

    assert "Проверка: найдены внешние источники по выбранным конфигурациям" in text
    assert "Доказательная проверка: явных конфликтов по найденным источникам не выявлено." in text
    assert "Проверка: найдены внешние источники" not in without_evidence_text


def test_format_match_summary_splits_confidence_without_evidence() -> None:
    recommendation = {
        "source_type": "build_from_parts",
        "title": "Preliminary high commercial fit",
        "total_price_value": "1000",
        "total_price_currency": "USD",
        "confidence": "high",
        "commercial_fit_confidence": "high",
        "evidence_summary": {
            "status": "disabled",
            "sources_count": 0,
            "confidence": "unknown",
        },
        "why_selected_short": "Коммерчески подходит по цене и наличию.",
        "components": [
            {
                "role": "server_platform",
                "producer": "Gooxi",
                "part_number": "PLATFORM",
                "quantity_required": 1,
                "available_quantity": 1,
            }
        ],
    }

    text = format_match_summary(
        {
            "match_run_id": 93,
            "llm_configurator_enabled": True,
            "llm_configurator_used": True,
            "web_evidence_pack": {"enabled": False},
            "ai_recommendations": [recommendation],
        }
    )

    assert "Коммерческое соответствие: высокое" in text
    assert "инженерная подтвержденность: предварительно, требуется проверка" in text
    assert "инженерная подтвержденность: проверено" not in text


def test_format_match_summary_hides_internal_evidence_messages() -> None:
    summary = {
        "match_run_id": 92,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "web_evidence_pack": {
            "enabled": True,
            "total_tasks": 4,
            "completed_tasks": 0,
            "error_count": 0,
        },
        "ai_recommendations": [
            {
                "source_type": "build_from_parts",
                "title": "Evidence not confirmed build",
                "total_price_value": "1000",
                "total_price_currency": "USD",
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "Gooxi",
                        "part_number": "0.95.002.0103",
                        "quantity_required": 1,
                        "available_quantity": 1,
                    }
                ],
                "evidence_summary": {
                    "status": "not_found",
                    "sources_count": 0,
                    "missing": ["источник не найден"],
                    "fatal_concerns": [],
                },
                "critical_checks": [
                    "Llm_rec_1: web evidence not found for Платформа; keep engineer проверить"
                ],
                "why_selected_short": "Подходит по цене и наличию.",
            }
        ],
    }

    text = format_match_summary(summary)

    assert (
        "Внешние источники не подтвердили совместимость выбранных связок; "
        "инженерная проверка обязательна."
    ) in text
    assert "Доказательная проверка: совместимость не подтверждена" not in text
    assert "Llm_rec_" not in text
    assert "web evidence not found" not in text
    assert "keep engineer" not in text


def test_format_match_summary_online_composer_no_sources_uses_single_notice() -> None:
    recommendation = {
        "source_type": "build_from_parts",
        "title": "No sources build",
        "quantity_required": 2,
        "total_price_value": "8800",
        "total_price_currency": "USD",
        "why_selected_short": "Закрывает требования по складу.",
        "evidence_summary": {
            "status": "not_confirmed",
            "sources_count": 0,
            "not_confirmed": ["support list"],
            "source_domains": [],
        },
        "components": [
            {
                "role": "server_platform",
                "producer": "ASUS",
                "part_number": "PLATFORM",
                "quantity_required": 2,
                "available_quantity": 2,
            }
        ],
    }
    summary = {
        "match_run_id": 95,
        "llm_configurator_used": True,
        "web_evidence_pack": {
            "enabled": True,
            "diagnostics": {
                "evidence_mode": "online_composer",
                "evidence_sources_count": 0,
            },
        },
        "ai_recommendations": [recommendation, {**recommendation, "title": "No sources 2"}],
    }

    text = format_match_summary(summary)

    assert (
        text.count(
            "Внешние источники не подтвердили совместимость выбранных связок; "
            "инженерная проверка обязательна."
        )
        == 1
    )
    assert "Доказательная проверка: совместимость не подтверждена" not in text


def test_format_match_summary_not_confirmed_relation_hides_internal_labels() -> None:
    recommendation = {
        "source_type": "build_from_parts",
        "title": "Normal not confirmed build",
        "quantity_required": 1,
        "total_price_value": "1200",
        "total_price_currency": "USD",
        "why_selected_short": "Needs engineer review.",
        "evidence_summary": {
            "status": "not_confirmed",
            "sources_count": 0,
            "not_confirmed": ["CPU support list was not found."],
            "engineering_checks": ["Check the platform CPU support list with an engineer."],
            "relation_evidence": [
                {
                    "relation_type": "platform_cpu",
                    "recommendation_id": "llm_rec_1",
                    "status": "not_confirmed",
                    "component_candidate_id": "platform-secret-id",
                    "sources_count": 0,
                    "missing": ["CPU support list was not found."],
                }
            ],
        },
        "components": [
            {
                "role": "server_platform",
                "producer": "ASUS",
                "part_number": "PLATFORM",
                "quantity_required": 1,
                "available_quantity": 1,
            }
        ],
    }
    summary = {
        "match_run_id": 98,
        "llm_configurator_used": True,
        "web_evidence_pack": {
            "enabled": True,
            "diagnostics": {
                "evidence_mode": "online_composer",
                "evidence_sources_count": 0,
                "evidence_status_summary": {"not_confirmed": 1},
            },
        },
        "ai_recommendations": [recommendation],
    }

    text = format_match_summary(summary)

    assert "status=error" not in text
    assert "error" not in text.casefold()
    assert "component_candidate_id" not in text
    assert "platform-secret-id" not in text
    assert "llm_rec_" not in text
    assert "raw JSON" not in text
    assert "provider" not in text.casefold()


def test_format_match_summary_relation_partially_confirmed_notice() -> None:
    recommendation = {
        "source_type": "build_from_parts",
        "title": "Relation partial build",
        "quantity_required": 2,
        "total_price_value": "8800",
        "total_price_currency": "USD",
        "why_selected_short": "Подходит по складу.",
        "evidence_summary": {
            "status": "partially_confirmed",
            "sources_count": 2,
            "confirmed": ["DDR5", "NVMe"],
            "not_confirmed": ["CPU support list"],
            "relation_evidence": [
                {"relation_type": "platform_cpu", "status": "partially_confirmed"}
            ],
        },
        "components": [
            {
                "role": "server_platform",
                "producer": "ASUS",
                "part_number": "PLATFORM",
                "quantity_required": 2,
                "available_quantity": 2,
            }
        ],
    }
    summary = {
        "match_run_id": 97,
        "llm_configurator_used": True,
        "web_evidence_pack": {"enabled": True, "diagnostics": {"evidence_sources_count": 2}},
        "ai_recommendations": [recommendation],
    }

    text = format_match_summary(summary)

    assert "часть совместимости подтверждена внешними источниками" in text
    assert "support list CPU/RAM нужно сверить инженеру" in text


def test_format_match_summary_online_composer_confirmed_facts_are_concise() -> None:
    summary = {
        "match_run_id": 96,
        "llm_configurator_used": True,
        "web_evidence_pack": {"enabled": True, "diagnostics": {"evidence_sources_count": 3}},
        "ai_recommendations": [
            {
                "source_type": "build_from_parts",
                "title": "Confirmed build",
                "quantity_required": 2,
                "total_price_value": "8800",
                "total_price_currency": "USD",
                "why_selected_short": "Закрывает требования по складу.",
                "evidence_summary": {
                    "status": "confirmed",
                    "sources_count": 3,
                    "confirmed": ["DDR5", "LGA4677", "NVMe"],
                    "missing": [],
                },
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "PLATFORM",
                        "quantity_required": 2,
                        "available_quantity": 2,
                    }
                ],
            }
        ],
    }

    text = format_match_summary(summary)

    assert "Доказательная проверка: подтверждено DDR5, LGA4677, NVMe" in text
    assert "http" not in text.casefold()


def test_format_match_summary_uses_only_selected_recommendations_for_stock_checks() -> None:
    selected = {
        "source_type": "build_from_parts",
        "title": "Selected stocked build",
        "quantity_required": 2,
        "total_price_value": "8800",
        "total_price_currency": "USD",
        "why_selected_short": "Закрывает требования по складу.",
        "components": [
            {
                "role": "server_platform",
                "producer": "ASUS",
                "part_number": "PLATFORM",
                "quantity_required": 2,
                "available_quantity": 2,
            },
            {
                "role": "ram",
                "producer": "Samsung",
                "part_number": "RAM",
                "quantity_required": 16,
                "available_quantity": 16,
            },
        ],
    }
    summary = {
        "match_run_id": 93,
        "llm_configurator_used": True,
        "risk_flags": ["По одному варианту не хватает остатка: доступно 2 шт., требуется 16 шт."],
        "missing_requirements": [],
        "ai_recommendations": [selected],
    }

    text = format_match_summary(summary)

    assert "По одному варианту не хватает остатка" not in text
    assert "доступно 2 шт., требуется 16 шт." not in text


def test_format_match_summary_shows_stock_check_for_selected_recommendation() -> None:
    selected = {
        "source_type": "build_from_parts",
        "title": "Selected short-stock build",
        "quantity_required": 2,
        "total_price_value": "8800",
        "total_price_currency": "USD",
        "why_selected_short": "Закрывает требования по складу.",
        "compatibility_warnings": [
            "RAM: остаток ниже требования, доступно 2 шт., требуется 16 шт."
        ],
        "components": [
            {
                "role": "server_platform",
                "producer": "ASUS",
                "part_number": "PLATFORM",
                "quantity_required": 2,
                "available_quantity": 2,
            },
            {
                "role": "ram",
                "producer": "Samsung",
                "part_number": "RAM",
                "quantity_required": 16,
                "available_quantity": 2,
            },
        ],
    }
    summary = {
        "match_run_id": 94,
        "llm_configurator_used": True,
        "risk_flags": [],
        "missing_requirements": [],
        "ai_recommendations": [selected],
    }

    text = format_match_summary(summary)

    assert "Что проверить инженеру:" in text
    assert "По одному варианту не хватает остатка: доступно 2 шт., требуется 16 шт." in text


def test_format_match_summary_shows_ai_recommendation_for_duplicate_valid_pool() -> None:
    summary = {
        "match_run_id": 55,
        "llm_configurator_enabled": True,
        "llm_configurator_used": True,
        "ai_recommendation_mode": "ai_success",
        "llm_proposals_count": 5,
        "valid_proposals_count": 5,
        "ai_recommendations_count": 1,
        "rejected_ai_recommendations_count": 4,
        "selection_skipped_count": 4,
        "ai_validation_summary": {
            "accepted": 1,
            "accepted_after_validation": 5,
            "validation_rejected_count": 0,
            "selection_skipped_count": 4,
            "selection_skipped_duplicate": 4,
        },
        "ai_recommendations": [
            {
                "candidate_type": "build_from_parts",
                "total_price_value": "8800",
                "total_price_currency": "USD",
                "right_size_note": "Подбор: минимально подходящий по требованиям",
                "compatibility_warnings": ["Проверить CPU support list."],
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "LLM-PLATFORM",
                        "quantity_required": 2,
                    },
                    {
                        "role": "cpu",
                        "producer": "Intel",
                        "part_number": "CPU",
                        "quantity_required": 4,
                    },
                ],
            }
        ],
    }

    text = format_match_summary(summary)

    assert text.startswith("AI-подбор по складу №55")
    assert "Рекомендации" in text
    assert "Показан 1 безопасный вариант." in text
    assert "Остальные AI-варианты были отклонены валидатором" in text
    assert "AI не смог сформировать безопасные рекомендации" not in text
    assert "Готовые варианты" not in text
    assert "\nСборка из комплектующих\n" not in text


def test_format_match_summary_separates_llm_recommendations_with_blank_lines() -> None:
    base_candidate = {
        "source_type": "build_from_parts",
        "quantity_required": 2,
        "total_price_value": "8800",
        "total_price_currency": "USD",
        "why_selected_short": "Закрывает требования по складу.",
        "right_size_note": (
            "Подбор: CPU выше требования, требуется проверить альтернативы "
            "в матрице компонентов."
        ),
        "components": [
            {
                "role": "server_platform",
                "producer": "ASUS",
                "part_number": "PLATFORM",
                "quantity_required": 2,
                "available_quantity": 7,
            },
            {
                "role": "cpu",
                "producer": "Intel",
                "part_number": "CPU",
                "quantity_required": 4,
                "available_quantity": 10,
            },
        ],
    }
    summary = {
        "match_run_id": 55,
        "llm_configurator_used": True,
        "risk_flags": [],
        "missing_requirements": [],
        "ai_recommendations": [
            {**base_candidate, "title": "Оптимальный по цене вариант"},
            {**base_candidate, "title": "Резервный складской вариант"},
        ],
    }

    text = format_match_summary(summary)

    assert "\n   Сборный сервер - 2 шт.\n\n   В составе 1 сервера:" in text
    assert "\n   - CPU: Intel CPU - 2 шт.\n\n   Всего к подбору:" in text
    assert "\n   Нужна инженерная проверка\n\n2." in text
    assert "на заказ" not in text


def test_format_match_summary_does_not_cut_why_selected_mid_sentence() -> None:
    summary = {
        "match_run_id": 48,
        "llm_configurator_used": True,
        "risk_flags": [],
        "missing_requirements": [],
        "ai_recommendations": [
            {
                "source_type": "build_from_parts",
                "title": "Оптимальный по цене вариант",
                "display_name": "ASUS PLATFORM",
                "total_price_value": "8800",
                "total_price_currency": "USD",
                "quantity_required": 2,
                "why_selected": (
                    "Первое предложение полностью объясняет выбор. "
                    "Второе предложение очень длинное и не должно попасть в Telegram "
                    "обрезанным посередине слова Intel."
                ),
                "right_size_note": "Подбор: минимально подходящий по требованиям",
                "component_summary": {"cpu": "Intel CPU", "ram": "512 ГБ", "storage": "SSD"},
            }
        ],
    }

    text = format_match_summary(summary)

    assert "Почему выбрана: Первое предложение полностью объясняет выбор." in text
    assert "Inte..." not in text


def test_format_match_summary_prefers_why_selected_short() -> None:
    summary = {
        "match_run_id": 50,
        "llm_configurator_used": True,
        "risk_flags": [],
        "missing_requirements": [],
        "ai_recommendations": [
            {
                "source_type": "build_from_parts",
                "title": "Оптимальный вариант",
                "quantity_required": 2,
                "total_price_value": "8800",
                "total_price_currency": "USD",
                "why_selected": "Длинное объяснение не должно попасть в Telegram.",
                "why_selected_short": "Коротко закрывает основные требования.",
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "PLATFORM",
                        "quantity_required": 2,
                    }
                ],
            }
        ],
    }

    text = format_match_summary(summary)

    assert "Почему выбрана: Коротко закрывает основные требования." in text
    assert "Длинное объяснение" not in text


def test_format_match_summary_truncates_why_selected_on_word_boundary() -> None:
    long_text = " ".join(["Очень длинное объяснение"] * 20)
    summary = {
        "match_run_id": 52,
        "llm_configurator_used": True,
        "risk_flags": [],
        "missing_requirements": [],
        "ai_recommendations": [
            {
                "source_type": "build_from_parts",
                "title": "Вариант",
                "quantity_required": 2,
                "total_price_value": "8800",
                "total_price_currency": "USD",
                "why_selected": long_text,
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "PLATFORM",
                        "quantity_required": 2,
                    }
                ],
            }
        ],
    }

    text = format_match_summary(summary)
    why_line = next(line for line in text.splitlines() if "Почему выбрана:" in line)

    assert why_line.endswith("...")
    assert "объясне..." not in why_line


def test_make_telegram_sentence_avoids_weak_mid_phrase_truncation() -> None:
    text = (
        "Подбор: CPU выше требования при достаточном складском остатке и более низкой "
        "цене относительно альтернатив, но итоговую совместимость нужно проверить."
    )

    sentence = make_telegram_sentence(text, max_chars=49)

    assert sentence.endswith("...")
    assert "при достаточном..." not in sentence
    assert "точ..." not in sentence


def test_make_telegram_sentence_falls_back_for_empty_or_noisy_text() -> None:
    assert (
        make_telegram_sentence("")
        == "Выбрано по сочетанию цены, наличия и соответствия требованиям."
    )
    assert (
        make_telegram_sentence("... !!!")
        == "Выбрано по сочетанию цены, наличия и соответствия требованиям."
    )


def test_format_match_summary_humanizes_same_platform_generic_titles() -> None:
    base_candidate = {
        "source_type": "build_from_parts",
        "quantity_required": 2,
        "total_price_value": "8800",
        "total_price_currency": "USD",
        "why_selected_short": "Закрывает требования по складу.",
        "components": [
            {
                "role": "server_platform",
                "producer": "ASUS",
                "part_number": "PLATFORM",
                "quantity_required": 2,
            },
            {
                "role": "ram",
                "producer": "Samsung",
                "part_number": "RAM-64G",
                "quantity_required": 16,
                "facts": {"ram_capacity_gb": 64},
            },
        ],
    }
    summary = {
        "match_run_id": 56,
        "llm_configurator_used": True,
        "risk_flags": [],
        "missing_requirements": [],
        "ai_recommendations": [
            {
                **base_candidate,
                "title": "Оптимальный по цене вариант",
                "components": [
                    *base_candidate["components"],
                    {
                        "role": "cpu",
                        "producer": "Intel",
                        "part_number": "CPU-16C",
                        "quantity_required": 4,
                        "facts": {"cpu_cores": 16},
                    },
                ],
            },
            {
                **base_candidate,
                "title": "Технически более чистый вариант",
                "components": [
                    *base_candidate["components"],
                    {
                        "role": "cpu",
                        "producer": "Intel",
                        "part_number": "CPU-32C",
                        "quantity_required": 4,
                        "facts": {"cpu_cores": 32},
                    },
                ],
            },
        ],
    }

    text = format_match_summary(summary)

    assert "1. Вариант с CPU 16 ядер" in text
    assert "2. Вариант с CPU 32 ядра" in text
    assert "Технически более чистый вариант" not in text


def test_format_match_summary_does_not_use_llm_display_name_as_components() -> None:
    summary = {
        "match_run_id": 51,
        "llm_configurator_used": True,
        "risk_flags": [],
        "missing_requirements": [],
        "ai_recommendations": [
            {
                "source_type": "build_from_parts",
                "title": "Сборка из validated components",
                "display_name": "ASUS + Intel Xeon + Micron 512GB RAM",
                "quantity_required": 2,
                "total_price_value": "6400",
                "total_price_currency": "USD",
                "why_selected_short": "Платформа и CPU взяты из validated components.",
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "PLATFORM",
                        "quantity_required": 2,
                    },
                    {
                        "role": "cpu",
                        "producer": "Intel",
                        "part_number": "CPU",
                        "quantity_required": 4,
                    },
                    {
                        "role": "ssd",
                        "producer": "Samsung",
                        "part_number": "SSD",
                        "quantity_required": 4,
                    },
                ],
            }
        ],
    }

    text = format_match_summary(summary)

    assert "ASUS PLATFORM" in text
    assert "CPU: Intel CPU - 2 шт." in text
    assert "SSD: Samsung SSD - 2 шт." in text
    assert "Всего к подбору" in text
    assert "на заказ" not in text
    assert "Micron" not in text
    assert "RAM:" not in text


def test_format_match_summary_marks_incomplete_llm_build_as_partial() -> None:
    summary = {
        "match_run_id": 47,
        "status": "partial_stock_matched",
        "engineer_review_required": True,
        "total_candidates": 1,
        "matched_items": 0,
        "risk_flags": [],
        "missing_requirements": [],
        "llm_configurator_used": True,
        "llm_recommended_build_candidates": [
            {
                "candidate_type": "build_from_parts",
                "total_price_value": "6400",
                "total_price_currency": "USD",
                "total_price_note": "без RAM",
                "completeness_status": "incomplete",
                "missing_component_roles": ["ram"],
                "right_size_note": (
                    "Подбор: CPU выше требования. В матрице не найден более близкий "
                    "CPU с достаточным остатком."
                ),
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "LLM-PLATFORM",
                        "quantity_required": 2,
                    },
                    {
                        "role": "cpu",
                        "producer": "Intel",
                        "part_number": "CPU",
                        "quantity_required": 4,
                    },
                    {
                        "role": "ssd",
                        "producer": "Samsung",
                        "part_number": "SSD",
                        "quantity_required": 4,
                    },
                ],
            }
        ],
        "candidates": [],
    }

    text = format_match_summary(summary)

    assert "Частичная сборка" in text
    assert "Тип: частичная сборка" in text
    assert "Частичная сборка - не хватает RAM" in text
    assert "В составе 1 сервера" in text
    assert "Всего к подбору" in text
    assert "ASUS LLM-PLATFORM" in text
    assert "Ориентировочно за 2 сервера без RAM: 6 400 USD" in text
    assert "Подбор: CPU выше требования. В матрице не найден более близкий CPU" in text
    assert "Рекомендованная сборка" not in text
    assert "Готовые варианты" not in text


def test_format_match_summary_shows_ready_server_as_unified_ai_recommendation() -> None:
    summary = {
        "match_run_id": 49,
        "llm_configurator_used": True,
        "risk_flags": [],
        "missing_requirements": [],
        "ai_recommendations": [
            {
                "source_type": "ready_server",
                "title": "Готовый складской вариант с проверками",
                "display_name": "Ready Server PN-1",
                "quantity_required": 2,
                "total_price_value": "13800",
                "total_price_currency": "USD",
                "why_selected_short": "Закрывает запрос быстрее остальных вариантов.",
                "right_size_note": "Подбор: готовый складской вариант с проверками",
                "component_summary": {
                    "platform": "Ready Server PN-1",
                    "ram": "512 ГБ",
                    "storage": "SSD",
                },
            }
        ],
    }

    text = format_match_summary(summary)

    assert "Рекомендации" in text
    assert "Тип: готовый сервер" in text
    assert "Готовая позиция - 2 шт." in text
    assert "Цена за весь запрос: 13 800 USD" in text
    assert "Ready Server PN-1" in text
    assert "Готовые варианты" not in text


def test_format_match_summary_keeps_rule_based_fallback_when_llm_is_unavailable() -> None:
    summary = {
        "match_run_id": 46,
        "status": "partial_stock_matched",
        "engineer_review_required": True,
        "total_candidates": 1,
        "matched_items": 0,
        "risk_flags": [],
        "missing_requirements": [],
        "llm_configurator_used": False,
        "llm_configurator_enabled": False,
        "ready_stock_candidates": [
            {
                "producer": "NERPA",
                "part_number": "D5720-181125SA04",
                "available_quantity": 3,
                "price_value": "6900",
                "price_currency": "USD",
            }
        ],
        "build_candidates": [],
        "candidates": [],
    }

    text = format_match_summary(summary)

    assert "Подбор по складу №46" in text
    assert "Готовые варианты" in text
    assert "Сборка из комплектующих" in text
    assert "NERPA D5720-181125SA04" in text
    assert "Расширенный AI-анализ выключен." in text
    assert "AI-подбор" not in text


def test_format_match_summary_explains_safe_ai_rejection_fallback() -> None:
    summary = {
        "match_run_id": 53,
        "status": "partial_stock_matched",
        "engineer_review_required": True,
        "total_candidates": 1,
        "matched_items": 0,
        "risk_flags": [],
        "missing_requirements": [],
        "llm_configurator_used": False,
        "llm_configurator_enabled": True,
        "llm_fallback_reason": "llm_configurator_all_recommendations_rejected",
        "ready_stock_candidates": [],
        "build_candidates": [
            {
                "candidate_type": "build_from_parts",
                "total_price_value": "6400",
                "total_price_currency": "USD",
                "components": [
                    {
                        "role": "server_platform",
                        "producer": "ASUS",
                        "part_number": "PLATFORM",
                        "quantity_required": 2,
                    }
                ],
            }
        ],
    }

    text = format_match_summary(summary)

    assert text.startswith("AI-подбор по складу №53")
    assert "Безопасную складскую рекомендацию дать нельзя" in text
    assert "Подробная матрица компонентов отправлена Excel-файлом" in text
    assert "Что проверить вручную:" in text
    assert "Готовые варианты" not in text
    assert "Сборка из комплектующих" not in text


def test_format_match_summary_network_no_recommendation_uses_network_checks() -> None:
    summary = {
        "match_run_id": 77,
        "product_group": "network",
        "llm_configurator_used": False,
        "llm_configurator_enabled": True,
        "llm_fallback_reason": "llm_configurator_all_recommendations_rejected",
        "ai_recommendation_mode": "ai_no_safe_recommendations",
        "valid_proposals_count": 0,
        "ai_recommendations_count": 0,
        "no_recommendation_reason": {
            "product_group": "network",
            "missing_roles": ["switch"],
            "manual_checks": [
                "Проверить CPU support list платформы и версию BIOS.",
                "Проверить QVL RAM и правила заполнения DIMM.",
                "Проверить NVMe/U.2/U.3 backplane.",
                "Проверить PoE budget и uplink SFP+.",
            ],
        },
    }

    text = format_match_summary(summary)

    assert "PoE" in text
    assert "uplink" in text or "access/uplink" in text
    for forbidden in ("CPU support", "QVL RAM", "DIMM", "NVMe/U.2/U.3", "backplane"):
        assert forbidden not in text


def test_format_match_summary_network_primary_remains_clean() -> None:
    summary = {
        "match_run_id": 77,
        "product_group": "network",
        "llm_configurator_used": True,
        "llm_configurator_enabled": True,
        "primary_recommendation_status": "valid",
        "primary_recommendation": {
            "product_group": "network",
            "candidate_type": "build_from_parts",
            "decision": "recommend",
            "components": [
                {
                    "role": "switch",
                    "producer": "Origo",
                    "part_number": "OS3254P/370W/A1A",
                    "quantity_required": 1,
                    "server_quantity": 1,
                    "available_quantity": 17,
                    "port_count": 48,
                    "port_speed": "1GbE",
                    "port_media": "RJ45",
                    "uplink_count": 4,
                    "uplink_speed": "10GbE",
                    "uplink_media": "SFP+",
                    "poe_supported": True,
                    "poe_budget_w": 370,
                    "poe_standard": "PoE+",
                    "l3_supported": True,
                }
            ],
            "total_price_value": "483.84",
            "total_price_currency": "USD",
            "critical_checks": [
                "Проверить CPU support list платформы и версию BIOS.",
                "Проверить QVL RAM и правила заполнения DIMM.",
                "Проверить NVMe backplane.",
                "Проверить PoE budget и uplink SFP+.",
            ],
        },
    }

    text = format_match_summary(summary)

    assert "Сетевое оборудование - 1 шт." in text
    assert "Коммутатор: Origo OS3254P/370W/A1A" in text
    assert "Ориентировочно за 1 шт. сетевого оборудования: 483.84 USD" in text
    assert "PoE" in text
    for forbidden in ("Количество серверов", "CPU support", "QVL RAM", "DIMM", "NVMe"):
        assert forbidden not in text


def test_format_match_summary_uses_safe_ai_unavailable_message() -> None:
    summary = {
        "match_run_id": 54,
        "llm_configurator_enabled": True,
        "llm_configurator_used": False,
        "llm_fallback_reason": "llm_configurator_invalid_json",
        "ready_stock_candidates": [
            {
                "producer": "NERPA",
                "part_number": "READY-SERVER",
                "available_quantity": 2,
            }
        ],
        "build_candidates": [
            {
                "candidate_type": "build_from_parts",
                "components": [
                    {"role": "server_platform", "producer": "ASUS", "part_number": "PLATFORM"}
                ],
            }
        ],
    }

    text = format_match_summary(summary)

    assert text.startswith("AI-подбор по складу №54")
    assert "Расширенный AI-анализ сейчас недоступен" in text
    assert "Готовые варианты" not in text
    assert "Сборка из комплектующих" not in text
    assert "READY-SERVER" not in text


def test_format_match_summary_has_safe_unknown_status_fallback() -> None:
    text = format_match_summary(
        {
            "match_run_id": 7,
            "status": "new_status",
            "engineer_review_required": False,
            "total_candidates": 0,
            "matched_items": 0,
            "risk_flags": [],
            "missing_requirements": [],
            "candidates": [],
        }
    )

    assert "Итог: Статус подбора: new_status" in text


def test_excel_report_delivery_uses_xlsx_filename() -> None:
    delivery = choose_excel_report_delivery(b"xlsx-bytes", match_run_id=42)

    assert delivery.mode == "file"
    assert delivery.text is None
    assert delivery.filename == "stock_match_42.xlsx"
    assert delivery.content == b"xlsx-bytes"


def test_markdown_report_delivery_remains_for_compatibility() -> None:
    delivery = choose_report_delivery("# Report", match_run_id=42)

    assert delivery.mode == "file"
    assert delivery.filename == "report_42.md"
    assert delivery.content == b"# Report"


def _telegram_grouped_presales_group() -> dict[str, object]:
    return {
        "group_id": "cfg_group_1",
        "group_title": "Intel LGA4677 / DDR5 / NVMe",
        "component_base": {
            "cpu": {
                "role": "cpu",
                "producer": "Intel",
                "part_number": "Xeon Gold 6530",
                "item_name": "Intel Xeon Gold 6530",
                "quantity_required": 4,
                "per_server_quantity": 2,
                "available_quantity": 20,
            },
            "ram": {
                "role": "ram",
                "producer": "Micron",
                "part_number": "MTC20F1045S1RC48BA2",
                "item_name": "Micron MTC20F1045S1RC48BA2 32GB DDR5 RDIMM",
                "quantity_required": 32,
                "per_server_quantity": 16,
                "server_quantity": 2,
                "available_quantity": 100,
                "ram_module_capacity_gb": 32,
                "ram_total_gb_per_server": 512,
            },
            "storage": {
                "role": "ssd",
                "producer": "KIOXIA",
                "part_number": "KCD8XRUG3T84",
                "item_name": "KIOXIA CD8-R 3.84TB U.3 NVMe",
                "quantity_required": 4,
                "per_server_quantity": 2,
                "available_quantity": 20,
                "storage_capacity_tb": 3.84,
                "facts": {"storage_interface": "U.3 NVMe"},
            },
        },
        "platform_options": [
            {
                "option_id": "platform_option_1_1",
                "role": "cheapest_quote",
                "platform": {
                    "role": "server_platform",
                    "producer": "ASUS",
                    "part_number": "PLATFORM-CHEAP",
                    "quantity_required": 2,
                    "available_quantity": 2,
                },
                "total_price_value": "8600",
                "total_price_currency": "USD",
                "why_this_platform": "Minimal cost build meeting all core specs.",
                "engineer_checks": ["Проверить CPU support list / BIOS."],
                "engineering_confidence": "preliminary_requires_engineer_review",
            },
            {
                "option_id": "platform_option_1_2",
                "role": "preferred_for_database",
                "platform": {
                    "role": "server_platform",
                    "producer": "Supermicro",
                    "part_number": "SYS-621C-TN12R",
                    "quantity_required": 2,
                },
                "total_price_value": "10200",
                "total_price_currency": "USD",
                "why_this_platform": (
                    "Proven Supermicro architecture with cost-optimized components."
                ),
                "engineer_checks": ["Проверить QVL RAM."],
                "engineering_confidence": "preliminary_requires_engineer_review",
            },
        ],
    }


def _telegram_primary_recommendation() -> dict[str, object]:
    group = _telegram_grouped_presales_group()
    component_base = group["component_base"]
    platform = group["platform_options"][0]["platform"]  # type: ignore[index]
    components = [
        platform,
        component_base["cpu"],  # type: ignore[index]
        component_base["ram"],  # type: ignore[index]
        component_base["storage"],  # type: ignore[index]
    ]
    return {
        "candidate_type": "build_from_parts",
        "title": "Cheapest valid complete stock build",
        "component_candidate_ids": {
            "platform": "platform-cheap",
            "cpu": "cpu-xeon",
            "ram": "ram-micron",
            "ssd": "ssd-kioxia",
        },
        "why_selected": "Minimal cost build meeting all core specs.",
        "assumptions": [],
        "engineer_checks": ["component_candidate_id should not leak"],
        "components": components,
        "total_price_value": "8600",
        "total_price_currency": "USD",
    }
