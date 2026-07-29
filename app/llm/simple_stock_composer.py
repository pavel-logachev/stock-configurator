from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from app.core.config import LlmSettings, get_llm_settings
from app.llm.base import LlmClient, LlmError, LlmHttpError
from app.llm.full_category_composer import FullCategoryComposerOutcome
from app.matching.full_category_matrix import (
    MATRIX_EMPTY_AFTER_CATEGORY_SELECTION,
    MATRIX_READY_FOR_LLM,
    MATRIX_TOO_LARGE_FOR_MODEL,
)
from app.matching.simple_stock_matrix import SimpleStockMatrixPackage
from app.matching.simple_stock_reconciler import reconcile_simple_stock_quote

SIMPLE_STOCK_QUOTE_PIPELINE_VERSION = "simple_stock_quote"
SIMPLE_STOCK_COMPOSER_MODE = "simple_stock_composer"
SIMPLE_STOCK_COMPOSER_PROMPT_VERSION = "simple_stock_composer_v70"
SIMPLE_STOCK_QUOTE_ACCEPTED = "simple_stock_quote_llm_accepted"
SIMPLE_STOCK_NO_RECOMMENDATION = "simple_stock_quote_no_recommendation"
SIMPLE_STOCK_SCHEMA_FAILED = "simple_stock_quote_schema_failed"
SIMPLE_STOCK_MECHANICAL_ERROR = "simple_stock_quote_mechanical_error"


@dataclass(frozen=True)
class SimpleStockQuotePromptBundle:
    system_prompt: str
    user_prompt: str
    prompt_char_count: int
    composer_profile: str = "role_first"


def compose_simple_stock_quote(
    *,
    user_request: str,
    resolved_request: Mapping[str, Any] | None,
    matrix_package: SimpleStockMatrixPackage,
    settings: LlmSettings | None = None,
    llm_client: LlmClient | None = None,
) -> FullCategoryComposerOutcome:
    effective_settings = settings or get_llm_settings()
    diagnostics: dict[str, Any] = {
        "composer_prompt_version": SIMPLE_STOCK_COMPOSER_PROMPT_VERSION,
        "matrix_schema_version": matrix_package.payload.get("schema_version"),
        "matrix_status": matrix_package.status,
        "matrix_char_count": matrix_package.char_count,
        "matrix_row_count": matrix_package.payload.get("diagnostics", {}).get("row_count"),
        "matrix_position_count": matrix_package.payload.get("diagnostics", {}).get(
            "position_count"
        ),
        "model": effective_settings.llm_model,
        "post_llm_validation": "mechanical_integrity_only",
        "post_llm_materialization": "product_candidate_to_stock_row_reconciliation",
        "validation_policy": (
            "simple route accepts LLM quote semantics; code materializes selected "
            "product candidates into exact stock rows, prices, quantities and totals"
        ),
    }

    if matrix_package.status != MATRIX_READY_FOR_LLM:
        final_status = (
            MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
            if matrix_package.status == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
            else MATRIX_TOO_LARGE_FOR_MODEL
        )
        return FullCategoryComposerOutcome(
            pipeline_version=SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
            used=False,
            status="no_recommendation",
            final_status_source=final_status,
            primary_recommendation_status="no_recommendation",
            no_recommendation_reason={
                "summary": "Складская матрица не готова для LLM в простом контуре.",
                "fallback_reason": final_status,
            },
            diagnostics=diagnostics,
        )

    if llm_client is None:
        return FullCategoryComposerOutcome(
            pipeline_version=SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
            used=False,
            status="no_recommendation",
            final_status_source="simple_stock_quote_provider_not_configured",
            primary_recommendation_status="no_recommendation",
            no_recommendation_reason={
                "summary": "LLM provider is not configured for the simple stock quote route.",
                "fallback_reason": "provider_not_configured",
            },
            diagnostics=diagnostics,
        )

    prompts = build_simple_stock_quote_prompts(
        user_request=user_request,
        resolved_request=resolved_request or {},
        matrix_package=matrix_package,
        model=effective_settings.llm_model,
    )
    diagnostics["prompt_char_count"] = prompts.prompt_char_count
    diagnostics["system_prompt_chars"] = len(prompts.system_prompt)
    diagnostics["user_prompt_chars"] = len(prompts.user_prompt)
    diagnostics["composer_profile"] = prompts.composer_profile

    try:
        raw_output = llm_client.generate_json(prompts.system_prompt, prompts.user_prompt)
    except LlmHttpError as exc:
        return FullCategoryComposerOutcome(
            pipeline_version=SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
            used=True,
            status="provider_error",
            final_status_source="simple_stock_quote_provider_error",
            primary_recommendation_status="provider_error",
            no_recommendation_reason={
                "summary": "LLM provider error before simple stock quote completed.",
                "fallback_reason": "provider_error",
            },
            diagnostics=diagnostics,
            error_type=type(exc).__name__,
            http_status=exc.status_code,
        )
    except LlmError as exc:
        return FullCategoryComposerOutcome(
            pipeline_version=SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
            used=True,
            status="provider_error",
            final_status_source="simple_stock_quote_provider_error",
            primary_recommendation_status="provider_error",
            no_recommendation_reason={
                "summary": "LLM error before simple stock quote completed.",
                "fallback_reason": "provider_error",
            },
            diagnostics=diagnostics,
            error_type=type(exc).__name__,
        )

    status = str(raw_output.get("status") or "").strip().lower()
    quote = _mapping(raw_output.get("quote"))
    if status != "no_recommendation" and quote:
        integrity = reconcile_simple_stock_quote(quote, matrix_package)
        diagnostics.update(integrity.diagnostics)
        if integrity.status == "mechanical_error":
            return FullCategoryComposerOutcome(
                pipeline_version=SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
                used=True,
                status="schema_failed",
                final_status_source=SIMPLE_STOCK_MECHANICAL_ERROR,
                primary_recommendation_status="schema_failed",
                llm_output=raw_output,
                validation_failure_reason={
                    "summary": (
                        "КП не сформировано: выбранные позиции не удалось механически "
                        "сопоставить с текущим складом."
                    ),
                    "fallback_reason": SIMPLE_STOCK_MECHANICAL_ERROR,
                    "next_actions": [
                        "Повторить запрос после обновления склада.",
                        "Проверить технические детали на листе диагностики.",
                    ],
                },
                validation_errors=integrity.errors,
                validation_error_details=integrity.error_details,
                diagnostics=diagnostics,
            )
        validated_quote = integrity.quote
        return FullCategoryComposerOutcome(
            pipeline_version=SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
            used=True,
            status="quote",
            final_status_source=SIMPLE_STOCK_QUOTE_ACCEPTED,
            primary_recommendation_status="llm_final",
            llm_output=raw_output,
            validated_quote=validated_quote,
            validation_warnings=[*integrity.warnings],
            diagnostics=diagnostics,
        )

    no_recommendation = _mapping(raw_output.get("no_recommendation"))
    if status == "no_recommendation" or no_recommendation:
        return FullCategoryComposerOutcome(
            pipeline_version=SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
            used=True,
            status="no_recommendation",
            final_status_source=SIMPLE_STOCK_NO_RECOMMENDATION,
            primary_recommendation_status="no_recommendation",
            llm_output=raw_output,
            no_recommendation_reason=no_recommendation
            or {
                "summary": "LLM did not find a useful stock offer.",
                "fallback_reason": SIMPLE_STOCK_NO_RECOMMENDATION,
            },
            diagnostics=diagnostics,
        )

    return FullCategoryComposerOutcome(
        pipeline_version=SIMPLE_STOCK_QUOTE_PIPELINE_VERSION,
        used=True,
        status="schema_failed",
        final_status_source=SIMPLE_STOCK_SCHEMA_FAILED,
        primary_recommendation_status="schema_failed",
        llm_output=raw_output,
        validation_failure_reason={
            "summary": "LLM returned neither quote nor no_recommendation.",
            "fallback_reason": SIMPLE_STOCK_SCHEMA_FAILED,
        },
        validation_errors=["simple_schema.quote_or_no_recommendation_missing"],
        diagnostics=diagnostics,
    )


def build_simple_stock_quote_prompts(
    *,
    user_request: str,
    resolved_request: Mapping[str, Any],
    matrix_package: SimpleStockMatrixPackage,
    model: str | None = None,
) -> SimpleStockQuotePromptBundle:
    _ = model
    system_prompt = """
You are Stock Configurator Composer.

Inputs:
- original_request_text: customer request and source of truth;
- resolved_request: compact interpretation from the category router;
- stock_matrix: complete product cards from the selected distributor categories.

Use only stock_matrix.category_sections[].positions[] as product facts. Each position has
component_candidate_id, part_number, description and offers. Do not invent products, IDs,
prices, stock, features, bundle contents or compatibility.

Task: build the most useful Russian draft quote from current stock. Choose exact stocked
products or reasonable stocked analogs. If no useful stocked product exists, return
no_recommendation.

Rules:
- Split the request into material quote roles: target/base system, CPU, RAM, drives,
  controllers, NIC/HBA, optics, PSU, licenses, high-value functional options, consumables
  and spares. Low-value mounting/install details for a larger system (rails, CMA, generic
  cables, brackets, caddies, generic risers) are engineer_checks or fitment notes, not separate
  quote/gap roles, unless they are the primary purchase or function-critical.
- Work target-first. For a complete object/system/kit/assembly request, first evaluate stocked
  ready, preconfigured or assembled products as the primary base candidate. Do not reject such
  a base only because its internal specs differ from the requested configuration; if it matches
  the core target object, include it as analog/partial/needs_check and put differences into
  key_deviations, procurement_gaps or engineer_checks. Use component-only coverage only when
  no stocked base reasonably matches the target object.
- One lines[] item is one selected product candidate. Return component_candidate_id only;
  do not return stock_row_id. Code will allocate real stock rows, prices, quantities and totals.
- Calculate quantities as requested target quantity times per-target quantity. If one suitable
  product cannot cover the quantity, use additional same-role products with the same key
  characteristics when available. Put only the remaining delta into procurement_gaps.
- For each role, compare all relevant stocked products. Function and required characteristics
  come before price. Within the same role/category, numeric and structural matrix facts are
  stronger evidence than exact wording: capacity, speed, interface, protocol, form/type, rank,
  ports, cache, power and license term. If a cheaper same-role item matches the main
  numeric/structural facts but lacks one expected wording token, do not reject it only for that
  missing token. Missing wording is an engineer_check, not a
  disqualifier, unless the matrix has a concrete conflicting fact.
- Use each category's technical_price_index as a mechanical recall aid before selecting a line
  or declaring a gap. Check 2-4 specific tokens for the role (capacity/type/speed or
  protocol/ports/interface). A candidate appearing across multiple specific token buckets is a
  strong analog candidate; select the cheapest sufficient stocked analog or name a concrete
  matrix conflict. Do not select or reject by one generic token alone.
- Concrete conflicts override price: different capacity, lower required speed, wrong
  interface/protocol, wrong form/type, port count, cache, power class, license term or explicit
  incompatibility. Exact part number, exact OEM or no-analog requests also override price. Use
  price_rank_in_currency and price_delta_vs_cheapest when present as price visibility hints.
  Brand reputation or a familiar vendor is not a premium reason unless the customer explicitly
  forbids analogs or requires exact OEM.
  In every line reason write either "Цена: выбран самый дешевый сопоставимый вариант." or
  "Цена: выбран дороже <part_number, price>, потому что <matrix fact makes the cheaper item
  unsuitable>."
  If no matrix-grounded disqualifier exists, select the cheaper product.
- Analogs are allowed unless the customer forbids substitution. Put known differences in
  key_deviations. Put unknown compatibility, firmware, mounting, licensing, bundle contents,
  lower-bound stock such as "1+", and support questions in engineer_checks. Uncertainty is not
  a procurement gap by itself.
- Platform-proprietary fit overrides generic similarity. For PSU, cooling, risers, rails/CMA,
  backplanes, carriers, embedded controllers, vendor-qualified memory or firmware-locked
  adapters, do not count a generic analog as covered only by wattage, speed, ports or
  dimensions. If the matrix does not show platform/vendor fit for the requested system, put it
  into procurement_gaps or alternatives plus engineer_checks, not included lines[]. Do not say a
  role is covered when the line itself depends on unresolved platform-fit.
- The platform-fit rule is two-sided. If a stocked row explicitly names the requested vendor,
  platform family, model family, generation, form factor or bundle family for a proprietary
  role, treat it as a strong positive candidate. Select it when quantity and role coverage are
  sufficient, or cite it in alternatives/engineer_checks with the concrete remaining concern.
  Do not create an "absent from stock" procurement_gap while such a platform-positive row exists.
- Create procurement_gaps only after selected lines: one missing role/function per gap, only
  the uncovered quantity or item. Do not create a gap for a role already reasonably covered by
  an exact item or chosen analog.
- Totals include selected lines only. You may fill price fields from the product card, but code
  will overwrite prices and totals mechanically.

Return only JSON. Write user-facing text in Russian.

Quote format:
{
  "status": "quote",
  "quote": {
    "title": "КП draft",
    "selection_mode": "stock_exact | stock_analog | partial_stock_offer",
    "completeness_status": "complete | partial",
    "operational_status": "operational_after_review | incomplete_needs_sourcing",
    "client_summary": "...",
    "coverage_summary": "...",
    "why_selected": "...",
    "lines": [
      {
        "line_id": "L1",
        "requirement_id": "R1",
        "component_candidate_id": "...",
        "part_number": "...",
        "item_name": "...",
        "role": "...",
        "fit_status": "exact | analog | partial | needs_check",
        "quantity": 1,
        "selected_currency": "optional",
        "reason": "..."
      }
    ],
    "total_price_value": "...",
    "total_price_currency": "...",
    "key_deviations": [],
    "procurement_gaps": [
      {
        "requirement_id": "R1",
        "item": "...",
        "quantity": 0,
        "reason": "..."
      }
    ],
    "assumptions": [],
    "engineer_checks": []
  }
}

No recommendation format:
{
  "status": "no_recommendation",
  "no_recommendation": {
    "summary": "...",
    "next_actions": []
  }
}
"""
    user_prompt = _stable_json(
        {
            "original_request_text": user_request.strip(),
            "resolved_request": dict(resolved_request),
            "composer_profile": "clean_role_first",
            "stock_matrix": matrix_package.payload,
        }
    )
    return SimpleStockQuotePromptBundle(
        system_prompt=system_prompt.strip(),
        user_prompt=user_prompt,
        prompt_char_count=len(system_prompt.strip()) + len(user_prompt),
        composer_profile="clean_role_first",
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=False, separators=(",", ":"))
