from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.catalog.ocs_anchor_categories import load_ocs_anchor_categories
from app.core.config import LlmSettings, get_llm_settings
from app.llm.base import LlmClient, LlmError, LlmHttpError
from app.matching.full_category_matrix import (
    MATRIX_EMPTY_AFTER_CATEGORY_SELECTION,
    MATRIX_READY_FOR_LLM,
    MATRIX_TOO_LARGE_FOR_MODEL,
    FullCategoryMatrixPackage,
)

V3_FULL_CATEGORY_MATRIX_MODE = "v3_full_category_matrix"
V3_VALIDATED = "v3_full_category_quote_validated"
V3_NO_RECOMMENDATION = "v3_full_category_no_recommendation"
V3_CODE_VALIDATION_BYPASSED = "v3_full_category_quote_code_validation_bypassed"
V3_SCHEMA_VALIDATION_FAILED = "v3_full_category_schema_validation_failed"
V3_MECHANICAL_VALIDATION_FAILED = "v3_full_category_mechanical_validation_failed"
V3_PROVIDER_ERROR = "v3_full_category_provider_error"
V3_PROVIDER_NOT_CONFIGURED = "v3_full_category_provider_not_configured"
COMPOSER_PROMPT_VERSION_V7 = "composer_v7"
COMPOSER_OUTPUT_SCHEMA_VERSION_V7 = "composer_output_schema_v7"
SELECTION_CONTRACT_VERSION_V7 = "selection_contract_v7"
COMPOSER_PROMPT_VERSION_V7_1 = "composer_v7_1"
COMPOSER_OUTPUT_SCHEMA_VERSION_V7_1 = "composer_output_schema_v7_1"
SELECTION_CONTRACT_VERSION_V7_1 = "selection_contract_v7_1"
RESOLVED_REQUEST_SCHEMA_VERSION_V7_1 = "resolved_request_schema_v7_1"


MECHANICAL_ERROR_MESSAGES_RU: dict[str, str] = {
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
    "contract.object_results_coverage_mismatch": "object_results не покрывает все объекты запроса.",
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
    "contract.compatibility_line_check_reference_mismatch": (
        "Проверка совместимости ссылается не на ту строку, товар или "
        "складскую строку."
    ),
    "contract.dominance_audit_missing": "Модель не вернула dominance_audit по строкам КП.",
    "contract.dominance_audit_line_coverage_mismatch": (
        "dominance_audit не покрывает все строки КП."
    ),
    "contract.no_recommendation_not_allowed": (
        "Модель вернула no_recommendation, хотя контракт требует показать "
        "лучший доступный складской вариант."
    ),
    "contract.line_id_missing": "В строке КП отсутствует обязательный line_id.",
    "contract.line_fact_ids_missing": "В строке КП отсутствуют fact_ids из матрицы.",
    "contract.line_compatibility_statement_missing": (
        "В строке КП отсутствует краткое объяснение совместимости."
    ),
    "contract.duplicate_line_for_anchor_covered_item": (
        "Модель докупила отдельно позицию, которую уже покрыл выбранный anchor."
    ),
    "reference.unknown_component_candidate_id": (
        "Модель выбрала component_candidate_id, которого нет в переданной матрице."
    ),
    "reference.unknown_stock_row_id": (
        "Модель выбрала stock_row_id, которого нет в переданной матрице."
    ),
    "reference.stock_row_product_mismatch": "Складская строка относится к другому товару.",
    "reference.unknown_or_foreign_fact_id": (
        "Строка КП ссылается на fact_id, которого нет у выбранного товара."
    ),
    "stock.insufficient_quantity": "Выбрано больше единиц, чем есть в складской строке.",
    "stock.row_overallocated": "Одна складская строка использована сверх доступного остатка.",
    "price.unit_price_mismatch": "Цена в строке не совпадает со складской ценой.",
    "price.currency_mismatch": "Валюта в строке не совпадает с валютой складской цены.",
    "price.price_missing": "По выбранной складской строке нет пригодной цены.",
    "arithmetic.line_total_mismatch": "Сумма строки посчитана неверно.",
    "arithmetic.quote_total_mismatch": "Итоговая сумма не совпадает с суммой строк.",
    "arithmetic.multiple_currencies": "В КП смешаны разные валюты.",
}


class FullCategoryQuoteLinePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line_id: str | None = None
    component_candidate_id: str = Field(min_length=1)
    stock_row_id: str | None = None
    role: str | None = None
    quantity: int = Field(gt=0)
    title: str | None = None
    reason: str | None = None
    unit_price_value: Decimal | None = None
    unit_price_currency: str | None = None
    line_total_value: Decimal | None = None
    line_total_currency: str | None = None
    satisfies_requirement_ids: list[str] = Field(default_factory=list)
    object_id: str | None = None
    covered_item_ids: list[str] = Field(default_factory=list)
    covered_requirement_ids: list[str] = Field(default_factory=list)
    technical_status: str | None = None
    coverage_contributions: list[dict[str, Any]] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    compatibility_statement: str | None = None


class FullCategoryCompatibilityCheckPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str | None = None
    checked_facts: list[Any] = Field(default_factory=list)
    blocking_mismatches: list[str] = Field(default_factory=list)
    selected_line_conflicts: list[str] = Field(default_factory=list)
    unresolved_risks: list[str] = Field(default_factory=list)


class FullCategoryQuotePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    client_status_label: str | None = None
    solution_scope: str | None = None
    substitution_policy: str | None = None
    selection_mode: str | None = None
    completeness_status: str | None = None
    operational_status: str | None = None
    anchor_component_candidate_id: str | None = None
    lines: list[FullCategoryQuoteLinePayload] = Field(default_factory=list)
    total_price_value: Decimal | None = None
    total_price_currency: str | None = None
    total_price: dict[str, Any] = Field(default_factory=dict)
    client_summary: str = ""
    coverage_summary: str = ""
    why_selected: str = ""
    object_results: list[dict[str, Any]] = Field(default_factory=list)
    anchor_search_audit: list[dict[str, Any]] = Field(default_factory=list)
    coverage: list[dict[str, Any]] = Field(default_factory=list)
    requirement_coverage: list[dict[str, Any]] = Field(default_factory=list)
    key_deviations: list[Any] = Field(default_factory=list)
    procurement_gaps: list[Any] = Field(default_factory=list)
    deviation_notes: list[Any] = Field(default_factory=list)
    price_audit: list[Any] = Field(default_factory=list)
    dominance_audit: list[Any] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    engineer_checks: list[str] = Field(default_factory=list)
    compatibility_check: FullCategoryCompatibilityCheckPayload | None = None


class FullCategoryComposerPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["quote", "no_recommendation"] | None = None
    quote: FullCategoryQuotePayload | None = None
    no_recommendation: dict[str, Any] | None = None
    general_notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class FullCategoryQuoteValidation:
    status: str
    errors: list[str] = field(default_factory=list)
    error_details: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_quote: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FullCategoryComposerOutcome:
    pipeline_version: str
    used: bool
    status: str
    final_status_source: str
    primary_recommendation_status: str
    llm_output: dict[str, Any] = field(default_factory=dict)
    validated_quote: dict[str, Any] = field(default_factory=dict)
    no_recommendation_reason: dict[str, Any] = field(default_factory=dict)
    validation_failure_reason: dict[str, Any] = field(default_factory=dict)
    validation_errors: list[str] = field(default_factory=list)
    validation_error_details: list[dict[str, Any]] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    http_status: int | None = None

    def to_report_json(self) -> dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "composer_mode": self.pipeline_version,
            "llm_configurator_used": self.used,
            "primary_recommendation_status": self.primary_recommendation_status,
            "final_status_source": self.final_status_source,
            "validated_quote": self.validated_quote,
            "primary_recommendation": self.validated_quote,
            "no_recommendation_reason": self.no_recommendation_reason,
            "validation_failure_reason": self.validation_failure_reason,
            "v3_llm_output": self.llm_output,
            "v3_validation_errors": self.validation_errors,
            "v3_validation_error_details": self.validation_error_details,
            "v3_validation_warnings": self.validation_warnings,
            "diagnostics": self.diagnostics,
            "llm_error_type": self.error_type,
            "llm_http_status": self.http_status,
        }


def compose_full_category_quote(
    *,
    user_request: str,
    resolved_request: Mapping[str, Any] | None = None,
    matrix_package: FullCategoryMatrixPackage,
    settings: LlmSettings | None = None,
    llm_client: LlmClient | None = None,
) -> FullCategoryComposerOutcome:
    effective_settings = settings or get_llm_settings()
    normalized_resolved_request = _jsonable_mapping(resolved_request or {})
    diagnostics = _matrix_diagnostics(matrix_package, effective_settings)
    diagnostics["resolved_request"] = normalized_resolved_request
    anchor_candidate_manifest = _anchor_candidate_manifest(
        normalized_resolved_request,
        matrix_package=matrix_package,
    )
    selection_contract = _selection_contract_payload(
        normalized_resolved_request,
        matrix_package=matrix_package,
        anchor_candidate_manifest=anchor_candidate_manifest,
    )
    prompt_version = _composer_prompt_version(effective_settings)
    output_schema_version = (
        COMPOSER_OUTPUT_SCHEMA_VERSION_V7_1
        if prompt_version == COMPOSER_PROMPT_VERSION_V7_1
        else COMPOSER_OUTPUT_SCHEMA_VERSION_V7
    )
    if matrix_package.status != MATRIX_READY_FOR_LLM:
        fallback_reason = (
            MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
            if matrix_package.status == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
            else MATRIX_TOO_LARGE_FOR_MODEL
        )
        summary = (
            "Selected v3 category matrix has no stocked/priced rows."
            if fallback_reason == MATRIX_EMPTY_AFTER_CATEGORY_SELECTION
            else "v3 full category matrix composer was not attempted."
        )
        return _not_used_outcome(
            final_status_source=fallback_reason,
            fallback_reason=fallback_reason,
            diagnostics=diagnostics,
            summary=summary,
        )
    if llm_client is None:
        return _not_used_outcome(
            final_status_source=V3_PROVIDER_NOT_CONFIGURED,
            fallback_reason="llm_provider_not_configured",
            diagnostics=diagnostics,
        )

    system_prompt, user_prompt = build_full_category_quote_prompts(
        user_request=user_request,
        resolved_request=normalized_resolved_request,
        matrix_package=matrix_package,
        settings=effective_settings,
    )
    prompt_char_count = len(system_prompt) + len(user_prompt)
    diagnostics = {
        **diagnostics,
        "composer_prompt_version": prompt_version,
        "composer_output_schema_version": output_schema_version,
        "selection_contract_version": selection_contract.get("contract_version"),
        "selection_contract": selection_contract,
        "anchor_candidate_manifest_count": len(anchor_candidate_manifest),
        "anchor_candidate_manifest_preview": anchor_candidate_manifest[:50],
        "matrix_payload_schema_version": matrix_package.payload.get(
            "matrix_payload_schema_version",
        ),
        "canonical_input_hash": _sha256_text(user_prompt),
        "estimated_input_tokens": _conservative_token_estimate(system_prompt + user_prompt),
        "llm_provider": effective_settings.llm_provider,
        "llm_params": {
            "max_output_tokens": effective_settings.llm_configurator_max_output_tokens,
            "timeout_seconds": effective_settings.llm_configurator_timeout_seconds,
            "read_timeout_seconds": (
                effective_settings.llm_configurator_read_timeout_seconds
            ),
            "thinking_enabled": effective_settings.llm_configurator_thinking_enabled,
            "contract_version": effective_settings.v3_full_category_contract_version,
        },
        "system_prompt_chars": len(system_prompt),
        "user_prompt_chars": len(user_prompt),
        "prompt_char_count": prompt_char_count,
    }
    if prompt_char_count > effective_settings.llm_configurator_max_package_chars:
        return _not_used_outcome(
            final_status_source=MATRIX_TOO_LARGE_FOR_MODEL,
            fallback_reason=MATRIX_TOO_LARGE_FOR_MODEL,
            diagnostics=diagnostics,
        )

    try:
        raw_output = llm_client.generate_json(system_prompt, user_prompt)
    except LlmHttpError as exc:
        return FullCategoryComposerOutcome(
            pipeline_version=V3_FULL_CATEGORY_MATRIX_MODE,
            used=True,
            status="error",
            final_status_source=V3_PROVIDER_ERROR,
            primary_recommendation_status="no_recommendation",
            no_recommendation_reason={
                "summary": "LLM provider error before v3 quote composition completed.",
                "fallback_reason": V3_PROVIDER_ERROR,
            },
            diagnostics=diagnostics,
            error_type=type(exc).__name__,
            http_status=exc.status_code,
        )
    except LlmError as exc:
        return FullCategoryComposerOutcome(
            pipeline_version=V3_FULL_CATEGORY_MATRIX_MODE,
            used=True,
            status="error",
            final_status_source=V3_PROVIDER_ERROR,
            primary_recommendation_status="no_recommendation",
            no_recommendation_reason={
                "summary": "LLM error before v3 quote composition completed.",
                "fallback_reason": V3_PROVIDER_ERROR,
            },
            diagnostics=diagnostics,
            error_type=type(exc).__name__,
        )

    try:
        payload = parse_full_category_composer_payload(raw_output)
    except ValidationError as exc:
        return FullCategoryComposerOutcome(
            pipeline_version=V3_FULL_CATEGORY_MATRIX_MODE,
            used=True,
            status="schema_error",
            final_status_source=V3_SCHEMA_VALIDATION_FAILED,
            primary_recommendation_status="no_recommendation",
            llm_output=_jsonable_mapping(raw_output),
            no_recommendation_reason={
                "summary": (
                    "LLM returned a v3 response that did not match "
                    "quote/no-recommendation shape."
                ),
                "fallback_reason": V3_SCHEMA_VALIDATION_FAILED,
            },
            validation_errors=[_validation_error_summary(exc)],
            diagnostics=diagnostics,
            error_type=type(exc).__name__,
        )

    if payload.status == "no_recommendation" or payload.no_recommendation:
        reason = payload.no_recommendation or {
            "summary": "LLM returned no safe quote for the full category matrix.",
            "fallback_reason": V3_NO_RECOMMENDATION,
        }
        return FullCategoryComposerOutcome(
            pipeline_version=V3_FULL_CATEGORY_MATRIX_MODE,
            used=True,
            status="no_recommendation",
            final_status_source=V3_NO_RECOMMENDATION,
            primary_recommendation_status="no_recommendation",
            llm_output=_jsonable_mapping(raw_output),
            no_recommendation_reason=_jsonable_mapping(reason),
            diagnostics=diagnostics,
        )

    if payload.quote is None:
        return FullCategoryComposerOutcome(
            pipeline_version=V3_FULL_CATEGORY_MATRIX_MODE,
            used=True,
            status="schema_error",
            final_status_source=V3_SCHEMA_VALIDATION_FAILED,
            primary_recommendation_status="no_recommendation",
            llm_output=_jsonable_mapping(raw_output),
            no_recommendation_reason={
                "summary": "LLM returned quote status without a quote object.",
                "fallback_reason": V3_SCHEMA_VALIDATION_FAILED,
            },
            validation_errors=["schema.quote_missing"],
            diagnostics=diagnostics,
        )

    return FullCategoryComposerOutcome(
        pipeline_version=V3_FULL_CATEGORY_MATRIX_MODE,
        used=True,
        status="valid_unchecked",
        final_status_source=V3_CODE_VALIDATION_BYPASSED,
        primary_recommendation_status="valid",
        llm_output=_jsonable_mapping(raw_output),
        validated_quote=_unvalidated_quote_payload(
            payload.quote,
            matrix_package=matrix_package,
        ),
        validation_warnings=["code_validation_bypassed"],
        diagnostics={
            **diagnostics,
            "code_validation_bypassed": True,
            "validation_policy": (
                "final code-side mechanical validation is disabled; LLM quote "
                "is passed through for manager review"
            ),
        },
    )


def build_full_category_quote_prompts(
    *,
    user_request: str,
    resolved_request: Mapping[str, Any] | None = None,
    matrix_package: FullCategoryMatrixPackage,
    settings: LlmSettings | None = None,
) -> tuple[str, str]:
    effective_settings = settings or get_llm_settings()
    if _composer_prompt_version(effective_settings) in {
        COMPOSER_PROMPT_VERSION_V7,
        COMPOSER_PROMPT_VERSION_V7_1,
    }:
        return _build_full_category_quote_prompts_v7(
            user_request=user_request,
            resolved_request=resolved_request,
            matrix_package=matrix_package,
        )

    resolved_request_payload = _jsonable_mapping(resolved_request or {})
    selection_contract_payload = _selection_contract_payload(
        resolved_request_payload,
        matrix_package=matrix_package,
    )
    system_prompt = """
You are Stock Configurator Composer v6.

You receive original_request_text, resolved_request, selection_contract and the
complete stocked/priced matrix for the selected distributor category or
category group.
The matrix is grouped by the distributor's own category/subcategory sections.
Inside each section, every stocked/priced product is shown as one complete
product block with its catalog wording and all stock rows. Code has not
semantically trimmed, ranked, shortlisted, assigned BOM roles or
compatibility-prefiltered the matrix.

Your task is to create the best available stocked commercial offer using only
matrix products and stock rows. This is the only paid selection pass for this
matrix. Return JSON only.

Most important rule:
- The default result is a quote, not a rejection.
- When procurement_mode="best_available" and at least one stocked/priced anchor
  product of the requested functional class exists, return status="quote".
- If a complete configuration cannot be assembled, return a partial quote:
  select the closest stocked anchor and every requested component that is
  safely compatible and stocked, then list every missing, short, downgraded or
  unverified item in requirement_coverage and procurement_gaps.
- Do not return no_recommendation merely because the matrix lacks the exact
  vendor, model, generation, CPU, RAM amount, drive, controller, HBA, NIC,
  license, rail kit, adapter, module, option or requested quantity.

Language:
- Write every natural-language field in Russian: title, reason,
  requirement_coverage, deviation_notes, why_selected, price_audit,
  assumptions, engineer_checks, compatibility_check and no_recommendation text.
- Keep vendor names, catalog product names, interfaces, standards and part
  numbers in their original catalog form where appropriate.

Sources of truth:
- original_request_text is the primary source for customer intent.
- Matrix product wording and stock rows are the primary source for product,
  stock, price and included-component facts.
- resolved_request is only a clarified reading of the request. It is not a BOM,
  shortlist, category filter or product recommendation.
- If resolved_request conflicts with original_request_text, trust
  original_request_text. If either conflicts with the matrix, trust the matrix.
- Every product block in full_category_matrix.category_sections[].products
  remains eligible if it technically fits.
- Do not treat category placement as proof of technical compatibility.
- Do not invent product capabilities, included components, compatibility,
  stock, prices, warehouses, licenses, adapters or accessories.

Selection contract:
- Obey selection_contract.no_recommendation_allowed when it is false.
- If no_recommendation_allowed=false, status="no_recommendation" is invalid:
  choose the best stocked anchor and return at least an anchor_only quote.
- If selection_contract.anchor_candidate_ids is empty, do not assume code has
  preselected anchors. Find anchor products yourself from the full matrix.
- Obey allowed_currencies and requested_warehouse constraints when supplied.
- renderer_policy tells you which structured fields are shown in Telegram and
  Excel; it is not a product-selection rule.

Definitions:
- Anchor product is the principal stocked item of the requested functional
  class: for example a complete server, server platform or server barebone for
  a server request; a monitor for a monitor request; a storage array, NAS or
  storage platform for a storage request; or the requested component class for
  a standalone component request.
- Partial quote is a commercially useful stocked subset centered on a valid
  anchor. It may require later purchase of CPU, RAM, drives, controllers,
  licenses, modules, rails, adapters, quantity balance or other items. It must
  never conceal those gaps.
- In partial_build mode, maximize useful concrete coverage from stocked rows
  after the anchor is chosen. Do not reduce a partial quote to the anchor alone
  when stocked products directly match explicit requested component roles.
  Include those stocked requirement-covering components when the component row
  itself matches the requested role, quantity and facts, then disclose missing
  enablement, adapters, licenses, controller/cabling, mounting or compatibility
  proof as separate procurement_gaps or engineer_checks.
- If the selected anchor already includes part of a requested component
  quantity but not enough of it, treat the missing quantity as an expansion
  target. Search the whole matrix for stocked rows that can cover the shortfall.
  Prefer the same included model or a clearly compatible same-role component;
  if the exact requested model is absent and substitutes are allowed, quote the
  best stocked analog when matrix facts support the role, and disclose the
  model/generation/vendor difference plus any enablement gap.
- Compatibility of selected lines and completeness of the requested solution
  are separate concepts. Selected lines may be mutually compatible while the
  overall build is partial.
- A compound requirement is satisfied only when all explicit sub-attributes
  are satisfied or consciously substituted. Matching only the aggregate value is
  not enough when the request also states per-unit size, module count, port
  count, interface, protocol, form factor, speed, generation or role.

Requirement classification:
- Internally build a complete requirement ledger from original_request_text and
  resolved_request before selecting products.
- If resolved_request.requirements exists, treat it as the primary ledger and
  preserve its requirement_id values in requirement_coverage.
- Classify explicit requirements as:
  1. non_negotiable: may not be violated in a quote. This includes explicit
     deliverable quantity, exact-only/no-analog wording, budget ceilings, "только",
     "обязательно", "строго", "без аналогов", "не менее", "не более", and
     physical/electrical/protocol facts that are necessary for the selected
     BOM to be usable for the stated core task.
  2. target: requested characteristic that should be matched exactly where
     possible, but may be substituted when substitution policy permits it.
  3. preference: non-blocking preference such as vendor, model, series,
     generation, color, appearance or ecosystem when exact-only wording is absent.
- A vendor, model, generation, ecosystem, option name or part number is a
  non_negotiable requirement only when original_request_text explicitly locks
  it or forbids substitutes. Otherwise it is a target or preference.
- If a phrase combines quantity with a preferred model, split it. Example:
  "1 x <brand/model> server" means non_negotiable quantity=1 complete server,
  while <brand/model> remains a target/preference unless exact-only wording is
  present.
- If resolved_request marks a model/generation as exact but the original text
  did not forbid analogs, treat that model as a preferred reference and keep
  only the measurable technical requirement as non_negotiable when appropriate.
- A requested option, license, rail kit, adapter, controller, interface,
  protocol or management feature is not automatically a non_negotiable blocker.
  Unless the original request explicitly locks it, treat it as a target. If the
  closest stocked analog is still useful for the customer's core task, quote it
  and disclose the difference in requirement_coverage and deviation_notes.
- Preserve compound technical requirements as compound facts. For example,
  "8 x 1.92TB SAS 2.5 SSD + 4 spare" contains quantity, role, per-unit
  capacity, protocol/interface, form factor and spare/operational split. Do not
  collapse it into capacity only. Exact/equivalent matching for such a target
  must consider every explicit attribute before allowing a downgrade.
- Unknown information must remain unknown. Do not invent a customer requirement
  or customer-owned component.

Substitution policy:
- Determine substitution_policy from original_request_text:
  forbidden: customer explicitly requires exact item or forbids analogs.
  allowed_no_downgrade: customer allows analogs but says "не хуже" or equivalent.
  allowed_with_disclosed_downgrade: default when substitutes are not explicitly
  forbidden and no no-downgrade restriction is present.
- Under all policies, non_negotiable requirements, core functional usability
  and technical compatibility may not be violated.
- For integrator requests, named models, brands, generations, CPUs, drives,
  adapters, controllers and vendor options are preferred references by default.
  This includes phrases like "<brand/model> в составе". Do not stop merely
  because the exact preferred ecosystem is absent when substitutes are allowed.
- Under allowed_with_disclosed_downgrade, an unavailable requested target should
  normally become a visible deviation, not no_recommendation, when a coherent
  stocked analog can still perform the core task.

Deliverable scope:
- Determine solution_scope before building the quote:
  complete_system, configured_system, standalone_product, replacement_component,
  expansion_or_upgrade, accessory or multi_product_solution.
- Build the highest available stocked offer for that scope. Prefer a complete
  BOM when possible. If a complete BOM is not possible but a useful stocked
  base exists, return partial_build or anchor_only and disclose what is missing.
- A request for a complete device must not be presented as fully satisfied by a
  barebone alone. A barebone/platform may be quoted only as partial_build or
  anchor_only with visible procurement_gaps when the rest cannot be safely
  selected from stock.
- A request for a standalone component must not add unrelated base systems,
  peripherals or optional accessories.
- A replacement or expansion request should include only the requested item and
  any mandatory stocked enablement parts.
- A complete or preconfigured system may satisfy several BOM roles at once.
  Do not buy CPU, RAM, storage, PSU, NIC, license or another item separately
  when the selected system explicitly includes enough of it.
- Optional or supported is not the same as included.
- When adding components to a complete or preconfigured stocked system, check
  the mandatory enablement for those added components separately. Do not assume
  cooling, power/cabling, carriers, trays, brackets, risers, licenses, firmware
  enablement or mounting kits are included unless the selected product block
  says so. If the matrix does not contain a required enablement item, disclose
  it in procurement_gaps or engineer_checks according to whether it blocks
  physical usability or only needs confirmation.
- Missing enablement for one selected or requested component must not erase
  other stocked lines that directly cover customer-requested roles. For
  example, if requested stocked drives, memory modules, adapters or other
  components are available with sufficient quantity and price, include them in
  partial_build when honest, and separately report the missing controller,
  cable, carrier, license, firmware, bracket or mounting requirement.
- A missing exact model is not by itself a reason to leave a requested component
  quantity short in partial_build. When the matrix contains a stocked same-role
  analog, or the same component model already included by the selected anchor,
  include the analog line if it can credibly cover the missing quantity and make
  the substitution visible.
- Do not add operating systems, services, cables, rails, adapters, licenses,
  support or peripherals unless requested or required for the selected BOM to
  be physically usable according to matrix facts.
- "Requested" means the line may be considered as a target. It does not make
  the exact requested option a mandatory enablement blocker unless the selected
  BOM would be physically unusable without that exact item.
- Low-value standard accessories that are not requested and are not clearly
  required by matrix wording should not dominate the recommendation. Put them
  into assumptions or engineer_checks instead of rejecting an otherwise coherent
  core configuration.

Fulfilment modes:
- Evaluate configurations in this strict order:
  1. exact_complete: complete stocked offer; every request requirement is met
     exactly.
  2. equivalent_complete: complete stocked offer; differences are equivalent
     or better for the stated task.
  3. downgraded_complete: operational complete offer with one or more
     disclosed degradations.
  4. partial_build: valid anchor plus a compatible stocked subset; one or more
     requested or operational components are missing, short or deliberately
     omitted because compatibility cannot be proved.
  5. anchor_only: only the best valid stocked anchor can be safely quoted.
  6. no_recommendation: allowed only under the narrow rules below.
- Never choose a lower fulfilment mode merely to reduce price. First choose the
  highest available mode and closest fit; then minimize price inside that fit
  class.
- A more expensive complete/equivalent configuration outranks a cheaper
  downgraded or partial configuration.

Optimization order:
1. physical and technical workability;
2. all non_negotiable requirements;
3. best available fulfilment mode;
4. lowest number and severity of deviations;
5. lowest total price inside that mode and deviation class;
6. fewest unnecessary BOM lines, assumptions and customer approvals.

Commercial objective:
- The product goal is to beat competitors on price inside the best available
  fulfilment mode. Study the whole matrix and compare complete configurations,
  not isolated cheapest rows.
- Rows inside each subcategory are mechanically ordered by quote-friendly
  currency priority and ascending row price only as a reading aid. Do not use
  that order as a greedy algorithm or compatibility proof.
- Complete/preconfigured systems, barebones, individual components and
  accessories are all valid product blocks. Compare configured systems against
  barebone-plus-components builds before choosing.
- A selected configuration is dominated and must not be chosen when another
  configuration has the same or better fulfilment mode, no more severe
  deviations, satisfies the same scope and has a lower total price.

Matrix search:
- Inspect all product blocks in all supplied category sections. Do not stop
  after the first workable configuration.
- Consider exact rows, stocked analogs, complete systems, configurable
  platforms, required components and required enablement items.
- For each explicit target that may be downgraded, first search the whole
  supplied matrix for exact or better candidates for the same functional role
  and explicit attributes: quantity, capacity, interface/protocol, form factor,
  speed/class, port count, generation, redundancy and role such as operational
  versus spare. A cheaper degraded candidate is invalid while an exact or
  better candidate with sufficient stock and credible compatibility remains.
- Do not write "not in matrix", "absent" or "unavailable" for a requested
  attribute if any supplied product block contains that attribute. If such a
  candidate exists but is not selected, explain the concrete reason: insufficient
  quantity, incompatible selected anchor, missing mandatory enablement, higher
  priority conflict, currency conflict, or worse total fulfilment mode.
- In partial_build, "missing mandatory enablement" is normally a reason to
  disclose a gap around that enablement, not a reason to omit the stocked
  requirement-covering component itself. Omit that component only when the
  component row is clearly incompatible with the chosen anchor or selected
  lines, has insufficient stock, lacks a usable price/currency, or selecting it
  would misrepresent the quoted subset as operationally complete.
- Do not report a requested component quantity as missing or to-be-procured
  until you have searched the supplied matrix for rows that can cover the
  shortfall. If such a row exists but is not selected, state the concrete
  matrix-grounded reason: incompatible with the chosen anchor, unsafe mixed
  homogeneous component model, insufficient stock, missing price/currency, or
  selected-line fit conflict.
- Interface/protocol downgrades are material by default. For example, SAS to
  SATA, FC speed downgrade, NVMe to SATA, optical to copper, ECC to non-ECC or
  RDIMM to UDIMM may be quoted only after better-fit stocked candidates were
  considered and rejected for a stated matrix-grounded reason.
- Do not combine different product models into one homogeneous requested line
  merely to reach quantity unless the customer explicitly allows mixed supply.
- If the customer separates operational quantity from spare/reserve/ZIP/extra
  quantity, analyze those quantities separately. If the operational quantity is
  coverable but spare/reserve is short, quote the coherent available build only
  when it remains honest and useful, and report the reserve shortfall clearly.
- If compatibility of a candidate component cannot be established, omit that
  component, keep the best anchor when useful, and report the component as a
  procurement gap instead of rejecting the whole offer.
- For partial_build, distinguish component compatibility from solution
  completeness. If the candidate component itself matches an explicit requested
  role and has stock/price, but the complete solution still lacks a separate
  required enabler, quote the component line when it is commercially useful and
  mark the enabler as the gap. Do not label the component attribute as absent
  merely because the enabling item is absent.

Compatibility:
- Before returning status="quote", verify the selected stocked lines against
  each other and against the stated facts in the matrix.
- A quote is allowed when compatibility_check.status is "compatible",
  "compatible_selected_lines" or "anchor_only".
- compatibility_check.status="compatible" means the selected rows form an
  internally coherent complete stocked offer.
- compatibility_check.status="compatible_selected_lines" means selected rows
  are internally coherent, while the overall requested solution is partial and
  missing items are disclosed in procurement_gaps.
- compatibility_check.status="anchor_only" means only the anchor is quoted and
  no separate configurable components were safely selected.
- For every selected quote line, checked_facts must cite component_candidate_id
  or stock_row_id, state the relevant matrix fact and explain how it satisfies
  a customer requirement, another selected component or a mandatory enablement
  role.
- Check the natural compatibility dimensions for the product group: function,
  physical form factor, interface/protocol, socket/connector, memory/storage
  type, power, mounting/cooling, quantity relationships, included versus
  separately selected components, licensing and enablement where relevant.
- Do not infer compatibility merely from vendor family, product naming or
  category placement. Do not make a matrix fact more specific than the row says.
- If a missing fact could make a selected line physically unusable or
  incompatible with other selected lines, omit that line or choose another
  stocked row. Do not hide selected-line incompatibility in procurement_gaps.
- Distinguish two cases carefully:
  1. missing fact about the selected BOM's own usability or fit -> blocking;
  2. missing exact requested target while the selected analog remains usable ->
     non-blocking deviation that must be disclosed.
- compatibility_check.unresolved_risks and engineer_checks are only for
  non-blocking installation, deployment, environment, firmware, licensing or
  manager-review checks. Do not use engineer review to rescue a hard BOM fit
  gap or a conflicting generation/model/form-factor accessory.

Stock and pricing:
- Use only component_candidate_id values present in
  full_category_matrix.category_sections[].products.
- Use only stock_row_id values belonging to the selected product block.
- Every quote line must have stock_row_id.
- quantity must be a positive integer and must fit the selected stock row.
- A stock_row_id is a hard stock bucket. Total quantity across all quote lines
  using the same stock_row_id must fit that row's stock quantity.
- If stock.quantity_is_greater_than=true, stock.quantity_value is a lower
  bound, not an exact count. For a safe quote, rely only on the guaranteed
  minimum stock.quantity_value + 1 unless the matrix explicitly states more.
- unit_price_value and unit_price_currency must be copied from
  price_order_value and price_order_currency.
- line_total_value must equal unit_price_value * quantity.
- All quote lines must use one currency. Do not perform currency conversion.
- quote.total_price_value must equal the exact sum of all line totals.
- Do not add tax, shipping, discounts or fees absent from the matrix.
- For the same product and required quantity, use the lowest-priced eligible
  stock row. If one row cannot satisfy quantity, split the same
  component_candidate_id across stock rows only when technically equivalent and
  same-currency.

Requirement coverage:
- Every explicit measurable requirement and every explicit preference from
  original_request_text and resolved_request must appear in
  quote.requirement_coverage.
- Each selected quote line must either satisfy at least one requirement_id via
  satisfies_requirement_ids or be clearly explained in reason as a mandatory
  enablement/component dependency.
- Allowed outcomes are met, exceeded, equivalent, degraded, partially_met,
  missing, unknown and not_applicable. Backward-compatible substituted and
  not_met are also allowed but prefer the new outcomes.
- Every locked/non_negotiable requirement must be met for complete modes.
- Do not mark a compound target as met merely because its total capacity or
  headline count matches. If explicit sub-attributes differ, such as total RAM
  matching while module count/per-module capacity differs, report equivalent,
  degraded or different behavior through requirement_coverage and
  key_deviations/deviation_notes.
- In partial_build and anchor_only, missing important targets and component
  shortages must be listed in both requirement_coverage and procurement_gaps.
- Every degraded, partially_met, missing, unknown, substituted or not_met item
  must have a corresponding key_deviations/deviation_notes or procurement_gaps
  entry.
- Do not hide a downgrade inside why_selected, assumptions or engineer_checks.

Procurement gaps:
- For every missing, short, omitted or unproven requested item, return a
  structured procurement_gaps object with requirement_id, role, requested,
  status (not_in_matrix, out_of_stock, quantity_shortage,
  no_compatible_item_proven, not_included or unknown), required_for
  (operational_readiness, requested_spec or optional_preference), impact and
  next_action.
- Missing unselected items in procurement_gaps do not by themselves force
  no_recommendation in best_available mode.

Deviation reporting:
- For every non-exact target, degraded line or unmet preference, return a
  structured key_deviations/deviation_notes object:
  requirement_id, requested, offered, direction ("upgrade", "downgrade" or
  "different"), severity ("minor" or "material"), impact and reason.
- A material downgrade may be quoted only when it still credibly performs the
  customer's core task and the substitution policy permits it.

Price audit:
- price_audit is manager-facing, not a mechanical gate. Include it when useful.
- It should state the selected fulfilment mode, why lower modes were not chosen
  merely for price, and notable cheaper rejected alternatives when they exist.
- If you reject an apparent cheaper complete/preconfigured system, price_audit
  must state the row and the concrete technical or stock reason.

No recommendation policy:
- Return status="no_recommendation" only after checking exact fulfilment and all
  policy-permitted analog modes.
- Do not return no_recommendation merely because an exact product, vendor,
  model, generation or target value is unavailable when substitutes are allowed.
- Return no_recommendation only when: an exact-only request cannot be fulfilled;
  no stocked/priced anchor product of the requested functional class exists;
  every anchor violates a locked safety/legal/hard installation boundary;
  no usable price/stock row exists for a relevant anchor; required lines cannot
  be quoted in one currency; or every available analog fails the core task.
- failed_requirements must contain only non_negotiable blockers or core-task
  blockers. Unavailable targets/preferences belong in deviation_notes for a
  quote, or in analog_attempt/best_near_miss for a true no_recommendation.
- If there is a coherent stocked analog but it misses vendor, generation,
  exact CPU model, exact controller, exact license, exact rail kit, exact HBA,
  exact storage interface or exact spare quantity targets, return a quote with
  selection_mode="downgraded_complete", "partial_build" or "anchor_only" as
  appropriate unless that miss makes the customer's core task impossible or the
  original request forbids substitutes.
- For no_recommendation, report exact_attempt, analog_attempt,
  failed_requirements, best_near_miss and recommended_next_actions when known.

Final self-check:
- Before returning JSON, verify: every selected ID exists, every stock_row_id
  belongs to its component_candidate_id, quantities fit stock, prices and
  currencies match the matrix, line totals and quote total are correct, all
  lines use one currency, every line is covered by checked_facts, every line
  maps to a requirement or mandatory enablement role, requirement_coverage is
  complete, every downgrade appears in deviation_notes, blocking_mismatches is
  empty, no included component is purchased again unnecessarily, no clearly
  required mounting/carrier/cooling/cable/adapter/license/enablement item for
  the selected BOM's physical usability is missing, the selected result belongs
  to the best available fulfilment mode, and engineer_checks is
  configuration-specific.
- If any self-check fails, choose another valid configuration or return
  no_recommendation. Do not return a rough draft for later cleanup.

Return one of these shapes:
{
  "status": "quote",
  "quote": {
    "title": "...",
    "client_status_label": "Ближайший складской вариант требует добора",
    "solution_scope": "complete_system",
    "substitution_policy": "allowed_with_disclosed_downgrade",
    "selection_mode": "exact_complete",
    "completeness_status": "complete | partial | anchor_only",
    "operational_status": "incomplete_needs_procurement",
    "anchor_component_candidate_id": "...",
    "lines": [
      {
        "role": "...",
        "component_candidate_id": "...",
        "stock_row_id": "...",
        "quantity": 1,
        "unit_price_value": "0.00",
        "unit_price_currency": "USD",
        "line_total_value": "0.00",
        "line_total_currency": "USD",
        "satisfies_requirement_ids": ["R1"],
        "reason": "..."
      }
    ],
    "total_price_value": "0.00",
    "total_price_currency": "USD",
    "client_summary": "...",
    "coverage_summary": "...",
    "requirement_coverage": [
      {
        "requirement_id": "R1",
        "requirement": "...",
        "priority": "non_negotiable",
        "outcome": "met",
        "requested": "...",
        "offered": "...",
        "evidence": ["..."],
        "impact": "..."
      }
    ],
    "compatibility_check": {
      "status": "compatible_selected_lines",
      "checked_facts": ["..."],
      "blocking_mismatches": [],
      "selected_line_conflicts": [],
      "unresolved_risks": []
    },
    "why_selected": "...",
    "key_deviations": [
      {
        "requirement_id": "R2",
        "requested": "...",
        "offered": "...",
        "direction": "downgrade",
        "severity": "minor",
        "impact": "...",
        "reason": "..."
      }
    ],
    "procurement_gaps": [
      {
        "requirement_id": "R3",
        "role": "...",
        "requested": "...",
        "status": "not_in_matrix",
        "required_for": "requested_spec",
        "impact": "...",
        "next_action": "..."
      }
    ],
    "deviation_notes": [
      {
        "requirement_id": "R2",
        "requested": "...",
        "offered": "...",
        "direction": "downgrade",
        "severity": "minor",
        "impact": "...",
        "reason": "..."
      }
    ],
    "price_audit": [
      {
        "scope": "configuration",
        "result": "...",
        "evidence": ["..."]
      }
    ],
    "assumptions": [],
    "engineer_checks": ["..."]
  },
  "general_notes": []
}

or:
{
  "status": "no_recommendation",
  "no_recommendation": {
    "summary": "...",
    "substitution_policy": "allowed_with_disclosed_downgrade",
    "exact_attempt": {"result": "failed", "blockers": ["..."]},
    "analog_attempt": {"result": "failed", "blockers": ["..."]},
    "failed_requirements": [
      {"requirement_id": "R1", "requirement": "...", "reason": "..."}
    ],
    "best_near_miss": [
      {
        "component_candidate_ids": ["..."],
        "stock_row_ids": ["..."],
        "why_not_quoteable": "..."
      }
    ],
    "recommended_next_actions": ["..."]
  },
  "general_notes": []
}
""".strip()
    user_prompt = json.dumps(
        {
            "task": "choose_best_available_stocked_quote_or_narrow_no_recommendation",
            "original_request_text": user_request,
            "resolved_request": resolved_request_payload,
            "selection_contract": selection_contract_payload,
            "matrix_policy": matrix_package.payload.get("matrix_policy", {}),
            "row_legend": matrix_package.payload.get("row_legend", {}),
            "full_category_matrix": matrix_package.payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return system_prompt, user_prompt


def _composer_prompt_version(settings: LlmSettings) -> str:
    version = str(settings.v3_full_category_contract_version or "").strip().lower()
    if version in {"v7_1", "v7.1", "7.1"}:
        return COMPOSER_PROMPT_VERSION_V7_1
    return COMPOSER_PROMPT_VERSION_V7 if version == "v7" else "composer_v6"


def _build_full_category_quote_prompts_v7(
    *,
    user_request: str,
    resolved_request: Mapping[str, Any] | None,
    matrix_package: FullCategoryMatrixPackage,
) -> tuple[str, str]:
    resolved_request_payload = _jsonable_mapping(resolved_request or {})
    anchor_candidate_manifest = _anchor_candidate_manifest(
        resolved_request_payload,
        matrix_package=matrix_package,
    )
    selection_contract_payload = _selection_contract_payload(
        resolved_request_payload,
        matrix_package=matrix_package,
        anchor_candidate_manifest=anchor_candidate_manifest,
    )
    system_prompt = """
You are Stock Configurator Composer v7.1.

You receive a customer request, a v7 resolved_request ledger, a commercial
selection contract, a compact matrix_index and the full lossless category
matrix grouped by distributor category sections.

Architecture boundary:
- Use only products and stock rows present in the supplied matrix.
- Do not invent stock, prices, capabilities, included items, compatibility,
  warehouses, licenses, accessories or customer-owned components.
- Code has not ranked, shortlisted or compatibility-filtered the matrix.
- You are the single semantic selection pass. There is no repair pass.
- Return JSON only. All natural-language fields must be in Russian.

Normal result:
- For best_available, the default output is status="quote".
- If complete fulfilment is impossible, return the best honest partial stocked
  offer with gaps and deviations.
- no_recommendation is allowed only when the selection contract allows it and
  there is no useful stocked subset, every relevant row violates a locked
  boundary, or an exact-only request cannot be fulfilled.

ANCHOR GATE - HARD OUTPUT CONTRACT

For every resolved_request object, read its object_contract before selecting
components.

When anchor_policy is "required" or "self" and anchor_candidate_count is
greater than zero:
- you MUST select exactly one anchor line for that object before selecting
  residual components;
- the selected anchor component_candidate_id MUST belong to
  object_contract.anchor_candidate_ids;
- selection_mode="partial_without_anchor" is forbidden for that object;
- omission of the anchor is not a procurement gap and is not an acceptable
  shortcut;
- if the closest anchor is incomplete or materially different, select it and
  use partial_with_anchor with explicit deviations and gaps.

When anchor_candidate_count is zero:
- do not invent an anchor;
- partial_without_anchor is allowed only when
  object_contract.partial_without_anchor_allowed=true;
- anchor_search_audit must state outcome="none_available_in_manifest".

A configured_system or complete_system quote without an anchor is invalid when
its object_contract lists one or more anchor candidates.

Selection ladder:
1. exact_complete
2. equivalent_complete
3. degraded_complete
4. partial_with_anchor
5. partial_without_anchor
6. no_recommendation
Pick the highest available class first, then minimize deviation severity, then
price. Never choose a worse class only because it is cheaper.

Universal algorithm for every object:
A. Candidate discovery:
- Inspect every product in matrix_index, then verify facts in full product
  blocks. The index is not a shortlist.
- Consider complete/preconfigured products, platforms, standalone items,
  components, accessories, consumables, licenses and services when relevant.

B1. ANCHOR SEARCH CHECKPOINT

For every object:
1. Read every candidate in anchor_candidate_manifest for that object.
2. Select the closest permissible anchor under the lexicographic fit order.
3. Record the decision in anchor_search_audit.
4. Only after an anchor is selected may you subtract included coverage and
   choose residual component lines.

Do not proceed directly from request requirements to component selection for an
object whose anchor_policy is required.

B. Included coverage subtraction:
- If a selected stocked product explicitly includes a requested function,
  component, port, PSU, RAM, storage or license in sufficient quantity, mark
  that coverage and do not buy it again.
- Supported or optional is not included.

C. Residual fulfilment:
- For every requested item, compute requested, included_covered,
  selected_covered and remaining.
- Search the whole matrix for stocked rows that can cover remaining quantity
  before creating a gap.
- Operational and spare quantities are separate items.
- Missing enablement does not erase useful stocked requested lines. Include
  safe useful lines and put the missing enabler into procurement_gaps.

D. Compatibility decision for each selected line:
- technical_status must be one of anchor, confirmed_fit, review_required,
  independent_item.
- confirmed_fit requires direct matrix facts.
- review_required is allowed when no explicit conflict exists but vendor
  certification, firmware, carrier, licensing or customer environment needs
  checking.
- independent_item is useful stocked partial coverage without claiming anchor
  integration.
- If there is an explicit conflict, do not include the line. Create a gap.

E. Same-role dominance:
- For every selected line, compare plausible same-role candidates from the
  whole matrix.
- A cheaper candidate dominates when it covers the same residual item, meets
  locked constraints, is no worse on explicit attributes, has sufficient stock
  or no less partial coverage, has same/better technical_status, has no
  matrix-supported defect and uses the selected currency.
- A more expensive line is allowed only with a matrix-supported reason.
- If two candidates have equally unknown certification, choose the cheaper one
  and attach the same engineer check.

F. Choose complete or partial result:
- Compare whole configurations, not isolated anchors.
- If complete is impossible, choose the maximum useful stocked coverage with
  honest gaps. Total price covers only included stocked lines.

Coverage and reporting:
- Every requested_item and every requirement_id from resolved_request must
  appear exactly once in coverage.
- Return one object_results entry and one anchor_search_audit entry for every
  resolved_request object.
- Each quote line must have line_id, object_id, role, covered_item_ids,
  covered_requirement_ids, technical_status, coverage_contributions, fact_ids
  and compatibility_statement.
- compatibility_check.checked_facts must contain one structured entry per
  quote line with matching line_id, component_candidate_id, stock_row_id,
  fact_ids, relationship and conclusion.
- dominance_audit must contain one entry per quote line with matching line_id.
- Every downgrade, partial, missing, different or unknown item must be visible
  in key_deviations or procurement_gaps. Do not hide it in assumptions.
- procurement_gaps must state requested, missing quantity/reason, required_for,
  impact and next_action.
- compatibility_check.status must be one of confirmed_selected_set,
  review_required_selected_set, independent_partial_set.
- engineer_checks must be concrete and tied to review_required lines or
  non-blocking environment checks.

Pricing:
- Every line must use an existing component_candidate_id and stock_row_id.
- quantity is a positive integer and must fit the selected stock row.
- unit price and currency must exactly match the selected stock row.
- line total equals unit price multiplied by quantity.
- All included lines use one currency. Do not convert currencies.
- total_price is the exact sum of included lines only.

Output shape for quote:
{
  "status": "quote",
  "quote": {
    "selection_mode": "exact_complete",
    "completeness_status": "complete",
    "operational_status": "operational",
    "solution_scope": "configured_system",
    "substitution_policy": "allowed_with_disclosed_downgrade",
    "client_status_label": "...",
    "object_results": [],
    "lines": [
      {
        "object_id": "O1",
        "role": "...",
        "line_id": "L1",
        "component_candidate_id": "...",
        "stock_row_id": "...",
        "quantity": 1,
        "unit_price_value": "0.0000",
        "unit_price_currency": "USD",
        "line_total_value": "0.0000",
        "line_total_currency": "USD",
        "covered_item_ids": ["I1"],
        "covered_requirement_ids": ["R1"],
        "satisfies_requirement_ids": ["R1"],
        "technical_status": "confirmed_fit",
        "coverage_contributions": [],
        "fact_ids": ["F:...:item_name"],
        "compatibility_statement": "...",
        "reason": "..."
      }
    ],
    "total_price": {"value": "0.0000", "currency": "USD"},
    "total_price_value": "0.0000",
    "total_price_currency": "USD",
    "client_summary": "...",
    "coverage_summary": "...",
    "coverage": [],
    "requirement_coverage": [],
    "key_deviations": [],
    "procurement_gaps": [],
    "compatibility_check": {
      "status": "review_required_selected_set",
      "checked_facts": [
        {
          "line_id": "L1",
          "component_candidate_id": "...",
          "stock_row_id": "...",
          "fact_ids": ["F:...:item_name"],
          "relationship": "anchor_identity",
          "conclusion": "..."
        }
      ],
      "blocking_mismatches": [],
      "selected_line_conflicts": [],
      "unresolved_risks": []
    },
    "anchor_search_audit": [
      {
        "object_id": "O1",
        "anchor_policy": "required",
        "anchor_candidate_count": 1,
        "outcome": "selected",
        "selected_anchor_line_id": "L1",
        "selected_anchor_component_candidate_id": "...",
        "reason": "..."
      }
    ],
    "dominance_audit": [],
    "why_selected": "...",
    "assumptions": [],
    "engineer_checks": []
  },
  "general_notes": []
}

Output shape for no_recommendation:
{
  "status": "no_recommendation",
  "no_recommendation": {
    "reason_code": "exact_only_unavailable",
    "summary": "...",
    "failed_requirements": [],
    "best_near_miss": [],
    "recommended_next_actions": []
  },
  "general_notes": []
}
""".strip()
    full_matrix = _matrix_payload_for_prompt_v7_1(
        matrix_package,
        anchor_candidate_manifest=anchor_candidate_manifest,
    )
    user_payload = {
        "TASK_CAPSULE": {
            "task": (
                "Select one best available stocked quote. Quote first. Use "
                "anchor_candidate_manifest and selection_contract before components. "
                "Subtract included coverage, fill residuals, run same-role price "
                "challenge, disclose gaps, JSON only."
            ),
            "original_request_text": user_request,
        },
        "resolved_request": resolved_request_payload,
        "selection_contract": selection_contract_payload,
        "anchor_candidate_manifest": anchor_candidate_manifest,
        "matrix_index": matrix_package.payload.get("matrix_index", []),
        "row_legend": matrix_package.payload.get("row_legend", {}),
        "BEGIN_FULL_CATEGORY_MATRIX": "BEGIN FULL CATEGORY MATRIX",
        "full_category_matrix": full_matrix,
        "END_FULL_CATEGORY_MATRIX": "END FULL CATEGORY MATRIX",
        "FINAL_RESPONSE_GATE": [
            (
                "For each required-anchor object with anchor_candidate_count > 0, "
                "select one anchor from anchor_candidate_ids. partial_without_anchor is forbidden."
            ),
            "Return one object_result and one anchor_search_audit entry for every object.",
            (
                "Return one compatibility checked_facts entry and one dominance_audit "
                "entry for every quote line."
            ),
            "IDs in those entries must exactly match the corresponding quote line.",
            "Return JSON only.",
        ],
    }
    return system_prompt, json.dumps(
        user_payload,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
    )


def _matrix_payload_for_prompt_v7_1(
    matrix_package: FullCategoryMatrixPackage,
    *,
    anchor_candidate_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_category_ids = {
        str(item.get("category_id") or "").strip()
        for item in anchor_candidate_manifest
        if str(item.get("category_role") or "").strip() == "anchor"
    }
    fallback_category_ids = {
        str(item.get("category_id") or "").strip()
        for item in anchor_candidate_manifest
        if str(item.get("category_role") or "").strip() == "fallback_anchor"
    }
    sections = matrix_package.payload.get("category_sections", [])
    ordered_sections = _ordered_category_sections(
        sections,
        anchor_category_ids=anchor_category_ids,
        fallback_category_ids=fallback_category_ids,
    )
    return {
        "schema_version": matrix_package.payload.get("schema_version"),
        "matrix_payload_schema_version": matrix_package.payload.get(
            "matrix_payload_schema_version",
        ),
        "distributor_code": matrix_package.payload.get("distributor_code"),
        "category_id": matrix_package.payload.get("category_id"),
        "category_ids": matrix_package.payload.get("category_ids"),
        "model": matrix_package.payload.get("model"),
        "category_sections": ordered_sections,
        "diagnostics": matrix_package.payload.get("diagnostics", {}),
    }


def _ordered_category_sections(
    sections: Any,
    *,
    anchor_category_ids: set[str],
    fallback_category_ids: set[str],
) -> list[Any]:
    if not isinstance(sections, Sequence) or isinstance(
        sections,
        (str, bytes, bytearray),
    ):
        return []
    indexed = list(enumerate(sections))

    def section_rank(item: tuple[int, Any]) -> tuple[int, int]:
        index, section = item
        category_id = (
            str(section.get("category_id") or "").strip()
            if isinstance(section, Mapping)
            else ""
        )
        if category_id in anchor_category_ids:
            return (0, index)
        if category_id in fallback_category_ids:
            return (1, index)
        return (2, index)

    return [section for _, section in sorted(indexed, key=section_rank)]


def _selection_contract_payload(
    resolved_request: Mapping[str, Any],
    *,
    matrix_package: FullCategoryMatrixPackage,
    anchor_candidate_manifest: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    procurement_mode = str(resolved_request.get("procurement_mode") or "").strip()
    request_mode = str(resolved_request.get("request_mode") or "").strip()
    substitution_policy = str(resolved_request.get("substitution_policy") or "").strip()
    exact_only = (
        procurement_mode == "exact_only"
        or request_mode == "exact_only"
        or substitution_policy == "forbidden"
    )
    allow_partial_quote = bool(
        resolved_request.get(
            "allow_partial_quote",
            resolved_request.get("allow_partial_offer", not exact_only),
        )
    )
    diagnostics = _safe_mapping(matrix_package.payload.get("diagnostics"))
    manifest = list(anchor_candidate_manifest or [])
    object_contracts = _object_contracts_payload(
        resolved_request,
        anchor_candidate_manifest=manifest,
        allow_partial_offer=allow_partial_quote,
    )
    flattened_anchor_candidate_ids = sorted(
        {
            str(item.get("component_candidate_id") or "").strip()
            for item in manifest
            if str(item.get("component_candidate_id") or "").strip()
        }
    )
    return {
        "contract_version": SELECTION_CONTRACT_VERSION_V7_1,
        "quote_first": True,
        "allow_partial_offer": allow_partial_quote,
        "allow_partial_without_anchor": all(
            bool(item.get("partial_without_anchor_allowed"))
            for item in object_contracts
        )
        if object_contracts
        else allow_partial_quote,
        "allow_review_required_lines": True,
        "single_currency_required": True,
        "engineer_review_required": True,
        "no_recommendation_allowed": bool(
            exact_only or matrix_package.status != MATRIX_READY_FOR_LLM
        ),
        "no_recommendation_policy": (
            "In best_available mode return quote when any useful stocked subset "
            "exists. no_recommendation is for exact-only failure, empty matrix, "
            "or locked boundary failure only."
        ),
        "object_contracts": object_contracts,
        "anchor_candidate_ids": flattened_anchor_candidate_ids,
        "anchor_detection": "backend_mechanical_manifest_from_category_roles",
        "anchor_candidate_manifest_count": len(manifest),
        "allowed_currencies": _matrix_price_currencies(matrix_package),
        "requested_warehouse": _requested_warehouse(resolved_request),
        "renderer_policy": "client_sheet_plus_internal_sheets",
        "partial_quote_allowed": allow_partial_quote,
        "exact_only": exact_only,
        "matrix_has_stocked_priced_rows": matrix_package.status == MATRIX_READY_FOR_LLM,
        "matrix_component_count": diagnostics.get("component_count"),
        "matrix_stock_row_count": diagnostics.get("stock_row_count"),
        "matrix_index_count": diagnostics.get("matrix_index_count"),
    }


def _anchor_candidate_manifest(
    resolved_request: Mapping[str, Any],
    *,
    matrix_package: FullCategoryMatrixPackage,
) -> list[dict[str, Any]]:
    objects = _resolved_request_objects_for_contract(resolved_request)
    if not objects:
        return []
    role_index = _category_role_index(matrix_package.payload.get("distributor_code"))
    if not role_index:
        return []
    sections_by_category_id = _category_sections_by_id(matrix_package)
    manifest: list[dict[str, Any]] = []
    for request_object in objects:
        object_id = request_object["object_id"]
        anchor_category_ids, fallback_category_ids = _object_anchor_category_ids(
            request_object,
            resolved_request=resolved_request,
            role_index=role_index,
            sections_by_category_id=sections_by_category_id,
        )
        manifest.extend(
            _anchor_manifest_entries_for_object(
                object_id=object_id,
                category_ids=anchor_category_ids,
                category_role="anchor",
                sections_by_category_id=sections_by_category_id,
                role_index=role_index,
            )
        )
        manifest.extend(
            _anchor_manifest_entries_for_object(
                object_id=object_id,
                category_ids=fallback_category_ids,
                category_role="fallback_anchor",
                sections_by_category_id=sections_by_category_id,
                role_index=role_index,
            )
        )
    return manifest


def _category_role_index(distributor_code: Any) -> dict[str, dict[str, Any]]:
    if str(distributor_code or "").strip().casefold() != "ocs":
        return {}
    result: dict[str, dict[str, Any]] = {}
    for anchor in load_ocs_anchor_categories():
        if not anchor.enable_allowed:
            continue
        result[anchor.category_id] = {
            "group": anchor.group,
            "role": anchor.role,
            "category_id": anchor.category_id,
            "category_kind": anchor.category_kind,
            "base_device_allowed": anchor.base_device_allowed,
        }
    return result


def _category_sections_by_id(
    matrix_package: FullCategoryMatrixPackage,
) -> dict[str, Mapping[str, Any]]:
    sections: dict[str, Mapping[str, Any]] = {}
    for raw_section in matrix_package.payload.get("category_sections", []):
        if not isinstance(raw_section, Mapping):
            continue
        category_id = str(raw_section.get("category_id") or "").strip()
        if category_id:
            sections[category_id] = raw_section
    return sections


def _object_anchor_category_ids(
    request_object: Mapping[str, Any],
    *,
    resolved_request: Mapping[str, Any],
    role_index: Mapping[str, Mapping[str, Any]],
    sections_by_category_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    object_id = str(request_object.get("object_id") or "").strip()
    object_plan = _retrieval_plan_for_object(resolved_request, object_id)
    explicit_anchor_ids = _existing_category_ids(
        object_plan.get("anchor_category_ids"),
        sections_by_category_id=sections_by_category_id,
    )
    explicit_fallback_ids = _existing_category_ids(
        object_plan.get("fallback_category_ids"),
        sections_by_category_id=sections_by_category_id,
    )
    if explicit_anchor_ids or explicit_fallback_ids:
        return explicit_anchor_ids, explicit_fallback_ids

    anchor_policy = _object_anchor_policy(request_object)
    if anchor_policy not in {"required", "self"}:
        return [], []
    anchor_ids: list[str] = []
    for category_id in sections_by_category_id:
        role = role_index.get(category_id)
        if role and bool(role.get("base_device_allowed")):
            anchor_ids.append(category_id)
    return anchor_ids, []


def _retrieval_plan_for_object(
    resolved_request: Mapping[str, Any],
    object_id: str,
) -> dict[str, Any]:
    retrieval_plan = _safe_mapping(resolved_request.get("retrieval_plan"))
    objects = retrieval_plan.get("objects")
    if isinstance(objects, Sequence) and not isinstance(objects, (str, bytes, bytearray)):
        for raw_item in objects:
            if not isinstance(raw_item, Mapping):
                continue
            if str(raw_item.get("object_id") or "").strip() == object_id:
                return dict(raw_item)
    return {
        "anchor_category_ids": _non_empty_text_items(
            retrieval_plan.get("anchor_category_ids") or []
        ),
        "component_category_ids": _non_empty_text_items(
            retrieval_plan.get("component_category_ids") or []
        ),
        "fallback_category_ids": _non_empty_text_items(
            retrieval_plan.get("fallback_category_ids") or []
        ),
    }


def _existing_category_ids(
    value: Any,
    *,
    sections_by_category_id: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    result: list[str] = []
    for category_id in _non_empty_text_items(value or []):
        if category_id in sections_by_category_id and category_id not in result:
            result.append(category_id)
    return result


def _anchor_manifest_entries_for_object(
    *,
    object_id: str,
    category_ids: Sequence[str],
    category_role: str,
    sections_by_category_id: Mapping[str, Mapping[str, Any]],
    role_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for category_id in category_ids:
        section = sections_by_category_id.get(category_id)
        if not section:
            continue
        role = role_index.get(category_id, {})
        for raw_product in section.get("products", []) or []:
            if not isinstance(raw_product, Mapping):
                continue
            stock_rows = _manifest_stock_rows(raw_product.get("stock_rows"))
            if not stock_rows:
                continue
            product = _safe_mapping(raw_product.get("product"))
            component_candidate_id = str(
                raw_product.get("component_candidate_id") or ""
            ).strip()
            if not component_candidate_id:
                continue
            entries.append(
                {
                    "object_id": object_id,
                    "component_candidate_id": component_candidate_id,
                    "category_id": category_id,
                    "category_role": category_role,
                    "catalog_identity": _catalog_identity(product),
                    "category_role_name": role.get("role"),
                    "category_kind": role.get("category_kind"),
                    "stock_rows": stock_rows,
                }
            )
    return entries


def _manifest_stock_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return rows
    for raw_row in value:
        if not isinstance(raw_row, Mapping):
            continue
        quantity_value = _int_or_none(raw_row.get("quantity_value"))
        price_order_value = raw_row.get("price_order_value")
        price_order_currency = str(raw_row.get("price_order_currency") or "").strip()
        stock_row_id = str(raw_row.get("stock_row_id") or "").strip()
        if not stock_row_id or quantity_value is None or quantity_value <= 0:
            continue
        if price_order_value in (None, "") or not price_order_currency:
            continue
        rows.append(
            {
                "stock_row_id": stock_row_id,
                "quantity_value": quantity_value,
                "quantity_is_greater_than": _bool_or_false(
                    raw_row.get("quantity_is_greater_than")
                ),
                "price_order_value": str(price_order_value),
                "price_order_currency": price_order_currency,
            }
        )
    return rows


def _catalog_identity(product: Mapping[str, Any]) -> str:
    parts = _non_empty_text_items(
        [
            str(product.get("producer") or ""),
            str(product.get("part_number") or ""),
            str(product.get("item_name") or product.get("product_name") or ""),
        ]
    )
    return " | ".join(parts)[:500]


def _object_contracts_payload(
    resolved_request: Mapping[str, Any],
    *,
    anchor_candidate_manifest: Sequence[Mapping[str, Any]],
    allow_partial_offer: bool,
) -> list[dict[str, Any]]:
    objects = _resolved_request_objects_for_contract(resolved_request)
    if not objects:
        return []
    anchor_ids_by_object: dict[str, list[str]] = {}
    for raw_entry in anchor_candidate_manifest:
        object_id = str(raw_entry.get("object_id") or "").strip()
        component_id = str(raw_entry.get("component_candidate_id") or "").strip()
        if not object_id or not component_id:
            continue
        anchor_ids_by_object.setdefault(object_id, [])
        if component_id not in anchor_ids_by_object[object_id]:
            anchor_ids_by_object[object_id].append(component_id)

    contracts: list[dict[str, Any]] = []
    for request_object in objects:
        object_id = str(request_object.get("object_id") or "").strip()
        anchor_policy = _object_anchor_policy(request_object)
        candidate_ids = sorted(anchor_ids_by_object.get(object_id, []))
        if anchor_policy in {"required", "self"} and candidate_ids:
            partial_without_anchor_allowed = False
        elif anchor_policy == "required":
            partial_without_anchor_allowed = allow_partial_offer
        elif anchor_policy == "not_required":
            partial_without_anchor_allowed = True
        else:
            partial_without_anchor_allowed = allow_partial_offer
        contracts.append(
            {
                "object_id": object_id,
                "deliverable_scope": request_object.get("deliverable_scope"),
                "object_quantity": request_object.get("object_quantity"),
                "primary_item_id": request_object.get("primary_item_id"),
                "anchor_policy": anchor_policy,
                "anchor_candidate_count": len(candidate_ids),
                "anchor_candidate_ids": candidate_ids,
                "partial_without_anchor_allowed": partial_without_anchor_allowed,
            }
        )
    return contracts


def _resolved_request_objects_for_contract(
    resolved_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    root_scope = str(resolved_request.get("deliverable_scope") or "").strip()
    raw_objects = _raw_mapping_sequence(resolved_request.get("objects"))
    for index, raw_object in enumerate(raw_objects, start=1):
        object_id = str(raw_object.get("object_id") or f"O{index}").strip()
        if not object_id:
            object_id = f"O{index}"
        deliverable_scope = str(
            raw_object.get("deliverable_scope") or root_scope or "standalone_product"
        ).strip()
        primary_item_id = str(raw_object.get("primary_item_id") or "").strip()
        if not primary_item_id:
            for raw_item in _raw_mapping_sequence(raw_object.get("requested_items")):
                item_id = str(raw_item.get("item_id") or "").strip()
                item_kind = str(raw_item.get("item_kind") or "").strip()
                if item_id and item_kind == "primary_product":
                    primary_item_id = item_id
                    break
            if not primary_item_id:
                for raw_item in _raw_mapping_sequence(raw_object.get("requested_items")):
                    item_id = str(raw_item.get("item_id") or "").strip()
                    if item_id:
                        primary_item_id = item_id
                        break
        result.append(
            {
                **dict(raw_object),
                "object_id": object_id,
                "deliverable_scope": deliverable_scope,
                "object_quantity": _int_or_none(
                    raw_object.get("object_quantity")
                    or raw_object.get("requested_quantity")
                    or resolved_request.get("requested_quantity")
                )
                or 1,
                "primary_item_id": primary_item_id,
                "anchor_policy": str(raw_object.get("anchor_policy") or "").strip()
                or _anchor_policy_for_scope(deliverable_scope),
            }
        )
    return result


def _object_anchor_policy(request_object: Mapping[str, Any]) -> str:
    policy = str(request_object.get("anchor_policy") or "").strip()
    if policy in {"required", "self", "not_required"}:
        return policy
    return _anchor_policy_for_scope(str(request_object.get("deliverable_scope") or ""))


def _anchor_policy_for_scope(scope: str) -> str:
    normalized = str(scope or "").strip()
    if normalized in {"complete_system", "configured_system", "multi_product_solution"}:
        return "required"
    if normalized == "standalone_product":
        return "self"
    if normalized in {"accessory", "replacement_component", "expansion_or_upgrade"}:
        return "not_required"
    return "required"


def _matrix_price_currencies(matrix_package: FullCategoryMatrixPackage) -> list[str]:
    currencies: list[str] = []
    for raw_section in matrix_package.payload.get("category_sections", []):
        if not isinstance(raw_section, Mapping):
            continue
        for raw_product in raw_section.get("products", []):
            if not isinstance(raw_product, Mapping):
                continue
            for raw_stock_row in raw_product.get("stock_rows", []):
                if not isinstance(raw_stock_row, Mapping):
                    continue
                currency = str(raw_stock_row.get("price_order_currency") or "").strip()
                if currency and currency not in currencies:
                    currencies.append(currency)
    return currencies


def _requested_warehouse(resolved_request: Mapping[str, Any]) -> str | None:
    for key in ("requested_warehouse", "warehouse", "shipment_city", "location"):
        value = str(resolved_request.get(key) or "").strip()
        if value:
            return value
    for requirement in resolved_request.get("requirements", []) or []:
        if not isinstance(requirement, Mapping):
            continue
        dimension = str(requirement.get("dimension") or requirement.get("key") or "")
        if dimension not in {"warehouse", "delivery", "location"}:
            continue
        value = str(
            requirement.get("value")
            or requirement.get("requested")
            or requirement.get("source_phrase")
            or ""
        ).strip()
        if value:
            return value
    return None


def parse_full_category_composer_payload(
    payload: Mapping[str, Any],
) -> FullCategoryComposerPayload:
    normalized: dict[str, Any] = dict(payload)
    if "quote" not in normalized:
        for key in ("recommendation", "selected_quote", "commercial_quote"):
            if isinstance(normalized.get(key), Mapping):
                normalized["quote"] = normalized[key]
                break
    if normalized.get("status") not in {"quote", "no_recommendation"}:
        if isinstance(normalized.get("no_recommendation"), Mapping):
            normalized["status"] = "no_recommendation"
        elif isinstance(normalized.get("quote"), Mapping):
            normalized["status"] = "quote"
    quote = normalized.get("quote")
    if isinstance(quote, Mapping):
        normalized["quote"] = _normalize_quote_shape(dict(quote))
    return FullCategoryComposerPayload.model_validate(normalized)


def _normalize_quote_shape(quote: dict[str, Any]) -> dict[str, Any]:
    raw_lines = quote.get("lines")
    if isinstance(raw_lines, list):
        quote["lines"] = [
            _normalize_quote_line_shape(line) if isinstance(line, Mapping) else line
            for line in raw_lines
        ]
    return quote


def _normalize_quote_line_shape(line: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(line)
    raw_contributions = normalized.get("coverage_contributions")
    if isinstance(raw_contributions, str):
        normalized["coverage_contributions"] = [
            {"description": raw_contributions},
        ]
    elif isinstance(raw_contributions, list):
        normalized["coverage_contributions"] = [
            contribution
            if isinstance(contribution, Mapping)
            else {"description": str(contribution)}
            for contribution in raw_contributions
            if contribution is not None and str(contribution).strip()
        ]
    return normalized


def _unvalidated_quote_payload(
    quote: FullCategoryQuotePayload,
    *,
    matrix_package: FullCategoryMatrixPackage,
) -> dict[str, Any]:
    payload = _jsonable_mapping(quote.model_dump(exclude_none=True))
    total_price = _safe_mapping(payload.get("total_price"))
    if total_price:
        payload.setdefault(
            "total_price_value",
            total_price.get("value") or total_price.get("amount"),
        )
        payload.setdefault("total_price_currency", total_price.get("currency"))

    stock_rows_by_id, _ = _package_row_indexes(matrix_package)
    lines = _raw_mapping_sequence(payload.get("lines"))
    normalized_lines: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        normalized_line = dict(line)
        normalized_line.setdefault("line_id", f"L{line_index + 1}")
        stock_row_id = str(normalized_line.get("stock_row_id") or "").strip()
        row = _safe_mapping(stock_rows_by_id.get(stock_row_id))
        if row:
            stock = _safe_mapping(row.get("stock"))
            product = _safe_mapping(row.get("product"))
            normalized_line.setdefault("producer", product.get("producer"))
            normalized_line.setdefault("part_number", product.get("part_number"))
            normalized_line.setdefault(
                "item_name",
                product.get("item_name") or product.get("product_name"),
            )
            normalized_line.setdefault("category_id", product.get("category_id"))
            normalized_line.setdefault("shipment_city", stock.get("shipment_city"))
            normalized_line.setdefault("location", stock.get("location"))
            normalized_line.setdefault(
                "available_quantity",
                _stock_effective_available_quantity(stock),
            )
            normalized_line.setdefault(
                "quantity_value",
                _int_or_none(stock.get("quantity_value")),
            )
            normalized_line.setdefault(
                "quantity_is_greater_than",
                _bool_or_false(stock.get("quantity_is_greater_than")),
            )
        normalized_lines.append(normalized_line)
    payload["lines"] = normalized_lines
    payload["engineering_review_required"] = True
    payload["code_validation_bypassed"] = True
    return payload


def validate_full_category_quote_payload(
    payload: FullCategoryComposerPayload,
    *,
    matrix_package: FullCategoryMatrixPackage,
    resolved_request: Mapping[str, Any] | None = None,
    selection_contract: Mapping[str, Any] | None = None,
    anchor_candidate_manifest: Sequence[Mapping[str, Any]] | None = None,
) -> FullCategoryQuoteValidation:
    error_details: list[dict[str, Any]] = []
    if payload.quote is None:
        return FullCategoryQuoteValidation(
            status="mechanically_invalid",
            errors=["schema.quote_missing"],
            error_details=[
                _validation_error_detail(
                    "schema.quote_missing",
                    path="quote",
                    stage="schema",
                )
            ],
        )
    if not payload.quote.lines:
        return FullCategoryQuoteValidation(
            status="mechanically_invalid",
            errors=["schema.quote_lines_missing"],
            error_details=[
                _validation_error_detail(
                    "schema.quote_lines_missing",
                    path="quote.lines",
                    stage="schema",
                )
            ],
        )

    stock_rows_by_id, stock_rows_by_component_id = _package_row_indexes(matrix_package)
    fact_ids_by_component_id = _package_fact_id_index(matrix_package)
    ledger_item_ids, ledger_requirement_ids = _resolved_request_id_sets(
        resolved_request or {},
    )
    stock_rows_by_unique_suffix = _stock_row_unique_suffix_index(stock_rows_by_id)
    errors: list[str] = []
    warnings: list[str] = []
    validated_lines: list[dict[str, Any]] = []
    used_quantity_by_stock_row_id: dict[str, int] = {}
    total_by_currency: dict[str, Decimal] = {}
    contract_payload = _safe_mapping(selection_contract)
    object_contracts = _raw_mapping_sequence(contract_payload.get("object_contracts"))
    if not contract_payload:
        manifest = list(anchor_candidate_manifest or [])
        contract_payload = _selection_contract_payload(
            resolved_request or {},
            matrix_package=matrix_package,
            anchor_candidate_manifest=manifest,
        )
        object_contracts = _raw_mapping_sequence(contract_payload.get("object_contracts"))

    _extend_validation_errors(
        errors,
        error_details,
        _quote_contract_errors_v7_1(
            payload.quote,
            selection_contract=contract_payload,
            resolved_request=resolved_request or {},
        ),
    )
    _extend_validation_errors(
        errors,
        error_details,
        _quote_compatibility_errors(
            payload.quote.compatibility_check,
            lines=payload.quote.lines,
            strict_contract=bool(object_contracts),
        ),
    )
    warnings.extend(
        _quote_compatibility_warnings(
            payload.quote.compatibility_check,
            lines=payload.quote.lines,
        )
    )
    warnings.extend(_quote_contract_warnings(payload.quote))
    _extend_validation_errors(
        errors,
        error_details,
        _dominance_audit_reference_errors(
            payload.quote.dominance_audit,
            lines=payload.quote.lines,
            stock_rows_by_id=stock_rows_by_id,
            stock_rows_by_component_id=stock_rows_by_component_id,
            fact_ids_by_component_id=fact_ids_by_component_id,
            strict_contract=bool(object_contracts),
        )
    )

    for line_index, line in enumerate(payload.quote.lines):
        line_errors: list[str] = []
        component_candidate_id = line.component_candidate_id.strip()
        stock_row_id = str(line.stock_row_id or "").strip()
        covered_item_ids = _non_empty_text_items(line.covered_item_ids)
        covered_requirement_ids = _line_requirement_ids(line)
        if ledger_item_ids:
            for item_id in covered_item_ids:
                if item_id not in ledger_item_ids:
                    line_errors.append(f"line_{line_index}:unknown_covered_item_id:{item_id}")
        if ledger_requirement_ids:
            for requirement_id in covered_requirement_ids:
                if requirement_id not in ledger_requirement_ids:
                    line_errors.append(
                        f"line_{line_index}:unknown_covered_requirement_id:{requirement_id}"
                    )
        if str(line.technical_status or "").strip() == "review_required" and not any(
            str(check or "").strip() for check in payload.quote.engineer_checks
        ):
            line_errors.append(f"line_{line_index}:review_required_without_engineer_check")
        if not stock_row_id and component_candidate_id in stock_rows_by_component_id:
            component_stock_rows = stock_rows_by_component_id[component_candidate_id]
            if len(component_stock_rows) == 1:
                stock_row_id = str(component_stock_rows[0].get("stock_row_id") or "")
                warnings.append(
                    f"line_{line_index}:stock_row_id_resolved_from_unique_component_candidate"
                )
            else:
                line_errors.append(f"line_{line_index}:stock_row_id_missing")
        row = stock_rows_by_id.get(stock_row_id)
        if row is None and stock_row_id:
            resolved_row = _resolve_stock_row_by_unique_suffix(
                stock_row_id,
                stock_rows_by_unique_suffix,
            )
            if resolved_row is not None:
                row = resolved_row
                stock_row_id = str(row.get("stock_row_id") or "").strip()
                warnings.append(
                    f"line_{line_index}:stock_row_id_resolved_from_unique_suffix"
                )
        if row is not None:
            row_component_candidate_id = str(
                row.get("component_candidate_id") or ""
            ).strip()
            if row_component_candidate_id and row_component_candidate_id != component_candidate_id:
                if component_candidate_id not in stock_rows_by_component_id:
                    component_candidate_id = row_component_candidate_id
                    warnings.append(
                        f"line_{line_index}:component_candidate_id_resolved_from_stock_row"
                    )
                else:
                    line_errors.append(f"line_{line_index}:stock_row_component_mismatch")
        if component_candidate_id not in stock_rows_by_component_id:
            line_errors.append(f"line_{line_index}:unknown_component_candidate_id")
        line_errors.extend(
            _line_fact_reference_errors(
                line,
                line_index=line_index,
                component_candidate_id=component_candidate_id,
                fact_ids_by_component_id=fact_ids_by_component_id,
            )
        )
        if row is None:
            line_errors.append(f"line_{line_index}:unknown_stock_row_id")
            errors.extend(line_errors)
            continue
        if str(row.get("component_candidate_id") or "").strip() != component_candidate_id:
            line_errors.append(f"line_{line_index}:stock_row_component_mismatch")

        stock = _safe_mapping(row.get("stock"))
        product = _safe_mapping(row.get("product"))
        available_quantity = _stock_effective_available_quantity(stock)
        if available_quantity is None:
            line_errors.append(f"line_{line_index}:stock_quantity_missing")
        elif line.quantity > available_quantity:
            line_errors.append(f"line_{line_index}:quantity_exceeds_stock")
        used_quantity_by_stock_row_id[stock_row_id] = (
            used_quantity_by_stock_row_id.get(stock_row_id, 0) + line.quantity
        )

        unit_price = _decimal_or_none(stock.get("price_order_value"))
        currency = str(stock.get("price_order_currency") or "").strip()
        if unit_price is None:
            line_errors.append(f"line_{line_index}:price_order_value_missing")
        if not currency:
            line_errors.append(f"line_{line_index}:price_order_currency_missing")
        if (
            unit_price is not None
            and line.unit_price_value is not None
            and line.unit_price_value != unit_price
        ):
            line_errors.append(f"line_{line_index}:unit_price_mismatch")
        if (
            currency
            and line.unit_price_currency
            and line.unit_price_currency.strip() != currency
        ):
            line_errors.append(f"line_{line_index}:unit_price_currency_mismatch")

        line_total = unit_price * line.quantity if unit_price is not None else None
        if (
            line_total is not None
            and line.line_total_value is not None
            and line.line_total_value != line_total
        ):
            line_errors.append(f"line_{line_index}:line_total_mismatch")
        if (
            currency
            and line.line_total_currency
            and line.line_total_currency.strip() != currency
        ):
            line_errors.append(f"line_{line_index}:line_total_currency_mismatch")

        errors.extend(line_errors)
        for line_error in line_errors:
            error_details.append(
                _validation_error_detail(
                    _mechanical_error_code(line_error),
                    path=f"quote.lines[{line_index}]",
                    details={"raw_error": line_error},
                )
            )
        if line_errors:
            continue
        if line_total is not None and currency:
            total_by_currency[currency] = total_by_currency.get(currency, Decimal("0")) + line_total
        validated_lines.append(
            {
                "line_id": line.line_id or f"L{line_index + 1}",
                "role": line.role,
                "title": line.title,
                "component_candidate_id": component_candidate_id,
                "stock_row_id": stock_row_id,
                "quantity": line.quantity,
                "unit_price_value": _decimal_json(unit_price),
                "unit_price_currency": currency,
                "line_total_value": _decimal_json(line_total),
                "line_total_currency": currency,
                "object_id": line.object_id,
                "covered_item_ids": covered_item_ids,
                "covered_requirement_ids": covered_requirement_ids,
                "satisfies_requirement_ids": covered_requirement_ids,
                "technical_status": line.technical_status,
                "coverage_contributions": _jsonable_sequence(
                    line.coverage_contributions,
                ),
                "fact_ids": list(line.fact_ids),
                "compatibility_statement": line.compatibility_statement,
                "producer": product.get("producer"),
                "part_number": product.get("part_number"),
                "item_name": product.get("item_name") or product.get("product_name"),
                "category_id": product.get("category_id"),
                "shipment_city": stock.get("shipment_city"),
                "location": stock.get("location"),
                "available_quantity": available_quantity,
                "quantity_value": _int_or_none(stock.get("quantity_value")),
                "quantity_is_greater_than": _bool_or_false(
                    stock.get("quantity_is_greater_than")
                ),
                "reason": line.reason,
            }
        )

    for stock_row_id, used_quantity in used_quantity_by_stock_row_id.items():
        row = stock_rows_by_id.get(stock_row_id)
        stock = _safe_mapping(row.get("stock") if row else None)
        available_quantity = _stock_effective_available_quantity(stock)
        if available_quantity is not None and used_quantity > available_quantity:
            errors.append(f"stock_row_overallocated:{stock_row_id}")
            error_details.append(
                _validation_error_detail(
                    "stock.row_overallocated",
                    path=f"stock_rows.{stock_row_id}",
                    details={
                        "stock_row_id": stock_row_id,
                        "used_quantity": used_quantity,
                        "available_quantity": available_quantity,
                        "quantity_value": _int_or_none(stock.get("quantity_value")),
                        "quantity_is_greater_than": _bool_or_false(
                            stock.get("quantity_is_greater_than")
                        ),
                    },
                )
            )

    if len(total_by_currency) > 1:
        errors.append("multiple_currencies_in_quote")
        error_details.append(
            _validation_error_detail(
                "arithmetic.multiple_currencies",
                path="quote.lines",
                details={"currencies": sorted(total_by_currency)},
            )
        )
    total_currency = next(iter(total_by_currency), None)
    total_price = total_by_currency.get(total_currency, Decimal("0")) if total_currency else None
    declared_total_price = _quote_total_price_value(payload.quote)
    declared_total_currency = _quote_total_price_currency(payload.quote)
    if (
        total_price is not None
        and declared_total_price is not None
        and declared_total_price != total_price
    ):
        errors.append("total_price_mismatch")
        error_details.append(
            _validation_error_detail(
                "arithmetic.quote_total_mismatch",
                path="quote.total_price_value",
                details={
                    "declared_total_price": str(declared_total_price),
                    "computed_total_price": str(total_price),
                },
            )
        )
    if (
        total_currency
        and declared_total_currency
        and declared_total_currency.strip() != total_currency
    ):
        errors.append("total_price_currency_mismatch")
        error_details.append(
            _validation_error_detail(
                "price.currency_mismatch",
                path="quote.total_price_currency",
                details={
                    "declared_total_currency": declared_total_currency,
                    "computed_total_currency": total_currency,
                },
            )
        )

    if errors:
        return FullCategoryQuoteValidation(
            status="mechanically_invalid",
            errors=errors,
            error_details=error_details,
            warnings=warnings,
        )
    return FullCategoryQuoteValidation(
        status="mechanically_valid",
        warnings=warnings,
        validated_quote={
            "title": payload.quote.title,
            "client_status_label": payload.quote.client_status_label,
            "solution_scope": payload.quote.solution_scope,
            "substitution_policy": payload.quote.substitution_policy,
            "selection_mode": payload.quote.selection_mode,
            "completeness_status": payload.quote.completeness_status,
            "operational_status": payload.quote.operational_status,
            "anchor_component_candidate_id": payload.quote.anchor_component_candidate_id,
            "object_results": _jsonable_sequence(payload.quote.object_results),
            "anchor_search_audit": _jsonable_sequence(payload.quote.anchor_search_audit),
            "lines": validated_lines,
            "total_price": {
                "value": _decimal_json(total_price),
                "currency": total_currency,
            },
            "total_price_value": _decimal_json(total_price),
            "total_price_currency": total_currency,
            "client_summary": payload.quote.client_summary,
            "coverage_summary": payload.quote.coverage_summary,
            "coverage": _jsonable_sequence(payload.quote.coverage),
            "requirement_coverage": _jsonable_sequence(
                payload.quote.requirement_coverage
            ),
            "compatibility_check": _compatibility_check_json(
                payload.quote.compatibility_check
            ),
            "why_selected": payload.quote.why_selected,
            "key_deviations": _jsonable_sequence(payload.quote.key_deviations),
            "procurement_gaps": _jsonable_sequence(payload.quote.procurement_gaps),
            "deviation_notes": _jsonable_sequence(payload.quote.deviation_notes),
            "price_audit": _jsonable_sequence(payload.quote.price_audit),
            "dominance_audit": _jsonable_sequence(payload.quote.dominance_audit),
            "assumptions": list(payload.quote.assumptions),
            "engineer_checks": list(payload.quote.engineer_checks),
            "engineering_review_required": True,
        },
    )


def _matrix_diagnostics(
    matrix_package: FullCategoryMatrixPackage,
    settings: LlmSettings,
) -> dict[str, Any]:
    payload = matrix_package.payload
    diagnostics = _safe_mapping(payload.get("diagnostics"))
    return {
        "schema_version": payload.get("schema_version"),
        "matrix_payload_schema_version": payload.get("matrix_payload_schema_version"),
        "matrix_status": matrix_package.status,
        "matrix_char_count": matrix_package.char_count,
        "matrix_row_count": diagnostics.get("row_count"),
        "matrix_component_count": diagnostics.get("component_count"),
        "matrix_stock_row_count": diagnostics.get("stock_row_count"),
        "matrix_index_count": diagnostics.get("matrix_index_count"),
        "fact_reference_count": diagnostics.get("fact_reference_count"),
        "matrix_category_count": diagnostics.get("category_count"),
        "max_package_chars": settings.llm_configurator_max_package_chars,
        "model": settings.llm_model,
        "distributor_code": payload.get("distributor_code"),
        "category_ids": payload.get("category_ids") or [payload.get("category_id")],
    }


def _not_used_outcome(
    *,
    final_status_source: str,
    fallback_reason: str,
    diagnostics: dict[str, Any],
    summary: str = "v3 full category matrix composer was not attempted.",
) -> FullCategoryComposerOutcome:
    return FullCategoryComposerOutcome(
        pipeline_version=V3_FULL_CATEGORY_MATRIX_MODE,
        used=False,
        status="not_attempted",
        final_status_source=final_status_source,
        primary_recommendation_status="no_recommendation",
        no_recommendation_reason={
            "summary": summary,
            "fallback_reason": fallback_reason,
        },
        diagnostics=diagnostics,
    )


def _mechanical_failure_details(
    payload: FullCategoryComposerPayload,
    *,
    matrix_package: FullCategoryMatrixPackage,
    validation: FullCategoryQuoteValidation,
) -> list[dict[str, Any]]:
    if payload.quote is None or not payload.quote.lines:
        return []

    stock_rows_by_id, stock_rows_by_component_id = _package_row_indexes(matrix_package)
    overallocated_stock_row_ids = {
        error.removeprefix("stock_row_overallocated:")
        for error in validation.errors
        if error.startswith("stock_row_overallocated:")
    }
    used_quantity_by_stock_row_id: dict[str, int] = {}
    for line in payload.quote.lines:
        stock_row_id = str(line.stock_row_id or "").strip()
        if stock_row_id:
            used_quantity_by_stock_row_id[stock_row_id] = (
                used_quantity_by_stock_row_id.get(stock_row_id, 0) + line.quantity
            )

    details: list[dict[str, Any]] = []
    included_stock_row_ids: set[str] = set()
    for line_index, line in enumerate(payload.quote.lines):
        stock_row_id = str(line.stock_row_id or "").strip()
        component_candidate_id = line.component_candidate_id.strip()
        line_errors = [
            error
            for error in validation.errors
            if error.startswith(f"line_{line_index}:")
        ]
        if stock_row_id in overallocated_stock_row_ids:
            line_errors.append(f"stock_row_overallocated:{stock_row_id}")
        if not line_errors:
            continue

        row = stock_rows_by_id.get(stock_row_id)
        details.append(
            _mechanical_failure_line_detail(
                line_index=line_index,
                component_candidate_id=component_candidate_id,
                stock_row_id=stock_row_id,
                requested_quantity=line.quantity,
                total_used_quantity=used_quantity_by_stock_row_id.get(stock_row_id),
                errors=line_errors,
                selected_row=row,
                same_component_rows=stock_rows_by_component_id.get(
                    component_candidate_id,
                    [],
                ),
            )
        )
        included_stock_row_ids.add(stock_row_id)

    for stock_row_id in sorted(overallocated_stock_row_ids - included_stock_row_ids):
        row = stock_rows_by_id.get(stock_row_id)
        details.append(
            _mechanical_failure_line_detail(
                line_index=None,
                component_candidate_id=str(row.get("component_candidate_id") or "")
                if row
                else "",
                stock_row_id=stock_row_id,
                requested_quantity=None,
                total_used_quantity=used_quantity_by_stock_row_id.get(stock_row_id),
                errors=[f"stock_row_overallocated:{stock_row_id}"],
                selected_row=row,
                same_component_rows=[],
            )
        )

    return json.loads(json.dumps(details, ensure_ascii=False, default=str))


def _mechanical_failure_line_detail(
    *,
    line_index: int | None,
    component_candidate_id: str,
    stock_row_id: str,
    requested_quantity: int | None,
    total_used_quantity: int | None,
    errors: list[str],
    selected_row: Mapping[str, Any] | None,
    same_component_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_stock = _safe_mapping(selected_row.get("stock") if selected_row else None)
    selected_product = _safe_mapping(selected_row.get("product") if selected_row else None)
    available_quantity = _stock_effective_available_quantity(selected_stock)
    return {
        "line_index": line_index,
        "errors": errors,
        "component_candidate_id": component_candidate_id,
        "stock_row_id": stock_row_id,
        "requested_quantity": requested_quantity,
        "total_used_quantity": total_used_quantity,
        "available_quantity": available_quantity,
        "quantity_value": _int_or_none(selected_stock.get("quantity_value")),
        "quantity_is_greater_than": _bool_or_false(
            selected_stock.get("quantity_is_greater_than")
        ),
        "unit_price_value": selected_stock.get("price_order_value"),
        "unit_price_currency": selected_stock.get("price_order_currency"),
        "shipment_city": selected_stock.get("shipment_city"),
        "location": selected_stock.get("location"),
        "product": {
            "producer": selected_product.get("producer"),
            "part_number": selected_product.get("part_number"),
            "item_name": selected_product.get("item_name")
            or selected_product.get("product_name"),
            "category_id": selected_product.get("category_id"),
        },
        "same_component_stock_rows": [
            _stock_row_availability_hint(row) for row in same_component_rows[:8]
        ],
    }


def _stock_row_availability_hint(row: Mapping[str, Any]) -> dict[str, Any]:
    stock = _safe_mapping(row.get("stock"))
    return {
        "stock_row_id": row.get("stock_row_id"),
        "available_quantity": _stock_effective_available_quantity(stock),
        "quantity_value": _int_or_none(stock.get("quantity_value")),
        "quantity_is_greater_than": _bool_or_false(
            stock.get("quantity_is_greater_than")
        ),
        "unit_price_value": stock.get("price_order_value"),
        "unit_price_currency": stock.get("price_order_currency"),
        "shipment_city": stock.get("shipment_city"),
        "location": stock.get("location"),
    }


def _mechanically_invalid_outcome(
    *,
    raw_output: Mapping[str, Any],
    validation: FullCategoryQuoteValidation,
    diagnostics: dict[str, Any],
    error_type: str | None = None,
    http_status: int | None = None,
) -> FullCategoryComposerOutcome:
    summary = _validation_failure_summary(validation.error_details or validation.errors)
    return FullCategoryComposerOutcome(
        pipeline_version=V3_FULL_CATEGORY_MATRIX_MODE,
        used=True,
        status="mechanically_invalid",
        final_status_source=V3_MECHANICAL_VALIDATION_FAILED,
        primary_recommendation_status="mechanically_invalid",
        llm_output=_jsonable_mapping(raw_output),
        validation_failure_reason={
            "_legacy_summary": (
                "LLM собрала вариант, но он не прошел проверку ID, остатков "
                "или цен. КП не показано."
            ),
            "summary": summary,
            "fallback_reason": V3_MECHANICAL_VALIDATION_FAILED,
            "failed_requirements": validation.errors,
            "error_details": validation.error_details,
        },
        validation_errors=validation.errors,
        validation_error_details=validation.error_details,
        validation_warnings=validation.warnings,
        diagnostics=diagnostics,
        error_type=error_type,
        http_status=http_status,
    )


def _validation_failure_summary(errors: Sequence[Any]) -> str:
    codes = {
        str(item.get("code") or "").strip()
        for item in errors
        if isinstance(item, Mapping)
    }
    if not codes:
        codes = {str(item or "").strip() for item in errors}
    stages = {code.split(".", 1)[0] for code in codes if "." in code}
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


def _package_row_indexes(
    matrix_package: FullCategoryMatrixPackage,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_stock_row_id: dict[str, dict[str, Any]] = {}
    by_component_id: dict[str, list[dict[str, Any]]] = {}
    for raw_section in matrix_package.payload.get("category_sections", []):
        if not isinstance(raw_section, Mapping):
            continue
        for raw_product in raw_section.get("products", []):
            if not isinstance(raw_product, Mapping):
                continue
            component_candidate_id = str(
                raw_product.get("component_candidate_id") or ""
            ).strip()
            product = _safe_mapping(raw_product.get("product"))
            for raw_stock_row in raw_product.get("stock_rows", []):
                if not isinstance(raw_stock_row, Mapping):
                    continue
                row = {
                    "component_candidate_id": component_candidate_id,
                    "stock_row_id": str(raw_stock_row.get("stock_row_id") or "").strip(),
                    "product": product,
                    "stock": dict(raw_stock_row),
                }
                stock_row_id = str(row.get("stock_row_id") or "").strip()
                if stock_row_id:
                    by_stock_row_id[stock_row_id] = row
                if component_candidate_id:
                    by_component_id.setdefault(component_candidate_id, []).append(row)
    if by_stock_row_id or by_component_id:
        return by_stock_row_id, by_component_id

    for raw_component in matrix_package.payload.get("components", []):
        if not isinstance(raw_component, Mapping):
            continue
        component_candidate_id = str(
            raw_component.get("component_candidate_id") or ""
        ).strip()
        product = _safe_mapping(raw_component.get("product"))
        for raw_stock_row in raw_component.get("stock_rows", []):
            if not isinstance(raw_stock_row, Mapping):
                continue
            row = {
                "component_candidate_id": component_candidate_id,
                "stock_row_id": str(raw_stock_row.get("stock_row_id") or "").strip(),
                "product": product,
                "stock": dict(raw_stock_row),
            }
            stock_row_id = str(row.get("stock_row_id") or "").strip()
            if stock_row_id:
                by_stock_row_id[stock_row_id] = row
            if component_candidate_id:
                by_component_id.setdefault(component_candidate_id, []).append(row)
    if by_stock_row_id or by_component_id:
        return by_stock_row_id, by_component_id

    for raw_row in matrix_package.payload.get("rows", []):
        if not isinstance(raw_row, Mapping):
            continue
        row = dict(raw_row)
        stock_row_id = str(row.get("stock_row_id") or "").strip()
        component_candidate_id = str(row.get("component_candidate_id") or "").strip()
        if stock_row_id:
            by_stock_row_id[stock_row_id] = row
        if component_candidate_id:
            by_component_id.setdefault(component_candidate_id, []).append(row)
    return by_stock_row_id, by_component_id


def _stock_row_unique_suffix_index(
    stock_rows_by_id: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    index: dict[str, dict[str, Any] | None] = {}
    for stock_row_id, row in stock_rows_by_id.items():
        suffix = _stock_row_id_suffix(stock_row_id)
        if not suffix:
            continue
        if suffix in index:
            index[suffix] = None
        else:
            index[suffix] = row
    return index


def _package_fact_id_index(
    matrix_package: FullCategoryMatrixPackage,
) -> dict[str, set[str]]:
    by_component_id: dict[str, set[str]] = {}
    for raw_section in matrix_package.payload.get("category_sections", []):
        if not isinstance(raw_section, Mapping):
            continue
        for raw_product in raw_section.get("products", []):
            if not isinstance(raw_product, Mapping):
                continue
            component_candidate_id = str(
                raw_product.get("component_candidate_id") or ""
            ).strip()
            if not component_candidate_id:
                continue
            for raw_fact in raw_product.get("fact_refs", []) or []:
                if not isinstance(raw_fact, Mapping):
                    continue
                fact_id = str(raw_fact.get("fact_id") or "").strip()
                if fact_id:
                    by_component_id.setdefault(component_candidate_id, set()).add(fact_id)
    return by_component_id


def _resolved_request_id_sets(
    resolved_request: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    item_ids: set[str] = set()
    requirement_ids: set[str] = set()
    for raw_object in _raw_mapping_sequence(resolved_request.get("objects")):
        for raw_item in _raw_mapping_sequence(raw_object.get("requested_items")):
            item_id = str(raw_item.get("item_id") or "").strip()
            if item_id:
                item_ids.add(item_id)
            quantity_requirement_id = str(
                raw_item.get("quantity_requirement_id") or ""
            ).strip()
            if quantity_requirement_id:
                requirement_ids.add(quantity_requirement_id)
            for raw_constraint in _raw_mapping_sequence(raw_item.get("constraints")):
                requirement_id = str(
                    raw_constraint.get("requirement_id")
                    or raw_constraint.get("id")
                    or ""
                ).strip()
                if requirement_id:
                    requirement_ids.add(requirement_id)
    for raw_requirement in _raw_mapping_sequence(resolved_request.get("requirements")):
        requirement_id = str(
            raw_requirement.get("requirement_id") or raw_requirement.get("id") or ""
        ).strip()
        if requirement_id:
            requirement_ids.add(requirement_id)
    return item_ids, requirement_ids


def _raw_mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _validation_error_detail(
    code: str,
    *,
    path: str = "",
    stage: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_code = str(code or "contract.unknown").strip()
    stage_value = stage or _validation_stage_from_code(normalized_code)
    return {
        "stage": stage_value,
        "code": normalized_code,
        "path": path,
        "message_ru": MECHANICAL_ERROR_MESSAGES_RU.get(
            normalized_code,
            "Ответ модели не прошёл механическую проверку контракта.",
        ),
        "details": _jsonable_mapping(details or {}),
    }


def _validation_stage_from_code(code: str) -> str:
    prefix = str(code or "").split(".", 1)[0]
    if prefix in {"schema", "contract", "reference", "stock", "price", "arithmetic"}:
        return prefix
    return "contract"


def _extend_validation_errors(
    errors: list[str],
    error_details: list[dict[str, Any]],
    items: Sequence[Mapping[str, Any] | str],
) -> None:
    for item in items:
        if isinstance(item, Mapping):
            code = str(item.get("code") or "").strip()
            if not code:
                continue
            errors.append(code)
            error_details.append(dict(item))
        else:
            code = str(item or "").strip()
            if code:
                errors.append(code)
                error_details.append(_validation_error_detail(code))


def _mechanical_error_code(error: str) -> str:
    text = str(error or "")
    if "unknown_component_candidate_id" in text:
        return "reference.unknown_component_candidate_id"
    if "unknown_stock_row_id" in text or "stock_row_id_missing" in text:
        return "reference.unknown_stock_row_id"
    if "stock_row_component_mismatch" in text:
        return "reference.stock_row_product_mismatch"
    if "quantity_exceeds_stock" in text or "stock_quantity_missing" in text:
        return "stock.insufficient_quantity"
    if "unit_price_mismatch" in text:
        return "price.unit_price_mismatch"
    if "price_order_value_missing" in text:
        return "price.price_missing"
    if "currency" in text:
        return "price.currency_mismatch"
    if "line_total_mismatch" in text:
        return "arithmetic.line_total_mismatch"
    if "total_price_mismatch" in text:
        return "arithmetic.quote_total_mismatch"
    if "unknown_or_foreign_fact_id" in text or "fact_refs_not_available" in text:
        return "reference.unknown_or_foreign_fact_id"
    return "contract.unknown"


def _quote_contract_errors_v7_1(
    quote: FullCategoryQuotePayload,
    *,
    selection_contract: Mapping[str, Any],
    resolved_request: Mapping[str, Any],
) -> list[dict[str, Any]]:
    object_contracts = _raw_mapping_sequence(selection_contract.get("object_contracts"))
    if not object_contracts:
        return []
    errors: list[dict[str, Any]] = []
    line_ids = _quote_line_ids(quote.lines)
    line_ids_set = set(line_ids)
    lines_by_id = {
        line_id: line
        for line_id, line in zip(line_ids, quote.lines, strict=False)
        if line_id
    }
    line_component_ids = {
        line_id: str(line.component_candidate_id or "").strip()
        for line_id, line in lines_by_id.items()
    }
    line_statuses = {
        line_id: str(line.technical_status or "").strip()
        for line_id, line in lines_by_id.items()
    }

    object_ids = [
        str(item.get("object_id") or "").strip()
        for item in object_contracts
        if str(item.get("object_id") or "").strip()
    ]
    object_result_ids = {
        str(item.get("object_id") or "").strip()
        for item in _raw_mapping_sequence(quote.object_results)
        if str(item.get("object_id") or "").strip()
    }
    if not quote.object_results:
        errors.append(
            _validation_error_detail(
                "contract.object_results_missing",
                path="quote.object_results",
            )
        )
    elif set(object_ids) != object_result_ids:
        errors.append(
            _validation_error_detail(
                "contract.object_results_coverage_mismatch",
                path="quote.object_results",
                details={
                    "expected_object_ids": object_ids,
                    "actual_object_ids": sorted(object_result_ids),
                },
            )
        )

    anchor_audit = _raw_mapping_sequence(quote.anchor_search_audit)
    anchor_audit_ids = {
        str(item.get("object_id") or "").strip()
        for item in anchor_audit
        if str(item.get("object_id") or "").strip()
    }
    if not anchor_audit:
        errors.append(
            _validation_error_detail(
                "contract.anchor_search_audit_missing",
                path="quote.anchor_search_audit",
            )
        )
    elif set(object_ids) != anchor_audit_ids:
        errors.append(
            _validation_error_detail(
                "contract.anchor_manifest_coverage_mismatch",
                path="quote.anchor_search_audit",
                details={
                    "expected_object_ids": object_ids,
                    "actual_object_ids": sorted(anchor_audit_ids),
                },
            )
        )

    audit_by_object = {
        str(item.get("object_id") or "").strip(): item for item in anchor_audit
    }
    object_result_by_id = {
        str(item.get("object_id") or "").strip(): item
        for item in _raw_mapping_sequence(quote.object_results)
    }
    quote_selection_mode = str(quote.selection_mode or "").strip()
    for contract in object_contracts:
        object_id = str(contract.get("object_id") or "").strip()
        anchor_policy = str(contract.get("anchor_policy") or "").strip()
        candidate_ids = _non_empty_text_items(contract.get("anchor_candidate_ids") or [])
        candidate_count = _int_or_none(contract.get("anchor_candidate_count")) or len(
            candidate_ids
        )
        partial_without_anchor_allowed = bool(
            contract.get("partial_without_anchor_allowed")
        )
        object_result = object_result_by_id.get(object_id, {})
        object_selection_mode = str(object_result.get("selection_mode") or "").strip()
        if (
            anchor_policy in {"required", "self"}
            and candidate_count > 0
            and (
                quote_selection_mode == "partial_without_anchor"
                or object_selection_mode == "partial_without_anchor"
                or partial_without_anchor_allowed
            )
        ):
            errors.append(
                _validation_error_detail(
                    "contract.partial_without_anchor_forbidden",
                    path=f"selection_contract.object_contracts[{object_id}]",
                    details={"object_id": object_id, "anchor_candidate_count": candidate_count},
                )
            )
        if anchor_policy in {"required", "self"} and candidate_count > 0:
            selected_anchor_line_id = str(
                object_result.get("selected_anchor_line_id") or ""
            ).strip()
            selected_anchor_component_id = str(
                object_result.get("anchor_component_candidate_id")
                or object_result.get("selected_anchor_component_candidate_id")
                or quote.anchor_component_candidate_id
                or ""
            ).strip()
            audit_item = audit_by_object.get(object_id, {})
            audit_outcome = str(audit_item.get("outcome") or "").strip()
            if audit_outcome and audit_outcome != "selected":
                errors.append(
                    _validation_error_detail(
                        "contract.anchor_required_not_selected",
                        path=f"quote.anchor_search_audit[{object_id}]",
                        details={"object_id": object_id, "outcome": audit_outcome},
                    )
                )
            if not selected_anchor_line_id:
                selected_anchor_line_id = str(
                    audit_item.get("selected_anchor_line_id") or ""
                ).strip()
            if not selected_anchor_component_id:
                selected_anchor_component_id = str(
                    audit_item.get("selected_anchor_component_candidate_id")
                    or audit_item.get("anchor_component_candidate_id")
                    or ""
                ).strip()
            if (
                not selected_anchor_line_id
                or selected_anchor_line_id not in line_ids_set
                or selected_anchor_component_id not in candidate_ids
                or line_component_ids.get(selected_anchor_line_id)
                != selected_anchor_component_id
                or line_statuses.get(selected_anchor_line_id) != "anchor"
            ):
                errors.append(
                    _validation_error_detail(
                        "contract.anchor_required_not_selected",
                        path=f"quote.object_results[{object_id}]",
                        details={
                            "object_id": object_id,
                            "selected_anchor_line_id": selected_anchor_line_id,
                            "selected_anchor_component_candidate_id": selected_anchor_component_id,
                            "anchor_candidate_ids": candidate_ids,
                        },
                    )
                )
        if anchor_policy in {"required", "self"} and candidate_count == 0:
            audit_item = audit_by_object.get(object_id, {})
            if audit_item and str(audit_item.get("outcome") or "").strip() != (
                "none_available_in_manifest"
            ):
                errors.append(
                    _validation_error_detail(
                        "contract.anchor_manifest_coverage_mismatch",
                        path=f"quote.anchor_search_audit[{object_id}]",
                        details={"expected_outcome": "none_available_in_manifest"},
                    )
                )

    for line_index, line in enumerate(quote.lines):
        line_id = str(line.line_id or "").strip()
        if not line_id:
            errors.append(
                _validation_error_detail(
                    "contract.line_id_missing",
                    path=f"quote.lines[{line_index}].line_id",
                )
            )
        if not _non_empty_text_items(line.fact_ids):
            errors.append(
                _validation_error_detail(
                    "contract.line_fact_ids_missing",
                    path=f"quote.lines[{line_index}].fact_ids",
                )
            )
        if not str(line.compatibility_statement or "").strip():
            errors.append(
                _validation_error_detail(
                    "contract.line_compatibility_statement_missing",
                    path=f"quote.lines[{line_index}].compatibility_statement",
                )
            )

    errors.extend(_anchor_duplicate_item_errors(quote))
    return errors


def _quote_line_ids(lines: Sequence[FullCategoryQuoteLinePayload]) -> list[str]:
    result: list[str] = []
    for index, line in enumerate(lines, start=1):
        result.append(str(line.line_id or f"L{index}").strip())
    return result


def _anchor_duplicate_item_errors(
    quote: FullCategoryQuotePayload,
) -> list[dict[str, Any]]:
    anchor_covered_items: set[str] = set()
    for line in quote.lines:
        if str(line.technical_status or "").strip() == "anchor":
            anchor_covered_items.update(_non_empty_text_items(line.covered_item_ids))
    if not anchor_covered_items:
        return []
    errors: list[dict[str, Any]] = []
    for line_index, line in enumerate(quote.lines):
        if str(line.technical_status or "").strip() == "anchor":
            continue
        overlap = sorted(
            anchor_covered_items.intersection(_non_empty_text_items(line.covered_item_ids))
        )
        if not overlap:
            continue
        contribution_text = json.dumps(
            _jsonable_sequence(line.coverage_contributions),
            ensure_ascii=False,
            default=str,
        ).casefold()
        if "residual" in contribution_text or "remaining" in contribution_text:
            continue
        errors.append(
            _validation_error_detail(
                "contract.duplicate_line_for_anchor_covered_item",
                path=f"quote.lines[{line_index}].covered_item_ids",
                details={"covered_item_ids": overlap},
            )
        )
    return errors


def _line_requirement_ids(line: FullCategoryQuoteLinePayload) -> list[str]:
    return _non_empty_text_items(
        [*line.satisfies_requirement_ids, *line.covered_requirement_ids],
    )


def _line_fact_reference_errors(
    line: FullCategoryQuoteLinePayload,
    *,
    line_index: int,
    component_candidate_id: str,
    fact_ids_by_component_id: Mapping[str, set[str]],
) -> list[str]:
    if not line.fact_ids:
        return []
    known_fact_ids = fact_ids_by_component_id.get(component_candidate_id, set())
    if not known_fact_ids:
        return [
            f"line_{line_index}:fact_refs_not_available_for_component:{component_candidate_id}"
        ]
    errors: list[str] = []
    for fact_id in _non_empty_text_items(line.fact_ids):
        if fact_id not in known_fact_ids:
            errors.append(f"line_{line_index}:unknown_or_foreign_fact_id:{fact_id}")
    return errors


def _dominance_audit_reference_errors(
    dominance_audit: list[Any],
    *,
    lines: list[FullCategoryQuoteLinePayload],
    stock_rows_by_id: Mapping[str, dict[str, Any]],
    stock_rows_by_component_id: Mapping[str, list[dict[str, Any]]],
    fact_ids_by_component_id: Mapping[str, set[str]],
    strict_contract: bool = False,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    line_ids = set(_quote_line_ids(lines))
    if strict_contract and lines and not dominance_audit:
        return [
            _validation_error_detail(
                "contract.dominance_audit_missing",
                path="quote.dominance_audit",
            )
        ]
    if strict_contract:
        audit_line_ids = {
            str(item.get("line_id") or "").strip()
            for item in dominance_audit
            if isinstance(item, Mapping)
        }
        if line_ids != audit_line_ids:
            errors.append(
                _validation_error_detail(
                    "contract.dominance_audit_line_coverage_mismatch",
                    path="quote.dominance_audit",
                    details={
                        "expected_line_ids": sorted(line_ids),
                        "actual_line_ids": sorted(audit_line_ids),
                    },
                )
            )
    for audit_index, item in enumerate(dominance_audit):
        if not isinstance(item, Mapping):
            continue
        for key, value in item.items():
            key_text = str(key or "")
            values = value if isinstance(value, list) else [value]
            if key_text.endswith("component_candidate_id") or key_text.endswith(
                "component_candidate_ids"
            ):
                for component_id in _non_empty_text_items([str(raw or "") for raw in values]):
                    if component_id not in stock_rows_by_component_id:
                        errors.append(
                            _validation_error_detail(
                                "reference.unknown_component_candidate_id",
                                path=f"quote.dominance_audit[{audit_index}]",
                                details={"component_candidate_id": component_id},
                            )
                        )
            if key_text.endswith("stock_row_id") or key_text.endswith("stock_row_ids"):
                for stock_row_id in _non_empty_text_items([str(raw or "") for raw in values]):
                    if stock_row_id not in stock_rows_by_id:
                        errors.append(
                            _validation_error_detail(
                                "reference.unknown_stock_row_id",
                                path=f"quote.dominance_audit[{audit_index}]",
                                details={"stock_row_id": stock_row_id},
                            )
                        )
            if key_text.endswith("fact_id") or key_text.endswith("fact_ids"):
                known_fact_ids = (
                    set().union(*fact_ids_by_component_id.values())
                    if fact_ids_by_component_id
                    else set()
                )
                for fact_id in _non_empty_text_items([str(raw or "") for raw in values]):
                    if fact_id not in known_fact_ids:
                        errors.append(
                            _validation_error_detail(
                                "reference.unknown_or_foreign_fact_id",
                                path=f"quote.dominance_audit[{audit_index}]",
                                details={"fact_id": fact_id},
                            )
                        )
    return errors


def _resolve_stock_row_by_unique_suffix(
    stock_row_id: str,
    stock_rows_by_unique_suffix: Mapping[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    suffix = _stock_row_id_suffix(stock_row_id)
    if not suffix:
        return None
    return stock_rows_by_unique_suffix.get(suffix)


def _stock_row_id_suffix(stock_row_id: str) -> str:
    value = str(stock_row_id or "").strip()
    if ":" not in value:
        return ""
    return value.rsplit(":", 1)[-1].strip()


def _quote_compatibility_errors(
    compatibility_check: FullCategoryCompatibilityCheckPayload | None,
    *,
    lines: list[FullCategoryQuoteLinePayload],
    strict_contract: bool = False,
) -> list[dict[str, Any]]:
    if compatibility_check is None:
        return [
            _validation_error_detail(
                "contract.compatibility_line_checks_missing",
                path="quote.compatibility_check",
            )
        ]

    errors: list[dict[str, Any]] = []
    compatible_statuses = {
        "compatible",
        "compatible_selected_lines",
        "anchor_only",
        "confirmed_selected_set",
        "review_required_selected_set",
        "independent_partial_set",
    }
    status = str(compatibility_check.status or "").strip()
    if status not in compatible_statuses:
        status = compatibility_check.status or "missing"
        errors.append(
            _validation_error_detail(
                "contract.compatibility_line_checks_missing",
                path="quote.compatibility_check.status",
                details={"status": status},
            )
        )
    if not compatibility_check.checked_facts:
        errors.append(
            _validation_error_detail(
                "contract.compatibility_line_checks_missing",
                path="quote.compatibility_check.checked_facts",
            )
        )
    elif strict_contract:
        line_ids = set(_quote_line_ids(lines))
        checked_line_ids = {
            str(item.get("line_id") or "").strip()
            for item in compatibility_check.checked_facts
            if isinstance(item, Mapping)
        }
        if line_ids != checked_line_ids:
            errors.append(
                _validation_error_detail(
                    "contract.compatibility_line_check_coverage_mismatch",
                    path="quote.compatibility_check.checked_facts",
                    details={
                        "expected_line_ids": sorted(line_ids),
                        "actual_line_ids": sorted(checked_line_ids),
                    },
                )
            )
        lines_by_id = {
            line_id: line
            for line_id, line in zip(_quote_line_ids(lines), lines, strict=False)
            if line_id
        }
        for check_index, raw_check in enumerate(compatibility_check.checked_facts):
            if not isinstance(raw_check, Mapping):
                errors.append(
                    _validation_error_detail(
                        "contract.compatibility_line_check_reference_mismatch",
                        path=f"quote.compatibility_check.checked_facts[{check_index}]",
                        details={"reason": "checked_facts entry is not an object"},
                    )
                )
                continue
            line_id = str(raw_check.get("line_id") or "").strip()
            line = lines_by_id.get(line_id)
            if line is None:
                continue
            component_id = str(raw_check.get("component_candidate_id") or "").strip()
            stock_row_id = str(raw_check.get("stock_row_id") or "").strip()
            fact_ids = _non_empty_text_items(raw_check.get("fact_ids") or [])
            if (
                component_id != str(line.component_candidate_id or "").strip()
                or stock_row_id != str(line.stock_row_id or "").strip()
                or not fact_ids
                or not set(fact_ids).issubset(set(_non_empty_text_items(line.fact_ids)))
            ):
                errors.append(
                    _validation_error_detail(
                        "contract.compatibility_line_check_reference_mismatch",
                        path=f"quote.compatibility_check.checked_facts[{check_index}]",
                        details={
                            "line_id": line_id,
                            "component_candidate_id": component_id,
                            "stock_row_id": stock_row_id,
                        },
                    )
                )
    if compatibility_check.blocking_mismatches:
        errors.append(
            _validation_error_detail(
                "contract.compatibility_line_checks_missing",
                path="quote.compatibility_check.blocking_mismatches",
                details={"blocking_mismatches": compatibility_check.blocking_mismatches},
            )
        )
    if compatibility_check.selected_line_conflicts:
        errors.append(
            _validation_error_detail(
                "contract.compatibility_line_checks_missing",
                path="quote.compatibility_check.selected_line_conflicts",
                details={
                    "selected_line_conflicts": compatibility_check.selected_line_conflicts
                },
            )
        )
    return errors


def _quote_compatibility_line_coverage_warnings(
    compatibility_check: FullCategoryCompatibilityCheckPayload,
    lines: list[FullCategoryQuoteLinePayload],
) -> list[str]:
    checked_facts_text = "\n".join(
        _non_empty_text_items(
            [
                json.dumps(item, ensure_ascii=False, default=str)
                if isinstance(item, Mapping)
                else str(item or "")
                for item in compatibility_check.checked_facts
            ]
        )
    )
    if not checked_facts_text:
        return []
    warnings: list[str] = []
    for line_index, line in enumerate(lines):
        component_candidate_id = line.component_candidate_id.strip()
        stock_row_id = str(line.stock_row_id or "").strip()
        covered = bool(stock_row_id and stock_row_id in checked_facts_text) or bool(
            component_candidate_id and component_candidate_id in checked_facts_text
        )
        if not covered:
            warnings.append(f"compatibility_check_line_{line_index}_coverage_missing")
    return warnings


def _quote_compatibility_warnings(
    compatibility_check: FullCategoryCompatibilityCheckPayload | None,
    *,
    lines: list[FullCategoryQuoteLinePayload],
) -> list[str]:
    if compatibility_check is None:
        return []
    warnings: list[str] = []
    status = str(compatibility_check.status or "").strip()
    if status == "anchor_only":
        warnings.append("compatibility_check_anchor_only")
    elif status in {
        "compatible_selected_lines",
        "review_required_selected_set",
        "independent_partial_set",
    }:
        warnings.append("compatibility_check_selected_lines_only")
    if compatibility_check.unresolved_risks:
        warnings.append("compatibility_check_unresolved_risks_present")
    warnings.extend(_quote_compatibility_line_coverage_warnings(compatibility_check, lines))
    return warnings


def _quote_contract_warnings(quote: FullCategoryQuotePayload) -> list[str]:
    warnings: list[str] = []
    if not (quote.solution_scope or "").strip():
        warnings.append("solution_scope_missing")
    if not (quote.substitution_policy or "").strip():
        warnings.append("substitution_policy_missing")
    if not (quote.selection_mode or "").strip():
        warnings.append("selection_mode_missing")
    if not quote.requirement_coverage:
        warnings.append("requirement_coverage_missing")
    if not quote.engineer_checks:
        warnings.append("engineer_checks_missing")
    selection_mode = str(quote.selection_mode or "").strip()
    completeness_status = str(quote.completeness_status or "").strip()
    if (
        selection_mode
        in {
            "partial_build",
            "partial_with_anchor",
            "partial_without_anchor",
            "anchor_only",
        }
        or completeness_status in {"partial", "anchor_only"}
    ) and not quote.procurement_gaps:
        warnings.append("procurement_gaps_missing_for_partial_quote")
    for line_index, line in enumerate(quote.lines):
        if not _line_requirement_ids(line):
            warnings.append(f"line_{line_index}:satisfies_requirement_ids_missing")
    return warnings


def _quote_total_price_value(quote: FullCategoryQuotePayload) -> Decimal | None:
    if quote.total_price_value is not None:
        return quote.total_price_value
    return _decimal_or_none(
        quote.total_price.get("value")
        or quote.total_price.get("amount")
        or quote.total_price.get("total_price_value")
    )


def _quote_total_price_currency(quote: FullCategoryQuotePayload) -> str | None:
    if quote.total_price_currency:
        return quote.total_price_currency
    value = (
        quote.total_price.get("currency")
        or quote.total_price.get("total_price_currency")
    )
    text = str(value or "").strip()
    return text or None


def _non_empty_text_items(values: list[str]) -> list[str]:
    return [item.strip() for item in values if item and item.strip()]


def _jsonable_sequence(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append(text)
            continue
        result.append(json.loads(json.dumps(item, ensure_ascii=False, default=str)))
    return result


def _compatibility_check_json(
    compatibility_check: FullCategoryCompatibilityCheckPayload | None,
) -> dict[str, Any]:
    if compatibility_check is None:
        return {}
    return {
        "status": compatibility_check.status,
        "checked_facts": list(compatibility_check.checked_facts),
        "blocking_mismatches": list(compatibility_check.blocking_mismatches),
        "selected_line_conflicts": list(compatibility_check.selected_line_conflicts),
        "unresolved_risks": list(compatibility_check.unresolved_risks),
    }


def _validation_error_summary(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "schema_validation_failed"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg") or "invalid")
    return f"{location}:{message}" if location else message


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _stock_effective_available_quantity(stock: Mapping[str, Any]) -> int | None:
    quantity_value = _int_or_none(stock.get("quantity_value"))
    if quantity_value is None:
        return None
    if _bool_or_false(stock.get("quantity_is_greater_than")):
        return quantity_value + 1
    return quantity_value


def _bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "y", "да"}:
            return True
        if normalized in {"0", "false", "no", "n", "нет"}:
            return False
    if isinstance(value, int | float):
        return value != 0
    return False


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False, default=str))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _conservative_token_estimate(value: str) -> int:
    return max(1, (len(value) + 2) // 3)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _decimal_json(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
