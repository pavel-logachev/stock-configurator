from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

FIT_TIER_STRONG = "strong_fit"
FIT_TIER_POSSIBLE = "possible_fit"
FIT_TIER_FALLBACK_UNKNOWN = "fallback_unknown"
FIT_TIER_EXPLICIT_MISMATCH = "explicit_mismatch"
FIT_TIER_WRONG_ROLE = "wrong_role"

POLICY_NAME = "broad_pre_llm_for_ai_reasoning"

OBJECTIVE_REJECT_DISTRIBUTOR_MISMATCH = "distributor_mismatch"
OBJECTIVE_REJECT_CATEGORY_NOT_SELECTED = "category_not_selected"
OBJECTIVE_REJECT_NO_STOCK = "no_stock"
OBJECTIVE_REJECT_NO_PRICE = "no_price"
OBJECTIVE_REJECT_BROKEN_ROW = "broken_row"
OBJECTIVE_REJECT_WRONG_ROLE = "wrong_role_objective"

OBJECTIVE_REJECT_REASONS = {
    OBJECTIVE_REJECT_DISTRIBUTOR_MISMATCH,
    OBJECTIVE_REJECT_CATEGORY_NOT_SELECTED,
    OBJECTIVE_REJECT_NO_STOCK,
    OBJECTIVE_REJECT_NO_PRICE,
    OBJECTIVE_REJECT_BROKEN_ROW,
    OBJECTIVE_REJECT_WRONG_ROLE,
}


@dataclass(frozen=True)
class BroadPreLlmDecision:
    include: bool
    fit_tier: str
    objective_reject_reason: str | None = None
    match_warnings: tuple[str, ...] = field(default_factory=tuple)
    uncertainty_reasons: tuple[str, ...] = field(default_factory=tuple)
    evidence_summary: str = ""

    def to_report_json(self) -> dict[str, Any]:
        return {
            "policy": POLICY_NAME,
            "include": self.include,
            "fit_tier": self.fit_tier,
            "objective_reject_reason": self.objective_reject_reason,
            "match_warnings": list(self.match_warnings),
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "evidence_summary": self.evidence_summary,
        }


def broad_pre_llm_for_ai_reasoning(
    *,
    product_group: str,
    role: str,
    distributor_code: str | None,
    selected_distributor_code: str | None = None,
    category_id: str | None = None,
    selected_category_ids: Sequence[str] | None = None,
    has_stock: bool = True,
    has_price: bool = True,
    broken_row: bool = False,
    objective_role_reject_reason: str | None = None,
    technical_reject_reason: str | None = None,
    technical_warnings: Sequence[str] = (),
    uncertainty_reasons: Sequence[str] = (),
    default_fit_tier: str = FIT_TIER_POSSIBLE,
) -> BroadPreLlmDecision:
    """Decide whether a candidate may reach AI reasoning.

    This policy is intentionally generic: application code may reject objective
    invalidity, but technical uncertainty is represented as fit/warning metadata.
    """

    objective_reason = _objective_reject_reason(
        distributor_code=distributor_code,
        selected_distributor_code=selected_distributor_code,
        category_id=category_id,
        selected_category_ids=selected_category_ids,
        has_stock=has_stock,
        has_price=has_price,
        broken_row=broken_row,
        objective_role_reject_reason=objective_role_reject_reason,
    )
    if objective_reason:
        return BroadPreLlmDecision(
            include=False,
            fit_tier=FIT_TIER_WRONG_ROLE
            if objective_reason == OBJECTIVE_REJECT_WRONG_ROLE
            else FIT_TIER_FALLBACK_UNKNOWN,
            objective_reject_reason=objective_reason,
        )

    warnings = _unique_strings(
        [
            *technical_warnings,
            *([technical_reject_reason] if technical_reject_reason else []),
        ]
    )
    uncertainties = _unique_strings(uncertainty_reasons)
    fit_tier = _fit_tier_for_technical_state(
        technical_reject_reason,
        uncertainty_reasons=uncertainties,
        default_fit_tier=default_fit_tier,
    )
    evidence_summary = _evidence_summary(
        product_group=product_group,
        role=role,
        fit_tier=fit_tier,
        warnings=warnings,
        uncertainties=uncertainties,
    )
    return BroadPreLlmDecision(
        include=True,
        fit_tier=fit_tier,
        match_warnings=tuple(warnings),
        uncertainty_reasons=tuple(uncertainties),
        evidence_summary=evidence_summary,
    )


def objective_role_reject_reason(local_reason: str | None) -> str | None:
    reason = str(local_reason or "").strip()
    if not reason:
        return None
    if _is_objective_wrong_role_reason(reason):
        return OBJECTIVE_REJECT_WRONG_ROLE
    return None


def technical_fit_tier_from_reason(reason: str | None) -> str:
    return _fit_tier_for_technical_state(
        reason,
        uncertainty_reasons=(),
        default_fit_tier=FIT_TIER_POSSIBLE,
    )


def _objective_reject_reason(
    *,
    distributor_code: str | None,
    selected_distributor_code: str | None,
    category_id: str | None,
    selected_category_ids: Sequence[str] | None,
    has_stock: bool,
    has_price: bool,
    broken_row: bool,
    objective_role_reject_reason: str | None,
) -> str | None:
    if broken_row:
        return OBJECTIVE_REJECT_BROKEN_ROW
    if (
        selected_distributor_code
        and distributor_code
        and distributor_code.strip().casefold() != selected_distributor_code.strip().casefold()
    ):
        return OBJECTIVE_REJECT_DISTRIBUTOR_MISMATCH
    if selected_category_ids is not None:
        selected = {str(value).strip() for value in selected_category_ids if str(value).strip()}
        if selected and str(category_id or "").strip() not in selected:
            return OBJECTIVE_REJECT_CATEGORY_NOT_SELECTED
    if not has_stock:
        return OBJECTIVE_REJECT_NO_STOCK
    if not has_price:
        return OBJECTIVE_REJECT_NO_PRICE
    if objective_role_reject_reason:
        return OBJECTIVE_REJECT_WRONG_ROLE
    return None


def _fit_tier_for_technical_state(
    reason: str | None,
    *,
    uncertainty_reasons: Sequence[str],
    default_fit_tier: str,
) -> str:
    reason_text = str(reason or "").strip()
    if _is_explicit_mismatch_reason(reason_text):
        return FIT_TIER_EXPLICIT_MISMATCH
    if reason_text or uncertainty_reasons:
        if _is_uncertainty_reason(reason_text) or uncertainty_reasons:
            return FIT_TIER_FALLBACK_UNKNOWN
        return FIT_TIER_POSSIBLE
    return default_fit_tier


def _is_objective_wrong_role_reason(reason: str) -> bool:
    lowered = reason.casefold()
    if any(
        marker in lowered
        for marker in (
            "role_mismatch",
            "not_base",
            "not_standalone",
            "accessory_or",
            "base_device_mismatch",
            "controller_only",
            "wifi_controller",
            "hardware_mismatch",
            "not_access_point",
            "not_base_switch",
            "not_base_router",
        )
    ):
        return True
    return lowered in {
        "support_role_mismatch",
        "license_role_mismatch",
        "network_cable_base_device_mismatch",
        "storage_drive_not_standalone_drive",
    }


def _is_explicit_mismatch_reason(reason: str) -> bool:
    lowered = reason.casefold()
    if not lowered:
        return False
    if _is_uncertainty_reason(lowered):
        return False
    return any(
        marker in lowered
        for marker in (
            "below_requirement",
            "contradiction",
            "mismatch",
            "unmanaged_l3_stacking",
            "tiny_desktop",
        )
    )


def _is_uncertainty_reason(reason: str) -> bool:
    lowered = reason.casefold()
    return any(
        marker in lowered
        for marker in (
            "unknown",
            "not_confirmed",
            "not_proven",
            "missing_facts",
            "incomplete_facts",
            "requires_engineer",
        )
    )


def _evidence_summary(
    *,
    product_group: str,
    role: str,
    fit_tier: str,
    warnings: Sequence[str],
    uncertainties: Sequence[str],
) -> str:
    parts = [f"product_group={product_group}", f"role={role}", f"fit_tier={fit_tier}"]
    if warnings:
        parts.append("warnings=" + ",".join(warnings[:3]))
    if uncertainties:
        parts.append("uncertainty=" + ",".join(uncertainties[:3]))
    return "; ".join(parts)


def _unique_strings(values: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def is_objective_reject_row(row: Mapping[str, Any]) -> bool:
    reason = str(row.get("objective_reject_reason") or "").strip()
    return reason in OBJECTIVE_REJECT_REASONS
