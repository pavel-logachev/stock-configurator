from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

REQ_CLASS_PURCHASABLE_COMPONENT_ROLE = "purchasable_component_role"
REQ_CLASS_PRIMARY_OBJECT_FEATURE = "primary_object_feature"
REQ_CLASS_ACCESSORY_OR_CONSUMABLE = "accessory_or_consumable"
REQ_CLASS_SERVICE_OR_SUPPORT = "service_or_support"
REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT = "logistics_or_commercial_constraint"
REQ_CLASS_ENGINEERING_CHECK = "engineering_check"
REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING = "out_of_scope_or_unmapped_non_blocking"
REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE = "blocking_unmapped_purchasable_role"

REQ_HARD = "hard"
REQ_OPTIONAL = "optional"

COMPOSER_REJECTED_BY_VALIDATION = "composer_rejected_by_validation"
COMPOSER_VALIDATED = "composer_validated"
COMPOSER_NO_RECOMMENDATION = "composer_no_recommendation"
COMPOSER_NOT_ATTEMPTED = "composer_not_attempted"
COMPOSER_FAILURE_SAFE_NO_RECOMMENDATION = "composer_failure_safe_no_recommendation"
COMPOSER_SCHEMA_VALIDATION_FAILED = "composer_schema_validation_failed"
COMPOSER_PROVIDER_TIMEOUT = "composer_provider_timeout"


class FulfillmentMode(StrEnum):
    SEPARATE_COMPONENT_REQUIRED = "separate_component_required"
    INCLUDED_IN_PRIMARY_OBJECT = "included_in_primary_object"
    INCLUDED_IN_SELECTED_COMPONENT = "included_in_selected_component"
    INCLUDED_IN_BUNDLE_OR_KIT = "included_in_bundle_or_kit"
    SERVICE_OR_SUPPORT = "service_or_support"
    LOGISTICS_CONSTRAINT = "logistics_constraint"
    ENGINEERING_CHECK_ONLY = "engineering_check_only"
    UNVERIFIED_REQUIRES_CONFIRMATION = "unverified_requires_confirmation"
    OPTIONAL_PREFERENCE = "optional_preference"
    NOT_APPLICABLE = "not_applicable"


NON_BLOCKING_CLASSES = {
    REQ_CLASS_PRIMARY_OBJECT_FEATURE,
    REQ_CLASS_ACCESSORY_OR_CONSUMABLE,
    REQ_CLASS_SERVICE_OR_SUPPORT,
    REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT,
    REQ_CLASS_ENGINEERING_CHECK,
    REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING,
}
MANDATORY_CLASSES = {
    REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
    REQ_CLASS_BLOCKING_UNMAPPED_PURCHASABLE_ROLE,
}
NON_BLOCKING_FULFILLMENT_MODES = {
    FulfillmentMode.INCLUDED_IN_PRIMARY_OBJECT.value,
    FulfillmentMode.INCLUDED_IN_SELECTED_COMPONENT.value,
    FulfillmentMode.INCLUDED_IN_BUNDLE_OR_KIT.value,
    FulfillmentMode.SERVICE_OR_SUPPORT.value,
    FulfillmentMode.LOGISTICS_CONSTRAINT.value,
    FulfillmentMode.ENGINEERING_CHECK_ONLY.value,
    FulfillmentMode.UNVERIFIED_REQUIRES_CONFIRMATION.value,
    FulfillmentMode.OPTIONAL_PREFERENCE.value,
    FulfillmentMode.NOT_APPLICABLE.value,
}
MANDATORY_FULFILLMENT_MODES = {FulfillmentMode.SEPARATE_COMPONENT_REQUIRED.value}
NON_BLOCKING_LIFECYCLE_REASONS = {
    "sent_to_composer",
    "included_in_primary_object",
    "included_in_selected_component",
    "included_in_bundle_or_kit",
    "optional_only",
    "optional_preference",
    "engineering_check_only",
    "logistics_constraint",
    "service_or_support",
    "not_applicable",
    "satisfied_by_platform",
    "satisfied_by_ready_server",
}
MISSING_LIFECYCLE_REASONS = {
    "no_category_found",
    "no_stock_candidates",
    "role_not_purchasable",
    "planner_dropped",
    "validation_dropped",
    "missing_category",
    "missing_candidates",
}


@dataclass(frozen=True)
class RequirementNode:
    requirement_id: str
    source_text: str
    classification: str
    fulfillment_mode: str
    target_role: str
    hardness: str = REQ_HARD
    primary_object: str = ""
    should_create_bom_role: bool = False
    should_validate_after_composer: bool = False
    reason: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_optional(self) -> bool:
        return self.hardness == REQ_OPTIONAL

    @property
    def is_mandatory(self) -> bool:
        if self.is_optional:
            return False
        if self.classification not in MANDATORY_CLASSES:
            return False
        if self.fulfillment_mode not in MANDATORY_FULFILLMENT_MODES:
            return False
        return bool(self.target_role)

    @property
    def can_be_hard_missing_role(self) -> bool:
        return self.is_mandatory and self.target_role != ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "source_text": self.source_text,
            "classification": self.classification,
            "fulfillment_mode": self.fulfillment_mode,
            "target_role": self.target_role,
            "hardness": self.hardness,
            "primary_object": self.primary_object,
            "should_create_bom_role": self.should_create_bom_role,
            "should_validate_after_composer": self.should_validate_after_composer,
            "is_mandatory": self.is_mandatory,
            "can_be_hard_missing_role": self.can_be_hard_missing_role,
            "reason": self.reason,
            "source": self.source,
            "metadata": _jsonable(self.metadata),
        }


@dataclass(frozen=True)
class RequirementGraph:
    product_group: str
    primary_object: str
    nodes: tuple[RequirementNode, ...]

    @property
    def mandatory_nodes(self) -> tuple[RequirementNode, ...]:
        return tuple(node for node in self.nodes if node.is_mandatory)

    @property
    def mandatory_roles(self) -> tuple[str, ...]:
        return tuple(_unique(node.target_role for node in self.mandatory_nodes))

    def mandatory_node_ids_for_role(self, role: str) -> list[str]:
        normalized = _normalize_role(role)
        return [
            node.requirement_id
            for node in self.mandatory_nodes
            if _normalize_role(node.target_role) == normalized
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "product_group": self.product_group,
            "primary_object": self.primary_object,
            "nodes": [node.as_dict() for node in self.nodes],
            "mandatory_node_ids": [node.requirement_id for node in self.mandatory_nodes],
            "mandatory_roles": list(self.mandatory_roles),
        }


@dataclass(frozen=True)
class CandidateUniverseLedger:
    role_candidate_count: dict[str, int]
    roles_sent_to_composer: tuple[str, ...]
    roles_dropped_reason_by_role: dict[str, str]
    role_lifecycle_trace: tuple[dict[str, Any], ...]
    role_fulfillment_diagnostics: tuple[dict[str, Any], ...]
    hard_missing_roles: tuple[str, ...]
    corrected_trace_contradictions: tuple[dict[str, Any], ...] = ()
    invariant_violations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "role_candidate_count": dict(self.role_candidate_count),
            "roles_sent_to_composer": list(self.roles_sent_to_composer),
            "roles_dropped_reason_by_role": dict(self.roles_dropped_reason_by_role),
            "role_lifecycle_trace": list(self.role_lifecycle_trace),
            "role_fulfillment_diagnostics": list(self.role_fulfillment_diagnostics),
            "hard_missing_roles": list(self.hard_missing_roles),
            "corrected_trace_contradictions": list(self.corrected_trace_contradictions),
            "invariant_violations": list(self.invariant_violations),
        }


@dataclass(frozen=True)
class ComposerExecutionState:
    should_attempt: bool
    attempted: bool
    returned_bom: bool
    returned_structured_no_recommendation: bool
    validation_rejected: bool
    execution_failure: bool
    final_status_source: str
    fallback_reason: str
    blocked_by: tuple[str, ...] = ()
    llm_call_stages: tuple[str, ...] = ()
    invalid_or_empty_output: bool = False
    schema_validation_failed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_attempt": self.should_attempt,
            "attempted": self.attempted,
            "returned_bom": self.returned_bom,
            "returned_structured_no_recommendation": (
                self.returned_structured_no_recommendation
            ),
            "validation_rejected": self.validation_rejected,
            "execution_failure": self.execution_failure,
            "final_status_source": self.final_status_source,
            "fallback_reason": self.fallback_reason,
            "blocked_by": list(self.blocked_by),
            "llm_call_stages": list(self.llm_call_stages),
            "invalid_or_empty_output": self.invalid_or_empty_output,
            "schema_validation_failed": self.schema_validation_failed,
        }


@dataclass(frozen=True)
class CoverageEvidence:
    requirement_id: str
    role: str
    status: str
    fulfillment_mode: str
    evidence_source: str
    component_candidate_id: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "role": self.role,
            "status": self.status,
            "fulfillment_mode": self.fulfillment_mode,
            "evidence_source": self.evidence_source,
            "component_candidate_id": self.component_candidate_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ValidationLedger:
    validation_rejected: bool
    rejected_candidates: tuple[dict[str, Any], ...]
    hard_mismatches: tuple[dict[str, Any], ...]
    unverified_requirements: tuple[dict[str, Any], ...]
    validation_summary: dict[str, Any]
    concrete_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "validation_rejected": self.validation_rejected,
            "rejected_candidates": list(self.rejected_candidates),
            "hard_mismatches": list(self.hard_mismatches),
            "unverified_requirements": list(self.unverified_requirements),
            "validation_summary": dict(self.validation_summary),
            "concrete_reasons": list(self.concrete_reasons),
        }


@dataclass(frozen=True)
class SafeNoRecommendation:
    structured_no_recommendation: bool
    summary: str
    fallback_reason: str
    final_status_source: str
    product_group: str
    primary_object: str
    missing_roles: tuple[str, ...]
    role_failures: tuple[dict[str, Any], ...]
    blockers: tuple[dict[str, Any], ...]
    validation_rejections: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "structured_no_recommendation": self.structured_no_recommendation,
            "summary": self.summary,
            "fallback_reason": self.fallback_reason,
            "final_status_source": self.final_status_source,
            "product_group": self.product_group,
            "primary_object": self.primary_object,
            "missing_roles": list(self.missing_roles),
            "missing_required_capabilities": [],
            "hard_mismatches": list(self.validation_rejections),
            "stock_shortages": _stock_shortages_from_rejections(self.validation_rejections),
            "role_failures": list(self.role_failures),
            "role_analysis": list(self.role_failures),
            "blockers": list(self.blockers),
            "validation_rejections": list(self.validation_rejections),
            "diagnostics": _jsonable(self.diagnostics),
            "diagnostic_notes": [
                f"fallback_reason={self.fallback_reason}",
                f"final_status_source={self.final_status_source}",
            ],
            "recommended_next_actions": [
                "Inspect execution ledger, Composer parse diagnostics, and validation ledger.",
                "Retry the bounded v2 Composer-first path after provider/output issues are fixed.",
            ],
        }


def build_requirement_graph(
    *,
    product_group: str | None,
    primary_object: str | None,
    classified_requirements: Any,
    hard_roles: Any = (),
) -> RequirementGraph:
    nodes: list[RequirementNode] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(_mapping_rows(classified_requirements), start=1):
        node = _requirement_node_from_row(
            row,
            index=index,
            primary_object=str(primary_object or "").strip(),
        )
        if node.requirement_id in seen_ids:
            continue
        seen_ids.add(node.requirement_id)
        nodes.append(node)

    existing_mandatory_roles = {
        _normalize_role(node.target_role) for node in nodes if node.is_mandatory
    }
    for role in _string_list(hard_roles):
        normalized_role = _normalize_role(role)
        if not normalized_role or normalized_role in existing_mandatory_roles:
            continue
        requirement_id = f"req_role_{normalized_role}"
        suffix = 2
        while requirement_id in seen_ids:
            requirement_id = f"req_role_{normalized_role}_{suffix}"
            suffix += 1
        seen_ids.add(requirement_id)
        nodes.append(
            RequirementNode(
                requirement_id=requirement_id,
                source_text=normalized_role,
                classification=REQ_CLASS_PURCHASABLE_COMPONENT_ROLE,
                fulfillment_mode=FulfillmentMode.SEPARATE_COMPONENT_REQUIRED.value,
                target_role=normalized_role,
                hardness=REQ_HARD,
                primary_object=str(primary_object or "").strip(),
                should_create_bom_role=True,
                should_validate_after_composer=True,
                reason="Mandatory role from v2 hard purchasable BOM role contract.",
                source="hard_purchasable_bom_roles",
            )
        )

    return RequirementGraph(
        product_group=str(product_group or "").strip(),
        primary_object=str(primary_object or "").strip(),
        nodes=tuple(nodes),
    )


def build_candidate_universe_ledger(
    *,
    requirement_graph: RequirementGraph,
    role_candidate_count: Any,
    roles_sent_to_composer: Any,
    role_lifecycle_trace: Any,
    role_fulfillment_diagnostics: Any,
    roles_dropped_reason_by_role: Any,
) -> CandidateUniverseLedger:
    counts = {
        _normalize_role(role): int(count or 0)
        for role, count in _safe_mapping(role_candidate_count).items()
        if _normalize_role(role) and _int_or_none(count) is not None
    }
    sent_roles = _unique(
        [
            *_string_list(roles_sent_to_composer),
            *[role for role, count in counts.items() if count > 0],
            *[
                _normalize_role(row.get("role"))
                for row in _mapping_rows(role_fulfillment_diagnostics)
                if str(row.get("lifecycle_reason") or "").strip() == "sent_to_composer"
            ],
        ]
    )
    sent_role_set = set(sent_roles)
    corrected: list[dict[str, Any]] = []
    raw_drop_reasons = {
        _normalize_role(role): str(reason or "").strip()
        for role, reason in _safe_mapping(roles_dropped_reason_by_role).items()
        if _normalize_role(role) and str(reason or "").strip()
    }
    drop_reasons: dict[str, str] = {}
    for role, reason in raw_drop_reasons.items():
        if role in sent_role_set:
            corrected.append(
                {
                    "role": role,
                    "field": "roles_dropped_reason_by_role",
                    "dropped_reason": reason,
                    "correction": "removed because the role was sent to Composer",
                }
            )
            continue
        drop_reasons[role] = reason

    trace: list[dict[str, Any]] = []
    for row in _mapping_rows(role_lifecycle_trace):
        normalized = dict(row)
        role = _normalize_role(normalized.get("role"))
        if not role:
            continue
        normalized["role"] = role
        if role in sent_role_set and str(normalized.get("dropped_reason") or "").strip():
            corrected.append(
                {
                    "role": role,
                    "field": "role_lifecycle_trace.dropped_reason",
                    "dropped_reason": normalized.get("dropped_reason"),
                    "correction": "cleared because the role was sent to Composer",
                }
            )
            normalized["dropped_reason"] = None
        trace.append(normalized)

    diagnostics: list[dict[str, Any]] = []
    hard_missing_roles: list[str] = []
    for row in _mapping_rows(role_fulfillment_diagnostics):
        normalized = dict(row)
        role = _normalize_role(normalized.get("role"))
        if not role:
            continue
        normalized["role"] = role
        reason = str(normalized.get("lifecycle_reason") or "").strip()
        candidate_count = _int_or_none(normalized.get("candidate_count"))
        if candidate_count is None:
            candidate_count = counts.get(role, 0)
            normalized["candidate_count"] = candidate_count
        if role in sent_role_set and reason != "sent_to_composer":
            corrected.append(
                {
                    "role": role,
                    "field": "role_fulfillment_diagnostics.lifecycle_reason",
                    "lifecycle_reason": reason,
                    "correction": (
                        "set to sent_to_composer because the role has Composer candidates"
                    ),
                }
            )
            normalized["lifecycle_reason"] = "sent_to_composer"
        elif not reason and role in sent_role_set:
            normalized["lifecycle_reason"] = "sent_to_composer"
        diagnostics.append(normalized)

    diagnostic_by_role = {str(row.get("role")): row for row in diagnostics}
    for role in requirement_graph.mandatory_roles:
        row = diagnostic_by_role.get(role, {})
        reason = str(row.get("lifecycle_reason") or drop_reasons.get(role) or "").strip()
        candidate_count = _int_or_none(row.get("candidate_count"))
        if candidate_count is None:
            candidate_count = counts.get(role, 0)
        if role in sent_role_set or candidate_count > 0:
            continue
        if reason in MISSING_LIFECYCLE_REASONS or not reason:
            hard_missing_roles.append(role)

    return CandidateUniverseLedger(
        role_candidate_count=dict(sorted(counts.items())),
        roles_sent_to_composer=tuple(sent_roles),
        roles_dropped_reason_by_role=dict(sorted(drop_reasons.items())),
        role_lifecycle_trace=tuple(trace),
        role_fulfillment_diagnostics=tuple(diagnostics),
        hard_missing_roles=tuple(_unique(hard_missing_roles)),
        corrected_trace_contradictions=tuple(corrected),
        invariant_violations=(),
    )


def build_validation_ledger(
    *,
    validation_hard_mismatches: Any,
    validation_unverified_requirements: Any,
    validation_summary: Any,
    rejected_recommendations_debug_safe: Any,
    final_status_source: str | None = None,
) -> ValidationLedger:
    rejected_candidates = tuple(
        _normalize_rejected_candidate(row)
        for row in _mapping_rows(rejected_recommendations_debug_safe)
    )
    hard_mismatches = tuple(dict(row) for row in _mapping_rows(validation_hard_mismatches))
    unverified = tuple(
        dict(row) for row in _mapping_rows(validation_unverified_requirements)
    )
    summary = _safe_mapping(validation_summary)
    concrete_reasons = tuple(
        _unique(
            [
                *[
                    reason
                    for candidate in rejected_candidates
                    for reason in _string_list(candidate.get("concrete_validation_reasons"))
                ],
                *[
                    _validation_row_reason(row)
                    for row in [*hard_mismatches, *unverified]
                    if _validation_row_reason(row)
                ],
            ]
        )
    )
    validation_rejected = (
        str(final_status_source or "") == COMPOSER_REJECTED_BY_VALIDATION
        or bool(rejected_candidates)
        or bool(hard_mismatches)
        or _summary_has_rejections(summary)
    )
    return ValidationLedger(
        validation_rejected=validation_rejected,
        rejected_candidates=rejected_candidates,
        hard_mismatches=hard_mismatches,
        unverified_requirements=unverified,
        validation_summary=summary,
        concrete_reasons=concrete_reasons,
    )


def build_coverage_evidence(
    *,
    requirement_graph: RequirementGraph,
    candidate_ledger: CandidateUniverseLedger,
    selected_components_by_role: Any,
) -> list[CoverageEvidence]:
    selected = {
        _normalize_role(role): str(component_id or "").strip()
        for role, component_id in _safe_mapping(selected_components_by_role).items()
        if _normalize_role(role)
    }
    evidence: list[CoverageEvidence] = []
    for node in requirement_graph.nodes:
        role = _normalize_role(node.target_role)
        if not role:
            evidence.append(
                CoverageEvidence(
                    requirement_id=node.requirement_id,
                    role=node.target_role,
                    status="not_applicable",
                    fulfillment_mode=node.fulfillment_mode,
                    evidence_source="requirement_graph",
                    reason="Requirement has no purchasable target role.",
                )
            )
            continue
        if node.fulfillment_mode == FulfillmentMode.INCLUDED_IN_PRIMARY_OBJECT.value:
            evidence.append(
                CoverageEvidence(
                    requirement_id=node.requirement_id,
                    role=role,
                    status="satisfied_by_primary_object",
                    fulfillment_mode=node.fulfillment_mode,
                    evidence_source="primary_object",
                    reason="LLM classified this requirement as fulfilled by the primary object.",
                )
            )
            continue
        if node.fulfillment_mode in {
            FulfillmentMode.INCLUDED_IN_SELECTED_COMPONENT.value,
            FulfillmentMode.INCLUDED_IN_BUNDLE_OR_KIT.value,
        }:
            component_id = selected.get(role, "")
            evidence.append(
                CoverageEvidence(
                    requirement_id=node.requirement_id,
                    role=role,
                    status=(
                        "satisfied_by_selected_component"
                        if component_id
                        else "unverified_selected_component"
                    ),
                    fulfillment_mode=node.fulfillment_mode,
                    evidence_source="selected_component" if component_id else "composer",
                    component_candidate_id=component_id,
                    reason=(
                        "Selected BOM contains the target component."
                        if component_id
                        else "Composer did not expose a selected target component."
                    ),
                )
            )
            continue
        if not node.is_mandatory:
            evidence.append(
                CoverageEvidence(
                    requirement_id=node.requirement_id,
                    role=role,
                    status="non_blocking",
                    fulfillment_mode=node.fulfillment_mode,
                    evidence_source="requirement_graph",
                    reason=(
                        "Requirement node is optional, accessory, service, logistics, "
                        "engineering, or out of scope."
                    ),
                )
            )
            continue
        component_id = selected.get(role, "")
        if component_id:
            status = "satisfied_by_selected_component"
            source = "selected_component"
            reason = "Selected BOM contains the required role."
        elif role in candidate_ledger.roles_sent_to_composer:
            status = "sent_to_composer"
            source = "candidate_universe_ledger"
            reason = "Candidate universe sent this mandatory role to Composer."
        else:
            status = "missing_candidate"
            source = "candidate_universe_ledger"
            reason = "Mandatory role was not available in the Composer candidate universe."
        evidence.append(
            CoverageEvidence(
                requirement_id=node.requirement_id,
                role=role,
                status=status,
                fulfillment_mode=node.fulfillment_mode,
                evidence_source=source,
                component_candidate_id=component_id,
                reason=reason,
            )
        )
    return evidence


def build_composer_execution_state(
    *,
    attempt_decision: Any,
    validation_ledger: ValidationLedger,
    primary_recommendation_status: str | None,
    primary_recommendation: Any,
    recommended_builds: Any,
    no_recommendation_reason: Any,
    fallback_reason: str | None,
    error_type: str | None,
    parse_diagnostics: Any,
    proposal_count: Any,
    valid_proposals_count: Any,
    final_status_source: str | None,
    llm_call_stages: Any,
) -> ComposerExecutionState:
    decision = _safe_mapping(attempt_decision)
    stages = tuple(_string_list(llm_call_stages or decision.get("llm_call_stages")))
    attempted = any(
        stage in {"main_composer", "bom_composition"}
        for stage in stages
    )
    attempted = attempted or _int_or_none(proposal_count) not in (None, 0)
    attempted = attempted or bool(_safe_mapping(primary_recommendation))
    attempted = attempted or bool(_mapping_rows(recommended_builds))
    should_attempt = bool(decision.get("should_attempt"))
    blocked_by = tuple(_string_list(decision.get("blocked_by")))
    current_final = str(final_status_source or "").strip()
    fallback = str(fallback_reason or "").strip()
    parse = _safe_mapping(parse_diagnostics)
    error = str(error_type or parse.get("error_type") or "").strip()
    primary_status = str(primary_recommendation_status or "").strip()
    schema_validation_failed = _schema_validation_failed(
        current_final=current_final,
        fallback_reason=fallback,
        error_type=error,
        parse_diagnostics=parse,
    )
    returned_bom = (
        not schema_validation_failed
        and (
            bool(_safe_mapping(primary_recommendation))
            or bool(_mapping_rows(recommended_builds))
            or (_int_or_none(proposal_count) or 0) > 0
            or bool(validation_ledger.rejected_candidates)
        )
    )
    returned_structured = (
        primary_status == "no_recommendation"
        and current_final == COMPOSER_NO_RECOMMENDATION
        and bool(_safe_mapping(no_recommendation_reason))
    )
    validation_rejected = (
        attempted
        and returned_bom
        and not schema_validation_failed
        and validation_ledger.validation_rejected
    )
    invalid_or_empty = (
        attempted
        and not returned_bom
        and not returned_structured
        and (
            bool(error)
            or "invalid" in fallback
            or "no_proposals" in fallback
            or "empty" in fallback
            or schema_validation_failed
        )
    )
    execution_failure = (
        attempted
        and (schema_validation_failed or (not returned_bom and not returned_structured))
    )
    normalized_final = _execution_final_status_source(
        current=current_final,
        fallback_reason=fallback,
        attempted=attempted,
        should_attempt=should_attempt,
        validation_rejected=validation_rejected,
        primary_recommendation_status=primary_status,
        returned_structured_no_recommendation=returned_structured,
        execution_failure=execution_failure,
        schema_validation_failed=schema_validation_failed,
    )
    return ComposerExecutionState(
        should_attempt=should_attempt,
        attempted=attempted,
        returned_bom=returned_bom,
        returned_structured_no_recommendation=returned_structured,
        validation_rejected=validation_rejected,
        execution_failure=execution_failure,
        final_status_source=normalized_final,
        fallback_reason=fallback,
        blocked_by=blocked_by,
        llm_call_stages=stages,
        invalid_or_empty_output=invalid_or_empty,
        schema_validation_failed=schema_validation_failed,
    )


def build_safe_no_recommendation(
    *,
    requirement_graph: RequirementGraph,
    candidate_ledger: CandidateUniverseLedger,
    execution_state: ComposerExecutionState,
    validation_ledger: ValidationLedger,
    coverage_evidence: Sequence[CoverageEvidence],
    original_request_text: str | None = None,
) -> SafeNoRecommendation:
    blockers: list[dict[str, Any]] = []
    role_failures: list[dict[str, Any]] = []
    missing_roles: list[str] = []

    if execution_state.validation_rejected:
        for rejection in validation_ledger.rejected_candidates:
            blockers.append(
                {
                    "type": "validation_rejected_candidate",
                    "recommendation_id": rejection.get("recommendation_id"),
                    "rejection_category": rejection.get("rejection_category"),
                    "concrete_validation_reasons": _string_list(
                        rejection.get("concrete_validation_reasons")
                    ),
                }
            )
        for row in validation_ledger.hard_mismatches:
            role = _normalize_role(row.get("role"))
            blockers.append(
                {
                    "type": "validation_hard_mismatch",
                    "role": role,
                    "requirement_node_ids": (
                        requirement_graph.mandatory_node_ids_for_role(role)
                        if role
                        else []
                    ),
                    "reason": _validation_row_reason(row),
                }
            )
        role_failures = _validation_role_failures(
            validation_ledger,
            requirement_graph=requirement_graph,
        )
    else:
        for role in candidate_ledger.hard_missing_roles:
            node_ids = requirement_graph.mandatory_node_ids_for_role(role)
            if not node_ids:
                continue
            missing_roles.append(role)
            failure = {
                "role": role,
                "reason": (
                    "Mandatory requirement role is missing from the Composer "
                    "candidate universe."
                ),
                "requirement_node_ids": node_ids,
                "candidate_coverage": _candidate_coverage_for_role(
                    candidate_ledger,
                    role,
                ),
            }
            role_failures.append(failure)
            blockers.append({"type": "mandatory_role_missing", **failure})

        if execution_state.execution_failure and not execution_state.returned_bom:
            for role in requirement_graph.mandatory_roles:
                if not role or role in missing_roles:
                    continue
                role_failures.append(
                    {
                        "role": role,
                        "reason": (
                            "Composer execution did not produce a validated BOM or "
                            "structured business no-recommendation for this mandatory requirement."
                        ),
                        "requirement_node_ids": (
                            requirement_graph.mandatory_node_ids_for_role(role)
                        ),
                        "candidate_coverage": _candidate_coverage_for_role(
                            candidate_ledger,
                            role,
                        ),
                    }
                )
            blockers.append(
                {
                    "type": "composer_execution_failure",
                    "fallback_reason": execution_state.fallback_reason,
                    "attempted": execution_state.attempted,
                    "returned_bom": execution_state.returned_bom,
                }
            )
        elif not execution_state.attempted:
            blockers.append(
                {
                    "type": "composer_not_attempted",
                    "blocked_by": list(execution_state.blocked_by),
                    "should_attempt": execution_state.should_attempt,
                }
            )

    diagnostics = {
        "original_request_text": str(original_request_text or "").strip(),
        "requirement_graph": requirement_graph.as_dict(),
        "candidate_universe_ledger": candidate_ledger.as_dict(),
        "composer_execution_state": execution_state.as_dict(),
        "coverage_evidence": [row.as_dict() for row in coverage_evidence],
        "validation_ledger": validation_ledger.as_dict(),
    }
    return SafeNoRecommendation(
        structured_no_recommendation=True,
        summary=(
            "Composer did not produce a safe validated recommendation; the v2 "
            "execution ledger returned a structured fail-closed no-recommendation."
        ),
        fallback_reason=execution_state.fallback_reason,
        final_status_source=execution_state.final_status_source,
        product_group=requirement_graph.product_group,
        primary_object=requirement_graph.primary_object,
        missing_roles=tuple(_unique(missing_roles)),
        role_failures=tuple(_unique_mapping_rows(role_failures)),
        blockers=tuple(_unique_mapping_rows(blockers)),
        validation_rejections=validation_ledger.rejected_candidates,
        diagnostics=diagnostics,
    )


def build_execution_contract(
    *,
    product_group: str | None,
    primary_object: str | None,
    classified_requirements: Any,
    hard_roles: Any,
    role_candidate_count: Any,
    roles_sent_to_composer: Any,
    role_lifecycle_trace: Any,
    role_fulfillment_diagnostics: Any,
    roles_dropped_reason_by_role: Any,
    attempt_decision: Any,
    selected_components_by_role: Any = None,
    primary_recommendation_status: str | None = None,
    primary_recommendation: Any = None,
    recommended_builds: Any = None,
    no_recommendation_reason: Any = None,
    fallback_reason: str | None = None,
    error_type: str | None = None,
    parse_diagnostics: Any = None,
    proposal_count: Any = None,
    valid_proposals_count: Any = None,
    final_status_source: str | None = None,
    llm_call_stages: Any = None,
    validation_hard_mismatches: Any = None,
    validation_unverified_requirements: Any = None,
    validation_summary: Any = None,
    rejected_recommendations_debug_safe: Any = None,
    original_request_text: str | None = None,
) -> dict[str, Any]:
    graph = build_requirement_graph(
        product_group=product_group,
        primary_object=primary_object,
        classified_requirements=classified_requirements,
        hard_roles=hard_roles,
    )
    candidate_ledger = build_candidate_universe_ledger(
        requirement_graph=graph,
        role_candidate_count=role_candidate_count,
        roles_sent_to_composer=roles_sent_to_composer,
        role_lifecycle_trace=role_lifecycle_trace,
        role_fulfillment_diagnostics=role_fulfillment_diagnostics,
        roles_dropped_reason_by_role=roles_dropped_reason_by_role,
    )
    validation_ledger = build_validation_ledger(
        validation_hard_mismatches=validation_hard_mismatches,
        validation_unverified_requirements=validation_unverified_requirements,
        validation_summary=validation_summary,
        rejected_recommendations_debug_safe=rejected_recommendations_debug_safe,
        final_status_source=final_status_source,
    )
    coverage = build_coverage_evidence(
        requirement_graph=graph,
        candidate_ledger=candidate_ledger,
        selected_components_by_role=selected_components_by_role,
    )
    execution_state = build_composer_execution_state(
        attempt_decision=attempt_decision,
        validation_ledger=validation_ledger,
        primary_recommendation_status=primary_recommendation_status,
        primary_recommendation=primary_recommendation,
        recommended_builds=recommended_builds,
        no_recommendation_reason=no_recommendation_reason,
        fallback_reason=fallback_reason,
        error_type=error_type,
        parse_diagnostics=parse_diagnostics,
        proposal_count=proposal_count,
        valid_proposals_count=valid_proposals_count,
        final_status_source=final_status_source,
        llm_call_stages=llm_call_stages,
    )
    safe_no_recommendation = build_safe_no_recommendation(
        requirement_graph=graph,
        candidate_ledger=candidate_ledger,
        execution_state=execution_state,
        validation_ledger=validation_ledger,
        coverage_evidence=coverage,
        original_request_text=original_request_text,
    )
    return {
        "requirement_graph": graph.as_dict(),
        "candidate_universe_ledger": candidate_ledger.as_dict(),
        "composer_execution_state": execution_state.as_dict(),
        "coverage_evidence": [row.as_dict() for row in coverage],
        "validation_ledger": validation_ledger.as_dict(),
        "safe_no_recommendation": safe_no_recommendation.as_dict(),
        "execution_ledger": {
            "final_status_source": execution_state.final_status_source,
            "composer_attempted": execution_state.attempted,
            "composer_returned_bom": execution_state.returned_bom,
            "validation_rejected": execution_state.validation_rejected,
            "mandatory_requirement_node_ids": graph.as_dict()["mandatory_node_ids"],
            "mandatory_roles": list(graph.mandatory_roles),
            "hard_missing_roles": list(candidate_ledger.hard_missing_roles),
            "invariant_violations": [
                *candidate_ledger.invariant_violations,
                *_contract_invariant_violations(
                    graph=graph,
                    candidate_ledger=candidate_ledger,
                    execution_state=execution_state,
                    safe_no_recommendation=safe_no_recommendation,
                ),
            ],
        },
    }


def selected_components_by_role_from_recommendation(value: Any) -> dict[str, str]:
    recommendation = _safe_mapping(value)
    result: dict[str, str] = {}
    for row in _mapping_rows(recommendation.get("components")):
        role = _normalize_role(row.get("role"))
        component_id = str(row.get("component_candidate_id") or "").strip()
        if role and component_id:
            result[role] = component_id
    component_ids = _safe_mapping(recommendation.get("component_candidate_ids"))
    for role, component_id in component_ids.items():
        normalized_role = _normalize_role(role)
        text = str(component_id or "").strip()
        if normalized_role and text:
            result.setdefault(normalized_role, text)
    return result


def _requirement_node_from_row(
    row: Mapping[str, Any],
    *,
    index: int,
    primary_object: str,
) -> RequirementNode:
    classification = str(row.get("classification") or "").strip()
    fulfillment_mode = _normalize_fulfillment_mode(
        row.get("fulfillment_mode"),
        classification=classification,
    )
    role = _normalize_role(
        row.get("target_role")
        or row.get("fulfillment_target_role")
        or row.get("role")
        or row.get("role_id")
    )
    hardness = _normalize_hardness(row)
    should_create = (
        _truthy(row.get("should_create_bom_role"))
        if "should_create_bom_role" in row
        else (
            classification in MANDATORY_CLASSES
            and fulfillment_mode in MANDATORY_FULFILLMENT_MODES
            and hardness == REQ_HARD
        )
    )
    should_validate = (
        _truthy(row.get("should_validate_after_composer"))
        if "should_validate_after_composer" in row
        else _truthy(row.get("should_be_validated_after_composer"))
    )
    requirement_id = str(row.get("requirement_id") or row.get("id") or "").strip()
    if not requirement_id:
        stem = role or classification or "requirement"
        requirement_id = f"req_{index}_{stem}"
    return RequirementNode(
        requirement_id=requirement_id,
        source_text=str(
            row.get("source_text")
            or row.get("requirement_text")
            or row.get("text")
            or ""
        ).strip(),
        classification=classification or _classification_from_fulfillment(fulfillment_mode),
        fulfillment_mode=fulfillment_mode,
        target_role=role,
        hardness=hardness,
        primary_object=primary_object,
        should_create_bom_role=should_create,
        should_validate_after_composer=should_validate,
        reason=str(row.get("reason") or "").strip(),
        source=str(row.get("source") or "").strip(),
        metadata={
            key: value
            for key, value in row.items()
            if key
            not in {
                "requirement_id",
                "id",
                "source_text",
                "requirement_text",
                "text",
                "classification",
                "fulfillment_mode",
                "target_role",
                "fulfillment_target_role",
                "role",
                "role_id",
                "hardness",
                "required",
                "optional",
                "should_create_bom_role",
                "should_validate_after_composer",
                "should_be_validated_after_composer",
                "reason",
                "source",
            }
        },
    )


def _normalize_fulfillment_mode(value: Any, *, classification: str) -> str:
    text = str(value or "").strip()
    allowed = {mode.value for mode in FulfillmentMode}
    if text in allowed:
        return text
    if text == "optional":
        return FulfillmentMode.OPTIONAL_PREFERENCE.value
    if text == "primary_object":
        return FulfillmentMode.INCLUDED_IN_PRIMARY_OBJECT.value
    if text in {"selected_component", "fulfilled_by_selected_component"}:
        return FulfillmentMode.INCLUDED_IN_SELECTED_COMPONENT.value
    if classification == REQ_CLASS_PRIMARY_OBJECT_FEATURE:
        return FulfillmentMode.INCLUDED_IN_PRIMARY_OBJECT.value
    if classification == REQ_CLASS_SERVICE_OR_SUPPORT:
        return FulfillmentMode.SERVICE_OR_SUPPORT.value
    if classification == REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT:
        return FulfillmentMode.LOGISTICS_CONSTRAINT.value
    if classification == REQ_CLASS_ENGINEERING_CHECK:
        return FulfillmentMode.ENGINEERING_CHECK_ONLY.value
    if classification == REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING:
        return FulfillmentMode.NOT_APPLICABLE.value
    return FulfillmentMode.SEPARATE_COMPONENT_REQUIRED.value


def _classification_from_fulfillment(fulfillment_mode: str) -> str:
    if fulfillment_mode == FulfillmentMode.INCLUDED_IN_PRIMARY_OBJECT.value:
        return REQ_CLASS_PRIMARY_OBJECT_FEATURE
    if fulfillment_mode == FulfillmentMode.SERVICE_OR_SUPPORT.value:
        return REQ_CLASS_SERVICE_OR_SUPPORT
    if fulfillment_mode == FulfillmentMode.LOGISTICS_CONSTRAINT.value:
        return REQ_CLASS_LOGISTICS_OR_COMMERCIAL_CONSTRAINT
    if fulfillment_mode == FulfillmentMode.ENGINEERING_CHECK_ONLY.value:
        return REQ_CLASS_ENGINEERING_CHECK
    if fulfillment_mode == FulfillmentMode.NOT_APPLICABLE.value:
        return REQ_CLASS_OUT_OF_SCOPE_OR_UNMAPPED_NON_BLOCKING
    return REQ_CLASS_PURCHASABLE_COMPONENT_ROLE


def _normalize_hardness(row: Mapping[str, Any]) -> str:
    text = str(
        row.get("hardness")
        or row.get("requirement_hardness")
        or row.get("priority")
        or ""
    ).strip().casefold()
    if text in {"optional", "desirable", "preferred", "nice_to_have"}:
        return REQ_OPTIONAL
    if row.get("optional") is True or row.get("required") is False:
        return REQ_OPTIONAL
    return REQ_HARD


def _execution_final_status_source(
    *,
    current: str,
    fallback_reason: str,
    attempted: bool,
    should_attempt: bool,
    validation_rejected: bool,
    primary_recommendation_status: str,
    returned_structured_no_recommendation: bool,
    execution_failure: bool,
    schema_validation_failed: bool,
) -> str:
    if fallback_reason == "llm_call_budget_exceeded" or current == "llm_call_budget_exceeded":
        return "llm_call_budget_exceeded"
    if fallback_reason == COMPOSER_PROVIDER_TIMEOUT or current == COMPOSER_PROVIDER_TIMEOUT:
        return COMPOSER_PROVIDER_TIMEOUT
    if "context_limit" in fallback_reason or current == "provider_context_limit":
        return "provider_context_limit"
    if schema_validation_failed:
        return COMPOSER_SCHEMA_VALIDATION_FAILED
    if not attempted:
        if current and current != COMPOSER_REJECTED_BY_VALIDATION:
            return current
        return COMPOSER_NOT_ATTEMPTED if not should_attempt else current or COMPOSER_NOT_ATTEMPTED
    if validation_rejected:
        return COMPOSER_REJECTED_BY_VALIDATION
    if primary_recommendation_status == "valid":
        return COMPOSER_VALIDATED
    if returned_structured_no_recommendation:
        return COMPOSER_NO_RECOMMENDATION
    if execution_failure:
        return COMPOSER_FAILURE_SAFE_NO_RECOMMENDATION
    return current or COMPOSER_NO_RECOMMENDATION


def _schema_validation_failed(
    *,
    current_final: str,
    fallback_reason: str,
    error_type: str,
    parse_diagnostics: Mapping[str, Any],
) -> bool:
    if current_final == COMPOSER_SCHEMA_VALIDATION_FAILED:
        return True
    if error_type != "ValidationError":
        return False
    if fallback_reason in {
        "llm_configurator_validation_failed",
        "multi_pass_validation_failed",
    }:
        return True
    return str(parse_diagnostics.get("parse_status") or "").strip() == "validation_error"


def _normalize_rejected_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    reasons = _unique(
        [
            str(result.get("rejection_code") or "").strip(),
            str(result.get("rejection_category") or "").strip(),
            str(result.get("rejection_message_ru") or result.get("message") or "").strip(),
            *[
                (
                    "unknown_component_id:"
                    f"{item.get('component_candidate_id') or item.get('candidate_id')}"
                )
                for item in _mapping_rows(result.get("unknown_component_ids"))
            ],
            *[
                (
                    f"stock_shortage:{item.get('role')} "
                    f"{item.get('required_quantity')} > {item.get('available_quantity')}"
                )
                for item in _mapping_rows(result.get("stock_shortages"))
            ],
            *[
                _validation_row_reason(item)
                for item in _mapping_rows(result.get("validation_hard_mismatches"))
                if _validation_row_reason(item)
            ],
            *[
                _validation_row_reason(item)
                for item in _mapping_rows(result.get("hard_capability_validation"))
                if _validation_row_reason(item)
            ],
        ]
    )
    result["concrete_validation_reasons"] = [reason for reason in reasons if reason]
    return result


def _validation_role_failures(
    validation_ledger: ValidationLedger,
    *,
    requirement_graph: RequirementGraph,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in validation_ledger.rejected_candidates:
        for role in _roles_from_validation_rejection(candidate):
            node_ids = requirement_graph.mandatory_node_ids_for_role(role)
            rows.append(
                {
                    "role": role,
                    "reason": "; ".join(
                        _string_list(candidate.get("concrete_validation_reasons"))
                    )
                    or "Composer candidate was rejected by deterministic validation.",
                    "requirement_node_ids": node_ids,
                    "candidate_coverage": {
                        "recommendation_id": candidate.get("recommendation_id"),
                        "rejection_category": candidate.get("rejection_category"),
                    },
                }
            )
    for row in validation_ledger.hard_mismatches:
        role = _normalize_role(row.get("role"))
        if not role:
            continue
        rows.append(
            {
                "role": role,
                "reason": _validation_row_reason(row),
                "requirement_node_ids": requirement_graph.mandatory_node_ids_for_role(role),
                "candidate_coverage": {
                    "component_candidate_id": row.get("component_candidate_id"),
                    "status": row.get("status"),
                },
            }
        )
    return _unique_mapping_rows([row for row in rows if row.get("role")])


def _roles_from_validation_rejection(row: Mapping[str, Any]) -> list[str]:
    roles = [
        *_string_list(row.get("missing_roles")),
        *[
            str(item.get("role") or "").strip()
            for item in _mapping_rows(row.get("stock_shortages"))
        ],
        *[
            str(item.get("role") or "").strip()
            for item in _mapping_rows(row.get("validation_hard_mismatches"))
        ],
        *[
            str(item.get("role") or "").strip()
            for item in _mapping_rows(row.get("hard_capability_validation"))
            if str(item.get("status") or "").strip()
            in {"hard_mismatch", "missing_component", "unverified_hard_requirement"}
        ],
        *[
            str(item.get("role") or item.get("component_role") or "").strip()
            for item in _mapping_rows(row.get("unknown_component_ids"))
        ],
    ]
    return _unique(_normalize_role(role) for role in roles if _normalize_role(role))


def _candidate_coverage_for_role(
    candidate_ledger: CandidateUniverseLedger,
    role: str,
) -> dict[str, Any]:
    role = _normalize_role(role)
    row = next(
        (
            item
            for item in candidate_ledger.role_fulfillment_diagnostics
            if _normalize_role(item.get("role")) == role
        ),
        {},
    )
    return {
        "candidate_count": candidate_ledger.role_candidate_count.get(role, 0),
        "lifecycle_reason": row.get("lifecycle_reason"),
        "selected_category_ids": _string_list(row.get("selected_category_ids")),
        "sent_to_composer": role in candidate_ledger.roles_sent_to_composer,
    }


def _contract_invariant_violations(
    *,
    graph: RequirementGraph,
    candidate_ledger: CandidateUniverseLedger,
    execution_state: ComposerExecutionState,
    safe_no_recommendation: SafeNoRecommendation,
) -> list[str]:
    violations: list[str] = []
    mandatory_roles = set(graph.mandatory_roles)
    for role in safe_no_recommendation.missing_roles:
        if role not in mandatory_roles:
            violations.append(f"missing_role_not_mandatory_requirement_node:{role}")
    if (
        not execution_state.attempted
        and execution_state.final_status_source == COMPOSER_REJECTED_BY_VALIDATION
    ):
        violations.append("composer_not_attempted_but_validation_rejected")
    sent = set(candidate_ledger.roles_sent_to_composer)
    for role, reason in candidate_ledger.roles_dropped_reason_by_role.items():
        if role in sent and reason:
            violations.append(f"role_sent_to_composer_marked_dropped:{role}")
    return violations


def _stock_shortages_from_rejections(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(shortage)
        for row in rows
        for shortage in _mapping_rows(row.get("stock_shortages"))
    ]


def _validation_row_reason(row: Mapping[str, Any]) -> str:
    return str(
        row.get("reason")
        or row.get("message")
        or row.get("user_message")
        or row.get("status")
        or row.get("type")
        or row.get("capability_id")
        or ""
    ).strip()


def _summary_has_rejections(summary: Mapping[str, Any]) -> bool:
    for key in (
        "validation_rejected_count",
        "rejected",
        "rejected_fatal",
        "rejected_stock",
        "rejected_stock_shortage",
    ):
        value = _int_or_none(summary.get(key))
        if value is not None and value > 0:
            return True
    return False


def _normalize_role(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "platform": "server_platform",
        "storage": "drive",
        "disk": "drive",
        "disks": "drive",
        "drives": "drive",
        "storage_drive": "drive",
        "power_cord": "cable",
        "power_cable": "cable",
        "nic": "network_adapter",
        "network_card": "network_adapter",
    }
    return aliases.get(text, text)


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _safe_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _unique_mapping_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value
