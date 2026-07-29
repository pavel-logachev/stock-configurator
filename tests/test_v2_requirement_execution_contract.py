from __future__ import annotations

from typing import Any

from app.matching.requirement_execution_contract import build_execution_contract


def test_final_status_source_matches_execution_state() -> None:
    contract = _contract(
        final_status_source="composer_validated",
        primary_recommendation_status="valid",
        primary_recommendation={
            "components": [
                {"role": "switch", "component_candidate_id": "switch-1"},
            ],
        },
        llm_call_stages=["candidate_universe_planner", "main_composer"],
    )

    state = contract["composer_execution_state"]
    assert contract["execution_ledger"]["final_status_source"] == state[
        "final_status_source"
    ]
    assert state["final_status_source"] == "composer_validated"


def test_optional_and_commercial_nodes_cannot_be_hard_missing_roles() -> None:
    contract = _contract(
        classified_requirements=[
            {
                "requirement_id": "req_vendor",
                "source_text": "prefer QNAP",
                "classification": "accessory_or_consumable",
                "fulfillment_mode": "optional_preference",
                "target_role": "vendor",
                "hardness": "optional",
            },
            {
                "requirement_id": "req_budget",
                "source_text": "cheapest option for quote",
                "classification": "logistics_or_commercial_constraint",
                "fulfillment_mode": "logistics_constraint",
                "target_role": "commercial",
                "hardness": "hard",
            },
        ],
        hard_roles=["storage_system"],
        role_candidate_count={"storage_system": 1},
        roles_sent_to_composer=["storage_system"],
        role_fulfillment_diagnostics=[
            {"role": "vendor", "lifecycle_reason": "optional_only", "candidate_count": 0},
            {
                "role": "commercial",
                "lifecycle_reason": "logistics_constraint",
                "candidate_count": 0,
            },
            {
                "role": "storage_system",
                "lifecycle_reason": "sent_to_composer",
                "candidate_count": 1,
            },
        ],
        fallback_reason="multi_pass_invalid_json",
        final_status_source="composer_failure_safe_no_recommendation",
        llm_call_stages=["main_composer"],
    )

    graph = contract["requirement_graph"]
    reason = contract["safe_no_recommendation"]
    assert "vendor" not in graph["mandatory_roles"]
    assert "commercial" not in graph["mandatory_roles"]
    assert "vendor" not in reason["missing_roles"]
    assert "commercial" not in reason["missing_roles"]


def test_feature_requirement_can_be_satisfied_by_primary_object() -> None:
    contract = _contract(
        classified_requirements=[
            {
                "requirement_id": "req_nvme",
                "source_text": "4TB on NVMe",
                "classification": "primary_object_feature",
                "fulfillment_mode": "included_in_primary_object",
                "target_role": "storage_system",
                "hardness": "hard",
            }
        ],
        hard_roles=["storage_system"],
        role_candidate_count={"storage_system": 1},
        roles_sent_to_composer=["storage_system"],
    )

    evidence = {
        row["requirement_id"]: row
        for row in contract["coverage_evidence"]
    }
    assert evidence["req_nvme"]["status"] == "satisfied_by_primary_object"


def test_separate_component_requirement_can_be_satisfied_by_selected_component() -> None:
    contract = _contract(
        classified_requirements=[
            {
                "requirement_id": "req_drive",
                "source_text": "separate NVMe drive",
                "classification": "purchasable_component_role",
                "fulfillment_mode": "separate_component_required",
                "target_role": "drive",
                "hardness": "hard",
            }
        ],
        hard_roles=["storage_system", "drive"],
        role_candidate_count={"storage_system": 1, "drive": 2},
        roles_sent_to_composer=["storage_system", "drive"],
        selected_components_by_role={"storage_system": "nas-1", "drive": "drive-1"},
        primary_recommendation_status="valid",
        primary_recommendation={
            "components": [
                {"role": "storage_system", "component_candidate_id": "nas-1"},
                {"role": "drive", "component_candidate_id": "drive-1"},
            ],
        },
        final_status_source="composer_validated",
        llm_call_stages=["main_composer"],
    )

    evidence = {
        row["requirement_id"]: row
        for row in contract["coverage_evidence"]
    }
    assert evidence["req_drive"]["status"] == "satisfied_by_selected_component"
    assert evidence["req_drive"]["component_candidate_id"] == "drive-1"


def test_role_lifecycle_and_fulfillment_diagnostics_are_normalized() -> None:
    contract = _contract(
        role_candidate_count={"switch": 1},
        roles_sent_to_composer=["switch"],
        roles_dropped_reason_by_role={"switch": "planner_dropped"},
        role_lifecycle_trace=[
            {"role": "switch", "composer_package": True, "dropped_reason": "planner_dropped"}
        ],
        role_fulfillment_diagnostics=[
            {"role": "switch", "lifecycle_reason": "planner_dropped", "candidate_count": 1}
        ],
        primary_recommendation_status="valid",
        primary_recommendation={
            "components": [
                {"role": "switch", "component_candidate_id": "switch-1"},
            ],
        },
        final_status_source="composer_validated",
        llm_call_stages=["main_composer"],
    )

    ledger = contract["candidate_universe_ledger"]
    assert "switch" not in ledger["roles_dropped_reason_by_role"]
    assert ledger["role_lifecycle_trace"][0]["dropped_reason"] is None
    assert ledger["role_fulfillment_diagnostics"][0]["lifecycle_reason"] == (
        "sent_to_composer"
    )
    assert contract["execution_ledger"]["invariant_violations"] == []


def test_invalid_empty_composer_output_is_execution_failure_not_fake_missing_roles() -> None:
    contract = _contract(
        fallback_reason="multi_pass_invalid_json",
        final_status_source="composer_failure_safe_no_recommendation",
        llm_call_stages=["candidate_universe_planner", "main_composer"],
    )

    state = contract["composer_execution_state"]
    reason = contract["safe_no_recommendation"]
    assert state["execution_failure"] is True
    assert state["returned_bom"] is False
    assert reason["missing_roles"] == []
    assert [row["role"] for row in reason["role_failures"]] == ["switch"]


def test_validation_rejection_contains_concrete_validation_reasons_only() -> None:
    contract = _contract(
        fallback_reason="composer_no_safe_complete_bom",
        final_status_source="composer_rejected_by_validation",
        primary_recommendation_status="no_recommendation",
        proposal_count=1,
        validation_summary={"validation_rejected_count": 1},
        rejected_recommendations_debug_safe=[
            {
                "recommendation_id": "bad-bom",
                "rejection_category": "stock_quantity_mismatch",
                "rejection_code": "stock_quantity_mismatch",
                "stock_shortages": [
                    {
                        "role": "switch",
                        "component_candidate_id": "switch-1",
                        "required_quantity": 2,
                        "available_quantity": 1,
                    }
                ],
            }
        ],
        llm_call_stages=["candidate_universe_planner", "main_composer"],
    )

    state = contract["composer_execution_state"]
    validation = contract["validation_ledger"]
    reason = contract["safe_no_recommendation"]
    assert state["final_status_source"] == "composer_rejected_by_validation"
    assert validation["concrete_reasons"]
    assert reason["missing_roles"] == []
    assert reason["validation_rejections"][0]["concrete_validation_reasons"] == [
        "stock_quantity_mismatch",
        "stock_shortage:switch 2 > 1",
    ]


def test_schema_validation_failure_is_execution_failure_not_bom_rejection() -> None:
    contract = _contract(
        fallback_reason="llm_configurator_validation_failed",
        error_type="ValidationError",
        parse_diagnostics={
            "llm_parse_stage": "main_composer",
            "parse_status": "validation_error",
            "llm_schema_validation_errors": [
                {
                    "loc": ["recommendations", "0", "proposal_role"],
                    "type": "string_type",
                    "message": "Input should be a valid string",
                }
            ],
        },
        final_status_source="composer_schema_validation_failed",
        proposal_count=1,
        validation_summary={"validation_rejected_count": 1},
        rejected_recommendations_debug_safe=[
            {
                "recommendation_id": "schema_invalid",
                "rejection_category": "invalid_schema",
                "rejection_code": "invalid_schema",
                "concrete_validation_reasons": ["invalid_schema"],
            }
        ],
        llm_call_stages=["candidate_universe_planner", "main_composer"],
    )

    state = contract["composer_execution_state"]
    reason = contract["safe_no_recommendation"]
    assert state["final_status_source"] == "composer_schema_validation_failed"
    assert state["attempted"] is True
    assert state["returned_bom"] is False
    assert state["schema_validation_failed"] is True
    assert state["validation_rejected"] is False
    assert state["execution_failure"] is True
    assert reason["blockers"][0]["type"] == "composer_execution_failure"


def test_provider_timeout_has_explicit_execution_status() -> None:
    contract = _contract(
        fallback_reason="composer_provider_timeout",
        error_type="LlmReadTimeoutError",
        parse_diagnostics={
            "composer_failure_stage": "main_composer",
            "composer_failure_error_type": "LlmReadTimeoutError",
            "composer_timeout_fallback_attempted": True,
            "composer_timeout_fallback_success": False,
        },
        final_status_source="composer_provider_timeout",
        llm_call_stages=["candidate_universe_planner", "main_composer"],
    )

    state = contract["composer_execution_state"]
    reason = contract["safe_no_recommendation"]
    assert state["final_status_source"] == "composer_provider_timeout"
    assert state["execution_failure"] is True
    assert state["returned_bom"] is False
    assert reason["fallback_reason"] == "composer_provider_timeout"
    assert reason["blockers"][0]["type"] == "composer_execution_failure"


def _contract(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "product_group": "network",
        "primary_object": "switch",
        "classified_requirements": [
            {
                "requirement_id": "req_switch",
                "source_text": "one 48 port PoE switch",
                "classification": "purchasable_component_role",
                "fulfillment_mode": "separate_component_required",
                "target_role": "switch",
                "hardness": "hard",
            }
        ],
        "hard_roles": ["switch"],
        "role_candidate_count": {"switch": 1},
        "roles_sent_to_composer": ["switch"],
        "role_lifecycle_trace": [
            {"role": "switch", "composer_package": True, "dropped_reason": None}
        ],
        "role_fulfillment_diagnostics": [
            {"role": "switch", "lifecycle_reason": "sent_to_composer", "candidate_count": 1}
        ],
        "roles_dropped_reason_by_role": {},
        "attempt_decision": {
            "should_attempt": True,
            "blocked_by": [],
            "llm_call_stages": ["candidate_universe_planner"],
        },
        "selected_components_by_role": {},
        "primary_recommendation_status": "no_recommendation",
        "primary_recommendation": {},
        "recommended_builds": [],
        "no_recommendation_reason": {},
        "fallback_reason": "",
        "error_type": "",
        "parse_diagnostics": {},
        "proposal_count": 0,
        "valid_proposals_count": 0,
        "final_status_source": "composer_no_recommendation",
        "llm_call_stages": ["candidate_universe_planner", "main_composer"],
        "validation_hard_mismatches": [],
        "validation_unverified_requirements": [],
        "validation_summary": {},
        "rejected_recommendations_debug_safe": [],
        "original_request_text": "Need one switch",
    }
    defaults.update(overrides)
    return build_execution_contract(**defaults)
