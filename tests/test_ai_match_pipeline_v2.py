from __future__ import annotations

import asyncio
import json
from collections.abc import Generator, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import LlmSettings
from app.core.database import Base
from app.db.models import DistributorCategory, DistributorProduct, DistributorStockPrice
from app.llm.base import LlmInvalidJsonError, LlmReadTimeoutError, LlmServerError
from app.matching.ai_match_orchestrator import (
    AiMatchOrchestratorRequest,
    run_ai_match_orchestrator,
)
from app.matching.ai_match_pipeline_v2 import (
    COMPOSER_REJECTED_BY_VALIDATION,
    PIPELINE_VERSION,
    run_ai_match_pipeline_v2,
)
from app.matching.spec_schema import StockSpec, StockSpecItem

SERVER_78_TEXT = """
Execution: 1U
Sockets: 2
CPU: Intel 6th generation, 2 pcs, at least 24 cores
RAM: 256 GB DDR5 RDIMM
Disks: 6 x SSD 1920 GB SATA, 2 x SSD 480 GB SATA
Controller: LSI Logic 9400-8i / LSI 9500-8i
Network adapter: Intel X710-DA2 2x10GbE SFP+
Power: 2 x 2000W hot-swap Platinum, C13-C14 cables, C13-Schuko cables
Cooling: 8 fans N+1
Interfaces: USB 3.0, serial RJ-45, VGA, remote management RJ-45
""".strip()


class AsyncSessionAdapter:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def execute(self, statement: Any) -> Any:
        return self._session.execute(statement)


class RecordingComposerClient:
    def __init__(
        self,
        responder: Any,
        *,
        planner_responder: Any | None = None,
    ) -> None:
        self._composer_responder = responder
        self._planner_responder = planner_responder or _default_planner_response
        self.packages: list[dict[str, Any]] = []
        self.planner_payloads: list[dict[str, Any]] = []
        self.planner_outputs: list[dict[str, Any]] = []
        self.system_prompts: list[str] = []

    @property
    def calls(self) -> int:
        return len(self.packages)

    @property
    def planner_calls(self) -> int:
        return len(self.planner_payloads)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.system_prompts.append(system_prompt)
        payload = json.loads(user_prompt)
        if "category_catalog" in payload:
            self.planner_payloads.append(payload)
            response = self._planner_response(payload)
            self.planner_outputs.append(response)
            return response
        self.packages.append(payload)
        stage = payload.get("multi_pass_stage")
        responder_name = getattr(self._composer_responder, "__name__", "")
        responder_is_stage_aware = "multi_pass" in responder_name
        if stage == "requirement_contract" and not responder_is_stage_aware:
            return _requirement_contract_for_payload(payload)
        if stage == "completeness_critic" and not responder_is_stage_aware:
            return {
                "all_hard_requirements_covered": True,
                "missing_roles": [],
                "insufficient_quantities": [],
                "unverified_requirements": [],
                "hard_mismatch_risks": [],
                "recommended_repair_actions": [],
            }
        return self._composer_responder(payload)

    def _planner_response(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        responder = self._planner_responder
        if isinstance(responder, list):
            index = min(len(self.planner_payloads) - 1, len(responder) - 1)
            return responder[index]
        return responder(payload)


class SequencedPlannerResponder:
    def __init__(self, *responders: Any) -> None:
        self._responders = responders
        self.calls = 0

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        index = min(self.calls, len(self._responders) - 1)
        self.calls += 1
        responder = self._responders[index]
        if callable(responder):
            return responder(payload)
        return responder


class SequencedComposerResponder:
    def __init__(self, *responders: Any) -> None:
        self._responders = responders
        self.calls = 0

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        index = min(self.calls, len(self._responders) - 1)
        self.calls += 1
        responder = self._responders[index]
        if isinstance(responder, BaseException):
            raise responder
        if callable(responder):
            return responder(payload)
        return responder


@pytest.fixture()
def db_session() -> Generator[Session]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield session
    engine.dispose()


def test_v2_does_not_call_old_requirement_classifier_gates_before_composer(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_basic_server_catalog(db_session)

    def fail_old_planner(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("v1 semantic planner should not run in v2")

    monkeypatch.setattr(
        "app.matching.match_engine.plan_semantic_matrix_roles",
        fail_old_planner,
    )
    client = RecordingComposerClient(_valid_server_response)

    result = _run_v2(db_session, client=client)

    assert client.calls == 2
    assert client.planner_calls == 1
    assert result.report_fields["composer_mode"] == "composer_cascade"
    assert result.report_fields["role_evaluation_used"] is False
    assert result.report_fields["role_evaluation_skipped_reason"] == (
        "default_composer_cascade"
    )
    assert result.report_fields["llm_call_stages"] == [
        "candidate_universe_planner",
        "requirement_contract",
        "main_composer",
    ]
    assert result.report_fields["requirement_contract_used"] is True
    assert result.report_fields["main_composer_used"] is True
    assert result.report_fields["pipeline_version"] == PIPELINE_VERSION
    assert result.report_fields["pre_composer_requirement_classifier_status"] is None
    assert result.report_fields["pre_composer_semantic_diagnostics_are_blocking"] is False


def test_v2_attempts_composer_when_known_group_universe_candidates_under_limit(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(_valid_server_response)

    result = _run_v2(db_session, client=client)

    assert client.calls == 2
    decision = result.report_fields["composer_attempt_decision"]
    assert decision["should_attempt"] is True
    assert decision["blocked_by"] == []
    assert result.match_result.final_status_source == "composer_validated"
    assert result.report_fields["final_status_source"] == result.report_fields[
        "composer_execution_state"
    ]["final_status_source"]


def test_v2_preserves_full_matrix_candidate_exposure(db_session: Session) -> None:
    _seed_basic_server_catalog(db_session, cpu_count=4)

    result = _run_v2(db_session, preview_only=True)

    package = result.package
    assert package["composer_package_full_matrix_used"] is True
    assert package["full_candidate_matrix_count_by_role"]["cpu"] == 4
    assert package["composer_package_candidate_count_by_role"]["cpu"] == 4
    assert len(package["component_candidate_matrix"]["cpu"]) == 4


def test_v2_composer_package_contains_original_request_text(db_session: Session) -> None:
    _seed_basic_server_catalog(db_session)

    result = _run_v2(db_session, preview_only=True)

    assert result.package["original_request_text"] == SERVER_78_TEXT
    assert result.package["user_request"] == SERVER_78_TEXT


def test_v2_composer_package_contains_all_candidates_under_limit(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session, cpu_count=3, ram_count=2, ssd_count=2)

    result = _run_v2(db_session, preview_only=True)

    package = result.package
    expected_total = sum(package["full_candidate_matrix_count_by_role"].values())
    assert package["composer_package_candidate_total"] == expected_total
    assert package["dropped_before_composer_count_by_role"] == {
        role: 0
        for role in package["composer_package_candidate_count_by_role"]
    }


def test_v2_rejects_invented_candidate_ids(db_session: Session) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(_invented_id_response)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source == "composer_no_recommendation"
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.report_fields["code_completeness_result"]["unknown_component_ids"]


def test_v2_rejects_stock_quantity_mismatch(db_session: Session) -> None:
    _seed_basic_server_catalog(db_session, ram_quantity=1)
    client = RecordingComposerClient(_stock_shortage_response)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source == COMPOSER_REJECTED_BY_VALIDATION
    assert result.report_fields["validation_repair_attempted"] is True
    assert result.report_fields["validation_repair_success"] is False
    assert any(
        row.get("stock_shortages")
        for row in result.match_result.rejected_ai_recommendations_debug_safe
    )
    validation = result.report_fields["validation_ledger"]
    assert validation["validation_rejected"] is True
    assert validation["rejected_candidates"]
    assert validation["rejected_candidates"][0]["concrete_validation_reasons"]


def test_v2_stock_override_ignores_normalized_optional_accessory_quantity(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    fan_stock = (
        db_session.query(DistributorStockPrice)
        .filter(DistributorStockPrice.item_id == "fan-1")
        .one()
    )
    fan_stock.quantity_value = 1
    db_session.commit()
    client = RecordingComposerClient(_server_with_overstated_optional_accessory_response)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source == "composer_validated"
    accessory = next(
        component
        for component in result.match_result.primary_recommendation["components"]
        if component["role"] == "other_accessory"
    )
    assert accessory["optional_component"] is True
    assert accessory["quantity_required"] == 1
    assert accessory["llm_quantity"] == 8
    assert not result.match_result.validation_hard_mismatches


def test_v2_rejects_obvious_hard_mismatch_in_86_like_bom(db_session: Session) -> None:
    _seed_basic_server_catalog(
        db_session,
        platform_name="AMD EPYC SP5 DDR5 server platform",
        cpu_name="Intel Xeon Gold 24 core LGA4677 processor",
    )
    client = RecordingComposerClient(_platform_cpu_mismatch_response)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source == COMPOSER_REJECTED_BY_VALIDATION
    assert result.report_fields["validation_repair_attempted"] is True
    assert result.report_fields["validation_repair_success"] is False
    assert result.match_result.ai_validation_summary["rejected_platform_cpu_mismatch"] == 1
    rejected_attempts = {
        row.get("validation_attempt")
        for row in result.match_result.rejected_ai_recommendations_debug_safe
    }
    assert {"initial", "repair"}.issubset(rejected_attempts)
    assert "repeated_rejected_component_combination" in json.dumps(
        result.match_result.rejected_ai_recommendations_debug_safe,
        ensure_ascii=False,
    )


def test_v2_validation_rejection_platform_cpu_mismatch_repairs_to_valid_bom(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        platform_name="Intel Xeon LGA4710 DDR5 1U dual socket server platform",
        cpu_name="Intel Xeon Gold 24 core LGA4677 processor",
    )
    _seed_product(
        db_session,
        item_id="cpu-compatible",
        category_id="cat-cpu",
        category_name="CPU Processors",
        part_number="CPU-4710",
        item_name="Intel Xeon 6th generation 24 core LGA4710 processor",
        quantity=4,
        price=Decimal("900"),
    )
    db_session.commit()
    client = RecordingComposerClient(
        SequencedComposerResponder(
            _platform_cpu_mismatch_response,
            _platform_cpu_compatible_repair_response,
        )
    )

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source == "composer_validated"
    assert result.match_result.primary_recommendation_status == "valid"
    assert result.report_fields["validation_repair_attempted"] is True
    assert result.report_fields["validation_repair_success"] is True
    assert result.match_result.llm_repair_used is True
    assert "post_validation_repair" in result.report_fields["llm_call_stages"]
    compatible_cpu_id = next(
        row["component_candidate_id"]
        for row in result.package["component_candidate_matrix"]["cpu"]
        if row["part_number"] == "CPU-4710"
    )
    assert (
        result.match_result.primary_recommendation["component_candidate_ids"]["cpu"]
        == compatible_cpu_id
    )


def test_v2_validation_repair_prompt_gets_concrete_rejected_ids_and_reason(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        platform_name="Intel Xeon LGA4710 DDR5 1U dual socket server platform",
        cpu_name="Intel Xeon Gold 24 core LGA4677 processor",
    )
    _seed_product(
        db_session,
        item_id="cpu-compatible",
        category_id="cat-cpu",
        category_name="CPU Processors",
        part_number="CPU-4710",
        item_name="Intel Xeon 6th generation 24 core LGA4710 processor",
        quantity=4,
        price=Decimal("900"),
    )
    db_session.commit()
    client = RecordingComposerClient(
        SequencedComposerResponder(
            _platform_cpu_mismatch_response,
            _platform_cpu_compatible_repair_response,
        )
    )

    _run_v2(db_session, client=client)

    repair_payload = next(
        payload
        for payload in client.packages
        if payload.get("multi_pass_stage") == "validation_repair"
    )
    serialized = json.dumps(repair_payload, ensure_ascii=False)
    rejected_ids = set(repair_payload["rejected_component_candidate_ids"])
    forbidden_ids = set(
        repair_payload["forbidden_component_combinations"][0][
            "component_candidate_ids"
        ].values()
    )
    assert forbidden_ids.issubset(rejected_ids)
    assert any(component_id.startswith("server_platform-") for component_id in rejected_ids)
    assert any(component_id.startswith("cpu-") for component_id in rejected_ids)
    assert "fatal socket mismatch" in serialized
    assert repair_payload["validator_errors"]
    assert repair_payload["forbidden_component_combinations"]
    assert repair_payload["candidate_exposure_policy"]["silent_trimming"] is False
    repair_prompt = next(
        prompt
        for prompt in client.system_prompts
        if "Validation-Aware Repair pass" in prompt
    )
    assert "general_notes-only" in repair_prompt
    assert "vendor-specific option kits" in repair_prompt.casefold()
    assert "must match platform vendor" in repair_prompt


def test_v2_validation_repair_structured_no_recommendation_is_preserved(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        platform_name="Intel Xeon LGA4710 DDR5 1U dual socket server platform",
        cpu_name="Intel Xeon Gold 24 core LGA4677 processor",
    )
    client = RecordingComposerClient(
        SequencedComposerResponder(
            _platform_cpu_mismatch_response,
            _validation_repair_no_recommendation_response,
        )
    )

    result = _run_v2(db_session, client=client)

    reason = result.match_result.no_recommendation_reason
    assert result.match_result.final_status_source == "composer_no_recommendation"
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert reason["structured_no_recommendation"] is True
    assert reason["validator_errors"]
    assert result.report_fields["validation_repair_attempted"] is True
    assert result.report_fields["validation_repair_returned_no_recommendation"] is True


def test_v2_validation_repair_empty_response_has_explicit_diagnostics(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        platform_name="Intel Xeon LGA4710 DDR5 1U dual socket server platform",
        cpu_name="Intel Xeon Gold 24 core LGA4677 processor",
    )
    client = RecordingComposerClient(
        SequencedComposerResponder(
            _platform_cpu_mismatch_response,
            _validation_repair_empty_response,
        )
    )

    result = _run_v2(db_session, client=client)

    reason_text = json.dumps(
        result.match_result.no_recommendation_reason,
        ensure_ascii=False,
    )
    assert result.match_result.final_status_source == COMPOSER_REJECTED_BY_VALIDATION
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.report_fields["validation_repair_attempted"] is True
    assert result.report_fields["validation_repair_success"] is False
    assert result.report_fields["validation_repair_empty_output"] is True
    assert (
        result.report_fields["validation_repair_failure_reason"]
        == "empty_without_no_recommendation"
    )
    assert result.match_result.no_recommendation_reason["structured_no_recommendation"]
    assert "fatal socket mismatch" in reason_text
    assert "empty_without_no_recommendation" in reason_text


def test_v2_validation_repair_general_notes_only_is_not_accepted(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        platform_name="Intel Xeon LGA4710 DDR5 1U dual socket server platform",
        cpu_name="Intel Xeon Gold 24 core LGA4677 processor",
    )
    client = RecordingComposerClient(
        SequencedComposerResponder(
            _platform_cpu_mismatch_response,
            _validation_repair_general_notes_only_response,
        )
    )

    result = _run_v2(db_session, client=client)

    final_bom = result.report_fields["final_bom_after_repair"]
    assert result.match_result.final_status_source == COMPOSER_REJECTED_BY_VALIDATION
    assert result.report_fields["validation_repair_empty_output"] is True
    assert (
        result.report_fields["validation_repair_failure_reason"]
        == "empty_without_no_recommendation"
    )
    assert final_bom["recommendations"] == []
    assert final_bom["no_recommendation"]
    assert "BOM assembled" not in json.dumps(final_bom, ensure_ascii=False)


def test_v2_returns_structured_no_recommendation_when_composer_declines(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(_structured_no_recommendation_response)

    result = _run_v2(db_session, client=client)

    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.match_result.no_recommendation_reason["structured_no_recommendation"]
    assert result.match_result.final_status_source == "composer_no_recommendation"


def test_v2_network_79_like_request_still_works(db_session: Session) -> None:
    _seed_network_catalog(db_session)
    client = RecordingComposerClient(_valid_network_response)

    result = _run_v2(
        db_session,
        spec=_network_spec(),
        client=client,
    )

    assert client.calls == 2
    assert result.report_fields["product_group"] == "network"
    assert result.match_result.final_status_source == "composer_validated"
    assert result.report_fields["main_composer_used"] is True
    assert result.report_fields["composer_execution_state"]["attempted"] is True
    assert result.report_fields["composer_execution_state"]["blocked_by"] == []
    assert result.package["composer_package_candidate_count_by_role"]["switch"] == 1
    assert result.report_fields["hard_purchasable_bom_roles"] == ["switch"]
    assert not result.report_fields.get("validation_repair_attempted")


def test_v2_multi_pass_requirement_contract_extracts_server_78_roles(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_complete_response)

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(multi_pass=True),
    )

    contract = result.report_fields["requirement_contract"]
    assert result.report_fields["composer_mode"] == "deep_audit"
    assert {
        "server_platform",
        "cpu",
        "ram",
        "ssd",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }.issubset(set(contract["required_roles"]))
    assert contract["required_quantities_by_role"]["cpu"]["count"] == 2
    assert contract["required_quantities_by_role"]["ram"]["module_count"] == 8


def test_v2_requirement_contract_validation_error_does_not_block_main_composer(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(
        _multi_pass_invalid_requirement_contract_then_valid_bom
    )

    result = _run_v2(db_session, client=client)

    stages = [payload.get("multi_pass_stage") for payload in client.packages]
    assert stages == ["requirement_contract", "bom_composition"]
    assert result.report_fields["composer_attempt_decision"]["should_attempt"] is True
    assert result.report_fields["composer_attempt_decision"]["blocked_by"] == []
    assert result.report_fields["requirement_contract_used"] is False
    assert result.report_fields["requirement_contract_fallback_used"] is True
    assert result.report_fields["requirement_contract_error_stage"] == (
        "requirement_contract"
    )
    assert result.report_fields["requirement_contract_error_type"] == "ValidationError"
    assert result.report_fields["requirement_contract_validation_errors"]
    assert result.report_fields["main_composer_used"] is True
    assert result.report_fields["bom_composer_used"] is True
    assert result.report_fields["composer_execution_state"]["attempted"] is True
    assert result.match_result.final_status_source == "composer_validated"


def test_v2_multi_pass_role_evaluation_covers_all_chunked_candidates(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        cpu_count=5,
        ram_count=3,
        ssd_count=4,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_complete_response)

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(multi_pass=True, multi_pass_chunk_size=2),
    )

    coverage = result.report_fields["role_evaluation_coverage_by_role"]
    assert coverage["cpu"]["candidate_count"] == 5
    assert coverage["cpu"]["considered_count"] == 5
    assert coverage["cpu"]["all_candidates_considered"] is True
    role_payloads = [
        payload
        for payload in client.packages
        if payload.get("multi_pass_stage") == "role_evaluation"
        and payload.get("role") == "cpu"
    ]
    assert len(role_payloads) == 3
    assert sorted(
        candidate_id
        for payload in role_payloads
        for candidate_id in payload["candidate_ids_for_chunk"]
    ) == sorted(result.package["composer_package_candidate_ids_by_role"]["cpu"])


def test_v2_multi_pass_bom_uses_role_summaries_and_validates_complete_bom(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_complete_response)

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(multi_pass=True),
    )

    bom_payload = next(
        payload
        for payload in client.packages
        if payload.get("multi_pass_stage") == "bom_composition"
    )
    assert "role_evaluation_summaries" in bom_payload
    assert result.report_fields["bom_composer_used"] is True
    assert result.match_result.final_status_source == "composer_validated"
    component_roles = {
        component["role"]
        for component in result.match_result.primary_recommendation["components"]
    }
    assert {
        "server_platform",
        "cpu",
        "ram",
        "ssd",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }.issubset(component_roles)
    quantities = result.match_result.primary_recommendation["quantities"]
    assert quantities["cpu"] == 2
    assert quantities["ram"] == 8
    assert quantities["ssd"] == 8
    assert quantities["power_supply"] == 2


def test_v2_multi_pass_critic_repairs_90_like_incomplete_bom(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_incomplete_then_repair_response)

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(multi_pass=True),
    )

    critic = result.report_fields["completeness_critic_result"]
    assert critic["missing_roles"] == [
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    ]
    assert result.report_fields["repair_composer_used"] is True
    assert result.match_result.final_status_source == "composer_validated"
    component_roles = {
        component["role"]
        for component in result.match_result.primary_recommendation["components"]
    }
    assert "storage_controller" in component_roles
    assert "network_adapter" in component_roles
    assert "power_supply" in component_roles
    assert "cable" in component_roles


def test_v2_default_composer_cascade_repairs_90_like_incomplete_bom(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_incomplete_then_repair_response)

    result = _run_v2(db_session, client=client)

    assert result.report_fields["composer_mode"] == "composer_cascade"
    assert result.report_fields["role_evaluation_used"] is False
    assert result.report_fields["repair_composer_used"] is True
    assert result.match_result.final_status_source == "composer_validated"


def test_v2_multi_pass_repair_can_return_structured_no_recommendation(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_incomplete_then_no_recommendation)

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(multi_pass=True),
    )

    assert result.report_fields["repair_composer_used"] is True
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.match_result.final_status_source == "composer_no_recommendation"
    assert result.match_result.no_recommendation_reason["structured_no_recommendation"]
    assert not result.report_fields.get("validation_repair_attempted")


def test_v2_call_budget_stops_before_hidden_composer_loop(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(_valid_server_response)

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(max_calls=2),
    )

    assert client.planner_calls == 1
    assert client.calls == 1
    assert result.report_fields["llm_call_budget_exceeded"] is True
    assert result.report_fields["llm_call_count"] == 2
    assert result.report_fields["max_llm_calls_per_match"] == 2
    assert result.report_fields["role_fulfillment_diagnostics"]
    assert result.report_fields["role_lifecycle_trace"]
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.match_result.llm_fallback_reason == "llm_call_budget_exceeded"


def test_v2_orchestrator_report_json_normalizes_repair_alias_no_recommendation(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_incomplete_then_alias_no_recommendation)

    result = asyncio.run(
        run_ai_match_orchestrator(
            AiMatchOrchestratorRequest(spec=_server_spec(), pipeline_v2=True),
            AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            llm_configurator_client=client,
            llm_settings=_composer_settings(multi_pass=True),
        )
    )

    reason = result.report_json["no_recommendation_reason"]
    assert result.report_json["composer_mode"] == "deep_audit"
    assert result.report_json["primary_recommendation_status"] == "no_recommendation"
    assert result.report_json["llm_fallback_reason"] == (
        "composer_structured_no_recommendation"
    )
    assert reason["summary"] == "Repair pass found no safe complete BOM."
    assert {row["role"] for row in reason["role_failures"]} >= {
        "compute_node",
        "fabric_adapter",
    }
    assert "Request compatible replacements." in reason["recommended_next_actions"]
    assert result.report_json["final_bom_after_repair"]["no_recommendation"][
        "role_level_reasons"
    ]


def test_v2_large_matrix_uses_compact_full_matrix_without_candidate_loss(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        cpu_count=180,
        ram_count=140,
        ssd_count=110,
    )

    result = _run_v2(db_session, preview_only=True)

    package = result.package
    assert package["v2_package_mode"] == "compact_full_matrix"
    assert package["compact_package_full_matrix_used"] is True
    assert package["package_candidate_loss"] is False
    assert package["compact_candidate_total"] == package[
        "composer_package_candidate_total"
    ]
    assert package["compact_candidate_count_by_role"] == package[
        "composer_package_candidate_count_by_role"
    ]
    assert package["compact_context_chars"] < package["verbose_context_chars"]
    assert "package_json" in package["removed_verbose_fields"]


def test_v2_provider_context_limit_on_verbose_retries_compact_and_validates(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    responder = SequencedComposerResponder(
        LlmServerError("maximum context length 32768 exceeded", status_code=503),
        _valid_server_response,
    )
    client = RecordingComposerClient(responder)

    result = _run_v2(db_session, client=client)

    assert client.calls == 3
    assert client.packages[1]["v2_package_mode"] == "verbose_full_matrix"
    assert client.packages[2]["v2_package_mode"] == "compact_full_matrix"
    assert result.match_result.final_status_source == "composer_validated"
    assert result.report_fields["provider_context_limit_retry_compact_attempted"] is True
    assert result.report_fields["provider_context_limit_retry_compact_success"] is True
    assert result.report_fields["provider_context_limit_after_compact"] is False
    assert result.report_fields["provider_context_limit_compact_chars"] < result.report_fields[
        "provider_context_limit_original_chars"
    ]


def test_v2_provider_context_limit_after_compact_returns_explicit_fallback(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    responder = SequencedComposerResponder(
        LlmServerError("maximum context length 32768 exceeded", status_code=503),
        LlmServerError("maximum context length 32768 exceeded", status_code=503),
    )
    client = RecordingComposerClient(responder)

    result = _run_v2(db_session, client=client)

    assert client.calls == 3
    assert result.match_result.llm_fallback_reason == "compact_full_matrix_context_limit"
    assert result.match_result.final_status_source == "provider_context_limit"
    assert result.report_fields["provider_context_limit_retry_compact_attempted"] is True
    assert result.report_fields["provider_context_limit_retry_compact_success"] is False
    assert result.report_fields["provider_context_limit_after_compact"] is True


def test_v2_candidate_universe_planner_server_78_treats_x710_as_component(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(_valid_server_response)

    result = _run_v2(db_session, client=client, preview_only=True)

    assert client.planner_calls == 1
    assert result.report_fields["product_group"] == "server"
    assert result.report_fields["primary_object"] == "server"
    assert result.report_fields["candidate_universe_planner_mode"] == (
        "llm_candidate_universe_planner_v2"
    )
    component_roles = {
        row["role"]: row["source_text"]
        for row in result.report_fields["component_role_indicators"]
    }
    assert "network_adapter" in component_roles
    assert "X710" in component_roles["network_adapter"]
    assert result.report_fields["competing_product_groups"][0]["product_group"] == (
        "network"
    )


def test_v2_composer_cascade_preserves_component_feature_diagnostics(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(
        _valid_server_response,
        planner_responder=_server_planner_with_embedded_requirements,
    )

    result = _run_v2(db_session, client=client)

    contract_payload = client.packages[0]
    main_payload = client.packages[1]
    planner_context = contract_payload["planner_context"]
    assert contract_payload["multi_pass_stage"] == "requirement_contract"
    assert any(
        row["role"] == "network_adapter"
        for row in planner_context["component_role_indicators"]
    )
    assert any(
        row.get("requirement_text") == "NVMe-capable front storage"
        for row in planner_context["embedded_requirements"]
    )
    assert any(
        row["role"] == "network_adapter"
        for row in planner_context["role_fulfillment_diagnostics"]
    )
    assert any(
        row["role"] == "network_adapter"
        for row in main_payload["component_role_indicators"]
    )
    assert any(
        row.get("requirement_text") == "NVMe-capable front storage"
        for row in main_payload["embedded_requirements"]
    )
    assert result.report_fields["role_fulfillment_diagnostics"]
    assert result.report_fields["role_lifecycle_trace"]


def test_v2_storage_nas_request_sends_system_and_drive_roles_to_composer(
    db_session: Session,
) -> None:
    _seed_storage_catalog(db_session)
    client = RecordingComposerClient(
        _valid_storage_response,
        planner_responder=_storage_nas_planner_response,
    )

    result = _run_v2(
        db_session,
        spec=_storage_nas_spec(),
        client=client,
        settings=_composer_settings(max_package_chars=100000),
    )

    main_payload = client.packages[1]
    assert result.report_fields["product_group"] == "storage"
    assert result.report_fields["primary_object"] == "storage_system"
    assert main_payload["component_candidate_matrix"]["storage_system"]
    assert main_payload["component_candidate_matrix"]["drive"]
    assert result.package["composer_package_candidate_count_by_role"][
        "storage_system"
    ] == 1
    assert result.package["composer_package_candidate_count_by_role"]["drive"] == 2
    lifecycle = {
        row["role"]: row["lifecycle_reason"]
        for row in result.report_fields["role_fulfillment_diagnostics"]
    }
    assert lifecycle["storage_system"] == "sent_to_composer"
    assert lifecycle["drive"] == "sent_to_composer"
    assert any(
        row["status"] == "satisfied_by_selected_component"
        and row["role"] == "drive"
        for row in result.report_fields["coverage_evidence"]
    )


def test_v2_optional_vendor_preference_is_not_hard_blocker(
    db_session: Session,
) -> None:
    _seed_storage_catalog(db_session)
    client = RecordingComposerClient(
        _valid_storage_response,
        planner_responder=_storage_nas_planner_with_optional_vendor,
    )

    result = _run_v2(
        db_session,
        spec=_storage_nas_spec(),
        client=client,
        preview_only=True,
    )

    lifecycle = {
        row["role"]: row["lifecycle_reason"]
        for row in result.report_fields["role_fulfillment_diagnostics"]
    }
    assert lifecycle["vendor"] == "optional_only"
    assert result.report_fields["roles_dropped_reason_by_role"]["vendor"] == (
        "optional_only"
    )
    assert "vendor" not in result.report_fields["category_plan"]
    assert client.planner_calls == 1


def test_v2_nas_feature_package_under_limit_attempts_composer(
    db_session: Session,
) -> None:
    _seed_storage_system_only_catalog(db_session)
    client = RecordingComposerClient(
        _valid_storage_system_only_response,
        planner_responder=_storage_nas_feature_planner_response,
    )

    result = _run_v2(
        db_session,
        spec=_storage_nas_spec(),
        client=client,
        settings=_composer_settings(max_package_chars=100000),
    )

    decision = result.report_fields["composer_attempt_decision"]
    assert client.calls == 2
    assert decision["should_attempt"] is True
    assert "package_over_budget" not in decision["blocked_by"]
    assert "package_too_large_and_chunking_failed" not in decision["blocked_by"]
    assert result.match_result.final_status_source == "composer_validated"
    assert result.report_fields["hard_purchasable_bom_roles"] == ["storage_system"]
    assert result.report_fields["primary_object_feature_requirements"]
    assert any(
        row["status"] == "satisfied_by_primary_object"
        and row["role"] == "drive"
        for row in result.report_fields["coverage_evidence"]
    )
    assert result.report_fields["package_budget_selected_context_chars"] <= (
        result.report_fields["effective_max_package_chars"]
    )


def test_v2_server_compact_package_under_1500000_attempts_composer(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        cpu_count=180,
        ram_count=140,
        ssd_count=110,
    )
    client = RecordingComposerClient(_valid_server_response)

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(max_package_chars=1_500_000),
    )

    decision = result.report_fields["composer_attempt_decision"]
    assert result.package["v2_package_mode"] == "compact_full_matrix"
    assert result.package["package_candidate_loss"] is False
    assert decision["selected_context_chars"] <= decision["effective_max_package_chars"]
    assert decision["should_attempt"] is True
    assert decision["blocked_by"] == []
    assert result.report_fields["composer_execution_state"]["should_attempt"] is True
    assert result.report_fields["composer_execution_state"]["attempted"] is True
    assert client.calls == 2


def test_v2_network_optional_uplink_roles_do_not_block_composer(
    db_session: Session,
) -> None:
    _seed_network_catalog(db_session)
    client = RecordingComposerClient(
        _valid_network_response,
        planner_responder=_network_planner_with_optional_uplink_accessories,
    )

    result = _run_v2(db_session, spec=_network_spec(), client=client)

    lifecycle = {
        row["role"]: row["lifecycle_reason"]
        for row in result.report_fields["role_fulfillment_diagnostics"]
    }
    decision = result.report_fields["composer_attempt_decision"]
    assert result.match_result.final_status_source == "composer_validated"
    assert result.report_fields["hard_purchasable_bom_roles"] == ["switch"]
    assert lifecycle["transceiver"] == "optional_only"
    assert lifecycle["dac_cable"] == "optional_only"
    assert lifecycle["stacking_module"] == "engineering_check_only"
    assert decision["should_attempt"] is True
    assert decision["blocked_by"] == []


def test_v2_optional_accessory_candidates_sent_to_composer_are_not_blockers(
    db_session: Session,
) -> None:
    _seed_network_catalog(db_session, include_transceiver=True)
    client = RecordingComposerClient(
        _valid_network_response,
        planner_responder=_network_planner_with_optional_transceiver_category,
    )

    result = _run_v2(db_session, spec=_network_spec(), client=client)

    main_payload = client.packages[1]
    assert main_payload["component_candidate_matrix"]["transceiver"]
    assert result.package["composer_package_candidate_count_by_role"]["transceiver"] == 1
    assert result.report_fields["hard_purchasable_bom_roles"] == ["switch"]
    assert result.match_result.final_status_source == "composer_validated"


def test_v2_network_missing_switch_no_recommendation_uses_primary_role_repair(
    db_session: Session,
) -> None:
    _seed_network_catalog(db_session)
    _seed_product(
        db_session,
        item_id="switch-cheap-bad",
        category_id="cat-switch",
        category_name="Network Switches PoE",
        part_number="BAD-5PORT",
        item_name="Unmanaged 5 port 100M desktop switch without PoE or L3",
        quantity=10,
        price=Decimal("5"),
    )
    db_session.commit()
    client = RecordingComposerClient(
        _network_missing_switch_no_recommendation_response,
        planner_responder=_network_planner_response,
    )

    result = _run_v2(db_session, spec=_network_spec(), client=client)

    assert result.match_result.final_status_source == "composer_validated"
    assert result.match_result.primary_recommendation_status == "valid"
    switch = result.match_result.primary_recommendation["components"][0]
    assert switch["part_number"] == "SW-48P-4SFP"
    diagnostics = result.match_result.web_evidence_pack["diagnostics"]
    assert diagnostics["deterministic_primary_role_repair_attempted"] is True
    assert diagnostics["deterministic_primary_role_repair_success"] is True
    assert diagnostics["deterministic_primary_role_repair_skipped_by_text_requirement"] >= 1


def test_v2_drive_requirement_can_be_fulfilled_by_ssd_alias(
    db_session: Session,
) -> None:
    _seed_storage_ssd_alias_catalog(db_session)
    client = RecordingComposerClient(
        _valid_storage_system_with_ssd_response,
        planner_responder=_storage_drive_alias_planner_response,
    )

    result = _run_v2(
        db_session,
        spec=_storage_nas_spec(),
        client=client,
        preview_only=True,
    )

    lifecycle = {
        row["role"]: row
        for row in result.report_fields["role_fulfillment_diagnostics"]
    }
    assert lifecycle["drive"]["lifecycle_reason"] == "sent_to_composer"
    assert lifecycle["drive"]["fulfilled_by_role"] == "ssd"
    assert result.report_fields["product_group"] == "storage"
    assert result.report_fields["primary_object"] == "storage_system"
    assert result.package["composer_package_candidate_count_by_role"]["storage_system"] == 1
    assert result.package["composer_package_candidate_count_by_role"]["ssd"] == 1
    assert result.report_fields["roles_dropped_reason_by_role"].get("drive") != (
        "role_not_purchasable"
    )


def test_v2_server_drive_requirement_with_ssd_matrix_does_not_block_composer(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        cpu_count=130,
        ram_count=130,
        ssd_count=130,
    )
    client = RecordingComposerClient(
        _valid_server_response,
        planner_responder=_server_planner_with_drive_requirement_ssd_category,
    )

    result = _run_v2(db_session, client=client)

    decision = result.report_fields["composer_attempt_decision"]
    assert "drive" in result.report_fields["hard_purchasable_bom_roles"]
    assert result.package["v2_package_mode"] == "compact_full_matrix"
    assert result.package["composer_package_candidate_count_by_role"]["ssd"] == 130
    assert "drive" not in result.package["package_candidate_exposure_incomplete_roles"]
    assert result.package["package_candidate_exposure_incomplete"] is False
    assert decision["blocked_by"] == []
    assert decision["should_attempt"] is True
    assert result.match_result.final_status_source == "composer_validated"


def test_v2_structured_no_recommendation_omits_optional_role_failures(
    db_session: Session,
) -> None:
    _seed_network_catalog(db_session)
    client = RecordingComposerClient(
        SequencedComposerResponder(
            LlmInvalidJsonError(
                "invalid json",
                parse_stage="main_composer",
                invalid_json_reason="empty_or_truncated",
            )
        ),
        planner_responder=_network_planner_with_optional_uplink_accessories,
    )

    result = _run_v2(db_session, spec=_network_spec(), client=client)

    reason = result.match_result.no_recommendation_reason
    failure_roles = [row["role"] for row in reason["role_failures"]]
    assert failure_roles
    assert len(failure_roles) == len(set(failure_roles))
    assert "switch" in failure_roles
    assert "transceiver" not in failure_roles
    assert "dac_cable" not in failure_roles
    assert "stacking_module" not in failure_roles


def test_v2_invalid_composer_json_normalizes_to_safe_no_recommendation(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(
        SequencedComposerResponder(
            LlmInvalidJsonError(
                "invalid json",
                parse_stage="main_composer",
                invalid_json_reason="empty_or_truncated",
            )
        )
    )

    result = _run_v2(db_session, client=client)

    reason = result.match_result.no_recommendation_reason
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.match_result.final_status_source == (
        "composer_failure_safe_no_recommendation"
    )
    assert result.match_result.llm_fallback_reason == "multi_pass_invalid_json"
    assert reason["structured_no_recommendation"] is True
    assert reason["role_failures"]
    assert reason["missing_roles"] == []
    assert result.report_fields["composer_execution_state"]["execution_failure"] is True
    assert result.report_fields["composer_execution_state"]["returned_bom"] is False
    assert reason["diagnostics"]["llm_error_type"] == "LlmInvalidJsonError"
    assert result.report_fields["no_recommendation_reason"] == reason


def test_v2_composer_schema_validation_failure_is_not_validation_rejection(
    db_session: Session,
) -> None:
    _seed_network_catalog(db_session)
    client = RecordingComposerClient(
        _multi_pass_schema_invalid_bom,
        planner_responder=_network_planner_response,
    )

    result = _run_v2(db_session, spec=_network_spec(), client=client)

    reason = result.match_result.no_recommendation_reason
    assert result.report_fields["composer_attempt_decision"]["should_attempt"] is True
    assert result.report_fields["composer_attempt_decision"]["blocked_by"] == []
    assert result.report_fields["main_composer_used"] is True
    assert result.report_fields["composer_execution_state"]["attempted"] is True
    assert result.report_fields["composer_execution_state"]["execution_failure"] is True
    assert result.report_fields["composer_execution_state"]["validation_rejected"] is False
    assert result.match_result.final_status_source == "composer_schema_validation_failed"
    assert result.match_result.final_status_source != COMPOSER_REJECTED_BY_VALIDATION
    assert result.report_fields["validation_ledger"]["validation_rejected"] is True
    assert reason["diagnostics"]["composer_execution_state"]["final_status_source"] == (
        "composer_schema_validation_failed"
    )
    assert result.report_fields["llm_parse_stage"] == "main_composer"
    assert result.report_fields["llm_schema_validation_errors"]


def test_v2_network_requirement_summary_strings_do_not_fatal_fail_valid_ids(
    db_session: Session,
) -> None:
    _seed_network_catalog(db_session, include_transceiver=True)
    client = RecordingComposerClient(
        _network_string_requirement_summary_response,
        planner_responder=_network_planner_with_optional_transceiver_category,
    )

    result = _run_v2(db_session, spec=_network_spec(), client=client)

    assert result.match_result.final_status_source == "composer_validated"
    assert result.report_fields.get("llm_schema_validation_errors", []) == []
    recommendation = result.match_result.primary_recommendation
    assert recommendation["component_candidate_ids"]["switch"].startswith("switch-")
    assert recommendation["requirement_fulfillment_summary"]
    assert all(
        isinstance(row, dict)
        for row in recommendation["requirement_fulfillment_summary"]
    )
    assert any(
        row.get("closed_by") == "llm_string_summary"
        for row in recommendation["requirement_fulfillment_summary"]
    )


def test_v2_composer_selected_components_alias_normalizes_when_ids_are_valid(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_selected_components_alias_bom)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source == "composer_validated"
    assert result.report_fields.get("llm_schema_validation_errors", []) == []
    component_roles = {
        component["role"]
        for component in result.match_result.primary_recommendation["components"]
    }
    assert {
        "server_platform",
        "cpu",
        "ram",
        "ssd",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    }.issubset(component_roles)
    assert any(
        "temporary note" in note
        for note in result.report_fields["composer_assumptions"]
    )
    assert result.report_fields["composer_unverified_requirements"] == [
        {"role": "cable", "source_text": "confirm exact cable types"}
    ]


def test_v2_composer_component_candidate_ids_role_map_defaults_source_type_safely(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_component_map_without_source_type)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source == "composer_validated"
    assert result.report_fields.get("llm_schema_validation_errors", []) == []
    assert (
        result.match_result.primary_recommendation["source_type"]
        == "build_from_parts"
    )


def test_v2_composer_role_specific_ssd_keys_preserve_multiple_lines(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ssd_count=2,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_role_specific_ssd_bom)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source == "composer_validated"
    ssd_components = [
        component
        for component in result.match_result.primary_recommendation["components"]
        if component["role"] == "ssd"
    ]
    assert [component["quantity_required"] for component in ssd_components] == [6, 2]
    assert len({component["component_candidate_id"] for component in ssd_components}) == 2
    assert result.match_result.primary_recommendation["quantities"][
        "ssd_1920"
    ] == 6
    assert result.match_result.primary_recommendation["quantities"]["ssd_480"] == 2


def test_v2_composer_does_not_default_source_type_for_invented_ids(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(_multi_pass_missing_source_type_invented_id)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source != "composer_validated"
    assert result.match_result.primary_recommendation_status == "no_recommendation"


def test_v2_composer_empty_repair_after_missing_cable_becomes_no_recommendation(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_missing_cable_then_empty_repair)

    result = _run_v2(db_session, client=client)

    assert result.report_fields["repair_composer_used"] is True
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.match_result.final_status_source == "composer_no_recommendation"
    assert result.match_result.final_status_source != "composer_validated"
    assert "cable" in result.match_result.no_recommendation_reason["missing_roles"]
    assert result.report_fields["code_completeness_after_repair"]["repair_required"]
    assert result.report_fields["final_bom_after_repair"]["no_recommendation"]


def test_v2_composer_insufficient_ssd_quantity_triggers_repair_not_valid(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        ram_name="DDR5 RDIMM 32GB server memory module",
    )
    client = RecordingComposerClient(_multi_pass_insufficient_ssd_then_no_recommendation)

    result = _run_v2(db_session, client=client)

    assert result.match_result.final_status_source != "composer_validated"
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.report_fields["repair_composer_used"] is True
    insufficient = result.report_fields["code_completeness_result"][
        "insufficient_quantities"
    ]
    assert any(
        row["role"] == "ssd" and row["selected_quantity"] == 6
        for row in insufficient
    )


def test_v2_storage_nas_nvme_uncertain_no_recommendation_still_passes(
    db_session: Session,
) -> None:
    _seed_storage_catalog(db_session)
    client = RecordingComposerClient(
        _storage_nas_nvme_uncertain_no_recommendation_response,
        planner_responder=_storage_nas_planner_response,
    )

    result = _run_v2(
        db_session,
        spec=_storage_nas_spec(),
        client=client,
        settings=_composer_settings(max_package_chars=100000),
    )

    assert result.report_fields["product_group"] == "storage"
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.match_result.final_status_source == "composer_no_recommendation"
    assert result.match_result.no_recommendation_reason["structured_no_recommendation"]
    assert not result.report_fields.get("validation_repair_attempted")


def test_v2_storage_ready_server_source_type_normalizes_to_build(
    db_session: Session,
) -> None:
    _seed_storage_system_only_catalog(db_session)
    client = RecordingComposerClient(
        _storage_ready_server_source_type_response,
        planner_responder=_storage_nas_feature_planner_response,
    )

    result = _run_v2(
        db_session,
        spec=_storage_nas_spec(),
        client=client,
        settings=_composer_settings(max_package_chars=100000),
    )

    assert result.match_result.final_status_source == "composer_validated"
    assert result.match_result.primary_recommendation["source_type"] == (
        "build_from_parts"
    )
    assert "ready_server source_candidate_id required" not in json.dumps(
        result.match_result.rejected_ai_recommendations_debug_safe,
        ensure_ascii=False,
    )


def test_v2_storage_budget_constraint_is_commercial_not_hard_capability(
    db_session: Session,
) -> None:
    _seed_storage_system_only_catalog(db_session)
    client = RecordingComposerClient(
        _valid_storage_system_only_response,
        planner_responder=_storage_budget_constraint_planner_response,
    )

    result = _run_v2(
        db_session,
        spec=_storage_nas_spec(),
        client=client,
        settings=_composer_settings(max_package_chars=100000),
    )

    assert result.match_result.final_status_source == "composer_validated"
    logistics_rows = result.report_fields[
        "logistics_or_commercial_constraint_requirements"
    ]
    assert any("350" in row.get("source_text", "") for row in logistics_rows)
    assert not any(
        "350" in row.get("source_text", "")
        for row in result.report_fields["primary_object_feature_requirements"]
    )
    assert not any(
        "350" in row.get("source_text", "")
        for row in result.report_fields["validation_unverified_requirements"]
    )
    assert not any(
        "350" in row.get("source_text", "")
        for row in result.match_result.primary_recommendation.get(
            "missing_required_capabilities",
            [],
        )
    )


def test_v2_storage_missing_nvme_drive_is_not_ready_server_artifact(
    db_session: Session,
) -> None:
    _seed_storage_catalog(db_session)
    client = RecordingComposerClient(
        _multi_pass_storage_missing_nvme_ready_source_response,
        planner_responder=_storage_nas_planner_response,
    )

    result = _run_v2(
        db_session,
        spec=_storage_nas_spec(),
        client=client,
        settings=_composer_settings(max_package_chars=100000),
    )

    serialized_debug = json.dumps(
        result.match_result.rejected_ai_recommendations_debug_safe,
        ensure_ascii=False,
    )
    reason = result.match_result.no_recommendation_reason
    assert result.match_result.primary_recommendation_status == "no_recommendation"
    assert result.match_result.final_status_source in {
        "composer_no_recommendation",
        COMPOSER_REJECTED_BY_VALIDATION,
    }
    assert "ready_server source_candidate_id required" not in serialized_debug
    assert "drive" in json.dumps(reason, ensure_ascii=False)
    assert reason["structured_no_recommendation"] is True


def test_v2_server_read_timeout_reports_bounded_provider_timeout(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(
        SequencedComposerResponder(
            LlmReadTimeoutError("LLM request read timed out."),
        )
    )

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(max_package_chars=1_500_000),
    )

    policy = result.report_fields["package_candidate_exposure_policy"]
    assert result.match_result.final_status_source == "composer_provider_timeout"
    assert result.match_result.llm_fallback_reason == "composer_provider_timeout"
    assert result.report_fields["composer_timeout_fallback_attempted"] is True
    assert result.report_fields["composer_timeout_fallback_type"] == (
        "compact_full_matrix_retry"
    )
    assert result.report_fields["composer_timeout_fallback_success"] is False
    assert policy["timeout_fallback"]["silent_trimming"] is False
    assert result.report_fields["package_candidate_loss"] is False
    assert result.report_fields["package_candidate_exposure_incomplete"] is False
    assert result.report_fields["composer_execution_state"]["execution_failure"] is True


def test_v2_large_compact_timeout_uses_role_aware_reduced_fallback(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        cpu_count=130,
        ram_count=130,
        ssd_count=130,
    )
    client = RecordingComposerClient(
        SequencedComposerResponder(
            LlmReadTimeoutError("LLM request read timed out."),
            _valid_server_response,
        )
    )

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(max_package_chars=1_500_000),
    )

    fields = result.report_fields
    policy = fields["package_candidate_exposure_policy"]
    assert result.match_result.final_status_source == "composer_validated"
    assert fields["composer_timeout_fallback_attempted"] is True
    assert fields["composer_timeout_fallback_type"] == "role_aware_reduced_package"
    assert fields["composer_timeout_fallback_success"] is True
    assert fields["composer_timeout_fallback_reason"] == (
        "role_aware_reduced_package_succeeded"
    )
    assert policy["mode"] == "timeout_fallback_role_aware_reduced_matrix"
    assert policy["silent_trimming"] is False
    assert policy["timeout_fallback"]["attempted"] is True
    assert policy["timeout_fallback"]["silent_trimming"] is False
    assert policy["timeout_fallback"]["type"] == "role_aware_reduced_package"
    assert fields["original_candidate_count_by_role"]["cpu"] == 130
    assert fields["fallback_candidate_count_by_role"]["cpu"] < 130
    assert fields["dropped_before_fallback_count_by_role"]["cpu"] > 0
    assert fields["dropped_before_fallback_reasons"]["cpu"][0]["reason"] == (
        "timeout_fallback_bounded_role_aware_reduction"
    )
    assert 0 < fields["timeout_fallback_coverage_ratio_by_role"]["cpu"] < 1

    fallback_payload = client.packages[-1]
    assert fallback_payload["multi_pass_stage"] == "bom_composition"
    assert fallback_payload["v2_package_mode"] == "compact_full_matrix"
    matrix = fallback_payload["component_candidate_matrix"]
    for role in (
        "platform",
        "cpu",
        "ram",
        "ssd",
        "storage_controller",
        "network_adapter",
        "power_supply",
        "cable",
    ):
        assert matrix[role], role


def test_v2_timeout_reduced_fallback_invalid_bom_is_still_rejected(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        cpu_count=130,
        ram_count=130,
        ssd_count=130,
    )
    client = RecordingComposerClient(
        SequencedComposerResponder(
            LlmReadTimeoutError("LLM request read timed out."),
            _stock_shortage_response,
        )
    )

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(max_package_chars=1_500_000),
    )

    assert result.report_fields["composer_timeout_fallback_attempted"] is True
    assert result.report_fields["composer_timeout_fallback_type"] == (
        "role_aware_reduced_package"
    )
    assert result.report_fields["composer_timeout_fallback_success"] is True
    assert result.match_result.final_status_source == COMPOSER_REJECTED_BY_VALIDATION
    assert result.match_result.primary_recommendation_status == "no_recommendation"


def test_v2_timeout_reduced_fallback_second_timeout_stays_provider_timeout(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(
        db_session,
        cpu_count=130,
        ram_count=130,
        ssd_count=130,
    )
    client = RecordingComposerClient(
        SequencedComposerResponder(
            LlmReadTimeoutError("LLM request read timed out."),
            LlmReadTimeoutError("LLM fallback request read timed out."),
        )
    )

    result = _run_v2(
        db_session,
        client=client,
        settings=_composer_settings(max_package_chars=1_500_000),
    )

    assert result.match_result.final_status_source == "composer_provider_timeout"
    assert result.match_result.llm_fallback_reason == "composer_provider_timeout"
    assert result.report_fields["composer_timeout_fallback_attempted"] is True
    assert result.report_fields["composer_timeout_fallback_type"] == (
        "role_aware_reduced_package"
    )
    assert result.report_fields["composer_timeout_fallback_success"] is False
    assert result.report_fields["composer_timeout_fallback_reason"] == (
        "role_aware_reduced_package_read_timeout"
    )
    policy = result.report_fields["package_candidate_exposure_policy"]
    assert policy["timeout_fallback"]["silent_trimming"] is False


def test_v2_post_normalization_fix_has_no_case_specific_runtime_hardcode() -> None:
    runtime_source = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "app/llm/configuration_composer.py",
            "app/matching/ai_match_pipeline_v2.py",
            "app/matching/requirement_execution_contract.py",
        )
    )
    forbidden_fragments = (
        "switch-38c1b898fa01",
        "transceiver-23c8aa2e78c1",
        "rec-nas-nvme-001",
        "SW-48P-4SFP",
        "SFP-10G-SR",
        "NAS-1",
        "category_id == 'cat-",
        'category_id == "cat-',
    )

    assert not any(fragment in runtime_source for fragment in forbidden_fragments)


def test_v2_server_universe_excludes_global_network_device_roles(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    _seed_network_catalog(db_session)
    client = RecordingComposerClient(_valid_server_response)

    result = _run_v2(db_session, client=client, preview_only=True)

    category_plan = result.report_fields["category_plan"]
    assert {"server_platform", "cpu", "ram", "ssd", "network_adapter"}.issubset(
        category_plan
    )
    assert "switch" not in category_plan
    assert "router" not in category_plan
    assert "access_point" not in category_plan
    assert "switch" not in result.package["full_candidate_matrix_count_by_role"]
    assert "cable" in category_plan
    assert result.report_fields["primary_object"] == "server"


def test_v2_planner_repairs_invented_category_ids(db_session: Session) -> None:
    _seed_basic_server_catalog(db_session)
    planner = SequencedPlannerResponder(
        _server_planner_with_invented_category,
        _server_planner_response,
    )
    client = RecordingComposerClient(
        _valid_server_response,
        planner_responder=planner,
    )

    result = _run_v2(db_session, client=client, preview_only=True)

    assert client.planner_calls == 2
    assert result.report_fields["planner_repair_attempted"] is True
    assert result.report_fields["planner_repair_success"] is True
    assert "invented-category-id" not in json.dumps(
        result.report_fields["category_plan"],
        ensure_ascii=False,
    )
    assert result.report_fields["category_plan"]["server_platform"] == [
        "cat-platform"
    ]


def test_v2_suspicious_primary_group_mismatch_triggers_repair(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    _seed_network_catalog(db_session)
    planner = SequencedPlannerResponder(
        _network_planner_with_server_indicators,
        _server_planner_response,
    )
    client = RecordingComposerClient(
        _valid_server_response,
        planner_responder=planner,
    )

    result = _run_v2(db_session, client=client, preview_only=True)

    assert client.planner_calls == 2
    assert result.report_fields["product_group"] == "server"
    assert result.report_fields["planner_repair_attempted"] is True
    assert result.report_fields["planner_repair_success"] is True
    assert any(
        reason.startswith("suspicious_primary_group_mismatch")
        for reason in result.report_fields["planner_suspicion_reasons"]
    )


def test_v2_successful_llm_primary_group_not_overridden_by_spec_hint(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(
        _valid_server_response,
        planner_responder=_server_planner_response,
    )

    result = _run_v2(db_session, spec=_network_spec(), client=client, preview_only=True)

    assert result.report_fields["product_group"] == "server"
    assert result.report_fields["primary_object"] == "server"


def test_v2_diagnostic_fallback_does_not_choose_network_from_sfp_keywords(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)

    result = asyncio.run(
        run_ai_match_pipeline_v2(
            _server_spec(),
            AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            preview_only=True,
            llm_configurator_client=None,
            llm_settings=LlmSettings(llm_provider="disabled"),
        )
    )

    assert result.report_fields["product_group"] == "unknown"
    assert result.report_fields["candidate_universe_planner_mode"] == (
        "diagnostic_fallback"
    )
    assert "llm_provider_disabled" in result.report_fields[
        "candidate_universe_planner_output"
    ]["planner_warnings"]


def test_v2_candidate_universe_prompt_contract_has_guiding_examples(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    client = RecordingComposerClient(_valid_server_response)

    _run_v2(db_session, client=client, preview_only=True)

    planner_prompt = client.system_prompts[0]
    assert "Example" in planner_prompt
    assert "10GbE SFP+ X710" in planner_prompt
    assert '"category_plan_entries"' in planner_prompt
    assert '"primary_product_group"' in planner_prompt


def test_v2_category_planner_receives_full_compact_category_catalog(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    _seed_category(db_session, "cat-empty-support", "Empty Support Category")
    db_session.commit()
    client = RecordingComposerClient(_valid_server_response)

    result = _run_v2(db_session, client=client, preview_only=True)

    planner_catalog = client.planner_payloads[0]["category_catalog"]
    by_id = {row["category_id"]: row for row in planner_catalog}
    assert "cat-empty-support" in by_id
    assert by_id["cat-empty-support"]["product_count"] == 0
    assert by_id["cat-empty-support"]["stocked_count"] == 0
    assert by_id["cat-empty-support"]["priced_count"] == 0

    cpu_category = by_id["cat-cpu"]
    assert cpu_category["distributor"] == "ocs"
    assert cpu_category["category_kind"] == "component"
    assert cpu_category["allowed_roles"] == ["cpu"]
    assert cpu_category["stocked_count"] == 1
    assert cpu_category["priced_count"] == 1
    assert cpu_category["sample_producers"] == ["TestVendor"]
    assert cpu_category["sample_part_numbers"] == ["CPU-0"]
    assert "category_id_is_distributor_fact_not_business_policy" in cpu_category[
        "warnings"
    ]
    assert result.report_fields["category_catalog_total"] == len(planner_catalog)
    assert result.report_fields["category_catalog_sent_to_ai_count"] == len(
        planner_catalog
    )
    assert result.report_fields["category_catalog_truncated"] is False


def test_v2_invented_category_ids_are_rejected_with_diagnostics(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    planner = SequencedPlannerResponder(
        _server_planner_with_invented_category,
        _server_planner_response,
    )
    client = RecordingComposerClient(
        _valid_server_response,
        planner_responder=planner,
    )

    result = _run_v2(db_session, client=client, preview_only=True)

    rejected = result.report_fields["candidate_universe_planner_output"][
        "rejected_category_reasons"
    ]
    assert {
        "category_id": "invented-category-id",
        "role": "server_platform",
        "reason": "not_in_supplied_catalog",
        "detail": "AI selected a category_id absent from the supplied catalog.",
    } in rejected
    assert result.report_fields["rejected_category_count"] == 1
    assert result.report_fields["planner_repair_attempted"] is True


def test_v2_wrong_product_group_category_is_rejected_before_matrix(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    _seed_network_catalog(db_session)

    def planner(payload: Mapping[str, Any]) -> dict[str, Any]:
        response = _server_planner_response(payload)
        response["category_plan_entries"].append(
            {
                "role": "network_adapter",
                "selected_category_ids": ["cat-switch"],
                "purpose": "candidate_universe",
                "reason": "Bad planner tried to satisfy server NIC from switch category.",
                "confidence": "high",
            }
        )
        return response

    client = RecordingComposerClient(_valid_server_response, planner_responder=planner)

    result = _run_v2(db_session, client=client, preview_only=True)

    rejected = result.report_fields["rejected_category_reasons"]
    assert any(
        row["category_id"] == "cat-switch"
        and row["role"] == "network_adapter"
        and row["reason"] == "category_context_not_compatible_with_product_group"
        for row in rejected
    )
    assert "cat-switch" not in result.report_fields["category_plan"]["network_adapter"]
    assert "switch" not in result.package["full_candidate_matrix_count_by_role"]


def test_v2_matrix_keeps_uncertain_plausible_rows_before_composer(
    db_session: Session,
) -> None:
    _seed_basic_server_catalog(db_session)
    _seed_category(db_session, "cat-generic-options", "Generic Expansion Options")
    _seed_product(
        db_session,
        item_id="generic-option-1",
        category_id="cat-generic-options",
        category_name="Generic Expansion Options",
        part_number="GEN-OPTION-1",
        item_name="Generic option module with incomplete technical facts",
        quantity=3,
        price=Decimal("42"),
    )
    db_session.commit()

    def planner(payload: Mapping[str, Any]) -> dict[str, Any]:
        response = _server_planner_response(payload)
        response["category_plan_entries"].append(
            {
                "role": "other_accessory",
                "selected_category_ids": ["cat-generic-options"],
                "purpose": "candidate_universe",
                "reason": "Generic stocked option may be relevant; Composer must decide.",
                "confidence": "medium",
            }
        )
        return response

    client = RecordingComposerClient(_valid_server_response, planner_responder=planner)

    result = _run_v2(db_session, client=client, preview_only=True)

    rows = result.package["component_candidate_matrix"]["other_accessory"]
    generic_rows = [
        row for row in rows if row["part_number"] == "GEN-OPTION-1"
    ]
    assert len(generic_rows) == 1
    assert generic_rows[0]["fit_tier"] == "possible_fit"
    assert generic_rows[0]["selection_bucket"] == "v2_full_matrix"
    assert result.report_fields["matrix_materialized_count_by_role"][
        "other_accessory"
    ] >= 1


def _run_v2(
    db_session: Session,
    *,
    spec: StockSpec | None = None,
    client: RecordingComposerClient | None = None,
    preview_only: bool = False,
    settings: LlmSettings | None = None,
):
    effective_client = client or RecordingComposerClient(_valid_server_response)
    return asyncio.run(
        run_ai_match_pipeline_v2(
            spec or _server_spec(),
            AsyncSessionAdapter(db_session),  # type: ignore[arg-type]
            preview_only=preview_only,
            llm_configurator_client=effective_client,
            llm_settings=settings or _composer_settings(),
        )
    )


def _composer_settings(
    *,
    multi_pass: bool = False,
    multi_pass_chunk_size: int = 80,
    max_calls: int | None = None,
    max_package_chars: int | None = None,
) -> LlmSettings:
    call_budget = max_calls if max_calls is not None else (50 if multi_pass else 6)
    return LlmSettings(
        llm_provider="openai-compatible",
        llm_base_url="https://llm.example.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_configurator_enabled=True,
        llm_configurator_mode="composer",
        llm_configurator_output_mode="single_best_cost_valid",
        llm_configurator_repair_enabled=False,
        llm_composer_multi_pass=multi_pass,
        llm_composer_multi_pass_chunk_size=multi_pass_chunk_size,
        llm_max_calls_per_match=call_budget,
        **(
            {"llm_configurator_max_package_chars": max_package_chars}
            if max_package_chars is not None
            else {}
        ),
    )


def _server_spec() -> StockSpec:
    return StockSpec(
        source_text=SERVER_78_TEXT,
        items=[StockSpecItem(item_type="server", quantity=1, name="server")],
    )


def _network_spec() -> StockSpec:
    return StockSpec(
        source_text=(
            "Need network #79: one 48 port PoE switch with 4x10G SFP+ uplinks"
        ),
        items=[StockSpecItem(item_type="network", quantity=1, name="switch")],
    )


def _storage_nas_spec() -> StockSpec:
    return StockSpec(
        source_text="Need NAS 4TB on NVMe, preferably QNAP",
        items=[StockSpecItem(item_type="storage", quantity=1, name="NAS")],
    )


def _default_planner_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    text = str(payload.get("original_request_text") or "").casefold()
    if "switch" in text or "коммутатор" in text:
        return _network_planner_response(payload)
    if "storage array" in text or "usable capacity" in text or "nas" in text:
        return _storage_planner_response(payload)
    return _server_planner_response(payload)


def _requirement_contract_for_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    text = str(payload.get("original_request_text") or "").casefold()
    if "switch" in text:
        return {
            "primary_object": "switch",
            "required_roles": ["switch"],
            "required_quantities_by_role": {"switch": {"count": 1}},
            "hard_requirements": ["48 port PoE switch with SFP+ uplinks"],
            "optional_requirements": [],
            "primary_object_features": ["48 port PoE", "4x10G SFP+"],
            "purchasable_component_roles": ["switch"],
            "accessories": [],
            "services_support": [],
            "logistics_commercial_constraints": [],
            "fulfillment_expectations": [],
            "engineer_checks": ["Confirm switch optics/cables before quote."],
        }
    if "nas" in text or "storage array" in text:
        return _storage_nas_requirement_contract()
    return _server_78_requirement_contract()


def _server_planner_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    categories = _catalog_by_name(payload)
    return {
        "primary_product_group": "server",
        "primary_object": "server",
        "confidence": "high",
        "procurement_intent": "Build one rack server BOM from platform and components.",
        "selected_group_reason": (
            "1U chassis, sockets, CPU, RAM, SSD, storage controller, PSU and cooling "
            "make the procurement object a server BOM."
        ),
        "competing_product_groups": [
            {
                "product_group": "network",
                "reason": "Ethernet, SFP+ and X710 are present.",
                "why_not_primary": "They describe the requested NIC role inside the server BOM.",
            }
        ],
        "primary_object_indicators": ["1U", "Sockets: 2", "CPU", "RAM DDR5", "SSD"],
        "component_role_indicators": [
            {
                "role": "server_platform",
                "source_text": "Execution: 1U",
                "reason": "server chassis",
            },
            {
                "role": "cpu",
                "source_text": "CPU: Intel 6th generation, 2 pcs",
                "reason": "server CPUs",
            },
            {"role": "ram", "source_text": "RAM: 256 GB DDR5 RDIMM", "reason": "server memory"},
            {"role": "ssd", "source_text": "SSD SATA", "reason": "server drives"},
            {
                "role": "storage_controller",
                "source_text": "LSI 9400-8i / 9500-8i",
                "reason": "storage controller",
            },
            {
                "role": "network_adapter",
                "source_text": "Intel X710-DA2 2x10GbE SFP+",
                "reason": "NIC inside server BOM",
            },
            {"role": "power_supply", "source_text": "2 x 2000W PSU", "reason": "server power"},
        ],
        "accessory_indicators": ["C13-C14 cables", "cooling fans"],
        "service_support_indicators": [],
        "logistics_commercial_constraints": ["Moscow stock if provided"],
        "broad_role_hints": [
            "server_platform",
            "cpu",
            "ram",
            "ssd",
            "storage_controller",
            "network_adapter",
            "power_supply",
            "cable",
            "other_accessory",
        ],
        "category_plan_entries": [
            _planner_entry("server_platform", categories, "platform"),
            _planner_entry("cpu", categories, "cpu"),
            _planner_entry("ram", categories, "memory"),
            _planner_entry("ssd", categories, "ssd"),
            _planner_entry("storage_controller", categories, "controller"),
            _planner_entry("network_adapter", categories, "adapter"),
            _planner_entry("power_supply", categories, "power supply"),
            _planner_entry("cable", categories, "cable"),
            _planner_entry("other_accessory", categories, "accessor"),
        ],
        "excluded_category_groups": [
            {
                "category_id_or_group": "switch/router/access_point",
                "reason": "not primary network procurement",
            }
        ],
        "needs_repair": False,
    }


def _server_planner_with_embedded_requirements(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    response = _server_planner_response(payload)
    response["embedded_requirements"] = [
        {
            "requirement_text": "NVMe-capable front storage",
            "role": "ssd",
            "classification": "primary_object_feature",
            "fulfillment_mode": "included_in_primary_object",
            "hardness": "hard",
        }
    ]
    response["requirement_fulfillment_decision"] = [
        {
            "requirement_text": "Intel X710-DA2 2x10GbE SFP+",
            "role": "network_adapter",
            "fulfillment_mode": "separate_component_required",
            "hardness": "hard",
        }
    ]
    return response


def _server_planner_with_drive_requirement_ssd_category(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    response = _server_planner_response(payload)
    response["component_role_indicators"] = [
        (
            {
                **row,
                "role": "drive",
                "reason": "generic disk requirement fulfilled by SSD candidates",
            }
            if row.get("role") == "ssd"
            else row
        )
        for row in response["component_role_indicators"]
    ]
    response["broad_role_hints"] = [
        "drive" if role == "ssd" else role
        for role in response["broad_role_hints"]
    ]
    return response


def _server_planner_with_invented_category(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    response = _server_planner_response(payload)
    response["category_plan_entries"][0] = {
        **response["category_plan_entries"][0],
        "selected_category_ids": ["invented-category-id"],
    }
    return response


def _network_planner_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    categories = _catalog_by_name(payload)
    return {
        "primary_product_group": "network",
        "primary_object": "switch",
        "confidence": "high",
        "procurement_intent": "Buy one access switch.",
        "selected_group_reason": "48 ports, PoE, L3 and stacking are switch requirements.",
        "competing_product_groups": [],
        "primary_object_indicators": ["48 port PoE switch", "L3", "stacking"],
        "component_role_indicators": [
            {
                "role": "switch",
                "source_text": "48 port PoE switch",
                "reason": "primary network device",
            }
        ],
        "accessory_indicators": ["SFP+ uplinks if optics or DACs are needed"],
        "service_support_indicators": [],
        "logistics_commercial_constraints": ["Moscow stock if provided"],
        "broad_role_hints": ["switch"],
        "category_plan_entries": [_planner_entry("switch", categories, "switch")],
        "excluded_category_groups": [
            {"category_id_or_group": "server/storage", "reason": "not part of switch procurement"}
        ],
        "needs_repair": False,
    }


def _network_planner_with_optional_uplink_accessories(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    response = _network_planner_response(payload)
    response["component_role_indicators"].extend(
        [
            {
                "role": "transceiver",
                "source_text": "4x10G SFP+ uplinks",
                "classification": "accessory_or_consumable",
                "fulfillment_mode": "separate_component_required",
                "hardness": "hard",
                "reason": (
                    "Uplink media may need optics, but no separate optics line "
                    "was requested."
                ),
            },
            {
                "role": "dac_cable",
                "source_text": "4x10G SFP+ uplinks",
                "classification": "accessory_or_consumable",
                "fulfillment_mode": "separate_component_required",
                "hardness": "hard",
                "reason": "Uplink media may need DAC, but no separate DAC line was requested.",
            },
            {
                "role": "stacking_module",
                "source_text": "stacking desirable",
                "classification": "engineering_check",
                "fulfillment_mode": "engineering_check_only",
                "hardness": "optional",
                "reason": "Stacking is desirable and should be checked on the switch.",
            },
        ]
    )
    response["embedded_requirements"] = [
        {
            "source_text": "4x10G SFP+ uplinks",
            "role": "switch",
            "classification": "primary_object_feature",
            "fulfillment_mode": "included_in_primary_object",
            "hardness": "hard",
        },
        {
            "source_text": "stacking desirable",
            "role": "switch",
            "classification": "engineering_check",
            "fulfillment_mode": "engineering_check_only",
            "hardness": "optional",
        },
    ]
    response["broad_role_hints"] = [
        "switch",
        "transceiver",
        "dac_cable",
        "stacking_module",
    ]
    return response


def _network_planner_with_optional_transceiver_category(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    response = _network_planner_with_optional_uplink_accessories(payload)
    categories = _catalog_by_name(payload)
    response["category_plan_entries"].append(
        _planner_entry("transceiver", categories, "transceiver")
    )
    return response


def _storage_budget_constraint_planner_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    response = _storage_nas_feature_planner_response(payload)
    budget_row = {
        "source_text": "budget 350k",
        "role": "storage_system",
        "requirement_classification": "commercial_constraint",
        "fulfillment_mode": "included_in_primary_object",
        "hardness": "hard",
        "reason": "Commercial budget ceiling for quote filtering.",
    }
    response["embedded_requirements"].append(budget_row)
    response["requirement_fulfillment_decision"].append(budget_row)
    response["logistics_commercial_constraints"] = ["budget 350k"]
    return response


def _network_planner_with_server_indicators(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    response = _network_planner_response(payload)
    response.update(
        {
            "primary_product_group": "network",
            "primary_object": "network_device",
            "selected_group_reason": "Previous planner overfocused on Ethernet/SFP+.",
            "primary_object_indicators": [
                "Execution: 1U",
                "Sockets: 2",
                "CPU Intel",
                "RAM DDR5",
            ],
            "component_role_indicators": [
                {"role": "cpu", "source_text": "CPU Intel", "reason": "server CPU"},
                {"role": "ram", "source_text": "RAM DDR5", "reason": "server RAM"},
                {
                    "role": "network_adapter",
                    "source_text": "Intel X710-DA2 2x10GbE SFP+",
                    "reason": "server NIC",
                },
            ],
        }
    )
    return response


def _storage_planner_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    categories = _catalog_by_name(payload)
    return {
        "primary_product_group": "storage",
        "primary_object": "storage_system",
        "confidence": "high",
        "procurement_intent": "Buy a storage array.",
        "selected_group_reason": "Storage array, usable capacity and controllers indicate storage.",
        "competing_product_groups": [],
        "primary_object_indicators": ["storage array", "usable capacity"],
        "component_role_indicators": [
            {"role": "storage_system", "source_text": "storage array", "reason": "primary object"}
        ],
        "accessory_indicators": [],
        "service_support_indicators": [],
        "logistics_commercial_constraints": [],
        "broad_role_hints": ["storage_system"],
        "category_plan_entries": [_planner_entry("storage_system", categories, "storage")],
        "excluded_category_groups": [],
        "needs_repair": False,
    }


def _storage_nas_planner_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    categories = _catalog_by_name(payload)
    return {
        "primary_product_group": "storage",
        "primary_object": "storage_system",
        "confidence": "high",
        "procurement_intent": "Buy a NAS/storage system with drives.",
        "selected_group_reason": "NAS primary object plus capacity/media requirements.",
        "competing_product_groups": [],
        "primary_object_indicators": ["NAS", "4TB"],
        "component_role_indicators": [
            {
                "role": "storage_system",
                "source_text": "NAS",
                "reason": "primary storage system",
                "hardness": "hard",
            },
            {
                "role": "drive",
                "source_text": "4TB on NVMe",
                "reason": "capacity/media may require purchasable drives",
                "hardness": "hard",
            },
        ],
        "embedded_requirements": [
            {
                "requirement_text": "4TB on NVMe",
                "role": "drive",
                "classification": "purchasable_component_role",
                "fulfillment_mode": "separate_component_required",
                "hardness": "hard",
            }
        ],
        "requirement_fulfillment_decision": [
            {
                "requirement_text": "4TB on NVMe",
                "role": "drive",
                "fulfillment_mode": "separate_component_required",
                "hardness": "hard",
            }
        ],
        "accessory_indicators": [],
        "service_support_indicators": [],
        "logistics_commercial_constraints": [],
        "broad_role_hints": ["storage_system", "drive"],
        "category_plan_entries": [
            _planner_entry("storage_system", categories, "storage system"),
            _planner_entry("drive", categories, "drive"),
        ],
        "excluded_category_groups": [],
        "needs_repair": False,
    }


def _storage_nas_planner_with_optional_vendor(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    response = _storage_nas_planner_response(payload)
    response["component_role_indicators"].append(
        {
            "role": "vendor",
            "source_text": "preferably QNAP",
            "reason": "optional brand preference",
            "hardness": "optional",
            "fulfillment_mode": "optional_preference",
        }
    )
    response["embedded_requirements"].append(
        {
            "requirement_text": "preferably QNAP",
            "role": "vendor",
            "classification": "optional_preference",
            "fulfillment_mode": "optional_preference",
            "hardness": "optional",
        }
    )
    return response


def _storage_nas_feature_planner_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    categories = _catalog_by_name(payload)
    return {
        "primary_product_group": "storage",
        "primary_object": "storage_system",
        "confidence": "high",
        "procurement_intent": "Buy a NAS storage system with included capacity.",
        "selected_group_reason": "NAS is the primary storage object.",
        "competing_product_groups": [],
        "primary_object_indicators": ["NAS", "4TB on NVMe"],
        "component_role_indicators": [
            {
                "role": "storage_system",
                "source_text": "NAS",
                "reason": "primary storage system",
                "hardness": "hard",
            }
        ],
        "embedded_requirements": [
            {
                "source_text": "4TB on NVMe",
                "role": "drive",
                "classification": "primary_object_feature",
                "fulfillment_mode": "included_in_primary_object",
                "hardness": "hard",
            },
            {
                "source_text": "preferably QNAP",
                "role": "vendor",
                "classification": "optional_preference",
                "fulfillment_mode": "optional_preference",
                "hardness": "optional",
            },
        ],
        "requirement_fulfillment_decision": [
            {
                "source_text": "4TB on NVMe",
                "role": "drive",
                "fulfillment_mode": "included_in_primary_object",
                "hardness": "hard",
            }
        ],
        "accessory_indicators": [],
        "service_support_indicators": [],
        "logistics_commercial_constraints": ["Budget is a commercial filter."],
        "broad_role_hints": ["storage_system", "drive"],
        "category_plan_entries": [
            _planner_entry("storage_system", categories, "storage system")
        ],
        "excluded_category_groups": [],
        "needs_repair": False,
    }


def _storage_drive_alias_planner_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    categories = _catalog_by_name(payload)
    return {
        "primary_product_group": "storage",
        "primary_object": "storage_system",
        "confidence": "high",
        "procurement_intent": "Buy a storage system with separate SSD media.",
        "selected_group_reason": "Storage system plus explicit SSD drive line.",
        "competing_product_groups": [],
        "primary_object_indicators": ["storage system"],
        "component_role_indicators": [
            {
                "role": "storage_system",
                "source_text": "storage system",
                "reason": "primary storage system",
                "hardness": "hard",
            },
            {
                "role": "drive",
                "source_text": "2 x SSD drives",
                "classification": "purchasable_component_role",
                "fulfillment_mode": "separate_component_required",
                "reason": "explicit drive line can be fulfilled by SSD role candidates",
                "hardness": "hard",
            },
        ],
        "embedded_requirements": [],
        "requirement_fulfillment_decision": [
            {
                "role": "drive",
                "source_text": "2 x SSD drives",
                "fulfillment_mode": "separate_component_required",
                "hardness": "hard",
            }
        ],
        "accessory_indicators": [],
        "service_support_indicators": [],
        "logistics_commercial_constraints": [],
        "broad_role_hints": ["storage_system", "drive"],
        "category_plan_entries": [
            _planner_entry("storage_system", categories, "storage system"),
            _planner_entry("ssd", categories, "ssd"),
        ],
        "excluded_category_groups": [],
        "needs_repair": False,
    }


def _catalog_by_name(payload: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in payload.get("category_catalog") or []:
        if not isinstance(row, Mapping):
            continue
        category_id = str(row.get("category_id") or "")
        text = " ".join(
            str(part)
            for part in (
                row.get("category_name"),
                " ".join(str(item) for item in row.get("category_path") or []),
                " ".join(str(item) for item in row.get("sample_product_names") or []),
            )
            if part
        ).casefold()
        result[category_id] = text
    return result


def _planner_entry(
    role: str,
    categories: Mapping[str, str],
    needle: str,
) -> dict[str, Any]:
    category_ids = [
        category_id
        for category_id, text in categories.items()
        if needle.casefold() in text
    ][:2]
    return {
        "role": role,
        "selected_category_ids": category_ids,
        "purpose": "candidate_universe",
        "reason": f"Catalog text matches {role}.",
        "confidence": "high" if category_ids else "low",
    }


def _valid_server_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "unverified_requirements": [],
        },
        "recommendations": [
            {
                "recommendation_id": "v2_valid_server",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "platform": matrix["platform"][0]["component_candidate_id"],
                    "cpu": matrix["cpu"][0]["component_candidate_id"],
                    "ram": matrix["ram"][0]["component_candidate_id"],
                    "storage": matrix["ssd"][0]["component_candidate_id"],
                    "storage_controller": matrix["storage_controller"][0][
                        "component_candidate_id"
                    ],
                    "network_adapter": matrix["network_adapter"][0][
                        "component_candidate_id"
                    ],
                    "power_supply": matrix["power_supply"][0]["component_candidate_id"],
                    "cable": matrix["cable"][0]["component_candidate_id"],
                },
                "quantities": {
                    "platform": 1,
                    "cpu": 2,
                    "ram": 8,
                    "storage": 8,
                    "storage_controller": 1,
                    "network_adapter": 1,
                    "power_supply": 2,
                    "cable": 2,
                },
                "decision": "recommend",
                "title": "V2 valid server BOM",
                "why_selected": "Cheapest safe v2 BOM from the full matrix.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _invented_id_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "invented",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "platform": "invented-platform-id",
                    "cpu": matrix["cpu"][0]["component_candidate_id"],
                },
                "quantities": {"platform": 1, "cpu": 1},
                "decision": "recommend",
                "title": "Invented BOM",
                "why_selected": "Should be rejected.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _stock_shortage_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "stock_shortage",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "platform": matrix["platform"][0]["component_candidate_id"],
                    "cpu": matrix["cpu"][0]["component_candidate_id"],
                    "ram": matrix["ram"][0]["component_candidate_id"],
                    "storage": matrix["ssd"][0]["component_candidate_id"],
                    "storage_controller": matrix["storage_controller"][0][
                        "component_candidate_id"
                    ],
                    "network_adapter": matrix["network_adapter"][0][
                        "component_candidate_id"
                    ],
                    "power_supply": matrix["power_supply"][0]["component_candidate_id"],
                    "cable": matrix["cable"][0]["component_candidate_id"],
                },
                "quantities": {
                    "platform": 1,
                    "cpu": 2,
                    "ram": 99,
                    "storage": 8,
                    "storage_controller": 1,
                    "network_adapter": 1,
                    "power_supply": 2,
                    "cable": 2,
                },
                "decision": "recommend",
                "title": "Stock shortage BOM",
                "why_selected": "Should be rejected.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _server_with_overstated_optional_accessory_response(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    response = _valid_server_response(package)
    recommendation = dict(response["recommendations"][0])
    component_ids = dict(recommendation["component_candidate_ids"])
    component_ids["other_accessory"] = package["component_candidate_matrix"][
        "other_accessory"
    ][0]["component_candidate_id"]
    quantities = dict(recommendation["quantities"])
    quantities["other_accessory"] = 8
    recommendation["component_candidate_ids"] = component_ids
    recommendation["quantities"] = quantities
    response["recommendations"] = [recommendation]
    return response


def _platform_cpu_mismatch_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "mismatch",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "platform": matrix["platform"][0]["component_candidate_id"],
                    "cpu": matrix["cpu"][0]["component_candidate_id"],
                    "ram": matrix["ram"][0]["component_candidate_id"],
                    "storage": matrix["ssd"][0]["component_candidate_id"],
                    "storage_controller": matrix["storage_controller"][0][
                        "component_candidate_id"
                    ],
                    "network_adapter": matrix["network_adapter"][0][
                        "component_candidate_id"
                    ],
                    "power_supply": matrix["power_supply"][0]["component_candidate_id"],
                    "cable": matrix["cable"][0]["component_candidate_id"],
                },
                "quantities": {
                    "platform": 1,
                    "cpu": 2,
                    "ram": 8,
                    "storage": 8,
                    "storage_controller": 1,
                    "network_adapter": 1,
                    "power_supply": 2,
                    "cable": 2,
                },
                "decision": "recommend",
                "title": "Mismatched BOM",
                "why_selected": "Should be rejected.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _platform_cpu_compatible_repair_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    response = _valid_server_response(package)
    recommendation = dict(response["recommendations"][0])
    component_ids = dict(recommendation["component_candidate_ids"])
    component_ids["cpu"] = next(
        row["component_candidate_id"]
        for row in matrix["cpu"]
        if row["part_number"] == "CPU-4710"
    )
    recommendation["recommendation_id"] = "validation_repaired_platform_cpu"
    recommendation["component_candidate_ids"] = component_ids
    recommendation["why_selected"] = (
        "Validation repair replaced the rejected CPU with a socket-compatible CPU."
    )
    response["recommendations"] = [recommendation]
    return response


def _validation_repair_no_recommendation_response(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    considered = {
        role: [row["component_candidate_id"] for row in rows]
        for role, rows in package["component_candidate_matrix"].items()
        if isinstance(rows, list)
    }
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "unverified_requirements": [
                {
                    "role": "cpu",
                    "source_text": "No compatible platform/CPU pair after validation.",
                }
            ],
        },
        "recommendations": [],
        "no_recommendation": {
            "summary": "No compatible platform/CPU pair after validator rejection.",
            "missing_roles": [],
            "missing_required_capabilities": [],
            "hard_mismatches": [
                {
                    "role": "cpu",
                    "reason": "Validator rejected the available platform/CPU socket pair.",
                }
            ],
            "stock_shortages": [],
            "role_analysis": [
                {
                    "role": role,
                    "considered_candidate_ids": ids,
                    "decision": "no_safe_choice" if role in {"platform", "cpu"} else "available",
                }
                for role, ids in considered.items()
            ],
            "considered_candidate_ids": considered,
            "explanation_ru": "No safe compatible platform/CPU pair was found.",
            "recommended_next_actions": ["Ask an engineer for a compatible platform/CPU pair."],
        },
        "general_notes": [],
    }


def _validation_repair_empty_response(package: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [],
        "no_recommendation": None,
        "general_notes": [],
    }


def _validation_repair_general_notes_only_response(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [],
        "no_recommendation": None,
        "general_notes": [
            "BOM assembled with a total estimated value, but selected components were omitted."
        ],
    }


def _structured_no_recommendation_response(package: Mapping[str, Any]) -> dict[str, Any]:
    considered = {
        role: [row["component_candidate_id"] for row in rows]
        for role, rows in package["component_candidate_matrix"].items()
        if isinstance(rows, list)
    }
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "unverified_requirements": [{"source_text": "No safe config"}],
        },
        "recommendations": [],
        "no_recommendation": {
            "summary": "No safe v2 configuration can be built.",
            "missing_roles": [],
            "missing_required_capabilities": [],
            "hard_mismatches": [
                {"role": "server_platform", "reason": "requires engineer check"}
            ],
            "stock_shortages": [],
            "role_analysis": [
                {
                    "role": role,
                    "considered_candidate_ids": ids,
                    "decision": "no_safe_choice",
                }
                for role, ids in considered.items()
            ],
            "considered_candidate_ids": considered,
            "explanation_ru": "Нет безопасной конфигурации.",
            "recommended_next_actions": ["Escalate to engineer."],
        },
        "general_notes": [],
    }


def _storage_nas_nvme_uncertain_no_recommendation_response(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    considered = {
        role: [row["component_candidate_id"] for row in rows]
        for role, rows in package["component_candidate_matrix"].items()
        if isinstance(rows, list)
    }
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "unverified_requirements": [
                {"role": "drive", "source_text": "NVMe SSD compatibility not proven"}
            ],
        },
        "recommendations": [],
        "no_recommendation": {
            "summary": "No safe NAS BOM because NVMe SSD support is not proven.",
            "missing_roles": [],
            "missing_required_capabilities": [
                {
                    "role": "drive",
                    "reason": "NVMe SSD support was not verified from provided facts.",
                }
            ],
            "hard_mismatches": [],
            "stock_shortages": [],
            "role_analysis": [
                {
                    "role": role,
                    "considered_candidate_ids": ids,
                    "decision": "no_safe_choice" if role == "drive" else "available",
                }
                for role, ids in considered.items()
            ],
            "considered_candidate_ids": considered,
            "explanation_ru": "NVMe SSD РЅРµ РґРѕРєР°Р·Р°РЅС‹ РґР»СЏ NAS.",
            "recommended_next_actions": ["Escalate to storage engineer."],
        },
        "general_notes": [],
    }


def _valid_network_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "network_79",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "switch": matrix["switch"][0]["component_candidate_id"],
                },
                "quantities": {"switch": 1},
                "decision": "recommend",
                "title": "Network switch",
                "why_selected": "Cheapest matching switch.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _network_missing_switch_no_recommendation_response(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    considered = [
        row["component_candidate_id"]
        for row in package["component_candidate_matrix"]["switch"]
    ]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [],
        "no_recommendation": {
            "summary": "Composer returned an incomplete switch BOM.",
            "missing_roles": ["switch"],
            "missing_required_capabilities": [],
            "hard_mismatches": [],
            "stock_shortages": [],
            "role_analysis": [
                {
                    "role": "switch",
                    "status": "missing",
                    "considered_candidate_ids": considered,
                }
            ],
            "considered_candidate_ids": {"switch": considered},
            "recommended_next_actions": ["Escalate to engineer."],
        },
        "general_notes": [],
    }


def _network_string_requirement_summary_response(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "network_79_strings",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "switch": matrix["switch"][0]["component_candidate_id"],
                },
                "optional_component_candidate_ids": {
                    "transceiver": matrix["transceiver"][0][
                        "component_candidate_id"
                    ],
                },
                "quantities": {"switch": 1, "transceiver": 4},
                "decision": "recommend",
                "title": "Network switch with optional optics",
                "why_selected": "Cheapest matching switch; optics are optional.",
                "requirement_fulfillment_summary": [
                    "48 ports and PoE are covered by selected switch.",
                    "4x10G SFP+ uplinks need engineer confirmation for optics.",
                ],
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _valid_storage_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [
                {
                    "role": "drive",
                    "fulfillment_mode": "separate_component_required",
                    "requirement_text": "4TB on NVMe",
                }
            ],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "storage_nas",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "storage_system": matrix["storage_system"][0][
                        "component_candidate_id"
                    ],
                    "drive": matrix["drive"][0]["component_candidate_id"],
                },
                "quantities": {"storage_system": 1, "drive": 2},
                "decision": "recommend",
                "title": "Storage NAS",
                "why_selected": "Cheapest NAS with separate NVMe drive candidates.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _valid_storage_system_only_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [
                {
                    "role": "drive",
                    "fulfillment_mode": "included_in_primary_object",
                    "requirement_text": "4TB on NVMe",
                }
            ],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "storage_nas_system_only",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "storage_system": matrix["storage_system"][0][
                        "component_candidate_id"
                    ],
                },
                "quantities": {"storage_system": 1},
                "decision": "recommend",
                "title": "Storage NAS",
                "why_selected": "NAS capacity is treated as a primary object feature.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _storage_ready_server_source_type_response(
    package: Mapping[str, Any],
) -> dict[str, Any]:
    result = _valid_storage_system_only_response(package)
    recommendation = dict(result["recommendations"][0])
    recommendation["source_type"] = "ready_server"
    result["recommendations"] = [recommendation]
    return result


def _multi_pass_storage_missing_nvme_ready_source_response(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _storage_nas_requirement_contract()
    if stage == "bom_composition":
        matrix = payload["candidate_facts_by_role"]
        return {
            "requirement_analysis": {
                "fulfillment_decisions": [
                    {
                        "role": "drive",
                        "fulfillment_mode": "separate_component_required",
                        "requirement_text": "4TB on NVMe",
                    }
                ],
                "requirement_contract_used": True,
            },
            "recommendations": [
                {
                    "recommendation_id": "nas_missing_nvme",
                    "proposal_role": "cheapest_fit",
                    "source_type": "ready_server",
                    "component_candidate_ids": {
                        "storage_system": matrix["storage_system"][0][
                            "component_candidate_id"
                        ],
                    },
                    "quantities": {"storage_system": 1},
                    "decision": "recommend",
                    "title": "NAS without selected NVMe drives",
                    "why_selected": "Intentionally omits required drive role.",
                    "confidence": "medium",
                }
            ],
            "no_recommendation": None,
            "general_notes": [],
        }
    if stage == "completeness_critic":
        return {
            "all_hard_requirements_covered": False,
            "missing_roles": ["drive"],
            "insufficient_quantities": [],
            "unverified_requirements": [
                {"role": "drive", "reason": "NVMe drives are not selected."}
            ],
            "hard_mismatch_risks": [],
            "recommended_repair_actions": ["Return structured no_recommendation."],
        }
    if stage == "repair":
        return {
            "recommendations": [],
            "no_recommendation": {
                "summary": "No safe NAS BOM because required NVMe drives are missing.",
                "missing_roles": ["drive"],
                "missing_required_capabilities": [
                    {
                        "role": "drive",
                        "reason": "Required NVMe drive role was not selected.",
                    }
                ],
                "hard_mismatches": [],
                "stock_shortages": [],
                "role_analysis": [
                    {
                        "role": "drive",
                        "decision": "missing",
                        "explanation": "NVMe drives are required for this NAS request.",
                    }
                ],
                "recommended_next_actions": ["Select compatible NVMe drives."],
            },
            "general_notes": [],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _valid_storage_system_with_ssd_response(package: Mapping[str, Any]) -> dict[str, Any]:
    matrix = package["component_candidate_matrix"]
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [
                {
                    "role": "drive",
                    "fulfilled_by_role": "ssd",
                    "fulfillment_mode": "separate_component_required",
                }
            ],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "storage_drive_alias",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "storage_system": matrix["storage_system"][0][
                        "component_candidate_id"
                    ],
                    "ssd": matrix["ssd"][0]["component_candidate_id"],
                },
                "quantities": {"storage_system": 1, "ssd": 2},
                "decision": "recommend",
                "title": "Storage with SSD alias",
                "why_selected": "Drive requirement is fulfilled by SSD role candidates.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _multi_pass_complete_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _server_78_requirement_contract()
    if stage == "role_evaluation":
        return _role_evaluation_response(payload)
    if stage == "bom_composition":
        return _complete_multi_pass_bom(payload)
    if stage == "completeness_critic":
        return {
            "all_hard_requirements_covered": True,
            "missing_roles": [],
            "insufficient_quantities": [],
            "unverified_requirements": [],
            "hard_mismatch_risks": [],
            "recommended_repair_actions": [],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_invalid_requirement_contract_then_valid_bom(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return {
            "primary_object": "server",
            "required_roles": {"invalid": "shape"},
            "required_quantities_by_role": {},
        }
    if stage == "bom_composition":
        return _complete_multi_pass_bom(payload)
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_schema_invalid_bom(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _requirement_contract_for_payload(payload)
    if stage == "bom_composition":
        return {
            "recommendations": [
                {
                    "recommendation_id": "schema_invalid",
                    "proposal_role": ["not", "a", "string"],
                    "source_type": "build_from_parts",
                    "component_candidate_ids": {},
                    "decision": "recommend",
                    "title": "Schema invalid BOM",
                    "why_selected": "Should be classified as schema failure.",
                    "confidence": "medium",
                }
            ],
            "no_recommendation": None,
            "general_notes": [],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_selected_components_alias_bom(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _requirement_contract_for_payload(payload)
    if stage == "bom_composition":
        candidates = payload["candidate_facts_by_role"]
        return {
            "requirement_analysis": {
                "fulfillment_decisions": [],
                "requirement_contract_used": True,
            },
            "recommendations": [
                {
                    "recommendation_id": "selected_components_alias",
                    "proposal_role": "cheapest_fit",
                    "candidate_type": "build_from_parts",
                    "selected_components": [
                        {
                            "role": "platform",
                            "component_candidate_id": _first_candidate_id(
                                candidates, "server_platform"
                            ),
                            "quantity": 1,
                        },
                        {
                            "role": "cpu",
                            "component_candidate_id": _first_candidate_id(
                                candidates, "cpu"
                            ),
                            "quantity": 2,
                        },
                        {
                            "role": "ram",
                            "component_candidate_id": _first_candidate_id(
                                candidates, "ram"
                            ),
                            "quantity": 8,
                        },
                        {
                            "role": "ssd",
                            "component_candidate_id": _first_candidate_id(
                                candidates, "ssd"
                            ),
                            "quantity": 8,
                        },
                        {
                            "role": "storage_controller",
                            "component_candidate_id": _first_candidate_id(
                                candidates, "storage_controller"
                            ),
                            "quantity": 1,
                        },
                        {
                            "role": "network_adapter",
                            "component_candidate_id": _first_candidate_id(
                                candidates, "network_adapter"
                            ),
                            "quantity": 1,
                        },
                        {
                            "role": "power_supply",
                            "component_candidate_id": _first_candidate_id(
                                candidates, "power_supply"
                            ),
                            "quantity": 2,
                        },
                        {
                            "role": "cable",
                            "component_candidate_id": _first_candidate_id(
                                candidates, "cable"
                            ),
                            "quantity": 2,
                        },
                    ],
                    "general_notes": ["temporary note from model"],
                    "tradeoffs": ["uses cheapest available platform"],
                    "unverified_requirements": [
                        {"role": "cable", "source_text": "confirm exact cable types"}
                    ],
                    "decision": "recommend",
                    "title": "Selected components alias BOM",
                    "why_selected": "Complete BOM assembled from selected_components.",
                    "confidence": "medium",
                }
            ],
            "no_recommendation": None,
            "general_notes": [],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_component_map_without_source_type(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _requirement_contract_for_payload(payload)
    if stage == "bom_composition":
        bom = _complete_multi_pass_bom(payload)
        recommendation = dict(bom["recommendations"][0])
        recommendation.pop("source_type")
        recommendation.pop("candidate_type", None)
        bom["recommendations"] = [recommendation]
        return bom
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_role_specific_ssd_bom(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _requirement_contract_for_payload(payload)
    if stage == "bom_composition":
        candidates = payload["candidate_facts_by_role"]
        ssd_ids = [
            row["component_candidate_id"]
            for row in candidates["ssd"]
            if isinstance(row, Mapping)
        ]
        assert len(ssd_ids) >= 2
        return {
            "requirement_analysis": {
                "fulfillment_decisions": [],
                "requirement_contract_used": True,
            },
            "recommendations": [
                {
                    "recommendation_id": "role_specific_ssd",
                    "proposal_role": "cheapest_fit",
                    "source_type": "build_from_parts",
                    "component_candidate_ids": {
                        "platform": _first_candidate_id(candidates, "server_platform"),
                        "cpu": _first_candidate_id(candidates, "cpu"),
                        "ram": _first_candidate_id(candidates, "ram"),
                        "ssd_1920": ssd_ids[0],
                        "ssd_480": ssd_ids[1],
                        "storage_controller": _first_candidate_id(
                            candidates, "storage_controller"
                        ),
                        "network_adapter": _first_candidate_id(
                            candidates, "network_adapter"
                        ),
                        "power_supply": _first_candidate_id(candidates, "power_supply"),
                        "cable": _first_candidate_id(candidates, "cable"),
                    },
                    "quantities": {
                        "platform": 1,
                        "cpu": 2,
                        "ram": 8,
                        "ssd_1920": 6,
                        "ssd_480": 2,
                        "storage_controller": 1,
                        "network_adapter": 1,
                        "power_supply": 2,
                        "cable": 2,
                    },
                    "decision": "recommend",
                    "title": "Role-specific SSD BOM",
                    "why_selected": "Uses two separate SSD lines from the matrix.",
                    "confidence": "medium",
                }
            ],
            "no_recommendation": None,
            "general_notes": [],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_missing_source_type_invented_id(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _requirement_contract_for_payload(payload)
    if stage == "bom_composition":
        candidates = payload["candidate_facts_by_role"]
        return {
            "recommendations": [
                {
                    "recommendation_id": "invented_missing_source_type",
                    "proposal_role": "cheapest_fit",
                    "component_candidate_ids": {
                        "platform": "invented-platform-id",
                        "cpu": _first_candidate_id(candidates, "cpu"),
                    },
                    "quantities": {"platform": 1, "cpu": 2},
                    "decision": "recommend",
                    "title": "Invented missing source type",
                    "why_selected": "Should not receive a safe default source_type.",
                    "confidence": "medium",
                }
            ],
            "no_recommendation": None,
            "general_notes": [],
        }
    if stage == "completeness_critic":
        return {
            "all_hard_requirements_covered": False,
            "missing_roles": ["server_platform"],
            "insufficient_quantities": [],
            "unverified_requirements": [],
            "hard_mismatch_risks": [
                {
                    "role": "server_platform",
                    "reason": "invented component_candidate_id",
                }
            ],
            "recommended_repair_actions": ["Return structured no_recommendation."],
        }
    if stage == "repair":
        return {
            "recommendations": [],
            "no_recommendation": {
                "summary": "Invented component IDs cannot be used.",
                "missing_roles": ["server_platform"],
                "hard_mismatches": [
                    {
                        "role": "server_platform",
                        "reason": "invented component_candidate_id",
                    }
                ],
                "recommended_next_actions": ["Use only provided candidate IDs."],
            },
            "general_notes": [],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_missing_cable_then_empty_repair(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _requirement_contract_for_payload(payload)
    if stage == "bom_composition":
        bom = _complete_multi_pass_bom(payload)
        recommendation = dict(bom["recommendations"][0])
        component_ids = dict(recommendation["component_candidate_ids"])
        quantities = dict(recommendation["quantities"])
        component_ids.pop("cable", None)
        quantities.pop("cable", None)
        recommendation["component_candidate_ids"] = component_ids
        recommendation["quantities"] = quantities
        bom["recommendations"] = [recommendation]
        return bom
    if stage == "completeness_critic":
        return {
            "all_hard_requirements_covered": False,
            "missing_roles": ["cable"],
            "insufficient_quantities": [],
            "unverified_requirements": [],
            "hard_mismatch_risks": [],
            "recommended_repair_actions": ["Add required cable or decline."],
        }
    if stage == "repair":
        return {"recommendations": [], "general_notes": []}
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_insufficient_ssd_then_no_recommendation(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage == "requirement_contract":
        return _requirement_contract_for_payload(payload)
    if stage == "bom_composition":
        bom = _complete_multi_pass_bom(payload)
        recommendation = dict(bom["recommendations"][0])
        quantities = dict(recommendation["quantities"])
        quantities["storage"] = 6
        recommendation["quantities"] = quantities
        bom["recommendations"] = [recommendation]
        return bom
    if stage == "completeness_critic":
        return {
            "all_hard_requirements_covered": False,
            "missing_roles": [],
            "insufficient_quantities": [{"role": "ssd", "expected": 8, "actual": 6}],
            "unverified_requirements": [],
            "hard_mismatch_risks": [],
            "recommended_repair_actions": ["Add two SSDs or decline."],
        }
    if stage == "repair":
        return {
            "recommendations": [],
            "no_recommendation": {
                "summary": "No safe complete BOM with only six SSDs.",
                "missing_roles": [],
                "missing_required_capabilities": [],
                "hard_mismatches": [],
                "stock_shortages": [],
                "role_analysis": [
                    {
                        "role": "ssd",
                        "status": "insufficient_quantity",
                        "considered_candidate_ids": [],
                        "explanation": "Eight SSDs are required, six were selected.",
                    }
                ],
                "considered_candidate_ids": {},
                "recommended_next_actions": ["Add SSD quantity or escalate."],
            },
            "general_notes": [],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_incomplete_then_repair_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage in {"requirement_contract", "role_evaluation"}:
        return _multi_pass_complete_response(payload)
    if stage == "bom_composition":
        return _incomplete_90_like_bom(payload)
    if stage == "completeness_critic":
        return _missing_accessory_critic()
    if stage == "repair":
        return _complete_multi_pass_bom(payload)
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_incomplete_then_no_recommendation(payload: Mapping[str, Any]) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage in {"requirement_contract", "role_evaluation"}:
        return _multi_pass_complete_response(payload)
    if stage == "bom_composition":
        return _incomplete_90_like_bom(payload)
    if stage == "completeness_critic":
        return _missing_accessory_critic()
    if stage == "repair":
        considered = {
            role: [
                row["component_candidate_id"]
                for row in rows
                if isinstance(row, Mapping)
            ]
            for role, rows in payload["candidate_facts_by_role"].items()
        }
        return {
            "recommendations": [],
            "no_recommendation": {
                "summary": "No safe complete multi-pass BOM can be repaired.",
                "missing_roles": ["network_adapter"],
                "missing_required_capabilities": [],
                "hard_mismatches": [],
                "stock_shortages": [],
                "role_analysis": [
                    {
                        "role": role,
                        "status": "missing" if role == "network_adapter" else "satisfied",
                        "considered_candidate_ids": ids,
                        "explanation": "Repair could not verify a complete safe BOM.",
                    }
                    for role, ids in considered.items()
                ],
                "considered_candidate_ids": considered,
                "explanation_ru": "РќРµ СѓРґР°Р»РѕСЃСЊ Р±РµР·РѕРїР°СЃРЅРѕ РёСЃРїСЂР°РІРёС‚СЊ BOM.",
                "recommended_next_actions": ["Escalate to engineer."],
            },
            "general_notes": [],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _multi_pass_incomplete_then_alias_no_recommendation(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    stage = payload.get("multi_pass_stage")
    if stage in {"requirement_contract", "role_evaluation"}:
        return _multi_pass_complete_response(payload)
    if stage == "bom_composition":
        return _incomplete_90_like_bom(payload)
    if stage == "completeness_critic":
        return {
            "all_hard_requirements_covered": False,
            "missing_roles": ["support_pack"],
            "insufficient_quantities": [],
            "unverified_requirements": [{"role": "fabric_adapter"}],
            "hard_mismatch_risks": [{"role": "compute_node", "reason": "socket"}],
            "recommended_repair_actions": ["Request compatible replacements."],
        }
    if stage == "repair":
        return {
            "requirement_analysis": {
                "hard_requirements_met": ["baseline accessories available"]
            },
            "recommendations": [],
            "no_recommendation": {
                "reason": "Repair pass found no safe complete BOM.",
                "role_level_reasons": [
                    {"role": "compute_node", "reason": "Socket mismatch."},
                    {"role": "fabric_adapter", "reason": "Stock shortage."},
                ],
                "failures_by_role": {
                    "license_pack": "No safe license candidate was verified."
                },
            },
            "general_notes": ["Partial platform information is usable."],
        }
    raise AssertionError(f"unexpected multi-pass stage: {stage}")


def _server_78_requirement_contract() -> dict[str, Any]:
    return {
        "primary_object": "server",
        "required_roles": [
            "server_platform",
            "cpu",
            "ram",
            "ssd",
            "storage_controller",
            "network_adapter",
            "power_supply",
            "cable",
        ],
        "required_quantities_by_role": {
            "server_platform": {"count": 1, "features": ["1U", "2 sockets"]},
            "cpu": {
                "count": 2,
                "vendor": "Intel",
                "generation": "6th",
                "min_cores_per_cpu": 24,
                "min_frequency_ghz": 2.2,
            },
            "ram": {
                "module_count": 8,
                "module_capacity_gb": 32,
                "total_gb": 256,
                "type": "DDR5 RDIMM",
                "speed_mhz": 6400,
            },
            "storage": {
                "groups": [
                    {"count": 6, "capacity_gb": 1920, "interface": "SATA"},
                    {"count": 2, "capacity_gb": 480, "interface": "SATA"},
                ]
            },
            "storage_controller": {"count": 1, "mode": "JBOD/hot-swap"},
            "network_adapter": {
                "count": 1,
                "min_ports_per_server": 2,
                "speed": "10GbE",
                "media": "SFP+",
            },
            "power_supply": {"count": 2, "wattage": 2000, "redundancy": "1+1"},
            "cable": {"count": 2, "type": "C13-C14 or C13-Schuko"},
        },
        "hard_requirements": [],
        "optional_requirements": [],
        "primary_object_features": [
            "1U",
            "2 sockets",
            "8 SFF front bays",
            "N+1 cooling",
            "USB/VGA/serial/management",
        ],
        "purchasable_component_roles": [
            "server_platform",
            "cpu",
            "ram",
            "ssd",
            "storage_controller",
            "network_adapter",
            "power_supply",
            "cable",
        ],
        "accessories": ["power cables"],
        "services_support": [],
        "logistics_commercial_constraints": [],
        "fulfillment_expectations": [],
        "engineer_checks": ["Confirm platform compatibility before quote."],
    }


def _storage_nas_requirement_contract() -> dict[str, Any]:
    return {
        "primary_object": "storage_system",
        "required_roles": ["storage_system", "drive"],
        "required_quantities_by_role": {
            "storage_system": {"count": 1},
            "drive": {"usable_capacity_tb": 4, "interface": "NVMe"},
        },
        "hard_requirements": ["NAS with 4TB on NVMe"],
        "optional_requirements": ["preferably QNAP"],
        "primary_object_features": ["NAS enclosure"],
        "purchasable_component_roles": ["storage_system", "drive"],
        "accessories": [],
        "services_support": [],
        "logistics_commercial_constraints": [],
        "fulfillment_expectations": [
            {
                "role": "drive",
                "fulfillment_mode": "separate_component_required",
                "requirement_text": "4TB on NVMe",
            }
        ],
        "engineer_checks": ["Confirm drive compatibility before quote."],
    }


def _role_evaluation_response(payload: Mapping[str, Any]) -> dict[str, Any]:
    ids = list(payload.get("candidate_ids_for_chunk") or [])
    return {
        "role": payload["role"],
        "considered_candidate_ids": ids,
        "best_candidate_ids": ids[:1],
        "rejected_candidate_ids": [],
        "uncertain_candidate_ids": ids[1:],
        "missing_facts": [],
        "role_specific_risks": [],
        "cheapest_safe_candidates": ids[:1],
        "exact_or_equivalent_candidates": ids[:1],
    }


def _complete_multi_pass_bom(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidates = payload["candidate_facts_by_role"]
    component_ids = {
        "platform": _first_candidate_id(candidates, "server_platform"),
        "cpu": _first_candidate_id(candidates, "cpu"),
        "ram": _first_candidate_id(candidates, "ram"),
        "storage": _first_candidate_id(candidates, "ssd"),
        "storage_controller": _first_candidate_id(candidates, "storage_controller"),
        "network_adapter": _first_candidate_id(candidates, "network_adapter"),
        "power_supply": _first_candidate_id(candidates, "power_supply"),
        "cable": _first_candidate_id(candidates, "cable"),
    }
    return {
        "requirement_analysis": {
            "fulfillment_decisions": [],
            "requirement_contract_used": True,
        },
        "recommendations": [
            {
                "recommendation_id": "multi_pass_complete",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": component_ids,
                "quantities": {
                    "platform": 1,
                    "cpu": 2,
                    "ram": 8,
                    "storage": 8,
                    "storage_controller": 1,
                    "network_adapter": 1,
                    "power_supply": 2,
                    "cable": 2,
                },
                "decision": "recommend",
                "title": "Multi-pass complete server BOM",
                "why_selected": "Complete BOM assembled from role evaluation summaries.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _incomplete_90_like_bom(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidates = payload["candidate_facts_by_role"]
    return {
        "recommendations": [
            {
                "recommendation_id": "multi_pass_incomplete",
                "proposal_role": "cheapest_fit",
                "source_type": "build_from_parts",
                "component_candidate_ids": {
                    "platform": _first_candidate_id(candidates, "server_platform"),
                    "cpu": _first_candidate_id(candidates, "cpu"),
                    "ram": _first_candidate_id(candidates, "ram"),
                    "storage": _first_candidate_id(candidates, "ssd"),
                },
                "quantities": {"platform": 1, "cpu": 1, "ram": 1, "storage": 1},
                "decision": "recommend",
                "title": "Incomplete #90-like BOM",
                "why_selected": "Intentionally incomplete for critic test.",
                "confidence": "medium",
            }
        ],
        "no_recommendation": None,
        "general_notes": [],
    }


def _missing_accessory_critic() -> dict[str, Any]:
    return {
        "all_hard_requirements_covered": False,
        "missing_roles": [
            "storage_controller",
            "network_adapter",
            "power_supply",
            "cable",
        ],
        "insufficient_quantities": [
            {"role": "cpu", "expected": 2, "actual": 1},
            {"role": "ram", "expected": 8, "actual": 1},
        ],
        "unverified_requirements": [],
        "hard_mismatch_risks": [],
        "recommended_repair_actions": ["Add missing roles and fix quantities."],
    }


def _first_candidate_id(
    candidates: Mapping[str, Any],
    role: str,
) -> str:
    rows = candidates[role]
    return rows[0]["component_candidate_id"]


def _seed_basic_server_catalog(
    db_session: Session,
    *,
    cpu_count: int = 1,
    ram_count: int = 1,
    ssd_count: int = 1,
    ram_quantity: int = 16,
    platform_name: str = "Intel Xeon LGA4677 DDR5 1U dual socket server platform",
    cpu_name: str = "Intel Xeon Gold 24 core LGA4677 processor",
    ram_name: str = "DDR5 RDIMM 64GB server memory module",
) -> None:
    categories = {
        "cat-platform": "Server Platforms",
        "cat-cpu": "CPU Processors",
        "cat-ram": "DDR5 RDIMM Memory",
        "cat-ssd": "SATA SSD Drives",
        "cat-hba": "LSI RAID HBA Storage Controllers",
        "cat-nic": "Network Adapters 10GbE SFP+",
        "cat-psu": "Power Supply PSU",
        "cat-cable": "C13 C14 Power Cables",
        "cat-accessory": "Fans Rails Accessories",
    }
    for category_id, name in categories.items():
        _seed_category(db_session, category_id, name)
    _seed_product(
        db_session,
        item_id="platform-1",
        category_id="cat-platform",
        category_name=categories["cat-platform"],
        part_number="PLATFORM-1",
        item_name=platform_name,
        quantity=4,
        price=Decimal("2000"),
    )
    for index in range(cpu_count):
        _seed_product(
            db_session,
            item_id=f"cpu-{index}",
            category_id="cat-cpu",
            category_name=categories["cat-cpu"],
            part_number=f"CPU-{index}",
            item_name=f"{cpu_name} {index}",
            quantity=4,
            price=Decimal("500") + index,
        )
    for index in range(ram_count):
        _seed_product(
            db_session,
            item_id=f"ram-{index}",
            category_id="cat-ram",
            category_name=categories["cat-ram"],
            part_number=f"RAM-{index}",
            item_name=f"{ram_name} {index}",
            quantity=ram_quantity,
            price=Decimal("150") + index,
        )
    for index in range(ssd_count):
        _seed_product(
            db_session,
            item_id=f"ssd-{index}",
            category_id="cat-ssd",
            category_name=categories["cat-ssd"],
            part_number=f"SSD-{index}",
            item_name=f"Enterprise SATA SSD 1.92TB drive {index}",
            quantity=8,
            price=Decimal("180") + index,
        )
    _seed_product(
        db_session,
        item_id="hba-1",
        category_id="cat-hba",
        category_name=categories["cat-hba"],
        part_number="HBA-1",
        item_name="LSI 9500-8i tri-mode HBA storage controller",
        quantity=2,
        price=Decimal("350"),
    )
    _seed_product(
        db_session,
        item_id="nic-1",
        category_id="cat-nic",
        category_name=categories["cat-nic"],
        part_number="X710-DA2",
        item_name="Intel X710-DA2 2x10GbE SFP+ network adapter",
        quantity=2,
        price=Decimal("250"),
    )
    _seed_product(
        db_session,
        item_id="psu-1",
        category_id="cat-psu",
        category_name=categories["cat-psu"],
        part_number="PSU-2000W",
        item_name="2000W hot-swap Platinum power supply PSU",
        quantity=4,
        price=Decimal("300"),
    )
    _seed_product(
        db_session,
        item_id="cable-1",
        category_id="cat-cable",
        category_name=categories["cat-cable"],
        part_number="C13-C14",
        item_name="C13-C14 power cable",
        quantity=10,
        price=Decimal("10"),
    )
    _seed_product(
        db_session,
        item_id="fan-1",
        category_id="cat-accessory",
        category_name=categories["cat-accessory"],
        part_number="FAN-1",
        item_name="server cooling fan accessory kit",
        quantity=10,
        price=Decimal("20"),
    )
    db_session.commit()


def _seed_network_catalog(
    db_session: Session,
    *,
    include_transceiver: bool = False,
) -> None:
    _seed_category(db_session, "cat-switch", "Network Switches PoE")
    _seed_product(
        db_session,
        item_id="switch-1",
        category_id="cat-switch",
        category_name="Network Switches PoE",
        part_number="SW-48P-4SFP",
        item_name="48 port 1G PoE switch 4x10G SFP+ uplinks L3 stacking",
        quantity=1,
        price=Decimal("1200"),
    )
    if include_transceiver:
        _seed_category(db_session, "cat-transceiver", "SFP+ Transceiver Optics")
        _seed_product(
            db_session,
            item_id="transceiver-1",
            category_id="cat-transceiver",
            category_name="SFP+ Transceiver Optics",
            part_number="SFP-10G-SR",
            item_name="10G SFP+ transceiver module",
            quantity=8,
            price=Decimal("80"),
        )
    db_session.commit()


def _seed_storage_catalog(db_session: Session) -> None:
    _seed_category(db_session, "cat-storage-system", "NAS Storage Systems")
    _seed_category(db_session, "cat-drive", "NVMe Drive Media")
    _seed_product(
        db_session,
        item_id="nas-1",
        category_id="cat-storage-system",
        category_name="NAS Storage Systems",
        part_number="NAS-1",
        item_name="Universal 4-bay NAS storage system",
        quantity=1,
        price=Decimal("800"),
    )
    for index in range(2):
        _seed_product(
            db_session,
            item_id=f"drive-{index}",
            category_id="cat-drive",
            category_name="NVMe Drive Media",
            part_number=f"NVME-2TB-{index}",
            item_name=f"2TB NVMe drive media {index}",
            quantity=2,
            price=Decimal("220") + index,
        )
    db_session.commit()


def _seed_storage_system_only_catalog(db_session: Session) -> None:
    _seed_category(db_session, "cat-storage-system", "NAS Storage Systems")
    _seed_product(
        db_session,
        item_id="nas-1",
        category_id="cat-storage-system",
        category_name="NAS Storage Systems",
        part_number="NAS-1",
        item_name="QNAP-like 4TB NVMe NAS storage system",
        quantity=1,
        price=Decimal("800"),
    )
    db_session.commit()


def _seed_storage_ssd_alias_catalog(db_session: Session) -> None:
    _seed_category(db_session, "cat-storage-system", "Storage Systems")
    _seed_category(db_session, "cat-ssd", "SSD Drive Media")
    _seed_product(
        db_session,
        item_id="storage-system-1",
        category_id="cat-storage-system",
        category_name="Storage Systems",
        part_number="STORAGE-1",
        item_name="Universal storage system",
        quantity=1,
        price=Decimal("1200"),
    )
    _seed_product(
        db_session,
        item_id="ssd-drive-1",
        category_id="cat-ssd",
        category_name="SSD Drive Media",
        part_number="SSD-2TB",
        item_name="2TB SSD drive media",
        quantity=4,
        price=Decimal("200"),
    )
    db_session.commit()


def _seed_category(db_session: Session, category_id: str, name: str) -> None:
    db_session.add(
        DistributorCategory(
            distributor_code="ocs",
            category_id=category_id,
            parent_category_id=None,
            name=name,
            level=1,
            path_json=[{"category_id": category_id, "name": name}],
            enabled_for_sync=True,
            raw_json={"id": category_id, "name": name},
            synced_at=datetime.now(UTC),
        )
    )


def _seed_product(
    db_session: Session,
    *,
    item_id: str,
    category_id: str,
    category_name: str,
    part_number: str,
    item_name: str,
    quantity: int,
    price: Decimal,
) -> None:
    now = datetime.now(UTC)
    db_session.add(
        DistributorProduct(
            distributor_code="ocs",
            item_id=item_id,
            product_key=f"ocs:{item_id}",
            part_number=part_number,
            producer="TestVendor",
            category_id=category_id,
            item_name=item_name,
            item_name_rus=item_name,
            product_name=item_name,
            product_description=item_name,
            product_notes=None,
            catalog_path_json=[{"category_id": category_id, "name": category_name}],
            package_json={},
            raw_json={},
            synced_at=now,
        )
    )
    db_session.add(
        DistributorStockPrice(
            distributor_code="ocs",
            item_id=item_id,
            product_key=f"ocs:{item_id}",
            shipment_city="Moscow",
            location="main",
            location_description="main",
            location_type="stock",
            quantity_value=quantity,
            quantity_is_greater_than=False,
            can_reserve=True,
            price_order_value=price,
            price_order_currency="USD",
            price_list_value=price,
            price_list_currency="USD",
            end_user_value=price,
            end_user_currency="USD",
            raw_json={},
            synced_at=now,
        )
    )
