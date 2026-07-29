from __future__ import annotations

from pathlib import Path

from app.policies.product_group_policy import (
    NETWORK_PRODUCT_GROUP_PROFILE,
    SERVER_PRODUCT_GROUP_PROFILE,
    STORAGE_PRODUCT_GROUP_PROFILE,
    get_product_group_profile,
)


def test_server_product_group_profile_documents_baseline_roles() -> None:
    profile = get_product_group_profile("server")

    assert profile is SERVER_PRODUCT_GROUP_PROFILE
    assert profile.product_group_id == "server"
    assert profile.required_roles == ("server_platform", "cpu", "ram", "storage")
    for role_id in (
        "ready_server",
        "server_platform",
        "cpu",
        "ram",
        "storage",
        "network_adapter",
        "storage_controller",
        "gpu",
        "transceiver",
        "cable",
        "power_supply",
        "rail_kit",
        "license",
        "support",
        "other_accessory",
    ):
        assert role_id in profile.role_catalog
    assert "socket" in profile.compatibility_dimensions
    assert "ram_type" in profile.compatibility_dimensions
    assert "nvme_support" in profile.compatibility_dimensions
    assert "psu/completeness" in profile.compatibility_dimensions


def test_server_product_group_profile_keeps_repair_equivalence_role_scoped() -> None:
    profile = SERVER_PRODUCT_GROUP_PROFILE

    assert any("within the same role eligibility" in rule for rule in profile.equivalence_rules)
    assert profile.commercial_output_template["default_output_mode"] == (
        "single_best_cost_valid"
    )
    assert profile.commercial_output_template["excel_sheets"] == (
        "AI-рекомендации",
        "Матрица компонентов",
    )


def test_network_product_group_profile_documents_mvp_contract() -> None:
    profile = get_product_group_profile("network")

    assert profile is NETWORK_PRODUCT_GROUP_PROFILE
    assert profile.product_group_id == "network"
    assert profile.required_roles == ()
    for role_id in (
        "switch",
        "router",
        "firewall",
        "access_point",
        "transceiver",
        "dac_cable",
        "cable",
        "license",
        "support",
        "power_supply",
        "stacking_module",
        "other_accessory",
    ):
        assert role_id in profile.role_catalog
    for dimension in (
        "port_count",
        "port_speed",
        "port_media",
        "poe_budget",
        "l2/l3",
        "stacking",
        "license/support completeness",
    ):
        assert dimension in profile.compatibility_dimensions
    assert profile.quantity_rules["power_supply"].startswith("hard only when explicit")


def test_storage_product_group_profile_documents_mvp_contract() -> None:
    profile = get_product_group_profile("storage")

    assert profile is STORAGE_PRODUCT_GROUP_PROFILE
    assert profile.product_group_id == "storage"
    for role_id in (
        "storage_system",
        "controller",
        "controller_module",
        "disk_shelf",
        "drive",
        "ssd",
        "hdd",
        "cache",
        "host_port",
        "protocol_module",
        "transceiver",
        "cable",
        "license",
        "support",
        "power_supply",
        "rail_kit",
        "other_accessory",
    ):
        assert role_id in profile.role_catalog
    validation_capabilities = {
        capability
        for entry in profile.role_catalog.values()
        for capability in entry.validation_capabilities
    }
    for capability in (
        "raw_capacity_tb",
        "usable_capacity_tb",
        "host_protocol",
        "host_port_speed",
        "support_required",
    ):
        assert capability in validation_capabilities
    for rule in (
        "usable/raw capacity cannot be safely satisfied",
        "LLM proposes component_candidate_id not in matrix",
    ):
        assert rule in profile.no_recommendation_rules
    assert "СХД - N шт." in profile.commercial_output_template["telegram"]
    assert profile.commercial_output_template["no_recommendation"] == (
        "Безопасную складскую рекомендацию дать нельзя."
    )


def test_ocs_category_ids_are_not_product_policy_logic() -> None:
    policy_source = Path("app/policies/product_group_policy.py").read_text(
        encoding="utf-8"
    )

    for category_id in ("V1100", "V110100", "V110103", "V120116"):
        assert category_id not in policy_source
