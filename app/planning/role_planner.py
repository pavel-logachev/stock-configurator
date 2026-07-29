from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.matching.spec_schema import StockSpec
from app.planning.requirement_planner import (
    RequirementPlannerClient,
    plan_semantic_matrix_requirements,
    plan_universal_requirements,
)
from app.policies.product_group_policy import SERVER_ROLE_CATALOG as POLICY_SERVER_ROLE_CATALOG
from app.policies.product_group_policy import get_product_group_profile

SERVER_PRODUCT_GROUP = "server"
NETWORK_PRODUCT_GROUP = "network"
STORAGE_PRODUCT_GROUP = "storage"
BASE_REQUIRED_ROLES = ("server_platform", "cpu", "ram", "storage")
DYNAMIC_SERVER_ROLES = tuple(
    role_id
    for role_id, role in POLICY_SERVER_ROLE_CATALOG.items()
    if role.behavior == "hard_when_requested"
)
SERVER_ROLE_CATALOG_IDS = tuple(POLICY_SERVER_ROLE_CATALOG)


def plan_server_roles(
    spec: StockSpec | Mapping[str, Any] | str,
    *,
    distributor_code: str | None = None,
    planner_client: RequirementPlannerClient | None = None,
) -> dict[str, Any]:
    profile = get_product_group_profile(SERVER_PRODUCT_GROUP)
    return plan_universal_requirements(
        spec,
        product_group_profile=profile,
        distributor_code=distributor_code,
        planner_client=planner_client,
    )


def plan_product_group_roles(
    spec: StockSpec | Mapping[str, Any] | str,
    *,
    product_group: str | None = None,
    distributor_code: str | None = None,
    planner_client: RequirementPlannerClient | None = None,
) -> dict[str, Any]:
    profile = get_product_group_profile(product_group or "") if product_group else None
    return plan_universal_requirements(
        spec,
        product_group_profile=profile,
        distributor_code=distributor_code,
        planner_client=planner_client,
    )


def plan_semantic_matrix_roles(
    spec: StockSpec | Mapping[str, Any] | str,
    *,
    distributor_code: str | None = None,
    planner_client: RequirementPlannerClient | None = None,
    deterministic_product_group_hint: str | None = None,
    semantic_planner_max_seconds: float | None = None,
    semantic_planner_stage_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    return plan_semantic_matrix_requirements(
        spec,
        distributor_code=distributor_code,
        planner_client=planner_client,
        deterministic_product_group_hint=deterministic_product_group_hint,
        semantic_planner_max_seconds=semantic_planner_max_seconds,
        semantic_planner_stage_timeout_seconds=semantic_planner_stage_timeout_seconds,
    )


def plan_network_roles(
    spec: StockSpec | Mapping[str, Any] | str,
    *,
    distributor_code: str | None = None,
    planner_client: RequirementPlannerClient | None = None,
) -> dict[str, Any]:
    return plan_product_group_roles(
        spec,
        product_group=NETWORK_PRODUCT_GROUP,
        distributor_code=distributor_code,
        planner_client=planner_client,
    )


def plan_storage_roles(
    spec: StockSpec | Mapping[str, Any] | str,
    *,
    distributor_code: str | None = None,
    planner_client: RequirementPlannerClient | None = None,
) -> dict[str, Any]:
    return plan_product_group_roles(
        spec,
        product_group=STORAGE_PRODUCT_GROUP,
        distributor_code=distributor_code,
        planner_client=planner_client,
    )


# Backward-compatible name used by older planning tests/imports.
SERVER_ROLE_CATALOG = SERVER_ROLE_CATALOG_IDS
