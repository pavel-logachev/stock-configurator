from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.policies.product_group_policy import get_product_group_profile

SERVER_PRODUCT_GROUP = "server"
UNMAPPED_ROLE = "unmapped"

ROLE_SOURCE_STAGE_A = "stage_a"
ROLE_SOURCE_PRODUCT_GROUP_PROFILE = "product_group_profile"
ROLE_SOURCE_REQUIREMENT_CLASSIFIER = "requirement_classifier"
ROLE_SOURCE_ACCESSORY_HINT = "accessory_hint"
ROLE_SOURCE_FALLBACK_PROFILE = "fallback_profile"
ROLE_SOURCE_CATEGORY_PLANNER = "category_planner"

ROLE_SOURCE_VALUES = (
    ROLE_SOURCE_STAGE_A,
    ROLE_SOURCE_PRODUCT_GROUP_PROFILE,
    ROLE_SOURCE_REQUIREMENT_CLASSIFIER,
    ROLE_SOURCE_ACCESSORY_HINT,
    ROLE_SOURCE_FALLBACK_PROFILE,
    ROLE_SOURCE_CATEGORY_PLANNER,
)


def normalize_role(value: Any, *, product_group: str | None = None) -> str:
    text = str(value or "").strip()
    normalized = text.casefold().replace("-", "_").replace("/", "_").replace(" ", "_")
    aliases = {
        "platform": "server_platform",
        "barebone": "server_platform",
        "chassis": "server_platform",
        "processor": "cpu",
        "processors": "cpu",
        "memory": "ram",
        "nic": "network_adapter",
        "network_card": "network_adapter",
        "network_interface_card": "network_adapter",
        "psu": "power_supply",
        "power": "power_supply",
        "power_cable": "cable",
        "power_cord": "cable",
        "power_cords": "cable",
        "c13_c14": "cable",
        "c13_schuko": "cable",
        "accessory": "other_accessory",
        "accessories": "other_accessory",
        "unknown": UNMAPPED_ROLE,
        "unmapped": UNMAPPED_ROLE,
    }
    if product_group == SERVER_PRODUCT_GROUP:
        aliases.update(
            {
                "drive": "storage",
                "drives": "storage",
                "disk": "storage",
                "disks": "storage",
                "ssd": "storage",
                "storage_ssd": "storage",
                "storage_drive": "storage",
                "hdd": "storage",
                "controller": "storage_controller",
                "hba": "storage_controller",
                "raid_controller": "storage_controller",
            }
        )
    return aliases.get(normalized, normalized or text)


def unique_roles(values: Sequence[Any], *, product_group: str | None = None) -> list[str]:
    result: list[str] = []
    for value in values:
        role = normalize_role(value, product_group=product_group)
        if role and role not in result:
            result.append(role)
    return result


def role_set(value: Any, *, product_group: str | None = None) -> set[str]:
    if not isinstance(value, list | tuple | set):
        return set()
    return set(unique_roles(list(value), product_group=product_group))


def roles_from_blueprint(
    matrix_blueprint: Any,
    *,
    product_group: str | None = None,
) -> list[str]:
    blueprint = matrix_blueprint if isinstance(matrix_blueprint, Mapping) else {}
    return unique_roles(
        [
            row.get("role") or row.get("role_id")
            for row in _mapping_rows(blueprint.get("roles"))
        ],
        product_group=product_group,
    )


def roles_from_capabilities(
    capabilities: Any,
    *,
    product_group: str | None = None,
) -> list[str]:
    return unique_roles(
        [row.get("role") or row.get("role_id") for row in _mapping_rows(capabilities)],
        product_group=product_group,
    )


def roles_from_classified_requirements(
    classified_requirements: Any,
    *,
    product_group: str | None = None,
) -> list[str]:
    roles: list[str] = []
    for row in _mapping_rows(classified_requirements):
        role = normalize_role(
            row.get("target_role") or row.get("role") or row.get("role_id"),
            product_group=product_group,
        )
        if not role:
            continue
        if role == UNMAPPED_ROLE:
            continue
        classification = str(row.get("classification") or "").strip()
        if classification in {
            "primary_object_feature",
            "logistics_or_commercial_constraint",
            "engineering_check",
            "out_of_scope_or_unmapped_non_blocking",
            "blocking_unmapped_purchasable_role",
        }:
            continue
        fulfillment_mode = str(row.get("fulfillment_mode") or "").strip()
        if fulfillment_mode in {
            "included_in_primary_object",
            "included_in_selected_component",
            "included_in_bundle_or_kit",
            "engineering_check_only",
            "logistics_constraint",
            "not_applicable",
        }:
            continue
        if _explicit_false(row.get("should_create_bom_role")):
            continue
        if role not in roles:
            roles.append(role)
    return roles


def roles_from_category_plan(
    category_plan: Any,
    *,
    product_group: str | None = None,
) -> list[str]:
    if not isinstance(category_plan, Mapping):
        return []
    return unique_roles(list(category_plan.keys()), product_group=product_group)


def product_group_profile_broad_roles(product_group: str | None) -> list[str]:
    profile = get_product_group_profile(str(product_group or "").strip())
    if profile is None:
        return []
    return unique_roles(list(profile.required_roles), product_group=profile.product_group_id)


def accessory_hint_roles(intent: Mapping[str, Any], *, product_group: str) -> list[str]:
    roles: list[str] = []
    for hint in _list_rows(intent.get("accessory_hints")):
        if isinstance(hint, Mapping):
            role = normalize_role(
                hint.get("target_role") or hint.get("role") or hint.get("role_id"),
                product_group=product_group,
            )
            if role:
                roles.append(role)
    return unique_roles(roles, product_group=product_group)


def merge_role_sources(
    *sources: tuple[str, Sequence[str]],
    existing: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if isinstance(existing, Mapping):
        for role, source_values in existing.items():
            normalized_role = normalize_role(role)
            result[normalized_role] = [
                source
                for source in _string_list(source_values)
                if source in ROLE_SOURCE_VALUES
            ]
    for source, roles in sources:
        if source not in ROLE_SOURCE_VALUES:
            continue
        for role in roles:
            normalized_role = normalize_role(role)
            if not normalized_role:
                continue
            result.setdefault(normalized_role, [])
            if source not in result[normalized_role]:
                result[normalized_role].append(source)
    return {role: result[role] for role in sorted(result)}


def roles_from_sources(role_source_by_role: Mapping[str, Any]) -> list[str]:
    return sorted(
        role
        for role, sources in role_source_by_role.items()
        if _string_list(sources)
    )


def dropped_roles(before: Sequence[str], after: Sequence[str]) -> list[str]:
    after_set = set(after)
    return [role for role in unique_roles(list(before)) if role not in after_set]


def merge_drop_reasons(
    *reason_maps: Mapping[str, Any],
    existing: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    result = {
        str(role): str(reason)
        for role, reason in (existing or {}).items()
        if str(role).strip() and str(reason).strip()
    }
    for reason_map in reason_maps:
        for role, reason in reason_map.items():
            role_text = normalize_role(role)
            reason_text = str(reason or "").strip()
            if role_text and reason_text:
                result[role_text] = reason_text
    return dict(sorted(result.items()))


def build_role_lifecycle_trace(
    roles: Sequence[str],
    *,
    role_source_by_role: Mapping[str, Any] | None = None,
    stage_a_roles: Sequence[str] = (),
    semantic_matrix_blueprint_roles: Sequence[str] = (),
    requirement_classifier_roles: Sequence[str] = (),
    before_category_planner_roles: Sequence[str] = (),
    category_planner_input_roles: Sequence[str] = (),
    category_planner_output_roles: Sequence[str] = (),
    validated_category_plan_roles: Sequence[str] = (),
    materialized_matrix_roles: Sequence[str] = (),
    composer_package_roles: Sequence[str] = (),
    dropped_reason_by_role: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    stage_sets = {
        "stage_a": set(stage_a_roles),
        "semantic_matrix_blueprint": set(semantic_matrix_blueprint_roles),
        "requirement_classifier": set(requirement_classifier_roles),
        "before_category_planner": set(before_category_planner_roles),
        "category_planner_input": set(category_planner_input_roles),
        "category_planner_output": set(category_planner_output_roles),
        "validated_category_plan": set(validated_category_plan_roles),
        "materialized_matrix": set(materialized_matrix_roles),
        "composer_package": set(composer_package_roles),
    }
    all_roles = unique_roles(
        [
            *roles,
            *stage_a_roles,
            *semantic_matrix_blueprint_roles,
            *requirement_classifier_roles,
            *before_category_planner_roles,
            *category_planner_input_roles,
            *category_planner_output_roles,
            *validated_category_plan_roles,
            *materialized_matrix_roles,
            *composer_package_roles,
        ]
    )
    sources = role_source_by_role or {}
    reasons = dropped_reason_by_role or {}
    trace: list[dict[str, Any]] = []
    for role in all_roles:
        trace.append(
            {
                "role": role,
                "sources": _string_list(sources.get(role)),
                **{stage: role in values for stage, values in stage_sets.items()},
                "dropped_reason": str(reasons.get(role) or "").strip() or None,
            }
        )
    return trace


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _list_rows(value: Any) -> list[Any]:
    if not isinstance(value, list | tuple):
        return []
    return list(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple | set):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _explicit_false(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"false", "0", "no"}
    return False
