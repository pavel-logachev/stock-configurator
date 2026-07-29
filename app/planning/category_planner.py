from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.catalog.ocs_anchor_categories import OcsAnchorCategory, load_ocs_anchor_categories
from app.catalog.ocs_server_categories import load_server_category_profile
from app.llm.base import LlmClient, LlmError
from app.planning import role_lifecycle
from app.policies.product_group_policy import (
    NETWORK_ROLE_CATALOG as NETWORK_PROFILE_ROLE_CATALOG,
)
from app.policies.product_group_policy import (
    SERVER_ROLE_CATALOG as SERVER_PROFILE_ROLE_CATALOG,
)
from app.policies.product_group_policy import (
    STORAGE_ROLE_CATALOG as STORAGE_PROFILE_ROLE_CATALOG,
)
from app.policies.product_group_policy import (
    get_product_group_profile,
)

MAX_CATEGORIES_PER_ROLE = 8
MAX_FULL_CATALOG_FOR_LLM = 80
UNMAPPED_ROLE = "unmapped"

ROLE_SYNONYMS: Mapping[str, tuple[str, ...]] = {
    "server_platform": ("platform", "barebone", "chassis", "серверные платформы", "платформы"),
    "cpu": ("cpu", "processor", "процессор", "серверные процессоры", "epyc", "xeon"),
    "ram": ("ram", "memory", "rdimm", "ddr", "оперативная память", "память"),
    "storage": ("ssd", "hdd", "drive", "disk", "nvme", "sas", "sata", "накопитель", "диск"),
    "storage_controller": ("controller", "raid", "hba", "tri-mode", "контроллер"),
    "network_adapter": ("network", "adapter", "nic", "ethernet", "sfp", "qsfp", "сетевые адаптеры"),
    "gpu": ("gpu", "nvidia", "accelerator", "видеокарта"),
    "transceiver": ("transceiver", "sfp module", "qsfp module", "трансивер"),
    "cable": ("dac", "aoc", "cable", "кабель"),
    "license/support": ("license", "support", "лицензия", "поддержка"),
    "switch": ("switch", "ethernet switch", "коммутатор", "свитч", "poe"),
    "router": ("router", "маршрутизатор", "роутер", "wan", "lan"),
    "firewall": ("firewall", "ngfw", "utm", "межсетевой экран", "фаервол"),
    "access_point": ("access point", "ap", "wi-fi", "wifi", "точка доступа"),
    "dac_cable": ("dac", "direct attach", "dac cable", "dac-кабель"),
    "power_supply": ("psu", "power supply", "блок питания", "бп"),
    "stacking_module": ("stacking", "stack module", "модуль стекирования", "стек"),
    "storage_system": (
        "storage system",
        "storage array",
        "disk array",
        "san",
        "nas",
        "схд",
        "система хранения",
    ),
    "controller": ("controller", "storage controller", "контроллер", "контроллер схд"),
    "controller_module": (
        "controller module",
        "control module",
        "модуль контроллера",
    ),
    "disk_shelf": (
        "disk shelf",
        "drive shelf",
        "expansion shelf",
        "полка",
        "дисковая полка",
    ),
    "drive": ("drive", "disk", "ssd", "hdd", "nvme", "sas", "sata", "диск", "накопитель"),
    "cache": ("cache", "cache module", "flash cache", "кэш", "cache card"),
    "host_port": (
        "host port",
        "fc port",
        "iscsi port",
        "sas port",
        "порт",
        "host interface",
    ),
    "protocol_module": (
        "protocol module",
        "fc module",
        "iscsi module",
        "nvme-of",
        "модуль fc",
        "модуль iscsi",
    ),
}


@dataclass(frozen=True)
class CompactCategory:
    distributor_code: str
    category_id: str
    category_name: str
    category_path: str = ""
    parent_id: str | None = None
    product_count: int | None = None
    sample_product_names: tuple[str, ...] = ()
    mapped_role: str | None = None
    inferred_category_kind: str = "unknown"
    allowed_candidate_roles: tuple[str, ...] = ()
    product_group_context: str = "unknown"
    confidence: str = "low"
    source: str = "inferred"
    base_device_allowed: bool | None = None
    review_status: str = ""

    def to_prompt_json(self) -> dict[str, Any]:
        return {
            "distributor_code": self.distributor_code,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "category_path": self.category_path,
            "parent_id": self.parent_id,
            "product_count": self.product_count,
            "sample_product_names": list(self.sample_product_names),
            "mapped_role_hint": self.mapped_role,
            "inferred_category_kind": self.inferred_category_kind,
            "allowed_candidate_roles": list(self.allowed_candidate_roles),
            "product_group_context": self.product_group_context,
            "confidence": self.confidence,
            "source": self.source,
            "base_device_allowed": self.base_device_allowed,
            "review_status": self.review_status,
        }


@dataclass(frozen=True)
class CategoryPlanResult:
    category_plan: dict[str, list[str]]
    category_plan_entries: list[dict[str, Any]] = field(default_factory=list)
    missing_category_roles: list[str] = field(default_factory=list)
    category_plan_warnings: list[str] = field(default_factory=list)
    role_coverage_summary: dict[str, Any] = field(default_factory=dict)
    category_catalog_summary: dict[str, Any] = field(default_factory=dict)
    category_plan_source: str = "none"
    category_planner_source: str = "none"
    category_planner_input_roles: list[str] = field(default_factory=list)
    category_planner_output_roles: list[str] = field(default_factory=list)
    category_planner_missing_required_roles: list[str] = field(default_factory=list)
    category_planner_repair_attempted: bool = False
    category_planner_repair_success: bool = False
    category_planner_repair_reason: str | None = None
    category_planner_repaired_roles: list[str] = field(default_factory=list)
    category_planner_unresolved_required_roles: list[str] = field(default_factory=list)
    validated_category_plan_roles: list[str] = field(default_factory=list)
    roles_dropped_before_category_planner: list[str] = field(default_factory=list)
    roles_dropped_after_category_planner: list[str] = field(default_factory=list)
    roles_dropped_reason_by_role: dict[str, str] = field(default_factory=dict)
    role_source_by_role: dict[str, list[str]] = field(default_factory=dict)
    role_lifecycle_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_report_json(self) -> dict[str, Any]:
        return {
            "category_plan": self.category_plan,
            "category_plan_entries": self.category_plan_entries,
            "missing_category_roles": self.missing_category_roles,
            "category_plan_warnings": self.category_plan_warnings,
            "role_coverage_summary": self.role_coverage_summary,
            "category_catalog_summary": self.category_catalog_summary,
            "category_plan_source": self.category_plan_source,
            "category_planner_source": self.category_planner_source,
            "category_planner_input_roles": self.category_planner_input_roles,
            "category_planner_output_roles": self.category_planner_output_roles,
            "category_planner_missing_required_roles": (
                self.category_planner_missing_required_roles
            ),
            "category_planner_repair_attempted": self.category_planner_repair_attempted,
            "category_planner_repair_success": self.category_planner_repair_success,
            "category_planner_repair_reason": self.category_planner_repair_reason,
            "category_planner_repaired_roles": self.category_planner_repaired_roles,
            "category_planner_unresolved_required_roles": (
                self.category_planner_unresolved_required_roles
            ),
            "validated_category_plan_roles": self.validated_category_plan_roles,
            "roles_dropped_before_category_planner": (
                self.roles_dropped_before_category_planner
            ),
            "roles_dropped_after_category_planner": (
                self.roles_dropped_after_category_planner
            ),
            "roles_dropped_reason_by_role": self.roles_dropped_reason_by_role,
            "role_source_by_role": self.role_source_by_role,
            "role_lifecycle_trace": self.role_lifecycle_trace,
        }


def build_compact_category_catalog(
    *,
    distributor_code: str,
    category_rows: Sequence[Any] = (),
    product_rows: Sequence[Any] = (),
    product_group: str = "server",
    matrix_roles: Sequence[str] = (),
) -> list[CompactCategory]:
    by_id: dict[str, CompactCategory] = {}
    product_counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for product in product_rows:
        category_id = _object_value(product, "category_id")
        if not category_id:
            continue
        category_id = str(category_id).strip()
        product_counts[category_id] = product_counts.get(category_id, 0) + 1
        sample = _product_sample_name(product)
        if sample:
            samples.setdefault(category_id, [])
            if len(samples[category_id]) < 3 and sample not in samples[category_id]:
                samples[category_id].append(sample)

    for row in category_rows:
        if str(_object_value(row, "distributor_code") or distributor_code) != distributor_code:
            continue
        category_id = str(_object_value(row, "category_id") or "").strip()
        if not category_id:
            continue
        by_id[category_id] = _compact_category_from_parts(
            distributor_code=distributor_code,
            category_id=category_id,
            category_name=str(_object_value(row, "name") or category_id),
            category_path=_category_path_text(_object_value(row, "path_json")),
            parent_id=_text_or_none(
                _object_value(row, "parent_category_id") or _object_value(row, "parent_id")
            ),
            product_count=product_counts.get(category_id),
            sample_product_names=tuple(samples.get(category_id, [])),
            product_group=product_group,
        )

    for category_id in sorted(set(product_counts).difference(by_id)):
        fallback_name = _fallback_category_name(category_id)
        by_id[category_id] = _compact_category_from_parts(
            distributor_code=distributor_code,
            category_id=category_id,
            category_name=fallback_name,
            category_path=fallback_name,
            parent_id=None,
            product_count=product_counts.get(category_id),
            sample_product_names=tuple(samples.get(category_id, [])),
            product_group=product_group,
        )

    categories = sorted(by_id.values(), key=lambda item: (item.category_path, item.category_id))
    roles = _unique([_normalize_role(role) for role in matrix_roles])
    if product_group == "server" and roles:
        return [
            category
            for category in categories
            if _category_relevant_for_product_group_roles(category, product_group, roles)
        ]
    if product_group == "server":
        return categories
    return [
        category
        for category in categories
        if _category_relevant_for_product_group_roles(category, product_group, roles)
    ]


def _compact_category_from_parts(
    *,
    distributor_code: str,
    category_id: str,
    category_name: str,
    category_path: str,
    parent_id: str | None,
    product_count: int | None,
    sample_product_names: tuple[str, ...],
    product_group: str,
) -> CompactCategory:
    anchor = _ocs_anchor_for_category(category_id, product_group, distributor_code)
    inferred = _infer_category_metadata(
        category_id=category_id,
        category_name=category_name,
        category_path=category_path,
        sample_product_names=sample_product_names,
        product_group=product_group,
        anchor=anchor,
    )
    return CompactCategory(
        distributor_code=distributor_code,
        category_id=category_id,
        category_name=category_name,
        category_path=category_path or inferred.get("category_path", ""),
        parent_id=parent_id,
        product_count=product_count,
        sample_product_names=sample_product_names,
        mapped_role=str(inferred.get("mapped_role") or "") or None,
        inferred_category_kind=str(inferred.get("category_kind") or "unknown"),
        allowed_candidate_roles=tuple(_string_list(inferred.get("allowed_roles"))),
        product_group_context=str(inferred.get("product_group_context") or "unknown"),
        confidence=str(inferred.get("confidence") or "low"),
        source=str(inferred.get("source") or "inferred"),
        base_device_allowed=(
            inferred.get("base_device_allowed")
            if isinstance(inferred.get("base_device_allowed"), bool)
            else None
        ),
        review_status=str(inferred.get("review_status") or ""),
    )


def _ocs_anchor_for_category(
    category_id: str,
    product_group: str,
    distributor_code: str,
) -> OcsAnchorCategory | None:
    if distributor_code.strip().casefold() != "ocs":
        return None
    anchors = [
        anchor
        for anchor in load_ocs_anchor_categories()
        if anchor.category_id == category_id and anchor.review_status != "rejected"
    ]
    if not anchors:
        return None
    for anchor in anchors:
        if anchor.group == product_group:
            return anchor
    if product_group == "server":
        for anchor in anchors:
            if anchor.group == "server":
                return anchor
    for anchor in anchors:
        if anchor.group in {"support_license", "accessory"}:
            return anchor
    return anchors[0]


def _infer_category_metadata(
    *,
    category_id: str,
    category_name: str,
    category_path: str,
    sample_product_names: tuple[str, ...],
    product_group: str,
    anchor: OcsAnchorCategory | None,
) -> dict[str, Any]:
    if anchor is not None:
        allowed_roles = list(anchor.allowed_roles or ((anchor.role,) if anchor.role else ()))
        category_kind = anchor.category_kind or _kind_from_roles(allowed_roles)
        return {
            "mapped_role": anchor.role or (allowed_roles[0] if allowed_roles else None),
            "allowed_roles": allowed_roles,
            "category_kind": category_kind,
            "product_group_context": anchor.group,
            "confidence": "high" if anchor.review_status == "approved" else "medium",
            "source": "ocs_anchor_categories",
            "base_device_allowed": anchor.base_device_allowed,
            "review_status": anchor.review_status,
            "category_path": anchor.category_path,
        }

    haystack = _category_haystack(
        category_id=category_id,
        category_name=category_name,
        category_path=category_path,
        sample_product_names=sample_product_names,
    )
    roles = _infer_roles_from_haystack(haystack, product_group)
    category_kind = _kind_from_haystack(haystack, roles)
    return {
        "mapped_role": roles[0] if roles else _fallback_role_for_category(category_id),
        "allowed_roles": roles,
        "category_kind": category_kind,
        "product_group_context": _product_group_from_roles(roles) or product_group,
        "confidence": "medium" if roles else "low",
        "source": "lexical_inference" if roles else "unknown",
        "base_device_allowed": category_kind == "base_device",
        "review_status": "",
        "category_path": category_path,
    }


def _category_visible_for_product_group(
    category: CompactCategory,
    product_group: str,
) -> bool:
    context = str(category.product_group_context or "").strip()
    if context in {product_group, "shared", "support_license", "accessory"}:
        return True
    if context and context != "unknown":
        return False
    profile = get_product_group_profile(product_group)
    allowed_profile_roles = set(profile.role_catalog) if profile is not None else set()
    return bool(set(category.allowed_candidate_roles).intersection(allowed_profile_roles))


def _category_relevant_for_product_group_roles(
    category: CompactCategory,
    product_group: str,
    roles: Sequence[str],
) -> bool:
    if not _category_visible_for_product_group(category, product_group):
        return False
    if not roles:
        return True
    if set(category.allowed_candidate_roles).intersection(roles):
        return True
    return any(_category_score_for_role(role, category) > 0 for role in roles)


def _category_haystack(
    *,
    category_id: str,
    category_name: str,
    category_path: str,
    sample_product_names: tuple[str, ...],
) -> str:
    return " ".join(
        [
            category_id,
            category_name,
            category_path,
            " ".join(sample_product_names),
        ]
    ).casefold()


def _infer_roles_from_haystack(haystack: str, product_group: str) -> list[str]:
    profile = get_product_group_profile(product_group)
    role_ids = list(profile.role_catalog) if profile is not None else []
    if product_group == "server":
        role_ids = list(SERVER_PROFILE_ROLE_CATALOG)
    roles: list[str] = []
    for role in role_ids:
        if _category_text_matches_role(role, haystack):
            roles.append(role)
    if product_group == "storage" and (
        "wifi" in haystack or "wi-fi" in haystack or "wireless" in haystack
    ):
        return [role for role in roles if role not in {"host_port", "protocol_module"}]
    return roles


def _category_text_matches_role(role: str, haystack: str) -> bool:
    if any(synonym.casefold() in haystack for synonym in _role_synonyms(role)):
        return True
    extra_patterns = {
        "switch": r"\bswitch(?:es)?\b|ethernet\s+switch",
        "router": r"\brouter(?:s)?\b|gateway",
        "access_point": r"\baccess\s*point\b|\bwi-?fi\s+ap\b",
        "transceiver": r"\b(?:transceiver|sfp|sfp\+|sfp28|qsfp|optic)\b",
        "dac_cable": r"\b(?:dac|direct\s+attach|twinax)\b",
        "cable": r"\b(?:cable|patch\s*cord|aoc)\b",
        "storage_system": r"\b(?:storage\s+system|storage\s+array|disk\s+array|nas|san)\b",
        "disk_shelf": r"\b(?:disk|drive|expansion)\s+(?:shelf|enclosure)\b|\bjbod\b",
        "drive": r"\b(?:drive|disk|ssd|hdd)\b",
        "ssd": r"\bssd\b",
        "hdd": r"\bhdd\b|nearline|nl-?sas",
        "host_port": r"\b(?:fc|fibre\s+channel|iscsi|nvme-?of|sas)\b",
        "protocol_module": r"\b(?:fc|iscsi|nvme-?of|protocol)\s+module\b",
        "support": r"\b(?:support|service|warranty|maintenance)\b",
        "license": r"\b(?:license|licence|subscription)\b",
        "power_supply": r"\b(?:psu|power\s+supply)\b",
        "rail_kit": r"\brails?\b",
    }
    pattern = extra_patterns.get(role)
    return bool(pattern and re.search(pattern, haystack, re.I))


def _kind_from_roles(roles: Sequence[str]) -> str:
    role_set = set(roles)
    if role_set.intersection(
        {
            "switch",
            "router",
            "firewall",
            "access_point",
            "storage_system",
            "ready_server",
            "server_platform",
        }
    ):
        return "base_device"
    if role_set.intersection({"ssd", "hdd", "drive"}):
        return "drive"
    if "transceiver" in role_set:
        return "transceiver"
    if role_set.intersection({"cable", "dac_cable"}):
        return "cable"
    if "support" in role_set:
        return "support"
    if "license" in role_set:
        return "license"
    if role_set.intersection(
        {"stacking_module", "controller_module", "protocol_module", "host_port"}
    ):
        return "module"
    if role_set:
        return "component"
    return "unknown"


def _kind_from_haystack(haystack: str, roles: Sequence[str]) -> str:
    if re.search(r"\b(?:license|licence|subscription)\b|лиценз|подписк", haystack, re.I):
        return "license"
    if re.search(r"\b(?:support|service|warranty|maintenance)\b|поддержк|гарант", haystack, re.I):
        return "support"
    if re.search(r"\b(?:cable|dac|twinax|patch\s*cord)\b|кабель", haystack, re.I):
        return "cable"
    if set(roles).intersection(_BASE_DEVICE_ROLES):
        return "base_device"
    if re.search(r"\b(?:transceiver|optic|sfp|qsfp)\b|трансивер|оптик", haystack, re.I):
        return "transceiver"
    if re.search(r"\b(?:module|option|accessor|accessory|adapter)\b", haystack, re.I):
        role_kind = _kind_from_roles(roles)
        if role_kind == "base_device":
            return "mixed"
        return "module" if role_kind in {"unknown", "component"} else role_kind
    return _kind_from_roles(roles)


def _product_group_from_roles(roles: Sequence[str]) -> str | None:
    role_set = set(roles)
    if role_set.intersection(NETWORK_PROFILE_ROLE_CATALOG):
        return "network"
    if role_set.intersection(STORAGE_PROFILE_ROLE_CATALOG):
        return "storage"
    if role_set.intersection(SERVER_PROFILE_ROLE_CATALOG):
        return "server"
    return None


def plan_distributor_categories(
    *,
    distributor_code: str,
    product_group: str,
    role_plan: Mapping[str, Any],
    compact_catalog: Sequence[CompactCategory | Mapping[str, Any]],
    llm_client: LlmClient | None = None,
) -> CategoryPlanResult:
    catalog = [_coerce_category(row) for row in compact_catalog]
    required_capabilities = _capabilities(role_plan.get("required_capabilities"))
    optional_capabilities = _capabilities(role_plan.get("optional_capabilities"))
    roles = _planned_roles(role_plan)
    effective_roles = role_lifecycle.unique_roles(
        [
            *roles,
            *_string_list(role_plan.get("effective_matrix_roles_before_category_planner")),
            *_string_list(role_plan.get("category_planner_input_roles")),
        ],
        product_group=product_group,
    )
    roles = effective_roles
    shortlist, shortlist_used = _planner_catalog_for_role_plan(role_plan, roles, catalog)
    planner_catalog = shortlist
    catalog_summary = _category_catalog_summary(
        distributor_code=distributor_code,
        product_group=product_group,
        full_catalog=catalog,
        planner_catalog=planner_catalog,
        shortlist_used=shortlist_used,
    )
    warnings: list[str] = []
    llm_plan: Mapping[str, Any] | None = None
    source = "deterministic"
    planner_source = "fallback_category_planner"
    if llm_client is not None and planner_catalog:
        try:
            response = llm_client.generate_json(
                _category_planner_system_prompt(product_group),
                json.dumps(
                    {
                        "distributor_code": distributor_code,
                        "product_group": product_group,
                        "primary_product_group": role_plan.get("primary_product_group")
                        or product_group,
                        "primary_object": role_plan.get("primary_object"),
                        "matrix_blueprint": role_plan.get("matrix_blueprint")
                        if isinstance(role_plan.get("matrix_blueprint"), Mapping)
                        else {"roles": []},
                        "matrix_blueprint_roles": _string_list(
                            role_plan.get("matrix_blueprint_roles")
                        ),
                        "stage_a_broad_roles": _string_list(
                            role_plan.get("stage_a_broad_roles")
                        ),
                        "semantic_matrix_blueprint_roles": _string_list(
                            role_plan.get("semantic_matrix_blueprint_roles")
                        ),
                        "requirement_classifier_roles": _string_list(
                            role_plan.get("requirement_classifier_roles")
                        ),
                        "effective_matrix_roles_before_category_planner": roles,
                        "category_planner_input_roles": roles,
                        "role_source_by_role": role_plan.get("role_source_by_role")
                        if isinstance(role_plan.get("role_source_by_role"), Mapping)
                        else {},
                        "required_capabilities": required_capabilities,
                        "optional_capabilities": optional_capabilities,
                        "role_plan": _safe_role_plan(role_plan),
                        "category_catalog": [
                            item.to_prompt_json() for item in planner_catalog
                        ],
                        "category_catalog_summary": catalog_summary,
                    },
                    ensure_ascii=False,
                ),
            )
            llm_plan = response if isinstance(response, Mapping) else None
            source = "llm"
            planner_source = "ai_category_planner"
        except (LlmError, ValueError, TypeError) as exc:
            warnings.append(f"category_planner_llm_fallback:{type(exc).__name__}")

    raw_plan = _extract_raw_category_plan(llm_plan) if isinstance(llm_plan, Mapping) else None
    raw_plan = raw_plan if raw_plan is not None else _deterministic_category_plan(
        role_plan,
        roles,
        planner_catalog,
    )
    validation = validate_category_plan(
        category_plan=raw_plan,
        distributor_code=distributor_code,
        product_group=product_group,
        compact_catalog=planner_catalog,
        role_plan=role_plan,
    )
    if validation["rejected"] and not validation["category_plan"]:
        warnings.extend(validation["warnings"])
        fallback = _deterministic_category_plan(role_plan, roles, planner_catalog)
        validation = validate_category_plan(
            category_plan=fallback,
            distributor_code=distributor_code,
            product_group=product_group,
            compact_catalog=planner_catalog,
            role_plan=role_plan,
        )
        source = "deterministic_fallback"
        planner_source = "fallback_category_planner"
        warnings.extend(validation["warnings"])
    else:
        warnings.extend(validation["warnings"])
    required_lifecycle_roles = _required_lifecycle_roles_for_category_plan(
        role_plan=role_plan,
        product_group=product_group,
        category_planner_input_roles=roles,
    )
    category_planner_missing_required_roles = _missing_required_lifecycle_roles(
        validation,
        required_lifecycle_roles=required_lifecycle_roles,
    )
    repair_attempted = False
    repair_success = False
    repair_reason: str | None = None
    repaired_roles: list[str] = []
    unresolved_required_roles = list(category_planner_missing_required_roles)
    if category_planner_missing_required_roles and llm_client is not None:
        repair_attempted = True
        repair_reason = "missing_required_roles"
        repair_catalog = _repair_catalog_for_missing_roles(
            missing_roles=category_planner_missing_required_roles,
            full_catalog=catalog,
            planner_catalog=planner_catalog,
        )
        try:
            repair_response = llm_client.generate_json(
                _category_planner_repair_system_prompt(product_group),
                json.dumps(
                    {
                        "distributor_code": distributor_code,
                        "product_group": product_group,
                        "primary_product_group": role_plan.get("primary_product_group")
                        or product_group,
                        "primary_object": role_plan.get("primary_object"),
                        "missing_required_roles": category_planner_missing_required_roles,
                        "category_planner_input_roles": roles,
                        "required_lifecycle_roles": required_lifecycle_roles,
                        "original_category_planner_output": (
                            llm_plan if isinstance(llm_plan, Mapping) else raw_plan
                        ),
                        "validated_category_plan_before_repair": validation[
                            "category_plan"
                        ],
                        "role_coverage_summary_before_repair": validation[
                            "role_coverage_summary"
                        ],
                        "role_plan": _safe_role_plan(role_plan),
                        "category_catalog": [
                            item.to_prompt_json() for item in repair_catalog
                        ],
                        "category_catalog_summary": {
                            **catalog_summary,
                            "repair_catalog_size": len(repair_catalog),
                            "repair_missing_roles": category_planner_missing_required_roles,
                        },
                    },
                    ensure_ascii=False,
                ),
            )
            repair_raw_plan = (
                _extract_raw_category_plan(repair_response)
                if isinstance(repair_response, Mapping)
                else None
            )
            if repair_raw_plan is None and isinstance(repair_response, list):
                repair_raw_plan = repair_response
            if repair_raw_plan is not None:
                merged_raw_plan = _merge_category_repair_plan(
                    raw_plan,
                    repair_raw_plan,
                    missing_roles=category_planner_missing_required_roles,
                    product_group=product_group,
                )
                repaired_validation = validate_category_plan(
                    category_plan=merged_raw_plan,
                    distributor_code=distributor_code,
                    product_group=product_group,
                    compact_catalog=catalog,
                    role_plan=role_plan,
                )
                repair_repaired_roles = [
                    role
                    for role in category_planner_missing_required_roles
                    if repaired_validation["category_plan"].get(role)
                    and not validation["category_plan"].get(role)
                ]
                repair_unresolved_roles = _missing_required_lifecycle_roles(
                    repaired_validation,
                    required_lifecycle_roles=required_lifecycle_roles,
                )
                warnings.extend(repaired_validation["warnings"])
                if repair_repaired_roles:
                    raw_plan = merged_raw_plan
                    validation = repaired_validation
                    repaired_roles = repair_repaired_roles
                    unresolved_required_roles = repair_unresolved_roles
                    repair_success = not repair_unresolved_roles
                    repair_reason = (
                        "repaired_missing_required_roles"
                        if repair_success
                        else "partially_repaired_missing_required_roles"
                    )
                    if source == "llm":
                        source = "llm_with_repair"
                else:
                    unresolved_required_roles = repair_unresolved_roles
                    repair_reason = _repair_no_category_reason(
                        repair_response,
                        missing_roles=category_planner_missing_required_roles,
                    )
                    warnings.append("category_planner_repair_no_category_found")
            else:
                repair_reason = _repair_no_category_reason(
                    repair_response,
                    missing_roles=category_planner_missing_required_roles,
                )
                warnings.append("category_planner_repair_no_category_plan")
        except (LlmError, ValueError, TypeError) as exc:
            repair_reason = f"repair_error:{type(exc).__name__}"
            warnings.append(repair_reason)
    else:
        unresolved_required_roles = category_planner_missing_required_roles
    if not validation["category_plan"] and not validation["missing_category_roles"]:
        planner_source = "none" if not planner_catalog else planner_source
    output_roles = _raw_category_plan_roles(raw_plan, product_group=product_group)
    validated_roles = role_lifecycle.roles_from_category_plan(
        validation["category_plan"],
        product_group=product_group,
    )
    missing_roles = role_lifecycle.unique_roles(
        validation["missing_category_roles"],
        product_group=product_group,
    )
    roles_dropped_before_category_planner = role_lifecycle.dropped_roles(
        _string_list(role_plan.get("effective_matrix_roles_before_category_planner")),
        roles,
    )
    roles_dropped_after_category_planner = [
        role
        for role in role_lifecycle.dropped_roles(roles, [*validated_roles, *missing_roles])
        if not _coverage_can_be_satisfied_by_platform(
            validation["role_coverage_summary"],
            role,
        )
    ]
    role_source_by_role = role_lifecycle.merge_role_sources(
        (
            role_lifecycle.ROLE_SOURCE_CATEGORY_PLANNER,
            [*output_roles, *validated_roles],
        ),
        existing=role_plan.get("role_source_by_role")
        if isinstance(role_plan.get("role_source_by_role"), Mapping)
        else {},
    )
    reason_by_role = role_lifecycle.merge_drop_reasons(
        {
            role: "missing_from_category_planner_input"
            for role in roles_dropped_before_category_planner
        },
        {role: "missing_category" for role in missing_roles},
        {
            role: "not_returned_by_category_planner"
            for role in roles_dropped_after_category_planner
        },
        existing=role_plan.get("roles_dropped_reason_by_role")
        if isinstance(role_plan.get("roles_dropped_reason_by_role"), Mapping)
        else {},
    )
    return CategoryPlanResult(
        category_plan=validation["category_plan"],
        category_plan_entries=validation["category_plan_entries"],
        missing_category_roles=validation["missing_category_roles"],
        category_plan_warnings=_unique(warnings),
        role_coverage_summary=validation["role_coverage_summary"],
        category_catalog_summary=catalog_summary,
        category_plan_source=source,
        category_planner_source=planner_source,
        category_planner_input_roles=roles,
        category_planner_output_roles=output_roles,
        category_planner_missing_required_roles=category_planner_missing_required_roles,
        category_planner_repair_attempted=repair_attempted,
        category_planner_repair_success=repair_success,
        category_planner_repair_reason=repair_reason,
        category_planner_repaired_roles=repaired_roles,
        category_planner_unresolved_required_roles=unresolved_required_roles,
        validated_category_plan_roles=validated_roles,
        roles_dropped_before_category_planner=roles_dropped_before_category_planner,
        roles_dropped_after_category_planner=roles_dropped_after_category_planner,
        roles_dropped_reason_by_role=reason_by_role,
        role_source_by_role=role_source_by_role,
        role_lifecycle_trace=role_lifecycle.build_role_lifecycle_trace(
            roles,
            role_source_by_role=role_source_by_role,
            stage_a_roles=_string_list(role_plan.get("stage_a_broad_roles")),
            semantic_matrix_blueprint_roles=_string_list(
                role_plan.get("semantic_matrix_blueprint_roles")
            ),
            requirement_classifier_roles=_string_list(
                role_plan.get("requirement_classifier_roles")
            ),
            before_category_planner_roles=_string_list(
                role_plan.get("effective_matrix_roles_before_category_planner")
            ),
            category_planner_input_roles=roles,
            category_planner_output_roles=output_roles,
            validated_category_plan_roles=validated_roles,
            dropped_reason_by_role=reason_by_role,
        ),
    )


def validate_category_plan(
    *,
    category_plan: Mapping[str, Any] | list[Any],
    distributor_code: str,
    product_group: str,
    compact_catalog: Sequence[CompactCategory | Mapping[str, Any]],
    role_plan: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = [_coerce_category(row) for row in compact_catalog]
    catalog_by_id = {item.category_id: item for item in catalog}
    profile = get_product_group_profile(product_group)
    allowed_roles = set(profile.role_catalog) if profile is not None else set()
    if not allowed_roles and product_group == "server":
        allowed_roles = set(SERVER_PROFILE_ROLE_CATALOG)
    allowed_roles.add(UNMAPPED_ROLE)
    cleaned: dict[str, list[str]] = {}
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    rejected = False
    for raw_row in _iter_raw_category_plan_rows(category_plan):
        raw_role_text = str(raw_row.get("role") or "").strip()
        role_text = _normalize_role(raw_role_text)
        if product_group == "storage" and raw_role_text == "storage":
            role_text = "storage_system"
        if role_text not in allowed_roles:
            warnings.append(f"category_plan_role_not_allowed:{role_text}")
            rejected = True
            continue
        raw_ids = raw_row.get("selected_category_ids")
        ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
        selected_ids: list[str] = []
        for category_id_value in ids:
            category_id = str(category_id_value or "").strip()
            if not category_id:
                continue
            category = catalog_by_id.get(category_id)
            if category is None:
                warnings.append(f"category_plan_id_not_in_catalog:{category_id}")
                rejected = True
                continue
            if category.distributor_code != distributor_code:
                warnings.append(f"category_plan_wrong_distributor:{category_id}")
                rejected = True
                continue
            compatibility_reason = _category_compatibility_rejection(
                category=category,
                role=role_text,
                product_group=product_group,
                purpose=str(raw_row.get("purpose") or "").strip(),
            )
            if compatibility_reason:
                warnings.append(
                    "category_plan_category_incompatible:"
                    f"{category_id}:{role_text}:{compatibility_reason}"
                )
                rejected = True
                continue
            cleaned.setdefault(role_text, [])
            if category_id not in cleaned[role_text]:
                cleaned[role_text].append(category_id)
            if category_id not in selected_ids:
                selected_ids.append(category_id)
        if selected_ids:
            entries.append(
                {
                    "capability_id": str(raw_row.get("capability_id") or "").strip()
                    or _capability_id_for_role(
                        role_plan,
                        role_text,
                    ),
                    "role": role_text,
                    "selected_category_ids": selected_ids,
                    "purpose": _normalized_purpose(
                        str(raw_row.get("purpose") or "").strip(),
                        role_text,
                    ),
                    "capability_ids": _string_list(raw_row.get("capability_ids")),
                    "hard_optional_relation": str(
                        raw_row.get("hard_optional_relation") or ""
                    ).strip(),
                    "reason": str(raw_row.get("reason") or "").strip(),
                    "confidence": str(raw_row.get("confidence") or "").strip(),
                }
            )

    required_roles = _required_roles_for_validation(role_plan, allowed_roles)
    coverage: dict[str, Any] = {}
    missing_category_roles: list[str] = []
    for role in required_roles:
        ids = cleaned.get(role, [])
        can_be_satisfied_by_platform = _role_can_be_satisfied_by_platform(
            role_plan,
            role,
        )
        missing_category = not ids and not can_be_satisfied_by_platform
        if missing_category:
            missing_category_roles.append(role)
        coverage[role] = {
            "required": True,
            "category_ids": ids,
            "category_count": len(ids),
            "missing_category": missing_category,
            "can_be_satisfied_by_platform": can_be_satisfied_by_platform,
        }
    return {
        "category_plan": cleaned,
        "category_plan_entries": entries,
        "missing_category_roles": missing_category_roles,
        "warnings": _unique(warnings),
        "rejected": rejected,
        "role_coverage_summary": coverage,
    }


def _required_lifecycle_roles_for_category_plan(
    *,
    role_plan: Mapping[str, Any],
    product_group: str,
    category_planner_input_roles: Sequence[str],
) -> list[str]:
    profile = get_product_group_profile(product_group)
    allowed_roles = set(profile.role_catalog) if profile is not None else set()
    allowed_roles.add(UNMAPPED_ROLE)
    input_roles = role_lifecycle.unique_roles(
        list(category_planner_input_roles),
        product_group=product_group,
    )
    input_role_set = set(input_roles)
    roles: list[str] = []
    for role in _string_list(role_plan.get("required_roles")):
        normalized = role_lifecycle.normalize_role(role, product_group=product_group)
        if normalized in allowed_roles or normalized == UNMAPPED_ROLE:
            roles.append(normalized)
    for capability in _capabilities(role_plan.get("required_capabilities")):
        if not capability.get("hard", True):
            continue
        normalized = role_lifecycle.normalize_role(
            capability.get("role"),
            product_group=product_group,
        )
        if normalized in allowed_roles or normalized == UNMAPPED_ROLE:
            roles.append(normalized)
    if profile is not None:
        for role in profile.required_roles:
            if role in input_role_set:
                roles.append(role)
    return role_lifecycle.unique_roles(roles, product_group=product_group)


def _missing_required_lifecycle_roles(
    validation: Mapping[str, Any],
    *,
    required_lifecycle_roles: Sequence[str],
) -> list[str]:
    category_plan = validation.get("category_plan")
    category_plan = category_plan if isinstance(category_plan, Mapping) else {}
    coverage = validation.get("role_coverage_summary")
    coverage = coverage if isinstance(coverage, Mapping) else {}
    missing: list[str] = []
    for role in required_lifecycle_roles:
        if _coverage_can_be_satisfied_by_platform(coverage, role):
            continue
        if _category_plan_has_role(category_plan, role):
            continue
        row = coverage.get(role)
        if isinstance(row, Mapping) and row.get("missing_category") is False:
            continue
        missing.append(role)
    return role_lifecycle.unique_roles(missing)


def _category_plan_has_role(category_plan: Mapping[str, Any], role: str) -> bool:
    aliases = [role]
    if role == "storage":
        aliases.extend(["drive", "ssd", "hdd"])
    if role in {"drive", "ssd", "hdd"}:
        aliases.append("storage")
    for alias in _unique(aliases):
        ids = category_plan.get(alias)
        if isinstance(ids, list) and any(str(item or "").strip() for item in ids):
            return True
    return False


def _repair_catalog_for_missing_roles(
    *,
    missing_roles: Sequence[str],
    full_catalog: Sequence[CompactCategory],
    planner_catalog: Sequence[CompactCategory],
) -> list[CompactCategory]:
    selected = {category.category_id: category for category in planner_catalog}
    for category in _shortlisted_catalog(missing_roles, full_catalog):
        selected[category.category_id] = category
    return sorted(selected.values(), key=lambda item: item.category_id)


def _merge_category_repair_plan(
    original_plan: Mapping[str, Any] | list[Any],
    repair_plan: Mapping[str, Any] | list[Any],
    *,
    missing_roles: Sequence[str],
    product_group: str,
) -> list[dict[str, Any]]:
    missing = set(
        role_lifecycle.unique_roles(list(missing_roles), product_group=product_group)
    )
    rows = [
        row
        for row in _iter_raw_category_plan_rows(original_plan)
        if role_lifecycle.normalize_role(
            row.get("role"),
            product_group=product_group,
        )
        not in missing
    ]
    for row in _iter_raw_category_plan_rows(repair_plan):
        role = role_lifecycle.normalize_role(
            row.get("role"),
            product_group=product_group,
        )
        if role not in missing:
            continue
        raw_ids = row.get("selected_category_ids")
        ids = raw_ids if isinstance(raw_ids, list) else [raw_ids]
        if not any(str(category_id or "").strip() for category_id in ids):
            continue
        rows.append(row)
    return rows


def _repair_no_category_reason(
    response: Any,
    *,
    missing_roles: Sequence[str],
) -> str:
    response_map = response if isinstance(response, Mapping) else {}
    reason_rows = response_map.get("no_category_found")
    if not isinstance(reason_rows, list):
        reason_rows = response_map.get("missing_category_roles")
    reasons: list[str] = []
    if isinstance(reason_rows, list):
        for row in reason_rows:
            if isinstance(row, Mapping):
                role = str(row.get("role") or "").strip()
                reason = str(row.get("reason") or "").strip()
                if role or reason:
                    reasons.append(f"{role}:{reason}".strip(":"))
            else:
                text = str(row or "").strip()
                if text:
                    reasons.append(text)
    if reasons:
        return "no_category_found:" + ",".join(reasons)
    return "no_category_found:" + ",".join(_string_list(list(missing_roles)))


def _shortlisted_catalog(
    roles: Sequence[str],
    catalog: Sequence[CompactCategory],
) -> list[CompactCategory]:
    selected: dict[str, CompactCategory] = {}
    for role in roles:
        scored = sorted(
            (
                (_category_score_for_role(role, category), category)
                for category in catalog
            ),
            key=lambda item: (-item[0], item[1].category_id),
        )
        role_matches = [category for score, category in scored if score > 0]
        if not role_matches:
            role_matches = [category for category in catalog if category.mapped_role == role]
        for category in role_matches[:MAX_CATEGORIES_PER_ROLE]:
            selected[category.category_id] = category
    return sorted(selected.values(), key=lambda item: item.category_id)


def _planner_catalog_for_roles(
    roles: Sequence[str],
    catalog: Sequence[CompactCategory],
) -> tuple[list[CompactCategory], bool]:
    if len(catalog) <= MAX_FULL_CATALOG_FOR_LLM:
        return list(catalog), False
    return _shortlisted_catalog(roles, catalog), True


def _planner_catalog_for_role_plan(
    role_plan: Mapping[str, Any],
    roles: Sequence[str],
    catalog: Sequence[CompactCategory],
) -> tuple[list[CompactCategory], bool]:
    if len(catalog) <= MAX_FULL_CATALOG_FOR_LLM:
        return list(catalog), False
    selected = {
        category.category_id: category for category in _shortlisted_catalog(roles, catalog)
    }
    for category in _shortlisted_catalog_for_capabilities(
        [
            *_capabilities(role_plan.get("required_capabilities")),
            *_capabilities(role_plan.get("optional_capabilities")),
        ],
        catalog,
    ):
        selected[category.category_id] = category
    return sorted(selected.values(), key=lambda item: item.category_id), True


def _shortlisted_catalog_for_capabilities(
    capabilities: Sequence[Mapping[str, Any]],
    catalog: Sequence[CompactCategory],
) -> list[CompactCategory]:
    selected: dict[str, CompactCategory] = {}
    for capability in capabilities:
        scored = sorted(
            (
                (_category_score_for_capability(capability, category), category)
                for category in catalog
            ),
            key=lambda item: (-item[0], item[1].category_id),
        )
        for score, category in scored[:MAX_CATEGORIES_PER_ROLE]:
            if score <= 0:
                continue
            selected[category.category_id] = category
    return sorted(selected.values(), key=lambda item: item.category_id)


def _category_catalog_summary(
    *,
    distributor_code: str,
    product_group: str,
    full_catalog: Sequence[CompactCategory],
    planner_catalog: Sequence[CompactCategory],
    shortlist_used: bool,
) -> dict[str, Any]:
    kind_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    for category in planner_catalog:
        kind = category.inferred_category_kind or "unknown"
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        for role in category.allowed_candidate_roles:
            role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "distributor_code": distributor_code,
        "product_group": product_group,
        "total_categories": len(full_catalog),
        "catalog_sent_to_planner": len(planner_catalog),
        "shortlist_used": shortlist_used,
        "shortlist_strategy": "lexical_role_retrieval" if shortlist_used else "full_catalog",
        "category_kind_counts": dict(sorted(kind_counts.items())),
        "allowed_role_counts": dict(sorted(role_counts.items())),
    }


def _deterministic_category_plan(
    role_plan: Mapping[str, Any],
    roles: Sequence[str],
    catalog: Sequence[CompactCategory],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    capabilities_by_role: dict[str, list[dict[str, Any]]] = {}
    for capability in [
        *_capabilities(role_plan.get("required_capabilities")),
        *_capabilities(role_plan.get("optional_capabilities")),
    ]:
        role = _normalize_role(str(capability.get("role") or ""))
        if role:
            capabilities_by_role.setdefault(role, []).append(capability)
    for role in roles:
        scored = [
            (
                (
                    max(
                        [
                            _category_score_for_capability(capability, category)
                            for capability in capabilities_by_role.get(role, [])
                        ]
                        or [0]
                    )
                    if role == UNMAPPED_ROLE
                    else _category_score_for_role(role, category)
                ),
                category,
            )
            for category in catalog
        ]
        scored = [(score, category) for score, category in scored if score > 0]
        scored.sort(key=lambda item: (-item[0], item[1].category_id))
        ids = [category.category_id for _, category in scored[:MAX_CATEGORIES_PER_ROLE]]
        if ids:
            result[role] = ids
    return result


def _extract_raw_category_plan(plan: Mapping[str, Any]) -> Mapping[str, Any] | list[Any] | None:
    raw = plan.get("category_plan")
    if isinstance(raw, Mapping | list):
        return raw
    return None


def _raw_category_plan_roles(
    category_plan: Mapping[str, Any] | list[Any],
    *,
    product_group: str,
) -> list[str]:
    return role_lifecycle.unique_roles(
        [
            row.get("role") or row.get("role_id")
            for row in _iter_raw_category_plan_rows(category_plan)
        ],
        product_group=product_group,
    )


def _iter_raw_category_plan_rows(
    category_plan: Mapping[str, Any] | list[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(category_plan, Mapping):
        for role, raw_ids in category_plan.items():
            rows.append({"role": str(role), "selected_category_ids": raw_ids})
        return rows
    if isinstance(category_plan, list):
        for item in category_plan:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "").strip()
            raw_ids = item.get("selected_category_ids")
            if raw_ids is None:
                raw_ids = item.get("category_ids")
            rows.append(
                {
                    "role": role,
                    "selected_category_ids": raw_ids or [],
                    "purpose": str(item.get("purpose") or "unknown").strip(),
                    "capability_id": str(item.get("capability_id") or "").strip(),
                    "capability_ids": item.get("capability_ids") or [],
                    "hard_optional_relation": str(
                        item.get("hard_optional_relation")
                        or item.get("hard_optional")
                        or ""
                    ).strip(),
                    "reason": str(item.get("reason") or "").strip(),
                    "confidence": str(item.get("confidence") or "").strip(),
                }
            )
    return rows


def _coverage_can_be_satisfied_by_platform(
    role_coverage_summary: Mapping[str, Any],
    role: str,
) -> bool:
    row = role_coverage_summary.get(role)
    if not isinstance(row, Mapping):
        return False
    return bool(row.get("can_be_satisfied_by_platform"))


def _category_compatibility_rejection(
    *,
    category: CompactCategory,
    role: str,
    product_group: str,
    purpose: str,
) -> str | None:
    if role == UNMAPPED_ROLE:
        return None
    context = str(category.product_group_context or "").strip()
    if context and context not in {
        "unknown",
        product_group,
        "shared",
        "support_license",
        "accessory",
    }:
        return f"product_group_context_{context}"
    allowed_roles = set(category.allowed_candidate_roles)
    if allowed_roles and role not in allowed_roles:
        return "role_not_allowed_by_category_metadata"
    if not allowed_roles and _category_score_for_role(role, category) <= 0:
        return "role_not_supported_by_category_text"

    normalized_purpose = _normalized_purpose(purpose, role)
    kind = str(category.inferred_category_kind or "unknown").strip()
    if role in _BASE_DEVICE_ROLES and normalized_purpose == "base_device":
        if category.base_device_allowed is False:
            return f"{kind}_not_allowed_as_base_device"
        if kind in {"accessory", "support", "license", "drive", "cable", "transceiver", "module"}:
            return f"{kind}_not_base_device"
    if role == "support" and kind not in {"support", "license", "mixed", "unknown"}:
        return f"{kind}_not_support"
    if role == "license" and kind not in {"license", "support", "mixed", "unknown"}:
        return f"{kind}_not_license"
    return None


_BASE_DEVICE_ROLES = {
    "ready_server",
    "server_platform",
    "switch",
    "router",
    "firewall",
    "access_point",
    "storage_system",
}


def _normalized_purpose(purpose: str, role: str) -> str:
    value = purpose.strip() or "unknown"
    if value == "unknown" and role in _BASE_DEVICE_ROLES:
        return "base_device"
    if value in {
        "base_device",
        "component",
        "module",
        "accessory",
        "support",
        "license",
        "drive",
        "cable",
        "transceiver",
        "unknown",
    }:
        return value
    return "unknown"


def _capabilities(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _capability_id_for_role(role_plan: Mapping[str, Any], role: str) -> str:
    for capability in _capabilities(role_plan.get("required_capabilities")):
        if _normalize_role(str(capability.get("role") or "")) == role:
            return str(capability.get("capability_id") or "").strip()
    return f"{role}.required"


def _category_score_for_role(role: str, category: CompactCategory) -> int:
    if role == UNMAPPED_ROLE:
        return 0
    score = 0
    kind = str(category.inferred_category_kind or "unknown")
    if role in _BASE_DEVICE_ROLES and kind in {
        "accessory",
        "support",
        "license",
        "drive",
        "cable",
        "transceiver",
        "module",
    }:
        return 0
    if kind == "base_device" and role not in _BASE_DEVICE_ROLES:
        return 0
    if kind == "transceiver" and role != "transceiver":
        return 0
    if kind == "cable" and role not in {"cable", "dac_cable"}:
        return 0
    if kind == "support" and role != "support":
        return 0
    if kind == "license" and role != "license":
        return 0
    if kind == "drive" and role not in {"drive", "ssd", "hdd", "storage"}:
        return 0
    allowed_roles = set(category.allowed_candidate_roles)
    if allowed_roles:
        if role not in allowed_roles:
            return 0
        score += 100
    elif category.mapped_role:
        if category.mapped_role == role or (
            role == "storage" and category.mapped_role in {"ssd", "hdd"}
        ) or (
            role == "drive" and category.mapped_role in {"ssd", "hdd", "storage"}
        ):
            score += 100
        else:
            return 0
    haystack = " ".join(
        [
            category.category_id,
            category.category_name,
            category.category_path,
            " ".join(category.sample_product_names),
        ]
    ).casefold()
    for synonym in _role_synonyms(role):
        if synonym.casefold() in haystack:
            score += 20
    if score and category.product_count:
        score += min(category.product_count, 20)
    return score


def _category_score_for_capability(
    capability: Mapping[str, Any],
    category: CompactCategory,
) -> int:
    query = _capability_search_text(capability)
    if not query:
        return 0
    haystack = " ".join(
        [
            category.category_id,
            category.category_name,
            category.category_path,
            " ".join(category.sample_product_names),
        ]
    ).casefold()
    score = 0
    for token in _search_tokens(query):
        if token in haystack:
            score += 12
    role = _normalize_role(str(capability.get("role") or ""))
    if role and role != UNMAPPED_ROLE:
        role_score = _category_score_for_role(role, category)
        if role_score:
            score += min(role_score, 40)
    if score and category.product_count:
        score += min(category.product_count, 20)
    return score


def _capability_search_text(capability: Mapping[str, Any]) -> str:
    parsed = capability.get("parsed_requirements")
    parsed_values = []
    if isinstance(parsed, Mapping):
        parsed_values = [
            str(value)
            for value in parsed.values()
            if isinstance(value, str | int | float)
        ]
    return " ".join(
        part
        for part in [
            str(capability.get("category_search_intent") or ""),
            str(capability.get("source_text") or ""),
            str(capability.get("requirement_text") or ""),
            str(capability.get("capability_id") or ""),
            " ".join(parsed_values),
        ]
        if part
    ).casefold()


def _search_tokens(value: str) -> list[str]:
    tokens = [
        token
        for token in value.replace("_", " ").replace(".", " ").replace("-", " ").split()
        if len(token) >= 3
    ]
    return _unique(tokens)


def _role_synonyms(role: str) -> tuple[str, ...]:
    profile_role = SERVER_PROFILE_ROLE_CATALOG.get(role)
    if profile_role is None:
        profile_role = NETWORK_PROFILE_ROLE_CATALOG.get(role)
    if profile_role is None:
        profile_role = STORAGE_PROFILE_ROLE_CATALOG.get(role)
    profile_synonyms = profile_role.synonyms if profile_role is not None else ()
    return tuple(
        _unique(
            [
                role,
                *ROLE_SYNONYMS.get(role, ()),
                *profile_synonyms,
            ]
        )
    )


def _category_planner_system_prompt(product_group: str) -> str:
    return (
        f"You are Distributor Category Planner for product_group={product_group}. "
        "Return JSON only. Use primary_product_group, matrix_blueprint.roles, "
        "required_capabilities and optional_capabilities from the package; do not "
        "infer a top-level product group from isolated raw-text keywords and do not "
        "re-extract hard requirements from raw user text. "
        "Select category_id values only from category_catalog. Do not invent categories. "
        "Use classified_requirements to distinguish separate purchasable roles from "
        "primary_object_feature, engineering_check, logistics/commercial, and "
        "out_of_scope non-blocking requirements. Do not select or require a category "
        "for primary object features such as platform/device ports, cooling, bays, "
        "PoE/L3/stacking, RAID/capacity, UPS runtime, or similar feature constraints. "
        "Map each required capability to distributor-specific categories. If a hard role "
        "has no suitable category, return an empty selected_category_ids list for that "
        "capability and include the role in missing_category_roles. Role must be a known "
        "role from role_catalog, or safe role=\"unmapped\" when the Requirement Planner "
        "preserved an unmapped hard capability. For every selected category include "
        "purpose one of base_device, component, module, accessory, support, license, "
        "drive, cable, transceiver, unknown; capability_ids it may help satisfy; "
        "hard_optional_relation; reason; and confidence. "
        "Shape: {\"category_plan\":[{\"capability_id\":\"...\",\"role\":\"...\","
        "\"selected_category_ids\":[\"...\"],\"purpose\":\"base_device\","
        "\"capability_ids\":[\"...\"],\"hard_optional_relation\":\"hard\","
        "\"reason\":\"...\",\"confidence\":\"medium\"}],"
        "\"missing_category_roles\":[],\"category_plan_warnings\":[]}."
    )


def _category_planner_repair_system_prompt(product_group: str) -> str:
    return (
        f"You are Distributor Category Planner Repair for product_group={product_group}. "
        "Return JSON only. Repair only missing_required_roles from the package. "
        "Choose category_id values only from category_catalog; never invent category IDs. "
        "For each missing required lifecycle role, either select real category IDs that "
        "belong to that role or explicitly return no_category_found with a concise reason. "
        "Do not change roles that already have selected categories in "
        "validated_category_plan_before_repair. Do not add vendor, SKU, or distributor-"
        "specific patches. Shape: {\"category_plan\":[{\"role\":\"...\","
        "\"selected_category_ids\":[\"...\"],\"purpose\":\"base_device\","
        "\"capability_ids\":[\"...\"],\"hard_optional_relation\":\"hard\","
        "\"reason\":\"...\",\"confidence\":\"medium\"}],"
        "\"no_category_found\":[{\"role\":\"...\",\"reason\":\"...\"}],"
        "\"category_plan_warnings\":[]}."
    )


def _planned_roles(role_plan: Mapping[str, Any]) -> list[str]:
    roles = [
        *_string_list(role_plan.get("effective_matrix_roles_before_category_planner")),
        *_string_list(role_plan.get("category_planner_input_roles")),
        *_string_list(role_plan.get("required_roles")),
        *_string_list(role_plan.get("optional_roles")),
    ]
    for capability in _capabilities(role_plan.get("required_capabilities")):
        roles.append(str(capability.get("role") or "").strip())
    for capability in _capabilities(role_plan.get("optional_capabilities")):
        roles.append(str(capability.get("role") or "").strip())
    return _unique([_normalize_role(role) for role in roles])


def _safe_role_plan(role_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "product_group": role_plan.get("product_group"),
        "primary_product_group": role_plan.get("primary_product_group"),
        "primary_object": role_plan.get("primary_object"),
        "semantic_planner_source": role_plan.get("semantic_planner_source"),
        "semantic_planner_used": role_plan.get("semantic_planner_used"),
        "semantic_planner_confidence": role_plan.get("semantic_planner_confidence"),
        "semantic_planner_error_type": role_plan.get("semantic_planner_error_type"),
        "semantic_planner_http_status": role_plan.get("semantic_planner_http_status"),
        "semantic_planner_parse_status": role_plan.get(
            "semantic_planner_parse_status"
        ),
        "semantic_planner_fallback_reason": role_plan.get(
            "semantic_planner_fallback_reason"
        ),
        "semantic_planner_model": role_plan.get("semantic_planner_model"),
        "semantic_planner_provider": role_plan.get("semantic_planner_provider"),
        "selected_product_group_reason": role_plan.get("selected_product_group_reason"),
        "matrix_blueprint": role_plan.get("matrix_blueprint")
        if isinstance(role_plan.get("matrix_blueprint"), Mapping)
        else {"roles": []},
        "matrix_blueprint_roles": _string_list(role_plan.get("matrix_blueprint_roles")),
        "stage_a_broad_roles": _string_list(role_plan.get("stage_a_broad_roles")),
        "semantic_matrix_blueprint_roles": _string_list(
            role_plan.get("semantic_matrix_blueprint_roles")
        ),
        "requirement_classifier_roles": _string_list(
            role_plan.get("requirement_classifier_roles")
        ),
        "effective_matrix_roles_before_category_planner": _string_list(
            role_plan.get("effective_matrix_roles_before_category_planner")
        ),
        "category_planner_input_roles": _string_list(
            role_plan.get("category_planner_input_roles")
        ),
        "role_source_by_role": role_plan.get("role_source_by_role")
        if isinstance(role_plan.get("role_source_by_role"), Mapping)
        else {},
        "embedded_requirements": role_plan.get("embedded_requirements")
        if isinstance(role_plan.get("embedded_requirements"), list)
        else [],
        "classified_requirements": role_plan.get("classified_requirements")
        if isinstance(role_plan.get("classified_requirements"), list)
        else [],
        "primary_object_feature_requirements": role_plan.get(
            "primary_object_feature_requirements"
        )
        if isinstance(role_plan.get("primary_object_feature_requirements"), list)
        else [],
        "engineering_check_requirements": role_plan.get(
            "engineering_check_requirements"
        )
        if isinstance(role_plan.get("engineering_check_requirements"), list)
        else [],
        "not_primary_product_groups": role_plan.get("not_primary_product_groups")
        if isinstance(role_plan.get("not_primary_product_groups"), list)
        else [],
        "required_capabilities": role_plan.get("required_capabilities")
        if isinstance(role_plan.get("required_capabilities"), list)
        else [],
        "optional_capabilities": role_plan.get("optional_capabilities")
        if isinstance(role_plan.get("optional_capabilities"), list)
        else [],
        "required_roles": _string_list(role_plan.get("required_roles")),
        "optional_roles": _string_list(role_plan.get("optional_roles")),
        "requirements_by_role": role_plan.get("requirements_by_role")
        if isinstance(role_plan.get("requirements_by_role"), Mapping)
        else {},
        "role_catalog": _string_list(role_plan.get("role_catalog")),
    }


def _required_roles_for_validation(
    role_plan: Mapping[str, Any],
    allowed_roles: set[str],
) -> list[str]:
    roles = [
        role
        for role in _string_list(role_plan.get("required_roles"))
        if role in allowed_roles
    ]
    for capability in _capabilities(role_plan.get("required_capabilities")):
        if not capability.get("hard", True):
            continue
        role = _normalize_role(str(capability.get("role") or ""))
        if role in allowed_roles or role == UNMAPPED_ROLE:
            roles.append(role)
    product_group = str(
        role_plan.get("product_group") or role_plan.get("primary_product_group") or ""
    ).strip()
    profile = get_product_group_profile(product_group)
    if profile is not None:
        input_roles = set(
            role_lifecycle.unique_roles(
                [
                    *_string_list(role_plan.get("category_planner_input_roles")),
                    *_string_list(
                        role_plan.get("effective_matrix_roles_before_category_planner")
                    ),
                ],
                product_group=product_group,
            )
        )
        for role in profile.required_roles:
            if role in allowed_roles and role in input_roles:
                roles.append(role)
    return _unique(roles)


def _role_can_be_satisfied_by_platform(role_plan: Mapping[str, Any], role: str) -> bool:
    matching = [
        capability
        for capability in _capabilities(role_plan.get("required_capabilities"))
        if _normalize_role(str(capability.get("role") or "")) == role
        and capability.get("hard", True)
    ]
    if not matching:
        return False
    return all(bool(capability.get("can_be_satisfied_by_platform")) for capability in matching)


def _coerce_category(row: CompactCategory | Mapping[str, Any]) -> CompactCategory:
    if isinstance(row, CompactCategory):
        return row
    return CompactCategory(
        distributor_code=str(row.get("distributor_code") or ""),
        category_id=str(row.get("category_id") or ""),
        category_name=str(row.get("category_name") or row.get("name") or ""),
        category_path=str(row.get("category_path") or row.get("path") or ""),
        parent_id=_text_or_none(row.get("parent_id") or row.get("parent_category_id")),
        product_count=_int_value(row.get("product_count")),
        sample_product_names=tuple(_string_list(row.get("sample_product_names"))),
        mapped_role=str(row.get("mapped_role") or "") or None,
        inferred_category_kind=str(
            row.get("inferred_category_kind") or row.get("category_kind") or "unknown"
        ),
        allowed_candidate_roles=tuple(_string_list(row.get("allowed_candidate_roles"))),
        product_group_context=str(row.get("product_group_context") or "unknown"),
        confidence=str(row.get("confidence") or "low"),
        source=str(row.get("source") or "inferred"),
        base_device_allowed=(
            row.get("base_device_allowed")
            if isinstance(row.get("base_device_allowed"), bool)
            else None
        ),
        review_status=str(row.get("review_status") or ""),
    )


def _fallback_role_for_category(category_id: str) -> str | None:
    for category in load_server_category_profile():
        if category.category_id == category_id:
            return category.role
    return None


def _fallback_category_name(category_id: str) -> str:
    for category in load_server_category_profile():
        if category.category_id == category_id:
            return category.name_ru
    return category_id


def _object_value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _product_sample_name(product: Any) -> str:
    values = [
        _object_value(product, "item_name"),
        _object_value(product, "product_name"),
        _object_value(product, "part_number"),
    ]
    return " ".join(str(value).strip() for value in values if str(value or "").strip())[:160]


def _normalize_role(role: str) -> str:
    text = str(role or "").strip()
    normalized = text.casefold().replace("-", "_").replace(" ", "_")
    if normalized == "platform":
        return "server_platform"
    if normalized == "license/support":
        return "support"
    if normalized in {"storage_array", "system"}:
        return "storage_system"
    if normalized in {"shelf", "shelves", "drive_shelf", "expansion_shelf"}:
        return "disk_shelf"
    if normalized in {"drives", "disks"}:
        return "drive"
    if normalized in {"host_ports", "ports", "host_interface"}:
        return "host_port"
    if normalized in {"protocol", "protocol_adapter", "interface_module"}:
        return "protocol_module"
    if normalized in {"power_cable", "power_cord", "power_cords", "c13_c14", "c13_schuko"}:
        return "cable"
    if normalized in {"unknown", "unmapped"}:
        return UNMAPPED_ROLE
    return normalized or text


def _text_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _category_path_text(path_json: Any) -> str:
    if isinstance(path_json, list):
        names = [
            str(item.get("name") or item.get("category_id") or "").strip()
            for item in path_json
            if isinstance(item, Mapping)
        ]
        return " / ".join(name for name in names if name)
    if isinstance(path_json, str):
        return path_json
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
