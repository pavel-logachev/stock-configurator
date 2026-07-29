from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class V3FullCategoryProfile:
    name: str
    category_ids: tuple[str, ...]
    description: str


# These profiles are transport presets for the production bot. They define which
# complete stocked/priced category matrix is sent to the LLM; they do not rank or
# filter products inside the selected categories.
V3_FULL_CATEGORY_PROFILES: dict[str, V3FullCategoryProfile] = {
    "network": V3FullCategoryProfile(
        name="network",
        category_ids=("V120100",),
        description="Network switches",
    ),
    "storage": V3FullCategoryProfile(
        name="storage",
        category_ids=(
            "V2101",
            "V2103",
            "V2105",
            "V2104",
            "V3100",
            "V3104",
            "V110106",
            "V110112",
        ),
        description="Storage, NAS and storage drives",
    ),
    "server": V3FullCategoryProfile(
        name="server",
        category_ids=(
            "V1100",
            "V110100",
            "V110103",
            "V110104",
            "V110106",
            "V110112",
            "V110107",
            "V110108",
            "V110109",
            "V120116",
        ),
        description="Servers and server components",
    ),
}

V3_FULL_CATEGORY_PROFILE_ALIASES: dict[str, str] = {
    "lan": "network",
    "network": "network",
    "switch": "network",
    "switches": "network",
    "net": "network",
    "nas": "storage",
    "storage": "storage",
    "shd": "storage",
    "схд": "storage",
    "хранилище": "storage",
    "server": "server",
    "servers": "server",
    "srv": "server",
    "сервер": "server",
    "серверы": "server",
}


def resolve_v3_full_category_profile(
    *,
    profile: str | None = None,
    category_ids: Sequence[str] | None = None,
) -> tuple[str | None, list[str]]:
    cleaned_category_ids = _clean_category_ids(category_ids or [])
    if cleaned_category_ids:
        return _clean_profile_name(profile), cleaned_category_ids

    profile_name = _clean_profile_name(profile)
    if profile_name is None:
        raise ValueError("Either profile or category_ids must be provided.")

    resolved_profile = V3_FULL_CATEGORY_PROFILES.get(profile_name)
    if resolved_profile is None:
        known_profiles = ", ".join(sorted(V3_FULL_CATEGORY_PROFILES))
        raise ValueError(
            f"Unknown v3 full-category profile: {profile_name}. Known: {known_profiles}."
        )

    return resolved_profile.name, list(resolved_profile.category_ids)


def _clean_profile_name(profile: str | None) -> str | None:
    value = str(profile or "").strip().lower()
    if not value:
        return None
    return V3_FULL_CATEGORY_PROFILE_ALIASES.get(value, value)


def _clean_category_ids(category_ids: Sequence[str]) -> list[str]:
    return [
        category_id
        for category_id in dict.fromkeys(str(value or "").strip() for value in category_ids)
        if category_id
    ]
