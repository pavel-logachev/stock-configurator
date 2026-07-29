from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select

from app.core.database import get_session_factory
from app.db.models import DistributorCategory, DistributorProduct

DISTRIBUTOR_CODE = "ocs"
DISCOVERY_GROUPS = ("server", "network", "storage", "support_license", "accessory")


@dataclass(frozen=True)
class AnchorCandidate:
    group: str
    suggested_role: str
    category_id: str
    category_name_path: str
    parent_id: str
    product_count: int | None
    matched_terms: tuple[str, ...]


SEARCH_TERMS: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "server": {
        "ready_server": ("server", "сервер", "готовые серверы"),
        "server_platform": ("server platform", "barebone", "chassis", "платформа"),
        "cpu": ("cpu", "processor", "xeon", "epyc", "процессор"),
        "ram": ("ram", "memory", "rdimm", "ddr", "память"),
        "ssd": ("ssd", "nvme", "u.2", "u.3"),
        "hdd": ("hdd", "hard disk", "жесткий диск"),
        "storage_controller": ("raid", "hba", "storage controller", "контроллер"),
        "network_adapter": ("network adapter", "nic", "ethernet", "сетевой адаптер"),
    },
    "network": {
        "switch": ("switch", "ethernet switch", "коммутатор", "свитч"),
        "router": ("router", "маршрутизатор", "роутер"),
        "firewall": ("firewall", "ngfw", "utm", "межсетевой экран"),
        "access_point": ("access point", "wi-fi", "wifi", "точка доступа"),
        "transceiver": ("transceiver", "sfp", "qsfp", "трансивер"),
        "dac_cable": ("dac", "direct attach"),
        "cable": ("cable", "aoc", "кабель"),
        "power_supply": ("power supply", "psu", "блок питания"),
        "stacking_module": ("stacking", "stack module", "стекирование"),
    },
    "storage": {
        "storage_system": ("storage array", "storage system", "схд", "система хранения"),
        "controller": ("storage controller", "контроллер схд"),
        "controller_module": ("controller module", "модуль контроллера"),
        "disk_shelf": ("disk shelf", "drive shelf", "полка", "дисковая полка"),
        "drive": ("drive", "disk", "накопитель", "диск"),
        "ssd": ("ssd", "nvme", "solid state"),
        "hdd": ("hdd", "hard disk"),
        "cache": ("cache", "кэш"),
        "host_port": ("host port", "fc port", "iscsi", "nvme-of"),
        "protocol_module": ("fc module", "iscsi module", "protocol module"),
        "transceiver": ("transceiver", "sfp", "qsfp", "трансивер"),
        "cable": ("cable", "dac", "aoc", "кабель"),
        "license": ("license", "лицензия"),
        "support": ("support", "поддержка"),
        "power_supply": ("power supply", "psu", "блок питания"),
        "rail_kit": ("rail", "rails", "рейки", "рельсы"),
        "other_accessory": ("accessory", "аксессуар"),
    },
    "support_license": {
        "license": ("license", "subscription", "лицензия"),
        "support": ("support", "warranty", "поддержка", "сервис"),
    },
    "accessory": {
        "transceiver": ("transceiver", "sfp", "qsfp", "трансивер"),
        "cable": ("cable", "dac", "aoc", "кабель"),
        "power_supply": ("power supply", "psu", "блок питания"),
        "rail_kit": ("rail", "rails", "рейки", "рельсы"),
        "other_accessory": ("accessory", "аксессуар"),
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover local OCS anchor category candidates for review."
    )
    parser.add_argument(
        "--group",
        choices=(*DISCOVERY_GROUPS, "all"),
        default="all",
        help="Anchor group to discover.",
    )
    return parser.parse_args(argv)


async def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    session_factory = get_session_factory()
    async with session_factory() as session:
        category_result = await session.execute(
            select(DistributorCategory)
            .where(DistributorCategory.distributor_code == DISTRIBUTOR_CODE)
            .order_by(DistributorCategory.level, DistributorCategory.name)
        )
        count_result = await session.execute(
            select(DistributorProduct.category_id, func.count(DistributorProduct.id))
            .where(DistributorProduct.distributor_code == DISTRIBUTOR_CODE)
            .group_by(DistributorProduct.category_id)
        )
        product_counts = {
            str(category_id): int(count)
            for category_id, count in count_result.all()
            if category_id
        }
        candidates = discover_anchor_candidates(
            category_result.scalars().all(),
            product_counts=product_counts,
            group=args.group,
        )
    _print_candidates(candidates)
    return 0


def discover_anchor_candidates(
    categories: Sequence[Any],
    *,
    product_counts: Mapping[str, int] | None = None,
    group: str = "all",
) -> list[AnchorCandidate]:
    groups = DISCOVERY_GROUPS if group == "all" else (group,)
    counts = product_counts or {}
    results: dict[tuple[str, str, str], AnchorCandidate] = {}
    for category in categories:
        category_id = str(_object_value(category, "category_id") or "").strip()
        if not category_id:
            continue
        name_path = _category_name_path(category)
        haystack = " ".join([category_id, name_path]).casefold()
        for group_name in groups:
            for role, terms in SEARCH_TERMS[group_name].items():
                matched_terms = tuple(
                    term for term in terms if term.casefold() in haystack
                )
                if not matched_terms:
                    continue
                key = (group_name, role, category_id)
                current = results.get(key)
                merged_terms = matched_terms
                if current is not None:
                    merged_terms = tuple(dict.fromkeys([*current.matched_terms, *matched_terms]))
                results[key] = AnchorCandidate(
                    group=group_name,
                    suggested_role=role,
                    category_id=category_id,
                    category_name_path=name_path,
                    parent_id=str(
                        _object_value(category, "parent_category_id")
                        or _object_value(category, "parent_id")
                        or ""
                    ),
                    product_count=counts.get(category_id),
                    matched_terms=merged_terms,
                )
    return sorted(
        results.values(),
        key=lambda item: (item.group, item.suggested_role, item.category_name_path),
    )


def _print_candidates(candidates: Sequence[AnchorCandidate]) -> None:
    print(
        "group\tsuggested_role\tcategory_id\tcategory_name/path\tparent_id\t"
        "product_count\tmatched_terms"
    )
    for candidate in candidates:
        product_count = "" if candidate.product_count is None else str(candidate.product_count)
        print(
            "\t".join(
                [
                    candidate.group,
                    candidate.suggested_role,
                    candidate.category_id,
                    candidate.category_name_path,
                    candidate.parent_id,
                    product_count,
                    ",".join(candidate.matched_terms),
                ]
            )
        )


def _category_name_path(category: Any) -> str:
    name = str(_object_value(category, "name") or "").strip()
    path_text = _path_text(_object_value(category, "path_json"))
    if path_text and name and name not in path_text:
        return f"{path_text} / {name}"
    return path_text or name


def _path_text(path_json: Any) -> str:
    if isinstance(path_json, list):
        parts = [
            str(row.get("name") or row.get("category_id") or "").strip()
            for row in path_json
            if isinstance(row, Mapping)
        ]
        return " / ".join(part for part in parts if part)
    if isinstance(path_json, str):
        return path_json
    return ""


def _object_value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
