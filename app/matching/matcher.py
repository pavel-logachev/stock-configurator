from dataclasses import dataclass

from app.catalog.normalizer import normalize_text
from app.distributors.base import DistributorOffer
from app.matching.spec_schema import NormalizedSpecItem


@dataclass(frozen=True)
class MatchResult:
    spec_item: NormalizedSpecItem
    offer: DistributorOffer
    score: float
    reason: str


def match_offer(spec_item: NormalizedSpecItem, offer: DistributorOffer) -> MatchResult | None:
    if (
        spec_item.part_number
        and offer.part_number
        and normalize_text(spec_item.part_number) == normalize_text(offer.part_number)
    ):
        return MatchResult(spec_item, offer, 1.0, "Совпадение по part number")

    if spec_item.brand and offer.brand:
        brand_matches = normalize_text(spec_item.brand) == normalize_text(offer.brand)
    else:
        brand_matches = True

    name_matches = normalize_text(spec_item.name) in normalize_text(offer.name)
    if brand_matches and name_matches:
        return MatchResult(spec_item, offer, 0.7, "Совпадение по названию и бренду")

    return None


def match_offers(
    spec_items: list[NormalizedSpecItem],
    offers: list[DistributorOffer],
) -> list[MatchResult]:
    results: list[MatchResult] = []

    for spec_item in spec_items:
        candidates = [match_offer(spec_item, offer) for offer in offers]
        candidates = [candidate for candidate in candidates if candidate is not None]
        if candidates:
            results.append(max(candidates, key=lambda item: item.score))

    return results
