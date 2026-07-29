from app.distributors.base import DistributorOffer
from app.distributors.ocs.schemas import OcsCatalogItem


def map_ocs_item_to_offer(item: OcsCatalogItem) -> DistributorOffer:
    return DistributorOffer(
        distributor="ocs",
        sku=item.sku,
        name=item.name,
        brand=item.brand,
        part_number=item.part_number,
        price=item.price,
        currency=item.currency,
        stock_quantity=item.stock_quantity,
    )
