from decimal import Decimal
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class OcsFlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class OcsStockLocation(OcsFlexibleModel):
    id: str | int | None = None
    code: str | None = None
    name: str | None = None
    city: str | None = None
    address: str | None = None


class OcsCategory(OcsFlexibleModel):
    id: str | int | None = None
    name: str | None = None
    parent_id: str | int | None = Field(
        default=None,
        validation_alias=AliasChoices("parent_id", "parentId", "parentid"),
    )
    children: list["OcsCategory"] = Field(default_factory=list)


class OcsProduct(OcsFlexibleModel):
    item_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("item_id", "itemId", "itemid", "id"),
    )
    sku: str | None = None
    name: str | None = None
    brand: str | None = None
    part_number: str | None = Field(
        default=None,
        validation_alias=AliasChoices("part_number", "partNumber", "partnumber"),
    )
    price: Decimal | None = None
    currency: str | None = None
    stock_quantity: int | None = Field(
        default=None,
        validation_alias=AliasChoices("stock_quantity", "stockQuantity", "stockquantity"),
    )
    raw: dict[str, Any] | None = None


class OcsProductWrapper(OcsFlexibleModel):
    product: OcsProduct | None = None
    item: OcsProduct | None = None
    products: list[OcsProduct] | None = None


class OcsCatalogItem(BaseModel):
    sku: str
    name: str
    brand: str | None = None
    part_number: str | None = None
    price: Decimal | None = None
    currency: str | None = None
    stock_quantity: int = Field(default=0, ge=0)
