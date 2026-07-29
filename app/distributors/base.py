from abc import ABC, abstractmethod
from datetime import UTC, datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class DistributorOffer(BaseModel):
    distributor: str
    sku: str
    name: str
    brand: str | None = None
    part_number: str | None = None
    price: Decimal | None = None
    currency: str | None = None
    stock_quantity: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DistributorClient(ABC):
    name: str

    @abstractmethod
    async def fetch_offers(self) -> list[DistributorOffer]:
        """Вернуть актуальные предложения дистрибьютора."""
