from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class CategoryRefreshResult:
    distributor_code: str
    status: str
    category_count: int
    products_processed: int = 0
    stock_rows_inserted: int = 0
    sync_run_id: int | None = None
    error_message: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "success"

    def to_diagnostics(self) -> dict[str, object]:
        return {
            "distributor_code": self.distributor_code,
            "status": self.status,
            "category_count": self.category_count,
            "products_processed": self.products_processed,
            "stock_rows_inserted": self.stock_rows_inserted,
            "sync_run_id": self.sync_run_id,
            "error_message": self.error_message,
        }


CategoryRefreshHandler = Callable[
    [AsyncSession, Sequence[str]],
    Awaitable[CategoryRefreshResult],
]


@dataclass(frozen=True)
class DistributorCapabilities:
    code: str
    display_name: str
    supports_category_refresh: bool = False
    supports_canonical_stock_prices: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DistributorConnector:
    code: str
    display_name: str
    capabilities: DistributorCapabilities
    refresh_categories: CategoryRefreshHandler | None = None

    def __post_init__(self) -> None:
        normalized_code = _normalize_code(self.code)
        object.__setattr__(self, "code", normalized_code)
        if self.capabilities.code != normalized_code:
            object.__setattr__(
                self,
                "capabilities",
                DistributorCapabilities(
                    code=normalized_code,
                    display_name=self.capabilities.display_name,
                    supports_category_refresh=self.capabilities.supports_category_refresh,
                    supports_canonical_stock_prices=(
                        self.capabilities.supports_canonical_stock_prices
                    ),
                    notes=self.capabilities.notes,
                ),
            )


@dataclass
class DistributorConnectorRegistry:
    _connectors: dict[str, DistributorConnector] = field(default_factory=dict)

    def register(self, connector: DistributorConnector) -> None:
        code = _normalize_code(connector.code)
        if not code:
            raise ValueError("Distributor connector code must not be empty.")
        if code in self._connectors:
            raise ValueError(f"Distributor connector {code!r} is already registered.")
        self._connectors[code] = connector

    def get(self, distributor_code: str) -> DistributorConnector | None:
        return self._connectors.get(_normalize_code(distributor_code))

    def capabilities(self) -> dict[str, DistributorCapabilities]:
        return {
            code: connector.capabilities
            for code, connector in sorted(self._connectors.items())
        }

    def refreshable_codes(self) -> tuple[str, ...]:
        return tuple(
            code
            for code, connector in sorted(self._connectors.items())
            if connector.refresh_categories is not None
        )


def build_default_distributor_registry() -> DistributorConnectorRegistry:
    from app.distributors.ocs.sync_products import sync_ocs_products
    from app.distributors.treolan.sync_products import sync_treolan_products

    registry = DistributorConnectorRegistry()
    registry.register(
        DistributorConnector(
            code="ocs",
            display_name="OCS",
            capabilities=DistributorCapabilities(
                code="ocs",
                display_name="OCS",
                supports_category_refresh=True,
                supports_canonical_stock_prices=True,
            ),
            refresh_categories=_wrap_product_sync("ocs", sync_ocs_products),
        )
    )
    registry.register(
        DistributorConnector(
            code="treolan",
            display_name="Treolan",
            capabilities=DistributorCapabilities(
                code="treolan",
                display_name="Treolan",
                supports_category_refresh=True,
                supports_canonical_stock_prices=True,
            ),
            refresh_categories=_wrap_product_sync("treolan", sync_treolan_products),
        )
    )
    return registry


def _wrap_product_sync(
    distributor_code: str,
    sync_func: Callable[..., Awaitable[object]],
) -> CategoryRefreshHandler:
    async def refresh(
        session: AsyncSession,
        category_ids: Sequence[str],
    ) -> CategoryRefreshResult:
        result = await sync_func(session, category_ids=category_ids)
        return CategoryRefreshResult(
            distributor_code=distributor_code,
            status=result.status,
            category_count=len(category_ids),
            products_processed=result.products_processed,
            stock_rows_inserted=result.stock_rows_inserted,
            sync_run_id=result.sync_run_id,
            error_message=result.error_message,
        )

    return refresh


def _normalize_code(distributor_code: str) -> str:
    return str(distributor_code or "").strip().lower()
