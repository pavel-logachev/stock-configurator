from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.distributors import category_refresh as category_refresh_module
from app.distributors.category_refresh import refresh_distributor_categories
from app.distributors.registry import (
    CategoryRefreshResult,
    DistributorCapabilities,
    DistributorConnector,
    DistributorConnectorRegistry,
    build_default_distributor_registry,
)


def test_default_distributor_registry_exposes_current_refresh_sources() -> None:
    registry = build_default_distributor_registry()

    assert registry.refreshable_codes() == ("ocs", "treolan")
    assert registry.get(" OCS ") is not None
    assert registry.get(" treolan ") is not None
    assert registry.get("ocs").capabilities.supports_canonical_stock_prices is True


def test_distributor_registry_normalizes_codes_and_rejects_duplicates() -> None:
    registry = DistributorConnectorRegistry()
    connector = _connector(" TestWarehouse ")

    registry.register(connector)

    assert registry.get(" testwarehouse ") is connector
    assert registry.capabilities()["testwarehouse"].code == "testwarehouse"
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_connector("TESTWAREHOUSE"))


def test_refresh_distributor_categories_uses_registered_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_refresh(
        session: AsyncSession,
        category_ids: Sequence[str],
    ) -> CategoryRefreshResult:
        calls.append(list(category_ids))
        return CategoryRefreshResult(
            distributor_code="testwarehouse",
            status="success",
            category_count=len(category_ids),
            products_processed=5,
            stock_rows_inserted=7,
            sync_run_id=123,
        )

    registry = DistributorConnectorRegistry()
    registry.register(_connector("testwarehouse", refresh_categories=fake_refresh))
    monkeypatch.setattr(
        category_refresh_module,
        "build_default_distributor_registry",
        lambda: registry,
    )

    result = asyncio.run(
        refresh_distributor_categories(
            object(),
            distributor_code=" TestWarehouse ",
            category_ids=["cat-a", "cat-a", "", " cat-b "],
        )
    )

    assert calls == [["cat-a", "cat-b"]]
    assert result.to_diagnostics() == {
        "distributor_code": "testwarehouse",
        "status": "success",
        "category_count": 2,
        "products_processed": 5,
        "stock_rows_inserted": 7,
        "sync_run_id": 123,
        "error_message": None,
    }


def test_refresh_distributor_categories_reports_unsupported_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        category_refresh_module,
        "build_default_distributor_registry",
        DistributorConnectorRegistry,
    )

    result = asyncio.run(
        refresh_distributor_categories(
            object(),
            distributor_code="missing",
            category_ids=["cat-a"],
        )
    )

    assert result.status == "unsupported_distributor"
    assert result.success is False
    assert result.category_count == 1
    assert "on-demand category refresh" in str(result.error_message)


def _connector(
    code: str,
    *,
    refresh_categories=None,
) -> DistributorConnector:
    normalized_code = code.strip().lower()
    return DistributorConnector(
        code=code,
        display_name=code.strip(),
        capabilities=DistributorCapabilities(
            code=normalized_code,
            display_name=code.strip(),
            supports_category_refresh=refresh_categories is not None,
            supports_canonical_stock_prices=True,
        ),
        refresh_categories=refresh_categories,
    )
