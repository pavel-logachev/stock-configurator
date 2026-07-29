from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class Distributor(Base):
    __tablename__ = "distributors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=true(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class DistributorCategory(Base):
    __tablename__ = "distributor_categories"
    __table_args__ = (
        UniqueConstraint(
            "distributor_code",
            "category_id",
            name="uq_distributor_categories_distributor_code_category_id",
        ),
        Index("ix_distributor_categories_distributor_code", "distributor_code"),
        Index("ix_distributor_categories_category_id", "category_id"),
        Index("ix_distributor_categories_parent_category_id", "parent_category_id"),
        Index("ix_distributor_categories_enabled_for_sync", "enabled_for_sync"),
        Index("ix_distributor_categories_synced_at", "synced_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    distributor_code: Mapped[str] = mapped_column(String(64), nullable=False)
    category_id: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_category_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    path_json: Mapped[list[dict[str, str]]] = mapped_column(JSON, nullable=False)
    enabled_for_sync: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class DistributorProduct(Base):
    __tablename__ = "distributor_products"
    __table_args__ = (
        UniqueConstraint(
            "distributor_code",
            "item_id",
            name="uq_distributor_products_distributor_code_item_id",
        ),
        Index("ix_distributor_products_distributor_code", "distributor_code"),
        Index("ix_distributor_products_item_id", "item_id"),
        Index("ix_distributor_products_product_key", "product_key"),
        Index("ix_distributor_products_part_number", "part_number"),
        Index("ix_distributor_products_producer", "producer"),
        Index("ix_distributor_products_category_id", "category_id"),
        Index("ix_distributor_products_synced_at", "synced_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    distributor_code: Mapped[str] = mapped_column(String(64), nullable=False)
    item_id: Mapped[str] = mapped_column(String(255), nullable=False)
    product_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    part_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    producer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    category_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    item_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    item_name_rus: Mapped[str | None] = mapped_column(Text, nullable=True)
    product_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    product_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    product_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    hscode: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ean: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_in_mpt_registry: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_project_item: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    traceable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    condition: Mapped[str | None] = mapped_column(String(255), nullable=True)
    warranty: Mapped[str | None] = mapped_column(String(255), nullable=True)
    original_country_iso_code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    vat_percent: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    serial_number_availability: Mapped[str | None] = mapped_column(String(255), nullable=True)
    catalog_path_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    package_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class DistributorStockPrice(Base):
    __tablename__ = "distributor_stock_prices"
    __table_args__ = (
        Index(
            "ix_distributor_stock_prices_distributor_code_item_id",
            "distributor_code",
            "item_id",
        ),
        Index("ix_distributor_stock_prices_product_key", "product_key"),
        Index("ix_distributor_stock_prices_shipment_city", "shipment_city"),
        Index("ix_distributor_stock_prices_location_type", "location_type"),
        Index("ix_distributor_stock_prices_can_reserve", "can_reserve"),
        Index("ix_distributor_stock_prices_synced_at", "synced_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    distributor_code: Mapped[str] = mapped_column(String(64), nullable=False)
    item_id: Mapped[str] = mapped_column(String(255), nullable=False)
    product_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    shipment_city: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    location_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    location_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    quantity_value: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quantity_is_greater_than: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    can_reserve: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    departure_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    arrival_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    delivery_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    price_order_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    price_order_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    price_list_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    price_list_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    end_user_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    end_user_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )


class MatchRun(Base):
    __tablename__ = "match_runs"
    __table_args__ = (
        Index("ix_match_runs_status", "status"),
        Index("ix_match_runs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    engineer_review_required: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=true(),
    )
    total_candidates: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    matched_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    missing_requirements_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    risk_flags_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    report_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    candidates: Mapped[list[MatchCandidate]] = relationship(
        back_populates="match_run",
        cascade="all, delete-orphan",
    )


class MatchCandidate(Base):
    __tablename__ = "match_candidates"
    __table_args__ = (
        Index("ix_match_candidates_match_run_id", "match_run_id"),
        Index("ix_match_candidates_distributor_code_item_id", "distributor_code", "item_id"),
        Index("ix_match_candidates_confidence_score", "confidence_score"),
        Index("ix_match_candidates_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_run_id: Mapped[int] = mapped_column(
        ForeignKey("match_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    distributor_code: Mapped[str] = mapped_column(String(64), nullable=False)
    item_id: Mapped[str] = mapped_column(String(255), nullable=False)
    product_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    part_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    producer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    category_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    item_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[int] = mapped_column(Integer, nullable=False)
    price_value: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    price_currency: Mapped[str | None] = mapped_column(String(16), nullable=True)
    available_quantity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reservable_locations: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    matched_requirements_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    missing_requirements_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    risk_flags_json: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default=text("'[]'"),
    )
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    match_run: Mapped[MatchRun] = relationship(back_populates="candidates")


class SyncRun(Base):
    __tablename__ = "sync_runs"
    __table_args__ = (
        Index(
            "ix_sync_runs_distributor_code_sync_type_started_at",
            "distributor_code",
            "sync_type",
            "started_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    distributor_code: Mapped[str] = mapped_column(String(64), nullable=False)
    sync_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    items_processed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
