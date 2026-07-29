"""create distributor product stock tables

Revision ID: 20260509_0002
Revises: 20260509_0001
Create Date: 2026-05-09 12:30:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260509_0002"
down_revision: str | None = "20260509_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "distributor_products",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("distributor_code", sa.String(length=64), nullable=False),
        sa.Column("item_id", sa.String(length=255), nullable=False),
        sa.Column("product_key", sa.String(length=255), nullable=True),
        sa.Column("part_number", sa.String(length=255), nullable=True),
        sa.Column("producer", sa.String(length=255), nullable=True),
        sa.Column("category_id", sa.String(length=255), nullable=True),
        sa.Column("item_name", sa.Text(), nullable=True),
        sa.Column("item_name_rus", sa.Text(), nullable=True),
        sa.Column("product_name", sa.Text(), nullable=True),
        sa.Column("product_description", sa.Text(), nullable=True),
        sa.Column("product_notes", sa.Text(), nullable=True),
        sa.Column("hscode", sa.String(length=64), nullable=True),
        sa.Column("ean", sa.String(length=64), nullable=True),
        sa.Column("is_in_mpt_registry", sa.Boolean(), nullable=True),
        sa.Column("is_project_item", sa.Boolean(), nullable=True),
        sa.Column("traceable", sa.Boolean(), nullable=True),
        sa.Column("condition", sa.String(length=255), nullable=True),
        sa.Column("warranty", sa.String(length=255), nullable=True),
        sa.Column("original_country_iso_code", sa.String(length=16), nullable=True),
        sa.Column("vat_percent", sa.Numeric(5, 2), nullable=True),
        sa.Column("serial_number_availability", sa.String(length=255), nullable=True),
        sa.Column("catalog_path_json", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("package_json", sa.JSON(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "distributor_code",
            "item_id",
            name="uq_distributor_products_distributor_code_item_id",
        ),
    )
    op.create_index(
        "ix_distributor_products_distributor_code",
        "distributor_products",
        ["distributor_code"],
    )
    op.create_index("ix_distributor_products_item_id", "distributor_products", ["item_id"])
    op.create_index(
        "ix_distributor_products_product_key",
        "distributor_products",
        ["product_key"],
    )
    op.create_index(
        "ix_distributor_products_part_number",
        "distributor_products",
        ["part_number"],
    )
    op.create_index("ix_distributor_products_producer", "distributor_products", ["producer"])
    op.create_index(
        "ix_distributor_products_category_id",
        "distributor_products",
        ["category_id"],
    )
    op.create_index("ix_distributor_products_synced_at", "distributor_products", ["synced_at"])

    op.create_table(
        "distributor_stock_prices",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("distributor_code", sa.String(length=64), nullable=False),
        sa.Column("item_id", sa.String(length=255), nullable=False),
        sa.Column("product_key", sa.String(length=255), nullable=True),
        sa.Column("shipment_city", sa.String(length=255), nullable=False),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column("location_description", sa.Text(), nullable=True),
        sa.Column("location_type", sa.String(length=255), nullable=True),
        sa.Column("quantity_value", sa.Integer(), nullable=True),
        sa.Column("quantity_is_greater_than", sa.Boolean(), nullable=True),
        sa.Column("can_reserve", sa.Boolean(), nullable=True),
        sa.Column("departure_date", sa.Date(), nullable=True),
        sa.Column("arrival_date", sa.Date(), nullable=True),
        sa.Column("delivery_date", sa.Date(), nullable=True),
        sa.Column("price_order_value", sa.Numeric(18, 4), nullable=True),
        sa.Column("price_order_currency", sa.String(length=16), nullable=True),
        sa.Column("price_list_value", sa.Numeric(18, 4), nullable=True),
        sa.Column("price_list_currency", sa.String(length=16), nullable=True),
        sa.Column("end_user_value", sa.Numeric(18, 4), nullable=True),
        sa.Column("end_user_currency", sa.String(length=16), nullable=True),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_distributor_stock_prices_distributor_code_item_id",
        "distributor_stock_prices",
        ["distributor_code", "item_id"],
    )
    op.create_index(
        "ix_distributor_stock_prices_product_key",
        "distributor_stock_prices",
        ["product_key"],
    )
    op.create_index(
        "ix_distributor_stock_prices_shipment_city",
        "distributor_stock_prices",
        ["shipment_city"],
    )
    op.create_index(
        "ix_distributor_stock_prices_location_type",
        "distributor_stock_prices",
        ["location_type"],
    )
    op.create_index(
        "ix_distributor_stock_prices_can_reserve",
        "distributor_stock_prices",
        ["can_reserve"],
    )
    op.create_index(
        "ix_distributor_stock_prices_synced_at",
        "distributor_stock_prices",
        ["synced_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_distributor_stock_prices_synced_at", table_name="distributor_stock_prices")
    op.drop_index(
        "ix_distributor_stock_prices_can_reserve",
        table_name="distributor_stock_prices",
    )
    op.drop_index(
        "ix_distributor_stock_prices_location_type",
        table_name="distributor_stock_prices",
    )
    op.drop_index(
        "ix_distributor_stock_prices_shipment_city",
        table_name="distributor_stock_prices",
    )
    op.drop_index(
        "ix_distributor_stock_prices_product_key",
        table_name="distributor_stock_prices",
    )
    op.drop_index(
        "ix_distributor_stock_prices_distributor_code_item_id",
        table_name="distributor_stock_prices",
    )
    op.drop_table("distributor_stock_prices")

    op.drop_index("ix_distributor_products_synced_at", table_name="distributor_products")
    op.drop_index("ix_distributor_products_category_id", table_name="distributor_products")
    op.drop_index("ix_distributor_products_producer", table_name="distributor_products")
    op.drop_index("ix_distributor_products_part_number", table_name="distributor_products")
    op.drop_index("ix_distributor_products_product_key", table_name="distributor_products")
    op.drop_index("ix_distributor_products_item_id", table_name="distributor_products")
    op.drop_index(
        "ix_distributor_products_distributor_code",
        table_name="distributor_products",
    )
    op.drop_table("distributor_products")
