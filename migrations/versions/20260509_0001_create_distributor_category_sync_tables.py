"""create distributor category sync tables

Revision ID: 20260509_0001
Revises:
Create Date: 2026-05-09 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260509_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "distributors",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )

    op.create_table(
        "distributor_categories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("distributor_code", sa.String(length=64), nullable=False),
        sa.Column("category_id", sa.String(length=255), nullable=False),
        sa.Column("parent_category_id", sa.String(length=255), nullable=True),
        sa.Column("name", sa.String(length=500), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("path_json", sa.JSON(), nullable=False),
        sa.Column(
            "enabled_for_sync",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "distributor_code",
            "category_id",
            name="uq_distributor_categories_distributor_code_category_id",
        ),
    )
    op.create_index(
        "ix_distributor_categories_distributor_code",
        "distributor_categories",
        ["distributor_code"],
    )
    op.create_index(
        "ix_distributor_categories_category_id",
        "distributor_categories",
        ["category_id"],
    )
    op.create_index(
        "ix_distributor_categories_parent_category_id",
        "distributor_categories",
        ["parent_category_id"],
    )
    op.create_index(
        "ix_distributor_categories_enabled_for_sync",
        "distributor_categories",
        ["enabled_for_sync"],
    )
    op.create_index(
        "ix_distributor_categories_synced_at",
        "distributor_categories",
        ["synced_at"],
    )

    op.create_table(
        "sync_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("distributor_code", sa.String(length=64), nullable=False),
        sa.Column("sync_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("items_processed", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_sync_runs_distributor_code_sync_type_started_at",
        "sync_runs",
        ["distributor_code", "sync_type", "started_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_sync_runs_distributor_code_sync_type_started_at", table_name="sync_runs")
    op.drop_table("sync_runs")

    op.drop_index("ix_distributor_categories_synced_at", table_name="distributor_categories")
    op.drop_index(
        "ix_distributor_categories_enabled_for_sync",
        table_name="distributor_categories",
    )
    op.drop_index(
        "ix_distributor_categories_parent_category_id",
        table_name="distributor_categories",
    )
    op.drop_index("ix_distributor_categories_category_id", table_name="distributor_categories")
    op.drop_index(
        "ix_distributor_categories_distributor_code",
        table_name="distributor_categories",
    )
    op.drop_table("distributor_categories")
    op.drop_table("distributors")
