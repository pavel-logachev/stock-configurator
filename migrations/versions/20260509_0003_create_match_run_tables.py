"""create match run tables

Revision ID: 20260509_0003
Revises: 20260509_0002
Create Date: 2026-05-09 14:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260509_0003"
down_revision: str | None = "20260509_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "match_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=255), nullable=True),
        sa.Column("source_text", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column(
            "engineer_review_required",
            sa.Boolean(),
            server_default=sa.true(),
            nullable=False,
        ),
        sa.Column("total_candidates", sa.Integer(), server_default="0", nullable=False),
        sa.Column("matched_items", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "missing_requirements_json",
            sa.JSON(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column("risk_flags_json", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=False),
        sa.Column("report_json", sa.JSON(), nullable=False),
        sa.Column("report_markdown", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_match_runs_status", "match_runs", ["status"])
    op.create_index("ix_match_runs_created_at", "match_runs", ["created_at"])

    op.create_table(
        "match_candidates",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_run_id", sa.Integer(), nullable=False),
        sa.Column("distributor_code", sa.String(length=64), nullable=False),
        sa.Column("item_id", sa.String(length=255), nullable=False),
        sa.Column("product_key", sa.String(length=255), nullable=True),
        sa.Column("part_number", sa.String(length=255), nullable=True),
        sa.Column("producer", sa.String(length=255), nullable=True),
        sa.Column("category_id", sa.String(length=255), nullable=True),
        sa.Column("item_name", sa.Text(), nullable=True),
        sa.Column("confidence_score", sa.Integer(), nullable=False),
        sa.Column("price_value", sa.Numeric(18, 4), nullable=True),
        sa.Column("price_currency", sa.String(length=16), nullable=True),
        sa.Column("available_quantity", sa.Integer(), nullable=True),
        sa.Column("reservable_locations", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "matched_requirements_json",
            sa.JSON(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "missing_requirements_json",
            sa.JSON(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column("risk_flags_json", sa.JSON(), server_default=sa.text("'[]'"), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["match_run_id"], ["match_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_match_candidates_match_run_id", "match_candidates", ["match_run_id"])
    op.create_index(
        "ix_match_candidates_distributor_code_item_id",
        "match_candidates",
        ["distributor_code", "item_id"],
    )
    op.create_index(
        "ix_match_candidates_confidence_score",
        "match_candidates",
        ["confidence_score"],
    )
    op.create_index("ix_match_candidates_created_at", "match_candidates", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_match_candidates_created_at", table_name="match_candidates")
    op.drop_index("ix_match_candidates_confidence_score", table_name="match_candidates")
    op.drop_index("ix_match_candidates_distributor_code_item_id", table_name="match_candidates")
    op.drop_index("ix_match_candidates_match_run_id", table_name="match_candidates")
    op.drop_table("match_candidates")

    op.drop_index("ix_match_runs_created_at", table_name="match_runs")
    op.drop_index("ix_match_runs_status", table_name="match_runs")
    op.drop_table("match_runs")
