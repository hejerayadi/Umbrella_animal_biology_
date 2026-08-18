"""Create the reconstruction_runs audit table.

Revision ID: 0001_reconstruction_runs
Revises:
Create Date: 2026-08-18
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_reconstruction_runs"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "reconstruction"


def upgrade() -> None:
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')

    op.create_table(
        "reconstruction_runs",
        # The orchestrator's trace id, not the per-call run id: it is what
        # stays stable across CONTINUE retries, so it is the only key under
        # which the slices of one logical run can be gathered.
        sa.Column("trace_id", sa.String(length=128), primary_key=True),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("sequence_id", sa.String(length=255), nullable=False),
        sa.Column("organism", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stop_reason", sa.String(length=64), nullable=True),
        sa.Column("slice_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("iterations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gaps_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gaps_resolved", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_calls_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_tokens_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema=SCHEMA,
    )

    # The two questions actually asked of this table: "what ran recently?" and
    # "how often do we abstain or run out of budget?".
    op.create_index(
        "ix_reconstruction_runs_created_at",
        "reconstruction_runs",
        ["created_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_reconstruction_runs_status",
        "reconstruction_runs",
        ["status", "stop_reason"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index("ix_reconstruction_runs_status", "reconstruction_runs", schema=SCHEMA)
    op.drop_index("ix_reconstruction_runs_created_at", "reconstruction_runs", schema=SCHEMA)
    op.drop_table("reconstruction_runs", schema=SCHEMA)
    # The schema itself is left in place: LangGraph's checkpoint tables live
    # there too, and dropping it would take them with it.
