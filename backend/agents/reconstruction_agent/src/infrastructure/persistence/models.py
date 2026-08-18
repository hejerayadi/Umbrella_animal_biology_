"""The agent's own tables.

LangGraph manages its checkpoint tables itself. This adds the one thing raw
checkpoints do not give: a durable, queryable record of what each run did -
how many slices it took, what it spent, why it stopped. That is what you read
when someone asks why a reconstruction came back unresolved last Tuesday.

Kept in a dedicated schema so the agent migrates independently of the backend,
which is the same isolation the separate `.venv` buys at the Python level.
"""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, MetaData, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: Overridden at runtime from `RECONSTRUCTION_DB_SCHEMA`; the default is what
#: the migration creates.
DEFAULT_SCHEMA = "reconstruction"


class Base(DeclarativeBase):
    metadata = MetaData(schema=DEFAULT_SCHEMA)


class ReconstructionRun(Base):
    """One reconstruction, across every slice it took.

    Keyed by `trace_id` rather than `run_id`: the orchestrator's trace id is
    what stays stable across CONTINUE retries, so it is the only identifier
    under which the slices of one logical run can be gathered.
    """

    __tablename__ = "reconstruction_runs"

    trace_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    #: The id of the most recent slice. Changes per HTTP call, kept for
    #: cross-referencing the logs of that specific attempt.
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)

    sequence_id: Mapped[str] = mapped_column(String(255), nullable=False)
    organism: Mapped[str | None] = mapped_column(String(255), nullable=True)

    #: The orchestrator-facing status of the latest slice: completed,
    #: continue, needs_agent or failed.
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stop_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)

    slice_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    gaps_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gaps_resolved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    tool_calls_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Free-text summary as shown to the user, for spot-checking without
    #: replaying the run.
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<ReconstructionRun {self.trace_id} {self.status} "
            f"{self.gaps_resolved}/{self.gaps_total} gaps>"
        )
