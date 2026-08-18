"""Reading and writing the run audit table.

Every slice upserts one row keyed by trace id, so the record reflects the
latest state of a logical run rather than accumulating a row per HTTP call.

Persistence failures never propagate. An audit row is valuable, but losing one
must not fail a reconstruction that otherwise succeeded - the result still goes
back to the orchestrator, and the loss is logged.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from configuration.logging import get_logger
from configuration.settings import DatabaseSettings
from infrastructure.persistence.models import ReconstructionRun

_log = get_logger(__name__)


def _async_url(url: str) -> str:
    """Normalise a connection URL onto SQLAlchemy's async psycopg driver.

    Operators write `postgresql://...` out of habit, and the sync driver would
    block the event loop this agent runs on.
    """
    if url.startswith("postgresql+"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


class RunRepository:
    """Upserts run rows. A no-op when no database is configured."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[Any] | None = None

        if settings.configured:
            assert settings.database_url is not None
            self._engine = create_async_engine(
                _async_url(settings.database_url),
                pool_pre_ping=True,
                # Same reason as the checkpointer: the default wait is longer
                # than the orchestrator's whole request budget.
                connect_args={"connect_timeout": settings.connect_timeout_seconds},
            )
            self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def enabled(self) -> bool:
        return self._sessions is not None

    async def record(
        self,
        *,
        trace_id: str,
        run_id: str,
        sequence_id: str,
        organism: str | None,
        status: str,
        stop_reason: str | None,
        slice_count: int,
        iterations: int,
        gaps_total: int,
        gaps_resolved: int,
        tool_calls_used: int,
        llm_tokens_used: int,
        summary: str | None,
    ) -> None:
        """Insert or update the row for one logical run."""
        if self._sessions is None:
            return

        values: dict[str, Any] = {
            "trace_id": trace_id,
            "run_id": run_id,
            "sequence_id": sequence_id,
            "organism": organism,
            "status": status,
            "stop_reason": stop_reason,
            "slice_count": slice_count,
            "iterations": iterations,
            "gaps_total": gaps_total,
            "gaps_resolved": gaps_resolved,
            "tool_calls_used": tool_calls_used,
            "llm_tokens_used": llm_tokens_used,
            "summary": summary,
        }

        try:
            statement = insert(ReconstructionRun).values(**values)
            # `created_at` is deliberately absent from the update set: the row
            # should keep the moment the run started, not the moment its last
            # slice finished.
            statement = statement.on_conflict_do_update(
                index_elements=[ReconstructionRun.trace_id],
                set_={key: value for key, value in values.items() if key != "trace_id"},
            )
            async with self._sessions() as session:
                await session.execute(statement)
                await session.commit()

        except Exception as error:  # noqa: BLE001 - auditing must not fail a run
            _log.warning("run_audit_failed", trace_id=trace_id, error=str(error))

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
