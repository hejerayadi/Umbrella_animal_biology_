"""Reading and writing the run audit table.

Every slice upserts one row keyed by trace id, so the record reflects the
latest state of a logical run rather than accumulating a row per HTTP call.

Persistence failures never propagate. An audit row is valuable, but losing one
must not fail a reconstruction that otherwise succeeded - the result still goes
back to the orchestrator, and the loss is logged.
"""
from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from configuration.logging import get_logger
from configuration.runtime import on_proactor_loop
from configuration.settings import DatabaseSettings
from infrastructure.persistence.models import ReconstructionRun

_log = get_logger(__name__)


def psycopg_url(url: str) -> str:
    """Normalise a connection URL onto SQLAlchemy's psycopg (v3) driver.

    A bare `postgresql://` - which is what Supabase hands out, and what most
    people write from habit - makes SQLAlchemy reach for psycopg2, which this
    agent does not install. The failure is a bare `ModuleNotFoundError:
    psycopg2` that says nothing about the URL.

    The same `postgresql+psycopg` dialect serves both `create_engine` and
    `create_async_engine`, so one rule covers the runtime engine and Alembic.
    """
    if url.startswith("postgresql+"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


#: Kept as the historical name used inside this module.
_async_url = psycopg_url


class RunRepository:
    """Upserts run rows. A no-op when no database is configured."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[Any] | None = None

        self._sync_engine: Engine | None = None
        self._sync_sessions: sessionmaker[Any] | None = None

        if not settings.configured:
            return

        assert settings.database_url is not None
        url = _async_url(settings.database_url)
        # Same reason as the checkpointer: the default wait is longer than the
        # orchestrator's whole request budget.
        connect_args = {"connect_timeout": settings.connect_timeout_seconds}

        if on_proactor_loop():
            # psycopg's async driver cannot run on the loop uvicorn built for
            # us on Windows. The synchronous engine has no such constraint, and
            # `record` drives it from a worker thread. Without this the audit
            # trail was silently empty on every developer machine - the write
            # failed, and failing quietly is what this repository is for.
            self._sync_engine = create_engine(
                url, pool_pre_ping=True, connect_args=connect_args
            )
            self._sync_sessions = sessionmaker(self._sync_engine, expire_on_commit=False)
            _log.info("run_audit_engine", mode="sync", detail="Windows proactor loop detected.")
            return

        self._engine = create_async_engine(url, pool_pre_ping=True, connect_args=connect_args)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    @property
    def enabled(self) -> bool:
        return self._sessions is not None or self._sync_sessions is not None

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
        if not self.enabled:
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
            if self._sync_sessions is not None:
                await asyncio.to_thread(self._execute_sync, statement)
            elif self._sessions is not None:
                async with self._sessions() as session:
                    await session.execute(statement)
                    await session.commit()

        except Exception as error:  # noqa: BLE001 - auditing must not fail a run
            _log.warning("run_audit_failed", trace_id=trace_id, error=str(error))

    def _execute_sync(self, statement: Any) -> None:
        """One upsert on the synchronous engine, run off the event loop."""
        assert self._sync_sessions is not None
        with self._sync_sessions() as session:
            session.execute(statement)
            session.commit()

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
        if self._sync_engine is not None:
            await asyncio.to_thread(self._sync_engine.dispose)
