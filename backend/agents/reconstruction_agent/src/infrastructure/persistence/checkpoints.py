"""Where LangGraph stores run state between steps - and between HTTP calls.

This is what makes CONTINUE work. The orchestrator allows 120 s per call and
retries a CONTINUE three times, so one reconstruction is spread over up to four
HTTP calls. Each picks up from the checkpoint written by the last, keyed by the
orchestrator's trace id.

In-memory checkpointing is kept for tests and for running without
infrastructure. It is genuinely per-process: with it, a CONTINUE that lands on
a different worker - or arrives after a restart - starts from nothing.
"""
from __future__ import annotations

from typing import Any

from configuration.logging import get_logger
from configuration.settings import DatabaseSettings

_log = get_logger(__name__)


class CheckpointerHandle:
    """An open checkpointer plus the connection pool behind it.

    LangGraph's Postgres saver owns a pool that has to be opened before use and
    closed on shutdown. Bundling the two means the application can manage that
    lifetime without knowing which implementation it got.
    """

    def __init__(self, saver: Any, closer: Any | None = None, *, durable: bool) -> None:
        self.saver = saver
        self._closer = closer
        #: False for the in-memory saver, whose state does not survive a
        #: restart. The API layer reports this on /health so a deployment that
        #: silently lost durability is visible.
        self.durable = durable

    async def aclose(self) -> None:
        if self._closer is not None:
            await self._closer.__aexit__(None, None, None)


def _libpq_url(url: str, connect_timeout: int = 5) -> str:
    """The connection string LangGraph's saver needs.

    Two corrections, both of which fail quietly rather than loudly:

    1. **Dialect suffix.** One URL is configured for the whole agent, but
       SQLAlchemy wants `postgresql+psycopg://` while LangGraph hands the
       string straight to libpq, which rejects that with a misleading
       `missing "=" after ...` and falls back to in-memory checkpointing.

    2. **Connect timeout.** psycopg waits ~130 s by default, which outlives
       the orchestrator's whole 120 s request budget - so an unreachable
       database would blow the deadline rather than degrade to in-memory.

    No search path is set: the agent shares one schema with the rest of
    Umbrella, and its tables are told apart by name.
    """
    scheme, separator, rest = url.partition("://")
    base = url if not separator else f"{scheme.split('+', 1)[0]}://{rest}"

    if "connect_timeout=" in base:
        return base

    separator = "&" if "?" in base else "?"
    return f"{base}{separator}connect_timeout={connect_timeout}"


async def build_checkpointer(settings: DatabaseSettings) -> CheckpointerHandle:
    """Open the configured checkpointer.

    Falls back to in-memory when no database is configured, or when Postgres
    cannot be reached. The fallback is deliberate: an agent that refuses to
    start because a checkpoint store is down is worse than one that runs and
    cannot resume - the first slice would have failed either way, and most
    reconstructions finish inside it.
    """
    if not settings.configured:
        _log.info("checkpointer_memory", reason="no RECONSTRUCTION_DATABASE_URL configured")
        return CheckpointerHandle(_memory_saver(), durable=False)

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        assert settings.database_url is not None
        context = AsyncPostgresSaver.from_conn_string(
            _libpq_url(settings.database_url, settings.connect_timeout_seconds)
        )
        saver = await context.__aenter__()
        # Creates LangGraph's own checkpoint tables if absent. Idempotent, so
        # it is safe on every boot; our Alembic migration owns only the
        # `reconstruction_runs` audit table alongside them.
        await saver.setup()

        _log.info("checkpointer_postgres")
        return CheckpointerHandle(saver, context, durable=True)

    except Exception as error:  # noqa: BLE001 - never block startup on this
        _log.warning(
            "checkpointer_postgres_unavailable",
            error=str(error),
            detail="Falling back to in-memory; CONTINUE will not resume across calls.",
        )
        return CheckpointerHandle(_memory_saver(), durable=False)


def _memory_saver() -> Any:
    """The in-process saver, or None when LangGraph is not installed."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ImportError:  # pragma: no cover - depends on install
        try:
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        except ImportError:
            _log.warning("checkpointing_unavailable")
            return None
