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
from configuration.runtime import on_proactor_loop
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

    assert settings.database_url is not None
    conninfo = _libpq_url(settings.database_url, settings.connect_timeout_seconds)

    if on_proactor_loop():
        # Windows only, and only under uvicorn: it builds its loop from a
        # factory (`uvicorn/loops/asyncio.py` -> ProactorEventLoop) rather than
        # from the event-loop policy, so `configuration.runtime` cannot reach
        # it. psycopg's async driver refuses to run there.
        #
        # The sync saver has no such constraint. It blocks the loop briefly on
        # each checkpoint write, which is the right trade against losing
        # resumable CONTINUE entirely on a developer's machine. Linux - where
        # this actually deploys - gets the async path.
        return _sync_postgres_handle(conninfo)

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        context = AsyncPostgresSaver.from_conn_string(conninfo)
        saver = await context.__aenter__()
        # Creates LangGraph's own checkpoint tables if absent. Idempotent, so
        # it is safe on every boot; our Alembic migration owns only the
        # `reconstruction_runs` audit table alongside them.
        await saver.setup()

        _log.info("checkpointer_postgres", mode="async")
        return CheckpointerHandle(saver, context, durable=True)

    except Exception as error:  # noqa: BLE001 - never block startup on this
        _log.warning(
            "checkpointer_postgres_unavailable",
            error=str(error),
            detail="Falling back to in-memory; CONTINUE will not resume across calls.",
        )
        return CheckpointerHandle(_memory_saver(), durable=False)


def _sync_postgres_handle(conninfo: str) -> CheckpointerHandle:
    """Durable checkpointing via the synchronous saver.

    Opened in a worker thread: `PostgresSaver.setup()` runs DDL, and doing that
    inline would block the event loop through a network round trip at startup.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        context = PostgresSaver.from_conn_string(conninfo)
        saver = context.__enter__()
        saver.setup()

        _log.info(
            "checkpointer_postgres",
            mode="sync",
            detail=(
                "Windows proactor loop detected; using the synchronous saver so "
                "CONTINUE can still resume."
            ),
        )
        return CheckpointerHandle(_threaded_saver(saver), _SyncCloser(context), durable=True)

    except Exception as error:  # noqa: BLE001 - never block startup on this
        _log.warning(
            "checkpointer_postgres_unavailable",
            error=str(error),
            detail="Falling back to in-memory; CONTINUE will not resume across calls.",
        )
        return CheckpointerHandle(_memory_saver(), durable=False)


def _threaded_saver(saver: Any) -> Any:
    """Give a synchronous saver the async API LangGraph actually calls.

    The graph is driven with `ainvoke`, so LangGraph reaches for `aget_tuple`,
    `aput` and `aput_writes`. `PostgresSaver` implements only the synchronous
    half, and `BaseCheckpointSaver` answers the async half with a bare
    `NotImplementedError` - whose message is the empty string, so handing the
    sync saver to the graph directly failed every run with a blank error.

    Each call is delegated to a worker thread rather than run inline: the
    checkpoint write is a network round trip, and blocking the event loop
    through it would stall the concurrent tool calls the loop depends on.
    `PostgresSaver` guards its connection with its own lock, so several
    threads are safe.
    """
    from langgraph.checkpoint.base import BaseCheckpointSaver

    class _ThreadedSaver(BaseCheckpointSaver):  # type: ignore[type-arg]
        def __init__(self, inner: Any) -> None:
            super().__init__(serde=inner.serde)
            self._inner = inner

        # --- synchronous half: straight through ---------------------------
        def get_tuple(self, config: Any) -> Any:
            return self._inner.get_tuple(config)

        def list(self, config: Any, **kwargs: Any) -> Any:
            return self._inner.list(config, **kwargs)

        def put(self, *args: Any, **kwargs: Any) -> Any:
            return self._inner.put(*args, **kwargs)

        def put_writes(self, *args: Any, **kwargs: Any) -> None:
            self._inner.put_writes(*args, **kwargs)

        def delete_thread(self, thread_id: str) -> None:
            self._inner.delete_thread(thread_id)

        def get_next_version(self, *args: Any, **kwargs: Any) -> Any:
            return self._inner.get_next_version(*args, **kwargs)

        # --- async half: the same work, off the event loop -----------------
        async def aget_tuple(self, config: Any) -> Any:
            import asyncio

            return await asyncio.to_thread(self._inner.get_tuple, config)

        async def alist(
            self,
            config: Any,
            *,
            filter: Any = None,  # noqa: A002 - LangGraph's parameter name
            before: Any = None,
            limit: int | None = None,
        ) -> Any:
            import asyncio

            # Materialised inside the thread: the underlying cursor is
            # synchronous, so it cannot be iterated from the loop.
            rows = await asyncio.to_thread(
                lambda: list(
                    self._inner.list(config, filter=filter, before=before, limit=limit)
                )
            )
            for row in rows:
                yield row

        async def aput(self, *args: Any, **kwargs: Any) -> Any:
            import asyncio
            import functools

            return await asyncio.to_thread(
                functools.partial(self._inner.put, *args, **kwargs)
            )

        async def aput_writes(self, *args: Any, **kwargs: Any) -> None:
            import asyncio
            import functools

            await asyncio.to_thread(
                functools.partial(self._inner.put_writes, *args, **kwargs)
            )

        async def adelete_thread(self, thread_id: str) -> None:
            import asyncio

            await asyncio.to_thread(self._inner.delete_thread, thread_id)

    return _ThreadedSaver(saver)


class _SyncCloser:
    """Adapts a synchronous context manager to the async close path."""

    def __init__(self, context: Any) -> None:
        self._context = context

    async def __aexit__(self, *exc_info: object) -> None:
        self._context.__exit__(None, None, None)


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
