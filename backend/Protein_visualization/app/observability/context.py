"""Request/workflow scoped fields that every structured log line inherits."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

CONTEXT_FIELDS = ("request_id", "trace_id", "task_id", "analysis_id", "node", "capability")

_log_context: ContextVar[dict[str, Any] | None] = ContextVar("log_context", default=None)
_request_started_at: ContextVar[float | None] = ContextVar("request_started_at", default=None)


def get_log_context() -> dict[str, Any]:
    return dict(_log_context.get() or {})


def bind_log_context(**fields: Any) -> Token[dict[str, Any] | None]:
    """Add fields to the current context and return a token for ``reset_log_context``."""
    present = {key: value for key, value in fields.items() if value is not None}
    return _log_context.set({**get_log_context(), **present})


def reset_log_context(token: Token[dict[str, Any] | None]) -> None:
    _log_context.reset(token)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    token = bind_log_context(**fields)
    try:
        yield
    finally:
        reset_log_context(token)


def start_request_timer() -> Token[float | None]:
    return _request_started_at.set(time.perf_counter())


def reset_request_timer(token: Token[float | None]) -> None:
    _request_started_at.reset(token)


def request_duration_ms() -> int | None:
    """Elapsed time since the request entered the service, or ``None`` outside a request."""
    started = _request_started_at.get()
    return None if started is None else int((time.perf_counter() - started) * 1000)
