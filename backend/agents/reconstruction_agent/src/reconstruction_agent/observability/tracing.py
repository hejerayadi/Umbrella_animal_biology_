"""Run correlation, and optional LangSmith tracing.

Every log line and event carries a run id so one reconstruction can be
followed across the graph, the tools and the HTTP clients. The repo already
traces the protein agent's LangGraph workflow with LangSmith; this wires the
same thing here, behind a flag.
"""
from __future__ import annotations

import os
import uuid
from contextvars import ContextVar

from ..configuration.logging import get_logger
from ..configuration.settings import ObservabilitySettings

_log = get_logger(__name__)

# A ContextVar rather than a plain global: concurrent runs in one process each
# see their own id, including across await points.
_current_run_id: ContextVar[str | None] = ContextVar("reconstruction_run_id", default=None)


def new_run_id() -> str:
    """A fresh, short correlation id."""
    return uuid.uuid4().hex[:12]


def set_run_id(run_id: str) -> None:
    _current_run_id.set(run_id)


def current_run_id() -> str | None:
    return _current_run_id.get()


def configure_tracing(settings: ObservabilitySettings) -> bool:
    """Enable LangSmith tracing if it is switched on and configured.

    Returns whether tracing is active. LangSmith reads its configuration from
    the environment, so this sets those variables rather than holding a client.
    """
    if not settings.langsmith_tracing:
        return False

    if not settings.langsmith_api_key:
        _log.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty; not tracing.")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

    _log.info("LangSmith tracing enabled for project %s.", settings.langsmith_project)
    return True
