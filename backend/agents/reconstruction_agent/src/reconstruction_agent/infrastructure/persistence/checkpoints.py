"""Where LangGraph stores run state between steps.

A reconstruction can run for minutes across several external job submissions.
Checkpointing lets a run be inspected mid-flight and resumed by thread id
rather than restarted from the first NCBI call.
"""
from __future__ import annotations

from typing import Any

from ...configuration.logging import get_logger

_log = get_logger(__name__)


def build_checkpointer(kind: str = "memory") -> Any | None:
    """The checkpointer to compile the graph with, or None for no checkpointing.

    In-memory is the default and is per-process: it survives steps within one
    agent process, not a restart. That is the right trade for a stateless HTTP
    worker behind the orchestrator - durable checkpointing would mean a shared
    store, which this agent does not own.

    Returns None when LangGraph is not installed, so the graph still compiles
    in a minimal environment (the tests exercise nodes directly).
    """
    if kind == "none":
        return None

    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:  # pragma: no cover - depends on install
        _log.warning("langgraph checkpointing unavailable; running without checkpoints.")
        return None

    return MemorySaver()
