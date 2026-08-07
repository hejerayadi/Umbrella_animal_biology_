"""Shared helpers for workflow nodes."""

import logging
from typing import Any

from app.domain.exceptions import ProteinAgentError

logger = logging.getLogger("app.workflow")


def executed(node: str, **updates: Any) -> dict[str, Any]:
    return {"executed_nodes": {node}, **updates}


def degraded(node: str, code: str, exc: Exception, **updates: Any) -> dict[str, Any]:
    """Record a recoverable provider failure as a coded warning instead of raising."""
    return {
        "executed_nodes": {node},
        "warnings": [f"{code}: {exc}"],
        "errors": [f"{node}: {type(exc).__name__}"],
        "retry_counts": {node: 1},
        **updates,
    }


def failure_code(exc: Exception, unavailable: str, timeout: str) -> str:
    message = str(exc).lower()
    return timeout if "timeout" in message or "timed out" in message else unavailable


__all__ = ["ProteinAgentError", "degraded", "executed", "failure_code", "logger"]
