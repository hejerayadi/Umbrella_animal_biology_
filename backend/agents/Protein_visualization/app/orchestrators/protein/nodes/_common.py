"""Shared helpers for workflow nodes."""

import logging
from typing import Any

from backend.agents.Protein_visualization.app.domain.exceptions import ProteinAgentError
from backend.agents.Protein_visualization.app.observability.logging import log_event

logger = logging.getLogger("app.workflow")


def executed(node: str, **updates: Any) -> dict[str, Any]:
    return {"executed_nodes": {node}, **updates}


def degraded(node: str, code: str, exc: Exception, **updates: Any) -> dict[str, Any]:
    """Record a recoverable provider failure as a coded warning instead of raising."""
    log_event(
        logger,
        "protein.node.degraded",
        logging.WARNING,
        node=node,
        status="degraded",
        error_code=code,
        exception_type=type(exc).__name__,
        retry_count=1,
    )
    return {
        "executed_nodes": {node},
        "warnings": [f"{code}: {exc}"],
        "errors": [f"{node}: {type(exc).__name__}"],
        "retry_counts": {node: 1},
        **updates,
    }


def failed(node: str, code: str, exc: Exception, **updates: Any) -> dict[str, Any]:
    """Record a non-recoverable node outcome without exposing payload contents."""
    log_event(
        logger,
        "protein.node.failed",
        logging.ERROR,
        node=node,
        status="failed",
        error_code=code,
        exception_type=type(exc).__name__,
    )
    return {
        "executed_nodes": {node},
        "warnings": [f"{code}: {exc}"],
        "errors": [f"{node}: {type(exc).__name__}"],
        **updates,
    }


def failure_code(exc: Exception, unavailable: str, timeout: str) -> str:
    message = str(exc).lower()
    return timeout if "timeout" in message or "timed out" in message else unavailable


__all__ = ["ProteinAgentError", "degraded", "executed", "failed", "failure_code", "logger"]
