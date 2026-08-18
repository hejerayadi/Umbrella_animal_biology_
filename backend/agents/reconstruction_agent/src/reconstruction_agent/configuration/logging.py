"""Logging setup for the agent process.

Two formats, chosen by `LOG_JSON`: human-readable lines for local development,
JSON lines for anywhere the output gets shipped to a log aggregator.

Every record carries the correlation id of the run it belongs to (see
`observability.tracing`), so one reconstruction can be followed end to end
across the graph, the tools and the HTTP clients.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .settings import ObservabilitySettings

_CONFIGURED = False

# Attributes present on every LogRecord. Anything else a caller attached via
# `extra=` is application context and belongs in the emitted payload.
_STANDARD_FIELDS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class JSONFormatter(logging.Formatter):
    """One JSON object per line, with `extra=` fields merged in at the top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Readable single line, with the run id when one is set."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        run_id = getattr(record, "run_id", None)
        return f"{line}  [run={run_id}]" if run_id else line


def configure_logging(settings: ObservabilitySettings | None = None) -> None:
    """Install the agent's handler on the root logger. Safe to call twice.

    Idempotent because uvicorn may import the app more than once (reload mode),
    and stacking handlers would duplicate every line.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = settings or ObservabilitySettings()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if settings.log_json else TextFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level.upper())

    # These two are chatty at INFO and say nothing we do not already log at the
    # tool boundary; the agent's own request logging is more useful.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Use `get_logger(__name__)`."""
    return logging.getLogger(name)
