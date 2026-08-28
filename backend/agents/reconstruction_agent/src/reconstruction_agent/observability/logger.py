"""Structured logging.

Every log record is a set of fields rather than a sentence, because the
questions asked of this agent afterwards are structured ones: which collections
were ranked and why, which tool ran, how long it took, what it returned, why a
replan fired, why one candidate won. A message that has to be parsed with a
regular expression cannot answer those.

What is never logged is the private reasoning of a language model. Decisions
are recorded as their outcome and their stated basis - the ranked candidates,
the deficit the critic named - not as hidden chain-of-thought.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog

#: Third-party loggers that are informative once and noise thereafter.
_NOISY = ("httpx", "httpcore", "urllib3", "asyncio", "hpack")

_configured = False


def configure_logging(
    *,
    level: str = "INFO",
    log_format: Literal["pretty", "json"] = "pretty",
) -> None:
    """Set up structlog and the standard library to render the same way.

    Idempotent: uvicorn imports the application module more than once under
    reload, and configuring twice would stack duplicate processors onto every
    record.
    """
    global _configured
    if _configured:
        return

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper(), force=True)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A logger that tags every record with `name`.

    The module name is bound as a field rather than taken from a standard
    library logger, because records are rendered straight to stdout without
    going through the logging hierarchy.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger().bind(logger=name)
    return logger


def bind_run_context(**fields: Any) -> None:
    """Attach fields to every record emitted for the rest of this run.

    Used for the correlation ids so that a whole reconstruction can be pulled
    out of the log stream by trace id, across every tool and service it touched.
    """
    structlog.contextvars.bind_contextvars(**fields)


def clear_run_context() -> None:
    """Drop the run-scoped fields. Always called in a `finally`."""
    structlog.contextvars.clear_contextvars()
