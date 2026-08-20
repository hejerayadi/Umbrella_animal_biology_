"""Structured logging, on structlog with rich rendering in development.

Every log line is a set of key/value pairs rather than a formatted string, so
the same event is readable in a terminal and queryable once shipped. Two
renderers, chosen by `LOG_FORMAT`:

- `pretty` - rich-rendered colour and aligned columns, for a developer.
- `json`   - one object per line, for a log aggregator.

The correlation id set by asgi-correlation-id is bound onto every record by a
processor, so one HTTP request - and the whole reconstruction it triggers -
can be followed across the graph, the tools and the HTTP clients without any
call site passing an id around.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from asgi_correlation_id import correlation_id
from rich.console import Console
from rich.logging import RichHandler

from configuration.settings import LogFormat, ObservabilitySettings

_CONFIGURED = False

# Chatty at INFO and duplicating what we already log at the tool boundary.
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio")


def add_correlation_id(
    _logger: object, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Bind the current request's correlation id onto the record.

    A processor rather than an argument at each call site: the id lives in a
    ContextVar that asgi-correlation-id sets per request, and threading it
    through every function signature would be noise.
    """
    request_id = correlation_id.get()
    if request_id:
        event_dict.setdefault("request_id", request_id)
    return event_dict


def drop_color_message(
    _logger: object, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Discard uvicorn's pre-coloured duplicate of the message.

    Uvicorn logs `color_message` alongside `event`; keeping both means every
    uvicorn line is rendered twice in JSON output.
    """
    event_dict.pop("color_message", None)
    return event_dict


def _shared_processors(settings: ObservabilitySettings) -> list[Any]:
    """Processors applied to records from structlog and stdlib logging alike.

    Order matters: context and metadata are added before any rendering, and
    exception formatting must come after the event is fully assembled.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_correlation_id,
        drop_color_message,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def configure_logging(
    settings: ObservabilitySettings | None = None,
    *,
    log_format: LogFormat | None = None,
) -> None:
    """Install structlog and route stdlib logging through it. Safe to call twice.

    Idempotent because uvicorn may import the app more than once (reload mode),
    and stacking handlers would duplicate every line.

    `log_format` is passed explicitly by `create_app`, which resolves it from
    `APP_ENV` when `LOG_FORMAT` is unset - this section cannot see the app
    environment on its own. Falling back to JSON rather than pretty when
    nothing is known keeps colour codes out of a log pipeline by default.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = settings or ObservabilitySettings()
    log_format = log_format or settings.log_format or LogFormat.JSON
    level = settings.log_level.upper()

    shared = _shared_processors(settings)

    structlog.configure(
        processors=[
            *shared,
            # Hands off to the stdlib formatter below, so third-party libraries
            # logging through `logging` get the same treatment as our own calls.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if log_format is LogFormat.JSON:
        renderer: Any = structlog.processors.JSONRenderer()
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
    else:
        # `colors=False`: RichHandler paints the line itself, and letting
        # structlog emit ANSI codes too would double them up.
        renderer = structlog.dev.ConsoleRenderer(colors=False)
        handler = RichHandler(
            console=Console(stderr=False),
            show_time=True,
            show_level=True,
            show_path=False,
            rich_tracebacks=True,
            markup=False,
            log_time_format="[%H:%M:%S]",
        )

    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=[
                *shared,
                structlog.processors.format_exc_info,
            ],
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Uvicorn installs its own handlers; clearing them makes it propagate to
    # ours instead of printing a second, differently-formatted copy.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """A bound logger for one module. Use `get_logger(__name__)`.

    Call it with an event name and keyword pairs rather than an f-string:

        log.info("gap_detected", gap_id=gap.identifier, length=gap.length)

    That is what makes the field queryable once the logs are shipped.
    """
    return structlog.stdlib.get_logger(name)


def bind_run_context(**values: Any) -> None:
    """Attach values to every log record emitted by this task from now on.

    Used at the start of a reconstruction so the run id and sequence id appear
    on every line the run produces, including from deep inside a tool.
    """
    structlog.contextvars.bind_contextvars(**values)


def clear_run_context() -> None:
    """Drop everything `bind_run_context` attached.

    Contextvars survive across awaits, so a run that did not clear up would
    leak its ids onto the next one handled by the same worker.
    """
    structlog.contextvars.clear_contextvars()
