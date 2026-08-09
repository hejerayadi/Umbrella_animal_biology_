"""Structured logging.

Every record is emitted as a single JSON object carrying the workflow fields the
implementation document requires: ``timestamp``, ``level``, ``trace_id``,
``task_id``, ``analysis_id``, ``node``, ``capability``, ``status``,
``duration_ms`` and ``error_code``. Context fields come from
``app.observability.context``; per-event fields are passed through ``extra``.
"""

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from backend.agents.Protein_visualization.app.observability.context import CONTEXT_FIELDS, get_log_context

EXTRA_KEY = "umbrella_fields"

FIELD_ORDER = (
    "timestamp",
    "level",
    "logger",
    "event",
    *CONTEXT_FIELDS,
    "status",
    "duration_ms",
    "error_code",
)

NOISY_LOGGERS = ("httpx", "httpcore", "qdrant_client", "urllib3", "sentence_transformers")

_BASE_FIELDS = {"timestamp", "level", "logger", "event", *CONTEXT_FIELDS}
_COMPACT_CONTEXT = {
    "request_id": "request",
    "trace_id": "trace",
    "task_id": "task",
    "analysis_id": "analysis",
}
_SUCCESS_STATUSES = {"completed", "success", "ready", "accept"}
_WARNING_STATUSES = {"partial", "revise", "rejected", "abstain", "abstained", "degraded"}


def _record_payload(record: logging.LogRecord) -> dict[str, Any]:
    """Build the canonical structured payload shared by every renderer."""
    payload: dict[str, Any] = {
        "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "event": record.getMessage(),
    }
    payload.update(get_log_context())
    payload.update(getattr(record, EXTRA_KEY, {}))
    if record.exc_info:
        payload["error"] = logging.Formatter().formatException(record.exc_info)
    return payload


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = _record_payload(record)
        ordered = {key: payload[key] for key in FIELD_ORDER if key in payload}
        ordered.update({key: value for key, value in payload.items() if key not in ordered})
        return json.dumps(ordered, default=str)


class KeyValueFormatter(logging.Formatter):
    """Human-readable variant for local development (``LOG_FORMAT=text``)."""

    def format(self, record: logging.LogRecord) -> str:
        fields = {**get_log_context(), **getattr(record, EXTRA_KEY, {})}
        rendered = " ".join(f"{key}={value}" for key, value in fields.items())
        line = f"[{record.levelname}] {record.getMessage()}"
        if rendered:
            line = f"{line} {rendered}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


class RichStructuredHandler(logging.Handler):
    """TTY-aware local renderer: compact success lines and detailed failure panels."""

    def __init__(self, console: Console | None = None) -> None:
        super().__init__()
        self.console = console or Console(stderr=True, no_color="NO_COLOR" in os.environ)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = _record_payload(record)
            renderable = (
                self._detail_panel(payload, record)
                if record.levelno >= logging.WARNING
                else self._compact_line(payload, record)
            )
            # logging.Handler.handle() holds the handler lock for this entire call,
            # so concurrent analyses cannot interleave a line or a panel.
            self.console.print(renderable)
        except Exception:
            self.handleError(record)

    def _compact_line(self, payload: dict[str, Any], record: logging.LogRecord) -> Text:
        status = str(payload.get("status", "")).lower()
        event = str(payload["event"])
        verdicts = {
            str(payload.get(key, "")).lower()
            for key in ("validation_status", "verdict", "deterministic_verdict")
        }
        warning = status in _WARNING_STATUSES or bool(verdicts & _WARNING_STATUSES)
        success = status in _SUCCESS_STATUSES or event.endswith(".completed")
        symbol, style = (
            ("!", "bold dark_orange") if warning else ("✓", "bold green") if success else ("•", "bold cyan")
        )

        line = Text(no_wrap=True, overflow="ellipsis")
        local_time = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        line.append(local_time, style="dim")
        line.append(f"  {symbol} ", style=style)
        line.append(f"{record.levelname:<5}", style=style)
        line.append(f"  {event}", style="bold white")

        duration = payload.get("duration_ms")
        if duration is not None:
            line.append(f"  {duration} ms", style="bright_cyan")

        excluded = _BASE_FIELDS | {"status", "duration_ms", "error_code", "error"}
        for key, value in payload.items():
            if key in excluded or value is None:
                continue
            line.append(f"  {key}=", style="dim")
            line.append(_compact_value(value), style="white")

        for key, label in (("node", "node"), ("capability", "cap")):
            value = payload.get(key)
            if value is not None and str(value) not in event:
                line.append(f"  {label}=", style="dim")
                line.append(_compact_value(value), style="bright_black")

        for key, label in _COMPACT_CONTEXT.items():
            if value := payload.get(key):
                line.append(f"  {label}=", style="dim")
                line.append(_short_id(value), style="bright_black")
        return line

    def _detail_panel(self, payload: dict[str, Any], record: logging.LogRecord) -> Panel:
        failed = record.levelno >= logging.ERROR or str(payload.get("status", "")).lower() == "failed"
        color = "red" if failed else "dark_orange"
        symbol = "✕" if failed else "!"
        title = Text(f" {symbol} {record.levelname} · {payload['event']} ", style=f"bold {color}")

        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(style="dim", width=14, no_wrap=True)
        table.add_column(style="white", overflow="fold")
        ordered_keys = (
            "timestamp",
            "logger",
            "status",
            "duration_ms",
            "error_code",
            *CONTEXT_FIELDS,
        )
        displayed = {"event", "level", "error"}
        for key in ordered_keys:
            if key in payload and payload[key] is not None:
                table.add_row(key, _full_value(payload[key]))
                displayed.add(key)
        for key, value in payload.items():
            if key not in displayed and value is not None:
                table.add_row(key, _full_value(value))
        components: list[Any] = [table]
        if error := payload.get("error"):
            components.extend([Text(""), Text(str(error), style="red" if failed else "yellow")])
        return Panel(Group(*components), title=title, title_align="left", border_style=color, expand=True)


def _short_id(value: Any) -> str:
    rendered = str(value)
    return rendered[:8] if len(rendered) > 8 else rendered


def _compact_value(value: Any, limit: int = 48) -> str:
    rendered = _full_value(value)
    return rendered if len(rendered) <= limit else f"{rendered[: limit - 1]}…"


def _full_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


def configure_logging(level: str = "INFO", log_format: str | bool = "json") -> None:
    """Configure JSON deployment logs or TTY-aware pretty development logs.

    Boolean values remain accepted for callers compiled against the previous
    ``json_format`` argument: True selects JSON and False selects pretty output.
    """
    normalized = ("json" if log_format else "pretty") if isinstance(log_format, bool) else log_format.lower()
    if normalized not in {"json", "pretty", "text"}:
        raise ValueError("LOG_FORMAT must be one of: json, pretty, text")

    if normalized == "json":
        handler: logging.Handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = RichStructuredHandler()
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True  # replaced by the request middleware
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Emit ``event`` with structured ``fields`` merged into the JSON payload.

    The active context is snapshotted onto the record so the fields survive even
    when a handler formats the record after the context has been reset.
    """
    payload = {**get_log_context(), **{key: value for key, value in fields.items() if value is not None}}
    logger.log(level, event, exc_info=exc_info, extra={EXTRA_KEY: payload})


@contextmanager
def log_stage(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log ``<event>`` start/completion with ``duration_ms``, and failures with ``error_code``.

    Extra fields collected in the yielded dict are attached to the closing record,
    so a node can report what it produced::

        with log_stage(logger, "search_pdb", node="search_pdb") as result:
            result["candidates"] = len(candidates)
    """
    started = time.perf_counter()
    result: dict[str, Any] = {}
    log_event(logger, f"{event}.started", logging.DEBUG, **fields)
    try:
        yield result
    except Exception as exc:
        log_event(
            logger,
            f"{event}.failed",
            logging.WARNING,
            exc_info=True,
            status="failed",
            duration_ms=_elapsed_ms(started),
            error_code=type(exc).__name__,
            **fields,
            **result,
        )
        raise
    log_event(
        logger,
        f"{event}.completed",
        status="completed",
        duration_ms=_elapsed_ms(started),
        **fields,
        **result,
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
