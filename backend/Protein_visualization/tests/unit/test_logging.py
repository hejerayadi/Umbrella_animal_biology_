import json
import logging
from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from app.observability.context import get_log_context, log_context
from app.observability.logging import (
    JsonFormatter,
    RichStructuredHandler,
    configure_logging,
    log_event,
    log_stage,
)


def _record(caplog: pytest.LogCaptureFixture) -> dict[str, object]:
    assert len(caplog.records) == 1
    return json.loads(JsonFormatter().format(caplog.records[0]))


def test_context_fields_are_merged_into_every_record(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test.context")
    with (
        caplog.at_level(logging.INFO, logger="test.context"),
        log_context(trace_id="trace-1", task_id="task-1"),
    ):
        log_event(logger, "node.completed", node="resolve_identity", status="completed")

    payload = _record(caplog)
    assert payload["event"] == "node.completed"
    assert payload["trace_id"] == "trace-1"
    assert payload["task_id"] == "task-1"
    assert payload["node"] == "resolve_identity"
    assert payload["level"] == "INFO"


def _pretty_logger(*, colors: bool) -> tuple[logging.Logger, StringIO]:
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=colors,
        no_color=not colors,
        color_system="standard" if colors else None,
        width=220,
    )
    logger = logging.getLogger(f"test.pretty.{colors}")
    logger.handlers = [RichStructuredHandler(console)]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger, stream


def test_pretty_info_is_compact_and_shortens_correlation_ids() -> None:
    logger, stream = _pretty_logger(colors=False)
    trace_id = "14e1d2c8-1465-4b76-8afa-fd66e4160000"
    task_id = "e6d14e1d-2c81-465f-b766-8afafd66e416"
    with log_context(trace_id=trace_id, task_id=task_id):
        log_event(
            logger,
            "fetch_annotations.completed",
            status="completed",
            duration_ms=231,
            annotations=25,
        )

    output = stream.getvalue()
    assert "✓ INFO" in output
    assert "fetch_annotations.completed" in output
    assert "231 ms" in output
    assert "annotations=25" in output
    assert "trace=14e1d2c8" in output
    assert "task=e6d14e1d" in output
    assert trace_id not in output
    assert "\x1b[" not in output


def test_pretty_warning_panel_keeps_full_ids_and_exception() -> None:
    logger, stream = _pretty_logger(colors=False)
    trace_id = "14e1d2c8-1465-4b76-8afa-fd66e4160000"
    with log_context(trace_id=trace_id):
        try:
            raise TimeoutError("RCSB timed out")
        except TimeoutError:
            log_event(
                logger,
                "search_experimental_structures.failed",
                logging.WARNING,
                exc_info=True,
                status="failed",
                duration_ms=15000,
                error_code="TimeoutError",
            )

    output = stream.getvalue()
    assert "WARNING" in output
    assert "search_experimental_structures.failed" in output
    assert trace_id in output
    assert "TimeoutError" in output
    assert "RCSB timed out" in output


def test_pretty_uses_ansi_only_when_console_supports_color() -> None:
    logger, stream = _pretty_logger(colors=True)
    log_event(logger, "workflow.completed", status="completed", duration_ms=10)

    assert "\x1b[" in stream.getvalue()


def test_pretty_marks_revise_verdict_as_warning() -> None:
    logger, stream = _pretty_logger(colors=False)
    log_event(
        logger,
        "run_scientific_critic.completed",
        status="completed",
        verdict="REVISE",
        duration_ms=20,
    )

    assert "! INFO" in stream.getvalue()


def test_text_alias_selects_rich_handler_and_json_keeps_json_formatter() -> None:
    with patch("app.observability.logging.logging.basicConfig") as basic_config:
        configure_logging("INFO", "text")
        pretty_handler = basic_config.call_args.kwargs["handlers"][0]
    assert isinstance(pretty_handler, RichStructuredHandler)

    with patch("app.observability.logging.logging.basicConfig") as basic_config:
        configure_logging("INFO", "json")
        json_handler = basic_config.call_args.kwargs["handlers"][0]
    assert isinstance(json_handler.formatter, JsonFormatter)


def test_context_is_restored_after_the_block() -> None:
    with log_context(trace_id="trace-1"):
        assert get_log_context()["trace_id"] == "trace-1"
    assert get_log_context() == {}


def test_log_stage_reports_duration_and_outcome(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test.stage")
    with caplog.at_level(logging.INFO, logger="test.stage"):
        with log_stage(logger, "search_pdb", node="search_pdb") as outcome:
            outcome["candidates"] = 3

    payload = _record(caplog)
    assert payload["event"] == "search_pdb.completed"
    assert payload["status"] == "completed"
    assert payload["candidates"] == 3
    assert isinstance(payload["duration_ms"], int)


def test_log_stage_records_error_code_and_reraises(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("test.stage.failure")
    with caplog.at_level(logging.WARNING, logger="test.stage.failure"):
        with pytest.raises(TimeoutError):
            with log_stage(logger, "search_pdb", node="search_pdb"):
                raise TimeoutError("RCSB timed out")

    payload = _record(caplog)
    assert payload["event"] == "search_pdb.failed"
    assert payload["status"] == "failed"
    assert payload["error_code"] == "TimeoutError"
    assert "RCSB timed out" in str(payload["error"])
