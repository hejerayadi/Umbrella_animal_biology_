import logging

import httpx
import pytest

from backend.agents.Protein_visualization.app.domain.exceptions import UpstreamServiceError
from backend.agents.Protein_visualization.app.observability.logging import (
    EXTRA_KEY,
    JsonFormatter,
    log_stage,
)
from backend.agents.Protein_visualization.app.tools.http import JsonHttpClient


async def test_http_tool_logs_retry_completion_and_safe_route(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, request=request)
        return httpx.Response(200, json={"results": []}, request=request)

    async with httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    ) as client:
        tool = JsonHttpClient("RCSB", "https://example.test", 1.0, 1, client)
        with caplog.at_level(logging.DEBUG):
            with log_stage(logging.getLogger("test.node"), "protein.node.search"):
                result = await tool.request_json("get", "/search?token=must-not-be-logged")

    assert result == {"results": []}
    records = {
        record.getMessage(): getattr(record, EXTRA_KEY, {})
        for record in caplog.records
        if record.name == "app.tools.http"
    }
    retried = records["protein.tool.request.retried"]
    completed = records["protein.tool.request.completed"]
    assert retried["attempt"] == 1
    assert retried["status_code"] == 503
    assert retried["backoff_ms"] == 200
    assert completed["attempts"] == 2
    assert completed["status_code"] == 200
    assert completed["route"] == "/search"
    assert "must-not-be-logged" not in str(records)
    assert completed["parent_span_id"]


async def test_http_tool_logs_terminal_failure_and_wraps_provider_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    async with httpx.AsyncClient(
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
    ) as client:
        tool = JsonHttpClient("AlphaFold", "https://example.test", 1.0, 0, client)
        with (
            caplog.at_level(logging.WARNING, logger="app.tools.http"),
            pytest.raises(UpstreamServiceError),
        ):
            await tool.request_json("GET", "/prediction/P04637?api_key=must-not-be-logged")

    failed = next(record for record in caplog.records if record.getMessage() == "protein.tool.request.failed")
    fields = getattr(failed, EXTRA_KEY)
    assert fields["status"] == "failed"
    assert fields["provider"] == "AlphaFold"
    assert fields["route"] == "/prediction/P04637"
    assert fields["error_code"] == "UpstreamServiceError"
    assert fields["upstream_error_code"] == "RetryableHttpError"
    assert "must-not-be-logged" not in JsonFormatter().format(failed)
