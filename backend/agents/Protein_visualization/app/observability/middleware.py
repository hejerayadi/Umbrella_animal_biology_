"""Request correlation and access logging."""

import logging
from collections.abc import Awaitable, Callable
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.agents.Protein_visualization.app.observability.context import (
    bind_log_context,
    request_duration_ms,
    reset_log_context,
    reset_request_timer,
    start_request_timer,
)
from backend.agents.Protein_visualization.app.observability.logging import log_event

REQUEST_ID_HEADER = "X-Request-Id"
TRACE_ID_HEADER = "X-Trace-Id"

logger = logging.getLogger("app.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Binds request/trace ids to the log context and logs one line per request.

    The Grand Orchestrator propagates its ``trace_id`` through the header; when it
    is absent a new one is generated so every log line stays correlatable.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        trace_id = request.headers.get(TRACE_ID_HEADER) or str(uuid4())
        token = bind_log_context(request_id=request_id, trace_id=trace_id)
        started_token = start_request_timer()
        request.state.request_id = request_id
        request.state.trace_id = trace_id

        log_event(
            logger,
            "http_request.started",
            logging.DEBUG,
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            log_event(
                logger,
                "http_request.failed",
                logging.ERROR,
                exc_info=True,
                method=request.method,
                path=request.url.path,
                status="failed",
                status_code=500,
                duration_ms=request_duration_ms(),
                error_code=type(exc).__name__,
            )
            raise
        else:
            log_event(
                logger,
                "http_request.completed",
                logging.INFO if response.status_code < 500 else logging.ERROR,
                method=request.method,
                path=request.url.path,
                status="completed" if response.status_code < 400 else "rejected",
                status_code=response.status_code,
                duration_ms=request_duration_ms(),
            )
            response.headers[REQUEST_ID_HEADER] = request_id
            response.headers[TRACE_ID_HEADER] = trace_id
            return response
        finally:
            reset_request_timer(started_token)
            reset_log_context(token)
