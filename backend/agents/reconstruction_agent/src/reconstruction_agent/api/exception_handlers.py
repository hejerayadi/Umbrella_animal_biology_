"""Turning domain failures into correct HTTP responses.

Two rules decide the status code.

HTTP semantics are used properly: a missing accession is 404, a structurally
valid request with impossible values is 422, an upstream service that broke is
502, one that is merely unavailable is 503, and a deadline that passed is 504.
Answering 200 for everything and hiding the outcome in the body would make the
status code useless to anyone reading it.

And insufficient evidence is not a failure. A gap that cannot be reconstructed
comes back 200 with `status: UNRESOLVED` and a reason, because refusing to
invent a sequence is the correct scientific answer, not an error condition.
That is why no code here maps to "we could not find enough homologues".
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from reconstruction_agent.api.v1.schemas.envelope import error_payload
from reconstruction_agent.domain.enums import ErrorCode
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.observability.logger import get_logger

_log = get_logger(__name__)

#: Domain error code -> HTTP status. Anything unlisted is a 500, which is the
#: honest answer for a failure the agent did not anticipate.
_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.INVALID_REQUEST: status.HTTP_400_BAD_REQUEST,
    ErrorCode.INVALID_GAP_COORDINATES: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.ASSEMBLY_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.SEQUENCE_NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.SEQUENCE_PROVIDER_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.TAXONOMY_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.DATABASE_CATALOGUE_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.BLAST_SUBMISSION_FAILED: status.HTTP_502_BAD_GATEWAY,
    ErrorCode.BLAST_TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
    ErrorCode.ALIGNMENT_FAILED: status.HTTP_502_BAD_GATEWAY,
    ErrorCode.EVO2_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.EVO2_TIMEOUT: status.HTTP_504_GATEWAY_TIMEOUT,
    ErrorCode.BUDGET_EXHAUSTED: status.HTTP_429_TOO_MANY_REQUESTS,
    ErrorCode.DEADLINE_EXCEEDED: status.HTTP_504_GATEWAY_TIMEOUT,
    ErrorCode.RATE_LIMITED: status.HTTP_429_TOO_MANY_REQUESTS,
    ErrorCode.UPSTREAM_UNAVAILABLE: status.HTTP_502_BAD_GATEWAY,
    ErrorCode.INTERNAL_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def http_status_for(code: ErrorCode) -> int:
    """The HTTP status that expresses `code`."""
    return _STATUS_BY_CODE.get(code, status.HTTP_500_INTERNAL_SERVER_ERROR)


#: The orchestrator's route. Handlers registered on the application are
#: global, so anything answering for `/api/v1` answers for this path too
#: unless it says otherwise - and a `{"detail": ...}` body here reaches the
#: user as "agent unreachable", with the real reason discarded.
ORCHESTRATOR_PATH = "/execute"


def _is_orchestrator(request: Request) -> bool:
    return request.url.path.rstrip("/") == ORCHESTRATOR_PATH


def _agent_failure(message: str) -> JSONResponse:
    """A bare `AgentResult`, which is the only shape the orchestrator parses.

    Always HTTP 200: the worker node turns any non-200 into a local transport
    failure, so an honest, well-described error returned as a 422 arrives as a
    connectivity problem instead.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "failed",
            "output": f"Reconstruction Agent error: {message}",
            "error": message,
            "retryable": False,
        },
    )


def register(app: FastAPI, *, request_id_header: str, trace_id_header: str) -> None:
    """Install the handlers on `app`.

    Everything below answers `/api/v1` with the `{data, meta, error}` envelope
    and the HTTP status the error code maps to - except on `POST /execute`,
    which always answers 200 with an `AgentResult`. That exception is enforced
    here rather than left to the endpoint, because request-body validation
    fails *before* any endpoint code runs, so a route-local try/except never
    sees it.
    """

    def _ids(request: Request) -> tuple[str | None, str | None]:
        return (
            request.headers.get(request_id_header),
            request.headers.get(trace_id_header),
        )

    @app.exception_handler(ReconstructionError)
    async def _domain_error(request: Request, exc: ReconstructionError) -> JSONResponse:
        if _is_orchestrator(request):
            _log.warning("request_failed_at_orchestrator_boundary", code=exc.code.value)
            return _agent_failure(exc.message)

        request_id, trace_id = _ids(request)
        http_status = http_status_for(exc.code)
        _log.warning(
            "request_failed",
            code=exc.code.value,
            status=http_status,
            path=request.url.path,
            retryable=exc.retryable,
        )
        return JSONResponse(
            status_code=http_status,
            content=error_payload(
                exc.code,
                exc.message,
                retryable=exc.retryable,
                details=exc.details,
                request_id=request_id,
                trace_id=trace_id,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        if _is_orchestrator(request):
            # The one case the endpoint cannot catch itself: FastAPI validates
            # the body before dispatching, so `execute` never runs.
            return _agent_failure("The request body did not match the expected schema.")

        request_id, trace_id = _ids(request)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=error_payload(
                ErrorCode.INVALID_REQUEST,
                "The request body did not match the expected schema.",
                details={"errors": exc.errors()},
                request_id=request_id,
                trace_id=trace_id,
            ),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        _log.exception("unhandled_error", path=request.url.path)
        if _is_orchestrator(request):
            # Reached only if something fails outside the endpoint's own try -
            # a dependency, or the response serialiser. The contract holds
            # regardless of where the failure happened.
            return _agent_failure("The agent failed unexpectedly.")

        request_id, trace_id = _ids(request)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_payload(
                ErrorCode.INTERNAL_ERROR,
                # The exception text is not echoed: it can carry internal
                # detail, and the correlation id is what actually lets someone
                # find this failure in the logs.
                "The agent failed unexpectedly. Quote the request id when reporting this.",
                request_id=request_id,
                trace_id=trace_id,
            ),
        )
