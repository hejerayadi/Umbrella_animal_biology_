"""HTTP boundary for the Multimodal Recognition Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation in `agent.py`, and returns whatever `AgentResult` comes
back. The orchestrator is the only caller; the frontend never reaches an agent
directly.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.multimodal_recognition_agent.api:app --port 8005
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Header

from .agent import RecognitionAgent
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal Recognition Agent")

# The code for "something we did not anticipate went wrong here".
#
# It is a literal rather than an `ErrorCode` member because `domain/errors.py`
# is outside this correction's authorized boundary, and it follows the
# precedent already set by `workflows/graph.py`, which emits the bare string
# "WORKFLOW_INCOMPLETE" for its own defensive cases. `WORKFLOW_INCOMPLETE` was
# deliberately NOT reused: it means "the workflow ran and produced no result",
# which is a different and more specific claim than "an unexpected exception
# escaped". Its canonical home is the ErrorCode enum, once that file may change.
INTERNAL_ERROR_CODE = "INTERNAL_ERROR"

# Fixed, data-free, and identical for every unexpected failure. The whole point
# is that two different exceptions produce byte-identical text, so nothing about
# what went wrong - or about what was sent - can be inferred from the response.
INTERNAL_ERROR_MESSAGE = (
    "The Recognition Agent encountered an internal error and could not complete "
    "this request. No species was identified. The failure has been logged for "
    "the operators of the service."
)

# HTTP client libraries that log the request URL, and therefore the connection
# details in it. This agent's own logger records the exception type and nothing
# else, but the libraries underneath it are not so careful:
#
#   - at INFO, httpx logs the full Azure request URL, which carries the endpoint
#     host and the resource name;
#   - at DEBUG, httpcore and urllib3 additionally log the deployment name and
#     the NCBI email, the latter percent-encoded in the Entrez query string -
#     which is why a leak scan looking for "name@host" reports clean while the
#     address is sitting in the log.
#
# The Azure API key is not affected either way: it travels in a header, never in
# a URL. Pinned to WARNING so an operator raising the ROOT level for diagnosis
# does not silently start writing endpoints and an address to disk.
QUIETED_HTTP_LOGGERS = ("httpx", "httpcore", "urllib3", "openai")
QUIETED_HTTP_LOGGER_LEVEL = logging.WARNING


def quiet_dependency_http_loggers() -> None:
    """Stop the HTTP client libraries logging connection details.

    Applied at the service entry point. A caller that imports `RecognitionAgent`
    directly rather than serving it over HTTP - the smoke scripts do - does not
    pass through here and keeps its libraries' own levels.
    """
    for name in QUIETED_HTTP_LOGGERS:
        logging.getLogger(name).setLevel(QUIETED_HTTP_LOGGER_LEVEL)


quiet_dependency_http_loggers()

# Built once at startup rather than per request: mocks are free to construct,
# but real implementations load models and open connections, and this keeps
# that cost out of the request path.
_agent = RecognitionAgent()


@app.post("/execute", response_model=AgentResult)
def execute(
    request: AgentRequest,
    x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`.

    `X-Trace-Id` is optional and read-only: the Global Orchestrator sends one
    on every worker call (see `worker_node.py`), but this agent must behave
    identically whether it is present, absent, or malformed - including when
    called directly (smoke tests, `RecognitionAgent()` used standalone). It is
    never validated, parsed, or trusted for anything beyond being echoed into
    this agent's own trace event, so an Orchestrator-side format change can
    never break Recognition.
    """

    try:
        return _agent.run(request, trace_id=x_trace_id)
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # This used to interpolate the exception into the response body, so any
        # exception that was not a RecognitionError returned its message - which
        # may carry provider internals, a URL or request data - to the caller
        # verbatim. Nothing derived from the exception may cross this boundary:
        # not str(exc), not repr(exc), not its args, and not a traceback.
        #
        # A correlation id ties the caller's failure to the logged one without
        # putting anything about the failure into the response.
        request_id = uuid.uuid4().hex
        _logger.error(
            "[Recognition] stage=api.execute request_id=%s unexpected %s; "
            "returning a controlled internal error.",
            request_id, type(exc).__name__,
        )

        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        #
        # The output is the same {error_code, error} dict every other failure
        # returns, so a caller can read output["error_code"] on every branch
        # rather than having to guess whether this one is a string.
        return AgentResult(
            status=AgentStatus.FAILED,
            output={
                "error_code": INTERNAL_ERROR_CODE,
                "error": INTERNAL_ERROR_MESSAGE,
            },
        )
