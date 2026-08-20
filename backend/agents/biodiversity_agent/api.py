"""HTTP boundary for the Biodiversity Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation in `mock.py`, and returns whatever `AgentResult` comes
back. The orchestrator is the only caller; the frontend never reaches an agent
directly.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.biodiversity_agent.api:app --port 8003
"""
from __future__ import annotations

from fastapi import FastAPI

from .mock import BiodiversityMock
from .schema import AgentRequest, AgentResult, AgentStatus

app = FastAPI(title="Biodiversity Agent")

# Built once at startup rather than per request: mocks are free to construct,
# but real implementations load models and open connections, and this keeps
# that cost out of the request path.
_agent = BiodiversityMock()


# The Global Orchestrator knows four statuses. M2 and M3 use two more
# internally, per their design documents, so they are narrowed here - the one
# place that faces the orchestrator. The original value is preserved in the
# payload under `status_detail`, so nothing is lost on the way out.
_WIRE_STATUS = {
    AgentStatus.PARTIAL: AgentStatus.COMPLETED,
    AgentStatus.NEEDS_CLARIFICATION: AgentStatus.FAILED,
}


def _to_wire(result: AgentResult) -> AgentResult:
    """Narrow an internal status to the platform contract, keeping the detail."""

    wire = _WIRE_STATUS.get(result.status)
    if wire is None:
        return result

    detail = result.status.value
    if isinstance(result.output, dict):
        result.output = {**result.output, "status_detail": detail}
    elif result.output is None:
        result.output = {"status_detail": detail}
    else:
        result.output = {"status_detail": detail, "payload": result.output}
    result.status = wire
    return result


@app.post("/execute", response_model=AgentResult)
def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        return _to_wire(_agent.run(request))
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        return AgentResult(status=AgentStatus.FAILED, output=f"Biodiversity Agent error: {exc}")
