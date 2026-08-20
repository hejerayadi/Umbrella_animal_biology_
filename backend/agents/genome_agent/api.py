"""HTTP boundary for the Genome Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation in `mock.py`, and returns whatever `AgentResult` comes
back. The orchestrator is the only caller; the frontend never reaches an agent
directly.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.genome_agent.api:app --port 8001
"""
from __future__ import annotations

from fastapi import FastAPI

from .mock import GenomeMock
from .schema import AgentRequest, AgentResult, AgentStatus

app = FastAPI(title="Genome Agent")

# Built once at startup rather than per request: mocks are free to construct,
# but real implementations load models and open connections, and this keeps
# that cost out of the request path.
_agent = GenomeMock()


@app.post("/execute", response_model=AgentResult)
def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        return _agent.run(request)
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        return AgentResult(status=AgentStatus.FAILED, output=f"Genome Agent error: {exc}")
