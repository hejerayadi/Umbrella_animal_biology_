"""HTTP boundary for the Trait Discovery Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation, and returns whatever `AgentResult` comes back. The
orchestrator is the only caller; the frontend never reaches an agent directly.

This wraps the mock implementation in `mock.py`, deliberately - NOT the
LangGraph sub-orchestrator under `workflows/`. That workflow has its own
richer input/output schemas (`schemas/`) and its own dependency set; wiring it
in behind this endpoint is a separate step.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.trait_discovery_agent.api:app --port 8007
"""
from __future__ import annotations

from fastapi import FastAPI

from .mock import TraitMock
from .schema import AgentRequest, AgentResult, AgentStatus
from kb.qdrant_store import ensure_collections

app = FastAPI(title="Trait Discovery Agent")

# Built once at startup rather than per request: mocks are free to construct,
# but real implementations load models and open connections, and this keeps
# that cost out of the request path.
_agent = TraitMock()


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
        return AgentResult(
            status=AgentStatus.FAILED, output=f"Trait Discovery Agent error: {exc}"
        )




@app.on_event("startup")
async def startup():
    await ensure_collections()

