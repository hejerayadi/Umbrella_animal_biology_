"""HTTP boundary for the Literature Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
LangGraph orchestrator, and returns whatever `AgentResult` comes back. The
orchestrator is the only caller; the frontend never reaches an agent directly.

`registry.py` reaches every worker as an independent service over `POST
/execute` - there is no in-process import path from the Global Orchestrator to
this package - so this FastAPI app is the agent's only way of being reachable.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.Literature_Agent.api:app --port 8004
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .orchestrator.graph import LiteratureOrchestrator
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(title="Literature Agent")

# Built once at startup rather than per request: constructing it compiles the
# LangGraph state machine, and that cost does not belong in the request path.
_agent = LiteratureOrchestrator()


@app.get("/health")
def health() -> dict:
    """Confirms the service is up and which implementation it serves."""
    return {"agent": "Literature", "implementation": type(_agent).__name__}


@app.post("/execute", response_model=AgentResult)
def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        return _agent.run_request(request)
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        _logger.warning("literature agent request failed", exc_info=True)
        return AgentResult(status=AgentStatus.FAILED, output=f"Literature Agent error: {exc}")
