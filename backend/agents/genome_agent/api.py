"""HTTP boundary for the Genome Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to
`orchestrator_adapter.py`, and returns whatever `AgentResult` comes back. The
orchestrator is the only caller; the frontend never reaches an agent directly.

There is one implementation, and it does real work: every request resolves the
species against NCBI and fetches live assembly, metadata and gene data. The
`mock.py` stub that used to sit behind an env flag has been removed - it
returned a fabricated string like "Genome sequence of mouse", and on a platform
whose whole premise is never inventing scientific results, silently serving
fake data when NCBI is unreachable is the wrong failure. A real error the
Responder can explain honestly is better.

That means NCBI being down, slow, or rate-limiting shows up as a FAILED result.
That is intended.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.genome_agent.api:app --port 8001
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .orchestrator_adapter import OrchestratorGenomeAgent
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(title="Genome Agent")

# Built once at startup rather than per request: constructing it compiles the
# LangGraph state machine, and that cost does not belong in the request path.
_agent = OrchestratorGenomeAgent()

print("[Genome] serving the LangGraph orchestrator (live NCBI)", flush=True)


@app.get("/health")
def health() -> dict:
    """Confirms the service is up and which implementation it serves."""
    return {"agent": "Genome", "implementation": type(_agent).__name__}


@app.post("/execute", response_model=AgentResult)
async def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        return await _agent.run(request)
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        _logger.warning("genome agent request failed", exc_info=True)
        return AgentResult(status=AgentStatus.FAILED, output=f"Genome Agent error: {exc}")
