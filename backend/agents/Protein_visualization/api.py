"""HTTP boundary for the Protein Visualization Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to
`orchestrator_adapter.py`, and returns whatever `AgentResult` comes back. The
orchestrator is the only caller; the frontend never reaches an agent directly.

There is one implementation, and it does real work: every request resolves the
gene and species against UniProt, then runs the LangGraph workflow that picks
an experimental or predicted structure and builds the Mol* scene from it. The
`mock.py` stub that used to sit here has been retired - it answered every
request by demanding a genome and then a trait, and finished by returning the
fixed string "Predicted 3D Protein Structure". On a platform whose premise is
never inventing scientific results, that is the wrong thing to serve, and it
made the agent look broken for a different reason than it actually was.

That means UniProt, RCSB, AlphaFold or InterPro being down shows up as a
FAILED result the Responder explains honestly. That is intended.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.Protein_visualization.api:app --port 8008
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .orchestrator_adapter import OrchestratorProteinAgent
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(title="Protein Visualization Agent")

# Built once at startup rather than per request: constructing it compiles the
# LangGraph state machine and opens the pooled UniProt connection, and neither
# cost belongs in the request path.
_agent = OrchestratorProteinAgent()

print("[Protein] serving the LangGraph workflow (live UniProt/RCSB/AlphaFold)", flush=True)


@app.get("/health")
def health() -> dict:
    """Confirms the service is up and which implementation it serves."""
    return {"agent": "Protein", "implementation": type(_agent).__name__}


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
        _logger.warning("protein agent request failed", exc_info=True)
        return AgentResult(
            status=AgentStatus.FAILED, output=f"Protein Visualization Agent error: {exc}"
        )
