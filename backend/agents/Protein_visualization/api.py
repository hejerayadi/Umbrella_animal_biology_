"""HTTP boundary for the Protein Visualization Agent.

This is the service `backend/run_agents.py` starts on port 8008. One process
serves two contracts, because splitting them would mean two different answers
to the same question depending on who asked:

* `POST /execute` is the platform contract - `{instruction, context}` in, an
  `AgentResult` out. It holds no business logic: it validates the request and
  hands it to `orchestrator_adapter.py`, which resolves the gene and species
  named in the chat message against UniProt before running the workflow. The
  Global Orchestrator is the only caller.
* `/api/v1/...` is the scientific API, mounted from `app/main.py` - the full
  analysis with the Mol* scene, every annotation and every evidence record,
  for the frontend viewer and for scripts.
* `/health` answers the platform's liveness check, `/api/v1/ready` reports
  whether Qdrant, the LLM and persistence are actually reachable.
* `/docs` documents the versioned API.

There is one implementation and it does real work. The `mock.py` stub that used
to sit here has been retired - it answered every request by demanding a genome
and then a trait, and finished by returning the fixed string "Predicted 3D
Protein Structure". On a platform whose premise is never inventing scientific
results, that is the wrong thing to serve, and it made the agent look broken
for a different reason than it actually was.

That means UniProt, RCSB, AlphaFold or InterPro being down shows up as a FAILED
result the Responder explains honestly. That is intended.

Run it (from the repository root):

    uv run --project backend/agents/Protein_visualization \
        python -m uvicorn backend.agents.Protein_visualization.api:app --port 8008
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from .app.main import create_app
from .app.observability.logging import log_stage
from .orchestrator_adapter import OrchestratorProteinAgent
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app: FastAPI = create_app()

# Built once at startup rather than per request: constructing it compiles the
# LangGraph state machine and opens the pooled UniProt connection, and neither
# cost belongs in the request path.
_agent = OrchestratorProteinAgent()


@app.get("/health")
def health() -> dict:
    """Confirms the service is up and which implementation it serves."""
    return {"agent": "Protein", "implementation": type(_agent).__name__}


@app.post("/execute", response_model=AgentResult)
async def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        with log_stage(
            _logger,
            "protein.execute",
            start_level=logging.INFO,
            node="execute",
            capability="inter_agent",
            instruction_length=len(request.instruction),
            context_keys=sorted(request.context),
        ) as outcome:
            result = await _agent.run(request)
            outcome["agent_status"] = result.status.value
            outcome["target_agent"] = result.target_agent
            return result
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        return AgentResult(status=AgentStatus.FAILED, output=f"Protein Visualization Agent error: {exc}")
