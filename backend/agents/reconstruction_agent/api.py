"""HTTP boundary for the Reconstruction Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation in `agent.py`, and returns whatever `ReconstructionResult` comes
back. The orchestrator is the only caller; the frontend never reaches an agent
directly.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.reconstruction_agent.api:app --port 8006
"""
from __future__ import annotations

import uuid
import logging
from fastapi import FastAPI

from .agent import graph
from .schema import AgentRequest, AgentResult, AgentStatus, ReconstructionResult

app = FastAPI(title="Reconstruction Agent")
logger = logging.getLogger(__name__)


def get_session_id(request: AgentRequest) -> str:
    """Extracts session_id from context or generates one."""
    session_id = request.context.get("session_id")
    if not session_id:
        logger.warning("No session_id found in request context. Generating a fallback UUID.")
        session_id = str(uuid.uuid4())
        # TODO: session_id devrait toujours être fourni par l'Orchestrateur une fois core/contracts.py finalisé
    return session_id


def request_to_state(request: AgentRequest) -> dict:
    """Converts the AgentRequest into a complete initial state."""
    return {
        "request": request,
    }


@app.post("/execute", response_model=ReconstructionResult)
def execute(request: AgentRequest) -> ReconstructionResult:
    """The agent's single endpoint. Always answers with a `ReconstructionResult`."""

    try:
        session_id = get_session_id(request)
        state = request_to_state(request)
        
        final_state = graph.invoke(state, config={"configurable": {"thread_id": session_id}})
        
        result = final_state.get("result", None)
        if result is None:
            return ReconstructionResult(
                status=AgentStatus.FAILED,
                output="Graph did not return a result."
            )

        # When the graph exits early via NEEDS_AGENT (e.g. extinct species waiting for
        # Evolution Agent), the result is a plain AgentResult, not a ReconstructionResult.
        # Promote it so FastAPI's response_model validation always sees the full schema.
        if not isinstance(result, ReconstructionResult):
            return ReconstructionResult(
                status=result.status,
                target_agent=result.target_agent,
                prompt_to_target_agent=result.prompt_to_target_agent,
                output=result.output,
            )

        return result
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        logger.exception("Error during execution")
        return ReconstructionResult(status=AgentStatus.FAILED, output=f"Reconstruction Agent error: {exc}")

