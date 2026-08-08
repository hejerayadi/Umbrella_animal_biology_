"""HTTP boundary for the Evolution Agent.

Communication only — this layer holds no business logic. It receives a
request from the Global Orchestrator, validates it into ``AgentRequest``,
hands it to the agent implementation, and returns whatever ``AgentResult``
comes back. The orchestrator is the only caller; the frontend never
reaches an agent directly.

Which implementation answers is chosen by EVOLUTION_AGENT_IMPL:

    EVOLUTION_AGENT_IMPL=mock          (default) the stub in ``mock.py``
    EVOLUTION_AGENT_IMPL=orchestrator  the real LangGraph orchestrator,
                                       reached through
                                       ``orchestrator_adapter.py``

The default stays ``mock`` so this endpoint cannot start failing because
an LLM key is missing on someone's machine. The orchestrator path needs
this agent's own venv and a configured LLM backend for intent
classification.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.evolution_agent.api:app --port 8002
"""

from __future__ import annotations

import inspect
import logging
import os

from fastapi import FastAPI

from .mock import EvolutionMock
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(title="Evolution Agent")


def _build_agent():
    """Pick the implementation this process serves.

    A failure to build the orchestrator falls back to the mock rather than
    killing the service: an agent that answers something is more useful to
    the Global Orchestrator than a port that never opens.
    """
    impl = os.getenv("EVOLUTION_AGENT_IMPL", "mock").strip().lower()

    if impl != "orchestrator":
        print(
            f"[Evolution] serving MOCK (EVOLUTION_AGENT_IMPL={impl!r}). "
            f"Set EVOLUTION_AGENT_IMPL=orchestrator for the real orchestrator.",
            flush=True,
        )
        return EvolutionMock()

    try:
        from .orchestrator_adapter import OrchestratorEvolutionAgent

        agent = OrchestratorEvolutionAgent()
        print("[Evolution] serving the LangGraph ORCHESTRATOR", flush=True)
        return agent
    except Exception as exc:  # noqa: BLE001 — startup must not crash the service
        print(
            f"[Evolution] EVOLUTION_AGENT_IMPL=orchestrator but it could NOT be "
            f"built ({type(exc).__name__}: {exc}); falling back to MOCK.",
            flush=True,
        )
        _logger.warning("orchestrator build failed", exc_info=True)
        return EvolutionMock()


# Built once at startup rather than per request: mocks are free to construct,
# but real implementations compile LangGraph graphs and open connections, and
# this keeps that cost out of the request path.
_agent = _build_agent()


@app.get("/health")
def health() -> dict:
    """Which implementation this process is actually serving.

    Cheap way to answer "did the environment variable take?" without
    sending a real request and inferring the answer from the shape of
    the output.
    """
    return {
        "agent": "Evolution",
        "implementation": type(_agent).__name__,
        "is_orchestrator": type(_agent).__name__ == "OrchestratorEvolutionAgent",
    }


@app.post("/execute", response_model=AgentResult)
async def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an ``AgentResult``."""

    try:
        # The mock answers synchronously; the orchestrator is async.
        # Awaiting only when there is something to await lets one code path
        # serve both without duplicating the error handling below.
        result = _agent.run(request)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as exc:  # noqa: BLE001 — boundary must not leak exceptions
        # Deliberately not an HTTPException: the Global Orchestrator expects
        # one schema back every time, and it already knows how to handle
        # FAILED. A 500 with FastAPI's {"detail": ...} body would break that
        # contract.
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"Evolution Agent error: {exc}",
        )
