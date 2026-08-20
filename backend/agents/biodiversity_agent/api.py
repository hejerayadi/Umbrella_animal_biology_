"""HTTP boundary for the Biodiversity Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation, and returns whatever `AgentResult` comes back. The
orchestrator is the only caller; the frontend never reaches an agent directly.

Which implementation answers is chosen by BIODIVERSITY_AGENT_IMPL:

    BIODIVERSITY_AGENT_IMPL=mock          (default) the stub in `mock.py`
    BIODIVERSITY_AGENT_IMPL=orchestrator  the real LangGraph orchestrator,
                                          reached through `orchestrator_adapter.py`

The default stays `mock` so this endpoint cannot start failing because an LLM
key is missing on someone's machine. The orchestrator path needs this agent's
own venv and a configured LLM backend for intent classification.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.biodiversity_agent.api:app --port 8003
"""
from __future__ import annotations

import inspect
import logging
import os

from fastapi import FastAPI

from .mock import BiodiversityMock
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(title="Biodiversity Agent")


def _build_agent():
    """Pick the implementation this process serves.

    A failure to build the orchestrator falls back to the mock rather than
    killing the service: an agent that answers something is more useful to the
    Global Orchestrator than a port that never opens.
    """
    impl = os.getenv("BIODIVERSITY_AGENT_IMPL", "mock").strip().lower()
    if impl != "orchestrator":
        # print, not log: uvicorn owns the logging config and swallows module
        # INFO records, but stdout is streamed by run_agents with the agent's
        # name in front. Which implementation is serving is the single most
        # confusing thing to get wrong, so it must be impossible to miss.
        print(
            f"[Biodiversity] serving MOCK (BIODIVERSITY_AGENT_IMPL={impl!r}). "
            f"Set BIODIVERSITY_AGENT_IMPL=orchestrator for the real orchestrator.",
            flush=True,
        )
        return BiodiversityMock()

    try:
        from .orchestrator_adapter import OrchestratorBiodiversityAgent

        agent = OrchestratorBiodiversityAgent()
        print("[Biodiversity] serving the LangGraph ORCHESTRATOR", flush=True)
        return agent
    except Exception as exc:  # noqa: BLE001 - startup must not depend on it
        # Loud, because this looks exactly like the flag not working.
        print(
            f"[Biodiversity] BIODIVERSITY_AGENT_IMPL=orchestrator but it could NOT be "
            f"built ({type(exc).__name__}: {exc}); falling back to MOCK.",
            flush=True,
        )
        _logger.warning("orchestrator build failed", exc_info=True)
        return BiodiversityMock()


# Built once at startup rather than per request: mocks are free to construct,
# but real implementations load models and open connections, and this keeps
# that cost out of the request path.
_agent = _build_agent()


@app.get("/health")
def health() -> dict:
    """Which implementation this process is actually serving.

    Cheap way to answer "did the environment variable take?" without sending a
    real request and inferring the answer from the shape of the output.
    """
    return {
        "agent": "Biodiversity",
        "implementation": type(_agent).__name__,
        "is_orchestrator": type(_agent).__name__ == "OrchestratorBiodiversityAgent",
    }


@app.post("/execute", response_model=AgentResult)
async def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        # The mock answers synchronously, the orchestrator is async. Awaiting
        # only when there is something to await lets one code path serve both
        # without duplicating the error handling below.
        result = _agent.run(request)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        return AgentResult(status=AgentStatus.FAILED, output=f"Biodiversity Agent error: {exc}")
