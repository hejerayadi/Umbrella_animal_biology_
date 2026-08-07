"""HTTP boundary for the Trait Discovery Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation, and returns whatever `AgentResult` comes back. The
orchestrator is the only caller; the frontend never reaches an agent directly.

Which implementation answers is chosen by the TRAIT_AGENT_IMPL environment
variable:

    TRAIT_AGENT_IMPL=mock       (default) the canned stub in `mock.py`
    TRAIT_AGENT_IMPL=workflow   the real LangGraph pipeline under `workflows/`,
                                reached through `workflow_adapter.py`

The default stays `mock` so this endpoint cannot start failing just because the
workflow's dependencies or NIM key are missing on a given machine. Flip the
variable to run the real thing; flip it back if a demo goes wrong. Note the
workflow path needs this agent's own venv and a NIM key, the mock path needs
neither.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.trait_discovery_agent.api:app --port 8007
"""
from __future__ import annotations

import inspect
import logging
import os

from fastapi import FastAPI

from .mock import TraitMock
from .schema import AgentRequest, AgentResult, AgentStatus
from kb.qdrant_store import ensure_collections

_logger = logging.getLogger(__name__)

app = FastAPI(title="Trait Discovery Agent")


def _build_agent():
    """Pick the implementation this process will serve.

    A failure to build the workflow (missing dependency, missing NIM key) falls
    back to the mock rather than killing the service: a degraded agent that
    answers is more useful to the orchestrator than a port that refuses to open.
    """
    impl = os.getenv("TRAIT_AGENT_IMPL", "mock").strip().lower()
    if impl != "workflow":
        _logger.info("Trait Discovery Agent serving the mock implementation")
        return TraitMock()

    try:
        from .workflow_adapter import WorkflowTraitAgent

        agent = WorkflowTraitAgent()
        _logger.info("Trait Discovery Agent serving the LangGraph workflow")
        return agent
    except Exception as exc:  # noqa: BLE001 - startup must not depend on the workflow
        _logger.warning(
            "TRAIT_AGENT_IMPL=workflow but the workflow could not be built (%s); "
            "falling back to the mock",
            exc,
        )
        return TraitMock()


# Built once at startup rather than per request: mocks are free to construct,
# but real implementations load models and open connections, and this keeps
# that cost out of the request path.
_agent = _build_agent()


@app.post("/execute", response_model=AgentResult)
async def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        # The mock answers synchronously, the workflow is async all the way
        # down. Awaiting only when there is something to await lets one code
        # path serve both instead of duplicating the error handling below.
        result = _agent.run(request)
        if inspect.isawaitable(result):
            result = await result
        return result
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

