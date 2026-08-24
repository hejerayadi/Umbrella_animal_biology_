"""HTTP boundary for the Trait Discovery Agent.

Communication only - this layer holds no business logic. It receives a request
from the Global Orchestrator, validates it into `AgentRequest`, hands it to the
agent implementation, and returns whatever `AgentResult` comes back. The
orchestrator is the only caller; the frontend never reaches an agent directly.

Which implementation answers is chosen by TRAIT_AGENT_IMPL:
  workflow  (default) the real LangGraph pipeline under `workflows/`, via
            `workflow_adapter.py`
  mock      the canned stub in `mock.py`, for an offline demo

`backend/run_agents.py` sets TRAIT_AGENT_IMPL=workflow explicitly and anything
exported in the shell still wins, so `set TRAIT_AGENT_IMPL=mock` gets the stub
back. If the workflow cannot be built - no NIM key, a missing dependency - the
port still opens on the stub rather than failing to start, and says so loudly
on the console.

Run it (from the repository root, with this agent's venv active):

    python -m uvicorn backend.agents.trait_discovery_agent.api:app --port 8007
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
from pathlib import Path

# Everything under `workflows/`, `subagents/`, `schemas/` and `kb/` imports its
# siblings as top-level modules (`from kb.sources.go_client import ...`), which
# resolves when that code is run from inside this directory - the way its own
# CLI and pytest.ini do it. Uvicorn is started from the repository root with
# `backend.agents.trait_discovery_agent.api:app`, so this package's directory
# is NOT on the path and every one of those imports would fail. Adding it is
# what lets one module tree serve both entry points. `tests/conftest.py` does
# the same thing for the same reason.
_AGENT_DIR = str(Path(__file__).resolve().parent)
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from fastapi import FastAPI

from .mock import TraitMock
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trait Discovery Agent",
    description=(
        "Identifies the genes behind an observable trait.\n\n"
        "Pipeline: **Gene Mapper** (Gene Ontology) → **Functional Evidence** "
        "(KEGG pathways + UniProt proteins, in parallel) → **Literature "
        "Support** → an LLM-written explanation."
    ),
    version="2.0.0",
)


def _build_agent():
    """The implementation this process will serve, chosen once at startup."""

    impl = os.getenv("TRAIT_AGENT_IMPL", "workflow").strip().lower()

    if impl == "mock":
        print(
            "[Trait] serving the MOCK stub (TRAIT_AGENT_IMPL=mock). "
            "Unset it for the real LangGraph workflow.",
            flush=True,
        )
        return TraitMock()

    try:
        from .workflow_adapter import WorkflowTraitAgent

        agent = WorkflowTraitAgent()
        print("[Trait] serving the LangGraph WORKFLOW", flush=True)
        return agent
    except Exception as exc:  # noqa: BLE001 - a broken workflow must not close the port
        print(
            f"[Trait] workflow failed to build ({type(exc).__name__}: {exc}); "
            f"falling back to MOCK.",
            flush=True,
        )
        _logger.warning("workflow build failed", exc_info=True)
        return TraitMock()


# Built once at startup rather than per request: the graph pulls in every
# subagent and its LLM client, and that cost does not belong in the request
# path.
_agent = _build_agent()


@app.get("/health", summary="Health check")
def health() -> dict:
    """Reports which implementation is currently answering."""
    return {
        "agent": "Trait",
        "implementation": type(_agent).__name__,
        "is_workflow": type(_agent).__name__ == "WorkflowTraitAgent",
    }


@app.post("/execute", response_model=AgentResult)
async def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an `AgentResult`."""

    try:
        result = _agent.run(request)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as exc:  # noqa: BLE001 - the boundary must not leak exceptions
        # Deliberately not an HTTPException: the orchestrator's router expects
        # one schema back every time, and it already knows how to handle a
        # FAILED status. A 500 with FastAPI's {"detail": ...} body would break
        # that contract.
        _logger.warning("request failed", exc_info=True)
        return AgentResult(
            status=AgentStatus.FAILED, output=f"Trait Discovery Agent error: {exc}"
        )


@app.on_event("startup")
async def startup() -> None:
    """Create the Qdrant cache collections, if a cache is configured.

    The cache is an optimisation: `kb/qdrant_store.py` already runs without
    qdrant-client installed. Its `get_client()` reads QDRANT_URL out of
    `os.environ` directly though, so an unconfigured deployment raises KeyError
    here - and a startup handler that raises takes the whole service down,
    which would leave the orchestrator with an agent that never answers rather
    than one that answers without a cache.
    """
    try:
        from kb.qdrant_store import ensure_collections

        await ensure_collections()
    except KeyError as exc:
        _logger.info("Qdrant is not configured (%s); running without the cache", exc)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("Qdrant setup failed (%s); running without the cache", exc)
