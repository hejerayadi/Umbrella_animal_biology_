"""HTTP boundary for the Evolution Agent.

Follows the exact same pattern as multimodal_recognition_agent:
  - POST /execute accepts AgentRequest (instruction + context)
  - Returns AgentResult with status / target_agent / output
  - output is a rich dict with all domain fields
  - Never raises — exceptions become FAILED AgentResult

Everything this agent produces is published under ONE key, the way every
other Umbrella worker publishes its findings: the Global Orchestrator merges
`output` straight into the context shared by all nine agents, so a flat
payload would put names like `status` and `model` into that shared namespace.
`evolution_analysis` is also the key the Reconstruction agent waits for.

  {
    "evolution_analysis": {
      "status":             "completed",
      "decision":           "analysis_complete",
      "explanation":        "Analysed 3 species...",
      "score_is_mock":      true,
      "species_list":       [...],
      "overall_confidence": 0.93,
      "similarity_scores":  [...],
      "species_groups":     [...],
      "similarity_network": {...},
      "newick_tree":        "(...);",
      "model":              "LG+G4",
      "bootstrap_support":  {...},
      "confidence_values":  {...},
      "alignment_url":      "https://...",
      "tree_url":           "https://...",
      "source_agents":      [...]
    }
  }

Which implementation answers is chosen by EVOLUTION_AGENT_IMPL:
  mock         (default) the stub in mock.py
  orchestrator the real LangGraph pipeline via orchestrator_adapter.py

Run (from repo root):
  python -m uvicorn backend.agents.evolution_agent.api:app --port 8002 --reload
"""

from __future__ import annotations

import inspect
import logging
import os

from fastapi import FastAPI

from .mock import EvolutionMock
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evolution Agent",
    description=(
        "Analyses evolutionary relationships between species.\n\n"
        "Pipeline: **Molecular Comparison** (MAFFT + ESM-C) "
        "→ **Phylogenetic Reconstruction** (IQ-TREE + UFBoot).\n\n"
        "All tools are mocked in Sprint 2 (`score_is_mock: true`)."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def _build_agent():
    impl = os.getenv("EVOLUTION_AGENT_IMPL", "mock").strip().lower()

    if impl != "orchestrator":
        print(
            f"[Evolution] serving MOCK (EVOLUTION_AGENT_IMPL={impl!r}). "
            "Set EVOLUTION_AGENT_IMPL=orchestrator for the real orchestrator.",
            flush=True,
        )
        return EvolutionMock()

    try:
        from .orchestrator_adapter import OrchestratorEvolutionAgent
        agent = OrchestratorEvolutionAgent()
        print("[Evolution] serving the LangGraph ORCHESTRATOR", flush=True)
        return agent
    except Exception as exc:  # noqa: BLE001
        print(
            f"[Evolution] orchestrator failed to build "
            f"({type(exc).__name__}: {exc}); falling back to MOCK.",
            flush=True,
        )
        _logger.warning("orchestrator build failed", exc_info=True)
        return EvolutionMock()


_agent = _build_agent()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health() -> dict:
    """Returns which implementation is currently active."""
    return {
        "agent":           "Evolution",
        "implementation":  type(_agent).__name__,
        "is_orchestrator": type(_agent).__name__ == "OrchestratorEvolutionAgent",
    }


@app.post(
    "/execute",
    response_model=AgentResult,
    summary="Run evolutionary analysis",
    description=(
        "Send a species list and a question. "
        "Returns similarity scores, species groups, and a phylogenetic tree.\n\n"
        "**Minimal request:**\n"
        "```json\n"
        '{"instruction": "Compare human, chimp and mouse.", '
        '"context": {"species_list": ["homo sapiens", "pan troglodytes", "mus musculus"]}}\n'
        "```\n\n"
        "**With GPT-5-mini intent classification** "
        "(set `EVOLUTION_AGENT_IMPL=orchestrator`):\n"
        "```json\n"
        '{"instruction": "How are human and chimp related evolutionarily?", "context": {}}\n'
        "```"
    ),
)
async def execute(request: AgentRequest) -> AgentResult:
    """The agent's single endpoint. Always answers with an AgentResult."""
    try:
        result = _agent.run(request)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"Evolution Agent error: {exc}",
        )
