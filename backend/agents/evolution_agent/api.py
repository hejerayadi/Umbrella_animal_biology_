"""HTTP boundary for the Evolution Agent.

Follows the exact same pattern as multimodal_recognition_agent:
  - POST /execute accepts AgentRequest (instruction + context)
  - Returns AgentResult with status / target_agent / output
  - output is a rich dict with all domain fields
  - Never raises — exceptions become FAILED AgentResult

The output dict shape mirrors what the recognition agent returns:
  {
    "evolution": {
      "decision": "analysis_complete",
      "text_alignment": "neutral",
      "score_is_mock": false,
      "explanation": "Analysed 3 species...",
      "clarification_question": null
    },
    "species_list":       [...],
    "closest_species":    [...],
    "species_groups":     [[...]],
    "similarity_network": {...},
    "similarity_scores":  [...],
    "evolutionary_tree":  "(...);",
    "model":              "LG+G4",
    "bootstrap_support":  {...},
    "confidence_values":  {...},
    "overall_confidence": 0.93,
    "alignment_url":      "https://...",
    "tree_url":           "https://...",
    "source_agents":      [...]
  }

The LangGraph pipeline in orchestrator_adapter.py always answers; the
Sprint 2 mock implementation and its EVOLUTION_AGENT_IMPL switch are gone.

Run (from repo root):
  python -m uvicorn backend.agents.evolution_agent.api:app --port 8002 --reload
"""

from __future__ import annotations

import inspect
import logging

from fastapi import FastAPI

from .framework.llm_client import load_env
from .orchestrator_adapter import OrchestratorEvolutionAgent
from .schema import AgentRequest, AgentResult, AgentStatus

_logger = logging.getLogger(__name__)

# Load .env unconditionally at startup. The Planner (LLM #1) also loads it,
# but only lazily inside its own function — a request that skips the
# Planner (an explicit "feature" in the payload) would otherwise never
# trigger it, leaving MAFFT_BINARY / IQTREE_BINARY / LLM credentials unset
# even when backend/agents/evolution_agent/.env defines them.
load_env()

app = FastAPI(
    title="Evolution Agent",
    description=(
        "Analyses evolutionary relationships between species.\n\n"
        "Pipeline: **Molecular Comparison** (UniProt + ESM-2 embeddings) "
        "and **Phylogenetic Reconstruction** (MAFFT + IQ-TREE/UFBoot), "
        "run concurrently when both are requested." + chr(10) + chr(10) +
        "No alignment feeds the embeddings: MAFFT gap characters would "
        "corrupt them, so only the tree branch aligns."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def _build_agent():
    print("[Evolution] serving the LangGraph ORCHESTRATOR", flush=True)
    return OrchestratorEvolutionAgent()


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
        "**Letting the Planner (LLM #1) infer the feature and species:**" + chr(10) +
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
