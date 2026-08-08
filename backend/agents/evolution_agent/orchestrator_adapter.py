"""Adapter between the HTTP boundary and the Evolution Orchestrator.

Sprint 2 responsibilities
--------------------------
1. Classify intent (LLM → RecognizedIntent) — or skip if the caller
   already provided context["feature"] / context["features"].
2. Validate the species list (at least 2 resolvable names required).
3. Build the enriched AgentRequest the orchestrator expects.
4. Run the sequential pipeline (MC → Phylo).
5. Reshape the EvolutionAnalysisResult into the platform contract:
       {"evolution_report": {...}, "alignment_url": "...", "tree_url": "..."}

Nothing under orchestrator/, workers/, or services/ is imported *into*
this module — only *from*.  That keeps the 49 existing tests and any
Streamlit dashboard unaffected.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from .intent import RecognizedIntent, classify_intent
from .orchestrator import EvolutionOrchestrator
from .schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    EvolutionAnalysisResult,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User-facing error messages
# ---------------------------------------------------------------------------

_NO_FEATURE_MESSAGE = (
    "The Evolution Agent could not determine what this question is asking for. "
    "It can run a full evolutionary analysis (sequence similarity + phylogenetic "
    "tree), a molecular comparison only, or a phylogenetic tree only. "
    "Could you rephrase the question?"
)

_NO_SPECIES_MESSAGE = (
    "The Evolution Agent needs at least 2 species to work with, but none were "
    "found in the request. Which species would you like to compare?"
)

_TOO_FEW_SPECIES_MESSAGE = (
    "The Evolution Agent needs at least 2 species to compare; "
    "only {count} {was_were} provided: {species}."
)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def resolve_species(
    context: dict[str, Any],
    intent:  RecognizedIntent,
) -> list[str]:
    """Return the species list, preferring what the platform already extracted.

    Priority: context["species_list"] > context["species"] > intent.species_list
    """
    ctx_species = (
        context.get("species_list")
        or context.get("species")
        or intent.species_list
        or []
    )
    if isinstance(ctx_species, str):
        ctx_species = [ctx_species]
    return [s for s in ctx_species if s]


def to_orchestrator_request(
    request: AgentRequest,
    intent:  RecognizedIntent,
) -> AgentRequest:
    """Build the enriched AgentRequest the orchestrator dispatches on."""
    context = dict(request.context or {})
    species = resolve_species(context, intent)

    return AgentRequest(
        instruction=request.instruction,
        context={
            **context,
            "feature": intent.feature or "full_analysis",
        },
        feature=intent.feature or "full_analysis",
        species_list=species,
        reference_species=(
            context.get("reference_species") or intent.reference_species
        ),
        session_id=request.session_id,
    )


# ---------------------------------------------------------------------------
# Result reshaping
# ---------------------------------------------------------------------------

def _analysis_to_dict(analysis: EvolutionAnalysisResult) -> dict[str, Any]:
    """Convert the EvolutionAnalysisResult dataclass tree to a JSON-safe dict.

    Uses dataclasses.asdict() for the nested structure so the serialisation
    is automatic even as the schema grows.
    """
    try:
        return asdict(analysis)
    except Exception:  # pragma: no cover — safety net only
        return {"error": "result serialisation failed"}


def to_platform_result(result: AgentResult) -> AgentResult:
    """Wrap the orchestrator result in the platform's output contract.

    card.json declares the output key as ``evolution_report``.  Downstream
    agents (e.g. Image Generation) branch on its presence — without this
    wrapping the key would never appear and those agents would stall.

    On FAILED / NEEDS_AGENT: pass through unchanged.
    """
    if result.status is not AgentStatus.COMPLETED:
        return result

    analysis: EvolutionAnalysisResult | None = (
        result.output
        if isinstance(result.output, EvolutionAnalysisResult)
        else None
    )

    report: dict[str, Any] = {}
    if analysis is not None:
        report = _analysis_to_dict(analysis)
    else:
        report = {"findings": result.output}

    # Top-level convenience keys read by the Global Orchestrator
    output: dict[str, Any] = {"evolution_report": report}
    if result.alignment_url:
        output["alignment_url"] = result.alignment_url
    if result.tree_url:
        output["tree_url"] = result.tree_url

    return AgentResult(
        status=result.status,
        target_agent=result.target_agent,
        prompt_to_target_agent=result.prompt_to_target_agent,
        output=output,
        newick_tree=result.newick_tree,
        tree_url=result.tree_url,
        similarity_scores=result.similarity_scores,
        alignment_url=result.alignment_url,
        confidence=result.confidence,
        source_agents=list(result.source_agents),
    )


def _failed(message: str) -> AgentResult:
    return AgentResult(
        status=AgentStatus.FAILED,
        output=message,
        source_agents=["Evolution Agent Orchestrator"],
    )


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

class OrchestratorEvolutionAgent:
    """Serves the sequential orchestrator behind the agent's HTTP endpoint.

    Same ``run(request) -> AgentResult`` shape as EvolutionMock, except
    async — the LangGraph pipeline is async all the way down.
    """

    def __init__(
        self, orchestrator: EvolutionOrchestrator | None = None
    ) -> None:
        # Built once at startup: compiles the LangGraph graph and opens
        # the species resolver backend.
        self._orchestrator = orchestrator or EvolutionOrchestrator()

    async def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}

        # ── Fast path: caller already knows what it wants ─────────────────
        # context["feature"] or context["features"] set by the Global
        # Orchestrator skips LLM classification entirely.
        has_feature = bool(
            context.get("feature")
            or context.get("features")
            or request.feature
        )
        if has_feature:
            _logger.info(
                "[Evolution] feature supplied by caller; skipping classification"
            )
            result = await self._orchestrator.run(request)
            return to_platform_result(result)

        # ── Intent classification ──────────────────────────────────────────
        intent = await classify_intent(request.instruction)
        if not intent.is_usable:
            _logger.info(
                "[Evolution] no feature resolved (%s)", intent.source
            )
            return _failed(_NO_FEATURE_MESSAGE)

        orchestrator_request = to_orchestrator_request(request, intent)

        # ── Pre-flight species check ───────────────────────────────────────
        species = orchestrator_request.species_list
        if not species:
            _logger.info("[Evolution] no species extracted")
            return _failed(_NO_SPECIES_MESSAGE)

        if len(species) < 2:
            was_were = "was" if len(species) == 1 else "were"
            return _failed(
                _TOO_FEW_SPECIES_MESSAGE.format(
                    count=len(species),
                    was_were=was_were,
                    species=", ".join(f"'{s}'" for s in species),
                )
            )

        # ── Run the pipeline ───────────────────────────────────────────────
        result = await self._orchestrator.run(orchestrator_request)
        return to_platform_result(result)
