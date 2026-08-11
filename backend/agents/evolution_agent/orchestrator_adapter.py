"""Adapter between the HTTP boundary and the Evolution Orchestrator."""

from __future__ import annotations

import logging
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

def _as_list(value: Any) -> list[str]:
    """Normalise a context entry that may be a single name or a list of them."""
    if isinstance(value, str):
        value = [value]
    return [s for s in (value or []) if s]


def resolve_species(
    context: dict[str, Any],
    intent:  RecognizedIntent,
) -> list[str]:
    """The species to compare, from the context or from the classifier.

    The context wins when it carries enough species, because a caller that
    named them explicitly means them literally. It cannot win on its own
    though: the Global Orchestrator's extractor seeds `context["species"]`
    with the ONE main subject of the question, and the Multimodal agent
    publishes the single species it recognised in a photo. Either would
    otherwise override the full list the classifier read out of "how are
    humans and chimps related?" and leave this agent refusing its own
    question with "needs at least 2 species".

    So a context that is short falls back to the classifier, and the two are
    merged when the classifier confirms the context's species and adds the
    rest.
    """

    from_context = _as_list(context.get("species_list") or context.get("species"))
    from_intent  = _as_list(intent.species_list)

    if len(from_context) >= 2 or not from_intent:
        return from_context

    # The classifier found more names than the context did. Keep any context
    # species it missed - that one is the subject the user actually asked
    # about - then add the rest in the order the classifier read them.
    lowered = {s.lower() for s in from_intent}
    return [s for s in from_context if s.lower() not in lowered] + from_intent


def to_orchestrator_request(
    request: AgentRequest,
    intent:  RecognizedIntent,
) -> AgentRequest:
    context = dict(request.context or {})
    species = resolve_species(context, intent)
    return AgentRequest(
        instruction=request.instruction,
        context={**context, "feature": intent.feature or "full_analysis"},
        feature=intent.feature or "full_analysis",
        species_list=species,
        reference_species=(
            context.get("reference_species") or intent.reference_species
        ),
        session_id=request.session_id,
        target_gene_or_protein=request.target_gene_or_protein,
        protein_inputs=request.protein_inputs,
    )


# ---------------------------------------------------------------------------
# Plain-language summary
# ---------------------------------------------------------------------------

def _build_summary(analysis: EvolutionAnalysisResult) -> str:
    species = analysis.species_list
    mc      = analysis.molecular
    phylo   = analysis.phylogenetic
    n       = len(species)

    if mc.similarity_scores:
        top     = max(mc.similarity_scores, key=lambda e: e.score)
        closest = (
            f"{top.species_a.capitalize()} and {top.species_b} "
            f"are the most closely related (similarity {top.score:.2f})"
        )
    else:
        closest = "Similarity scores unavailable"

    groups    = len(mc.species_groups)
    group_str = f"forming {groups} evolutionary group" + ("s" if groups != 1 else "")
    conf      = f"{analysis.overall_confidence * 100:.0f}%"

    return (
        f"Analysed {n} species. "
        f"{closest}. "
        f"The {n} species are {group_str}. "
        f"Phylogenetic tree built using {phylo.model} model "
        f"with {conf} overall confidence."
    )


# ---------------------------------------------------------------------------
# Result reshaping — EvolutionOutput published under one context key
# ---------------------------------------------------------------------------

# The single key this agent publishes its findings under.
#
# The Global Orchestrator merges `output` straight into the context every
# other agent reads (see its worker_node.py), so a flat payload puts generic
# names like `status`, `model` and `explanation` into a namespace shared with
# eight other agents, where the last writer wins and the Responder renders
# every one of them as a separate finding.
#
# The name is not free either: the Reconstruction agent waits specifically for
# `evolution_analysis` in the context before it will run, so this is the key
# that closes its dependency. Publishing anything else leaves it escalating to
# an agent that has already answered, until the orchestrator's loop-breaker
# turns the whole run into a failure.
EVOLUTION_OUTPUT_KEY = "evolution_analysis"


def to_platform_result(result: AgentResult) -> AgentResult:
    """Reshape the pipeline result into the platform's output contract."""
    if result.status is not AgentStatus.COMPLETED:
        return result

    analysis: EvolutionAnalysisResult | None = (
        result.output
        if isinstance(result.output, EvolutionAnalysisResult)
        else None
    )

    if analysis is None:
        return AgentResult(
            status=result.status,
            output={
                EVOLUTION_OUTPUT_KEY: {
                    "status":        "completed",
                    "decision":      "analysis_complete",
                    "explanation":   str(result.output),
                    "score_is_mock": True,
                }
            },
            confidence=result.confidence,
            source_agents=list(result.source_agents),
        )

    mc    = analysis.molecular
    phylo = analysis.phylogenetic

    findings: dict[str, Any] = {
        # Headline fields
        "status":             "completed",
        "decision":           "analysis_complete",
        "explanation":        _build_summary(analysis),
        "score_is_mock":      True,

        # Species
        "species_list":       analysis.species_list,
        "overall_confidence": analysis.overall_confidence,

        # Molecular comparison
        "similarity_scores": [
            {"species_a": e.species_a, "species_b": e.species_b, "score": e.score}
            for e in mc.similarity_scores
        ],
        "species_groups": [
            {"group_id": g.group_id, "species": g.species, "mean_score": g.mean_score}
            for g in mc.species_groups
        ],
        "similarity_network": mc.similarity_network,

        # Phylogenetic tree
        "newick_tree":        phylo.newick_tree,
        "model":              phylo.model,
        "bootstrap_support":  phylo.bootstrap_support,
        "confidence_values":  phylo.confidence_values,

        # URLs
        "alignment_url":      mc.alignment_url,
        "tree_url":           phylo.tree_url,

        # Provenance
        "source_agents":      analysis.source_agents,
    }

    return AgentResult(
        status=AgentStatus.COMPLETED,
        output={EVOLUTION_OUTPUT_KEY: findings},
        newick_tree=phylo.newick_tree,
        tree_url=phylo.tree_url,
        similarity_scores=[
            {"species_a": e.species_a, "species_b": e.species_b, "score": e.score}
            for e in mc.similarity_scores
        ],
        alignment_url=mc.alignment_url,
        confidence=analysis.overall_confidence,
        source_agents=analysis.source_agents,
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
    """Serves the sequential orchestrator behind the agent's HTTP endpoint."""

    def __init__(
        self, orchestrator: EvolutionOrchestrator | None = None
    ) -> None:
        self._orchestrator = orchestrator or EvolutionOrchestrator()

    async def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}

        has_feature = bool(
            context.get("feature")
            or context.get("features")
            or request.feature
        )
        if has_feature:
            _logger.info("[Evolution] feature supplied; skipping classification")
            result = await self._orchestrator.run(request)
            return to_platform_result(result)

        intent = await classify_intent(request.instruction)
        if not intent.is_usable:
            _logger.info("[Evolution] no feature resolved (%s)", intent.source)
            return _failed(_NO_FEATURE_MESSAGE)

        orchestrator_request = to_orchestrator_request(request, intent)

        species = orchestrator_request.species_list
        if not species:
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

        result = await self._orchestrator.run(orchestrator_request)
        return to_platform_result(result)
