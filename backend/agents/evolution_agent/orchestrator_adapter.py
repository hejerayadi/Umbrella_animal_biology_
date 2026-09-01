"""Adapter between the HTTP boundary and the Evolution Orchestrator."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from langsmith import traceable

from .planner import plan
from .orchestrator import EvolutionOrchestrator
from .schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    EvolutionAnalysisResult,
    PlannedFeature,
    PlannerDecision,
)

# Backward-compatible alias (tests monkeypatch this name on the module)
classify_intent = plan

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User-facing error messages
# ---------------------------------------------------------------------------

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
    decision: PlannerDecision,
) -> list[str]:
    ctx_species = (
        context.get("species_list")
        or context.get("species")
        or decision.species_list
        or []
    )
    if isinstance(ctx_species, str):
        ctx_species = [ctx_species]
    if not isinstance(ctx_species, list):
        ctx_species = []
    return [s for s in ctx_species if isinstance(s, str) and s]


def to_orchestrator_request(
    request: AgentRequest,
    decision: PlannerDecision,
) -> AgentRequest:
    context = dict(request.context or {})
    species = resolve_species(context, decision)
    return AgentRequest(
        instruction=request.instruction,
        context={
            **context,
            "planner_decision": decision,
            "feature": decision.feature.value,
        },
        feature=decision.feature.value,
        species_list=species,
        reference_species=(
            context.get("reference_species") or decision.reference_species
        ),
        session_id=request.session_id,
        target_gene_or_protein=request.target_gene_or_protein,
        protein_inputs=request.protein_inputs,
    )


# ---------------------------------------------------------------------------
# Branch-specific summaries
# ---------------------------------------------------------------------------

def _build_summary(analysis: EvolutionAnalysisResult, feature: str) -> str:
    species = analysis.species_list
    mc      = analysis.molecular
    phylo   = analysis.phylogenetic
    n       = len(species)

    if feature == PlannedFeature.PHYLOGENETIC_TREE.value and phylo:
        # overall_confidence is None when UFBoot did not run (e.g. too few
        # species) — never fabricate a percentage in that case.
        conf = (
            f"{analysis.overall_confidence * 100:.0f}%"
            if analysis.overall_confidence is not None
            else "not available (bootstrap not run)"
        )
        return (
            f"Phylogenetic tree built for {n} species "
            f"using {phylo.model} model with {conf} overall confidence."
        )

    if feature == PlannedFeature.MOLECULAR_COMPARISON.value and mc:
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
        # None when there's no separation signal to measure (e.g. every
        # species landed in one group) — never fabricate a percentage.
        conf = (
            f"{analysis.overall_confidence * 100:.0f}%"
            if analysis.overall_confidence is not None
            else "not available (no group separation to measure)"
        )

        return (
            f"Analysed {n} species. "
            f"{closest}. "
            f"The {n} species are {group_str}. "
            f"Overall confidence: {conf}."
        )

    # full_analysis: both
    parts: list[str] = []
    if mc:
        if mc.similarity_scores:
            top = max(mc.similarity_scores, key=lambda e: e.score)
            parts.append(
                f"{top.species_a.capitalize()} and {top.species_b} "
                f"are the most closely related (similarity {top.score:.2f})"
            )
        groups = len(mc.species_groups)
        parts.append(f"forming {groups} evolutionary group" + ("s" if groups != 1 else ""))
    if phylo:
        parts.append(f"Phylogenetic tree built using {phylo.model} model")
    conf = (
        f"{analysis.overall_confidence * 100:.0f}%"
        if analysis.overall_confidence is not None
        else "not available"
    )
    return f"Analysed {n} species. {'. '.join(parts)}. Overall confidence: {conf}."


# ---------------------------------------------------------------------------
# Result reshaping — flat output matching EvolutionOutput
# ---------------------------------------------------------------------------

def to_platform_result(result: AgentResult, feature: str = "full_analysis") -> AgentResult:
    """Flatten the pipeline result into a clean, readable response.

    Branch-specific: only includes fields relevant to the executed feature.
    """
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
                "status":        "completed",
                "decision":      "analysis_complete",
                "explanation":   str(result.output),
                # No structured analysis came back, so nothing here can be
                # vouched for as a real measurement.
                "score_is_mock": True,
            },
            confidence=result.confidence,
            source_agents=list(result.source_agents),
        )

    mc    = analysis.molecular
    phylo = analysis.phylogenetic

    # Reported from the workers that actually ran, not hard-coded: with the
    # real UniProt/ESM-2 and MAFFT/IQ-TREE workers wired in these numbers
    # are genuine measurements, and claiming otherwise misleads every
    # downstream consumer.
    mocked_flags = dict(analysis.providers_are_mocked)

    output: dict[str, Any] = {
        "status":             "completed",
        "decision":           "analysis_complete",
        "explanation":        _build_summary(analysis, feature),
        "score_is_mock":      any(mocked_flags.values()),
        "providers_are_mocked": mocked_flags,
        "species_list":       analysis.species_list,
        "overall_confidence": analysis.overall_confidence,
    }

    # Branch-specific: molecular comparison fields
    if mc:
        output["similarity_scores"] = [
            {"species_a": e.species_a, "species_b": e.species_b, "score": e.score}
            for e in mc.similarity_scores
        ]
        output["species_groups"] = [
            {"group_id": g.group_id, "species": g.species, "mean_score": g.mean_score}
            for g in mc.species_groups
        ]
        output["similarity_network"] = mc.similarity_network

    # Branch-specific: phylogenetic tree fields
    if phylo:
        output["newick_tree"]       = phylo.newick_tree
        output["model"]             = phylo.model
        output["bootstrap_support"] = phylo.bootstrap_support
        output["confidence_values"] = phylo.confidence_values
        output["tree_url"]          = phylo.tree_url

    output["source_agents"] = analysis.source_agents

    # Explainer (LLM #2) output — prose only, kept in its own key so it can
    # never be confused with, or overwrite, a value produced by a worker.
    output["interpretation"] = result.interpretation
    output["warnings"]       = list(result.warnings)
    output["llm_calls"]      = result.llm_calls

    return AgentResult(
        status=AgentStatus.COMPLETED,
        output=output,
        newick_tree=phylo.newick_tree if phylo else None,
        tree_url=phylo.tree_url if phylo else None,
        similarity_scores=[
            {"species_a": e.species_a, "species_b": e.species_b, "score": e.score}
            for e in mc.similarity_scores
        ] if mc else None,
        confidence=analysis.overall_confidence,
        source_agents=analysis.source_agents,
        interpretation=result.interpretation,
        warnings=list(result.warnings),
        llm_calls=result.llm_calls,
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
    """Serves the feature-dependent orchestrator behind the agent's HTTP endpoint."""

    def __init__(
        self, orchestrator: EvolutionOrchestrator | None = None
    ) -> None:
        self._orchestrator = orchestrator or EvolutionOrchestrator()

    @traceable(name="Evolution Agent request", run_type="chain")
    async def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}

        # LLM budget accounting: the Planner is call #1 when it runs, the
        # Explainer is call #2 (counted by the orchestrator). Total <= 2.
        planner_calls = 0

        # Check if a planner decision was already made (e.g. by tests)
        existing_decision: PlannerDecision | None = context.get("planner_decision")

        if existing_decision is not None:
            _logger.info("[Evolution] planner_decision supplied; skipping planning")
            decision = existing_decision
        elif context.get("feature") or context.get("features") or request.feature:
            # Legacy path: feature supplied directly (tests, direct callers)
            _logger.info("[Evolution] feature supplied; skipping classification")
            feature_str = context.get("feature") or context.get("features") or request.feature or "full_analysis"
            species_from_ctx = context.get("species_list") or context.get("species") or request.species_list or []
            if isinstance(species_from_ctx, str):
                species_from_ctx = [species_from_ctx]
            decision = PlannerDecision(
                feature=feature_str,
                species_list=species_from_ctx,
                reference_species=context.get("reference_species") or request.reference_species,
                source="caller",
            )
        else:
            try:
                decision = await classify_intent(request.instruction)
                planner_calls = 1
            except Exception as exc:
                _logger.warning("[Evolution] planner call failed: %s", exc)
                return _failed(
                    "The Evolution Agent could not process your request. "
                    "Could you rephrase your question about evolutionary analysis?"
                )

        # Clarification needed
        if not decision.is_usable:
            _logger.info("[Evolution] clarification required (%s)", decision.source)
            return AgentResult(
                status=AgentStatus.CONTINUE,
                output={
                    "decision": "clarification_required",
                    "clarification_question": decision.clarification_question or "Could you rephrase your question?",
                    "source": decision.source,
                    "llm_calls": planner_calls,
                },
                source_agents=["Evolution Agent Orchestrator"],
                llm_calls=planner_calls,
            )

        orchestrator_request = to_orchestrator_request(request, decision)

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

        try:
            result = await self._orchestrator.run(orchestrator_request)
        except Exception as exc:
            _logger.warning("[Evolution] orchestrator failed: %s", exc)
            return _failed(
                f"The Evolution Agent encountered an error: {exc}"
            )

        # Fold the Planner call into the count the orchestrator started
        # (it already counted the Explainer, if one ran).
        result = replace(result, llm_calls=result.llm_calls + planner_calls)

        feature_value = decision.feature.value if isinstance(decision.feature, PlannedFeature) else str(decision.feature)
        return to_platform_result(result, feature=feature_value)
