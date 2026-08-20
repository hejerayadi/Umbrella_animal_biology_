"""Evolution Agent Orchestrator — feature-dependent pipeline.

The orchestrator runs the correct worker(s) based on what the Planner
decided:

    • molecular_comparison  → Molecular Comparison Agent only
    • phylogenetic_tree     → Phylogenetic Tree Agent only
    • full_analysis         → both in parallel
    • clarification_required → clarify_node (ask the user a question)

LangGraph graph shape
---------------------
    START
      → plan_node
            "resolve"    → species_resolver_node
            "clarify"    → clarify_node
            "no_feature" → fail_node
      species_resolver_node
            "run"        → dispatch_node
            "failed"     → fail_node
      dispatch_node
            "assemble"   → assemble_node
            "failed"     → fail_node
      assemble_node → explain_node → END
      clarify_node  → END
      fail_node     → END

Only the assemble path reaches explain_node (LLM #2), so clarification,
failure and needs_agent escalation never spend an Explainer call.

State transitions
-----------------
Every node returns a partial dict that LangGraph merges into EvolutionState.
Nodes never read from state fields they themselves wrote — they read what
the previous node produced.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    EvolutionAnalysisResult,
    EvolutionaryFeature,
    MolecularComparisonResult,
    PhylogeneticResult,
    PlannedFeature,
    PlannerDecision,
)
from ..explainer import explain as _default_explainer, fallback_explanation
from ..workers.molecular_comparison.mock import MolecularComparisonMock
from ..workers.phylogenetic_tree.worker import PhylogeneticTreeWorker
from .services.species_resolver import SpeciesResolverService

_logger = logging.getLogger(__name__)


# Graph state


@dataclass
class EvolutionState:
    """Object passed through every LangGraph node.

    Fields are populated incrementally as the pipeline advances.
    ``result`` is the terminal field — set by assemble_node, clarify_node,
    or fail_node and read by the caller.
    """

    request: AgentRequest

    # Set by plan_node: the PlannedFeature enum value as a string
    planned_feature: str = ""

    # Set by plan_node (when clarification is needed)
    clarification_question: str = ""

    # Set by species_resolver_node
    resolved_species:   list[str] = field(default_factory=list)
    unresolved_species: list[str] = field(default_factory=list)

    # Set by the dispatch step (parallel fan-out)
    mc_result:    MolecularComparisonResult | None = None
    mc_error:     str | None = None

    # Set by the dispatch step (parallel fan-out)
    phylo_result: PhylogeneticResult | None = None
    phylo_error:  str | None = None

    # Set by explain_node (LLM #2) — prose only, never structured data
    interpretation: str | None = None
    warnings:       list[str]  = field(default_factory=list)

    # Terminal result — set by assemble_node, clarify_node, or fail_node,
    # then enriched with the interpretation by explain_node
    result: AgentResult | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class EvolutionOrchestrator:
    """Domain orchestrator: accepts an AgentRequest, returns an AgentResult.

    Wires the Molecular Comparison and Phylogenetic Tree subagents into a
    LangGraph pipeline that runs the correct workers based on the Planner's
    decision.
    """

    def __init__(
        self,
        mc_worker:    Any | None = None,
        phylo_worker: Any | None = None,
        resolver:     SpeciesResolverService | None = None,
        explainer:    Any | None = None,
    ) -> None:
        self._mc_worker    = mc_worker    or MolecularComparisonMock()
        self._phylo_worker = phylo_worker or PhylogeneticTreeWorker()
        self._resolver     = resolver     or SpeciesResolverService.from_env()
        # LLM #2. Injectable so tests never touch a real backend.
        self._explainer    = explainer    or _default_explainer
        self._graph        = self._build_graph()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, request: AgentRequest) -> AgentResult:
        state = EvolutionState(request=request)
        final = await self._graph.ainvoke(state)
        if isinstance(final, dict):
            return final["result"]
        return final.result  # pragma: no cover

    # ------------------------------------------------------------------
    # LangGraph wiring
    # ------------------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(EvolutionState)

        g.add_node("plan",             self._plan_node)
        g.add_node("species_resolver", self._species_resolver_node)
        g.add_node("dispatch",         self._dispatch_node)
        g.add_node("assemble",         self._assemble_node)
        g.add_node("explain",          self._explain_node)
        g.add_node("clarify",          self._clarify_node)
        g.add_node("fail",             self._fail_node)

        g.add_edge(START, "plan")

        g.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {
                "resolve":   "species_resolver",
                "clarify":   "clarify",
                "no_feature": "fail",
            },
        )
        g.add_conditional_edges(
            "species_resolver",
            self._route_after_resolve,
            {"run": "dispatch", "failed": "fail"},
        )
        g.add_conditional_edges(
            "dispatch",
            self._route_after_dispatch,
            {"assemble": "assemble", "failed": "fail"},
        )

        # Only a successful assembly is worth explaining. Clarification,
        # failure and escalation reach END without an Explainer call.
        g.add_edge("assemble", "explain")
        g.add_edge("explain",  END)
        g.add_edge("clarify",  END)
        g.add_edge("fail",     END)

        return g.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _plan_node(self, state: EvolutionState) -> dict:
        """Read the planned feature from request.

        Priority:
        1. ``request.context["planner_decision"]`` (new path via adapter)
        2. ``request.context["feature"]`` or ``request.feature`` (legacy path)
        """
        req = state.request
        ctx = req.context or {}
        decision: PlannerDecision | None = ctx.get("planner_decision")

        if decision is not None:
            feature = decision.feature
            if feature == PlannedFeature.CLARIFICATION_REQUIRED:
                return {
                    "planned_feature": "clarification_required",
                    "clarification_question": decision.clarification_question or "Could you rephrase your question?",
                }
            if feature in {PlannedFeature.MOLECULAR_COMPARISON, PlannedFeature.PHYLOGENETIC_TREE, PlannedFeature.FULL_ANALYSIS}:
                return {"planned_feature": feature.value}
            return {"planned_feature": "__invalid__"}

        # Legacy path: read feature string directly from request
        feature = (ctx.get("feature") or req.feature or "").strip()
        if not feature:
            return {"planned_feature": "__invalid__"}
        valid = {f.value for f in PlannedFeature} | {"full_analysis"}
        if feature not in valid:
            return {"planned_feature": "__invalid__"}
        return {"planned_feature": feature}

    def _route_after_plan(self, state: EvolutionState) -> str:
        if state.planned_feature == "clarification_required":
            return "clarify"
        if state.planned_feature in {PlannedFeature.MOLECULAR_COMPARISON.value, PlannedFeature.PHYLOGENETIC_TREE.value, PlannedFeature.FULL_ANALYSIS.value}:
            return "resolve"
        return "no_feature"

    def _species_resolver_node(self, state: EvolutionState) -> dict:
        """Normalise raw species names to canonical scientific names."""
        req = state.request
        ctx = req.context or {}

        raw: list[str] = []
        if req.species_list:
            raw = req.species_list
        else:
            ctx_sp = ctx.get("species_list") or ctx.get("species") or []
            raw    = [ctx_sp] if isinstance(ctx_sp, str) else list(ctx_sp)

        if not raw:
            return {
                "resolved_species":   [],
                "unresolved_species": ["<none provided>"],
            }

        try:
            canonical, unresolved = self._resolver.resolve_all(raw)
        except Exception as exc:
            _logger.warning("[Resolve] species resolution failed: %s", exc)
            return {
                "resolved_species":   [],
                "unresolved_species": list(raw),
            }

        req.species_list = canonical
        return {
            "resolved_species":   canonical,
            "unresolved_species": unresolved,
        }

    def _route_after_resolve(self, state: EvolutionState) -> str:
        if state.unresolved_species or not state.resolved_species:
            return "failed"
        return "run"

    async def _dispatch_node(self, state: EvolutionState) -> dict:
        """Dispatch the correct worker(s) based on planned_feature.

        Only dispatches the worker(s) needed for the planned feature.
        Each worker is offloaded to a thread via asyncio.to_thread.
        """
        req = state.request
        feature = state.planned_feature

        async def call(worker: Any) -> AgentResult:
            return await asyncio.to_thread(worker.run, req)

        results: dict[str, Any] = {}

        try:
            if feature in (PlannedFeature.MOLECULAR_COMPARISON.value, PlannedFeature.FULL_ANALYSIS.value):
                _logger.info("[Dispatch] running molecular_comparison")
                mc_result = await call(self._mc_worker)
                results["mc_result"] = mc_result.output
                if mc_result.status is AgentStatus.NEEDS_AGENT:
                    return {"result": mc_result}
                if mc_result.status is AgentStatus.FAILED:
                    results["mc_error"] = mc_result.output

            if feature in (PlannedFeature.PHYLOGENETIC_TREE.value, PlannedFeature.FULL_ANALYSIS.value):
                _logger.info("[Dispatch] running phylogenetic_tree")
                phylo_result = await call(self._phylo_worker)
                results["phylo_result"] = phylo_result.output
                if phylo_result.status is AgentStatus.NEEDS_AGENT:
                    return {"result": phylo_result}
                if phylo_result.status is AgentStatus.FAILED:
                    results["phylo_error"] = phylo_result.output

        except Exception as exc:
            _logger.warning("[Dispatch] worker call failed: %s", exc)
            return {
                "mc_error":    "Dispatch failed: " + str(exc) if feature in (PlannedFeature.MOLECULAR_COMPARISON.value, PlannedFeature.FULL_ANALYSIS.value) else None,
                "phylo_error": "Dispatch failed: " + str(exc) if feature in (PlannedFeature.PHYLOGENETIC_TREE.value, PlannedFeature.FULL_ANALYSIS.value) else None,
            }

        return results

    def _route_after_dispatch(self, state: EvolutionState) -> str:
        if state.result is not None:
            return "failed"
        if state.mc_error or state.phylo_error:
            return "failed"
        return "assemble"

    def _assemble_node(self, state: EvolutionState) -> dict:
        """Combine worker outputs into an EvolutionAnalysisResult.

        Branch-specific: no tree fields for molecular_comparison,
        no network fields for phylogenetic_tree.
        """
        mc    = state.mc_result
        phylo = state.phylo_result
        feature = state.planned_feature

        overall_confidence = 0.0
        if mc and phylo:
            overall_confidence = round(
                (self._mc_mean(mc) + phylo.overall_confidence) / 2, 4
            )
        elif mc:
            overall_confidence = round(self._mc_mean(mc), 4)
        elif phylo:
            overall_confidence = phylo.overall_confidence

        source_agents = ["Evolution Agent Orchestrator"]
        if mc:
            source_agents.append("Molecular Comparison Agent")
        if phylo:
            source_agents.append("Phylogenetic Tree Agent")

        analysis = EvolutionAnalysisResult(
            species_list=state.resolved_species,
            molecular=mc,
            phylogenetic=phylo,
            overall_confidence=overall_confidence,
            source_agents=source_agents,
        )

        # Branch-specific top-level fields
        newick_tree: str | None = None
        tree_url: str | None = None
        similarity_scores: list[dict] | None = None
        alignment_url: str | None = None

        if phylo:
            newick_tree = phylo.newick_tree
            tree_url    = phylo.tree_url
        if mc:
            similarity_scores = [
                {
                    "species_a": e.species_a,
                    "species_b": e.species_b,
                    "score":     e.score,
                }
                for e in mc.similarity_scores
            ]
            alignment_url = mc.alignment_url

        return {
            "result": AgentResult(
                status=AgentStatus.COMPLETED,
                output=analysis,
                newick_tree=newick_tree,
                tree_url=tree_url,
                similarity_scores=similarity_scores,
                alignment_url=alignment_url,
                confidence=overall_confidence,
                source_agents=source_agents,
            )
        }

    async def _explain_node(self, state: EvolutionState) -> dict:
        """LLM #2 — attach a grounded interpretation to a successful result.

        Reached only from ``assemble``: clarification, failure and
        escalation go straight to END, so no LLM call is spent on them.

        The Explainer receives a strict whitelist (instruction, feature,
        validated species, the structured output of the SELECTED worker(s),
        warnings, mocked/real flags) and gives back prose only.  The
        structured payload assembled upstream is never rebuilt from it.

        A missing, failing or ungrounded interpretation is not an error:
        the worker result is kept, ``status`` stays COMPLETED, and a
        ``interpretation_unavailable`` warning records what happened.
        """
        result = state.result
        if result is None or result.status is not AgentStatus.COMPLETED:
            return {}

        analysis = result.output
        if not isinstance(analysis, EvolutionAnalysisResult):
            return {}

        feature  = state.planned_feature
        species  = list(analysis.species_list)
        results  = self._explainer_payload(analysis)
        mocked   = self._providers_are_mocked(analysis)
        warnings = list(state.warnings)

        interpretation: str | None = None
        try:
            interpretation = await self._explainer(
                instruction=state.request.instruction,
                feature=feature,
                species=species,
                results=results,
                warnings=warnings,
                providers_are_mocked=mocked,
            )
        except Exception as exc:  # noqa: BLE001 — never break a good result
            _logger.warning(
                "[Explain] explainer raised (%s); keeping worker result",
                type(exc).__name__,
            )
            interpretation = None

        if not interpretation:
            warnings.append("interpretation_unavailable")
            interpretation = fallback_explanation(feature, species, results)

        return {
            "interpretation": interpretation,
            "warnings": warnings,
            "result": replace(
                result,
                interpretation=interpretation,
                warnings=warnings,
                llm_calls=result.llm_calls + 1,
            ),
        }

    @staticmethod
    def _explainer_payload(analysis: EvolutionAnalysisResult) -> dict:
        """Serialise ONLY the workers that actually ran."""
        payload: dict[str, Any] = {}

        mc = analysis.molecular
        if mc is not None:
            payload["similarity"] = {
                "similarity_scores": [
                    {"species_a": e.species_a,
                     "species_b": e.species_b,
                     "score": e.score}
                    for e in mc.similarity_scores
                ],
                "species_groups": [
                    {"group_id": g.group_id,
                     "species": g.species,
                     "mean_score": g.mean_score}
                    for g in mc.species_groups
                ],
                "network_nodes": len(mc.similarity_network),
            }

        phylo = analysis.phylogenetic
        if phylo is not None:
            payload["phylogeny"] = {
                "newick_tree": phylo.newick_tree,
                "model": phylo.model,
                "bootstrap_support": dict(phylo.bootstrap_support),
                "confidence_values": dict(phylo.confidence_values),
                "overall_confidence": phylo.overall_confidence,
            }

        return payload

    def _providers_are_mocked(self, analysis: EvolutionAnalysisResult) -> dict:
        """Report mocked/real per branch from the wired worker classes.

        Read-only inspection — it does not touch the providers themselves.
        """
        flags: dict[str, bool] = {}
        if analysis.molecular is not None:
            flags["similarity"] = "mock" in type(self._mc_worker).__name__.lower()
        if analysis.phylogenetic is not None:
            flags["phylogeny"] = "mock" in type(self._phylo_worker).__name__.lower()
        return flags

    def _clarify_node(self, state: EvolutionState) -> dict:
        """Return a clarification question to the user."""
        return {
            "result": AgentResult(
                status=AgentStatus.CONTINUE,
                output={
                    "decision": "clarification_required",
                    "clarification_question": state.clarification_question or "Could you rephrase your question?",
                },
                source_agents=["Evolution Agent Orchestrator"],
            )
        }

    def _fail_node(self, state: EvolutionState) -> dict:
        """Produce a FAILED AgentResult describing what went wrong."""
        if state.result is not None:
            return {"result": state.result}

        if state.unresolved_species:
            names = ", ".join(
                f"'{s}'" for s in state.unresolved_species
                if s != "<none provided>"
            )
            msg = (
                f"Could not resolve species to scientific names: {names}. "
                "Please use scientific names (e.g. 'Homo sapiens') or "
                "common names from the supported catalogue."
            ) if names else (
                "No species were provided. Please supply at least 2 species."
            )
            return {
                "result": AgentResult(
                    status=AgentStatus.FAILED,
                    output=msg,
                    source_agents=["Evolution Agent Orchestrator"],
                )
            }

        if state.mc_error:
            return {
                "result": AgentResult(
                    status=AgentStatus.FAILED,
                    output=f"Molecular comparison failed: {state.mc_error}",
                    source_agents=[
                        "Evolution Agent Orchestrator",
                        "Molecular Comparison Agent",
                    ],
                )
            }

        if state.phylo_error:
            return {
                "result": AgentResult(
                    status=AgentStatus.FAILED,
                    output=f"Phylogenetic reconstruction failed: {state.phylo_error}",
                    source_agents=[
                        "Evolution Agent Orchestrator",
                        "Phylogenetic Tree Agent",
                    ],
                )
            }

        return {
            "result": AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    "Evolution Orchestrator: no valid feature to run. "
                    "Expected one of: molecular_comparison, phylogenetic_tree, "
                    "or full_analysis."
                ),
                source_agents=["Evolution Agent Orchestrator"],
            )
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mc_mean(mc: MolecularComparisonResult) -> float:
        scores = mc.similarity_scores
        if not scores:
            return 0.0
        return sum(e.score for e in scores) / len(scores)
