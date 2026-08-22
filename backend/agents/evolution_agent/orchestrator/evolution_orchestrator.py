"""Evolution Agent Orchestrator — Sprint 2 sequential pipeline.

Sprint 2 design
---------------
The orchestrator runs two subagents in a fixed order:

    1. molecular_comparison   — fetch sequences, align, embed, score, cluster
    2. phylogenetic_tree      — build tree from the alignment MC produced

This is a *dependent* pipeline, not a fan-out:
  • Step 2 only runs if Step 1 completed successfully.
  • The alignment string produced by Step 1 is injected into the request
    context before Step 2 is called (context["alignment"]).
  • A failure at either step short-circuits the whole pipeline.

LangGraph graph shape
---------------------
    START
      → plan_node
            "resolve"    → species_resolver_node
            "no_feature" → fail_node
      species_resolver_node
            "run"        → molecular_comparison_node
            "failed"     → fail_node
      molecular_comparison_node
            "next"       → phylogenetic_tree_node
            "failed"     → fail_node
      phylogenetic_tree_node
            "assemble"   → assemble_node
            "failed"     → fail_node
      assemble_node → END
      fail_node     → END

State transitions
-----------------
Every node returns a partial dict that LangGraph merges into EvolutionState.
Nodes never read from state fields they themselves wrote — they read what
the previous node produced.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
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
)
from ..workers.molecular_comparison.mock import MolecularComparisonMock
from ..workers.phylogenetic_tree.mock import PhylogeneticTreeMock
from .services.species_resolver import SpeciesResolverService



# Graph state



@dataclass
class EvolutionState:
    """Object passed through every LangGraph node.

    Fields are populated incrementally as the pipeline advances.
    ``result`` is the terminal field — set by either assemble_node or
    fail_node and read by the caller.
    """

    request: AgentRequest

    # Set by _plan_node: the normalised feature string, or "__invalid__"
    planned_feature: str = ""

    # Set by species_resolver_node
    resolved_species:   list[str] = field(default_factory=list)
    unresolved_species: list[str] = field(default_factory=list)

    # Set by molecular_comparison_node
    mc_result:    MolecularComparisonResult | None = None
    mc_error:     str | None = None            # non-None → pipeline aborts

    # Set by phylogenetic_tree_node
    phylo_result: PhylogeneticResult | None = None
    phylo_error:  str | None = None            # non-None → pipeline aborts

    # Terminal result — set by assemble_node or fail_node
    result: AgentResult | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class EvolutionOrchestrator:
    """Domain orchestrator: accepts an AgentRequest, returns an AgentResult.

    Wires the Molecular Comparison and Phylogenetic Tree subagents into a
    sequential LangGraph pipeline.  The pipeline is recompiled once at
    construction time — cheap for mocks, important for real workers that
    open network connections.
    """

    def __init__(
        self,
        mc_worker:    Any | None = None,
        phylo_worker: Any | None = None,
        resolver:     SpeciesResolverService | None = None,
    ) -> None:
        self._mc_worker    = mc_worker    or MolecularComparisonMock()
        self._phylo_worker = phylo_worker or PhylogeneticTreeMock()
        self._resolver     = resolver     or SpeciesResolverService.from_env()
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

        g.add_node("plan",                    self._plan_node)
        g.add_node("species_resolver",        self._species_resolver_node)
        g.add_node("molecular_comparison",    self._molecular_comparison_node)
        g.add_node("phylogenetic_tree",       self._phylogenetic_tree_node)
        g.add_node("assemble",                self._assemble_node)
        g.add_node("fail",                    self._fail_node)

        g.add_edge(START, "plan")

        g.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"resolve": "species_resolver", "no_feature": "fail"},
        )
        g.add_conditional_edges(
            "species_resolver",
            self._route_after_resolve,
            {"run": "molecular_comparison", "failed": "fail"},
        )
        g.add_conditional_edges(
            "molecular_comparison",
            self._route_after_mc,
            {"next": "phylogenetic_tree", "failed": "fail"},
        )
        g.add_conditional_edges(
            "phylogenetic_tree",
            self._route_after_phylo,
            {"assemble": "assemble", "failed": "fail"},
        )

        g.add_edge("assemble", END)
        g.add_edge("fail",     END)

        return g.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    # Features the plan node accepts.
    _VALID_FEATURES = (
        {f.value for f in EvolutionaryFeature} | {"full_analysis"}
    )

    def _plan_node(self, state: EvolutionState) -> dict:
        """Validate the requested feature and store it in planned_feature.

        An absent or empty feature is treated as 'full_analysis' — run
        the whole pipeline.  Returns a non-empty dict so LangGraph accepts
        the update.
        """
        req     = state.request
        feature = (req.context or {}).get("feature") or req.feature or ""
        if feature and feature not in self._VALID_FEATURES:
            return {"planned_feature": "__invalid__"}
        return {"planned_feature": feature or "full_analysis"}

    def _route_after_plan(self, state: EvolutionState) -> str:
        return "resolve" if state.planned_feature != "__invalid__" else "no_feature"

    def _species_resolver_node(self, state: EvolutionState) -> dict:
        """Normalise raw species names to canonical scientific names.

        Source priority: request.species_list > context["species_list"]
        > context["species"].

        Any unresolvable name causes a hard stop — we never pass a partial
        species list to workers.
        """
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

        canonical, unresolved = self._resolver.resolve_all(raw)
        req.species_list = canonical          # mutate in-place for workers
        return {
            "resolved_species":   canonical,
            "unresolved_species": unresolved,
        }

    def _route_after_resolve(self, state: EvolutionState) -> str:
        if state.unresolved_species or not state.resolved_species:
            return "failed"
        return "run"

    async def _molecular_comparison_node(self, state: EvolutionState) -> dict:
        """Run the Molecular Comparison subagent.

        Wraps the synchronous mock in asyncio.to_thread so the event loop
        stays free.  Real async workers drop the wrapper.
        """
        result: AgentResult = await asyncio.to_thread(
            self._mc_worker.run, state.request
        )

        if result.status is AgentStatus.FAILED:
            return {"mc_error": result.output}

        if result.status is AgentStatus.NEEDS_AGENT:
            # Propagate escalation immediately — pack it as the terminal result.
            return {
                "result": AgentResult(
                    status=AgentStatus.NEEDS_AGENT,
                    target_agent=result.target_agent,
                    prompt_to_target_agent=result.prompt_to_target_agent,
                    output=result.output,
                    source_agents=["Evolution Agent Orchestrator",
                                   *result.source_agents],
                )
            }

        mc: MolecularComparisonResult = result.output
        return {"mc_result": mc}

    def _route_after_mc(self, state: EvolutionState) -> str:
        # NEEDS_AGENT path sets result directly; we still reach this router.
        if state.result is not None:
            return "failed"          # re-route to fail to surface the result
        if state.mc_error:
            return "failed"
        return "next"

    async def _phylogenetic_tree_node(self, state: EvolutionState) -> dict:
        """Run the Phylogenetic Tree subagent.

        Injects the alignment from the MC step into the request context
        so the phylo worker receives it as its primary input.
        """
        req = state.request
        mc  = state.mc_result

        # Hand the alignment forward — this is the core of the sequential
        # handoff that Sprint 2 requires.
        if mc is not None:
            req.context = {**(req.context or {}), "alignment": mc.alignment}

        result: AgentResult = await asyncio.to_thread(
            self._phylo_worker.run, req
        )

        if result.status is AgentStatus.FAILED:
            return {"phylo_error": result.output}

        if result.status is AgentStatus.NEEDS_AGENT:
            return {
                "result": AgentResult(
                    status=AgentStatus.NEEDS_AGENT,
                    target_agent=result.target_agent,
                    prompt_to_target_agent=result.prompt_to_target_agent,
                    output=result.output,
                    source_agents=["Evolution Agent Orchestrator",
                                   *result.source_agents],
                )
            }

        phylo: PhylogeneticResult = result.output
        return {"phylo_result": phylo}

    def _route_after_phylo(self, state: EvolutionState) -> str:
        if state.result is not None:
            return "failed"
        if state.phylo_error:
            return "failed"
        return "assemble"

    def _assemble_node(self, state: EvolutionState) -> dict:
        """Combine MC and phylo outputs into an EvolutionAnalysisResult."""
        mc    = state.mc_result
        phylo = state.phylo_result

        overall_confidence = round(
            (self._mc_mean(mc) + phylo.overall_confidence) / 2, 4
        )

        analysis = EvolutionAnalysisResult(
            species_list=state.resolved_species,
            molecular=mc,
            phylogenetic=phylo,
            overall_confidence=overall_confidence,
            source_agents=[
                "Evolution Agent Orchestrator",
                "Molecular Comparison Agent",
                "Phylogenetic Tree Agent",
            ],
        )

        return {
            "result": AgentResult(
                status=AgentStatus.COMPLETED,
                output=analysis,
                newick_tree=phylo.newick_tree,
                tree_url=phylo.tree_url,
                similarity_scores=[
                    {
                        "species_a": e.species_a,
                        "species_b": e.species_b,
                        "score":     e.score,
                    }
                    for e in mc.similarity_scores
                ],
                alignment_url=mc.alignment_url,
                confidence=overall_confidence,
                source_agents=analysis.source_agents,
            )
        }

    def _fail_node(self, state: EvolutionState) -> dict:
        """Produce a FAILED AgentResult describing what went wrong.

        If a NEEDS_AGENT result was set by a worker node it passes through
        unchanged — the fail_node is also the escalation exit.
        """
        # NEEDS_AGENT was set by a worker node → pass it straight through.
        if state.result is not None:
            return {"result": state.result}

        # Species resolution failed.
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

        # Molecular comparison step failed.
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

        # Phylogenetic reconstruction step failed.
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

        # No valid feature supplied.
        return {
            "result": AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    "Evolution Orchestrator: no valid feature to run. "
                    "Expected one of: "
                    + ", ".join(f.value for f in EvolutionaryFeature)
                    + ", or 'full_analysis'."
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
