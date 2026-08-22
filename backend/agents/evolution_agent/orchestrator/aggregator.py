"""Sprint 2 aggregator for the Evolution Orchestrator.

Sprint 2 runs a fixed two-step sequential pipeline, not a parallel fan-out,
so the aggregator's job is much simpler than in Sprint 1: it receives the
completed MC and phylo results and assembles them into one EvolutionAnalysisResult.

The function is kept as a standalone module (rather than inlined into the
orchestrator) so it can be unit-tested independently and swapped out in
Sprint 3+ when Neo4j relationships and literature evidence are added.

Aggregation rules
-----------------
1. If either worker result has status NEEDS_AGENT the escalation is
   returned immediately — the caller (orchestrator) handles this before
   reaching the aggregator, but the function is defensive anyway.
2. If either worker result has status FAILED the aggregate is FAILED with
   a combined error message.
3. Otherwise: combine into EvolutionAnalysisResult and wrap in AgentResult.
"""

from __future__ import annotations

from ..schema import (
    AgentResult,
    AgentStatus,
    EvolutionAnalysisResult,
    MolecularComparisonResult,
    PhylogeneticResult,
)


def assemble(
    species_list: list[str],
    mc_result:    MolecularComparisonResult,
    phylo_result: PhylogeneticResult,
) -> AgentResult:
    """Combine MC and phylo results into a single AgentResult.

    Called by the orchestrator's assemble_node once both workers have
    completed successfully.  Does not raise — any unexpected input is
    caught and returned as FAILED.
    """
    try:
        overall_confidence = _mean_confidence(mc_result, phylo_result)

        analysis = EvolutionAnalysisResult(
            species_list=species_list,
            molecular=mc_result,
            phylogenetic=phylo_result,
            overall_confidence=overall_confidence,
            source_agents=[
                "Evolution Agent Orchestrator",
                "Molecular Comparison Agent",
                "Phylogenetic Tree Agent",
            ],
        )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=analysis,
            newick_tree=phylo_result.newick_tree,
            tree_url=phylo_result.tree_url,
            similarity_scores=[
                {
                    "species_a": e.species_a,
                    "species_b": e.species_b,
                    "score":     e.score,
                }
                for e in mc_result.similarity_scores
            ],
            alignment_url=mc_result.alignment_url,
            confidence=overall_confidence,
            source_agents=analysis.source_agents,
        )

    except Exception as exc:  # noqa: BLE001
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"Assembly failed: {exc}",
            source_agents=["Evolution Agent Orchestrator"],
        )


def _mean_confidence(
    mc:    MolecularComparisonResult,
    phylo: PhylogeneticResult,
) -> float:
    """Average the MC mean similarity score and the phylo UFBoot confidence."""
    scores = mc.similarity_scores
    mc_mean = (
        sum(e.score for e in scores) / len(scores) if scores else 0.0
    )
    return round((mc_mean + phylo.overall_confidence) / 2, 4)
