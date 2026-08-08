"""Manual test script — run from the repo root:

    python -m backend.agents.evolution_agent.test_manual

Two modes:

  MODE 1 — Pipeline only (no LLM, no .env needed)
    Tests the full MC → Phylo pipeline with the offline mock orchestrator.
    Always works, no API key required.

  MODE 2 — Full stack with Azure GPT-5 intent classification
    Reads your .env, calls Azure to classify the instruction, then runs
    the pipeline.  Requires a valid .env with Azure credentials.
    Set EVOLUTION_AGENT_IMPL=orchestrator to enable.

Usage:
    python -m backend.agents.evolution_agent.test_manual          # mode 1
    $env:EVOLUTION_AGENT_IMPL="orchestrator"; python -m ...       # mode 2
"""

import asyncio
import os

from backend.agents.evolution_agent.orchestrator import EvolutionOrchestrator
from backend.agents.evolution_agent.orchestrator.services.species_resolver import (
    SpeciesResolverService,
    _OfflineBackend,
)
from backend.agents.evolution_agent.schema import (
    AgentRequest,
    EvolutionAnalysisResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("=" * 60)


def _as_analysis(result) -> EvolutionAnalysisResult:
    """Normalise result.output (dataclass or wrapped evolution_report dict)."""
    output = result.output
    if isinstance(output, EvolutionAnalysisResult):
        return output
    if isinstance(output, dict) and "evolution_report" in output:
        output = output["evolution_report"]
    if isinstance(output, dict) and "molecular" in output:
        from backend.agents.evolution_agent.schema import (
            MolecularComparisonResult,
            PhylogeneticResult,
            SimilarityEdge,
            SpeciesGroup,
        )
        m = output["molecular"]
        p = output["phylogenetic"]
        return EvolutionAnalysisResult(
            species_list=output["species_list"],
            molecular=MolecularComparisonResult(
                species_list=m["species_list"],
                alignment=m["alignment"],
                alignment_url=m["alignment_url"],
                similarity_scores=[SimilarityEdge(**e) for e in m["similarity_scores"]],
                species_groups=[SpeciesGroup(**g) for g in m["species_groups"]],
                similarity_network=m["similarity_network"],
            ),
            phylogenetic=PhylogeneticResult(
                newick_tree=p["newick_tree"],
                tree_url=p["tree_url"],
                model=p["model"],
                bootstrap_support=p["bootstrap_support"],
                confidence_values=p["confidence_values"],
                overall_confidence=p["overall_confidence"],
            ),
            overall_confidence=output["overall_confidence"],
            source_agents=output.get("source_agents", []),
        )
    raise TypeError(f"cannot interpret output of type {type(output)!r}")


def _print_result(result) -> None:
    print(f"Status     : {result.status.value}")
    print(f"Confidence : {result.confidence}")

    if result.status.value != "completed":
        print(f"Error      : {result.output}")
        return

    analysis = _as_analysis(result)

    print(f"\n--- Molecular Comparison ---")
    print(f"Species    : {analysis.molecular.species_list}")
    print(f"Pairs      : {len(analysis.molecular.similarity_scores)}")
    for e in analysis.molecular.similarity_scores:
        bar = "█" * int(e.score * 20)
        print(f"  {e.species_a:<20} vs {e.species_b:<20} {e.score:.2f}  {bar}")
    print(f"Groups     : {len(analysis.molecular.species_groups)}")
    for g in analysis.molecular.species_groups:
        print(f"  Group {g.group_id}: {g.species}  (mean: {g.mean_score})")

    print(f"\n--- Phylogenetic Tree ---")
    print(f"Model      : {analysis.phylogenetic.model}")
    print(f"Newick     : {analysis.phylogenetic.newick_tree}")
    print(f"Bootstrap  : {analysis.phylogenetic.bootstrap_support}")
    print(f"Per-leaf   : {analysis.phylogenetic.confidence_values}")
    print(f"Tree URL   : {analysis.phylogenetic.tree_url}")

    print(f"\n--- Summary ---")
    print(f"Overall confidence : {analysis.overall_confidence}")
    print(f"Source agents      : {analysis.source_agents}")


# ---------------------------------------------------------------------------
# MODE 1 — pipeline only, no LLM
# ---------------------------------------------------------------------------

async def run_pipeline(species: list[str], label: str) -> None:
    _sep(label)
    print(f"Input species: {species}")

    orch = EvolutionOrchestrator(
        resolver=SpeciesResolverService(_OfflineBackend())
    )
    result = await orch.run(
        AgentRequest(instruction="manual test", context={}, species_list=species)
    )
    _print_result(result)


async def pipeline_tests() -> None:
    await run_pipeline(
        ["human", "chimp", "mouse"],
        "Test 1 — Common names, 3 species (happy path)",
    )
    await run_pipeline(
        ["Homo sapiens", "Pan troglodytes", "Mus musculus",
         "Gallus gallus", "Danio rerio"],
        "Test 2 — All 5 species, scientific names",
    )
    await run_pipeline(
        ["Homo sapiens"],
        "Test 3 — Only 1 species (should fail: need at least 2)",
    )
    await run_pipeline(
        ["Homo sapiens", "Draco magicus"],
        "Test 4 — Unknown species (should fail)",
    )
    await run_pipeline(
        ["human", "Gallus gallus", "zebrafish"],
        "Test 5 — Mixed common + scientific names",
    )


# ---------------------------------------------------------------------------
# MODE 2 — full stack with Azure GPT-5 intent classification
# ---------------------------------------------------------------------------

async def run_llm(instruction: str, label: str) -> None:
    """Classify a free-form instruction with Azure GPT-5, then run the pipeline."""
    _sep(label)
    print(f"Instruction: {instruction!r}")

    from backend.agents.evolution_agent.intent import classify_intent

    print("Calling Azure GPT-5 for intent classification...")
    intent = await classify_intent(instruction)
    print(f"Intent     : feature={intent.feature!r}")
    print(f"            species={intent.species_list}")
    print(f"            reference={intent.reference_species!r}")
    print(f"            source={intent.source!r}")

    if not intent.is_usable:
        print("→ Not a recognisable evolutionary question — skipping pipeline.")
        return

    from backend.agents.evolution_agent.orchestrator_adapter import (
        OrchestratorEvolutionAgent,
    )

    agent  = OrchestratorEvolutionAgent()
    result = await agent.run(
        AgentRequest(instruction=instruction, context={})
    )
    _print_result(result)


async def llm_tests() -> None:
    await run_llm(
        "How similar are human and chimpanzee proteins?",
        "LLM Test 1 — Molecular comparison question",
    )
    await run_llm(
        "Show me the evolutionary tree for human, mouse, chicken and zebrafish.",
        "LLM Test 2 — Phylogenetic tree question",
    )
    await run_llm(
        "Compare homo sapiens, pan troglodytes and mus musculus evolutionarily.",
        "LLM Test 3 — Full analysis question",
    )
    await run_llm(
        "What is the capital of France?",
        "LLM Test 4 — Non-evolution question (should return no feature)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    use_llm = os.getenv("EVOLUTION_AGENT_IMPL", "mock").lower() == "orchestrator"

    print("\n" + "█" * 60)
    if use_llm:
        print("  MODE 2 — Full stack with Azure GPT-5")
        print("  (set EVOLUTION_AGENT_IMPL=mock to run pipeline only)")
    else:
        print("  MODE 1 — Pipeline only (no LLM)")
        print("  (set EVOLUTION_AGENT_IMPL=orchestrator for Azure GPT-5)")
    print("█" * 60)

    await pipeline_tests()

    if use_llm:
        await llm_tests()


if __name__ == "__main__":
    asyncio.run(main())
