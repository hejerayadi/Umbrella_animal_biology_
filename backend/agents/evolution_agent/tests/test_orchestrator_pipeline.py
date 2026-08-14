"""End-to-end tests for the Sprint 2 parallel pipeline.

Tests verify:
  1. The happy path produces a complete EvolutionAnalysisResult.
  2. Both workers run concurrently (parallel fan-out, no alignment handoff).
  3. MC failure short-circuits the pipeline.
  4. Phylo failure is surfaced correctly.
  5. Species resolution failures stop the pipeline before any worker.
  6. Common names are resolved to scientific names.
  7. NEEDS_AGENT escalations from either worker are propagated.
"""

from __future__ import annotations

import pytest

from backend.agents.evolution_agent.schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    EvolutionAnalysisResult,
    MolecularComparisonResult,
    PhylogeneticResult,
)


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_pipeline_returns_evolution_analysis_result(
    orchestrator,
) -> None:
    request = AgentRequest(
        instruction="Full evolutionary analysis of human, chimp and mouse.",
        context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
    )
    result = await orchestrator.run(request)

    assert result.status is AgentStatus.COMPLETED
    assert isinstance(result.output, EvolutionAnalysisResult)


@pytest.mark.asyncio
async def test_analysis_result_contains_molecular_and_phylo(orchestrator) -> None:
    request = AgentRequest(
        instruction="Compare human, chimp and mouse.",
        context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
    )
    result = await orchestrator.run(request)
    analysis: EvolutionAnalysisResult = result.output

    assert isinstance(analysis.molecular, MolecularComparisonResult)
    assert isinstance(analysis.phylogenetic, PhylogeneticResult)


@pytest.mark.asyncio
async def test_analysis_result_species_list_matches_input(orchestrator) -> None:
    species = ["homo sapiens", "pan troglodytes", "mus musculus"]
    request = AgentRequest(
        instruction="test", context={}, species_list=list(species)
    )
    result = await orchestrator.run(request)
    # The resolver normalises to title-cased scientific names, so compare
    # case-insensitively.
    resolved = {s.lower() for s in result.output.species_list}
    assert resolved == {s.lower() for s in species}


@pytest.mark.asyncio
async def test_convenience_fields_populated_on_result(orchestrator) -> None:
    request = AgentRequest(
        instruction="test", context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
    )
    result = await orchestrator.run(request)

    # Top-level AgentResult convenience fields
    assert result.newick_tree is not None
    assert result.tree_url    is not None
    assert result.alignment_url is not None
    assert result.similarity_scores is not None
    assert result.confidence is not None


@pytest.mark.asyncio
async def test_overall_confidence_is_mean_of_mc_and_phylo(orchestrator) -> None:
    request = AgentRequest(
        instruction="test", context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
    )
    result = await orchestrator.run(request)
    analysis: EvolutionAnalysisResult = result.output

    # Verify it's the arithmetic mean of mc mean-score and phylo confidence
    mc   = analysis.molecular
    phylo = analysis.phylogenetic
    mc_mean = sum(e.score for e in mc.similarity_scores) / len(mc.similarity_scores)
    expected = round((mc_mean + phylo.overall_confidence) / 2, 4)
    assert analysis.overall_confidence == expected


@pytest.mark.asyncio
async def test_source_agents_lists_both_workers(orchestrator) -> None:
    request = AgentRequest(
        instruction="test", context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
    )
    result = await orchestrator.run(request)
    sa = result.source_agents

    assert "Evolution Agent Orchestrator" in sa
    assert "Molecular Comparison Agent"   in sa
    assert "Phylogenetic Tree Agent"      in sa


@pytest.mark.asyncio
async def test_all_five_species_full_pipeline(orchestrator) -> None:
    request = AgentRequest(
        instruction="test", context={},
        species_list=[
            "homo sapiens", "pan troglodytes", "mus musculus",
            "gallus gallus", "danio rerio",
        ],
    )
    result = await orchestrator.run(request)
    assert result.status is AgentStatus.COMPLETED
    assert "danio rerio" in result.output.phylogenetic.newick_tree


# ---------------------------------------------------------------------------
# 2. Parallel fan-out: both workers run concurrently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_both_workers_are_called_in_parallel(make_orchestrator) -> None:
    """MC and phylo must both run on the same request (fan-out)."""
    calls: list[str] = []

    class _MCSpy:
        def run(self, request: AgentRequest) -> AgentResult:
            calls.append("mc")
            return AgentResult(
                status=AgentStatus.COMPLETED,
                output=MolecularComparisonResult(
                    species_list=request.species_list,
                    alignment="",
                    alignment_url="http://alignment",
                    similarity_scores=[],
                    species_groups=[],
                    similarity_network={},
                ),
            )

    class _PhyloSpy:
        def run(self, request: AgentRequest) -> AgentResult:
            calls.append("phylo")
            return AgentResult(
                status=AgentStatus.COMPLETED,
                output=PhylogeneticResult(
                    newick_tree="('x':1.0,'y':1.0,'z':1.0);",
                    tree_url="http://tree",
                    model="LG+G4",
                    bootstrap_support={"node_root": 80},
                    confidence_values={s: 0.8 for s in request.species_list},
                    overall_confidence=0.8,
                ),
            )

    orch = make_orchestrator(mc_worker=_MCSpy(), phylo_worker=_PhyloSpy())
    await orch.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
        )
    )

    assert sorted(calls) == ["mc", "phylo"], "Both workers must be dispatched"


@pytest.mark.asyncio
async def test_phylo_runs_without_mc_alignment(make_orchestrator) -> None:
    """Phylo no longer consumes MC's alignment — it reconstructs its own."""
    received_alignment: list[str] = []

    class _PhyloSpy:
        from backend.agents.evolution_agent.workers.phylogenetic_tree.mock import (
            PhylogeneticTreeMock as _Real,
        )
        _real = _Real()

        def run(self, request: AgentRequest) -> AgentResult:
            received_alignment.append(
                (request.context or {}).get("alignment", "")
            )
            return self._real.run(request)

    orch = make_orchestrator(phylo_worker=_PhyloSpy())
    result = await orch.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
        )
    )

    assert result.status is AgentStatus.COMPLETED
    assert received_alignment, "Phylo worker was never called"
    assert not received_alignment[0], "Phylo must not receive MC's alignment in parallel mode"


# ---------------------------------------------------------------------------
# 3. MC failure short-circuits pipeline
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mc_failure_fails_pipeline(make_orchestrator) -> None:
    """MC failure must fail the whole pipeline even though phylo was dispatched."""
    orch = make_orchestrator()
    result = await orch.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens"],  # < 2 → MC fails
        )
    )

    assert result.status is AgentStatus.FAILED


@pytest.mark.asyncio
async def test_mc_failure_message_surfaces_in_result(orchestrator) -> None:
    result = await orchestrator.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens"],
        )
    )
    assert result.status is AgentStatus.FAILED
    assert "molecular comparison" in result.output.lower()


@pytest.mark.asyncio
async def test_unknown_species_fails_before_mc_runs(make_orchestrator) -> None:
    mc_called: list[bool] = []

    class _MCSpy:
        def run(self, request: AgentRequest) -> AgentResult:
            mc_called.append(True)
            return AgentResult(status=AgentStatus.COMPLETED, output=None)

    orch = make_orchestrator(mc_worker=_MCSpy())
    result = await orch.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens", "draco magicus"],
        )
    )

    assert result.status is AgentStatus.FAILED
    assert not mc_called, "MC must not run when species resolution fails"
    assert "draco magicus" in result.output.lower()


# ---------------------------------------------------------------------------
# 4. Phylo failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phylo_failure_surfaces_correctly(make_orchestrator) -> None:
    class _PhyloFail:
        def run(self, request: AgentRequest) -> AgentResult:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="IQ-TREE exploded",
                source_agents=["Phylogenetic Tree Agent"],
            )

    orch    = make_orchestrator(phylo_worker=_PhyloFail())
    result  = await orch.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
        )
    )

    assert result.status is AgentStatus.FAILED
    assert "phylogenetic reconstruction" in result.output.lower()
    assert "iq-tree exploded" in result.output.lower()


@pytest.mark.asyncio
async def test_phylo_failure_includes_mc_source_agent(make_orchestrator) -> None:
    """MC ran and succeeded, so its source agent should be mentioned."""
    class _PhyloFail:
        def run(self, request: AgentRequest) -> AgentResult:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="tree failed",
                source_agents=["Phylogenetic Tree Agent"],
            )

    orch   = make_orchestrator(phylo_worker=_PhyloFail())
    result = await orch.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
        )
    )

    assert "Phylogenetic Tree Agent" in result.source_agents


# ---------------------------------------------------------------------------
# 5. Species resolution failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_species_fails_with_helpful_message(orchestrator) -> None:
    result = await orchestrator.run(
        AgentRequest(instruction="analyse evolution", context={})
    )
    assert result.status is AgentStatus.FAILED
    assert "species" in result.output.lower()


@pytest.mark.asyncio
async def test_partially_unknown_species_list_fails(orchestrator) -> None:
    result = await orchestrator.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens", "unicornus fabulus"],
        )
    )
    assert result.status is AgentStatus.FAILED
    assert "unicornus fabulus" in result.output.lower()


# ---------------------------------------------------------------------------
# 6. Common-name resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_common_names_resolve_to_scientific(orchestrator) -> None:
    result = await orchestrator.run(
        AgentRequest(
            instruction="compare human, chimp and mouse",
            context={},
            species_list=["human", "chimp", "mouse"],
        )
    )
    assert result.status is AgentStatus.COMPLETED
    resolved = result.output.species_list
    assert "Homo sapiens"    in resolved
    assert "Pan troglodytes" in resolved
    assert "Mus musculus"    in resolved


@pytest.mark.asyncio
async def test_mixed_common_and_scientific_names(orchestrator) -> None:
    result = await orchestrator.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["human", "Mus musculus", "zebrafish"],
        )
    )
    assert result.status is AgentStatus.COMPLETED


# ---------------------------------------------------------------------------
# 7. NEEDS_AGENT escalation propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mc_needs_agent_propagates_to_caller(make_orchestrator) -> None:
    class _MCEscalate:
        def run(self, request: AgentRequest) -> AgentResult:
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Genome Agent",
                prompt_to_target_agent="Retrieve sequences for Homo sapiens.",
                output={"partial": True},
                source_agents=["Molecular Comparison Agent"],
            )

    orch   = make_orchestrator(mc_worker=_MCEscalate())
    result = await orch.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens", "mus musculus"],
        )
    )

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Genome Agent"
    assert "Homo sapiens" in result.prompt_to_target_agent


@pytest.mark.asyncio
async def test_phylo_needs_agent_propagates_to_caller(make_orchestrator) -> None:
    class _PhyloEscalate:
        def run(self, request: AgentRequest) -> AgentResult:
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Literature Agent",
                prompt_to_target_agent="Find phylogeny papers for these species.",
                output={"partial": True},
                source_agents=["Phylogenetic Tree Agent"],
            )

    orch   = make_orchestrator(phylo_worker=_PhyloEscalate())
    result = await orch.run(
        AgentRequest(
            instruction="test", context={},
            species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
        )
    )

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Literature Agent"
