import pytest

from schemas.inputs import TraitDiscoveryInput
from schemas.common import AgentStatus
from subagents.gene_mapper import mock_gene_mapper
from subagents.pathways import mock_pathways_agent
from subagents.protein_data import mock_protein_data_agent
from subagents.literature_support import mock_literature_support
from sub_orchestrator.functional_evidence_orchestrator import FunctionalEvidenceOrchestrator
from orchestrator.trait_discovery_orchestrator import TraitDiscoveryOrchestrator


def _build_orchestrator():
    functional_evidence = FunctionalEvidenceOrchestrator(
        pathways_agent=mock_pathways_agent,
        protein_data_agent=mock_protein_data_agent,
    )
    return TraitDiscoveryOrchestrator(
        gene_mapper=mock_gene_mapper,
        functional_evidence_orchestrator=functional_evidence,
        literature_support=mock_literature_support,
    )


@pytest.mark.asyncio
async def test_full_completion_fur_growth():
    """fur growth has 2 literature records -> not thin -> should fully complete."""
    orchestrator = _build_orchestrator()
    result = await orchestrator.run(TraitDiscoveryInput(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "KRT71", "HR"]},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert any(a.gene_symbol == "FGF5" for a in result.go_annotations)
    assert result.explanation


@pytest.mark.asyncio
async def test_thin_evidence_escalates_to_literature_agent():
    """cold adaptation has only 1 literature record -> thin -> NEEDS_AGENT bubbles up."""
    orchestrator = _build_orchestrator()
    result = await orchestrator.run(TraitDiscoveryInput(
        trait_name="cold adaptation",
        species_name="human",
        instruction="Which genes are involved in cold adaptation?",
        context={"gene_list": ["UCP1"]},
    ))

    assert result.status == AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Literature Agent"
    assert result.prompt_to_target_agent is not None
    # partial data should still be attached, not discarded
    assert len(result.pathway_data) == 1
    assert result.pathway_data[0].pathway_name == "Fatty acid degradation"


@pytest.mark.asyncio
async def test_unknown_genes_fail_at_gene_mapper():
    orchestrator = _build_orchestrator()
    result = await orchestrator.run(TraitDiscoveryInput(
        trait_name="unknown trait",
        species_name="cat",
        instruction="Random question",
        context={"gene_list": ["ZZZ999"]},
    ))

    assert result.status == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_missing_gene_list_bubbles_needs_agent_to_genome_agent():
    orchestrator = _build_orchestrator()
    result = await orchestrator.run(TraitDiscoveryInput(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={},  # no gene_list resolved yet
    ))

    assert result.status == AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Genome Agent"
    assert result.prompt_to_target_agent is not None