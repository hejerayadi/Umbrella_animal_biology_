"""
§8: malformed_ids (Pathways) and missing_genes (Protein Data) must survive
the trip from worker output all the way up to the Trait Discovery
Orchestrator's state — not get dropped at the Functional Evidence
sub-orchestrator boundary the way they were before this patch.

These tests monkeypatch at the pathways_agent/protein_data_agent boundary
(the same names workflows.nodes.functional_evidence_nodes imports), so they
never touch the real KEGG/UniProt clients or Qdrant — same isolation style
as tests/test_functional_evidence_merge.py.
"""
import pytest

import workflows.nodes.functional_evidence_nodes as fe_nodes
import workflows.nodes.gene_mapper_node as gm_node
import workflows.nodes.literature_support_node as lit_node
import workflows.trait_discovery_graph as td_graph_module
from schemas.common import AgentStatus
from schemas.outputs import (
    GeneMapperOutput,
    GOAnnotation,
    LiteratureSupportOutput,
    PathwayEntry,
    PathwaysOutput,
    ProteinDataOutput,
)
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.state import FunctionalEvidenceState, TraitDiscoveryState


async def _fake_pathways_agent(input):
    return PathwaysOutput(
        status=AgentStatus.COMPLETED,
        pathways=[PathwayEntry(pathway_id="hsa04010", pathway_name="MAPK signaling")],
        malformed_ids=["BADGENE"],
    )


async def _fake_protein_data_agent(input):
    return ProteinDataOutput(status=AgentStatus.FAILED, proteins=[], missing_genes=["FGF5", "UCP1"])


async def _fake_gene_mapper_agent(input):
    return GeneMapperOutput(
        status=AgentStatus.COMPLETED,
        go_annotations=[GOAnnotation(gene_symbol="FGF5", go_id="GO:0001", go_name="growth")],
    )


async def _fake_literature_support_agent(input):
    return LiteratureSupportOutput(status=AgentStatus.COMPLETED, evidence=[])


@pytest.mark.asyncio
async def test_pathways_node_returns_malformed_ids(monkeypatch):
    monkeypatch.setattr(fe_nodes, "pathways_agent", _fake_pathways_agent)
    result = await fe_nodes.pathways_node(
        FunctionalEvidenceState(gene_list=["FGF5", "BADGENE"], instruction="x")
    )
    assert result["malformed_ids"] == ["BADGENE"]


@pytest.mark.asyncio
async def test_protein_data_node_returns_missing_genes(monkeypatch):
    monkeypatch.setattr(fe_nodes, "protein_data_agent", _fake_protein_data_agent)
    result = await fe_nodes.protein_data_node(
        FunctionalEvidenceState(gene_list=["FGF5", "UCP1"], instruction="x")
    )
    assert result["missing_genes"] == ["FGF5", "UCP1"]


@pytest.mark.asyncio
async def test_functional_evidence_graph_surfaces_both_detail_fields(monkeypatch):
    """Pathways partially fails (one malformed id) while Protein Data fails
    outright (all genes missing) — both details must reach the sub-graph's
    final state, keyed separately, with no cross-contamination."""
    monkeypatch.setattr(fe_nodes, "pathways_agent", _fake_pathways_agent)
    monkeypatch.setattr(fe_nodes, "protein_data_agent", _fake_protein_data_agent)

    app = build_functional_evidence_graph()
    result = await app.ainvoke(
        FunctionalEvidenceState(gene_list=["FGF5", "UCP1", "BADGENE"], instruction="x")
    )

    assert result["malformed_ids"] == ["BADGENE"]
    assert result["missing_genes"] == ["FGF5", "UCP1"]
    # §0.2/§8: Pathways alone still resolved something -> non-critical overall.
    assert result["status"] == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_parent_orchestrator_surfaces_partial_failure_detail(monkeypatch):
    """End-to-end at the Trait Discovery Orchestrator level: the sub-
    orchestrator's partial-failure detail must land on TraitDiscoveryState
    under the contracted field names, not just its aggregated pathway_data/
    protein_data lists. Gene Mapper, Literature Support, and the explanation
    writer are all faked so this test is self-contained (no network/Qdrant/
    Neo4j dependency) and isolates the one thing under test: propagation of
    malformed_ids/missing_genes through the Functional Evidence boundary."""
    monkeypatch.setattr(fe_nodes, "pathways_agent", _fake_pathways_agent)
    monkeypatch.setattr(fe_nodes, "protein_data_agent", _fake_protein_data_agent)
    monkeypatch.setattr(gm_node, "gene_mapper_agent", _fake_gene_mapper_agent)
    monkeypatch.setattr(lit_node, "literature_support_agent", _fake_literature_support_agent)

    async def fake_explain(**kwargs):
        return f"Explanation for {kwargs['trait_name']}."
    monkeypatch.setattr(td_graph_module, "write_explanation", fake_explain)

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "UCP1", "BADGENE"]},
    ))

    assert result["malformed_ids"] == ["BADGENE"]
    assert result["missing_genes"] == ["FGF5", "UCP1"]
    # Partial Functional Evidence failure is still non-critical at the top level.
    assert result["status"] == AgentStatus.COMPLETED
