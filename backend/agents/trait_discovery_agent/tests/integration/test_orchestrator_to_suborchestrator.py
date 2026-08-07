
import pytest

import subagents.pathways as pathways_module
import subagents.protein_data as protein_data_module
import workflows.trait_discovery_graph as td_graph_module
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.state import FunctionalEvidenceState, TraitDiscoveryState
from schemas.common import AgentStatus

from .fakes import CallLog, make_fake_kegg_client, make_fake_uniprot_client


@pytest.fixture
def wired_functional_evidence(monkeypatch):

    kegg_log, uniprot_log = CallLog(), CallLog()
    monkeypatch.setattr(pathways_module, "fetch_pathway", make_fake_kegg_client(kegg_log))
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", make_fake_uniprot_client(uniprot_log))
    return kegg_log, uniprot_log


@pytest.mark.asyncio
async def test_functional_evidence_graph_runs_standalone(wired_functional_evidence):
    app = build_functional_evidence_graph()
    result = await app.ainvoke(FunctionalEvidenceState(
        gene_list=["FGF5", "UCP1"],
        instruction="collect functional evidence",
        context={
            "kegg_gene_ids": {"FGF5": "hsa:FGF5", "UCP1": "hsa:UCP1"},
            "tax_id": 9606,
        },
    ))

    assert {p.pathway_id for p in result["pathway_data"]} == {"hsa04010", "hsa00071"}
    assert {p.gene_symbol for p in result["protein_data"]} == {"FGF5", "UCP1"}
    assert result["status"] == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_parent_passes_gene_list_and_merges_child_output(
    wired_functional_evidence, patch_llm_boundary, monkeypatch
):
    kegg_log, uniprot_log = wired_functional_evidence

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={
            "gene_list": ["FGF5"],
            "kegg_gene_ids": {"FGF5": "hsa:FGF5"},
            "tax_id": 9606,
        },
    ))

    assert kegg_log.calls == [("hsa:FGF5",)]
    assert uniprot_log.calls == [("FGF5", 9606)]

    # Its output landed in the parent state under the contracted field names.
    assert [p.pathway_id for p in result["pathway_data"]] == ["hsa04010"]
    assert [p.gene_symbol for p in result["protein_data"]] == ["FGF5"]
    assert result["functional_evidence_status"] == AgentStatus.COMPLETED
    assert result["status"] == AgentStatus.COMPLETED



@pytest.mark.asyncio
async def test_suborchestrator_failure_propagates_and_short_circuits_parent(
    patch_llm_boundary, monkeypatch
):
    async def empty_uniprot(gene_symbol, tax_id):
        return None  # Protein Data resolves nothing -> FAILED
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", empty_uniprot)
    monkeypatch.setattr(
        pathways_module, "fetch_pathway",
        make_fake_kegg_client(CallLog()),
    )

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={
            "gene_list": ["FGF5"],
            "kegg_gene_ids": {"FGF5": "hsa:FGF5"},
            "tax_id": 9606,
        },
    ))

    assert result["functional_evidence_status"] == AgentStatus.FAILED
    assert result["status"] == AgentStatus.FAILED
    assert patch_llm_boundary["explain"] == []