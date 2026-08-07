
import pytest

import subagents.pathways as pathways_module
import subagents.protein_data as protein_data_module
from schemas.inputs import PathwaysInput, ProteinDataInput
from schemas.common import AgentStatus
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.nodes.functional_evidence_nodes import pathways_node, protein_data_node
from workflows.state import FunctionalEvidenceState

from .fakes import CallLog, make_fake_kegg_client, make_fake_uniprot_client


@pytest.mark.asyncio
async def test_pathways_node_calls_real_agent_with_correct_request(monkeypatch):
    log = CallLog()
    monkeypatch.setattr(pathways_module, "fetch_pathway", make_fake_kegg_client(log))

    state = FunctionalEvidenceState(
        gene_list=["FGF5", "UCP1"],
        instruction="find pathways",
        context={"kegg_gene_ids": {"FGF5": "hsa:FGF5", "UCP1": "hsa:UCP1"}},
    )
    result = await pathways_node(state)

    assert log.calls == [("hsa:FGF5",), ("hsa:UCP1",)]
    assert {p.pathway_id for p in result["pathway_data"]} == {"hsa04010", "hsa00071"}
    assert result["pathways_status"] == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_protein_data_node_calls_real_agent_with_correct_request(monkeypatch):
    log = CallLog()
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", make_fake_uniprot_client(log))

    state = FunctionalEvidenceState(
        gene_list=["FGF5"],
        instruction="find protein data",
        context={"tax_id": 9606},
    )
    result = await protein_data_node(state)

    assert log.calls == [("FGF5", 9606)]
    assert result["protein_data"][0].source_accession == "P12034"
    assert result["protein_data_status"] == AgentStatus.COMPLETED



@pytest.mark.asyncio
async def test_one_agent_failing_does_not_affect_the_other(monkeypatch):
    async def empty_kegg(kegg_gene_id):
        return None  # Pathways resolves nothing -> FAILED
    monkeypatch.setattr(pathways_module, "fetch_pathway", empty_kegg)
    monkeypatch.setattr(
        protein_data_module, "fetch_uniprot", make_fake_uniprot_client(CallLog())
    )

    app = build_functional_evidence_graph()
    result = await app.ainvoke(FunctionalEvidenceState(
        gene_list=["FGF5"],
        instruction="find functional evidence",
        context={"kegg_gene_ids": {"FGF5": "hsa:FGF5"}, "tax_id": 9606},
    ))

    assert result["pathways_status"] == AgentStatus.FAILED
    assert result["pathway_data"] == []
    # Protein Data still ran and still succeeded, independent of Pathways.
    assert result["protein_data_status"] == AgentStatus.COMPLETED
    assert result["protein_data"][0].gene_symbol == "FGF5"
    # merge_node's rule: either child FAILED -> whole sub-orchestrator FAILED.
    assert result["status"] == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_pathways_agent_reembeds_nothing_on_repeat_call(monkeypatch, embed_calls):
    log = CallLog()
    monkeypatch.setattr(pathways_module, "fetch_pathway", make_fake_kegg_client(log))

    request = PathwaysInput(
        gene_list=["FGF5"], instruction="find pathways",
        context={"kegg_gene_ids": {"FGF5": "hsa:FGF5"}},
    )

    first = await pathways_module.pathways_agent(request)
    second = await pathways_module.pathways_agent(request)

    assert len(log) == 2, "fetch_pathway has no request-level cache and runs on every call"
    assert len(embed_calls) == 1, "but the second run's dedup_key/text is unchanged, so upsert_point must skip re-embedding"
    assert second.pathways[0].pathway_name == first.pathways[0].pathway_name



@pytest.mark.asyncio
async def test_protein_data_agent_reports_missing_genes(monkeypatch):
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", make_fake_uniprot_client(CallLog()))

    request = ProteinDataInput(gene_list=["NOT_A_REAL_GENE"], instruction="x", context={"tax_id": 9606})
    result = await protein_data_module.protein_data_agent(request)

    assert result.missing_genes == ["NOT_A_REAL_GENE"]
    assert result.status == AgentStatus.FAILED