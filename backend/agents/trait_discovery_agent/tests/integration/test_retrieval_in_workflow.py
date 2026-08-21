
import pytest

import subagents.pathways as pathways_module
import subagents.protein_data as protein_data_module
from kb.retrieval import semantic_search, validate_document
from schemas.common import AgentStatus
from schemas.inputs import PathwaysInput, ProteinDataInput

from .fakes import CallLog, make_fake_kegg_client, make_fake_uniprot_client


@pytest.mark.asyncio
async def test_protein_data_agent_reembeds_nothing_on_repeat_run(monkeypatch, embed_calls):
    log = CallLog()
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", make_fake_uniprot_client(log))

    request = ProteinDataInput(
        gene_list=["FGF5"], trait_name="test", instruction="x", context={"tax_id": 9606}
    )

    first = await protein_data_module.protein_data_agent(request)
    second = await protein_data_module.protein_data_agent(request)

    assert first.status == second.status == AgentStatus.COMPLETED
    assert len(log) == 2, "fetch_uniprot has no request-level cache and runs on every call"
    assert len(embed_calls) == 1, "the second run's payload is unchanged, so it must not be re-embedded"


@pytest.mark.asyncio
async def test_documents_written_during_workflow_are_retrievable_and_valid(monkeypatch):
    monkeypatch.setattr(pathways_module, "fetch_pathway", make_fake_kegg_client(CallLog()))

    await pathways_module.pathways_agent(PathwaysInput(
        gene_list=["FGF5", "UCP1"], instruction="x",
        context={"kegg_gene_ids": {"FGF5": "hsa:FGF5", "UCP1": "hsa:UCP1"}},
    ))

    hits = await semantic_search("kegg_pathways", "MAPK signaling pathway", top_k=3)
    assert hits, "the document the workflow just wrote should be retrievable"
    assert any(h.payload.get("gene_symbol") == "FGF5" for h in hits)

    for hit in hits:
        result = validate_document("kegg_pathways", hit.payload)
        assert result.is_valid, f"payload written by the workflow failed schema validation: {result.missing_fields}"



@pytest.mark.asyncio
async def test_metadata_filter_isolates_gene_after_two_workflow_runs(monkeypatch):
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", make_fake_uniprot_client(CallLog()))

    await protein_data_module.protein_data_agent(
        ProteinDataInput(gene_list=["FGF5"], trait_name="test", instruction="x", context={"tax_id": 9606})
    )
    await protein_data_module.protein_data_agent(
        ProteinDataInput(gene_list=["UCP1"], trait_name="test", instruction="x", context={"tax_id": 9606})
    )

    hits = await semantic_search(
        "uniprot_proteins", "protein function", top_k=5, filters={"gene_symbol": "UCP1"},
    )

    assert hits, "expected at least one hit for UCP1"
    assert all(h.payload.get("gene_symbol") == "UCP1" for h in hits)


@pytest.mark.asyncio
async def test_unresolved_gene_writes_nothing_to_the_kb(monkeypatch):
    monkeypatch.setattr(pathways_module, "fetch_pathway", make_fake_kegg_client(CallLog()))

    await pathways_module.pathways_agent(PathwaysInput(
        gene_list=["NOT_A_REAL_GENE"], instruction="x",
        context={"kegg_gene_ids": {"NOT_A_REAL_GENE": "hsa:doesnotexist"}},
    ))

    hits = await semantic_search("kegg_pathways", "NOT_A_REAL_GENE", top_k=5)
    assert not any(h.payload.get("gene_symbol") == "NOT_A_REAL_GENE" for h in hits)