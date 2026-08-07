
import pytest

import subagents.pathways as pathways_module
import subagents.protein_data as protein_data_module
import workflows.trait_discovery_graph as td_graph_module
from schemas.common import AgentStatus
from schemas.outputs import GOAnnotation, LiteratureRecord, PathwayEntry, ProteinEntry
from workflows.state import TraitDiscoveryState

from .fakes import CallLog, make_fake_kegg_client, make_fake_uniprot_client


@pytest.fixture
def wired_pipeline(monkeypatch):
    monkeypatch.setattr(pathways_module, "fetch_pathway", make_fake_kegg_client(CallLog()))
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", make_fake_uniprot_client(CallLog()))


# ---------------------------------------------------------------------------
# 1. Happy path: data enters via context["gene_list"] and comes out the other
#    end matching TraitDiscoveryOutput's shape field-for-field.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_pipeline_data_shape_matches_output_contract(wired_pipeline, patch_llm_boundary):
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

    # gene_list entered state correctly from context
    assert result["gene_list"] == ["FGF5"]

    # every stage produced the type schemas/outputs.py promises for it
    assert result["go_annotations"] and all(isinstance(g, GOAnnotation) for g in result["go_annotations"])
    assert result["pathway_data"] and all(isinstance(p, PathwayEntry) for p in result["pathway_data"])
    assert result["protein_data"] and all(isinstance(p, ProteinEntry) for p in result["protein_data"])
    assert result["evidence"] and all(isinstance(e, LiteratureRecord) for e in result["evidence"])
    assert isinstance(result["explanation"], str) and result["explanation"]

    assert result["status"] == AgentStatus.COMPLETED
    # aggregate() ran exactly once, and it ran last (after every stage populated data)
    assert len(patch_llm_boundary["explain"]) == 1
    explained_with = patch_llm_boundary["explain"][0]
    assert "FGF5" in explained_with["genes"]


# ---------------------------------------------------------------------------
# 2. Escalation path: no gene_list in context -> capability resolver picks the
#    Genome Agent, and the workflow halts *before* any downstream stage runs
#    (no go_annotations/pathway_data/protein_data/evidence should exist).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_gene_list_halts_pipeline_before_any_stage_runs(patch_llm_boundary):
    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={},
    ))

    assert result["gene_list"] == []
    assert result["go_annotations"] == []
    assert result["pathway_data"] == []
    assert result["protein_data"] == []
    assert result["evidence"] == []
    assert result["status"] == AgentStatus.NEEDS_AGENT
    assert result["target_agent"] == "Genome Agent"
    assert patch_llm_boundary["explain"] == [], "aggregate must not run when the pipeline escalates"


# ---------------------------------------------------------------------------
# 3. Failure path: a hard failure partway through (Protein Data resolves
#    nothing) must reach the terminal `failed` node with the failure reason
#    visible on functional_evidence_status, and must not fabricate a status
#    of COMPLETED or silently drop the failure.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_midpipeline_failure_reaches_failed_terminal_node(patch_llm_boundary, monkeypatch):
    monkeypatch.setattr(pathways_module, "fetch_pathway", make_fake_kegg_client(CallLog()))
    async def empty_uniprot(gene_symbol, tax_id):
        return None
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", empty_uniprot)

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

    # upstream data that DID resolve is preserved, not discarded, on failure
    assert result["go_annotations"], "Gene Mapper output should still be present"
    assert result["pathway_data"], "Pathways output should still be present"
    assert result["protein_data"] == [], "Protein Data found nothing"
    assert result["functional_evidence_status"] == AgentStatus.FAILED
    assert result["status"] == AgentStatus.FAILED
    assert patch_llm_boundary["explain"] == [], "aggregate must not run on a failed pipeline"


# ---------------------------------------------------------------------------
# 4. Literature-escalation path: functional evidence completes normally, but
#    literature support finds only thin evidence, so the pipeline must escalate
#    to the Literature Agent instead of completing.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_thin_literature_evidence_triggers_escalation_not_completion(
    wired_pipeline, patch_llm_boundary
):
    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="cold adaptation",   # thin (1-record) entry in the literature mock DB
        species_name="mouse",
        instruction="Which genes drive cold adaptation?",
        context={
            "gene_list": ["UCP1"],
            "kegg_gene_ids": {"UCP1": "hsa:UCP1"},
            "tax_id": 9606,
        },
    ))

    assert result["go_annotations"] and result["pathway_data"] and result["protein_data"]
    assert result["evidence"], "thin evidence must still be carried through, not dropped"
    assert result["status"] == AgentStatus.NEEDS_AGENT
    assert result["target_agent"] == "Literature Agent"
    assert patch_llm_boundary["explain"] == []