
import os

import pytest

import subagents.pathways as pathways_module
import subagents.protein_data as protein_data_module
from schemas.inputs import PathwaysInput, ProteinDataInput
from schemas.common import AgentStatus
from workflows.functional_evidence_graph import build_functional_evidence_graph
from workflows.nodes.functional_evidence_nodes import pathways_node, protein_data_node
from workflows.state import FunctionalEvidenceState

from .fakes import (
    CallLog,
    make_fake_kegg_client,
    make_fake_uniprot_client,
    make_fake_uniprot_list_client,
    make_fake_uniprot_multi_client,
)

requires_live_llm = pytest.mark.skipif(
    not os.getenv("NVIDIA_NIM_API_KEY") and not os.getenv("NIM_API_KEY"),
    reason="NVIDIA_NIM_API_KEY not set — needed for the real LLM disambiguation call",
)


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
    """§10, single-hit branch: kept as the 'no decision needed' case. The
    candidate list (the primary §8 path, replacing the old direct
    fetch_uniprot call) is faked with exactly one reviewed hit, and the
    LLM boundary is asserted untouched — the real agent must resolve
    straight through without ever calling _llm_pick_protein."""
    log = CallLog()
    monkeypatch.setattr(
        protein_data_module, "list_uniprot_candidates", make_fake_uniprot_list_client(log)
    )

    async def fail_if_llm_called(*args, **kwargs):
        raise AssertionError(
            "single reviewed hit must skip the LLM entirely (§8) — _llm_pick_protein "
            "should never be called"
        )
    monkeypatch.setattr(protein_data_module, "_llm_pick_protein", fail_if_llm_called)

    state = FunctionalEvidenceState(
        gene_list=["FGF5"],
        trait_name="hair follicle growth",
        instruction="find protein data",
        context={"tax_id": 9606},
    )
    result = await protein_data_node(state)

    assert log.calls == [("FGF5", 9606)]
    assert result["protein_data"][0].source_accession == "P12034"
    assert result["protein_data_status"] == AgentStatus.COMPLETED


@requires_live_llm
@pytest.mark.asyncio
async def test_protein_data_multi_hit_real_llm_picks_trait_relevant_entry(monkeypatch):
    """§8/§10: two reviewed UniProt hits, one genuinely trait-relevant and
    one a plausible-looking distractor placed FIRST in the list. Nothing
    about the LLM call is mocked here — this exercises the real bind_tools
    loop against the real NVIDIA NIM endpoint end to end (prompt, tool
    binding, JSON parsing, grounding check). A pass proves the model
    reasoned over function_summary text rather than defaulting to array
    order. Only the UniProt candidate list is faked, since forcing a real
    gene to have exactly two reviewed hits on demand isn't reliable."""
    log = CallLog()
    monkeypatch.setattr(
        protein_data_module, "list_uniprot_candidates", make_fake_uniprot_multi_client(log)
    )

    state = FunctionalEvidenceState(
        gene_list=["UCP1"],
        trait_name="cold adaptation",
        instruction="find protein data",
        context={"tax_id": 9606},
    )
    result = await protein_data_node(state)

    assert log.calls == [("UCP1", 9606)]
    assert result["protein_data_status"] == AgentStatus.COMPLETED
    picked = result["protein_data"][0]
    # The trait-relevant entry (thermogenesis) must win over the distractor
    # pseudogene entry, which was placed FIRST in the candidate list — a
    # naive "take array order" implementation would get this wrong.
    assert picked.source_accession == "P25874"
    assert picked.protein_name == "Uncoupling protein 1"


@pytest.mark.asyncio
async def test_protein_data_llm_unavailable_falls_back_deterministically(monkeypatch):
    """§9/§10: kept as the LLM-unavailable fallback test. Two reviewed hits
    are available (so the LLM boundary would normally be used), but the LLM
    call itself fails — simulating a real NIM outage — and the agent must
    fall back to the deterministic first-hit fetch rather than dropping the
    gene entirely."""
    candidate_log = CallLog()
    fallback_log = CallLog()
    monkeypatch.setattr(
        protein_data_module, "list_uniprot_candidates",
        make_fake_uniprot_multi_client(candidate_log),
    )
    monkeypatch.setattr(
        protein_data_module, "fetch_uniprot", make_fake_uniprot_client(fallback_log)
    )

    async def simulate_nim_outage(*args, **kwargs):
        raise RuntimeError("simulated NIM outage: connection refused")
    monkeypatch.setattr(protein_data_module, "_llm_pick_protein", simulate_nim_outage)

    state = FunctionalEvidenceState(
        gene_list=["FGF5"],
        trait_name="cold adaptation",
        instruction="find protein data",
        context={"tax_id": 9606},
    )
    result = await protein_data_node(state)

    assert candidate_log.calls == [("FGF5", 9606)], "multi-hit branch must still be attempted first"
    assert fallback_log.calls == [("FGF5", 9606)], "deterministic fallback must run on LLM failure"
    assert result["protein_data_status"] == AgentStatus.COMPLETED
    assert result["protein_data"][0].source_accession == "P12034"


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
    # merge_node's rule (§0.2): a SINGLE child failing is non-critical — the
    # sub-orchestrator only fails when BOTH children fail.
    assert result["status"] == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_both_agents_succeeding_completes_the_suborchestrator(monkeypatch):
    """§10 inverse case: both children resolve real data -> merge_node's
    AND-of-failure rule (§0.2/§8) has nothing to fail on, so the
    sub-orchestrator status is COMPLETED. Pathways uses the real KEGG fake;
    Protein Data uses the single-hit candidate-list fake so it resolves
    without ever touching the LLM boundary (§8's 'no decision needed' case),
    keeping this purely a merge-logic test rather than an LLM test."""
    kegg_log = CallLog()
    uniprot_log = CallLog()
    monkeypatch.setattr(pathways_module, "fetch_pathway", make_fake_kegg_client(kegg_log))
    monkeypatch.setattr(
        protein_data_module, "list_uniprot_candidates",
        make_fake_uniprot_list_client(uniprot_log),
    )

    app = build_functional_evidence_graph()
    result = await app.ainvoke(FunctionalEvidenceState(
        gene_list=["FGF5"],
        instruction="find functional evidence",
        context={"kegg_gene_ids": {"FGF5": "hsa:FGF5"}, "tax_id": 9606},
    ))

    assert result["pathways_status"] == AgentStatus.COMPLETED
    assert result["protein_data_status"] == AgentStatus.COMPLETED
    assert result["status"] == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_both_agents_failing_fails_the_suborchestrator(monkeypatch):
    async def empty_kegg(kegg_gene_id):
        return None
    async def empty_uniprot(gene_symbol, tax_id):
        return None
    monkeypatch.setattr(pathways_module, "fetch_pathway", empty_kegg)
    monkeypatch.setattr(protein_data_module, "fetch_uniprot", empty_uniprot)

    app = build_functional_evidence_graph()
    result = await app.ainvoke(FunctionalEvidenceState(
        gene_list=["FGF5"],
        instruction="find functional evidence",
        context={"kegg_gene_ids": {"FGF5": "hsa:FGF5"}, "tax_id": 9606},
    ))

    assert result["pathways_status"] == AgentStatus.FAILED
    assert result["protein_data_status"] == AgentStatus.FAILED
    # merge_node's rule (§0.2): BOTH children FAILED -> whole sub-orchestrator FAILED.
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

    request = ProteinDataInput(
        gene_list=["NOT_A_REAL_GENE"], trait_name="test", instruction="x", context={"tax_id": 9606}
    )
    result = await protein_data_module.protein_data_agent(request)

    assert result.missing_genes == ["NOT_A_REAL_GENE"]
    assert result.status == AgentStatus.FAILED