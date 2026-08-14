import pytest
from schemas.inputs import PathwaysInput
from schemas.common import AgentStatus
from subagents.pathways import mock_pathways_agent, list_pathway_candidates, fetch_pathway_name


@pytest.mark.asyncio
async def test_mock_pathways_agent_trait_aware():
    inp = PathwaysInput(
        gene_list=["UCP1", "PRDM16", "UNKNOWN"],
        trait_name="cold adaptation",
        instruction="test",
        context={"kegg_gene_ids": {"UCP1": "hsa:7350", "PRDM16": "hsa:63976", "UNKNOWN": "hsa:99999"}},
    )
    out = await mock_pathways_agent(inp)
    assert out.status == AgentStatus.COMPLETED
    assert len(out.pathways) == 2
    assert "UNKNOWN" in out.malformed_ids
    prdm16 = next(p for p in out.pathways if p.pathway_id == "ko04928")
    assert "Thermogenesis" in prdm16.pathway_name


@pytest.mark.asyncio
async def test_list_pathway_candidates_real_kegg():
    candidates = await list_pathway_candidates("hsa:7350")
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert all("pathway_id" in c for c in candidates)


@pytest.mark.asyncio
async def test_fetch_pathway_name_real_kegg():
    name = await fetch_pathway_name("ko04928")
    assert isinstance(name, str)
    assert len(name) > 0  # KEGG returned a valid name, whatever it is


@pytest.mark.asyncio
async def test_pathways_agent_no_kegg_id():
    inp = PathwaysInput(
        gene_list=["NO_ID"],
        trait_name="test trait",
        instruction="test",
        context={"kegg_gene_ids": {}},
    )
    out = await mock_pathways_agent(inp)
    assert out.status == AgentStatus.FAILED
    assert "NO_ID" in out.malformed_ids