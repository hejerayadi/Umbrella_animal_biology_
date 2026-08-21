import pytest
from schemas.inputs import PathwaysInput
from schemas.common import AgentStatus
from schemas.outputs import PathwayEntry
from subagents.pathways import mock_pathways_agent, list_pathway_candidates, fetch_pathway_name
import subagents.pathways as pw_module
from subagents.pathways import pathways_agent


# --------------------------------------------------------------------------- #
#  Helpers for the real pathways_agent (mirrors tests/test_gene_mapper.py)
# --------------------------------------------------------------------------- #
async def _fake_list_single(kegg_gene_id):
    return [{"pathway_id": "hsa04010", "pathway_name": "MAPK signaling pathway"}]

async def _fake_list_multi(kegg_gene_id):
    return [
        {"pathway_id": "hsa00071", "pathway_name": "Fatty acid degradation"},
        {"pathway_id": "hsa04928", "pathway_name": "Thermogenesis"},
    ]

async def _fake_fetch_name(pathway_id):
    mapping = {
        "hsa04010": "MAPK signaling pathway",
        "hsa00071": "Fatty acid degradation",
        "hsa04928": "Thermogenesis",
    }
    return mapping.get(pathway_id)

async def _fake_fetch_pathway(kegg_gene_id):
    return PathwayEntry(pathway_id="hsa00071", pathway_name="Fatty acid degradation")


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


# --------------------------------------------------------------------------- #
#  Real agent tests (mirrors tests/test_gene_mapper.py's scenario suite)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_single_pathway_skips_llm(monkeypatch):
    monkeypatch.setattr(pw_module, "list_pathway_candidates", _fake_list_single)
    monkeypatch.setattr(pw_module, "fetch_pathway_name", _fake_fetch_name)

    result = await pathways_agent(PathwaysInput(
        trait_name="hair growth",
        gene_list=["FGF5"],
        instruction="test",
        context={"kegg_gene_ids": {"FGF5": "hsa:FGF5"}},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.pathways[0].pathway_id == "hsa04010"
    assert result.malformed_ids == []


@pytest.mark.asyncio
async def test_multi_pathway_uses_llm_pick(monkeypatch):
    """Gene with more than one KEGG pathway link: the LLM's selection must be
    used over the naive first-result pick."""
    monkeypatch.setattr(pw_module, "list_pathway_candidates", _fake_list_multi)
    monkeypatch.setattr(pw_module, "fetch_pathway_name", _fake_fetch_name)

    async def fake_llm_pick(trait, gene, candidates, *args, **kwargs):
        assert len(candidates) == 2
        # Deliberately NOT the first candidate, to prove the naive
        # first-result pick isn't what's being used.
        return "hsa04928", "Thermogenesis", "directly relevant to cold adaptation"

    monkeypatch.setattr(pw_module, "_llm_pick_pathway", fake_llm_pick)

    result = await pathways_agent(PathwaysInput(
        trait_name="cold adaptation",
        gene_list=["UCP1"],
        instruction="test",
        context={"kegg_gene_ids": {"UCP1": "hsa:UCP1"}},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.pathways[0].pathway_id == "hsa04928"
    assert result.pathways[0].pathway_name == "Thermogenesis"
    # Naive first-result pick would have been "hsa00071" - assert we didn't use it.
    assert result.pathways[0].pathway_id != "hsa00071"


@pytest.mark.asyncio
async def test_llm_unavailable_falls_back_to_first_pathway(monkeypatch):
    """LLM unavailable -> the deterministic fetch_pathway fallback still
    returns a usable (unranked) result."""
    monkeypatch.setattr(pw_module, "list_pathway_candidates", _fake_list_multi)
    monkeypatch.setattr(pw_module, "fetch_pathway", _fake_fetch_pathway)

    async def fake_llm_pick(*args, **kwargs):
        raise RuntimeError("NIM unreachable")

    monkeypatch.setattr(pw_module, "_llm_pick_pathway", fake_llm_pick)

    result = await pathways_agent(PathwaysInput(
        trait_name="cold adaptation",
        gene_list=["UCP1"],
        instruction="test",
        context={"kegg_gene_ids": {"UCP1": "hsa:UCP1"}},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.pathways[0].pathway_id == "hsa00071"


@pytest.mark.asyncio
async def test_no_kegg_id_never_calls_llm(monkeypatch):
    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("list_pathway_candidates should not be called")

    async def _fail_llm_if_called(*args, **kwargs):
        raise AssertionError("_llm_pick_pathway should not be called")

    monkeypatch.setattr(pw_module, "list_pathway_candidates", _fail_if_called)
    monkeypatch.setattr(pw_module, "_llm_pick_pathway", _fail_llm_if_called)

    result = await pathways_agent(PathwaysInput(
        trait_name="test",
        gene_list=["NO_ID"],
        instruction="test",
        context={"kegg_gene_ids": {}},
    ))

    assert result.status == AgentStatus.FAILED
    assert result.malformed_ids == ["NO_ID"]
    assert result.pathways == []


@pytest.mark.asyncio
async def test_one_pathway_missing_one_present_only_missing_is_malformed(monkeypatch):
    """One gene has zero KEGG links (malformed), the other has a real link.
    Only the missing one should end up in malformed_ids, and the overall
    status should still be COMPLETED because at least one pathway resolved."""
    async def list_candidates(kegg_gene_id):
        if kegg_gene_id == "hsa:MISSING":
            return []
        return [{"pathway_id": "hsa04010", "pathway_name": "MAPK signaling pathway"}]

    monkeypatch.setattr(pw_module, "list_pathway_candidates", list_candidates)
    monkeypatch.setattr(pw_module, "fetch_pathway_name", _fake_fetch_name)

    result = await pathways_agent(PathwaysInput(
        trait_name="test",
        gene_list=["MISSING_GENE", "FGF5"],
        instruction="test",
        context={"kegg_gene_ids": {"MISSING_GENE": "hsa:MISSING", "FGF5": "hsa:FGF5"}},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.malformed_ids == ["MISSING_GENE"]
    assert len(result.pathways) == 1
    assert result.pathways[0].pathway_id == "hsa04010"


@pytest.mark.asyncio
async def test_grounding_rule_rejects_invented_pathway_id(monkeypatch):
    monkeypatch.setattr(pw_module, "list_pathway_candidates", _fake_list_multi)
    monkeypatch.setattr(pw_module, "fetch_pathway_name", _fake_fetch_name)
    monkeypatch.setattr(pw_module, "fetch_pathway", _fake_fetch_pathway)

    async def fake_llm_pick(*args, **kwargs):
        return "hsa99999", "invented pathway", "hallucinated"

    monkeypatch.setattr(pw_module, "_llm_pick_pathway", fake_llm_pick)

    result = await pathways_agent(PathwaysInput(
        trait_name="test",
        gene_list=["UCP1"],
        instruction="test",
        context={"kegg_gene_ids": {"UCP1": "hsa:UCP1"}},
    ))

    # Fallback should have kicked in because the invented ID was rejected.
    assert result.status == AgentStatus.COMPLETED
    assert result.pathways[0].pathway_id == "hsa00071"