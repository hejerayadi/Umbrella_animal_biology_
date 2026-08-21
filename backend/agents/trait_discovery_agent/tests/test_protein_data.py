import pytest
from schemas.inputs import ProteinDataInput
from schemas.common import AgentStatus
from schemas.outputs import ProteinEntry
from subagents.protein_data import mock_protein_data_agent, list_uniprot_candidates
import subagents.protein_data as pd_module
from subagents.protein_data import protein_data_agent


# --------------------------------------------------------------------------- #
#  Helpers for the real protein_data_agent (mirrors tests/test_pathways.py)
# --------------------------------------------------------------------------- #
async def _fake_list_single(gene_symbol, tax_id):
    return [{
        "source_accession": "P12034",
        "protein_name": "Fibroblast growth factor 5",
        "function_summary": "Regulates hair follicle growth cycle.",
    }]

async def _fake_list_multi(gene_symbol, tax_id):
    return [
        {
            "source_accession": "P25874",
            "protein_name": "Uncoupling protein 1",
            "function_summary": "Mitochondrial proton channel, generates heat.",
        },
        {
            "source_accession": "Q99999",
            "protein_name": "Uncoupling protein 1, isoform b",
            "function_summary": "Superseded/stale reviewed entry, unrelated function noted.",
        },
    ]

async def _fake_fetch_uniprot(gene_symbol, tax_id):
    return ProteinEntry(
        gene_symbol=gene_symbol,
        protein_name="Uncoupling protein 1",
        function_summary="Mitochondrial proton channel, generates heat.",
        source_accession="P25874",
    )


@pytest.mark.asyncio
async def test_mock_protein_data_agent_trait_aware():
    inp = ProteinDataInput(
        gene_list=["UCP1", "FGF5", "UNKNOWN"],
        trait_name="cold adaptation",
        instruction="test",
        context={"tax_id": 9606},
    )
    out = await mock_protein_data_agent(inp)
    assert out.status == AgentStatus.COMPLETED
    assert len(out.proteins) == 2
    assert "UNKNOWN" in out.missing_genes
    ucp1 = next(p for p in out.proteins if p.gene_symbol == "UCP1")
    assert "heat" in ucp1.function_summary


@pytest.mark.asyncio
async def test_list_uniprot_candidates_real_uniprot():
    candidates = await list_uniprot_candidates("UCP1", 9606)
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert all("source_accession" in c for c in candidates)


# --------------------------------------------------------------------------- #
#  Real agent tests (mirrors tests/test_pathways.py's scenario suite)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_single_hit_skips_llm(monkeypatch):
    monkeypatch.setattr(pd_module, "list_uniprot_candidates", _fake_list_single)

    result = await protein_data_agent(ProteinDataInput(
        trait_name="hair growth",
        gene_list=["FGF5"],
        instruction="test",
        context={"tax_id": 9606},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.proteins[0].source_accession == "P12034"
    assert result.missing_genes == []


@pytest.mark.asyncio
async def test_multi_hit_uses_llm_pick(monkeypatch):
    """Gene with more than one reviewed UniProt entry: the LLM's selection
    must be used over the naive first-result pick."""
    monkeypatch.setattr(pd_module, "list_uniprot_candidates", _fake_list_multi)

    async def fake_llm_pick(trait, gene, candidates, *args, **kwargs):
        assert len(candidates) == 2
        # Deliberately NOT the first candidate, to prove the naive
        # first-result pick isn't what's being used.
        return (
            "Q99999",
            "Uncoupling protein 1, isoform b",
            "Superseded/stale reviewed entry, unrelated function noted.",
            "picked the isoform record as more trait-relevant",
        )

    monkeypatch.setattr(pd_module, "_llm_pick_protein", fake_llm_pick)

    result = await protein_data_agent(ProteinDataInput(
        trait_name="cold adaptation",
        gene_list=["UCP1"],
        instruction="test",
        context={"tax_id": 9606},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.proteins[0].source_accession == "Q99999"
    # Naive first-result pick would have been "P25874" - assert we didn't use it.
    assert result.proteins[0].source_accession != "P25874"


@pytest.mark.asyncio
async def test_llm_unavailable_falls_back_to_first_hit(monkeypatch):
    """LLM unavailable -> the deterministic fetch_uniprot fallback still
    returns a usable (unranked) result."""
    monkeypatch.setattr(pd_module, "list_uniprot_candidates", _fake_list_multi)
    monkeypatch.setattr(pd_module, "fetch_uniprot", _fake_fetch_uniprot)

    async def fake_llm_pick(*args, **kwargs):
        raise RuntimeError("NIM unreachable")

    monkeypatch.setattr(pd_module, "_llm_pick_protein", fake_llm_pick)

    result = await protein_data_agent(ProteinDataInput(
        trait_name="cold adaptation",
        gene_list=["UCP1"],
        instruction="test",
        context={"tax_id": 9606},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.proteins[0].source_accession == "P25874"


@pytest.mark.asyncio
async def test_zero_hits_never_calls_llm(monkeypatch):
    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("list_uniprot_candidates should not be called")

    async def _fail_llm_if_called(*args, **kwargs):
        raise AssertionError("_llm_pick_protein should not be called")

    async def _empty_list(gene_symbol, tax_id):
        return []

    monkeypatch.setattr(pd_module, "list_uniprot_candidates", _empty_list)
    monkeypatch.setattr(pd_module, "_llm_pick_protein", _fail_llm_if_called)
    monkeypatch.setattr(pd_module, "fetch_uniprot", _fail_if_called)

    result = await protein_data_agent(ProteinDataInput(
        trait_name="test",
        gene_list=["NOT_A_REAL_GENE"],
        instruction="test",
        context={"tax_id": 9606},
    ))

    assert result.status == AgentStatus.FAILED
    assert result.missing_genes == ["NOT_A_REAL_GENE"]
    assert result.proteins == []


@pytest.mark.asyncio
async def test_one_missing_one_present_only_missing_is_recorded(monkeypatch):
    """One gene has zero reviewed hits (missing), the other has a real hit.
    Only the missing one should end up in missing_genes, and the overall
    status should still be COMPLETED because at least one protein resolved."""
    async def list_candidates(gene_symbol, tax_id):
        if gene_symbol == "MISSING_GENE":
            return []
        return [{
            "source_accession": "P12034",
            "protein_name": "Fibroblast growth factor 5",
            "function_summary": "Regulates hair follicle growth cycle.",
        }]

    monkeypatch.setattr(pd_module, "list_uniprot_candidates", list_candidates)

    result = await protein_data_agent(ProteinDataInput(
        trait_name="test",
        gene_list=["MISSING_GENE", "FGF5"],
        instruction="test",
        context={"tax_id": 9606},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.missing_genes == ["MISSING_GENE"]
    assert len(result.proteins) == 1
    assert result.proteins[0].source_accession == "P12034"


@pytest.mark.asyncio
async def test_grounding_rule_rejects_invented_accession(monkeypatch):
    monkeypatch.setattr(pd_module, "list_uniprot_candidates", _fake_list_multi)
    monkeypatch.setattr(pd_module, "fetch_uniprot", _fake_fetch_uniprot)

    async def fake_llm_pick(*args, **kwargs):
        return "P00000", "invented protein", "hallucinated", "hallucinated"

    monkeypatch.setattr(pd_module, "_llm_pick_protein", fake_llm_pick)

    result = await protein_data_agent(ProteinDataInput(
        trait_name="test",
        gene_list=["UCP1"],
        instruction="test",
        context={"tax_id": 9606},
    ))

    # Fallback should have kicked in because the invented accession was rejected.
    assert result.status == AgentStatus.COMPLETED
    assert result.proteins[0].source_accession == "P25874"
