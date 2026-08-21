import pytest

from schemas.inputs import GeneMapperInput
from schemas.outputs import GOAnnotation
from schemas.common import AgentStatus
from subagents import gene_mapper as gm_module
from subagents.gene_mapper import gene_mapper_agent, mock_gene_mapper


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
async def _fake_list_single(gene, acc):
    return [{"go_id": "GO:0042633", "go_name": "hair cycle"}]

async def _fake_list_multi(gene, acc):
    return [
        {"go_id": "GO:0009408", "go_name": "response to heat"},
        {"go_id": "GO:0009409", "go_name": "response to cold"},
    ]

async def _fake_resolve(go_id):
    mapping = {
        "GO:0042633": "hair cycle",
        "GO:0009408": "response to heat",
        "GO:0009409": "response to cold",
    }
    return mapping.get(go_id)

async def _fake_fetch(gene, acc):
    return GOAnnotation(gene_symbol=gene, go_id="GO:0009408", go_name="response to heat")


# --------------------------------------------------------------------------- #
#  Real agent tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_single_candidate_skips_llm(monkeypatch):
    monkeypatch.setattr(gm_module, "list_go_candidates", _fake_list_single)
    monkeypatch.setattr(gm_module, "resolve_go_term_name", _fake_resolve)

    result = await gene_mapper_agent(GeneMapperInput(
        trait_name="hair growth",
        gene_list=["HR"],
        species_name="human",
        instruction="test",
        context={"uniprot_accessions": {"HR": "P56524"}},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.go_annotations[0].go_id == "GO:0042633"
    assert result.unmatched_genes == []


@pytest.mark.asyncio
async def test_multi_candidate_uses_llm(monkeypatch):
    monkeypatch.setattr(gm_module, "list_go_candidates", _fake_list_multi)
    monkeypatch.setattr(gm_module, "resolve_go_term_name", _fake_resolve)

    async def fake_llm_pick(trait, gene, candidates, *args, **kwargs):
        return "GO:0009409", "response to cold", "trait is cold adaptation"

    monkeypatch.setattr(gm_module, "_llm_pick_candidate", fake_llm_pick)

    result = await gene_mapper_agent(GeneMapperInput(
        trait_name="cold adaptation",
        gene_list=["UCP1"],
        species_name="human",
        instruction="test",
        context={"uniprot_accessions": {"UCP1": "P25874"}},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.go_annotations[0].go_id == "GO:0009409"
    assert result.go_annotations[0].go_name == "response to cold"


@pytest.mark.asyncio
async def test_three_candidates_one_trait_relevant_uses_llm_pick(monkeypatch):
    """Gene with 3 GO candidates, only one of which is trait-relevant: the
    LLM's pick must be the one used in the output, and the gene must not
    land in unmatched_genes."""
    async def _fake_list_three(gene, acc):
        return [
            {"go_id": "GO:0001", "go_name": "unrelated process one"},
            {"go_id": "GO:0042633", "go_name": "hair cycle"},
            {"go_id": "GO:0002", "go_name": "unrelated process two"},
        ]

    monkeypatch.setattr(gm_module, "list_go_candidates", _fake_list_three)
    monkeypatch.setattr(gm_module, "resolve_go_term_name", _fake_resolve)

    async def fake_llm_pick(trait, gene, candidates, *args, **kwargs):
        assert len(candidates) == 3
        return "GO:0042633", "hair cycle", "most trait-relevant of the three"

    monkeypatch.setattr(gm_module, "_llm_pick_candidate", fake_llm_pick)

    result = await gene_mapper_agent(GeneMapperInput(
        trait_name="hair growth",
        gene_list=["HR"],
        species_name="human",
        instruction="test",
        context={"uniprot_accessions": {"HR": "P56524"}},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.go_annotations[0].go_id == "GO:0042633"
    assert result.go_annotations[0].go_name == "hair cycle"
    assert result.unmatched_genes == []


@pytest.mark.asyncio
async def test_llm_unreachable_falls_back_to_first_candidate(monkeypatch):
    monkeypatch.setattr(gm_module, "list_go_candidates", _fake_list_multi)
    monkeypatch.setattr(gm_module, "fetch_go_annotation", _fake_fetch)

    async def fake_llm_pick(*args, **kwargs):
        raise RuntimeError("NIM unreachable")

    monkeypatch.setattr(gm_module, "_llm_pick_candidate", fake_llm_pick)

    result = await gene_mapper_agent(GeneMapperInput(
        trait_name="heat response",
        gene_list=["UCP1"],
        species_name="human",
        instruction="test",
        context={"uniprot_accessions": {"UCP1": "P25874"}},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert result.go_annotations[0].go_id == "GO:0009408"


@pytest.mark.asyncio
async def test_no_uniprot_goes_unmatched():
    result = await gene_mapper_agent(GeneMapperInput(
        trait_name="test",
        gene_list=["UNKNOWN"],
        species_name="human",
        instruction="test",
        context={},
    ))
    assert result.status == AgentStatus.FAILED
    assert result.unmatched_genes == ["UNKNOWN"]
    assert result.go_annotations == []


@pytest.mark.asyncio
async def test_no_uniprot_accession_never_calls_llm(monkeypatch):
    """No uniprot_accession in context means there's nothing to disambiguate,
    so the gene must be marked unmatched WITHOUT ever reaching the LLM pick
    (list_go_candidates should not even be called)."""
    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("list_go_candidates should not be called")

    async def _fail_llm_if_called(*args, **kwargs):
        raise AssertionError("_llm_pick_candidate should not be called")

    monkeypatch.setattr(gm_module, "list_go_candidates", _fail_if_called)
    monkeypatch.setattr(gm_module, "_llm_pick_candidate", _fail_llm_if_called)

    result = await gene_mapper_agent(GeneMapperInput(
        trait_name="test",
        gene_list=["NO_ACCESSION_GENE"],
        species_name="human",
        instruction="test",
        context={"uniprot_accessions": {}},
    ))

    assert result.status == AgentStatus.FAILED
    assert result.unmatched_genes == ["NO_ACCESSION_GENE"]
    assert result.go_annotations == []


@pytest.mark.asyncio
async def test_quickgo_retry_then_unmatched(monkeypatch):
    calls = []
    async def flaky(gene, acc):
        calls.append(1)
        raise RuntimeError("timeout")

    monkeypatch.setattr(gm_module, "list_go_candidates", flaky)

    result = await gene_mapper_agent(GeneMapperInput(
        trait_name="test",
        gene_list=["HR"],
        species_name="human",
        instruction="test",
        context={"uniprot_accessions": {"HR": "P56524"}},
    ))
    assert len(calls) == 2  # initial + one retry
    assert result.status == AgentStatus.FAILED
    assert "HR" in result.unmatched_genes


@pytest.mark.asyncio
async def test_grounding_rule_rejects_invented_go_id(monkeypatch):
    monkeypatch.setattr(gm_module, "list_go_candidates", _fake_list_multi)
    monkeypatch.setattr(gm_module, "resolve_go_term_name", _fake_resolve)

    async def fake_llm_pick(*args, **kwargs):
        return "GO:9999999", "fake term", "invented"

    monkeypatch.setattr(gm_module, "_llm_pick_candidate", fake_llm_pick)
    monkeypatch.setattr(gm_module, "fetch_go_annotation", _fake_fetch)

    result = await gene_mapper_agent(GeneMapperInput(
        trait_name="test",
        gene_list=["UCP1"],
        species_name="human",
        instruction="test",
        context={"uniprot_accessions": {"UCP1": "P25874"}},
    ))

    # Fallback should have kicked in because the invented ID was rejected
    assert result.status == AgentStatus.COMPLETED
    assert result.go_annotations[0].go_id == "GO:0009408"


# --------------------------------------------------------------------------- #
#  Mock contract test (ensures CI backward compat)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_mock_gene_mapper_contract():
    result = await mock_gene_mapper(GeneMapperInput(
        trait_name="coat color",
        gene_list=["FGF5", "KRT71", "HR", "UNKNOWN"],
        species_name="mouse",
        instruction="test",
        context={},
    ))
    assert result.status == AgentStatus.FAILED  # UNKNOWN is unmatched
    assert len(result.go_annotations) == 3
    assert "UNKNOWN" in result.unmatched_genes