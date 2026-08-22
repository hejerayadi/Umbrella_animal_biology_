import pytest

from schemas.inputs import LiteratureSupportInput
from schemas.common import AgentStatus
from subagents import literature_support as ls_module
from subagents.literature_support import literature_support_agent, mock_literature_support


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
_TWO_RECORDS = [
    {"pmid": "18239092", "title": "FGF5 and hair cycle regulation", "year": 2008,
     "short_summary": "Links FGF5 mutation to hair length in mammals."},
    {"pmid": "30112233", "title": "Follicle regulatory network in rodents", "year": 2019,
     "short_summary": "Broader gene network context around FGF5 signaling."},
]

_ONE_RECORD = [
    {"pmid": "26123456", "title": "UCP1 and non-shivering thermogenesis", "year": 2016,
     "short_summary": "UCP1 role in brown fat heat production."},
]


async def _fake_fetch_two(trait, genes):
    return _TWO_RECORDS

async def _fake_fetch_one(trait, genes):
    return _ONE_RECORD

async def _fake_fetch_none(trait, genes):
    return []


# --------------------------------------------------------------------------- #
#  Real agent tests
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_two_solid_records_llm_judges_sufficient(monkeypatch):
    monkeypatch.setattr(ls_module, "request_literature_evidence", _fake_fetch_two)

    async def fake_judge(trait, genes, evidence, thin_flag):
        assert len(evidence) == 2
        assert thin_flag is False
        return True, ["18239092"], "clear FGF5 link", ""

    monkeypatch.setattr(ls_module, "_llm_judge_sufficiency", fake_judge)

    result = await literature_support_agent(LiteratureSupportInput(
        trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert len(result.evidence) == 2
    assert result.target_agent is None


@pytest.mark.asyncio
async def test_one_thin_record_llm_recommends_escalation(monkeypatch):
    monkeypatch.setattr(ls_module, "request_literature_evidence", _fake_fetch_one)

    async def fake_judge(trait, genes, evidence, thin_flag):
        assert thin_flag is True
        return False, [], "only one record, insufficient", "need more UCP1 evidence"

    monkeypatch.setattr(ls_module, "_llm_judge_sufficiency", fake_judge)

    result = await literature_support_agent(LiteratureSupportInput(
        trait_name="cold adaptation", gene_list=["UCP1"], instruction="test", context={},
    ))

    assert result.status == AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Literature Agent"
    assert result.prompt_to_target_agent == "need more UCP1 evidence"
    assert len(result.evidence) == 1


@pytest.mark.asyncio
async def test_no_evidence_fails_without_calling_llm(monkeypatch):
    monkeypatch.setattr(ls_module, "request_literature_evidence", _fake_fetch_none)

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("_llm_judge_sufficiency should not be called")

    monkeypatch.setattr(ls_module, "_llm_judge_sufficiency", _fail_if_called)

    result = await literature_support_agent(LiteratureSupportInput(
        trait_name="unknown trait", gene_list=["XYZ"], instruction="test", context={},
    ))

    assert result.status == AgentStatus.FAILED
    assert result.evidence == []


@pytest.mark.asyncio
async def test_llm_unavailable_falls_back_to_count_threshold_sufficient(monkeypatch):
    monkeypatch.setattr(ls_module, "request_literature_evidence", _fake_fetch_two)

    async def fake_judge(*args, **kwargs):
        raise RuntimeError("NIM unreachable")

    monkeypatch.setattr(ls_module, "_llm_judge_sufficiency", fake_judge)

    result = await literature_support_agent(LiteratureSupportInput(
        trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
    ))

    # 2 records > threshold (1) -> deterministic fallback treats as sufficient
    assert result.status == AgentStatus.COMPLETED
    assert len(result.evidence) == 2


@pytest.mark.asyncio
async def test_llm_unavailable_falls_back_to_count_threshold_thin(monkeypatch):
    monkeypatch.setattr(ls_module, "request_literature_evidence", _fake_fetch_one)

    async def fake_judge(*args, **kwargs):
        raise RuntimeError("NIM unreachable")

    monkeypatch.setattr(ls_module, "_llm_judge_sufficiency", fake_judge)

    result = await literature_support_agent(LiteratureSupportInput(
        trait_name="cold adaptation", gene_list=["UCP1"], instruction="test", context={},
    ))

    assert result.status == AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Literature Agent"


@pytest.mark.asyncio
async def test_literature_agent_retry_then_failed(monkeypatch):
    calls = []

    async def flaky(trait, genes):
        calls.append(1)
        raise RuntimeError("timeout")

    monkeypatch.setattr(ls_module, "request_literature_evidence", flaky)

    result = await literature_support_agent(LiteratureSupportInput(
        trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
    ))

    assert len(calls) == 2  # initial + one retry
    assert result.status == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_grounding_rule_rejects_invented_pmid(monkeypatch):
    monkeypatch.setattr(ls_module, "request_literature_evidence", _fake_fetch_two)

    async def fake_judge(trait, genes, evidence, thin_flag):
        # cites a pmid that was never in the retrieved evidence
        return True, ["99999999"], "invented citation", ""

    monkeypatch.setattr(ls_module, "_llm_judge_sufficiency", fake_judge)

    result = await literature_support_agent(LiteratureSupportInput(
        trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
    ))

    # Grounding failure -> deterministic fallback kicks in; 2 records is not
    # thin, so it still resolves to COMPLETED via the fallback path.
    assert result.status == AgentStatus.COMPLETED
    assert len(result.evidence) == 2


@pytest.mark.asyncio
async def test_dedupe_by_pmid_collapses_exact_duplicates(monkeypatch):
    duplicated = _TWO_RECORDS + [_TWO_RECORDS[0]]

    async def fake_fetch(trait, genes):
        return duplicated

    monkeypatch.setattr(ls_module, "request_literature_evidence", fake_fetch)

    async def fake_judge(trait, genes, evidence, thin_flag):
        assert len(evidence) == 2  # duplicate pmid already collapsed
        return True, [], "ok", ""

    monkeypatch.setattr(ls_module, "_llm_judge_sufficiency", fake_judge)

    result = await literature_support_agent(LiteratureSupportInput(
        trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
    ))

    assert result.status == AgentStatus.COMPLETED
    assert len(result.evidence) == 2


# --------------------------------------------------------------------------- #
#  Mock contract test (ensures CI backward compat)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_mock_literature_support_contract():
    result = await mock_literature_support(LiteratureSupportInput(
        trait_name="fur growth", gene_list=["FGF5"], instruction="test", context={},
    ))
    assert result.status == AgentStatus.COMPLETED
    assert len(result.evidence) == 2

    result_thin = await mock_literature_support(LiteratureSupportInput(
        trait_name="cold adaptation", gene_list=["UCP1"], instruction="test", context={},
    ))
    assert result_thin.status == AgentStatus.NEEDS_AGENT
    assert result_thin.target_agent == "Literature Agent"


# --------------------------------------------------------------------------- #
#  Neo4j write-tool wiring (previously a logged no-op stub — see
#  kb/neo4j_store.py). These confirm the LLM's bound write_trait_gene_
#  relationship tool now actually reaches Neo4j (or fails soft), not that
#  the graph write logic itself is correct — that's kb/test_neo4j_store.py.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_write_tool_is_wired_to_neo4j_not_a_stub(monkeypatch):
    import kb.neo4j_store as neo4j_store
    from kb.sources.literature_agent_client import write_trait_gene_relationship

    calls = []

    async def fake_upsert(trait_name, gene_symbol, pmid):
        calls.append((trait_name, gene_symbol, pmid))
        return True

    monkeypatch.setattr(neo4j_store, "upsert_trait_gene_relationship", fake_upsert)
    # literature_agent_client.py imported the function by name, so patch it there too.
    import kb.sources.literature_agent_client as lac_module
    monkeypatch.setattr(lac_module, "upsert_trait_gene_relationship", fake_upsert)

    result = await write_trait_gene_relationship.ainvoke({
        "trait_name": "fur growth", "gene_symbol": "FGF5", "pmid": "18239092",
    })

    assert result is True
    assert calls == [("fur growth", "FGF5", "18239092")]


@pytest.mark.asyncio
async def test_write_tool_failure_does_not_raise_or_change_evidence_status(monkeypatch):
    """A Neo4j write failure must fail soft (§9) — evidence is already valid
    and already returned regardless of whether the graph write lands."""
    import kb.sources.literature_agent_client as lac_module

    async def failing_upsert(trait_name, gene_symbol, pmid):
        return False  # kb/neo4j_store.py never raises out of this function

    monkeypatch.setattr(lac_module, "upsert_trait_gene_relationship", failing_upsert)

    result = await lac_module.write_trait_gene_relationship.ainvoke({
        "trait_name": "fur growth", "gene_symbol": "FGF5", "pmid": "18239092",
    })

    assert result is False  # surfaced, but nothing raised
