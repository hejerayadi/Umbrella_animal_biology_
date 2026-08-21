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
