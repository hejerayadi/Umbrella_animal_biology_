"""
Unit tests for workflows.nodes.functional_evidence_nodes.merge_node.

§4/§10: merge_node is a pure coordination function — a fixed rule over two
already-computed statuses, not an LLM agent. These tests exercise the node
directly (no graph, no child agents) to keep the decision logic and the
"no LLM anywhere in this node" guarantee independently verifiable, the same
way the Genome Agent guide tests its deterministic protein_structure branch.
"""
import pytest

from schemas.common import AgentStatus
from workflows.nodes.functional_evidence_nodes import merge_node
from workflows.state import FunctionalEvidenceState


def _state(pathways_status, protein_data_status) -> FunctionalEvidenceState:
    return FunctionalEvidenceState(
        gene_list=["FGF5"],
        instruction="find functional evidence",
        pathways_status=pathways_status,
        protein_data_status=protein_data_status,
    )


@pytest.mark.asyncio
async def test_both_completed_merges_to_completed():
    result = await merge_node(_state(AgentStatus.COMPLETED, AgentStatus.COMPLETED))
    assert result["status"] == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_one_failed_one_completed_still_merges_to_completed():
    """§0.2/§8: a single child failing is non-critical — the other still has
    standalone value, so the sub-orchestrator only fails when BOTH children
    fail. Covered in both directions since the rule isn't symmetric-by-luck,
    it's a deliberate `and`."""
    result = await merge_node(_state(AgentStatus.FAILED, AgentStatus.COMPLETED))
    assert result["status"] == AgentStatus.COMPLETED

    result = await merge_node(_state(AgentStatus.COMPLETED, AgentStatus.FAILED))
    assert result["status"] == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_both_failed_merges_to_failed():
    result = await merge_node(_state(AgentStatus.FAILED, AgentStatus.FAILED))
    assert result["status"] == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_merge_node_never_touches_the_llm(monkeypatch):
    """§4.2/§10 negative test, protein_structure-style: assert there is no
    code path in merge_node that could invoke an LLM at all. Every LLM
    invocation in this codebase (bind_tools loops and plain completions
    alike) funnels through workflows.llm.get_llm before it can reach NIM
    (see workflows/llm/__init__.py) — poisoning that single choke point
    means any accidental LLM call here fails loudly instead of silently
    making a real network request during a unit test.
    """
    import workflows.llm as llm_module

    def _fail_if_called(*args, **kwargs):
        raise AssertionError(
            "merge_node has no decision to make (§4.2) — it must never "
            "construct an LLM client"
        )

    monkeypatch.setattr(llm_module, "get_llm", _fail_if_called)

    for pathways_status, protein_data_status in [
        (AgentStatus.COMPLETED, AgentStatus.COMPLETED),
        (AgentStatus.FAILED, AgentStatus.COMPLETED),
        (AgentStatus.COMPLETED, AgentStatus.FAILED),
        (AgentStatus.FAILED, AgentStatus.FAILED),
    ]:
        result = await merge_node(_state(pathways_status, protein_data_status))
        assert result["status"] in (AgentStatus.COMPLETED, AgentStatus.FAILED)
