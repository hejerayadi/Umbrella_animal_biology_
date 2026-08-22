"""
Full-scope Trait Discovery Orchestrator integration checks (Level 1 of the
full-orchestrator checklist):

  - Functional Evidence and Literature Support must run in PARALLEL after
    Gene Mapper, not sequentially.
  - Gene Mapper resolving *some* genes (partial match) must not halt the
    whole pipeline the way resolving *nothing* does.

All external agents are faked at the same import-site names the node modules
use, so these tests need no network, Qdrant, or Neo4j access.
"""
import asyncio

import pytest

import workflows.nodes.functional_evidence_nodes as fe_nodes
import workflows.nodes.gene_mapper_node as gm_node
import workflows.nodes.literature_support_node as lit_node
import workflows.trait_discovery_graph as td_graph_module
from schemas.common import AgentStatus
from schemas.outputs import (
    GeneMapperOutput,
    GOAnnotation,
    LiteratureSupportOutput,
    PathwayEntry,
    PathwaysOutput,
    ProteinDataOutput,
)
from workflows.state import TraitDiscoveryState


def _patch_downstream_agents(monkeypatch, *, pathways=None, protein_data=None, literature=None):
    monkeypatch.setattr(
        fe_nodes, "pathways_agent",
        pathways or (lambda input: _ok(PathwaysOutput(status=AgentStatus.COMPLETED, pathways=[])))
    )
    monkeypatch.setattr(
        fe_nodes, "protein_data_agent",
        protein_data or (lambda input: _ok(ProteinDataOutput(status=AgentStatus.COMPLETED, proteins=[])))
    )
    monkeypatch.setattr(
        lit_node, "literature_support_agent",
        literature or (lambda input: _ok(LiteratureSupportOutput(status=AgentStatus.COMPLETED, evidence=[])))
    )


async def _ok(value):
    return value


@pytest.mark.asyncio
async def test_functional_evidence_and_literature_support_run_concurrently(monkeypatch):
    """Level 1, item 4: 'Parallel: Functional Evidence + Literature Support
    (no dependency on each other, both only need Gene Mapper's output)'.

    Proven by having each branch block until it has observed that the OTHER
    branch has also started. If the graph were still sequential
    (gene_mapper -> functional_evidence -> literature_support, as it was
    before this patch), Pathways would run to completion inside Functional
    Evidence before Literature Support's node even started, so
    lit_started would never fire in time and this test would time out
    instead of passing.
    """
    fe_started = asyncio.Event()
    lit_started = asyncio.Event()

    async def fake_gene_mapper_agent(input):
        return GeneMapperOutput(
            status=AgentStatus.COMPLETED,
            go_annotations=[GOAnnotation(gene_symbol="FGF5", go_id="GO:0001", go_name="growth")],
        )

    async def fake_pathways_agent(input):
        fe_started.set()
        await asyncio.wait_for(lit_started.wait(), timeout=2)
        return PathwaysOutput(status=AgentStatus.COMPLETED, pathways=[])

    async def fake_literature_support_agent(input):
        lit_started.set()
        await asyncio.wait_for(fe_started.wait(), timeout=2)
        return LiteratureSupportOutput(status=AgentStatus.COMPLETED, evidence=[])

    monkeypatch.setattr(gm_node, "gene_mapper_agent", fake_gene_mapper_agent)
    _patch_downstream_agents(
        monkeypatch, pathways=fake_pathways_agent, literature=fake_literature_support_agent
    )

    async def fake_explain(**kwargs):
        return "ok"
    monkeypatch.setattr(td_graph_module, "write_explanation", fake_explain)

    app = td_graph_module.build_trait_discovery_graph()
    result = await asyncio.wait_for(
        app.ainvoke(TraitDiscoveryState(
            trait_name="fur growth",
            species_name="mouse",
            instruction="Which genes cause fur growth?",
            context={"gene_list": ["FGF5"]},
        )),
        timeout=5,
    )

    assert fe_started.is_set() and lit_started.is_set()
    assert result["status"] == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_gene_mapper_partial_match_does_not_halt_pipeline(monkeypatch):
    """Level 1, item 8: 'Gene Mapper fails / all unmatched -> still pass
    empty/partial go_annotations downstream, flag in final answer, don't
    crash.' A gene list with SOME matches and SOME unmatched genes must
    reach aggregate(), not the failed terminal node."""
    async def fake_gene_mapper_agent(input):
        return GeneMapperOutput(
            status=AgentStatus.FAILED,  # subagent's own contract: any unmatched -> FAILED
            go_annotations=[GOAnnotation(gene_symbol="FGF5", go_id="GO:0001", go_name="growth")],
            unmatched_genes=["BADGENE"],
        )

    monkeypatch.setattr(gm_node, "gene_mapper_agent", fake_gene_mapper_agent)
    _patch_downstream_agents(monkeypatch)

    async def fake_explain(**kwargs):
        return "ok"
    monkeypatch.setattr(td_graph_module, "write_explanation", fake_explain)

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "BADGENE"]},
    ))

    assert result["go_annotations"], "partial annotations must still flow downstream"
    assert result["unmatched_genes"] == ["BADGENE"]
    assert result["gene_mapper_status"] == AgentStatus.COMPLETED, (
        "orchestrator-level routing status: a partial match is non-critical"
    )
    assert result["status"] == AgentStatus.COMPLETED, "pipeline must not halt on a partial match"


@pytest.mark.asyncio
async def test_gene_mapper_total_failure_still_halts_pipeline(monkeypatch):
    """Boundary check the other direction (§0.2): resolving NOTHING is still
    the one Gene Mapper failure mode critical enough to hard-fail the run —
    this patch narrows the halt condition, it doesn't remove it."""
    async def fake_gene_mapper_agent(input):
        return GeneMapperOutput(
            status=AgentStatus.FAILED, go_annotations=[], unmatched_genes=["BADGENE"]
        )

    monkeypatch.setattr(gm_node, "gene_mapper_agent", fake_gene_mapper_agent)
    _patch_downstream_agents(monkeypatch)

    async def fail_if_called(**kwargs):
        raise AssertionError("write_explanation must not run on a failed workflow")
    monkeypatch.setattr(td_graph_module, "write_explanation", fail_if_called)

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["BADGENE"]},
    ))

    assert result["go_annotations"] == []
    assert result["unmatched_genes"] == ["BADGENE"]
    assert result["gene_mapper_status"] == AgentStatus.FAILED
    assert result["status"] == AgentStatus.FAILED
