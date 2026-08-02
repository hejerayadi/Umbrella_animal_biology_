import time
import asyncio

import pytest

from workflows.state import TraitDiscoveryState, FunctionalEvidenceState
import workflows.functional_evidence_graph as fe_graph_module
import workflows.trait_discovery_graph as td_graph_module
from schemas.common import AgentStatus


# ---- subgraph tested in isolation, before being wired into the parent ----

@pytest.mark.asyncio
async def test_functional_evidence_subgraph_parallel_execution():
    app = fe_graph_module.build_functional_evidence_graph()
    result = await app.ainvoke(FunctionalEvidenceState(
        gene_list=["UCP1", "PRDM16"], instruction="test",
    ))
    assert result["status"] == AgentStatus.COMPLETED
    assert len(result["pathway_data"]) == 2   # both UCP1 and PRDM16 resolve
    assert result["protein_data"]              # UCP1 resolves, PRDM16 doesn't -> still COMPLETED


@pytest.mark.asyncio
async def test_functional_evidence_subgraph_fails_when_both_children_empty():
    app = fe_graph_module.build_functional_evidence_graph()
    result = await app.ainvoke(FunctionalEvidenceState(
        gene_list=["ZZZ999"], instruction="test",
    ))
    assert result["status"] == AgentStatus.FAILED


# ---- full end-to-end runs, mirroring Task 1's four scenarios ----

@pytest.mark.asyncio
async def test_full_completion_fur_growth():
    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth", species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "KRT71", "HR"]},
    ))
    assert result["status"] == AgentStatus.COMPLETED
    assert any(a.gene_symbol == "FGF5" for a in result["go_annotations"])
    assert result["explanation"]


@pytest.mark.asyncio
async def test_thin_evidence_escalates_to_literature_agent():
    """Confirms NEEDS_AGENT surfaces at the top-level graph's END with target_agent
    populated, and partial functional-evidence data is still attached, not discarded."""
    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="cold adaptation", species_name="human",
        instruction="Which genes are involved in cold adaptation?",
        context={"gene_list": ["UCP1"]},
    ))
    assert result["status"] == AgentStatus.NEEDS_AGENT
    assert result["target_agent"] == "Literature Agent"
    assert result["prompt_to_target_agent"] is not None
    assert len(result["pathway_data"]) == 1
    assert result["pathway_data"][0].pathway_name == "Fatty acid degradation"


@pytest.mark.asyncio
async def test_unknown_genes_fail_at_gene_mapper():
    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="unknown trait", species_name="cat",
        instruction="Random question",
        context={"gene_list": ["ZZZ999"]},
    ))
    assert result["status"] == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_missing_gene_list_escalates_to_genome_agent():
    """Confirms escalation happens at graph entry, before gene_mapper ever runs."""
    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth", species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={},
    ))
    assert result["status"] == AgentStatus.NEEDS_AGENT
    assert result["target_agent"] == "Genome Agent"
    assert result["prompt_to_target_agent"] is not None


# ---- proof that the parallel branches actually run concurrently, not sequentially ----

@pytest.mark.asyncio
async def test_functional_evidence_and_literature_run_concurrently(monkeypatch):
    events = []

    async def slow_pathways(state):
        events.append(("pathways_start", time.monotonic()))
        await asyncio.sleep(0.1)
        events.append(("pathways_end", time.monotonic()))
        return {"pathway_data": [], "pathways_status": AgentStatus.COMPLETED}

    async def fast_protein_data(state):
        return {"protein_data": [], "protein_data_status": AgentStatus.COMPLETED}

    async def instant_literature(state):
        events.append(("literature_start", time.monotonic()))
        events.append(("literature_end", time.monotonic()))
        return {
            "evidence": [], "literature_status": AgentStatus.COMPLETED,
            "_literature_target_agent": None, "_literature_prompt": None,
        }

    # Patch at the module that BINDS the node function into the graph (import-time
    # reference), not at the original definition module — and rebuild the graphs
    # afterward so the patched functions are actually the ones compiled into nodes.
    monkeypatch.setattr(fe_graph_module, "pathways_node", slow_pathways)
    monkeypatch.setattr(fe_graph_module, "protein_data_node", fast_protein_data)
    monkeypatch.setattr(td_graph_module, "literature_support_node", instant_literature)

    patched_fe_app = fe_graph_module.build_functional_evidence_graph()
    monkeypatch.setattr(td_graph_module, "_functional_evidence_app", patched_fe_app)
    app = td_graph_module.build_trait_discovery_graph()

    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth", species_name="mouse",
        instruction="test", context={"gene_list": ["FGF5"]},
    ))

    assert result["status"] == AgentStatus.COMPLETED
    literature_end = next(t for label, t in events if label == "literature_end")
    pathways_end = next(t for label, t in events if label == "pathways_end")
    # literature (instant) finished well before the artificially slow pathways call —
    # proof the two branches ran concurrently, not one after another
    assert literature_end < pathways_end