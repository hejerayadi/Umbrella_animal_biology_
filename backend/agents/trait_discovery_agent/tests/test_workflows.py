import pytest

import workflows.capability_resolver as resolver_module
import workflows.nodes.escalation_nodes as esc_module
import workflows.trait_discovery_graph as td_graph_module
from workflows.capability_resolver import CapabilityResolution
from workflows.state import TraitDiscoveryState
from schemas.common import AgentStatus


async def fake_resolve_genome(*, waiting_agent, need_description, known_context):
    return CapabilityResolution(
        target_agent="Genome Agent",
        prompt_to_target_agent=f"Resolve genes: {need_description}",
        reasoning="fake",
    )


async def fake_write_explanation(**kwargs):
    return f"Mock explanation for {kwargs['trait_name']}"


@pytest.fixture
def patch_llm_nodes(monkeypatch):
    monkeypatch.setattr(esc_module, "resolve_capability", fake_resolve_genome)
    monkeypatch.setattr(td_graph_module, "write_explanation", fake_write_explanation)


@pytest.mark.asyncio
async def test_escalation_and_aggregate_nodes_work(patch_llm_nodes):
    state = TraitDiscoveryState(
        trait_name="coat color",
        species_name="mouse",
        instruction="find relevant genes",
        go_annotations=[],
        pathway_data=[],
        protein_data=[],
        evidence=[],
    )

    genome_result = await esc_module.escalate_genome_agent_node(state)
    assert genome_result["status"].value == "needs_agent"
    assert genome_result["target_agent"] == "Genome Agent"

    aggregate_result = await td_graph_module.aggregate_node(state)
    assert aggregate_result["status"].value == "completed"
    assert "Mock explanation" in aggregate_result["explanation"]


@pytest.mark.asyncio
async def test_trait_discovery_graph_propagates_gene_list_and_runs_nodes(monkeypatch):
    monkeypatch.setattr(td_graph_module, "write_explanation", fake_write_explanation)
    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "KRT71", "HR"]},
    ))

    assert result["gene_list"] == ["FGF5", "KRT71", "HR"]
    assert result["go_annotations"]
    assert result["pathway_data"]
    assert result["protein_data"]
    assert result["evidence"]
    assert result["status"] == AgentStatus.COMPLETED


def test_catalog_text_uses_json_card_fields():
    from workflows.agent_catalog import build_catalog_text

    catalog = build_catalog_text()
    assert "### Genome Agent" in catalog
    assert "Role:" in catalog
    assert "Call when:" in catalog


@pytest.mark.asyncio
async def test_resolve_capability_parses_valid_json(monkeypatch):
    class _Resp:
        content = (
            '{"target_agent":"Genome Agent",'
            '"prompt_to_target_agent":"Resolve genes for fur growth in mouse.",'
            '"reasoning":"Genome Agent provides missing gene candidates."}'
        )

    async def _fake_invoke(*args, **kwargs):
        return _Resp()

    monkeypatch.setattr(resolver_module, "invoke_with_fallback", _fake_invoke)

    result = await resolver_module.resolve_capability(
        waiting_agent="Trait Discovery Agent",
        need_description="Need a candidate gene list for fur growth.",
        known_context="trait=fur growth, species=mouse",
    )

    assert result.target_agent == "Genome Agent"
    assert "Resolve genes" in result.prompt_to_target_agent


@pytest.mark.asyncio
async def test_resolve_capability_falls_back_on_non_json(monkeypatch):
    class _Resp:
        content = (
            "We should escalate this to the Literature Agent because the evidence is too thin "
            "and more papers are required before concluding."
        )

    async def _fake_invoke(*args, **kwargs):
        return _Resp()

    monkeypatch.setattr(resolver_module, "invoke_with_fallback", _fake_invoke)

    result = await resolver_module.resolve_capability(
        waiting_agent="Literature Support",
        need_description="Evidence for cold adaptation is too thin.",
        known_context="genes so far=['UCP1']",
    )

    assert result.target_agent == "Literature Agent"
    assert "Known context:" in result.prompt_to_target_agent
    assert "fallback" in result.reasoning.lower()