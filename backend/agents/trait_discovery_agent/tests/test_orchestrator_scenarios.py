"""
Scenario tests for the Trait Discovery orchestrator.

These are not node-level unit tests (see test_workflows.py for that) — each
case here runs the *whole compiled LangGraph graph* end to end and prints a
readable trace of what happened. The point is to demonstrate, not just assert:

    - LangGraph's conditional edges control which path is taken. Nothing here
      changes trait_discovery_graph.py or functional_evidence_graph.py.
    - The LLM boundary is used for exactly two things: (1) resolving which
      external agent to escalate to, via workflows.capability_resolver, and
      (2) writing the final explanation, via workflows.explanation_writer.
      Both are stubbed at the LLM call itself (get_llm / write_explanation),
      never at the orchestration layer around them — the prompt templates,
      the dynamic agent_cards catalog, and the graph routing all run for real.
    - Case 4 asserts the LLM boundary is *not* touched on a hard failure.

Run all four scenarios with readable output:
    pytest tests/test_orchestrator_scenarios.py -v -s

Run just one:
    pytest tests/test_orchestrator_scenarios.py -v -s -k case2
"""
import logging

import pytest

import workflows.capability_resolver as resolver_module
import workflows.trait_discovery_graph as td_graph_module
from workflows.agent_catalog import build_catalog_text
from workflows.capability_resolver import CapabilityResolution
from workflows.state import TraitDiscoveryState
from schemas.common import AgentStatus

# Node-level INFO logs are useful in isolation but drown out the -s trace below.
logging.getLogger().setLevel(logging.WARNING)

_WIDTH = 40


def _print_case(title, input_lines, workflow_lines, llm_decision, final_status):
    print("\n" + "=" * _WIDTH)
    print(f"CASE: {title}")
    print("=" * _WIDTH)
    print("Input:")
    for line in input_lines:
        print(f"  {line}")
    print("Workflow:")
    for line in workflow_lines:
        print(f"  {line}")
    print("LLM decision:")
    print(f"  {llm_decision}")
    print("Final status:")
    print(f"  {final_status}")
    print("=" * _WIDTH)


class _FakeMessage:
    """Minimal stand-in for the AIMessage-like object invoke_with_fallback
    returns — capability_resolver only ever reads `.content` off it."""

    def __init__(self, content: str):
        self.content = content


class _FakeInvokeWithFallback:
    """Stands in for `workflows.llm.invoke_with_fallback` (imported into
    capability_resolver's own namespace as `invoke_with_fallback`). Records
    every (prompt, variables) pair it receives so a test can assert the agent
    catalog it saw was built live from agent_cards/, not hardcoded in the
    orchestrator — reconstructed by rendering the same prompt template the
    real call site uses."""

    def __init__(self, result: CapabilityResolution):
        self.result = result
        self.calls: list[dict] = []

    async def __call__(self, prompt, variables: dict):
        self.calls.append(variables)
        return _FakeMessage(self.result.model_dump_json())

    def rendered_catalog(self, prompt, call_index: int = 0) -> str:
        """Re-render the prompt for a given call so tests can assert on the
        catalog text exactly as the LLM would have seen it."""
        return prompt.format_prompt(**self.calls[call_index]).to_string()


def _patch_resolver(monkeypatch, result: CapabilityResolution) -> _FakeInvokeWithFallback:
    fake = _FakeInvokeWithFallback(result)
    monkeypatch.setattr(resolver_module, "invoke_with_fallback", fake)
    return fake


# ---------------------------------------------------------------------------
# Case 1 — normal successful workflow
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_case1_normal_success(monkeypatch):
    explanation_calls = []

    async def fake_write_explanation(**kwargs):
        explanation_calls.append(kwargs)
        return f"Mock explanation for {kwargs['trait_name']}: genes {kwargs['genes']}."

    monkeypatch.setattr(td_graph_module, "write_explanation", fake_write_explanation)

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["FGF5", "KRT71", "HR"]},
    ))

    assert result["gene_list"] == ["FGF5", "KRT71", "HR"]
    assert result["go_annotations"], "Gene Mapper did not run"
    assert result["pathway_data"], "Functional Evidence (pathways) did not run"
    assert result["protein_data"], "Functional Evidence (protein data) did not run"
    assert result["evidence"], "Literature Support did not run"
    assert len(explanation_calls) == 1, "Explanation Writer LLM should run exactly once"
    assert result["status"] == AgentStatus.COMPLETED

    _print_case(
        "Normal successful workflow",
        ["Trait: fur growth", "Species: mouse", "Genes: FGF5, KRT71, HR"],
        [
            "check_gene_list -> gene_mapper",
            "gene_mapper -> functional_evidence (pathways + protein_data, parallel)",
            "functional_evidence -> literature_support",
            "literature_support -> join_and_route -> aggregate (LLM)",
        ],
        f"Explanation Writer called once: \"{result['explanation'][:60]}...\"",
        result["status"],
    )


# ---------------------------------------------------------------------------
# Case 2 — missing gene list -> capability resolution
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_case2_missing_gene_list_escalation(monkeypatch):
    fake = _patch_resolver(monkeypatch, CapabilityResolution(
        target_agent="Genome Agent",
        prompt_to_target_agent="Resolve a candidate gene list for fur growth in mouse.",
        reasoning="No gene_list is available yet; Genome Agent is the only agent that produces one.",
    ))

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={},  # no gene_list -> nothing for Gene Mapper to work with
    ))

    assert result["gene_list"] == []
    assert not result["go_annotations"], "Gene Mapper must not run without a gene list"
    assert result["status"] == AgentStatus.NEEDS_AGENT
    assert result["target_agent"] == "Genome Agent"
    assert len(fake.calls) == 1, "Capability Resolver LLM should run exactly once"

    # Prove the catalog handed to the LLM was built dynamically from agent_cards/
    # at call time, not a hardcoded string anywhere in the orchestrator. The fake
    # is spliced in at `invoke_with_fallback` (capability_resolver's own imported
    # name), so what it receives is the exact `agent_catalog` variable that gets
    # formatted into the prompt — built fresh from build_catalog_text() each call.
    catalog_seen = fake.calls[0]["agent_catalog"]
    assert "### Genome Agent" in catalog_seen
    assert "### Trait Discovery Agent" not in catalog_seen, "waiting agent must be excluded"
    assert build_catalog_text(exclude="Trait Discovery Agent") in catalog_seen

    _print_case(
        "Missing gene list",
        ["Trait: fur growth", "Genes: []"],
        ["check_gene_list -> escalate_genome_agent -> capability_resolver (LLM)"],
        result["target_agent"],
        result["status"],
    )


# ---------------------------------------------------------------------------
# Case 3 — literature evidence too thin -> capability resolution
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_case3_literature_escalation(monkeypatch):
    fake = _patch_resolver(monkeypatch, CapabilityResolution(
        target_agent="Literature Agent",
        prompt_to_target_agent="Find deeper peer-reviewed evidence for cold adaptation and UCP1.",
        reasoning="Existing literature evidence is below the thin-evidence threshold.",
    ))

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="cold adaptation",
        species_name="mouse",
        instruction="Which genes drive cold adaptation?",
        context={"gene_list": ["UCP1"]},  # resolves fine; literature DB for this trait is thin
    ))

    assert result["go_annotations"], "Gene Mapper should have completed"
    assert result["pathway_data"] and result["protein_data"], "Functional Evidence should have completed"
    assert result["evidence"], "thin evidence must still be returned, not dropped"
    assert result["status"] == AgentStatus.NEEDS_AGENT
    assert result["target_agent"] == "Literature Agent"
    assert len(fake.calls) == 1

    _print_case(
        "Literature evidence too thin",
        ["Trait: cold adaptation", "Genes: UCP1", f"Literature records found: {len(result['evidence'])}"],
        [
            "check_gene_list -> gene_mapper -> functional_evidence -> literature_support",
            "literature_support -> join_and_route -> escalate_literature_agent -> capability_resolver (LLM)",
        ],
        result["target_agent"],
        result["status"],
    )


# ---------------------------------------------------------------------------
# Case 4 — hard subagent failure, no LLM call at all
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_case4_subagent_failure_no_llm_call(monkeypatch):
    async def _fail_if_called_explanation(**kwargs):
        raise AssertionError("write_explanation must not run on a failed workflow")
    monkeypatch.setattr(td_graph_module, "write_explanation", _fail_if_called_explanation)

    async def _guard_invoke_with_fallback(prompt, variables):
        raise AssertionError("capability_resolver must not run for a hard subagent failure")
    monkeypatch.setattr(resolver_module, "invoke_with_fallback", _guard_invoke_with_fallback)

    app = td_graph_module.build_trait_discovery_graph()
    result = await app.ainvoke(TraitDiscoveryState(
        trait_name="fur growth",
        species_name="mouse",
        instruction="Which genes cause fur growth?",
        context={"gene_list": ["ZZZTOP"]},  # not in any mock DB -> Gene Mapper fails
    ))

    assert result["gene_mapper_status"] == AgentStatus.FAILED
    assert result["status"] == AgentStatus.FAILED

    _print_case(
        "Subagent failure (Gene Mapper resolves nothing)",
        ["Trait: fur growth", "Genes: ZZZTOP (not in mock GO database)"],
        [
            "check_gene_list -> gene_mapper (FAILED)",
            "functional_evidence, literature_support still run (both fail on this gene too)",
            "join_and_route -> failed",
        ],
        "(not invoked — assertions above would fail loudly if it were)",
        result["status"],
    )
