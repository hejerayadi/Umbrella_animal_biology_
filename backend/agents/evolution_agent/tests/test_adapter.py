"""Tests for the HTTP → orchestrator adapter and intent classifier.

No network, no API key — the LLM is always stubbed.
"""

from __future__ import annotations

import json
import pytest

from backend.agents.evolution_agent.intent import (
    RecognizedIntent,
    _extract_json,
    _to_intent,
    classify_intent,
)
from backend.agents.evolution_agent.orchestrator_adapter import (
    OrchestratorEvolutionAgent,
    resolve_species,
    to_orchestrator_request,
    to_platform_result,
)
from backend.agents.evolution_agent.schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    EvolutionAnalysisResult,
    PlannedFeature,
)


# ---------------------------------------------------------------------------
# Stub LLM (shared across all async tests)
# ---------------------------------------------------------------------------

class _StubLLM:
    """Returns one canned reply (or raises) for every ainvoke call."""

    def __init__(self, content: str | Exception):
        self._content = content

    async def ainvoke(self, _messages):
        if isinstance(self._content, Exception):
            raise self._content
        return type("R", (), {"content": self._content})()


# ---------------------------------------------------------------------------
# Fixture helper for a minimal EvolutionAnalysisResult
# ---------------------------------------------------------------------------

def _make_mock_analysis() -> EvolutionAnalysisResult:
    from backend.agents.evolution_agent.schema import (
        MolecularComparisonResult,
        PhylogeneticResult,
        SimilarityEdge,
        SpeciesGroup,
    )
    mc = MolecularComparisonResult(
        species_list=["homo sapiens", "mus musculus"],
        similarity_scores=[SimilarityEdge("homo sapiens", "mus musculus", 0.85)],
        species_groups=[SpeciesGroup(0, ["homo sapiens", "mus musculus"], 0.85)],
        similarity_network={
            "homo sapiens": [{"neighbour": "mus musculus", "score": 0.85}],
            "mus musculus": [{"neighbour": "homo sapiens", "score": 0.85}],
        },
    )
    phylo = PhylogeneticResult(
        newick_tree="('homo sapiens':9.0,'mus musculus':9.0);",
        tree_url="https://evolution.umbrella.local/tree/test.svg",
        model="LG+G4",
        bootstrap_support={"node_root": 90},
        confidence_values={"homo sapiens": 0.97, "mus musculus": 0.93},
        overall_confidence=0.90,
    )
    return EvolutionAnalysisResult(
        species_list=["homo sapiens", "mus musculus"],
        molecular=mc,
        phylogenetic=phylo,
        overall_confidence=0.875,
        source_agents=["Evolution Agent Orchestrator"],
    )


# ---------------------------------------------------------------------------
# intent: JSON parsing helpers
# ---------------------------------------------------------------------------

def test_extract_json_bare_object() -> None:
    assert _extract_json('{"feature":"full_analysis"}') == {"feature": "full_analysis"}


def test_extract_json_survives_code_fences() -> None:
    assert _extract_json(
        '```json\n{"feature":"molecular_comparison"}\n```'
    )["feature"] == "molecular_comparison"


def test_extract_json_embedded_in_prose() -> None:
    assert _extract_json(
        'Sure! {"feature":"phylogenetic_tree"} Hope that helps.'
    )["feature"] == "phylogenetic_tree"


def test_extract_json_returns_none_for_no_json() -> None:
    assert _extract_json("no json here") is None


def test_invented_feature_is_rejected() -> None:
    intent = _to_intent(
        {"feature": "protein_folding", "species_list": ["Homo sapiens"]}
    )
    assert intent.feature == PlannedFeature.CLARIFICATION_REQUIRED
    assert not intent.is_usable


def test_null_species_entries_are_dropped() -> None:
    intent = _to_intent({
        "feature": "full_analysis",
        "species_list": ["Homo sapiens", "null", "  "],
    })
    assert intent.species_list == ["Homo sapiens"]


def test_divergence_time_rejected_as_unknown() -> None:
    """divergence_time was removed in Sprint 2 — must not be a valid feature."""
    intent = _to_intent({"feature": "divergence_time", "species_list": []})
    assert intent.feature == PlannedFeature.CLARIFICATION_REQUIRED


# ---------------------------------------------------------------------------
# intent: classify_intent with stubbed LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backend_failure_never_raises() -> None:
    intent = await classify_intent("anything", llm=_StubLLM(RuntimeError("down")))
    assert intent.feature == PlannedFeature.CLARIFICATION_REQUIRED
    assert intent.source == "error"


@pytest.mark.asyncio
async def test_unparsable_output_returns_none_feature() -> None:
    intent = await classify_intent("anything", llm=_StubLLM("no json"))
    assert intent.feature == PlannedFeature.CLARIFICATION_REQUIRED
    assert intent.source == "unparsable"


@pytest.mark.asyncio
async def test_full_analysis_parsed() -> None:
    reply = json.dumps({
        "feature": "full_analysis",
        "species_list": ["Homo sapiens", "Mus musculus"],
        "reference_species": None,
        "explicitly_requested_both": True,
    })
    intent = await classify_intent("compare", llm=_StubLLM(reply))
    assert intent.feature == PlannedFeature.FULL_ANALYSIS
    assert intent.is_usable
    assert intent.species_list == ["Homo sapiens", "Mus musculus"]


@pytest.mark.asyncio
async def test_molecular_comparison_parsed() -> None:
    reply = json.dumps({
        "feature": "molecular_comparison",
        "species_list": ["Homo sapiens", "Pan troglodytes"],
        "reference_species": None,
    })
    intent = await classify_intent("how similar", llm=_StubLLM(reply))
    assert intent.feature == PlannedFeature.MOLECULAR_COMPARISON


@pytest.mark.asyncio
async def test_phylogenetic_tree_parsed() -> None:
    reply = json.dumps({
        "feature": "phylogenetic_tree",
        "species_list": ["Homo sapiens", "Mus musculus", "Gallus gallus"],
        "reference_species": "Gallus gallus",
    })
    intent = await classify_intent("show tree", llm=_StubLLM(reply))
    assert intent.feature == PlannedFeature.PHYLOGENETIC_TREE
    assert intent.reference_species == "Gallus gallus"


# ---------------------------------------------------------------------------
# adapter: request construction
# ---------------------------------------------------------------------------

def test_context_species_list_beats_classifier() -> None:
    species = resolve_species(
        {"species_list": ["Homo sapiens", "Mus musculus"]},
        RecognizedIntent(feature="full_analysis", species_list=["human", "mouse"]),
    )
    assert species == ["Homo sapiens", "Mus musculus"]


def test_classifier_species_used_when_context_empty() -> None:
    species = resolve_species(
        {},
        RecognizedIntent(
            feature="full_analysis",
            species_list=["Homo sapiens", "Pan troglodytes"],
        ),
    )
    assert species == ["Homo sapiens", "Pan troglodytes"]


def test_context_species_string_accepted() -> None:
    species = resolve_species(
        {"species": "Homo sapiens"},
        RecognizedIntent(feature="full_analysis"),
    )
    assert species == ["Homo sapiens"]


def test_to_orchestrator_request_sets_feature_and_species() -> None:
    req = to_orchestrator_request(
        AgentRequest(
            instruction="compare",
            context={"species_list": ["Homo sapiens", "Pan troglodytes"]},
        ),
        RecognizedIntent(
            feature="full_analysis",
            species_list=["human", "chimp"],
            reference_species="Homo sapiens",
        ),
    )
    assert req.feature == "full_analysis"
    assert req.species_list == ["Homo sapiens", "Pan troglodytes"]   # context wins
    assert req.reference_species == "Homo sapiens"
    assert req.context["feature"] == "full_analysis"


# ---------------------------------------------------------------------------
# adapter: result reshaping
# ---------------------------------------------------------------------------

def test_completed_result_publishes_flat_evolution_output() -> None:
    analysis = _make_mock_analysis()
    mapped = to_platform_result(
        AgentResult(
            status=AgentStatus.COMPLETED,
            output=analysis,
            tree_url=analysis.phylogenetic.tree_url,
            confidence=0.875,
            source_agents=["Evolution Agent Orchestrator"],
        )
    )
    assert mapped.status is AgentStatus.COMPLETED
    # Flat structure — all keys at top level, no nested "evolution" wrapper
    assert mapped.output["status"]        == "completed"
    assert mapped.output["decision"]      == "analysis_complete"
    assert mapped.output["explanation"]
    # Reported from the workers that actually ran. The fixture builds an
    # analysis with no mocked-provider flags set, i.e. nothing claims to be
    # mocked, so the published output must not say the scores are.
    assert mapped.output["score_is_mock"] is False
    assert mapped.output["providers_are_mocked"] == {}
    assert mapped.output["species_list"]
    assert mapped.output["newick_tree"]
    assert mapped.output["model"]
    assert mapped.output["tree_url"]


def test_evolution_output_is_json_serialisable() -> None:
    mapped = to_platform_result(
        AgentResult(status=AgentStatus.COMPLETED, output=_make_mock_analysis())
    )
    json.dumps(mapped.output)   # raises if anything non-serialisable leaked in


def test_failed_result_passes_through() -> None:
    original = AgentResult(status=AgentStatus.FAILED, output="error")
    assert to_platform_result(original) is original


def test_needs_agent_passes_through() -> None:
    original = AgentResult(
        status=AgentStatus.NEEDS_AGENT,
        target_agent="Genome Agent",
        prompt_to_target_agent="retrieve",
        output={},
    )
    result = to_platform_result(original)
    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Genome Agent"


# ---------------------------------------------------------------------------
# adapter: pre-flight refusals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unclassifiable_prompt_fails_with_useful_message(
    monkeypatch,
) -> None:
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _no(_p, **_k):
        return RecognizedIntent(feature=None, source="unparsable")

    monkeypatch.setattr(mod, "classify_intent", _no)
    agent  = OrchestratorEvolutionAgent(orchestrator=object())
    result = await agent.run(AgentRequest(instruction="hello", context={}))
    assert result.status is AgentStatus.CONTINUE
    assert "clarification" in str(result.output).lower()


@pytest.mark.asyncio
async def test_no_species_fails_early(monkeypatch) -> None:
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _fa(_p, **_k):
        return RecognizedIntent(feature="full_analysis", species_list=None)

    monkeypatch.setattr(mod, "classify_intent", _fa)
    agent  = OrchestratorEvolutionAgent(orchestrator=object())
    result = await agent.run(AgentRequest(instruction="compare", context={}))
    assert result.status is AgentStatus.FAILED
    assert "species" in result.output.lower()


@pytest.mark.asyncio
async def test_one_species_fails_with_count_message(monkeypatch) -> None:
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _one(_p, **_k):
        return RecognizedIntent(
            feature="full_analysis", species_list=["Homo sapiens"]
        )

    monkeypatch.setattr(mod, "classify_intent", _one)
    agent  = OrchestratorEvolutionAgent(orchestrator=object())
    result = await agent.run(AgentRequest(instruction="analyse", context={}))
    assert result.status is AgentStatus.FAILED
    assert "1" in result.output


@pytest.mark.asyncio
async def test_caller_supplied_feature_skips_classification(
    orchestrator, monkeypatch
) -> None:
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _must_not_run(_p, **_k):
        raise AssertionError("classify_intent must be skipped")

    monkeypatch.setattr(mod, "classify_intent", _must_not_run)
    agent  = OrchestratorEvolutionAgent(orchestrator=orchestrator)
    result = await agent.run(
        AgentRequest(
            instruction="compare",
            context={"feature": "full_analysis"},
            species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
        )
    )
    assert result.status in {AgentStatus.COMPLETED, AgentStatus.NEEDS_AGENT}


@pytest.mark.asyncio
async def test_request_feature_field_skips_classification(
    orchestrator, monkeypatch
) -> None:
    import backend.agents.evolution_agent.orchestrator_adapter as mod

    async def _must_not_run(_p, **_k):
        raise AssertionError("classify_intent must be skipped")

    monkeypatch.setattr(mod, "classify_intent", _must_not_run)
    agent  = OrchestratorEvolutionAgent(orchestrator=orchestrator)
    result = await agent.run(
        AgentRequest(
            instruction="compare",
            context={},
            feature="full_analysis",
            species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
        )
    )
    assert result.status in {AgentStatus.COMPLETED, AgentStatus.NEEDS_AGENT}
