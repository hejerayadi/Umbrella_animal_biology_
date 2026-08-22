"""Tests for the HTTP -> orchestrator adapter and the intent classifier.

Narrow on purpose: the mapping functions and the classifier's parsing, with a
stubbed LLM. No network, no API key, no HTTP server - so these stay fast and
cannot fail for reasons unrelated to the mapping.

The orchestrator's own behaviour is covered by the existing suite, which knows
nothing about this layer.
"""

from __future__ import annotations

import pytest

from backend.agents.biodiversity_agent.intent import (
    RecognizedIntent,
    _extract_json,
    _to_intent,
    classify_intent,
)
from backend.agents.biodiversity_agent.orchestrator_adapter import (
    OrchestratorBiodiversityAgent,
    resolve_species,
    to_orchestrator_request,
    to_platform_result,
)
from backend.agents.biodiversity_agent.schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    BiodiversityFeature,
)

_DISTRIBUTION = BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value
_HOTSPOTS = BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value


class _StubLLM:
    """Returns one canned model reply, whatever it is asked."""

    def __init__(self, content: str | Exception):
        self._content = content

    async def ainvoke(self, _messages):
        if isinstance(self._content, Exception):
            raise self._content
        return type("Response", (), {"content": self._content})()


# --------------------------------------------------------------------------
# intent: parsing the model's answer
# --------------------------------------------------------------------------
def test_extract_json_accepts_bare_object():
    assert _extract_json('{"feature": "biodiversity_hotspots"}') == {
        "feature": "biodiversity_hotspots"
    }


def test_extract_json_survives_code_fences():
    text = '```json\n{"feature": "migration_analysis"}\n```'
    assert _extract_json(text)["feature"] == "migration_analysis"


def test_extract_json_survives_surrounding_prose():
    text = 'Sure! Here you go:\n{"feature": "habitat_visualization"}\nHope that helps.'
    assert _extract_json(text)["feature"] == "habitat_visualization"


def test_extract_json_returns_none_when_there_is_no_json():
    assert _extract_json("I think you want a map of some kind.") is None


def test_invented_feature_is_rejected_not_repaired():
    """The router would raise on it, and a wrong skill is worse than none."""
    intent = _to_intent({"feature": "species_photo_gallery", "species_name": "wolf"})
    assert intent.feature is None
    assert not intent.is_usable


def test_null_like_strings_are_treated_as_absent():
    intent = _to_intent(
        {"feature": _HOTSPOTS, "species_name": "null", "region": "  "}
    )
    assert intent.species_name is None
    assert intent.region == "global"


@pytest.mark.asyncio
async def test_classify_intent_never_raises_when_the_backend_fails():
    intent = await classify_intent("anything", llm=_StubLLM(RuntimeError("backend down")))
    assert intent.feature is None
    assert intent.source == "error"


@pytest.mark.asyncio
async def test_classify_intent_reports_unparsable_output():
    intent = await classify_intent("anything", llm=_StubLLM("no json here"))
    assert intent.feature is None
    assert intent.source == "unparsable"


@pytest.mark.asyncio
async def test_classify_intent_reads_a_good_answer():
    reply = f'{{"feature": "{_DISTRIBUTION}", "species_name": "wolf", "region": "Europe"}}'
    intent = await classify_intent("where do wolves live?", llm=_StubLLM(reply))
    assert intent.feature == _DISTRIBUTION
    assert intent.species_name == "wolf"
    assert intent.region == "Europe"
    assert intent.is_usable


# --------------------------------------------------------------------------
# adapter: request mapping
# --------------------------------------------------------------------------
def test_context_species_beats_the_classifier():
    """The Global Orchestrator's extractor is purpose-built for this; trust it."""
    species = resolve_species(
        {"species": "Canis lupus"},
        RecognizedIntent(feature=_DISTRIBUTION, species_name="wolf"),
    )
    assert species == "Canis lupus"


def test_classifier_species_used_when_context_has_none():
    species = resolve_species({}, RecognizedIntent(feature=_DISTRIBUTION, species_name="wolf"))
    assert species == "wolf"


def test_to_orchestrator_request_fills_in_the_missing_feature():
    built = to_orchestrator_request(
        AgentRequest(instruction="where do wolves live?", context={"species": "Canis lupus"}),
        RecognizedIntent(feature=_DISTRIBUTION, species_name="wolf", region="Europe"),
    )
    assert built.feature == _DISTRIBUTION
    assert built.species_name == "Canis lupus"
    assert built.region == "Europe"
    assert built.instruction == "where do wolves live?"


# --------------------------------------------------------------------------
# adapter: result mapping
# --------------------------------------------------------------------------
def test_completed_result_publishes_biodiversity_report():
    """Image Generation branches on this key; losing it strands that agent."""
    mapped = to_platform_result(
        AgentResult(
            status=AgentStatus.COMPLETED,
            output={"points": [[1.0, 2.0]]},
            map_url="/maps/wolf.html",
            observation_count=42,
            source_agents=["Species Distribution Agent"],
        )
    )
    assert mapped.status is AgentStatus.COMPLETED
    assert "biodiversity_report" in mapped.output
    assert mapped.output["biodiversity_report"]["findings"] == {"points": [[1.0, 2.0]]}
    assert mapped.output["biodiversity_report"]["observation_count"] == 42
    assert mapped.output["map_url"] == "/maps/wolf.html"


def test_result_output_is_json_serialisable():
    import json

    mapped = to_platform_result(
        AgentResult(
            status=AgentStatus.COMPLETED,
            output={"route": "ok"},
            migration_route=[(1.0, 2.0), (3.0, 4.0)],
        )
    )
    json.dumps(mapped.output)  # raises if something unserialisable leaked in


def test_escalation_keeps_its_target_and_prompt():
    mapped = to_platform_result(
        AgentResult(
            status=AgentStatus.NEEDS_AGENT,
            target_agent="Literature Agent",
            prompt_to_target_agent="Find papers on wolf range contraction.",
            output={"partial": True},
        )
    )
    assert mapped.status is AgentStatus.NEEDS_AGENT
    assert mapped.target_agent == "Literature Agent"
    assert "wolf range" in mapped.prompt_to_target_agent


def test_failed_result_passes_through_untouched():
    original = AgentResult(status=AgentStatus.FAILED, output="no worker produced a result")
    assert to_platform_result(original) is original


# --------------------------------------------------------------------------
# adapter: the two refusals
# --------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unclassifiable_prompt_fails_with_a_useful_message(monkeypatch):
    import backend.agents.biodiversity_agent.orchestrator_adapter as adapter_module

    async def _no_feature(_prompt):
        return RecognizedIntent(feature=None, source="unparsable")

    monkeypatch.setattr(adapter_module, "classify_intent", _no_feature)

    agent = OrchestratorBiodiversityAgent(orchestrator=object())
    result = await agent.run(AgentRequest(instruction="hello there", context={}))

    assert result.status is AgentStatus.FAILED
    # Names what it *can* do rather than an internal field name.
    assert "migration" in result.output and "hotspots" in result.output


@pytest.mark.asyncio
async def test_species_scoped_feature_without_a_species_asks_for_one(monkeypatch):
    import backend.agents.biodiversity_agent.orchestrator_adapter as adapter_module

    async def _distribution(_prompt):
        return RecognizedIntent(feature=_DISTRIBUTION, species_name=None)

    monkeypatch.setattr(adapter_module, "classify_intent", _distribution)

    agent = OrchestratorBiodiversityAgent(orchestrator=object())
    result = await agent.run(AgentRequest(instruction="show me a distribution map", context={}))

    assert result.status is AgentStatus.FAILED
    assert "species" in result.output.lower()


@pytest.mark.asyncio
async def test_hotspots_runs_without_a_species(orchestrator, monkeypatch):
    """Hotspots is region-scoped - the species gate must not catch it."""
    import backend.agents.biodiversity_agent.orchestrator_adapter as adapter_module

    async def _hotspots(_prompt):
        return RecognizedIntent(feature=_HOTSPOTS, species_name=None, region="Africa")

    monkeypatch.setattr(adapter_module, "classify_intent", _hotspots)

    agent = OrchestratorBiodiversityAgent(orchestrator=orchestrator)
    result = await agent.run(
        AgentRequest(instruction="biodiversity hotspots in Africa?", context={})
    )

    assert result.status is AgentStatus.COMPLETED
    assert "biodiversity_report" in result.output


@pytest.mark.asyncio
async def test_caller_supplied_features_skip_classification(orchestrator, monkeypatch):
    """context['features'] is the orchestrator's own parallel path."""
    import backend.agents.biodiversity_agent.orchestrator_adapter as adapter_module

    async def _must_not_run(_prompt):
        raise AssertionError("classification must be skipped when features are supplied")

    monkeypatch.setattr(adapter_module, "classify_intent", _must_not_run)

    agent = OrchestratorBiodiversityAgent(orchestrator=orchestrator)
    result = await agent.run(AgentRequest(
        instruction="distribution and migration of the arctic tern",
        context={"features": [_DISTRIBUTION, BiodiversityFeature.MIGRATION_ANALYSIS.value],
                 "species": "Sterna paradisaea"},
        species_name="Sterna paradisaea",
    ))

    assert result.status in {AgentStatus.COMPLETED, AgentStatus.NEEDS_AGENT}
