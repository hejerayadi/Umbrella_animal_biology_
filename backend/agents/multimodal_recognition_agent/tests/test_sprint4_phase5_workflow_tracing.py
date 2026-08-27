"""Phase 5 (Sprint 4) - Recognition workflow and provider tracing.

Tracing is disabled by default (Phase 4's foundation), so these tests prove:

1. disabled tracing calls the metadata pipeline zero times - not "sends nothing",
   but never even reaches the point where anything could be sent;
2. enabled tracing produces exactly one record per instrumented node, each
   carrying only allowlisted, non-sensitive fields, with a duration and a
   correctly reported fallback/provider-mode/error-code where applicable;
3. whether or not tracing is enabled, the AgentResult returned to the caller is
   byte-for-byte identical - tracing observes the request, it never shapes it;
4. an exporter failure at any point cannot turn a completed request into a
   failed one.
"""
from __future__ import annotations

import json

import pytest

from .. import observability
from ..agent import RecognitionAgent
from ..config import LangSmithConfig
from ..domain.errors import ErrorCode, RecognitionError
from ..observability import RecognitionTracer
from ..schema import AgentRequest, AgentStatus
from .conftest import StubClassifier, image_entry, make_config, png_bytes, prediction

INSTRUCTION = "Identify this animal and explain the result."


def _context():
    return {"recognition_image": image_entry(png_bytes())}


@pytest.fixture
def enabled_config():
    return make_config(
        langsmith=LangSmithConfig(tracing_enabled=True, api_key="x", project="y")
    )


@pytest.fixture
def spy_events(monkeypatch):
    """Force every tracer to report enabled, and capture every record() call
    exactly as safe_metadata would sanitize it - exercising the real filter."""
    events: list[dict] = []
    monkeypatch.setattr(
        RecognitionTracer, "_build_client", staticmethod(lambda config: object())
    )
    original_record = RecognitionTracer.record

    def spy_record(self, event):
        if self.enabled:
            events.append(observability.safe_metadata(event))
        original_record(self, event)

    monkeypatch.setattr(RecognitionTracer, "record", spy_record)
    return events


# --- disabled by default: touches nothing ----------------------------------

def test_disabled_tracing_never_reaches_the_metadata_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(observability, "safe_metadata", lambda raw: calls.append(raw) or {})

    agent = RecognitionAgent(
        make_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    assert calls == []


# --- one record per instrumented node, correctly shaped --------------------

def test_an_identified_run_records_one_event_per_instrumented_node(
    enabled_config, spy_events
):
    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    nodes_seen = [event["node"] for event in spy_events]
    assert nodes_seen == ["plan", "classify", "taxonomy", "explain", "finalize"]


def test_every_event_carries_a_non_negative_duration(enabled_config, spy_events):
    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    for event in spy_events:
        assert event["duration_ms"] >= 0


def test_the_finalize_event_reports_status_and_decision(enabled_config, spy_events):
    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    finalize_event = spy_events[-1]
    assert finalize_event["status"] == "completed"
    assert finalize_event["decision"] == "identified"


def test_a_classification_failure_records_the_fixed_error_code_only(
    enabled_config, spy_events
):
    agent = RecognitionAgent(
        enabled_config,
        classifier=StubClassifier([], raises=ErrorCode.CLASSIFICATION_UNAVAILABLE),
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    classify_event = next(e for e in spy_events if e["node"] == "classify")
    assert classify_event["error_code"] == "CLASSIFICATION_UNAVAILABLE"
    # No exception body, no traceback text, nothing beyond the fixed code.
    assert "Traceback" not in json.dumps(classify_event)

    finalize_event = spy_events[-1]
    assert finalize_event["status"] == "failed"
    assert finalize_event["error_code"] == "CLASSIFICATION_UNAVAILABLE"


def test_no_candidates_is_recorded_as_not_identified_not_a_failure(
    enabled_config, spy_events
):
    agent = RecognitionAgent(enabled_config, classifier=StubClassifier([]))
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    finalize_event = spy_events[-1]
    assert finalize_event["status"] == "completed"
    assert finalize_event["decision"] == "not_identified"


def test_a_deterministic_plan_is_recorded_as_a_fallback(enabled_config, spy_events):
    """No reasoning LLM is wired in `make_config()`, so the planner always
    falls back to the deterministic plan - this is the default path, and it
    must be reported as a fallback, not silently reported as if it were llm."""
    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    plan_event = next(e for e in spy_events if e["node"] == "plan")
    assert plan_event["llm_role"] == "planner"
    assert plan_event["llm_call_count"] == 0
    assert plan_event["fallback"] is True

    explain_event = next(e for e in spy_events if e["node"] == "explain")
    assert explain_event["llm_role"] == "explainer"
    assert explain_event["llm_call_count"] == 0
    assert explain_event["fallback"] is True


def test_taxonomy_reports_its_provider_mode_and_availability(enabled_config, spy_events):
    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    taxonomy_event = next(e for e in spy_events if e["node"] == "taxonomy")
    assert taxonomy_event["taxonomy_provider_mode"] == "mock"
    assert taxonomy_event["taxonomy_available"] is True


def test_classify_reports_provider_mode_and_candidate_count(enabled_config, spy_events):
    agent = RecognitionAgent(
        enabled_config,
        classifier=StubClassifier(
            [prediction("panthera_leo", 0.95), prediction("panthera_tigris", 0.3)]
        ),
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    classify_event = next(e for e in spy_events if e["node"] == "classify")
    assert classify_event["bioclip_provider_mode"] == "mock_classification"
    assert classify_event["candidate_count"] == 2


# --- the allowlist holds even under real instrumentation --------------------

def test_no_traced_event_carries_an_instruction_prompt_or_image_field(
    enabled_config, spy_events
):
    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    for event in spy_events:
        for forbidden in ("instruction", "context", "image_bytes", "data_url",
                          "prompt", "explanation", "response"):
            assert forbidden not in event


def test_no_retry_count_is_ever_reported_in_phase_5(enabled_config, spy_events):
    """Zero retries anywhere in this agent; Phase 5 emits no retry evidence at
    all rather than a hardcoded zero that could drift from reality."""
    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    assert all("retry_count" not in event for event in spy_events)


# --- tracing never shapes the result ----------------------------------------

def test_traced_and_untraced_results_are_identical(enabled_config):
    disabled_agent = RecognitionAgent(
        make_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    enabled_agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )

    disabled_result = disabled_agent.run(
        AgentRequest(instruction=INSTRUCTION, context=_context())
    )
    enabled_result = enabled_agent.run(
        AgentRequest(instruction=INSTRUCTION, context=_context())
    )

    assert disabled_result.status == enabled_result.status
    assert disabled_result.output == enabled_result.output


def test_an_exporter_failure_cannot_turn_a_success_into_a_failure(
    enabled_config, monkeypatch
):
    monkeypatch.setattr(
        RecognitionTracer, "_build_client", staticmethod(lambda config: object())
    )

    def _boom(_raw):
        raise RuntimeError("simulated exporter outage")

    monkeypatch.setattr(observability, "safe_metadata", _boom)

    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    result = agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    assert result.status is AgentStatus.COMPLETED
    assert result.output["recognition"]["decision"] == "identified"


# --- two-call ceiling and provider behaviour are unaffected -----------------

def test_the_two_call_ceiling_still_holds_with_tracing_enabled(enabled_config, spy_events):
    agent = RecognitionAgent(
        enabled_config, classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    result = agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))

    provenance = result.output["recognition_provenance"]
    assert provenance["reasoning_llm_calls"] <= 2