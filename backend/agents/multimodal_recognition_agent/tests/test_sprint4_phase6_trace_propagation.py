"""Phase 6 (Sprint 4) - trace continuity with the Global Orchestrator.

Scope, per the plan: Recognition-owned changes only. No Orchestrator file is
touched or imported here - `worker_node.py` already sends `X-Trace-Id` on
every call (confirmed by direct inspection); these tests prove the receiving
half, entirely within this agent's own boundary.

Three properties matter:

1. a trace id supplied at any entry point (API header, or a direct call with
   `trace_id=`) reaches the one `finalize` event Phase 5 already emits, and
   nowhere else;
2. its absence changes nothing - no new field, no different behaviour,
   identical AgentResult to before this phase existed;
3. it is never validated, parsed or trusted - an empty, malformed, or
   suspicious value is either passed through verbatim or dropped, never
   allowed to affect the response.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from .. import observability
from ..agent import RecognitionAgent
from ..api import app
from ..config import LangSmithConfig
from ..observability import RecognitionTracer
from ..schema import AgentRequest
from .conftest import StubClassifier, image_entry, make_config, png_bytes, prediction

INSTRUCTION = "Identify this animal and explain the result."
client = TestClient(app)


def _context():
    return {"recognition_image": image_entry(png_bytes())}


def _enabled_config():
    return make_config(
        langsmith=LangSmithConfig(tracing_enabled=True, api_key="x", project="y")
    )


def _spy(monkeypatch):
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


# --- the trace id reaches exactly the finalize event -----------------------

def test_a_trace_id_passed_directly_reaches_only_the_finalize_event(monkeypatch):
    events = _spy(monkeypatch)
    agent = RecognitionAgent(
        _enabled_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )

    agent.run(
        AgentRequest(instruction=INSTRUCTION, context=_context()),
        trace_id="trace-abc-123",
    )

    carrying_events = [e for e in events if "trace_id" in e]
    assert len(carrying_events) == 1
    assert carrying_events[0]["node"] == "finalize"
    assert carrying_events[0]["trace_id"] == "trace-abc-123"


def test_the_api_header_reaches_the_same_finalize_event(monkeypatch):
    events = _spy(monkeypatch)
    monkeypatch.setattr("backend.agents.multimodal_recognition_agent.api._agent",
                         RecognitionAgent(_enabled_config(),
                                          classifier=StubClassifier([prediction("panthera_leo", 0.95)])))

    response = client.post(
        "/execute",
        json={"instruction": INSTRUCTION, "context": _context()},
        headers={"X-Trace-Id": "trace-from-orchestrator"},
    )

    assert response.status_code == 200
    carrying_events = [e for e in events if "trace_id" in e]
    assert len(carrying_events) == 1
    assert carrying_events[0]["trace_id"] == "trace-from-orchestrator"


# --- absence changes nothing ------------------------------------------------

def test_no_header_means_no_trace_id_field_at_all(monkeypatch):
    events = _spy(monkeypatch)
    monkeypatch.setattr("backend.agents.multimodal_recognition_agent.api._agent",
                         RecognitionAgent(_enabled_config(),
                                          classifier=StubClassifier([prediction("panthera_leo", 0.95)])))

    client.post("/execute", json={"instruction": INSTRUCTION, "context": _context()})

    assert all("trace_id" not in e for e in events)


def test_the_agent_result_is_identical_with_and_without_a_trace_id():
    agent = RecognitionAgent(
        make_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    context = _context()

    without = agent.run(AgentRequest(instruction=INSTRUCTION, context=context))
    with_id = agent.run(
        AgentRequest(instruction=INSTRUCTION, context=context), trace_id="anything"
    )

    assert without.status == with_id.status
    assert without.output == with_id.output


def test_existing_call_sites_that_omit_trace_id_are_unaffected():
    """Smoke scripts and direct RecognitionAgent() use call .run(request) with
    no trace_id argument at all - this must keep working exactly as before."""
    agent = RecognitionAgent(
        make_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    result = agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()))
    assert result.status.value == "completed"


# --- never trusted, never validated -----------------------------------------

def test_an_empty_string_trace_id_is_treated_as_absent(monkeypatch):
    events = _spy(monkeypatch)
    agent = RecognitionAgent(
        _enabled_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )

    agent.run(AgentRequest(instruction=INSTRUCTION, context=_context()), trace_id="")

    assert all("trace_id" not in e for e in events)


def test_an_arbitrary_or_malformed_trace_id_never_raises(monkeypatch):
    events = _spy(monkeypatch)
    agent = RecognitionAgent(
        _enabled_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )

    weird = "not-a-uuid; DROP TABLE traces; <script>alert(1)</script>"
    result = agent.run(
        AgentRequest(instruction=INSTRUCTION, context=_context()), trace_id=weird
    )

    assert result.status.value == "completed"
    finalize_event = next(e for e in events if "trace_id" in e)
    assert finalize_event["trace_id"] == weird  # echoed verbatim, never parsed


def test_a_trace_id_can_never_appear_outside_the_allowlist(monkeypatch):
    events = _spy(monkeypatch)
    agent = RecognitionAgent(
        _enabled_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )

    agent.run(
        AgentRequest(instruction=INSTRUCTION, context=_context()), trace_id="trace-xyz"
    )

    for event in events:
        if event["node"] != "finalize":
            assert "trace_id" not in event


def test_disabled_tracing_ignores_the_trace_id_entirely(monkeypatch):
    """Disabled is still disabled: a trace_id argument must not switch
    anything on, or reach any pipeline, when tracing itself is off."""
    calls = []
    monkeypatch.setattr(observability, "safe_metadata", lambda raw: calls.append(raw) or {})

    agent = RecognitionAgent(
        make_config(), classifier=StubClassifier([prediction("panthera_leo", 0.95)])
    )
    agent.run(
        AgentRequest(instruction=INSTRUCTION, context=_context()), trace_id="trace-xyz"
    )

    assert calls == []