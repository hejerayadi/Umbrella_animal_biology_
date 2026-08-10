"""Worker node: what an agent is told, and when to stop asking.

These reproduce the Arctic fox run that spun for 36 steps: the Trait agent
asked for a gene list, the Genome agent was called with "Draw an Arctic fox",
returned no gene list, and Trait asked again until NVIDIA rate-limited us.

No network and no LLM - the HTTP client is replaced with a stub, so these run
in milliseconds and pin the orchestrator's behaviour on its own.
"""

from __future__ import annotations

import pytest

from backend.orchestrator.langgraph.nodes import worker_node
from backend.orchestrator.state import WorkflowState


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Stands in for the pooled httpx client, recording what was sent."""

    def __init__(self, payload: dict):
        self._payload = payload
        self.sent: list[dict] = []

    def post(self, url, json=None, **kwargs):
        self.sent.append(json)
        return _FakeResponse(self._payload)


@pytest.fixture
def fake_client(monkeypatch):
    def _install(payload: dict) -> _FakeClient:
        client = _FakeClient(payload)
        monkeypatch.setattr(worker_node, "_client", client)
        return client

    return _install


_NEEDS_GENES = {
    "status": "needs_agent",
    "target_agent": "Genome",
    "prompt_to_target_agent": "Resolve a validated list of candidate genes",
    "output": None,
}


def test_agent_started_by_the_planner_gets_the_user_question(fake_client):
    client = fake_client({"status": "completed", "output": {}})

    node = worker_node.make_worker_node("ImageGeneration", "http://x")
    node(WorkflowState(user_query="Draw an Arctic fox", context={}))

    assert client.sent[0]["instruction"] == "Draw an Arctic fox"


def test_agent_fetched_as_a_dependency_gets_the_request_not_the_user_question(fake_client):
    """The bug behind the loop: Genome used to be told "Draw an Arctic fox"."""
    client = fake_client({"status": "completed", "output": {"gene_list": ["FGF5"]}})

    node = worker_node.make_worker_node("Genome", "http://x")
    node(
        WorkflowState(
            user_query="Draw an Arctic fox",
            context={"species": "Arctic fox"},
            agent_instructions={"Genome": "Resolve a validated list of candidate genes"},
        )
    )

    assert client.sent[0]["instruction"] == "Resolve a validated list of candidate genes"


def test_first_escalation_is_allowed_and_records_a_signature(fake_client):
    fake_client(_NEEDS_GENES)

    node = worker_node.make_worker_node("Trait", "http://x")
    updates = node(WorkflowState(user_query="Draw an Arctic fox", context={"species": "Arctic fox"}))

    assert updates["last_result"].status.value == "needs_agent"
    assert updates["waiting_stack"] == ["Trait"]
    assert updates["escalation_signatures"]["Trait"] == ["species"]


def test_escalating_again_with_new_context_is_still_allowed(fake_client):
    """The helper delivered something - the agent may legitimately ask again."""
    fake_client(_NEEDS_GENES)

    node = worker_node.make_worker_node("Trait", "http://x")
    updates = node(
        WorkflowState(
            user_query="Draw an Arctic fox",
            # Genome came back with an assembly, so context grew.
            context={"species": "Arctic fox", "genome": "GCF_018345385.1"},
            escalation_signatures={"Trait": ["species"]},
        )
    )

    assert updates["last_result"].status.value == "needs_agent"


def test_escalating_again_with_nothing_new_is_stopped(fake_client):
    """The exact loop: Genome answered but never supplied gene_list."""
    fake_client(_NEEDS_GENES)

    node = worker_node.make_worker_node("Trait", "http://x")
    updates = node(
        WorkflowState(
            user_query="Draw an Arctic fox",
            context={"species": "Arctic fox", "genome": "GCF_018345385.1"},
            escalation_signatures={"Trait": ["genome", "species"]},
        )
    )

    result = updates["last_result"]
    assert result.status.value == "failed"
    assert "without receiving anything new" in result.output
    # A stopped agent must not be pushed onto the waiting stack, or the graph
    # would try to resume something that already gave up.
    assert "waiting_stack" not in updates
    assert updates["execution_history"] == ["Trait -> failed"]


def test_resolver_hands_the_request_to_the_agent_it_picked():
    """The other half of the fix - the resolver must record what it resolved."""
    from backend.orchestrator.langgraph.nodes.resolver_node import make_resolver_node
    from backend.orchestrator.schema import AgentResult, AgentStatus

    class _FakeResolver:
        def resolve(self, current_agent, prompt_to_target_agent):
            return "Genome"

    node = make_resolver_node(_FakeResolver())
    updates = node(
        WorkflowState(
            user_query="Draw an Arctic fox",
            current_agent="Trait",
            last_result=AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Genome",
                prompt_to_target_agent="Resolve a validated list of candidate genes",
            ),
        )
    )

    assert updates["resolved_agent"] == "Genome"
    assert updates["agent_instructions"] == {
        "Genome": "Resolve a validated list of candidate genes"
    }


def test_the_guard_is_per_agent(fake_client):
    """One agent looping must not silence a different agent's first request."""
    fake_client(_NEEDS_GENES)

    node = worker_node.make_worker_node("ImageGeneration", "http://x")
    updates = node(
        WorkflowState(
            user_query="Draw an Arctic fox",
            context={"species": "Arctic fox"},
            escalation_signatures={"Trait": ["species"]},
        )
    )

    assert updates["last_result"].status.value == "needs_agent"
