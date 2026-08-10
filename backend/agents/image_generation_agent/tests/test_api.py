"""The HTTP contract the orchestrator's worker_node depends on.

`backend/orchestrator/langgraph/nodes/worker_node.py` POSTs {instruction, context}
to /execute and reads four keys back: status, target_agent, prompt_to_target_agent
and output. These tests pin that shape, because a change here breaks the
orchestrator silently - the graph would just see FAILED and route to the responder.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.agents.image_generation_agent.api import app

client = TestClient(app)


def _execute(instruction: str, context: dict) -> dict:
    """Exactly the request worker_node._call_agent builds."""
    response = client.post(
        "/execute", json={"instruction": instruction, "context": context}
    )
    assert response.status_code == 200
    return response.json()


def test_response_carries_every_key_worker_node_reads():
    payload = _execute("Draw a woolly mammoth", {"species": "Mammuthus primigenius"})
    assert set(payload) >= {
        "status",
        "target_agent",
        "prompt_to_target_agent",
        "output",
    }


def test_species_without_traits_draws_over_http(monkeypatch):
    """The end-to-end shape of "Draw an Arctic fox": one hop, no escalation."""
    monkeypatch.setattr(
        "backend.agents.image_generation_agent.orchestrator_logic.FluxClient",
        lambda: type(
            "FakeFlux", (), {"generate_image": lambda self, prompt: "https://x/fox.jpg"}
        )(),
    )

    payload = _execute("Draw an Arctic fox", {"species": "Vulpes lagopus"})

    assert payload["status"] == "completed"
    assert payload["target_agent"] is None
    assert payload["output"]["image"] == "https://x/fox.jpg"


def test_no_subject_fails_without_raising_http_500():
    """A failure must come back as a 200 + FAILED body, never a 500.

    worker_node treats any HTTP error as "agent unreachable", which would hide
    the real reason from the user.
    """
    payload = _execute("Draw something", {})

    assert payload["status"] == "failed"
    assert "Missing subject to illustrate" in payload["output"]


def test_completed_output_matches_card_json(monkeypatch):
    """The keys promised by card.json are what get merged into shared context."""
    monkeypatch.setattr(
        "backend.agents.image_generation_agent.orchestrator_logic.FluxClient",
        lambda: type(
            "FakeFlux", (), {"generate_image": lambda self, prompt: "https://x/i.jpg"}
        )(),
    )

    payload = _execute(
        "Draw a woolly mammoth",
        {"species": "Mammuthus primigenius", "traits": ["Long curved tusks"]},
    )

    assert payload["status"] == "completed"
    assert set(payload["output"]) == {"image", "traits_used", "confidence_score"}
    assert payload["output"]["image"] == "https://x/i.jpg"


def test_unhandled_exception_still_answers_with_the_agent_schema(monkeypatch):
    """api.py's except-branch: the orchestrator must never get a bare 500."""

    def boom(request):
        raise RuntimeError("flux exploded")

    monkeypatch.setattr(
        "backend.agents.image_generation_agent.api._logic.run", boom, raising=True
    )

    payload = _execute("Draw a mammoth", {"species": "Mammuthus primigenius"})

    assert payload["status"] == "failed"
    assert "flux exploded" in payload["output"]
