"""The live commentary the graph emits while it runs.

`/chat` answers once, at the end, which for a question routed through several
agents is minutes of silence followed by the whole story at once. The nodes now
narrate themselves through `backend/orchestrator/events.py`, and the streaming
route turns that into Server-Sent Events.

Two things have to stay true, and both are tested here:

- With nobody listening - every existing caller, `python -m backend.main`, the
  blocking `/chat` route - the graph behaves exactly as it did before.
- With a listener, every step that starts also finishes, under the same id, so
  the UI can render one line that updates rather than two that pile up.

The production graph, router and worker HTTP parsing run unchanged; only the
LLM decisions and the remote agent services are scripted.
"""

from __future__ import annotations

from typing import Any

from backend.agent_card import AgentCard
from backend.orchestrator import events
from backend.orchestrator.langgraph.graph import build_orchestrator_graph
from backend.orchestrator.planner import ExecutionPlan
from backend.orchestrator.state import WorkflowState

_QUERY = "Which genes give the polar bear its white coat?"

_CARDS = {
    name: AgentCard(name, f"{name} agent", [name], [], [])
    for name in ("Trait", "Genome", "ImageGeneration")
}
_ENDPOINTS = {name: f"http://stub/{name}" for name in _CARDS}


class _Planner:
    def plan(self, user_query: str, *, has_image: bool = False) -> ExecutionPlan:
        return ExecutionPlan(
            initial_agent="Trait",
            reasoning="The question is about a trait and the genes behind it.",
        )


class _Extractor:
    def extract(self, user_query: str) -> dict[str, Any]:
        return {"species": "Ursus maritimus"}


class _Resolver:
    def resolve(self, current_agent: str, prompt_to_target_agent: str) -> str:
        return "Genome"


class _StreamingResponder:
    """Writes its answer in pieces, the way the real one streams from Azure."""

    CHUNKS = ("The polar bear ", "owes its white coat ", "to two genes.")

    def answer_directly(self, user_query: str) -> str:
        return "unused"

    def synthesize(self, **kwargs: Any) -> str:
        return "".join(self.CHUNKS)

    def synthesize_stream(self, **kwargs: Any) -> Any:
        yield from self.CHUNKS


class _BlockingResponder:
    """No streaming methods at all - the shape every existing test double has."""

    def answer_directly(self, user_query: str) -> str:
        return "unused"

    def synthesize(self, **kwargs: Any) -> str:
        return "one whole answer"


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _Client:
    """Trait needs help, Genome supplies it, Trait finishes. Two hops."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def post(self, url: str, json: dict[str, Any], **kwargs: Any) -> _Response:
        agent = url.rsplit("/", 2)[-2]
        self.calls.append(agent)

        if agent == "Trait" and self.calls.count("Trait") == 1:
            return _Response(
                {
                    "status": "needs_agent",
                    "target_agent": "Genome",
                    "prompt_to_target_agent": "the annotated gene list for this assembly",
                }
            )
        if agent == "Genome":
            return _Response(
                {"status": "completed", "output": {"annotated_genes": ["MC1R", "ASIP"]}}
            )
        return _Response({"status": "completed", "output": {"traits": ["white coat"]}})


def _run(responder: Any) -> WorkflowState:
    graph = build_orchestrator_graph(
        planner=_Planner(),
        extractor=_Extractor(),
        resolver=_Resolver(),
        responder=responder,
        agent_cards=_CARDS,
        agent_endpoints=_ENDPOINTS,
        worker_client=_Client(),
        sleep=lambda seconds: None,
    )
    final = graph.invoke(WorkflowState(user_query=_QUERY))
    return final if isinstance(final, WorkflowState) else WorkflowState(**final)


def _collect(responder: Any) -> tuple[WorkflowState, list[dict[str, Any]]]:
    captured: list[dict[str, Any]] = []
    with events.emitting_to(captured.append):
        state = _run(responder)
    return state, captured


def test_a_run_with_nobody_listening_is_unchanged() -> None:
    # The blocking /chat route binds no emitter, so every `events.*` call in
    # every node has to be a no-op rather than an error.
    state = _run(_StreamingResponder())

    assert state.final_answer == "The polar bear owes its white coat to two genes."
    assert state.execution_history[0] == "Planner -> Trait"
    assert state.execution_history[-1] == "Responder -> answer ready"


def test_every_step_that_starts_also_finishes_under_the_same_id() -> None:
    # The UI keys on `id` and renders one line per step. A step that only ever
    # reports "running" leaves a spinner on screen for the rest of the session.
    _, captured = _collect(_StreamingResponder())
    steps = [event for event in captured if event["type"] == "step"]

    started = {step["id"] for step in steps if step["state"] == "running"}
    finished = {step["id"] for step in steps if step["state"] in ("done", "failed")}
    assert started == finished
    assert len(started) == len(steps) / 2


def test_the_steps_narrate_the_whole_route_in_order() -> None:
    _, captured = _collect(_StreamingResponder())
    done = [
        event["text"]
        for event in captured
        if event["type"] == "step" and event["state"] != "running"
    ]

    assert done == [
        "Starting with the Trait agent",
        "Found species='Ursus maritimus'",
        "The Trait agent needs help from another agent",
        "Bringing in the Genome agent",
        "The Genome agent finished",
        "The Trait agent finished",
        "Answer ready",
    ]


def test_the_planners_reasoning_reaches_the_user() -> None:
    # It was already required of the planner and already produced on every run;
    # until this existed it went to the server log and nowhere else.
    _, captured = _collect(_StreamingResponder())
    thoughts = [event["text"] for event in captured if event["type"] == "thought"]

    assert "The question is about a trait and the genes behind it." in thoughts
    # The escalating agent's request is a thought too - it is what the resolver
    # is reading when it picks the next agent.
    assert "the annotated gene list for this assembly" in thoughts


def test_the_answer_arrives_in_the_pieces_the_model_wrote_it_in() -> None:
    state, captured = _collect(_StreamingResponder())
    deltas = [event["delta"] for event in captured if event["type"] == "answer"]

    assert deltas == list(_StreamingResponder.CHUNKS)
    # Whatever the pieces, the committed answer is the join of them.
    assert "".join(deltas) == state.final_answer


def test_a_responder_that_cannot_stream_still_answers() -> None:
    # `build_orchestrator_graph` takes an injected responder, and the doubles
    # in the other tests here implement only `answer_directly`/`synthesize`.
    state, captured = _collect(_BlockingResponder())

    assert state.final_answer == "one whole answer"
    assert [event for event in captured if event["type"] == "answer"] == []
    # The step itself is still reported, so the UI knows the answer is coming.
    assert any(
        event["type"] == "step" and event["text"] == "Answer ready" for event in captured
    )


def test_agent_labels_are_spelled_for_a_human() -> None:
    # Registry keys are single words so they can double as graph node names.
    assert events.agent_label("Genome") == "Genome agent"
    assert events.agent_label("ImageGeneration") == "Image Generation agent"
