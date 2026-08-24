"""The planner's follow-up slot: "answer my question, and draw it".

Such a message is two requests in one sentence, but the graph had exactly one
way in - `initial_agent` - and exactly one way to add an agent afterwards:
another agent escalating with `needs_agent`. The Image Generation Agent is
reachable by neither. No agent escalates *to* it (none of them can know a
picture was asked for - that fact lives in the user's sentence, which only the
planner reads), and it deliberately no longer escalates *from* itself, for the
reasons written at the top of its `orchestrator_logic.py`.

So the drawing half was dropped, silently, without even appearing in the
execution history. Worse on the run that exposed it: the research half FAILED
first, `failed` routes straight to the responder, and the responder - seeing a
request to draw and no drawing anywhere in context - apologised that it could
not produce images at all.

The production graph, router and worker HTTP parsing run unchanged here. Only
the LLM decisions and the remote agent services are scripted, so no provider,
network or API key is touched.
"""

from __future__ import annotations

from typing import Any

from backend.agent_card import AgentCard
from backend.orchestrator.langgraph.graph import build_orchestrator_graph
from backend.orchestrator.planner import ExecutionPlan
from backend.orchestrator.state import WorkflowState

_QUERY = "What traits let the Arctic fox survive extreme cold, and draw it"

_CARDS = {
    name: AgentCard(name, f"{name} agent", [name], [], [])
    for name in ("Trait", "Genome", "ImageGeneration")
}
_ENDPOINTS = {name: f"http://stub/{name}" for name in _CARDS}


class _Planner:
    def __init__(self, follow_up: str | None) -> None:
        self.follow_up = follow_up

    def plan(self, user_query: str, *, has_image: bool = False) -> ExecutionPlan:
        return ExecutionPlan(
            initial_agent="Trait",
            reasoning="traits question",
            follow_up_agent=self.follow_up,
        )


class _Extractor:
    def extract(self, user_query: str) -> dict[str, Any]:
        return {"species": "Vulpes lagopus"}


class _Resolver:
    def resolve(self, current_agent: str, prompt_to_target_agent: str) -> str:
        return "Genome"


class _Responder:
    """Records what the responder was told, rather than writing prose."""

    def __init__(self) -> None:
        self.failure: str | None = None
        self.context: dict[str, Any] = {}

    def answer_directly(self, user_query: str) -> str:
        return "unused"

    def synthesize(
        self,
        user_query: str,
        context: dict[str, Any],
        execution_history: list[str],
        failure: str | None = None,
    ) -> str:
        self.failure = failure
        self.context = context
        return "final answer"


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _Client:
    """Trait keeps asking for genes; the assembly has none. The real run.

    Genome answers with an assembly and an empty gene list, which is what
    GCF_018345385.1 actually returns, so Trait escalates again, the worker
    node's loop breaker fires, and Trait ends in `failed`.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def post(self, url: str, json: dict[str, Any], **kwargs: Any) -> _Response:
        agent = url.rsplit("/", 2)[-2]
        self.calls.append(agent)

        if agent == "Trait":
            return _Response(
                {
                    "status": "needs_agent",
                    "target_agent": "Genome",
                    "prompt_to_target_agent": "a validated list of candidate genes",
                }
            )
        if agent == "Genome":
            return _Response(
                {
                    "status": "completed",
                    "output": {
                        "genome_assembly": "GCF_018345385.1",
                        "annotated_genes": [],
                    },
                }
            )
        return _Response(
            {
                "status": "completed",
                "output": {
                    "image": "data:image/jpeg;base64,AAAA",
                    "traits_used": [],
                    "confidence_score": 0.35,
                },
            }
        )


def _run(follow_up: str | None) -> tuple[WorkflowState, _Responder, _Client]:
    responder = _Responder()
    client = _Client()
    graph = build_orchestrator_graph(
        planner=_Planner(follow_up),
        extractor=_Extractor(),
        resolver=_Resolver(),
        responder=responder,
        agent_cards=_CARDS,
        agent_endpoints=_ENDPOINTS,
        worker_client=client,
        sleep=lambda seconds: None,
    )
    final = graph.invoke(WorkflowState(user_query=_QUERY))
    state = final if isinstance(final, WorkflowState) else WorkflowState(**final)
    return state, responder, client


def test_the_drawing_still_happens_when_the_research_half_fails() -> None:
    # The whole point. Trait dies on an unannotated assembly; the illustration
    # needs only the species name, which the extractor already seeded, so there
    # is no reason for the user to lose it too.
    state, responder, _ = _run("ImageGeneration")

    assert "Trait -> failed" in state.execution_history
    assert "ImageGeneration -> completed" in state.execution_history
    assert state.context["image"] == "data:image/jpeg;base64,AAAA"
    assert responder.context["image"] == "data:image/jpeg;base64,AAAA"


def test_the_failure_is_still_reported_even_though_the_drawing_succeeded() -> None:
    # The follow-up runs last, so `last_result` is its success. Reporting from
    # that alone would tell the user everything worked while the genomics half
    # was on the floor.
    _, responder, _ = _run("ImageGeneration")

    assert responder.failure is not None
    assert "Trait" in responder.failure


def test_the_follow_up_runs_exactly_once() -> None:
    # Its own `completed` routes back through the same branch that dispatched
    # it. Without the worker node removing it from the pending list, that is an
    # infinite loop of FLUX calls rather than one picture.
    _, _, client = _run("ImageGeneration")

    assert client.calls.count("ImageGeneration") == 1


def test_without_a_follow_up_a_failure_goes_straight_to_the_responder() -> None:
    # The pre-existing path, unchanged: nothing scheduled, nothing extra runs.
    state, _, client = _run(None)

    assert "ImageGeneration" not in client.calls
    assert state.execution_history[-1] == "Responder -> answer ready"
    assert "image" not in state.context
