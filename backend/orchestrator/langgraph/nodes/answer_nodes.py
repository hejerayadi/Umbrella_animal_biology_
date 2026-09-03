"""The two nodes that produce the text the user actually reads.

Both wrap the same `Responder` and both are terminal (their only outgoing
edge is END), which is why they live together:

- `direct_answer` - the planner decided no research agent was needed.
- `responder`     - agents ran, and their findings need writing up.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ... import events
from ...responder import Responder
from ...state import WorkflowState


def _write(responder: Responder, stream_method: str, fallback_method: str, **kwargs: Any) -> str:
    """Produce the final answer, streaming it out to any listener as it comes.

    Falls back to the plain blocking call when the responder has no streaming
    method. That is not defensive padding: `build_orchestrator_graph` accepts
    an injected responder, and the test doubles that use it implement only
    `answer_directly` and `synthesize`. Nothing listens to events in those
    tests either, so both halves of this stay honest.
    """

    stream = getattr(responder, stream_method, None)
    if stream is None:
        return str(getattr(responder, fallback_method)(**kwargs))

    chunks: list[str] = []
    for chunk in stream(**kwargs):
        chunks.append(chunk)
        events.answer_delta(chunk)
    return "".join(chunks)


def make_direct_answer_node(
    responder: Responder,
) -> Callable[[WorkflowState], dict[str, Any]]:
    """Build the node that replies conversationally, with no agents involved.

    Reached when the planner decides the message is a greeting, small talk,
    or a question about the platform rather than a research request.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        step_id = events.step_started("Responder", "Replying")
        answer = _write(
            responder,
            "answer_directly_stream",
            "answer_directly",
            user_query=state.user_query,
        )
        events.step_finished(step_id, "Responder", "Replied directly")
        return {
            "final_answer": answer,
            "execution_history": [
                *state.execution_history,
                "Responder -> answered directly",
            ],
        }

    return _node


def make_responder_node(
    responder: Responder,
) -> Callable[[WorkflowState], dict[str, Any]]:
    """Build the node that writes the final answer once the agents are done.

    Every research path ends here, including failed ones - so the user always
    gets a written explanation rather than a raw dictionary or error.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        # Every failure from the run, so the responder can explain honestly
        # instead of inventing an answer.
        #
        # Read from `state.failures` rather than `last_result`: a
        # planner-scheduled follow-up runs after a failure, so on a request
        # like "explain X and draw it" the drawing succeeds last and
        # `last_result` no longer carries the failure that the user still
        # needs to be told about.
        failures = list(state.failures)

        # `last_result` still contributes, for the runs that never recorded a
        # failure into state (a stub worker in a test, an older state object
        # restored from elsewhere).
        result = state.last_result
        if result is not None and result.status.value == "failed":
            text = str(result.output)
            if not any(text in recorded for recorded in failures):
                failures.append(text)

        failure = "; ".join(failures) if failures else None

        step_id = events.step_started(
            "Responder", "Writing the answer from what the agents found"
        )
        answer = _write(
            responder,
            "synthesize_stream",
            "synthesize",
            user_query=state.user_query,
            context=state.context,
            execution_history=state.execution_history,
            failure=failure,
        )
        events.step_finished(step_id, "Responder", "Answer ready")
        return {
            "final_answer": answer,
            "execution_history": [
                *state.execution_history,
                "Responder -> answer ready",
            ],
        }

    return _node
