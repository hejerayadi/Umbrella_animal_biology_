"""The two nodes that produce the text the user actually reads.

Both wrap the same `Responder` and both are terminal (their only outgoing
edge is END), which is why they live together:

- `direct_answer` - the planner decided no research agent was needed.
- `responder`     - agents ran, and their findings need writing up.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...responder import Responder
from ...state import WorkflowState


def make_direct_answer_node(
    responder: Responder,
) -> Callable[[WorkflowState], dict[str, Any]]:
    """Build the node that replies conversationally, with no agents involved.

    Reached when the planner decides the message is a greeting, small talk,
    or a question about the platform rather than a research request.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        answer = responder.answer_directly(state.user_query)
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

        answer = responder.synthesize(
            user_query=state.user_query,
            context=state.context,
            execution_history=state.execution_history,
            failure=failure,
        )
        return {
            "final_answer": answer,
            "execution_history": [
                *state.execution_history,
                "Responder -> answer ready",
            ],
        }

    return _node
