"""The graph node that seeds the shared context from the user's message.

Runs once, between the planner and the first worker agent, so every agent
that starts by reading `context["species"]` finds it already there. Only on
the research path - a greeting goes planner -> direct_answer and never pays
for this extra LLM call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...extractor import Extractor
from ...state import WorkflowState


# Every execution-history entry is rendered into the Responder's prompt in
# full - unlike a context finding, which that module caps. A pasted nucleotide
# sequence is the one extracted value that can be arbitrarily long, so it is
# abbreviated here rather than spending thousands of prompt tokens restating
# what is already in `context`.
_MAX_HISTORY_CHARS = 60


def _abbreviate(value: Any) -> str:
    text = repr(value)
    if len(text) <= _MAX_HISTORY_CHARS:
        return text
    return f"{text[:_MAX_HISTORY_CHARS]}... ({len(text)} chars)"


def make_extractor_node(
    extractor: Extractor,
) -> Callable[[WorkflowState], dict[str, Any]]:
    """Build the graph node that runs the Extractor.

    Same "node factory" shape as the planner node: takes the extractor object
    once, hands back the small function LangGraph calls each run.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        facts = extractor.extract(state.user_query)

        if not facts:
            # Nothing named in the message. Leave context untouched rather
            # than writing empty keys - agents check for a key's presence.
            return {
                "execution_history": [
                    *state.execution_history,
                    "Extractor -> nothing named",
                ],
            }

        step = ", ".join(f"{key}={_abbreviate(value)}" for key, value in facts.items())
        return {
            # Merge, never replace: same pattern the worker node uses when an
            # agent completes, so nothing already in context is lost.
            "context": {**state.context, **facts},
            "execution_history": [*state.execution_history, f"Extractor -> {step}"],
        }

    return _node
