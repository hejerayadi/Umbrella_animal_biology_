"""The graph node that seeds the shared context from the user's message.

Runs once, between the planner and the first worker agent, so every agent
that starts by reading `context["species"]` finds it already there. Only on
the research path - a greeting goes planner -> direct_answer and never pays
for this extra LLM call.
"""
from __future__ import annotations

from typing import Any

from ...extractor import Extractor
from ...state import WorkflowState


def make_extractor_node(extractor: Extractor):
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
                "execution_history": [*state.execution_history, "Extractor -> nothing named"],
            }

        step = ", ".join(f"{key}={value!r}" for key, value in facts.items())
        return {
            # Merge, never replace: same pattern the worker node uses when an
            # agent completes, so nothing already in context is lost.
            "context": {**state.context, **facts},
            "execution_history": [*state.execution_history, f"Extractor -> {step}"],
        }

    return _node
