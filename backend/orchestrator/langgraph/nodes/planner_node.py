"""The graph node that runs the Planner.

This is the first node of every run: it decides whether the message needs a
research agent at all, and if so which agent starts the work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...planner import Planner
from ...state import WorkflowState


def make_planner_node(planner: Planner) -> Callable[[WorkflowState], dict[str, Any]]:
    """Build the graph node that runs the Planner.

    This is a "node factory": it takes the planner object once and hands
    back a small function (`_node`) that LangGraph will call every time this
    step in the graph runs. That inner function is the actual node.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        # Ask the planner: "does this even need an agent, and if so, who
        # should go first?" The image flag travels separately from the query
        # because the planner reads text and the attachment is not in it.
        plan = planner.plan(state.user_query, has_image=state.has_image)

        # `initial_agent` is None when the message needs no research agent -
        # the router sends those straight to the direct-answer node.
        step = plan.initial_agent or "Direct answer"

        # LangGraph nodes don't mutate the state directly - they return a
        # dict of "here's what changed", and LangGraph merges it in.
        # A follow-up agent is scheduled here and dispatched by the router once
        # the initial agent's line of work ends - see `route_after_worker`.
        follow_ups = [plan.follow_up_agent] if plan.follow_up_agent else []
        if follow_ups:
            step = f"{step} then {plan.follow_up_agent}"

        return {
            "current_agent": plan.initial_agent,
            "follow_up_agents": follow_ups,
            "execution_history": [*state.execution_history, f"Planner -> {step}"],
        }

    return _node
