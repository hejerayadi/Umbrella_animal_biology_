"""Router: decides which graph node runs next.

The router never calls an LLM - it is pure "if/else" logic, the traffic cop
described in the project spec. It only reads what is already in the state.

`route_after_worker` compares the agent's status by its string `.value`
("completed", "needs_agent", ...) rather than by enum identity, because each
agent currently defines its own local `AgentStatus` enum class - two
different classes with the same value are not `==` to each other, so
comparing by value keeps this router decoupled from any single agent's class.
"""
from __future__ import annotations

from .state import WorkflowState


def route_after_planner(state: WorkflowState) -> str:
    """Return the next node after the planner has made its decision.

    The planner sets `current_agent` to the agent that should start, or
    leaves it as None when the message needs no research agent at all (a
    greeting, small talk, a question about the platform).
    """

    if state.current_agent is None:
        return "direct_answer"
    return state.current_agent


def route_after_worker(state: WorkflowState) -> str:
    """Return the name of the next node to run after a worker executes."""

    # Whatever the agent that just ran handed back to us.
    result = state.last_result
    if result is None:
        # This would only happen if something calls the router before any
        # agent has actually run - a bug, not a normal workflow state.
        raise ValueError("route_after_worker called before any worker produced a result")

    status = result.status.value

    # The agent said "I can't finish this myself, I need help."
    # -> Go ask the capability resolver who can help.
    if status == "needs_agent":
        return "capability_resolver"

    # The agent said "I'm not done yet, run me again."
    # -> Go straight back to the same agent.
    if status == "continue":
        assert state.current_agent is not None
        return state.current_agent

    # The agent said "something went wrong, I give up."
    # -> Stop running agents, but still go through the responder so the user
    #    gets a real explanation instead of a raw error.
    if status == "failed":
        return "responder"

    # The agent said "I'm done."
    if status == "completed":
        # If someone else was paused waiting on this agent, go back and
        # resume them now that they have what they needed. Otherwise all the
        # work is finished, so hand off to the responder to write the answer.
        return state.waiting_agent if state.waiting_agent is not None else "responder"

    # Any other status value is unexpected - fail loudly instead of guessing.
    raise ValueError(f"Unhandled agent status: {status!r}")
