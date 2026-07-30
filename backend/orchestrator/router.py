"""Router: interprets an AgentResult and decides the next graph node.

The router never calls an LLM; it only reads `state.last_result.status`. It
compares that status by its string `.value` ("completed", "needs_agent", ...)
rather than by enum identity, because each agent currently defines its own
local `AgentStatus` enum class - two different classes with the same value
are not `==` to each other, so comparing by value keeps this router decoupled
from any single agent's class. For resuming a paused caller it also reads
`state.waiting_agent`, which the worker node has already updated.
"""
from __future__ import annotations

from langgraph.graph import END

from .state import WorkflowState


def route_after_worker(state: WorkflowState) -> str:
    """Return the name of the next node to run after a worker executes.

    This is plain "if/else" logic, no AI involved - it's the traffic cop
    described in the project spec: it just reads the last agent's status and
    picks where to go next.
    """

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
    # -> Stop the whole workflow.
    if status == "failed":
        return END

    # The agent said "I'm done."
    if status == "completed":
        # If someone else was paused waiting on this agent, go back and
        # resume them now that they have what they needed. Otherwise, there's
        # nobody left waiting, so the whole workflow is finished.
        return state.waiting_agent if state.waiting_agent is not None else END

    # Any other status value is unexpected - fail loudly instead of guessing.
    raise ValueError(f"Unhandled agent status: {status!r}")
