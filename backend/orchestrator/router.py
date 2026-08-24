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

    The research path goes through the extractor first, so the agent the
    planner chose finds the subject of the question already in `context`.
    Non-research messages skip it - a greeting names no species, and the
    extra LLM call would buy nothing.
    """

    if state.current_agent is None:
        return "direct_answer"
    return "extractor"


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
    # -> Stop the research line, but run any follow-up the planner scheduled
    #    before handing over to the responder.
    #
    # A failure does NOT cancel the follow-up. The two halves of "what traits
    # let the Arctic fox survive the cold, and draw it" are independent
    # requests: the genomics half needs an annotated assembly, the drawing
    # needs only a species name. Skipping the drawing because the genomics
    # failed is what made the whole request come back as an apology.
    if status == "failed":
        return _next_follow_up(state) or "responder"

    # The agent said "I'm done."
    if status == "completed":
        # If someone else was paused waiting on this agent, go back and
        # resume them now that they have what they needed.
        if state.waiting_agent is not None:
            return state.waiting_agent
        # Nobody is waiting, so the research line is finished. Run the
        # follow-up if there is one, otherwise write the answer.
        return _next_follow_up(state) or "responder"

    # Any other status value is unexpected - fail loudly instead of guessing.
    raise ValueError(f"Unhandled agent status: {status!r}")


def _next_follow_up(state: WorkflowState) -> str | None:
    """The next planner-scheduled agent that has not run yet, if any.

    `worker_node` removes each entry as it dispatches it, so this returns None
    once the follow-up has run and the graph moves on to the responder. That
    bookkeeping is what stops a follow-up whose own result routes back here
    from being dispatched forever.
    """

    return state.follow_up_agents[0] if state.follow_up_agents else None
