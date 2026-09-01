"""The graph node that runs the Capability Resolver.

Only ever reached right after a worker returned `needs_agent`, so this node
can assume `state.last_result` describes exactly what is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ....registry import resolve_agent_name
from ...capability_resolver import CapabilityResolver
from ...state import WorkflowState


def make_resolver_node(
    resolver: CapabilityResolver,
) -> Callable[[WorkflowState], dict[str, Any]]:
    """Build the graph node that runs the Capability Resolver.

    This node only ever runs right after a worker said `needs_agent`, so we
    know `state.last_result` describes exactly what's missing.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        result = state.last_result
        # These are sanity checks confirming the graph reached this node the
        # way it's supposed to (only after a "needs_agent" result).
        assert result is not None and result.status.value == "needs_agent"
        assert state.current_agent is not None

        request = result.prompt_to_target_agent or state.user_query

        # A worker that pauses usually already knows who it needs, and says so
        # in `target_agent`. Take that answer when it names a real agent.
        #
        # Until this check existed, `target_agent` was read by nobody: the
        # resolver ran unconditionally and re-derived the same choice from the
        # prose in `prompt_to_target_agent`, at the cost of an LLM call. The
        # Genome Agent's handoff worked only because the second answer happened
        # to agree with the first - its stated target, "Reconstruction Agent",
        # was not even a key in this catalog, so nothing would have noticed if
        # they had disagreed. `resolve_agent_name` is what accepts that
        # spelling; see registry.py for why there are two.
        declared = resolve_agent_name(result.target_agent)

        if declared is not None and declared != state.current_agent:
            target = declared
            step = f"Resolver -> {target} (declared by {state.current_agent})"
        else:
            # Either the worker named nobody, named something unknown, or named
            # itself. Ask the resolver: "agent X is stuck on this - who can
            # help?" A self-nomination is a loop, so it is treated as no answer.
            target = resolver.resolve(
                current_agent=state.current_agent,
                prompt_to_target_agent=request,
            )
            step = f"Resolver -> {target}"

        updates: dict[str, Any] = {
            "resolved_agent": target,
            "execution_history": [*state.execution_history, step],
        }

        # Hand the request itself to the agent that was picked. Choosing the
        # right agent is only half the job: until this was recorded, the
        # request was used to select `target` and then dropped, so `target`
        # was called with the user's original question and had no idea what
        # the waiting agent actually needed from it.
        updates["agent_instructions"] = {**state.agent_instructions, target: request}

        return updates

    return _node
