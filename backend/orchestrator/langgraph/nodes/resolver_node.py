"""The graph node that runs the Capability Resolver.

Only ever reached right after a worker returned `needs_agent`, so this node
can assume `state.last_result` describes exactly what is missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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

        # Ask the resolver: "agent X is stuck on this - who can help?"
        request = result.prompt_to_target_agent or state.user_query
        target = resolver.resolve(
            current_agent=state.current_agent,
            prompt_to_target_agent=request,
        )

        updates: dict[str, Any] = {
            "resolved_agent": target,
            "execution_history": [*state.execution_history, f"Resolver -> {target}"],
        }

        # Hand the request itself to the agent that was picked. Choosing the
        # right agent is only half the job: until this was recorded, the
        # request was used to select `target` and then dropped, so `target`
        # was called with the user's original question and had no idea what
        # the waiting agent actually needed from it.
        updates["agent_instructions"] = {**state.agent_instructions, target: request}

        return updates

    return _node
