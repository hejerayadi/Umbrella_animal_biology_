"""The graph node that runs one worker agent.

One node is built per entry in `AGENT_REGISTRY`, and this is the only place
in the whole orchestrator where a worker agent's own code actually runs.

Requests are built with `SimpleNamespace` rather than a shared `AgentRequest`
class: every mock agent only ever reads `.instruction` / `.context` off the
object it receives (duck typing), so no dependency on any one agent's local
schema is needed to call it.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from ....registry import WorkerAgent
from ...state import WorkflowState

# Every worker agent's status funnels through this one logger, so watching
# the console shows the orchestrator's whole train of thought in order.
_logger = logging.getLogger(__name__)


def make_worker_node(agent_name: str, agent: WorkerAgent):
    """Build the graph node for one specific worker agent (e.g. "Genome").

    `agent_name` and `agent` are captured here once when the graph is built,
    so every time this node runs later, it already knows which agent it is
    and which real object to call.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        # Build the request object the agent expects. We use a plain
        # SimpleNamespace (just an object with attributes) instead of a
        # shared class, since the mock agents only ever read
        # `.instruction` and `.context` off of it.
        request = SimpleNamespace(instruction=state.user_query, context=state.context)

        # Actually call the agent. This is the one place in the whole
        # orchestrator where a worker agent's code runs.
        result = agent.run(request)
        status = result.status.value

        if status == "needs_agent":
            _logger.info("[%s] needs_agent -> %r", agent_name, result.prompt_to_target_agent)
        elif status == "failed":
            _logger.info("[%s] failed -> %r", agent_name, result.output)
        else:
            _logger.info("[%s] %s -> %r", agent_name, status, result.output)

        # Start building the state updates every worker produces, no matter
        # its status: which agent just ran, what it returned, and a log entry.
        updates: dict[str, Any] = {
            "current_agent": agent_name,
            "last_result": result,
            "execution_history": [*state.execution_history, f"{agent_name} -> {status}"],
        }

        if status == "needs_agent":
            # This agent can't finish without help. Remember that it's
            # paused and waiting, so we can come back to it later.
            updates["waiting_stack"] = [*state.waiting_stack, agent_name]
            updates["waiting_agent"] = agent_name

        elif status == "completed":
            # Merge whatever this agent produced into the shared context, so
            # every later agent can see it too (e.g. {"genome": "..."}).
            if isinstance(result.output, dict):
                updates["context"] = {**state.context, **result.output}

            if state.waiting_stack:
                # Resume whoever was waiting on this agent - that's the top
                # of the stack, not whatever remains after popping it off.
                updates["waiting_agent"] = state.waiting_stack[-1]
                updates["waiting_stack"] = state.waiting_stack[:-1]
            else:
                # Nobody is waiting on this agent, so there's nothing left
                # to resume - the whole workflow can finish here.
                updates["waiting_agent"] = None
                updates["waiting_stack"] = []

        # "continue" and "failed" don't need any extra bookkeeping beyond
        # the basic updates above - the router (in router.py) decides what
        # to do with those statuses.

        return updates

    return _node
