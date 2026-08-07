"""The graph node that calls one worker agent over HTTP.

One node is built per entry in `AGENT_ENDPOINTS`. Agents are independent
services: this node POSTs to `<agent>/execute` and parses the reply back into
the orchestrator's own `AgentResult`, so nothing downstream (`router.py`,
`resolver_node.py`, `answer_nodes.py`) needs to know the call left the
process.

The request body matches every agent's `AgentRequest`: `instruction` is the
user's original question, `context` is everything the agents have produced so
far.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from ...schema import AgentResult, AgentStatus
from ...state import WorkflowState

# Every worker agent's status funnels through this one logger, so watching
# the console shows the orchestrator's whole train of thought in order.
_logger = logging.getLogger(__name__)

# Connect fast (a missing agent should fail immediately, not hang the graph),
# but allow a slow agent plenty of time to actually do its work.
_TIMEOUT = httpx.Timeout(120.0, connect=5.0)

# One pooled client for the whole process, reused across every agent call.
_client = httpx.Client(timeout=_TIMEOUT)


def _call_agent(agent_name: str, base_url: str, state: WorkflowState) -> AgentResult:
    """POST to one agent and parse its reply. All transport concerns live here."""

    try:
        response = _client.post(
            f"{base_url}/execute",
            json={"instruction": state.user_query, "context": state.context},
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        # Agent not running, unreachable, timed out, or returned 5xx. None of
        # these exist for an in-process call, and none of them should crash
        # the graph - the router already knows how to handle FAILED.
        _logger.info("[%s] unreachable -> %s", agent_name, exc)
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"{agent_name} agent unreachable at {base_url}: {exc}",
        )

    try:
        status = AgentStatus(payload["status"])
    except (KeyError, ValueError) as exc:
        # The agent answered, but not with a status this orchestrator knows.
        _logger.info("[%s] unusable response -> %r", agent_name, payload)
        return AgentResult(
            status=AgentStatus.FAILED,
            output=f"{agent_name} agent returned an unusable response ({exc}): {payload!r}",
        )

    return AgentResult(
        status=status,
        target_agent=payload.get("target_agent"),
        prompt_to_target_agent=payload.get("prompt_to_target_agent"),
        output=payload.get("output"),
    )


def make_worker_node(agent_name: str, base_url: str):
    """Build the graph node for one specific worker agent (e.g. "Genome").

    `agent_name` and `base_url` are captured here once when the graph is
    built, so every time this node runs later it already knows which agent it
    is and where to reach it.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        # The one place in the whole orchestrator that talks to an agent.
        result = _call_agent(agent_name, base_url, state)
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
