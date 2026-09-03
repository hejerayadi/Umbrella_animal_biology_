"""The graph node that calls one worker agent over HTTP.

One node is built per entry in `AGENT_ENDPOINTS`. Agents are independent
services: this node POSTs to `<agent>/execute` and parses the reply back into
the orchestrator's own `AgentResult`, so nothing downstream (`router.py`,
`resolver_node.py`, `answer_nodes.py`) needs to know the call left the
process.

The request body matches every agent's `AgentRequest`: `instruction` is the
user's original question, `context` is everything the agents have produced so
far.

One exception to "context is sent as-is": an attached image travels as a short
id, and only the agent that can actually look at it is given the bytes. See
`_context_for` below.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import httpx

from ....image_store import IMAGE_STORE
from ... import events
from ...schema import AgentResult, AgentStatus
from ...state import WorkflowState

# The context key holding the id of an image the user attached, and the agent
# allowed to receive the image itself.
IMAGE_ID_CONTEXT_KEY = "recognition_image_id"
_IMAGE_CONSUMER = "Multimodal"

# The key the Recognition agent reads the image from. Its own `validation.py`
# reads this one key and no other, deliberately, so the name has to match.
_RECOGNITION_IMAGE_KEY = "recognition_image"

# Every worker agent's status funnels through this one logger, so watching
# the console shows the orchestrator's whole train of thought in order.
_logger = logging.getLogger(__name__)

# Connect fast (a missing agent should fail immediately, not hang the graph),
# but allow a slow agent plenty of time to actually do its work.
#
# 120s was not "plenty" for the agents that call an LLM per item rather than
# once per request. Measured: the Trait Discovery Agent takes ~300s to resolve
# a single gene against a rate-limited NVIDIA NIM key, because Gene Mapper,
# Pathways, Protein Data, Literature Support and the explanation writer each
# make their own call and a free-tier 429 costs 10-60s of backoff apiece. It
# never returned inside the old limit, so the orchestrator reported it as
# unreachable and the Responder told the user the agent was down - a transport
# error standing in for work that was still running and would have succeeded.
#
# The read timeout is the one that moves. `connect` stays low so an agent that
# is genuinely not running still fails in seconds, which is what stops a
# missing service from hanging the graph.
_READ_TIMEOUT_SECONDS = float(os.getenv("AGENT_READ_TIMEOUT_SECONDS", "600"))
_TIMEOUT = httpx.Timeout(_READ_TIMEOUT_SECONDS, connect=5.0)

# One pooled client for the whole process, reused across every agent call.
_client = httpx.Client(timeout=_TIMEOUT)

# Three retries after the initial call. The values are also the delays before
# each retry, making the policy deterministic and directly testable.
CONTINUE_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0)


def _context_for(agent_name: str, context: dict[str, Any]) -> dict[str, Any]:
    """The context to send to one agent, resolving or stripping the image.

    The id is swapped for the real bytes for the Recognition agent, and removed
    entirely for everyone else. Broadcasting a multi-megabyte data URL to nine
    agents would be wasteful; the reason it is actively harmful is that the
    Responder renders every context key into an LLM prompt, so a stray image
    would arrive as hundreds of thousands of tokens.
    """

    image_id = context.get(IMAGE_ID_CONTEXT_KEY)
    if not image_id:
        return context

    trimmed = {
        key: value for key, value in context.items() if key != IMAGE_ID_CONTEXT_KEY
    }
    if agent_name != _IMAGE_CONSUMER:
        return trimmed

    stored = IMAGE_STORE.get(str(image_id))
    if stored is None:
        # Evicted, or a stale id from an old browser tab. Send the request
        # without it: the agent answers MISSING_IMAGE, which is a clearer
        # result than a transport error here would be.
        _logger.info("[%s] image id %r is unknown or expired", agent_name, image_id)
        return trimmed

    return {
        **trimmed,
        _RECOGNITION_IMAGE_KEY: {
            "data_url": stored.as_data_url(),
            "filename": stored.filename,
        },
    }


# An agent's own words are shown to the user under the step that produced
# them. A full result dict is not prose - it can be a whole gene table - so
# only the opening of it is worth showing, and the answer itself follows
# minutes later anyway.
_MAX_THOUGHT_CHARS = 240


def _summarise(value: Any) -> str:
    """One short line of an agent's output, for display under its step."""

    text = " ".join(str(value or "").split())
    if len(text) <= _MAX_THOUGHT_CHARS:
        return text
    return f"{text[:_MAX_THOUGHT_CHARS]}…"


def _call_agent(
    agent_name: str,
    base_url: str,
    state: WorkflowState,
    client: Any | None = None,
) -> AgentResult:
    """POST to one agent and parse its reply. All transport concerns live here."""

    # An agent brought in to satisfy another agent's `needs_agent` is sent that
    # request; the agent the planner started with has no entry and is sent the
    # user's question. See `WorkflowState.agent_instructions`.
    instruction = state.agent_instructions.get(agent_name) or state.user_query
    request_id = str(uuid4())

    try:
        response = (client or _client).post(
            f"{base_url}/execute",
            headers={
                "X-Trace-Id": state.trace_id,
                "X-Request-Id": request_id,
            },
            json={
                "instruction": instruction,
                "context": _context_for(agent_name, state.context),
            },
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
        continuation_reason=payload.get("continuation_reason"),
        # Legacy agents only returned the status. Treat their CONTINUE as
        # retryable, while allowing newer agents to explicitly opt out.
        retryable=bool(payload.get("retryable", status is AgentStatus.CONTINUE)),
        error=payload.get("error"),
    )


def make_worker_node(
    agent_name: str,
    base_url: str,
    *,
    client: Any | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retry_delays: tuple[float, ...] = CONTINUE_RETRY_DELAYS,
) -> Callable[[WorkflowState], dict[str, Any]]:
    """Build the graph node for one specific worker agent (e.g. "Genome").

    `agent_name` and `base_url` are captured here once when the graph is
    built, so every time this node runs later it already knows which agent it
    is and where to reach it.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        label = events.agent_label(agent_name)
        retries_used = state.continue_retry_counts.get(agent_name, 0)
        if retries_used:
            delay = retry_delays[retries_used - 1]
            _logger.info(
                "[%s] retry %d/%d after %.1fs",
                agent_name,
                retries_used,
                len(retry_delays),
                delay,
            )
            # Emitted before the sleep, not after: the whole point is that the
            # user can see why nothing is happening for the next few seconds.
            step_id = events.step_started(
                agent_name,
                f"Waiting {delay:.0f}s, then asking the {label} again "
                f"(retry {retries_used} of {len(retry_delays)})",
            )
            sleep(delay)
        else:
            step_id = events.step_started(agent_name, f"Asking the {label}")

        # The one place in the whole orchestrator that talks to an agent.
        result = _call_agent(agent_name, base_url, state, client)
        status = result.status.value

        if status == "continue" and not result.retryable:
            reason = result.continuation_reason or str(
                result.output or "no reason supplied"
            )
            result = AgentResult(
                status=AgentStatus.FAILED,
                output=f"{agent_name} cannot continue automatically: {reason}",
                error=reason,
            )
            status = "failed"
        elif status == "continue" and retries_used >= len(retry_delays):
            reason = result.continuation_reason or str(
                result.output or "no reason supplied"
            )
            result = AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    f"{agent_name} remained incomplete after {len(retry_delays)} retries. "
                    f"Last reason: {reason}"
                ),
                error=reason,
            )
            status = "failed"

        # The context keys present right now. Compared against the snapshot
        # taken the last time this agent escalated, this answers "did the
        # helper we fetched actually bring anything back?".
        signature = sorted(state.context)
        looping = (
            status == "needs_agent"
            and state.escalation_signatures.get(agent_name) == signature
        )

        if looping:
            # Same agent, same request, and nothing new in context since last
            # time: the helper cannot produce what this agent is waiting for.
            # Retrying is what turned one unmet dependency into 36 steps of
            # NCBI and NIM calls, so stop and let the responder explain.
            _logger.info(
                "[%s] escalated again with no new context (%s) - breaking the loop",
                agent_name,
                signature,
            )
            result = AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    f"{agent_name} asked for help again without receiving anything "
                    f"new. Its request ({result.prompt_to_target_agent!r}) cannot be "
                    f"satisfied by the agents available, so the workflow was stopped "
                    f"rather than retrying indefinitely."
                ),
            )
            status = "failed"

        if status == "needs_agent":
            _logger.info(
                "[%s] needs_agent -> %r", agent_name, result.prompt_to_target_agent
            )
        elif status == "failed":
            _logger.info("[%s] failed -> %r", agent_name, result.output)
        else:
            _logger.info("[%s] %s -> %r", agent_name, status, result.output)

        if status == "needs_agent":
            events.thought(step_id, _summarise(result.prompt_to_target_agent))
            events.step_finished(step_id, agent_name, f"The {label} needs help from another agent")
        elif status == "failed":
            events.thought(step_id, _summarise(result.output))
            events.step_finished(step_id, agent_name, f"The {label} could not finish", failed=True)
        elif status == "continue":
            events.step_finished(step_id, agent_name, f"The {label} is not done yet")
        else:
            events.step_finished(step_id, agent_name, f"The {label} finished")

        # Start building the state updates every worker produces, no matter
        # its status: which agent just ran, what it returned, and a log entry.
        updates: dict[str, Any] = {
            "current_agent": agent_name,
            "last_result": result,
            "execution_history": [
                *state.execution_history,
                f"{agent_name} -> {status}",
            ],
        }

        # This agent has now had its turn, so it is no longer pending as a
        # planner-scheduled follow-up. Removing it here rather than in the
        # router is what keeps the router pure: the router only reads state,
        # and something has to write the fact that the step was taken. Without
        # it, a completed follow-up would route back through
        # `_next_follow_up`, still find itself listed, and run again forever.
        if agent_name in state.follow_up_agents:
            updates["follow_up_agents"] = [
                pending for pending in state.follow_up_agents if pending != agent_name
            ]

        if status == "failed":
            # Kept for the responder, which can no longer read the failure off
            # `last_result`: a follow-up runs after this point and would
            # overwrite it. See `WorkflowState.failures`.
            updates["failures"] = [*state.failures, f"{agent_name}: {result.output}"]

            # The research line is over. Anyone still parked on the waiting
            # stack was waiting on work that just died, so a follow-up
            # completing must not "resume" them into a branch whose dependency
            # never arrived - it would send the graph back into the same
            # unsatisfiable escalation the failure just ended.
            if state.follow_up_agents:
                updates["waiting_agent"] = None
                updates["waiting_stack"] = []

        retry_counts = dict(state.continue_retry_counts)
        if status == "continue":
            retry_counts[agent_name] = retries_used + 1
        else:
            retry_counts.pop(agent_name, None)
        updates["continue_retry_counts"] = retry_counts

        if status == "needs_agent":
            # This agent can't finish without help. Remember that it's
            # paused and waiting, so we can come back to it later.
            updates["waiting_stack"] = [*state.waiting_stack, agent_name]
            updates["waiting_agent"] = agent_name
            # Remember what context looked like when it asked, so the next
            # escalation can tell whether the helper actually delivered.
            updates["escalation_signatures"] = {
                **state.escalation_signatures,
                agent_name: sorted(
                    {
                        *state.context,
                        *(result.output if isinstance(result.output, dict) else {}),
                    }
                ),
            }

            # Findings produced before an escalation are still useful to the
            # helper and to the resumed agent. They must enter shared context
            # just like a completed result.
            if isinstance(result.output, dict):
                updates["context"] = {**state.context, **result.output}

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
