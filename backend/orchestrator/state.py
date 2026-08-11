from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowState:
    """Mutable state shared by every node in the orchestrator graph.

    Think of this as a "clipboard" that gets passed from agent to agent.
    Every node in the graph (planner, resolver, each worker) can read it and
    hand back an updated version of it.

    `waiting_stack` is an implementation detail on top of the required
    `waiting_agent` field: it records the full chain of agents paused while
    waiting on a dependency, so nested requests (A waits on B, B waits on C)
    resume the correct caller once each dependency completes. `waiting_agent`
    always mirrors whichever agent should be resumed next.

    `last_result` is typed loosely (`Any`) rather than a shared `AgentResult`
    class: each agent still defines its own local result type, so the graph
    reads it structurally (`.status.value`, `.output`, ...) instead of
    depending on one common dataclass.
    """

    # The original question the user asked, in plain English. This never
    # changes once the workflow starts.
    user_query: str

    # Set when the user attached an image to this message. The image itself is
    # NOT here - only its id, in `context["recognition_image_id"]`, with the
    # bytes held server-side by `backend/image_store.py`.
    #
    # This flag exists because the Planner reads only the text. "What animal is
    # this?" with a photo and without one are the same sentence, but only one
    # of them can be answered by the Recognition agent - and that agent is
    # useless without an image. The planner needs to be told.
    has_image: bool = False

    # The name of the agent that is running (or about to run) right now,
    # e.g. "Genome" or "Protein".
    current_agent: str | None = None

    # The name of the agent we should go BACK to once the current one
    # finishes - i.e. who is "on hold" waiting for help. None means nobody
    # is waiting, so when the current agent finishes, the whole workflow is
    # done.
    waiting_agent: str | None = None

    # The full list of agents that are paused, in the order they paused.
    # Example: if Protein pauses to wait on Trait, and Trait itself pauses to
    # wait on Literature, this list would be ["Protein", "Trait"] while
    # Literature is running. It's how we know who to resume, one at a time,
    # once each helper finishes.
    waiting_stack: list[str] = field(default_factory=list)

    # Filled in by the capability resolver: "here is the agent that can help
    # with what you're missing." Read by the graph to know which worker node
    # to jump to next.
    resolved_agent: str | None = None

    # The instruction each agent should be sent, keyed by agent name.
    #
    # An agent pulled in to satisfy someone else's `needs_agent` must be told
    # what was actually asked for, not the user's original sentence. Without
    # this, the Trait agent asks for a gene list and the Genome agent receives
    # "Draw an Arctic fox" - it fetches an assembly, never runs gene
    # annotation, and Trait asks again forever.
    #
    # Kept per agent rather than as a single "next request" so that an agent
    # resumed after its dependency finishes replays the same instruction it
    # was originally given, instead of silently reverting to the user's query.
    # An agent with no entry here (the one the planner started with) is sent
    # `user_query`, which is the right thing for it.
    agent_instructions: dict[str, str] = field(default_factory=dict)

    # For each agent, the context keys that existed the last time it said
    # `needs_agent`. Used to stop an unsatisfiable dependency from looping:
    # if an agent escalates again and nothing new has arrived in context, the
    # helper cannot give it what it needs and retrying would spin forever.
    escalation_signatures: dict[str, list[str]] = field(default_factory=dict)

    # Number of retryable CONTINUE results already returned by each agent.
    # The worker node uses this to apply the bounded 1/2/4-second retry policy
    # and removes an entry as soon as that agent returns any other status.
    continue_retry_counts: dict[str, int] = field(default_factory=dict)

    # All the facts gathered so far, shared by every agent. For example,
    # after the Genome agent runs, this might contain {"genome": "..."}.
    # Every agent reads from this and adds to it.
    context: dict[str, Any] = field(default_factory=dict)

    # A simple, human-readable log of everything that happened, in order.
    # Useful for debugging and for showing the user what the orchestrator did.
    execution_history: list[str] = field(default_factory=list)

    # Whatever the most recently run agent just returned (its AgentResult).
    # The router looks at this to decide what happens next.
    last_result: Any | None = None

    # The finished, human-readable answer written by the Responder - this is
    # the text the user actually sees. Stays None until the workflow reaches
    # either the `direct_answer` or `responder` node at the very end.
    final_answer: str | None = None
