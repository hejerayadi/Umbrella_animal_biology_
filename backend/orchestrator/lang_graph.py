"""LangGraph execution engine for the Global Scientific Orchestrator.

Graph shape::

    START -> planner -> <worker> -> (conditional) -> capability_resolver -> <worker> -> ... -> END

Each worker agent is one node. The planner and capability resolver are each
one node. All routing between them is decided by conditional edges backed by
`router.route_after_worker`, which never calls an LLM - only the planner and
capability resolver nodes do.

Requests are built with `SimpleNamespace` rather than a shared `AgentRequest`
class: every mock agent only ever reads `.instruction` / `.context` off the
object it receives (duck typing), so no dependency on any one agent's local
schema is needed to call it.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..registry import AGENT_CARDS, AGENT_REGISTRY, WorkerAgent
from .capability_resolver import CapabilityResolver
from .planner import Planner
from .router import route_after_worker
from .state import WorkflowState


def _make_planner_node(planner: Planner):
    """Build the graph node that runs the Planner.

    This is a "node factory": it takes the planner object once and hands
    back a small function (`_node`) that LangGraph will call every time this
    step in the graph runs. That inner function is the actual node.
    """

    def _node(state: WorkflowState) -> dict[str, Any]:
        # Ask the planner: "given the user's question, who should go first?"
        plan = planner.plan(state.user_query)

        # LangGraph nodes don't mutate the state directly - they return a
        # dict of "here's what changed", and LangGraph merges it in.
        return {
            "current_agent": plan.initial_agent,
            "execution_history": [*state.execution_history, f"Planner -> {plan.initial_agent}"],
        }

    return _node


def _make_resolver_node(resolver: CapabilityResolver):
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
        target = resolver.resolve(
            current_agent=state.current_agent,
            prompt_to_target_agent=result.prompt_to_target_agent or state.user_query,
        )
        return {
            "resolved_agent": target,
            "execution_history": [*state.execution_history, f"Resolver -> {target}"],
        }

    return _node


def _make_worker_node(agent_name: str, agent: WorkerAgent):
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


def build_orchestrator_graph() -> CompiledStateGraph:
    """Assemble and compile the LangGraph execution graph.

    This function does the actual wiring: create one node per agent, plus
    the planner and resolver nodes, then connect them with the rules for
    "what happens next" (the conditional edges).
    """

    # These do the real work behind the planner/resolver nodes below.
    planner = Planner(AGENT_CARDS)
    resolver = CapabilityResolver(AGENT_CARDS)

    # StateGraph(WorkflowState) means: every node in this graph reads and
    # writes a WorkflowState object (the "clipboard" described in state.py).
    graph = StateGraph(WorkflowState)
    graph.add_node("planner", _make_planner_node(planner))
    graph.add_node("capability_resolver", _make_resolver_node(resolver))

    # One node per worker agent (Genome, Evolution, Protein, ...), all built
    # the same way via the factory function above.
    for name, agent in AGENT_REGISTRY.items():
        graph.add_node(name, _make_worker_node(name, agent))

    # The workflow always starts by running the planner first.
    graph.add_edge(START, "planner")

    worker_names = list(AGENT_REGISTRY)
    # A lookup table LangGraph uses to know "if the chosen next-step is named
    # X, go to the node named X" - it's just an identity mapping since our
    # node names already match the agent names.
    dispatch_map = {name: name for name in worker_names}

    # After the planner runs, jump straight to whichever agent it picked
    # (read from `state.current_agent`).
    graph.add_conditional_edges("planner", lambda s: s.current_agent, dispatch_map)

    # After the resolver runs, jump straight to whichever agent it picked
    # (read from `state.resolved_agent`).
    graph.add_conditional_edges("capability_resolver", lambda s: s.resolved_agent, dispatch_map)

    # After ANY worker agent runs, use the pure "traffic cop" function from
    # router.py to decide what happens next: another worker, the resolver,
    # or the end of the workflow.
    post_worker_map = {**dispatch_map, "capability_resolver": "capability_resolver", END: END}
    for name in worker_names:
        graph.add_conditional_edges(name, route_after_worker, post_worker_map)

    # Turn the graph definition into something that can actually be run.
    return graph.compile()


class GlobalOrchestrator:
    """Public entry point: runs the compiled LangGraph workflow for a query.

    This is the single object the rest of the app (e.g. main.py, or an API
    endpoint) is meant to use. It hides all the graph-building details above
    behind one simple method: `run(user_query)`.
    """

    def __init__(self) -> None:
        # Build the graph once when the orchestrator is created, not on
        # every single query - building it is the "expensive" one-time setup.
        self._graph = build_orchestrator_graph()

    def run(self, user_query: str, initial_context: dict[str, Any] | None = None) -> WorkflowState:
        """Execute the full plan -> worker -> resolver loop until COMPLETED or FAILED."""

        # Start with a fresh clipboard: just the user's question and any
        # extra known facts (e.g. {"species": "woolly mammoth"}).
        initial_state = WorkflowState(user_query=user_query, context=dict(initial_context or {}))

        # Hand it to LangGraph, which runs planner -> workers -> resolver ->
        # workers -> ... automatically until the graph reaches END.
        final = self._graph.invoke(initial_state)

        # Depending on the LangGraph version, the result may come back as a
        # plain dict instead of a WorkflowState object - normalize it here so
        # callers always get a real WorkflowState either way.
        return final if isinstance(final, WorkflowState) else WorkflowState(**final)
