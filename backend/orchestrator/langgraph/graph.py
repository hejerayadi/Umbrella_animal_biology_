"""Assembles the orchestrator's LangGraph execution graph.

Graph shape::

    START -> planner -+-> direct_answer ----------------------------------> END
                      |
                      +-> <worker> -> (conditional) -+-> capability_resolver -> <worker> -> ...
                                                     |
                                                     +-> responder ---------> END

Each worker agent is one node. The planner, capability resolver, and the two
answer-writing nodes are each one node. All routing between them is decided
by conditional edges backed by `router.py`, which never calls an LLM - only
the planner, capability resolver, and responder nodes do.

A message that needs no research agent (a greeting, a question about the
platform) goes planner -> direct_answer -> END. Everything else runs agents
and finishes at `responder`, which writes the findings up as prose.

This module only does the wiring; the nodes themselves live in `nodes/`.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ...registry import AGENT_CARDS, AGENT_REGISTRY
from ..capability_resolver import CapabilityResolver
from ..planner import Planner
from ..responder import Responder
from ..router import route_after_planner, route_after_worker
from ..state import WorkflowState
from .nodes import (
    make_direct_answer_node,
    make_planner_node,
    make_resolver_node,
    make_responder_node,
    make_worker_node,
)


def build_orchestrator_graph() -> CompiledStateGraph:
    """Assemble and compile the LangGraph execution graph.

    This function does the actual wiring: create one node per agent, plus
    the planner and resolver nodes, then connect them with the rules for
    "what happens next" (the conditional edges).
    """

    # These do the real work behind the planner/resolver/responder nodes below.
    planner = Planner(AGENT_CARDS)
    resolver = CapabilityResolver(AGENT_CARDS)
    responder = Responder(AGENT_CARDS)

    # StateGraph(WorkflowState) means: every node in this graph reads and
    # writes a WorkflowState object (the "clipboard" described in state.py).
    graph = StateGraph(WorkflowState)
    graph.add_node("planner", make_planner_node(planner))
    graph.add_node("capability_resolver", make_resolver_node(resolver))
    graph.add_node("direct_answer", make_direct_answer_node(responder))
    graph.add_node("responder", make_responder_node(responder))

    # One node per worker agent (Genome, Evolution, Protein, ...), all built
    # the same way via the factory function above.
    for name, agent in AGENT_REGISTRY.items():
        graph.add_node(name, make_worker_node(name, agent))

    # The workflow always starts by running the planner first.
    graph.add_edge(START, "planner")

    worker_names = list(AGENT_REGISTRY)
    # A lookup table LangGraph uses to know "if the chosen next-step is named
    # X, go to the node named X" - it's just an identity mapping since our
    # node names already match the agent names.
    dispatch_map = {name: name for name in worker_names}

    # After the planner runs, either jump to the agent it picked, or - when it
    # decided no agent is needed - go straight to the direct-answer node.
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {**dispatch_map, "direct_answer": "direct_answer"},
    )

    # After the resolver runs, jump straight to whichever agent it picked
    # (read from `state.resolved_agent`).
    graph.add_conditional_edges("capability_resolver", lambda s: s.resolved_agent, dispatch_map)

    # After ANY worker agent runs, use the pure "traffic cop" function from
    # router.py to decide what happens next: another worker, the resolver,
    # or the responder that writes the final answer.
    post_worker_map = {
        **dispatch_map,
        "capability_resolver": "capability_resolver",
        "responder": "responder",
    }
    for name in worker_names:
        graph.add_conditional_edges(name, route_after_worker, post_worker_map)

    # Both ways of producing an answer are the last step before finishing.
    graph.add_edge("direct_answer", END)
    graph.add_edge("responder", END)

    # Turn the graph definition into something that can actually be run.
    return graph.compile()
