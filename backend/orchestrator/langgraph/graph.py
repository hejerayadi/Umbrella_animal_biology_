"""Assembles the orchestrator's LangGraph execution graph.

Graph shape::

    START -> planner -+-> direct_answer ----------------------------------> END
                      |
                      +-> extractor -> <worker> -> (conditional) -+-> capability_resolver -> <worker> -> ...
                                                                  |
                                                                  +-> responder ---------> END

Each worker agent is one node. The planner, extractor, capability resolver,
and the two answer-writing nodes are each one node. All routing between them
is decided by conditional edges backed by `router.py`, which never calls an
LLM - only the planner, extractor, capability resolver, and responder nodes
do.

The extractor sits between the planner and the first agent so that the
subject of the question (species, trait, gene) is already in `context` by the
time any agent reads it - see `extractor.py` for why that step is needed.

A message that needs no research agent (a greeting, a question about the
platform) goes planner -> direct_answer -> END. Everything else runs agents
and finishes at `responder`, which writes the findings up as prose.

This module only does the wiring; the nodes themselves live in `nodes/`.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ...registry import AGENT_CARDS, AGENT_ENDPOINTS
from ..capability_resolver import CapabilityResolver
from ..extractor import Extractor
from ..planner import Planner
from ..responder import Responder
from ..router import route_after_planner, route_after_worker
from ..state import WorkflowState
from .nodes import (
    make_direct_answer_node,
    make_extractor_node,
    make_planner_node,
    make_resolver_node,
    make_responder_node,
    make_worker_node,
)
from .nodes.worker_node import CONTINUE_RETRY_DELAYS


def build_orchestrator_graph(
    *,
    planner: Any | None = None,
    extractor: Any | None = None,
    resolver: Any | None = None,
    responder: Any | None = None,
    agent_cards: dict[str, Any] | None = None,
    agent_endpoints: dict[str, str] | None = None,
    worker_client: Any | None = None,
    sleep: Callable[[float], None] | None = None,
    retry_delays: tuple[float, ...] = CONTINUE_RETRY_DELAYS,
) -> CompiledStateGraph:
    """Assemble and compile the LangGraph execution graph.

    This function does the actual wiring: create one node per agent, plus
    the planner and resolver nodes, then connect them with the rules for
    "what happens next" (the conditional edges).
    """

    # Optional dependencies make the complete graph deterministic in tests.
    # Production callers pass nothing and get the same real components and
    # registered HTTP endpoints as before.
    cards = AGENT_CARDS if agent_cards is None else agent_cards
    endpoints = AGENT_ENDPOINTS if agent_endpoints is None else agent_endpoints
    planner = Planner(cards) if planner is None else planner
    extractor = Extractor() if extractor is None else extractor
    resolver = CapabilityResolver(cards) if resolver is None else resolver
    responder = Responder(cards) if responder is None else responder

    # StateGraph(WorkflowState) means: every node in this graph reads and
    # writes a WorkflowState object (the "clipboard" described in state.py).
    graph = StateGraph(WorkflowState)
    graph.add_node("planner", make_planner_node(planner))
    graph.add_node("extractor", make_extractor_node(extractor))
    graph.add_node("capability_resolver", make_resolver_node(resolver))
    graph.add_node("direct_answer", make_direct_answer_node(responder))
    graph.add_node("responder", make_responder_node(responder))

    # One node per worker agent (Genome, Evolution, Protein, ...), all built
    # the same way via the factory function above. Each node holds the URL of
    # that agent's service, not an instance of it.
    for name, base_url in endpoints.items():
        worker_options: dict[str, Any] = {
            "client": worker_client,
            "retry_delays": retry_delays,
        }
        if sleep is not None:
            worker_options["sleep"] = sleep
        graph.add_node(name, make_worker_node(name, base_url, **worker_options))

    # The workflow always starts by running the planner first.
    graph.add_edge(START, "planner")

    worker_names = list(endpoints)
    # A lookup table LangGraph uses to know "if the chosen next-step is named
    # X, go to the node named X" - it's just an identity mapping since our
    # node names already match the agent names.
    dispatch_map: dict[Hashable, str] = {name: name for name in worker_names}

    # After the planner runs, either head into the extractor (which seeds the
    # shared context before any agent sees it) or - when no agent is needed -
    # go straight to the direct-answer node.
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"extractor": "extractor", "direct_answer": "direct_answer"},
    )

    # The extractor always hands off to the agent the planner already chose.
    graph.add_conditional_edges("extractor", lambda s: s.current_agent, dispatch_map)

    # After the resolver runs, jump straight to whichever agent it picked
    # (read from `state.resolved_agent`).
    graph.add_conditional_edges(
        "capability_resolver", lambda s: s.resolved_agent, dispatch_map
    )

    # After ANY worker agent runs, use the pure "traffic cop" function from
    # router.py to decide what happens next: another worker, the resolver,
    # or the responder that writes the final answer.
    post_worker_map: dict[Hashable, str] = {
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
