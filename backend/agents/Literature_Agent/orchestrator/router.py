"""Which node runs next, given the route the classifier chose.

The single definition of the top-level graph's branching. `graph.py` passes
these straight to `add_conditional_edges`, and the tests exercise them
directly, so the behaviour under test is the behaviour that ships.
"""
from __future__ import annotations

from ..routing.router import classify_route_llm


def decide_next_after_routing(state: dict) -> list[str]:
    """Fan out from the router.

    Returns a list because `both_parallel` starts two nodes in the same
    LangGraph superstep. `both_sequential` starts only discovery - writing is
    reached afterwards, from `decide_after_discovery`, so that it can be handed
    the papers that were found.
    """
    route = state.get("route", "discovery")

    if route == "writing":
        return ["writing"]
    if route == "both_parallel":
        return ["discovery", "writing"]
    # "discovery", "both_sequential", and anything unrecognised: start with a
    # search. Discovery is the safe default - it retrieves rather than asserts.
    return ["discovery"]


def decide_after_discovery(state: dict) -> str:
    """Only the sequential route continues into writing; everything else ends."""
    return "writing" if state.get("route") == "both_sequential" else "aggregate"


__all__ = ["classify_route_llm", "decide_next_after_routing", "decide_after_discovery"]
