from __future__ import annotations

from ..routing.router import classify_route_llm


def decide_next_after_routing(state: dict):
    route = state.get("route", "discovery")
    if route == "discovery":
        return ["discovery"]
    if route == "writing":
        return ["writing"]
    if route == "both_sequential":
        return ["discovery"]
    if route == "both_parallel":
        return ["discovery", "writing"]
    return ["discovery"]


def decide_after_discovery(state: dict):
    return "writing" if state.get("route") == "both_sequential" else "aggregate"


__all__ = ["classify_route_llm", "decide_next_after_routing", "decide_after_discovery"]
