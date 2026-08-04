"""Node factories for the orchestrator graph.

Each function here takes the object that does the real work (a Planner, a
Responder, a worker agent, ...) and returns the small function LangGraph
calls when that step of the graph runs. `graph.py` wires them together; none
of them know about each other.
"""
from __future__ import annotations

from .answer_nodes import make_direct_answer_node, make_responder_node
from .planner_node import make_planner_node
from .resolver_node import make_resolver_node
from .worker_node import make_worker_node

__all__ = [
    "make_direct_answer_node",
    "make_planner_node",
    "make_resolver_node",
    "make_responder_node",
    "make_worker_node",
]
