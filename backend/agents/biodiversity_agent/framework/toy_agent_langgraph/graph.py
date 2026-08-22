"""Toy LangGraph agent for the Biodiversity Agent framework validation.

Sprint 2 task 2 requires us to "test a small agent first, choose the
LLM, set up the project structure". This module is exactly that: a
throwaway graph that proves LangGraph runs, the LLM wrapper works
(or degrades cleanly), and the parallel + conditional primitives we
need in the real orchestrator behave as expected.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    call_llm_node,
    escalate_node,
    merge_node,
    parallel_task_a,
    parallel_task_b,
    route_after_llm,
)
from .state import ToyState


def build_toy_graph():
    graph = StateGraph(ToyState)

    graph.add_node("call_llm", call_llm_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("parallel_a", parallel_task_a)
    graph.add_node("parallel_b", parallel_task_b)
    graph.add_node("merge", merge_node)

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {"escalate": "escalate", "finish": "parallel_a"},
    )
    # ``call_llm`` also fans out to parallel_b so parallel_a and parallel_b
    # run concurrently, then merge waits for both.
    graph.add_edge("call_llm", "parallel_b")
    graph.add_edge("parallel_a", "merge")
    graph.add_edge("parallel_b", "merge")

    graph.add_edge("escalate", END)
    graph.add_edge("merge", END)

    return graph.compile()
