from langgraph.graph import StateGraph, START, END
from framework.toy_agent_langgraph.state import ToyState
from framework.toy_agent_langgraph.nodes import (
    call_llm_node, route_after_llm, escalate_node,
    parallel_task_a, parallel_task_b, merge_node,
)

def build_toy_graph():
    graph = StateGraph(ToyState)

    graph.add_node("call_llm", call_llm_node)
    graph.add_node("escalate", escalate_node)


    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm", route_after_llm,
        {"escalate": "escalate", "finish": "parallel_a"},
    )

    graph.add_node("parallel_a", parallel_task_a)
    graph.add_node("parallel_b", parallel_task_b)
    graph.add_node("merge", merge_node)

    
    graph.add_edge("call_llm", "parallel_b")
    graph.add_edge("parallel_a", "merge")
    graph.add_edge("parallel_b", "merge")

    graph.add_edge("escalate", END)
    graph.add_edge("merge", END)

    return graph.compile()