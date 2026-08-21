from langgraph.graph import StateGraph, START, END

from workflows.state import FunctionalEvidenceState
from workflows.nodes.functional_evidence_nodes import pathways_node, protein_data_node, merge_node


def build_functional_evidence_graph():
    graph = StateGraph(FunctionalEvidenceState)

    graph.add_node("pathways", pathways_node)
    graph.add_node("fetch_protein_data", protein_data_node)
    graph.add_node("merge", merge_node)

    # fan-out from START, fan-in to merge — same pattern validated with a real NIM call
    # in the Task 2 toy agent
    graph.add_edge(START, "pathways")
    graph.add_edge(START, "fetch_protein_data")
    graph.add_edge("pathways", "merge")
    graph.add_edge("fetch_protein_data", "merge")
    graph.add_edge("merge", END)

    return graph.compile()