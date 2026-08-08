import operator
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    # The add reducer appends parallel outputs into a single list
    aggregate: Annotated[list[str], operator.add]


def node_x(state: State) -> State:
    print("Node X running")
    return {"aggregate": ["X done"]}


def node_y(state: State) -> State:
    print("Node Y running")
    return {"aggregate": ["Y done"]}


graph = StateGraph(State)
graph.add_node("node_x", node_x)
graph.add_node("node_y", node_y)

# Both nodes start directly from START — this is what makes them run in parallel
graph.add_edge(START, "node_x")
graph.add_edge(START, "node_y")

# Both feed into END — this is the "join" point where results are collected
graph.add_edge("node_x", END)
graph.add_edge("node_y", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"aggregate": []})
    print("Final result:", result)