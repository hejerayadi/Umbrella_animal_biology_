from langgraph.graph import StateGraph, END
from typing import TypedDict

class ToyState(TypedDict):
    input_value: int
    result: str

def node_a(state: ToyState) -> ToyState:
    print("Node A received:", state["input_value"])
    return {"input_value": state["input_value"], "result": "processed by A"}

def node_b(state: ToyState) -> ToyState:
    print("Node B received:", state["input_value"])
    return {"input_value": state["input_value"], "result": "processed by B"}

# This function decides WHICH node to go to next, based on the state
def route_after_a(state: ToyState) -> str:
    if state["input_value"] > 10:
        return "node_b"
    return END

graph = StateGraph(ToyState)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.set_entry_point("node_a")

# conditional edge: after node_a, call route_after_a to decide what's next
graph.add_conditional_edges("node_a", route_after_a, {"node_b": "node_b", END: END})
graph.add_edge("node_b", END)

app = graph.compile()

if __name__ == "__main__":
    print("--- Test 1: input_value = 5 (should skip node_b) ---")
    result = app.invoke({"input_value": 5, "result": ""})
    print("Final result:", result)

    print("\n--- Test 2: input_value = 15 (should route to node_b) ---")
    result = app.invoke({"input_value": 15, "result": ""})
    print("Final result:", result)