import os
from dataclasses import dataclass

import pytest
from langgraph.graph import StateGraph, START, END

from workflows.llm import get_llm

@dataclass
class PingState:
    question: str
    answer: str = ""

async def ping_node(state: PingState) -> dict:
    llm = get_llm(temperature=0.0)
    response = await llm.ainvoke(state.question)
    return {"answer": response.content}

def build_ping_graph():
    graph = StateGraph(PingState)
    graph.add_node("ping", ping_node)
    graph.add_edge(START, "ping")
    graph.add_edge("ping", END)
    return graph.compile()

@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("NVIDIA_NIM_API_KEY") and not os.getenv("NIM_API_KEY"),
    reason="NVIDIA_NIM_API_KEY not set",
)
async def test_nim_smoke():
    """Framework-selection sanity check: one node, one LLM call, before the real graph."""
    app = build_ping_graph()
    try:
        result = await app.ainvoke(PingState(question="Reply with exactly the word: pong"))
    except Exception as exc:
        message = str(exc).lower()
        if "401" in message or "unauthorized" in message or "authentication failed" in message:
            pytest.skip(f"NVIDIA NIM authentication failed: {exc}")
        # ---- ADD THIS ----
        if "404" in message and ("not found" in message or "function" in message):
            pytest.skip(f"NVIDIA NIM model not available on this account: {exc}")
        # ------------------
        raise
    assert "pong" in result["answer"].lower()