"""Nodes for the toy LangGraph agent.

The toy graph exercises three LangGraph primitives we need in the real
orchestrator:

1. **A single LLM call node** - proves the Azure OpenAI wrapper works.
2. **A conditional edge** - proves ``add_conditional_edges`` routing.
3. **Parallel fan-out + merge** - proves independent branches join.

The LLM call gracefully degrades to a deterministic stub when Azure
credentials are absent so the toy graph is fully runnable offline.
"""

from __future__ import annotations

import os

from ..llm_client import LLMUnavailable, get_azure_llm
from .state import ToyState


async def call_llm_node(state: ToyState) -> dict:
    """One LLM call through the configured backend, with an offline fallback.

    Real errors from the LLM SDK are re-raised in verbose mode so the
    developer can see them - the silent fallback only fires when the
    credentials are missing (``LLMUnavailable``) or when the offline
    mode is explicitly forced via ``LLM_OFFLINE=1``.
    """

    if os.environ.get("LLM_OFFLINE") == "1":
        text = _offline_answer(state.question)
    else:
        try:
            llm = get_azure_llm()
            response = await llm.ainvoke(state.question)
            text = response.content
        except LLMUnavailable:
            # No credentials at all - deterministic stub good enough to
            # exercise the graph shape.
            text = _offline_answer(state.question)

    needs_escalation = "i don't know" in text.lower() or len(text) < 5
    return {"answer": text, "needs_escalation": needs_escalation}


def route_after_llm(state: ToyState) -> str:
    """Conditional edge — LangGraph equivalent of ``status == NEEDS_AGENT``."""

    return "escalate" if state.needs_escalation else "finish"


async def escalate_node(state: ToyState) -> dict:
    return {"escalation_target": "human_reviewer"}


async def parallel_task_a(state: ToyState) -> dict:
    return {"parallel_result_a": f"A processed: {state.question[:20]}"}


async def parallel_task_b(state: ToyState) -> dict:
    return {"parallel_result_b": f"B processed: {state.question[::-1][:20]}"}


async def merge_node(state: ToyState) -> dict:
    return {"merged_result": f"{state.parallel_result_a} | {state.parallel_result_b}"}


# ---------- helpers ----------


def _offline_answer(question: str) -> str:
    """Trivial rule-based response so the graph runs without Azure keys."""

    q = question.lower()
    if "pathway" in q or "ucp1" in q:
        return "UCP1 is involved in the thermogenesis pathway."
    if "species" in q or "biodiversity" in q:
        return "Species distributions are tracked in GBIF."
    return "Offline stub response — the graph shape is what matters here."
