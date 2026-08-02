from framework.toy_agent_langgraph.state import ToyState
from framework.llm_client import get_nim_llm

_llm = get_nim_llm()

async def call_llm_node(state: ToyState) -> dict:


    """One real LLM call through NVIDIA NIM."""

    response = await _llm.ainvoke(state.question)
    text = response.content
    needs_escalation = "i don't know" in text.lower() or len(text) < 5
    return {"answer": text, "needs_escalation": needs_escalation}




def route_after_llm(state: ToyState) -> str:
    """Conditional edge — the LangGraph equivalent of checking `status == NEEDS_AGENT`."""
    
    return "escalate" if state.needs_escalation else "finish"




async def escalate_node(state: ToyState) -> dict:
    return {"escalation_target": "human_reviewer"}



async def parallel_task_a(state: ToyState) -> dict:
    return {"parallel_result_a": f"A processed: {state.question[:20]}"}



async def parallel_task_b(state: ToyState) -> dict:
    return {"parallel_result_b": f"B processed: {state.question[::-1][:20]}"}



async def merge_node(state: ToyState) -> dict:
    return {"merged_result": f"{state.parallel_result_a} | {state.parallel_result_b}"}