"""
Workflow de raisonnement de l'agent Scientific Analysis, structure avec LangGraph.

Schema : Input -> Agent (API Responses) -> Decision -> Tool(s) si besoin -> Agent -> Output
"""
import json
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from .llm_client import get_llm_client, get_deployment_name
from .system_prompt import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, TOOL_DISPATCH

MAX_ITERATIONS = 4


class AgentState(TypedDict):
    input_items: list[dict]
    last_output: list[Any]
    tool_calls_made: list[str]
    iterations: int
    final_text: str


def _agent_node(state: AgentState) -> AgentState:
    client = get_llm_client()
    deployment = get_deployment_name()

    response = client.responses.create(
        model=deployment,
        input=state["input_items"],
        tools=TOOL_SCHEMAS,
    )

    state["last_output"] = response.output
    state["iterations"] += 1

    items_to_replay = []
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            items_to_replay.append({
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            })
    state["input_items"] = state["input_items"] + items_to_replay

    for item in response.output:
        if getattr(item, "type", None) == "message":
            for content in item.content:
                if getattr(content, "type", None) == "output_text":
                    state["final_text"] = content.text

    return state


def _tools_node(state: AgentState) -> AgentState:
    outputs = []
    for item in state["last_output"]:
        if getattr(item, "type", None) == "function_call":
            tool_name = item.name
            call_id = item.call_id
            try:
                args = json.loads(item.arguments) if item.arguments else {}
            except json.JSONDecodeError:
                args = {}

            fn = TOOL_DISPATCH.get(tool_name)
            if fn is None:
                result_text = f"ERREUR_OUTIL: unknown tool '{tool_name}'"
            else:
                try:
                    result_text = fn(**args)
                except TypeError as e:
                    result_text = f"ERREUR_OUTIL: invalid arguments for '{tool_name}' ({e})"

            state["tool_calls_made"].append(tool_name)
            outputs.append({"type": "function_call_output", "call_id": call_id, "output": result_text})

    state["input_items"] = state["input_items"] + outputs
    return state


def _should_continue(state: AgentState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return END
    has_function_call = any(getattr(item, "type", None) == "function_call" for item in state["last_output"])
    return "tools" if has_function_call else END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def run_scientific_analysis_agent(query: str, context_papers: list | None = None, context_records: list | None = None) -> dict:
    """
    Point d'entree interne. Retourne un dict simple (pas un objet Pydantic)
    pour rester coherent avec le reste du repo (search_literature() etc.
    retournent aussi des dicts). Ne leve jamais d'exception.

    Args:
        query: The scientific question/claim to analyze.
        context_papers: Optional list of paper citation strings to enrich the query context.
        context_records: Optional list of record dicts (with title/short_summary) for enrichment.
    """
    if not query or not query.strip():
        return {"status": "error", "answer": "", "tool_calls_made": [], "error": "empty query"}

    # Enrich the query with paper context if available
    enriched_query = query.strip()
    context_parts = []
    
    if context_papers:
        # context_papers is a list of strings (citation lines)
        papers_summary = "\n".join([f"- {p}" for p in context_papers[:5] if isinstance(p, str)])
        if papers_summary:
            context_parts.append(f"Papers found:\n{papers_summary}")
    
    if context_records:
        # context_records is a list of dicts with title, short_summary, year, etc.
        summaries = []
        for r in context_records[:5]:
            if isinstance(r, dict):
                title = r.get("title", "?")
                summary = r.get("short_summary", "")[:200]  # First 200 chars
                year = r.get("year", "")
                if title:
                    summaries.append(f"- {title} ({year}): {summary}")
        if summaries:
            context_parts.append(f"Summaries:\n" + "\n".join(summaries))
    
    if context_parts:
        enriched_query = f"{query.strip()}\n\nContext from search results:\n" + "\n\n".join(context_parts)

    try:
        compiled_graph = build_graph()
        initial_state: AgentState = {
            "input_items": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": enriched_query},
            ],
            "last_output": [],
            "tool_calls_made": [],
            "iterations": 0,
            "final_text": "",
        }
        final_state = compiled_graph.invoke(initial_state)

        answer_text = final_state.get("final_text", "").strip()
        status = "insufficient_info" if not answer_text or "AUCUNE_INFO" in answer_text else "ok"

        return {
            "status": status,
            "answer": answer_text or "No answer could be generated.",
            "tool_calls_made": final_state["tool_calls_made"],
        }

    except Exception as e:
        return {
            "status": "error",
            "answer": "",
            "tool_calls_made": [],
            "error": f"{type(e).__name__}: {e}",
        }
