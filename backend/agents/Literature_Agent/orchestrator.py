from __future__ import annotations

from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .subagents.discovery.knowledge_discovery import KnowledgeDiscoveryOrchestrator
from .routing.router import classify_route_llm
from .schema import AgentRequest, AgentResult, AgentStatus
from .subagents.writing.scientific_writing import ScientificWritingOrchestrator

load_dotenv()


class OrchestratorState(TypedDict):
    request: AgentRequest
    route: str
    discovery_result: AgentResult | None
    writing_result: AgentResult | None
    final_result: AgentResult | None


knowledge_discovery_orchestrator = KnowledgeDiscoveryOrchestrator()
scientific_writing_orchestrator = ScientificWritingOrchestrator()


def route_with_llm(state: OrchestratorState) -> dict:
    """Choose the route, falling back safely when Azure is unavailable."""
    instruction = state["request"].instruction
    print(f"\n🤖 [ROUTER] Classification LLM pour: '{instruction[:50]}...'")

    try:
        route = classify_route_llm(instruction)
        if not route:
            print("⚠️ [ROUTER] LLM incertain, fallback vers 'discovery'")
            route = "discovery"
        print(f"✅ [ROUTER] Route décidée: {route}")
        return {"route": route}
    except Exception as exc:  # pragma: no cover - safety net
        print(f"❌ [ROUTER] Erreur critique LLM: {exc}")
        return {"route": "discovery"}


def call_discovery(state: OrchestratorState) -> dict:
    print("🔍 [DISCOVERY] Recherche littéraire en cours...")
    result = knowledge_discovery_orchestrator.run(state["request"])
    print(f"📚 [DISCOVERY] Trouvé {len(result.output.get('papers', []))} papiers")
    return {"discovery_result": result}


def call_writing(state: OrchestratorState) -> dict:
    context = dict(state["request"].context or {})

    if state.get("discovery_result") and state["discovery_result"].status == AgentStatus.COMPLETED:
        context["discovery_output"] = state["discovery_result"].output
        print("✍️ [WRITING] Rédaction avec contexte enrichi (mode sequential)")
    else:
        print("✍️ [WRITING] Rédaction sans contexte préalable")

    enriched_request = AgentRequest(instruction=state["request"].instruction, context=context)
    result = scientific_writing_orchestrator.run(enriched_request)
    return {"writing_result": result}


def aggregate_results(state: OrchestratorState) -> dict:
    discovery = state.get("discovery_result")
    writing = state.get("writing_result")
    combined_output: dict[str, Any] = {
        "discovery": discovery.output if discovery else None,
        "writing": writing.output if writing else None,
    }

    return {
        "final_result": AgentResult(
            status=AgentStatus.COMPLETED,
            output=combined_output,
        )
    }


def decide_next_after_routing(state: OrchestratorState):
    route = state["route"]

    if route == "discovery":
        return ["discovery"]
    if route == "writing":
        return ["writing"]
    if route == "both_sequential":
        return ["discovery"]
    if route == "both_parallel":
        return ["discovery", "writing"]
    return ["discovery"]


def decide_after_discovery(state: OrchestratorState):
    if state["route"] == "both_sequential":
        return "writing"
    return "aggregate"


graph = StateGraph(OrchestratorState)

graph.add_node("router_llm", route_with_llm)
graph.add_node("discovery", call_discovery)
graph.add_node("writing", call_writing)
graph.add_node("aggregate", aggregate_results)

graph.set_entry_point("router_llm")

graph.add_conditional_edges("router_llm", decide_next_after_routing)
graph.add_conditional_edges("discovery", decide_after_discovery)
graph.add_edge("writing", "aggregate")
graph.add_edge("aggregate", END)

app = graph.compile()


def run_orchestrator(instruction: str, context: dict | None = None) -> dict:
    initial_state = {
        "request": AgentRequest(instruction=instruction, context=context or {}),
        "route": "",
        "discovery_result": None,
        "writing_result": None,
        "final_result": None,
    }
    return app.invoke(initial_state)
