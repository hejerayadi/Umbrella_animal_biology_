from __future__ import annotations

from typing import Any, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from ..schema import AgentRequest, AgentResult, AgentStatus
from ..subagents.discovery import KnowledgeDiscoveryOrchestrator
from ..subagents.writing import ScientificWritingOrchestrator
from ..routing.router import classify_route_llm

load_dotenv()


class OrchestratorState(TypedDict):
    request: AgentRequest
    route: str
    discovery_result: AgentResult | None
    writing_result: AgentResult | None
    final_result: AgentResult | None


class LiteratureOrchestrator:
    def __init__(self) -> None:
        self._discovery = KnowledgeDiscoveryOrchestrator()
        self._writing = ScientificWritingOrchestrator()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)

        graph.add_node("router", self._route)
        graph.add_node("discovery", self._run_discovery)
        graph.add_node("writing", self._run_writing)
        graph.add_node("aggregate", self._aggregate)

        graph.set_entry_point("router")
        graph.add_conditional_edges("router", self._route_after_router)
        graph.add_conditional_edges("discovery", self._route_after_discovery)
        graph.add_edge("writing", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    def _route(self, state: OrchestratorState) -> dict:
        instruction = state["request"].instruction
        route = classify_route_llm(instruction) or "discovery"
        return {"route": route}

    def _run_discovery(self, state: OrchestratorState) -> dict:
        result = self._discovery.run(state["request"])
        return {"discovery_result": result}

    def _run_writing(self, state: OrchestratorState) -> dict:
        request = state["request"]
        context = dict(request.context or {})
        if state.get("discovery_result") and state["discovery_result"].status == AgentStatus.COMPLETED:
            context["discovery_output"] = state["discovery_result"].output
        result = self._writing.run(
            AgentRequest(instruction=request.instruction, context=context)
        )
        return {"writing_result": result}

    def _aggregate(self, state: OrchestratorState) -> dict:
        discovery = state.get("discovery_result")
        writing = state.get("writing_result")

        return {
            "final_result": AgentResult(
                status=AgentStatus.COMPLETED,
                output={
                    "discovery": discovery.output if discovery else None,
                    "writing": writing.output if writing else None,
                },
            )
        }

    def _route_after_router(self, state: OrchestratorState):
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

    def _route_after_discovery(self, state: OrchestratorState):
        return "writing" if state["route"] == "both_sequential" else "aggregate"

    def run(self, instruction: str, context: dict | None = None) -> AgentResult:
        initial_state: OrchestratorState = {
            "request": AgentRequest(instruction=instruction, context=context or {}),
            "route": "",
            "discovery_result": None,
            "writing_result": None,
            "final_result": None,
        }
        result = self._graph.invoke(initial_state)
        return result["final_result"]
