from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ...mock import LiteratureMock
from ...schema import AgentRequest, AgentResult, AgentStatus


class DiscoveryState(TypedDict):
    request: AgentRequest
    raw_results: dict | None
    final_result: AgentResult | None


class KnowledgeDiscoveryOrchestrator:
    """Knowledge Discovery sub-orchestrator.

    Internal LangGraph pipeline:
        search → format → END
    """

    def __init__(self) -> None:
        self._agent = LiteratureMock()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(DiscoveryState)

        graph.add_node("search", self._search)
        graph.add_node("format", self._format_results)

        graph.set_entry_point("search")
        graph.add_edge("search", "format")
        graph.add_edge("format", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _search(self, state: DiscoveryState) -> dict:
        """Call the literature source (mock for now) to retrieve papers."""
        result = self._agent.run(state["request"])
        return {"raw_results": result.output}

    def _format_results(self, state: DiscoveryState) -> dict:
        """Enrich and normalise the raw search output."""
        raw = state.get("raw_results") or {}
        papers = raw.get("papers", [])
        return {
            "final_result": AgentResult(
                status=AgentStatus.COMPLETED,
                output={
                    "papers": papers,
                    "total_found": len(papers),
                    "source": "literature_search",
                },
            )
        }

    # ------------------------------------------------------------------
    # Public interface (called by the parent LiteratureOrchestrator)
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResult:
        result = self._graph.invoke(
            {
                "request": request,
                "raw_results": None,
                "final_result": None,
            }
        )
        return result["final_result"]
