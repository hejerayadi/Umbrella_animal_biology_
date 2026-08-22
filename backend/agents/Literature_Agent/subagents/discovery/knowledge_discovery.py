from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ...schema import AgentRequest, AgentResult, AgentStatus
from .sources import search_papers


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
        """Ask the literature source for papers matching the instruction."""
        return {"raw_results": search_papers(state["request"].instruction)}

    def _format_results(self, state: DiscoveryState) -> dict:
        """Normalise the raw search output into the agent's discovery payload.

        `is_placeholder` and `notice` are carried through untouched when the
        source sets them: whoever reads this output has to be able to tell
        retrieved literature from a stand-in.
        """
        raw = state.get("raw_results") or {}
        papers = raw.get("papers", [])

        output = {
            "papers": papers,
            "total_found": raw.get("total_found", len(papers)),
            "source": raw.get("source", "literature_search"),
        }
        if raw.get("is_placeholder"):
            output["is_placeholder"] = True
            output["notice"] = raw.get("notice")

        return {"final_result": AgentResult(status=AgentStatus.COMPLETED, output=output)}

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
