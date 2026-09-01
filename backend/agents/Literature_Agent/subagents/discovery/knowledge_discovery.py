from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ...schema import AgentRequest, AgentResult, AgentStatus
from .sources import search_papers
from ..scientific_analysis import run_scientific_analysis


class DiscoveryState(TypedDict):
    request: AgentRequest
    raw_results: dict | None
    sci_result: dict | None
    final_result: AgentResult | None


class KnowledgeDiscoveryOrchestrator:
    """Knowledge Discovery sub-orchestrator.

    Internal LangGraph pipeline:

        START → search → scientific_analysis → format → END

    Sequential execution: `search` (Retrieval & Knowledge Processing) finds
    papers first, then passes them as context to `scientific_analysis`
    (QA / Contradiction / Gap Detection). This enables the Scientific Analysis
    agent to use paper titles/abstracts as enrichment for its reasoning,
    improving answer quality over the pure query alone.

    Both agents degrade gracefully (never raise), so one failure does not
    cascade nor prevent the other from attempting its work.
    """

    def __init__(self) -> None:
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(DiscoveryState)

        graph.add_node("search", self._search)
        graph.add_node("scientific_analysis", self._run_scientific_analysis)
        graph.add_node("format", self._format_results)

        # Sequential: search first, then scientific_analysis with context.
        graph.add_edge(START, "search")
        graph.add_edge("search", "scientific_analysis")
        graph.add_edge("scientific_analysis", "format")

        graph.add_edge("format", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _search(self, state: DiscoveryState) -> dict:
        """Ask the literature source (Retrieval & Knowledge Processing) for papers."""
        return {"raw_results": search_papers(state["request"].instruction)}

    def _run_scientific_analysis(self, state: DiscoveryState) -> dict:
        """Ask the Scientific Analysis agent (QA / Contradiction / Gap).

        Passes the papers found by `search` as enrichment context.

        `run_scientific_analysis` never raises (see subagents/scientific_analysis/agent.py):
        an unconfigured or failing engine comes back as a dict with
        status="not_configured" or "error", never an exception - so one
        agent failing here cannot crash the whole discovery pipeline.
        """
        try:
            raw_results = state.get("raw_results") or {}
            papers = raw_results.get("papers", [])
            records = raw_results.get("records", [])
            result = run_scientific_analysis(
                state["request"].instruction,
                context_papers=papers,
                context_records=records
            )
        except Exception as exc:  # noqa: BLE001 - a leaf agent must never crash discovery
            result = {"status": "error", "answer": "", "tool_calls_made": [], "error": str(exc)}
        return {"sci_result": result}

    def _format_results(self, state: DiscoveryState) -> dict:
        """Normalise both branches' output into the agent's discovery payload.

        Existing keys (`papers`, `records`, `total_found`, `source`,
        `is_placeholder`, `notice`) are untouched, so nothing that already
        reads `output["discovery"]["records"]` (e.g. the Trait Discovery
        Agent) breaks. `scientific_analysis` is new: a nested dict with its
        own `status` so a caller can tell "no gap/contradiction found" apart
        from "the agent could not run".
        """
        raw = state.get("raw_results") or {}
        papers = raw.get("papers", [])

        output = {
            "papers": papers,
            "records": raw.get("records", []),
            "total_found": raw.get("total_found", len(papers)),
            "source": raw.get("source", "literature_search"),
            "scientific_analysis": state.get("sci_result") or {"status": "not_configured"},
        }
        if raw.get("is_placeholder"):
            output["is_placeholder"] = True
            output["notice"] = raw.get("notice")

        # COMPLETED as long as this node ran: both `search_papers` and
        # `run_scientific_analysis` degrade gracefully instead of raising, so
        # reaching this point always means SOME payload is available - a
        # branch's own failure is visible inside `output`, not by failing the
        # whole discovery step (see graph.py's aggregate_results for how a
        # harder failure - e.g. writing - still propagates at the top level).
        return {"final_result": AgentResult(status=AgentStatus.COMPLETED, output=output)}

    # ------------------------------------------------------------------
    # Public interface (called by the parent LiteratureOrchestrator)
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResult:
        result = self._graph.invoke(
            {
                "request": request,
                "raw_results": None,
                "sci_result": None,
                "final_result": None,
            }
        )
        return result["final_result"]
