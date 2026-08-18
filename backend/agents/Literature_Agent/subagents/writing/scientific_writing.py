from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from ...schema import AgentRequest, AgentResult, AgentStatus


# ---------------------------------------------------------------------------
# Leaf agents
# ---------------------------------------------------------------------------


class WritingSupportAgent:
    """Drafts scientific text (abstract, introduction, section …)."""

    def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}
        discovery_output = context.get("discovery_output", {})
        papers = (
            discovery_output.get("papers", [])
            if isinstance(discovery_output, dict)
            else []
        )
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "draft": (
                    f"Scientific abstract based on {len(papers)} references: "
                    f"{', '.join(papers[:3])}..."
                ),
                "references_used": papers,
                "style": "academic",
            },
        )


class PublicationSupportAgent:
    """Recommends journals / venues for a given draft."""

    def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}
        writing_output = context.get("writing_output", {})
        draft = (
            writing_output.get("draft", "")
            if isinstance(writing_output, dict)
            else ""
        )
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "recommended_journals": [
                    "Journal of Animal Genomics",
                    "Biodiversity Research Letters",
                ],
                "filtering_criteria": "impact_factor, scope_match, open_access",
                "explanation": (
                    f"Recommendations based on: '{draft[:60]}...'"
                    if draft
                    else "Recommendations based on the paper topic."
                ),
            },
        )


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------


class WritingState(TypedDict):
    request: AgentRequest
    route: str                        # "writing" | "publication" | "both"
    writing_result: AgentResult | None
    publication_result: AgentResult | None
    final_result: AgentResult | None


# ---------------------------------------------------------------------------
# Sub-orchestrator
# ---------------------------------------------------------------------------


class ScientificWritingOrchestrator:
    """Scientific Writing sub-orchestrator.

    Internal LangGraph pipeline (sequential when both tasks are needed):

        classify
          ├─ writing    → writing_support → aggregate → END
          ├─ publication → publication_support → aggregate → END
          └─ both        → writing_support → publication_support → aggregate → END
    """

    def __init__(self) -> None:
        self._writing_support = WritingSupportAgent()
        self._publication_support = PublicationSupportAgent()
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(WritingState)

        graph.add_node("classify", self._classify)
        graph.add_node("writing_support", self._run_writing)
        graph.add_node("publication_support", self._run_publication)
        graph.add_node("aggregate", self._aggregate)

        graph.set_entry_point("classify")

        # After classify → branch to first relevant node
        graph.add_conditional_edges("classify", self._route_after_classify)

        # After writing_support → either move to publication (both) or aggregate
        graph.add_conditional_edges("writing_support", self._route_after_writing)

        # publication_support always leads to aggregate
        graph.add_edge("publication_support", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    def _classify(self, state: WritingState) -> dict:
        """Decide which writing tasks are required."""
        instruction = state["request"].instruction.lower()
        needs_pub = any(
            k in instruction
            for k in ("journal", "venue", "publication", "submit", "reviewer")
        )
        needs_write = any(
            k in instruction
            for k in ("write", "draft", "abstract", "introduction", "section", "rewrite")
        )
        if needs_write and needs_pub:
            route = "both"
        elif needs_write:
            route = "writing"
        else:
            route = "publication"
        return {"route": route}

    def _run_writing(self, state: WritingState) -> dict:
        result = self._writing_support.run(state["request"])
        return {"writing_result": result}

    def _run_publication(self, state: WritingState) -> dict:
        """Run publication support, enriching context with writing output if available."""
        request = state["request"]
        context = dict(request.context or {})
        if state.get("writing_result") and state["writing_result"].status == AgentStatus.COMPLETED:
            context["writing_output"] = state["writing_result"].output
        result = self._publication_support.run(
            AgentRequest(instruction=request.instruction, context=context)
        )
        return {"publication_result": result}

    def _aggregate(self, state: WritingState) -> dict:
        """Merge writing + publication outputs into a single AgentResult."""
        writing = state.get("writing_result")
        publication = state.get("publication_result")
        route = state["route"]

        if route == "writing":
            final_output = writing.output if writing else {}
        elif route == "publication":
            final_output = publication.output if publication else {}
        else:
            final_output = {
                "writing": writing.output if writing else None,
                "publication": publication.output if publication else None,
            }

        return {
            "final_result": AgentResult(
                status=AgentStatus.COMPLETED,
                output=final_output,
            )
        }

    # ------------------------------------------------------------------
    # Conditional edge functions
    # ------------------------------------------------------------------

    def _route_after_classify(self, state: WritingState) -> str:
        if state["route"] == "publication":
            return "publication_support"
        return "writing_support"  # writing or both

    def _route_after_writing(self, state: WritingState) -> str:
        if state["route"] == "both":
            return "publication_support"
        return "aggregate"

    # ------------------------------------------------------------------
    # Public interface (called by the parent LiteratureOrchestrator)
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResult:
        result = self._graph.invoke(
            {
                "request": request,
                "route": "",
                "writing_result": None,
                "publication_result": None,
                "final_result": None,
            }
        )
        return result["final_result"]
