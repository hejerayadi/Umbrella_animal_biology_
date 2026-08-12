from __future__ import annotations

from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from ..schema import AgentRequest, AgentResult, AgentStatus

load_dotenv()


class WritingSupportAgent:
    """Agent for scientific writing and drafting."""

    def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}
        discovery_output = context.get("discovery_output", {})
        papers = discovery_output.get("papers", []) if isinstance(discovery_output, dict) else []

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "draft": f"Scientific abstract based on {len(papers)} references: {', '.join(papers[:3])}...",
                "references_used": papers,
                "style": "academic",
            },
        )


class PublicationSupportAgent:
    """Agent for publication venue recommendations."""

    def run(self, request: AgentRequest) -> AgentResult:
        context = request.context or {}
        writing_output = context.get("writing_output", {})
        draft = writing_output.get("draft", "") if isinstance(writing_output, dict) else ""

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "recommended_journals": [
                    "Journal of Animal Genomics",
                    "Biodiversity Research Letters",
                ],
                "filtering_criteria": "impact_factor, scope_match, open_access",
                "explanation": f"Recommendations based on: '{draft[:60]}...'" if draft else "Recommendations based on the paper topic.",
            },
        )


class WritingState(TypedDict):
    """State for the Writing workflow."""

    request: AgentRequest
    writing_path: str  # "writing_only" | "publication_only" | "writing_and_publication"
    writing_result: AgentResult | None
    publication_result: AgentResult | None
    final_result: AgentResult | None


class ScientificWritingOrchestrator:
    """Scientific Writing Orchestrator using LangGraph."""

    def __init__(self) -> None:
        self._writing_support = WritingSupportAgent()
        self._publication_support = PublicationSupportAgent()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(WritingState)

        graph.add_node("determine_path", self._determine_writing_path)
        graph.add_node("writing", self._run_writing)
        graph.add_node("publication", self._run_publication)
        graph.add_node("aggregate", self._aggregate)

        graph.set_entry_point("determine_path")
        graph.add_conditional_edges("determine_path", self._route_after_determination)
        graph.add_edge("writing", "aggregate")
        graph.add_edge("publication", "aggregate")
        graph.add_edge("aggregate", END)

        return graph.compile()

    def _determine_writing_path(self, state: WritingState) -> dict:
        """Determine which writing services are needed based on instruction."""
        instruction = state["request"].instruction.lower()
        needs_publication = any(
            keyword in instruction
            for keyword in ("journal", "venue", "publication", "submit", "reviewer")
        )
        needs_writing = any(
            keyword in instruction for keyword in ("write", "draft", "abstract", "introduction", "section", "rewrite")
        )

        if needs_writing and needs_publication:
            path = "writing_and_publication"
        elif needs_writing:
            path = "writing_only"
        else:
            path = "publication_only"

        return {"writing_path": path}

    def _run_writing(self, state: WritingState) -> dict:
        """Execute the writing support agent."""
        result = self._writing_support.run(state["request"])
        return {"writing_result": result}

    def _run_publication(self, state: WritingState) -> dict:
        """Execute the publication support agent."""
        # If writing was done, add its output to context
        request = state["request"]
        context = dict(request.context or {})
        if state.get("writing_result"):
            context["writing_output"] = state["writing_result"].output

        enriched_request = AgentRequest(instruction=request.instruction, context=context)
        result = self._publication_support.run(enriched_request)
        return {"publication_result": result}

    def _aggregate(self, state: WritingState) -> dict:
        """Aggregate results based on the path taken."""
        path = state["writing_path"]
        writing = state.get("writing_result")
        publication = state.get("publication_result")

        if path == "writing_and_publication":
            output = {
                "writing": writing.output if writing else None,
                "publication": publication.output if publication else None,
            }
        elif path == "writing_only":
            output = writing.output if writing else {}
        else:  # publication_only
            output = publication.output if publication else {}

        return {
            "final_result": AgentResult(
                status=AgentStatus.COMPLETED,
                output=output,
            )
        }

    def _route_after_determination(self, state: WritingState):
        """Route to the appropriate writing services."""
        path = state["writing_path"]

        if path == "writing_and_publication":
            return "writing"  # Start with writing, then go to publication
        elif path == "writing_only":
            return "writing"
        else:  # publication_only
            return "publication"

    def run(self, request: AgentRequest) -> AgentResult:
        """Execute the writing orchestrator graph."""
        result = self._graph.invoke(
            {
                "request": request,
                "writing_path": "",
                "writing_result": None,
                "publication_result": None,
                "final_result": None,
            }
        )
        return result.get("final_result", AgentResult(status=AgentStatus.FAILED, output="No final result"))
