from __future__ import annotations

from ...schema import AgentRequest, AgentResult, AgentStatus


class WritingSupportAgent:
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


class ScientificWritingOrchestrator:
    """Scientific Writing Orchestrator."""

    def __init__(self) -> None:
        self._writing_support = WritingSupportAgent()
        self._publication_support = PublicationSupportAgent()

    def run(self, request: AgentRequest) -> AgentResult:
        instruction = request.instruction.lower()
        needs_publication = any(keyword in instruction for keyword in ("journal", "venue", "publication", "submit", "reviewer"))
        needs_writing = any(keyword in instruction for keyword in ("write", "draft", "abstract", "introduction", "section", "rewrite"))

        if needs_writing and needs_publication:
            writing_result = self._writing_support.run(request)
            enriched_context = dict(request.context or {})
            enriched_context["writing_output"] = writing_result.output
            publication_result = self._publication_support.run(
                AgentRequest(instruction=request.instruction, context=enriched_context)
            )
            return AgentResult(
                status=AgentStatus.COMPLETED,
                output={
                    "writing": writing_result.output,
                    "publication": publication_result.output,
                },
            )

        if needs_writing:
            return self._writing_support.run(request)

        return self._publication_support.run(request)
