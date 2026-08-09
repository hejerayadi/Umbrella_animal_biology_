from .schema import AgentRequest, AgentResult, AgentStatus


class ProteinMock:
    def run(self, request: AgentRequest) -> AgentResult:

        if "genome" not in request.context:
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Genome",
                prompt_to_target_agent="""
Retrieve the complete genome of the requested species.
""",
            )

        if "traits" not in request.context:
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Trait",
                prompt_to_target_agent="""
Identify the genes associated with the requested trait.
""",
            )

        return AgentResult(
            status=AgentStatus.COMPLETED, output={"protein_structure": "Predicted 3D Protein Structure"}
        )
