from .schema import AgentRequest, AgentResult, AgentStatus


class ReconstructionMock:

    def run(self, request: AgentRequest) -> AgentResult:

        if "genome" not in request.context:

            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Genome",
                prompt_to_target_agent="""
Retrieve the incomplete genome sequence.
"""
            )

        if "evolution_analysis" not in request.context:

            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Evolution",
                prompt_to_target_agent="""
Analyze closely related species to improve genome reconstruction.
"""
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "reconstructed_genome": "Completed genome sequence"
            }
        )