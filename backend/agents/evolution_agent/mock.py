from .schema import AgentRequest, AgentResult, AgentStatus


class EvolutionMock:

    def run(self, request: AgentRequest) -> AgentResult:

        if "genome" not in request.context:

            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Genome",
                prompt_to_target_agent="""
Retrieve the genome of the studied species.
"""
            )

        if "papers" not in request.context:

            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Literature",
                prompt_to_target_agent="""
Retrieve scientific papers about the evolution of this species.
"""
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "evolution_analysis": "Evolutionary relationship completed."
            }
        )