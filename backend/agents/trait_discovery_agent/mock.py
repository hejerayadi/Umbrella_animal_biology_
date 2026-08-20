from .schema import AgentRequest, AgentResult, AgentStatus


class TraitMock:

    def run(self, request: AgentRequest) -> AgentResult:

        if "genome" not in request.context:

            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Genome",
                prompt_to_target_agent="""
Retrieve the genome of the requested species.
"""
            )

        if "papers" not in request.context:

            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Literature",
                prompt_to_target_agent="""
Retrieve papers about genes associated with the requested trait.
"""
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "traits": [
                    "Gene A",
                    "Gene B"
                ]
            }
        )