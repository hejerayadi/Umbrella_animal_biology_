from .schema import AgentRequest, AgentResult, AgentStatus


class GenomeMock:

    def run(self, request: AgentRequest) -> AgentResult:

        species = request.context.get("species")

        if species is None:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="No species provided."
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "genome": f"Genome sequence of {species}"
            }
        )