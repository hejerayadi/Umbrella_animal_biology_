from .schema import AgentRequest, AgentResult, AgentStatus


class BiodiversityMock:

    def run(self, request: AgentRequest) -> AgentResult:

        species = request.context.get("species")

        if species is None:

            return AgentResult(
                status=AgentStatus.FAILED,
                output="Species not specified."
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "biodiversity_report": {
                    "species": species,
                    "status": "Endangered",
                    "habitat": "Savannah"
                }
            }
        )   