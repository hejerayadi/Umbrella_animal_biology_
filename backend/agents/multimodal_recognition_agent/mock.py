from .schema import AgentRequest, AgentResult, AgentStatus


class MultimodalMock:

    def run(self, request: AgentRequest) -> AgentResult:

        if "image" not in request.context:

            return AgentResult(
                status=AgentStatus.FAILED,
                output="No image was provided."
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "species": "Loxodonta africana"
            }
        )