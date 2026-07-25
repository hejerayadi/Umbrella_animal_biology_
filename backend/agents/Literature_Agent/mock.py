from .schema import AgentRequest, AgentResult, AgentStatus


class LiteratureMock:

    def run(self, request: AgentRequest) -> AgentResult:

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "papers": [
                    "Paper 1",
                    "Paper 2",
                    "Paper 3"
                ]
            }
        )