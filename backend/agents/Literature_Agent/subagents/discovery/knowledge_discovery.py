from __future__ import annotations

from ...mock import LiteratureMock
from ...schema import AgentRequest, AgentResult


class KnowledgeDiscoveryOrchestrator:
    """Knowledge Discovery & Analysis sub-orchestrator."""

    def __init__(self) -> None:
        self._agent = LiteratureMock()

    def run(self, request: AgentRequest) -> AgentResult:
        return self._agent.run(request)
