"""Backward-compatible mock entry point delegating to real logic."""

from __future__ import annotations

from .logic import ProteinVisualizationLogic
from .schema import AgentRequest, AgentResult


class ProteinVisualizationMock(ProteinVisualizationLogic):
    """Legacy mock class name — runs the orchestrator integration logic."""

    def run(self, request: AgentRequest) -> AgentResult:
        return super().run(request)
