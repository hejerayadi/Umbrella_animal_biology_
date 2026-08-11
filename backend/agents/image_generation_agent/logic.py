"""Main agent entry point — delegates to orchestrator integration logic."""

from __future__ import annotations

from .orchestrator_logic import run_orchestrator_logic
from .schema import AgentRequest, AgentResult


class ImageGenerationLogic:
    """Image generation agent: routing controller + FLUX.2-pro generation."""

    def run(self, request: AgentRequest) -> AgentResult:
        return run_orchestrator_logic(request)
