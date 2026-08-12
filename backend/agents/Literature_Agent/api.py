"""Pure Python entry point for the Literature Agent.

This module intentionally contains no FastAPI dependency. The agent logic is
kept in its orchestrator and mock implementations, and this wrapper exists only
as a simple callable interface for direct Python usage.
"""
from __future__ import annotations

from .orchestrator.graph import LiteratureOrchestrator
from .schema import AgentRequest, AgentResult, AgentStatus

_orchestrator = LiteratureOrchestrator()


def execute(request: AgentRequest) -> AgentResult:
    """Execute the Literature Agent directly in-process."""
    try:
        result = _orchestrator._graph.invoke({"request": request, "route": "", "discovery_result": None, "writing_result": None, "final_result": None})
        return result.get("final_result", AgentResult(status=AgentStatus.FAILED, output="No final result"))
    except Exception as exc:  # pragma: no cover - safety net
        return AgentResult(status=AgentStatus.FAILED, output=f"Literature Agent error: {exc}")

