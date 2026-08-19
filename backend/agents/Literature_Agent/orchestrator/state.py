from __future__ import annotations

from typing import TypedDict

from ..schema import AgentRequest, AgentResult


class OrchestratorState(TypedDict):
    request: AgentRequest
    route: str
    discovery_result: AgentResult | None
    writing_result: AgentResult | None
    final_result: AgentResult | None
