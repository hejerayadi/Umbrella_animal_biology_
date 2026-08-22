from __future__ import annotations

from typing import TypedDict

from ..schema import AgentRequest, AgentResult


class OrchestratorState(TypedDict):
    """What flows between the nodes of the top-level literature graph.

    The single definition - `graph.py` builds its `StateGraph` from this rather
    than declaring its own copy.
    """

    request: AgentRequest
    route: str
    discovery_result: AgentResult | None
    writing_result: AgentResult | None
    final_result: AgentResult | None


def initial_state(request: AgentRequest) -> OrchestratorState:
    """A fresh state for one run. Every key is set, as LangGraph expects."""
    return {
        "request": request,
        "route": "",
        "discovery_result": None,
        "writing_result": None,
        "final_result": None,
    }
