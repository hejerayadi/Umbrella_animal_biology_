"""The Literature Agent's top-level orchestrator.

Re-exports only - every definition lives in `graph.py`, `router.py` or
`state.py`, so there is one implementation of each and the tests exercise it.
"""
from __future__ import annotations

from ..schema import AgentRequest, AgentResult
from .graph import LiteratureOrchestrator, aggregate_results
from .router import decide_after_discovery, decide_next_after_routing
from .state import OrchestratorState, initial_state


def run_orchestrator(instruction: str, context: dict | None = None) -> AgentResult:
    """Run one instruction through a throwaway orchestrator."""
    return LiteratureOrchestrator().run(instruction, context)


__all__ = [
    "LiteratureOrchestrator",
    "OrchestratorState",
    "aggregate_results",
    "decide_after_discovery",
    "decide_next_after_routing",
    "initial_state",
    "run_orchestrator",
]
