from __future__ import annotations

from typing import Any

from ..schema import AgentResult, AgentStatus
from .graph import LiteratureOrchestrator


def decide_next_after_routing(state: dict):
    route = state.get("route", "discovery")
    if route == "discovery":
        return ["discovery"]
    if route == "writing":
        return ["writing"]
    if route == "both_sequential":
        return ["discovery"]
    if route == "both_parallel":
        return ["discovery", "writing"]
    return ["discovery"]


def decide_after_discovery(state: dict):
    return "writing" if state.get("route") == "both_sequential" else "aggregate"


def aggregate_results(state: dict) -> dict:
    discovery = state.get("discovery_result")
    writing = state.get("writing_result")
    return {
        "final_result": AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "discovery": discovery.output if discovery else None,
                "writing": writing.output if writing else None,
            },
        )
    }


def run_orchestrator(instruction: str, context: dict | None = None) -> AgentResult:
    return LiteratureOrchestrator().run(instruction, context)


app = LiteratureOrchestrator()

__all__ = [
    "LiteratureOrchestrator",
    "aggregate_results",
    "decide_after_discovery",
    "decide_next_after_routing",
    "run_orchestrator",
    "app",
]
