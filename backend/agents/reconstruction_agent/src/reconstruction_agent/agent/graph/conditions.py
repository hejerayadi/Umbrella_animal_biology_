"""Predicates the graph's conditional edges branch on.

Plain functions of state, deliberately free of LangGraph imports, so the
routing logic is unit-testable by calling them with a dict.
"""
from __future__ import annotations

from ..state.state import ReconstructionState

# Node names, defined once so edges and conditions cannot drift apart.
DETECT_GAPS = "detect_gaps"
PLAN = "plan"
EXECUTE_TOOLS = "execute_tools"
REASON = "reason"
CRITIQUE = "critique"
FINISH = "__end__"


def has_work(state: ReconstructionState) -> bool:
    """Whether any gap is worth attempting.

    False when the sequence had no gaps at all, or when every gap was declined
    by the validation policy - both are complete answers, reached without
    spending a tool call.
    """
    contexts = state.get("gap_contexts") or []
    if not contexts:
        return False

    skipped = state.get("skipped") or {}
    return any(context.identifier not in skipped for context in contexts)


def should_continue(state: ReconstructionState) -> bool:
    """Whether to run another plan/act/critique cycle."""
    if not state.get("should_continue", True):
        return False
    return state.get("iteration", 0) < state.get("max_iterations", 6)


def route_after_detection(state: ReconstructionState) -> str:
    """Straight to the end when there is nothing to reconstruct."""
    return PLAN if has_work(state) else FINISH


def route_after_critique(state: ReconstructionState) -> str:
    """Loop back for more evidence, or stop."""
    return PLAN if should_continue(state) else FINISH
