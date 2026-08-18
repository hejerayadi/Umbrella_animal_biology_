"""Predicates the graph's conditional edges branch on.

Plain functions of state, deliberately free of LangGraph imports, so the
routing logic is unit-testable by calling them with a dict.
"""
from __future__ import annotations

from agent.state.state import ReconstructionState

# Node names, defined once so edges and conditions cannot drift apart.
LOAD_OR_INIT = "load_or_init"
DETECT_GAPS = "detect_gaps"
PLAN = "plan"
SELECT_TOOLS = "select_tools"
EXECUTE_TOOLS = "execute_tools"
OBSERVE = "observe"
REASON = "reason"
VALIDATE = "validate"
CRITIQUE = "critique"
DECIDE = "decide"
FINALIZE = "finalize"
END = "__end__"


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
    return PLAN if has_work(state) else FINALIZE


def route_after_selection(state: ReconstructionState) -> str:
    """Skip execution when selection produced nothing runnable.

    A plan whose every step was filtered out - preconditions gone, budget
    spent - must not fall through to a no-op tool round. Going straight to
    the critic lets it record why and decide whether to abstain.
    """
    return EXECUTE_TOOLS if state.get("pending_invocations") else CRITIQUE


def route_after_decision(state: ReconstructionState) -> str:
    """Loop back for more evidence, or finish.

    `decide` has already folded the critic's verdict, the budgets and the
    wall clock into `should_continue`; this only reads the outcome so the
    branch stays trivially testable.
    """
    return PLAN if should_continue(state) else FINALIZE
