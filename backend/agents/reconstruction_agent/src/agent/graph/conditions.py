"""Predicates the graph's conditional edges branch on.

Plain functions of state, deliberately free of LangGraph imports, so the
routing logic is unit-testable by calling them with a dict.
"""
from __future__ import annotations

from agent.state.state import ReconstructionState, open_gap_ids

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
    """Whether any gap still has a live question.

    False when the sequence had no gaps at all, when every gap was declined by
    the validation policy, and - importantly - when every gap already has a
    final outcome. That last case is a resumed or replayed trace id whose run
    already finished: without it the graph would enter a full plan/critique
    round, spending two LLM calls to rediscover that there was nothing to do.
    """
    if not state.get("gap_contexts"):
        return False

    return bool(open_gap_ids(state))


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
