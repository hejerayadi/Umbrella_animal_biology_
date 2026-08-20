"""The agent's working state and how it is updated."""
from agent.state.reducers import (
    accumulate_references,
    extend_by_gap,
    merge_by_gap,
    replace,
    unique_extend,
)
from agent.state.state import ReconstructionState, initial_state
from agent.state.transitions import (
    advance_iteration,
    delegate_to,
    halt,
    with_alignment,
    with_candidates,
    with_critique,
    with_error,
    with_gaps,
    with_plan,
    with_reconstruction,
    with_references,
    with_skipped,
    with_tool_call,
    with_warning,
)

__all__ = [
    "ReconstructionState",
    "accumulate_references",
    "advance_iteration",
    "delegate_to",
    "extend_by_gap",
    "halt",
    "initial_state",
    "merge_by_gap",
    "replace",
    "unique_extend",
    "with_alignment",
    "with_candidates",
    "with_critique",
    "with_error",
    "with_gaps",
    "with_plan",
    "with_reconstruction",
    "with_references",
    "with_skipped",
    "with_tool_call",
    "with_warning",
]
