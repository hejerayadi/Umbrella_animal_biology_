"""When the agent loop should stop.

Separated from the graph's edges so the halting rules can be unit-tested
directly, and so there is one authoritative answer to "why did this run end?".
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..state.state import ReconstructionState


class StopReason(str, Enum):
    ALL_RESOLVED = "all_gaps_resolved"
    NO_PROGRESS = "no_progress"
    MAX_ITERATIONS = "max_iterations_reached"
    NOTHING_TO_DO = "nothing_to_do"
    DELEGATED = "delegated_to_another_agent"
    FATAL_ERROR = "fatal_error"


@dataclass(frozen=True, slots=True)
class StopPolicy:
    """Decides whether to run another plan/act/critique cycle.

    Every branch returns a reason, because a run that stops silently is
    indistinguishable from one that stopped by accident.
    """

    def should_stop(self, state: ReconstructionState) -> tuple[bool, StopReason | None]:
        if state.get("needs_agent"):
            return True, StopReason.DELEGATED

        if state.get("errors") and not state.get("reconstructions"):
            # Errors alone are not fatal - partial results still ship - but
            # errors with nothing to show for the run are.
            return True, StopReason.FATAL_ERROR

        contexts = state.get("gap_contexts") or []
        if not contexts:
            return True, StopReason.NOTHING_TO_DO

        attemptable = {
            context.identifier
            for context in contexts
            if context.identifier not in (state.get("skipped") or {})
        }
        if not attemptable:
            return True, StopReason.NOTHING_TO_DO

        if attemptable <= set(state.get("reconstructions") or {}):
            return True, StopReason.ALL_RESOLVED

        iteration = state.get("iteration", 0)
        if iteration >= state.get("max_iterations", 6):
            return True, StopReason.MAX_ITERATIONS

        # An iteration that ran tools and produced no new evidence will not
        # produce any on a replay of the same plan either.
        if iteration > 0 and not self._made_progress(state):
            return True, StopReason.NO_PROGRESS

        return False, None

    @staticmethod
    def _made_progress(state: ReconstructionState) -> bool:
        """Whether the last iteration added anything usable.

        Evidence counts as progress even when it did not yield a
        reconstruction: references found this round can align next round.
        """
        return bool(state.get("references")) or bool(state.get("candidates"))
