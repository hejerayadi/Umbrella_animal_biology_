"""When the agent loop should stop, and why.

Separated from the graph's edges so the halting rules can be unit-tested
directly, and so there is one authoritative answer to "why did this run end?".

Two kinds of stopping are distinguished, and conflating them would be a bug:

- **Terminal** - the work is over (resolved, abstained, out of budget). The
  agent reports a result.
- **Yield** - this HTTP slice is over but the work is not. The agent
  checkpoints and returns CONTINUE for the orchestrator to retry.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent.state.state import ReconstructionState, is_last_slice
from domain.policies.budget_policy import BudgetPolicy, BudgetUsage


class StopReason(str, Enum):
    ALL_RESOLVED = "all_gaps_resolved"
    NO_PROGRESS = "no_progress"
    MAX_ITERATIONS = "max_iterations_reached"
    NOTHING_TO_DO = "nothing_to_do"
    DELEGATED = "delegated_to_another_agent"
    FATAL_ERROR = "fatal_error"
    #: A budget ran out. The run reports what it managed to gather.
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: The critic concluded no further tool call would help. A scientific
    #: result, reported as COMPLETED with the gaps marked unresolved.
    ABSTAINED = "abstained"
    #: The wall clock for this slice ran out. Not terminal - becomes CONTINUE.
    YIELDED = "yielded_to_orchestrator"


#: Reasons that end the run rather than pausing it.
TERMINAL_REASONS = frozenset(StopReason) - {StopReason.YIELDED}


@dataclass(frozen=True, slots=True)
class StopPolicy:
    """Decides whether to run another plan/act/critique cycle.

    Every branch returns a reason, because a run that stops silently is
    indistinguishable from one that stopped by accident.
    """

    def should_stop(
        self,
        state: ReconstructionState,
        *,
        budgets: BudgetPolicy | None = None,
        usage: BudgetUsage | None = None,
    ) -> tuple[bool, StopReason | None]:
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

        # Every open gap has been abandoned by the critic: more tool calls
        # cannot change the answer, so stopping is the honest move.
        verdicts = state.get("verdicts") or {}
        open_gaps = attemptable - set(state.get("skipped") or {})
        if open_gaps and all(verdicts.get(gap) == "abstain" for gap in open_gaps):
            return True, StopReason.ABSTAINED

        if budgets is not None and usage is not None:
            if budgets.exhausted(usage) is not None:
                return True, StopReason.BUDGET_EXHAUSTED

            # Checked after the terminal reasons: a run that is actually
            # finished should say so, not report a yield it does not need.
            # And on the last slice yielding is not available - another
            # CONTINUE would be converted to FAILED.
            if budgets.should_yield(usage) and not is_last_slice(state):
                return True, StopReason.YIELDED

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
