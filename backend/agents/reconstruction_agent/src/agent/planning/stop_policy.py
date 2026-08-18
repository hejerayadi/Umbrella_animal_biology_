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

from agent.state.state import (
    ReconstructionState,
    final_gap_outcomes,
    is_last_slice,
    open_gap_ids,
)
from contracts.observation import ObservationStatus
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
            # Every gap the sequence has was declined by the validation policy
            # before a single tool ran. Nothing was attempted, so nothing was
            # resolved or abandoned - the honest word is "nothing to do".
            return True, StopReason.NOTHING_TO_DO

        # Which gaps still have a live question, and which are settled. A gap
        # merely present in `reconstructions` is NOT settled: an UNRESOLVED
        # entry is this round's best answer, not a verdict.
        outcomes = final_gap_outcomes(state)
        still_open = open_gap_ids(state)

        if not still_open:
            # Everything has a final outcome. Which reason that is depends on
            # what those outcomes are, and conflating them is what made the
            # agent report "all_gaps_resolved" for a run that resolved nothing.
            attempted = {gap: outcome for gap, outcome in outcomes.items() if gap in attemptable}
            if attempted and all(outcome == "reconstructed" for outcome in attempted.values()):
                return True, StopReason.ALL_RESOLVED
            if any(outcome == "abstained" for outcome in attempted.values()):
                return True, StopReason.ABSTAINED
            return True, StopReason.ALL_RESOLVED

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

        # An iteration that ran and produced no new evidence will not produce
        # any on a replay of the same plan either.
        if iteration > 0 and not self._made_progress(state):
            return True, StopReason.NO_PROGRESS

        return False, None

    @staticmethod
    def _made_progress(state: ReconstructionState) -> bool:
        """Whether the loop is still getting somewhere.

        Measured per round from the observation trail, not from accumulated
        state. Reading `bool(state["references"])` instead would answer "has
        this run ever found a reference?", which stays True for the rest of the
        run after the first success - so a run stuck on one unresolvable gap
        would burn every remaining iteration before MAX_ITERATIONS noticed, and
        NO_PROGRESS could only ever fire for a run that found nothing at all.

        **Two** barren rounds are required, not one. A tool that finds nothing
        is entitled to a second attempt with relaxed parameters, and that retry
        is planned on the round *after* the barren one - so stopping at the
        first empty round would cancel the retry before it ever ran, and the
        relaxed search this agent promises would be dead code.
        """
        observations = state.get("observations") or []
        if not observations:
            return False

        # `decide` increments `iteration` before probing the policy, so the
        # round whose observations we want is one behind the state's counter.
        current_round = max(0, state.get("iteration", 0) - 1)

        def productive(round_index: int) -> bool | None:
            """True/False for a round that ran; None for one that did not."""
            recent = [o for o in observations if o.iteration == round_index]
            if not recent:
                return None
            # A tool that ran cleanly counts, even when it added no
            # *references*: `evolutionary_context` returns relatedness rather
            # than evidence, and that genuinely moves the run forward. EMPTY,
            # FAILED and SKIPPED do not.
            return any(o.status is ObservationStatus.OK for o in recent)

        latest = productive(current_round)
        if latest is None:
            # Nothing ran this round at all - the plan produced no runnable
            # calls. There is no retry pending, so this is a true stall.
            return False
        if latest:
            return True

        # This round was barren. Allow one more only if the previous round was
        # not - that is the retry window.
        if current_round == 0:
            return True
        return productive(current_round - 1) is not False
