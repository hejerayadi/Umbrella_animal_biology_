"""The conditional edges, tested as what they are: functions over a dict.

Routing is where an agent loops forever, finalises with nothing, or replans
past its budget. Testing it directly - no graph compiled, no service faked -
is what makes those three behaviours provable rather than hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reconstruction_agent.agent import router
from reconstruction_agent.domain.enums import CriticDeficit, EvaluationDecision


@dataclass
class _Clock:
    """A deadline whose expiry is stated rather than waited for."""

    over: bool = False

    def expired(self) -> bool:
        return self.over


@dataclass
class _Ledger:
    spent: bool = False

    @property
    def exhausted(self) -> bool:
        return self.spent


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "plan": ("search_homologs",),
        "deadline": _Clock(),
        "budget": _Ledger(),
        "replan_count": 0,
        "max_replans": 2,
    }
    return {**base, **overrides}


class TestTheGuardIsTheOnlyPlaceWorkIsRefused:
    def test_an_affordable_action_is_dispatched(self) -> None:
        assert router.after_guard(_state()) == router.ACT

    def test_an_empty_plan_finalises(self) -> None:
        assert router.after_guard(_state(plan=())) == router.FINALIZE

    def test_a_closed_window_finalises_rather_than_dispatching(self) -> None:
        """A call whose answer arrives after the response has been sent is
        worse than no call: it spends the reserve finalisation needs."""
        assert router.after_guard(_state(deadline=_Clock(over=True))) == router.FINALIZE

    def test_an_exhausted_budget_finalises(self) -> None:
        assert router.after_guard(_state(budget=_Ledger(spent=True))) == router.FINALIZE


class TestEvaluationOnlyKeepsTheLoopOpenForMissingEvidence:
    def test_more_evidence_required_reaches_the_critic(self) -> None:
        state = _state(decision=EvaluationDecision.MORE_EVIDENCE_REQUIRED)
        assert router.after_evaluate(state) == router.CRITIC

    def test_a_ready_candidate_finalises(self) -> None:
        state = _state(decision=EvaluationDecision.CANDIDATE_READY)
        assert router.after_evaluate(state) == router.FINALIZE

    def test_the_replan_limit_stops_the_loop(self) -> None:
        """The termination guarantee: a deficit that survives its allowance is
        reported, not retried forever."""
        state = _state(
            decision=EvaluationDecision.MORE_EVIDENCE_REQUIRED,
            replan_count=2,
            max_replans=2,
        )
        assert router.after_evaluate(state) == router.FINALIZE

    def test_a_closed_window_outranks_missing_evidence(self) -> None:
        state = _state(
            decision=EvaluationDecision.MORE_EVIDENCE_REQUIRED,
            deadline=_Clock(over=True),
        )
        assert router.after_evaluate(state) == router.FINALIZE

    def test_every_terminal_decision_finalises(self) -> None:
        for decision in (
            EvaluationDecision.RESOLVED,
            EvaluationDecision.UNRESOLVED,
            EvaluationDecision.BUDGET_EXHAUSTED,
            EvaluationDecision.FAILED,
        ):
            assert router.after_evaluate(_state(decision=decision)) == router.FINALIZE


class TestReplanningIsRefusedWithoutSomethingToActOn:
    def test_a_named_deficit_reaches_the_replanner(self) -> None:
        state = _state(deficit=CriticDeficit.NO_HOMOLOGS)
        assert router.after_critic(state) == router.REPLAN

    def test_no_deficit_finalises_rather_than_replanning_blindly(self) -> None:
        """Replanning without a diagnosis re-runs the strategy that just failed."""
        assert router.after_critic(_state(deficit=None)) == router.FINALIZE

    def test_a_replan_with_no_new_actions_finalises(self) -> None:
        assert router.after_replan(_state(plan=())) == router.FINALIZE

    def test_a_replan_with_actions_re_enters_the_loop(self) -> None:
        assert router.after_replan(_state(plan=("search_homologs",))) == router.REASON
