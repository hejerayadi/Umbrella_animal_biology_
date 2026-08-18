"""Budgets, and the stop reasons they produce.

These matter more than ordinary cost control: the orchestrator grants four HTTP
slices and force-fails the fifth, so a loop that overspends does not just cost
money - it loses every finding gathered.
"""
from __future__ import annotations

import time

import pytest

from agent.planning.stop_policy import StopPolicy, StopReason
from contracts.observation import Observation, ObservationStatus
from domain.models import Gap, GapContext
from domain.policies.budget_policy import BudgetKind, BudgetPolicy, Budgets, BudgetUsage


def make_context(gap_id: str = "gap_1") -> GapContext:
    return GapContext(
        gap=Gap(gap_id, 100, 140), left_flank="A" * 100, right_flank="C" * 100
    )


def open_state(**overrides: object) -> dict:
    """A state with one unresolved gap - the loop would otherwise stop."""
    state = {
        "gap_contexts": [make_context()],
        "skipped": {},
        "reconstructions": {},
        "references": {"gap_1": [object()]},
        # The last round found something. Without this the policy would stop
        # with NO_PROGRESS before reaching the branch each test is about.
        "observations": [
            Observation(tool="blast_search", gap_id="gap_1",
                        status=ObservationStatus.OK, iteration=0, evidence_added=1),
        ],
        "verdicts": {},
        "iteration": 1,
        "max_iterations": 6,
        "slice_index": 0,
        "max_slices": 4,
    }
    state.update(overrides)
    return state


class TestBudgetPolicy:
    def test_allows_a_call_within_the_allowance(self) -> None:
        policy = BudgetPolicy(Budgets(max_tool_calls=5))

        allowed, kind = policy.may_run("blast_search", BudgetUsage(tool_calls=2))

        assert allowed
        assert kind is None

    def test_refuses_once_the_global_cap_is_reached(self) -> None:
        policy = BudgetPolicy(Budgets(max_tool_calls=5))

        allowed, kind = policy.may_run("blast_search", BudgetUsage(tool_calls=5))

        assert not allowed
        assert kind is BudgetKind.TOOL_CALLS

    def test_refuses_once_a_per_tool_cap_is_reached(self) -> None:
        """BLAST is the expensive one; the global cap alone would not stop it."""
        policy = BudgetPolicy(Budgets(max_tool_calls=20, per_tool={"blast_search": 2}))
        usage = BudgetUsage(tool_calls=2, tool_calls_by_name={"blast_search": 2})

        blast_allowed, kind = policy.may_run("blast_search", usage)
        mafft_allowed, _ = policy.may_run("mafft_align", usage)

        assert not blast_allowed
        assert kind is BudgetKind.TOOL_CALLS_FOR_TOOL
        # An unrelated tool is unaffected by another's cap.
        assert mafft_allowed

    def test_reports_which_allowance_is_spent(self) -> None:
        policy = BudgetPolicy(Budgets(max_tool_calls=5, max_llm_tokens=100))

        assert policy.exhausted(BudgetUsage(tool_calls=5)) is BudgetKind.TOOL_CALLS
        assert policy.exhausted(BudgetUsage(llm_tokens=100)) is BudgetKind.LLM_TOKENS
        assert policy.exhausted(BudgetUsage(tool_calls=1, llm_tokens=1)) is None

    def test_a_zero_token_budget_means_unmetered_not_exhausted(self) -> None:
        """Clients that cannot report usage return zeros; that must not stall."""
        policy = BudgetPolicy(Budgets(max_llm_tokens=0))

        assert policy.exhausted(BudgetUsage(llm_tokens=0)) is None

    def test_yields_once_the_wall_clock_is_spent(self) -> None:
        policy = BudgetPolicy(Budgets(yield_after_seconds=0.05))
        usage = BudgetUsage(slice_started_at=time.monotonic() - 0.2)

        assert policy.should_yield(usage)

    def test_does_not_yield_before_the_wall_clock_is_spent(self) -> None:
        policy = BudgetPolicy(Budgets(yield_after_seconds=60.0))
        usage = BudgetUsage(slice_started_at=time.monotonic())

        assert not policy.should_yield(usage)

    def test_elapsed_is_zero_outside_a_slice(self) -> None:
        assert BudgetUsage().elapsed_seconds == 0.0


class TestStopReasons:
    @pytest.fixture
    def policy(self) -> StopPolicy:
        return StopPolicy()

    def test_budget_exhaustion_stops_the_loop(self, policy: StopPolicy) -> None:
        budgets = BudgetPolicy(Budgets(max_tool_calls=4))

        stop, reason = policy.should_stop(
            open_state(), budgets=budgets, usage=BudgetUsage(tool_calls=4)
        )

        assert stop
        assert reason is StopReason.BUDGET_EXHAUSTED

    def test_wall_clock_yields_rather_than_stopping(self, policy: StopPolicy) -> None:
        budgets = BudgetPolicy(Budgets(max_tool_calls=99, yield_after_seconds=0.01))
        usage = BudgetUsage(slice_started_at=time.monotonic() - 1.0)

        stop, reason = policy.should_stop(open_state(), budgets=budgets, usage=usage)

        assert stop
        assert reason is StopReason.YIELDED

    def test_the_last_slice_never_yields(self, policy: StopPolicy) -> None:
        """A CONTINUE on the last slice is converted to FAILED by the
        orchestrator, discarding everything gathered."""
        budgets = BudgetPolicy(Budgets(max_tool_calls=99, yield_after_seconds=0.01))
        usage = BudgetUsage(slice_started_at=time.monotonic() - 1.0)

        stop, reason = policy.should_stop(
            open_state(slice_index=3, max_slices=4), budgets=budgets, usage=usage
        )

        assert reason is not StopReason.YIELDED

    def test_abstaining_on_every_open_gap_stops_the_loop(self, policy: StopPolicy) -> None:
        stop, reason = policy.should_stop(open_state(verdicts={"gap_1": "abstain"}))

        assert stop
        assert reason is StopReason.ABSTAINED

    def test_a_revise_verdict_keeps_the_loop_running(self, policy: StopPolicy) -> None:
        stop, _ = policy.should_stop(open_state(verdicts={"gap_1": "revise"}))

        assert not stop

    def test_terminal_reasons_exclude_only_the_yield(self) -> None:
        from agent.planning.stop_policy import TERMINAL_REASONS

        assert StopReason.YIELDED not in TERMINAL_REASONS
        assert StopReason.ABSTAINED in TERMINAL_REASONS
        assert StopReason.BUDGET_EXHAUSTED in TERMINAL_REASONS
