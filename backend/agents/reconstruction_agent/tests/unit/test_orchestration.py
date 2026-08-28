"""Budget and deadline enforcement.

Both exist because the language model cannot be relied on to respect a limit,
and the cost of it getting that wrong is an unbounded loop against metered
external services. These are the deterministic guard rails that sit outside it.
"""

from __future__ import annotations

import pytest

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.orchestration.budget import BudgetLedger, BudgetLimits
from reconstruction_agent.orchestration.deadline import Deadline, PhaseBudget


class TestBudgetReservation:
    def test_a_concurrent_round_cannot_overrun_the_limit(self) -> None:
        """The property the whole design rests on.

        Slots are claimed before the searches are dispatched. Counting them
        afterwards would let all of them run and discover the overrun once the
        money had already been spent.
        """
        ledger = BudgetLedger(limits=BudgetLimits(max_blast_calls=3, max_tool_calls=10))

        assert ledger.reserve(ToolName.SEARCH_HOMOLOGS, 5) == 3
        assert ledger.reserve(ToolName.SEARCH_HOMOLOGS, 1) == 0
        assert ledger.blast_calls == 3

    def test_a_partial_grant_is_normal(self) -> None:
        """Asking for three with two left runs two, rather than refusing."""
        ledger = BudgetLedger(limits=BudgetLimits(max_blast_calls=2))
        assert ledger.reserve(ToolName.SEARCH_HOMOLOGS, 3) == 2

    def test_the_overall_ceiling_binds_as_well_as_the_specific_one(self) -> None:
        ledger = BudgetLedger(limits=BudgetLimits(max_tool_calls=2, max_blast_calls=8))
        assert ledger.reserve(ToolName.SEARCH_HOMOLOGS, 8) == 2
        assert ledger.exhausted is True

    def test_tools_draw_on_their_own_allowances(self) -> None:
        """Exhausting the search budget must not block alignment."""
        ledger = BudgetLedger(limits=BudgetLimits(max_blast_calls=1, max_mafft_calls=2))
        ledger.reserve(ToolName.SEARCH_HOMOLOGS, 1)

        assert ledger.can_afford(ToolName.SEARCH_HOMOLOGS) is False
        assert ledger.can_afford(ToolName.ALIGN_HOMOLOGS) is True

    def test_iterations_are_capped(self) -> None:
        ledger = BudgetLedger(limits=BudgetLimits(max_iterations=2))
        assert [ledger.start_iteration() for _ in range(3)] == [True, True, False]

    def test_snapshot_reports_what_was_spent(self) -> None:
        """This is what the API response `meta` carries back to the caller."""
        ledger = BudgetLedger()
        ledger.reserve(ToolName.SEARCH_HOMOLOGS, 2)
        ledger.reserve_llm(1)

        snapshot = ledger.snapshot()
        assert snapshot["blast_calls"] == 2
        assert snapshot["llm_calls"] == 1
        assert snapshot["budget_exhausted"] is False


class TestDeadline:
    def test_the_finalisation_reserve_is_withheld_from_tools(self) -> None:
        """Tools may only spend what still leaves room to build a result.

        Scoring and serialising after the clock has run out produces nothing at
        all, which is worse than a partial answer delivered on time.
        """
        deadline = Deadline(total_seconds=100.0, reserve_seconds=30.0)

        assert deadline.remaining_for_work() < deadline.remaining()
        assert deadline.remaining_for_work() == max(deadline.remaining() - 30.0, 0.0)

    def test_expiry_is_reached_while_wall_clock_time_remains(self) -> None:
        """Expiry means "start finalising", not "the run is over"."""
        deadline = Deadline(total_seconds=10.0, reserve_seconds=10.0)

        assert deadline.expired() is True
        assert deadline.remaining() > 0.0

    def test_an_operation_longer_than_the_window_is_refused(self) -> None:
        deadline = Deadline(total_seconds=60.0, reserve_seconds=30.0)

        assert deadline.allows(10.0) is True
        assert deadline.allows(120.0) is False

    def test_a_phase_never_outlives_the_run(self) -> None:
        """A phase allowance is clipped to the time that actually remains."""
        deadline = Deadline(total_seconds=50.0, reserve_seconds=10.0)
        phases = PhaseBudget(homology_seconds=200.0)

        assert phases.for_homology(deadline) <= deadline.remaining_for_work()


class TestRateLimiting:
    """Pacing has to work below one request per second, not only above it."""

    def test_a_sub_one_rate_is_expressed_as_one_per_interval(self) -> None:
        """aiolimiter grants nothing when capacity is under one.

        A rate of 0.1/s written as "0.1 per second" gives a capacity of 0.1, so
        every single acquisition raises rather than waiting - which rules out
        exactly the pacing the strictest services ask for. NCBI BLAST wants no
        more than one request every ten seconds.
        """
        from reconstruction_agent.integrations.http.client import _limiter_for

        limiter = _limiter_for(0.1)

        assert limiter.max_rate >= 1.0
        assert limiter.time_period == pytest.approx(10.0)

    def test_a_rate_above_one_stays_per_second(self) -> None:
        from reconstruction_agent.integrations.http.client import _limiter_for

        limiter = _limiter_for(6.0)

        assert limiter.max_rate == pytest.approx(6.0)
        assert limiter.time_period == pytest.approx(1.0)

    def test_a_zero_rate_does_not_divide_by_zero(self) -> None:
        from reconstruction_agent.integrations.http.client import _limiter_for

        assert _limiter_for(0.0).time_period > 0
