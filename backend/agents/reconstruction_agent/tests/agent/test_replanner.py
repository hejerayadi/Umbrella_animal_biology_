"""Every replan must change what gets measured.

That is the whole termination argument. A strategy that re-runs the same call
with the same arguments returns the same answer and burns an iteration, so the
tests here check two things for each deficit: that the first attempt changes an
input, and that a second attempt on the same lever declines rather than
repeating itself.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.agent.replanner import Replanner
from reconstruction_agent.domain.enums import CriticDeficit, ToolName
from reconstruction_agent.domain.models.homology import HomologySearchOutcome
from reconstruction_agent.domain.models.sequence import Gap, GapContext


def _state(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "gap_id": "gap_1",
        "gap": Gap(gap_id="gap_1", start=10, end=55),
        "overrides": {},
        "exhausted_scopes": frozenset(),
        "measured_scopes": (),
        "hits": (),
        "candidates": (),
    }
    return {**base, **overrides}


def _scope(code: str, *, hits: int, spanning: int) -> HomologySearchOutcome:
    return HomologySearchOutcome(
        database_code=code, database_label=code, total_hits=hits, gap_spanning_hits=spanning
    )


class TestScopingDeficitsMoveOntoScopesNotYetTried:
    def test_database_mismatch_excludes_the_weak_scope(self) -> None:
        """The scope moved onto is one the lineage offers, never an invented one."""
        state = _state(measured_scopes=(_scope("scope@txid1", hits=50, spanning=0),))
        strategy = Replanner().plan_for(CriticDeficit.DATABASE_MISMATCH, state)

        assert strategy.actionable
        assert "scope@txid1" in strategy.exhaust_scopes
        assert ToolName.SEARCH_HOMOLOGS in strategy.actions

    def test_a_productive_scope_is_never_excluded(self) -> None:
        state = _state(measured_scopes=(_scope("scope@txid1", hits=50, spanning=50),))
        strategy = Replanner().plan_for(CriticDeficit.DATABASE_MISMATCH, state)

        assert not strategy.actionable

    def test_replanning_stops_once_every_scope_is_measured(self) -> None:
        state = _state(
            measured_scopes=(_scope("scope@txid1", hits=50, spanning=0),),
            exhausted_scopes=frozenset({"scope@txid1"}),
        )
        strategy = Replanner().plan_for(CriticDeficit.DATABASE_MISMATCH, state)

        assert not strategy.actionable
        assert "measured" in strategy.rationale


class TestSearchDeficitsRelaxTheSearch:
    def test_no_homologs_relaxes_the_expect_value(self) -> None:
        strategy = Replanner().plan_for(CriticDeficit.NO_HOMOLOGS, _state())
        search = strategy.overrides[ToolName.SEARCH_HOMOLOGS.value]

        assert search["expect"] > 1e-5
        assert search["scope_limit"] > 3

    def test_relaxation_stops_before_the_search_admits_noise(self) -> None:
        state = _state(overrides={ToolName.SEARCH_HOMOLOGS.value: {"expect": 1e-1}})
        strategy = Replanner().plan_for(CriticDeficit.NO_HOMOLOGS, state)

        assert not strategy.actionable

    def test_insufficient_coverage_widens_both_the_search_and_the_fetch(self) -> None:
        strategy = Replanner().plan_for(CriticDeficit.INSUFFICIENT_COVERAGE, _state())

        assert strategy.overrides[ToolName.SEARCH_HOMOLOGS.value]["max_hits"] > 50
        assert strategy.overrides[ToolName.GET_HOMOLOG_SEQUENCES.value]["limit"] > 12


class TestAlignmentDeficitsAddReferences:
    def test_no_gap_spanning_homolog_fetches_further_down_the_list(self) -> None:
        strategy = Replanner().plan_for(CriticDeficit.NO_GAP_SPANNING_HOMOLOG, _state())

        assert strategy.actions[0] is ToolName.GET_HOMOLOG_SEQUENCES
        assert strategy.overrides[ToolName.GET_HOMOLOG_SEQUENCES.value]["limit"] > 12

    def test_the_same_lever_is_not_pulled_twice(self) -> None:
        """Pulling it again would re-run the identical call."""
        state = _state(overrides={ToolName.GET_HOMOLOG_SEQUENCES.value: {"limit": 25}})
        strategy = Replanner().plan_for(CriticDeficit.NO_GAP_SPANNING_HOMOLOG, state)

        assert not strategy.actionable

    def test_insufficient_context_widens_the_flanks_and_restarts(self) -> None:
        state = _state(
            context=GapContext(
                gap=Gap(gap_id="gap_1", start=10, end=55),
                left_flank="A" * 500,
                right_flank="C" * 500,
            )
        )
        strategy = Replanner().plan_for(CriticDeficit.INSUFFICIENT_CONTEXT, state)

        assert strategy.actions[0] is ToolName.GET_SEQUENCE_CONTEXT
        assert strategy.overrides[ToolName.GET_SEQUENCE_CONTEXT.value]["flank_size"] == 1000

    def test_flanks_are_not_widened_past_the_ceiling(self) -> None:
        state = _state(overrides={ToolName.GET_SEQUENCE_CONTEXT.value: {"flank_size": 2000}})
        strategy = Replanner().plan_for(CriticDeficit.INSUFFICIENT_CONTEXT, state)

        assert not strategy.actionable


class TestDeficitsNoSearchCanFix:
    def test_biological_validation_failure_is_not_retried(self) -> None:
        """More homologues do not make an implausible sequence plausible."""
        strategy = Replanner().plan_for(CriticDeficit.BIOLOGICAL_VALIDATION_FAILED, _state())

        assert not strategy.actionable
        assert strategy.rationale

    def test_every_deficit_has_a_strategy_and_a_rationale(self) -> None:
        """A deficit with no handler would replan into an empty plan silently."""
        for deficit in CriticDeficit:
            strategy = Replanner().plan_for(deficit, _state())
            assert strategy.rationale, f"{deficit.value} produced no rationale"
