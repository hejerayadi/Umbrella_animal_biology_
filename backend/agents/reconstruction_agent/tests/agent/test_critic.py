"""Rule-based diagnosis, and the line between a failure and a finding.

The distinction this file exists to protect was found by a live run, not by
reasoning: a search that timed out was diagnosed as NO_HOMOLOGS, the replanner
relaxed the e-value and widened the scopes, and the retry - which did find 50
gap-spanning hits - arrived with no time left to align them. The run reported
an absence it had never measured.

A transport failure is a RETRY concern and belongs to the HTTP layer. Only a
completed measurement may produce a scientific deficit.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.agent.critic import diagnose_by_rule
from reconstruction_agent.domain.enums import CriticDeficit
from reconstruction_agent.domain.models.alignment import AlignmentSupport, ReferenceFill
from reconstruction_agent.domain.models.homology import HomologHit, HomologySearchOutcome
from reconstruction_agent.domain.models.sequence import Gap, GapContext

_GAP = Gap(gap_id="gap_1", start=20, end=65)


def _state(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "gap_id": "gap_1",
        "gap": _GAP,
        "measured_scopes": (),
        "hits": (),
        "candidates": (),
    }
    return {**base, **overrides}


def _scope(
    code: str, *, hits: int = 0, spanning: int = 0, error: str | None = None
) -> HomologySearchOutcome:
    # `total_hits` is derived from the hit list, so the count has to be real.
    return HomologySearchOutcome(
        database_code=code,
        database_label=code,
        hits=tuple(_hit(0.9, accession=f"{code}-{i}") for i in range(hits)),
        gap_spanning_hits=spanning,
        error=error,
    )


def _hit(identity: float, *, accession: str = "X1") -> HomologHit:
    return HomologHit(
        accession=accession,
        description="a homologue",
        identity=identity,
        source_database="scope",
    )


class TestATimedOutSearchIsNotAScientificFinding:
    def test_searches_that_all_failed_produce_no_deficit(self) -> None:
        """The measured regression: this used to read as NO_HOMOLOGS and
        trigger a replan that spent the rest of the run."""
        state = _state(
            measured_scopes=(
                _scope("scope@a", error="did not finish within 200s"),
                _scope("scope@b", error="did not finish within 200s"),
            )
        )
        assert diagnose_by_rule(state) is None

    def test_a_completed_search_returning_nothing_is_no_homologs(self) -> None:
        """Absence measured is a finding, and it is worth replanning against."""
        state = _state(measured_scopes=(_scope("scope@a", hits=0),))
        assert diagnose_by_rule(state) is CriticDeficit.NO_HOMOLOGS

    def test_one_completed_search_is_enough_to_diagnose(self) -> None:
        """A partial failure still leaves a real measurement to act on."""
        state = _state(
            measured_scopes=(
                _scope("scope@a", error="timed out"),
                _scope("scope@b", hits=0),
            )
        )
        assert diagnose_by_rule(state) is CriticDeficit.NO_HOMOLOGS


class TestScopingIsDistinguishedFromAbsence:
    def test_distant_hits_that_do_not_span_are_a_scoping_problem(self) -> None:
        """The homologues exist; they were not being looked for."""
        state = _state(
            measured_scopes=(_scope("scope@a", hits=50, spanning=0),),
            hits=(_hit(0.67),),
        )
        assert diagnose_by_rule(state) is CriticDeficit.DATABASE_MISMATCH

    def test_close_hits_that_do_not_span_are_not_a_scoping_problem(self) -> None:
        state = _state(
            measured_scopes=(_scope("scope@a", hits=50, spanning=0),),
            hits=(_hit(0.96),),
        )
        assert diagnose_by_rule(state) is CriticDeficit.NO_GAP_SPANNING_HOMOLOG


class TestAlignmentDeficits:
    def test_unanchorable_flanks_outrank_everything_downstream(self) -> None:
        state = _state(context=GapContext(gap=_GAP, left_flank="", right_flank=""))
        assert diagnose_by_rule(state) is CriticDeficit.INSUFFICIENT_CONTEXT

    def test_disagreeing_references_are_ambiguous(self) -> None:
        support = AlignmentSupport(
            gap_id="gap_1",
            fills=tuple(
                ReferenceFill(accession=f"X{i}", bases="ACGT", spans_gap=True) for i in range(4)
            ),
            conflicting_positions=(2,),
        )
        state = _state(measured_scopes=(_scope("scope@a", hits=50, spanning=50),), support=support)
        assert diagnose_by_rule(state) is CriticDeficit.AMBIGUOUS_ALIGNMENT

    def test_too_few_spanning_references_is_a_coverage_deficit(self) -> None:
        support = AlignmentSupport(
            gap_id="gap_1",
            fills=(ReferenceFill(accession="X1", bases="ACGT", spans_gap=True),),
        )
        state = _state(measured_scopes=(_scope("scope@a", hits=50, spanning=50),), support=support)
        assert diagnose_by_rule(state) is CriticDeficit.INSUFFICIENT_COVERAGE


def test_sufficient_evidence_produces_no_deficit() -> None:
    """Nothing to act on means finalise, not replan."""
    support = AlignmentSupport(
        gap_id="gap_1",
        fills=tuple(
            ReferenceFill(accession=f"X{i}", bases="ACGT", spans_gap=True) for i in range(5)
        ),
    )
    state = _state(measured_scopes=(_scope("scope@a", hits=50, spanning=50),), support=support)
    assert diagnose_by_rule(state) is None
