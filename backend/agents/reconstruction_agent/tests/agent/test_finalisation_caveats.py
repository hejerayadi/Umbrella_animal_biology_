"""What finalisation refuses, and what it accepts with a caveat.

Both behaviours were wrong at once, and a live run showed it. A 26-base fill
was accepted for a 10-base gap, reported RESOLVED with no caveat, and a
consumer substituting it would have shifted every base after that point.

The cause was a gate reading the aggregate validation score instead of the
`fatal` flag each check carries. `ValidationVerdict.fatal` exists so that an
advisory check reaching zero - length is deliberately advisory, because gap
lengths are estimates and real indels change them - does not reject a candidate
it was only meant to caution about. Reading the aggregate inverted that: it
refused only when *every* check scored zero, which is never.
"""

from __future__ import annotations

from reconstruction_agent.domain.enums import CandidateOrigin, GapStatus
from reconstruction_agent.domain.models.candidate import (
    Candidate,
    CandidateScores,
    ValidationVerdict,
)
from reconstruction_agent.domain.models.sequence import Gap
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.tools.finalize.finalize_result import (
    FinalizeResultInput,
    FinalizeResultTool,
)

_GAP = Gap(gap_id="gap_1", start=100, end=110)


def _candidate(
    sequence: str = "ACGTACGTAC",
    *,
    verdicts: tuple[ValidationVerdict, ...] = (),
    origin: CandidateOrigin = CandidateOrigin.HOMOLOGY,
    confidence: float = 0.8,
) -> Candidate:
    return Candidate(
        candidate_id="c1",
        gap_id="gap_1",
        origin=origin,
        sequence=sequence,
        scores=CandidateScores(homology=confidence, validation=0.5),
        validation_results=verdicts,
        final_confidence=confidence,
    )


async def _finalize(candidate: Candidate) -> object:
    outcome = await FinalizeResultTool(ConfidenceEngine()).run(
        FinalizeResultInput(gap_id="gap_1", gap=_GAP, candidates=(candidate,))
    )
    assert outcome.data is not None
    return outcome.data.reconstruction


class TestOnlyFatalChecksDisqualify:
    async def test_a_failed_fatal_check_refuses_the_candidate(self) -> None:
        result = await _finalize(
            _candidate(
                verdicts=(
                    ValidationVerdict(
                        check="alphabet", passed=False, fatal=True, reason="contains N"
                    ),
                )
            )
        )

        assert result.status is GapStatus.UNRESOLVED  # type: ignore[attr-defined]
        assert "alphabet" in result.explanation  # type: ignore[attr-defined]

    async def test_a_failed_advisory_check_does_not(self) -> None:
        """Length is advisory by design: gap lengths are estimates."""
        result = await _finalize(
            _candidate(
                sequence="ACGTACGTACGTACGTACGTACGTAC",
                verdicts=(
                    ValidationVerdict(
                        check="length", passed=False, fatal=False, reason="26 vs 10 bases"
                    ),
                ),
            )
        )

        assert result.status is GapStatus.RESOLVED  # type: ignore[attr-defined]

    async def test_a_fatal_check_outranks_passing_ones(self) -> None:
        """The old gate refused only when every check scored zero - so a single
        passing check rescued any candidate."""
        result = await _finalize(
            _candidate(
                verdicts=(
                    ValidationVerdict(check="gc_consistency", passed=True),
                    ValidationVerdict(check="alphabet", passed=False, fatal=True),
                )
            )
        )

        assert result.status is GapStatus.UNRESOLVED  # type: ignore[attr-defined]


class TestAcceptedFillsCarryTheirCaveats:
    async def test_a_length_mismatch_is_reported(self) -> None:
        """A consumer writes the fill in at the gap's coordinates; a different
        length shifts every base after it, and nothing in the sequence itself
        records that."""
        result = await _finalize(_candidate(sequence="ACGTACGTACGTACGTACGTACGTAC"))
        warnings = " ".join(result.warnings)  # type: ignore[attr-defined]

        assert "26 bases" in warnings
        assert "10" in warnings
        assert "shift" in warnings

    async def test_a_matching_length_produces_no_such_caveat(self) -> None:
        result = await _finalize(_candidate())
        assert not any("shift" in warning for warning in result.warnings)  # type: ignore[attr-defined]

    async def test_an_advisory_failure_is_reported(self) -> None:
        result = await _finalize(
            _candidate(
                verdicts=(
                    ValidationVerdict(
                        check="gc_consistency",
                        passed=False,
                        fatal=False,
                        reason="GC differs from the flanks by 40%",
                    ),
                )
            )
        )
        warnings = " ".join(result.warnings)  # type: ignore[attr-defined]

        assert "gc_consistency" in warnings
        assert "40%" in warnings

    async def test_a_model_fill_says_no_organism_carries_it(self) -> None:
        result = await _finalize(_candidate(origin=CandidateOrigin.MODEL))
        warnings = " ".join(result.warnings)  # type: ignore[attr-defined]

        assert "genome model" in warnings
        assert "sequenced organism" in warnings

    async def test_an_unremarkable_fill_carries_no_caveats(self) -> None:
        """Warnings a reader learns to ignore protect nobody."""
        result = await _finalize(
            _candidate(verdicts=(ValidationVerdict(check="alphabet", passed=True),))
        )

        assert result.warnings == ()  # type: ignore[attr-defined]
