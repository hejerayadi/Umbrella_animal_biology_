"""Deterministic confidence, and the biological checks that gate it.

The property under test throughout is that a number comes from evidence and
nothing else. Nothing here calls a model, and the same inputs always give the
same output - which is what makes a returned confidence something a reviewer
can check rather than something they have to trust.
"""

from __future__ import annotations

import pytest

from reconstruction_agent.domain.enums import ConfidenceLevel
from reconstruction_agent.domain.models.candidate import CandidateScores
from reconstruction_agent.domain.models.sequence import Gap, GapContext
from reconstruction_agent.services.scoring.confidence_engine import (
    ConfidenceEngine,
    ConfidenceThresholds,
    alignment_score,
    homology_score,
)
from reconstruction_agent.services.validation.validators import (
    ValidationThresholds,
    aggregate_score,
    validate,
)


@pytest.fixture
def engine() -> ConfidenceEngine:
    return ConfidenceEngine()


def _context(gap_length: int = 10, flank: str = "ACGTACGTAC") -> GapContext:
    return GapContext(
        gap=Gap(gap_id="g1", start=10, end=10 + gap_length),
        left_flank=flank,
        right_flank=flank,
    )


class TestDeterminism:
    def test_the_same_evidence_always_gives_the_same_confidence(
        self, engine: ConfidenceEngine
    ) -> None:
        scores = CandidateScores(homology=0.9, alignment=0.8, conservation=0.7, evolutionary=0.6)
        assert engine.confidence(scores) == engine.confidence(scores)

    def test_confidence_stays_within_bounds(self, engine: ConfidenceEngine) -> None:
        perfect = CandidateScores(
            homology=1.0, alignment=1.0, conservation=1.0, evolutionary=1.0, evo2=1.0
        )
        empty = CandidateScores()

        assert engine.confidence(perfect) == pytest.approx(1.0)
        assert engine.confidence(empty) == 0.0

    def test_better_evidence_scores_higher(self, engine: ConfidenceEngine) -> None:
        weak = CandidateScores(homology=0.3, alignment=0.3, conservation=0.3)
        strong = CandidateScores(homology=0.9, alignment=0.9, conservation=0.9)

        assert engine.confidence(strong) > engine.confidence(weak)


class TestEvo2IsOptional:
    def test_an_unconsulted_evo2_does_not_penalise_a_candidate(
        self, engine: ConfidenceEngine
    ) -> None:
        """ "Not asked" and "asked and disagreed" are different findings.

        Evo 2 is invoked only to break a tie, so most candidates never see it.
        Scoring its absence as zero would penalise every candidate whose
        evidence was clear enough not to need arbitration.
        """
        base = CandidateScores(homology=0.9, alignment=0.9, conservation=0.9, evolutionary=0.9)
        disagreed = base.model_copy(update={"evo2": 0.0})

        assert engine.confidence(base) > engine.confidence(disagreed)

    def test_agreement_raises_confidence_and_disagreement_lowers_it(
        self, engine: ConfidenceEngine
    ) -> None:
        base = CandidateScores(homology=0.6, alignment=0.6, conservation=0.6, evolutionary=0.6)

        agrees = engine.confidence(base.model_copy(update={"evo2": 1.0}))
        disagrees = engine.confidence(base.model_copy(update={"evo2": 0.0}))

        assert disagrees < engine.confidence(base) < agrees

    def test_evo2_cannot_outvote_clear_homology(self, engine: ConfidenceEngine) -> None:
        """A modelling opinion must not override observed evidence.

        Strong homology with Evo 2 disagreeing still beats weak homology with
        Evo 2 agreeing.
        """
        strong_but_doubted = CandidateScores(
            homology=0.95, alignment=0.95, conservation=0.95, evolutionary=0.95, evo2=0.0
        )
        weak_but_endorsed = CandidateScores(
            homology=0.2, alignment=0.2, conservation=0.2, evolutionary=0.2, evo2=1.0
        )

        assert engine.confidence(strong_but_doubted) > engine.confidence(weak_but_endorsed)


class TestValidationGates:
    def test_failed_validation_collapses_the_confidence(self, engine: ConfidenceEngine) -> None:
        """Validation gates rather than trades off.

        A sequence that fails a biological check must not be rescued by scoring
        well on every other signal.
        """
        excellent = CandidateScores(homology=1.0, alignment=1.0, conservation=1.0, evolutionary=1.0)
        assert engine.confidence(excellent.model_copy(update={"validation": 0.0})) == 0.0


class TestReportedBands:
    @pytest.mark.parametrize(
        ("confidence", "expected"),
        [
            (0.95, ConfidenceLevel.HIGH),
            (0.75, ConfidenceLevel.HIGH),
            (0.5, ConfidenceLevel.MEDIUM),
            (0.2, ConfidenceLevel.LOW),
        ],
    )
    def test_bands_follow_the_thresholds(
        self, engine: ConfidenceEngine, confidence: float, expected: ConfidenceLevel
    ) -> None:
        assert engine.level(confidence) is expected

    def test_below_the_floor_a_candidate_is_not_returnable(self) -> None:
        """The honest answer below the floor is that the region is unresolved.

        Returning a low-confidence sequence with a warning invites it to be
        used anyway.
        """
        engine = ConfidenceEngine(thresholds=ConfidenceThresholds(minimum=0.15))

        assert engine.is_returnable(0.14) is False
        assert engine.is_returnable(0.15) is True


class TestComponentScores:
    def test_support_depth_saturates(self) -> None:
        """The fifth agreeing homologue matters; the fiftieth does not."""
        five = homology_score(identity=0.9, coverage=1.0, supporting_hits=5)
        fifty = homology_score(identity=0.9, coverage=1.0, supporting_hits=50)

        assert five == pytest.approx(fifty)

    def test_identity_and_coverage_both_bind(self) -> None:
        """A perfect match over a tenth of the query is not good evidence."""
        full = homology_score(identity=1.0, coverage=1.0, supporting_hits=5)
        shallow = homology_score(identity=1.0, coverage=0.1, supporting_hits=5)

        assert shallow < full

    def test_alignment_scores_zero_when_nothing_spans_the_gap(self) -> None:
        """Whatever else the alignment looks like."""
        assert alignment_score(spanning_references=0, total_references=40) == 0.0

    def test_alignment_scores_the_spanning_fraction(self) -> None:
        assert alignment_score(spanning_references=5, total_references=10) == 0.5


class TestBiologicalChecks:
    def test_an_ambiguous_base_is_fatal(self) -> None:
        """A sequence that is itself unresolved answers nothing."""
        verdicts = validate("ACGTNACGTA", _context())
        alphabet = next(v for v in verdicts if v.check == "alphabet")

        assert alphabet.passed is False
        assert aggregate_score(verdicts) == 0.0

    def test_a_clean_fill_passes_every_check(self) -> None:
        verdicts = validate("ACGTACGTAC", _context())

        assert all(v.passed for v in verdicts)
        assert aggregate_score(verdicts) > 0.9

    def test_a_wrong_length_lowers_the_score_without_being_fatal(self) -> None:
        """A real indel legitimately changes a gap length."""
        verdicts = validate("ACGTACGTACGTACGTACGTACGT", _context(gap_length=10))
        length = next(v for v in verdicts if v.check == "length")

        assert length.passed is False
        assert 0.0 < aggregate_score(verdicts) < 1.0

    def test_a_homopolymer_is_rejected(self) -> None:
        """The shape a degenerate consensus takes when evidence has collapsed."""
        verdicts = validate("AAAAAAAAAA", _context())
        complexity = next(v for v in verdicts if v.check == "low_complexity")

        assert complexity.passed is False

    def test_composition_unlike_the_flanks_is_flagged(self) -> None:
        verdicts = validate("GCGCGCGCGC", _context(flank="ATATATATAT"))
        gc = next(v for v in verdicts if v.check == "gc_consistency")

        assert gc.passed is False

    def test_an_empty_sequence_is_not_a_reconstruction(self) -> None:
        verdicts = validate("", _context())
        assert aggregate_score(verdicts) == 0.0

    def test_thresholds_are_configurable(self) -> None:
        """Nothing here hardcodes where a line is drawn."""
        strict = ValidationThresholds(length_tolerance=0.0)
        verdicts = validate("ACGTACGTA", _context(gap_length=10), strict)

        assert next(v for v in verdicts if v.check == "length").passed is False
