"""Confidence scoring, validation, and which gaps get attempted."""
from __future__ import annotations

import pytest

from contracts.output import ReconstructionStatus
from domain.models import Candidate, Gap, GapContext
from domain.policies import ConfidencePolicy, ValidationPolicy
from domain.services import ReconstructionValidator


def make_context(length: int = 100, *, left: int = 200, right: int = 200) -> GapContext:
    return GapContext(
        gap=Gap("gap_1", 500, 500 + length),
        left_flank="A" * left,
        right_flank="C" * right,
    )


class TestConfidencePolicy:
    def test_strong_evidence_scores_above_the_threshold(self) -> None:
        policy = ConfidencePolicy()
        candidate = Candidate(
            gap_id="gap_1",
            sequence="ACGT" * 10,
            support=0.95,
            supporting_references=["A", "B", "C", "D", "E"],
        )

        score = policy.score(candidate, make_context(40), mean_identity=0.95)

        assert score >= policy.minimum_confidence
        assert policy.classify(score) is ReconstructionStatus.RECONSTRUCTED

    def test_single_reference_scores_below_the_threshold(self) -> None:
        """One reference carrying the answer alone is thin evidence."""
        policy = ConfidencePolicy()
        candidate = Candidate(
            gap_id="gap_1", sequence="ACGT", support=1.0, supporting_references=["A"]
        )

        score = policy.score(candidate, make_context(4), mean_identity=0.99)

        assert policy.classify(score) is ReconstructionStatus.LOW_CONFIDENCE

    def test_longer_gaps_score_lower_on_identical_evidence(self) -> None:
        """Flanking evidence constrains a long span less than a short one."""
        policy = ConfidencePolicy()
        candidate = Candidate(
            gap_id="gap_1",
            sequence="ACGT" * 10,
            support=0.9,
            supporting_references=["A", "B", "C"],
        )

        short = policy.score(candidate, make_context(50), mean_identity=0.9)
        long = policy.score(candidate, make_context(1500), mean_identity=0.9)

        assert long < short

    def test_length_penalty_never_reaches_zero(self) -> None:
        """A long gap with good evidence stays reportable, just not confident."""
        policy = ConfidencePolicy()
        candidate = Candidate(
            gap_id="gap_1", sequence="A" * 100, support=1.0, supporting_references=["A", "B"]
        )

        assert policy.score(candidate, make_context(100_000), mean_identity=1.0) > 0.0

    def test_one_flank_scores_lower_than_two(self) -> None:
        policy = ConfidencePolicy()
        candidate = Candidate(
            gap_id="gap_1", sequence="ACGT", support=0.9, supporting_references=["A", "B"]
        )

        both = policy.score(candidate, make_context(4), mean_identity=0.9)
        one = policy.score(candidate, make_context(4, right=0), mean_identity=0.9)

        assert one < both

    def test_aggregate_takes_the_weakest_gap(self) -> None:
        """A reconstruction is only as good as its weakest filled gap."""
        assert ConfidencePolicy().aggregate([0.9, 0.6, 0.95]) == pytest.approx(0.6)

    def test_aggregate_of_nothing_is_zero(self) -> None:
        assert ConfidencePolicy().aggregate([]) == 0.0


class TestValidationPolicy:
    def test_attempts_a_normal_gap(self) -> None:
        attempt, reason = ValidationPolicy().should_attempt(make_context(100))

        assert attempt
        assert reason is None

    def test_declines_a_gap_over_the_length_cap(self) -> None:
        attempt, reason = ValidationPolicy(max_gap_length=1000).should_attempt(make_context(5000))

        assert not attempt
        assert reason is not None and "5000" in reason

    def test_declines_a_gap_without_usable_flanks(self) -> None:
        context = GapContext(gap=Gap("gap_1", 0, 100), left_flank="AC", right_flank="GT")

        attempt, reason = ValidationPolicy().should_attempt(context)

        assert not attempt
        assert reason is not None and "flank" in reason.lower()


class TestReconstructionValidator:
    def test_accepts_a_plausible_reconstruction(self) -> None:
        candidate = Candidate(
            gap_id="gap_1", sequence="ACGTACGTAC", supporting_references=["A", "B"]
        )

        report = ReconstructionValidator().validate(candidate, make_context(10))

        assert report.is_valid
        assert not report.warnings

    def test_rejects_a_reconstruction_containing_unknowns(self) -> None:
        """Filling a gap of N with N does not resolve it."""
        candidate = Candidate(gap_id="gap_1", sequence="ACGTNNACGT")

        report = ReconstructionValidator().validate(candidate, make_context(10))

        assert not report.is_valid
        assert any("unknown" in error.lower() for error in report.errors)

    def test_rejects_a_length_far_from_the_gap(self) -> None:
        candidate = Candidate(gap_id="gap_1", sequence="ACGT")

        report = ReconstructionValidator(length_tolerance=0.5).validate(
            candidate, make_context(100)
        )

        assert not report.is_valid

    def test_tolerates_a_small_indel(self) -> None:
        """A real indel makes gap width and reconstruction length differ."""
        candidate = Candidate(
            gap_id="gap_1", sequence="ACGT" * 22, supporting_references=["A", "B"]
        )

        report = ReconstructionValidator(length_tolerance=0.5).validate(
            candidate, make_context(100)
        )

        assert report.is_valid

    def test_warns_about_a_long_homopolymer(self) -> None:
        candidate = Candidate(
            gap_id="gap_1", sequence="A" * 30, supporting_references=["A", "B"]
        )

        report = ReconstructionValidator(max_homopolymer_run=20).validate(
            candidate, make_context(30)
        )

        assert report.is_valid  # a warning, not a disqualification
        assert any("homopolymer" in warning for warning in report.warnings)
