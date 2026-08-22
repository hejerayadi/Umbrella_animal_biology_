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

    def test_single_reference_scores_below_corroborated_evidence(self) -> None:
        """One reference carrying the answer alone is thin evidence.

        It is no longer *barred* from being reported: the old depth factor
        capped one reference at 0.50 and two at 0.625, which put them under any
        usable threshold however perfect the alignment - so a gap with a single
        excellent homologue was unreconstructable by arithmetic rather than by
        judgement. What must still hold is the ordering: thin evidence scores
        lower than corroborated evidence, and the critic still objects to it
        (see `_THIN_EVIDENCE_REFERENCES` in the critic).
        """
        policy = ConfidencePolicy()
        alone = Candidate(
            gap_id="gap_1", sequence="ACGT", support=1.0, supporting_references=["A"]
        )
        corroborated = Candidate(
            gap_id="gap_1",
            sequence="ACGT",
            support=1.0,
            supporting_references=["A", "B", "C"],
        )

        thin = policy.score(alone, make_context(4), mean_identity=0.99)
        thick = policy.score(corroborated, make_context(4), mean_identity=0.99)

        assert thin < thick

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


class TestAlignmentReferenceSelection:
    """Which references reach MAFFT.

    A reference that stops short of the gap contributes no bases to the columns
    being reconstructed, and pulls the multiple alignment towards the shape with
    no insertion at the junction. One such row was enough to break a real run:
    it covered a single flank at 0.979 identity, so it ranked first, and the
    alignment came back with no gap columns while three genuine donors sat in
    the same input carrying the correct fill.
    """

    @staticmethod
    def _context(gap_length: int = 21):
        from domain.models import Sequence
        from domain.services import ContextExtractor, GapDetector

        residues = "ACGT" * 30 + "N" * gap_length + "ACGT" * 30
        sequence = Sequence.parse("t", residues)
        gaps = GapDetector().detect(sequence)
        return ContextExtractor().extract_all(sequence, gaps)[0]

    @staticmethod
    def _reference(accession: str, gap_bases: int | None, identity: float = 0.95):
        from domain.models import Reference

        return Reference(
            accession=accession,
            residues="ACGT" * 60,
            identity=identity,
            coverage=1.0,
            source="blast",
            gap_bases=gap_bases,
        )

    def _select(self, references):
        from agent.planning.planner import PlanStep
        from agent.planning.tool_selector import ToolSelector
        from domain.services import ReferenceRanker

        context = self._context()
        state = {"references": {context.identifier: references}, "gap_contexts": [context]}
        selector = ToolSelector(ReferenceRanker())
        invocation = selector.build(
            PlanStep(tool="mafft_align", gap_id=context.identifier), state
        )
        return list(invocation.payload.references) if invocation else []

    def test_non_carriers_are_excluded_when_a_carrier_exists(self) -> None:
        chosen = self._select(
            [
                # Ranks top on identity and carries nothing - the exact profile
                # that broke the real run.
                self._reference("FLANK_ONLY.1", gap_bases=0, identity=0.99),
                self._reference("DONOR_A.1", gap_bases=21, identity=0.90),
                self._reference("DONOR_B.1", gap_bases=21, identity=0.88),
            ]
        )

        assert "FLANK_ONLY.1" not in chosen
        assert set(chosen) == {"DONOR_A.1", "DONOR_B.1"}

    def test_everything_is_kept_when_no_hit_carries_the_gap(self) -> None:
        # Nothing spans it, so there is nothing better to fall back to. The
        # alignment still runs: that is what tells the critic the evidence is
        # absent rather than untried.
        chosen = self._select(
            [
                self._reference("A.1", gap_bases=0),
                self._reference("B.1", gap_bases=0),
            ]
        )

        assert set(chosen) == {"A.1", "B.1"}

    def test_unmeasured_references_are_still_alignable(self) -> None:
        # NCBI records never went through a homology search, so they have no
        # measurement. They must not be discarded for lacking one.
        chosen = self._select(
            [
                self._reference("NCBI_A.1", gap_bases=None),
                self._reference("NCBI_B.1", gap_bases=None),
            ]
        )

        assert set(chosen) == {"NCBI_A.1", "NCBI_B.1"}
