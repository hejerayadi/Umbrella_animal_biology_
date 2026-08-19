"""The rebuilt confidence score, and the properties it has to hold.

The score exists to rank correct reconstructions above incorrect ones. Measured
on the sweep, the previous formula managed 0.575 where 0.5 means no signal at
all - so raising the threshold discarded good answers at nearly the same rate
as bad ones. These tests pin the properties that produced the rebuilt score's
0.998, so a future tuning pass cannot quietly undo them.
"""
from __future__ import annotations

import pytest

from domain.models import Candidate, Gap, GapContext, Reference
from domain.policies import ConfidencePolicy, ReferenceQualityPolicy

FLANK = "ACGT" * 20


def context_for(gap_length: int = 30, *, both_flanks: bool = True) -> GapContext:
    return GapContext(
        gap=Gap("gap_1", 80, 80 + gap_length),
        left_flank=FLANK,
        right_flank=FLANK if both_flanks else "",
    )


def candidate_with(support: float, references: int = 3) -> Candidate:
    return Candidate(
        gap_id="gap_1",
        sequence="ACGT" * 5,
        support=support,
        supporting_references=[f"REF_{i}" for i in range(references)],
    )


def reference(accession: str, **kwargs: object) -> Reference:
    return Reference(accession=accession, **kwargs)  # type: ignore[arg-type]


class TestAmbiguityIsAFactorNotATerm:
    """Every confidently-wrong answer in the sweep was an evenly split vote."""

    def test_a_coin_flip_is_crushed_however_good_the_flanks_are(self) -> None:
        policy = ConfidencePolicy()
        split = policy.score(candidate_with(0.02), context_for(), mean_identity=1.0)
        decisive = policy.score(candidate_with(1.0), context_for(), mean_identity=1.0)

        assert split < decisive / 2, (
            "averaged against strong identity, a split consensus used to score respectably"
        )

    def test_the_penalty_is_continuous_between_the_two_ends(self) -> None:
        policy = ConfidencePolicy()
        scores = [
            policy.score(candidate_with(support), context_for(), mean_identity=0.9)
            for support in (0.1, 0.2, 0.3, 0.4, 0.5)
        ]

        assert scores == sorted(scores), "confidence must rise with decisiveness"

    def test_a_decisive_consensus_pays_no_ambiguity_penalty(self) -> None:
        policy = ConfidencePolicy()

        assert policy._decisiveness_factor(0.5) == 1.0
        assert policy._decisiveness_factor(1.0) == 1.0


class TestDepthNoLongerBlocksGoodEvidence:
    def test_two_excellent_references_can_clear_a_usable_threshold(self) -> None:
        """Previously impossible: two references capped the score at 0.625."""
        policy = ConfidencePolicy()
        score = policy.score(
            candidate_with(1.0, references=2),
            context_for(),
            mean_identity=0.98,
            references=[
                reference("REF_0", identity=0.98, coverage=0.95, relatedness=0.9),
                reference("REF_1", identity=0.97, coverage=0.95, relatedness=0.9),
            ],
        )

        assert score > 0.65

    def test_a_lone_reference_is_still_capped_below_certainty(self) -> None:
        policy = ConfidencePolicy()
        score = policy.score(
            candidate_with(1.0, references=1), context_for(), mean_identity=1.0
        )

        assert score < 1.0

    def test_no_references_scores_nothing(self) -> None:
        policy = ConfidencePolicy()
        score = policy.score(
            candidate_with(1.0, references=0), context_for(), mean_identity=1.0
        )

        assert score == 0.0


class TestEvidenceQualityMovesTheScore:
    """"Five sequences agreed" and "five good sequences agreed" must differ."""

    def test_well_matched_close_relatives_beat_distant_poor_matches(self) -> None:
        policy = ConfidencePolicy()
        shared = dict(mean_identity=0.9)

        strong = policy.score(
            candidate_with(0.9),
            context_for(),
            references=[
                reference(f"REF_{i}", identity=0.98, coverage=0.95, relatedness=0.95)
                for i in range(3)
            ],
            **shared,  # type: ignore[arg-type]
        )
        weak = policy.score(
            candidate_with(0.9),
            context_for(),
            references=[
                reference(f"REF_{i}", identity=0.72, coverage=0.4, relatedness=0.3)
                for i in range(3)
            ],
            **shared,  # type: ignore[arg-type]
        )

        assert strong > weak

    def test_unmeasured_references_are_neutral_not_bad(self) -> None:
        """An NCBI record carries no alignment statistics; that is not a fault."""
        policy = ConfidencePolicy()
        unmeasured = policy.score(
            candidate_with(0.9),
            context_for(),
            mean_identity=0.9,
            references=[reference(f"REF_{i}") for i in range(3)],
        )

        assert unmeasured > 0.4

    def test_only_the_supporting_references_are_read(self) -> None:
        """A sequence that voted against the fill says nothing good about it."""
        policy = ConfidencePolicy()
        candidate = Candidate(
            gap_id="gap_1",
            sequence="ACGT",
            support=0.9,
            supporting_references=["REF_GOOD"],
        )

        score = policy.score(
            candidate,
            context_for(),
            mean_identity=0.9,
            references=[
                reference("REF_GOOD", identity=0.99, coverage=0.99, relatedness=0.99),
                reference("REF_DISSENT", identity=0.1, coverage=0.1, relatedness=0.1),
            ],
        )
        only_good = policy.score(
            candidate,
            context_for(),
            mean_identity=0.9,
            references=[
                reference("REF_GOOD", identity=0.99, coverage=0.99, relatedness=0.99)
            ],
        )

        assert score == pytest.approx(only_good)


class TestLengthAgreesWithTheAdmissionPolicy:
    def test_a_gap_the_agent_accepts_can_still_be_reported(self) -> None:
        """At scale 2000 against a 5000 cap, the two policies contradicted."""
        policy = ConfidencePolicy()
        score = policy.score(
            candidate_with(1.0, references=5),
            context_for(gap_length=1500),
            mean_identity=0.98,
            references=[
                reference(f"REF_{i}", identity=0.98, coverage=0.95, relatedness=0.9)
                for i in range(5)
            ],
        )

        assert score > 0.5

    def test_longer_gaps_still_score_lower(self) -> None:
        policy = ConfidencePolicy()
        short = policy.score(candidate_with(1.0), context_for(30), mean_identity=0.9)
        long = policy.score(candidate_with(1.0), context_for(3000), mean_identity=0.9)

        assert short > long


class TestPlausibilityCanOnlyDoubt:
    """A language model must not manufacture confidence the alignment lacks."""

    def test_agreement_does_not_raise_the_score(self) -> None:
        policy = ConfidencePolicy()
        without = policy.score(candidate_with(0.9), context_for(), mean_identity=0.9)
        with_agreement = policy.score(
            candidate_with(0.9), context_for(), mean_identity=0.9, plausibility=1.0
        )

        assert with_agreement == pytest.approx(without)

    def test_disagreement_lowers_it(self) -> None:
        policy = ConfidencePolicy()
        without = policy.score(candidate_with(0.9), context_for(), mean_identity=0.9)
        doubted = policy.score(
            candidate_with(0.9), context_for(), mean_identity=0.9, plausibility=0.05
        )

        assert doubted < without

    def test_it_cannot_zero_a_well_evidenced_fill(self) -> None:
        """Evo 2 is a check, not a veto - the alignment evidence still counts."""
        policy = ConfidencePolicy()
        without = policy.score(candidate_with(0.9), context_for(), mean_identity=0.9)
        doubted = policy.score(
            candidate_with(0.9), context_for(), mean_identity=0.9, plausibility=0.0
        )

        assert doubted >= without * 0.5


class TestReferenceQuality:
    def test_a_pseudogene_is_discounted(self) -> None:
        policy = ReferenceQualityPolicy()
        pseudogene = reference("REF_1", description="MT-CO1 pseudogene 58 (MTCO1P58)")

        assert policy.penalty(pseudogene) < 0.5

    def test_a_numt_is_discounted(self) -> None:
        """The failure seen live: NUMTs are what NCBI returns for MT-CO1."""
        policy = ReferenceQualityPolicy()
        numt = reference("REF_1", description="nuclear mitochondrial insertion, NUMT")

        assert policy.penalty(numt) < 0.5

    def test_a_partial_record_is_only_mildly_discounted(self) -> None:
        policy = ReferenceQualityPolicy()
        partial = reference("REF_1", description="cytochrome oxidase, partial cds")

        assert 0.5 < policy.penalty(partial) < 1.0

    def test_a_curated_complete_record_is_not_penalised(self) -> None:
        policy = ReferenceQualityPolicy()
        clean = reference("REF_1", description="cytochrome c oxidase subunit I, complete cds")

        assert policy.penalty(clean) == 1.0

    def test_an_implausibly_short_record_cannot_carry_the_fill(self) -> None:
        policy = ReferenceQualityPolicy()
        stub = reference("REF_1", description="complete cds", residues="ACGT")

        assert policy.penalty(stub, expected_length=1000) < 1.0

    def test_length_is_not_judged_before_the_sequence_is_fetched(self) -> None:
        """A hit is not poor quality merely for not having been retrieved yet."""
        policy = ReferenceQualityPolicy()
        unfetched = reference("REF_1", description="complete cds")

        assert policy.penalty(unfetched, expected_length=1000) == 1.0

    def test_a_pseudogene_is_downweighted_rather_than_discarded(self) -> None:
        """It cannot outvote a functional homologue, but it is not nothing.

        Excluding it outright would leave a gap whose only hits are pseudogenes
        with no answer at all. Kept and heavily discounted, it produces a
        low-confidence reconstruction the score reports honestly - which is
        worth more to a biologist than silence.
        """
        policy = ReferenceQualityPolicy()
        pseudogene = reference("REF_1", description="pseudogene, partial")

        assert policy.is_usable(pseudogene)
        assert policy.penalty(pseudogene) < 0.5

    def test_a_pseudogene_of_implausible_length_is_excluded(self) -> None:
        """Two independent reasons to doubt it is enough to leave it out."""
        policy = ReferenceQualityPolicy()
        hopeless = reference("REF_1", description="pseudogene", residues="ACGT")

        assert not policy.is_usable(hopeless, expected_length=5000)

    def test_it_says_why_it_discounted_something(self) -> None:
        policy = ReferenceQualityPolicy()

        assert policy.describe(reference("REF_1", description="a pseudogene")) is not None
        assert policy.describe(reference("REF_2", description="complete cds")) is None
