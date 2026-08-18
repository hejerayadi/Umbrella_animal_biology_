"""Consensus building, ranking, and the stop policy."""
from __future__ import annotations

import pytest

from agent.planning.stop_policy import StopPolicy, StopReason
from agent.reasoning.evidence_synthesizer import EvidenceSynthesizer
from agent.reasoning.reasoner import Reasoner
from contracts.observation import Observation, ObservationStatus
from contracts.output import GapReconstruction, ReconstructionStatus
from domain.models import (
    AlignedPair,
    Alignment,
    Gap,
    GapContext,
    Reference,
    Sequence,
)
from domain.policies import ConfidencePolicy
from domain.services import CandidateRanker, ReconstructionValidator


def make_context(length: int = 4) -> GapContext:
    return GapContext(
        gap=Gap("gap_1", 4, 4 + length), left_flank="A" * 100, right_flank="C" * 100
    )


def solid(gap_id: str = "gap_1") -> GapReconstruction:
    """A gap whose outcome is genuinely settled - reconstructed, not merely present."""
    return GapReconstruction(
        gap_id=gap_id,
        start=4,
        end=8,
        length=4,
        status=ReconstructionStatus.RECONSTRUCTED,
        reconstructed_sequence="GGGG",
        confidence=0.9,
    )


def make_alignment(*reference_rows: tuple[str, str]) -> Alignment:
    """An alignment whose gap sits at columns 4..8."""
    target = "AAAA----CCCC"
    return Alignment(
        gap_id="gap_1",
        pairs=[
            AlignedPair("target", accession, target, row) for accession, row in reference_rows
        ],
        gap_column_start=4,
        gap_column_end=8,
    )


class TestCandidateRanker:
    def test_unanimous_references_give_full_support(self) -> None:
        alignment = make_alignment(("REF_1", "AAAAGGGGCCCC"), ("REF_2", "AAAAGGGGCCCC"))

        candidate = CandidateRanker().build_consensus(make_context(), alignment, [])

        assert candidate is not None
        assert candidate.sequence == "GGGG"
        assert candidate.support == pytest.approx(1.0)

    def test_disagreement_is_carried_into_support(self) -> None:
        """Three say G, one says T: the majority wins, and support is the
        margin over the runner-up - (3-1)/4 - not the winner's 0.75 share."""
        alignment = make_alignment(
            ("REF_1", "AAAAGGGGCCCC"),
            ("REF_2", "AAAAGGGGCCCC"),
            ("REF_3", "AAAAGGGGCCCC"),
            ("REF_4", "AAAATTTTCCCC"),
        )

        candidate = CandidateRanker().build_consensus(make_context(), alignment, [])

        assert candidate is not None
        assert candidate.sequence == "GGGG"
        assert candidate.support == pytest.approx(0.5)

    def test_a_near_tie_scores_close_to_zero_support(self) -> None:
        """Three vs two is nearly a coin flip and must not read as 0.6.

        Every confidently-wrong case in the tuning sweep had this shape.
        """
        alignment = make_alignment(
            ("REF_1", "AAAAGGGGCCCC"),
            ("REF_2", "AAAAGGGGCCCC"),
            ("REF_3", "AAAAGGGGCCCC"),
            ("REF_4", "AAAATTTTCCCC"),
            ("REF_5", "AAAATTTTCCCC"),
        )

        candidate = CandidateRanker().build_consensus(make_context(), alignment, [])

        assert candidate is not None
        assert candidate.support == pytest.approx(0.2)

    def test_alignment_not_spanning_the_gap_yields_no_candidate(self) -> None:
        """Inventing a filling from the flanks alone would be fabrication."""
        alignment = Alignment(gap_id="gap_1", pairs=[], gap_column_start=None)

        assert CandidateRanker().build_consensus(make_context(), alignment, []) is None

    def test_records_which_references_contributed(self) -> None:
        alignment = make_alignment(("REF_1", "AAAAGGGGCCCC"), ("REF_2", "AAAAGGGGCCCC"))
        references = [Reference(accession="REF_1", organism="Testus organismus")]

        candidate = CandidateRanker().build_consensus(make_context(), alignment, references)

        assert candidate is not None
        assert candidate.supporting_references == ["REF_1", "REF_2"]
        assert candidate.evidence[0].organism == "Testus organismus"


class TestReasoner:
    @pytest.fixture
    def reasoner(self) -> Reasoner:
        return Reasoner(
            ranker=CandidateRanker(),
            validator=ReconstructionValidator(),
            confidence=ConfidencePolicy(),
        )

    def test_produces_a_scored_reconstruction(self, reasoner: Reasoner) -> None:
        alignment = make_alignment(
            ("REF_1", "AAAAGGGGCCCC"),
            ("REF_2", "AAAAGGGGCCCC"),
            ("REF_3", "AAAAGGGGCCCC"),
        )
        context = make_context()

        candidates = reasoner.build_candidates(context, alignment, [])
        result = reasoner.finalise(context, candidates)

        assert result.reconstructed_sequence == "GGGG"
        assert result.confidence > 0.0
        assert result.explanation is not None

    def test_reports_unresolved_when_there_are_no_candidates(self, reasoner: Reasoner) -> None:
        """An unresolvable gap is a result, not an absence."""
        context = make_context()

        result = reasoner.finalise(context, [])

        assert result.status is ReconstructionStatus.UNRESOLVED
        assert result.reconstructed_sequence is None
        assert result.explanation is not None


class TestStopPolicy:
    def test_stops_when_there_is_nothing_to_do(self) -> None:
        should_stop, reason = StopPolicy().should_stop({"gap_contexts": []})

        assert should_stop
        assert reason is StopReason.NOTHING_TO_DO

    def test_stops_once_every_gap_is_resolved(self) -> None:
        """ALL_RESOLVED means reconstructed, not merely 'has an entry'."""
        state = {
            "gap_contexts": [make_context()],
            "skipped": {},
            "reconstructions": {"gap_1": solid()},
            "iteration": 1,
            "max_iterations": 6,
        }

        should_stop, reason = StopPolicy().should_stop(state)

        assert should_stop
        assert reason is StopReason.ALL_RESOLVED

    def test_an_unresolved_entry_is_not_all_resolved(self) -> None:
        """The old policy called this ALL_RESOLVED, which was a lie: the gap
        has an entry, but that entry says it could not be reconstructed."""
        unresolved = GapReconstruction(
            gap_id="gap_1",
            start=100,
            end=140,
            length=40,
            status=ReconstructionStatus.UNRESOLVED,
        )
        state = {
            "gap_contexts": [make_context()],
            "skipped": {},
            "reconstructions": {"gap_1": unresolved},
            "verdicts": {},
            "observations": [
                Observation(tool="mafft_align", status=ObservationStatus.OK,
                            iteration=0, evidence_added=2),
            ],
            "iteration": 1,
            "max_iterations": 6,
        }

        should_stop, reason = StopPolicy().should_stop(state)

        assert reason is not StopReason.ALL_RESOLVED
        assert not should_stop, "an unresolved gap is still open work"

    def test_stops_at_the_iteration_ceiling(self) -> None:
        state = {
            "gap_contexts": [make_context()],
            "skipped": {},
            "reconstructions": {},
            "references": {"gap_1": [Reference(accession="A")]},
            "iteration": 6,
            "max_iterations": 6,
        }

        should_stop, reason = StopPolicy().should_stop(state)

        assert should_stop
        assert reason is StopReason.MAX_ITERATIONS

    def test_stops_when_an_iteration_found_nothing(self) -> None:
        state = {
            "gap_contexts": [make_context()],
            "skipped": {},
            "reconstructions": {},
            "references": {},
            "candidates": {},
            "iteration": 1,
            "max_iterations": 6,
        }

        should_stop, reason = StopPolicy().should_stop(state)

        assert should_stop
        assert reason is StopReason.NO_PROGRESS

    def test_continues_while_evidence_is_accumulating(self) -> None:
        state = {
            "gap_contexts": [make_context()],
            "skipped": {},
            "reconstructions": {},
            "references": {"gap_1": [Reference(accession="A")]},
            # Progress is measured per round, from the observation trail - not
            # from accumulated `references`, which stays truthy forever once
            # any round succeeds.
            "observations": [
                Observation(tool="blast_search", gap_id="gap_1",
                            status=ObservationStatus.OK, iteration=0, evidence_added=1),
            ],
            "iteration": 1,
            "max_iterations": 6,
        }

        should_stop, _ = StopPolicy().should_stop(state)

        assert not should_stop


class TestEvidenceSynthesizer:
    def test_splices_confident_reconstructions_into_the_sequence(self) -> None:
        target = Sequence.parse("seq", "AAAA" + "N" * 4 + "CCCC")
        reconstruction = GapReconstruction(
            gap_id="gap_1",
            start=4,
            end=8,
            length=4,
            status=ReconstructionStatus.RECONSTRUCTED,
            reconstructed_sequence="GGGG",
            confidence=0.9,
        )

        applied = EvidenceSynthesizer().apply(target, [reconstruction])

        assert applied.residues == "AAAAGGGGCCCC"

    def test_leaves_unresolved_gaps_untouched(self) -> None:
        target = Sequence.parse("seq", "AAAA" + "N" * 4 + "CCCC")
        reconstruction = GapReconstruction(
            gap_id="gap_1", start=4, end=8, length=4, status=ReconstructionStatus.UNRESOLVED
        )

        applied = EvidenceSynthesizer().apply(target, [reconstruction])

        assert applied.residues == target.residues

    def test_applies_multiple_gaps_without_corrupting_offsets(self) -> None:
        """Right-to-left application keeps earlier offsets valid."""
        target = Sequence.parse("seq", "AAAA" + "NN" + "TTTT" + "NN" + "CCCC")
        reconstructions = [
            GapReconstruction(
                gap_id="gap_1",
                start=4,
                end=6,
                length=2,
                status=ReconstructionStatus.RECONSTRUCTED,
                # Longer than the gap it fills - an indel.
                reconstructed_sequence="GGGGGG",
                confidence=0.9,
            ),
            GapReconstruction(
                gap_id="gap_2",
                start=10,
                end=12,
                length=2,
                status=ReconstructionStatus.RECONSTRUCTED,
                reconstructed_sequence="AA",
                confidence=0.9,
            ),
        ]

        applied = EvidenceSynthesizer().apply(target, reconstructions)

        assert applied.residues == "AAAA" + "GGGGGG" + "TTTT" + "AA" + "CCCC"
