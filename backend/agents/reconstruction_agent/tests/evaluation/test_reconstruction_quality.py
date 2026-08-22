"""Scores the agent's reconstruction against a known answer.

The method: take a complete sequence, punch a hole in it, and check whether
the agent puts back what was there. Accuracy is the only measure that matters
for this agent, and it cannot be read off unit tests of the parts.

The harness runs offline against synthetic references. Marked `evaluation` so
`pytest -m "not evaluation"` skips it in a fast loop.
"""
from __future__ import annotations

import pytest

from agent.reasoning.reasoner import Reasoner
from domain.models import AlignedPair, Alignment, Sequence
from domain.policies import ConfidencePolicy
from domain.services import (
    CandidateRanker,
    ContextExtractor,
    GapDetector,
    ReconstructionValidator,
)

pytestmark = pytest.mark.evaluation


def punch_gap(sequence: str, start: int, length: int) -> tuple[str, str]:
    """Replace a span with N, returning the damaged sequence and the truth."""
    return sequence[:start] + "N" * length + sequence[start + length :], sequence[
        start : start + length
    ]


def identity(left: str, right: str) -> float:
    """Fraction of positions that agree, over the shorter of the two."""
    if not left or not right:
        return 0.0
    comparable = min(len(left), len(right))
    matches = sum(
        1
        for a, b in zip(left[:comparable], right[:comparable], strict=True)
        if a == b
    )
    return matches / comparable


@pytest.fixture
def reasoner() -> Reasoner:
    return Reasoner(
        ranker=CandidateRanker(),
        validator=ReconstructionValidator(),
        confidence=ConfidencePolicy(),
    )


class TestReconstructionAccuracy:
    def test_recovers_a_gap_from_agreeing_references(self, reasoner: Reasoner) -> None:
        """The end-to-end claim: identical references restore the truth exactly."""
        original = "ACGTTGCA" * 20
        damaged, truth = punch_gap(original, 80, 8)

        sequence = Sequence.parse("damaged", damaged)
        gap = GapDetector().detect(sequence)[0]
        context = ContextExtractor(flank_length=40).extract(sequence, gap)

        # Three references that all carry the true bases across the gap.
        target_row = "A" * 40 + "-" * 8 + "C" * 40
        alignment = Alignment(
            gap_id=context.identifier,
            pairs=[
                AlignedPair("target", f"REF_{index}", target_row, "A" * 40 + truth + "C" * 40)
                for index in range(3)
            ],
            gap_column_start=40,
            gap_column_end=48,
        )

        candidates = reasoner.build_candidates(context, alignment, [])
        result = reasoner.finalise(context, candidates)

        assert result.reconstructed_sequence == truth
        assert identity(result.reconstructed_sequence or "", truth) == 1.0

    def test_majority_outvotes_a_single_divergent_reference(
        self, reasoner: Reasoner
    ) -> None:
        original = "ACGTTGCA" * 20
        damaged, truth = punch_gap(original, 80, 8)

        sequence = Sequence.parse("damaged", damaged)
        gap = GapDetector().detect(sequence)[0]
        context = ContextExtractor(flank_length=40).extract(sequence, gap)

        # Differs from the truth at every position, so the 3:1 split is
        # uniform across the columns and support is exactly 0.75. A row that
        # happened to share a base would raise support on that column alone.
        divergent = "TGCAACGT"
        assert all(a != b for a, b in zip(truth, divergent, strict=True))

        target_row = "A" * 40 + "-" * 8 + "C" * 40
        rows = ["A" * 40 + truth + "C" * 40] * 3 + ["A" * 40 + divergent + "C" * 40]
        alignment = Alignment(
            gap_id=context.identifier,
            pairs=[
                AlignedPair("target", f"REF_{index}", target_row, row)
                for index, row in enumerate(rows)
            ],
            gap_column_start=40,
            gap_column_end=48,
        )

        candidates = reasoner.build_candidates(context, alignment, [])
        result = reasoner.finalise(context, candidates)

        assert result.reconstructed_sequence == truth
        # Support is the winner's margin over the runner-up - (3-1)/4 - so the
        # disagreement is recorded rather than hidden behind a 0.75 share.
        assert candidates[0].support == pytest.approx(0.5)

    def test_confidence_tracks_disagreement(self, reasoner: Reasoner) -> None:
        """Scores must fall as the references stop agreeing, or they mislead."""
        original = "ACGTTGCA" * 20
        damaged, truth = punch_gap(original, 80, 8)

        sequence = Sequence.parse("damaged", damaged)
        gap = GapDetector().detect(sequence)[0]
        context = ContextExtractor(flank_length=40).extract(sequence, gap)
        target_row = "A" * 40 + "-" * 8 + "C" * 40

        def score_for(rows: list[str]) -> float:
            alignment = Alignment(
                gap_id=context.identifier,
                pairs=[
                    AlignedPair("target", f"REF_{index}", target_row, row)
                    for index, row in enumerate(rows)
                ],
                gap_column_start=40,
                gap_column_end=48,
            )
            candidates = reasoner.build_candidates(context, alignment, [])
            return candidates[0].confidence or 0.0

        unanimous = score_for(["A" * 40 + truth + "C" * 40] * 4)
        split = score_for(
            ["A" * 40 + truth + "C" * 40] * 2 + ["A" * 40 + "TTTTTTTT" + "C" * 40] * 2
        )

        assert split < unanimous
