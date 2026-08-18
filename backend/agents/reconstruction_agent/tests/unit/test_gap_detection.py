"""Gap detection and context extraction - the agent's first step."""
from __future__ import annotations

import pytest

from reconstruction_agent.domain.exceptions import InvalidSequenceError
from reconstruction_agent.domain.models import Sequence
from reconstruction_agent.domain.services import ContextExtractor, GapDetector


class TestGapDetector:
    def test_finds_each_run_of_unknowns(self, gapped_sequence: Sequence) -> None:
        gaps = GapDetector().detect(gapped_sequence)

        assert [gap.length for gap in gaps] == [10, 5]
        assert [gap.identifier for gap in gaps] == ["gap_1", "gap_2"]

    def test_reports_no_gaps_for_a_complete_sequence(self, ungapped_sequence: Sequence) -> None:
        assert GapDetector().detect(ungapped_sequence) == []

    def test_offsets_slice_out_exactly_the_unknown_run(self, gapped_sequence: Sequence) -> None:
        for gap in GapDetector().detect(gapped_sequence):
            assert set(gapped_sequence.residues[gap.start : gap.end]) == {"N"}

    def test_minimum_length_filters_incidental_ambiguity(self) -> None:
        # A single N is sequencing noise, not an assembly gap.
        sequence = Sequence.parse("noisy", "ACGT" * 10 + "N" + "ACGT" * 10 + "NNNN" + "ACGT" * 10)

        assert len(GapDetector(minimum_length=1).detect(sequence)) == 2
        assert len(GapDetector(minimum_length=2).detect(sequence)) == 1

    def test_other_iupac_codes_are_not_gaps(self) -> None:
        """R and Y express constrained uncertainty and carry information."""
        sequence = Sequence.parse("ambiguous", "ACGT" * 10 + "RYSW" + "ACGT" * 10)

        assert GapDetector().detect(sequence) == []


class TestSequenceParsing:
    def test_normalises_whitespace_and_case(self) -> None:
        sequence = Sequence.parse("wrapped", "acgt\nACGT\n  acgt  ")

        assert sequence.residues == "ACGTACGTACGT"

    def test_rejects_non_nucleotide_characters(self) -> None:
        with pytest.raises(InvalidSequenceError, match="non-nucleotide"):
            Sequence.parse("protein", "MKVLAAGIVGL")

    def test_rejects_an_empty_sequence(self) -> None:
        with pytest.raises(InvalidSequenceError):
            Sequence.parse("empty", "   \n  ")

    def test_completeness_reflects_the_unknown_fraction(self) -> None:
        sequence = Sequence.parse("half", "ACGTACGTNN")

        assert sequence.unknown_count == 2
        assert sequence.completeness == pytest.approx(0.8)


class TestContextExtractor:
    def test_extracts_flanks_on_both_sides(self, gapped_sequence: Sequence) -> None:
        gaps = GapDetector().detect(gapped_sequence)
        context = ContextExtractor(flank_length=50).extract(gapped_sequence, gaps[0])

        assert len(context.left_flank) == 50
        assert len(context.right_flank) == 50
        assert context.has_both_flanks

    def test_flank_stops_at_a_neighbouring_gap(self, gapped_sequence: Sequence) -> None:
        """Two nearby gaps must not pollute each other's search query."""
        gaps = GapDetector().detect(gapped_sequence)
        # A flank window wide enough to reach across the first gap.
        context = ContextExtractor(flank_length=200).extract(gapped_sequence, gaps[1])

        assert "N" not in context.left_flank
        assert "N" not in context.right_flank

    def test_query_sequence_omits_the_gap_itself(self, gapped_sequence: Sequence) -> None:
        """N scores as a mismatch, penalising the references we most want."""
        gaps = GapDetector().detect(gapped_sequence)
        context = ContextExtractor(flank_length=50).extract(gapped_sequence, gaps[0])

        query = context.query_sequence()
        assert "N" not in query
        assert query == context.left_flank + context.right_flank

    def test_short_flanks_are_marked_unusable(self) -> None:
        sequence = Sequence.parse("tiny", "ACGT" + "N" * 10 + "ACGT")
        gaps = GapDetector().detect(sequence)
        context = ContextExtractor(flank_length=500, minimum_flank=50).extract(sequence, gaps[0])

        assert not context.has_usable_flanks
