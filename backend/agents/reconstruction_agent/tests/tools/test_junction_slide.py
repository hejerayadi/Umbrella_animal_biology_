"""Reading the fill when the aligner slides the indel.

The last defect in the evidence path, and the subtlest. BLAST returned fifty
references all carrying the missing segment, MAFFT aligned eight of them and
inserted exactly the right 45 columns - and the run still reported "no reference
sequence aligned across gap_1", because the mapper demanded those columns begin
at *exactly* the flank junction.

They began one base early. The left flank of the polar bear case ends in `A` and
the removed segment `TTTG...CGTA` also ends in `A`, so two placements of the
indel describe the identical sequence and MAFFT is free to choose either. The
strings below are that real alignment in miniature.
"""
from __future__ import annotations

from domain.models import Gap, GapContext
from domain.services import CandidateRanker
from tools.mafft.mapper import to_alignment

#: `AAAA` + `TTTGA` + `CCCC`, with the fill ending in the same base the left
#: flank ends in - which is what makes the placement ambiguous.
LEFT = "GGGA"
FILL = "TTTGA"
RIGHT = "CCCC"


def aligned_fasta(target_row: str, *reference_rows: str) -> str:
    rows = [f">target\n{target_row}"]
    rows += [f">REF_{i}\n{row}" for i, row in enumerate(reference_rows)]
    return "\n".join(rows) + "\n"


def alignment_for(target_row: str, *reference_rows: str):
    return to_alignment(
        aligned_fasta(target_row, *reference_rows),
        gap_id="gap_1",
        target_id="target",
        left_flank_length=len(LEFT),
    )


def context() -> GapContext:
    return GapContext(
        gap=Gap(identifier="gap_1", start=len(LEFT), end=len(LEFT) + len(FILL)),
        left_flank=LEFT,
        right_flank=RIGHT,
    )


class TestExactPlacement:
    def test_a_run_at_the_junction_is_found(self) -> None:
        alignment = alignment_for(
            "GGGA-----CCCC",
            "GGGATTTGACCCC",
        )
        assert alignment.spans_gap
        assert alignment.junction_slide == 0
        assert alignment.gap_column_start == 4

    def test_the_fill_is_read_out_unchanged(self) -> None:
        alignment = alignment_for("GGGA-----CCCC", "GGGATTTGACCCC")
        candidate = CandidateRanker().build_consensus(context(), alignment, [])
        assert candidate is not None
        assert candidate.sequence == FILL


class TestSlidPlacement:
    """The real failure: `GGG` + run + `A`, instead of `GGGA` + run."""

    def test_a_run_one_base_early_is_still_found(self) -> None:
        alignment = alignment_for(
            "GGG-----ACCCC",
            "GGGATTTGACCCC",
        )
        assert alignment.spans_gap, "the run slid one base and was missed entirely"
        assert alignment.junction_slide == 1

    def test_the_fill_is_rotated_back_between_the_flanks(self) -> None:
        """The columns hold `ATTTG`; the bases actually missing are `TTTGA`.

        Both describe the same sequence once spliced, but only the rotated form
        can be compared against the truth or handed back as "the missing bases".
        """
        alignment = alignment_for("GGG-----ACCCC", "GGGATTTGACCCC")
        candidate = CandidateRanker().build_consensus(context(), alignment, [])
        assert candidate is not None
        assert candidate.sequence == FILL

    def test_the_rotated_fill_reconstructs_the_original(self) -> None:
        """The property that matters: flanks + fill == the sequence removed from."""
        alignment = alignment_for("GGG-----ACCCC", "GGGATTTGACCCC")
        candidate = CandidateRanker().build_consensus(context(), alignment, [])
        assert candidate is not None
        assert LEFT + candidate.sequence + RIGHT == LEFT + FILL + RIGHT

    def test_a_run_placed_late_is_also_recovered(self) -> None:
        """The mirror case - the aligner may slide either way."""
        alignment = alignment_for(
            "GGGAT-----CCC",
            "GGGATTTGACCCC",
        )
        assert alignment.spans_gap
        assert alignment.junction_slide == -1


class TestStillHonestWhenThereIsNothing:
    def test_an_alignment_with_no_inserted_run_does_not_span(self) -> None:
        """The references agree the target is missing nothing here."""
        alignment = alignment_for("GGGACCCC", "GGGACCCC")
        assert not alignment.spans_gap

    def test_a_run_far_from_the_junction_is_not_claimed_as_the_gap(self) -> None:
        """An indel elsewhere in the flank is not the missing segment, and
        reading it as one would fabricate a fill from unrelated columns."""
        target = "GGGA" + "C" * 40 + "-----" + "CCCC"
        reference = "GGGA" + "C" * 40 + "TTTGA" + "CCCC"
        alignment = alignment_for(target, reference)
        assert not alignment.spans_gap
