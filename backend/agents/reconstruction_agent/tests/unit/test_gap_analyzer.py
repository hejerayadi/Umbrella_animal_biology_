"""Reading the alignment columns that correspond to a gap.

The slide test below is the one that matters. It reproduces a failure that cost
real debugging time: eight references carried every base of the answer and were
reported as "no reference aligned across the gap", because the inserted columns
began one base earlier than the computed junction.
"""

from __future__ import annotations

from reconstruction_agent.domain.models.alignment import AlignedRow, Alignment
from reconstruction_agent.domain.models.sequence import Gap, GapContext
from reconstruction_agent.services.alignment.gap_analyzer import analyze_gap

LEFT = "AAAA"
RIGHT = "CCCC"
TRUTH = "GGGGG"


def _context() -> GapContext:
    return GapContext(gap=Gap(gap_id="gap_1", start=4, end=9), left_flank=LEFT, right_flank=RIGHT)


def _alignment(*rows: tuple[str, str]) -> Alignment:
    return Alignment(
        query_id="query",
        rows=tuple(AlignedRow(identifier=name, aligned_sequence=seq) for name, seq in rows),
    )


class TestFillExtraction:
    def test_the_inserted_columns_are_read_as_the_fill(self) -> None:
        """The straightforward case: the indel sits exactly at the junction."""
        support = analyze_gap(
            _alignment(
                ("query", "AAAA-----CCCC"),
                ("ref1", "AAAAGGGGGCCCC"),
            ),
            _context(),
        )

        assert support.has_support is True
        assert support.spanning_fills[0].bases == TRUTH

    def test_a_slid_indel_still_yields_the_correct_fill(self) -> None:
        """The aligner may place the indel one base either side of the junction.

        Here the final base of the left flank is aligned *after* the inserted
        run. Requiring an exact junction match would report this reference as
        not spanning the gap, discarding a homologue that carries the complete
        answer. The run is found with a tolerance and the fill re-anchored
        against the flanks, so the same bases come back.
        """
        support = analyze_gap(
            _alignment(
                ("query", "AAA-----ACCCC"),
                ("ref1", "AAAAGGGGGCCCC"),
            ),
            _context(),
        )

        assert support.has_support is True
        assert support.spanning_fills[0].bases == TRUTH

    def test_a_reference_that_does_not_span_contributes_nothing(self) -> None:
        """Matching one flank proves the flank is conserved, nothing more."""
        support = analyze_gap(
            _alignment(
                ("query", "AAAA-----CCCC"),
                ("ref1", "AAAA---------"),
            ),
            _context(),
        )

        assert support.spanning_count == 0
        assert support.has_support is False
        assert support.fills[0].spans_gap is False

    def test_homologues_present_but_none_spanning_is_not_support(self) -> None:
        """The case that must never be mistaken for usable evidence.

        Every reference aligned; not one of them covers the missing region.
        """
        support = analyze_gap(
            _alignment(
                ("query", "AAAA-----CCCC"),
                ("ref1", "AAAA---------"),
                ("ref2", "---------CCCC"),
            ),
            _context(),
        )

        assert len(support.fills) == 2
        assert support.has_support is False


class TestCompetingReferences:
    def test_distinct_fills_are_kept_apart(self) -> None:
        """Disagreement is a finding, not noise to be averaged away.

        Each distinct proposal stays available to become its own candidate.
        """
        support = analyze_gap(
            _alignment(
                ("query", "AAAA-----CCCC"),
                ("ref1", "AAAAGGGGGCCCC"),
                ("ref2", "AAAAGGGGGCCCC"),
                ("ref3", "AAAAGGTGGCCCC"),
            ),
            _context(),
        )

        assert support.distinct_fills() == {"GGGGG": 2, "GGTGG": 1}

    def test_disagreeing_positions_are_reported(self) -> None:
        support = analyze_gap(
            _alignment(
                ("query", "AAAA-----CCCC"),
                ("ref1", "AAAAGGGGGCCCC"),
                ("ref2", "AAAAGGTGGCCCC"),
            ),
            _context(),
        )

        assert support.conflicting_positions == (2,)
        assert 0.0 < support.conservation < 1.0

    def test_unanimous_references_score_full_conservation(self) -> None:
        support = analyze_gap(
            _alignment(
                ("query", "AAAA-----CCCC"),
                ("ref1", "AAAAGGGGGCCCC"),
                ("ref2", "AAAAGGGGGCCCC"),
            ),
            _context(),
        )

        assert support.conservation == 1.0
        assert support.conflicting_positions == ()

    def test_a_lone_reference_is_not_treated_as_conserved(self) -> None:
        """One reference agrees with itself, which is not evidence."""
        support = analyze_gap(
            _alignment(("query", "AAAA-----CCCC"), ("ref1", "AAAAGGGGGCCCC")),
            _context(),
        )

        assert support.conservation == 0.5


class TestDegenerateInput:
    def test_an_alignment_without_the_query_row_yields_no_support(self) -> None:
        support = analyze_gap(_alignment(("ref1", "AAAAGGGGGCCCC")), _context())
        assert support.has_support is False

    def test_an_alignment_with_no_references_yields_no_support(self) -> None:
        support = analyze_gap(_alignment(("query", "AAAA-----CCCC")), _context())
        assert support.has_support is False

    def test_an_alignment_with_no_inserted_columns_yields_no_support(self) -> None:
        """Nothing was inserted between the flanks, so nothing fills the gap."""
        support = analyze_gap(_alignment(("query", "AAAACCCC"), ("ref1", "AAAACCCC")), _context())

        assert support.has_support is False
