"""Multiple sequence alignment, and what it says about a gap.

MAFFT aligns; it does not reconstruct. The scientific content is entirely in
reading its output: which references actually cross the missing region, what
each of them puts there, and whether they agree.

The distinction that matters most is between "homologues were found" and "a
homologue spans the gap". They look alike in a hit table and are not remotely
the same evidence - the second is the only one that can support a fill.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field


class AlignedRow(BaseModel):
    """One sequence in an alignment, with gap characters included."""

    model_config = ConfigDict(frozen=True)

    identifier: str
    #: Alignment-length string over the DNA alphabet plus the gap character.
    aligned_sequence: str

    def column(self, index: int) -> str:
        if 0 <= index < len(self.aligned_sequence):
            return self.aligned_sequence[index]
        return "-"

    def columns(self, start: int, end: int) -> str:
        return self.aligned_sequence[max(0, start) : max(0, end)]


class Alignment(BaseModel):
    """A finished multiple sequence alignment."""

    model_config = ConfigDict(frozen=True)

    #: Identifier of the row holding the target flanks.
    query_id: str
    rows: tuple[AlignedRow, ...] = ()

    @property
    def width(self) -> int:
        return max((len(row.aligned_sequence) for row in self.rows), default=0)

    @property
    def query_row(self) -> AlignedRow | None:
        return next((row for row in self.rows if row.identifier == self.query_id), None)

    @property
    def reference_rows(self) -> tuple[AlignedRow, ...]:
        return tuple(row for row in self.rows if row.identifier != self.query_id)


class ReferenceFill(BaseModel):
    """What one reference sequence puts where the target has nothing."""

    model_config = ConfigDict(frozen=True)

    accession: str
    organism: str | None = None
    organism_tax_id: int | None = None
    #: The reference bases across the gap columns, gap characters stripped.
    bases: str
    #: False when the reference is present in the alignment but does not cover
    #: both flanks. Such a row contributes nothing and must never be counted as
    #: support merely because it appeared in the results.
    spans_gap: bool = True


class AlignmentSupport(BaseModel):
    """The analysis of one alignment with respect to one gap."""

    model_config = ConfigDict(frozen=True)

    gap_id: str
    #: Alignment columns judged to correspond to the missing region, half-open.
    #:
    #: These are located from the flank junctions rather than by arithmetic on
    #: the query coordinates, because an aligner may place an indel one base
    #: either side of the junction when the flank and the missing segment share
    #: an end base. Requiring an exact junction match reports references that
    #: carry every base of the answer as "nothing aligned across the gap".
    gap_column_start: int = 0
    gap_column_end: int = 0

    fills: tuple[ReferenceFill, ...] = ()
    #: Column agreement among spanning references, 0..1.
    conservation: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Positions where spanning references disagree on the base.
    conflicting_positions: tuple[int, ...] = ()

    @property
    def spanning_fills(self) -> tuple[ReferenceFill, ...]:
        return tuple(fill for fill in self.fills if fill.spans_gap)

    @property
    def spanning_count(self) -> int:
        return len(self.spanning_fills)

    @property
    def has_support(self) -> bool:
        """Whether anything here can support a reconstruction at all."""
        return self.spanning_count > 0

    def distinct_fills(self) -> dict[str, int]:
        """Distinct proposed sequences and how many references back each.

        Competing entries here are exactly the ambiguity the agent must not
        collapse early - they become separate candidates, not a majority vote.
        """
        return dict(Counter(fill.bases for fill in self.spanning_fills if fill.bases))
