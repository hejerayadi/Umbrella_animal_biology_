"""Alignments between the target's flanking context and a reference."""
from __future__ import annotations

from dataclasses import dataclass, field

GAP_CHARACTER = "-"


@dataclass(frozen=True, slots=True)
class AlignedPair:
    """One target/reference row pair from a pairwise or multiple alignment.

    Both strings are the same length and may contain `-` for alignment gaps.
    """

    target_id: str
    reference_id: str
    target_aligned: str
    reference_aligned: str

    def __post_init__(self) -> None:
        if len(self.target_aligned) != len(self.reference_aligned):
            raise ValueError(
                f"Aligned rows for {self.reference_id} differ in length "
                f"({len(self.target_aligned)} vs {len(self.reference_aligned)})."
            )

    @property
    def length(self) -> int:
        return len(self.target_aligned)

    @property
    def identity(self) -> float:
        """Fraction of aligned columns where both rows carry the same base.

        Columns where either row has an alignment gap are excluded from the
        denominator, so this is identity over the aligned region rather than
        over the full span.
        """
        comparable = 0
        matches = 0
        for target_base, reference_base in zip(
            self.target_aligned, self.reference_aligned, strict=True
        ):
            if target_base == GAP_CHARACTER or reference_base == GAP_CHARACTER:
                continue
            comparable += 1
            if target_base == reference_base:
                matches += 1
        return matches / comparable if comparable else 0.0


@dataclass(frozen=True, slots=True)
class Alignment:
    """The alignment of one gap's context against a set of references.

    This is the direct input to candidate generation: the reference bases
    sitting in the columns that span the gap are the proposed reconstruction.
    """

    gap_id: str
    pairs: list[AlignedPair] = field(default_factory=list)
    # Column offsets in the aligned coordinate system that correspond to the
    # gap itself, rather than to the flanks.
    gap_column_start: int | None = None
    gap_column_end: int | None = None
    #: How many bases earlier than the flank junction the gap's columns begin.
    #: Non-zero when the aligner slid the indel, which it may legitimately do
    #: whenever the flank and the missing segment share end bases.
    junction_slide: int = 0
    tool: str = "mafft"

    @property
    def reference_count(self) -> int:
        return len(self.pairs)

    @property
    def spans_gap(self) -> bool:
        """Whether the gap's columns were located in this alignment.

        Without them there is nothing to read a reconstruction out of, however
        good the flank identity looks.
        """
        return self.gap_column_start is not None and self.gap_column_end is not None

    def anchor_fill(self, fill: str) -> str:
        """`fill` expressed between the flanks, undoing any indel slide.

        A fill read from columns the aligner placed early is still correct - but
        correct *there*, not between the flanks the caller holds. Rotating it
        back is what makes it splice cleanly.

        Measured on the polar bear mitogenome: the columns yielded
        `ATTTGAAAG...CGT` where the removed bases were `TTTGAAAG...CGTA`. Both
        describe the same sequence once placed, and only the rotated form can be
        compared against the truth or handed to a caller as "the missing bases".

        The bases moved across come from the *target* row, which every reference
        shares, so one rotation is correct for the consensus as a whole.
        """
        slide = self.junction_slide
        if not slide or not fill or not self.pairs:
            return fill

        target_row = self.pairs[0].target_aligned
        start, end = self.gap_column_start, self.gap_column_end
        if start is None or end is None:
            return fill

        if slide > 0:
            # The run began `slide` bases early: drop that many from the front
            # and take the target bases that followed the run instead.
            trailing = _residues(target_row[end:], slide)
            if len(trailing) < slide or len(fill) <= slide:
                return fill
            return fill[slide:] + trailing

        # The run began late: the bases just before it belong at the front.
        leading = _residues(target_row[:start][::-1], -slide)[::-1]
        if len(leading) < -slide or len(fill) <= -slide:
            return fill
        return leading + fill[:slide]

    def mean_identity(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(pair.identity for pair in self.pairs) / len(self.pairs)


def _residues(row: str, count: int) -> str:
    """The first `count` non-gap characters of an aligned row."""
    out: list[str] = []
    for character in row:
        if character == GAP_CHARACTER:
            continue
        out.append(character)
        if len(out) == count:
            break
    return "".join(out)
