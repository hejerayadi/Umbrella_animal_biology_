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

    def mean_identity(self) -> float:
        if not self.pairs:
            return 0.0
        return sum(pair.identity for pair in self.pairs) / len(self.pairs)
