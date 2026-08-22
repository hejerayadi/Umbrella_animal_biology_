"""An unresolved region of the target sequence, plus the context around it."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Gap:
    """A run of unknown bases to reconstruct.

    Offsets are 0-based, `start` inclusive and `end` exclusive, matching Python
    slicing. Every consumer assumes that convention.
    """

    identifier: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"Gap {self.identifier}: end ({self.end}) must exceed start.")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class GapContext:
    """A gap together with the known sequence flanking it.

    The flanks are what makes reconstruction possible: they are the anchors
    searched against reference databases, and the alignment of a reference to
    both flanks is what tells us which reference bases span the gap.

    A gap with no usable flank on either side is not reconstructable, which
    `has_usable_flanks` is there to state explicitly.
    """

    gap: Gap
    left_flank: str
    right_flank: str
    # Minimum flank length that makes a homology search meaningful. Below this
    # a BLAST hit is more likely chance similarity than real homology.
    minimum_flank: int = 50

    @property
    def identifier(self) -> str:
        return self.gap.identifier

    @property
    def has_usable_flanks(self) -> bool:
        """True when at least one flank is long enough to anchor a search."""
        return max(len(self.left_flank), len(self.right_flank)) >= self.minimum_flank

    @property
    def has_both_flanks(self) -> bool:
        """True when the gap is bracketed on both sides.

        Two anchors constrain the reconstruction far better than one, so
        candidate ranking scores these higher.
        """
        return (
            len(self.left_flank) >= self.minimum_flank
            and len(self.right_flank) >= self.minimum_flank
        )

    def query_sequence(self) -> str:
        """Flanks joined, for use as a homology-search query.

        The gap's own unknown bases are omitted rather than sent as a run of
        `N`: search services score `N` as a mismatch, which would penalise
        exactly the references we most want to find.
        """
        return f"{self.left_flank}{self.right_flank}"
