"""Pull the known sequence flanking each gap.

The flanks are the agent's only handle on a gap: they are what gets searched
against reference databases and what anchors the alignment that spans the gap.
"""
from __future__ import annotations

from dataclasses import dataclass

from domain.models import Gap, GapContext, Sequence
from domain.models.sequence import UNKNOWN_BASE


@dataclass(frozen=True, slots=True)
class ContextExtractor:
    """Builds `GapContext` objects from a sequence and its gaps.

    `flank_length` trades specificity against availability: longer flanks give
    a more specific homology search, but gaps clustered close together will not
    have that much clean sequence between them.
    """

    flank_length: int = 500
    minimum_flank: int = 50

    def extract(self, sequence: Sequence, gap: Gap) -> GapContext:
        """Context for one gap, with flanks truncated at neighbouring unknowns.

        A flank must be *known* sequence to be useful, so each side stops at
        the first unknown base encountered walking away from the gap. Without
        that, two nearby gaps would each include the other's N run in their
        flank and both searches would be polluted.
        """
        left_raw = sequence.slice(gap.start - self.flank_length, gap.start)
        right_raw = sequence.slice(gap.end, gap.end + self.flank_length)

        return GapContext(
            gap=gap,
            # Walking outward from the gap means trimming the left flank from
            # its start and the right flank from its end.
            left_flank=self._trim_leading_unknowns(left_raw),
            right_flank=self._trim_trailing_unknowns(right_raw),
            minimum_flank=self.minimum_flank,
        )

    def extract_all(self, sequence: Sequence, gaps: list[Gap]) -> list[GapContext]:
        return [self.extract(sequence, gap) for gap in gaps]

    @staticmethod
    def _trim_leading_unknowns(flank: str) -> str:
        """Keep only the run of known bases immediately before the gap."""
        index = flank.rfind(UNKNOWN_BASE)
        return flank[index + 1 :] if index != -1 else flank

    @staticmethod
    def _trim_trailing_unknowns(flank: str) -> str:
        """Keep only the run of known bases immediately after the gap."""
        index = flank.find(UNKNOWN_BASE)
        return flank[:index] if index != -1 else flank
