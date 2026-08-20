"""Find the unresolved regions in a sequence.

Fully deterministic and offline - this is the first thing that runs, and it
decides whether there is any work to do at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from domain.models import Gap, Sequence
from domain.models.sequence import UNKNOWN_BASE

# A gap is a maximal run of N. Other IUPAC ambiguity codes (R, Y, ...) express
# a *constrained* uncertainty and are left alone: they still carry information,
# and overwriting them would discard it.
_GAP_PATTERN = re.compile(f"{UNKNOWN_BASE}+")


@dataclass(frozen=True, slots=True)
class GapDetector:
    """Locates runs of unknown bases.

    `minimum_length` filters out incidental single-base ambiguities, which are
    sequencing noise rather than assembly gaps and are not worth a BLAST round
    trip. Raise it to focus on structural gaps only.
    """

    minimum_length: int = 1

    def detect(self, sequence: Sequence) -> list[Gap]:
        """Every gap in the sequence, ordered by position.

        Gap identifiers are positional (`gap_1`, `gap_2`, ...) and stable for a
        given input, so they can be referenced in logs and output without
        carrying the offsets around.
        """
        gaps: list[Gap] = []
        for index, match in enumerate(_GAP_PATTERN.finditer(sequence.residues), start=1):
            start, end = match.span()
            if end - start < self.minimum_length:
                continue
            gaps.append(Gap(identifier=f"gap_{index}", start=start, end=end))
        return gaps

    def total_unknown_bases(self, sequence: Sequence) -> int:
        return sequence.unknown_count
