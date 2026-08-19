"""Which gaps the agent will attempt, and when a result is fit to return.

The confidence policy scores what was produced; this one decides what should
be attempted at all and whether the finished run is worth reporting.
"""
from __future__ import annotations

from dataclasses import dataclass

from domain.models import GapContext


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Admission rules for gaps and for the finished run."""

    max_gap_length: int = 5000
    # Below this a "gap" is sequencing ambiguity, not an assembly gap.
    min_gap_length: int = 1
    min_flank_length: int = 50

    def should_attempt(self, context: GapContext) -> tuple[bool, str | None]:
        """Whether to spend external calls on this gap.

        Returns the decision and, when negative, the reason - which is reported
        to the caller rather than discarded, so a skipped gap is visibly
        skipped instead of silently absent.
        """
        length = context.gap.length

        if length < self.min_gap_length:
            return False, f"Gap is {length} base(s), below the {self.min_gap_length}-base minimum."

        if length > self.max_gap_length:
            return False, (
                f"Gap is {length} bases, above the {self.max_gap_length}-base limit; "
                "reconstruction over spans this long is not reliable enough to report."
            )

        if not context.has_usable_flanks:
            return False, (
                f"No flank reaches the {self.min_flank_length}-base minimum needed to "
                "anchor a homology search."
            )

        return True, None

    def is_reportable(self, attempted: int, resolved: int) -> bool:
        """Whether a finished run produced anything worth returning.

        A run that attempted gaps and resolved none is still reportable: the
        finding that these gaps could not be reconstructed from available
        references is itself the answer to the orchestrator's question.
        """
        return attempted > 0 or resolved > 0
