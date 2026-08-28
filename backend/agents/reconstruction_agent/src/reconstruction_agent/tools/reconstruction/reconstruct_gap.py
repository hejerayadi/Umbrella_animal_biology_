"""Apply an accepted fill to the record, producing the repaired sequence.

The step that turns a scored candidate into a repaired genome. It is separate
from `finalize_result` because the two answer different questions: finalisation
decides *whether* a candidate may be reported, this one performs the
substitution and reports the region as repaired.

It refuses rather than corrupts. A fill of the wrong length, an ambiguous fill,
or a region that turns out to contain called bases all come back as a stated
refusal with the record untouched - see `services/reconstruction/gap_replacer.py`
for why each of those would be undetectable downstream.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.domain.models.sequence import Gap
from reconstruction_agent.services.reconstruction.gap_replacer import apply_fill
from reconstruction_agent.tools.base import Tool, ToolOutcome


class ReconstructGapInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str
    gap: Gap
    candidate: Candidate
    #: The record being repaired, as fetched.
    residues: str


class ReconstructGapOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: The repaired region only, not the whole record: returning a whole
    #: scaffold through the orchestrator's shared context would put megabytes
    #: into every downstream agent's prompt.
    filled_sequence: str = ""
    #: Enough surrounding sequence to see the junctions the fill was placed at.
    repaired_window: str = ""
    applied: bool = False


class ReconstructGapTool(Tool[ReconstructGapInput, ReconstructGapOutput]):
    """Substitute an accepted candidate into the record."""

    name = ToolName.RECONSTRUCT_GAP
    description = (
        "Write an accepted candidate into the target record and return the repaired "
        "region. Refuses if the fill would shift coordinates or overwrite called bases."
    )
    input_model = ReconstructGapInput

    #: Bases of context returned either side of the fill, so a reviewer can see
    #: both junctions without the whole record being shipped.
    WINDOW = 60

    async def run(self, request: ReconstructGapInput) -> ToolOutcome[ReconstructGapOutput]:
        result = apply_fill(request.residues, request.gap, request.candidate.sequence)

        if not result.applied:
            return ToolOutcome(tool=self.name, ok=False, reason=result.reason)

        start = max(request.gap.start - self.WINDOW, 0)
        end = min(request.gap.end + self.WINDOW, len(result.residues))

        return ToolOutcome(
            tool=self.name,
            ok=True,
            data=ReconstructGapOutput(
                filled_sequence=request.candidate.sequence,
                repaired_window=result.residues[start:end],
                applied=True,
            ),
        )

    def summarise(self, outcome: ToolOutcome[ReconstructGapOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        return {
            "applied": outcome.data.applied,
            "filled_bases": len(outcome.data.filled_sequence),
        }
