"""Input/output shapes for the MAFFT alignment tool."""
from __future__ import annotations

from pydantic import Field

from domain.models import Alignment
from tools.contracts import ToolInput, ToolOutput


class AlignmentInput(ToolInput):
    """Align a gap's context against its selected references."""

    gap_id: str
    # The target row is the gap's flanks joined; the gap's own bases are
    # excluded so alignment gaps in the result mark where the gap sits.
    target_id: str = "target"
    target_sequence: str = Field(description="The gap's flanking context.")
    left_flank_length: int = Field(
        description="Where the left flank ends in `target_sequence`; locates the gap columns."
    )
    references: dict[str, str] = Field(
        default_factory=dict, description="accession -> residues for each reference to align."
    )


class AlignmentOutput(ToolOutput):
    alignment: Alignment | None = Field(
        default=None, description="The alignment, when the job succeeded."
    )
    job_id: str | None = None
    aligned_count: int = 0
