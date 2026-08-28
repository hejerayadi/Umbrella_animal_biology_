"""Fetch the record being repaired and cut the flanks around one gap.

First action of every plan, and the only one whose failure ends a run outright:
without the record there is no query to search with, no anchor to align to and
no coordinates to trust.

The flank width is an argument rather than a constant because widening it is
the replanner's answer to INSUFFICIENT_CONTEXT - a gap in a repetitive region
needs a longer anchor to place uniquely.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.domain.models.evidence import EvidenceContribution
from reconstruction_agent.domain.models.sequence import Gap, GapContext, SequenceRecord
from reconstruction_agent.services.sequence.sequence_service import SequenceService, build_context
from reconstruction_agent.tools.base import Tool, ToolOutcome

#: Bases either side of a gap. Long enough to place an alignment uniquely,
#: short enough that the BLAST query stays fast.
DEFAULT_FLANK_SIZE = 500


class SequenceContextInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str
    gap: Gap
    accession: str | None = None
    #: Used instead of fetching when the caller pasted a sequence.
    residues: str | None = None
    flank_size: int = Field(default=DEFAULT_FLANK_SIZE, ge=20, le=5000)


class SequenceContextOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: GapContext
    record: SequenceRecord


class GetSequenceContextTool(Tool[SequenceContextInput, SequenceContextOutput]):
    """The gap plus the sequence flanking it."""

    name = ToolName.GET_SEQUENCE_CONTEXT
    description = (
        "Fetch the target record and cut the flanking sequence around one gap. "
        "Run this first; widen flank_size when the flanks are too repetitive to anchor."
    )
    input_model = SequenceContextInput

    def __init__(self, sequences: SequenceService) -> None:
        self._sequences = sequences

    async def run(self, request: SequenceContextInput) -> ToolOutcome[SequenceContextOutput]:
        try:
            record = await self._resolve(request)
        except ReconstructionError as error:
            return ToolOutcome(tool=self.name, ok=False, reason=str(error), transport_error=True)

        context = build_context(record, request.gap, flank_size=request.flank_size)
        if not context.has_usable_flanks:
            return ToolOutcome(
                tool=self.name,
                ok=False,
                reason=(
                    f"Gap {request.gap_id} has no usable flanking sequence to anchor "
                    "an alignment; it sits at a record boundary or inside a longer "
                    "unresolved region."
                ),
            )

        return ToolOutcome(
            tool=self.name,
            ok=True,
            data=SequenceContextOutput(context=context, record=record),
        )

    async def _resolve(self, request: SequenceContextInput) -> SequenceRecord:
        if request.accession:
            return await self._sequences.fetch(request.accession)
        if request.residues:
            return SequenceRecord(accession="supplied-sequence", residues=request.residues)
        raise ReconstructionError("No accession or residues were supplied.")

    def summarise(self, outcome: ToolOutcome[SequenceContextOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        context = outcome.data.context
        return {
            "accession": outcome.data.record.accession,
            "left_flank_bp": len(context.left_flank),
            "right_flank_bp": len(context.right_flank),
            "gap_length": context.gap.length,
        }

    def contribute(self, outcome: ToolOutcome[SequenceContextOutput]) -> EvidenceContribution:
        """Fetching the record is a consultation of NCBI, answer or not."""
        if outcome.data is None:
            return EvidenceContribution()
        return EvidenceContribution(providers=("NCBI",))
