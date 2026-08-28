"""Fetch the actual residues of the hits the search returned.

BLAST reports where a hit is and how well it matches; it does not hand back
enough sequence to fill a gap with. This is the step that turns a list of
accessions into alignable evidence.

Its failure mode has its own name for a reason. Hits were found and crossed the
gap, and then not one of their sequences could be retrieved: that is a
retrieval failure, and reporting it as an absence of gap-spanning homologues
would contradict a measurement this run has already made.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.orchestration.deadline import Deadline
from reconstruction_agent.services.sequence.sequence_service import SequenceService
from reconstruction_agent.tools.base import Tool, ToolOutcome


class HomologSequencesInput(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    gap_id: str
    hits: tuple[HomologHit, ...]
    deadline: Deadline
    #: Raised by the replanner when an alignment was too thin to be conclusive.
    limit: int = Field(default=12, ge=1, le=50)


class HomologSequencesOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    hits: tuple[HomologHit, ...]


class GetHomologSequencesTool(Tool[HomologSequencesInput, HomologSequencesOutput]):
    """The residues of the top hits, ready to align."""

    name = ToolName.GET_HOMOLOG_SEQUENCES
    description = (
        "Fetch the sequence of each homologue found, so it can be aligned. "
        "Raise limit when an alignment had too few references to be conclusive."
    )
    input_model = HomologSequencesInput

    def __init__(self, sequences: SequenceService) -> None:
        self._sequences = sequences

    async def run(self, request: HomologSequencesInput) -> ToolOutcome[HomologSequencesOutput]:
        if not request.hits:
            return ToolOutcome(
                tool=self.name, ok=False, reason="No homologues were supplied to fetch."
            )

        try:
            fetched = await self._sequences.fetch_homolog_sequences(
                request.hits[: request.limit], deadline=request.deadline
            )
        except ReconstructionError as error:
            return ToolOutcome(tool=self.name, ok=False, reason=str(error), transport_error=True)

        if not fetched:
            return ToolOutcome(
                tool=self.name,
                ok=False,
                reason=(
                    f"{len(request.hits)} homologues crossed this region but none of "
                    "their sequences could be retrieved, so no alignment was possible."
                ),
                transport_error=True,
            )

        return ToolOutcome(tool=self.name, ok=True, data=HomologSequencesOutput(hits=fetched))

    def summarise(self, outcome: ToolOutcome[HomologSequencesOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        return {"fetched": len(outcome.data.hits)}
