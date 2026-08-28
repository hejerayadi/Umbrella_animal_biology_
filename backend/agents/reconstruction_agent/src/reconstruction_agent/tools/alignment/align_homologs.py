"""Align the flanks and the homologues at EMBL-EBI MAFFT.

The one step still run at EBI: NCBI publishes no alignment service and no MAFFT
binary is installed here.

The query is the two flanks concatenated, not the whole record. The junction
between them is where the missing bases belong, and every reference that aligns
across that junction is one vote on what they are.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.domain.models.alignment import Alignment
from reconstruction_agent.domain.models.evidence import EvidenceContribution
from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.domain.models.result import AlignmentEvidence
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.integrations.mafft.client import MafftClient
from reconstruction_agent.integrations.mafft.parser import build_fasta, parse_alignment
from reconstruction_agent.orchestration.deadline import Deadline
from reconstruction_agent.tools.base import Tool, ToolOutcome


class AlignHomologsInput(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    gap_id: str
    context: GapContext
    hits: tuple[HomologHit, ...]
    deadline: Deadline
    time_budget: float | None = None


class AlignHomologsOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    alignment: Alignment


class AlignHomologsTool(Tool[AlignHomologsInput, AlignHomologsOutput]):
    """A multiple alignment of the flanks against the homologues."""

    name = ToolName.ALIGN_HOMOLOGS
    description = (
        "Align the gap's flanks with the fetched homologues, so the bases each "
        "homologue carries across the junction can be read off."
    )
    input_model = AlignHomologsInput

    def __init__(self, mafft: MafftClient) -> None:
        self._mafft = mafft

    async def run(self, request: AlignHomologsInput) -> ToolOutcome[AlignHomologsOutput]:
        if not request.hits:
            return ToolOutcome(
                tool=self.name, ok=False, reason="No homologue sequences were supplied to align."
            )

        allowed = request.deadline.remaining_for_work()
        if request.time_budget is not None:
            allowed = min(allowed, request.time_budget)
        if allowed <= 0:
            return ToolOutcome(
                tool=self.name,
                ok=False,
                reason="The run deadline closed before an alignment could be started.",
            )

        try:
            aligned = await self._mafft.align(
                build_fasta(request.context.query_sequence(), request.hits),
                timeout=allowed,
            )
        except ReconstructionError as error:
            return ToolOutcome(tool=self.name, ok=False, reason=str(error), transport_error=True)

        alignment = parse_alignment(aligned)
        if alignment.query_row is None or not alignment.reference_rows:
            return ToolOutcome(
                tool=self.name,
                ok=False,
                reason="The aligner returned no usable alignment of the flanks against the hits.",
            )

        return ToolOutcome(tool=self.name, ok=True, data=AlignHomologsOutput(alignment=alignment))

    def summarise(self, outcome: ToolOutcome[AlignHomologsOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        alignment = outcome.data.alignment
        return {"references": len(alignment.reference_rows)}

    def contribute(self, outcome: ToolOutcome[AlignHomologsOutput]) -> EvidenceContribution:
        """The aligner ran at EMBL-EBI; how many rows it returned is evidence."""
        if outcome.data is None:
            return EvidenceContribution(providers=("EMBL-EBI MAFFT",))
        return EvidenceContribution(
            providers=("EMBL-EBI MAFFT",),
            alignment=AlignmentEvidence(
                references_aligned=len(outcome.data.alignment.reference_rows)
            ),
        )
