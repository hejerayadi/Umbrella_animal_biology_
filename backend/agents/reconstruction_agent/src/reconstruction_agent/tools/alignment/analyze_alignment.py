"""Read what each aligned reference says about the missing bases.

Separate from producing the alignment because it fails for a completely
different reason and the distinction is what the critic acts on. An alignment
that could not be computed is a transport problem; an alignment that was
computed and shows no reference crossing the junction is a scientific finding.

This is also where MAFFT's indel slide is absorbed. When the flank and the
missing segment share an end base, the aligner may place the gap one base
either side of the junction; `analyze_gap` re-anchors the fill between the
flanks. Without that tolerance, references carrying every base of the answer
are reported as "no reference aligned across the gap".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.models.alignment import Alignment, AlignmentSupport
from reconstruction_agent.domain.models.evidence import EvidenceContribution
from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.domain.models.result import AlignmentEvidence
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.services.alignment.gap_analyzer import analyze_gap
from reconstruction_agent.tools.base import Tool, ToolOutcome


class AnalyzeAlignmentInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str
    alignment: Alignment
    context: GapContext
    hits: tuple[HomologHit, ...] = ()


class AnalyzeAlignmentOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    support: AlignmentSupport


class AnalyzeAlignmentTool(Tool[AnalyzeAlignmentInput, AnalyzeAlignmentOutput]):
    """The bases each reference carries across the gap, and how far they agree."""

    name = ToolName.ANALYZE_ALIGNMENT
    description = (
        "Read the alignment columns spanning the gap: which references cross it, what "
        "each proposes, and where they disagree."
    )
    input_model = AnalyzeAlignmentInput

    async def run(self, request: AnalyzeAlignmentInput) -> ToolOutcome[AnalyzeAlignmentOutput]:
        support = analyze_gap(request.alignment, request.context, request.hits)

        if not support.has_support:
            return ToolOutcome(
                tool=self.name,
                ok=False,
                data=AnalyzeAlignmentOutput(support=support),
                reason=(
                    "Homologues were found and aligned, but none of them spans the "
                    "missing region, so nothing supports a fill."
                ),
            )

        return ToolOutcome(tool=self.name, ok=True, data=AnalyzeAlignmentOutput(support=support))

    def summarise(self, outcome: ToolOutcome[AnalyzeAlignmentOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        support = outcome.data.support
        return {
            "spanning_references": len(support.spanning_fills),
            "conservation": round(support.conservation, 3),
            "conflicting_positions": len(support.conflicting_positions),
        }

    def contribute(self, outcome: ToolOutcome[AnalyzeAlignmentOutput]) -> EvidenceContribution:
        """How many references actually crossed the gap, and how far they agree.

        Contributed on failure too: "aligned, but nothing spans the gap" is the
        measurement that separates a scoping problem from an absence.
        """
        if outcome.data is None:
            return EvidenceContribution()
        support = outcome.data.support
        return EvidenceContribution(
            alignment=AlignmentEvidence(
                references_aligned=len(support.fills),
                references_spanning_gap=support.spanning_count,
                conservation=support.conservation,
                competing_fills=len(support.distinct_fills()),
            )
        )
