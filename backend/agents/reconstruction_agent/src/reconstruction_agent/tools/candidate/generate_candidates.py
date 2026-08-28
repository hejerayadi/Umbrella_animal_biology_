"""Turn alignment support into competing, scored proposals.

Competing is the operative word. References that disagree produce distinct
candidates rather than a plurality consensus, and every one of them is kept and
scored. Collapsing early would throw away exactly the evidence the critic needs
to recognise COMPETING_CANDIDATES and arbitrate - and it would let a run report
one answer with high confidence when the evidence actually supported two.

Every score here is computed by a deterministic scorer. No language model
contributes a number at any point on this path.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.exceptions import ReconstructionError
from reconstruction_agent.domain.models.alignment import AlignmentSupport
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.domain.models.evidence import EvidenceContribution
from reconstruction_agent.domain.models.homology import HomologHit
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.domain.models.taxonomy import TargetProfile
from reconstruction_agent.services.candidate.candidate_builder import CandidateBuilder
from reconstruction_agent.tools.base import Tool, ToolOutcome


class GenerateCandidatesInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str
    support: AlignmentSupport
    context: GapContext
    profile: TargetProfile
    hits: tuple[HomologHit, ...] = ()


class GenerateCandidatesOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Best-scoring first. Never truncated to one: the alternatives are
    #: evidence about how settled the answer is.
    candidates: tuple[Candidate, ...] = ()


class GenerateCandidatesTool(Tool[GenerateCandidatesInput, GenerateCandidatesOutput]):
    """Every distinct sequence the evidence supports, scored and ranked."""

    name = ToolName.GENERATE_CANDIDATES
    description = (
        "Build the competing fill sequences the alignment supports, each scored on "
        "homology, alignment, conservation, taxonomy and biological validity."
    )
    input_model = GenerateCandidatesInput

    def __init__(self, builder: CandidateBuilder) -> None:
        self._builder = builder

    async def run(self, request: GenerateCandidatesInput) -> ToolOutcome[GenerateCandidatesOutput]:
        try:
            candidates = await self._builder.build(
                request.support, request.context, request.profile, request.hits
            )
        except ReconstructionError as error:
            return ToolOutcome(tool=self.name, ok=False, reason=str(error), transport_error=True)

        if not candidates:
            return ToolOutcome(
                tool=self.name,
                ok=False,
                reason="No candidate sequence could be assembled from the alignment support.",
            )

        return ToolOutcome(
            tool=self.name,
            ok=True,
            data=GenerateCandidatesOutput(candidates=candidates),
        )

    def summarise(self, outcome: ToolOutcome[GenerateCandidatesOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        candidates = outcome.data.candidates
        best = candidates[0] if candidates else None
        return {
            "candidates": len(candidates),
            "best_confidence": round(best.final_confidence, 3) if best else None,
            #: The margin is what decides whether arbitration is worth a call.
            "margin": (
                round(candidates[0].final_confidence - candidates[1].final_confidence, 3)
                if len(candidates) > 1
                else None
            ),
        }

    def contribute(self, outcome: ToolOutcome[GenerateCandidatesOutput]) -> EvidenceContribution:
        """Which validation checks the leading candidate passed, and which it failed."""
        if outcome.data is None or not outcome.data.candidates:
            return EvidenceContribution()
        best = outcome.data.candidates[0]
        return EvidenceContribution(
            validation_checks=tuple(result.check for result in best.validation_results),
            validation_failures=best.failed_checks(),
        )
