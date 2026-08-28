"""Ask Evo 2 what belongs in the gap - and, where candidates exist, whether it agrees.

One call, two uses. The continuation Evo 2 writes is a proposed fill in its own
right, and the same continuation is what existing candidates are scored
against. Which use matters depends on what homology found:

**Nothing spanned the gap.** The continuation is the only proposal there will
be, and this tool is what turns an unresolved region into an answer. It is
always called in that case - no margin, no borderline band, no condition beyond
the region being short enough to generate coherently.

**Candidates already exist.** The continuation is a second opinion, and the
tool declines unless there is a genuine tie to break. Running it on a fill a
dozen conspecific genomes agree on cannot improve the answer.

A generated fill is returned marked as a prediction. Downstream it is scored on
a separate, capped path, and its rationale says in words that no sequenced
organism is known to carry it - because a caller reading a summary rather than
a schema still has to be told.

Unavailability is never a failure of the gap: a gap homology answered keeps its
answer, and one it did not is reported unresolved rather than filled by
nothing. The reason is recorded either way.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.domain.models.evidence import EvidenceContribution
from reconstruction_agent.domain.models.result import Evo2Evidence
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.services.candidate.model_candidate import build_model_candidate
from reconstruction_agent.services.plausibility.evo2_service import Evo2Service
from reconstruction_agent.services.plausibility.plausibility_policy import ArbitrationPolicy
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.tools.base import Tool, ToolOutcome


class Evo2Input(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str
    candidates: tuple[Candidate, ...]
    #: Needed to validate a generated fill against the region it will occupy -
    #: its length, and the GC content of the sequence either side of it.
    context: GapContext
    left_flank: str
    gap_length: int
    #: Set by the replanner to arbitrate even when the policy would decline -
    #: used when the critic named COMPETING_CANDIDATES and nothing else is left
    #: to try.
    force: bool = False


class Evo2Output(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Agreement per candidate id. Empty when there was nothing to judge.
    agreement: dict[str, float] = {}
    #: The model's proposed fill, already validated and scored, present when
    #: the continuation was usable. Carries `origin=MODEL`, so nothing
    #: downstream can mistake it for an observation.
    generated_candidate: Candidate | None = None
    mean_model_confidence: float | None = None
    consulted: bool = False
    reason: str = ""


class EvaluateWithEvo2Tool(Tool[Evo2Input, Evo2Output]):
    """A second opinion on which candidate a genome model would have written."""

    name = ToolName.EVALUATE_WITH_EVO2
    description = (
        "Break a tie between candidates by comparing them with Evo 2's own continuation. "
        "Only worth calling when two candidates are close or confidence is borderline."
    )
    input_model = Evo2Input

    def __init__(
        self,
        service: Evo2Service,
        engine: ConfidenceEngine,
        policy: ArbitrationPolicy | None = None,
    ) -> None:
        self._service = service
        self._engine = engine
        self._policy = policy or ArbitrationPolicy()

    async def run(self, request: Evo2Input) -> ToolOutcome[Evo2Output]:
        decision = self._policy.decide(request.candidates, gap_length=request.gap_length)
        if not decision.arbitrate and not request.force:
            # Declining is a result, and the reason is part of the audit trail.
            return ToolOutcome(
                tool=self.name,
                ok=True,
                data=Evo2Output(consulted=False, reason=decision.reason),
                reason=decision.reason,
            )

        arbitration = await self._service.arbitrate(
            request.candidates,
            left_flank=request.left_flank,
            gap_length=request.gap_length,
        )

        if not arbitration.consulted:
            reason = arbitration.unavailable_reason or "Evo 2 produced no usable answer."
            return ToolOutcome(
                tool=self.name,
                ok=True,
                data=Evo2Output(consulted=False, reason=reason),
                reason=reason,
            )

        generated = build_model_candidate(
            arbitration.generated,
            request.context,
            engine=self._engine,
            model_certainty=arbitration.mean_model_confidence,
        )

        return ToolOutcome(
            tool=self.name,
            ok=True,
            data=Evo2Output(
                agreement=arbitration.agreement,
                generated_candidate=generated,
                mean_model_confidence=arbitration.mean_model_confidence,
                consulted=True,
                reason=decision.reason,
            ),
        )

    def summarise(self, outcome: ToolOutcome[Evo2Output]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        data = outcome.data
        generated = data.generated_candidate
        return {
            "consulted": data.consulted,
            "agreement": {key: round(value, 3) for key, value in data.agreement.items()},
            "mean_model_confidence": data.mean_model_confidence,
            "generated_bases": len(generated.sequence) if generated else 0,
            "generated_confidence": (round(generated.final_confidence, 3) if generated else None),
            "reason": data.reason,
        }

    def contribute(self, outcome: ToolOutcome[Evo2Output]) -> EvidenceContribution:
        """Recorded whether or not the call happened.

        A gap whose evidence says nothing about Evo 2 is indistinguishable from
        one where arbitration silently failed.
        """
        if outcome.data is None:
            return EvidenceContribution()
        data = outcome.data
        return EvidenceContribution(
            providers=("NVIDIA Evo 2",) if data.consulted else (),
            evo2=Evo2Evidence(
                consulted=data.consulted,
                agreement=max(data.agreement.values(), default=None) if data.agreement else None,
                mean_model_confidence=data.mean_model_confidence,
                unavailable_reason=None if data.consulted else data.reason,
            ),
        )
