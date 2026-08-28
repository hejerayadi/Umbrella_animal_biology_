"""Decide what one gap's evidence actually supports, and refuse when it is thin.

The last action of every plan, and the only one permitted to declare a gap
resolved. Concentrating that decision here means there is exactly one place
where a sequence becomes an answer - so the confidence floor cannot be
bypassed by a path that reached the end some other way.

Refusal is a result. A candidate below the floor, or one that failed biological
validation, comes back UNRESOLVED with its original coordinates preserved and a
stated reason. Nothing here ever invents a base to avoid an empty answer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import GapStatus, ToolName, UnresolvedReason
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.domain.models.result import GapEvidence, GapReconstruction, Provenance
from reconstruction_agent.domain.models.sequence import Gap
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.tools.base import Tool, ToolOutcome


class FinalizeResultInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str
    gap: Gap
    candidates: tuple[Candidate, ...] = ()
    provenance: Provenance = Provenance()
    evidence: GapEvidence = GapEvidence()
    #: Why nothing was produced, when the caller already knows. Set when no
    #: search completed: reporting that as an absence of gap-spanning
    #: homologues would claim a measurement the run never made.
    unresolved_hint: UnresolvedReason | None = None
    hint_explanation: str = ""


class FinalizeResultOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    reconstruction: GapReconstruction


class FinalizeResultTool(Tool[FinalizeResultInput, FinalizeResultOutput]):
    """Accept the best candidate, or refuse and say why."""

    name = ToolName.FINALIZE_RESULT
    description = (
        "Commit to the best candidate for a gap, or report it UNRESOLVED with a reason. "
        "Always the last action for a gap."
    )
    input_model = FinalizeResultInput
    #: A run that cannot afford to report its own findings has spent its
    #: whole budget for nothing.
    budget_exempt = True

    def __init__(self, engine: ConfidenceEngine) -> None:
        self._engine = engine

    async def run(self, request: FinalizeResultInput) -> ToolOutcome[FinalizeResultOutput]:
        reconstruction = self._decide(request)
        return ToolOutcome(
            tool=self.name,
            ok=reconstruction.status is GapStatus.RESOLVED,
            data=FinalizeResultOutput(reconstruction=reconstruction),
            reason=reconstruction.explanation,
        )

    def _decide(self, request: FinalizeResultInput) -> GapReconstruction:
        if not request.candidates:
            return self._unresolved(
                request,
                request.unresolved_hint or UnresolvedReason.INSUFFICIENT_GAP_SPANNING_HOMOLOGS,
                request.hint_explanation
                or "No candidate sequence could be assembled from the evidence gathered.",
            )

        best, alternatives = request.candidates[0], request.candidates[1:]

        # Gated on the checks that declare themselves disqualifying, not on the
        # aggregate score. `ValidationVerdict.fatal` exists precisely so an
        # advisory check reaching zero - a fill twice the length of its gap
        # scores nothing on length - does not reject a candidate it was only
        # meant to caution about. Reading the aggregate instead let a candidate
        # through whenever any other check passed, which is every candidate.
        fatal = [v.check for v in best.validation_results if v.fatal and not v.passed]
        if fatal:
            return self._unresolved(
                request,
                UnresolvedReason.BIOLOGICAL_VALIDATION_FAILED,
                f"The best candidate failed validation: {', '.join(fatal)}.",
                alternatives=alternatives,
            )

        if not self._engine.is_returnable(best.final_confidence):
            return self._unresolved(
                request,
                UnresolvedReason.CONFIDENCE_BELOW_THRESHOLD,
                (
                    f"The best candidate reached only {best.final_confidence:.2f} confidence, "
                    f"below the floor for a reported reconstruction. {best.rationale}"
                ),
                alternatives=alternatives,
            )

        return GapReconstruction(
            gap=request.gap,
            status=GapStatus.RESOLVED,
            selected_candidate=best,
            alternatives=alternatives,
            explanation=best.rationale,
            provenance=request.provenance,
            evidence=request.evidence,
            warnings=_caveats(best, request.gap),
        )

    def _unresolved(
        self,
        request: FinalizeResultInput,
        reason: UnresolvedReason,
        explanation: str,
        *,
        alternatives: tuple[Candidate, ...] = (),
    ) -> GapReconstruction:
        """A gap left open, with its coordinates intact and the reason stated."""
        return GapReconstruction(
            gap=request.gap,
            status=GapStatus.UNRESOLVED,
            unresolved_reason=reason,
            alternatives=alternatives,
            explanation=explanation,
            provenance=request.provenance,
            evidence=request.evidence,
        )

    def summarise(self, outcome: ToolOutcome[FinalizeResultOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        result = outcome.data.reconstruction
        return {
            "status": result.status.value,
            "confidence": round(result.confidence, 3) if result.confidence else None,
            "unresolved_reason": (
                result.unresolved_reason.value if result.unresolved_reason else None
            ),
        }


def _caveats(best: Candidate, gap: Gap) -> tuple[str, ...]:
    """What a caller must know before substituting this fill.

    A consumer writes the fill into the record at the gap's coordinates. A fill
    of a different length shifts every base after it, and the only place that
    can be noticed is here - the sequence itself carries no evidence of it.
    Measured on NW_007907101: a 26-base fill accepted for a 10-base gap.
    """
    caveats: list[str] = []

    if best.length != gap.length:
        caveats.append(
            f"The fill is {best.length} bases and the gap is {gap.length}. "
            "Assembly gap lengths are estimates and a real indel changes them, so this is "
            "not necessarily wrong - but substituting it will shift every downstream "
            "coordinate in the record."
        )

    for verdict in best.validation_results:
        if not verdict.passed and not verdict.fatal:
            caveats.append(f"Advisory check {verdict.check} did not pass: {verdict.reason}")

    if best.is_model_generated:
        caveats.append(
            "This fill was written by a genome model, not read from any sequenced organism."
        )

    return tuple(caveats)
