"""Re-score candidates once new evidence has arrived.

Its whole reason to exist is the failure mode the plan singles out: **an Evo 2
result computed and then lost.** Arbitration produces an agreement score, and
unless something folds that score back into the confidence, the run has spent a
call on a number nobody reads. Rescoring is that step, and it is a tool rather
than a side effect of arbitration so it is visible in the history and testable
on its own.

The engine here is the same one that scored the candidates in the first place.
Nothing is recomputed from evidence - the deterministic signals are unchanged -
only the Evo 2 term is added and the weighted sum retaken.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.tools.base import Tool, ToolOutcome


class ScoreCandidateInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str
    candidates: tuple[Candidate, ...]
    #: Agreement per candidate id, as `evaluate_with_evo2` reported it.
    evo2_agreement: dict[str, float] = {}


class ScoreCandidateOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: Re-ranked: arbitration can change which candidate leads, which is the
    #: entire point of having asked.
    candidates: tuple[Candidate, ...] = ()
    #: True when the leading candidate is not the one that led before.
    order_changed: bool = False


class ScoreCandidateTool(Tool[ScoreCandidateInput, ScoreCandidateOutput]):
    """Fold arbitration back into the confidence and re-rank."""

    name = ToolName.SCORE_CANDIDATE
    description = (
        "Recompute candidate confidence after arbitration and re-rank them. "
        "Run this after evaluate_with_evo2, or its result is never used."
    )
    input_model = ScoreCandidateInput

    def __init__(self, engine: ConfidenceEngine) -> None:
        self._engine = engine

    async def run(self, request: ScoreCandidateInput) -> ToolOutcome[ScoreCandidateOutput]:
        if not request.candidates:
            return ToolOutcome(tool=self.name, ok=False, reason="There are no candidates to score.")
        if not request.evo2_agreement:
            # Nothing new to fold in. Re-ranking on unchanged inputs would be
            # a no-op that looks like work in the history.
            return ToolOutcome(
                tool=self.name,
                ok=True,
                data=ScoreCandidateOutput(candidates=request.candidates),
                reason="No arbitration result to fold in; scores are unchanged.",
            )

        leader_before = request.candidates[0].candidate_id
        rescored = tuple(
            self._rescore(candidate, request.evo2_agreement.get(candidate.candidate_id))
            for candidate in request.candidates
        )
        ranked = tuple(sorted(rescored, key=lambda item: -item.final_confidence))

        return ToolOutcome(
            tool=self.name,
            ok=True,
            data=ScoreCandidateOutput(
                candidates=ranked,
                order_changed=ranked[0].candidate_id != leader_before,
            ),
        )

    def _rescore(self, candidate: Candidate, agreement: float | None) -> Candidate:
        """One candidate with its Evo 2 term applied.

        A candidate arbitration did not cover keeps `None` rather than being
        given a zero: it was not asked about, and scoring it as though the
        model disagreed would penalise it for the tool's own coverage.
        """
        if agreement is None:
            return candidate

        scores = candidate.scores.model_copy(update={"evo2": agreement})
        confidence = self._engine.confidence(scores)
        return candidate.model_copy(
            update={
                "scores": scores,
                "final_confidence": confidence,
                "confidence_level": self._engine.level(confidence),
                "rationale": self._engine.explain(scores, confidence),
            }
        )

    def summarise(self, outcome: ToolOutcome[ScoreCandidateOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        candidates = outcome.data.candidates
        return {
            "candidates": len(candidates),
            "best_confidence": round(candidates[0].final_confidence, 3) if candidates else None,
            "order_changed": outcome.data.order_changed,
        }
