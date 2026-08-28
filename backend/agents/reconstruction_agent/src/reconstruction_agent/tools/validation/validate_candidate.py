"""Run the biological checks against a candidate, on demand.

Validation already runs inside candidate generation, so this tool exists for
the case where it needs running *again*: after arbitration has re-ranked the
candidates and a different sequence now leads, or when the critic named
BIOLOGICAL_VALIDATION_FAILED and the replanner wants the verdicts stated
explicitly rather than inferred from a score.

Validation is a gate, not a signal. A sequence that fails a disqualifying check
is not rescued by scoring well on homology, which is why the aggregate is
multiplied into the confidence rather than weighted alongside it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.enums import ToolName
from reconstruction_agent.domain.models.candidate import Candidate, ValidationVerdict
from reconstruction_agent.domain.models.evidence import EvidenceContribution
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.services.validation.validators import aggregate_score, validate
from reconstruction_agent.tools.base import Tool, ToolOutcome


class ValidateCandidateInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    gap_id: str
    candidates: tuple[Candidate, ...]
    context: GapContext


class ValidateCandidateOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidates: tuple[Candidate, ...] = ()
    #: Checks the leading candidate failed. Empty when it passed everything.
    failures: tuple[str, ...] = ()


class ValidateCandidateTool(Tool[ValidateCandidateInput, ValidateCandidateOutput]):
    """Deterministic biological checks, with a reason for each verdict."""

    name = ToolName.VALIDATE_CANDIDATE
    description = (
        "Re-run the biological checks - alphabet, length, GC consistency, complexity - "
        "against the current candidates and fold the result into their confidence."
    )
    input_model = ValidateCandidateInput

    def __init__(self, engine: ConfidenceEngine) -> None:
        self._engine = engine

    async def run(self, request: ValidateCandidateInput) -> ToolOutcome[ValidateCandidateOutput]:
        if not request.candidates:
            return ToolOutcome(
                tool=self.name, ok=False, reason="There are no candidates to validate."
            )

        validated = tuple(
            self._apply(candidate, request.context) for candidate in request.candidates
        )
        ranked = tuple(sorted(validated, key=lambda item: -item.final_confidence))
        failures = ranked[0].failed_checks()

        return ToolOutcome(
            tool=self.name,
            # A candidate that fails validation is a finding, not a tool error:
            # the run learned something true about the sequence.
            ok=not failures,
            data=ValidateCandidateOutput(candidates=ranked, failures=failures),
            reason=(f"The leading candidate failed: {', '.join(failures)}." if failures else ""),
        )

    def _apply(self, candidate: Candidate, context: GapContext) -> Candidate:
        verdicts: tuple[ValidationVerdict, ...] = validate(candidate.sequence, context)
        scores = candidate.scores.model_copy(update={"validation": aggregate_score(verdicts)})
        confidence = self._engine.confidence(scores)
        return candidate.model_copy(
            update={
                "validation_results": verdicts,
                "scores": scores,
                "final_confidence": confidence,
                "confidence_level": self._engine.level(confidence),
                "rationale": self._engine.explain(scores, confidence),
            }
        )

    def summarise(self, outcome: ToolOutcome[ValidateCandidateOutput]) -> dict[str, Any]:
        if outcome.data is None:
            return {}
        return {
            "candidates": len(outcome.data.candidates),
            "failures": list(outcome.data.failures),
        }

    def contribute(self, outcome: ToolOutcome[ValidateCandidateOutput]) -> EvidenceContribution:
        if outcome.data is None or not outcome.data.candidates:
            return EvidenceContribution()
        best = outcome.data.candidates[0]
        return EvidenceContribution(
            validation_checks=tuple(verdict.check for verdict in best.validation_results),
            validation_failures=outcome.data.failures,
        )
