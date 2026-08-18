"""Turns gathered evidence into scored, validated candidate reconstructions.

This is the deterministic core of the agent: the LLM chooses what to gather
and reviews the result, but what the bases actually are comes from the
alignment, through this module, with no model in the path.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.prompts import deterministic_explanation
from configuration.logging import get_logger
from contracts.output import GapReconstruction, ReconstructionStatus
from domain.models import Alignment, Candidate, GapContext, Reference
from domain.policies.confidence_policy import ConfidencePolicy
from domain.services import CandidateRanker, ReconstructionValidator

_log = get_logger(__name__)


@dataclass(slots=True)
class Reasoner:
    """Builds candidates from alignments, scores them, and validates them."""

    ranker: CandidateRanker
    validator: ReconstructionValidator
    confidence: ConfidencePolicy

    def build_candidates(
        self,
        context: GapContext,
        alignment: Alignment,
        references: list[Reference],
    ) -> list[Candidate]:
        """Scored candidates for one gap.

        Currently one consensus candidate per alignment. The list return is not
        speculative: per-reference candidates and a second method both belong
        here, and the ranker already exists to choose between them.
        """
        consensus = self.ranker.build_consensus(context, alignment, references)
        if consensus is None:
            return []

        score = self.confidence.score(
            consensus, context, mean_identity=alignment.mean_identity()
        )
        explanation = deterministic_explanation(
            gap_id=context.identifier,
            length=consensus.length,
            confidence=score,
            references=consensus.supporting_references,
            mean_identity=alignment.mean_identity(),
        )
        return [consensus.scored(score, explanation=explanation)]

    def finalise(
        self,
        context: GapContext,
        candidates: list[Candidate],
        *,
        threshold: float | None = None,
    ) -> GapReconstruction:
        """The reportable outcome for one gap.

        Every path returns a `GapReconstruction` rather than None: a gap that
        could not be resolved is a result the caller needs, not an absence.
        """
        best = self.ranker.best(candidates)

        if best is None:
            return GapReconstruction(
                gap_id=context.identifier,
                start=context.gap.start,
                end=context.gap.end,
                length=context.gap.length,
                status=ReconstructionStatus.UNRESOLVED,
                explanation=deterministic_explanation(
                    gap_id=context.identifier,
                    length=context.gap.length,
                    confidence=0.0,
                    references=[],
                    mean_identity=None,
                ),
            )

        report = self.validator.validate(best, context)
        if not report.is_valid:
            # A candidate that fails a hard check is reported as unresolved
            # with the reasons attached, never silently downgraded.
            _log.info(
                "candidate_rejected", gap_id=context.identifier, errors=report.errors
            )
            return GapReconstruction(
                gap_id=context.identifier,
                start=context.gap.start,
                end=context.gap.end,
                length=context.gap.length,
                status=ReconstructionStatus.UNRESOLVED,
                confidence=0.0,
                evidence=best.evidence,
                explanation=(
                    "A candidate was produced but failed validation: "
                    + "; ".join(report.errors)
                ),
            )

        confidence = best.confidence or 0.0
        return GapReconstruction(
            gap_id=context.identifier,
            start=context.gap.start,
            end=context.gap.end,
            length=context.gap.length,
            status=self.confidence.classify(confidence, threshold=threshold),
            reconstructed_sequence=best.sequence,
            confidence=confidence,
            evidence=best.evidence,
            explanation=self._with_warnings(best.explanation, report.warnings),
        )

    @staticmethod
    def _with_warnings(explanation: str | None, warnings: list[str]) -> str | None:
        if not warnings:
            return explanation
        joined = " ".join(warnings)
        return f"{explanation} Caveats: {joined}" if explanation else f"Caveats: {joined}"
