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
        """Scored candidates for one gap, best-first.

        One when the consensus is clean, two when it is contested. The second
        is not a courtesy: where the references split evenly, the plurality
        string is one hypothesis and the runner-up is another, and the vote by
        itself cannot say which is right. Carrying both is what gives an
        independent check - Evo 2 - something to arbitrate between, and it is
        why a 50/50 gap can now be settled instead of only distrusted.
        """
        candidates = self.ranker.build_alternatives(context, alignment, references)
        if not candidates:
            return []

        scored: list[Candidate] = []
        for candidate in candidates:
            score = self.confidence.score(
                candidate,
                context,
                mean_identity=alignment.mean_identity(),
                # The references carry the measured homology and the
                # phylogenetic proximity; without them the score cannot tell
                # "five sequences agreed" from "five well-matched close
                # relatives agreed".
                references=references,
            )
            explanation = deterministic_explanation(
                gap_id=context.identifier,
                length=candidate.length,
                confidence=score,
                references=candidate.supporting_references,
                mean_identity=alignment.mean_identity(),
            )
            scored.append(candidate.scored(score, explanation=explanation))

        return self.ranker.rank(scored)

    def arbitrate(
        self,
        context: GapContext,
        candidates: list[Candidate],
        alignment: Alignment,
        references: list[Reference],
        *,
        scores: dict[str, float],
    ) -> list[Candidate]:
        """Re-decide a contested gap using Evo 2's read of the sequence context.

        The candidates are keyed positionally - `candidate_0`, `candidate_1` -
        because that is how they were handed to the tool, and re-deriving the
        mapping from sequence text would break the moment two candidates shared
        a prefix.

        Each candidate is re-scored with its own plausibility, so the model's
        opinion enters through the confidence policy rather than around it. That
        matters: the policy lets plausibility *lower* a score and never raise
        one, so a genome model cannot promote a fill the alignment does not
        support - it can only break a tie between fills the alignment already
        allows, and cast doubt on one it dislikes.
        """
        if not scores:
            return candidates

        # Relative, not absolute. Evo 2's agreement with any real sequence is
        # low in absolute terms, so scoring candidates on the raw figure would
        # penalise both equally and settle nothing. Against the best of them,
        # the preferred fill keeps its score intact and the rejected one is
        # discounted by exactly how much less the model expected it.
        ordered = sorted(scores.values(), reverse=True)
        best = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else 0.0
        # How cleanly the model separated the leaders. A near-tie in its own
        # opinion settles nothing and earns no relief from the ambiguity
        # penalty; a decisive preference earns it in proportion.
        margin = ((best - runner_up) / best) if best > 0 else 0.0

        rescored: list[Candidate] = []
        for index, candidate in enumerate(candidates):
            raw = scores.get(f"candidate_{index}")
            plausibility = (raw / best) if raw is not None and best > 0 else raw
            if plausibility is None:
                rescored.append(candidate)
                continue

            score = self.confidence.score(
                candidate,
                context,
                mean_identity=alignment.mean_identity(),
                references=references,
                plausibility=plausibility,
                # Only the fill the model actually preferred has its ambiguity
                # resolved; the ones it rejected remain as unsettled as the
                # vote left them.
                arbitration_margin=margin if raw == best else None,
            )
            explanation = deterministic_explanation(
                gap_id=context.identifier,
                length=candidate.length,
                confidence=score,
                references=candidate.supporting_references,
                mean_identity=alignment.mean_identity(),
            )
            rescored.append(
                candidate.scored(
                    score,
                    explanation=(
                        f"{explanation} Evo 2 scored this fill "
                        f"{plausibility:.2f} against its own prediction."
                    ),
                )
            )

        return self.ranker.rank(rescored)

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
