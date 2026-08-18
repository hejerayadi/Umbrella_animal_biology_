"""How a candidate's confidence is computed, and what it must clear to ship.

Isolated from the ranker on purpose: scoring is the part most likely to be
retuned as the evaluation suite grows, and keeping it here means that happens
without touching how candidates are built.
"""
from __future__ import annotations

from dataclasses import dataclass

from contracts.output import ReconstructionStatus
from domain.models import Candidate, GapContext


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """Scores a candidate on the evidence behind it.

    Three signals are combined by weighted mean, each on 0..1:

    - `support`  - how much the aligned references agreed per column.
    - `identity` - mean alignment identity over the flanking context.
    - `flanks`   - whether the gap was anchored on one side or both.

    That base is then scaled by two multiplicative factors, for gap length and
    for evidence depth. Both are multiplicative rather than weighted terms
    because each can invalidate the others outright:

    - A long gap is less constrained by its flanks however well they align, so
      a score ignoring length would be most overconfident on exactly the cases
      that matter most.
    - Depth cannot be a weighted term because `support` is degenerate when only
      one reference contributed: a lone reference always agrees with itself and
      scores 1.0. Weighted, that gave a single-reference consensus ~0.87 -
      confidently reporting a guess. As a factor, it caps what thin evidence
      can ever reach, no matter how clean that one alignment looks.
    """

    support_weight: float = 0.4
    identity_weight: float = 0.4
    flank_weight: float = 0.2

    # References beyond this add no further confidence.
    saturating_reference_count: int = 5
    # What a single-reference consensus is capped at, before length is applied.
    minimum_depth_factor: float = 0.5

    # Gap length at which the length penalty reaches its floor.
    length_penalty_scale: float = 2000.0
    minimum_length_factor: float = 0.4

    # See `scripts/tune_settings.py`: the lowest threshold that accepts no
    # incorrect reconstruction across a 360-scenario sweep with known answers.
    minimum_confidence: float = 0.65

    def score(
        self,
        candidate: Candidate,
        context: GapContext,
        *,
        mean_identity: float = 0.0,
    ) -> float:
        support = candidate.support or 0.0
        flanks = 1.0 if context.has_both_flanks else (0.5 if context.has_usable_flanks else 0.0)

        total_weight = self.support_weight + self.identity_weight + self.flank_weight
        base = (
            support * self.support_weight
            + mean_identity * self.identity_weight
            + flanks * self.flank_weight
        ) / total_weight

        scaled = (
            base
            * self._length_factor(context.gap.length)
            * self._depth_factor(len(candidate.supporting_references))
        )
        return round(scaled, 4)

    def _length_factor(self, gap_length: int) -> float:
        """Multiplier that decays with gap length, floored so it never reaches 0.

        A long gap with excellent evidence should still be reportable - just
        never at full confidence.
        """
        decay = 1.0 - (gap_length / self.length_penalty_scale)
        return max(self.minimum_length_factor, min(1.0, decay))

    def _depth_factor(self, reference_count: int) -> float:
        """Multiplier rising from `minimum_depth_factor` at one reference to 1.0.

        Zero references means the candidate came from nowhere traceable, which
        is not evidence at all.
        """
        if reference_count <= 0:
            return 0.0
        if reference_count >= self.saturating_reference_count:
            return 1.0

        span = self.saturating_reference_count - 1
        growth = (1.0 - self.minimum_depth_factor) * ((reference_count - 1) / span)
        return self.minimum_depth_factor + growth

    def classify(
        self, confidence: float, *, threshold: float | None = None
    ) -> ReconstructionStatus:
        """Whether a score is good enough to report as a reconstruction."""
        floor = self.minimum_confidence if threshold is None else threshold
        return (
            ReconstructionStatus.RECONSTRUCTED
            if confidence >= floor
            else ReconstructionStatus.LOW_CONFIDENCE
        )

    def aggregate(self, confidences: list[float]) -> float:
        """Overall confidence for a whole sequence.

        The minimum, not the mean: a reconstruction is only as trustworthy as
        its weakest filled gap, and averaging would let one solid gap mask
        several poor ones.
        """
        return round(min(confidences), 4) if confidences else 0.0
