"""How a candidate's confidence is computed, and what it must clear to ship.

Isolated from the ranker on purpose: scoring is the part most likely to be
retuned as the evaluation suite grows, and keeping it here means that happens
without touching how candidates are built.

The score has to do one job: rank correct reconstructions above incorrect ones,
so that a threshold can trade recall for precision. Measured on the 360-scenario
sweep, the previous formula ranked them at 0.575 - barely above the 0.5 that
means no signal at all. Raising the threshold therefore discarded good answers
at almost the same rate as bad ones, which is how the agent ended up refusing
seven gaps in ten to avoid twenty-two mistakes.

The rebuild keeps the same shape - a weighted mean of evidence terms, scaled by
factors that can each invalidate the rest - but widens what counts as evidence
and, crucially, makes ambiguity a *factor* rather than a term. An evenly split
consensus is not weak evidence to be averaged against strong flanking identity;
it is a coin flip, and no amount of flanking identity redeems it.
"""
from __future__ import annotations

from dataclasses import dataclass

from contracts.output import ReconstructionStatus
from domain.models import Candidate, GapContext, Reference


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """Scores a candidate on the evidence behind it.

    Five signals are combined by weighted mean, each on 0..1:

    - `support`      - how decisively the references agreed, per column.
    - `conservation` - mean alignment identity over the flanking context.
    - `flanks`       - whether the gap was anchored on one side or both.
    - `homology`     - measured BLAST identity and coverage of the references
                       that actually supported the fill.
    - `proximity`    - how closely related those references are to the target.

    The last two are new. They are what separates "five sequences agreed" from
    "five *well-matched, closely related* sequences agreed", which the score
    previously could not tell apart.

    That base is then scaled by three multiplicative factors - length, depth and
    decisiveness. Each is multiplicative rather than a weighted term because
    each can invalidate the others outright:

    - A long gap is less constrained by its flanks however well they align, so
      a score ignoring length would be most overconfident on exactly the cases
      that matter most.
    - Depth cannot be a weighted term because `support` is degenerate when only
      one reference contributed: a lone reference always agrees with itself.
      As a factor, it caps what thin evidence can reach.
    - Decisiveness is the explicit ambiguity penalty. A 50/50 split produced
      every confidently-wrong answer in the sweep, and averaging cannot express
      "this particular signal being bad makes the rest irrelevant".
    """

    support_weight: float = 0.30
    identity_weight: float = 0.25
    flank_weight: float = 0.15
    homology_weight: float = 0.15
    proximity_weight: float = 0.15

    # References beyond this add no further confidence. Lowered from five:
    # three independent, well-matched relatives that agree unanimously are
    # strong evidence, and demanding five meant most real gaps - where a
    # handful of good homologues exist - could never clear the bar.
    saturating_reference_count: int = 3
    # What a single-reference consensus is capped at, before length is applied.
    # Raised from 0.5, which put one and two references arithmetically below
    # any usable threshold however perfect the evidence. The ambiguity factor
    # below now carries the job of suppressing thin *and* split evidence, which
    # is what depth was really being used for.
    minimum_depth_factor: float = 0.75

    # Gap length at which the length penalty reaches its floor. Raised to match
    # `ValidationPolicy.max_gap_length`: at 2000 the two policies disagreed, and
    # a gap the admission policy accepted could not clear the reporting
    # threshold however good its evidence.
    length_penalty_scale: float = 5000.0
    minimum_length_factor: float = 0.4

    # Consensus decisiveness at or above which the evidence is treated as
    # settled, and below which confidence is actively crushed. 0.5 means the
    # winning base beat the runner-up by half the total vote weight.
    decisive_support: float = 0.5
    ambiguous_support: float = 0.1
    minimum_decisiveness_factor: float = 0.25

    # See `scripts/tune_settings.py`: derived from a stated precision floor
    # rather than picked, and re-derived whenever the score changes.
    minimum_confidence: float = 0.15

    #: Optional independent check. Not part of the weighted mean: a language
    #: model's opinion must not manufacture confidence that the alignment
    #: evidence does not support, so it can only scale an existing score down.
    #: This is the *floor* of that scaling - what a fill Evo 2 flatly
    #: contradicts retains, rather than losing everything the alignment earned.
    plausibility_floor: float = 0.6

    def score(
        self,
        candidate: Candidate,
        context: GapContext,
        *,
        mean_identity: float = 0.0,
        references: list[Reference] | None = None,
        plausibility: float | None = None,
        arbitration_margin: float | None = None,
    ) -> float:
        """Confidence for one candidate, on 0..1.

        `references` are those that actually supported the fill, not every
        reference gathered for the gap - a sequence that voted against the
        answer says nothing good about it.
        """
        support = candidate.support or 0.0
        flanks = 1.0 if context.has_both_flanks else (0.5 if context.has_usable_flanks else 0.0)
        supporting = self._supporting(candidate, references)

        total_weight = (
            self.support_weight
            + self.identity_weight
            + self.flank_weight
            + self.homology_weight
            + self.proximity_weight
        )
        base = (
            support * self.support_weight
            + mean_identity * self.identity_weight
            + flanks * self.flank_weight
            + self._homology(supporting) * self.homology_weight
            + self._proximity(supporting) * self.proximity_weight
        ) / total_weight

        scaled = (
            base
            * self._length_factor(context.gap.length)
            * self._depth_factor(len(candidate.supporting_references))
            * self._decisiveness_factor(
                self._settled_support(support, arbitration_margin)
            )
        )

        if plausibility is not None:
            scaled *= self._plausibility_factor(plausibility)

        return round(min(1.0, max(0.0, scaled)), 4)

    @staticmethod
    def _supporting(
        candidate: Candidate, references: list[Reference] | None
    ) -> list[Reference]:
        """The reference objects behind this candidate, when they are known.

        The candidate carries accessions; the metadata lives on the references.
        When they were not supplied - several call sites score without them -
        the homology and proximity terms fall back to neutral rather than zero,
        so an unmeasured reference is not treated as a bad one.
        """
        if not references:
            return []
        supporting = set(candidate.supporting_references)
        return [
            reference for reference in references if reference.accession in supporting
        ] or list(references)

    @staticmethod
    def _mean(values: list[float], default: float) -> float:
        return sum(values) / len(values) if values else default

    def _homology(self, references: list[Reference]) -> float:
        """Measured alignment quality of the supporting references.

        E-value is deliberately not averaged in directly: it spans many orders
        of magnitude and is already reflected in identity and coverage for any
        hit strong enough to have been retrieved. It is used as a veto instead
        - a hit that failed the search's own significance cut never arrives.
        """
        return self._mean(
            [r.quality for r in references if r.quality > 0.0], default=0.5
        )

    def _proximity(self, references: list[Reference]) -> float:
        """How closely related the supporting references are to the target."""
        return self._mean(
            [r.relatedness for r in references if r.relatedness is not None], default=0.5
        )

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

        span = max(1, self.saturating_reference_count - 1)
        growth = (1.0 - self.minimum_depth_factor) * ((reference_count - 1) / span)
        return self.minimum_depth_factor + growth

    def _settled_support(
        self, support: float, arbitration_margin: float | None
    ) -> float:
        """Decisiveness after an independent source has weighed in.

        The ambiguity penalty exists because an evenly split vote carries no
        information about which base is right. When something outside the vote
        answers that question - Evo 2 separating two candidate fills - the
        ambiguity is *resolved*, and continuing to charge the full penalty
        would reject a fill that has just been settled. That is not
        hypothetical: on a live run the model predicted the true fill exactly,
        scored it 1.0 against the decoy's 0.0, and the answer still came out
        below the reporting threshold.

        Relief is bounded at `decisive_support`, so arbitration can lift a gap
        to "settled" and no further. Everything else - depth, length, homology,
        proximity - still caps the result, which is what keeps a genome model
        from manufacturing confidence the alignment does not support.
        """
        if arbitration_margin is None:
            return support

        earned = min(1.0, max(0.0, arbitration_margin)) * self.decisive_support
        return max(support, earned)

    def _decisiveness_factor(self, support: float) -> float:
        """The ambiguity penalty: how much a split consensus costs.

        Linear between `ambiguous_support` and `decisive_support`, flat outside.
        Every confidently-wrong reconstruction in the sweep was an evenly split
        vote that scored respectably because good flanking identity averaged the
        split away. Applying it as a factor makes that impossible: a coin flip
        is crushed to a quarter of whatever the rest of the evidence earned.
        """
        if support >= self.decisive_support:
            return 1.0
        if support <= self.ambiguous_support:
            return self.minimum_decisiveness_factor

        span = self.decisive_support - self.ambiguous_support
        position = (support - self.ambiguous_support) / span
        return self.minimum_decisiveness_factor + position * (
            1.0 - self.minimum_decisiveness_factor
        )

    def _plausibility_factor(self, plausibility: float) -> float:
        """How an independent model's opinion may move the score.

        Only downward, and continuously. Evo 2 predicting a different
        continuation is a reason to doubt a fill the alignment supports; it
        agreeing is not a reason to trust one the alignment does not - so the
        factor reaches 1.0 at full agreement and never exceeds it.

        Continuity is what makes arbitration work. A step function that was
        flat above some floor could not separate two candidates the model
        scored 0.8 and 0.7, which is exactly the comparison a contested gap
        needs. Bounded below so a genome model can cast doubt on a
        well-evidenced fill without erasing it.
        """
        bounded = min(1.0, max(0.0, plausibility))
        return self.plausibility_floor + (1.0 - self.plausibility_floor) * bounded

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

    @staticmethod
    def aggregate(confidences: list[float]) -> float:
        """One number for a whole run: the weakest gap it reports.

        The minimum rather than the mean, because a result is only as
        trustworthy as its shakiest part.
        """
        return round(min(confidences), 4) if confidences else 0.0
