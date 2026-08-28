"""Turning evidence into a confidence number, deterministically.

No language model touches any value here. The same evidence always produces the
same confidence, a reviewer can recompute it by hand, and a low score can
always be explained by naming which signal was weak. An invented confidence
would be indistinguishable from a real one right up to the point where somebody
relied on it.

The shape is a weighted sum of the evidence signals, multiplied by validation.
The distinction is deliberate: homology, alignment, conservation, relatedness
and plausibility are all *degrees* of support and trade off against one
another, whereas validation is a gate - a sequence that fails a biological
check should not be rescued by scoring well elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from reconstruction_agent.domain.enums import CandidateOrigin, ConfidenceLevel
from reconstruction_agent.domain.models.candidate import CandidateScores


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    """How much each evidence signal counts. Configurable by design.

    Evo 2 carries the least weight of the evidence terms because it is a
    modelling opinion rather than an observation: it says what a genome model
    expects, not what any real organism has. It is worth having when homology
    is ambiguous and should never outvote homology when homology is clear.
    """

    homology: float = 0.30
    alignment: float = 0.25
    conservation: float = 0.20
    evolutionary: float = 0.15
    evo2: float = 0.10


@dataclass(frozen=True, slots=True)
class ConfidenceThresholds:
    """Where a numeric confidence becomes a reported band."""

    high: float = 0.75
    medium: float = 0.45
    #: Below this a candidate is never returned; the gap is reported unresolved.
    minimum: float = 0.15
    #: The most a model-generated fill may ever score. Set below `high` on
    #: purpose: a prediction is reportable, and is never reported as strongly
    #: as an observation. Raising this to 1.0 would let Evo 2 outrank the
    #: evidence it was brought in to supplement.
    model_ceiling: float = 0.60


class ConfidenceEngine:
    """Combines evidence signals into a final confidence."""

    def __init__(
        self,
        weights: ScoringWeights | None = None,
        thresholds: ConfidenceThresholds | None = None,
    ) -> None:
        self.weights = weights or ScoringWeights()
        self.thresholds = thresholds or ConfidenceThresholds()

    def confidence(
        self, scores: CandidateScores, origin: CandidateOrigin = CandidateOrigin.HOMOLOGY
    ) -> float:
        """The final confidence for one candidate, 0..1."""
        if origin is CandidateOrigin.MODEL:
            return self._model_confidence(scores)

        terms: list[tuple[float, float]] = [
            (self.weights.homology, scores.homology),
            (self.weights.alignment, scores.alignment),
            (self.weights.conservation, scores.conservation),
            (self.weights.evolutionary, scores.evolutionary),
        ]

        # Evo 2 is absent from most candidates. Its weight is redistributed
        # over the observed signals rather than scored as zero: "not asked" and
        # "asked and disagreed" are different findings, and conflating them
        # would penalise every candidate that never needed arbitration.
        if scores.evo2 is not None:
            terms.append((self.weights.evo2, scores.evo2))

        total_weight = sum(weight for weight, _ in terms)
        if total_weight <= 0:
            return 0.0

        weighted = sum(weight * value for weight, value in terms) / total_weight
        return _clamp(weighted * scores.validation)

    def _model_confidence(self, scores: CandidateScores) -> float:
        """Confidence for a sequence no organism is known to carry.

        Scored on the two things that actually apply to it - how sure the model
        was, and whether the result is plausible sequence - rather than on
        homology terms that were never measurable. Scoring the unmeasurable as
        zero would put every model prediction below the floor and make the
        whole capability inert.

        Then capped. A prediction must never be reportable as confidently as a
        fill twelve sequenced relatives agree on, however sure the model was:
        the ceiling is what keeps the two distinguishable in the response even
        if a reader ignores `origin` entirely.
        """
        certainty = scores.evo2 if scores.evo2 is not None else 0.0
        return _clamp(min(certainty * scores.validation, self.thresholds.model_ceiling))

    def level(self, confidence: float) -> ConfidenceLevel:
        """The reported band for a numeric confidence."""
        if confidence >= self.thresholds.high:
            return ConfidenceLevel.HIGH
        if confidence >= self.thresholds.medium:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def is_returnable(self, confidence: float) -> bool:
        """Whether a candidate may be returned as a reconstruction at all.

        Below the floor the honest answer is that the region could not be
        resolved. Returning a low-confidence sequence with a warning attached
        invites it to be used anyway.
        """
        return confidence >= self.thresholds.minimum

    def explain(
        self,
        scores: CandidateScores,
        confidence: float,
        origin: CandidateOrigin = CandidateOrigin.HOMOLOGY,
    ) -> str:
        """Why this candidate scored as it did, in one line."""
        if origin is CandidateOrigin.MODEL:
            certainty = scores.evo2 if scores.evo2 is not None else 0.0
            return (
                f"confidence {confidence:.2f} from an Evo 2 prediction "
                f"(model certainty {certainty:.2f}, validation {scores.validation:.2f}). "
                "No sequenced organism is known to carry this fill."
            )
        return self._explain_observed(scores, confidence)

    def _explain_observed(self, scores: CandidateScores, confidence: float) -> str:
        """Why this candidate scored as it did, in one line.

        Written from the numbers rather than by a model, so it can never
        describe reasoning that did not happen.
        """
        parts = [
            f"homology {scores.homology:.2f}",
            f"alignment {scores.alignment:.2f}",
            f"conservation {scores.conservation:.2f}",
            f"relatedness {scores.evolutionary:.2f}",
        ]
        if scores.evo2 is not None:
            parts.append(f"Evo 2 agreement {scores.evo2:.2f}")
        if scores.validation < 1.0:
            parts.append(f"validation {scores.validation:.2f}")
        return f"confidence {confidence:.2f} from " + ", ".join(parts)


def homology_score(
    identity: float, coverage: float, supporting_hits: int, *, saturation: int = 5
) -> float:
    """Support from the homologues proposing a fill.

    Depth saturates: five independent homologues agreeing is strong, and the
    fiftieth adds almost nothing, so the count is damped rather than linear.
    Identity and coverage multiply because both are necessary - a perfect match
    over a tenth of the query is not good evidence.
    """
    depth = min(supporting_hits, saturation) / saturation
    return _clamp(identity * max(coverage, 0.1) * (0.5 + 0.5 * depth))


def alignment_score(spanning_references: int, total_references: int) -> float:
    """How much of the aligned evidence actually crosses the gap.

    Zero when nothing spans it, whatever else the alignment looks like.
    """
    if spanning_references <= 0 or total_references <= 0:
        return 0.0
    return _clamp(spanning_references / total_references)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
