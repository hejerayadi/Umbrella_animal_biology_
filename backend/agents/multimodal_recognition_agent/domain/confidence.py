"""The confidence gate: which of the three outcomes this request earned.

Three rules matter more than the thresholds themselves:

1. A classification score is not a probability. Nothing here converts one into
   a percentage or a confidence claim, and the output says so explicitly. In
   Sprint 2 the scores are deterministic test values from a mock, which makes
   the point twice over.
2. Text can never promote a result. It can hold one back (`conflict`), but a
   species the classifier did not return cannot be identified because the user
   named it.
3. `uncertain` and `not_identified` are completed outcomes. They mean the
   workflow ran and reached an honest conclusion, not that the service failed.
"""
from __future__ import annotations

from ..config import ThresholdConfig
from .models import Decision, SpeciesCandidate, TextAlignment


def decide(
    candidates: list[SpeciesCandidate],
    margin: float | None,
    alignment: TextAlignment,
    thresholds: ThresholdConfig,
) -> Decision:
    """Pick `identified`, `uncertain` or `not_identified`."""

    # The classifier named nothing. There is nothing to be uncertain about.
    if not candidates:
        return "not_identified"

    top = candidates[0].classification_score

    # Below the floor, the best label is no better than noise.
    if top < thresholds.uncertain_min_score:
        return "not_identified"

    # The instruction names a species the image evidence does not support.
    # Downgrade, never resolve it in the text's favour.
    if alignment == "conflict":
        return "uncertain"

    if top >= thresholds.identified_min_score and (margin or 0.0) >= thresholds.identified_min_margin:
        return "identified"

    return "uncertain"


def visual_evidence_is_sufficient(
    candidates: list[SpeciesCandidate], thresholds: ThresholdConfig
) -> bool:
    """Did the image produce evidence worth reasoning about at all?

    Deterministic, and deliberately narrow: no candidate, or a best candidate
    below the floor, means the image gave the classifier nothing to work with.
    That is the only condition that may ask the user for a better photograph -
    it is not a general clarification route.
    """
    if not candidates:
        return False
    return candidates[0].classification_score >= thresholds.uncertain_min_score


def better_image_request(decision: Decision) -> str | None:
    """The message asking for a better image, or None when it does not apply."""
    if decision != "not_identified":
        return None
    return (
        "The image did not produce usable visual evidence. A sharper, closer photograph "
        "of the animal - ideally showing the whole body in good light - would give the "
        "classifier something to work with."
    )


def clarification_for(decision: Decision, candidates: list[SpeciesCandidate]) -> str | None:
    """A question worth asking the user, when the result is not conclusive."""

    if decision == "identified":
        return None

    if decision == "not_identified":
        # Provider-neutral on purpose. This used to name "the Sprint 2 classifier",
        # which was false the moment real remote BioCLIP-2 inference started
        # running - and this is the sentence a user sees precisely when their
        # photograph could not be identified. Describing the EVIDENCE rather than
        # the component that produced it is true in every mode, so no mode branch
        # is needed: it holds for remote BioCLIP-2, for the mock, for no
        # candidates at all, for weak candidates, and for candidates too close
        # together to separate.
        return (
            "The available visual evidence was not strong enough to identify a species "
            "confidently. Could you supply a clearer photograph of the animal, or tell me "
            "where the observation was made?"
        )

    names = ", ".join(candidate.scientific_name for candidate in candidates[:3])
    if not names:
        return "Could you provide more detail about the animal in the image?"
    return (
        f"The classification was inconclusive between: {names}. "
        "Could you confirm which of these it resembles, or where the observation was made?"
    )
