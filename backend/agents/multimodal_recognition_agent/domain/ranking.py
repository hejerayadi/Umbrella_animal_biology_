"""Validate, order and cap the classifier's taxon predictions.

The classification boundary returns labels that are *already ranked* - that is
part of its contract, and this module's job is to hold it to that rather than to
re-derive a ranking of its own. There is no grouping here, no aggregation over
reference points and no distance arithmetic, because none of those concepts
exist in a label classifier's output.

What this module does enforce, on whatever provider is plugged in:

1. every prediction carries the mandatory fields;
2. every score is a real number in [0, 1];
3. the list is in non-increasing score order;
4. no species appears twice;
5. no more than `top_k` candidates leave this module.

A provider that breaks any of those is not producing a smaller problem to be
patched up - it is not answering to contract, so the request fails with a
controlled code instead of being served a repaired answer.
"""
from __future__ import annotations

from typing import Any

from .errors import ErrorCode, RecognitionError
from .models import BioCLIPTaxonPrediction, SpeciesCandidate

# Required on every prediction. A label missing any of these cannot be reported
# honestly - we would not know which taxon it names, or how the classifier
# ranked it - so the whole fixture is rejected rather than guessed at.
REQUIRED_PREDICTION_FIELDS = ("species_id", "scientific_name", "classification_score")

# Scores are bounded because the whole downstream confidence gate is expressed
# in the same units. A score outside this range would silently walk past every
# threshold comparison.
MIN_SCORE = 0.0
MAX_SCORE = 1.0

# Floating-point slack when checking that a list is ordered. Two scores written
# as 0.7 and 0.7 in a fixture must not be called "out of order".
_ORDER_TOLERANCE = 1e-9


def validate_prediction_payload(payload: Any) -> bool:
    """True when a raw prediction record carries every mandatory field, with a
    usable score. Used when loading a fixture, before anything is constructed."""
    if not isinstance(payload, dict):
        return False
    for field in REQUIRED_PREDICTION_FIELDS:
        if payload.get(field) in (None, ""):
            return False
    if not isinstance(payload.get("species_id"), str):
        return False
    if not isinstance(payload.get("scientific_name"), str):
        return False

    score = payload.get("classification_score")
    # `bool` is a subclass of `int`; True would otherwise pass as the score 1.
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    if not MIN_SCORE <= float(score) <= MAX_SCORE:
        return False

    rank = payload.get("rank", "species")
    return rank == "species"


def assert_ranked_and_distinct(
    predictions: list[BioCLIPTaxonPrediction],
    *,
    error: ErrorCode = ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION,
) -> None:
    """Hold the provider to its "already ranked, distinct labels" promise.

    Raises rather than sorting. Silently re-ordering a provider's output would
    hide the fact that it is misbehaving, and in Sprint 2 the provider is a test
    oracle - an oracle that lies is worth failing on.
    """
    seen: set[str] = set()
    previous: float | None = None

    for prediction in predictions:
        if not MIN_SCORE <= prediction.classification_score <= MAX_SCORE:
            raise RecognitionError(error)
        if prediction.species_id in seen:
            raise RecognitionError(error)
        seen.add(prediction.species_id)

        if previous is not None and prediction.classification_score > previous + _ORDER_TOLERANCE:
            raise RecognitionError(error)
        previous = prediction.classification_score


def build_candidates(
    predictions: list[BioCLIPTaxonPrediction],
    *,
    top_k: int,
) -> list[SpeciesCandidate]:
    """Turn validated predictions into candidates, capped at `top_k`.

    Taxonomy fields are left unset here. The mocked GBIF and NCBI sources fill
    them in a later node, and inventing them at this point is exactly what the
    specification forbids.
    """
    if top_k <= 0:
        raise RecognitionError(ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION)

    assert_ranked_and_distinct(predictions)

    return [
        SpeciesCandidate(
            species_id=prediction.species_id,
            scientific_name=prediction.scientific_name,
            common_name=prediction.common_name,
            rank=prediction.rank,
            classification_score=prediction.classification_score,
            taxonomy_status="unverified",
        )
        for prediction in predictions[:top_k]
    ]


def top_margin(candidates: list[SpeciesCandidate]) -> float | None:
    """Gap between the best and second-best species.

    None when there is nothing to compare. A single candidate gets a margin of
    1.0 - there is no runner-up to be confused with.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return 1.0
    return candidates[0].classification_score - candidates[1].classification_score
