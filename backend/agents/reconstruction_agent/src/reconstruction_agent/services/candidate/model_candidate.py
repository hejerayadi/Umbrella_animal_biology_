"""Building a candidate from an Evo 2 continuation.

This is the path that answers a gap homology could not. It is deliberately kept
apart from `CandidateBuilder`, which assembles fills from references that
actually span the region: mixing the two would put a prediction and an
observation through the same constructor, and the difference between them is
the most consequential thing about either.

Everything here exists to keep that difference visible downstream:

- `origin` is `MODEL`, and the confidence engine scores it on a separate,
  capped path rather than on homology terms that were never measurable.
- `supporting_hits` and `supporting_organisms` are empty, and truthfully so.
  No sequenced organism is known to carry this fill.
- The rationale says as much in words, because a caller reading a summary and
  not a schema still has to be told.

Validation runs unchanged. A model is perfectly capable of writing a
low-complexity run or a sequence whose GC content is nothing like its flanks,
and being confident while doing it.
"""

from __future__ import annotations

from reconstruction_agent.domain.enums import CandidateOrigin
from reconstruction_agent.domain.models.candidate import Candidate, CandidateScores
from reconstruction_agent.domain.models.sequence import GapContext
from reconstruction_agent.services.scoring.confidence_engine import ConfidenceEngine
from reconstruction_agent.services.validation.validators import (
    ValidationThresholds,
    aggregate_score,
    validate,
)

#: Suffix marking the candidate in ids and logs. Chosen so that a reader
#: scanning a candidate list sees the provenance without opening the object.
MODEL_CANDIDATE_SUFFIX = "evo2"


def build_model_candidate(
    sequence: str,
    context: GapContext,
    *,
    engine: ConfidenceEngine,
    model_certainty: float | None,
    thresholds: ValidationThresholds | None = None,
) -> Candidate | None:
    """A candidate carrying an Evo 2 fill, or None if there is nothing usable.

    Returns None rather than a zero-confidence candidate when the continuation
    is the wrong length: a fill that does not match the gap would shift every
    coordinate after it, and offering it as a low-scoring option invites it to
    be taken.
    """
    fill = sequence.strip().upper()
    if not fill or len(fill) != context.gap.length:
        return None

    verdicts = validate(fill, context, thresholds)
    scores = CandidateScores(
        # Truthfully zero: no homologue proposed this, nothing aligned to
        # support it, no organism sits at any taxonomic distance from it. The
        # engine knows not to read these as a weak measurement, because
        # `origin` tells it they were never measurable.
        homology=0.0,
        alignment=0.0,
        conservation=0.0,
        evolutionary=0.0,
        evo2=model_certainty if model_certainty is not None else 0.0,
        validation=aggregate_score(verdicts),
    )
    confidence = engine.confidence(scores, CandidateOrigin.MODEL)

    return Candidate(
        candidate_id=f"{context.gap_id}_{MODEL_CANDIDATE_SUFFIX}",
        gap_id=context.gap_id,
        origin=CandidateOrigin.MODEL,
        sequence=fill,
        supporting_hits=(),
        supporting_organisms=(),
        scores=scores,
        validation_results=verdicts,
        final_confidence=confidence,
        confidence_level=engine.level(confidence),
        rationale=engine.explain(scores, confidence, CandidateOrigin.MODEL),
    )
