"""Evo 2 in the pipeline: it proposes the missing bases, and it judges them.

Evo 2 has no endpoint that scores an existing sequence. Both capabilities here
are therefore built on the one thing it does offer - continuing a prompt - and
they are the same call read two ways:

**Generation.** The continuation *is* a proposed fill. This is what lets a gap
with no gap-spanning homologue be answered at all: homology returns nothing,
and a model trained on genomes still has something to say about what belongs
between two flanks.

**Agreement.** Where candidates already exist, each is compared against that
same continuation, position by position. A high score means the model would
have written something similar. It is not a likelihood and it is not evidence
that the sequence is correct.

One call serves both. The candidates all occupy the same position after the
same flank, so asking per candidate would multiply the cost and introduce
sampling variance between comparisons meant to be directly comparable.

What the module will not do is let the two be confused downstream. A generated
fill is returned marked as a prediction, and the confidence engine scores it on
a separate, capped path - because a sequence no organism is known to carry must
never be reportable as strongly as one a dozen sequenced relatives agree on.

The agreement is weighted by the model's own sampling probabilities where the
endpoint returns them. Without that weighting a base the model emitted at 0.99
certainty and one it emitted at 0.26 would count the same, and a disagreement
on a base the model was unsure of would be read as evidence against a candidate
that real homology supports.
"""

from __future__ import annotations

from dataclasses import dataclass

from reconstruction_agent.domain.exceptions import ExternalServiceError, ReconstructionError
from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.integrations.evo2.client import Evo2Client
from reconstruction_agent.integrations.evo2.models import Evo2Generation
from reconstruction_agent.observability.logger import get_logger

_log = get_logger(__name__)

#: Below this, the model wrote something substantially different from the
#: candidate and the disagreement is worth reporting to the critic.
DISAGREEMENT_THRESHOLD = 0.4

#: How much of the left flank to prompt with. Long enough for the model to have
#: real context, short enough not to dominate the request.
DEFAULT_PROMPT_BASES = 500


@dataclass(frozen=True, slots=True)
class Arbitration:
    """What one Evo 2 call produced for one gap."""

    #: Agreement per candidate id, 0..1. Empty when there was nothing to judge.
    agreement: dict[str, float]
    #: The bases Evo 2 wrote. Usable as a fill in its own right when homology
    #: produced none - which is the case this capability exists for.
    generated: str = ""
    #: The model's mean certainty across its continuation, when reported.
    mean_model_confidence: float | None = None
    #: Set when Evo 2 was wanted but could not be consulted. The run continues
    #: without it: a gap that homology answered keeps its answer, and one it
    #: did not is reported unresolved rather than filled by nothing.
    unavailable_reason: str | None = None

    @property
    def consulted(self) -> bool:
        """Whether the model actually answered."""
        return self.unavailable_reason is None and bool(self.generated)

    def disagrees_with(self, candidate: Candidate) -> bool:
        """Whether the model wrote something substantially different."""
        score = self.agreement.get(candidate.candidate_id)
        return score is not None and score < DISAGREEMENT_THRESHOLD


class Evo2Service:
    """Arbitrates between candidates using Evo 2's own continuation."""

    def __init__(self, client: Evo2Client, *, prompt_bases: int = DEFAULT_PROMPT_BASES) -> None:
        self._client = client
        self._prompt_bases = prompt_bases

    @property
    def available(self) -> bool:
        return self._client.available

    async def arbitrate(
        self, candidates: tuple[Candidate, ...], *, left_flank: str, gap_length: int
    ) -> Arbitration:
        """Continue the flank, and score any existing candidates against it.

        Runs with no candidates at all: that is the case where the continuation
        is the answer rather than a cross-check, and refusing to call would
        leave the gap unresolved for want of asking.
        """
        if not self._client.available:
            return Arbitration(
                agreement={},
                unavailable_reason="NVIDIA_API_KEY is not configured; Evo 2 was not consulted.",
            )
        if not left_flank:
            return Arbitration(
                agreement={},
                unavailable_reason="No flanking sequence to prompt Evo 2 with.",
            )

        prompt = left_flank[-self._prompt_bases :]
        try:
            generation = await self._client.generate(prompt, num_tokens=gap_length)
        except ExternalServiceError as error:
            _log.info("evo2_unavailable", reason=str(error), code=error.code.value)
            return Arbitration(agreement={}, unavailable_reason=str(error))
        except ReconstructionError as error:
            _log.info("evo2_failed", reason=str(error))
            return Arbitration(agreement={}, unavailable_reason=str(error))

        if not generation.sequence:
            return Arbitration(
                agreement={}, unavailable_reason="Evo 2 returned no usable continuation."
            )

        agreement = {
            candidate.candidate_id: agreement_score(candidate.sequence, generation)
            for candidate in candidates
        }
        _log.info(
            "evo2_consulted",
            candidates=len(candidates),
            generated_bases=len(generation.sequence),
            mean_model_confidence=generation.mean_confidence,
            agreement={key: round(value, 3) for key, value in agreement.items()},
        )
        return Arbitration(
            agreement=agreement,
            generated=generation.sequence,
            mean_model_confidence=generation.mean_confidence,
        )


def agreement_score(candidate: str, generation: Evo2Generation) -> float:
    """How far `candidate` matches what Evo 2 wrote, 0..1.

    Position by position over the overlap, each match weighted by how sure the
    model was of the base it emitted there. A model that was unsure contributes
    little either way, which is what stops a low-certainty continuation from
    arguing against a candidate that homology supports.

    Length disagreement counts against the score: the comparison is over the
    longer of the two, so a candidate half the length of the continuation
    cannot score well by matching its first half perfectly.
    """
    written = generation.sequence
    if not candidate or not written:
        return 0.0

    probs = generation.sampled_probs
    span = max(len(candidate), len(written))
    overlap = min(len(candidate), len(written))

    total = 0.0
    weight_sum = 0.0
    for index in range(span):
        weight = probs[index] if index < len(probs) else 1.0
        weight_sum += weight
        if index < overlap and candidate[index] == written[index]:
            total += weight

    if weight_sum <= 0:
        return 0.0
    return round(total / weight_sum, 6)
