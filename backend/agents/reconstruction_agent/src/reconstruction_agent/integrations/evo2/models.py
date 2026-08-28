"""The wire shapes for Evo 2 on NVIDIA NIM.

The request parameters here are not defaults anyone chose for elegance; each
one is a constraint the endpoint imposes, measured against it:

- `temperature` must be strictly positive. The API rejects `0`, so near-greedy
  decoding is expressed as a very small value together with `top_k=1`.
- The model path is `arc/evo2-40b`. The `-forward` variant that would return
  logits is a 404, and `/forward` with `output_layers: ["logits"]` fails with
  "StripedHyena has no attribute logits" - it returns embeddings only.

The consequence shapes the whole design: **there is no endpoint that scores an
existing sequence.** Evo 2 cannot be asked "how likely is this candidate". It
can only be asked to continue the left flank itself, and the candidate compared
against what it produced. That is an agreement check, not a likelihood, and
everything downstream is named accordingly.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Evo2GenerationRequest(BaseModel):
    """One continuation request."""

    model_config = ConfigDict(frozen=True)

    #: The left flank. Evo 2 continues from here.
    sequence: str
    #: How many bases to generate - the length of the region being arbitrated.
    num_tokens: int = Field(ge=1)
    #: Strictly positive: the API rejects `temperature <= 0`. Combined with
    #: `top_k=1` this is as close to greedy as the endpoint permits, which is
    #: what makes the comparison reproducible.
    temperature: float = Field(default=0.01, gt=0.0)
    top_k: int = Field(default=1, ge=1)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    #: Asked for explicitly: without them the continuation carries no measure
    #: of how sure the model was, and an agreement score computed from bases
    #: alone would weight a confident base the same as a coin flip.
    enable_sampled_probs: bool = True

    def to_payload(self) -> dict[str, object]:
        """The JSON NIM expects."""
        return {
            "sequence": self.sequence,
            "num_tokens": self.num_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "enable_sampled_probs": self.enable_sampled_probs,
        }


class Evo2Generation(BaseModel):
    """What Evo 2 produced, parsed."""

    model_config = ConfigDict(frozen=True)

    #: The generated continuation, uppercased and stripped of anything that is
    #: not a base. NIM occasionally returns whitespace or a trailing newline.
    sequence: str = ""
    #: Per-base sampling probability, when the endpoint returned them. Empty is
    #: a valid answer and means the agreement is computed unweighted rather
    #: than not at all.
    sampled_probs: tuple[float, ...] = ()

    @property
    def mean_confidence(self) -> float | None:
        """The model's average certainty across the continuation."""
        if not self.sampled_probs:
            return None
        return sum(self.sampled_probs) / len(self.sampled_probs)
