"""Input/output shapes for the evolutionary-context and Evo 2 tools."""
from __future__ import annotations

from pydantic import Field

from tools.contracts import ToolInput, ToolOutput


class EvolutionaryContextInput(ToolInput):
    """Ask how closely a set of organisms relates to the target organism."""

    target_organism: str = Field(description="Scientific name of the sequence being repaired.")
    candidate_organisms: list[str] = Field(
        default_factory=list, description="Organisms to score for relatedness to the target."
    )


class EvolutionaryContextOutput(ToolOutput):
    # organism -> 0..1, where 1.0 is the same species.
    relatedness: dict[str, float] = Field(default_factory=dict)
    # Populated when the Evolution Agent answered rather than the local
    # fallback; lets the caller tell a real phylogeny from a heuristic.
    source: str = "heuristic"
    notes: list[str] = Field(default_factory=list)


class PlausibilityInput(ToolInput):
    """Check candidate reconstructions against what Evo 2 expects.

    The left flank is the prompt: Evo 2 is asked what follows it, and each
    candidate is compared against that prediction.
    """

    gap_id: str | None = None
    left_flank: str = Field(description="Known sequence immediately before the gap.")
    candidates: dict[str, str] = Field(
        default_factory=dict, description="candidate id -> proposed gap sequence."
    )
    # Evo 2 attends over long contexts, but a very long prompt costs latency
    # for little gain here; the bases nearest the gap carry the signal.
    max_prompt_bases: int = Field(default=1024, ge=32)


class CandidateAgreement(ToolOutput):
    """How one candidate compares to Evo 2's expectation."""

    candidate_id: str
    #: Fraction of positions where the candidate matches Evo 2's prediction.
    agreement: float = Field(ge=0.0, le=1.0)
    #: The same, weighted by how confident Evo 2 was at each position. A
    #: mismatch where the model was unsure counts for less than one where it
    #: was certain.
    weighted_agreement: float = Field(ge=0.0, le=1.0)


class PlausibilityOutput(ToolOutput):
    # candidate id -> weighted agreement, 0..1. Comparable between candidates
    # for the same gap; meaningless across gaps, and never an absolute
    # accept/reject threshold.
    scores: dict[str, float] = Field(default_factory=dict)
    agreements: list[CandidateAgreement] = Field(default_factory=list)
    best_candidate: str | None = None
    #: What Evo 2 itself predicted, for the run log. Never adopted as sequence.
    predicted_sequence: str | None = None
    model_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
