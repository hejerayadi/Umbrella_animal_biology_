"""Input/output shapes for the evolutionary-context tool."""
from __future__ import annotations

from pydantic import Field

from ..contracts import ToolInput, ToolOutput


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
