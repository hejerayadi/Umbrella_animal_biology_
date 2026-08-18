"""Evolutionary context for reference ranking.

Which relatives are informative is a phylogenetics question, and the Umbrella
system already has an agent for it. This tool provides the local answer and
tells the caller when the real one is worth asking for.
"""
from __future__ import annotations

from ...configuration.logging import get_logger
from ..contracts import Tool
from .mapper import heuristic_relatedness
from .schemas import EvolutionaryContextInput, EvolutionaryContextOutput

_log = get_logger(__name__)

# Below this, the local heuristic is admitting it cannot place the organism
# rather than asserting distance - the point at which delegating pays off.
_DELEGATION_THRESHOLD = 0.5


class EvolutionaryContextTool(Tool[EvolutionaryContextInput, EvolutionaryContextOutput]):
    """Scores how closely candidate organisms relate to the target.

    Runs a name-based heuristic locally. When that heuristic cannot separate
    the candidates, it says so in `notes`, and the graph may return
    NEEDS_AGENT to have the orchestrator route the question to the Evolution
    Agent - which is what `card.json` declares this agent may need.
    """

    name = "evolutionary_context"
    description = (
        "Score how closely candidate reference organisms relate to the target organism, so "
        "that references can be ranked by phylogenetic proximity as well as by alignment "
        "quality. Fast and local, but genus-level only; when it cannot separate candidates "
        "the Evolution Agent should be consulted instead."
    )
    estimated_seconds = 0.1

    async def run(self, payload: EvolutionaryContextInput) -> EvolutionaryContextOutput:
        if not payload.target_organism:
            return EvolutionaryContextOutput(
                succeeded=True,
                notes=["No target organism given; relatedness could not be estimated."],
            )

        relatedness = {
            organism: heuristic_relatedness(payload.target_organism, organism)
            for organism in payload.candidate_organisms
        }

        notes: list[str] = []
        if relatedness and max(relatedness.values()) < _DELEGATION_THRESHOLD:
            notes.append(
                "The name-based heuristic could not place any candidate near "
                f"{payload.target_organism}. A phylogeny from the Evolution Agent would "
                "rank these references far better."
            )

        return EvolutionaryContextOutput(
            succeeded=True,
            relatedness=relatedness,
            source="heuristic",
            notes=notes,
            diagnostics={"delegation_recommended": bool(notes)},
        )
