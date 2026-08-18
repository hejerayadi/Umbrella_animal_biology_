"""Turns a plan step into a concrete, validated tool invocation.

The planner names a tool and a gap; this builds the typed payload that tool
needs from the current state. Keeping that translation here means the planner
never has to know a tool's input schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...configuration.logging import get_logger
from ...domain.models import GapContext, Reference
from ...domain.services import ReferenceRanker
from ...tools.blast.schemas import BlastSearchInput
from ...tools.evo.schemas import EvolutionaryContextInput
from ...tools.mafft.schemas import AlignmentInput
from ...tools.ncbi.schemas import NCBISearchInput
from ..planning.planner import PlanStep
from ..state.state import ReconstructionState

_log = get_logger(__name__)

# How many references to carry into an alignment. MAFFT slows sharply with row
# count, and past roughly this many the consensus stops changing.
_MAX_ALIGNMENT_REFERENCES = 8


@dataclass(frozen=True, slots=True)
class ToolInvocation:
    """A tool name paired with the payload to run it on."""

    tool: str
    payload: Any
    gap_id: str | None = None


class ToolSelector:
    """Builds tool payloads from plan steps and state."""

    def __init__(self, ranker: ReferenceRanker | None = None) -> None:
        self._ranker = ranker or ReferenceRanker()

    def build(self, step: PlanStep, state: ReconstructionState) -> ToolInvocation | None:
        """The invocation for one step, or None when it cannot be run.

        Returning None rather than raising: a step whose preconditions are no
        longer met (its gap got skipped, its references never arrived) should
        be quietly dropped so the rest of the plan still executes.
        """
        context = self._context_for(step.gap_id, state)

        if step.tool == "blast_search":
            return self._blast(step, context)
        if step.tool == "mafft_align":
            return self._mafft(step, context, state)
        if step.tool == "ncbi_search":
            return self._ncbi(step, context, state)
        if step.tool == "evolutionary_context":
            return self._evo(step, state)

        _log.warning("No payload builder for tool %r; dropping the step.", step.tool)
        return None

    # --- per-tool builders --------------------------------------------------

    def _blast(self, step: PlanStep, context: GapContext | None) -> ToolInvocation | None:
        if context is None or not context.has_usable_flanks:
            return None
        return ToolInvocation(
            tool="blast_search",
            gap_id=context.identifier,
            payload=BlastSearchInput(
                sequence=context.query_sequence(),
                gap_id=context.identifier,
                **_allowed(step.arguments, {"database", "program", "max_hits", "expect"}),
            ),
        )

    def _mafft(
        self, step: PlanStep, context: GapContext | None, state: ReconstructionState
    ) -> ToolInvocation | None:
        if context is None:
            return None

        available = (state.get("references") or {}).get(context.identifier, [])
        usable = [
            reference
            for reference in self._ranker.rank(
                self._ranker.filter_usable(available), limit=_MAX_ALIGNMENT_REFERENCES
            )
            # A reference with no residues cannot be aligned, however well it
            # scored on the BLAST metadata alone.
            if reference.has_sequence
        ]
        if not usable:
            return None

        return ToolInvocation(
            tool="mafft_align",
            gap_id=context.identifier,
            payload=AlignmentInput(
                gap_id=context.identifier,
                target_sequence=context.query_sequence(),
                left_flank_length=len(context.left_flank),
                references={
                    reference.accession: reference.residues or "" for reference in usable
                },
            ),
        )

    def _ncbi(
        self, step: PlanStep, context: GapContext | None, state: ReconstructionState
    ) -> ToolInvocation:
        organisms = step.arguments.get("organisms") or state.get("requested_organisms") or []
        return ToolInvocation(
            tool="ncbi_search",
            gap_id=context.identifier if context else None,
            payload=NCBISearchInput(
                term=str(step.arguments.get("term") or state.get("organism") or ""),
                organisms=[str(organism) for organism in organisms],
                **_allowed(step.arguments, {"database", "limit", "fetch_sequences"}),
            ),
        )

    def _evo(self, step: PlanStep, state: ReconstructionState) -> ToolInvocation:
        organisms = {
            reference.organism
            for references in (state.get("references") or {}).values()
            for reference in references
            if reference.organism
        }
        return ToolInvocation(
            tool="evolutionary_context",
            payload=EvolutionaryContextInput(
                target_organism=state.get("organism") or "",
                candidate_organisms=sorted(organisms),
            ),
        )

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _context_for(gap_id: str | None, state: ReconstructionState) -> GapContext | None:
        if not gap_id:
            return None
        for context in state.get("gap_contexts") or []:
            if context.identifier == gap_id:
                return context
        return None

    @staticmethod
    def apply_relatedness(
        references: list[Reference], relatedness: dict[str, float]
    ) -> list[Reference]:
        """Attach relatedness scores so the ranker can weigh them.

        Returns new objects because `Reference` is frozen - the originals stay
        valid for anything already holding them.
        """
        return [
            Reference(
                accession=reference.accession,
                organism=reference.organism,
                description=reference.description,
                residues=reference.residues,
                identity=reference.identity,
                coverage=reference.coverage,
                e_value=reference.e_value,
                bit_score=reference.bit_score,
                relatedness=relatedness.get(reference.organism or "", reference.relatedness),
                source=reference.source,
                metadata=dict(reference.metadata),
            )
            for reference in references
        ]


def _allowed(arguments: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    """Keep only the argument keys a given tool schema accepts.

    The planner is a language model and will occasionally invent parameters;
    filtering here turns that into a no-op instead of a validation error.
    """
    return {key: value for key, value in arguments.items() if key in keys}
