"""Writing an accepted fill back into the record.

The only place in the agent where a sequence is modified. Everything upstream
gathers evidence and scores it; this substitutes bases, and it is deliberately
small, pure and separate so that the one operation that can corrupt a genome is
readable in full on one screen.

Three refusals, all of them silent-corruption risks rather than style:

**The region must be the length it claims.** A fill shorter or longer than the
gap shifts every downstream coordinate in the record, and nothing later would
detect it.

**The region being replaced must actually be unresolved.** Overwriting called
bases with a reconstruction is data loss, not repair.

**The fill must be unambiguous.** Substituting one N-run for another achieves
nothing while making the record claim it was repaired.
"""

from __future__ import annotations

from dataclasses import dataclass

from reconstruction_agent.domain.models.sequence import DNA_ALPHABET, Gap

#: Characters an assembler emits for an unresolved position.
AMBIGUITY = frozenset("NnXx")


@dataclass(frozen=True, slots=True)
class Replacement:
    """The outcome of substituting one fill into one record."""

    #: The repaired residues. Empty when the substitution was refused.
    residues: str = ""
    applied: bool = False
    reason: str = ""


def apply_fill(residues: str, gap: Gap, fill: str) -> Replacement:
    """Substitute `fill` for the bases `gap` covers in `residues`."""
    if gap.end > len(residues):
        return Replacement(
            reason=(
                f"Gap {gap.gap_id} ends at {gap.end} but the record is only "
                f"{len(residues)} bases long."
            )
        )

    if len(fill) != gap.length:
        return Replacement(
            reason=(
                f"The fill is {len(fill)} bases and gap {gap.gap_id} is {gap.length}; "
                "substituting it would shift every coordinate after this region."
            )
        )

    if not fill or not set(fill.upper()) <= DNA_ALPHABET:
        return Replacement(
            reason=(
                f"The fill for gap {gap.gap_id} is not unambiguous sequence, so applying "
                "it would mark the region repaired without resolving it."
            )
        )

    replaced = residues[gap.start : gap.end]
    resolved = [base for base in replaced if base not in AMBIGUITY]
    if resolved:
        return Replacement(
            reason=(
                f"The region {gap.start}-{gap.end} contains {len(resolved)} called "
                "bases; overwriting them would be data loss rather than repair."
            )
        )

    return Replacement(
        residues=residues[: gap.start] + fill.upper() + residues[gap.end :],
        applied=True,
    )
