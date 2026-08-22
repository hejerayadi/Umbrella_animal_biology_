"""Why a gap failed, in terms the planner can act on.

A critique that says only "not good enough" leaves the planner with nowhere to
go but the same tool call again, which produces the same evidence and the same
objection. That was the shape of the loop before this module existed: REVISE
was structurally indistinguishable from "retry", so a second iteration cost two
LLM calls and changed nothing.

A diagnosis names *which* link in the evidence chain broke. The planner reads it
and picks a different strategy - widen the search, fetch different references,
realign, or stop and ask another agent - rather than repeating itself.

Deliberately derived from the state, not from the critic's prose: the reason has
to be available when no LLM is configured, and it must mean the same thing every
time so the planner can branch on it.
"""
from __future__ import annotations

from enum import Enum

from agent.state.state import ReconstructionState, has_usable_alignment
from domain.models import Reference

#: Below this, an alignment's references are too divergent to read a fill out of
#: with any confidence. Matches the reference ranker's usability floor.
_WEAK_IDENTITY = 0.7

#: A consensus this evenly split is not a majority view, it is a coin flip - the
#: single largest source of confidently-wrong reconstructions in the sweep.
_AMBIGUOUS_SUPPORT = 0.25

#: Below this, every reference is a distant relative and the fill is a guess
#: about a lineage nothing in the evidence represents.
_DISTANT_RELATIVES = 0.5


class RevisionReason(str, Enum):
    """What to fix, not merely that something is wrong."""

    #: The homology search found nothing at all.
    NO_BLAST_HITS = "no_blast_hits"
    #: Hits exist but none can be aligned - no residues, or too divergent.
    BAD_REFERENCES = "bad_references"
    #: References aligned, but the alignment does not locate the gap's columns.
    WEAK_ALIGNMENT = "weak_alignment"
    #: The references disagree so evenly that no base wins on merit.
    AMBIGUOUS_CONSENSUS = "ambiguous_consensus"
    #: Everything available is too distant to be informative about this lineage.
    INSUFFICIENT_PHYLOGENETIC_SUPPORT = "insufficient_phylogenetic_support"


def diagnose(state: ReconstructionState, gap_id: str) -> RevisionReason | None:
    """The first broken link in one gap's evidence chain.

    Ordered by position in the chain rather than by severity: fixing a later
    link is pointless while an earlier one is still broken, so the earliest
    failure is the actionable one. Returns None when nothing is identifiably
    wrong - the evidence is present and coherent, and the gap simply scored
    below the reporting threshold.
    """
    references: list[Reference] = list((state.get("references") or {}).get(gap_id) or [])

    if not references:
        return RevisionReason.NO_BLAST_HITS

    alignable = [reference for reference in references if reference.has_sequence]
    if not alignable:
        # Hits were found and then could not be used - historically because
        # BLAST returned metadata without residues.
        return RevisionReason.BAD_REFERENCES

    alignment = (state.get("alignments") or {}).get(gap_id)
    if not has_usable_alignment(alignment):
        return RevisionReason.WEAK_ALIGNMENT

    identities = [r.identity for r in alignable if r.identity is not None]
    if identities and max(identities) < _WEAK_IDENTITY:
        return RevisionReason.BAD_REFERENCES

    candidates = (state.get("candidates") or {}).get(gap_id) or []
    support = max((getattr(c, "support", 0.0) or 0.0 for c in candidates), default=None)
    if support is not None and support < _AMBIGUOUS_SUPPORT:
        return RevisionReason.AMBIGUOUS_CONSENSUS

    scored = [r.relatedness for r in alignable if r.relatedness is not None]
    if scored and max(scored) < _DISTANT_RELATIVES:
        return RevisionReason.INSUFFICIENT_PHYLOGENETIC_SUPPORT

    return None


def diagnose_all(state: ReconstructionState, gap_ids: set[str]) -> dict[str, str]:
    """Diagnoses for every gap still in play, as plain strings for the state.

    Stored as values rather than enum members because state is checkpointed to
    Postgres and has to survive a JSON round trip.
    """
    reasons: dict[str, str] = {}
    for gap_id in gap_ids:
        reason = diagnose(state, gap_id)
        if reason is not None:
            reasons[gap_id] = reason.value
    return reasons
