"""Domain result to transport payload.

One direction only. Domain models describe biology and never learn what a
caller wants to see; transport payloads never become domain models. Keeping the
mapping in one place is what lets `/api/v2` present the same reconstruction
differently without touching a single service.

The top-level keys produced for the orchestrator are a cross-agent contract:
they are declared in `card.json`, other agents test for them in the shared
context, and renaming one silently breaks whoever reads it.
"""

from __future__ import annotations

from typing import Any

from reconstruction_agent.domain.models.candidate import Candidate
from reconstruction_agent.domain.models.result import GapReconstruction, ReconstructionResult


def to_agent_output(result: ReconstructionResult) -> dict[str, Any]:
    """The `output` dict merged into the shared orchestrator context.

    The highest-confidence resolved fill is surfaced at the top level rather
    than left buried in the per-gap detail, because it is the single most
    useful field for a downstream agent.

    It is called `reconstruction_best_fill`, and carries its own coordinates,
    because the previous name - `reconstruction_sequence`, a bare string - read
    as "the reconstructed sequence" and was nothing of the kind. Measured on
    the polar bear scaffold NW_024426341.1 it held `"ACCTTAC"`: seven bases,
    the one accepted fill, offered under a name that invited a consumer to
    treat it as the repaired 1,153,480 bp record. The bases are the same; what
    changed is that the payload now says what they are and where they go, so
    the mistake is no longer available to make.
    """
    resolved = [item for item in result.reconstructions if item.is_resolved]
    best = max(resolved, key=lambda item: item.confidence or 0.0, default=None)

    return {
        "reconstruction": to_data(result),
        "reconstruction_summary": _summary(result),
        "reconstruction_best_fill": _best_fill(best),
    }


def _best_fill(item: GapReconstruction | None) -> dict[str, Any] | None:
    """The winning fill, with the gap it closes and how much of it that is.

    `covers_bp` against `sequence_length_bp` is the part a caller most needs
    and cannot derive: it is the difference between "the record is repaired"
    and "seven of its bases are".
    """
    if item is None or item.sequence is None:
        return None
    return {
        "sequence": item.sequence,
        "length_bp": len(item.sequence),
        "gap_id": item.gap_id,
        "start": item.gap.start,
        "end": item.gap.end,
        "confidence": round(item.confidence, 4) if item.confidence is not None else None,
        "is_model_generated": (
            item.selected_candidate.is_model_generated if item.selected_candidate else None
        ),
    }


def to_data(result: ReconstructionResult) -> dict[str, Any]:
    """The business payload, for the `data` field of a v1 response."""
    profile = result.target_profile
    return {
        "request_id": result.request_id,
        "status": result.status.value,
        "scientific_name": profile.scientific_name if profile else None,
        "tax_id": profile.tax_id if profile else None,
        "molecule_type": profile.molecule_type.value if profile else None,
        "assembly_id": result.assembly_id,
        "sequence_accession": result.sequence_accession,
        "summary": {
            "requested_gaps": result.requested_gaps,
            "resolved_gaps": result.resolved_gaps,
            "unresolved_gaps": result.unresolved_gaps,
        },
        "reconstructions": [_gap(item) for item in result.reconstructions],
        "warnings": list(result.warnings),
    }


def _gap(item: GapReconstruction) -> dict[str, Any]:
    return {
        "gap_id": item.gap.gap_id,
        "start": item.gap.start,
        "end": item.gap.end,
        "length": item.gap.length,
        "status": item.status.value,
        "selected_candidate": _candidate(item.selected_candidate),
        "alternatives": [_candidate(alt) for alt in item.alternatives],
        "unresolved_reason": (item.unresolved_reason.value if item.unresolved_reason else None),
        "explanation": item.explanation,
        "evidence": {
            "homology": item.evidence.homology.model_dump(mode="json"),
            "alignment": item.evidence.alignment.model_dump(mode="json"),
            "evo2": item.evidence.evo2.model_dump(mode="json"),
            "validation": {
                "checks": list(item.evidence.validation_checks),
                "failures": list(item.evidence.validation_failures),
            },
        },
        "provenance": item.provenance.model_dump(mode="json"),
        "warnings": list(item.warnings),
    }


def _candidate(candidate: Candidate | None) -> dict[str, Any] | None:
    """One candidate, with the evidence behind its score.

    The component scores travel with it rather than only the total, so a
    reviewer can see which signal carried the confidence instead of having to
    take the number on trust.
    """
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id,
        # The single most consequential field here. A fill assembled from
        # sequenced relatives and one written by a genome model are both
        # plausible strings of the same length, and nothing else in this
        # payload separates them - `supporting_hits` being empty is a hint a
        # reader has to interpret, not a statement. Omitting this was a real
        # defect: a model prediction was reported with no marking at all.
        "origin": candidate.origin.value,
        "is_model_generated": candidate.is_model_generated,
        "sequence": candidate.sequence,
        "length": candidate.length,
        "confidence": round(candidate.final_confidence, 4),
        "confidence_level": candidate.confidence_level.value,
        "supporting_hits": list(candidate.supporting_hits),
        "supporting_organisms": list(candidate.supporting_organisms),
        "scores": candidate.scores.model_dump(mode="json"),
        "rationale": candidate.rationale,
    }


def _summary(result: ReconstructionResult) -> str:
    """A sentence the orchestrator responder can hand to the user.

    Written from the counts, so it can never claim more than was achieved.
    """
    if not result.reconstructions:
        return "No unresolved regions were found in the requested sequence."

    line = result.summary_line()
    if result.unresolved_gaps == 0:
        return line

    reasons = sorted(
        {
            item.unresolved_reason.value
            for item in result.reconstructions
            if not item.is_resolved and item.unresolved_reason
        }
    )
    return f"{line} Unresolved regions were left in place ({', '.join(reasons)})."
