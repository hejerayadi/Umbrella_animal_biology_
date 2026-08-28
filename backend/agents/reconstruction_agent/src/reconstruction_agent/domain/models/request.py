"""What was asked for, validated before any expensive work begins.

The orchestrator sends free-form `{instruction, context}`, and the context is a
shared dictionary every agent in the run has contributed to. Reading it
defensively here - and rejecting what cannot be worked with up front - is what
keeps a malformed request from being discovered three external services and two
hundred seconds into a run.

This agent needs a specific sequence. It does not pick a target on its own, and
`card.json` says so, because the alternative is reconstructing something nobody
asked about.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from reconstruction_agent.domain.exceptions import (
    InvalidGapCoordinatesError,
    InvalidRequestError,
)
from reconstruction_agent.domain.models.sequence import DNA_ALPHABET, Gap

#: Context keys other agents use for the same thing. The Genome agent seeds
#: some of these, so reading all of them is what lets a hand-off work.
_ACCESSION_KEYS = (
    "sequence_accession",
    "accession",
    "nucleotide_accession",
    "refseq_accession",
)
_ORGANISM_KEYS = ("scientific_name", "species", "organism", "species_name")


#: Characters a caller-supplied flank may contain. `N` is allowed - a flank
#: can legitimately carry ambiguity - but anything else means the value is not
#: sequence at all, most often a placeholder like "ACTG...".
_FLANK_ALPHABET = DNA_ALPHABET | {"N"}


class RequestedGap(BaseModel):
    """One gap as the caller described it, after its coordinates are settled.

    The caller sends more than the agent strictly needs - a length, and the
    flanking sequence. Those are redundant with `start`/`end` and with the
    record itself, which makes them worth reading rather than discarding: two
    descriptions of the same region that disagree are how a coordinate error
    announces itself.
    """

    model_config = ConfigDict(frozen=True)

    gap: Gap
    #: Flanks as supplied, kept only when they are actually sequence.
    left_flank: str = ""
    right_flank: str = ""
    #: What was done with the caller's fields, carried into the response so a
    #: coordinate conversion is never invisible.
    notes: tuple[str, ...] = ()


class ReconstructionRequest(BaseModel):
    """A validated request to reconstruct unresolved regions."""

    model_config = ConfigDict(frozen=True)

    instruction: str = ""
    sequence_accession: str | None = None
    #: A sequence pasted directly, when the caller has one rather than an id.
    residues: str | None = None
    assembly_id: str | None = None
    scientific_name: str | None = None
    assembly_level: str | None = None
    #: Regions the caller specifically wants repaired. Empty means "find them".
    target_gaps: tuple[RequestedGap, ...] = ()

    @property
    def gaps(self) -> tuple[Gap, ...]:
        """Just the coordinates, for the parts of the pipeline that need only those."""
        return tuple(item.gap for item in self.target_gaps)

    @property
    def has_target(self) -> bool:
        return bool(self.sequence_accession or self.residues)

    @classmethod
    def from_agent_request(
        cls, instruction: str, context: dict[str, Any] | None
    ) -> ReconstructionRequest:
        """Parse the orchestrator contract into something workable.

        Raises rather than guessing when the target is missing. Guessing would
        mean reconstructing a sequence nobody named.
        """
        data = context or {}

        accession = _first_string(data, _ACCESSION_KEYS)
        residues = _sequence_from(data)

        if not accession and not residues:
            raise InvalidRequestError(
                "No sequence to reconstruct. Provide a nucleotide accession, or the "
                "sequence itself, in the request context.",
                details={"context_keys": sorted(data)},
            )

        return cls(
            instruction=instruction or "",
            sequence_accession=accession,
            residues=residues,
            assembly_id=_first_string(data, ("assembly_id", "assembly")),
            scientific_name=_first_string(data, _ORGANISM_KEYS),
            assembly_level=_first_string(data, ("assembly_level",)),
            target_gaps=_gaps_from(data.get("target_gaps")),
        )


def _first_string(data: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """The first key present with a usable string value."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _sequence_from(data: dict[str, Any]) -> str | None:
    """A pasted sequence, if the context carries one.

    Accepts both a bare string and the nested shape other agents emit, and
    strips whitespace an interactive user will inevitably paste along with it.
    """
    for key in ("sequence", "residues", "nucleotide_sequence"):
        value = data.get(key)
        if isinstance(value, dict):
            value = value.get("residues") or value.get("sequence")
        if isinstance(value, str) and value.strip():
            return "".join(value.split()).upper()
    return None


def _gaps_from(raw: Any) -> tuple[RequestedGap, ...]:
    """Caller-supplied gap coordinates, with their redundant fields checked.

    Rejected loudly when malformed. These coordinates decide which bases get
    replaced, so an off-by-one accepted quietly here corrupts the answer in a
    way nothing downstream can detect.
    """
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise InvalidGapCoordinatesError(
            "target_gaps must be a list of gap objects.",
            details={"received": type(raw).__name__},
        )

    gaps: list[RequestedGap] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise InvalidGapCoordinatesError(
                f"target_gaps[{index - 1}] is not an object.",
                details={"received": type(item).__name__},
            )
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidGapCoordinatesError(
                f"target_gaps[{index - 1}] needs integer start and end values.",
                details={"gap": item},
            ) from error

        try:
            gap = Gap(gap_id=str(item.get("gap_id") or f"gap_{index}"), start=start, end=end)
        except ValueError as error:
            raise InvalidGapCoordinatesError(
                f"target_gaps[{index - 1}]: {error}", details={"gap": item}
            ) from error

        gaps.append(
            RequestedGap(
                gap=gap,
                left_flank=_flank_from(item, ("left_flank", "flank_left", "upstream")),
                right_flank=_flank_from(item, ("right_flank", "flank_right", "downstream")),
                notes=_reconcile(gap, item),
            )
        )

    return tuple(gaps)


def _flank_from(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    """A caller-supplied flank, kept only when it is actually sequence.

    Callers routinely send a truncated display string like "ACGT..." in these
    fields. Silently treating that as sequence would anchor an alignment on
    characters that were never in the record, so anything outside the alphabet
    disqualifies the whole value rather than being stripped out of it.
    """
    for key in keys:
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        cleaned = "".join(value.split()).upper()
        if set(cleaned) <= _FLANK_ALPHABET:
            return cleaned
    return ""


def _reconcile(gap: Gap, item: dict[str, Any]) -> tuple[str, ...]:
    """Note where the caller's own description of a gap contradicts itself.

    `length` is redundant with `end - start`. When the two disagree the
    coordinates are authoritative - they are what gets replaced - but the
    disagreement is recorded, because it is the only signal that the caller and
    this agent may be counting from different origins.
    """
    raw_length = item.get("length")
    if raw_length is None:
        return ()
    try:
        stated = int(raw_length)
    except (TypeError, ValueError):
        return (f"{gap.gap_id}: ignored a non-numeric length ({raw_length!r}).",)
    if stated == gap.length:
        return ()
    return (
        f"{gap.gap_id}: the stated length ({stated}) disagrees with the supplied "
        f"coordinates ({gap.start}-{gap.end}, {gap.length} bases); the coordinates were used.",
    )
