"""The target sequence, its metadata, and the holes to be filled.

Coordinates are the perennial source of off-by-one bugs here, so one convention
is fixed for the whole agent and stated at every boundary that touches it:

    `start` is 0-based and inclusive, `end` is 0-based and EXCLUSIVE.

That is Python slice semantics, so `residues[gap.start:gap.end]` is the gap and
`gap.end - gap.start` is its length, with no adjustment anywhere. Requests
arriving in 1-based inclusive form (which is what NCBI displays) are converted
once, at the API boundary, and never again.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reconstruction_agent.domain.enums import MoleculeType

#: The unambiguous DNA alphabet. `N` is deliberately excluded: it is the marker
#: of the thing being reconstructed, so a candidate containing one has not
#: answered the question.
DNA_ALPHABET = frozenset("ACGT")


class Gap(BaseModel):
    """One unresolved region of the target sequence."""

    model_config = ConfigDict(frozen=True)

    #: Stable within a run; used as the key for every piece of evidence.
    gap_id: str
    #: 0-based, inclusive.
    start: int = Field(ge=0)
    #: 0-based, exclusive.
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Gap:
        if self.end <= self.start:
            raise ValueError(
                f"gap {self.gap_id}: end ({self.end}) must exceed start ({self.start})"
            )
        return self

    @property
    def length(self) -> int:
        return self.end - self.start


class GapContext(BaseModel):
    """A gap together with the sequence flanking it.

    The flanks are the entire basis for reconstruction: they are what gets
    searched, what anchors the alignment, and what a candidate must sit between.
    A gap whose flanks are too short to align is not reconstructible, however
    many homologues exist.
    """

    model_config = ConfigDict(frozen=True)

    gap: Gap
    #: The record this gap is in. Carried so a search can exclude it: a record
    #: cannot be evidence about its own unresolved region.
    source_accession: str | None = None
    #: Sequence immediately before the gap, 5' -> 3', ending at `gap.start`.
    left_flank: str = ""
    #: Sequence immediately after the gap, 5' -> 3', starting at `gap.end`.
    right_flank: str = ""

    @property
    def gap_id(self) -> str:
        return self.gap.gap_id

    @property
    def length(self) -> int:
        return self.gap.length

    @property
    def has_usable_flanks(self) -> bool:
        """Whether both flanks carry enough signal to anchor an alignment.

        Both are required: a single flank can place one edge of the fill but
        cannot bound it, so the reconstruction would have no defined end.
        """
        return bool(self.left_flank) and bool(self.right_flank)

    def query_sequence(self) -> str:
        """The flanks joined across the gap, as submitted to a homology search.

        The missing bases are omitted rather than sent as a run of `N`: they
        carry no information, and a long ambiguous stretch degrades the search.
        What is wanted is a reference that matches both flanks, because that is
        what can span the hole between them.
        """
        return f"{self.left_flank}{self.right_flank}"


class AssemblyMetadata(BaseModel):
    """What is known about the assembly the target belongs to."""

    model_config = ConfigDict(frozen=True)

    assembly_id: str | None = None
    assembly_level: str | None = None
    organism: str | None = None
    tax_id: int | None = None


class SequenceRecord(BaseModel):
    """A retrieved nucleotide sequence and the identity that came with it."""

    model_config = ConfigDict(frozen=True)

    accession: str
    residues: str
    description: str = ""
    organism: str | None = None
    tax_id: int | None = None
    molecule_type: MoleculeType = MoleculeType.UNKNOWN

    @property
    def length(self) -> int:
        return len(self.residues)

    def slice(self, start: int, end: int) -> str:
        """Bases in `[start, end)`, clamped to the record."""
        return self.residues[max(0, start) : min(self.length, end)]

    def left_flank_of(self, gap: Gap, size: int) -> str:
        """Up to `size` bases immediately before `gap`."""
        return self.slice(gap.start - size, gap.start)

    def right_flank_of(self, gap: Gap, size: int) -> str:
        """Up to `size` bases immediately after `gap`."""
        return self.slice(gap.end, gap.end + size)
