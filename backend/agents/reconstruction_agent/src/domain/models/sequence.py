"""The target sequence being repaired."""
from __future__ import annotations

from dataclasses import dataclass, field

from domain.exceptions import InvalidSequenceError

# IUPAC nucleotide alphabet. `N` is "any base" and is what an assembly gap is
# written as; the other ambiguity codes are rarer but legal input.
IUPAC_NUCLEOTIDES = frozenset("ACGTUNRYSWKMBDHV")
UNKNOWN_BASE = "N"

#: Watson-Crick complements across the whole IUPAC alphabet, so a minus-strand
#: hit can be brought onto the target's strand without losing ambiguity codes.
_COMPLEMENTS = str.maketrans(
    "ACGTUNRYSWKMBDHVacgtunryswkmbdhv",
    "TGCAANYRSWMKVHDBtgcaanyrswmkvhdb",
)


def reverse_complement(residues: str) -> str:
    """The reverse complement of a nucleotide string.

    BLAST reports hits on either strand. A minus-strand hit describes the same
    homology but written backwards relative to the target, so aligning it as-is
    produces noise rather than evidence - it has to be brought onto the
    target's strand first.
    """
    return residues.translate(_COMPLEMENTS)[::-1]


@dataclass(frozen=True, slots=True)
class Sequence:
    """A nucleotide sequence with its provenance.

    Immutable: reconstruction produces new `Sequence` objects rather than
    mutating the input, so the original stays available for comparison and
    for the audit trail.
    """

    identifier: str
    residues: str
    organism: str | None = None
    description: str | None = None
    accession: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.residues:
            raise InvalidSequenceError(f"Sequence '{self.identifier}' is empty.")

    @classmethod
    def parse(
        cls,
        identifier: str,
        residues: str,
        *,
        organism: str | None = None,
        description: str | None = None,
        accession: str | None = None,
    ) -> Sequence:
        """Normalise raw residue text into a `Sequence`.

        Strips whitespace and uppercases, because FASTA arrives wrapped at 60 or
        80 columns and casing carries no meaning here (lowercase conventionally
        marks repeat-masked regions, which we do not treat differently).
        """
        cleaned = "".join(residues.split()).upper()
        if not cleaned:
            raise InvalidSequenceError(f"Sequence '{identifier}' has no residues.")

        illegal = set(cleaned) - IUPAC_NUCLEOTIDES
        if illegal:
            raise InvalidSequenceError(
                f"Sequence '{identifier}' contains non-nucleotide characters: "
                f"{''.join(sorted(illegal))}"
            )
        return cls(
            identifier=identifier,
            residues=cleaned,
            organism=organism,
            description=description,
            accession=accession,
        )

    def __len__(self) -> int:
        return len(self.residues)

    @property
    def unknown_count(self) -> int:
        return self.residues.count(UNKNOWN_BASE)

    @property
    def completeness(self) -> float:
        """Fraction of positions that are a known base. 1.0 means no gaps."""
        return 1.0 - (self.unknown_count / len(self.residues))

    def slice(self, start: int, end: int) -> str:
        """Residues in [start, end), clamped to the sequence bounds.

        Clamping rather than raising keeps flank extraction simple at the ends
        of a sequence, where a gap may have less context on one side.
        """
        return self.residues[max(0, start) : min(len(self.residues), end)]

    def with_residues(self, residues: str) -> Sequence:
        """Copy carrying new residues and the same provenance."""
        return Sequence(
            identifier=self.identifier,
            residues=residues,
            organism=self.organism,
            description=self.description,
            accession=self.accession,
            metadata=dict(self.metadata),
        )
