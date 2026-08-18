"""A homologous sequence from an external database, used as evidence."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Reference:
    """One candidate donor sequence retrieved from NCBI or found via BLAST.

    `relatedness` is how phylogenetically close the reference organism is to
    the target, on 0..1. It is separate from alignment identity on purpose: a
    close relative with a mediocre local alignment and a distant relative with
    a strong one are different kinds of evidence, and the ranker weighs them
    differently.
    """

    accession: str
    organism: str | None = None
    description: str | None = None
    residues: str | None = None

    # Populated when the reference came from a BLAST hit.
    identity: float | None = None
    coverage: float | None = None
    e_value: float | None = None
    bit_score: float | None = None

    relatedness: float | None = None
    source: str = "ncbi"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def has_sequence(self) -> bool:
        """Whether the residues were fetched, or only the hit metadata."""
        return bool(self.residues)

    def __len__(self) -> int:
        return len(self.residues) if self.residues else 0
