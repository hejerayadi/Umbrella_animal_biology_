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

    #: Which strand of the subject the homology was found on. A minus-strand
    #: hit has already been reverse-complemented onto the target's strand by
    #: the time it reaches here; this records that it happened.
    strand: int = 1

    # --- Provenance -------------------------------------------------------
    # Evidence accumulates across iterations and slices, so a reference has to
    # say where it came from: two rounds can return the same accession from
    # different tools with different quality, and merging them without knowing
    # which is which silently loses the better one.
    iteration: int | None = None
    attempt: int | None = None

    @property
    def has_sequence(self) -> bool:
        """Whether the residues were fetched, or only the hit metadata."""
        return bool(self.residues)

    @property
    def quality(self) -> float:
        """How good this reference is as evidence, on 0..1.

        A single number so references from different tools can be compared when
        deduplicating: BLAST hits carry alignment statistics, NCBI records carry
        none, and the one that actually measured its homology should win.
        Deliberately not the ranker's score, which weighs relatedness for a
        different purpose.
        """
        signals = [value for value in (self.identity, self.coverage) if value is not None]
        if not signals:
            # No measured homology at all - usable, but not preferable.
            return 0.0
        return sum(signals) / len(signals)

    def __len__(self) -> int:
        return len(self.residues) if self.residues else 0
