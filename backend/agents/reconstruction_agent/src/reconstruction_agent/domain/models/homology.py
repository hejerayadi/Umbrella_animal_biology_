"""Homologous sequences found for a gap, normalised away from any one provider.

A `HomologHit` describes a biological relationship - this other sequence
resembles our query, here, this well. Every field is a homology concept rather
than a BLAST concept, so a different search backend maps onto the same model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HomologHit(BaseModel):
    """One homologous sequence, as evidence for reconstructing one gap."""

    model_config = ConfigDict(frozen=True)

    accession: str
    description: str = ""

    #: The organism this hit belongs to.
    #:
    #: EMBL-EBI reports a structured organism field of the literal string "NA"
    #: on every real hit, so this is parsed out of the free-text description
    #: instead. Getting it wrong is expensive and silent: the evolutionary
    #: weight multiplies by zero and every hit scores as if unrelated.
    organism: str | None = None
    #: Filled in by resolving `organism` against NCBI Taxonomy after the search.
    #: This, not the name string, is what evolutionary distance is computed from.
    organism_tax_id: int | None = None

    #: Fraction of aligned positions that match, 0..1.
    identity: float = Field(ge=0.0, le=1.0)
    #: Fraction of the query covered by the alignment, 0..1.
    query_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    e_value: float = Field(default=0.0, ge=0.0)
    bit_score: float = 0.0

    #: Alignment span on the query, 0-based half-open.
    query_start: int = 0
    query_end: int = 0
    #: Alignment span on the subject, 0-based half-open.
    subject_start: int = 0
    subject_end: int = 0

    #: The retrieved subject sequence, when it has been fetched.
    subject_sequence: str | None = None

    #: Which collection this came from. Provenance only - nothing branches on
    #: it, and no code in the agent is written against a particular value.
    source_database: str | None = None
    source_provider: str = "EMBL-EBI"

    @property
    def alignment_length(self) -> int:
        return max(0, self.query_end - self.query_start)

    @property
    def has_sequence(self) -> bool:
        return bool(self.subject_sequence)

    @property
    def taxonomy_resolved(self) -> bool:
        return self.organism_tax_id is not None

    def spans(self, junction: int) -> bool:
        """Whether this hit aligns across the point where the flanks meet.

        The query submitted for a gap is the two flanks joined, so the missing
        region sits exactly at `junction` and a usable hit must align on both
        sides of it. A hit matching one flank beautifully and stopping there
        proves only that the flank is conserved - it carries no information
        about the bases in between, and counting it as support is the mistake
        that produces confident nonsense.
        """
        return self.query_start < junction < self.query_end


class HomologySearchOutcome(BaseModel):
    """What one search against one collection actually produced.

    This is the measurement that drives database selection. `gap_spanning_hits`
    is the only number that matters scientifically: a search can return fifty
    excellent hits and still be useless if none of them covers both flanks, and
    ranking on raw hit count hides exactly that failure.
    """

    model_config = ConfigDict(frozen=True)

    #: Opaque provider identifier for the collection searched. Recorded so a
    #: later round can avoid repeating it; never interpreted.
    database_code: str
    database_label: str = ""
    hits: tuple[HomologHit, ...] = ()
    #: Hits whose alignment covers the gap on both sides.
    gap_spanning_hits: int = 0
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def total_hits(self) -> int:
        return len(self.hits)

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def is_usable(self) -> bool:
        """Whether this search produced evidence a reconstruction can rest on."""
        return self.succeeded and self.gap_spanning_hits > 0
