"""Input/output shapes for the BLAST tool."""
from __future__ import annotations

from pydantic import Field

from contracts.output import EvidenceItem
from domain.models import Reference
from tools.contracts import ToolInput, ToolOutput


class BlastSearchInput(ToolInput):
    """Find sequences homologous to a gap's flanking context."""

    sequence: str = Field(description="Query residues - normally a gap's joined flanks.")
    gap_id: str | None = Field(default=None, description="Gap this search is on behalf of.")
    # Widens the subject region fetched for a hit: the missing segment sits
    # between the flanks, so it is absent from every HSP by definition.
    gap_length: int = Field(default=0, ge=0, description="Width of the gap being filled.")
    # EMBL-EBI partitions ENA by division and molecule type - there is no
    # single "em_rel" catch-all, and passing one is rejected as an invalid
    # parameter. Vertebrate coding sequence is the right default for an animal
    # genomics agent; `GET /ncbiblast/parameterdetails/database` lists the rest.
    database: str = Field(default="em_cds_std_vrt", description="EMBL-EBI database code.")
    program: str = "blastn"
    max_hits: int = Field(default=50, ge=1, le=1000)
    # 1e-5 is strict enough to exclude chance similarity over a few hundred
    # bases while still admitting genuinely divergent homologues.
    expect: float = 1e-5


class BlastHit(ToolOutput):
    """One alignment reported by BLAST."""

    accession: str
    description: str | None = None
    organism: str | None = None
    identity: float | None = Field(default=None, ge=0.0, le=1.0)
    coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    e_value: float | None = None
    bit_score: float | None = None


class BlastSearchOutput(ToolOutput):
    references: list[Reference] = Field(
        default_factory=list, description="Hits as domain references, best first."
    )
    evidence: list[EvidenceItem] = Field(default_factory=list)
    job_id: str | None = None
    total_hits: int = 0
