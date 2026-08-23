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
    # Where the flanks meet in `sequence`. A reference that still carries the
    # missing segment aligns with a gap run in the query row at exactly this
    # point, which is how the mapper tells a donor from a look-alike.
    left_flank_length: int | None = Field(
        default=None, ge=0, description="End of the left flank within `sequence`."
    )
    # EMBL-EBI partitions ENA by division and molecule type - there is no
    # single "em_rel" catch-all, and passing one is rejected as an invalid
    # parameter. `GET /ncbiblast/parameterdetails/database` lists them.
    #
    # `em_std_vrt` ("ENA Sequence Standard Vertebrate") rather than
    # `em_cds_std_vrt` ("ENA Coding Standard Vertebrate"), which was the
    # default and searched coding sequence only. A mitochondrial gene gap is
    # inside a CDS and matched fine; a gap in a nuclear scaffold - which is
    # what the Genome agent hands over, and what most real assemblies need -
    # is usually intronic or intergenic and could never match a coding-only
    # database, so those runs came back with no evidence whatever the query.
    database: str = Field(default="em_std_vrt", description="EMBL-EBI database code.")
    program: str = "blastn"
    max_hits: int = Field(default=50, ge=1, le=1000)
    # 1e-5 is strict enough to exclude chance similarity over a few hundred
    # bases while still admitting genuinely divergent homologues.
    expect: float = 1e-5
    # An EMBL-EBI job already submitted for this same search, to be polled
    # instead of submitted again.
    #
    # This field travels in BOTH directions, which is unusual for a tool input
    # and is the point. An EBI BLAST job takes ~205 s (measured); one slice
    # grants at most 75 s. The tool therefore writes the id here the moment
    # `submit` returns, BEFORE the long poll it is going to be cancelled in -
    # so when `asyncio.wait_for` kills the call, the caller still holds the
    # payload and can read the id off it. The next slice passes it back and the
    # tool resumes polling the same job. Without this the id died with the
    # cancelled coroutine, every slice resubmitted from zero, and four slices
    # of 70 s could never finish one 205 s job.
    job_id: str | None = Field(
        default=None, description="Resume this EMBL-EBI job instead of submitting a new one."
    )


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
