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
    # No default, on purpose. This used to be `em_std_vrt` - "ENA Sequence
    # Standard Vertebrate" - and that single constant was the agent's central
    # failure. EMBL divisions are MUTUALLY EXCLUSIVE: VRT means *other*
    # vertebrates and explicitly excludes mammals, human, mouse and rodent. For
    # a mammal target the correct homologues were not in the searched database
    # at all, so no e-value, retry or ranking change could surface them.
    #
    # Measured on the polar bear mitogenome, same query and parameters:
    # `em_std_vrt` returned 50 hits of which 7 crossed the gap, all fish at
    # ~67% identity; `em_std_mam` returned 50 hits of which 50 crossed it,
    # Ursus maritimus at 95.7%, recovering all 45 masked bases exactly.
    #
    # The database is now discovered from EBI's live catalogue, chosen by the
    # planner, and settled by measuring which candidate produces gap carriers.
    # `None` means "not chosen yet"; the tool resolves it and writes the answer
    # back here so a resumed slice reuses the same one.
    database: str | None = Field(default=None, description="EMBL-EBI database code.")
    # Filed against the caller's target so a run can attribute a result to a
    # database, and so the planner's next round can compare them.
    division: str | None = Field(
        default=None, description="Taxonomic division this database belongs to."
    )
    # DUST masking of low-complexity query regions, sent explicitly rather than
    # left to EBI's default. See `BlastClient.submit` for why this is standard
    # practice but not a timeout fix.
    low_complexity_filter: bool = Field(
        default=True, description="Mask low-complexity query regions (DUST)."
    )
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
    #: Which database produced this, and how well it did. The pair the planner
    #: compares when deciding where to search next: a database is chosen on
    #: whether it yields references that actually cross the gap, not on hit
    #: count, which can be high and entirely useless.
    database: str | None = None
    hits_carrying_gap: int = 0
