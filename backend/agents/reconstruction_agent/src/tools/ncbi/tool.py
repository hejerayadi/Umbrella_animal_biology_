"""Reference retrieval from NCBI, as a tool the planner can select."""
from __future__ import annotations

from configuration.logging import get_logger
from contracts.output import EvidenceItem
from domain.exceptions import ReconstructionError
from domain.models import Reference
from infrastructure.ncbi.client import NCBIClient
from tools.contracts import Tool
from tools.ncbi.mapper import to_references
from tools.ncbi.schemas import NCBISearchInput, NCBISearchOutput

_log = get_logger(__name__)


class NCBISearchTool(Tool[NCBISearchInput, NCBISearchOutput]):
    """Finds reference sequences by organism and description.

    This is the agent's entry point to reference data: BLAST finds what is
    homologous to a sequence, but when the caller already knows which relatives
    matter, going straight to NCBI is faster and cheaper.
    """

    name = "ncbi_search"
    description = (
        "Search NCBI nucleotide databases for reference sequences by organism name, gene "
        "or description, and fetch their residues. Use when the relevant relatives are "
        "already known by name. Cheaper and faster than BLAST, but it does not tell you "
        "whether a sequence is homologous to the region being reconstructed."
    )
    estimated_seconds = 3.0

    def __init__(self, client: NCBIClient) -> None:
        self._client = client

    async def run(self, payload: NCBISearchInput) -> NCBISearchOutput:
        try:
            identifiers = await self._client.search(
                self._build_term(payload), database=payload.database, limit=payload.limit
            )

            if not identifiers:
                return NCBISearchOutput(
                    succeeded=True,
                    total_found=0,
                    diagnostics={"term": self._build_term(payload)},
                )

            if not payload.fetch_sequences:
                return NCBISearchOutput(
                    succeeded=True,
                    total_found=len(identifiers),
                    diagnostics={"identifiers": identifiers},
                )

            raw_fasta = await self._client.fetch_fasta(identifiers, database=payload.database)
            references, oversized = self._within_length(
                to_references(raw_fasta), payload.max_reference_length
            )

            return NCBISearchOutput(
                succeeded=True,
                references=references,
                total_found=len(references),
                evidence=[
                    EvidenceItem(
                        source="ncbi",
                        reference_id=reference.accession,
                        organism=reference.organism,
                        note="Retrieved as a candidate reference sequence.",
                    )
                    for reference in references
                ],
                diagnostics={"identifiers": identifiers, "oversized_dropped": oversized},
            )

        except ReconstructionError as error:
            # Reported, not raised: the graph records the failure and continues
            # with whatever other evidence it has.
            _log.warning("ncbi_search_failed", error=str(error))
            return NCBISearchOutput(succeeded=False, error=str(error))

    @staticmethod
    def _build_term(payload: NCBISearchInput) -> str:
        """Combine the free-text term, organism restriction and length bound.

        The `[SLEN]` clause is what keeps whole-genome records out. Entrez
        sorts by relevance, not by size, so without it a gene name matches the
        chromosome carrying the gene just as well as the gene - and the
        chromosome wins on relevance often enough to fill the whole result set.
        Excluding them in the query is the only cheap place: by the time
        `efetch` has answered, the megabytes have already been paid for.
        """
        clauses: list[str] = []

        if payload.term:
            clauses.append(f"({payload.term})")

        if payload.organisms:
            organisms = " OR ".join(f'"{organism}"[Organism]' for organism in payload.organisms)
            clauses.append(f"({organisms})")

        clauses.append(f"1:{payload.max_reference_length}[SLEN]")
        return " AND ".join(clauses)

    @staticmethod
    def _within_length(
        references: list[Reference], limit: int
    ) -> tuple[list[Reference], list[str]]:
        """Drop references longer than the bound, naming what was dropped.

        The `[SLEN]` clause should already have excluded these. This is the
        second line: `[SLEN]` is not honoured identically by every Entrez
        database, and a caller may pass its own `term`. Nothing oversized may
        reach the state, because everything in the state is checkpointed.
        """
        kept: list[Reference] = []
        dropped: list[str] = []

        for reference in references:
            if len(reference.residues or "") > limit:
                dropped.append(reference.accession)
            else:
                kept.append(reference)

        if dropped:
            _log.warning("ncbi_oversized_references_dropped", accessions=dropped, limit=limit)

        return kept, dropped
