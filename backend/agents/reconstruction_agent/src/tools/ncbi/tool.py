"""Reference retrieval from NCBI, as a tool the planner can select."""
from __future__ import annotations

from configuration.logging import get_logger
from contracts.output import EvidenceItem
from domain.exceptions import ReconstructionError
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
            references = to_references(raw_fasta)

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
                diagnostics={"identifiers": identifiers},
            )

        except ReconstructionError as error:
            # Reported, not raised: the graph records the failure and continues
            # with whatever other evidence it has.
            _log.warning("ncbi_search_failed", error=str(error))
            return NCBISearchOutput(succeeded=False, error=str(error))

    @staticmethod
    def _build_term(payload: NCBISearchInput) -> str:
        """Combine the free-text term with any organism restriction."""
        if not payload.organisms:
            return payload.term

        organisms = " OR ".join(f'"{organism}"[Organism]' for organism in payload.organisms)
        return f"({payload.term}) AND ({organisms})" if payload.term else f"({organisms})"
