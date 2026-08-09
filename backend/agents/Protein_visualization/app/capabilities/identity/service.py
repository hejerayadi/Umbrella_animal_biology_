from datetime import UTC, datetime
from typing import Any

from backend.agents.Protein_visualization.app.domain.exceptions import ProteinNotFoundError
from backend.agents.Protein_visualization.app.domain.models import (
    EvidenceRef,
    ProteinStructureRequest,
    ResolvedProtein,
)
from backend.agents.Protein_visualization.app.tools.uniprot_client import UniProtClient


class IdentityCapability:
    def __init__(self, client: UniProtClient) -> None:
        self.client = client

    async def resolve(self, request: ProteinStructureRequest) -> tuple[ResolvedProtein, EvidenceRef]:
        if request.uniprot_accession:
            entry = await self.client.get_entry(request.uniprot_accession)
        else:
            records = await self.client.search(
                f"gene_exact:{request.resolved_gene_id}", request.species.scientific_name
            )
            entry = self._canonical(records)
        return self._normalize(entry, request)

    @staticmethod
    def _canonical(records: list[dict[str, Any]]) -> dict[str, Any]:
        """The one entry a gene symbol denotes, out of everything UniProt returns.

        A gene search never comes back with a single record for a well-studied
        gene: TP53 in human answers with the Swiss-Prot entry plus a handful of
        TrEMBL fragments and isoforms, all carrying the same protein name. That
        is not ambiguity - Swiss-Prot review is exactly the curation that marks
        one of them canonical, so a single reviewed entry settles it.

        Ambiguity is when curation cannot: no reviewed entry and several
        unreviewed candidates, or two reviewed entries for one symbol. Picking
        the first would silently model a fragment, so those still raise.
        """

        def is_reviewed(record: dict[str, Any]) -> bool:
            entry_type = str(record.get("entryType", "")).casefold()
            return "reviewed" in entry_type and "unreviewed" not in entry_type

        reviewed = [record for record in records if is_reviewed(record)]
        if len(reviewed) == 1:
            return reviewed[0]
        if not reviewed and len(records) == 1:
            return records[0]
        if not records:
            raise ProteinNotFoundError("UniProt returned no protein for this gene and species")
        raise ProteinNotFoundError(
            f"Protein identity is ambiguous: {len(reviewed)} reviewed of {len(records)} candidates"
        )

    @staticmethod
    def _normalize(
        entry: dict[str, Any], request: ProteinStructureRequest
    ) -> tuple[ResolvedProtein, EvidenceRef]:
        accession = entry.get("primaryAccession")
        organism = entry.get("organism", {})
        if not accession:
            raise ProteinNotFoundError("UniProt did not return a canonical accession")
        if organism.get("taxonId") and organism["taxonId"] != request.species.taxon_id:
            raise ProteinNotFoundError("UniProt identity does not match the requested species")
        genes = entry.get("genes", [])
        gene = genes[0].get("geneName", {}).get("value") if genes else request.resolved_gene_id
        protein_name = (
            entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
        )
        protein = ResolvedProtein(
            uniprot_accession=accession,
            gene_symbol=gene or request.resolved_gene_id,
            scientific_name=organism.get("scientificName", request.species.scientific_name),
            taxon_id=organism.get("taxonId", request.species.taxon_id),
            protein_name=protein_name,
            sequence=entry.get("sequence", {}).get("value") or request.protein_sequence,
        )
        evidence = EvidenceRef(
            provider="UniProt",
            external_id=accession,
            retrieved_at=datetime.now(UTC).isoformat(),
            source_url=f"https://www.uniprot.org/uniprotkb/{accession}",
        )
        return protein, evidence
