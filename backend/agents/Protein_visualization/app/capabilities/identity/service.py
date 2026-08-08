from datetime import UTC, datetime
from typing import Any

from backend.agents.Protein_visualization.app.domain.exceptions import ProteinNotFoundError
from backend.agents.Protein_visualization.app.domain.models import EvidenceRef, ProteinStructureRequest, ResolvedProtein
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
            if len(records) != 1:
                raise ProteinNotFoundError("Protein identity is ambiguous")
            entry = records[0]
        return self._normalize(entry, request)

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
