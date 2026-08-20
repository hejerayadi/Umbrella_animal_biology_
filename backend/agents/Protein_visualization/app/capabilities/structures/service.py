import asyncio
from typing import Any

from backend.agents.Protein_visualization.app.domain.enums import PreferredSource, StructureSource
from backend.agents.Protein_visualization.app.domain.models import (
    ProteinStructureRequest,
    ResolvedProtein,
    StructureCandidate,
)
from backend.agents.Protein_visualization.app.tools.alphafold_client import AlphaFoldClient
from backend.agents.Protein_visualization.app.tools.rcsb_client import RCSBClient


class StructureCapability:
    def __init__(self, rcsb: RCSBClient, alphafold: AlphaFoldClient, max_candidates: int = 10) -> None:
        self.rcsb = rcsb
        self.alphafold = alphafold
        self.max_candidates = max_candidates

    async def experimental(self, protein: ResolvedProtein) -> list[StructureCandidate]:
        entity_refs = await self.rcsb.find_by_uniprot(protein.uniprot_accession, self.max_candidates)
        results = await asyncio.gather(
            *(self._experimental_entity(reference, protein) for reference in entity_refs),
            return_exceptions=True,
        )
        candidates = [result for result in results if isinstance(result, StructureCandidate)]
        if not candidates:
            for result in results:
                if isinstance(result, Exception):
                    raise result
        return candidates

    async def _experimental_entity(
        self,
        reference: str,
        protein: ResolvedProtein,
    ) -> StructureCandidate | None:
        try:
            pdb_id, entity_id = reference.rsplit("_", 1)
        except ValueError:
            return None

        entity, entry = await asyncio.gather(
            self.rcsb.polymer_entity(pdb_id, entity_id),
            self.rcsb.entry(pdb_id),
        )
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers", {})
        references = identifiers.get("reference_sequence_identifiers") or []
        matching = [
            item
            for item in references
            if str(item.get("database_name", "")).lower() == "uniprot"
            and str(item.get("database_accession", "")).upper() == protein.uniprot_accession.upper()
        ]
        if not matching:
            return None

        coverage = self._reference_coverage(entity, matching, protein)
        auth_chains = identifiers.get("auth_asym_ids") or identifiers.get("asym_ids") or []
        chain_id = str(auth_chains[0]) if auth_chains else None
        resolution_values = entry.get("rcsb_entry_info", {}).get("resolution_combined") or []
        resolution = float(resolution_values[0]) if resolution_values else None
        method = (entry.get("exptl") or [{}])[0].get("method")
        return StructureCandidate(
            source=StructureSource.pdb,
            external_id=pdb_id.upper(),
            structure_type="EXPERIMENTAL",
            chain_id=chain_id,
            experimental_method=method,
            resolution_angstrom=resolution,
            sequence_coverage=coverage,
            mean_plddt=None,
            file_format="MMCIF",
            file_url=self.rcsb.structure_urls(pdb_id)["data"],
            selection_score=self.score_pdb(coverage, resolution),
        )

    @staticmethod
    def _reference_coverage(
        entity: dict[str, Any],
        matching_references: list[dict[str, Any]],
        protein: ResolvedProtein,
    ) -> float:
        reported = [
            float(item["reference_sequence_coverage"])
            for item in matching_references
            if item.get("reference_sequence_coverage") is not None
        ]
        if reported:
            return max(0.0, min(1.0, max(reported)))

        entity_length = entity.get("entity_poly", {}).get("rcsb_sample_sequence_length")
        if entity_length and protein.sequence:
            return max(0.0, min(1.0, float(entity_length) / len(protein.sequence)))
        return 0.0

    async def predicted(self, protein: ResolvedProtein) -> list[StructureCandidate]:
        records = await self.alphafold.predictions(protein.uniprot_accession)
        return [
            StructureCandidate(
                source=StructureSource.alphafold,
                external_id=item.get("entryId", f"AF-{protein.uniprot_accession}"),
                structure_type="PREDICTED",
                chain_id="A",
                experimental_method=None,
                resolution_angstrom=None,
                sequence_coverage=1.0,
                mean_plddt=float(item["globalMetricValue"]) if item.get("globalMetricValue") else None,
                file_format="MMCIF" if item.get("cifUrl") else "PDB",
                file_url=item.get("cifUrl") or item["pdbUrl"],
                selection_score=self.score_alphafold(item.get("globalMetricValue")),
                warnings=("Predicted structure; confidence varies by residue.",),
            )
            for item in records
        ]

    @staticmethod
    def score_pdb(coverage: float, resolution: float | None) -> float:
        resolution_score = max(0.0, 1.0 - ((resolution or 4.0) - 1.0) / 5.0)
        return round(0.7 * coverage + 0.3 * resolution_score, 4)

    @staticmethod
    def score_alphafold(mean_plddt: float | None) -> float:
        return round(0.75 * ((mean_plddt or 0.0) / 100.0), 4)

    @staticmethod
    def select(
        request: ProteinStructureRequest,
        pdb: list[StructureCandidate],
        alphafold: list[StructureCandidate],
    ) -> tuple[StructureCandidate | None, list[StructureCandidate]]:
        if request.preferred_source is PreferredSource.alphafold:
            pool = alphafold or pdb
        elif request.preferred_source is PreferredSource.pdb:
            pool = pdb
        else:
            pool = pdb or alphafold
        ranked = sorted(pool, key=lambda item: item.selection_score, reverse=True)
        return (
            (ranked[0], ranked[1:] + [item for item in pdb + alphafold if item not in ranked])
            if ranked
            else (None, [])
        )
