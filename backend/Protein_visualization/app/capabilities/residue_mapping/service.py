from app.domain.models import ProteinStructureRequest, ResidueMapping, ResolvedProtein, StructureCandidate
from app.tools.sifts_client import SiftsClient


class ResidueMappingCapability:
    def __init__(self, client: SiftsClient) -> None:
        self.client = client

    async def map(
        self,
        request: ProteinStructureRequest,
        protein: ResolvedProtein,
        structure: StructureCandidate,
    ) -> list[ResidueMapping]:
        if request.residue_position is None or structure.structure_type != "EXPERIMENTAL":
            return []
        payload = await self.client.mappings(structure.external_id)
        root = payload.get(structure.external_id.lower(), {})
        mappings = root.get("UniProt", {}).get(protein.uniprot_accession, {}).get("mappings", [])
        result = []
        for item in mappings:
            if item.get("unp_start", 0) <= request.residue_position <= item.get("unp_end", -1):
                offset = request.residue_position - item["unp_start"]
                pdb_start = item.get("start", {}).get("residue_number")
                result.append(
                    ResidueMapping(
                        uniprot_accession=protein.uniprot_accession,
                        uniprot_position=request.residue_position,
                        pdb_id=structure.external_id,
                        chain_id=item.get("chain_id", "?"),
                        pdb_residue_number=str(pdb_start + offset) if pdb_start is not None else None,
                        is_observed=pdb_start is not None,
                    )
                )
        return result
