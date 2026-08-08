from backend.agents.Protein_visualization.app.domain.models import Annotation, ResolvedProtein
from backend.agents.Protein_visualization.app.tools.interpro_client import InterProClient


class AnnotationCapability:
    def __init__(self, client: InterProClient) -> None:
        self.client = client

    async def annotate(self, protein: ResolvedProtein) -> list[Annotation]:
        records = await self.client.annotations(protein.uniprot_accession)
        annotations = []
        for record in records:
            metadata = record.get("metadata", {})
            for protein_record in record.get("proteins", []):
                for location in protein_record.get("entry_protein_locations", []):
                    for fragment in location.get("fragments", []):
                        annotations.append(
                            Annotation(
                                source="InterPro",
                                accession=metadata.get("accession", "unknown"),
                                kind=metadata.get("type", "entry"),
                                label=metadata.get("name", "Unlabelled InterPro entry"),
                                start=fragment.get("start"),
                                end=fragment.get("end"),
                            )
                        )
        return annotations
