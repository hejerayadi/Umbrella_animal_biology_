from typing import Any

from app.domain.models import Annotation, ResidueMapping, StructureCandidate, VisualizationSpec


class VisualizationCapability:
    """Builds the Mol* scene. Nothing is highlighted without a validated mapping."""

    def build(
        self, structure: StructureCandidate | None, mappings: list[ResidueMapping]
    ) -> VisualizationSpec | None:
        if structure is None:
            return None
        selections = tuple(
            {
                "label": f"UniProt residue {item.uniprot_position}",
                "chain": item.chain_id,
                "residue_number": item.pdb_residue_number,
                "color": "#f59e0b",
            }
            for item in mappings
            if item.is_observed and item.pdb_residue_number is not None
        )
        return VisualizationSpec(
            viewer="molstar",
            structure_id=structure.external_id,
            structure_url=structure.file_url,
            selections=selections,
        )

    def to_config(
        self,
        spec: VisualizationSpec | None,
        structure: StructureCandidate | None,
        annotations: list[Annotation],
    ) -> dict[str, Any]:
        """Serialize the scene for the frontend.

        Domain spans are InterPro/UniProt coordinates. They are only marked
        applicable when the structure is numbered in UniProt space (AlphaFold
        models); on an experimental entry they are reported untouched, because
        translating them would require a SIFTS mapping per span.
        """
        if spec is None or structure is None:
            return {}

        uniprot_numbering = structure.structure_type == "PREDICTED"
        domains = [
            {
                "label": annotation.label,
                "accession": annotation.accession,
                "kind": annotation.kind,
                "start": annotation.start,
                "end": annotation.end,
                "coordinate_space": "structure" if uniprot_numbering else "uniprot",
                "applicable": uniprot_numbering,
            }
            for annotation in annotations
            if annotation.start is not None and annotation.end is not None
        ]

        return {
            "viewer": spec.viewer,
            "structure": {
                "id": spec.structure_id,
                "url": spec.structure_url,
                "format": structure.file_format,
                "source": structure.source.value,
                "structure_type": structure.structure_type,
                "chain_id": structure.chain_id,
            },
            "representation": {"type": spec.representation, "color_theme": "chain-id"},
            "selections": list(spec.selections),
            "domains": domains,
        }
