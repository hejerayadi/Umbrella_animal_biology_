"""Builds the EvidencePack the explanation and the critic are allowed to see.

Only facts derived from data actually returned by a provider are emitted; nothing
here paraphrases, infers, or extrapolates. The LLM receives this pack and nothing
else, so an absent fact cannot become a generated claim.
"""

from app.domain.models import (
    Annotation,
    EvidencePack,
    EvidenceRef,
    KnowledgeHit,
    ProteinStructureRequest,
    ResidueMapping,
    ResolvedProtein,
    StructureCandidate,
)

MAX_ANNOTATION_FACTS = 8
MAX_KNOWLEDGE_FACTS = 5


class EvidenceCapability:
    def build(
        self,
        request: ProteinStructureRequest,
        protein: ResolvedProtein | None,
        structure: StructureCandidate | None,
        annotations: list[Annotation],
        mappings: list[ResidueMapping],
        knowledge: list[KnowledgeHit],
        evidence: list[EvidenceRef],
        warnings: list[str],
    ) -> EvidencePack:
        facts: list[str] = []
        limitations: list[str] = []

        if protein:
            facts.append(
                f"UniProt {protein.uniprot_accession} is the canonical entry for gene "
                f"{protein.gene_symbol} in {protein.scientific_name} (taxon {protein.taxon_id})."
            )
            if protein.protein_name:
                facts.append(f"UniProt records the protein name as {protein.protein_name}.")

        if structure is None:
            limitations.append("No structure satisfied the selection rules, so no viewer scene was produced.")
        elif structure.structure_type == "EXPERIMENTAL":
            method = structure.experimental_method or "an unreported method"
            resolution = (
                f" at {structure.resolution_angstrom} Å resolution"
                if structure.resolution_angstrom is not None
                else ""
            )
            facts.append(
                f"{structure.external_id} is an experimental structure from RCSB PDB determined by "
                f"{method}{resolution}, covering {structure.sequence_coverage:.0%} of the sequence."
            )
            if structure.sequence_coverage < 1.0:
                limitations.append(
                    f"{structure.external_id} covers only {structure.sequence_coverage:.0%} of the "
                    "UniProt sequence; regions outside that range are not represented."
                )
        else:
            plddt = (
                f" with a mean pLDDT of {structure.mean_plddt}" if structure.mean_plddt is not None else ""
            )
            facts.append(
                f"{structure.external_id} is a predicted AlphaFold model{plddt}; it is not "
                "experimentally determined."
            )
            limitations.append(
                "The selected structure is an AlphaFold prediction. Per-residue confidence varies and "
                "it must not be described as experimental."
            )

        for annotation in annotations[:MAX_ANNOTATION_FACTS]:
            span = (
                f" spanning residues {annotation.start}-{annotation.end}"
                if annotation.start is not None and annotation.end is not None
                else ""
            )
            facts.append(
                f"{annotation.source} {annotation.accession} ({annotation.kind}): {annotation.label}{span}."
            )
        if len(annotations) > MAX_ANNOTATION_FACTS:
            limitations.append(
                f"{len(annotations) - MAX_ANNOTATION_FACTS} further InterPro entries were returned "
                "but are not included in this summary."
            )

        for mapping in mappings:
            if mapping.is_observed:
                facts.append(
                    f"SIFTS maps UniProt position {mapping.uniprot_position} to residue "
                    f"{mapping.pdb_residue_number} of chain {mapping.chain_id} in {mapping.pdb_id}."
                )
            else:
                limitations.append(
                    f"UniProt position {mapping.uniprot_position} is not observed in "
                    f"{mapping.pdb_id}; it cannot be highlighted."
                )

        if request.residue_position is not None and not mappings:
            limitations.append(
                f"No SIFTS mapping was obtained for UniProt position {request.residue_position}; "
                "the residue is not highlighted."
            )
        if request.mutation and not mappings:
            limitations.append(
                f"The requested mutation {request.mutation} could not be located in the selected structure."
            )

        for hit in sorted(knowledge, key=lambda item: item.score, reverse=True)[:MAX_KNOWLEDGE_FACTS]:
            source = hit.metadata.get("source", "knowledge base")
            facts.append(f"Indexed {source} document {hit.id}: {hit.text}")
        if not knowledge:
            limitations.append("No indexed knowledge document matched this protein and species.")

        limitations.extend(warnings)

        return EvidencePack(
            evidence=tuple(evidence),
            facts=tuple(facts),
            limitations=tuple(dict.fromkeys(limitations)),
        )
