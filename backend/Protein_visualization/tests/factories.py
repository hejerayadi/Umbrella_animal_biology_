"""Builders shared by the unit, integration and end-to-end suites."""

from typing import Any
from uuid import uuid4

from app.contracts.agent_task import AgentTask
from app.domain.enums import PreferredSource, StructureSource
from app.domain.models import (
    Annotation,
    ProteinStructureRequest,
    ResidueMapping,
    ResolvedProtein,
    SpeciesRef,
    StructureCandidate,
)

P53 = ResolvedProtein(
    uniprot_accession="P04637",
    gene_symbol="TP53",
    scientific_name="Homo sapiens",
    taxon_id=9606,
    protein_name="Cellular tumor antigen p53",
    sequence="MEEPQSDPSV",
)


def agent_task(**input_overrides: Any) -> AgentTask:
    payload: dict[str, Any] = {
        "resolved_gene_id": "TP53",
        "species": {"scientific_name": "Homo sapiens", "taxon_id": 9606},
        "uniprot_accession": "P04637",
    }
    payload.update(input_overrides)
    return AgentTask.model_validate(
        {
            "task_id": str(uuid4()),
            "trace_id": str(uuid4()),
            "idempotency_key": f"test-{uuid4()}",
            "input": payload,
        }
    )


def structure_request(**overrides: Any) -> ProteinStructureRequest:
    defaults: dict[str, Any] = {
        "task_id": uuid4(),
        "trace_id": uuid4(),
        "resolved_gene_id": "TP53",
        "species": SpeciesRef("Homo sapiens", 9606),
        "uniprot_accession": "P04637",
    }
    defaults.update(overrides)
    # The dataclass does no coercion; the HTTP contract does. Mirror that here.
    defaults["preferred_source"] = PreferredSource(defaults.get("preferred_source", PreferredSource.auto))
    return ProteinStructureRequest(**defaults)


def pdb_candidate(**overrides: Any) -> StructureCandidate:
    defaults: dict[str, Any] = {
        "source": StructureSource.pdb,
        "external_id": "1TUP",
        "structure_type": "EXPERIMENTAL",
        "chain_id": "A",
        "experimental_method": "X-RAY DIFFRACTION",
        "resolution_angstrom": 2.2,
        "sequence_coverage": 0.72,
        "mean_plddt": None,
        "file_format": "MMCIF",
        "file_url": "https://files.rcsb.org/download/1TUP.cif",
        "selection_score": 0.65,
    }
    defaults.update(overrides)
    return StructureCandidate(**defaults)


def alphafold_candidate(**overrides: Any) -> StructureCandidate:
    defaults: dict[str, Any] = {
        "source": StructureSource.alphafold,
        "external_id": "AF-P04637-F1",
        "structure_type": "PREDICTED",
        "chain_id": "A",
        "experimental_method": None,
        "resolution_angstrom": None,
        "sequence_coverage": 1.0,
        "mean_plddt": 82.1,
        "file_format": "MMCIF",
        "file_url": "https://alphafold.ebi.ac.uk/files/AF-P04637-F1-model_v4.cif",
        "selection_score": 0.61,
        "warnings": ("Predicted structure; confidence varies by residue.",),
    }
    defaults.update(overrides)
    return StructureCandidate(**defaults)


def annotation(**overrides: Any) -> Annotation:
    defaults: dict[str, Any] = {
        "source": "InterPro",
        "accession": "IPR011615",
        "kind": "domain",
        "label": "p53 DNA-binding domain",
        "start": 102,
        "end": 292,
    }
    defaults.update(overrides)
    return Annotation(**defaults)


def residue_mapping(**overrides: Any) -> ResidueMapping:
    defaults: dict[str, Any] = {
        "uniprot_accession": "P04637",
        "uniprot_position": 273,
        "pdb_id": "1TUP",
        "chain_id": "A",
        "pdb_residue_number": "273",
        "is_observed": True,
    }
    defaults.update(overrides)
    return ResidueMapping(**defaults)
