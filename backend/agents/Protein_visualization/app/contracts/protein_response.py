from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from backend.agents.Protein_visualization.app.domain.enums import AnalysisStatus, ValidationStatus


class AnalysisAccepted(BaseModel):
    analysis_id: UUID
    task_id: UUID
    status: AnalysisStatus = AnalysisStatus.received


class ProteinSummary(BaseModel):
    uniprot_accession: str
    gene_symbol: str
    scientific_name: str
    taxon_id: int


class StructureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source: str
    external_id: str
    structure_type: str
    chain_id: str | None = None
    experimental_method: str | None = None
    resolution_angstrom: float | None = None
    sequence_coverage: float
    mean_plddt: float | None = None
    file_format: str
    file_url: str
    selection_score: float
    warnings: tuple[str, ...] = ()


class ExplanationResponse(BaseModel):
    summary: str
    limitations: list[str] = Field(default_factory=list)


class ProteinAnalysisResponse(BaseModel):
    analysis_id: UUID
    task_id: UUID
    status: AnalysisStatus
    validation_status: ValidationStatus
    protein: ProteinSummary | None = None
    selected_structure: StructureResponse | None = None
    alternative_structures: list[StructureResponse] = Field(default_factory=list)
    annotations: list[dict[str, Any]] = Field(default_factory=list)
    residue_mappings: list[dict[str, Any]] = Field(default_factory=list)
    molstar_config: dict[str, Any] = Field(default_factory=dict)
    explanation: ExplanationResponse | None = None
    warnings: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
