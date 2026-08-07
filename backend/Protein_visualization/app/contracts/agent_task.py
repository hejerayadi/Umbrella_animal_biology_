from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import PreferredSource
from app.domain.models import ProteinStructureRequest, SpeciesRef


class SpeciesContract(BaseModel):
    scientific_name: str = Field(min_length=2)
    taxon_id: int = Field(gt=0)


class ProteinTaskInput(BaseModel):
    resolved_gene_id: str = Field(min_length=1)
    species: SpeciesContract
    uniprot_accession: str | None = None
    protein_sequence: str | None = None
    requested_regions: list[str] = Field(default_factory=list)
    residue_position: int | None = Field(default=None, gt=0)
    mutation: str | None = None
    preferred_source: PreferredSource = PreferredSource.auto
    include_explanation: bool = True

    @model_validator(mode="after")
    def require_identity_anchor(self) -> "ProteinTaskInput":
        if not self.uniprot_accession and not self.protein_sequence and not self.resolved_gene_id:
            raise ValueError("uniprot_accession, protein_sequence, or resolved_gene_id is required")
        return self


class AgentTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    trace_id: UUID
    source_agent: str = "umbrella-main-orchestrator"
    target_capability: str = "protein-structure.orchestrate"
    schema_version: str = "1.0"
    idempotency_key: str = Field(min_length=1, max_length=200)
    input: ProteinTaskInput

    def to_domain(self) -> ProteinStructureRequest:
        return ProteinStructureRequest(
            task_id=self.task_id,
            trace_id=self.trace_id,
            resolved_gene_id=self.input.resolved_gene_id,
            species=SpeciesRef(**self.input.species.model_dump()),
            uniprot_accession=self.input.uniprot_accession,
            protein_sequence=self.input.protein_sequence,
            requested_regions=tuple(self.input.requested_regions),
            residue_position=self.input.residue_position,
            mutation=self.input.mutation,
            preferred_source=self.input.preferred_source,
            include_explanation=self.input.include_explanation,
        )
