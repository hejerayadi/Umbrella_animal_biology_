from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DocumentType = Literal[
    "protein_function",
    "protein_domain",
    "experimental_structure_description",
    "predicted_structure_description",
    "scientific_abstract",
    "curated_evidence",
]


class KnowledgeDocument(BaseModel):
    document_id: str
    text: str = Field(min_length=1)
    domain: Literal["protein"] = "protein"
    source: str
    source_record_id: str
    protein_id: str
    gene_symbol: str
    species_name: str
    taxonomy_id: int
    document_type: DocumentType
    section: str
    language: str = "en"
    content_sha256: str | None = None
    indexed_at: datetime | None = None


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    protein_id: str
    taxonomy_id: int
    document_types: list[DocumentType] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=50)


class IngestionResult(BaseModel):
    inserted: int
    skipped: int = 0
