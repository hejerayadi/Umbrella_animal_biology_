from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.persistence.db import Base


class OrchestratorTask(Base):
    __tablename__ = "orchestrator_tasks"
    __table_args__ = (UniqueConstraint("idempotency_key"),)
    analysis_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    task_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    trace_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    response_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), index=True)
    capability: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(30))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)


class ToolCall(Base):
    __tablename__ = "tool_calls"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), index=True)
    provider: Mapped[str] = mapped_column(String(80))
    status_code: Mapped[int | None] = mapped_column(Integer)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)


class ProteinEntity(Base):
    __tablename__ = "protein_entities"
    id: Mapped[int] = mapped_column(primary_key=True)
    accession: Mapped[str] = mapped_column(String(30), index=True)
    gene_symbol: Mapped[str] = mapped_column(String(80))
    species_name: Mapped[str] = mapped_column(String(200))
    taxonomy_id: Mapped[int] = mapped_column(Integer)


class StructureCandidateRecord(Base):
    __tablename__ = "structure_candidates"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), index=True)
    source: Mapped[str] = mapped_column(String(30))
    external_id: Mapped[str] = mapped_column(String(40))
    selection_score: Mapped[float] = mapped_column(Float)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class StructureSelection(Base):
    __tablename__ = "structure_selections"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), unique=True)
    external_id: Mapped[str] = mapped_column(String(40))
    reason: Mapped[str] = mapped_column(Text)


class ProteinAnnotationRecord(Base):
    __tablename__ = "protein_annotations"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), index=True)
    annotation_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class ResidueMappingRecord(Base):
    __tablename__ = "residue_mappings"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), index=True)
    mapping_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class RetrievalRun(Base):
    __tablename__ = "retrieval_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), index=True)
    query: Mapped[str] = mapped_column(Text)
    result_count: Mapped[int] = mapped_column(Integer)
    filters_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class VisualizationConfigRecord(Base):
    __tablename__ = "visualization_configs"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), unique=True)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)


class EvidenceRefRecord(Base):
    __tablename__ = "evidence_refs"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("orchestrator_tasks.analysis_id"), index=True)
    provider: Mapped[str] = mapped_column(String(80))
    external_id: Mapped[str] = mapped_column(String(100))
    source_url: Mapped[str | None] = mapped_column(Text)


class IngestionDocument(Base):
    __tablename__ = "ingestion_documents"
    document_id: Mapped[str] = mapped_column(String(200), primary_key=True)
    content_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(30))
