from dataclasses import dataclass
from typing import Any

from .common import AgentStatus, AgentResult


@dataclass
class SpeciesResolverOutput:
    assembly_id: str | None
    scientific_name: str | None
    common_name: str | None
    confidence: float


@dataclass
class GenomeMetadataOutput:
    genome_size_bp: int | None
    chromosome_count: int | None
    karyotype: str | None
    assembly_level: str | None


@dataclass
class GeneAnnotationOutput:
    gene_table: list[dict[str, Any]]
    gene_list: list[str]


@dataclass
class VisualizationOutput:
    status: str
    chart_data: bytes | None
    format: str | None
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None
    note: str | None = None
    comparisons: list[dict[str, Any]] | None = None


@dataclass
class GenomeAgentOutput:
    species: SpeciesResolverOutput | None
    metadata: GenomeMetadataOutput | None
    annotation: GeneAnnotationOutput | None
    visualization: VisualizationOutput | None
    explanation: str | None
    errors: list[str]
    status: AgentStatus