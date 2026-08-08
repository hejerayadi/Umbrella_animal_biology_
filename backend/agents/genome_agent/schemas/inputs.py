from dataclasses import dataclass
from typing import Any

from .common import AgentRequest


@dataclass
class SpeciesResolverRequest(AgentRequest):
    species_name: str = ""


@dataclass
class GenomeMetadataRequest(AgentRequest):
    assembly_id: str = ""


@dataclass
class GeneAnnotationRequest(AgentRequest):
    assembly_id: str = ""


@dataclass
class VisualizationRequest(AgentRequest):
    scope: str = "chromosome_map"
    genome_size_bp: int | None = None
    gene_table: list[dict[str, Any]] | None = None
