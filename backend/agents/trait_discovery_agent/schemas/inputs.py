from dataclasses import dataclass, field
from typing import Any, List
from .outputs import GOAnnotation


@dataclass
class TraitDiscoveryInput:
    trait_name: str
    species_name: str
    instruction: str
    context: dict = field(default_factory=dict)
    # NOTE: no gene_list field here — it's a Genome Agent cross-agent dependency,
    # expected to arrive via context["gene_list"] once resolved upstream.


@dataclass
class FunctionalEvidenceInput:
    gene_list: List[str]              # from Gene Mapper, via orchestrator
    go_annotations: List[GOAnnotation]  # from Gene Mapper
    instruction: str
    context: dict = field(default_factory=dict)


@dataclass
class GeneMapperInput:
    trait_name: str
    gene_list: List[str]      # from Genome Agent, not fetched here
    species_name: str         # for disambiguation
    instruction: str
    context: dict = field(default_factory=dict)


@dataclass
class PathwaysInput:
    gene_list: List[str]      # passed down from Gene Mapper via sub-orchestrator
    instruction: str
    context: dict = field(default_factory=dict)


@dataclass
class ProteinDataInput:
    gene_list: List[str]      # same list as Pathways, both run in parallel
    instruction: str
    context: dict = field(default_factory=dict)


@dataclass
class LiteratureSupportInput:
    trait_name: str
    gene_list: List[str]      # from Gene Mapper
    instruction: str
    context: dict = field(default_factory=dict)