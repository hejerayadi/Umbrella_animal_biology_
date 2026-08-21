from dataclasses import dataclass, field
from typing import Any, List
from .outputs import GOAnnotation

@dataclass
class TraitDiscoveryInput:
    trait_name: str
    species_name: str
    instruction: str
    context: dict = field(default_factory=dict)

@dataclass
class FunctionalEvidenceInput:
    gene_list: List[str]
    trait_name: str                # NEW — threaded from orchestrator
    go_annotations: List[GOAnnotation]
    instruction: str
    context: dict = field(default_factory=dict)

@dataclass
class GeneMapperInput:
    trait_name: str
    gene_list: List[str]
    species_name: str
    instruction: str
    context: dict = field(default_factory=dict)

@dataclass
class PathwaysInput:
    gene_list: List[str]
    trait_name: str                # NEW — needed for relevance judgment
    instruction: str
    context: dict = field(default_factory=dict)

@dataclass
class ProteinDataInput:
    gene_list: List[str]
    trait_name: str                # NEW — needed for relevance judgment
    instruction: str
    context: dict = field(default_factory=dict)

@dataclass
class LiteratureSupportInput:
    trait_name: str
    gene_list: List[str]
    instruction: str
    context: dict = field(default_factory=dict)