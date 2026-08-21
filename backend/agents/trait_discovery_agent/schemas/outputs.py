from dataclasses import dataclass, field
from typing import List, Optional
from .common import AgentStatus

@dataclass
class GOAnnotation:
    gene_symbol: str
    go_id: str
    go_name: str

@dataclass
class GeneMapperOutput:
    status: AgentStatus
    go_annotations: List[GOAnnotation] = field(default_factory=list)
    unmatched_genes: List[str] = field(default_factory=list)
    target_agent: Optional[str] = None
    prompt_to_target_agent: Optional[str] = None

@dataclass
class PathwayEntry:
    pathway_id: str
    pathway_name: str
    reasoning: str = ""          # NEW — why this pathway was selected

@dataclass
class PathwaysOutput:
    status: AgentStatus
    pathways: List[PathwayEntry] = field(default_factory=list)
    malformed_ids: List[str] = field(default_factory=list)
    target_agent: Optional[str] = None
    prompt_to_target_agent: Optional[str] = None

@dataclass
class ProteinEntry:
    gene_symbol: str
    protein_name: str
    function_summary: str
    source_accession: str = ""

@dataclass
class ProteinDataOutput:
    status: AgentStatus
    proteins: List[ProteinEntry] = field(default_factory=list)
    missing_genes: List[str] = field(default_factory=list)
    target_agent: Optional[str] = None
    prompt_to_target_agent: Optional[str] = None

@dataclass
class FunctionalEvidenceOutput:
    status: AgentStatus
    pathway_data: List[PathwayEntry] = field(default_factory=list)
    protein_data: List[ProteinEntry] = field(default_factory=list)
    target_agent: Optional[str] = None
    prompt_to_target_agent: Optional[str] = None

@dataclass
class LiteratureRecord:
    pmid: str
    title: str
    year: int
    short_summary: str

@dataclass
class LiteratureSupportOutput:
    status: AgentStatus
    evidence: List[LiteratureRecord] = field(default_factory=list)
    target_agent: Optional[str] = None
    prompt_to_target_agent: Optional[str] = None

@dataclass
class TraitDiscoveryOutput:
    status: AgentStatus
    go_annotations: List[GOAnnotation] = field(default_factory=list)
    pathway_data: List[PathwayEntry] = field(default_factory=list)
    protein_data: List[ProteinEntry] = field(default_factory=list)
    evidence: List[LiteratureRecord] = field(default_factory=list)
    explanation: str = ""
    target_agent: Optional[str] = None
    prompt_to_target_agent: Optional[str] = None