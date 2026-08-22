from dataclasses import dataclass, field
from typing import List, Optional

from schemas.common import AgentStatus
from schemas.outputs import GOAnnotation, PathwayEntry, ProteinEntry, LiteratureRecord

def _coerce_gene_list(context: dict | None, gene_list: Optional[List[str]] = None) -> List[str]:
    if gene_list:
        return gene_list
    if not context:
        return []
    raw = context.get("gene_list")
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if raw is not None:
        return [str(raw)]
    return []

@dataclass
class TraitDiscoveryState:
    trait_name: str
    species_name: str
    instruction: str
    context: dict = field(default_factory=dict)

    gene_list: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.gene_list = _coerce_gene_list(self.context, self.gene_list)

    go_annotations: List[GOAnnotation] = field(default_factory=list)
    pathway_data: List[PathwayEntry] = field(default_factory=list)
    protein_data: List[ProteinEntry] = field(default_factory=list)
    evidence: List[LiteratureRecord] = field(default_factory=list)
    explanation: str = ""

    # §8: surfaced from the sub-orchestrators so partial-failure detail is
    # visible at the top level, not swallowed at a sub-orchestrator boundary.
    # malformed_ids / missing_genes come from Functional Evidence (Pathways /
    # Protein Data respectively); unmatched_genes comes from Gene Mapper.
    malformed_ids: List[str] = field(default_factory=list)
    missing_genes: List[str] = field(default_factory=list)
    unmatched_genes: List[str] = field(default_factory=list)

    gene_mapper_status: Optional[AgentStatus] = None
    functional_evidence_status: Optional[AgentStatus] = None
    literature_status: Optional[AgentStatus] = None

    _literature_target_agent: Optional[str] = None
    _literature_prompt: Optional[str] = None

    resolution_reasoning: Optional[str] = None

    status: Optional[AgentStatus] = None
    target_agent: Optional[str] = None
    prompt_to_target_agent: Optional[str] = None

@dataclass
class FunctionalEvidenceState:
    gene_list: List[str]
    instruction: str
    trait_name: str = ""           # NEW — threaded from parent graph
    context: dict = field(default_factory=dict)

    pathway_data: List[PathwayEntry] = field(default_factory=list)
    protein_data: List[ProteinEntry] = field(default_factory=list)
    pathways_status: Optional[AgentStatus] = None
    protein_data_status: Optional[AgentStatus] = None

    # §8: per-agent partial-failure detail — malformed_ids comes from
    # Pathways, missing_genes from Protein Data. Populated independently by
    # each worker node (no key collision) and passed through by merge_node
    # unchanged, so a failure in one never has to touch the other's data.
    malformed_ids: List[str] = field(default_factory=list)
    missing_genes: List[str] = field(default_factory=list)

    status: Optional[AgentStatus] = None