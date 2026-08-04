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
    """Top-level graph state. Mirrors TraitDiscoveryOutput field-for-field so the final
    graph state IS the output — no separate assembly step needed at the end."""

    # ---- inputs, set once at entry ----
    trait_name: str
    species_name: str
    instruction: str
    context: dict = field(default_factory=dict)

    # ---- populated as nodes run ----
    gene_list: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.gene_list = _coerce_gene_list(self.context, self.gene_list)

    go_annotations: List[GOAnnotation] = field(default_factory=list)
    pathway_data: List[PathwayEntry] = field(default_factory=list)
    protein_data: List[ProteinEntry] = field(default_factory=list)
    evidence: List[LiteratureRecord] = field(default_factory=list)
    explanation: str = ""

    # ---- per-step status bookkeeping, so conditional edges have something to read ----
    gene_mapper_status: Optional[AgentStatus] = None
    functional_evidence_status: Optional[AgentStatus] = None
    literature_status: Optional[AgentStatus] = None

    # ---- scratch fields, private by convention (underscore prefix), not part of the
    # public output shape until an escalation node promotes them ----
    _literature_target_agent: Optional[str] = None
    _literature_prompt: Optional[str] = None

    # ---- final output fields, same names/shape as TraitDiscoveryOutput ----
    status: Optional[AgentStatus] = None
    target_agent: Optional[str] = None
    prompt_to_target_agent: Optional[str] = None


@dataclass
class FunctionalEvidenceState:
    """Subgraph state, kept independent of TraitDiscoveryState on purpose — the subgraph
    should only know about gene_list/instruction/context in, and
    pathway_data/protein_data/status out, exactly like FunctionalEvidenceInput/Output did
    in Task 1. Reusing the parent state here would silently couple the subgraph to it."""

    gene_list: List[str]
    instruction: str
    context: dict = field(default_factory=dict)

    pathway_data: List[PathwayEntry] = field(default_factory=list)
    protein_data: List[ProteinEntry] = field(default_factory=list)
    pathways_status: Optional[AgentStatus] = None
    protein_data_status: Optional[AgentStatus] = None

    status: Optional[AgentStatus] = None