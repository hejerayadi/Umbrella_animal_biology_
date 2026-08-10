from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from backend.agents.Protein_visualization.app.domain.enums import PreferredSource, StructureSource


@dataclass(frozen=True, slots=True)
class SpeciesRef:
    scientific_name: str
    taxon_id: int


@dataclass(frozen=True, slots=True)
class ProteinStructureRequest:
    task_id: UUID
    trace_id: UUID
    resolved_gene_id: str
    species: SpeciesRef
    uniprot_accession: str | None = None
    protein_sequence: str | None = None
    requested_regions: tuple[str, ...] = ()
    residue_position: int | None = None
    mutation: str | None = None
    preferred_source: PreferredSource = PreferredSource.auto
    include_explanation: bool = True


@dataclass(frozen=True, slots=True)
class ResolvedProtein:
    uniprot_accession: str
    gene_symbol: str
    scientific_name: str
    taxon_id: int
    protein_name: str | None = None
    sequence: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    provider: str
    external_id: str
    retrieved_at: str
    source_url: str | None = None
    checksum: str | None = None


@dataclass(frozen=True, slots=True)
class StructureCandidate:
    source: StructureSource
    external_id: str
    structure_type: Literal["EXPERIMENTAL", "PREDICTED"]
    chain_id: str | None
    experimental_method: str | None
    resolution_angstrom: float | None
    sequence_coverage: float
    mean_plddt: float | None
    file_format: Literal["MMCIF", "PDB"]
    file_url: str
    selection_score: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Annotation:
    source: str
    accession: str
    kind: str
    label: str
    start: int | None = None
    end: int | None = None


@dataclass(frozen=True, slots=True)
class ResidueMapping:
    uniprot_accession: str
    uniprot_position: int
    pdb_id: str
    chain_id: str
    pdb_residue_number: str | None
    is_observed: bool
    mapping_source: Literal["SIFTS"] = "SIFTS"


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidencePack:
    evidence: tuple[EvidenceRef, ...] = ()
    facts: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VisualizationSpec:
    viewer: Literal["molstar"]
    structure_id: str
    structure_url: str
    representation: str = "cartoon"
    selections: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class Explanation:
    summary: str
    limitations: tuple[str, ...] = ()
    generated: bool = False
    """True when a language model wrote the summary, False for the deterministic fallback."""


@dataclass(frozen=True, slots=True)
class CriticReport:
    verdict: Literal["ACCEPT", "REVISE", "ABSTAIN"]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """One Azure OpenAI call's cost and latency, for the node that made it.

    Token counts come from the provider's own response (`AIMessage.usage_metadata`
    via `include_raw=True`), never estimated locally - a local tokenizer would
    silently drift from whatever model the deployment actually routes to.
    """

    node: str
    model: str
    duration_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
