"""Shared data contracts for the Evolution Agent domain.

Sprint 2 introduces three rich result types that flow through the
sequential pipeline:

    MolecularComparisonResult   — output of the Molecular Comparison subagent
    PhylogeneticResult          — output of the Phylogenetic Reconstruction subagent
    EvolutionAnalysisResult     — the final assembled payload returned to the
                                  Global Orchestrator

The platform-wide contract (AgentRequest / AgentResult / AgentStatus)
is kept unchanged so the HTTP boundary and every existing caller continues
to work without modification.  EvolutionAnalysisResult is wrapped inside
AgentResult.output by the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Platform-wide status & feature enum
# ---------------------------------------------------------------------------


class AgentStatus(Enum):
    """Universal execution status used across every agent in the platform."""

    COMPLETED   = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE    = "continue"
    FAILED      = "failed"


class EvolutionaryFeature(str, Enum):
    """The skills the Evolution Agent can route to.

    Sprint 2: the full run always executes BOTH in sequence
    (molecular_comparison → phylogenetic_tree).  The single-feature
    values are kept so the router and adapter stay backwards-compatible.
    """

    MOLECULAR_COMPARISON = "molecular_comparison"
    PHYLOGENETIC_TREE    = "phylogenetic_tree"


# ---------------------------------------------------------------------------
# Platform-wide request / result
# ---------------------------------------------------------------------------


@dataclass
class AgentRequest:
    """Standard input contract for any agent, extended for evolution.

    ``instruction`` and ``context`` are the minimal platform contract.
    The remaining fields are optional evolution hints — when absent the
    orchestrator infers them from ``context`` or falls back to defaults.
    """

    instruction: str
    context: dict[str, Any] = field(default_factory=dict)

    # Which skill to run (set by adapter; "full_analysis" triggers the
    # full sequential pipeline).
    feature: str | None = None

    # Molecular comparison and phylogenetic tree both work on a list of
    # species (at least 2).
    species_list: list[str] = field(default_factory=list)

    # Optional anchor species for tree rooting.
    reference_species: str | None = None

    session_id: str | None = None


@dataclass
class AgentResult:
    """Standard output contract, extended with evolution payload fields.

    ``output`` carries the EvolutionAnalysisResult (or an error string).
    The domain-specific convenience fields let downstream agents read the
    most important results without unpacking ``output``.
    """

    status: AgentStatus

    # Escalation fields — populated when status == NEEDS_AGENT
    target_agent: str | None = None
    prompt_to_target_agent: str | None = None

    # Free-form payload — EvolutionAnalysisResult on success, str on failure
    output: Any | None = None

    # Convenience fields (mirrors what's inside output.phylogenetic)
    newick_tree: str | None = None
    tree_url:    str | None = None

    # Convenience fields (mirrors what's inside output.molecular)
    similarity_scores: list[dict[str, Any]] | None = None
    alignment_url:     str | None = None

    # Provenance / trust
    confidence:    float | None = None
    source_agents: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sprint 2 rich result types
# ---------------------------------------------------------------------------


@dataclass
class SimilarityEdge:
    """A single edge in the similarity network."""

    species_a: str
    species_b: str
    score:     float   # cosine similarity of ESMC embeddings, in [0, 1]


@dataclass
class SpeciesGroup:
    """A cluster of evolutionarily close species."""

    group_id:   int
    species:    list[str]
    mean_score: float   # average intra-group similarity


@dataclass
class MolecularComparisonResult:
    """Everything the Molecular Comparison subagent produces.

    Tools mocked in Sprint 2: NCBI, UniProt, MAFFT, ESM-C, NetworkX.

    Fields
    ------
    species_list    : canonical scientific names that were compared
    alignment       : mock FASTA-format multiple sequence alignment
    alignment_url   : URL to a rendered HTML alignment viewer
    similarity_scores : pairwise cosine-similarity scores (ESMC embeddings)
    species_groups  : clusters of evolutionarily similar species
    similarity_network : adjacency-list representation of the similarity graph
                         {species_name: [{"neighbour": str, "score": float}]}
    """

    species_list:       list[str]
    alignment:          str                      # FASTA text
    alignment_url:      str
    similarity_scores:  list[SimilarityEdge]
    species_groups:     list[SpeciesGroup]
    similarity_network: dict[str, list[dict[str, Any]]]


@dataclass
class PhylogeneticResult:
    """Everything the Phylogenetic Reconstruction subagent produces.

    Tools mocked in Sprint 2: IQ-TREE, ModelFinder, UFBoot.

    Fields
    ------
    newick_tree       : Newick-format tree string
    tree_url          : URL to rendered SVG/HTML tree
    model             : substitution model selected by ModelFinder (mock)
    bootstrap_support : per-node bootstrap values {node_label: int (0-100)}
    confidence_values : per-leaf confidence {species: float in [0,1]}
    overall_confidence: mean UFBoot support across all internal nodes
    """

    newick_tree:        str
    tree_url:           str
    model:              str                      # e.g. "LG+G4"
    bootstrap_support:  dict[str, int]           # node label → UFBoot %
    confidence_values:  dict[str, float]         # leaf → confidence
    overall_confidence: float


@dataclass
class EvolutionAnalysisResult:
    """The final assembled output of the full pipeline.

    Produced by the orchestrator's assemble node after both subagents
    have run.  This is what gets wrapped inside AgentResult.output and
    then inside {"evolution_report": ...} by the adapter.
    """

    species_list: list[str]
    molecular:    MolecularComparisonResult
    phylogenetic: PhylogeneticResult

    # Flat convenience fields for quick access by downstream agents
    overall_confidence: float           # mean of MC mean-score + phylo confidence
    source_agents:      list[str] = field(default_factory=list)
