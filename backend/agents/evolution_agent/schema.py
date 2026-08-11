"""Shared data contracts for the Evolution Agent domain.

Two layers of contracts live here:

  PUBLIC CONTRACT (Swagger-visible, presentation-aligned)
  -------------------------------------------------------
  EvolutionInput   — exactly what the presentation slide defines
  EvolutionOutput  — exactly what the presentation slide defines

  INTERNAL PIPELINE TYPES (Sprint 2)
  -----------------------------------
  MolecularComparisonResult  — output of Subagent 1
  PhylogeneticResult         — output of Subagent 2
  EvolutionAnalysisResult    — assembled result handed to the adapter

  PLATFORM CONTRACT (unchanged, compatible with Global Orchestrator)
  ------------------------------------------------------------------
  AgentRequest / AgentResult / AgentStatus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Platform-wide status & feature enum
# ---------------------------------------------------------------------------

class AgentStatus(Enum):
    COMPLETED   = "completed"
    NEEDS_AGENT = "needs_agent"
    CONTINUE    = "continue"
    FAILED      = "failed"


class EvolutionaryFeature(str, Enum):
    MOLECULAR_COMPARISON = "molecular_comparison"
    PHYLOGENETIC_TREE    = "phylogenetic_tree"


# ---------------------------------------------------------------------------
# PUBLIC CONTRACT — exactly as defined in the presentation
# ---------------------------------------------------------------------------

@dataclass
class EvolutionInput:
    """Public input contract shown in the presentation.

    The /execute endpoint accepts this shape from Swagger / the Global
    Orchestrator.

    Fields
    ------
    species                : list of species names to compare (min 2).
                             Common names are accepted ("human", "chimp").
    question               : the free-text question driving the analysis.
    target_gene_or_protein : optional gene or protein to focus on
                             (e.g. "cytochrome b", "hemoglobin").
    protein_inputs         : optional pre-fetched protein sequences in
                             FASTA format; skips NCBI/UniProt fetch when set.
    outgroup               : optional outgroup species for tree rooting.
    """

    species:                list[str]
    question:               str
    target_gene_or_protein: Optional[str]       = None
    protein_inputs:         Optional[list[str]] = None
    outgroup:               Optional[str]        = None


@dataclass
class EvolutionOutput:
    """Public output contract shown in the presentation.

    The /execute endpoint always returns this shape.

    Fields
    ------
    status              : "completed" | "failed"
    species             : canonical scientific names that were analysed
    molecular_comparison: plain-language summary of the MC step
    closest_species     : the two most similar species in the comparison
    species_groups      : clusters of evolutionarily close species
                          [[species_a, species_b], [species_c], ...]
    similarity_network  : JSON string of the full adjacency-list graph
    evolutionary_tree   : Newick tree string
    explanation         : one-sentence plain-language summary of the result
    """

    status:               str
    species:              list[str]
    molecular_comparison: Optional[str]             = None
    closest_species:      Optional[list[str]]        = None
    species_groups:       Optional[list[list[str]]]  = None
    similarity_network:   Optional[str]              = None
    evolutionary_tree:    Optional[str]              = None
    explanation:          Optional[str]              = None


# ---------------------------------------------------------------------------
# PLATFORM CONTRACT — minimal change, keeps Global Orchestrator compatible
# ---------------------------------------------------------------------------

@dataclass
class AgentRequest:
    """Standard platform input contract, extended with all EvolutionInput fields.

    The /execute endpoint converts EvolutionInput → AgentRequest before
    handing off to the orchestrator.
    """

    instruction: str
    context: dict[str, Any] = field(default_factory=dict)

    feature:          str | None  = None
    species_list:     list[str]   = field(default_factory=list)
    reference_species: str | None = None   # maps to EvolutionInput.outgroup
    session_id:       str | None  = None

    # EvolutionInput extras — carried through to workers
    target_gene_or_protein: str | None       = None
    protein_inputs:         list[str] | None = None


@dataclass
class AgentResult:
    """Standard platform output contract."""

    status: AgentStatus

    target_agent:            str | None = None
    prompt_to_target_agent:  str | None = None
    output:                  Any | None = None

    newick_tree:       str | None                 = None
    tree_url:          str | None                 = None
    similarity_scores: list[dict[str, Any]] | None = None
    alignment_url:     str | None                 = None

    confidence:    float | None  = None
    source_agents: list[str]     = field(default_factory=list)


# ---------------------------------------------------------------------------
# INTERNAL PIPELINE TYPES
# ---------------------------------------------------------------------------

@dataclass
class SimilarityEdge:
    species_a: str
    species_b: str
    score:     float


@dataclass
class SpeciesGroup:
    group_id:   int
    species:    list[str]
    mean_score: float


@dataclass
class MolecularComparisonResult:
    """Output of Subagent 1 (Molecular Comparison).

    Tools mocked in Sprint 2: NCBI, UniProt, MAFFT, ESM-C, NetworkX.
    """

    species_list:       list[str]
    alignment:          str
    alignment_url:      str
    similarity_scores:  list[SimilarityEdge]
    species_groups:     list[SpeciesGroup]
    similarity_network: dict[str, list[dict[str, Any]]]


@dataclass
class PhylogeneticResult:
    """Output of Subagent 2 (Phylogenetic Reconstruction).

    Tools mocked in Sprint 2: IQ-TREE, ModelFinder, UFBoot.
    """

    newick_tree:        str
    tree_url:           str
    model:              str
    bootstrap_support:  dict[str, int]
    confidence_values:  dict[str, float]
    overall_confidence: float


@dataclass
class EvolutionAnalysisResult:
    """Final assembled result — input to the adapter's to_platform_result()."""

    species_list:       list[str]
    molecular:          MolecularComparisonResult
    phylogenetic:       PhylogeneticResult
    overall_confidence: float
    source_agents:      list[str] = field(default_factory=list)
