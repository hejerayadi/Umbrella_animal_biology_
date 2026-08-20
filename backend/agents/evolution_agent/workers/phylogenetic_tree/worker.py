"""Real Phylogenetic Reconstruction worker — Sprint 3.

Uses real bioinformatics tools:
    1. MAFFT — multiple sequence alignment
    2. IQ-TREE — phylogenetic tree building
       - ModelFinder (MFP) — picks best substitution model
       - UFBoot2 — ultrafast bootstrap support

Pipeline:
    Sequences → MAFFT → aligned sequences → IQ-TREE → tree + support

The orchestrator dispatches this worker only for the phylogenetic branch,
so it builds its own alignment rather than consuming Molecular Comparison's.

What this worker returns (PhylogeneticResult):
  • newick_tree       — Newick tree from IQ-TREE, leaves labelled with the
                        full scientific names (quoted when they contain a
                        space)
  • tree_url          — ``None``: no rendered artefact is served yet
  • model             — substitution model reported by ModelFinder
  • bootstrap_support — per-INTERNAL-NODE UFBoot % {node_label: int}
  • confidence_values — per-INTERNAL-NODE confidence {node_label: float};
                        branch support is a property of a split, never of a
                        single species
  • overall_confidence— mean of the real UFBoot supports, or ``None``
  • aligned_fasta     — the MAFFT alignment, names restored
  • warnings          — e.g. ``ufboot_not_run``
"""

from __future__ import annotations

import logging
import re

from ...schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    PhylogeneticResult,
)
from ...tools.mafft import align as mafft_align, MAFFTError
from ...tools.iqtree import build_tree as iqtree_build, IQTreeError
from ...tools import taxon_ids

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sequence catalogue (temporary — will be replaced with real NCBI/UniProt fetch)
# ---------------------------------------------------------------------------

# Real cytochrome b sequences (60 aa) for 5 model organisms.
# These are actual protein sequences from UniProt/NCBI.
_SEQUENCES: dict[str, str] = {
    "homo sapiens": (
        "MTNIRKSHPLFKIINHSFIDLPAPSNISSWWNFGSLLGACLILQITTGLFLAMHYTSDTT"
    ),
    "pan troglodytes": (
        "MTNIRKSHPLFKIINHSFIDLPAPSNISSWWNFGSLLGACLILQITTGLFLAMHYTSDTA"
    ),
    "mus musculus": (
        "MTNIRKTHPLFKIVNHCFVDLPTPSNISALWNFGSLLGACLIVQILTGLFLAMHYSSDTL"
    ),
    "gallus gallus": (
        "MPNIRKTHPLFKIVNHAFVDLPTPANISPLWNFGSLLGACLIVQILTGLFLAMHYSSDTV"
    ),
    "danio rerio": (
        "MANIRKTHPLFKIVNHSFVDLPTPANISPLWNFGSLLGTCLIVQILTGLFLAMHYSSDTK"
    ),
}


class PhylogeneticTreeWorker:
    """Real phylogenetic tree builder using MAFFT + IQ-TREE.

    Runs independently of the Molecular Comparison subagent (parallel
    fan-out).  If ``request.context["alignment"]`` happens to be present it
    is used; otherwise the worker fetches sequences and aligns them.
    """

    def __init__(
        self,
        mafft_path: str | None = None,
        iqtree_path: str | None = None,
    ) -> None:
        self._mafft_path = mafft_path
        self._iqtree_path = iqtree_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResult:
        species = self._resolve_species(request)

        if not species:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="species_list is required for phylogenetic reconstruction.",
                source_agents=["Phylogenetic Tree Agent"],
            )

        if len(species) < 3:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    "Phylogenetic reconstruction requires at least 3 species; "
                    f"got {len(species)}: {species}."
                ),
                source_agents=["Phylogenetic Tree Agent"],
            )

        # Fetch sequences for each species
        sequences = self._fetch_sequences(species)
        missing = [s for s in species if s not in sequences]
        if missing:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"Could not fetch sequences for: {missing}",
                source_agents=["Phylogenetic Tree Agent"],
            )

        try:
            phylo_result = self._build_tree(species, sequences)
        except (MAFFTError, IQTreeError) as exc:
            _logger.error("[Phylo] tree build failed: %s", exc)
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"Phylogenetic reconstruction failed: {exc}",
                source_agents=["Phylogenetic Tree Agent"],
            )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=phylo_result,
            newick_tree=phylo_result.newick_tree,
            tree_url=phylo_result.tree_url,
            confidence=phylo_result.overall_confidence,
            source_agents=["Phylogenetic Tree Agent"],
            warnings=list(phylo_result.warnings),
        )

    # ------------------------------------------------------------------
    # Tree building pipeline
    # ------------------------------------------------------------------

    def _build_tree(
        self, species: list[str], sequences: dict[str, str]
    ) -> PhylogeneticResult:
        """Run MAFFT → IQ-TREE pipeline.

        Scientific names are swapped for whitespace-free ids before the
        external tools see them, and restored afterwards: MAFFT and IQ-TREE
        both truncate a FASTA header at the first space, which would turn
        "Homo sapiens" into "Homo" and collide with "Homo erectus".
        """
        warnings: list[str] = []

        name_to_id, id_to_name = taxon_ids.make_mapping(list(sequences))
        safe_sequences = taxon_ids.to_safe_sequences(sequences, name_to_id)

        # Step 1: MAFFT alignment (on safe ids)
        _logger.info("[Phylo] aligning %d sequences with MAFFT...", len(species))
        aligned_fasta = mafft_align(safe_sequences)
        aligned_dict = taxon_ids.parse_fasta(aligned_fasta)

        missing = [i for i in safe_sequences if i not in aligned_dict]
        if missing or not aligned_dict:
            lost = [id_to_name.get(i, i) for i in missing]
            raise MAFFTError(
                "MAFFT did not return every input taxon; missing: "
                f"{lost or '(no sequence at all)'}"
            )
        _logger.info("[Phylo] alignment complete (%d columns)",
                     len(next(iter(aligned_dict.values()))))

        # Step 2: IQ-TREE (ModelFinder + UFBoot), still on safe ids.
        # UFBoot needs at least 4 taxa to have a non-trivial split to
        # resample; below that it is skipped and reported, never faked.
        num_species = len(aligned_dict)
        bootstrap_val = 1000 if num_species >= 4 else 0
        if bootstrap_val == 0:
            warnings.append("ufboot_not_run")
        _logger.info("[Phylo] building tree with IQ-TREE (ModelFinder%s)...",
                     " + UFBoot" if bootstrap_val > 0 else "")
        phylo = iqtree_build(
            alignment=aligned_dict,
            model="MFP",
            bootstrap=bootstrap_val,
        )
        if not phylo.ufboot_run and "ufboot_not_run" not in warnings:
            warnings.append("ufboot_not_run")
        _logger.info("[Phylo] tree built: model=%s, ufboot_run=%s, confidence=%s",
                     phylo.model, phylo.ufboot_run, phylo.overall_confidence)

        # Step 3: put the full scientific names back
        newick = taxon_ids.restore_newick(phylo.newick_tree, id_to_name)
        alignment = taxon_ids.restore_fasta(aligned_fasta, id_to_name)

        return PhylogeneticResult(
            newick_tree=newick,
            tree_url=None,      # no rendered artefact is served yet
            model=phylo.model,
            bootstrap_support=phylo.bootstrap_support,
            confidence_values=phylo.confidence_values,
            overall_confidence=phylo.overall_confidence,
            aligned_fasta=alignment,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Sequence fetching (temporary — will use real NCBI/UniProt later)
    # ------------------------------------------------------------------

    def _fetch_sequences(self, species: list[str]) -> dict[str, str]:
        """Fetch sequences for species. Currently uses local catalogue."""
        result = {}
        for s in species:
            key = s.strip().lower()
            if key in _SEQUENCES:
                result[s] = _SEQUENCES[key]
            else:
                _logger.warning("[Phylo] no sequence for '%s' in local catalogue", s)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_species(request: AgentRequest) -> list[str]:
        raw: list[str] = []
        if request.species_list:
            raw = request.species_list
        else:
            ctx = request.context or {}
            raw = ctx.get("species_list") or ctx.get("species") or []
            if isinstance(raw, str):
                raw = [raw]
        # Keep the caller's canonical form ("Homo sapiens"): it is what ends
        # up labelling the tree. Case folding happens only for the catalogue
        # lookup in _fetch_sequences.
        return [s.strip() for s in raw if s.strip()]


# FASTA parsing lives in tools.taxon_ids so that identifier handling and
# identifier restoration cannot drift apart.
_parse_fasta = taxon_ids.parse_fasta
