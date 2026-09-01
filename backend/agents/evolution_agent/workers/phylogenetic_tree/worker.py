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
from typing import Callable

from langsmith import traceable

from ...schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    PhylogeneticResult,
)
from ...tools.mafft import align as mafft_align, MAFFTError
from ...tools.iqtree import build_tree as iqtree_build, IQTreeError
from ...tools import taxon_ids
from ...tools.sequences import (
    DEFAULT_GENE_CANDIDATES,
    fetch_uniprot_sequence,
    parse_fasta_inputs,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Offline fallback catalogue
# ---------------------------------------------------------------------------

# Cytochrome b fragments (60 aa) for 5 model organisms, kept ONLY as an
# offline fallback for when UniProt is unreachable, and as a fixture the
# tests can rely on without network access. The primary path is the same
# UniProt fetch the Molecular Comparison worker uses, so that a
# full_analysis builds its tree and its similarity network from the same
# sequences instead of two unrelated datasets.
#
# A tree built from these 60 columns is far weaker than one built from the
# full-length protein: any run that falls back here reports
# ``offline_sequence_fallback`` in its warnings.
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

    Runs concurrently with the Molecular Comparison subagent and builds
    its own alignment: MAFFT gap characters would corrupt ESM-2 embeddings,
    so the two branches share sequences but never share an alignment.
    """

    def __init__(
        self,
        mafft_path: str | None = None,
        iqtree_path: str | None = None,
        fetch_fn: Callable[[str, list[str]], str] | None = None,
    ) -> None:
        self._mafft_path = mafft_path
        self._iqtree_path = iqtree_path
        # Same fetcher as Molecular Comparison. Injectable so tests run
        # without network access.
        self._fetch = fetch_fn or fetch_uniprot_sequence

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @traceable(name="Phylogenetic Tree Agent", run_type="chain")
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

        # Same gene selection rule as Molecular Comparison, so a
        # full_analysis aligns and embeds the same protein.
        gene_candidates = (
            [request.target_gene_or_protein] if request.target_gene_or_protein
            else DEFAULT_GENE_CANDIDATES
        )
        presupplied = (
            parse_fasta_inputs(request.protein_inputs, species)
            if request.protein_inputs else {}
        )

        sequences, fetch_warnings = self._fetch_sequences(
            species, gene_candidates, presupplied
        )
        missing = [s for s in species if s not in sequences]
        if missing:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    "Could not obtain sequences for: "
                    f"{missing}. Tried UniProt (genes={gene_candidates}) "
                    "and the offline catalogue."
                ),
                source_agents=["Phylogenetic Tree Agent"],
            )

        try:
            phylo_result = self._build_tree(species, sequences, fetch_warnings)
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

    @traceable(name="MAFFT -> IQ-TREE pipeline", run_type="chain")
    def _build_tree(
        self,
        species: list[str],
        sequences: dict[str, str],
        extra_warnings: list[str] | None = None,
    ) -> PhylogeneticResult:
        """Run MAFFT → IQ-TREE pipeline.

        Scientific names are swapped for whitespace-free ids before the
        external tools see them, and restored afterwards: MAFFT and IQ-TREE
        both truncate a FASTA header at the first space, which would turn
        "Homo sapiens" into "Homo" and collide with "Homo erectus".
        """
        warnings: list[str] = list(extra_warnings or [])

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

    def _fetch_sequences(
        self,
        species: list[str],
        gene_candidates: list[str],
        presupplied: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], list[str]]:
        """Resolve one sequence per species.

        Order of preference, per species:
          1. a sequence supplied on the request (``protein_inputs``);
          2. a live UniProt fetch — the same call the Molecular Comparison
             worker makes, so both branches see identical data;
          3. the offline 60-aa catalogue.

        Returns ``(sequences, warnings)``. A species that none of the three
        can satisfy is simply absent from ``sequences``; the caller turns
        that into a FAILED result naming it.
        """
        resolved: dict[str, str] = dict(presupplied or {})
        warnings: list[str] = []
        fell_back: list[str] = []

        for s in species:
            if s in resolved:
                continue
            try:
                resolved[s] = self._fetch(s, gene_candidates)
                continue
            except Exception as exc:
                _logger.info("[Phylo] UniProt fetch failed for %r (%s)", s, exc)

            key = s.strip().lower()
            if key in _SEQUENCES:
                resolved[s] = _SEQUENCES[key]
                fell_back.append(s)
            else:
                _logger.warning(
                    "[Phylo] no sequence for '%s' from UniProt or the offline catalogue", s
                )

        if fell_back:
            warnings.append(
                "offline_sequence_fallback: used the 60-aa offline catalogue for "
                + ", ".join(fell_back)
            )
        return resolved, warnings

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
