"""Sprint 2 mock for the Phylogenetic Reconstruction subagent.

Simulates the full tree-building pipeline:
    MAFFT alignment  →  ModelFinder (substitution model)
    →  IQ-TREE ML tree  →  UFBoot bootstrap support

The Sprint 2 orchestrator runs this subagent in *parallel* with the Molecular
Comparison subagent (``asyncio.gather`` fan-out), so it does not consume MC's
alignment.  When ``request.context["alignment"]`` is absent it reconstructs
its own alignment from the species list (see ``alignment_source``).

What this mock returns (PhylogeneticResult):
  • newick_tree       — Newick-format tree string
  • tree_url          — URL to rendered SVG/HTML tree
  • model             — substitution model chosen by ModelFinder (mock: "LG+G4")
  • bootstrap_support — per-internal-node UFBoot % {node_label: int}
  • confidence_values — per-leaf confidence {species: float}
  • overall_confidence— mean UFBoot support (0-1 scale)

Swap for the real worker (Sprint 3+) without changing the orchestrator —
the run() signature and return type are stable.
"""

from __future__ import annotations

from ...schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    PhylogeneticResult,
)

# ---------------------------------------------------------------------------
# Fixture catalogue
# ---------------------------------------------------------------------------

_CATALOGUE: set[str] = {
    "homo sapiens",
    "pan troglodytes",
    "mus musculus",
    "gallus gallus",
    "danio rerio",
}

# Pre-built Newick trees keyed by frozenset of species.
# Topology reflects the accepted vertebrate phylogeny.
# Branch lengths are approximate (divergence in units of 10 Mya).
_FULL_NEWICK = (
    "((('homo sapiens':0.6,'pan troglodytes':0.6):8.4,'mus musculus':9.0):81.0,"
    "'gallus gallus':90.0,'danio rerio':360.0);"
)
_SUBTREES: dict[frozenset[str], str] = {
    frozenset({"homo sapiens", "pan troglodytes", "mus musculus"}): (
        "(('homo sapiens':0.6,'pan troglodytes':0.6):8.4,'mus musculus':9.0);"
    ),
    frozenset({"homo sapiens", "pan troglodytes", "mus musculus", "gallus gallus"}): (
        "((('homo sapiens':0.6,'pan troglodytes':0.6):8.4,'mus musculus':9.0)"
        ":81.0,'gallus gallus':90.0);"
    ),
    frozenset({
        "homo sapiens", "pan troglodytes", "mus musculus",
        "gallus gallus", "danio rerio",
    }): _FULL_NEWICK,
}

# Mock UFBoot support values for internal nodes.
# Keys match the internal labels one would see in a real IQ-TREE output.
_BOOTSTRAP: dict[frozenset[str], dict[str, int]] = {
    frozenset({"homo sapiens", "pan troglodytes", "mus musculus"}): {
        "node_human_chimp": 100,
        "node_mammals":     95,
    },
    frozenset({"homo sapiens", "pan troglodytes", "mus musculus", "gallus gallus"}): {
        "node_human_chimp":   100,
        "node_mammals":       95,
        "node_amniotes":      88,
    },
    frozenset({
        "homo sapiens", "pan troglodytes", "mus musculus",
        "gallus gallus", "danio rerio",
    }): {
        "node_human_chimp":   100,
        "node_mammals":       95,
        "node_amniotes":      88,
        "node_vertebrates":   82,
    },
}

# Per-leaf confidence values (0-1), derived from site-likelihood variance.
_LEAF_CONFIDENCE: dict[str, float] = {
    "homo sapiens":    0.97,
    "pan troglodytes": 0.97,
    "mus musculus":    0.93,
    "gallus gallus":   0.89,
    "danio rerio":     0.85,
}

# ModelFinder always picks LG+G4 for this mock protein dataset.
_MODEL = "LG+G4"


class PhylogeneticTreeMock:
    """Deterministic stand-in for the ModelFinder + IQ-TREE + UFBoot pipeline.

    Runs independently of the Molecular Comparison subagent (parallel
    fan-out).  If ``request.context["alignment"]`` happens to be present it
    is used; otherwise the worker falls back to the species list and the
    alignment is marked as reconstructed.
    """

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

        unknown = [s for s in species if s not in _CATALOGUE]
        if unknown:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    f"Species not in mock catalogue: {unknown}. "
                    "Available: " + ", ".join(sorted(_CATALOGUE)) + "."
                ),
                source_agents=["Phylogenetic Tree Agent"],
            )

        # Log whether the alignment was passed in from MC (expected) or
        # reconstructed from scratch (fallback).
        alignment_source = (
            "passed_from_mc"
            if (request.context or {}).get("alignment")
            else "reconstructed"
        )

        phylo_result = self._build_result(species, alignment_source)

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=phylo_result,
            newick_tree=phylo_result.newick_tree,
            tree_url=phylo_result.tree_url,
            confidence=phylo_result.overall_confidence,
            source_agents=["Phylogenetic Tree Agent"],
        )

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self, species: list[str], alignment_source: str
    ) -> PhylogeneticResult:
        key             = frozenset(species)
        newick          = _SUBTREES.get(key) or self._star_tree(species)
        tree_url        = self._tree_url(species)
        bootstrap       = _BOOTSTRAP.get(key) or self._default_bootstrap()
        conf_values     = {s: _LEAF_CONFIDENCE.get(s, 0.80) for s in species}
        overall_conf    = self._overall_confidence(bootstrap)

        return PhylogeneticResult(
            newick_tree=newick,
            tree_url=tree_url,
            model=_MODEL,
            bootstrap_support=bootstrap,
            confidence_values=conf_values,
            overall_confidence=overall_conf,
        )

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
        return [s.strip().lower() for s in raw if s.strip()]

    @staticmethod
    def _star_tree(species: list[str]) -> str:
        """Star topology fallback for subsets not explicitly pre-built."""
        leaves = ",".join(f"'{s}':1.0" for s in species)
        return f"({leaves});"

    @staticmethod
    def _default_bootstrap() -> dict[str, int]:
        """Fallback bootstrap values for subsets not in the catalogue."""
        return {"node_root": 80}

    @staticmethod
    def _overall_confidence(bootstrap: dict[str, int]) -> float:
        """Mean UFBoot support scaled to [0, 1]."""
        if not bootstrap:
            return 0.80
        return round(sum(bootstrap.values()) / len(bootstrap) / 100, 4)

    @staticmethod
    def _tree_url(species: list[str]) -> str:
        slug = "_".join(s.replace(" ", "_") for s in sorted(species))
        return f"https://evolution.umbrella.local/tree/{slug}.svg"
