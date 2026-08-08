"""Sprint 2 mock for the Molecular Comparison subagent.

Simulates the full pipeline:
    NCBI / UniProt  →  MAFFT alignment  →  ESM-C embeddings
    →  pairwise similarity scores  →  species groups  →  similarity network

Everything is deterministic and dependency-free so the orchestrator can be
exercised end-to-end without any external service.

What this mock returns (MolecularComparisonResult):
  • alignment          — mock FASTA multiple sequence alignment
  • alignment_url      — URL to a rendered alignment viewer
  • similarity_scores  — pairwise cosine-similarity scores (SimilarityEdge list)
  • species_groups     — clusters derived from the scores (SpeciesGroup list)
  • similarity_network — adjacency-list graph {species: [{neighbour, score}]}

Swap for the real worker (Sprint 3+) without changing the orchestrator —
the run() signature and return type are stable.
"""

from __future__ import annotations

from ...schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    MolecularComparisonResult,
    SimilarityEdge,
    SpeciesGroup,
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

# Mock FASTA sequences — length 60 aa, biologically plausible cytochrome-b
# proxies (same length so MAFFT would not need to insert gaps).
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

# Pairwise cosine-similarity scores (ESMC embedding proxies).
# Values are biologically calibrated: human-chimp very close, human-fish distant.
_SCORES: dict[frozenset[str], float] = {
    frozenset({"homo sapiens",    "pan troglodytes"}): 0.98,
    frozenset({"homo sapiens",    "mus musculus"}):    0.85,
    frozenset({"homo sapiens",    "gallus gallus"}):   0.72,
    frozenset({"homo sapiens",    "danio rerio"}):     0.54,
    frozenset({"pan troglodytes", "mus musculus"}):    0.84,
    frozenset({"pan troglodytes", "gallus gallus"}):   0.71,
    frozenset({"pan troglodytes", "danio rerio"}):     0.53,
    frozenset({"mus musculus",    "gallus gallus"}):   0.68,
    frozenset({"mus musculus",    "danio rerio"}):     0.51,
    frozenset({"gallus gallus",   "danio rerio"}):     0.62,
}

# Similarity threshold above which two species are placed in the same group.
_GROUP_THRESHOLD = 0.75


class MolecularComparisonMock:
    """Deterministic stand-in for the NCBI + MAFFT + ESM-C pipeline."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, request: AgentRequest) -> AgentResult:
        species = self._resolve_species(request)

        if not species:
            return AgentResult(
                status=AgentStatus.FAILED,
                output="species_list is required for molecular comparison.",
                source_agents=["Molecular Comparison Agent"],
            )

        if len(species) < 2:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    "Molecular comparison requires at least 2 species; "
                    f"got {len(species)}: {species}."
                ),
                source_agents=["Molecular Comparison Agent"],
            )

        unknown = [s for s in species if s not in _CATALOGUE]
        if unknown:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=(
                    f"Species not in mock catalogue: {unknown}. "
                    "Available: " + ", ".join(sorted(_CATALOGUE)) + "."
                ),
                source_agents=["Molecular Comparison Agent"],
            )

        mc_result = self._build_result(species)

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=mc_result,
            similarity_scores=[
                {"species_a": e.species_a, "species_b": e.species_b, "score": e.score}
                for e in mc_result.similarity_scores
            ],
            alignment_url=mc_result.alignment_url,
            confidence=self._mean_score(mc_result.similarity_scores),
            source_agents=["Molecular Comparison Agent"],
        )

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(self, species: list[str]) -> MolecularComparisonResult:
        alignment     = self._build_alignment(species)
        alignment_url = self._alignment_url(species)
        scores        = self._compute_scores(species)
        groups        = self._compute_groups(species, scores)
        network       = self._build_network(species, scores)

        return MolecularComparisonResult(
            species_list=species,
            alignment=alignment,
            alignment_url=alignment_url,
            similarity_scores=scores,
            species_groups=groups,
            similarity_network=network,
        )

    # ------------------------------------------------------------------
    # FASTA alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _build_alignment(species: list[str]) -> str:
        """Return a mock FASTA multiple sequence alignment.

        All sequences are the same length (60 aa), so no gaps are needed —
        MAFFT would produce this trivially.  The sequences differ only in
        the last few residues, reflecting real cytochrome-b divergence.
        """
        lines: list[str] = []
        for sp in species:
            header = f">{sp.replace(' ', '_')}"
            seq    = _SEQUENCES[sp]
            lines.append(header)
            lines.append(seq)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Similarity scores
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_scores(species: list[str]) -> list[SimilarityEdge]:
        edges: list[SimilarityEdge] = []
        for i in range(len(species)):
            for j in range(i + 1, len(species)):
                a, b  = species[i], species[j]
                score = _SCORES.get(frozenset({a, b}), 0.50)
                edges.append(SimilarityEdge(species_a=a, species_b=b, score=score))
        return edges

    # ------------------------------------------------------------------
    # Species groups (simple threshold clustering)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_groups(
        species: list[str], scores: list[SimilarityEdge]
    ) -> list[SpeciesGroup]:
        """Single-linkage clustering at _GROUP_THRESHOLD.

        Produces the minimum number of groups such that every pair of
        species within a group has similarity >= _GROUP_THRESHOLD.
        Small enough for the mock catalogue; not meant to scale.
        """
        # Build adjacency set above threshold
        neighbours: dict[str, set[str]] = {s: set() for s in species}
        for edge in scores:
            if edge.score >= _GROUP_THRESHOLD:
                neighbours[edge.species_a].add(edge.species_b)
                neighbours[edge.species_b].add(edge.species_a)

        # BFS to find connected components
        visited:  set[str]        = set()
        groups:   list[list[str]] = []
        for sp in species:
            if sp in visited:
                continue
            component: list[str] = []
            queue = [sp]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                queue.extend(neighbours[node] - visited)
            groups.append(component)

        # Compute mean intra-group similarity for each component
        score_map = {
            frozenset({e.species_a, e.species_b}): e.score for e in scores
        }
        result: list[SpeciesGroup] = []
        for gid, members in enumerate(groups):
            intra: list[float] = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    key = frozenset({members[i], members[j]})
                    if key in score_map:
                        intra.append(score_map[key])
            mean = round(sum(intra) / len(intra), 4) if intra else 1.0
            result.append(SpeciesGroup(group_id=gid, species=members, mean_score=mean))

        return result

    # ------------------------------------------------------------------
    # Similarity network (adjacency list)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_network(
        species: list[str], scores: list[SimilarityEdge]
    ) -> dict[str, list[dict]]:
        """Return the full similarity graph as an adjacency list.

        Every species node lists all its neighbours with their scores.
        Keeps all edges (not just those above threshold) so consumers can
        apply their own cutoff.
        """
        network: dict[str, list[dict]] = {s: [] for s in species}
        for edge in scores:
            network[edge.species_a].append(
                {"neighbour": edge.species_b, "score": edge.score}
            )
            network[edge.species_b].append(
                {"neighbour": edge.species_a, "score": edge.score}
            )
        return network

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
    def _alignment_url(species: list[str]) -> str:
        slug = "_vs_".join(s.replace(" ", "_") for s in sorted(species))
        return f"https://evolution.umbrella.local/alignment/{slug}.html"

    @staticmethod
    def _mean_score(scores: list[SimilarityEdge]) -> float:
        if not scores:
            return 0.0
        return round(sum(e.score for e in scores) / len(scores), 4)
