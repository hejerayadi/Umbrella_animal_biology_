"""Sprint 2 mock for the Molecular Comparison subagent.

Simulates the pipeline that will become real in Sprint 3+:
    NCBI / UniProt (raw sequences)  →  ESM-2 embeddings  →  pairwise
    similarity scores  →  species groups  →  similarity network (NetworkX)

No alignment step runs in this sub-agent. MAFFT is exclusive to the
Phylogenetic Reconstruction sub-agent — alignment gap characters would
corrupt ESM-2 embeddings. Both sub-agents receive the same raw, unaligned
sequences as parallel siblings.

Everything here is deterministic and dependency-free so the orchestrator
can be exercised end-to-end without any external service.

What this mock returns (MolecularComparisonResult):
  • similarity_scores  — pairwise cosine-similarity scores (SimilarityEdge list)
  • species_groups     — clusters derived from the scores (SpeciesGroup list)
  • similarity_network — nx.node_link_data(graph, edges="edges") output

Swap for the real worker (Sprint 3+) without changing the orchestrator —
the run() signature and return type are stable.
"""

from __future__ import annotations

import networkx as nx

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

# Raw (unaligned) mock sequences — cytochrome-b proxies. No gap characters;
# this is what the real fetch step would hand to ESM-2 directly.
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

# Pairwise cosine-similarity scores (ESM-2 embedding proxies, hand-set for
# the mock). Human-chimp very close, human-fish distant.
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
_GROUP_THRESHOLD = 0.6


class MolecularComparisonMock:
    """Deterministic stand-in for the NCBI/UniProt + ESM-2 pipeline."""

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
            confidence=self._mean_score(mc_result.similarity_scores),
            source_agents=["Molecular Comparison Agent"],
        )

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(self, species: list[str]) -> MolecularComparisonResult:
        scores  = self._compute_scores(species)
        groups  = self._compute_groups(species, scores)
        network = self._build_network(species, scores)

        # The mock stands in for the real contract, so it must populate
        # `confidence` too -- leaving it None made every orchestrator-level
        # confidence read None and hid the real aggregation logic.
        # Same separation measure the real agent uses.
        from .logic import compute_confidence

        return MolecularComparisonResult(
            species_list=species,
            similarity_scores=scores,
            species_groups=groups,
            similarity_network=network,
            confidence=compute_confidence(scores, groups),
        )

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
    # Species groups (connected-components clustering at threshold)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_groups(
        species: list[str], scores: list[SimilarityEdge]
    ) -> list[SpeciesGroup]:
        neighbours: dict[str, set[str]] = {s: set() for s in species}
        for edge in scores:
            if edge.score >= _GROUP_THRESHOLD:
                neighbours[edge.species_a].add(edge.species_b)
                neighbours[edge.species_b].add(edge.species_a)

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
    # Similarity network — built as a real NetworkX graph, serialized via
    # node_link_data so the shape matches MolecularComparisonResult's
    # documented contract exactly.
    # ------------------------------------------------------------------

    @staticmethod
    def _build_network(
        species: list[str], scores: list[SimilarityEdge]
    ) -> dict:
        graph = nx.Graph()
        graph.add_nodes_from(species)
        for edge in scores:
            graph.add_edge(edge.species_a, edge.species_b, score=edge.score)
        return nx.node_link_data(graph, edges="edges")

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
    def _mean_score(scores: list[SimilarityEdge]) -> float:
        if not scores:
            return 0.0
        return round(sum(e.score for e in scores) / len(scores), 4)