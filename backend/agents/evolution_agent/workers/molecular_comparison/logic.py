"""Real implementation of the Molecular Comparison sub-agent (Sprint 3+).

Pipeline: UniProt (raw sequence fetch) -> ESM-2 embeddings (mean-pooled) ->
batch mean-centered cosine similarity -> relative-threshold connected-
components clustering -> NetworkX similarity graph.

Centering matters: raw ESM-2 embeddings are anisotropic (they cluster
tightly around one dominant direction), which compresses raw cosine
similarity into a narrow, low-signal band regardless of true sequence
divergence -- see scripts/validate_esm2.py, which first demonstrated this
and the batch-mean-centering fix.

No alignment step runs here -- MAFFT is exclusive to Phylogenetic
Reconstruction. Fetch/embed functions are injectable so tests never need
network access or a loaded model.
"""

from __future__ import annotations

import statistics
from typing import Callable, Optional

import networkx as nx
import requests
import torch

from ...schema import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    MolecularComparisonResult,
    SimilarityEdge,
    SpeciesGroup,
)

# ---------------------------------------------------------------------------
# Sequence fetching (UniProt)
# ---------------------------------------------------------------------------

DEFAULT_GENE_CANDIDATES = ["CYTB", "COX1"]


def fetch_uniprot_sequence(species: str, gene_candidates: list[str]) -> str:
    """Fetch a reviewed protein sequence for `species` by organism name.

    Tries each gene symbol in order until one resolves. Raises ValueError
    if none do -- callers turn that into a FAILED AgentResult, they never
    let it propagate as an unhandled exception.
    """
    for gene in gene_candidates:
        params = {
            "query": f'organism_name:"{species}" AND gene:{gene} AND reviewed:true',
            "format": "fasta",
            "size": 1,
        }
        resp = requests.get(
            "https://rest.uniprot.org/uniprotkb/search", params=params, timeout=30
        )
        resp.raise_for_status()
        fasta = resp.text.strip()
        if fasta:
            lines = fasta.split("\n")
            return "".join(l for l in lines if not l.startswith(">"))
    raise ValueError(
        f"No reviewed UniProt sequence for species='{species}', "
        f"tried genes={gene_candidates}"
    )


def parse_fasta_inputs(protein_inputs: list[str], species_list: list[str]) -> dict[str, str]:
    """Match pre-supplied FASTA strings to species by header substring."""
    resolved: dict[str, str] = {}
    for fasta in protein_inputs:
        lines = [l for l in fasta.strip().split("\n") if l]
        if not lines or not lines[0].startswith(">"):
            continue
        header = lines[0][1:].lower()
        seq = "".join(l for l in lines[1:] if not l.startswith(">"))
        for sp in species_list:
            if sp.lower().replace(" ", "_") in header or sp.lower() in header:
                resolved[sp] = seq
                break
    return resolved


# ---------------------------------------------------------------------------
# ESM-2 embeddings (lazy singleton -- loaded once per process, not per call)
# ---------------------------------------------------------------------------

_MODEL = None
_ALPHABET = None
_BATCH_CONVERTER = None


def _get_model():
    global _MODEL, _ALPHABET, _BATCH_CONVERTER
    if _MODEL is None:
        import esm
        _MODEL, _ALPHABET = esm.pretrained.esm2_t12_35M_UR50D()
        _MODEL.eval()
        _BATCH_CONVERTER = _ALPHABET.get_batch_converter()
    return _MODEL, _BATCH_CONVERTER


def embed_sequences(sequences: dict[str, str]) -> dict[str, torch.Tensor]:
    """Mean-pooled ESM-2 embedding per species. Raw, uncentered."""
    model, batch_converter = _get_model()
    data = list(sequences.items())
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12], return_contacts=False)
    token_reps = results["representations"][12]

    pooled: dict[str, torch.Tensor] = {}
    for i, (species, seq) in enumerate(data):
        tokens_len = len(seq) + 2  # BOS/EOS
        pooled[species] = token_reps[i, 1 : tokens_len - 1].mean(0)
    return pooled


# ---------------------------------------------------------------------------
# Similarity, grouping, network
# ---------------------------------------------------------------------------

def compute_similarity_scores(embeddings: dict[str, torch.Tensor]) -> list[SimilarityEdge]:
    species = list(embeddings.keys())

    # Subtract the batch mean before computing cosine similarity -- raw
    # ESM-2 embeddings share a dominant direction (anisotropy) that makes
    # every pair look artificially similar otherwise. Centered relative to
    # this request's own species batch, matching scripts/validate_esm2.py.
    matrix = torch.stack([embeddings[s] for s in species])
    centered = matrix - matrix.mean(dim=0)

    edges: list[SimilarityEdge] = []
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            a, b = species[i], species[j]
            sim = torch.nn.functional.cosine_similarity(
                centered[i].unsqueeze(0), centered[j].unsqueeze(0)
            ).item()
            edges.append(SimilarityEdge(species_a=a, species_b=b, score=round(sim, 4)))
    return edges


def compute_species_groups(
    species: list[str], scores: list[SimilarityEdge]
) -> list[SpeciesGroup]:
    """Connected-components clustering at a threshold relative to this
    request's own score distribution -- not a fixed constant. Real ESM-2
    cosine scores live in a narrow, gene-dependent band, so an absolute
    cutoff calibrated on one gene won't generalize to another.
    """
    if len(scores) < 2:
        threshold = 0.0
    else:
        vals = [e.score for e in scores]
        threshold = statistics.mean(vals) + 0.5 * statistics.pstdev(vals)

    neighbours: dict[str, set[str]] = {s: set() for s in species}
    for edge in scores:
        if edge.score >= threshold:
            neighbours[edge.species_a].add(edge.species_b)
            neighbours[edge.species_b].add(edge.species_a)

    visited: set[str] = set()
    groups: list[list[str]] = []
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

    score_map = {frozenset({e.species_a, e.species_b}): e.score for e in scores}
    result: list[SpeciesGroup] = []
    for gid, members in enumerate(groups):
        intra = [
            score_map[frozenset({members[i], members[j]})]
            for i in range(len(members))
            for j in range(i + 1, len(members))
            if frozenset({members[i], members[j]}) in score_map
        ]
        mean = round(sum(intra) / len(intra), 4) if intra else 1.0
        result.append(SpeciesGroup(group_id=gid, species=members, mean_score=mean))
    return result


def compute_confidence(
    scores: list[SimilarityEdge], groups: list[SpeciesGroup]
) -> float | None:
    """How well-separated the reported groups actually are, in [0, 1].

    NOT the mean similarity score -- centered cosine similarity can be
    negative, so a plain mean can itself go negative and render as a
    nonsensical "-24% confidence". Instead this compares mean intra-group
    similarity against mean cross-group similarity: if species inside a
    group are much more similar to each other than to species outside it,
    the grouping is well-supported (confidence near 1). If there's no
    difference, the grouping carries no real signal (confidence near 0.5,
    a coin flip). If it's backwards, confidence approaches 0.

    Returns None -- never a fabricated number -- when there is nothing to
    compare: every species landed in one group (no cross-group pairs) or
    every species is its own singleton group (no intra-group pairs).
    """
    intra_pairs = {
        frozenset({a, b})
        for g in groups
        for i, a in enumerate(g.species)
        for b in g.species[i + 1:]
    }
    intra_scores = [
        e.score for e in scores if frozenset({e.species_a, e.species_b}) in intra_pairs
    ]
    cross_scores = [
        e.score for e in scores if frozenset({e.species_a, e.species_b}) not in intra_pairs
    ]

    if not intra_scores or not cross_scores:
        return None

    separation = statistics.mean(intra_scores) - statistics.mean(cross_scores)
    # Cosine similarity lives in [-1, 1], so separation lives in [-2, 2].
    # Map that onto [0, 1], centered at 0.5 for "no separation".
    return round(max(0.0, min(1.0, (separation + 2) / 4)), 4)


def build_similarity_network(species: list[str], scores: list[SimilarityEdge]) -> dict:
    graph = nx.Graph()
    graph.add_nodes_from(species)
    for edge in scores:
        graph.add_edge(edge.species_a, edge.species_b, score=edge.score)
    return nx.node_link_data(graph, edges="edges")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

FetchFn = Callable[[str, list[str]], str]
EmbedFn = Callable[[dict[str, str]], dict[str, torch.Tensor]]


class MolecularComparisonAgent:
    """Real Molecular Comparison sub-agent.

    fetch_fn/embed_fn are injectable so tests can run without network
    access or a loaded model -- pass fakes in tests, defaults are used
    in production.
    """

    def __init__(
        self,
        fetch_fn: Optional[FetchFn] = None,
        embed_fn: Optional[EmbedFn] = None,
    ):
        self._fetch = fetch_fn or fetch_uniprot_sequence
        self._embed = embed_fn or embed_sequences

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
                output=f"Molecular comparison requires at least 2 species; got {species}.",
                source_agents=["Molecular Comparison Agent"],
            )

        gene_candidates = (
            [request.target_gene_or_protein] if request.target_gene_or_protein
            else DEFAULT_GENE_CANDIDATES
        )

        sequences: dict[str, str] = {}
        if request.protein_inputs:
            sequences = parse_fasta_inputs(request.protein_inputs, species)

        missing = [s for s in species if s not in sequences]
        fetch_errors: list[str] = []
        for sp in missing:
            try:
                sequences[sp] = self._fetch(sp, gene_candidates)
            except Exception as exc:
                fetch_errors.append(f"{sp}: {exc}")

        if fetch_errors:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"Sequence fetch failed for: {'; '.join(fetch_errors)}",
                source_agents=["Molecular Comparison Agent"],
            )

        try:
            embeddings = self._embed(sequences)
        except Exception as exc:
            return AgentResult(
                status=AgentStatus.FAILED,
                output=f"Embedding computation failed: {exc}",
                source_agents=["Molecular Comparison Agent"],
            )

        scores = compute_similarity_scores(embeddings)
        groups = compute_species_groups(species, scores)
        network = build_similarity_network(species, scores)
        confidence = compute_confidence(scores, groups)

        mc_result = MolecularComparisonResult(
            species_list=species,
            similarity_scores=scores,
            species_groups=groups,
            similarity_network=network,
            confidence=confidence,
        )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=mc_result,
            similarity_scores=[
                {"species_a": e.species_a, "species_b": e.species_b, "score": e.score}
                for e in scores
            ],
            confidence=confidence,
            source_agents=["Molecular Comparison Agent"],
        )

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