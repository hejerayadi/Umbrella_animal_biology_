"""Species name resolution service for the Evolution Agent.

Takes a raw species name (common or scientific, any capitalisation) and
returns the canonical scientific name the downstream workers and external
APIs (GenBank, TimeTree) expect.

Two backends, same ``resolve`` contract:

- **Offline** (default): a small in-memory dict. Covers the Sprint 1
  mock catalogue plus common-name aliases. Zero dependencies.
- **Qdrant** (future): cosine similarity search on ESMC embeddings stored
  in a dedicated ``evolution_embeddings`` collection. Activated when
  ``QDRANT_URL`` and ``QDRANT_API_KEY`` are set in the environment.

The orchestrator calls ``resolve_all(species_list)`` which returns two
lists: successfully resolved canonical names and names that could not be
resolved. If the unresolved list is non-empty, the orchestrator stops and
returns FAILED — it never passes partial species lists to workers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Offline catalogue
# ---------------------------------------------------------------------------
# Maps lowercase query → canonical scientific name (title-cased).
# Covers the Sprint 1 mock catalogue plus the most common aliases.
# Add rows here as new species enter the mock fixtures.
_OFFLINE: dict[str, str] = {
    # ── Homo sapiens ──────────────────────────────────────────────
    "homo sapiens":         "Homo sapiens",
    "human":                "Homo sapiens",
    "humans":               "Homo sapiens",
    "human being":          "Homo sapiens",
    # ── Pan troglodytes ───────────────────────────────────────────
    "pan troglodytes":      "Pan troglodytes",
    "chimpanzee":           "Pan troglodytes",
    "chimp":                "Pan troglodytes",
    "common chimpanzee":    "Pan troglodytes",
    # ── Mus musculus ──────────────────────────────────────────────
    "mus musculus":         "Mus musculus",
    "mouse":                "Mus musculus",
    "house mouse":          "Mus musculus",
    "laboratory mouse":     "Mus musculus",
    # ── Gallus gallus ─────────────────────────────────────────────
    "gallus gallus":        "Gallus gallus",
    "chicken":              "Gallus gallus",
    "red junglefowl":       "Gallus gallus",
    # ── Danio rerio ───────────────────────────────────────────────
    "danio rerio":          "Danio rerio",
    "zebrafish":            "Danio rerio",
    "zebra fish":           "Danio rerio",
    "zebra danio":          "Danio rerio",
}


@dataclass
class ResolvedSpecies:
    """Result of a single name lookup."""
    query: str           # original query string
    canonical: str       # scientific name (title-cased)
    score: float         # 1.0 = exact match; < 1.0 = fuzzy/vector match


class SpeciesResolverService:
    """Uniform interface backed by either Qdrant or an in-memory dict.

    Usage:
        svc = SpeciesResolverService.from_env()
        canonical, unresolved = svc.resolve_all(["human", "Mus musculus"])
        # canonical = ["Homo sapiens", "Mus musculus"], unresolved = []
    """

    def __init__(self, backend: "_ResolverBackend") -> None:
        self._backend = backend

    @classmethod
    def from_env(cls) -> "SpeciesResolverService":
        """Pick Qdrant backend if credentials are configured, else offline."""
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if url and api_key:
            try:
                return cls(_QdrantBackend(url=url, api_key=api_key))
            except Exception:  # pragma: no cover — surfaces on first real call
                return cls(_OfflineBackend())
        return cls(_OfflineBackend())

    def resolve(self, query: str) -> ResolvedSpecies | None:
        """Resolve a single name. Returns None if completely unknown."""
        if not query or not query.strip():
            return None
        return self._backend.lookup(query.strip())

    def resolve_all(
        self, queries: list[str]
    ) -> tuple[list[str], list[str]]:
        """Resolve every name in ``queries``.

        Returns:
            canonical  — list of resolved scientific names (same order as input)
            unresolved — list of original query strings that could not be resolved

        The caller should treat a non-empty ``unresolved`` list as a hard
        failure and not proceed to any worker.
        """
        canonical: list[str] = []
        unresolved: list[str] = []
        for q in queries:
            match = self.resolve(q)
            if match:
                canonical.append(match.canonical)
            else:
                unresolved.append(q)
        return canonical, unresolved


# ---------------------------------------------------------------------------
# Backends (private)
# ---------------------------------------------------------------------------


class _ResolverBackend:
    def lookup(self, query: str) -> ResolvedSpecies | None:
        raise NotImplementedError


class _OfflineBackend(_ResolverBackend):
    """Dict lookup with a conservative, word-aware fallback.

    The fallback only accepts a catalogue entry whose *whole words* all
    appear as whole words in the query — "the common chimpanzee" still
    resolves via "common chimpanzee", but "mouse lemur" no longer collapses
    onto *Mus musculus*.

    Two rules keep it from inventing matches:

    * bare substring matching is not enough. ``"mouse" in "mouse lemur"``
      is true, yet a mouse lemur (*Microcebus*) is not a house mouse; a
      wrong canonical name is worse than an honest failure, because the
      caller has no way to notice the swap.
    * a query that adds a qualifying word to a catalogue name is rejected.
      Extra words normally narrow to a *different* taxon, so anything left
      over after the match has to be a filler word to be ignored.
    """

    # Words that carry no taxonomic meaning and may be dropped when
    # comparing a query against a catalogue key.
    _FILLER = frozenset({"the", "a", "an", "of", "species", "sp", "spp"})

    @staticmethod
    def _words(text: str) -> list[str]:
        return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]

    def lookup(self, query: str) -> ResolvedSpecies | None:
        q = query.lower().strip()

        # Exact match first.
        if q in _OFFLINE:
            return ResolvedSpecies(query=query, canonical=_OFFLINE[q], score=1.0)

        q_words = [w for w in self._words(q) if w not in self._FILLER]
        if not q_words:
            return None
        q_set = set(q_words)

        # Prefer the longest catalogue key that the query fully contains,
        # so "common chimpanzee" wins over the shorter "chimpanzee" and the
        # result cannot depend on dict ordering.
        best: tuple[int, str, str] | None = None
        for key, sci in _OFFLINE.items():
            key_words = [w for w in self._words(key) if w not in self._FILLER]
            if not key_words:
                continue
            key_set = set(key_words)
            if not key_set <= q_set:
                continue
            # Every leftover query word must be filler, else the query is
            # naming something more specific than the catalogue entry.
            if q_set - key_set:
                continue
            if best is None or len(key_words) > best[0]:
                best = (len(key_words), key, sci)

        if best is None:
            return None
        return ResolvedSpecies(query=query, canonical=best[2], score=0.9)


class _QdrantBackend(_ResolverBackend):  # pragma: no cover — requires live cluster
    """Qdrant-backed resolver using ESMC embeddings.

    Uses the ``evolution_embeddings`` collection (distinct from biodiversity's
    ``species_taxonomy``), which stores one vector per known species.
    """

    COLLECTION_NAME = "evolution_embeddings"
    VECTOR_DIM = 384  # all-MiniLM-L6-v2

    def __init__(self, url: str, api_key: str) -> None:
        from qdrant_client import QdrantClient  # type: ignore

        self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self._embedder = None

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def lookup(self, query: str) -> ResolvedSpecies | None:
        try:
            hits = self._client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=self._embed(query),
                limit=1,
            )
        except Exception:
            return None
        if not hits:
            return None
        best = hits[0]
        payload = best.payload or {}
        sci = payload.get("scientific_name")
        if not sci:
            return None
        return ResolvedSpecies(
            query=query,
            canonical=sci,
            score=float(best.score),
        )
