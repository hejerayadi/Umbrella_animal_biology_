"""Qdrant-backed species-name normalization service.

Species Distribution needs a *scientific* name (e.g. ``Loxodonta africana``)
to call GBIF, but users type free-form ("African elephant", "elefante
africano", "elephant"). We store a small taxonomy in a Qdrant collection
and retrieve the closest scientific name by cosine similarity on
sentence embeddings.

This module has two backends selectable by env var:

- **Online**: uses the real Qdrant cluster + a sentence-transformers
  embedder. Enabled when ``QDRANT_URL`` and ``QDRANT_API_KEY`` are set.
- **Offline fallback**: a small in-memory dict lookup so the orchestrator
  and its tests run without any external service. Enabled by default.

The two backends expose the same ``normalize(query: str) -> str | None``
contract, so the orchestrator does not need to know which one is live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ---------- offline fallback catalogue ----------

_OFFLINE_CATALOGUE: dict[str, str] = {
    # common names -> scientific name
    "african elephant": "Loxodonta africana",
    "elephant d'afrique": "Loxodonta africana",
    "elefante africano": "Loxodonta africana",
    "polar bear": "Ursus maritimus",
    "ours polaire": "Ursus maritimus",
    "oso polar": "Ursus maritimus",
    "tiger": "Panthera tigris",
    "tigre": "Panthera tigris",
    "wolf": "Canis lupus",
    "loup": "Canis lupus",
    "gray wolf": "Canis lupus",
    "arctic tern": "Sterna paradisaea",
    "sterne arctique": "Sterna paradisaea",
    # scientific names normalize to themselves
    "loxodonta africana": "Loxodonta africana",
    "ursus maritimus": "Ursus maritimus",
    "panthera tigris": "Panthera tigris",
    "canis lupus": "Canis lupus",
    "sterna paradisaea": "Sterna paradisaea",
}


@dataclass
class TaxonomyMatch:
    scientific_name: str
    common_names: list[str]
    score: float


class SpeciesTaxonomyService:
    """Uniform interface backed by either Qdrant or an in-memory dict.

    Callers do:
        ``svc = SpeciesTaxonomyService.from_env()``
        ``scientific = svc.normalize("elephant d'afrique")``
    and get ``"Loxodonta africana"`` back (or ``None`` if unknown).
    """

    def __init__(self, backend: "TaxonomyBackend") -> None:
        self._backend = backend

    @classmethod
    def from_env(cls) -> "SpeciesTaxonomyService":
        """Pick the Qdrant backend if credentials are configured, else offline."""
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if url and api_key:
            try:
                return cls(_QdrantBackend(url=url, api_key=api_key))
            except Exception:  # pragma: no cover — surfaces on first real call
                # Never crash the orchestrator because Qdrant is down; degrade.
                return cls(_OfflineBackend())
        return cls(_OfflineBackend())

    def normalize(self, query: str) -> str | None:
        """Return the scientific name for ``query`` or ``None`` if unknown."""
        if not query:
            return None
        return self._backend.lookup(query)

    def top_match(self, query: str) -> TaxonomyMatch | None:
        """Full match record (name + score) - useful when the orchestrator
        wants to threshold on confidence."""
        return self._backend.top_match(query)


# ---------- backends (private) ----------


class TaxonomyBackend:
    def lookup(self, query: str) -> str | None:
        raise NotImplementedError

    def top_match(self, query: str) -> TaxonomyMatch | None:
        raise NotImplementedError


class _OfflineBackend(TaxonomyBackend):
    """Dict lookup + naive substring fallback."""

    def lookup(self, query: str) -> str | None:
        m = self.top_match(query)
        return m.scientific_name if m else None

    def top_match(self, query: str) -> TaxonomyMatch | None:
        q = query.strip().lower()
        if not q:
            return None
        if q in _OFFLINE_CATALOGUE:
            sci = _OFFLINE_CATALOGUE[q]
            return TaxonomyMatch(sci, common_names=[q], score=1.0)
        # naive substring match - covers "the african elephant" -> "african elephant"
        for key, sci in _OFFLINE_CATALOGUE.items():
            if key in q or q in key:
                return TaxonomyMatch(sci, common_names=[key], score=0.6)
        return None


class _QdrantBackend(TaxonomyBackend):  # pragma: no cover — requires live cluster
    """Real Qdrant-backed backend.

    Kept intentionally thin: initialization only opens a client, and
    lookups do a single ``search`` call. The embedder is created lazily
    the first time ``lookup`` runs to keep import cheap.
    """

    COLLECTION_NAME = "species_taxonomy"
    VECTOR_DIM = 384  # matches all-MiniLM-L6-v2

    def __init__(self, url: str, api_key: str) -> None:
        from qdrant_client import QdrantClient  # type: ignore

        self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self._embedder = None

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def lookup(self, query: str) -> str | None:
        match = self.top_match(query)
        return match.scientific_name if match else None

    def top_match(self, query: str) -> TaxonomyMatch | None:
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
        return TaxonomyMatch(
            scientific_name=sci,
            common_names=payload.get("common_names", []),
            score=float(best.score),
        )
