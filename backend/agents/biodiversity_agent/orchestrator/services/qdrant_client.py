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

import logging
import os
from dataclasses import dataclass

_logger = logging.getLogger(__name__)


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
    # The three species the migration Random Forest is trained on.
    "white stork": "Ciconia ciconia",
    "cigogne blanche": "Ciconia ciconia",
    "humpback whale": "Megaptera novaeangliae",
    "baleine a bosse": "Megaptera novaeangliae",
    "baleine à bosse": "Megaptera novaeangliae",
    "monarch butterfly": "Danaus plexippus",
    "monarch": "Danaus plexippus",
    "papillon monarque": "Danaus plexippus",
    # scientific names normalize to themselves
    "loxodonta africana": "Loxodonta africana",
    "ursus maritimus": "Ursus maritimus",
    "panthera tigris": "Panthera tigris",
    "canis lupus": "Canis lupus",
    "sterna paradisaea": "Sterna paradisaea",
    "ciconia ciconia": "Ciconia ciconia",
    "megaptera novaeangliae": "Megaptera novaeangliae",
    "danaus plexippus": "Danaus plexippus",
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
        # Cheapest first: in-memory dict, then GBIF's resolver over the network.
        fallbacks: list[TaxonomyBackend] = [_OfflineBackend(), _GbifBackend()]
        if url and api_key:
            try:
                # Chained, not either/or: Qdrant stays authoritative when it
                # answers, and the fallbacks cover it when it does not.
                return cls(_ChainedBackend(
                    [_QdrantBackend(url=url, api_key=api_key), *fallbacks]))
            except Exception:  # pragma: no cover — surfaces on first real call
                # Never crash the orchestrator because Qdrant is down; degrade.
                return cls(_ChainedBackend(fallbacks))
        return cls(_ChainedBackend(fallbacks))

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




class _GbifBackend(TaxonomyBackend):
    """GBIF's own name resolver - the broadest backend, so it goes last.

    ``workers.common.gbif.match_species`` already does the work the taxonomy
    collection was meant to do, and does it for every animal GBIF knows rather
    than the handful in the catalogue above: a common-name table, ``/species/
    match`` for binomials and misspellings, an exact vernacular search, and
    plural handling ("African elephants" -> ``Loxodonta africana``). It costs a
    network call, which is why the in-memory catalogue is consulted first.
    """

    def lookup(self, query: str) -> str | None:
        match = self.top_match(query)
        return match.scientific_name if match else None

    def top_match(self, query: str) -> TaxonomyMatch | None:
        # Imported here, not at module scope: this package is imported by the
        # workers, and a top-level import would close the loop.
        from ...workers.common.gbif import match_species

        resolved = match_species(query)
        if not resolved:
            return None
        _key, scientific = resolved
        if not scientific:
            return None
        # No similarity score to report - GBIF either resolves a name or does
        # not, so a hit is a hit.
        return TaxonomyMatch(scientific_name=scientific, common_names=[query], score=1.0)


class _ChainedBackend(TaxonomyBackend):
    """Try each backend in order, first non-empty answer wins.

    Exists because a configured Qdrant URL is not the same thing as a usable
    taxonomy: the collection may be missing, empty, or unreachable. Treating
    "credentials present" as "Qdrant works" let a silent miss propagate all
    the way to GBIF as an unresolved common name.
    """

    def __init__(self, backends: list[TaxonomyBackend]) -> None:
        self._backends = backends

    def lookup(self, query: str) -> str | None:
        match = self.top_match(query)
        return match.scientific_name if match else None

    def top_match(self, query: str) -> TaxonomyMatch | None:
        for backend in self._backends:
            try:
                match = backend.top_match(query)
            except Exception as exc:  # noqa: BLE001 - a backend may fail anyhow
                _logger.warning(
                    "[Taxonomy] %s raised for %r (%s); trying the next backend",
                    type(backend).__name__, query, type(exc).__name__,
                )
                continue
            if match is not None:
                return match
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
        # A missing collection fails identically on every lookup; warn once
        # rather than once per request.
        self._warned = False

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
        except Exception as exc:  # noqa: BLE001 - any client/collection failure
            # Logged once per lookup rather than swallowed: a missing
            # ``species_taxonomy`` collection is indistinguishable from an
            # unknown species otherwise, and the two need different fixes.
            if not self._warned:
                self._warned = True
                _logger.warning(
                    "[Taxonomy] Qdrant lookup failed (%s: %s); falling back to "
                    "the offline catalogue for this and later lookups. If the "
                    "%r collection is missing, seed it or unset QDRANT_URL.",
                    type(exc).__name__, exc, self.COLLECTION_NAME,
                )
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
