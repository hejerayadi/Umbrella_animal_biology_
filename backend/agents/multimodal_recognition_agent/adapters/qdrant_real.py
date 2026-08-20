"""The real Qdrant query adapter - a validated boundary, not yet connectable.

This class is deliberately unable to construct itself until the Sprint 2
collection contract has been supplied. Every value it needs - the endpoint, the
collection name, the vector name, the dimension, the distance, the dataset
version, the Top-K - belongs to Chahd. Guessing any of them would mean querying
a collection whose shape we do not know, and quietly getting nonsense back.

So the constructor asks one question first: is the contract frozen? If not, it
raises `QDRANT_CONTRACT_NOT_FROZEN` and says which fields are missing. That is
the honest failure, and it is what makes it impossible to accidentally ship a
half-configured "real" path.

READ-ONLY BY CONSTRUCTION. There is no `create_collection`, no `upsert`, no
`delete`, no migration method here and there never will be. Ingestion is
Chahd's, and this class is shaped so that ownership cannot drift by accident.

`qdrant-client` is intentionally NOT pinned in requirements.txt yet: pinning it
would mean writing calls against a contract that does not exist. It is imported
lazily, only once the contract is complete, so its absence costs nothing today.
"""
from __future__ import annotations

import logging
from typing import Any

from ..config import QdrantConfig
from ..domain.errors import ErrorCode, RecognitionError
from ..domain.models import RetrievedReference
from ..domain.ranking import validate_payload
from .retrieval import RetrievalOutcome

_logger = logging.getLogger(__name__)


class RealQdrantRetriever:
    """Queries the Sprint 2 collection. Reads only."""

    provider_name = "Qdrant"
    mode = "real_minimal"

    def __init__(self, config: QdrantConfig, api_key: str | None = None) -> None:
        missing = config.missing_contract_fields
        if missing:
            # Named, so whoever sees this knows exactly what Chahd still owes -
            # but the values themselves are never logged.
            _logger.info(
                "Real Qdrant retrieval unavailable; unconfigured contract fields: %s",
                ", ".join(missing),
            )
            raise RecognitionError(ErrorCode.QDRANT_CONTRACT_NOT_FROZEN)

        self.config = config

        try:  # imported here, not at module scope, so the package is optional
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RecognitionError(ErrorCode.QDRANT_CLIENT_UNAVAILABLE) from exc

        self._client = QdrantClient(
            url=config.url,
            api_key=api_key,
            timeout=config.timeout_seconds,
        )

    def validate_collection_contract(self) -> None:
        """Compare the live collection against the configured contract.

        Runs before the first query. A mismatch means the vectors in the
        collection were not produced the way we are producing ours, so the
        similarities would be meaningless - we refuse rather than query.
        """
        try:
            info: Any = self._client.get_collection(self.config.collection)
        except Exception as exc:  # noqa: BLE001 - client raises a wide range
            raise RecognitionError(ErrorCode.RETRIEVAL_UNAVAILABLE) from exc

        params = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            raise RecognitionError(ErrorCode.COLLECTION_CONTRACT_MISMATCH)

        # Qdrant reports a single unnamed vector config, or a dict of named ones.
        if isinstance(vectors, dict):
            if self.config.vector_name is None or self.config.vector_name not in vectors:
                raise RecognitionError(ErrorCode.COLLECTION_CONTRACT_MISMATCH)
            spec = vectors[self.config.vector_name]
        else:
            if self.config.vector_name is not None:
                raise RecognitionError(ErrorCode.COLLECTION_CONTRACT_MISMATCH)
            spec = vectors

        size = getattr(spec, "size", None)
        distance = getattr(spec, "distance", None)
        distance_name = getattr(distance, "name", None) or str(distance)

        if size != self.config.expected_dimension:
            raise RecognitionError(ErrorCode.COLLECTION_CONTRACT_MISMATCH)
        if distance_name.lower() != str(self.config.expected_distance).lower():
            raise RecognitionError(ErrorCode.COLLECTION_CONTRACT_MISMATCH)

    def search(self, query_vector: list[float], *, top_k: int) -> RetrievalOutcome:
        if len(query_vector) != self.config.expected_dimension:
            raise RecognitionError(ErrorCode.EMBEDDING_DIMENSION_MISMATCH)

        self.validate_collection_contract()

        try:
            hits: Any = self._client.search(
                collection_name=self.config.collection,
                query_vector=(
                    (self.config.vector_name, query_vector)
                    if self.config.vector_name
                    else query_vector
                ),
                limit=top_k,
                # Payloads yes, vectors no: we never need the stored vectors
                # back, and not transferring them keeps responses small.
                with_payload=True,
                with_vectors=False,
                query_filter=self._dataset_filter(),
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            if "timeout" in message or "timed out" in message:
                raise RecognitionError(ErrorCode.RETRIEVAL_TIMEOUT) from exc
            raise RecognitionError(ErrorCode.RETRIEVAL_UNAVAILABLE) from exc

        references: list[RetrievedReference] = []
        rejected = 0
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            if not validate_payload(payload):
                rejected += 1
                continue
            references.append(
                RetrievedReference(
                    point_id=str(getattr(hit, "id", "")),
                    species_id=payload["species_id"],
                    scientific_name=payload["scientific_name"],
                    common_name=payload.get("common_name"),
                    similarity_score=float(getattr(hit, "score", 0.0)),
                    source=payload.get("source"),
                    dataset_version=payload["dataset_version"],
                    embedding_mode=payload["embedding_mode"],
                )
            )

        return RetrievalOutcome(
            references=references,
            provider=self.provider_name,
            mode=self.mode,
            collection=self.config.collection,
            dataset_version=self.config.dataset_version,
            rejected_payloads=rejected,
        )

    def _dataset_filter(self) -> Any:
        """Restrict results to the agreed dataset version.

        The exact filterable fields are Chahd's to confirm; this filters on
        `dataset_version` only, which the payload contract already requires.
        """
        try:
            from qdrant_client.http import models as rest
        except ImportError as exc:  # pragma: no cover - constructor already checked
            raise RecognitionError(ErrorCode.QDRANT_CLIENT_UNAVAILABLE) from exc

        return rest.Filter(
            must=[
                rest.FieldCondition(
                    key="dataset_version",
                    match=rest.MatchValue(value=self.config.dataset_version),
                )
            ]
        )
