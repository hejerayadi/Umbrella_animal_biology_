"""Local fixture retriever - development and unit tests ONLY.

This is NOT the Sprint 2 Qdrant deliverable and must never be presented as one.
It exists so the workflow, the ranking and the confidence gate can be built and
tested before Chahd's collection exists, and so failure branches can be
exercised offline.

Its provenance mode is `mock_local_development`. Nothing in this file can emit
`real_minimal`, so a response produced from fixtures cannot be mistaken for one
produced from the real collection - not in a log, not in a demo, not in a report.

The similarity it computes is a real cosine over real vectors; only the vectors
are fixtures. That keeps the ranking code honest: it is doing the same
arithmetic it will do against the real collection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..config import LOCAL_DATASET_VERSION
from ..domain.errors import ErrorCode, RecognitionError
from ..domain.models import RetrievedReference
from ..domain.ranking import validate_payload
from .bioclip import deterministic_vector
from .retrieval import RetrievalOutcome

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "mock_references.json"

# How much of a fixture reference's vector comes from its species rather than
# from the individual reference.
#
# This exists because two SHA-derived vectors are effectively orthogonal, so two
# references of the SAME species would score independently - and a species'
# mean-of-best-N would be dragged down by its own second reference. Real
# embeddings do not behave that way: images of one species cluster together.
#
# So a fixture reference is built as a blend of a per-species centre and a
# per-reference offset. This is FIXTURE CONSTRUCTION, local to this file and to
# offline development. It is not part of the query-side vector contract in
# `bioclip.py` - that algorithm is untouched, and it is the only thing the
# ingestion side has to reproduce.
_SPECIES_COHESION = 0.9


def _cosine(a: list[float], b: list[float]) -> float:
    """Both sides are L2-normalised, so the dot product is the cosine."""
    return sum(x * y for x, y in zip(a, b))


def _blend(base: list[float], offset: list[float], weight: float) -> list[float]:
    """`weight * base + (1 - weight) * offset`, L2-normalised."""
    mixed = [weight * b + (1.0 - weight) * o for b, o in zip(base, offset)]
    norm = math.sqrt(sum(component * component for component in mixed))
    return [component / norm for component in mixed]


class MockQdrantRetriever:
    """Fixture-backed retrieval. Never touches a network or a database."""

    provider_name = "MockQdrantRetriever"
    mode = "mock_local_development"

    def __init__(
        self,
        dimension: int,
        *,
        points: list[dict] | None = None,
        dataset_version: str = LOCAL_DATASET_VERSION,
        fail_with: ErrorCode | None = None,
    ) -> None:
        """`points` overrides the fixture file - that is how tests build exact
        scenarios (a clear winner, two close candidates, no hits at all).

        `fail_with` makes the unavailable/timeout branches testable without
        needing something to actually be unavailable.
        """
        self.dimension = dimension
        self.dataset_version = dataset_version
        self._fail_with = fail_with
        self._points = points if points is not None else self._load_fixture_points()

    @staticmethod
    def _load_fixture_points() -> list[dict]:
        data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
        return list(data.get("points", []))

    def _vector_for(self, point: dict) -> list[float]:
        """A stored point's vector.

        Explicit if the fixture gives one (tests use that to build exact
        scenarios); otherwise a blend of the species centre and this
        reference's own offset, so same-species references cluster the way real
        embeddings would.
        """
        if "vector" in point:
            return list(point["vector"])

        offset = deterministic_vector(point["vector_seed"], self.dimension)
        species_id = (point.get("payload") or {}).get("species_id")
        if not species_id:
            return offset
        centre = deterministic_vector(species_id, self.dimension)
        return _blend(centre, offset, _SPECIES_COHESION)

    def search(self, query_vector: list[float], *, top_k: int) -> RetrievalOutcome:
        if self._fail_with is not None:
            raise RecognitionError(self._fail_with)

        if len(query_vector) != self.dimension:
            raise RecognitionError(ErrorCode.EMBEDDING_DIMENSION_MISMATCH)

        scored: list[tuple[float, dict]] = []
        rejected = 0

        for point in self._points:
            payload = point.get("payload", point)
            # Same validation the real adapter applies. A reference we cannot
            # attribute to a species is dropped, not guessed at.
            if not validate_payload(payload):
                rejected += 1
                continue

            vector = self._vector_for(point)
            if len(vector) != self.dimension:
                rejected += 1
                continue

            scored.append((_cosine(query_vector, vector), payload))

        scored.sort(key=lambda item: (-item[0], str(item[1].get("point_id", ""))))

        references = [
            RetrievedReference(
                point_id=str(payload.get("point_id", "")),
                species_id=payload["species_id"],
                scientific_name=payload["scientific_name"],
                common_name=payload.get("common_name"),
                similarity_score=score,
                source=payload.get("source"),
                dataset_version=payload["dataset_version"],
                embedding_mode=payload["embedding_mode"],
            )
            for score, payload in scored[:top_k]
        ]

        return RetrievalOutcome(
            references=references,
            provider=self.provider_name,
            mode=self.mode,
            collection=None,
            dataset_version=self.dataset_version,
            rejected_payloads=rejected,
        )
