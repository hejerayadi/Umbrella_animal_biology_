"""The retrieval boundary.

The workflow asks for "the nearest reference points to this vector" and gets
back validated `RetrievedReference` objects. It does not know whether they came
from a real Qdrant collection or from a local fixture file, which is the whole
point: the same workflow, the same ranking, the same decisions, either way.

What the workflow DOES know is which provider answered, because that goes into
the provenance of every response. A result produced from local fixtures says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..domain.models import RetrievedReference


@dataclass(frozen=True)
class RetrievalOutcome:
    """What one retrieval attempt produced, plus how to describe it truthfully."""

    references: list[RetrievedReference]

    # Provenance. `mode` is deliberately not a boolean: "real_minimal" and
    # "mock_local_development" are different claims and must never be conflated.
    provider: str
    mode: str
    collection: str | None
    dataset_version: str | None

    # Hits dropped for missing mandatory payload fields. Surfaced rather than
    # silently swallowed, so a broken ingestion is visible.
    rejected_payloads: int = 0


class RetrievalProvider(Protocol):
    """What the workflow needs from a vector store."""

    provider_name: str
    mode: str

    def search(self, query_vector: list[float], *, top_k: int) -> RetrievalOutcome:
        ...
