"""Shared dataclasses for the reconstruction data ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CleanedScaffold:
    accession: str
    species_id: str
    scaffold_id: str
    sequence: str
    header: str


@dataclass
class Window:
    scaffold_id: str
    accession: str
    species_id: str
    sequence: str
    start: int
    end: int


@dataclass
class MaskedTriplet:
    window_id: str
    left_context: str
    masked_region: str
    right_context: str


@dataclass
class EmbeddingRecord:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class SpeciesConfig:
    accession: str
    species_id: str
    source: str
    is_mammoth: bool = False
    partition: str = "train"
    chromosome: str | None = None


@dataclass
class IngestionSummary:
    processed_accessions: list[str] = field(default_factory=list)
    failed_accessions: list[str] = field(default_factory=list)
    stored_points: int = 0


__all__ = [
    "CleanedScaffold",
    "Window",
    "MaskedTriplet",
    "EmbeddingRecord",
    "SpeciesConfig",
    "IngestionSummary",
]
