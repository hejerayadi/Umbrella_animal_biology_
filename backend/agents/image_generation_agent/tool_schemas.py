"""Structured input/output schemas for image-generation evidence tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class UniProtResult:
    success: bool
    accession: str | None = None
    protein_name: str | None = None
    organism: str | None = None
    sequence_length: int | None = None
    sequence: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class PDBStructureResult:
    found: bool
    pdb_id: str | None = None
    experimental_method: str | None = None
    resolution: float | None = None
    sequence_coverage: float | None = None
    title: str | None = None
    organism: str | None = None
    chain: str | None = None
    selection_reason: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class WebSearchHit:
    title: str
    link: str
    snippet: str


@dataclass(frozen=True)
class WebSearchResult:
    success: bool
    results: tuple[WebSearchHit, ...] = field(default_factory=tuple)
    error: str | None = None


CriticVerdict = Literal["ACCEPT", "REVISE", "ABSTAIN"]


@dataclass(frozen=True)
class CriticResult:
    verdict: CriticVerdict
    reasons: tuple[str, ...] = field(default_factory=tuple)
    source: str = "deterministic"


@dataclass(frozen=True)
class EvidenceBundle:
    uniprot: UniProtResult | None = None
    pdb: PDBStructureResult | None = None
    web_search: WebSearchResult | None = None
    critic: CriticResult | None = None
    skipped: bool = True

    @classmethod
    def empty(cls) -> EvidenceBundle:
        return cls(skipped=True)
