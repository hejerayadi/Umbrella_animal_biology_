"""Taxonomy lookup - mocked for Sprint 2.

GBIF, NCBI and IUCN are not called. Not once, not "just to check". The fixtures
below stand in for them, and every candidate they touch carries a
`taxonomy_status` saying how much of it was found.

The rule that matters: when a fixture has no GBIF ID or no NCBI taxid, the
answer is `None`. It is never filled in, never inferred from a sibling species,
never approximated. A missing identifier is a fact about our data, and inventing
one would turn a mock into a fabrication.

`mock_verified` means "the fixture had everything" - it does NOT mean the
identifier was verified against a live database. Nothing here was.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Protocol

from ..domain.errors import ErrorCode, RecognitionError
from ..domain.models import SpeciesCandidate

_FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "mock_taxonomy.json"


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


class TaxonomyProvider(Protocol):
    mode: str

    def enrich(self, candidate: SpeciesCandidate) -> SpeciesCandidate:
        ...


@dataclass(frozen=True)
class TaxonomyLookup:
    """One source's answer about one candidate.

    `accepted_name` lets a source resolve a synonym to the accepted name. It can
    only ever rename a species retrieval already returned - it cannot introduce
    a different one, because the workflow matches it back against the candidate.
    """

    available: bool
    matched: bool
    identifier: int | None = None
    accepted_name: str | None = None
    rank: str | None = None
    classification: dict[str, str] = dc_field(default_factory=dict)


class MockGBIFProvider:
    """GBIF, mocked. Verifies a scientific name and resolves synonyms.

    It never chooses a species and never invents an identifier: an absent value
    in the fixture comes back as `None`.
    """

    source = "GBIF"
    mode = "mock"

    def __init__(self, *, species: dict | None = None, synonyms: dict | None = None,
                 unavailable: bool = False, unavailable_ids: set[str] | None = None) -> None:
        data = species if species is not None else _load_fixture()
        self._species = data.get("species", {}) if "species" in data else data
        self._synonyms = synonyms if synonyms is not None else _load_fixture().get("synonyms", {})
        self._unavailable = unavailable
        self._unavailable_ids = unavailable_ids or set()

    def lookup(self, species_id: str, scientific_name: str) -> TaxonomyLookup:
        if self._unavailable or species_id in self._unavailable_ids:
            return TaxonomyLookup(available=False, matched=False)

        resolved_id = self._synonyms.get(species_id, species_id)
        record = self._species.get(resolved_id)
        if record is None:
            return TaxonomyLookup(available=True, matched=False)

        return TaxonomyLookup(
            available=True,
            matched=True,
            identifier=record.get("gbif_id"),
            accepted_name=record.get("scientific_name"),
            rank=record.get("rank", "SPECIES"),
            classification=dict(record.get("classification", {})),
        )


class MockNCBIProvider:
    """NCBI Taxonomy, mocked. Supplies a taxid when the fixture has one."""

    source = "NCBI"
    mode = "mock"

    def __init__(self, *, species: dict | None = None, unavailable: bool = False,
                 unavailable_ids: set[str] | None = None) -> None:
        data = species if species is not None else _load_fixture()
        self._species = data.get("species", {}) if "species" in data else data
        self._unavailable = unavailable
        self._unavailable_ids = unavailable_ids or set()

    def lookup(self, species_id: str, scientific_name: str) -> TaxonomyLookup:
        if self._unavailable or species_id in self._unavailable_ids:
            return TaxonomyLookup(available=False, matched=False)

        record = self._species.get(species_id)
        if record is None:
            return TaxonomyLookup(available=True, matched=False)

        return TaxonomyLookup(
            available=True,
            matched=True,
            identifier=record.get("ncbi_taxid"),
            accepted_name=record.get("scientific_name"),
            rank=record.get("rank", "SPECIES"),
        )


class MockTaxonomyProvider:
    """Fixture-backed taxonomy with four behaviours the workflow must handle:
    a complete record, a partial one, no match at all, and an unavailable
    service.

    Since Phase 4 this is a facade over two independent sources, so "GBIF up,
    NCBI down" is a state the workflow can actually be tested against. The
    single-provider API is unchanged.
    """

    mode = "mock"

    def __init__(
        self,
        *,
        fixtures: dict | None = None,
        simulate_unavailable: bool = False,
        gbif_unavailable: bool = False,
        ncbi_unavailable: bool = False,
    ) -> None:
        data = fixtures if fixtures is not None else _load_fixture()
        self._species: dict[str, dict] = data.get("species", {})
        self._unavailable_ids = set(data.get("unavailable_species_ids", []))
        self._simulate_unavailable = simulate_unavailable

        # The two sources are independent, so "GBIF answered, NCBI timed out" is
        # a real state rather than an all-or-nothing outage.
        self.gbif = MockGBIFProvider(
            species=data,
            synonyms=data.get("synonyms", {}),
            unavailable=simulate_unavailable or gbif_unavailable,
            unavailable_ids=self._unavailable_ids,
        )
        self.ncbi = MockNCBIProvider(
            species=data,
            unavailable=simulate_unavailable or ncbi_unavailable,
            unavailable_ids=self._unavailable_ids,
        )

    def validate_candidate(self, candidate: SpeciesCandidate) -> tuple[SpeciesCandidate, dict]:
        """Enrich through both sources, and report what each one did.

        Neither source may pick a species: `accepted_name` is applied only when
        it refers to the candidate retrieval already returned.
        """
        gbif = self.gbif.lookup(candidate.species_id, candidate.scientific_name)
        ncbi = self.ncbi.lookup(candidate.species_id, candidate.scientific_name)

        gbif_id = gbif.identifier if gbif.matched else None
        ncbi_taxid = ncbi.identifier if ncbi.matched else None

        if gbif_id is not None and ncbi_taxid is not None:
            status = "mock_verified"
        elif gbif_id is not None or ncbi_taxid is not None:
            status = "partial"
        else:
            status = "unverified"

        updates: dict[str, object] = {
            "gbif_id": gbif_id,
            "ncbi_taxid": ncbi_taxid,
            "taxonomy_status": status,
        }
        # A synonym may be normalised to its accepted name - never to a
        # different species.
        if gbif.matched and gbif.accepted_name and candidate.common_name is None:
            record = self._species.get(candidate.species_id, {})
            names = record.get("common_names") or []
            if names:
                updates["common_name"] = names[0]

        report = {
            "gbif": {"available": gbif.available, "matched": gbif.matched,
                     "identifier": gbif_id},
            "ncbi": {"available": ncbi.available, "matched": ncbi.matched,
                     "identifier": ncbi_taxid},
            "status": status,
        }
        return candidate.model_copy(update=updates), report

    # -- name resolution, used by the text analyser -------------------------

    def resolve_name(self, text: str) -> str | None:
        """Map a scientific or common name in free text to a species_id.

        Used only to understand what the user *said*. It can never add a
        candidate: the workflow compares the result against species that
        retrieval already returned.
        """
        needle = text.strip().lower()
        if not needle:
            return None
        for species_id, record in self._species.items():
            if record.get("scientific_name", "").lower() == needle:
                return species_id
            if needle in {name.lower() for name in record.get("common_names", [])}:
                return species_id
        return None

    def known_names(self) -> dict[str, str]:
        """Every name this fixture knows, lowercased, mapped to its species_id."""
        names: dict[str, str] = {}
        for species_id, record in self._species.items():
            scientific = record.get("scientific_name")
            if scientific:
                names[scientific.lower()] = species_id
            for common in record.get("common_names", []):
                names[common.lower()] = species_id
        return names

    # -- enrichment ---------------------------------------------------------

    def enrich(self, candidate: SpeciesCandidate) -> SpeciesCandidate:
        if self._simulate_unavailable or candidate.species_id in self._unavailable_ids:
            # The service being down is not a reason to fail the request. The
            # candidate survives, visibly unverified.
            raise RecognitionError(ErrorCode.RETRIEVAL_UNAVAILABLE)

        record = self._species.get(candidate.species_id)
        if record is None:
            # No match. Not an error - just nothing to add.
            return candidate.model_copy(update={"taxonomy_status": "unverified"})

        gbif_id = record.get("gbif_id")
        ncbi_taxid = record.get("ncbi_taxid")

        # Status derived from what is actually present, not from the fixture's
        # own claim, so the two can never drift apart.
        if gbif_id is not None and ncbi_taxid is not None:
            status = "mock_verified"
        elif gbif_id is not None or ncbi_taxid is not None:
            status = "partial"
        else:
            status = "unverified"

        return candidate.model_copy(
            update={
                "gbif_id": gbif_id,
                "ncbi_taxid": ncbi_taxid,
                "taxonomy_status": status,
                "common_name": candidate.common_name or (
                    record.get("common_names") or [None]
                )[0],
            }
        )

    def enrich_all(self, candidates: list[SpeciesCandidate]) -> tuple[list[SpeciesCandidate], bool]:
        """Enrich every candidate. Returns (candidates, taxonomy_degraded).

        A taxonomy outage degrades the answer; it does not fail the request.
        """
        enriched: list[SpeciesCandidate] = []
        degraded = False
        for candidate in candidates:
            try:
                enriched.append(self.enrich(candidate))
            except RecognitionError:
                degraded = True
                enriched.append(candidate.model_copy(update={"taxonomy_status": "unverified"}))
        return enriched, degraded
