"""Taxonomy validation - GBIF and NCBI, both mocked for Sprint 2.

GBIF, NCBI and IUCN are not called. Not once, not "just to check". The fixtures
below stand in for them, and every candidate they touch carries a
`taxonomy_status` saying how much of it was found.

Where these sources sit in the workflow matters as much as what they do. They
run **after** BioCLIP-2 classification, on the candidates it produced. They may
validate a candidate, supply its identifiers and normalise a synonym to its
accepted name. They may never create a species candidate the classifier did not
return, and they may never re-order or re-score one.

The rule that matters most: when a fixture has no GBIF ID or no NCBI taxid, the
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


def _identifier(raw: object) -> int | None:
    """An identifier the fixture actually supplied, or None.

    A non-integer value in the fixture is treated as absent rather than coerced.
    Coercing `"5219404"` or `5219404.0` into an int would be the agent deciding
    what a broken record meant, which is one short step from inventing one.
    """
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw


class TaxonomySource(Protocol):
    """One mocked taxonomy service."""

    source: str
    mode: str

    def lookup(self, species_id: str, scientific_name: str) -> TaxonomyLookup:
        ...


@dataclass(frozen=True)
class TaxonomyLookup:
    """One source's answer about one candidate.

    `available` is False when the source itself did not answer (down, timing
    out). `matched` is False when it answered but holds no record for this
    species. The two are different facts and the response reports both.

    `accepted_name` lets a source resolve a synonym to the accepted name. It can
    only ever rename a species the classifier already returned - it cannot
    introduce a different one, because the caller matches it back against the
    candidate.

    `inconsistent` marks a record that contradicts the candidate (a different
    scientific name under the same id). Such a record is used for nothing: no
    identifier is taken from it.
    """

    available: bool
    matched: bool
    identifier: int | None = None
    accepted_name: str | None = None
    rank: str | None = None
    classification: dict[str, str] = dc_field(default_factory=dict)
    inconsistent: bool = False


def _record_is_consistent(record: dict, species_id: str, scientific_name: str,
                          synonyms: dict) -> bool:
    """Does this fixture record actually describe the candidate we asked about?

    A record whose scientific name is neither the candidate's name nor a known
    synonym of it is inconsistent fixture data. We refuse to read an identifier
    out of it rather than attaching someone else's taxid to this species.
    """
    fixture_name = record.get("scientific_name")
    if not isinstance(fixture_name, str) or not fixture_name:
        return False
    if fixture_name.strip().lower() == scientific_name.strip().lower():
        return True
    # The candidate may be a synonym that resolves onto this accepted record.
    return synonyms.get(species_id) is not None


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
        # An unavailable source is not a failed request. It is a fact reported
        # in the response, and the candidate survives it visibly unverified.
        if self._unavailable or species_id in self._unavailable_ids:
            return TaxonomyLookup(available=False, matched=False)

        resolved_id = self._synonyms.get(species_id, species_id)
        record = self._species.get(resolved_id)
        if not isinstance(record, dict):
            return TaxonomyLookup(available=True, matched=False)

        if not _record_is_consistent(record, species_id, scientific_name, self._synonyms):
            return TaxonomyLookup(available=True, matched=False, inconsistent=True)

        return TaxonomyLookup(
            available=True,
            matched=True,
            identifier=_identifier(record.get("gbif_id")),
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
        if not isinstance(record, dict):
            return TaxonomyLookup(available=True, matched=False)

        # NCBI resolves no synonyms here, so the name must agree exactly.
        if not _record_is_consistent(record, species_id, scientific_name, {}):
            return TaxonomyLookup(available=True, matched=False, inconsistent=True)

        return TaxonomyLookup(
            available=True,
            matched=True,
            identifier=_identifier(record.get("ncbi_taxid")),
            accepted_name=record.get("scientific_name"),
            rank=record.get("rank", "SPECIES"),
        )


class MockTaxonomyProvider:
    """Fixture-backed taxonomy with the branches the workflow must handle:
    a complete record, a partial one, no match at all, an unavailable service,
    and an inconsistent record.

    It is a facade over two independent sources, so "GBIF up, NCBI down" is a
    state the workflow can actually be tested against, and the response reports
    each source's outcome separately.
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
        """Validate one classified candidate through both sources.

        Neither source may pick a species, change a score or change the order.
        `accepted_name` is applied only when it refers to the candidate the
        classifier already returned.
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
        # A common name may be supplied where the classifier had none. It is a
        # label, never an identity: `species_id` and `scientific_name` are the
        # classifier's and stay untouched.
        if gbif.matched and gbif.accepted_name and candidate.common_name is None:
            record = self._species.get(candidate.species_id, {})
            names = record.get("common_names") or []
            if names:
                updates["common_name"] = names[0]

        report = {
            "gbif": {"mode": self.gbif.mode, "available": gbif.available,
                     "matched": gbif.matched, "identifier": gbif_id,
                     "inconsistent_record": gbif.inconsistent},
            "ncbi": {"mode": self.ncbi.mode, "available": ncbi.available,
                     "matched": ncbi.matched, "identifier": ncbi_taxid,
                     "inconsistent_record": ncbi.inconsistent},
            "status": status,
        }
        return candidate.model_copy(update=updates), report

    # -- name resolution, used by the text analyser -------------------------

    def resolve_name(self, text: str) -> str | None:
        """Map a scientific or common name in free text to a species_id.

        Used only to understand what the user *said*. It can never add a
        candidate: the workflow compares the result against species the
        classifier already returned.
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

    # -- single-source enrichment (kept for direct callers and tests) -------

    def enrich(self, candidate: SpeciesCandidate) -> SpeciesCandidate:
        if self._simulate_unavailable or candidate.species_id in self._unavailable_ids:
            # The service being down is not a reason to fail the request. The
            # caller catches this and the candidate survives, visibly unverified.
            raise RecognitionError(ErrorCode.TAXONOMY_UNAVAILABLE)

        record = self._species.get(candidate.species_id)
        if record is None:
            # No match. Not an error - just nothing to add.
            return candidate.model_copy(update={"taxonomy_status": "unverified"})

        gbif_id = _identifier(record.get("gbif_id"))
        ncbi_taxid = _identifier(record.get("ncbi_taxid"))

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


def build_taxonomy_provider(config: "RecognitionConfig") -> MockTaxonomyProvider:
    """Construct the taxonomy provider the configured mode names, or refuse.

    Same contract as `bioclip.build_classifier`: the fixture-backed provider is
    reachable only through `mode == "mock"`, and a mode with no implementation
    behind it raises instead of degrading to fixture data. Fixture taxonomy
    served as though it were GBIF and NCBI would attach real-looking identifiers
    to a species nobody looked up.
    """
    from ..config import (
        TAXONOMY_PROVIDER_MODES,
        ConfigError,
    )

    mode = (config.taxonomy_provider_mode or "").strip().lower()

    if mode == "mock":
        return MockTaxonomyProvider()

    if mode in TAXONOMY_PROVIDER_MODES:
        raise ConfigError(
            f"TAXONOMY_PROVIDER_MODE={mode!r} has no implementation yet "
            "(live GBIF and NCBI lookups arrive in Phase 4). Refusing to start "
            "rather than serving fixture taxonomy as live lookups."
        )

    raise ConfigError(
        "TAXONOMY_PROVIDER_MODE must be one of: "
        + ", ".join(TAXONOMY_PROVIDER_MODES)
        + f". Got {mode!r}."
    )
