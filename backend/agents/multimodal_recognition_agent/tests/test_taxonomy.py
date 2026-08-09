"""`MockTaxonomyProvider` - all four branches, directly.

These assertions previously existed only inside the workflow suite. They belong
here: the four branches are a Phase 3 requirement in their own right, and a
taxonomy branch should be provable without running a whole request.

No live GBIF, NCBI or IUCN call is made. Every value comes from the local
fixture file.
"""
from __future__ import annotations

import pytest

from ..adapters.taxonomy import MockTaxonomyProvider
from ..domain.errors import ErrorCode, RecognitionError
from .conftest import candidate


@pytest.fixture
def provider():
    return MockTaxonomyProvider()


# --- branch 1: verified fixture -------------------------------------------

def test_verified_fixture_supplies_both_identifiers(provider):
    enriched = provider.enrich(candidate("panthera_leo", 0.9))

    assert enriched.taxonomy_status == "mock_verified"
    assert enriched.gbif_id is not None
    assert enriched.ncbi_taxid is not None


def test_verified_status_is_derived_from_what_is_present_not_claimed(provider):
    """Status is computed from the data, so the fixture cannot lie about itself."""
    fixtures = {"species": {"x": {"scientific_name": "X", "gbif_id": 1, "ncbi_taxid": 2}}}
    assert MockTaxonomyProvider(fixtures=fixtures).enrich(
        candidate("x", 0.9)
    ).taxonomy_status == "mock_verified"


# --- branch 2: partial fixture --------------------------------------------

def test_partial_fixture_is_reported_as_partial(provider):
    enriched = provider.enrich(candidate("ursus_maritimus", 0.9))

    assert enriched.taxonomy_status == "partial"
    assert enriched.gbif_id is None          # absent in the fixture
    assert enriched.ncbi_taxid is not None


def test_partial_when_only_gbif_is_present():
    fixtures = {"species": {"x": {"scientific_name": "X", "gbif_id": 1, "ncbi_taxid": None}}}
    assert MockTaxonomyProvider(fixtures=fixtures).enrich(
        candidate("x", 0.9)
    ).taxonomy_status == "partial"


# --- branch 3: missing match ----------------------------------------------

def test_species_absent_from_the_fixture_is_unverified(provider):
    enriched = provider.enrich(candidate("no_such_species_xyz", 0.9))

    assert enriched.taxonomy_status == "unverified"
    assert enriched.gbif_id is None
    assert enriched.ncbi_taxid is None


def test_species_present_but_with_no_identifiers_is_unverified(provider):
    enriched = provider.enrich(candidate("vulpes_lagopus", 0.9))

    assert enriched.taxonomy_status == "unverified"
    assert enriched.gbif_id is None
    assert enriched.ncbi_taxid is None


# --- branch 4: unavailable / timeout --------------------------------------

def test_simulated_outage_raises_a_controlled_error():
    with pytest.raises(RecognitionError) as caught:
        MockTaxonomyProvider(simulate_unavailable=True).enrich(candidate("panthera_leo", 0.9))
    assert caught.value.code is ErrorCode.RETRIEVAL_UNAVAILABLE


def test_outage_degrades_visibly_without_crashing():
    """The candidate survives, marked unverified, and the caller is told."""
    enriched, degraded = MockTaxonomyProvider(simulate_unavailable=True).enrich_all(
        [candidate("panthera_leo", 0.9), candidate("panthera_tigris", 0.5)]
    )

    assert degraded is True
    assert len(enriched) == 2
    assert all(c.taxonomy_status == "unverified" for c in enriched)
    assert all(c.gbif_id is None and c.ncbi_taxid is None for c in enriched)


def test_per_species_outage_degrades_only_that_species():
    fixtures = {
        "species": {"panthera_leo": {"scientific_name": "Panthera leo",
                                     "gbif_id": 1, "ncbi_taxid": 2}},
        "unavailable_species_ids": ["flaky_species"],
    }
    enriched, degraded = MockTaxonomyProvider(fixtures=fixtures).enrich_all(
        [candidate("panthera_leo", 0.9), candidate("flaky_species", 0.8)]
    )

    assert degraded is True
    assert enriched[0].taxonomy_status == "mock_verified"
    assert enriched[1].taxonomy_status == "unverified"


def test_healthy_provider_reports_no_degradation(provider):
    _, degraded = provider.enrich_all([candidate("panthera_leo", 0.9)])
    assert degraded is False


# --- identifiers are never invented ---------------------------------------

def test_a_missing_identifier_is_never_filled_in(provider):
    """The central rule: absent means null, not guessed."""
    for species_id in ("ursus_maritimus", "vulpes_lagopus", "no_such_species_xyz"):
        enriched = provider.enrich(candidate(species_id, 0.9))
        assert enriched.gbif_id is None or isinstance(enriched.gbif_id, int)
        if species_id != "ursus_maritimus":
            assert enriched.ncbi_taxid is None


def test_identifiers_are_never_borrowed_from_a_sibling_species(provider):
    """A species with no IDs must not inherit its genus-mate's."""
    leo = provider.enrich(candidate("panthera_leo", 0.9))
    fox = provider.enrich(candidate("vulpes_lagopus", 0.9))

    assert leo.gbif_id is not None
    assert fox.gbif_id is None
    assert fox.ncbi_taxid is None


def test_enrichment_never_changes_the_similarity_score(provider):
    before = candidate("panthera_leo", 0.8123)
    after = provider.enrich(before)
    assert after.similarity_score == before.similarity_score == 0.8123


def test_enrichment_never_changes_the_species_identity(provider):
    before = candidate("panthera_leo", 0.9, name="Panthera leo")
    after = provider.enrich(before)
    assert after.species_id == before.species_id
    assert after.scientific_name == before.scientific_name


# --- fixtures are visibly mock --------------------------------------------

def test_the_fixture_file_is_labelled_as_mock_and_unverified():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "fixtures" / "mock_taxonomy.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    note = data["_fixture_note"].lower()
    assert "fixture" in note
    assert "not" in note and "gbif" in note and "ncbi" in note
    assert data["dataset"].startswith("local-dev")


def test_provider_declares_mock_mode(provider):
    assert provider.mode == "mock"
