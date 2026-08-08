"""Aggregating reference hits into ranked species."""
from __future__ import annotations

import pytest

from ..domain.ranking import (
    REQUIRED_PAYLOAD_FIELDS,
    aggregate_by_species,
    top_margin,
    validate_payload,
)
from .conftest import candidate, reference


def _aggregate(references, per_species=3, top_k=5):
    return aggregate_by_species(
        references, max_references_per_species=per_species, top_k_species=top_k
    )


def test_duplicate_references_collapse_into_one_candidate():
    references = [
        reference("panthera_leo", 0.90, "p1"),
        reference("panthera_leo", 0.86, "p2"),
        reference("panthera_leo", 0.82, "p3"),
    ]
    candidates = _aggregate(references)

    assert len(candidates) == 1
    assert candidates[0].species_id == "panthera_leo"
    assert candidates[0].reference_count == 3


def test_score_is_the_mean_of_the_best_n_references():
    references = [
        reference("panthera_leo", 0.90, "p1"),
        reference("panthera_leo", 0.80, "p2"),
        reference("panthera_leo", 0.70, "p3"),
        reference("panthera_leo", 0.10, "p4"),  # excluded from the mean
    ]
    candidates = _aggregate(references, per_species=3)

    assert candidates[0].similarity_score == (0.90 + 0.80 + 0.70) / 3
    # ...but the count reports every matching reference.
    assert candidates[0].reference_count == 4


def test_score_is_smoothed_across_a_species_own_references():
    """One unusually high hit among a species' references does not carry it."""
    references = [
        reference("noisy_species", 0.95, "a"),   # one great hit...
        reference("noisy_species", 0.10, "b"),   # ...and two poor ones
        reference("noisy_species", 0.10, "c"),
        reference("steady_species", 0.90, "d"),
    ]
    candidates = _aggregate(references)

    assert candidates[0].species_id == "steady_species"
    # 0.95 alone would have won; the mean of its own best three does not.
    assert candidates[1].similarity_score == pytest.approx((0.95 + 0.10 + 0.10) / 3)


def test_a_single_reference_is_not_penalised_for_being_sparse():
    """Documented behaviour: weighting by evidence count would be calibration,
    and Sprint 2 makes none."""
    references = [
        reference("sparse_species", 0.95, "a"),
        reference("dense_species", 0.90, "b"),
        reference("dense_species", 0.89, "c"),
    ]
    assert _aggregate(references)[0].species_id == "sparse_species"


def test_candidates_are_ordered_by_score():
    references = [
        reference("c", 0.50, "1"),
        reference("a", 0.90, "2"),
        reference("b", 0.70, "3"),
    ]
    assert [c.species_id for c in _aggregate(references)] == ["a", "b", "c"]


def test_top_k_species_is_respected():
    references = [reference(f"species_{i}", 0.9 - i / 100, str(i)) for i in range(10)]
    assert len(_aggregate(references, top_k=5)) == 5


def test_ties_break_deterministically():
    references = [reference("zebra", 0.8, "1"), reference("aardvark", 0.8, "2")]
    first = _aggregate(references)
    second = _aggregate(list(reversed(references)))
    assert [c.species_id for c in first] == [c.species_id for c in second] == ["aardvark", "zebra"]


def test_taxonomy_is_not_invented_during_aggregation():
    candidates = _aggregate([reference("panthera_leo", 0.9)])
    assert candidates[0].gbif_id is None
    assert candidates[0].ncbi_taxid is None
    assert candidates[0].taxonomy_status == "unverified"


def test_no_references_gives_no_candidates():
    assert _aggregate([]) == []


# --- margin ----------------------------------------------------------------

def test_margin_is_none_without_candidates():
    assert top_margin([]) is None


def test_single_candidate_has_maximum_margin():
    assert top_margin([candidate("panthera_leo", 0.9)]) == 1.0


def test_margin_is_the_gap_to_the_runner_up():
    candidates = [candidate("a", 0.90), candidate("b", 0.85)]
    assert top_margin(candidates) == 0.90 - 0.85


# --- payload validation ----------------------------------------------------

def test_complete_payload_is_valid():
    payload = {field: "value" for field in REQUIRED_PAYLOAD_FIELDS}
    assert validate_payload(payload) is True


def test_payload_missing_a_mandatory_field_is_rejected():
    for missing in REQUIRED_PAYLOAD_FIELDS:
        payload = {field: "value" for field in REQUIRED_PAYLOAD_FIELDS}
        del payload[missing]
        assert validate_payload(payload) is False, missing


def test_payload_with_an_empty_mandatory_field_is_rejected():
    payload = {field: "value" for field in REQUIRED_PAYLOAD_FIELDS}
    payload["species_id"] = ""
    assert validate_payload(payload) is False
