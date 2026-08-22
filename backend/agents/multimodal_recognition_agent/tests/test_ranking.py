"""Validation, ordering and capping of the classifier's taxon predictions.

There is no aggregation here to test any more - a label classifier returns
labels, not reference points to be grouped. What this module owes is the
opposite: it must hold whatever provider is plugged in to the contract it
promised, and refuse rather than repair when the provider breaks it.
"""
from __future__ import annotations

import pytest

from ..domain import ranking
from ..domain.errors import ErrorCode, RecognitionError
from .conftest import prediction


# --- payload validation ----------------------------------------------------

def test_a_complete_payload_is_valid():
    assert ranking.validate_prediction_payload({
        "species_id": "panthera_leo",
        "scientific_name": "Panthera leo",
        "classification_score": 0.9,
    }) is True


@pytest.mark.parametrize("field", ranking.REQUIRED_PREDICTION_FIELDS)
def test_a_missing_mandatory_field_is_invalid(field):
    payload = {
        "species_id": "panthera_leo",
        "scientific_name": "Panthera leo",
        "classification_score": 0.9,
    }
    payload.pop(field)
    assert ranking.validate_prediction_payload(payload) is False


@pytest.mark.parametrize("score", [-0.01, 1.01, "0.9", None, True, float("nan")])
def test_an_unusable_score_is_invalid(score):
    assert ranking.validate_prediction_payload({
        "species_id": "a", "scientific_name": "Aaa aaa", "classification_score": score,
    }) is False


def test_a_rank_other_than_species_is_invalid():
    assert ranking.validate_prediction_payload({
        "species_id": "a", "scientific_name": "Aaa aaa",
        "classification_score": 0.5, "rank": "genus",
    }) is False


def test_a_non_dict_payload_is_invalid():
    for payload in (None, [], "x", 42):
        assert ranking.validate_prediction_payload(payload) is False


# --- ordering and distinctness are enforced, never repaired ----------------

def test_an_ordered_distinct_list_passes():
    ranking.assert_ranked_and_distinct([
        prediction("a", 0.9), prediction("b", 0.5), prediction("c", 0.5),
    ])


def test_an_out_of_order_list_is_refused():
    with pytest.raises(RecognitionError) as excinfo:
        ranking.assert_ranked_and_distinct([prediction("a", 0.4), prediction("b", 0.8)])
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION


def test_a_duplicated_species_is_refused():
    with pytest.raises(RecognitionError):
        ranking.assert_ranked_and_distinct([prediction("a", 0.9), prediction("a", 0.5)])


def test_equal_scores_are_not_treated_as_out_of_order():
    ranking.assert_ranked_and_distinct([prediction("a", 0.7), prediction("b", 0.7)])


def test_the_error_code_is_configurable_for_fixture_callers():
    with pytest.raises(RecognitionError) as excinfo:
        ranking.assert_ranked_and_distinct(
            [prediction("a", 0.1), prediction("b", 0.9)],
            error=ErrorCode.CLASSIFICATION_FIXTURE_INVALID,
        )
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_FIXTURE_INVALID


# --- building candidates ---------------------------------------------------

def test_candidates_preserve_the_classifier_order_and_scores():
    candidates = ranking.build_candidates(
        [prediction("a", 0.9, "Aaa aaa"), prediction("b", 0.4, "Bbb bbb")], top_k=5
    )
    assert [c.species_id for c in candidates] == ["a", "b"]
    assert [c.classification_score for c in candidates] == [0.9, 0.4]
    assert [c.scientific_name for c in candidates] == ["Aaa aaa", "Bbb bbb"]


@pytest.mark.parametrize("top_k, expected", [(1, 1), (2, 2), (3, 3), (10, 3)])
def test_top_k_caps_the_candidates(top_k, expected):
    predictions = [prediction("a", 0.9), prediction("b", 0.6), prediction("c", 0.3)]
    assert len(ranking.build_candidates(predictions, top_k=top_k)) == expected


def test_capping_keeps_the_highest_scoring_labels():
    predictions = [prediction("a", 0.9), prediction("b", 0.6), prediction("c", 0.3)]
    assert [c.species_id for c in ranking.build_candidates(predictions, top_k=2)] == ["a", "b"]


def test_no_candidate_carries_a_taxonomy_identifier_yet():
    """Taxonomy is a later node's job. Nothing is invented here."""
    for candidate in ranking.build_candidates([prediction("a", 0.9)], top_k=5):
        assert candidate.gbif_id is None
        assert candidate.ncbi_taxid is None
        assert candidate.taxonomy_status == "unverified"


def test_an_empty_prediction_list_yields_no_candidates():
    assert ranking.build_candidates([], top_k=5) == []


@pytest.mark.parametrize("top_k", [0, -3])
def test_a_non_positive_top_k_is_refused(top_k):
    with pytest.raises(RecognitionError) as excinfo:
        ranking.build_candidates([prediction("a", 0.9)], top_k=top_k)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION


def test_build_candidates_refuses_an_unranked_provider():
    with pytest.raises(RecognitionError):
        ranking.build_candidates([prediction("a", 0.2), prediction("b", 0.8)], top_k=5)


# --- margin ----------------------------------------------------------------

def test_margin_between_two_candidates():
    candidates = ranking.build_candidates(
        [prediction("a", 0.9), prediction("b", 0.6)], top_k=5
    )
    assert ranking.top_margin(candidates) == pytest.approx(0.3)


def test_a_single_candidate_has_no_runner_up():
    candidates = ranking.build_candidates([prediction("a", 0.9)], top_k=5)
    assert ranking.top_margin(candidates) == 1.0


def test_no_candidate_has_no_margin():
    assert ranking.top_margin([]) is None
