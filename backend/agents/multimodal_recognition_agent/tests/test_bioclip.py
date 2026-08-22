"""The mock BioCLIP-2 classification boundary.

Four properties matter, and every test below is one of them:

1. it is deterministic - the same image always produces the same labels;
2. a known fixture image produces its controlled, already-ranked Top-K;
3. an unknown image produces NOTHING - never a guessed taxon;
4. a broken oracle fails safely instead of being repaired into an answer.
"""
from __future__ import annotations

import json

import pytest

from ..adapters.bioclip import BioCLIP2Classifier, MockBioCLIP2Provider
from ..config import RECOGNITION_IMAGE_CONTEXT_KEY, ValidationConfig
from ..domain.errors import ErrorCode, RecognitionError
from ..domain.models import BioCLIPTaxonPrediction
from ..validation import validate_paired_request
from .conftest import image_entry, make_config, png_bytes, sha256_of

LION = [
    {"species_id": "panthera_leo", "scientific_name": "Panthera leo",
     "common_name": "lion", "rank": "species", "classification_score": 0.91},
    {"species_id": "panthera_pardus", "scientific_name": "Panthera pardus",
     "rank": "species", "classification_score": 0.42},
    {"species_id": "panthera_tigris", "scientific_name": "Panthera tigris",
     "rank": "species", "classification_score": 0.31},
]


def normalized(raw: bytes | None = None, filename: str = "observation.png"):
    raw = raw if raw is not None else png_bytes()
    return validate_paired_request(
        "Identify this animal.",
        {RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(raw, filename=filename)},
        make_config().validation,
    )


def provider_for(raw: bytes, entries=None, **kwargs) -> MockBioCLIP2Provider:
    return MockBioCLIP2Provider(
        predictions={sha256_of(raw): list(entries if entries is not None else LION)},
        **kwargs,
    )


# --- the interface ---------------------------------------------------------

def test_the_mock_satisfies_the_classifier_protocol():
    assert isinstance(MockBioCLIP2Provider(), BioCLIP2Classifier)


def test_the_provider_declares_its_mode_and_target_model():
    provider = MockBioCLIP2Provider()
    assert provider.recognition_mode == "mock_classification"
    assert provider.provider_name == "MockBioCLIP2Provider"
    assert provider.model_target == "BioCLIP-2"
    assert provider.version == "sprint2-mock-bioclip2-classifier-v1"


def test_the_provider_exposes_classify_and_no_embedding_method():
    provider = MockBioCLIP2Provider()
    assert callable(provider.classify)
    for removed in ("embed_image", "embed", "search", "query_vector", "seed_for"):
        assert not hasattr(provider, removed), removed


# --- determinism -----------------------------------------------------------

def test_the_same_image_always_produces_the_same_labels():
    raw = png_bytes()
    provider = provider_for(raw)
    image = normalized(raw)

    first = provider.classify(image, 5)
    second = provider.classify(image, 5)

    assert first == second
    assert [p.species_id for p in first] == [
        "panthera_leo", "panthera_pardus", "panthera_tigris"
    ]


def test_two_separately_built_providers_agree():
    raw = png_bytes()
    image = normalized(raw)
    assert provider_for(raw).classify(image, 5) == provider_for(raw).classify(image, 5)


def test_a_different_image_is_a_different_key():
    raw = png_bytes()
    provider = provider_for(raw)
    other = normalized(png_bytes(marker="something else"))
    assert provider.classify(other, 5) == []


# --- a known fixture returns ordered Top-K ---------------------------------

def test_a_known_image_returns_ordered_predictions():
    raw = png_bytes()
    predictions = provider_for(raw).classify(normalized(raw), 5)

    scores = [p.classification_score for p in predictions]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(p, BioCLIPTaxonPrediction) for p in predictions)
    assert all(p.rank == "species" for p in predictions)


@pytest.mark.parametrize("top_k, expected", [(1, 1), (2, 2), (3, 3), (5, 3), (50, 3)])
def test_top_k_caps_the_result(top_k, expected):
    raw = png_bytes()
    assert len(provider_for(raw).classify(normalized(raw), top_k)) == expected


def test_top_k_keeps_the_highest_scoring_labels():
    raw = png_bytes()
    assert [p.species_id for p in provider_for(raw).classify(normalized(raw), 2)] == [
        "panthera_leo", "panthera_pardus"
    ]


@pytest.mark.parametrize("top_k", [0, -1])
def test_a_non_positive_top_k_is_a_controlled_failure(top_k):
    raw = png_bytes()
    with pytest.raises(RecognitionError) as excinfo:
        provider_for(raw).classify(normalized(raw), top_k)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION


# --- an unknown image is answered with nothing -----------------------------

def test_an_unknown_image_returns_no_candidates():
    provider = MockBioCLIP2Provider(predictions={})
    assert provider.classify(normalized(), 5) == []


def test_an_unknown_image_never_borrows_a_label_from_a_known_one():
    raw = png_bytes()
    provider = provider_for(raw)
    unknown = normalized(png_bytes(marker="unknown-image"))

    assert provider.classify(unknown, 5) == []
    # And the known one still works, so the empty answer is not a broken oracle.
    assert provider.classify(normalized(raw), 5)


def test_no_randomness_is_involved():
    """Repeated across many distinct unknown images: always empty, never a
    sampled taxon."""
    provider = MockBioCLIP2Provider(predictions={})
    for index in range(25):
        image = normalized(png_bytes(marker=f"unknown-{index}"))
        assert provider.classify(image, 5) == []


# --- a broken oracle fails safely ------------------------------------------

@pytest.mark.parametrize(
    "entries",
    [
        # unordered: the second label outscores the first
        [{"species_id": "a", "scientific_name": "Aaa aaa", "classification_score": 0.3},
         {"species_id": "b", "scientific_name": "Bbb bbb", "classification_score": 0.9}],
        # the same species twice
        [{"species_id": "a", "scientific_name": "Aaa aaa", "classification_score": 0.9},
         {"species_id": "a", "scientific_name": "Aaa aaa", "classification_score": 0.5}],
    ],
)
def test_an_unordered_or_duplicated_fixture_fails_safely(entries):
    raw = png_bytes()
    provider = provider_for(raw, entries)
    with pytest.raises(RecognitionError) as excinfo:
        provider.classify(normalized(raw), 5)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_FIXTURE_INVALID


@pytest.mark.parametrize(
    "entry",
    [
        {"scientific_name": "Aaa aaa", "classification_score": 0.9},        # no id
        {"species_id": "a", "classification_score": 0.9},                   # no name
        {"species_id": "a", "scientific_name": "Aaa aaa"},                  # no score
        {"species_id": "a", "scientific_name": "Aaa aaa", "classification_score": 1.4},
        {"species_id": "a", "scientific_name": "Aaa aaa", "classification_score": -0.1},
        {"species_id": "a", "scientific_name": "Aaa aaa", "classification_score": "high"},
        {"species_id": "a", "scientific_name": "Aaa aaa", "classification_score": True},
        {"species_id": "a", "scientific_name": "Aaa aaa", "classification_score": 0.9,
         "rank": "genus"},
        {"species_id": 7, "scientific_name": "Aaa aaa", "classification_score": 0.9},
        "not an object",
    ],
)
def test_a_malformed_prediction_record_fails_safely(entry):
    raw = png_bytes()
    provider = provider_for(raw, [entry])
    with pytest.raises(RecognitionError) as excinfo:
        provider.classify(normalized(raw), 5)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_FIXTURE_INVALID


@pytest.mark.parametrize(
    "key",
    ["not-a-sha", "ABCDEF" * 10 + "abcd", "a" * 63, "a" * 65, ""],
)
def test_a_fixture_key_that_is_not_a_sha256_fails_safely(key):
    provider = MockBioCLIP2Provider(predictions={key: LION})
    with pytest.raises(RecognitionError) as excinfo:
        provider.classify(normalized(), 5)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_FIXTURE_INVALID


def test_a_fixture_entry_that_is_not_a_list_fails_safely():
    provider = MockBioCLIP2Provider(predictions={sha256_of(png_bytes()): "lion"})
    with pytest.raises(RecognitionError) as excinfo:
        provider.classify(normalized(), 5)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_FIXTURE_INVALID


def test_a_missing_fixture_file_fails_safely(tmp_path):
    provider = MockBioCLIP2Provider(fixture_path=tmp_path / "absent.json")
    with pytest.raises(RecognitionError) as excinfo:
        provider.classify(normalized(), 5)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_FIXTURE_INVALID


@pytest.mark.parametrize(
    "document",
    ['{"images": {}}',                   # no provider_version
     '{"provider_version": "", "images": {}}',
     '{"provider_version": "v1"}',       # no images
     '{"provider_version": "v1", "images": []}',
     '[]', 'not json at all'],
)
def test_a_malformed_fixture_document_fails_safely(tmp_path, document):
    path = tmp_path / "broken.json"
    path.write_text(document, encoding="utf-8")
    provider = MockBioCLIP2Provider(fixture_path=path)
    with pytest.raises(RecognitionError) as excinfo:
        provider.classify(normalized(), 5)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_FIXTURE_INVALID


def test_a_provider_outage_is_a_controlled_failure():
    provider = MockBioCLIP2Provider(fail_with=ErrorCode.CLASSIFICATION_UNAVAILABLE)
    with pytest.raises(RecognitionError) as excinfo:
        provider.classify(normalized(), 5)
    assert excinfo.value.code is ErrorCode.CLASSIFICATION_UNAVAILABLE


# --- the shipped test oracle -----------------------------------------------

def test_the_shipped_fixture_loads_and_validates():
    provider = MockBioCLIP2Provider()
    # `knows` forces the load, which is where every schema rule is applied.
    assert provider.knows("0" * 64) is False
    assert provider.fixture_version == "sprint2-mock-bioclip2-classifier-v1"


def test_the_shipped_fixture_declares_itself_a_test_oracle():
    import pathlib

    path = (pathlib.Path(__file__).resolve().parent.parent
            / "fixtures" / "mock_bioclip_predictions.json")
    document = json.loads(path.read_text(encoding="utf-8"))

    note = document["_fixture_note"].lower()
    assert "test oracle" in note
    assert "not bioclip-2 inference" in note
    assert document["recognition_mode"] == "mock_classification"
    assert document["score_kind"] == "deterministic_sprint2_test_score"


def test_every_shipped_fixture_entry_is_ranked_and_in_range():
    import pathlib

    path = (pathlib.Path(__file__).resolve().parent.parent
            / "fixtures" / "mock_bioclip_predictions.json")
    images = json.loads(path.read_text(encoding="utf-8"))["images"]

    assert images, "the shipped oracle should hold at least one demo image"
    for digest, entries in images.items():
        assert len(digest) == 64
        scores = [entry["classification_score"] for entry in entries]
        assert scores == sorted(scores, reverse=True), digest
        assert all(0.0 <= score <= 1.0 for score in scores), digest
        ids = [entry["species_id"] for entry in entries]
        assert len(ids) == len(set(ids)), digest


def test_the_demo_image_digests_match_the_shipped_oracle():
    """The generator and the oracle must agree, or the demo runbook is broken.

    If this ever fails, a different Pillow build is encoding the demo PNGs
    differently: re-run `fixtures/make_demo_images.py` and paste the printed
    digests into `mock_bioclip_predictions.json`.
    """
    from ..fixtures.make_demo_images import DEMO_IMAGES, render

    provider = MockBioCLIP2Provider()
    known = {name: provider.knows(sha256_of(render(size, colour, fmt)))
             for name, size, colour, fmt in DEMO_IMAGES}

    assert known["demo_identified.png"] is True
    assert known["demo_uncertain.png"] is True
    assert known["demo_partial_taxonomy.png"] is True
    # Deliberately absent, so the demo can show an honest not_identified.
    assert known["demo_unknown.png"] is False


# --- no real model is anywhere near this -----------------------------------

def test_the_module_imports_no_model_runtime():
    """Checked on import statements, not on prose: the docstring is allowed to
    say the word BioCLIP, the code is not allowed to import it."""
    import pathlib
    import re

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "adapters" / "bioclip.py").read_text(encoding="utf-8")

    pattern = re.compile(
        r"^\s*(?:import|from)\s+(torch|open_clip|bioclip|pybioclip|numpy|qdrant_client"
        r"|requests|httpx|urllib)\b",
        re.MULTILINE,
    )
    assert pattern.search(source) is None, pattern.search(source).group(0)


def test_validation_config_is_unchanged_by_classification():
    """Sanity: the classifier never touches validation bounds."""
    config = make_config().validation
    assert isinstance(config, ValidationConfig)
    assert config.min_image_width == 64
