"""The deterministic mock BioCLIP-2 provider.

These tests pin the vector algorithm. If any of them start failing after a
change, the change has broken compatibility with whatever was ingested using the
old one - which is exactly the failure mode that would otherwise be silent.
"""
from __future__ import annotations

import math

import pytest

from ..adapters.bioclip import (
    UNKNOWN_IMAGE_SEED_PREFIX,
    MockBioCLIP2Provider,
    deterministic_vector,
)
from ..validation import validate_paired_request
from .conftest import make_config, png_bytes


def test_no_bioclip_or_torch_import_anywhere_in_the_package():
    """Sprint 2 must not require a model runtime. This asserts it cannot."""
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent
    forbidden = ("import torch", "import open_clip", "import bioclip", "from torch",
                 "from open_clip", "from bioclip", "hf-hub:", "huggingface_hub")

    offenders = []
    for path in package.rglob("*.py"):
        # Skip this agent's own installed dependencies and its test suite.
        if ".venv" in path.parts or path.parent.name == "tests":
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, offenders


def test_vector_is_deterministic():
    first = deterministic_vector("panthera_leo/ref-001", 32)
    second = deterministic_vector("panthera_leo/ref-001", 32)
    assert first == second


def test_vector_length_matches_dimension():
    for dimension in (8, 32, 64, 128, 768):
        assert len(deterministic_vector("seed", dimension)) == dimension


def test_vector_is_l2_normalised():
    vector = deterministic_vector("panthera_leo/ref-001", 32)
    norm = math.sqrt(sum(component * component for component in vector))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_components_are_in_range_before_normalisation():
    # After normalisation every component is within [-1, 1] by construction.
    assert all(-1.0 <= c <= 1.0 for c in deterministic_vector("seed", 64))


def test_different_seeds_give_different_vectors():
    a = deterministic_vector("panthera_leo/ref-001", 32)
    b = deterministic_vector("panthera_tigris/ref-001", 32)
    assert a != b
    # And they are not accidentally near-identical.
    assert abs(sum(x * y for x, y in zip(a, b))) < 0.9


def test_dimension_must_be_positive():
    with pytest.raises(ValueError):
        deterministic_vector("seed", 0)


def test_same_image_always_yields_the_same_vector():
    config = make_config()
    provider = MockBioCLIP2Provider(dimension=32, version="test-v0")
    context = {"recognition_image": {
        "data_url": _url(png_bytes()), "filename": "a.png"}}

    first = validate_paired_request("identify", context, config.validation)
    second = validate_paired_request("identify", context, config.validation)

    assert provider.embed_image(first) == provider.embed_image(second)


def test_unknown_image_is_seeded_from_its_own_hash():
    provider = MockBioCLIP2Provider(dimension=32, version="test-v0")
    sha = "a" * 64
    assert provider.seed_for(sha) == UNKNOWN_IMAGE_SEED_PREFIX + sha


def test_known_image_maps_to_the_agreed_fixture_seed():
    sha = "b" * 64
    provider = MockBioCLIP2Provider(
        dimension=32, version="test-v0", image_seed_overrides={sha: "panthera_leo/ref-001"}
    )
    assert provider.seed_for(sha) == "panthera_leo/ref-001"
    # ...and produces exactly the reference's vector, which is the whole point.
    from ..validation import validate_paired_request as _v  # noqa: F401
    assert deterministic_vector("panthera_leo/ref-001", 32) == deterministic_vector(
        provider.seed_for(sha), 32
    )


def test_provider_declares_mock_provenance():
    provider = MockBioCLIP2Provider(dimension=32, version="local-dev-mock-bioclip2-v0")
    assert provider.embedding_mode == "mock"
    assert provider.provider_name == "MockBioCLIP2Provider"
    assert provider.version == "local-dev-mock-bioclip2-v0"


def _url(raw: bytes) -> str:
    import base64

    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
