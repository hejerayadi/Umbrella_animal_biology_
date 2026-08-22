"""Shared test fixtures.

Images are generated in-process with Pillow rather than committed as binary
files: the specification forbids committing raw images, and a generated one is
reproducible anyway.

Nothing here reaches a network, a database or a model. `StubClassifier` is how a
test states exactly which taxonomic labels the classification boundary returned,
which is what makes the confidence, text-fusion and taxonomy branches testable
one at a time.
"""
from __future__ import annotations

import base64
import hashlib
import io

import pytest
from PIL import Image, PngImagePlugin

from ..config import RecognitionConfig, ThresholdConfig, ValidationConfig
from ..domain.models import BioCLIPTaxonPrediction, SpeciesCandidate


# --- image builders --------------------------------------------------------

def png_bytes(width: int = 128, height: int = 128, marker: str | None = None) -> bytes:
    """A valid PNG. `marker` is embedded as a plaintext tEXt chunk, which is how
    the leak tests get a known string into the *decoded* bytes."""
    image = Image.new("RGB", (width, height), (10, 120, 200))
    buffer = io.BytesIO()
    info = None
    if marker is not None:
        info = PngImagePlugin.PngInfo()
        info.add_text("marker", marker)
    image.save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def jpeg_bytes(width: int = 128, height: int = 128) -> bytes:
    image = Image.new("RGB", (width, height), (200, 90, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def webp_bytes(width: int = 128, height: int = 128) -> bytes:
    image = Image.new("RGB", (width, height), (40, 190, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP")
    return buffer.getvalue()


def data_url(raw: bytes, media_type: str = "image/png") -> str:
    return f"data:{media_type};base64," + base64.b64encode(raw).decode("ascii")


def image_entry(raw: bytes, media_type: str = "image/png", filename: str = "observation.png"):
    return {"data_url": data_url(raw, media_type), "filename": filename}


def sha256_of(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# --- configuration ---------------------------------------------------------

def make_config(**overrides) -> RecognitionConfig:
    """A complete config built directly, so tests never touch the environment."""
    base = dict(
        validation=ValidationConfig(
            max_image_bytes=10_485_760,
            max_image_pixels=25_000_000,
            min_image_width=64,
            min_image_height=64,
        ),
        thresholds=ThresholdConfig(
            identified_min_score=0.75,
            identified_min_margin=0.08,
            uncertain_min_score=0.45,
        ),
        bioclip_provider_mode="mock",
        recognition_mode="mock_classification",
        mock_provider_version="sprint2-mock-bioclip2-classifier-v1",
        classification_fixture_path=None,
        taxonomy_provider_mode="mock",
        text_analyzer_mode="rules",
        top_k_species=5,
    )
    base.update(overrides)
    return RecognitionConfig(**base)


@pytest.fixture
def config() -> RecognitionConfig:
    return make_config()


@pytest.fixture
def validation_config(config) -> ValidationConfig:
    return config.validation


@pytest.fixture
def valid_png() -> bytes:
    return png_bytes()


@pytest.fixture
def valid_context(valid_png):
    return {"recognition_image": image_entry(valid_png)}


# --- domain builders -------------------------------------------------------

def prediction(species_id: str, score: float, name: str | None = None,
               common_name: str | None = None) -> BioCLIPTaxonPrediction:
    """One taxonomic label, as the classification boundary would return it."""
    return BioCLIPTaxonPrediction(
        species_id=species_id,
        scientific_name=name or species_id.replace("_", " ").capitalize(),
        common_name=common_name,
        rank="species",
        classification_score=score,
    )


def candidate(species_id: str, score: float, name: str | None = None) -> SpeciesCandidate:
    return SpeciesCandidate(
        species_id=species_id,
        scientific_name=name or species_id.replace("_", " ").capitalize(),
        classification_score=score,
        taxonomy_status="unverified",
    )


class StubClassifier:
    """Returns exactly the labels a test hands it, so scenarios are exact.

    It implements the same `classify(image, top_k)` signature the real
    BioCLIP-2 provider will, which is the point: every workflow test below runs
    against the production interface, not a special test path.
    """

    provider_name = "StubClassifier"
    recognition_mode = "mock_classification"
    version = "stub-classifier-v1"

    def __init__(self, predictions=None, *, raises=None):
        self._predictions = list(predictions or [])
        self._raises = raises
        self.last_top_k: int | None = None
        self.calls = 0

    def classify(self, image, top_k):
        from ..domain.errors import RecognitionError

        self.calls += 1
        if self._raises is not None:
            raise RecognitionError(self._raises)
        self.last_top_k = top_k
        return list(self._predictions)[:top_k]
