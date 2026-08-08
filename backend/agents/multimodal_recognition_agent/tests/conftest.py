"""Shared test fixtures.

Images are generated in-process with Pillow rather than committed as binary
files: the specification forbids committing raw images, and a generated one is
reproducible anyway.
"""
from __future__ import annotations

import base64
import hashlib
import io

import pytest
from PIL import Image, PngImagePlugin

from ..config import (
    QdrantConfig,
    RecognitionConfig,
    ThresholdConfig,
    ValidationConfig,
)
from ..domain.models import RetrievedReference, SpeciesCandidate


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
        qdrant=QdrantConfig(
            url=None,
            collection=None,
            vector_name=None,
            expected_dimension=None,
            expected_distance=None,
            dataset_version=None,
            top_k_references=None,
            timeout_seconds=10.0,
        ),
        bioclip_provider_mode="mock",
        mock_provider_version="local-dev-mock-bioclip2-v0",
        mock_embedding_dimension=32,
        mock_embedding_dimension_is_local_default=True,
        retrieval_mode="mock",
        taxonomy_provider_mode="mock",
        text_analyzer_mode="rules",
        top_k_species=5,
        max_references_per_species=3,
        image_seed_overrides={},
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

def reference(species_id: str, score: float, point_id: str = "p", name: str | None = None):
    return RetrievedReference(
        point_id=point_id,
        species_id=species_id,
        scientific_name=name or species_id.replace("_", " ").capitalize(),
        common_name=None,
        similarity_score=score,
        source="test",
        dataset_version="local-dev-fixtures-v0",
        embedding_mode="mock_bioclip2",
    )


def candidate(species_id: str, score: float, name: str | None = None, count: int = 1):
    return SpeciesCandidate(
        species_id=species_id,
        scientific_name=name or species_id.replace("_", " ").capitalize(),
        similarity_score=score,
        reference_count=count,
        taxonomy_status="unverified",
    )


class StubRetriever:
    """Returns exactly what a test hands it, so scenarios are exact."""

    provider_name = "StubRetriever"
    mode = "mock_local_development"

    def __init__(self, references, *, raises=None):
        self._references = references
        self._raises = raises
        self.last_top_k: int | None = None

    def search(self, query_vector, *, top_k):
        from ..adapters.retrieval import RetrievalOutcome
        from ..domain.errors import RecognitionError

        if self._raises is not None:
            raise RecognitionError(self._raises)
        self.last_top_k = top_k
        return RetrievalOutcome(
            references=list(self._references),
            provider=self.provider_name,
            mode=self.mode,
            collection=None,
            dataset_version="local-dev-fixtures-v0",
        )
