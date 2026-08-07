"""
Shared deterministic mock BioCLIP-2 provider — Sprint 2.

CRITICAL: this exact file (or a byte-identical copy) must be used on BOTH sides:
  - Chahd's ingestion pipeline  -> generates the reference vectors stored in Qdrant
  - Leith's Recognition Agent   -> generates the query vector at inference time

If the two sides diverge even slightly (different seed derivation, different
RNG, different normalization), retrieval will never produce meaningful
matches, even in this mocked Sprint 2 setup. Do not fork this file per agent.

embedding_mode        = "mock"
mock_provider_version = "sprint2-mock-bioclip2-v1"

These generated vectors are NOT scientific BioCLIP-2 embeddings and must
never be described or logged as such.
"""

import hashlib
import random

VECTOR_DIMENSION = 64
MOCK_PROVIDER_VERSION = "sprint2-mock-bioclip2-v1"
EMBEDDING_MODE = "mock"

# Optional fixture mapping for known demo images, keyed by sha256 of the raw
# image bytes. Populate this once the team agrees on the demo image set, so
# both ingestion and query sides return the EXACT same vector for those
# specific images (fully reproducible demo, no reliance on hash-seeding).
#
# Example:
#   "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a0": [0.12, -0.03, ...]
FIXTURE_VECTORS: dict[str, list[float]] = {
    # "<image_sha256>": [ ... VECTOR_DIMENSION floats ... ],
}


def _seeded_vector(seed_key: str, dimension: int = VECTOR_DIMENSION) -> list[float]:
    """Deterministically derive a unit-length vector from a seed string."""
    seed_int = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed_int)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
    norm = sum(x * x for x in raw) ** 0.5
    if norm == 0:
        raw[0] = 1.0
        norm = 1.0
    return [x / norm for x in raw]


def embed_image_bytes(image_bytes: bytes) -> list[float]:
    """
    Deterministic mock embedding for one image.

    - If the image's sha256 matches a FIXTURE_VECTORS entry, return that exact
      vector (used for the agreed demo images).
    - Otherwise, derive a seeded pseudo-random unit vector from the sha256
      hash of the image bytes.

    Test/demo use only — never represents real biological similarity.
    """
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    if image_sha256 in FIXTURE_VECTORS:
        return FIXTURE_VECTORS[image_sha256]
    return _seeded_vector(image_sha256)


def image_sha256_of(image_bytes: bytes) -> str:
    """Helper so both sides can compute/log the same hash key consistently."""
    return hashlib.sha256(image_bytes).hexdigest()
