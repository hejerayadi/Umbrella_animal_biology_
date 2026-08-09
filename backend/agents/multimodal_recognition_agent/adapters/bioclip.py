"""The image-embedding boundary.

BioCLIP-2 is the officially selected model for the real recognition phase. It is
NOT executed in Sprint 2: no package is imported, no weights are downloaded, no
GPU is required. `MockBioCLIP2Provider` stands in for it and produces
deterministic test vectors.

Those vectors are not embeddings. They encode nothing biological. Their only
job is to be reproducible, so that the same image always retrieves the same
references and the workflow can be tested end to end.

------------------------------------------------------------------------------
THE VECTOR ALGORITHM - the part that must match Chahd's ingestion exactly
------------------------------------------------------------------------------

Query and reference vectors must be produced by identical code, or the
similarity between them is meaningless. Not "similar code" - identical output
for identical input. This is specified without reference to any language's
random number generator, so it can be reimplemented anywhere and still agree:

    seed_material = utf8(seed_string)
    block(i)      = SHA256(seed_material || uint32_be(i))     for i = 0, 1, 2, ...
    stream        = block(0) || block(1) || ...               (truncated to 4*D bytes)
    raw[j]        = int32_be(stream[4j : 4j+4])               signed, big-endian
    vector[j]     = raw[j] / 2^31                             in [-1, 1)
    result        = vector / ||vector||2                      L2-normalised

`D` is the collection's vector dimension. The seed string is:

  - a reference point:  its agreed seed (e.g. "panthera_leo/ref-001")
  - a query image:      the agreed seed when the image's SHA-256 is a known
                        fixture, otherwise "sha256:" + the image's SHA-256

The SHA-256 in that last line is the one defined in `validation.decode_image`:
over the base64-decoded encoded file bytes. Both sides must agree on that too.

None of this is frozen. It is a concrete proposal for Chahd to accept, amend or
replace, and until that happens the retrieval path is local development only.

A note on choosing `D`, measured rather than guessed: two unrelated vectors from
this generator have a cosine standard deviation of roughly 1/sqrt(D). Over 500
unknown query images against the local fixture set, the highest AGGREGATED
species score was 0.483 at D=32, 0.425 at D=64, 0.329 at D=128 and 0.232 at
D=256, with zero false identifications at any of them. D=32 is therefore
workable, but it leaves the least headroom above the `uncertain` floor. A larger
dimension buys separation cheaply, which is worth weighing when the manifest
fixes the real value.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Protocol

from ..domain.errors import ErrorCode, RecognitionError
from ..domain.models import NormalizedRecognitionInput

# Prefix for images with no agreed fixture seed. Keeping it explicit means a
# fixture seed and a hash seed can never collide.
UNKNOWN_IMAGE_SEED_PREFIX = "sha256:"


def deterministic_vector(seed: str, dimension: int) -> list[float]:
    """The shared algorithm above. Pure, and identical on every platform."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")

    material = seed.encode("utf-8")
    needed = dimension * 4
    stream = bytearray()
    counter = 0
    while len(stream) < needed:
        stream.extend(hashlib.sha256(material + struct.pack(">I", counter)).digest())
        counter += 1

    raw = struct.unpack(f">{dimension}i", bytes(stream[:needed]))
    vector = [value / 2147483648.0 for value in raw]  # 2^31

    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        # Astronomically unlikely, but a zero vector has no direction and cosine
        # similarity against it is undefined.
        raise ValueError("degenerate zero vector")
    return [component / norm for component in vector]


class ImageEmbeddingProvider(Protocol):
    """What the workflow needs from an embedding model."""

    version: str

    def embed_image(self, image: NormalizedRecognitionInput) -> list[float]:
        ...


class MockBioCLIP2Provider:
    """Deterministic stand-in for BioCLIP-2. Executes no model.

    Two paths, both reproducible:

    - a known demo image (its SHA-256 appears in `image_seed_overrides`) is
      mapped to an agreed fixture seed, so it retrieves a known reference;
    - anything else is seeded from its own hash, which lands it far from every
      stored reference - the honest outcome for an image nothing was ingested for.
    """

    embedding_mode = "mock"
    provider_name = "MockBioCLIP2Provider"

    def __init__(
        self,
        dimension: int,
        version: str,
        image_seed_overrides: dict[str, str] | None = None,
    ) -> None:
        self.dimension = dimension
        self.version = version
        self._overrides = dict(image_seed_overrides or {})

    def seed_for(self, image_sha256: str) -> str:
        """The seed string this image resolves to. Hashes only - no image data."""
        return self._overrides.get(image_sha256, UNKNOWN_IMAGE_SEED_PREFIX + image_sha256)

    def embed_image(self, image: NormalizedRecognitionInput) -> list[float]:
        vector = deterministic_vector(self.seed_for(image.image_sha256), self.dimension)

        # Belt and braces: a query vector of the wrong length would be rejected
        # by the collection anyway, and failing here says why.
        if len(vector) != self.dimension:
            raise RecognitionError(ErrorCode.EMBEDDING_DIMENSION_MISMATCH)
        return vector
