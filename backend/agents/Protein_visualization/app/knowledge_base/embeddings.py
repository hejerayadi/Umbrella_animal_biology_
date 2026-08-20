import asyncio
import hashlib
import math
import re
from typing import Any, Protocol

DEFAULT_DIMENSIONS = 1024


class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int: ...

    async def embed(self, text: str) -> list[float]: ...


class HashEmbedding:
    """Deterministic offline embedding. Test and CI use only — never production recall."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest[:4], "big") % self._dimensions] += -1.0 if digest[4] & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class BgeM3Embedding:
    """BAAI/bge-m3 through sentence-transformers, loaded lazily on first use.

    The model is CPU-heavy, so encoding runs in a worker thread to keep the event
    loop free while the workflow's other branches are in flight.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self.model_name = model_name
        self._dimensions = dimensions
        self._model: Any | None = None

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _encode(self, text: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vector: list[float] = self._model.encode(text, normalize_embeddings=True).tolist()
        return vector

    async def embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._encode, text)
