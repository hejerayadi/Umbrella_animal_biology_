"""Generation centralisee des embeddings (meme modele pour ingestion et recherche)."""
from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_MODEL

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    return _get_model().encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _get_model().encode(texts)]
