"""
Embedding generation for the three Qdrant collections.
Swap this implementation for the team's shared embedding endpoint once it exists —
nothing else in kb/ or subagents/ needs to change (they only call embed_text()).
"""
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2 output size


async def embed_text(text: str) -> list[float]:
    return _model.encode(text, normalize_embeddings=True).tolist()