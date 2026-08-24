"""
Embedding generation for the three Qdrant collections.
Swap this implementation for the team's shared embedding endpoint once it exists —
nothing else in kb/ or subagents/ needs to change (they only call embed_text()).

sentence-transformers is optional at runtime, the same way qdrant-client is in
qdrant_store.py: both exist to make the KB cache work, and the agent answers
correctly without a cache — just more slowly, re-fetching from QuickGO, KEGG and
UniProt each time. Importing it at module scope made it mandatory in practice,
because kb.qdrant_store imports this module and every subagent imports that, so
one uninstalled optional package took the whole agent's HTTP service down.

Constructing the model is deferred for a second reason: SentenceTransformer
downloads ~90 MB of weights the first time it runs, and at module scope that
download happens during uvicorn startup, before the port is even open.
"""
import logging

EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2 output size
_MODEL_NAME = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ModuleNotFoundError:
    SentenceTransformer = None  # type: ignore[assignment]
    EMBEDDINGS_AVAILABLE = False

_model = None


def _get_model():
    global _model
    if _model is None:
        logger.info("loading embedding model %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


async def embed_text(text: str) -> list[float]:
    if not EMBEDDINGS_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is not installed; cannot embed text for the KB cache"
        )
    return _get_model().encode(text, normalize_embeddings=True).tolist()
