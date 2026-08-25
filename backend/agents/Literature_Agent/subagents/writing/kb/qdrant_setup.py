"""Qdrant connection for the writing knowledge base.

The client is built on first use, never at import time. `retrieval.py` is
imported by `scientific_writing.py`, which sits on the import chain behind
`api.py`, so anything that raises here takes the whole agent down before
FastAPI can start - and no caller can catch an ImportError. Deferring
construction keeps a missing or wrong QDRANT_URL a call-time problem that
`_tool_*` in `scientific_writing.py` already degrades from. Same reasoning as
`llm/client.py`, which documents it at length for the Azure client.

The `.env` is addressed explicitly rather than by walking up from the working
directory. The service is started from the repository root, and `python -m
agents.Literature_Agent...` from `backend/`, so a bare `load_dotenv()` picks up
`backend/.env` - which has no QDRANT_URL - and this module would silently fall
back to localhost.
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

# parents[3] is Literature_Agent/ (kb -> writing -> subagents -> agent root).
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

EMBEDDING_MODEL = "sentence-transformers/all-minilm-l6-v2"

COLLECTIONS = {
    "ghaya_papers_fulltext": 384,
    "ghaya_related_work": 384,
    "ghaya_literature_reviews": 384,
    "ghaya_citation_examples": 384,
}


def is_configured() -> bool:
    """Whether a Qdrant call can be attempted. Never raises."""
    return bool(os.getenv("QDRANT_URL"))


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Build the client on first use and reuse it afterwards.

    Raises RuntimeError when QDRANT_URL is unset, rather than letting
    QdrantClient default to localhost:6333 and fail later as a connection
    timeout that reads like the cloud instance being down.
    """
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError(
            "QDRANT_URL is not set - the Literature Agent's writing knowledge "
            "base is not configured. See .env.example."
        )

    return QdrantClient(
        url=url,
        api_key=os.getenv("QDRANT_API_KEY"),
        cloud_inference=True,
        timeout=120,
    )


def create_collections():
    client = get_client()
    for name, dim in COLLECTIONS.items():
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Collection creee : {name}")
        else:
            print(f"Deja existante : {name}")


def create_payload_indexes():
    client = get_client()
    for name in [
        "ghaya_papers_fulltext",
        "ghaya_related_work",
        "ghaya_literature_reviews",
    ]:
        client.create_payload_index(name, "domain", PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "section_type", PayloadSchemaType.KEYWORD)
    client.create_payload_index("ghaya_citation_examples", "intent", PayloadSchemaType.KEYWORD)


if __name__ == "__main__":
    create_collections()
    create_payload_indexes()
