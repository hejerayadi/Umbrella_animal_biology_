"""Connexion Qdrant et creation des collections du Scientific Analysis agent."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ..config import QDRANT_URL, QDRANT_API_KEY, VECTOR_SIZE, COLLECTIONS

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def create_all_collections(client: QdrantClient) -> None:
    for name in COLLECTIONS.values():
        if client.collection_exists(name):
            continue
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
