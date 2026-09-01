"""Chunking + construction des points + upsert batch vers Qdrant."""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ..config import CHUNK_MAX_CHARS, CHUNK_OVERLAP
from .embeddings import embed_batch

_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _to_qdrant_id(readable_id: str) -> str:
    """Qdrant exige un entier non-signe ou un UUID comme ID de point."""
    return str(uuid.uuid5(_UUID_NAMESPACE, readable_id))


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars - overlap
    return chunks


def ingest_documents(client: QdrantClient, collection_name: str, documents: list[dict]) -> int:
    points: list[PointStruct] = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        vectors = embed_batch(chunks)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            readable_id = f"{doc['id']}_{i}"
            points.append(PointStruct(
                id=_to_qdrant_id(readable_id),
                vector=vector,
                payload={**doc, "doc_chunk_id": readable_id, "chunk_index": i, "text": chunk},
            ))
    if not points:
        return 0
    client.upsert(collection_name=collection_name, points=points)
    return len(points)
