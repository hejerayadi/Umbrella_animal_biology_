"""
Shared Qdrant access layer for all three cacheable subagents
(go_annotations, kegg_pathways, uniprot_proteins).

  same-text hash check -> unchanged: skip
                        -> changed/new: embed -> upsert
"""
import hashlib
import logging
import os
from typing import Any, Optional

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    _QDRANT_AVAILABLE = True
except ModuleNotFoundError:
    AsyncQdrantClient = Any  # type: ignore[assignment]
    Distance = PointStruct = VectorParams = Any  # type: ignore[assignment]
    _QDRANT_AVAILABLE = False

from kb.embeddings import embed_text, EMBEDDING_DIM

logger = logging.getLogger(__name__)

COLLECTIONS = ["go_annotations", "kegg_pathways", "uniprot_proteins"]

_client: Optional[AsyncQdrantClient] = None


def _qdrant_unavailable_warning() -> None:
    if not _QDRANT_AVAILABLE:
        logger.warning("qdrant-client is not installed; running without cache")


def get_client() -> AsyncQdrantClient:
    if not _QDRANT_AVAILABLE:
        _qdrant_unavailable_warning()
        raise RuntimeError("qdrant-client is not installed")

    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _client


async def ensure_collections() -> None:
    """Idempotent. Run once at API startup (see Step 7)."""
    if not _QDRANT_AVAILABLE:
        _qdrant_unavailable_warning()
        return

    client = get_client()
    existing = {c.name for c in (await client.get_collections()).collections}
    for name in COLLECTIONS:
        if name in existing:
            continue
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info("created Qdrant collection %s", name)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _point_id(dedup_key: str) -> str:
    # Qdrant point IDs must be UUID or unsigned int; hash our human dedup key into one.
    return hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()[:32]


async def get_cached(collection: str, dedup_key: str) -> Optional[dict[str, Any]]:
    if not _QDRANT_AVAILABLE:
        return None

    client = get_client()
    result = await client.retrieve(collection_name=collection, ids=[_point_id(dedup_key)])
    if not result:
        return None
    return result[0].payload


async def upsert_point(
    collection: str,
    dedup_key: str,
    text_to_embed: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    if not text_to_embed:
        logger.warning("skip upsert for %s: empty text_to_embed", dedup_key)
        return None
    """Re-embeds only if `text_to_embed` changed since the last upsert (hash check)."""
    if not _QDRANT_AVAILABLE:
        return payload

    client = get_client()
    point_id = _point_id(dedup_key)
    new_hash = _text_hash(text_to_embed)

    existing = await client.retrieve(collection_name=collection, ids=[point_id])
    if existing and existing[0].payload.get("text_hash") == new_hash:
        logger.info("skip re-embed for %s (unchanged)", dedup_key)
        return existing[0].payload

    vector = await embed_text(text_to_embed)
    full_payload = {**payload, "text_hash": new_hash, "dedup_key": dedup_key}
    await client.upsert(
        collection_name=collection,
        points=[PointStruct(id=point_id, vector=vector, payload=full_payload)],
    )
    logger.info("upserted %s into %s", dedup_key, collection)
    return full_payload