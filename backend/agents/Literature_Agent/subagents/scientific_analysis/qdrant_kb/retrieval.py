"""Recherche semantique avec fallback online (PubMed) si score local insuffisant."""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from ..config import SCORE_THRESHOLD
from .embeddings import embed_text


def search_local(client: QdrantClient, collection_name: str, query: str, top_k: int = 5, filters: Filter | None = None):
    query_vector = embed_text(query)
    return client.query_points(
        collection_name=collection_name, query=query_vector, query_filter=filters, limit=top_k
    ).points


def is_sufficient(results, threshold: float = SCORE_THRESHOLD) -> bool:
    return bool(results) and results[0].score >= threshold


def retrieve_with_fallback(client: QdrantClient, collection_name: str, query: str, top_k: int = 5, filters: Filter | None = None):
    results = search_local(client, collection_name, query, top_k, filters)
    if not is_sufficient(results):
        from ..search_agent import run_search_agent
        results = run_search_agent(client, query, collection_name)
    return results
