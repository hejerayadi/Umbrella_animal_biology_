from .client import get_client, COLLECTION_NAME




def search_qdrant(query_vector, limit: int = 50, query_filter=None):
    """Return the `limit` nearest journals to an already-embedded query.

    This is deliberately the only thing this module does: embedding lives
    in embeddings.embedder, ranking lives in ranking.ranker, and the
    pipeline that joins them lives in ingestion.retrieval.

    `query_filter` is an optional qdrant_client.models.Filter, used to
    restrict the search to journals matching a payload condition.
    """

    results = get_client().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    return results.points
