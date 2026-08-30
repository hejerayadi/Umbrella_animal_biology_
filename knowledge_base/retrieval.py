"""
Recherche semantique dans Qdrant, avec logique de fallback :

    Requete -> Qdrant local
        -> score suffisant ?  OUI -> on retourne les resultats
                               NON -> Search Agent (online) -> on reindexe -> on retourne

C'est la traduction directe du schema "External Scientific Sources" fourni par l'utilisateur.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from config import SCORE_THRESHOLD
from knowledge_base.embeddings import embed_text


def search_local(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 5,
    filters: Filter | None = None,
):
    """Recherche vectorielle simple dans une collection Qdrant."""
    query_vector = embed_text(query)
    return client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=filters,
        limit=top_k,
    ).points


def is_sufficient(results, threshold: float = SCORE_THRESHOLD) -> bool:
    """Determine si le meilleur resultat local est assez bon pour ne PAS declencher le fallback."""
    if not results:
        return False
    return results[0].score >= threshold


def retrieve_with_fallback(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 5,
    filters: Filter | None = None,
):
    """
    Point d'entree principal a utiliser dans les modules de l'agent.

    1. Cherche d'abord dans Qdrant.
    2. Si le score est insuffisant, appelle le Search Agent (online),
       qui va interroger les APIs externes, parser les resultats,
       les indexer dans Qdrant, puis relancer la recherche locale.
    """
    results = search_local(client, collection_name, query, top_k, filters)

    if not is_sufficient(results):
        # Import local pour eviter une dependance circulaire au chargement du module.
        from pipelines.search_agent import run_search_agent

        print(f"[fallback] Rien de suffisant en local pour '{query}' -> recherche online...")
        results = run_search_agent(client, query, collection_name)

    return results
