"""
Search Agent (Online) - flux de droite dans le schema :
    Query APIs / Web Search -> Parse Returned Papers -> Metadata Extraction -> Vector Database

C'est un FALLBACK : appele uniquement par retrieve_with_fallback() quand
la recherche locale dans Qdrant n'est pas suffisante (voir knowledge_base/retrieval.py).
"""
from qdrant_client import QdrantClient

from data_sources.api_clients import query_pubmed, normalize_pubmed_result
from knowledge_base.ingestion import ingest_documents


def run_search_agent(client: QdrantClient, query: str, collection_name: str, max_results: int = 5):
    """
    1. Interroge les APIs externes (PubMed par defaut, extensible a OpenAlex/EuropePMC).
    2. Parse et normalise les resultats bruts.
    3. Les indexe dans Qdrant pour que la prochaine requete similaire les trouve localement.
    4. Relance une recherche locale et retourne les resultats.
    """
    raw_results = query_pubmed(query, max_results=max_results)

    if not raw_results:
        print(f"[search_agent] Aucun resultat trouve en ligne pour '{query}'.")
        return []

    papers = [normalize_pubmed_result(r) for r in raw_results]
    ingest_documents(client, collection_name, papers)

    # Import local pour eviter une dependance circulaire avec retrieval.py
    from knowledge_base.retrieval import search_local
    return search_local(client, collection_name, query)
