"""Fallback online : interroge PubMed quand Qdrant n'a pas assez d'info, puis indexe le resultat."""
import requests

from .qdrant_kb.ingestion import ingest_documents


def _query_pubmed(query: str, max_results: int = 5) -> list[dict]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    resp = requests.get(search_url, params=params, timeout=10)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    resp = requests.get(summary_url, params=params, timeout=10)
    resp.raise_for_status()
    result = resp.json().get("result", {})
    return [result[uid] for uid in ids if uid in result]


def _normalize(raw: dict) -> dict:
    return {
        "id": f"pubmed_{raw.get('uid')}",
        "text": raw.get("title", ""),
        "source": "PubMed",
        "publication_date": raw.get("pubdate"),
    }


def run_search_agent(client, query: str, collection_name: str, max_results: int = 5):
    try:
        raw_results = _query_pubmed(query, max_results=max_results)
    except Exception:
        raw_results = []

    if not raw_results:
        return []

    papers = [_normalize(r) for r in raw_results]
    ingest_documents(client, collection_name, papers)

    from .qdrant_kb.retrieval import search_local
    return search_local(client, collection_name, query)
