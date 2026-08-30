"""
Connecteurs vers les APIs scientifiques externes.
Utilises par le Search Agent (fallback online) quand Qdrant n'a pas assez d'info.
"""
import requests


def query_pubmed(query: str, max_results: int = 10) -> list[dict]:
    """Recherche sur PubMed via l'API E-utilities (NCBI)."""
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    resp = requests.get(search_url, params=params, timeout=10)
    resp.raise_for_status()
    ids = resp.json().get("esearchresult", {}).get("idlist", [])

    if not ids:
        return []

    # Recupere les resumes (titre + abstract) pour chaque id trouve
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    resp = requests.get(summary_url, params=params, timeout=10)
    resp.raise_for_status()
    result = resp.json().get("result", {})

    return [result[uid] for uid in ids if uid in result]


def query_openalex(query: str, max_results: int = 10) -> list[dict]:
    """Recherche sur OpenAlex."""
    url = "https://api.openalex.org/works"
    params = {"search": query, "per-page": max_results}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("results", [])


def query_europepmc(query: str, max_results: int = 10) -> list[dict]:
    """Recherche sur Europe PMC (texte integral souvent disponible)."""
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {"query": query, "format": "json", "pageSize": max_results}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json().get("resultList", {}).get("result", [])


def normalize_pubmed_result(raw: dict) -> dict:
    return {
        "id": f"pubmed_{raw.get('uid')}",
        "text": raw.get("title", ""),
        "source": "PubMed",
        "publication_date": raw.get("pubdate"),
    }


def normalize_openalex_result(raw: dict) -> dict:
    return {
        "id": f"openalex_{raw.get('id', '').split('/')[-1]}",
        "text": raw.get("title", ""),
        "source": "OpenAlex",
        "publication_date": raw.get("publication_date"),
    }


def normalize_europepmc_result(raw: dict) -> dict:
    return {
        "id": f"europepmc_{raw.get('id')}",
        "text": raw.get("title", "") + " " + raw.get("abstractText", ""),
        "source": "EuropePMC",
        "publication_date": raw.get("firstPublicationDate"),
    }
