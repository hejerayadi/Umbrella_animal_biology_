"""
Pipeline de recherche (retrieval) pour Writing Support -- recherche
vectorielle + filtrage par metadonnees sur les 3 collections Qdrant.

Usage (depuis backend/, avec le venv active) :
    python -m agents.Literature_Agent.subagents.writing.kb.retrieval
(lance une demo de recherche sur chaque collection)

Ou importe search_papers / search_related_work / search_citations
depuis un autre module.
"""

from qdrant_client.models import Document, Filter, FieldCondition, MatchValue

from .qdrant_setup import client, EMBEDDING_MODEL


def _build_filter(conditions: dict | None) -> Filter | None:
    """Construit un filtre Qdrant a partir d'un dict simple {champ: valeur}."""
    if not conditions:
        return None
    return Filter(
        must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in conditions.items()]
    )


def search_papers(query: str, section_type: str | None = None, domain: str | None = None, limit: int = 5):
    """Recherche dans ghaya_papers_fulltext (arXiv + PubMed).
    section_type: 'abstract' ou 'body'. domain: ex. 'biodiversity'."""
    filters = {}
    if section_type:
        filters["section_type"] = section_type
    if domain:
        filters["domain"] = domain

    return client.query_points(
        collection_name="ghaya_papers_fulltext",
        query=Document(text=query, model=EMBEDDING_MODEL),
        query_filter=_build_filter(filters),
        limit=limit,
    ).points


def search_related_work(query: str, limit: int = 5):
    """Recherche dans ghaya_related_work (Multi-XScience) -- pas de filtre
    thematique disponible (domain='general' partout)."""
    return client.query_points(
        collection_name="ghaya_related_work",
        query=Document(text=query, model=EMBEDDING_MODEL),
        limit=limit,
    ).points


def search_citations(query: str, intent: str | None = None, limit: int = 5):
    """Recherche dans ghaya_citation_examples (SciCite).
    intent: 'Background', 'Method', ou 'Result Comparison'."""
    filters = {"intent": intent} if intent else None

    return client.query_points(
        collection_name="ghaya_citation_examples",
        query=Document(text=query, model=EMBEDDING_MODEL),
        query_filter=_build_filter(filters),
        limit=limit,
    ).points


def print_results(points, label: str):
    print(f"\n{'='*60}", flush=True)
    print(f"{label} -- {len(points)} resultat(s)", flush=True)
    print("=" * 60, flush=True)
    for i, p in enumerate(points, 1):
        payload = p.payload
        preview = (payload.get("context") or payload.get("input_text") or "")[:200]
        print(f"\n[{i}] score={p.score:.3f}", flush=True)
        print(f"    {preview}...", flush=True)


if __name__ == "__main__":
    # Demo : une requete par collection pour valider que le retrieval fonctionne
    results = search_papers("wildlife conservation and habitat loss", section_type="abstract")
    print_results(results, "ghaya_papers_fulltext (abstract, filtre section_type)")

    results = search_related_work("temporal logic model checking")
    print_results(results, "ghaya_related_work")

    results = search_citations("previous studies have shown", intent="Background")
    print_results(results, "ghaya_citation_examples (filtre intent=Background)")