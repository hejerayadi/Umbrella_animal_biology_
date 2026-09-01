"""Tools de l'agent Scientific Analysis, branches sur le RAG pipeline Qdrant."""
from .qdrant_kb.client import get_client
from .qdrant_kb.retrieval import retrieve_with_fallback, search_local
from .config import COLLECTIONS

_client = None


def _get_shared_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def answer_scientific_question(question: str) -> str:
    """Answer a factual scientific question using the QA knowledge base, with PubMed fallback."""
    try:
        client = _get_shared_client()
        results = retrieve_with_fallback(client, COLLECTIONS["qa"], question, top_k=3)
        if not results:
            return "AUCUNE_INFO: no relevant source found for this question."
        context = "\n".join(r.payload.get("text", "") for r in results)
        sources = [f"{r.payload.get('source', '?')} (score={r.score:.2f})" for r in results]
        return f"CONTEXT:\n{context}\n\nSOURCES: {', '.join(sources)}"
    except Exception as e:
        return f"ERREUR_OUTIL: could not retrieve information ({type(e).__name__}: {e})"


def check_claim_contradiction(claim: str) -> str:
    """Check whether a scientific claim is supported or contradicted by available evidence (SciFact)."""
    try:
        client = _get_shared_client()
        evidences = retrieve_with_fallback(client, COLLECTIONS["contradiction"], claim, top_k=5)
        if not evidences:
            return "AUCUNE_INFO: no evidence found for this claim."
        lines = [
            f"[{e.payload.get('label', 'NEI')}] (score={e.score:.2f}) {e.payload.get('text', '')[:200]}"
            for e in evidences
        ]
        return "EVIDENCES:\n" + "\n".join(lines)
    except Exception as e:
        return f"ERREUR_OUTIL: could not check this claim ({type(e).__name__}: {e})"


def detect_knowledge_gap(concept: str) -> str:
    """Determine whether a biological concept is well covered by the structured ORKG knowledge base."""
    try:
        client = _get_shared_client()
        results = search_local(client, COLLECTIONS["gap"], concept, top_k=5)
        if not results or results[0].score < 0.5:
            score = results[0].score if results else None
            score_txt = f"{score:.2f}" if score is not None else "none"
            return f"GAP_DETECTED: concept '{concept}' is poorly covered (best score={score_txt})."
        details = "; ".join(f"{m.payload.get('source', '?')} (score={m.score:.2f})" for m in results)
        return f"NO_GAP: concept '{concept}' is already covered. Matches: {details}"
    except Exception as e:
        return f"ERREUR_OUTIL: could not evaluate this concept ({type(e).__name__}: {e})"


TOOL_DISPATCH = {
    "answer_scientific_question": answer_scientific_question,
    "check_claim_contradiction": check_claim_contradiction,
    "detect_knowledge_gap": detect_knowledge_gap,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "answer_scientific_question",
        "description": "Answer a factual scientific question using the QA knowledge base (PubMedQA/BioASQ), with PubMed fallback.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The scientific question to answer"}},
            "required": ["question"],
        },
    },
    {
        "type": "function",
        "name": "check_claim_contradiction",
        "description": "Check whether a scientific claim is supported or contradicted by available evidence (SciFact).",
        "parameters": {
            "type": "object",
            "properties": {"claim": {"type": "string", "description": "The scientific claim to verify"}},
            "required": ["claim"],
        },
    },
    {
        "type": "function",
        "name": "detect_knowledge_gap",
        "description": "Determine whether a biological concept is well covered by the structured ORKG knowledge base, or under-researched.",
        "parameters": {
            "type": "object",
            "properties": {"concept": {"type": "string", "description": "The biological concept or relationship to check"}},
            "required": ["concept"],
        },
    },
]
