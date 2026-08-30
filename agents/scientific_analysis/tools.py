"""
Tools de l'agent Scientific Analysis.
Chaque tool = 1 fonction Python d'execution + 1 schema JSON (format Responses API).
Wrappers fins autour du RAG pipeline deja construit, avec gestion d'erreur pour
que le LLM recoive toujours une reponse exploitable, meme en cas d'echec technique.
"""
from knowledge_base.qdrant_client import get_client
from agents.scientific_analysis.qa_module import answer_question
from agents.scientific_analysis.contradiction_module import check_contradiction
from agents.scientific_analysis.gap_detection_module import detect_gap

_client = None


def _get_shared_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


def answer_scientific_question(question: str) -> str:
    """Answer a factual scientific question using the QA knowledge base (PubMedQA/BioASQ), with PubMed fallback."""
    try:
        client = _get_shared_client()
        context, results = answer_question(client, question, top_k=3)

        if not context.strip():
            return "AUCUNE_INFO: no relevant source found for this question."

        sources = [f"{r.payload.get('source', '?')} (score={r.score:.2f})" for r in results]
        return f"CONTEXT:\n{context}\n\nSOURCES: {', '.join(sources)}"

    except Exception as e:
        return f"ERREUR_OUTIL: could not retrieve information ({type(e).__name__}: {e})"


def check_claim_contradiction(claim: str) -> str:
    """Check whether a scientific claim is supported or contradicted by available evidence (SciFact)."""
    try:
        client = _get_shared_client()
        evidences = check_contradiction(client, claim, top_k=5)

        if not evidences:
            return "AUCUNE_INFO: no evidence found for this claim."

        lines = []
        for e in evidences:
            label = e.payload.get("label", "NEI")
            text = e.payload.get("text", "")[:200]
            lines.append(f"[{label}] (score={e.score:.2f}) {text}")

        return "EVIDENCES:\n" + "\n".join(lines)

    except Exception as e:
        return f"ERREUR_OUTIL: could not check this claim ({type(e).__name__}: {e})"


def detect_knowledge_gap(concept: str) -> str:
    """Determine whether a biological concept is well covered by the structured ORKG knowledge base, or under-researched."""
    try:
        client = _get_shared_client()
        result = detect_gap(client, concept)

        if result["gap_detected"]:
            score = result.get("best_score")
            score_txt = f"{score:.2f}" if score is not None else "none"
            return f"GAP_DETECTED: concept '{concept}' is poorly covered (best score={score_txt})."

        matches = result.get("matches", [])
        details = "; ".join(f"{m.payload.get('source', '?')} (score={m.score:.2f})" for m in matches)
        return f"NO_GAP: concept '{concept}' is already covered. Matches: {details}"

    except Exception as e:
        return f"ERREUR_OUTIL: could not evaluate this concept ({type(e).__name__}: {e})"


# Dispatch : nom du tool -> fonction Python correspondante
TOOL_DISPATCH = {
    "answer_scientific_question": answer_scientific_question,
    "check_claim_contradiction": check_claim_contradiction,
    "detect_knowledge_gap": detect_knowledge_gap,
}

# Schemas au format attendu par l'API Responses (tools=[...])
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "answer_scientific_question",
        "description": "Answer a factual scientific question using the QA knowledge base (PubMedQA/BioASQ), with PubMed fallback.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The scientific question to answer"}
            },
            "required": ["question"],
        },
    },
    {
        "type": "function",
        "name": "check_claim_contradiction",
        "description": "Check whether a scientific claim is supported or contradicted by available evidence (SciFact).",
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {"type": "string", "description": "The scientific claim to verify"}
            },
            "required": ["claim"],
        },
    },
    {
        "type": "function",
        "name": "detect_knowledge_gap",
        "description": "Determine whether a biological concept is well covered by the structured ORKG knowledge base, or under-researched.",
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {"type": "string", "description": "The biological concept or relationship to check"}
            },
            "required": ["concept"],
        },
    },
]