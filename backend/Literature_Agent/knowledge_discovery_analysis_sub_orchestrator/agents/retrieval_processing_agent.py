"""
Retrieval and Knowledge Processing Sub Agent.

TODO: connecter à Qdrant et implémenter réellement les capacités
synthesis et summary. Pour l'instant, squelette (stub).
"""

from data_classes import AgentResponse


def run(capability: str, query: str) -> AgentResponse:
    if capability not in ("synthesis", "summary"):
        return AgentResponse(
            agent_name="retrieval_processing",
            capability=capability,
            result="",
            success=False,
            error_message=f"Capability inconnue: {capability}",
        )

    # --- STUB : à remplacer par la vraie logique RAG/Qdrant ---
    result_text = f"[STUB] Résultat simulé pour capability='{capability}' sur la requête: {query}"

    return AgentResponse(
        agent_name="retrieval_processing",
        capability=capability,
        result=result_text,
        claims=[],
        success=True,
    )
