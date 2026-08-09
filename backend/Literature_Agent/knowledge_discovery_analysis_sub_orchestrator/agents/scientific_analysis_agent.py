"""
Scientific Analysis Sub Agent.

TODO: connecter à Qdrant (recherche vectorielle sur le corpus scientifique)
et implémenter réellement les 3 capacités : qa, contradiction_detection, gap_detection.
Pour l'instant, squelette qui retourne une réponse simulée (stub)
pour valider la communication avec l'orchestrateur.
"""

from data_classes import AgentResponse


def run(capability: str, query: str) -> AgentResponse:
    """
    Point d'entrée de l'agent. Reçoit la capability et la query
    décidées par l'orchestrateur, exécute le traitement correspondant.
    """
    if capability not in ("qa", "contradiction_detection", "gap_detection"):
        return AgentResponse(
            agent_name="scientific_analysis",
            capability=capability,
            result="",
            success=False,
            error_message=f"Capability inconnue: {capability}",
        )

    # --- STUB : à remplacer par la vraie logique RAG/Qdrant ---
    result_text = f"[STUB] Résultat simulé pour capability='{capability}' sur la requête: {query}"

    return AgentResponse(
        agent_name="scientific_analysis",
        capability=capability,
        result=result_text,
        claims=[],
        success=True,
    )
