"""
Module Gap Detection : compare un concept a la couverture existante
dans gap_orkg (ORKG). Un score de similarite faible sur toutes les
correspondances = signal de gap potentiel dans la litterature couverte.
"""
from qdrant_client import QdrantClient

from config import COLLECTIONS
from knowledge_base.retrieval import search_local


def detect_gap(client: QdrantClient, concept: str, score_threshold: float = 0.5):
    """
    Note : ici on utilise volontairement search_local() et non le fallback,
    car un score faible EST le signal qu'on cherche (le gap), pas une erreur
    a corriger en allant chercher en ligne.
    """
    results = search_local(client, COLLECTIONS["gap"], concept, top_k=5)

    if not results or results[0].score < score_threshold:
        return {"gap_detected": True, "concept": concept, "best_score": results[0].score if results else None}

    return {"gap_detected": False, "concept": concept, "matches": results}
