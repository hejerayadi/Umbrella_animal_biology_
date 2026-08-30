"""
Module Contradiction Detection : recupere les evidences liees a un claim
depuis contradiction_claims (SciFact).

La classification finale SUPPORT / REFUTE / NEI necessite un modele NLI
(non inclus ici) applique sur chaque evidence recuperee.
"""
from qdrant_client import QdrantClient

from config import COLLECTIONS
from knowledge_base.retrieval import retrieve_with_fallback


def check_contradiction(client: QdrantClient, claim: str, top_k: int = 5):
    """Recupere les evidences les plus proches semantiquement du claim donne."""
    evidences = retrieve_with_fallback(client, COLLECTIONS["contradiction"], claim, top_k=top_k)
    return evidences
