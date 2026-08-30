"""
Tests basiques de validation.

Usage :
    python -m pytest tests/test_retrieval.py -v

Necessite un .env valide (QDRANT_URL / QDRANT_API_KEY) et que
run_offline_ingestion.py ait deja ete execute au moins une fois.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLLECTIONS
from knowledge_base.qdrant_client import get_client
from knowledge_base.retrieval import search_local, is_sufficient


def test_search_local_returns_results():
    client = get_client()
    results = search_local(client, COLLECTIONS["qa"], "muscle growth gene in cattle", top_k=3)
    assert len(results) > 0, "La recherche locale ne retourne aucun resultat."


def test_is_sufficient_logic():
    client = get_client()
    results = search_local(client, COLLECTIONS["qa"], "muscle growth gene in cattle", top_k=3)
    # On verifie juste que la fonction s'execute sans erreur et retourne un booleen
    assert isinstance(is_sufficient(results), bool)


if __name__ == "__main__":
    test_search_local_returns_results()
    test_is_sufficient_logic()
    print("Tous les tests sont passes.")
