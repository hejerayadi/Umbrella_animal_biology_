"""
Tests du routage de l'orchestrateur.
Reprend les cas déjà validés manuellement en Colab.

Lancer avec : pytest tests/test_routing.py -v
"""

import pytest
from orchestrators.knowledge_discovery_orchestrator import route


def test_routes_to_gap_detection():
    query = "Quelles sont les zones peu explorées dans la recherche sur la fertilité bovine ?"
    decision = route(query)

    assert len(decision.calls) >= 1
    call = decision.calls[0]
    assert call.agent_name == "scientific_analysis"
    assert call.capability == "gap_detection"


def test_routes_to_synthesis():
    query = "Peux-tu me faire une synthèse des articles récents sur la sélection génomique bovine ?"
    decision = route(query)

    assert len(decision.calls) >= 1
    call = decision.calls[0]
    assert call.agent_name == "retrieval_processing"
    assert call.capability == "synthesis"


def test_routes_to_contradiction_detection():
    query = "Y a-t-il des contradictions dans la littérature sur l'effet de la sélection génomique sur la fertilité bovine ?"
    decision = route(query)

    assert len(decision.calls) >= 1
    call = decision.calls[0]
    assert call.agent_name == "scientific_analysis"
    assert call.capability == "contradiction_detection"
