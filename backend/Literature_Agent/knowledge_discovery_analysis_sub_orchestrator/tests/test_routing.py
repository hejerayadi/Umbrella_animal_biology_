"""Tests du routage séquentiel. Lancer avec: pytest tests/test_routing.py -v"""

from orchestrators.graph import build_graph


def _run(query):
    app = build_graph()
    return app.invoke({"query": query, "agent_calls": [], "agent_responses": [], "final_answer": ""})


def test_single_agent_gap_detection():
    result = _run("Quelles sont les zones peu explorées dans la recherche sur la fertilité bovine ?")
    names = [c.agent_name for c in result["agent_calls"]]
    assert "scientific_analysis" in names


def test_single_agent_synthesis():
    result = _run("Peux-tu me faire une synthèse des articles récents sur la sélection génomique bovine ?")
    names = [c.agent_name for c in result["agent_calls"]]
    assert "retrieval_processing" in names


def test_multi_agent_sequential_order():
    result = _run(
        "Fais une synthèse des articles récents sur la sélection génomique bovine "
        "et dis-moi s'il existe des contradictions dans cette littérature."
    )
    names = [c.agent_name for c in result["agent_calls"]]
    assert "retrieval_processing" in names
    assert "scientific_analysis" in names
    # Vérifie que l'exécution s'est faite dans l'ordre (pas en parallèle) :
    # les 2 réponses doivent apparaître dans final_answer dans le même ordre que agent_calls.
    order_in_answer = [n for n in names if n in result["final_answer"]]
    assert order_in_answer == names
