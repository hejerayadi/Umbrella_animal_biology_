from orchestrators.registry import load_agent_cards
"""
Graphe LangGraph séquentiel.

    START -> orchestrator_node -> execute_agents_node -> END

Pas de parallélisme : execute_agents_node exécute les agents un par un,
dans l'ordre décidé par le LLM (voir knowledge_discovery_orchestrator.route()).

Task delegation + response aggregation + communication avec les agents
enfants sont tous les trois réalisés dans execute_agents_node :
- delegation : chaque AgentCall (agent_name, capability, query) est transmis
  tel quel au module de l'agent correspondant
- communication : appel direct de `module.run(capability, query)`
- aggregation : les résultats sont concaténés dans l'ordre d'exécution

Le dispatch "quel agent_name -> quel module Python" se fait via un
dictionnaire (SUBAGENTS), pas via une chaîne de if/else : ce n'est pas une
décision métier, c'est juste une table de correspondance nom -> module.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from orchestrators.knowledge_discovery_orchestrator import route
from data_classes import AgentCall, AgentResponse
from agents import retrieval_processing_agent, scientific_analysis_agent

SUBAGENTS = {
    "retrieval_processing": retrieval_processing_agent,
    "scientific_analysis": scientific_analysis_agent,
}


class GraphState(TypedDict):
    query: str
    agent_calls: list[AgentCall]
    agent_responses: list[dict]
    final_answer: str


def orchestrator_node(state: GraphState) -> dict:
    decision = route(state["query"])
    calls = list(decision.calls)

    called_names = {c.agent_name for c in calls}
    cards = {c["name"]: c for c in load_agent_cards()}  # importer load_agent_cards depuis registry

    for call in list(calls):
        card = cards.get(call.agent_name)
        if not card:
            continue
        for dep in card.get("depends_on", []):
            if dep not in called_names:
                # insère la dépendance manquante juste avant l'agent qui en a besoin
                idx = calls.index(call)
                calls.insert(idx, AgentCall(
                    agent_name=dep,
                    capability="synthesis",  # capacité par défaut du prérequis
                    query=call.query,        # réutilise la même query faute de mieux
                ))
                called_names.add(dep)

    return {"agent_calls": calls}

def execute_agents_node(state: GraphState) -> dict:
    responses = []
    for call in state["agent_calls"]:
        module = SUBAGENTS.get(call.agent_name)
        if module is None:
            continue
        responses.append(module.run(call.capability, call.query))

    parts = [f"[{r['agent_name']}/{r['capability']}] {r['result']}" for r in responses]
    return {
        "agent_responses": responses,
        "final_answer": "\n\n".join(parts) if parts else "Aucun agent n'a été sélectionné.",
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("execute_agents_node", execute_agents_node)

    graph.add_edge(START, "orchestrator_node")
    graph.add_edge("orchestrator_node", "execute_agents_node")
    graph.add_edge("execute_agents_node", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    test_queries = [
        "Quelles sont les zones peu explorées dans la recherche sur la fertilité bovine ?",
        "Fais une synthèse des articles récents sur la sélection génomique bovine et dis-moi s'il existe des contradictions dans cette littérature.",
    ]

    for q in test_queries:
        print("=" * 80)
        print(f"Question: {q}\n")
        result = app.invoke({"query": q, "agent_calls": [], "agent_responses": [], "final_answer": ""})
        print("Agents exécutés (dans l'ordre):", [c.agent_name for c in result["agent_calls"]])
        print("\nRéponse finale:\n", result["final_answer"])
