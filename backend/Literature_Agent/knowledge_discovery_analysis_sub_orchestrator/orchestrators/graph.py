"""
Graphe LangGraph du Knowledge Discovery & Analysis Sub-Orchestrator.

Structure :

    START → orchestrator_node → [retrieval_processing_node, scientific_analysis_node]* → aggregate_node → END

* Le nombre de branches activées est dynamique : décidé par orchestrator_node
  (via la fonction route() déjà validée), pas codé en dur. Si le LLM ne
  retourne qu'un seul AgentCall, une seule branche s'exécute. S'il en
  retourne deux (agents indépendants), LangGraph les exécute EN PARALLÈLE.

Pour un cas séquentiel (un agent dépend du résultat de l'autre), voir la
note en bas de fichier — ce graphe couvre le cas parallèle/simple d'abord.
"""

from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END

from orchestrators.knowledge_discovery_orchestrator import route
from data_classes import AgentCall, AgentResponse
from agents import retrieval_processing_agent, scientific_analysis_agent


# ---------------------------------------------------------------------------
# 1. État partagé du graphe
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    query: str
    agent_calls: list[AgentCall]
    # `operator.add` permet à plusieurs branches parallèles d'ajouter
    # chacune leur réponse à la même liste, sans écraser celle de l'autre.
    agent_responses: Annotated[list[AgentResponse], operator.add]
    final_answer: str


# ---------------------------------------------------------------------------
# 2. Nodes
# ---------------------------------------------------------------------------

def orchestrator_node(state: GraphState) -> dict:
    """Décide quel(s) agent(s) appeler via tool calling (route() déjà validé)."""
    decision = route(state["query"])
    return {"agent_calls": decision.calls}


def retrieval_processing_node(state: GraphState) -> dict:
    """Exécute l'agent retrieval_processing pour son AgentCall correspondant."""
    call = next(c for c in state["agent_calls"] if c.agent_name == "retrieval_processing")
    response = retrieval_processing_agent.run(call.capability, call.query)
    return {"agent_responses": [response]}


def scientific_analysis_node(state: GraphState) -> dict:
    """Exécute l'agent scientific_analysis pour son AgentCall correspondant."""
    call = next(c for c in state["agent_calls"] if c.agent_name == "scientific_analysis")
    response = scientific_analysis_agent.run(call.capability, call.query)
    return {"agent_responses": [response]}


def aggregate_node(state: GraphState) -> dict:
    """Fusionne les réponses de tous les agents appelés en une réponse finale."""
    parts = [f"[{r.agent_name}/{r.capability}] {r.result}" for r in state["agent_responses"]]
    final_answer = "\n\n".join(parts)
    return {"final_answer": final_answer}


# ---------------------------------------------------------------------------
# 3. Routage conditionnel (fan-out dynamique)
# ---------------------------------------------------------------------------

def dispatch(state: GraphState) -> list[str]:
    """
    Lit les agent_calls décidés par l'orchestrateur et retourne la liste
    des nodes à activer. Si 2 agents sont présents, LangGraph les lance
    EN PARALLÈLE automatiquement (fan-out natif). Si 1 seul, une seule
    branche s'exécute (comportement séquentiel simple).
    """
    agent_names = {c.agent_name for c in state["agent_calls"]}
    next_nodes = []
    if "retrieval_processing" in agent_names:
        next_nodes.append("retrieval_processing_node")
    if "scientific_analysis" in agent_names:
        next_nodes.append("scientific_analysis_node")

    if not next_nodes:
        # Aucun agent décidé (cas limite) → on saute directement à l'agrégation
        return ["aggregate_node"]
    return next_nodes


# ---------------------------------------------------------------------------
# 4. Construction du graphe
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("retrieval_processing_node", retrieval_processing_node)
    graph.add_node("scientific_analysis_node", scientific_analysis_node)
    graph.add_node("aggregate_node", aggregate_node)

    graph.add_edge(START, "orchestrator_node")

    # Fan-out dynamique : 1 ou 2 branches selon la décision de l'orchestrateur
    graph.add_conditional_edges(
        "orchestrator_node",
        dispatch,
        ["retrieval_processing_node", "scientific_analysis_node", "aggregate_node"],
    )

    # Les deux branches (qu'elles aient tourné ou non) convergent vers aggregate_node
    graph.add_edge("retrieval_processing_node", "aggregate_node")
    graph.add_edge("scientific_analysis_node", "aggregate_node")

    graph.add_edge("aggregate_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# 5. Test manuel
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    test_queries = [
        # Cas 1 : un seul agent (gap_detection)
        "Quelles sont les zones peu explorées dans la recherche sur la fertilité bovine ?",
        # Cas 2 : potentiellement les deux agents (à tester si le prompt orchestrateur
        # est adapté pour permettre plusieurs tool calls dans une seule réponse)
        "Fais-moi une synthèse des articles récents sur la sélection génomique bovine "
        "et dis-moi s'il existe des contradictions dans cette littérature.",
    ]

    for q in test_queries:
        print("=" * 80)
        print(f"Question: {q}\n")
        result = app.invoke({"query": q, "agent_calls": [], "agent_responses": [], "final_answer": ""})
        print("Agents appelés:", [c.agent_name for c in result["agent_calls"]])
        print("\nRéponse finale:\n", result["final_answer"])
