"""
Knowledge Discovery & Analysis Sub-Orchestrator.

Rôle : reçoit une question utilisateur, et décide (via tool calling)
quel(s) agent(s) enfant(s) doivent la traiter. Ne répond JAMAIS
directement à la question — il route uniquement.
"""

import json
from llm import get_azure_client, get_deployment_name
from orchestrators.registry import build_tools_from_cards
from data_classes import AgentCall, RoutingDecision

SYSTEM_PROMPT = (
    "Tu es le Knowledge Discovery & Analysis Sub-Orchestrator. "
    "Ton seul rôle est de déléguer la question de l'utilisateur à l'agent enfant "
    "le plus approprié, en utilisant les tools disponibles. "
    "Tu ne réponds jamais toi-même à la question sur le fond."
)


def route(user_query: str) -> RoutingDecision:
    """
    Envoie la question de l'utilisateur au LLM, qui décide via tool calling
    quel(s) agent(s) appeler. Retourne une RoutingDecision structurée.
    """
    client = get_azure_client()
    deployment_name = get_deployment_name()
    tools = build_tools_from_cards()

    response = client.responses.create(
        model=deployment_name,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        tools=tools,
    )

    calls = []
    reasoning_text = None

    for item in response.output:
        if item.type == "reasoning":
            # Le contenu du raisonnement n'est pas toujours exposé (souvent une liste vide)
            summary = getattr(item, "summary", None)
            if summary:
                reasoning_text = str(summary)
        elif item.type == "function_call":
            args = json.loads(item.arguments)
            calls.append(
                AgentCall(
                    agent_name=item.name,
                    capability=args["capability"],
                    query=args["query"],
                )
            )

    return RoutingDecision(calls=calls, reasoning=reasoning_text)


if __name__ == "__main__":
    # Test manuel rapide : python -m orchestrators.knowledge_discovery_orchestrator
    test_query = "Quelles sont les zones peu explorées dans la recherche sur la fertilité bovine ?"
    decision = route(test_query)

    print(f"Question: {test_query}\n")
    for call in decision.calls:
        print(f"→ Agent: {call.agent_name}")
        print(f"  Capability: {call.capability}")
        print(f"  Query: {call.query}")
