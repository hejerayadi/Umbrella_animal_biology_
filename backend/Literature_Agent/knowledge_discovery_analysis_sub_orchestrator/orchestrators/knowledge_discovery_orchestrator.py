# orchestrators/knowledge_discovery_orchestrator.py
import json
from llm import get_azure_client, get_deployment_name
from data_classes import AgentCall, RoutingDecision
from orchestrators.registry import load_agent_cards


def route(user_query: str) -> RoutingDecision:
    client = get_azure_client()
    deployment_name = get_deployment_name()
    cards = load_agent_cards()

    catalog_lines = []
    for card in cards:
        caps = ", ".join(c["id"] for c in card["capabilities"])
        deps = card.get("depends_on", [])
        dep_txt = f" (dépend de: {', '.join(deps)})" if deps else " (aucune dépendance)"
        catalog_lines.append(f"- {card['name']}{dep_txt} — capacités: {caps}")
    catalog = "\n".join(catalog_lines)

    system_prompt = (
        "Tu es un planificateur de tâches pour un système multi-agents.\n\n"
        f"Agents disponibles:\n{catalog}\n\n"
        "Détermine la séquence COMPLÈTE et ORDONNÉE des étapes nécessaires pour "
        "répondre à la question. Respecte les dépendances : un agent qui dépend "
        "d'un autre doit apparaître APRÈS lui dans la séquence.\n"
        "Vérifie bien que la DERNIÈRE étape couvre le vrai besoin de la question "
        "(ex: si on demande des lacunes de recherche, la séquence doit se terminer "
        "par gap_detection, pas s'arrêter à la préparation).\n\n"
        'Réponds UNIQUEMENT avec un JSON : {"steps": [{"agent_name": "...", '
        '"capability": "...", "query": "..."}]}'
    )

    response = client.responses.create(
        model=deployment_name,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )

    data = json.loads(response.output_text)
    calls = [AgentCall(**step) for step in data.get("steps", [])]
    return RoutingDecision(calls=calls)