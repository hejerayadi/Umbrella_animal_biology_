"""
Charge les agent cards (JSON) et les convertit au format 'tools' pour Azure OpenAI.
Ajouter un agent = ajouter un fichier JSON ici, sans toucher au code de l'orchestrateur.
"""

import json
from pathlib import Path

AGENT_CARDS_DIR = Path(__file__).parent.parent / "agent_cards"
CHILD_AGENT_FILES = ["retrieval_processing.json", "scientific_analysis.json"]


def load_agent_cards() -> list[dict]:
    cards = []
    for filename in CHILD_AGENT_FILES:
        path = AGENT_CARDS_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            cards.append(json.load(f))
    return cards


def build_tools_from_cards() -> list[dict]:
    cards = load_agent_cards()
    tools = []
    for card in cards:
        capability_ids = [c["id"] for c in card["capabilities"]]
        capability_descriptions = "; ".join(f"{c['id']} = {c['description']}" for c in card["capabilities"])

        description = card["description"]
        depends_on = card.get("depends_on", [])
        if depends_on:
            description += (
                f" DÉPENDANCE : nécessite que {', '.join(depends_on)} "
                f"ait été appelé AVANT, dans la même réponse."
            )

        tools.append({
            "type": "function",
            "name": card["name"],
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {
                        "type": "string",
                        "enum": capability_ids,
                        "description": capability_descriptions,
                    },
                    "query": {
                        "type": "string",
                        "description": "La requête reformulée à transmettre à cet agent",
                    },
                },
                "required": ["capability", "query"],
            },
        })
    return tools
