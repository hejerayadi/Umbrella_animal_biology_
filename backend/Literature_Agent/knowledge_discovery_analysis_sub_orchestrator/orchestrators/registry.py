"""
Registry : charge les agent cards (JSON) et les convertit
au format 'tools' attendu par l'API Azure OpenAI (function calling).

L'orchestrateur ne connaît jamais la liste des agents en dur dans son code :
il la lit depuis ce registry, qui lit lui-même les fichiers JSON dans agent_cards/.
Ça permet d'ajouter un nouvel agent sans toucher au code de l'orchestrateur.
"""

import json
from pathlib import Path

AGENT_CARDS_DIR = Path(__file__).parent.parent / "agent_cards"

# Les agents enfants gérés par CE sous-orchestrateur
# (l'orchestrateur lui-même n'est pas un tool ici, c'est un point d'entrée)
CHILD_AGENT_FILES = ["retrieval_processing.json", "scientific_analysis.json"]


def load_agent_cards() -> list[dict]:
    """Charge toutes les agent cards enfants depuis le dossier agent_cards/."""
    cards = []
    for filename in CHILD_AGENT_FILES:
        path = AGENT_CARDS_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            cards.append(json.load(f))
    return cards


def build_tools_from_cards() -> list[dict]:
    """
    Convertit les agent cards JSON en format 'tools' compatible
    avec l'API Azure OpenAI (function calling).
    """
    cards = load_agent_cards()
    tools = []

    for card in cards:
        capability_ids = [c["id"] for c in card["capabilities"]]
        capability_descriptions = "; ".join(
            f"{c['id']} = {c['description']}" for c in card["capabilities"]
        )

        tools.append({
            "type": "function",
            "name": card["name"],
            "description": card["description"],
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
