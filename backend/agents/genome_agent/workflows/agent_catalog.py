from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_AGENT_CARDS_DIR = Path(__file__).parent.parent / "agent_cards"


def load_agent_catalog() -> str:
    """Read all agent_cards/*.json into a single text block for the LLM."""
    catalog_parts: list[str] = []
    for card_path in sorted(_AGENT_CARDS_DIR.glob("*.json")):
        try:
            data = json.loads(card_path.read_text(encoding="utf-8"))
            name = data.get("name", card_path.stem)
            description = data.get("description", "")
            capabilities = ", ".join(data.get("capabilities", []))
            managed = ", ".join(data.get("managed_services", []))
            role = data.get("role", "internal")
            call_when = data.get("call_when", "")
            cannot_help = ", ".join(data.get("cannot_help_with", []))

            part = f"- {name} ({role}): {description}\n"
            part += f"  Capabilities: {capabilities}\n"
            if managed:
                part += f"  Managed services: {managed}\n"
            if call_when:
                part += f"  Call when: {call_when}\n"
            if cannot_help:
                part += f"  Cannot help with: {cannot_help}\n"

            catalog_parts.append(part)
        except Exception as exc:
            logger.warning("Failed to load agent card %s: %s", card_path, exc)

    return "\n".join(catalog_parts)


def _get_known_agent_names() -> list[str]:
    names: list[str] = []
    for card_path in _AGENT_CARDS_DIR.glob("*.json"):
        try:
            data = json.loads(card_path.read_text(encoding="utf-8"))
            names.append(data.get("name", ""))
        except Exception:
            pass
    return names
