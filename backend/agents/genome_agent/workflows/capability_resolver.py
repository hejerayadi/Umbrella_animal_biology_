from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .agent_catalog import _AGENT_CARDS_DIR, load_agent_catalog
from .llm import get_llm_client

logger = logging.getLogger(__name__)


class CapabilityResolverOutput(BaseModel):
    target_agent: str = Field(description="Name of the agent that can resolve the missing capability.")
    handoff_message: str = Field(
        description="A short message explaining what data or action is needed from the target agent."
    )


_RESOLVER_SYSTEM_PROMPT = (
    "You are the capability resolver for the Genome Agent. "
    "Given a description of what is missing and what is already known, "
    "choose the best agent from the catalog to fulfill the request.\n\n"
    "Available agents:\n{agent_catalog}\n\n"
    "Rules:\n"
    "- Only return an agent that is explicitly listed in the catalog.\n"
    "- If no agent in the catalog can help, return 'none' as the target_agent.\n"
    "- The handoff_message should be specific about what data or action is needed.\n"
)


def resolve_capability(
    current_agent: str,
    prompt_to_target_agent: str,
    known_context: dict[str, Any],
) -> CapabilityResolverOutput | None:
    """Use the LLM to pick a target agent from the catalog."""
    catalog = load_agent_catalog()
    prompt = _RESOLVER_SYSTEM_PROMPT.format(agent_catalog=catalog)

    user_message = (
        f"Current agent: {current_agent}\n"
        f"Missing capability: {prompt_to_target_agent}\n"
        f"Known context: {json.dumps(known_context, default=str)}\n"
        f"Which agent from the catalog should handle this?"
    )

    try:
        client = get_llm_client()
    except Exception as exc:
        logger.warning("LLM client unavailable: %s", exc)
        return None

    bound = client.bind_tools([CapabilityResolverOutput], tool_choice="CapabilityResolverOutput")

    try:
        response = bound.invoke([SystemMessage(content=prompt), HumanMessage(content=user_message)])
        tool_calls = response.tool_calls or []
        if tool_calls:
            call = tool_calls[0]
            result = CapabilityResolverOutput(**call["args"])
            if _is_valid_target(result.target_agent):
                return result
            logger.warning("LLM returned invalid target agent: %s", result.target_agent)
    except Exception as exc:
        logger.warning("LLM capability resolver failed: %s", exc)

    return None


def resolve_capability_fallback(
    current_agent: str,
    prompt_to_target_agent: str,
) -> CapabilityResolverOutput:
    """Keyword-based fallback when the LLM tool call fails."""
    known_targets = _get_known_agent_names()
    prompt_lower = prompt_to_target_agent.lower()

    for target in known_targets:
        if target.lower() in prompt_lower:
            return CapabilityResolverOutput(
                target_agent=target,
                handoff_message=f"Resolve missing capability: {prompt_to_target_agent}",
            )

    capability_keywords = {
        "3d": "Protein Structure Visualization Agent",
        "protein": "Protein Structure Visualization Agent",
        "structure": "Protein Structure Visualization Agent",
        "reconstruct": "Reconstruction Agent",
        "genome": "Reconstruction Agent",
        "literature": "Literature Agent",
        "paper": "Literature Agent",
    }

    for keyword, target in capability_keywords.items():
        if keyword in prompt_lower and target in known_targets:
            return CapabilityResolverOutput(
                target_agent=target,
                handoff_message=f"Resolve missing capability: {prompt_to_target_agent}",
            )

    return CapabilityResolverOutput(
        target_agent="none",
        handoff_message=f"No agent in the catalog can resolve: {prompt_to_target_agent}",
    )


def _is_valid_target(target_agent: str) -> bool:
    """Confirm the target agent is actually present in the catalog."""
    known_targets = _get_known_agent_names()
    return target_agent.lower() in {t.lower() for t in known_targets}


def _get_known_agent_names() -> list[str]:
    names: list[str] = []
    for card_path in _AGENT_CARDS_DIR.glob("*.json"):
        try:
            data = json.loads(card_path.read_text(encoding="utf-8"))
            names.append(data.get("name", ""))
        except Exception:
            pass
    return names
