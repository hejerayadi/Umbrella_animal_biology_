from __future__ import annotations

from ..registry import AGENT_REGISTRY


class CapabilityResolver:
    """Map a requested capability to the first matching agent.

    The resolver is intentionally simple: it scans the registered agents and
    returns the name of the first agent whose declared capabilities include
    the requested capability string. If no agent advertises that capability,
    the resolver returns `None` and the caller can decide how to recover.
    """

    def resolve(self, capability: str) -> str | None:
        # Walk through the global agent registry in registration order.
        # The first agent that explicitly exposes the capability wins.
        for agent_name, card in AGENT_REGISTRY.items():
            if capability in card.capabilities:
                return agent_name

        # No registered agent claims this capability.
        return None

    