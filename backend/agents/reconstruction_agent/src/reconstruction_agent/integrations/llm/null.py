"""The client used when no model is configured.

Not a stub and not a test double: this is the production client for
`LLM_PROVIDER=none`, and the mode has to stay genuinely usable. The agent's
scientific conclusions are computed by deterministic scorers, so a run without
a model loses action *selection*, not correctness - the planner falls back to
the standard evidence-gathering order and the critic to rule-based diagnosis.

Returning None from every call is what routes callers onto that fallback.
"""

from __future__ import annotations

from reconstruction_agent.integrations.llm.base import StructuredT


class NullLLMClient:
    """Answers nothing, so every caller takes its deterministic path."""

    @property
    def available(self) -> bool:
        return False

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[StructuredT],
    ) -> StructuredT | None:
        return None
