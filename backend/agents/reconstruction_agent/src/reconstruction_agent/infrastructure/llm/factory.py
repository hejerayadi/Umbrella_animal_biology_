"""Builds the configured LLM client.

One place that knows how `LLM_PROVIDER` maps onto an implementation, so the
rest of the package depends only on the `LLMClient` protocol.
"""
from __future__ import annotations

from ...configuration.logging import get_logger
from ...configuration.settings import LLMProvider, LLMSettings
from .client import LLMClient, NullLLMClient, OpenAILLMClient

_log = get_logger(__name__)


def build_llm_client(settings: LLMSettings) -> LLMClient:
    """The client for the configured provider.

    Falls back to `NullLLMClient` rather than raising when a provider is named
    but unusable (no key, package missing). The agent's deterministic path is a
    genuine fallback, so a misconfigured LLM degrades the run instead of
    failing it - but it logs a warning, because silently losing LLM planning is
    exactly the kind of thing that should be visible.
    """
    if settings.provider is LLMProvider.NONE:
        _log.info("LLM_PROVIDER=none; using the deterministic planner.")
        return NullLLMClient()

    if not settings.api_key:
        _log.warning(
            "LLM_PROVIDER=%s but LLM_API_KEY is empty; falling back to the "
            "deterministic planner.",
            settings.provider.value,
        )
        return NullLLMClient()

    try:
        if settings.provider in (LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI):
            return OpenAILLMClient(
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url,
                temperature=settings.temperature,
            )
    except RuntimeError as error:
        _log.warning("Could not build the %s client (%s); falling back.", settings.provider, error)
        return NullLLMClient()

    _log.warning("Unhandled LLM provider %r; falling back.", settings.provider)
    return NullLLMClient()
