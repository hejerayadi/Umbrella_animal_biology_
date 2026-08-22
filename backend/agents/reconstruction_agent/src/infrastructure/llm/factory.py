"""Builds the configured LLM client.

One place that knows how `LLM_PROVIDER` maps onto an implementation, so the
rest of the package depends only on the `LLMClient` protocol.
"""
from __future__ import annotations

from configuration.logging import get_logger
from configuration.settings import AzureSettings, LLMProvider, LLMSettings
from infrastructure.llm.client import (
    AzureOpenAIClient,
    LLMClient,
    NullLLMClient,
    OpenAILLMClient,
)

_log = get_logger(__name__)


def build_llm_client(
    settings: LLMSettings, azure: AzureSettings | None = None
) -> LLMClient:
    """The client for the configured provider.

    Falls back to `NullLLMClient` rather than raising when a provider is named
    but unusable (no key, package missing). The agent's deterministic path is a
    genuine fallback, so a misconfigured LLM degrades the run instead of
    failing it - but it logs a warning, because silently losing LLM planning is
    exactly the kind of thing that should be visible.
    """
    if settings.provider is LLMProvider.NONE:
        _log.info("llm_disabled", detail="Using the deterministic planner.")
        return NullLLMClient()

    if settings.provider in (LLMProvider.AZURE_FOUNDRY, LLMProvider.AZURE_OPENAI):
        return _build_azure(settings, azure)

    if settings.provider is LLMProvider.OPENAI:
        if not settings.api_key:
            _log.warning("llm_missing_key", provider=settings.provider.value)
            return NullLLMClient()
        try:
            return OpenAILLMClient(
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url,
                temperature=settings.temperature,
            )
        except RuntimeError as error:
            _log.warning("llm_build_failed", provider=settings.provider.value, error=str(error))
            return NullLLMClient()

    _log.warning("llm_unknown_provider", provider=str(settings.provider))
    return NullLLMClient()


def _build_azure(settings: LLMSettings, azure: AzureSettings | None) -> LLMClient:
    """Azure OpenAI / AI Foundry.

    Both provider values land here: Foundry fronts the same chat-completions
    surface, and the project endpoint is only needed for Foundry's other
    services, not for completions.
    """
    azure = azure or AzureSettings()

    if not azure.configured:
        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_BASE_URL", azure.openai_base_url),
                ("AZURE_OPENAI_DEPLOYMENT", azure.openai_deployment),
                ("AZURE_OPENAI_API_KEY", azure.openai_api_key),
            )
            if not value
        ]
        _log.warning(
            "azure_unconfigured",
            missing=missing,
            detail="Falling back to the deterministic planner.",
        )
        return NullLLMClient()

    assert azure.openai_base_url and azure.openai_deployment and azure.openai_api_key

    _log.info(
        "azure_llm_ready",
        deployment=azure.openai_deployment,
        api_version=azure.openai_api_version,
    )
    return AzureOpenAIClient(
        base_url=azure.openai_base_url,
        deployment=azure.openai_deployment,
        api_key=azure.openai_api_key,
        api_version=azure.openai_api_version,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
