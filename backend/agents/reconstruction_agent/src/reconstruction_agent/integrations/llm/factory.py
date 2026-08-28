"""Choosing the LLM client for a set of settings.

One decision, in one place: an unconfigured Azure deployment is the same thing
as `LLM_PROVIDER=none`, because a client that cannot answer is not usefully
different from one that will not. Deciding it here means no caller has to
re-derive it, and the health endpoint reports the same answer the graph acts on.
"""

from __future__ import annotations

from reconstruction_agent.config.settings import LlmProvider, Settings
from reconstruction_agent.integrations.llm.azure import AzureOpenAIClient
from reconstruction_agent.integrations.llm.base import LLMClient
from reconstruction_agent.integrations.llm.null import NullLLMClient


def build_llm_client(settings: Settings) -> LLMClient:
    """The client the agent should use, given what is configured."""
    if settings.llm.provider is LlmProvider.NONE:
        return NullLLMClient()

    client = AzureOpenAIClient(settings.azure_openai, settings.llm)
    return client if client.available else NullLLMClient()
