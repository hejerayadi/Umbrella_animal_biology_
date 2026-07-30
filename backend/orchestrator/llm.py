"""Reusable Azure OpenAI wrapper.

Every orchestrator component (Planner, Capability Resolver) must obtain its
LLM through `get_llm()` instead of constructing its own client, so only one
Azure OpenAI connection is ever created per process.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load the secret keys (API key, endpoint URL) from the .env file that sits
# right next to this file, so this works no matter what folder you run the
# program from.
load_dotenv(Path(__file__).parent / ".env")


# @lru_cache(maxsize=1) means: run this function once, remember the result,
# and give everyone that same result next time instead of running it again.
# That's how we guarantee only ONE connection to Azure OpenAI ever gets made.
@lru_cache(maxsize=1)
def get_llm() -> AzureChatOpenAI:
    """Return the single shared Azure OpenAI chat model instance."""

    return AzureChatOpenAI(
        # Where your Azure OpenAI resource lives (from the .env file).
        azure_endpoint=os.environ["azure_endpoint"],
        # The secret key that proves we're allowed to call it.
        api_key=os.environ["openai_key_azure"],
        # Which version of the Azure API we're speaking. Can be overridden
        # with an environment variable, otherwise falls back to a sane default.
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        # The name of the specific model deployment on Azure we want to use.
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1"),
        # temperature=0 means "always give the most predictable answer",
        # which matters here because we want consistent routing decisions,
        # not creative/random ones.
        temperature=0,
    )
