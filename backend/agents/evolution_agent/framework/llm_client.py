"""LLM client wrapper for the Evolution Agent.

Tries backends in priority order and raises LLMUnavailable if none work.

Priority
--------
1. Azure OpenAI  — AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
2. Groq          — GROQ_API_KEY
3. GitHub Models — GITHUB_TOKEN

Environment variables
---------------------
AZURE_OPENAI_ENDPOINT    https://<resource>.openai.azure.com/   ← base only, no path
AZURE_OPENAI_API_KEY     your Azure key
AZURE_OPENAI_DEPLOYMENT  deployment name (default: gpt-5-mini)
AZURE_OPENAI_API_VERSION API version     (default: 2024-12-01-preview)

GROQ_API_KEY             your Groq key
GROQ_MODEL               model name (default: llama-3.3-70b-versatile)

GITHUB_TOKEN             your GitHub PAT
GITHUB_MODEL             model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Walk up from this file until a .env is found and load it."""
    here = Path(__file__).resolve()
    for parent in [here.parent.parent, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


_load_env()


class LLMUnavailable(RuntimeError):
    """No LLM backend credentials are configured."""


def get_llm(temperature: float | None = None):
    """Return the first available chat model backend."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key  = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()

    if endpoint and api_key and "PASTE_YOUR" not in api_key:
        return _azure(endpoint, api_key, temperature)

    if os.environ.get("GROQ_API_KEY"):
        return _groq(temperature)

    if os.environ.get("GITHUB_TOKEN"):
        return _github(temperature)

    raise LLMUnavailable(
        "No LLM backend configured.\n"
        "Open backend/agents/evolution_agent/.env and replace "
        "PASTE_YOUR_NEW_KEY_HERE with your real Azure API key."
    )


# Backend factories

def _azure(endpoint: str, api_key: str, temperature: float | None):
    from langchain_openai import AzureChatOpenAI  # type: ignore

    deployment  = os.environ.get("AZURE_OPENAI_DEPLOYMENT",  "gpt-5-mini")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    # Ensure the endpoint is the bare base URL — strip any path suffixes
    # like /openai/v1 that the portal sometimes shows.
    base = endpoint.split("/openai")[0].rstrip("/") + "/"

    # GPT-5 family only accepts temperature=1 (its default). Pin it unless
    # the caller explicitly overrides.
    temp = temperature if temperature is not None else 1

    return AzureChatOpenAI(
        azure_endpoint=base,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        timeout=60,
        max_retries=2,
        temperature=temp,
    )


def _groq(temperature: float | None):
    from langchain_openai import ChatOpenAI  # type: ignore

    kw = {"temperature": temperature} if temperature is not None else {}
    return ChatOpenAI(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        timeout=30,
        max_retries=2,
        **kw,
    )


def _github(temperature: float | None):
    from langchain_openai import ChatOpenAI  # type: ignore

    kw = {"temperature": temperature} if temperature is not None else {}
    return ChatOpenAI(
        model=os.environ.get("GITHUB_MODEL", "gpt-4o-mini"),
        api_key=os.environ["GITHUB_TOKEN"],
        base_url="https://models.inference.ai.azure.com",
        timeout=30,
        max_retries=2,
        **kw,
    )
