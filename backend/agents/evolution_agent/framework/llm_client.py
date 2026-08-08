"""LLM client wrapper for the Evolution Agent.

Tries backends in this order, returning the first one that has valid
credentials. Raises ``LLMUnavailable`` only if none are configured.

Priority order
--------------
1. Azure OpenAI  — AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
                   Use AZURE_OPENAI_DEPLOYMENT=gpt-5 for GPT-5.
                   This is the project-wide standard.

2. Groq          — GROQ_API_KEY (starts with gsk_)
                   Free tier, ~500 tok/s, Llama-3.3-70b by default.
                   Best option for local dev without Azure access.

3. GitHub Models — GITHUB_TOKEN (classic PAT, no scope required)
                   Free tier, GPT-4o-mini by default.
                   Requires accepting marketplace terms once.

Environment variables
---------------------
AZURE_OPENAI_ENDPOINT    https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY     your Azure key
AZURE_OPENAI_DEPLOYMENT  deployment name  (default: gpt-5)
AZURE_OPENAI_API_VERSION API version      (default: 2024-08-01-preview)

GROQ_API_KEY             your Groq key
GROQ_MODEL               model name       (default: llama-3.3-70b-versatile)

GITHUB_TOKEN             your GitHub PAT
GITHUB_MODEL             model name       (default: gpt-4o-mini)

The module loads ``.env`` automatically at import time — callers never
need to call ``load_dotenv()`` themselves.
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
    """Raised when no LLM backend credentials are configured.

    Callers should catch this and fall back to a deterministic offline
    path (e.g. return RecognizedIntent(source='llm_unavailable')).
    """


def get_llm(temperature: float | None = None):
    """Return the first available chat model backend.

    Priority: Azure OpenAI → Groq → GitHub Models → LLMUnavailable.

    Parameters
    ----------
    temperature:
        Optional sampling temperature. Omitted entirely when None so
        GPT-5 (which rejects non-default temperature values) works
        without special-casing by callers.
    """
    if os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("azure_endpoint"):
        return _azure(temperature)
    if os.environ.get("GROQ_API_KEY"):
        return _groq(temperature)
    if os.environ.get("GITHUB_TOKEN"):
        return _github(temperature)
    raise LLMUnavailable(
        "No LLM backend configured. Set one of:\n"
        "  • AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY  (Azure / GPT-5)\n"
        "  • GROQ_API_KEY                                   (Groq / Llama)\n"
        "  • GITHUB_TOKEN                                   (GitHub Models)\n"
        "See backend/agents/evolution_agent/.env.example for details."
    )


# ---------------------------------------------------------------------------
# Private backend factories
# ---------------------------------------------------------------------------

def _kwargs(temperature: float | None) -> dict:
    """Only include temperature in kwargs when explicitly set.

    GPT-5 hard-errors on any non-default temperature value. Omitting
    the field lets every model use its own sensible default.
    """
    return {"temperature": temperature} if temperature is not None else {}


def _azure(temperature: float | None):
    endpoint   = (
        os.environ.get("AZURE_OPENAI_ENDPOINT")
        or os.environ.get("azure_endpoint")
    )
    api_key    = (
        os.environ.get("AZURE_OPENAI_API_KEY")
        or os.environ.get("openai_key_azure")
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not endpoint or not api_key:
        raise LLMUnavailable(
            "Azure OpenAI: both AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "must be set."
        )

    from langchain_openai import AzureChatOpenAI  # type: ignore

    # GPT-5 only accepts temperature=1 (its default). Pin it unless the
    # caller explicitly overrides, which keeps compatibility with other
    # models that accept custom temperatures.
    temp = temperature if temperature is not None else 1

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        timeout=60,
        max_retries=2,
        temperature=temp,
    )


def _groq(temperature: float | None):
    from langchain_openai import ChatOpenAI  # type: ignore

    return ChatOpenAI(
        model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
        timeout=30,
        max_retries=2,
        **_kwargs(temperature),
    )


def _github(temperature: float | None):
    from langchain_openai import ChatOpenAI  # type: ignore

    return ChatOpenAI(
        model=os.environ.get("GITHUB_MODEL", "gpt-4o-mini"),
        api_key=os.environ["GITHUB_TOKEN"],
        base_url="https://models.inference.ai.azure.com",
        timeout=30,
        max_retries=2,
        **_kwargs(temperature),
    )
