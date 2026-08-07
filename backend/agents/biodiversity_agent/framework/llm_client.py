"""LLM client wrapper for the Biodiversity Agent.

The wrapper tries backends in this order, returning the first one that
has valid credentials, and raising ``LLMUnavailable`` only if none does:

1. **Groq** (``GROQ_API_KEY``) - free tier, Llama 3.3 70B via the
   OpenAI-compatible endpoint at ``https://api.groq.com/openai/v1``.
   No terms of service to accept, no region restrictions, ~500 tok/s.
   This is the recommended dev backend.
2. **GitHub Models** (``GITHUB_TOKEN``) - free tier, GPT-4o-mini via
   ``https://models.inference.ai.azure.com``. Requires accepting the
   marketplace terms first.
3. **Azure OpenAI** (``AZURE_OPENAI_ENDPOINT`` + ``AZURE_OPENAI_API_KEY``) -
   the platform-wide standard per the repo README. Used in production
   when Hajer shares her Azure resource with the group.
4. If none is configured, ``LLMUnavailable`` is raised so callers can
   fall back to their deterministic offline path.

Environment variables consumed:

- ``GROQ_API_KEY``            - Groq API key (starts with ``gsk_``).
- ``GROQ_MODEL``              - optional, default ``llama-3.3-70b-versatile``.
- ``GITHUB_TOKEN``            - GitHub PAT (classic, no scope required).
- ``GITHUB_MODEL``            - optional, default ``gpt-4o-mini``.
- ``AZURE_OPENAI_ENDPOINT``   - the ``https://<resource>.openai.azure.com/`` URL.
- ``AZURE_OPENAI_API_KEY``    - the Azure key.
- ``AZURE_OPENAI_DEPLOYMENT`` - deployment name (default ``gpt-4o-mini``).
- ``AZURE_OPENAI_API_VERSION``- optional, default ``2024-08-01-preview``.

The module loads ``.env`` at import time via ``python-dotenv`` so tests
and scripts just work without the caller having to remember to call
``load_dotenv()`` themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Walk up from this file to the first ``.env`` and load it."""

    here = Path(__file__).resolve()
    for parent in [here.parent.parent, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


_load_env()


class LLMUnavailable(RuntimeError):
    """Raised when no LLM backend is configured.

    Callers should catch this and switch to a deterministic offline path.
    """


def get_llm(temperature: float | None = None):
    """Return the first available chat model backend.

    Priority: Azure OpenAI (repo standard) -> Groq -> GitHub Models ->
    raise ``LLMUnavailable``.

    Azure wins by default so the code matches the platform-wide standard
    documented in the root README. Groq and GitHub Models are automatic
    fallbacks if the Azure keys are missing, which is convenient during
    development and CI.
    """

    if os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("azure_endpoint"):
        return _get_azure_llm(temperature)
    if os.environ.get("GROQ_API_KEY"):
        return _get_groq_llm(temperature)
    if os.environ.get("GITHUB_TOKEN"):
        return _get_github_models_llm(temperature)
    raise LLMUnavailable(
        "No LLM backend configured. Set one of: AZURE_OPENAI_ENDPOINT + "
        "AZURE_OPENAI_API_KEY, GROQ_API_KEY, or GITHUB_TOKEN in your .env."
    )


# Backwards-compatible alias - the toy agent originally called this name.
def get_azure_llm(temperature: float | None = None):
    return get_llm(temperature)


# ---------- private backend factories ----------


def _optional_temp(temperature: float | None) -> dict:
    """Return kwargs dict with ``temperature`` set only if explicitly passed.

    GPT-5 family models on Azure reject any non-default temperature value
    (they hard-error with ``unsupported_value``). Omitting the field
    entirely lets the model use its own default and keeps compatibility
    with every other model that accepts a custom temperature.
    """

    return {"temperature": temperature} if temperature is not None else {}


def _get_groq_llm(temperature: float | None):
    """Groq via its OpenAI-compatible endpoint.

    Uses ``langchain_openai.ChatOpenAI`` with a custom ``base_url``.
    Groq speaks the OpenAI wire protocol so the rest of the code stays
    completely unchanged.
    """

    api_key = os.environ["GROQ_API_KEY"]
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    from langchain_openai import ChatOpenAI  # type: ignore

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=30,
        max_retries=2,
        **_optional_temp(temperature),
    )


def _get_github_models_llm(temperature: float | None):
    """GitHub Models via its OpenAI-compatible endpoint."""

    token = os.environ["GITHUB_TOKEN"]
    model = os.environ.get("GITHUB_MODEL", "gpt-4o-mini")

    from langchain_openai import ChatOpenAI  # type: ignore

    return ChatOpenAI(
        model=model,
        api_key=token,
        base_url="https://models.inference.ai.azure.com",
        timeout=30,
        max_retries=2,
        **_optional_temp(temperature),
    )


def _get_azure_llm(temperature: float | None):
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get(
        "azure_endpoint"
    )
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get(
        "openai_key_azure"
    )
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    if not endpoint or not api_key:
        raise LLMUnavailable(
            "Azure OpenAI credentials incomplete: endpoint and api_key both required."
        )

    from langchain_openai import AzureChatOpenAI  # type: ignore

    # GPT-5 family only accepts the default temperature (1). langchain's
    # own default is 0.7 which GPT-5 hard-rejects, so we pin it to 1
    # unless the caller explicitly wants something else - value 1 is
    # accepted by every OpenAI model, so this is safe across the board.
    temp = temperature if temperature is not None else 1

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        azure_deployment=deployment,
        api_version=api_version,
        timeout=30,
        max_retries=2,
        temperature=temp,
    )
