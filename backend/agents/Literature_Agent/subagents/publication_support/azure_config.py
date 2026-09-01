"""Azure OpenAI settings for the Publication Support subagent.

One resolver, shared by `ranking/llm_reranker.py` and
`ranking/query_interpreter.py`, which each carried a byte-identical copy that
read `AZURE_OPENAI_*` and nothing else. That was the reason this package
needed a `.env` of its own: the agent-wide file names its deployments per
role, and there was no role name this package would answer to.

Resolution order, first hit wins - the same chain as `llm/client.py` and
`subagents/scientific_analysis/config.py`:

    AZURE_LITERATURE_PUBLICATION_<FIELD>   this subagent only
    AZURE_LITERATURE_<FIELD>               every role in this agent
    AZURE_OPENAI_<FIELD>                   the shared fallback

The distinction matters here: `rerank_journals` and `interpret_query` send
`temperature` and `max_tokens`, which a reasoning deployment (gpt-5) rejects.
The PUBLICATION block keeps this package on a plain chat deployment even when
the shared fallback points at a reasoning one.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# The agent's single .env, addressed by absolute path: this package is imported
# from the orchestrator, whose CWD is the repository root, so a bare
# load_dotenv() would read backend/.env instead.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

_FIELD_ALIASES = {
    "endpoint": ("ENDPOINT",),
    "api_key": ("API_KEY",),
    "deployment": ("DEPLOYMENT", "DEPLOYMENT_NAME"),
}

_PREFIXES = ("AZURE_LITERATURE_PUBLICATION_", "AZURE_LITERATURE_", "AZURE_OPENAI_")


def _setting(field: str) -> str:
    """Resolve one setting across the prefix chain. Returns "" when unset."""
    for prefix in _PREFIXES:
        for suffix in _FIELD_ALIASES[field]:
            value = os.getenv(prefix + suffix)
            if value and value.strip():
                return value.strip().strip('"').strip("'")
    return ""


def _normalise_endpoint(endpoint: str) -> str:
    """Ensure the endpoint ends in /openai/v1.

    This package calls the plain `OpenAI` SDK with `base_url`, not
    `AzureOpenAI`, so it needs the v1 path that the role blocks for the router
    and writer deliberately omit. Normalising here means either spelling works
    in the .env.
    """
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/openai/v1"):
        return trimmed
    return trimmed.split("/openai/")[0] + "/openai/v1"


def get_endpoint() -> str:
    raw = _setting("endpoint")
    return _normalise_endpoint(raw) if raw else ""


def get_api_key() -> str:
    return _setting("api_key")


def get_deployment() -> str:
    return _setting("deployment")


def is_configured() -> bool:
    """Whether an LLM call can be attempted. Never raises."""
    return bool(get_endpoint() and get_api_key() and get_deployment())


# Built on first use. Raising at import time made this module - and therefore
# the whole publication_support package - unimportable without credentials,
# which no caller can catch, so the writing sub-orchestrator could not even
# fall back. A missing key is now a call-time error the caller degrades from.
@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    missing = [
        name
        for name, value in (
            ("endpoint", get_endpoint()),
            ("api_key", get_api_key()),
            ("deployment", get_deployment()),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Azure OpenAI settings for publication support in the "
            f"Literature_Agent .env: {', '.join(missing)}. Set the "
            "AZURE_LITERATURE_PUBLICATION_* block, or the shared AZURE_OPENAI_* one."
        )
    return OpenAI(api_key=get_api_key(), base_url=get_endpoint())


__all__ = [
    "get_client",
    "get_deployment",
    "get_endpoint",
    "get_api_key",
    "is_configured",
]
