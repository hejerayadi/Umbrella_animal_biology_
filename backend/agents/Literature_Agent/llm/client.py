"""Azure OpenAI access for the Literature Agent, one configuration per role.

The parts of this agent do not share a model: the router and the writing
support run on one deployment, discovery on another. So the settings are named
per role rather than per agent -

    AZURE_LITERATURE_ROUTER_ENDPOINT
    AZURE_LITERATURE_WRITING_DEPLOYMENT
    AZURE_LITERATURE_DISCOVERY_API_KEY      ... and so on

- because a single AZURE_OPENAI_* set cannot describe two deployments. Listing
it twice in a .env is not an error and raises nothing: the last line silently
wins and the other resource is simply never reached.

Each setting is looked up in three places, first hit wins:

    AZURE_LITERATURE_<ROLE>_<FIELD>   this role only
    AZURE_LITERATURE_<FIELD>          every role in this agent
    AZURE_OPENAI_<FIELD>              the shared backend configuration

So one agent-wide block still works for everything, and a role only needs its
own entries where it actually differs.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

# This agent's own .env, addressed explicitly rather than by walking up from
# the working directory: the service is started from the repository root
# ("python -m uvicorn backend.agents.Literature_Agent.api:app"), so a relative
# search can pick up a different .env than the one next to this code.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

ROUTER = "ROUTER"
WRITING = "WRITING"
DISCOVERY = "DISCOVERY"

# Alternate spellings accepted for one field, tried in order within each
# prefix. Other agents in this repo spell the deployment both ways.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "endpoint": ("ENDPOINT",),
    "api_key": ("API_KEY",),
    "api_version": ("API_VERSION",),
    "deployment": ("DEPLOYMENT", "DEPLOYMENT_NAME"),
}


def _setting(role: str, field: str) -> str | None:
    """Resolve one setting for one role: role, then agent, then shared."""
    prefixes = (f"AZURE_LITERATURE_{role}_", "AZURE_LITERATURE_", "AZURE_OPENAI_")
    for prefix in prefixes:
        for suffix in _FIELD_ALIASES[field]:
            value = os.getenv(prefix + suffix)
            if value and value.strip():
                return value.strip()
    return None


def _normalise_endpoint(endpoint: str) -> str:
    """Reduce an endpoint to the bare resource root the SDK expects.

    `AzureOpenAI` appends "/openai/deployments/{deployment}/..." to whatever it
    is given. An endpoint copied from the v1 API page ends in "/openai/v1", and
    leaving that on produces a 404 that reads like a missing deployment rather
    than a malformed URL - so it is trimmed here instead of being diagnosed
    over and over.
    """
    return endpoint.split("/openai/")[0].rstrip("/") + "/"


def is_configured(role: str = ROUTER) -> bool:
    """Whether an LLM call can be attempted for `role`. Never raises."""
    return all(
        _setting(role, field)
        for field in ("endpoint", "api_key", "api_version", "deployment")
    )


def describe_config(role: str) -> dict:
    """Which deployment a role resolved to. No secrets - for /health and logs."""
    endpoint = _setting(role, "endpoint")
    return {
        "configured": is_configured(role),
        "endpoint": _normalise_endpoint(endpoint) if endpoint else None,
        "deployment": _setting(role, "deployment"),
        "api_version": _setting(role, "api_version"),
    }


@lru_cache(maxsize=len((ROUTER, WRITING, DISCOVERY)))
def _get_client(role: str) -> AzureOpenAI:
    """Build a role's client on first use, never at import time.

    `AzureOpenAI(...)` raises when the key or endpoint is missing. At module
    scope that turns a missing env var into an ImportError, which no caller can
    catch and which kills the whole agent - the keyword fallback in
    `routing.router.classify_route` would never get to run, and the FastAPI
    service would refuse to start. Deferring construction keeps a missing key a
    call-time problem with a working fallback.
    """
    endpoint = _setting(role, "endpoint")
    api_key = _setting(role, "api_key")
    if not endpoint or not api_key:
        raise RuntimeError(
            f"Azure OpenAI is not configured for the Literature Agent's {role} role."
        )

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=_normalise_endpoint(endpoint),
        api_version=_setting(role, "api_version"),
    )


def call_llm(
    messages: list[dict],
    max_completion_tokens: int = 500,
    reasoning_effort: str = "low",
    role: str = ROUTER,
) -> str:
    """Call the deployment configured for `role`, return the text answer.

    Raises RuntimeError when that role has no Azure configuration, so callers
    can fall back to a deterministic path instead of failing the request.
    Unchanged from before - existing callers (routing, publication support)
    keep working exactly as before.
    """
    deployment = _setting(role, "deployment")
    if not deployment:
        raise RuntimeError(
            f"Azure OpenAI is not configured for the Literature Agent's {role} role."
        )

    response = _get_client(role).chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )
    content = response.choices[0].message.content
    return content or ""


def call_llm_with_tools(
    messages: list[dict],
    tools: list[dict],
    max_completion_tokens: int = 500,
    reasoning_effort: str = "low",
    role: str = ROUTER,
):
    """Call the deployment configured for `role`, with function-calling tools
    available. Returns the raw `message` object from the API response (not
    just text), because the caller needs to inspect `.tool_calls` to know
    whether the model asked to invoke a tool or produced a final answer.

    Raises RuntimeError when that role has no Azure configuration, same as
    `call_llm`.
    """
    deployment = _setting(role, "deployment")
    if not deployment:
        raise RuntimeError(
            f"Azure OpenAI is not configured for the Literature Agent's {role} role."
        )

    response = _get_client(role).chat.completions.create(
        model=deployment,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0].message