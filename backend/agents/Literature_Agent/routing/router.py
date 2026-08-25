from __future__ import annotations

from typing import Iterable

from ..llm.client import ROUTER, call_llm
from ..llm.prompts import ROUTING_SYSTEM_PROMPT

VALID_ROUTES = {"discovery", "writing", "both_sequential", "both_parallel"}

DISCOVERY_KEYWORDS = (
    "find", "search", "lookup", "retrieve", "discover", "summarize", "summary",
    "compare", "comparison", "papers", "literature", "studies", "reviews", "recent"
)
WRITING_KEYWORDS = (
    "write", "draft", "abstract", "introduction", "section", "paper", "journal",
    "venue", "publication", "manuscript", "review section", "rewrite", "edit"
)
SEQUENTIAL_KEYWORDS = (
    "based on", "using the results", "after the search", "then write", "grounded in",
    "drawn from", "using the literature", "first search", "after searching"
)


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def classify_route(text: str) -> str:
    """Heuristic router used as deterministic fallback when Azure is unavailable."""
    instruction = (text or "").strip()
    if not instruction:
        return "discovery"

    has_discovery = _contains_any(instruction, DISCOVERY_KEYWORDS)
    has_writing = _contains_any(instruction, WRITING_KEYWORDS)

    if has_discovery and has_writing:
        if _contains_any(instruction, SEQUENTIAL_KEYWORDS):
            return "both_sequential"
        return "both_parallel"
    if has_discovery:
        return "discovery"
    if has_writing:
        return "writing"
    return "discovery"


def classify_route_llm(instruction: str) -> str:
    """Preferred route classification via Azure OpenAI, with deterministic fallback.

    Always returns one of `VALID_ROUTES` - every failure path goes through
    `classify_route`, which does too.
    """
    try:
        route = call_llm(
            messages=[
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
            ],
            # Generous for a one-word answer on purpose: on a reasoning
            # deployment max_completion_tokens covers reasoning tokens too, and
            # a gpt-5 deployment intermittently spends 50+ of them even at the
            # lowest effort. When that exhausts the budget the call returns
            # finish_reason="length" with empty content, which would look like
            # an uncertain classifier rather than a truncated one and silently
            # drop every request to the "discovery" default.
            max_completion_tokens=256,
            reasoning_effort="none",
            role=ROUTER,
        )
        route = (route or "").strip().lower()
    except Exception:
        return classify_route(instruction)

    # An answer that is not one of the four routes falls back to the keyword
    # matcher too, not just a raised exception. The truncation described above
    # is not an exception: it returns finish_reason="length" with empty
    # content, which lands here. Returning None for that sent every such
    # request to the caller's "discovery" default and left `classify_route`
    # unreachable in exactly the case it was written for.
    return route if route in VALID_ROUTES else classify_route(instruction)
