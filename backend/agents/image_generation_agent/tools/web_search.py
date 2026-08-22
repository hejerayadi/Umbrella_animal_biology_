"""Web search via Serper API for complementary biological evidence."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from ..tool_schemas import WebSearchHit, WebSearchResult

logger = logging.getLogger(__name__)

_AGENT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(_AGENT_DIR / ".env", override=True)

SERPER_URL = "https://google.serper.dev/search"
DEFAULT_TIMEOUT = 30


def call_web_search(
    query: str,
    *,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    api_key: str | None = None,
) -> WebSearchResult:
    """Run a natural-language web search and return structured organic results."""
    query = query.strip()
    if not query:
        return WebSearchResult(success=False, error="Search query is required.")

    key = api_key if api_key is not None else os.getenv("SERPER_API_KEY")
    if not key:
        return WebSearchResult(success=False, error="SERPER_API_KEY is not configured.")

    http = session or requests.Session()
    headers = {
        "X-API-KEY": key,
        "Content-Type": "application/json",
    }

    try:
        response = http.post(
            SERPER_URL,
            headers=headers,
            json={"q": query, "num": 5},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        logger.warning("Serper web search failed: %s", type(exc).__name__)
        return WebSearchResult(success=False, error=f"Web search request failed: {exc}")

    hits = _parse_organic_results(payload)
    return WebSearchResult(success=True, results=tuple(hits))


def _parse_organic_results(payload: dict) -> list[WebSearchHit]:
    organic = payload.get("organic") or []
    hits: list[WebSearchHit] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if title and link:
            hits.append(WebSearchHit(title=title, link=link, snippet=snippet))
    return hits
