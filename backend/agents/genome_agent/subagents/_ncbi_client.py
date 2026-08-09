"""
NCBI eutils request helper: shared rate limiting + retry logic.

Shared by every real (non-mock) subagent in this package
(species_resolver, genome_metadata, gene_annotation, sequence_window) so
pacing/retry behavior against NCBI stays in one place.

Rate limiting matters here specifically because subagents/visualization.py's
size_comparison scope fans out several resolve_species()/get_genome_metadata()
calls concurrently via asyncio.gather() for its reference species. Without a
shared gate, those land on NCBI almost simultaneously — comfortably over
NCBI's unauthenticated cap of 3 requests/second — and eutils answers with
429s. A single retry per call (the original behavior) just re-triggers the
same burst problem a moment later. Serializing every call through one
process-wide gate, with a minimum spacing between calls, fixes this at the
source: no matter how many coroutines fire at once, only one request is
actually in flight and they're spread out enough to stay under the cap.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import requests

logger = logging.getLogger(__name__)

NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---------------------------------------------------------------------------
# Shared pacing gate (mirrors the approach in workflows/llm.py's NVIDIA
# client pacing): at most one in-flight eutils request at a time, with a
# minimum gap enforced between the start of one call and the next.
#
# NCBI's documented caps are 3 req/sec without an API key, 10 req/sec with
# one (https://www.ncbi.nlm.nih.gov/books/NBK25497/). Default spacing here
# is set a little under the unauthenticated cap's theoretical minimum
# (0.333s) to leave headroom; it tightens automatically once NCBI_API_KEY
# is set, since ncbi_get() then also attaches api_key to every request.
# ---------------------------------------------------------------------------
_NCBI_GATE = threading.Semaphore(1)
_pace_lock = threading.Lock()
_last_call_started_at = 0.0


def _default_min_interval() -> float:
    return 0.11 if os.getenv("NCBI_API_KEY") else 0.35


def _pace() -> None:
    """Block until at least the minimum interval has passed since the
    previous call was allowed to start."""
    global _last_call_started_at
    min_interval = float(os.getenv("NCBI_MIN_INTERVAL", str(_default_min_interval())))
    with _pace_lock:
        now = time.monotonic()
        wait = min_interval - (now - _last_call_started_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_started_at = time.monotonic()


def ncbi_get(
    params: dict,
    *,
    timeout: int = 30,
    max_retries: int = 2,
    retry_delay: float = 0.5,
) -> requests.Response:
    """Make a rate-limited, retrying GET request to NCBI eutils.

    `params` must include a "path" key (e.g. "esearch.fcgi") — it is popped
    off and used to build the URL, so the dict passed in is mutated.

    An NCBI_API_KEY env var, if set, is attached to every request
    automatically (raises the rate cap from 3 req/sec to 10 req/sec and
    tightens the pacing gate above to match — no other code needs to know
    about it).
    """
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params.setdefault("api_key", api_key)

    url = f"{NCBI_EUTILS_BASE}/{params.pop('path')}"

    def _request() -> requests.Response:
        with _NCBI_GATE:
            _pace()
            return requests.get(url, params=params, timeout=timeout)

    resp = _request()

    attempt = 0
    while resp.status_code in (429, 500) and attempt < max_retries:
        attempt += 1
        delay = retry_delay * (2 ** (attempt - 1))
        logger.warning(
            "NCBI returned %s, retrying (%d/%d) after %.1fs...",
            resp.status_code,
            attempt,
            max_retries,
            delay,
        )
        time.sleep(delay)
        resp = _request()

    resp.raise_for_status()
    return resp
