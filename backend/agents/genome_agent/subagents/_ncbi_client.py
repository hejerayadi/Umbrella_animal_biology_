"""
NCBI eutils request helper with basic retry logic.
"""

from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

NCBI_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def ncbi_get(params: dict, *, timeout: int = 30, max_retries: int = 1, retry_delay: float = 0.5) -> requests.Response:
    """Make a GET request to NCBI eutils with a single retry on 429/500."""
    url = f"{NCBI_EUTILS_BASE}/{params.pop('path')}"
    
    def _request() -> requests.Response:
        return requests.get(url, params=params, timeout=timeout)
    
    resp = _request()
    
    if resp.status_code in (429, 500) and max_retries > 0:
        logger.warning(
            "NCBI returned %s, retrying once after %.1fs...",
            resp.status_code,
            retry_delay,
        )
        time.sleep(retry_delay)
        resp = _request()
    
    resp.raise_for_status()
    return resp
