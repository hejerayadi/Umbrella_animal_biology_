from __future__ import annotations

import asyncio
import logging
import random

import httpx

logger = logging.getLogger(__name__)

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE_SECONDS = 1.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _retry_after_seconds(exc: Exception) -> float | None:
    """Honor a server-supplied Retry-After header (KEGG/QuickGO both send one
    on 429s) instead of guessing at a backoff."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    value = exc.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
    **kwargs,
) -> httpx.Response:
    """GET/POST/etc with retry on timeouts and 429/5xx, raising for status on
    the final attempt. Respects Retry-After when the server provides one,
    otherwise backs off with jitter."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = await client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == attempts - 1:
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Transient error on %s %s (attempt %d/%d), retrying in %.1fs: %s",
                method, url, attempt + 1, attempts, delay, exc,
            )
            await asyncio.sleep(delay)
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises
