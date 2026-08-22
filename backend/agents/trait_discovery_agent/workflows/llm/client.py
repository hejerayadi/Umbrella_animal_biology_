import asyncio
import logging
import os
import random
from functools import lru_cache

from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")
DEFAULT_BASE_URL = os.getenv("NIM_BASE_URL")
FALLBACK_MODELS = (DEFAULT_MODEL,)

MAX_CAPACITY_RETRIES = 3
CAPACITY_RETRY_BASE_SECONDS = 1.5

# 429s (request-rate quota) are a fundamentally different failure than 503s
# (momentary worker-pool saturation): a free-tier NIM key's quota window is
# typically per-minute, not per-request, so retrying after 1-2s (the 503
# backoff) just burns the retry budget for no benefit — it needs a much more
# patient wait, and more attempts, before the quota actually clears.
# Configurable via env since the right amount of patience depends entirely
# on your tier/quota, which this code has no way to introspect.
MAX_RATE_LIMIT_RETRIES = int(os.getenv("NIM_MAX_RATE_LIMIT_RETRIES", "5"))
RATE_LIMIT_RETRY_BASE_SECONDS = float(os.getenv("NIM_RATE_LIMIT_RETRY_BASE_SECONDS", "10"))
RATE_LIMIT_RETRY_MAX_SECONDS = float(os.getenv("NIM_RATE_LIMIT_RETRY_MAX_SECONDS", "60"))

# Optional proactive throttle: a minimum gap enforced between the *start* of
# consecutive NIM calls, process-wide, regardless of which subagent issues
# them. Off by default (0s) since most deployments aren't quota-constrained
# and shouldn't pay a latency tax for it — but for a free-tier key that's
# already hitting 429s, spacing requests out can avoid tripping the limit in
# the first place rather than only reacting to it after the fact.
MIN_CALL_INTERVAL_SECONDS = float(os.getenv("NIM_MIN_CALL_INTERVAL_SECONDS", "0"))
_last_call_started_at: float = 0.0
_throttle_lock = asyncio.Lock()


def _get_api_key() -> str | None:
    return os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NIM_API_KEY")


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.1, model: str | None = None) -> ChatNVIDIA:
    """Create a ChatNVIDIA client with the configured API key explicitly passed in."""
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "NVIDIA_NIM_API_KEY is not set. Copy .env.example to .env and add your NIM key."
        )
    return ChatNVIDIA(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=512,
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
    )


def _is_missing_model_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "404" in message and ("not found" in message or "function" in message)


def _is_capacity_error(exc: Exception) -> bool:
    """503s from a saturated local NIM worker pool ('request limit reached',
    ResourceExhausted) are transient — worth a short retry, not an immediate
    fallback or model swap."""
    message = str(exc).lower()
    return "503" in message or "resourceexhausted" in message or (
        "request limit" in message and "service unavailable" in message
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    """429s — a request-quota rate limit, distinct from worker-pool
    exhaustion (_is_capacity_error above) and needing a much longer backoff.
    langchain_nvidia_ai_endpoints raises a bare Exception (see its
    _common.py:_try_raise) with no structured status code or Retry-After
    header exposed to the caller — only this string — so string matching is
    all that's available here."""
    message = str(exc).lower()
    return "429" in message or "too many requests" in message


async def _throttle() -> None:
    """Optional proactive spacing between NIM calls (see
    MIN_CALL_INTERVAL_SECONDS). No-op unless explicitly configured."""
    if MIN_CALL_INTERVAL_SECONDS <= 0:
        return
    global _last_call_started_at
    async with _throttle_lock:
        wait = _last_call_started_at + MIN_CALL_INTERVAL_SECONDS - asyncio.get_event_loop().time()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_started_at = asyncio.get_event_loop().time()


async def _retry_on_capacity(coro_fn):
    """Call coro_fn() (a zero-arg async callable), retrying with jittered
    exponential backoff on transient worker-pool exhaustion (503) or
    request-rate limiting (429) — the latter gets a much longer backoff and
    a larger retry budget since it's a slower-clearing condition than the
    former (see MAX_RATE_LIMIT_RETRIES vs MAX_CAPACITY_RETRIES)."""
    last_error: Exception | None = None
    attempt = 0
    while True:
        await _throttle()
        try:
            return await coro_fn()
        except Exception as exc:
            is_rate_limited = _is_rate_limit_error(exc)
            is_capacity = (not is_rate_limited) and _is_capacity_error(exc)
            if not (is_rate_limited or is_capacity):
                raise
            max_retries = MAX_RATE_LIMIT_RETRIES if is_rate_limited else MAX_CAPACITY_RETRIES
            if attempt >= max_retries:
                raise
            last_error = exc
            if is_rate_limited:
                delay = min(
                    RATE_LIMIT_RETRY_BASE_SECONDS * (2 ** attempt), RATE_LIMIT_RETRY_MAX_SECONDS
                ) + random.uniform(0, 1.0)
                logger.warning(
                    "NIM rate limited [429] (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
            else:
                delay = CAPACITY_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "NIM worker pool exhausted (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
            await asyncio.sleep(delay)
            attempt += 1
    raise last_error  # pragma: no cover - unreachable


def _candidate_models(model: str | None) -> list[str]:
    """Build the [explicit model, *fallbacks] list shared by every invocation path."""
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in FALLBACK_MODELS if m not in candidate_models)
    return candidate_models


def _parse_json_object(content: str) -> dict | None:
    """Best-effort extraction of a single JSON object from a model's final text,
    tolerating markdown code fences."""
    import json

    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
