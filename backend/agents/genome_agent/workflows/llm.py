from __future__ import annotations

import logging
import os
import random
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv(Path(__file__).parent.parent / ".env")

# The NVIDIA client warns on every bind_tools() call for models it doesn't
# have hardcoded tool-support metadata for, and separately warns on every
# client construction when a model isn't in its known-type list. Both are
# one-time-relevant compatibility notices, not per-call signals — this
# model demonstrably works for both tool binding and inference in this
# codebase — so they're just noise on every single invocation. Silenced at
# the source rather than suppressed per-call so they can't leak from any
# import path.
warnings.filterwarnings(
    "ignore",
    message=r".*is not known to support tools.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*type is unknown and inference may fail.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*is unknown, check `available_models`.*",
    category=UserWarning,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "meta/llama-3.3-70b-instruct"

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Proactive pacing.
#
# The graph itself never calls the LLM concurrently (query_router,
# capability_resolver, and explanation_writer run one after another within
# a single graph run), so the "Worker local total request limit reached"
# errors come from this process's calls landing too close together in time
# — either back-to-back scenarios in one run, or another process sharing the
# same NVIDIA_API_KEY. Rather than let calls fire immediately and retry
# after they get rejected, every call is paced through a single global gate:
# at most one in-flight request at a time, with a minimum gap enforced
# between the start of one call and the next. This keeps this agent's own
# request pattern smooth and under the cap by construction, so the
# exponential-backoff retry below becomes a rarely-needed safety net rather
# than the primary defense.
# ---------------------------------------------------------------------------
_LLM_GATE = threading.Semaphore(1)
_pace_lock = threading.Lock()
_last_call_started_at = 0.0
MIN_CALL_INTERVAL_SECONDS = float(os.getenv("NVIDIA_LLM_MIN_INTERVAL", "0.5"))


def _pace() -> None:
    """Block until at least MIN_CALL_INTERVAL_SECONDS has passed since the
    previous call was allowed to start."""
    global _last_call_started_at
    with _pace_lock:
        now = time.monotonic()
        wait = MIN_CALL_INTERVAL_SECONDS - (now - _last_call_started_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_started_at = time.monotonic()


@lru_cache(maxsize=None)
def get_llm_client() -> BaseChatModel:
    """Build (once) and reuse the ChatNVIDIA client.

    Constructing ChatNVIDIA is not free: for a model not in the library's
    hardcoded registry (this one isn't — that's why you see the "type is
    unknown" warning), its constructor calls `self.available_models`,
    which makes a real HTTP GET to NVIDIA to list every model it offers,
    just to validate the name. Without caching, `route_query`,
    `resolve_capability`, and `write_explanation` would each rebuild the
    client — tripling both the network round trips per scenario and the
    requests counted against the free-tier cap, for zero benefit, since
    nothing about the client changes between calls in this process.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set")

    return ChatNVIDIA(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        # The library defaults this to 60s, and it isn't just an HTTP
        # timeout — on a 202 ("still processing") response it polls every
        # 0.02s for up to `timeout` seconds before giving up (see
        # `_wait()` in langchain_nvidia_ai_endpoints._common). Under
        # free-tier saturation that's exactly the response you get, so
        # the default means a single call can silently hang for up to a
        # minute before our fallback logic ever gets a chance to run.
        # Failing in a few seconds instead means the deterministic
        # fallback kicks in fast, which is the whole point of having one.
        timeout=float(os.getenv("NVIDIA_LLM_TIMEOUT", "8")),
    )


def summarize_llm_error(exc: Exception) -> str:
    """Collapse the NVIDIA client's raw error text into one short line.

    The client's own error formatting (`_format_error` in
    langchain_nvidia_ai_endpoints) builds its message as
    `"[status] {title_or_dict}\\n{full_response_dict}"`. When the response
    body has no "detail" key — which is the case for the free-tier
    ResourceExhausted 503 — that second line is just the entire raw
    response dict repeated. Logging `str(exc)` as-is prints that whole
    two-line blob on every failure. This pulls out just the useful part.
    """
    if _is_transient_error(exc):
        return "NVIDIA endpoint rate-limited (free-tier cap reached) — using fallback"
    # Fall back to the first line only, so unexpected errors still surface
    # without dragging the raw response dict into the log.
    return str(exc).splitlines()[0]


def _is_transient_error(exc: Exception) -> bool:
    """True for rate-limit / service-unavailable errors worth retrying.

    Anything else (auth errors, bad requests, malformed tool calls, etc.)
    is left alone so it re-raises immediately and the existing fallback
    logic in query_router / capability_resolver / explanation_writer
    kicks in exactly as before.
    """
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "resourceexhausted",
            "rate limit",
            "rate_limit",
            "too many requests",
            "503",
            "429",
            "service unavailable",
        )
    )


def invoke_with_retry(
    invoke_fn: Callable[[], _T],
    *,
    max_retries: int = 1,
    base_delay: float = 1.0,
) -> _T:
    """Call `invoke_fn()` through the pacing gate above.

    On a free-tier key, the shared "Worker local total request limit
    reached" ceiling is outside this process's control — no amount of
    local pacing or backoff avoids it when the endpoint is genuinely
    saturated. Every caller of this function (query_router,
    capability_resolver, explanation_writer) already has a fast,
    deterministic non-LLM fallback for exactly this case, so the default
    here is now a single attempt: on a transient error, raise immediately
    and let that fallback answer right away instead of burning several
    seconds retrying a call that's likely to fail again.

    Pass max_retries > 1 explicitly if you'd rather wait out a transient
    dip than fall back (e.g. for a call with no good fallback path).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        with _LLM_GATE:
            _pace()
            try:
                return invoke_fn()
            except Exception as exc:
                last_exc = exc
                if not _is_transient_error(exc) or attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "Transient NVIDIA endpoint error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises