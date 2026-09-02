from __future__ import annotations

import contextlib
import logging
import os
import random
import threading
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - Docker/Linux only in practice
    fcntl = None

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

# meta/llama-3.1-8b-instruct (and every other plain-instruct model this
# used to fall back to — 3.3-70b, Mistral, Granite, Phi-4, Qwen3-Instruct,
# etc.) now comes back 404/410 from NVIDIA NIM: those catalog entries were
# deprecated, not just congested. This is the same dead-model wave
# trait_discovery_agent hit (see its workflows/llm/client.py
# FALLBACK_MODELS comment and probe_all_models.py) — of 46 candidates
# probed there, only 5 were still alive, all hybrid-reasoning:
#   nvidia/nemotron-3-nano-30b-a3b   <- smallest/fastest of the 5, used here
#   nvidia/nemotron-3-super-120b-a12b
#   nvidia/nemotron-3-ultra-550b-a55b
#   openai/gpt-oss-20b
#   openai/gpt-oss-120b
# genome_agent doesn't have trait_discovery_agent's multi-model
# FALLBACK_MODELS chain — it only ever calls one model — so this switches
# straight to the smallest confirmed-alive one rather than the smallest
# *congested-but-alive* one. Override with NVIDIA_LLM_MODEL if a bigger
# model from that list is needed once a fallback chain is added here, or a
# paid/higher-tier key changes the calculus.
MODEL_NAME = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b")

_T = TypeVar("_T")

# Smaller tool-calling models frequently emit the JSON *keyword* null as a
# quoted *string* instead — e.g. {"assembly_id": "null"} instead of
# {"assembly_id": null}. Pydantic happily accepts that as the literal
# three-character string for any `str | None` field, so a caller checking
# `value is not None` never sees it as absent. This showed up concretely in
# species_resolver.py: the free-tier 8B model, on the "elephant" scenario,
# decided to give up (assembly_id=null) but encoded it this way, and burned
# its whole step budget hitting the grounding-rejection message over and
# over instead of ever reaching a valid submission.
_NULL_SENTINELS = {"null", "none", "nil", "n/a", "na", ""}


def coerce_null_sentinels(args: dict, nullable_fields: set[str]) -> dict:
    """Return a copy of `args` with any of `nullable_fields` whose value is
    a null-sentinel string (see _NULL_SENTINELS) replaced with real None.

    Only touches keys present in `args` and listed in `nullable_fields` —
    fields that are genuinely required (e.g. GenomeMetadataOutput's
    assembly_id_used) are deliberately left alone so a "null" there still
    fails pydantic validation and prompts a real resubmission, rather than
    silently becoming an invalid None for a required field.
    """
    cleaned = dict(args)
    for field in nullable_fields:
        value = cleaned.get(field)
        if isinstance(value, str) and value.strip().lower() in _NULL_SENTINELS:
            cleaned[field] = None
    return cleaned

# ---------------------------------------------------------------------------
# Proactive pacing / concurrency limiting.
#
# docker-compose.yml runs full_species_resolver_demo and
# full_genome_metadata_demo (plus genome-agent-scenarios, ncbi-live-check,
# etc.) as *separate containers*, all with `env_file: .env`, so they all
# share one free-tier NVIDIA_API_KEY. An in-memory threading.Semaphore only
# serializes calls made by this one process — it does nothing to stop two+
# containers from each calling NVIDIA at once.
#
# IMPORTANT: "Worker local total request limit reached (17/16)" is a
# *concurrent in-flight requests* cap, not a requests-per-minute cap.
# Spacing out when calls *start* (the original approach) doesn't fix this —
# if a call takes longer to complete than the gap between two call starts
# (very likely: NIM can take several seconds per turn, and the tool loop
# makes several turns), multiple calls end up in flight at once regardless
# of how evenly their starts were spaced. The only thing that actually
# bounds concurrency is holding a lock for the *entire* call, not just the
# gap before it.
#
# Every container here bind-mounts the same host directory
# (`.:/app/gene_agent`), so an flock on a small state file inside that
# mount gives every process — regardless of container — a real cross-process
# mutex: at most one NVIDIA call, from any of these processes, is ever in
# flight at a time. That's a stricter guarantee than the free-tier's
# 16-concurrent cap needs, but on a free tier "well under the limit" is the
# right target, not "right up against it". The min-interval pacing is kept
# on top of that as a courtesy gap between one call finishing and the next
# starting, not as the primary defense anymore.
# ---------------------------------------------------------------------------
_LLM_GATE = threading.Semaphore(1)
_pace_lock = threading.Lock()
_last_call_started_at = 0.0
# Bumped from 1.5s -> 2.5s: the tool loops in species_resolver.py (up to 6
# steps) and genome_metadata.py (up to 5 steps) each make that many
# sequential NVIDIA calls per scenario. A tighter gap made it more likely
# for a run to still be inside the free tier's worker-pool window when the
# next step's call started, which is what actually trips "Worker local
# total request limit reached" — this is a deliberate speed-for-reliability
# trade (slower demo runs, fewer skipped LLM steps), not a bugfix.
MIN_CALL_INTERVAL_SECONDS = float(os.getenv("NVIDIA_LLM_MIN_INTERVAL", "2.5"))

# Shared across every process that checks out this repo (dev machine or any
# docker-compose service), since they all mount the same gene_agent/ dir.
_PACE_STATE_FILE = Path(__file__).parent.parent / ".nim_pace_state"


@contextlib.contextmanager
def _llm_call_slot():
    """Cross-process mutex held for the *entire* duration of one NVIDIA
    call (including any retries/backoff inside it) — not just the pacing
    gap before it starts. This is what actually caps concurrent in-flight
    requests across every process sharing this key; see the module-level
    comment above for why pacing alone can't do that.

    Falls back to in-process-only locking (the original behavior) if flock
    isn't available (e.g. a non-POSIX dev environment) or the state file
    can't be opened, so a missing lock never becomes a hard failure — it
    just quietly loses the cross-process guarantee.
    """
    if fcntl is None:
        with _LLM_GATE:
            yield
        return

    try:
        _PACE_STATE_FILE.touch(exist_ok=True)
        f = open(_PACE_STATE_FILE, "r+")
    except OSError as exc:
        logger.warning(
            "Cross-process NVIDIA locking unavailable (%s) — falling back "
            "to in-process locking only for this call.",
            exc,
        )
        with _LLM_GATE:
            yield
        return

    try:
        # Blocks here until no other process anywhere has a call in flight.
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            raw = f.read().strip()
            last = float(raw) if raw else 0.0
            wait = MIN_CALL_INTERVAL_SECONDS - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            yield
        finally:
            f.seek(0)
            f.truncate()
            f.write(repr(time.time()))
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()


def get_llm_client() -> BaseChatModel:
    """Return a ChatNVIDIA client for the *current* NVIDIA_API_KEY.

    This function itself is deliberately NOT cached: it re-reads
    os.getenv("NVIDIA_API_KEY") on every call so that a key being unset
    (or rotated) mid-process is always observed immediately — e.g. by
    scripts/run_species_resolver_scenarios.py's "LLM client unavailable"
    scenario, which pops NVIDIA_API_KEY from os.environ and expects the
    very next call to raise. The actual expensive work is delegated to
    _build_llm_client, which IS cached — see its docstring.
    """
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set")
    return _build_llm_client(api_key)


@lru_cache(maxsize=None)
def _build_llm_client(api_key: str) -> BaseChatModel:
    """Build (once per distinct api_key) and reuse the ChatNVIDIA client.

    Constructing ChatNVIDIA is not free: for a model not in the library's
    hardcoded registry (this one isn't — that's why you see the "type is
    unknown" warning), its constructor calls `self.available_models`,
    which makes a real HTTP GET to NVIDIA to list every model it offers,
    just to validate the name. Without caching, `route_query`,
    `resolve_capability`, and `write_explanation` would each rebuild the
    client — tripling both the network round trips per scenario and the
    requests counted against the free-tier cap, for zero benefit, since
    nothing about the client changes between calls in this process.

    Cached by api_key (not zero-arg) so that a key rotation mid-process
    still gets a client built against the new key, rather than silently
    reusing whatever client happened to be built first. A previously-built
    client for an old key is never returned once get_llm_client() sees a
    different (or missing) key — the old bug this replaces was caching
    with no key at all, so removing NVIDIA_API_KEY from os.environ had no
    effect on an already-cached client.
    """
    return ChatNVIDIA(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
        top_p=0.7,
        # Bumped from 1024 -> 8192: MODEL_NAME is now a hybrid-reasoning
        # model (see above), which spends a chunk of its output on visible
        # chain-of-thought before ever reaching its actual tool-call/JSON
        # answer. 1024 was sized for the old plain-instruct 8B model and
        # would routinely get a reasoning completion cut off mid-thought,
        # before it produced anything callers here could parse. Matches the
        # same fix trait_discovery_agent made for the same model family
        # (see its workflows/llm/client.py MAX_TOKENS comment).
        max_tokens=int(os.getenv("NVIDIA_LLM_MAX_TOKENS", "8192")),
        # Bumped from 12s -> 60s: the old short timeout was deliberately
        # tuned to fail fast on a *congested* worker pool so the
        # deterministic fallback kicked in quickly (see the removed comment
        # below this one in git history). A hybrid-reasoning model doesn't
        # just queue slower, it also *thinks* for several seconds before
        # replying even once it has a worker slot — 12s was cutting off
        # calls that were succeeding, not just ones that were stuck, which
        # meant the LLM path barely ever completed. 60s matches the
        # langchain_nvidia_ai_endpoints library default and gives the model
        # real room to finish; the cross-process flock in _llm_call_slot
        # still bounds how many calls can be waiting at once, so this
        # doesn't reintroduce a pile-up risk.
        timeout=float(os.getenv("NVIDIA_LLM_TIMEOUT", "60")),
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
    if _is_rate_limited_error(exc):
        return "NVIDIA endpoint rate-limited (429, over the free-tier RPM cap) — using fallback"
    if _is_capacity_error(exc):
        return "NVIDIA endpoint worker pool saturated (503, transient) — using fallback"
    # Fall back to the first line only, so unexpected errors still surface
    # without dragging the raw response dict into the log.
    return str(exc).splitlines()[0]


def _is_capacity_error(exc: Exception) -> bool:
    """True for 503 / ResourceExhausted worker-pool saturation.

    This is a "busy right now" signal, not a rate-limit signal — the local
    NIM worker pool is momentarily full, and it commonly clears within a
    couple seconds. Worth a short retry. Matches trait_discovery_agent's
    workflows/llm/client.py::_is_capacity_error.
    """
    message = str(exc).lower()
    return "503" in message or "resourceexhausted" in message or (
        "request limit" in message and "service unavailable" in message
    )


def _is_rate_limited_error(exc: Exception) -> bool:
    """True for 429 — you are over your requests-per-minute cap.

    Gets exactly one delayed retry (see RATE_LIMIT_RETRY_DELAY_SECONDS /
    invoke_with_retry below), not the same immediate exponential schedule
    as a 503: hammering a bucket that's already over its limit doesn't
    help, but a free-tier key trips this often enough that always failing
    straight to the deterministic fallback means the LLM path almost never
    actually runs. One extra ~10s wait per call is the compromise — slower,
    but it gets a live answer far more often.
    """
    message = str(exc).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message or "rate_limit" in message


def _is_transient_error(exc: Exception) -> bool:
    """True for any error worth retrying at all — capacity (503) or rate
    limit (429), each on its own schedule (see invoke_with_retry). Anything
    else (auth errors, bad requests, malformed tool calls) is left alone so
    it re-raises immediately and the existing fallback logic in
    query_router / capability_resolver / explanation_writer kicks in right
    away.
    """
    return _is_capacity_error(exc) or _is_rate_limited_error(exc)


# One-time delay before the single 429 retry. NVIDIA's free-tier RPM window
# is short enough that this is usually enough for the bucket to have room
# again, without making a skipped-LLM-step demo run painfully slow.
RATE_LIMIT_RETRY_DELAY_SECONDS = float(os.getenv("NVIDIA_LLM_RATE_LIMIT_RETRY_DELAY", "10"))


def invoke_with_retry(
    invoke_fn: Callable[[], _T],
    *,
    max_retries: int = 1,
    base_delay: float = 2.5,
) -> _T:
    """Call `invoke_fn()` through the pacing gate above.

    Two different retry schedules, both counted against max_retries:
      - Capacity (503): exponential backoff (base_delay * 2**attempt), on
        the theory that a saturated worker pool clears within seconds.
      - Rate limit (429): capped at exactly one retry, ever, after a fixed
        RATE_LIMIT_RETRY_DELAY_SECONDS wait — a deliberate compromise for a
        free tier that trips this often (see _is_rate_limited_error).
        Retried at most once regardless of max_retries, since a second 429
        in a row means the wait didn't help and further attempts are very
        unlikely to either.

    Every caller of this function (query_router, capability_resolver,
    explanation_writer, and the species_resolver / genome_metadata
    subagents) already has a fast, deterministic non-LLM fallback, so once
    both budgets are exhausted this raises and the caller falls back
    immediately rather than retrying further.

    The whole attempt sequence (including backoff sleeps between retries)
    runs inside one _llm_call_slot() acquisition, so a retry here can't
    interleave with another process's call and re-trigger the same
    concurrent-request-limit error it's trying to recover from.
    """
    last_exc: Exception | None = None
    rate_limit_retry_used = False
    with _llm_call_slot():
        for attempt in range(max_retries):
            try:
                return invoke_fn()
            except Exception as exc:
                last_exc = exc
                is_last_attempt = attempt == max_retries - 1
                if _is_rate_limited_error(exc):
                    if rate_limit_retry_used or is_last_attempt:
                        raise
                    rate_limit_retry_used = True
                    delay = RATE_LIMIT_RETRY_DELAY_SECONDS + random.uniform(0, 1.0)
                    logger.warning(
                        "NVIDIA endpoint rate-limited (429), retrying once in %.1fs: %s",
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    continue
                if not _is_capacity_error(exc) or is_last_attempt:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 1.0)
                logger.warning(
                    "Transient NVIDIA endpoint error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises