import asyncio
import logging
import os
import random
from functools import lru_cache

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# python-dotenv is already a declared dependency (requirements.txt) but was
# never actually invoked anywhere in the codebase -- .env only got picked up
# automatically under docker-compose (env_file:). A bare `python
# evaluation/run_eval.py` (or any local, non-docker invocation) read nothing
# from .env and silently fell back to whatever's already exported in the
# shell -- or, worse, to DEFAULT_MODEL's hardcoded value below if nothing
# was exported at all. Loading it here, at import time of the one module
# every LLM call routes through, means every entry point gets it for free.
# override=False: real environment variables (e.g. a CI secret) still win
# over .env, matching docker-compose's own env_file semantics.
load_dotenv(override=False)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("NIM_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
DEFAULT_BASE_URL = os.getenv("NIM_BASE_URL")

# Additional models to try, in order, if DEFAULT_MODEL 404/410s (retired,
# renamed, or never enabled on this key), or if it keeps timing out/500ing
# after its retry budget (see _is_advance_worthy_error). Comma-separated env
# var so an operator can widen/reorder this without a code change;
# NIM_MODEL itself always tried first.
#
# probe_all_models.py (raw HTTP, no LangChain, no retry/fallback machinery --
# see that file for the full method) hit every plain-instruct model this
# chain used to lean on -- Llama 3.1, Granite, Phi-4, Mistral-Small,
# Qwen3-Instruct, Kimi-K2, mistral-nemotron, the old Nemotron-Super lines --
# and found 41 of 46 candidates dead (404/410/error) on this key as of Aug
# 2026. Only 5 came back ALIVE, and all 5 are hybrid-reasoning models:
#   nvidia/nemotron-3-nano-30b-a3b
#   nvidia/nemotron-3-super-120b-a12b
#   nvidia/nemotron-3-ultra-550b-a55b
#   openai/gpt-oss-20b
#   openai/gpt-oss-120b
# (The two GPT-OSS entries came back with `content: None` even on that
# probe's 5-token budget -- not an error, just the hidden "analysis" channel
# never finishing in time. See the MAX_TOKENS/timeout comments below and
# NonJSONFinalAnswerError for how that's handled here.)
#
# There is currently no live plain-instruct model on this key at all, so the
# old "plain instruct models first, reasoning models as a last resort" chain
# is no longer just non-optimal, it's *empty* -- every entry in it 404s
# before ever reaching a model that would actually answer. This chain is
# rebuilt entirely around the 5 confirmed-alive models instead: see
# MAX_TOKENS and get_llm's `timeout` for how the visible/hidden
# chain-of-thought these models produce is accommodated rather than avoided.
# _reasoning_off_preamble is still sent (harmless if ignored) but is no
# longer the primary defense -- nemotron-3-super-120b-a12b was observed
# reasoning fully despite it, so it can't be relied on alone.
#
# Ordering: smaller/lighter models first (faster, better shot at spare
# shared free-tier capacity), alternating Nemotron/GPT-OSS so one
# publisher's bad day doesn't take out consecutive candidates. The two
# `-a<N>b` MoE models with the smallest active-parameter counts lead;
# nemotron-3-ultra-550b-a55b -- by far the largest and slowest of the five --
# is the last resort.
# Re-run probe_all_models.py periodically (this catalog is being pruned
# unusually aggressively) and prune/reorder this if entries go stale, or if
# plain-instruct models come back alive and should retake the front of the
# chain for latency's sake.
_EXTRA_FALLBACKS = tuple(
    m.strip() for m in os.getenv(
        "NIM_FALLBACK_MODELS",
        "nvidia/nemotron-3-nano-30b-a3b,"
        "openai/gpt-oss-20b,"
        "nvidia/nemotron-3-super-120b-a12b,"
        "openai/gpt-oss-120b,"
        "nvidia/nemotron-3-ultra-550b-a55b",
    ).split(",")
    if m.strip()
)
FALLBACK_MODELS = tuple(dict.fromkeys((DEFAULT_MODEL, *_EXTRA_FALLBACKS)))

# 8192 rather than the previous 2048: every model in FALLBACK_MODELS is now a
# hybrid-reasoning model with no plain-instruct model shielding it, and
# nemotron-3-super-120b-a12b has been observed reasoning fully -- burning the
# *entire* previous 2048-token budget on visible chain-of-thought -- even
# with the reasoning-off preamble in place, so that preamble can no longer
# be trusted as the thing that keeps completions short. Token budget is now
# the primary defense: give these models enough room to think AND still
# reach the JSON final answer, rather than trying to suppress the thinking
# in the first place. GPT-OSS's hidden "analysis" channel is a related but
# distinct case -- see NonJSONFinalAnswerError below for what happens if it
# still doesn't finish in time.
# 16384 rather than 8192: BRCA1-style genes with a dozen+ GO candidates hit
# resolve_go_term_names(go_ids=[...]) as ONE batched tool call (see
# subagents/gene_mapper/llm_pick.py) -- when a reasoning model's visible
# chain-of-thought already ate deep into the budget before it starts
# emitting that call, there isn't enough left to finish a long go_ids list,
# and the completion gets cut off mid-argument (observed verbatim:
# go_ids='["GO:0051726", "GO:0008630", "GO:0' -- a truncated JSON string,
# not garbage). See _is_truncated_completion below for how that's now
# detected and treated as advance-worthy instead of crashing the whole
# decision with an opaque pydantic ValidationError three layers away from
# the actual cause.
MAX_TOKENS = int(os.getenv("NIM_MAX_TOKENS", "16384"))

# langchain_nvidia_ai_endpoints defaults to a 60s timeout (see its
# _NVIDIAClient.timeout) -- fine for a plain-instruct model's near-instant
# reply, but a real constraint now that every candidate in FALLBACK_MODELS
# produces thousands of tokens of chain-of-thought before its final answer.
# 180s gives the larger reasoning models (nemotron-3-ultra-550b-a55b,
# openai/gpt-oss-120b) realistic room to actually finish an 8192-token
# completion rather than getting cut off mid-thought and counted as a
# _is_timeout_error advance-worthy failure before they had a fair shot.
REQUEST_TIMEOUT_SECONDS = float(os.getenv("NIM_REQUEST_TIMEOUT_SECONDS", "180"))

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
        max_tokens=MAX_TOKENS,
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
        # Client transport option, not a generation param -- ChatNVIDIA pops
        # it out of kwargs itself (see REQUEST_TIMEOUT_SECONDS above for why
        # the library's 60s default no longer fits this model chain).
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _reasoning_off_preamble() -> str:
    """Prefix for the first system message, ASKING any reasoning-capable
    model in FALLBACK_MODELS to skip its chain-of-thought and answer
    directly. Kept because it's free and occasionally still helps, but it is
    NOT the primary defense against long/truncated completions any more --
    nemotron-3-super-120b-a12b has been observed reasoning fully despite it.
    Since every model in FALLBACK_MODELS is now a hybrid-reasoning model
    (see that comment), MAX_TOKENS and get_llm's `timeout` are what actually
    keep completions from getting cut off mid-thought; this preamble is best
    read as a latency optimization that sometimes pays off, not a guarantee.

    Stacks the different per-family directives NVIDIA documents (see
    docs.nvidia.com/nim/large-language-models/latest/reasoning-model.html
    and docs.nvidia.com/rag/2.5.0/enable-nemotron-thinking.html): Nemotron
    1.5 / Nano use '/no_think', older Nemotron reasoning models use the
    literal phrase 'detailed thinking off'. A model that doesn't recognize
    either is expected to just treat this as inert prefix text ahead of the
    real system prompt -- harmless for non-reasoning models. (NVIDIA's
    ChatNVIDIA client also exposes a structured `thinking_mode=False` kwarg
    on invoke/ainvoke that resolves the right per-model toggle from its own
    catalog metadata instead of this hand-rolled string -- worth trying if a
    future model in this chain ignores both phrases above -- but it hasn't
    been adopted here since the string form already covers every model
    currently in FALLBACK_MODELS and, per above, neither approach is being
    relied on as the primary defense right now.)"""
    return "/no_think\ndetailed thinking off\n\n"


def _is_missing_model_error(exc: Exception) -> bool:
    """True for '404 not found' (never enabled on this key), '410 Gone'
    (retired/EOL'd model, e.g. NVIDIA NIM catalog deprecations), and the
    malformed variant observed as '[###] Unknown Error\\npage not found' --
    a garbled/unparseable status code (literal '###') paired with what's
    almost certainly a truncated 'Page not found'. This happens when the
    gateway returns something that isn't clean JSON for a route that no
    longer resolves to a real model backend, so langchain_nvidia_ai_endpoints'
    own status-code parser has nothing valid to put in the brackets. All
    three mean the same thing -- 'this model id isn't servable right now' --
    so all three should advance to the next candidate in FALLBACK_MODELS
    rather than aborting the request. 'page not found' is matched
    unconditionally (not gated on a status-code digit, since here there
    isn't a usable one) -- specific enough a phrase that it shouldn't
    collide with an unrelated error."""
    message = str(exc).lower()
    if "410" in message and "gone" in message:
        return True
    if "404" in message and ("not found" in message or "function" in message):
        return True
    return "page not found" in message


class NonJSONFinalAnswerError(RuntimeError):
    """Raised by json_completion.py / tool_loop.py when a model's final
    answer isn't parseable JSON -- including a bare `None` content, observed
    on both GPT-OSS models: they route chain-of-thought through a separate
    hidden "analysis" channel (surfaced by langchain_nvidia_ai_endpoints as
    `additional_kwargs["reasoning_content"]`, not `.content`) and can hit
    max_tokens before that channel ever hands off to the visible completion
    -- `.content` is then simply None, not an error, so nothing upstream of
    _parse_json_object caught this until now.

    Treated as advance-worthy (see _is_advance_worthy_error): MAX_TOKENS is
    already generous account-wide, so if a model still doesn't get to a
    parseable final answer within it, trying the next candidate in
    FALLBACK_MODELS is strictly better than hard-failing the whole
    gene/pathway/protein decision -- same reasoning as a persistent timeout
    or 5xx."""


class TruncatedCompletionError(NonJSONFinalAnswerError):
    """Raised when a completion was cut off mid-generation by max_tokens --
    either the API says so directly (finish_reason == "length", see
    _is_truncated_completion) or the shape of a tool-call argument makes it
    unmistakable, e.g. a string that starts like a JSON array/object but
    fails to parse (go_ids='["GO:0051726", "GO:0008630", "GO:0' -- observed
    verbatim on a BRCA1 gene-mapper decision: a dozen+ candidate GO ids
    batched into one resolve_go_term_names call ran out of room mid-list).

    Previously this reached _coerce_stringified_json_args, which silently
    left the truncated string as-is (its docstring only ever anticipated a
    *complete* JSON-encoded string, not a cut-off one), and the tool's own
    Pydantic schema then raised a plain ValidationError several layers away
    from the actual cause -- a plain ValidationError isn't advance-worthy,
    so it aborted the entire gene/pathway/protein decision instead of
    trying the next candidate model, discarding a real LLM pick every time
    this happened. Subclassing NonJSONFinalAnswerError means the existing
    isinstance check in _is_advance_worthy_error covers this for free."""


def _is_truncated_completion(ai_msg) -> bool:
    """True if the API itself reports this completion was cut off by
    max_tokens (finish_reason == "length") rather than completing normally
    (finish_reason == "stop"/"tool_calls"). Checked immediately after every
    LLM turn in tool_loop.py, before any attempt to parse `.content` or
    execute a tool call with what might be truncated arguments -- catching
    this here is strictly more reliable than inferring it after the fact
    from a parse failure (see TruncatedCompletionError), since not every
    truncation produces an obviously-malformed value (a short go_ids list
    cut off exactly on an element boundary would look valid, just
    incomplete)."""
    return (getattr(ai_msg, "response_metadata", None) or {}).get("finish_reason") == "length"


def _is_advance_worthy_error(exc: Exception) -> bool:
    """Whether _candidate_models' caller should move on to the NEXT model in
    FALLBACK_MODELS rather than raising. True for missing-model errors
    (above), a timeout that _retry_on_capacity already retried and gave up
    on, a persistent 5xx server error, and a non-JSON/None final answer
    (NonJSONFinalAnswerError) -- each of these means the model is, for the
    purposes of finishing this request, exactly as unusable as one that
    404s, and there's no reason to sacrifice the whole gene/pathway/protein
    decision (falling back to a deterministic heuristic) when another model
    in the chain might just work. Rate-limit and capacity (503) errors are
    NOT included here: those are properties of your account/quota or of
    NIM's shared worker pool, not the specific model, so switching models
    wouldn't help and would just mask the real signal (slow down / check
    quota)."""
    return (
        isinstance(exc, NonJSONFinalAnswerError)
        or _is_missing_model_error(exc)
        or _is_timeout_error(exc)
        or _is_server_error(exc)
    )


def _is_capacity_error(exc: Exception) -> bool:
    """503s from a saturated local NIM worker pool ('request limit reached',
    ResourceExhausted) are transient — worth a short retry, not an immediate
    fallback or model swap."""
    message = str(exc).lower()
    return "503" in message or "resourceexhausted" in message or (
        "request limit" in message and "service unavailable" in message
    )


def _is_timeout_error(exc: Exception) -> bool:
    """Raw socket/connection timeouts ('Timeout on reading data from
    socket', asyncio.TimeoutError, etc.) rather than an HTTP status code --
    langchain_nvidia_ai_endpoints' default 60s socket timeout with no retry
    of its own means one slow/cold-starting request previously killed the
    whole gene outright. Treated like _is_capacity_error: worth a handful of
    short retries (a stuck socket is often just a stuck socket), and unlike
    capacity/rate-limit errors, ALSO an advance-worthy error for
    _candidate_models (see _is_missing_model_error) -- a model that keeps
    timing out even after retries is as unusable right now as one that
    404s, and there's no reason to give up on the whole request rather than
    trying the next candidate."""
    message = str(exc).lower()
    return "timeout" in message or isinstance(exc, (TimeoutError, asyncio.TimeoutError))


def _is_server_error(exc: Exception) -> bool:
    """Generic 5xx from NIM's own gateway -- observed as both '[500]
    Internal Server Error' and the more opaque '[500] Unknown Error', each
    paired with a body like 'Inference connection error while making
    inference request'. Deliberately excludes anything _is_capacity_error
    already owns (503 specifically): that's a distinct, well-understood
    worker-pool-exhaustion signal with its own retry budget. A bare 500 here
    means NIM's gateway itself failed to reach the underlying model
    backend -- usually just as transient as a 503, so it gets a short retry,
    and (like a persistent timeout) is advance-worthy on FALLBACK_MODELS if
    it doesn't clear."""
    if _is_capacity_error(exc):
        return False
    message = str(exc).lower()
    return (
        "500" in message
        or "internal server error" in message
        or "inference connection error" in message
    )


MAX_TIMEOUT_RETRIES = int(os.getenv("NIM_MAX_TIMEOUT_RETRIES", "2"))
TIMEOUT_RETRY_BASE_SECONDS = float(os.getenv("NIM_TIMEOUT_RETRY_BASE_SECONDS", "2"))

MAX_SERVER_ERROR_RETRIES = int(os.getenv("NIM_MAX_SERVER_ERROR_RETRIES", "2"))
SERVER_ERROR_RETRY_BASE_SECONDS = float(os.getenv("NIM_SERVER_ERROR_RETRY_BASE_SECONDS", "2"))


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
    exponential backoff on transient worker-pool exhaustion (503), request-
    rate limiting (429), or raw socket timeouts — the rate-limit case gets a
    much longer backoff and a larger retry budget since it's a
    slower-clearing condition than the other two (see MAX_RATE_LIMIT_RETRIES
    vs MAX_CAPACITY_RETRIES / MAX_TIMEOUT_RETRIES)."""
    last_error: Exception | None = None
    attempt = 0
    while True:
        await _throttle()
        try:
            return await coro_fn()
        except Exception as exc:
            is_rate_limited = _is_rate_limit_error(exc)
            is_capacity = (not is_rate_limited) and _is_capacity_error(exc)
            is_timeout = (not is_rate_limited) and (not is_capacity) and _is_timeout_error(exc)
            is_server_error = (
                not (is_rate_limited or is_capacity or is_timeout) and _is_server_error(exc)
            )
            if not (is_rate_limited or is_capacity or is_timeout or is_server_error):
                raise
            if is_rate_limited:
                max_retries = MAX_RATE_LIMIT_RETRIES
            elif is_capacity:
                max_retries = MAX_CAPACITY_RETRIES
            elif is_timeout:
                max_retries = MAX_TIMEOUT_RETRIES
            else:
                max_retries = MAX_SERVER_ERROR_RETRIES
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
            elif is_capacity:
                delay = CAPACITY_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "NIM worker pool exhausted (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
            elif is_timeout:
                delay = TIMEOUT_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "NIM socket timeout (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, exc,
                )
            else:
                delay = SERVER_ERROR_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(
                    "NIM server error [5xx] (attempt %d/%d), retrying in %.1fs: %s",
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


def _parse_json_object(content: str | None) -> dict | None:
    """Best-effort extraction of a single JSON object from a model's final text,
    tolerating markdown code fences.

    `content` can be None -- observed on both GPT-OSS models when their
    hidden "analysis" channel hasn't handed off to the visible completion
    channel yet (see NonJSONFinalAnswerError). Treated the same as any other
    unparseable answer rather than raising AttributeError on `.strip()`."""
    import json

    if not content:
        return None
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