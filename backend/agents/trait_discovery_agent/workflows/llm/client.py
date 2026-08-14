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


async def _retry_on_capacity(coro_fn):
    """Call coro_fn() (a zero-arg async callable), retrying with jittered
    exponential backoff on transient worker-pool exhaustion (503)."""
    last_error: Exception | None = None
    for attempt in range(MAX_CAPACITY_RETRIES + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            if not _is_capacity_error(exc) or attempt == MAX_CAPACITY_RETRIES:
                raise
            last_error = exc
            delay = CAPACITY_RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(
                "NIM worker pool exhausted (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1, MAX_CAPACITY_RETRIES, delay, exc,
            )
            await asyncio.sleep(delay)
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
