import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# The agent's .env lives at the package root, one level up from workflows/.
# Two things matter about where this call sits: it uses an explicit path
# rather than a search, so it works no matter which directory the command was
# run from; and it runs BEFORE the module-level os.getenv calls below, so
# NIM_MODEL and NIM_BASE_URL are picked up from .env too, not just the API
# key read later inside get_llm(). load_dotenv does not overwrite variables
# already set in the shell, so an explicit export still wins.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_MODEL = os.getenv("NIM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
DEFAULT_BASE_URL = os.getenv("NIM_BASE_URL")
FALLBACK_MODELS = (
    DEFAULT_MODEL,
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.3-70b-instruct",
)


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


async def invoke_with_fallback(
    prompt: ChatPromptTemplate,
    payload: dict,
    *,
    temperature: float = 0.1,
    model: str | None = None,
):
    """Invoke a prompt against NIM, retrying with a known-good fallback model on 404s.

    The account behind this workspace can expose different model/function IDs over time.
    When one alias disappears, the graph should still be able to complete with an
    alternate model rather than failing hard at the first escalation node.
    """
    candidate_models = []
    if model:
        candidate_models.append(model)
    candidate_models.extend(m for m in FALLBACK_MODELS if m not in candidate_models)

    last_error: Exception | None = None
    for candidate_model in candidate_models:
        llm = get_llm(temperature=temperature, model=candidate_model)
        chain = prompt | llm
        try:
            return await chain.ainvoke(payload)
        except Exception as exc:  # pragma: no cover - exercised in live NIM runs
            last_error = exc
            if not _is_missing_model_error(exc):
                raise

    if last_error is not None:
        raise last_error
    raise RuntimeError("NIM invocation failed without raising an exception")