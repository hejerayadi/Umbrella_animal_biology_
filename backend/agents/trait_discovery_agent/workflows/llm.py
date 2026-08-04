import os
from functools import lru_cache

from langchain_nvidia_ai_endpoints import ChatNVIDIA

DEFAULT_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.3-70b-instruct")


def _get_api_key() -> str | None:
    return os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")


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
    )