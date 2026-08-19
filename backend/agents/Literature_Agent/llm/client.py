from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(override=True)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def call_llm(messages: list[dict], max_completion_tokens: int = 500, reasoning_effort: str = "low") -> str:
    """Call the configured Azure OpenAI deployment."""
    if not DEPLOYMENT_NAME or not client.api_key:
        raise RuntimeError("Azure OpenAI is not configured for the Literature Agent.")

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )
    content = response.choices[0].message.content
    return content or ""
