"""Azure embeddings for the ingestion scripts.

NOT used by retrieval: `qdrant_setup.py` builds its client with
`cloud_inference=True`, so Qdrant embeds query and payload text itself and the
serving path never calls Azure. This module exists for the `ingest_*.py`
scripts and for computing vectors outside Qdrant.

The client is built on first use, not at import time - the same reason spelled
out in `qdrant_setup.py` and `llm/client.py`: a module-scope client turns a
missing credential into an ImportError that no caller can catch.
"""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

# parents[3] is Literature_Agent/, addressed explicitly rather than by walking
# up from the working directory.
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


@lru_cache(maxsize=1)
def _client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )


def _deployment() -> str:
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
    if not deployment:
        raise RuntimeError(
            "AZURE_EMBEDDING_DEPLOYMENT is not set. See .env.example."
        )
    return deployment


def embed_text(text: str) -> list[float]:
    response = _client().embeddings.create(input=text, model=_deployment())
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    response = _client().embeddings.create(input=texts, model=_deployment())
    return [d.embedding for d in response.data]
