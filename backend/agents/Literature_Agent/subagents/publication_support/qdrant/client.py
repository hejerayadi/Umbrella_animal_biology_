import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient


# Addressed by absolute path: this package is imported from the orchestrator,
# whose working directory is the repository root, so a bare load_dotenv() finds
# backend/.env - which has none of these keys - and the cluster URL would come
# back empty. Falls back to the Literature_Agent .env for the values shared
# with the other subagents.
_PKG = Path(__file__).resolve().parents[1]
load_dotenv(_PKG / ".env", override=False)
load_dotenv(_PKG.parents[1] / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    """Read a setting, tolerating the `KEY = "value"` spacing used in the .env
    already sitting in this folder."""
    return (os.getenv(name) or default).strip().strip('"').strip("'")


COLLECTION_NAME = _env("QDRANT_JOURNALS_COLLECTION", "journals")

# Vector size of BAAI/bge-small-en-v1.5, the model in embeddings.embedder.
# Change both together or the collection will reject the vectors.
VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Build the Qdrant client every module shares.

    QdrantClient(url=None) silently falls back to localhost, so a missing
    URL would look like a working cloud connection while actually reading a
    local instance. Fail loudly instead.
    """

    # QDRANT_URL is what the rest of the repository calls this. CLUSTER_ENDPOINT
    # is kept as an alias so an existing .env in this folder keeps working.
    url = _env("QDRANT_URL") or _env("CLUSTER_ENDPOINT")
    api_key = _env("QDRANT_API_KEY")

    if not url:
        raise RuntimeError(
            "QDRANT_URL is not set. Add it to .env:\n"
            "  QDRANT_URL=https://<cluster>.cloud.qdrant.io\n"
            "  QDRANT_API_KEY=<your-key>"
        )

    if not api_key:
        raise RuntimeError(
            "QDRANT_API_KEY is not set. Add it to .env."
        )

    # The 5s default is fine locally but too tight for batch upserts
    # over the network.
    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=60,
    )
