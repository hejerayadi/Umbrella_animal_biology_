import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient


load_dotenv()


COLLECTION_NAME = "journals"

# Vector size of BAAI/bge-small-en-v1.5, the model in embeddings.embedder.
# Change both together or the collection will reject the vectors.
VECTOR_SIZE = 384


def get_client() -> QdrantClient:
    """Build the Qdrant client every module shares.

    QdrantClient(url=None) silently falls back to localhost, so a missing
    CLUSTER_ENDPOINT would look like a working cloud connection while
    actually reading a local instance. Fail loudly instead.
    """

    url = os.getenv("CLUSTER_ENDPOINT")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise RuntimeError(
            "CLUSTER_ENDPOINT is not set. Add it to .env:\n"
            "  CLUSTER_ENDPOINT=https://<cluster>.cloud.qdrant.io\n"
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
