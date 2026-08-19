import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    cloud_inference=True,
    timeout=120,
)

EMBEDDING_MODEL = "sentence-transformers/all-minilm-l6-v2"

COLLECTIONS = {
    "ghaya_papers_fulltext": 384,
    "ghaya_related_work": 384,
    "ghaya_literature_reviews": 384,
    "ghaya_citation_examples": 384,
}


def create_collections():
    for name, dim in COLLECTIONS.items():
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Collection creee : {name}")
        else:
            print(f"Deja existante : {name}")


def create_payload_indexes():
    for name in [
        "ghaya_papers_fulltext",
        "ghaya_related_work",
        "ghaya_literature_reviews",
    ]:
        client.create_payload_index(name, "domain", PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "section_type", PayloadSchemaType.KEYWORD)
    client.create_payload_index("ghaya_citation_examples", "intent", PayloadSchemaType.KEYWORD)


if __name__ == "__main__":
    create_collections()
    create_payload_indexes()