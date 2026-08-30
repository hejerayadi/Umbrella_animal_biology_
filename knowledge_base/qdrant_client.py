"""
Connexion a Qdrant + creation des collections.
Etape 1 du cycle "Storing data in Qdrant" : Create a collection.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import QDRANT_URL, QDRANT_API_KEY, VECTOR_SIZE, COLLECTIONS


def get_client() -> QdrantClient:
    """Retourne un client connecte a l'instance Qdrant (Cloud ou locale)."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_all_collections(client: QdrantClient) -> None:
    """Cree les 3 collections du Scientific Analysis Agent si elles n'existent pas deja."""
    for key, name in COLLECTIONS.items():
        if client.collection_exists(name):
            print(f"[skip] Collection '{name}' existe deja.")
            continue

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[ok] Collection '{name}' creee ({key}).")


if __name__ == "__main__":
    # Test manuel : python -m knowledge_base.qdrant_client
    client = get_client()
    create_all_collections(client)
    print(client.get_collections())
