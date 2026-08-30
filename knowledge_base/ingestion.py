"""
Chunking + construction des points + upsert vers Qdrant.
Etapes 2, 3 et 4 du cycle "Storing data in Qdrant" :
  Generate embeddings -> Build the point -> Upsert into Qdrant.
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import CHUNK_MAX_CHARS, CHUNK_OVERLAP
from knowledge_base.embeddings import embed_batch

# Namespace fixe pour generer des UUID deterministes a partir de nos IDs texte.
# Deterministe = le meme doc_id/chunk donnera toujours le meme UUID,
# donc un upsert() repete met a jour le point existant au lieu d'en creer un doublon.
_UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def _to_qdrant_id(readable_id: str) -> str:
    """Convertit un ID lisible (ex: 'pubmedqa_10001_0') en UUID valide pour Qdrant."""
    return str(uuid.uuid5(_UUID_NAMESPACE, readable_id))


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Decoupe un texte long en morceaux avec chevauchement, pour ne pas couper le sens."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars - overlap
    return chunks


def ingest_documents(client: QdrantClient, collection_name: str, documents: list[dict]) -> int:
    """
    Ingere une liste de documents dans une collection Qdrant.

    Chaque document doit etre un dict avec au minimum :
      - "id"   : identifiant unique du document source
      - "text" : le texte a indexer
      (+ tout autre champ de metadata : source, species, date, label, etc.)

    Retourne le nombre de points inseres.
    """
    points: list[PointStruct] = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        vectors = embed_batch(chunks)

        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            readable_id = f"{doc['id']}_{i}"
            points.append(
                PointStruct(
                    id=_to_qdrant_id(readable_id),
                    vector=vector,
                    # On garde l'ID lisible dans le payload (utile pour debug/affichage),
                    # meme si l'ID Qdrant lui-meme doit etre un UUID/entier.
                    payload={**doc, "doc_chunk_id": readable_id, "chunk_index": i, "text": chunk},
                )
            )

    if not points:
        print(f"[ingestion] Aucun point a inserer pour '{collection_name}'.")
        return 0

    # Un seul appel batch : bien plus rapide que d'inserer point par point.
    client.upsert(collection_name=collection_name, points=points)
    print(f"[ingestion] {len(points)} points inseres dans '{collection_name}'.")
    return len(points)