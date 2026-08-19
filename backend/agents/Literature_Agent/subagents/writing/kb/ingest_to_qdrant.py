import time
import uuid
from qdrant_client.models import PointStruct, Document
from .qdrant_setup import client, EMBEDDING_MODEL

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def upsert_batch(collection_name: str, texts: list[str], payloads: list[dict], ids: list[int | str], batch_size: int = 20):
    """ids : identifiants lisibles (ex. "12345_body_3"), stockes dans le
    payload sous "chunk_id". Le point ID reel est un UUID deterministe
    derive de ce chunk_id (uuid5) : relancer le script met a jour les
    memes points au lieu de les dupliquer."""
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        points = [
            PointStruct(
                id=str(uuid.uuid5(NAMESPACE, str(readable_id))),
                vector=Document(text=text, model=EMBEDDING_MODEL),
                payload={**payload, "chunk_id": readable_id},
            )
            for readable_id, text, payload in zip(batch_ids, batch_texts, batch_payloads)
        ]

        for attempt in range(3):
            try:
                client.upsert(collection_name=collection_name, points=points)
                print(f"{i + len(batch_texts)}/{len(texts)} points inseres dans {collection_name}", flush=True)
                break
            except Exception as e:
                print(f"  Erreur sur ce batch (tentative {attempt + 1}/3) : {e}", flush=True)
                time.sleep(3)
        else:
            print(f"  Batch {i}-{i+len(batch_texts)} abandonne apres 3 tentatives", flush=True)