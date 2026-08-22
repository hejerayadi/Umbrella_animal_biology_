"""
Pipeline de retrieval pour le domaine "migration animale".

Étapes :
1. Prendre une question texte
2. La transformer en vecteur (même modèle que pour l'ingestion)
3. Chercher les documents les plus proches dans Qdrant
4. (Bonus) Filtrer par métadonnées
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "animal_migration"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant a besoin d'un index explicite sur un champ pour pouvoir filtrer dessus.
# On le crée une fois ici (si il existe déjà, Qdrant ignore silencieusement la demande).
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="species",
    field_schema="keyword",
)


def search(query: str, top_k: int = 3, species_filter: str = None):
    query_vector = model.encode(query).tolist()

    # Filtrage optionnel par métadonnées (ex: uniquement les baleines)
    query_filter = None
    if species_filter:
        query_filter = Filter(
            must=[FieldCondition(key="species", match=MatchValue(value=species_filter))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    )

    print(f"\n🔍 Requête: '{query}'" + (f" (filtré sur: {species_filter})" if species_filter else ""))
    print("-" * 60)
    for point in results.points:
        print(f"Score: {point.score:.3f} | Espèce: {point.payload['species']}")
        print(f"  -> {point.payload['text'][:120]}...")
    print()


if __name__ == "__main__":
    # Test 1 : recherche générale
    search("comment les oiseaux migrent en hiver")

    # Test 2 : recherche sur les mammifères marins
    search("animaux qui traversent l'océan")

    # Test 3 : recherche avec filtre sur une espèce précise
    search("migration", species_filter="saumon du pacifique")