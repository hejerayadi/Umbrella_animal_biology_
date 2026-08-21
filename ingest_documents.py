"""
Pipeline d'ingestion pour le domaine "migration animale".

Étapes :
1. Charger les documents (data.py)
2. Générer un embedding pour chaque document (sentence-transformers)
3. Stocker les vecteurs + métadonnées dans Qdrant
"""

import os
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from data_gbif import documents

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "animal_migration"

# 1. Connexion à Qdrant
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 2. Chargement du modèle d'embedding (léger, tourne en local, pas de clé API nécessaire)
print("Chargement du modèle d'embedding...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Génération des embeddings + préparation des points à insérer
points = []
for doc in documents:
    vector = model.encode(doc["text"]).tolist()

    point = PointStruct(
        id=str(uuid.uuid4()),  # identifiant unique pour chaque document
        vector=vector,
        payload={
            "text": doc["text"],
            "species": doc["species"],
            "region": doc["region"],
            "source": doc["source"],
        },
    )
    points.append(point)
    print(f"  -> embedding généré pour: {doc['species']}")

# 4. Envoi vers Qdrant
client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"\n✅ {len(points)} documents indexés dans la collection '{COLLECTION_NAME}'")