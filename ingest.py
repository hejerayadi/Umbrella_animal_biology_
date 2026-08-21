import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Charge les variables depuis le fichier .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Créer la collection pour le domaine "migration animale"
client.create_collection(
    collection_name="animal_migration",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

print("Collection créée ✅")
print(client.get_collections())