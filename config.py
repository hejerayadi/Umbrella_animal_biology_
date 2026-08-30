"""
Configuration centrale du projet.
Toutes les constantes sont lues ici une seule fois, puis importees partout ailleurs.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# IMPORTANT : cette taille doit correspondre exactement a la dimension de sortie
# du modele choisi ci-dessus.
#   all-MiniLM-L6-v2                          -> 384
#   pritamdeka/S-BioBert-snli-multinli-stsb   -> 768
VECTOR_SIZE = 384

# --- Collections Qdrant (une par module de l'agent Scientific Analysis) ---
COLLECTIONS = {
    "qa": "qa_corpus",                        # QA / RAG -> PubMedQA, BioASQ
    "contradiction": "contradiction_claims",  # Contradiction Detection -> SciFact
    "gap": "gap_orkg",                        # Gap Detection -> ORKG
}

# --- Fallback (Search Agent online) ---
# Si le meilleur score retourne par Qdrant est en dessous de ce seuil,
# on considere que la base locale n'a pas assez d'info et on declenche
# une recherche en ligne (PubMed, OpenAlex, etc.)
SCORE_THRESHOLD = 0.65

# --- Chunking ---
CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP = 100
