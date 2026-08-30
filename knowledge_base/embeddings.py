"""
Un seul endroit pour generer les embeddings.
Garantit que l'ingestion et la recherche utilisent exactement le meme modele
(erreur tres frequente sinon : dimensions ou espace vectoriel incoherents).
Etape 2 du cycle "Storing data in Qdrant" : Generate embeddings.
"""
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embeddings] Chargement du modele '{EMBEDDING_MODEL}'...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """Encode un seul texte en vecteur."""
    model = _get_model()
    return model.encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Encode une liste de textes en une seule passe (plus rapide que texte par texte)."""
    model = _get_model()
    vectors = model.encode(texts)
    return [v.tolist() for v in vectors]


if __name__ == "__main__":
    # Test manuel : python -m knowledge_base.embeddings
    vec = embed_text("does the MSTN gene affect muscle growth in cattle")
    print(f"Dimension du vecteur : {len(vec)}")
