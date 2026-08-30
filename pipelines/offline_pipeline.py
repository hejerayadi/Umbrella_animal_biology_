"""
Offline Ingestion Pipeline (flux de gauche dans le schema) :
    Sources -> Parsing & Cleaning -> Chunking -> Embedding -> Vector Database

Le chunking et l'embedding sont geres dans knowledge_base/ingestion.py ;
ce fichier s'occupe du nettoyage avant ingestion.
"""
from qdrant_client import QdrantClient

from knowledge_base.ingestion import ingest_documents


def clean_text(raw_text: str) -> str:
    """Nettoyage basique : espaces normalises, texte strippe."""
    text = raw_text.strip()
    text = " ".join(text.split())
    return text


def run_offline_ingestion(client: QdrantClient, documents: list[dict], collection_name: str) -> int:
    """
    Nettoie une liste de documents puis les ingere dans la collection donnee.
    A appeler une fois par source (PubMedQA -> qa_corpus, SciFact -> contradiction_claims, etc.)
    """
    cleaned = []
    for doc in documents:
        if not doc.get("text"):
            continue
        cleaned.append({**doc, "text": clean_text(doc["text"])})

    return ingest_documents(client, collection_name, cleaned)
