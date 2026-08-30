"""
Module QA / RAG : repond a une question en s'appuyant sur qa_corpus
(PubMedQA, BioASQ), avec fallback online si necessaire.
"""
from qdrant_client import QdrantClient

from config import COLLECTIONS
from knowledge_base.retrieval import retrieve_with_fallback


def answer_question(client: QdrantClient, question: str, top_k: int = 3):
    """
    Recupere le contexte pertinent pour une question donnee.
    Le contexte retourne est ensuite a passer a un LLM (Azure OpenAI)
    pour generer la reponse finale.
    """
    results = retrieve_with_fallback(client, COLLECTIONS["qa"], question, top_k=top_k)
    context = "\n".join(r.payload.get("text", "") for r in results)
    return context, results
