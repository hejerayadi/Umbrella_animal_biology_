"""
Peuple les 3 collections Qdrant du Scientific Analysis agent avec l'echantillon
PubMedQA fourni. A lancer une fois pour tester, puis remplacer par les vrais
corpus (PubMedQA complet, SciFact, ORKG) une fois telecharges.

Usage (depuis la racine du repo, ou le module est importable) :
    python -m backend.agents.Literature_Agent.subagents.scientific_analysis.run_offline_ingestion
"""
import json
from pathlib import Path

from .qdrant_kb.client import get_client, create_all_collections
from .config import COLLECTIONS


def load_pubmedqa(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    documents = []
    for doc_id, entry in raw.items():
        question = entry.get("QUESTION", "")
        contexts = " ".join(entry.get("CONTEXTS", []))
        documents.append({
            "id": f"pubmedqa_{doc_id}",
            "text": f"{question} {contexts}".strip(),
            "source": "PubMedQA",
        })
    return documents


def main():
    from .qdrant_kb.ingestion import ingest_documents

    client = get_client()
    print("=== Creation des collections ===")
    create_all_collections(client)

    sample_path = Path(__file__).parent / "data" / "pubmedqa_sample.json"
    print(f"\n=== Ingestion {sample_path.name} -> {COLLECTIONS['qa']} ===")
    docs = load_pubmedqa(str(sample_path))
    count = ingest_documents(client, COLLECTIONS["qa"], docs)
    print(f"{count} points inseres.")


if __name__ == "__main__":
    main()
