"""
Chargeurs des corpus statiques (fichiers telecharges dans data/).
Chaque fonction normalise sa source vers le meme format :
    {"id": ..., "text": ..., "source": ..., ...metadata}
Cette normalisation est ce que fait "Parsing & Cleaning" dans le schema
de l'Offline Ingestion Pipeline.
"""
import json


def load_pubmedqa(path: str) -> list[dict]:
    """
    Format attendu (ori_pqal.json de PubMedQA) :
        { "12345": {"QUESTION": "...", "CONTEXTS": ["...", "..."], ...}, ... }
    """
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


def load_scifact(claims_path: str, corpus_path: str) -> list[dict]:
    """
    Format attendu (fichiers .jsonl officiels de SciFact) :
        claims.jsonl : {"id": ..., "claim": "...", "cited_doc_ids": [...], "label": "..."}
        corpus.jsonl : {"doc_id": ..., "title": "...", "abstract": [...]}
    """
    documents = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            claim = json.loads(line)
            documents.append({
                "id": f"scifact_claim_{claim['id']}",
                "text": claim.get("claim", ""),
                "source": "SciFact",
                "label": claim.get("label", "NEI"),
                "cited_doc_ids": claim.get("cited_doc_ids", []),
            })
    return documents


def load_orkg_dump(path: str) -> list[dict]:
    """
    Format attendu : export JSON ORKG avec une liste de ressources/relations.
    A adapter selon le format exact de ton export (API ORKG ou dump officiel).
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    documents = []
    for entry in raw:
        documents.append({
            "id": f"orkg_{entry.get('id')}",
            "text": entry.get("label", "") + " " + entry.get("description", ""),
            "source": "ORKG",
            "entity_type": entry.get("type", "resource"),
        })
    return documents
