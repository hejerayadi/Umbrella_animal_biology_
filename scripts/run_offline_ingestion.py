"""
Point d'entree : ingere les datasets statiques dans Qdrant.

Usage :
    python scripts/run_offline_ingestion.py
"""
import sys
import os

# Permet d'executer ce script directement (python scripts/run_offline_ingestion.py)
# en trouvant les modules du projet a la racine.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLLECTIONS
from knowledge_base.qdrant_client import get_client, create_all_collections
from pipelines.offline_pipeline import run_offline_ingestion
from data_sources.datasets_loader import load_pubmedqa


def main():
    client = get_client()

    print("=== Etape 1 : creation des collections ===")
    create_all_collections(client)

    print("\n=== Etape 2 : ingestion PubMedQA -> qa_corpus ===")
    qa_docs = load_pubmedqa("data/pubmedqa_sample.json")
    run_offline_ingestion(client, qa_docs, COLLECTIONS["qa"])

    # Une fois que tu as telecharge les vrais fichiers SciFact et ORKG dans data/,
    # decommente et adapte ces lignes :
    #
    # from data_sources.datasets_loader import load_scifact, load_orkg_dump
    #
    # print("\n=== Ingestion SciFact -> contradiction_claims ===")
    # scifact_docs = load_scifact("data/scifact/claims_train.jsonl", "data/scifact/corpus.jsonl")
    # run_offline_ingestion(client, scifact_docs, COLLECTIONS["contradiction"])
    #
    # print("\n=== Ingestion ORKG -> gap_orkg ===")
    # orkg_docs = load_orkg_dump("data/orkg/orkg_dump.json")
    # run_offline_ingestion(client, orkg_docs, COLLECTIONS["gap"])

    print("\n=== Termine ===")


if __name__ == "__main__":
    main()
