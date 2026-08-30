"""
Point d'entree : teste une requete de bout en bout (recherche locale + fallback).

Usage :
    python scripts/run_search_agent.py "does MSTN gene affect muscle growth in cattle"
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLLECTIONS
from knowledge_base.qdrant_client import get_client
from agents.scientific_analysis.qa_module import answer_question


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_search_agent.py "ta question ici"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    client = get_client()

    context, results = answer_question(client, question)

    print(f"\nQuestion : {question}\n")
    print("=== Contexte recupere ===")
    print(context)
    print("\n=== Details des resultats ===")
    for r in results:
        print(f"score={r.score:.3f} | source={r.payload.get('source')} | id={r.id}")


if __name__ == "__main__":
    main()
