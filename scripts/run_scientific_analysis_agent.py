"""
Test independant de l'agent Scientific Analysis, AVANT integration
a l'orchestrateur LangGraph global.

Usage :
    python scripts/run_scientific_analysis_agent.py "ta question ici"
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.scientific_analysis.graph import run_scientific_analysis_agent


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/run_scientific_analysis_agent.py "ta question ici"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = run_scientific_analysis_agent(query)

    print(f"\nQuery    : {query}")
    print(f"Status   : {result.status}")
    print(f"Tools    : {result.tool_calls_made}")
    if result.error_message:
        print(f"Erreur   : {result.error_message}")
    print(f"\n=== Reponse ===\n{result.answer}")


if __name__ == "__main__":
    main()
