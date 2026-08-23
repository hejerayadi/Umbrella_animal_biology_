"""
Test manuel interactif -- tu colles une affirmation/un paragraphe, l'agent
suggere comment formuler une citation pour l'appuyer (en s'appuyant sur les
exemples de la base ghaya_citation_examples).

Usage (depuis backend/, avec le venv active) :
    python -m agents.Literature_Agent.manual_test_interactive_citation
"""

from agents.Literature_Agent.schema import AgentRequest
from agents.Literature_Agent.subagents.writing.scientific_writing import WritingSupportAgent

print("=" * 70)
print("Colle l'affirmation ou le paragraphe pour lequel tu veux une "
      "suggestion de citation, puis appuie sur Entree PUIS Ctrl+Z (Windows) "
      "et Entree a nouveau pour valider.")
print("=" * 70)
print()

lines = []
try:
    while True:
        lines.append(input())
except EOFError:
    pass

source_text = "\n".join(lines).strip()

if not source_text:
    print("Aucun texte fourni, arret.")
else:
    request = AgentRequest(
        instruction=(
            "Rewrite this claim as it would appear in a scientific paper, "
            "phrased as if introducing a citation to support it (do not "
            "invent a real reference -- just show the citation phrasing)"
        ),
        context={"source_content": source_text},
    )

    print()
    print("Appel en cours (Azure OpenAI + Qdrant + LanguageTool)...")
    print()

    result = WritingSupportAgent().run(request)

    print("=" * 70)
    print("REPONSE DE L'AGENT")
    print("=" * 70)
    print(f"Statut : {result.status}")
    print()

    if result.output.get("draft"):
        print("--- PHRASE AVEC CITATION SUGGEREE ---")
        print(result.output["draft"])
        print()
        print(f"Outils utilises : {result.output.get('tools_used', [])}")
        print(f"Correction de style appliquee : {result.output.get('style_corrected')}")
    else:
        print("Aucun draft produit.")
        print(f"Notice : {result.output.get('notice')}")