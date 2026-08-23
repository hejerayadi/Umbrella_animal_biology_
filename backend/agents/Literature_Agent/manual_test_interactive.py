"""
Test manuel interactif -- tu colles ton propre paragraphe dans le terminal,
l'agent ecrit un abstract a partir de ce contenu.

Usage (depuis backend/, avec le venv active) :
    python -m agents.Literature_Agent.manual_test_interactive
"""

from agents.Literature_Agent.schema import AgentRequest
from agents.Literature_Agent.subagents.writing.scientific_writing import WritingSupportAgent

print("=" * 70)
print("Colle ton paragraphe ci-dessous, puis appuie sur Entree PUIS Ctrl+Z "
      "(Windows) et Entree a nouveau pour valider.")
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
        instruction="Write an abstract from this content",
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
        print("--- ABSTRACT GENERE ---")
        print(result.output["draft"])
        print()
        print(f"Outils utilises : {result.output.get('tools_used', [])}")
        print(f"Correction de style appliquee : {result.output.get('style_corrected')}")
    else:
        print("Aucun draft produit.")
        print(f"Notice : {result.output.get('notice')}")