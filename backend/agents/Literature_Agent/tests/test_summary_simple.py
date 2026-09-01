"""Simple test to see Agent 1 summaries and Agent 2 response."""
from backend.agents.Literature_Agent.subagents.discovery import KnowledgeDiscoveryOrchestrator
from backend.agents.Literature_Agent.schema import AgentRequest


def test_summary_with_context():
    """Test orchestrator showing papers with summaries and Agent 2 analysis."""
    question = "does the MSTN gene affect muscle growth in cattle"
    print(f"\n❓ QUESTION:\n   {question}\n")
    
    # Run orchestrator
    orch = KnowledgeDiscoveryOrchestrator()
    result = orch.run(AgentRequest(instruction=question, context={}))
    
    # ===== AGENT 1: Papers + Summaries =====
    print("=" * 80)
    print("AGENT 1: RETRIEVAL & KNOWLEDGE PROCESSING (Papiers + Résumés)")
    print("=" * 80)
    
    papers = result.output.get("papers", [])
    records = result.output.get("records", [])
    total = result.output.get("total_found", 0)
    
    print(f"\n📊 Total trouvé: {total} papiers\n")
    
    if papers and records:
        for i, (paper, record) in enumerate(zip(papers, records), 1):
            print(f"\n{i}. PAPIER:")
            print(f"   {paper}\n")
            print(f"   📝 RÉSUMÉ:")
            summary = record.get("short_summary", "N/A")
            if summary:
                # Afficher le résumé avec indentation
                for line in summary.split("\n"):
                    print(f"      {line}")
            else:
                print("      (Aucun résumé disponible)")
    else:
        print("   Aucun papier trouvé")
    
    # ===== AGENT 2: Scientific Analysis =====
    print("\n" + "=" * 80)
    print("AGENT 2: SCIENTIFIC ANALYSIS (avec contexte des papiers)")
    print("=" * 80)
    
    sci = result.output.get("scientific_analysis", {})
    status = sci.get("status", "unknown")
    answer = sci.get("answer", "N/A")
    tools = sci.get("tool_calls_made", [])
    error = sci.get("error")
    
    print(f"\n🔍 STATUS: {status}")
    print(f"\n📋 RÉPONSE:\n{answer}")
    
    if tools:
        print(f"\n🛠️  OUTILS APPELÉS: {', '.join(tools)}")
    
    if error:
        print(f"\n❌ ERREUR: {error}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("RÉSUMÉ")
    print("=" * 80)
    print(f"✓ Orchestrator status: {result.status}")
    print(f"✓ Agent 1 trouvé: {len(papers)} papiers avec résumés")
    print(f"✓ Agent 2 statut: {status}")
    print(f"✓ Agent 2 tools: {len(tools)} appels")
    
    # Assertions
    assert result.status is not None
    assert papers is not None
    assert sci is not None


if __name__ == "__main__":
    test_summary_with_context()
