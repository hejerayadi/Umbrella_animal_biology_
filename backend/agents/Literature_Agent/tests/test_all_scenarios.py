"""Test the Scientific Analysis agent with multiple questions.

Tests both:
1. Summary/Answer (answer_scientific_question)
2. Contradiction checking (check_claim_contradiction)
"""
from backend.agents.Literature_Agent.subagents.discovery import KnowledgeDiscoveryOrchestrator
from backend.agents.Literature_Agent.schema import AgentRequest


def test_with_question():
    """Test with a "does" question -> answer_scientific_question"""
    print("\n" + "=" * 80)
    print("TEST 1: SUMMARY/ANSWER QUESTION")
    print("=" * 80)
    
    question = "does the MSTN gene affect muscle growth in cattle"
    print(f"\n❓ QUESTION:\n   {question}")
    
    orch = KnowledgeDiscoveryOrchestrator()
    result = orch.run(AgentRequest(instruction=question, context={}))
    
    # Agent 1: Papers
    papers = result.output.get("papers", [])
    print(f"\n📚 AGENT 1 - Papers Found: {len(papers)}")
    for i, paper in enumerate(papers[:3], 1):
        print(f"   {i}. {paper[:100]}...")
    
    # Agent 2: Analysis
    sci = result.output.get("scientific_analysis", {})
    print(f"\n🔬 AGENT 2 - Scientific Analysis:")
    print(f"   Status: {sci.get('status')}")
    print(f"   Answer: {sci.get('answer')}")
    print(f"   Tools Used: {sci.get('tool_calls_made')}")


def test_with_claim():
    """Test with a claim -> check_claim_contradiction"""
    print("\n" + "=" * 80)
    print("TEST 2: CONTRADICTION CHECK (CLAIM)")
    print("=" * 80)
    
    claim = "the MSTN gene has been shown to inhibit muscle growth in farm animals"
    print(f"\n📌 CLAIM:\n   {claim}")
    
    orch = KnowledgeDiscoveryOrchestrator()
    result = orch.run(AgentRequest(instruction=claim, context={}))
    
    # Agent 1: Papers
    papers = result.output.get("papers", [])
    print(f"\n📚 AGENT 1 - Papers Found: {len(papers)}")
    for i, paper in enumerate(papers[:3], 1):
        print(f"   {i}. {paper[:100]}...")
    
    # Agent 2: Analysis
    sci = result.output.get("scientific_analysis", {})
    print(f"\n🔬 AGENT 2 - Scientific Analysis:")
    print(f"   Status: {sci.get('status')}")
    print(f"   Answer: {sci.get('answer')}")
    print(f"   Tools Used: {sci.get('tool_calls_made')}")


def test_direct_tools():
    """Test the tools directly to see their responses"""
    print("\n" + "=" * 80)
    print("TEST 3: DIRECT TOOL TESTING")
    print("=" * 80)
    
    try:
        from backend.agents.Literature_Agent.subagents.scientific_analysis.tools import (
            answer_scientific_question,
            check_claim_contradiction,
            detect_knowledge_gap
        )
        
        # Test answer_scientific_question
        print("\n🛠️ Tool 1: answer_scientific_question")
        print("   Question: What is the MSTN gene?")
        result1 = answer_scientific_question("What is the MSTN gene?")
        print(f"   Response: {result1[:200]}...")
        
        # Test check_claim_contradiction
        print("\n🛠️ Tool 2: check_claim_contradiction")
        claim = "MSTN inhibits muscle growth"
        print(f"   Claim: {claim}")
        result2 = check_claim_contradiction(claim)
        print(f"   Response: {result2[:200]}...")
        
        # Test detect_knowledge_gap
        print("\n🛠️ Tool 3: detect_knowledge_gap")
        concept = "MSTN in cattle"
        print(f"   Concept: {concept}")
        result3 = detect_knowledge_gap(concept)
        print(f"   Response: {result3[:200]}...")
        
    except Exception as e:
        print(f"❌ Error testing tools: {e}")


if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("TESTING KNOWLEDGE DISCOVERY ORCHESTRATOR WITH DIFFERENT QUESTION TYPES")
    print("█" * 80)
    
    # Test 1: Question
    try:
        test_with_question()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Claim
    try:
        test_with_claim()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
    
    # Test 3: Direct tools
    try:
        test_direct_tools()
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
    
    print("\n" + "█" * 80)
    print("TESTS COMPLETED")
    print("█" * 80)
