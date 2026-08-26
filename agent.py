"""
Azure AI Foundry Agent for Journal Recommendations
Stateful agent with memory, multi-tool support, and conversational interface
"""

import os
import json
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI

# Import your existing pipeline
from ingestion.retrieval import retrieve_journals
from ranking.llm_reranker import rerank_journals, apply_preferences
from ranking.query_interpreter import interpret_query, build_llm_topic


load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not API_KEY or not ENDPOINT or not DEPLOYMENT:
    raise RuntimeError("Missing Azure OpenAI credentials")

client = OpenAI(
    api_key=API_KEY,
    base_url=ENDPOINT,
)

# ============================================================================
# AGENT MEMORY & STATE
# ============================================================================

class AgentMemory:
    """Maintains conversation history and search results for context."""
    
    def __init__(self):
        self.conversation_history = []
        self.search_results = {}  # Cache: topic -> results
        self.current_topic = None
        # The turn's raw text. Tool slot-filling strips stated preferences
        # ("open access", "not Elsevier") out of `topic`, so the constraints
        # have to be read from what the user actually wrote.
        self.last_user_input = None
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def cache_search(self, topic: str, results: list):
        """Cache search results by topic."""
        self.search_results[topic] = results
        self.current_topic = topic
    
    def get_cached_search(self, topic: str):
        """Retrieve cached search results."""
        return self.search_results.get(topic)
    
    def get_context(self) -> str:
        """Return conversation context for agent."""
        if self.current_topic:
            return f"Current research topic: {self.current_topic}"
        return "No recent search performed"


memory = AgentMemory()


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def retrieve_and_rank_journals(topic: str, top_k: int = 5) -> dict:
    """
    Retrieve journals from Qdrant and rank them with LLM re-ranking.
    
    Args:
        topic: Research topic/query
        top_k: Number of top results to return
    
    Returns:
        Dict with rankings and comparison data
    """
    
    # `topic` is what the agent chose to search for: it has the conversation
    # context, so it resolves follow-ups like "now find open access ones" into
    # a real subject. What it does NOT carry is the user's stated preferences,
    # which slot-filling drops on the way in. Both are passed to the
    # interpreter so the subject comes from the agent and the constraints come
    # from the user's own words.
    raw_input = memory.last_user_input

    if raw_input and raw_input.strip().lower() != topic.strip().lower():
        interpreter_input = (
            f"{topic}\n\n(The researcher's exact words: {raw_input})"
        )
    else:
        interpreter_input = topic

    interpretation = interpret_query(interpreter_input)

    # Get all ranked candidates from ML system
    all_ranked = retrieve_journals(
        query=interpretation["search_query"],
        candidate_limit=50,
        open_access_only=(
            interpretation["constraints"]["open_access"] is True
        ),
    )

    # Pass Top 30 to LLM re-ranker
    candidates_for_llm = all_ranked[:30]

    # Rank wider than we return, so apply_preferences has journals in
    # reserve to promote from.
    llm_results = rerank_journals(
        topic=build_llm_topic(interpretation),
        candidates=candidates_for_llm,
        top_k=max(top_k * 3, 15),
    )

    llm_results = apply_preferences(
        llm_results,
        interpretation["constraints"],
        top_k=top_k,
    )
    
    # Fallback if LLM fails. Built wider than we return and passed through
    # apply_preferences, so a stated constraint survives re-ranker failure.
    if not llm_results:
        fallback = [
            {
                "name": journal["name"],
                "publisher": journal["publisher"],
                "original_score": journal["final_score"],
                "llm_score": journal["final_score"],
                "reasoning": "Fallback to ML-based ranking",
                "is_oa": journal["is_oa"],
                "is_in_doaj": journal["is_in_doaj"],
                "apc_usd": journal["apc_usd"],
                "semantic_score": journal["semantic_score"],
                "topic_score": journal["topic_score"],
                "subfield_score": journal["subfield_score"],
                "field_score": journal["field_score"],
                "domain_score": journal["domain_score"],
            }
            for journal in all_ranked[:max(top_k * 3, 15)]
        ]

        llm_results = apply_preferences(
            fallback,
            interpretation["constraints"],
            top_k=top_k,
        )
    
    # Cache results for follow-up questions
    memory.cache_search(topic, llm_results)
    
    return {
        "topic": topic,
        "interpreted_as": interpretation["search_query"],
        "constraints": interpretation["constraints"],
        "total_candidates": len(all_ranked),
        "top_results": llm_results,
        "success": True
    }


def explain_journal_score(journal_name: str) -> dict:
    """
    Explain why a journal received its score.
    
    Args:
        journal_name: Name of the journal to explain
    
    Returns:
        Dict with detailed score breakdown and reasoning
    """
    
    if not memory.current_topic:
        return {
            "success": False,
            "error": "No recent search. Please search for a topic first."
        }
    
    results = memory.get_cached_search(memory.current_topic)
    if not results:
        return {
            "success": False,
            "error": "No cached results found"
        }
    
    journal = next(
        (j for j in results if j["name"].lower() == journal_name.lower()),
        None
    )
    
    if not journal:
        available = [j["name"] for j in results]
        return {
            "success": False,
            "error": f"Journal not found. Available journals: {', '.join(available)}"
        }
    
    return {
        "success": True,
        "journal": journal["name"],
        "publisher": journal["publisher"],
        "explanation": {
            "ml_analysis": {
                "semantic_score": journal["semantic_score"],
                "topic_score": journal["topic_score"],
                "subfield_score": journal["subfield_score"],
                "field_score": journal["field_score"],
                "domain_score": journal["domain_score"],
                "final_ml_score": journal["original_score"],
            },
            "llm_analysis": {
                "llm_score": journal["llm_score"],
                "reasoning": journal["reasoning"],
            }
        }
    }


def compare_journals(journal1: str, journal2: str) -> dict:
    """
    Compare two journals side-by-side.
    
    Args:
        journal1: First journal name
        journal2: Second journal name
    
    Returns:
        Dict with comparison data
    """
    
    if not memory.current_topic:
        return {
            "success": False,
            "error": "No recent search. Please search for a topic first."
        }
    
    results = memory.get_cached_search(memory.current_topic)
    if not results:
        return {
            "success": False,
            "error": "No cached results found"
        }
    
    j1 = next(
        (j for j in results if j["name"].lower() == journal1.lower()),
        None
    )
    j2 = next(
        (j for j in results if j["name"].lower() == journal2.lower()),
        None
    )
    
    if not j1 or not j2:
        return {
            "success": False,
            "error": f"One or both journals not found in results"
        }
    
    return {
        "success": True,
        "comparison": {
            "journal1": {
                "name": j1["name"],
                "ml_score": j1["original_score"],
                "llm_score": j1["llm_score"],
                "semantic": j1["semantic_score"],
                "topic": j1["topic_score"],
            },
            "journal2": {
                "name": j2["name"],
                "ml_score": j2["original_score"],
                "llm_score": j2["llm_score"],
                "semantic": j2["semantic_score"],
                "topic": j2["topic_score"],
            },
            "difference": {
                "ml_score_delta": j1["original_score"] - j2["original_score"],
                "llm_score_delta": j1["llm_score"] - j2["llm_score"],
                "winner_ml": j1["name"] if j1["original_score"] > j2["original_score"] else j2["name"],
                "winner_llm": j1["name"] if j1["llm_score"] > j2["llm_score"] else j2["name"],
            }
        }
    }


# ============================================================================
# TOOL REGISTRY & DISPATCHER
# ============================================================================

TOOLS = {
    "retrieve_and_rank_journals": {
        "function": retrieve_and_rank_journals,
        "description": "Search for and rank journals for a research topic",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The research topic to search for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return (default 5)",
                    "default": 5
                }
            },
            "required": ["topic"]
        }
    },
    "explain_journal_score": {
        "function": explain_journal_score,
        "description": "Explain why a journal received its score",
        "parameters": {
            "type": "object",
            "properties": {
                "journal_name": {
                    "type": "string",
                    "description": "The name of the journal to explain"
                }
            },
            "required": ["journal_name"]
        }
    },
    "compare_journals": {
        "function": compare_journals,
        "description": "Compare two journals side-by-side",
        "parameters": {
            "type": "object",
            "properties": {
                "journal1": {
                    "type": "string",
                    "description": "First journal name"
                },
                "journal2": {
                    "type": "string",
                    "description": "Second journal name"
                }
            },
            "required": ["journal1", "journal2"]
        }
    }
}


def execute_tool(tool_name: str, tool_input: dict) -> Any:
    """Execute a tool and return its result."""
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}
    
    tool_func = TOOLS[tool_name]["function"]
    return tool_func(**tool_input)


# ============================================================================
# AGENT LOOP
# ============================================================================

def run_agent():
    """Main agent loop with memory and tool use."""
    
    def build_system_prompt() -> str:
        """Rebuilt each turn so the context line reflects the current search.

        Interpolating memory.get_context() once, before the loop, froze it
        at "No recent search performed" for the whole session.
        """

        return f"""You are an expert research publication recommender agent.
You help researchers find the best academic journals for their work.

You have access to three tools:
1. retrieve_and_rank_journals(topic, top_k=5) - Search for and rank journals
2. explain_journal_score(journal_name) - Explain a journal's score
3. compare_journals(journal1, journal2) - Compare two journals

IMPORTANT OUTPUT FORMAT:
- When presenting results, use BULLET POINTS
- Keep explanations concise and clear
- Always reference specific scores when explaining rankings
- Use the format: "• [Journal Name] (Score: X.XX)"
- State whether a journal is open access, and its APC when one is listed
- If a result carries a preference_note, say so plainly rather than
  presenting the journal as if it matched what the user asked for

You remember previous searches in this conversation, so users can ask follow-up questions
about results without re-searching.

Current context: {memory.get_context()}
"""
    
    print("🔬 Journal Recommendation Agent")
    print("="*60)
    print("I can help you find the best journals for your research.")
    print("Commands: search [topic], explain [journal], compare [j1] vs [j2]")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 👋")
            break
        
        # Add user message to memory
        memory.add_message("user", user_input)

        # Recorded before the tool call so retrieve_and_rank_journals can read
        # the preferences that slot-filling strips out of its arguments.
        memory.last_user_input = user_input

        # Rebuilt now so the context line reflects searches made this session.
        system_prompt = build_system_prompt()
        
        # Get messages for API (limit to last 10 for context window)
        messages = memory.conversation_history[-10:]
        
        try:
            # Call Azure OpenAI with tools
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool["description"],
                            "parameters": tool["parameters"]
                        }
                    }
                    for name, tool in TOOLS.items()
                ],
                tool_choice="auto",
                temperature=0.3,
                max_tokens=2000,
            )
            
            # Process response
            assistant_message = response.choices[0].message
            
            # Check for tool calls
            if assistant_message.tool_calls:
                # Execute tools and collect results
                tool_results_text = ""
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    
                    print(f"\n🔧 Executing: {tool_name}")
                    result = execute_tool(tool_name, tool_input)
                    
                    # Format result as text for the model
                    tool_results_text += f"\nTool: {tool_name}\nResult: {json.dumps(result, indent=2)}\n"
                
                # Add assistant message to history
                memory.add_message("assistant", assistant_message.content or "")
                
                # Add tool results as a user message (simpler approach)
                messages.append({"role": "assistant", "content": assistant_message.content or ""})
                messages.append({
                    "role": "user",
                    "content": f"Tool results:\n{tool_results_text}\n\nPlease format the results in bullet points as requested."
                })
                
                # Get final response from agent
                final_response = client.chat.completions.create(
                    model=DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *messages
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                )
                
                final_text = final_response.choices[0].message.content
                memory.add_message("assistant", final_text)
                print(f"\n🤖 Agent: {final_text}\n")
            
            else:
                # No tool call, just respond
                response_text = assistant_message.content or "I couldn't process that request."
                memory.add_message("assistant", response_text)
                print(f"\n🤖 Agent: {response_text}\n")
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    run_agent()