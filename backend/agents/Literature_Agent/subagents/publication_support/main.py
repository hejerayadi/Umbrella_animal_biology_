from ingestion.retrieval import retrieve_journals
from ranking.llm_reranker import rerank_journals, apply_preferences
from ranking.query_interpreter import (
    interpret_query,
    describe_interpretation,
    build_llm_topic,
)


def main():

    query = input(
        "Enter your research topic: "
    ).strip()

    if not query:
        print("Please enter a research topic.")
        return

    print("\n[Step 1] Understanding your query...\n")

    interpretation = interpret_query(query)

    # Clarification is checked first: a vague topic like "AI" is often
    # reported as both invalid and clarifiable, and asking beats refusing.
    if interpretation["needs_clarification"]:

        print(interpretation["clarification_question"])

        follow_up = input("> ").strip()

        if not follow_up:
            print("No topic given.")
            return

        interpretation = interpret_query(f"{query}. {follow_up}")

    if not interpretation["is_valid"]:
        print(
            "That doesn't look like a research topic. "
            "Try describing your research area."
        )
        return

    print(describe_interpretation(interpretation))

    search_query = interpretation["search_query"]

    print("\n[Step 2] Retrieving candidates from Qdrant and ranking...\n")

    # Get all ranked candidates (Top 50)
    all_ranked = retrieve_journals(
        query=search_query,
        candidate_limit=50,
        open_access_only=(
            interpretation["constraints"]["open_access"] is True
        ),
    )

    # Take Top 20-30 for LLM re-ranking
    candidates_for_llm = all_ranked[:30]

    print(f"Retrieved and ranked {len(all_ranked)} journals.")
    print(f"Passing Top {len(candidates_for_llm)} to LLM re-ranker...\n")

    print("[Step 3] LLM re-ranking Top 30...\n")

    # Pass to LLM for re-ranking
    # Rank wider than we display: apply_preferences reorders within this
    # list, so it needs journals in reserve to promote from.
    llm_results = rerank_journals(
        topic=build_llm_topic(interpretation),
        candidates=candidates_for_llm,
        top_k=15,
    )

    llm_results = apply_preferences(
        llm_results,
        interpretation["constraints"],
        top_k=5,
    )

    if not llm_results:
        print("Error: LLM re-ranking failed. Falling back to ML-based ranking.\n")
        # Convert ML ranking to match LLM result format. Built wider than we
        # display and passed through apply_preferences, so a stated
        # constraint is still honoured when the re-ranker is unavailable.
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
            for journal in all_ranked[:15]
        ]

        llm_results = apply_preferences(
            fallback,
            interpretation["constraints"],
            top_k=5,
        )

    print("\n" + "="*60)
    print("FINAL RESULTS (LLM Re-Ranked)")
    print("="*60 + "\n")

    for i, journal in enumerate(llm_results, start=1):

        print(f"{i}. {journal['name']}")
        print(f"   Publisher: {journal['publisher']}")

        if journal.get("is_oa"):
            access = "Open access"

            if journal.get("apc_usd"):
                access += f" (APC ${journal['apc_usd']} USD)"
        else:
            access = "Not open access"

        print(f"   Access:    {access}")

        # Set when the journal does not match something the researcher
        # asked for; shown so a constraint miss is never silent.
        if journal.get("preference_note"):
            print(f"   Note:      {journal['preference_note']}")

        print()

        print(f"   Scores (ML-based):")
        print(f"      Semantic:  {journal['semantic_score']:.4f}")
        print(f"      Topic:     {journal['topic_score']:.4f}")
        print(f"      Subfield:  {journal['subfield_score']:.4f}")
        print(f"      Field:     {journal['field_score']:.4f}")
        print(f"      Domain:    {journal['domain_score']:.4f}")
        print(f"      Final:     {journal['original_score']:.4f}")
        print()

        print(f"   LLM Evaluation:")
        print(f"      LLM Score: {journal['llm_score']:.4f}")
        print(f"      Reasoning: {journal['reasoning']}")
        print()


if __name__ == "__main__":
    main()
