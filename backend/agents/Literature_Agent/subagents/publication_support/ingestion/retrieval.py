from qdrant_client.models import Filter, FieldCondition, MatchValue

from embeddings.embedder import embed_query
from qdrant.qdrant_search import search_qdrant
from ranking.ranker import (
    hierarchy_relevance,
    calculate_final_score,
    warm_cache,
    HIERARCHY_LEVELS,
)


def retrieve_journals(
    query: str,
    candidate_limit: int = 50,
    final_limit: int = 5,
    open_access_only: bool = False,
    oa_supplement: int = 25,
):
    """
    Retrieve and rank journals for a research query.

    Pipeline:

        Query
          ↓
        BGE embedding
          ↓
        Qdrant candidate retrieval
          ↓
        Hybrid ranking
          ↓
        Ranked journals (all candidates, caller slices)
    
    Args:
        query: Research topic
        candidate_limit: How many to pull from Qdrant (default 50)
        final_limit: DEPRECATED - now ignored. Caller should slice results.
    
    Returns:
        List of all ranked journals (full ranked list, not pre-sliced)
    """

    # --------------------------------------------------
    # 1. Convert the research query into a vector
    # --------------------------------------------------

    query_vector = embed_query(query)

    # --------------------------------------------------
    # 2. Retrieve semantic candidates from Qdrant
    # --------------------------------------------------

    candidates = search_qdrant(
        query_vector=query_vector,
        limit=candidate_limit,
    )

    # An unfiltered pull is dominated by the big subscription journals, so a
    # researcher who needs open access often finds only a couple of usable
    # options in it. When they asked for open access, top the pool up with
    # the nearest open-access journals so the re-ranker has real choices.
    #
    # This supplements rather than replaces: the strongest matches overall
    # stay in the pool, and the re-ranker decides what the preference is
    # worth. Journals already retrieved are not added twice.
    if open_access_only and oa_supplement > 0:

        seen_ids = {
            candidate.payload.get("id")
            for candidate in candidates
        }

        oa_candidates = search_qdrant(
            query_vector=query_vector,
            limit=oa_supplement,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="is_oa",
                        match=MatchValue(value=True),
                    )
                ]
            ),
        )

        # A filtered vector search always returns `limit` points however far
        # away they are, so this pool can contain open access journals with
        # no connection to the topic. That is tolerated on purpose: they are
        # only candidates, and ranking.llm_reranker.apply_preferences will
        # not promote anything the re-ranker scored as a weak topical match.
        #
        # Screening them out by distance here does not work: supplements are
        # by definition outside the unfiltered top-`candidate_limit`, so any
        # threshold drawn from that pool rejects all of them and turns the
        # supplement into dead code.
        candidates = list(candidates) + [
            candidate
            for candidate in oa_candidates
            if candidate.payload.get("id") not in seen_ids
        ]

    # --------------------------------------------------
    # 3. Apply hierarchical relevance scoring
    # --------------------------------------------------

    # Embed every label across all candidates in a single batch first;
    # the per-candidate scoring below then reads from the cache.
    warm_cache(
        [query] + [
            label
            for candidate in candidates
            for topic in candidate.payload.get("topics", [])
            for topic_key, _ in HIERARCHY_LEVELS
            for label in [topic.get(topic_key)]
            if label
        ]
    )

    ranked_journals = []

    for candidate in candidates:

        payload = candidate.payload

        topics = payload.get("topics", [])

        hierarchy_scores = hierarchy_relevance(
            query,
            topics,
        )

        # Qdrant's semantic similarity
        semantic_score = candidate.score

        # Hybrid score
        final_score = calculate_final_score(
            semantic_score=semantic_score,
            hierarchy_scores=hierarchy_scores,
        )

        ranked_journals.append(
            {
                "id": payload.get("id"),
                "name": payload.get("name"),
                "publisher": payload.get("publisher"),
                "issn": payload.get("issn"),

                # Defaults keep this working against points ingested before
                # these fields existed, rather than raising mid-search.
                "is_oa": payload.get("is_oa", False),
                "is_in_doaj": payload.get("is_in_doaj", False),
                "apc_usd": payload.get("apc_usd"),

                "semantic_score": semantic_score,

                "topic_score": hierarchy_scores["topic"],
                "subfield_score": hierarchy_scores["subfield"],
                "field_score": hierarchy_scores["field"],
                "domain_score": hierarchy_scores["domain"],

                "final_score": final_score,

                "topics": topics,
            }
        )

    # --------------------------------------------------
    # 4. Sort by final hybrid score
    # --------------------------------------------------

    ranked_journals.sort(
        key=lambda journal: journal["final_score"],
        reverse=True,
    )

    # --------------------------------------------------
    # 5. Return ALL ranked journals (no slicing here)
    # --------------------------------------------------
    # Caller now decides: slice to Top 20-30 for LLM, or Top 5 for final output

    return ranked_journals


if __name__ == "__main__":

    query = "machine learning and artificial intelligence"

    print(f"\nQuery: {query}\n")

    results = retrieve_journals(
        query=query,
        candidate_limit=50,
    )

    print("Top 10 recommendations:\n")

    for i, journal in enumerate(results[:10], start=1):

        print(
            f"{i}. {journal['name']}"
        )

        print(
            f"   Final Score: "
            f"{journal['final_score']:.4f}"
        )

        print(
            f"   Semantic: "
            f"{journal['semantic_score']:.4f}"
        )

        print(
            f"   Topic: "
            f"{journal['topic_score']:.4f}"
        )

        print(
            f"   Subfield: "
            f"{journal['subfield_score']:.4f}"
        )

        print(
            f"   Field: "
            f"{journal['field_score']:.4f}"
        )

        print(
            f"   Domain: "
            f"{journal['domain_score']:.4f}"
        )

        print()