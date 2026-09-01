import json

# Credentials, the .env lookup and the lazily-built client all live in
# `azure_config`, which resolves them across the AZURE_LITERATURE_PUBLICATION_*
# / AZURE_LITERATURE_* / AZURE_OPENAI_* chain in the agent's single .env.
# Re-exported here because `agent.py` has always imported them from this module.
from ..azure_config import get_client, get_deployment, is_configured  # noqa: F401


def rerank_journals(topic: str, candidates: list, top_k: int = 10):
    """
    Use Azure OpenAI GPT-4.1-mini to re-rank journal candidates.
    
    Args:
        topic: Research topic/query
        candidates: List of journal dicts with scores (from ranking pipeline)
                   Each must have: name, publisher, semantic_score, topic_score,
                   subfield_score, field_score, domain_score, final_score
        top_k: Number of top results to return (default 10)
    
    Returns:
        List of re-ranked journals with LLM scores and reasoning
    """

    if not candidates:
        return []

    def access_line(candidate: dict) -> str:
        """Describe a candidate's access model in plain terms.

        Shown so the model's reasoning text can state access and cost
        accurately instead of guessing from the journal name. The ordering
        itself ignores these — apply_preferences handles that in code.
        """

        if candidate.get("is_oa"):
            access = "Open access: YES"
        else:
            access = "Open access: NO"

        if candidate.get("is_in_doaj"):
            access += " (listed in DOAJ)"

        apc = candidate.get("apc_usd")

        if apc:
            access += f" | Article processing charge: ${apc} USD"
        elif candidate.get("is_oa"):
            access += " | Article processing charge: not published"

        return access

    # Build the candidate summary for the LLM
    candidates_text = "\n".join(
        f"""Journal {i+1}: {candidate['name']}
Publisher: {candidate['publisher']}
{access_line(candidate)}
Semantic Score: {candidate['semantic_score']:.4f}
Topic Score: {candidate['topic_score']:.4f}
Subfield Score: {candidate['subfield_score']:.4f}
Field Score: {candidate['field_score']:.4f}
Domain Score: {candidate['domain_score']:.4f}
Final Score (ML-based): {candidate['final_score']:.4f}"""
        for i, candidate in enumerate(candidates)
    )

    # Construct the prompt
    prompt = f"""You are an expert research publication recommender. 
Your task is to re-rank academic journals based on their relevance to a research topic.

Research Topic: {topic}

Below are {len(candidates)} candidate journals with their ML-based ranking scores:

{candidates_text}

Please evaluate and re-rank these journals based on:
1. How well the journal's scope aligns with the research topic
2. The quality of the component scores (semantic, topic, subfield, field, domain)
3. The journal's fit for publishing research in this area

Rank ONLY on how well each journal fits the research topic. Ignore open
access status, article processing charges and publisher when deciding the
order — those are applied separately, after your ranking. They are shown
to you only so your reasoning can mention them accurately. Never assume or
invent a journal's access status or its APC.

`llm_score` must express topical fit on a 0.0-1.0 scale, where scores above
0.8 mean a strong match and scores below 0.5 mean the journal is a poor
venue for this topic. Score honestly: if a journal is a weak match, give it
a low score even if it is the best remaining option.

Return a JSON array with exactly this format, ranking the TOP {min(top_k, len(candidates))} journals:

[
  {{
    "rank": 1,
    "name": "Journal Name",
    "llm_score": 0.95,
    "original_rank": 1,
    "original_score": 0.8092,
    "reasoning": "Short explanation of why this journal is well-suited"
  }},
  ...
]

Only return the JSON array, no other text."""

    try:
        response = get_client().chat.completions.create(
            model=get_deployment(),
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        response_text = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        # Parse the JSON response
        reranked = json.loads(response_text)

        # Enrich with original candidate data.
        #
        # Matching is on a normalised name: the model echoes journal names
        # back as free text, so casing and whitespace drift. An exact-match
        # lookup that misses used to yield an entry with None for every
        # score, which crashes the first caller that formats one.
        def normalise(name) -> str:
            return " ".join(str(name).split()).casefold()

        name_to_candidate = {
            normalise(candidate['name']): candidate
            for candidate in candidates
        }

        original_rank_by_name = {
            normalise(candidate['name']): position
            for position, candidate in enumerate(candidates, start=1)
        }

        enriched_results = []

        for position, item in enumerate(reranked, start=1):

            if not isinstance(item, dict) or not item.get('name'):
                continue

            key = normalise(item['name'])
            original = name_to_candidate.get(key)

            # A name that matches no candidate means the model invented or
            # mangled it. Dropping it is the only safe option — there is no
            # journal here to report scores for.
            if original is None:
                print(
                    "[reranker] Dropping unrecognised journal: "
                    f"{item['name']!r}"
                )
                continue

            try:
                llm_score = float(item.get('llm_score'))
            except (TypeError, ValueError):
                llm_score = 0.0

            enriched_results.append({
                # Canonical name and scores come from our own candidate
                # record, never from the model's echo of them.
                "name": original['name'],
                "publisher": original.get('publisher'),
                "original_rank": original_rank_by_name.get(key),
                "original_score": original.get('final_score'),
                "llm_score": llm_score,
                "final_rank": position,
                "reasoning": item.get('reasoning') or "",
                "is_oa": original.get('is_oa', False),
                "is_in_doaj": original.get('is_in_doaj', False),
                "apc_usd": original.get('apc_usd'),
                "semantic_score": original.get('semantic_score'),
                "topic_score": original.get('topic_score'),
                "subfield_score": original.get('subfield_score'),
                "field_score": original.get('field_score'),
                "domain_score": original.get('domain_score'),
            })

        return enriched_results

    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw response: {response_text}")
        return []
    except Exception as e:
        print(f"Error calling Azure OpenAI: {e}")
        return []

# Minimum topical fit a journal needs before a stated preference is allowed
# to move it up. Below this the journal is a poor venue regardless of how
# well it matches the preference.
RELEVANCE_FLOOR = 0.55


def apply_preferences(
    ranked: list,
    constraints: dict,
    top_k: int = 5,
    relevance_floor: float = RELEVANCE_FLOOR,
):
    """
    Re-order a relevance ranking to honour stated preferences.

    The LLM ranks purely on topical fit, then this applies the constraints
    deterministically. Doing it here rather than in the prompt is what
    stops a preference from dragging an unrelated journal into the results:
    only journals the model already scored as relevant can be promoted, and
    a preference can never introduce one that was not already there.

    Args:
        ranked: Output of rerank_journals, ordered by topical fit
        constraints: The `constraints` dict from ranking.query_interpreter
        top_k: How many journals to return
        relevance_floor: Minimum llm_score to be eligible for promotion

    Returns:
        List of at most `top_k` journals, each with a `preference_note`
        explaining any mismatch with what the researcher asked for.
    """

    if not ranked:
        return []

    wants_oa = constraints.get("open_access") is True

    excluded = {
        publisher.lower()
        for publisher in constraints.get("publisher_exclude") or []
    }

    if not wants_oa and not excluded:
        return ranked[:top_k]

    def is_excluded_publisher(journal: dict) -> bool:

        publisher = (journal.get("publisher") or "").lower()

        return any(name in publisher for name in excluded)

    def sort_key(item):

        index, journal = item

        score = journal.get("llm_score") or 0.0

        # Journals that are too weak a topical match keep their original
        # relevance order and stay below everything else. A preference is
        # a tiebreaker among genuinely relevant journals, never a promotion
        # out of irrelevance.
        if score < relevance_floor:
            return (2, index)

        satisfies = (
            (not wants_oa or journal.get("is_oa"))
            and not is_excluded_publisher(journal)
        )

        # Python's sort is stable, so relevance order is preserved inside
        # each of these groups.
        return (0 if satisfies else 1, index)

    reordered = [
        journal
        for _, journal in sorted(enumerate(ranked), key=sort_key)
    ]

    for journal in reordered[:top_k]:

        notes = []

        if wants_oa and not journal.get("is_oa"):
            notes.append("not open access")

        if is_excluded_publisher(journal):
            notes.append(f"published by {journal.get('publisher')}")

        if (journal.get("llm_score") or 0.0) < relevance_floor:
            notes.append("weak topical match")

        journal["preference_note"] = "; ".join(notes)

    return reordered[:top_k]
