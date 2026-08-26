"""
Offline regression tests for the recommendation pipeline.

Run with:

    python3 -m tests.test_pipeline

No network and no API calls: the Azure client is replaced with a stub in
the tests that need one, and Qdrant retrieval is stubbed where the code
under test only orchestrates it. That keeps this runnable on every change,
which is the point — the bugs this suite covers all looked correct on
inspection and only appeared when something was actually executed.
"""

import sys
import traceback


# ----------------------------------------------------------------------
# Tiny test runner (no pytest dependency)
# ----------------------------------------------------------------------

def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run_all():

    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]

    passed, failed = 0, []

    for name, function in tests:

        try:
            function()
            passed += 1
            print(f"  PASS  {name}")

        except Exception as error:
            failed.append((name, error))
            print(f"  FAIL  {name}: {error}")

    print(f"\n{passed} passed, {len(failed)} failed, {len(tests)} total")

    if failed:
        print("\nFailure detail:")
        for name, error in failed:
            print(f"\n--- {name} ---")
            traceback.print_exception(type(error), error, error.__traceback__)

    return 1 if failed else 0


# ----------------------------------------------------------------------
# Stub Azure client
# ----------------------------------------------------------------------

def stub_client(content: str):
    """Build an object shaped like the OpenAI client, returning `content`."""

    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    response = type("Response", (), {"choices": [choice]})()

    completions = type(
        "Completions", (),
        {"create": staticmethod(lambda **kwargs: response)},
    )()

    chat = type("Chat", (), {"completions": completions})()

    return type("Client", (), {"chat": chat})()


def journal(name, **overrides):
    """A candidate dict shaped like ingestion.retrieval output."""

    base = {
        "id": f"id-{name}",
        "name": name,
        "publisher": "Test Publisher",
        "issn": "0000-0000",
        "is_oa": False,
        "is_in_doaj": False,
        "apc_usd": None,
        "semantic_score": 0.70,
        "topic_score": 0.80,
        "subfield_score": 0.60,
        "field_score": 0.50,
        "domain_score": 0.40,
        "final_score": 0.75,
        "topics": [],
    }

    base.update(overrides)

    return base


# ======================================================================
# ranking.llm_reranker - enrichment robustness
# ======================================================================

def test_reranker_matches_names_despite_drift():
    """Casing and whitespace drift in the model's echo must still match."""

    import ranking.llm_reranker as reranker

    candidates = [journal("Nature Communications", final_score=0.81)]

    original_client = reranker.client
    reranker.client = stub_client(
        '[{"rank": 1, "name": "  nature   COMMUNICATIONS ", '
        '"llm_score": 0.9, "reasoning": "r"}]'
    )

    try:
        results = reranker.rerank_journals("topic", candidates, top_k=5)
    finally:
        reranker.client = original_client

    check(len(results) == 1, f"expected 1 result, got {len(results)}")
    check(
        results[0]["name"] == "Nature Communications",
        f"expected canonical name, got {results[0]['name']!r}",
    )


def test_reranker_drops_hallucinated_journal():
    """A name matching no candidate must be dropped, not emitted as None."""

    import ranking.llm_reranker as reranker

    candidates = [journal("Real Journal")]

    original_client = reranker.client
    reranker.client = stub_client(
        '[{"rank": 1, "name": "Invented Journal", "llm_score": 0.9, '
        '"reasoning": "r"}]'
    )

    try:
        results = reranker.rerank_journals("topic", candidates, top_k=5)
    finally:
        reranker.client = original_client

    check(results == [], f"expected no results, got {results}")


def test_reranker_scores_are_never_none():
    """Every score must be formattable — this is the crash that shipped."""

    import ranking.llm_reranker as reranker

    candidates = [journal("Real Journal")]

    original_client = reranker.client
    reranker.client = stub_client(
        '[{"rank": 1, "name": "Real Journal", "llm_score": 0.9, '
        '"reasoning": "r"}]'
    )

    try:
        results = reranker.rerank_journals("topic", candidates, top_k=5)
    finally:
        reranker.client = original_client

    for key in (
        "semantic_score", "topic_score", "subfield_score",
        "field_score", "domain_score", "original_score", "llm_score",
    ):
        value = results[0][key]
        check(value is not None, f"{key} is None")
        format(value, ".4f")


def test_reranker_ignores_model_echoed_scores():
    """original_score must come from our data, not the model's echo."""

    import ranking.llm_reranker as reranker

    candidates = [journal("Real Journal", final_score=0.81)]

    original_client = reranker.client
    reranker.client = stub_client(
        '[{"rank": 1, "name": "Real Journal", "llm_score": 0.9, '
        '"original_score": 0.11, "reasoning": "r"}]'
    )

    try:
        results = reranker.rerank_journals("topic", candidates, top_k=5)
    finally:
        reranker.client = original_client

    check(
        results[0]["original_score"] == 0.81,
        f"expected 0.81 from our data, got {results[0]['original_score']}",
    )


def test_reranker_coerces_string_score():

    import ranking.llm_reranker as reranker

    candidates = [journal("Real Journal")]

    original_client = reranker.client
    reranker.client = stub_client(
        '[{"rank": 1, "name": "Real Journal", "llm_score": "0.93", '
        '"reasoning": "r"}]'
    )

    try:
        results = reranker.rerank_journals("topic", candidates, top_k=5)
    finally:
        reranker.client = original_client

    check(
        isinstance(results[0]["llm_score"], float),
        "llm_score should be coerced to float",
    )


def test_reranker_survives_unparseable_response():
    """Bad JSON must return [] so the caller can fall back, not raise."""

    import ranking.llm_reranker as reranker

    original_client = reranker.client
    reranker.client = stub_client("not json at all")

    try:
        results = reranker.rerank_journals("topic", [journal("A")], top_k=5)
    finally:
        reranker.client = original_client

    check(results == [], f"expected [], got {results}")


def test_reranker_handles_no_candidates():

    from ranking.llm_reranker import rerank_journals

    check(rerank_journals("topic", [], top_k=5) == [], "expected []")


# ======================================================================
# ranking.llm_reranker.apply_preferences - deterministic constraints
# ======================================================================

def _preference_fixture():
    return [
        {"name": "ClosedBest", "llm_score": 0.95, "is_oa": False,
         "publisher": "Elsevier BV"},
        {"name": "OpenGood", "llm_score": 0.80, "is_oa": True,
         "publisher": "MDPI"},
        {"name": "OpenIrrelevant", "llm_score": 0.20, "is_oa": True,
         "publisher": "MDPI"},
    ]


def test_preferences_unconstrained_preserves_relevance_order():

    from ranking.llm_reranker import apply_preferences

    names = [j["name"] for j in apply_preferences(_preference_fixture(), {}, 3)]

    check(names[0] == "ClosedBest", f"relevance order broken: {names}")


def test_preferences_promote_relevant_open_access():

    from ranking.llm_reranker import apply_preferences

    names = [
        j["name"]
        for j in apply_preferences(
            _preference_fixture(), {"open_access": True}, 3
        )
    ]

    check(names[0] == "OpenGood", f"OA journal not promoted: {names}")


def test_preferences_never_promote_irrelevant_journal():
    """The bug that shipped twice: a preference must not beat relevance."""

    from ranking.llm_reranker import apply_preferences

    results = apply_preferences(
        _preference_fixture(), {"open_access": True}, 2
    )

    names = [j["name"] for j in results]

    check(
        "OpenIrrelevant" not in names,
        f"irrelevant OA journal promoted: {names}",
    )


def test_preferences_exclude_publisher():

    from ranking.llm_reranker import apply_preferences

    results = apply_preferences(
        _preference_fixture(), {"publisher_exclude": ["Elsevier"]}, 3
    )

    check(
        results[0]["name"] != "ClosedBest",
        "excluded publisher still ranked first",
    )


def test_preferences_annotate_mismatches():
    """A constraint miss must be stated, never silent."""

    from ranking.llm_reranker import apply_preferences

    results = apply_preferences(
        [{"name": "OnlyClosed", "llm_score": 0.9, "is_oa": False,
          "publisher": "X"}],
        {"open_access": True},
        1,
    )

    check(
        "not open access" in results[0]["preference_note"],
        f"missing note, got {results[0].get('preference_note')!r}",
    )


def test_preferences_handle_empty_input():

    from ranking.llm_reranker import apply_preferences

    check(apply_preferences([], {"open_access": True}, 5) == [], "expected []")


# ======================================================================
# ranking.query_interpreter
# ======================================================================

def test_interpreter_rejects_empty_input():

    from ranking.query_interpreter import interpret_query

    result = interpret_query("   ")

    check(result["is_valid"] is False, "blank input should be invalid")
    check(result["interpreted"] is False, "blank input should not be interpreted")


def test_interpreter_falls_back_to_raw_on_bad_json():
    """Interpretation is an enhancement layer; it must never break search."""

    import ranking.query_interpreter as interpreter

    original_client = interpreter.client
    interpreter.client = stub_client("this is not json")

    try:
        result = interpreter.interpret_query("plasma physics", use_cache=False)
    finally:
        interpreter.client = original_client

    check(
        result["search_query"] == "plasma physics",
        f"expected raw fallback, got {result['search_query']!r}",
    )
    check(result["interpreted"] is False, "should be marked uninterpreted")


def test_interpreter_search_query_is_never_blank():
    """A blank search_query would poison the embedding."""

    import ranking.query_interpreter as interpreter

    original_client = interpreter.client
    interpreter.client = stub_client('{"search_query": "   ", "is_valid": true}')

    try:
        result = interpreter.interpret_query("cell biology", use_cache=False)
    finally:
        interpreter.client = original_client

    check(
        result["search_query"].strip() != "",
        "search_query must never be blank",
    )


def test_interpreter_strips_code_fences():

    import ranking.query_interpreter as interpreter

    original_client = interpreter.client
    interpreter.client = stub_client(
        '```json\n{"search_query": "Marine biology", "is_valid": true}\n```'
    )

    try:
        result = interpreter.interpret_query("marine bio", use_cache=False)
    finally:
        interpreter.client = original_client

    check(
        result["search_query"] == "Marine biology",
        f"fence not stripped: {result['search_query']!r}",
    )


def test_interpreter_coerces_wrong_types():
    """Malformed field types must fall back, not propagate or raise."""

    import ranking.query_interpreter as interpreter

    original_client = interpreter.client
    interpreter.client = stub_client(
        '{"search_query": "Botany", "is_valid": "yes", "intent": "nonsense", '
        '"expanded_queries": "not a list", "constraints": "not a dict", '
        '"needs_clarification": true, "clarification_question": ""}'
    )

    try:
        result = interpreter.interpret_query("botany", use_cache=False)
    finally:
        interpreter.client = original_client

    check(result["is_valid"] is True, "non-bool is_valid should default True")
    check(result["intent"] == "search", "invalid intent should default")
    check(result["expanded_queries"] == [], "non-list should default to []")
    check(isinstance(result["constraints"], dict), "constraints must be a dict")
    check(
        result["needs_clarification"] is False,
        "clarification without a question must be suppressed",
    )


def test_build_llm_topic_excludes_preferences():
    """Preferences are applied in code; restating them fights the prompt."""

    from ranking.query_interpreter import build_llm_topic

    topic = build_llm_topic({
        "raw_input": "open access journal for CRISPR work",
        "search_query": "CRISPR gene editing",
        "interpreted": True,
        "constraints": {
            "open_access": True,
            "publisher_exclude": ["Elsevier"],
            "other": [],
        },
    })

    check("CRISPR gene editing" in topic, "cleaned query missing")
    check("Stated preferences" not in topic, "preferences leaked into topic")


# ======================================================================
# agent.py tools - previously never executed
# ======================================================================

def _seed_agent_memory():

    import agent

    results = [
        {"name": "Alpha Journal", "publisher": "A Press", "original_score": 0.80,
         "llm_score": 0.90, "reasoning": "a", "is_oa": True, "apc_usd": 1000,
         "is_in_doaj": True, "semantic_score": 0.70, "topic_score": 0.80,
         "subfield_score": 0.60, "field_score": 0.50, "domain_score": 0.40},
        {"name": "Beta Journal", "publisher": "B Press", "original_score": 0.70,
         "llm_score": 0.60, "reasoning": "b", "is_oa": False, "apc_usd": None,
         "is_in_doaj": False, "semantic_score": 0.60, "topic_score": 0.70,
         "subfield_score": 0.50, "field_score": 0.40, "domain_score": 0.30},
    ]

    agent.memory.cache_search("test topic", results)

    return agent


def test_compare_journals_returns_deltas():
    """Never executed before: it subtracts scores that used to be None."""

    agent = _seed_agent_memory()

    result = agent.compare_journals("Alpha Journal", "Beta Journal")

    check(result["success"] is True, f"comparison failed: {result}")

    difference = result["comparison"]["difference"]

    check(
        abs(difference["ml_score_delta"] - 0.10) < 1e-9,
        f"unexpected ml delta: {difference['ml_score_delta']}",
    )
    check(
        difference["winner_llm"] == "Alpha Journal",
        f"unexpected llm winner: {difference['winner_llm']}",
    )


def test_compare_journals_is_case_insensitive():

    agent = _seed_agent_memory()

    result = agent.compare_journals("alpha journal", "BETA JOURNAL")

    check(result["success"] is True, f"case handling failed: {result}")


def test_compare_journals_reports_unknown_journal():

    agent = _seed_agent_memory()

    result = agent.compare_journals("Alpha Journal", "Nonexistent")

    check(result["success"] is False, "should fail on unknown journal")


def test_explain_journal_score_returns_breakdown():

    agent = _seed_agent_memory()

    result = agent.explain_journal_score("Alpha Journal")

    check(result["success"] is True, f"explain failed: {result}")
    check(
        result["explanation"]["ml_analysis"]["final_ml_score"] == 0.80,
        "wrong ml score reported",
    )


def test_explain_journal_score_lists_alternatives_when_missing():

    agent = _seed_agent_memory()

    result = agent.explain_journal_score("Missing Journal")

    check(result["success"] is False, "should fail on unknown journal")
    check("Alpha Journal" in result["error"], "should list available journals")


def test_agent_reads_constraints_from_raw_user_input():
    """Slot-filling drops preferences from `topic`; raw input must carry them.

    This is the bug where the agent answered an open-access request with
    five closed-access journals.
    """

    import agent

    seen = {}

    def fake_interpret(text, use_cache=True):
        seen["text"] = text
        return {
            "raw_input": text,
            "search_query": "graphene supercapacitors",
            "interpreted": True,
            "expanded_queries": [],
            "field_hints": [],
            "intent": "search",
            "is_valid": True,
            "needs_clarification": False,
            "clarification_question": "",
            "constraints": {
                "open_access": True,
                "publisher_exclude": [],
                "other": [],
            },
        }

    originals = (
        agent.interpret_query,
        agent.retrieve_journals,
        agent.rerank_journals,
        agent.memory.last_user_input,
    )

    agent.interpret_query = fake_interpret
    agent.retrieve_journals = lambda **kwargs: [journal("X", is_oa=True)]
    agent.rerank_journals = lambda **kwargs: [
        {"name": "X", "llm_score": 0.9, "is_oa": True, "publisher": "P"}
    ]
    agent.memory.last_user_input = (
        "find me OPEN ACCESS journals for graphene supercapacitors"
    )

    try:
        agent.retrieve_and_rank_journals(topic="graphene supercapacitors")
    finally:
        (
            agent.interpret_query,
            agent.retrieve_journals,
            agent.rerank_journals,
            agent.memory.last_user_input,
        ) = originals

    check(
        "OPEN ACCESS" in seen.get("text", ""),
        "the user's own words never reached the interpreter",
    )


def test_agent_fallback_applies_constraints():
    """Never executed before: the ML fallback when re-ranking fails."""

    import agent

    def fake_interpret(text, use_cache=True):
        return {
            "raw_input": text,
            "search_query": text,
            "interpreted": True,
            "expanded_queries": [],
            "field_hints": [],
            "intent": "search",
            "is_valid": True,
            "needs_clarification": False,
            "clarification_question": "",
            "constraints": {
                "open_access": True,
                "publisher_exclude": [],
                "other": [],
            },
        }

    pool = [
        journal("ClosedStrong", is_oa=False, final_score=0.90),
        journal("OpenDecent", is_oa=True, final_score=0.80),
    ]

    originals = (
        agent.interpret_query,
        agent.retrieve_journals,
        agent.rerank_journals,
    )

    agent.interpret_query = fake_interpret
    agent.retrieve_journals = lambda **kwargs: pool
    agent.rerank_journals = lambda **kwargs: []   # force the fallback

    try:
        result = agent.retrieve_and_rank_journals(topic="anything", top_k=2)
    finally:
        (
            agent.interpret_query,
            agent.retrieve_journals,
            agent.rerank_journals,
        ) = originals

    names = [j["name"] for j in result["top_results"]]

    check(names, "fallback returned nothing")
    check(
        names[0] == "OpenDecent",
        f"fallback ignored the open access constraint: {names}",
    )


if __name__ == "__main__":
    print("Running pipeline tests (no network)\n")
    sys.exit(run_all())
